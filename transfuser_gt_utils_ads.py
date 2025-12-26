"""
ADS Data 专用 GT Label 生成工具
用于从华为ADS数据生成agent targets和BEV semantic map的GT labels

与 NavSim 版本的区别：
1. 不依赖 nuplan 地图 API
2. 使用 ADS 数据格式 (obj_label, static_obj_feat, curb_feat)
3. 类别数量更少 (无 centerline)

BEV 类别定义 (4类):
    0: background
    1: static_objects (barriers, cones等)
    2: vehicles
    3: pedestrians

Agent 类别定义 (4类):
    0: empty
    1: vehicle
    2: pedestrian
    3: other
"""

import numpy as np
import numpy.typing as npt
import cv2
import math
from typing import Dict, List, Tuple, Optional, Any
from enum import IntEnum
from dataclasses import dataclass


# ============ ADS Object Label 索引定义 ============
class ADSObjectIndex(IntEnum):
    """ADS obj_label 数组索引定义"""
    X = 0
    Y = 1
    Z = 2
    LX = 3  # length
    LY = 4  # width
    LZ = 5  # height
    HEADING = 6
    LABEL = 7
    STATE = 8
    VX = 9
    VY = 10


class BoundingBox2DIndex(IntEnum):
    """2D bounding boxes 输出索引"""
    X = 0
    Y = 1
    HEADING = 2
    LENGTH = 3
    WIDTH = 4

    @classmethod
    def size(cls):
        return 5


# ============ Agent Label 类别定义 ============
class AgentClassIndex(IntEnum):
    """Agent 多类标签定义 (4类)"""
    EMPTY = 0       # 空/背景（填充用）
    VEHICLE = 1     # 车辆
    PEDESTRIAN = 2  # 行人
    OTHER = 3       # 其他类型


# ADS 数据中的类别标签映射 (根据 FusionClassification 枚举)
# obj_label[7] (LABEL) 的类别定义如下：
# 0: UNKNOWN, 1: MICRO_CAR, 2: CAR, 3: VAN, 4: LIGHT_TRUCK, 5: TRUCK, 6: BUS
# 7: PEDESTRIAN, 8: CYCLIST_BIKE, 9: CYCLIST_MOTORCYCLE, 10: MOTORCYCLE, 11: BICYCLE
ADS_LABEL_TO_AGENT_CLASS = {
    0: AgentClassIndex.OTHER,        # CLASSIFICATION_UNKNOWN
    1: AgentClassIndex.VEHICLE,      # CLASSIFICATION_MICRO_CAR
    2: AgentClassIndex.VEHICLE,      # CLASSIFICATION_CAR
    3: AgentClassIndex.VEHICLE,      # CLASSIFICATION_VAN
    4: AgentClassIndex.VEHICLE,      # CLASSIFICATION_LIGHT_TRUCK
    5: AgentClassIndex.VEHICLE,      # CLASSIFICATION_TRUCK
    6: AgentClassIndex.VEHICLE,      # CLASSIFICATION_BUS
    7: AgentClassIndex.PEDESTRIAN,   # CLASSIFICATION_PEDESTRIAN
    8: AgentClassIndex.PEDESTRIAN,   # CLASSIFICATION_CYCLIST_BIKE (骑自行车的人归为行人)
    9: AgentClassIndex.PEDESTRIAN,   # CLASSIFICATION_CYCLIST_MOTORCYCLE (骑摩托车的人归为行人)
    10: AgentClassIndex.OTHER,       # CLASSIFICATION_MOTORCYCLE (无人摩托车)
    11: AgentClassIndex.OTHER,       # CLASSIFICATION_BICYCLE (无人自行车)
}


# ============ BEV 类别定义 ============
class BEVClassIndex(IntEnum):
    """BEV 语义图类别定义 (4类，无 centerline)"""
    BACKGROUND = 0
    STATIC_OBJECTS = 1
    VEHICLES = 2
    PEDESTRIANS = 3


@dataclass
class ADSGTConfig:
    """ADS GT生成配置"""
    # 范围 (米)
    # NOTE: default to a front BEV frame (x in [0, 32]) with y in [-32, 32],
    # matching bev_pixel_height=128, bev_pixel_size=0.25 -> 32m forward.
    lidar_min_x: float = 0.0
    lidar_max_x: float = 32.0
    lidar_min_y: float = -32.0
    lidar_max_y: float = 32.0
    
    # Agent detection
    num_bounding_boxes: int = 20
    
    # FOV 限制 (用于只有front view的情况)
    use_fov_filter: bool = True
    fov_angle_deg: float = 60.0  # 总FOV角度 (左右各30度)
    
    # BEV尺寸
    bev_pixel_width: int = 256
    bev_pixel_height: int = 128
    bev_pixel_size: float = 0.25  # 每像素对应的米数
    
    # BEV类别数
    num_bev_classes: int = 4  # background + static + vehicle + pedestrian
    
    @property
    def bev_semantic_frame(self) -> Tuple[int, int]:
        return (self.bev_pixel_height, self.bev_pixel_width)
    
    @property
    def bev_radius(self) -> float:
        values = [self.lidar_min_x, self.lidar_max_x, self.lidar_min_y, self.lidar_max_y]
        return max([abs(value) for value in values])
    
    @property
    def fov_half_angle_rad(self) -> float:
        return math.radians(self.fov_angle_deg / 2.0)


# ============ Agent Targets 生成 ============

def _is_invalid_obj_label_row(obj: npt.NDArray[np.float64]) -> bool:
    """
    Robust invalid row check for obj_label.
    Common paddings observed:
    - xyz all -1 (checkerPredict_vis.py uses mean(xyz) == -1)
    - xyz all around -1000 (legacy padding)
    """
    xyz = np.asarray(obj[:3], dtype=np.float64)
    if np.allclose(xyz, -1.0, atol=1e-3):
        return True
    if np.all(xyz < -900.0):
        return True
    return False


def compute_agent_targets_ads(
    obj_labels: npt.NDArray[np.float64],
    config: ADSGTConfig = None,
    valid_mask: Optional[npt.NDArray[np.bool_]] = None,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.int64]]:
    """
    从 ADS obj_label 计算 Agent Targets (多类标签)
    
    Args:
        obj_labels: shape (N, 11) - [x, y, z, lx, ly, lz, heading, label, state, vx, vy]
        config: 配置
        valid_mask: shape (N,) - 有效物体的 mask (可选)
        
    Returns:
        agent_states: shape (num_bounding_boxes, 5) - [x, y, heading, length, width]
        agent_labels: shape (num_bounding_boxes,) - 类别标签
    """
    if config is None:
        config = ADSGTConfig()
    
    max_agents = config.num_bounding_boxes
    agent_states_list: List[npt.NDArray[np.float32]] = []
    agent_labels_list: List[int] = []
    
    def _in_range(x: float, y: float) -> bool:
        """检查是否在范围内"""
        # 距离检查
        if not (config.lidar_min_x <= x <= config.lidar_max_x and
                config.lidar_min_y <= y <= config.lidar_max_y):
            return False
        
        # FOV 检查 (可选)
        if config.use_fov_filter:
            # 只考虑前方 (x > 0)
            if x <= 0:
                return False
            # 计算方位角
            azimuth = math.atan2(y, x)
            if abs(azimuth) > config.fov_half_angle_rad:
                return False
        
        return True
    
    for i, obj in enumerate(obj_labels):
        # 检查是否无效
        if _is_invalid_obj_label_row(obj):
            continue
        
        # 检查 valid_mask
        if valid_mask is not None and not valid_mask[i]:
            continue
        
        box_x = obj[ADSObjectIndex.X]
        box_y = obj[ADSObjectIndex.Y]
        box_heading = obj[ADSObjectIndex.HEADING]
        box_length = obj[ADSObjectIndex.LX]
        box_width = obj[ADSObjectIndex.LY]
        obj_label_class = int(obj[ADSObjectIndex.LABEL])
        
        # 检查是否在范围内
        if not _in_range(box_x, box_y):
            continue
        
        # 获取类别
        agent_class = ADS_LABEL_TO_AGENT_CLASS.get(obj_label_class, AgentClassIndex.OTHER)
        
        agent_states_list.append(
            np.array([box_x, box_y, box_heading, box_length, box_width], dtype=np.float32)
        )
        agent_labels_list.append(int(agent_class))
    
    # 初始化输出
    agent_states = np.zeros((max_agents, BoundingBox2DIndex.size()), dtype=np.float32)
    agent_labels = np.zeros(max_agents, dtype=np.int64)  # 默认为0 (EMPTY)
    
    if len(agent_states_list) > 0:
        agents_states_arr = np.array(agent_states_list)
        agents_labels_arr = np.array(agent_labels_list)
        
        # 按距离排序，取最近的
        distances = np.linalg.norm(agents_states_arr[:, :2], axis=-1)
        argsort = np.argsort(distances)[:max_agents]
        
        agents_states_arr = agents_states_arr[argsort]
        agents_labels_arr = agents_labels_arr[argsort]
        
        agent_states[:len(agents_states_arr)] = agents_states_arr
        agent_labels[:len(agents_labels_arr)] = agents_labels_arr
    
    return agent_states, agent_labels


# ============ BEV Semantic Map 生成 ============

def _coords_to_pixel(coords: npt.NDArray, config: ADSGTConfig) -> npt.NDArray[np.int32]:
    """
    将局部坐标转换为 BEV 图像像素坐标
    与 vis_utils.py 中的 draw_static_obj_bev 保持一致

    坐标系:
        - 车辆 x (前方) -> 图像 v (行，向上为负)
        - 车辆 y (左方) -> 图像 u (列，向左为负)

    Args:
        coords: shape (..., 2) - [x, y] 格式，单位为米
        config: ADSGTConfig

    Returns:
        pixel_coords: shape (..., 2) - [u, v] 格式，即 [列, 行]
    """
    H, W = config.bev_semantic_frame  # (128, 256)
    # 自车位置在图像底部中心
    center_u = W // 2   # 128
    center_v = H - 1    # 127 (底部)
    meter_pix_scale = 1.0 / config.bev_pixel_size  # 每米对应的像素数

    x = coords[..., 0]  # 前方距离
    y = coords[..., 1]  # 左右距离

    u = center_u - y * meter_pix_scale  # 图像列
    v = center_v - x * meter_pix_scale  # 图像行

    pixel_coords = np.stack([u, v], axis=-1).astype(np.int32)
    return pixel_coords


def _draw_box_on_bev(
    bev_map: npt.NDArray[np.uint8],
    x: float, y: float, heading: float,
    length: float, width: float,
    label: int,
    config: ADSGTConfig
) -> None:
    """
    在 BEV 图上绘制一个填充的 box
    """
    # 计算box的四个角点 (局部坐标)
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)

    # 四个角点相对于中心的偏移
    half_l = length / 2
    half_w = width / 2

    corners = np.array([
        [ half_l,  half_w],
        [ half_l, -half_w],
        [-half_l, -half_w],
        [-half_l,  half_w],
    ])

    # 旋转
    rot_matrix = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
    rotated_corners = corners @ rot_matrix.T

    # 平移到物体位置
    rotated_corners[:, 0] += x
    rotated_corners[:, 1] += y

    # 转换为像素坐标 [u, v] 格式
    pixel_coords = _coords_to_pixel(rotated_corners, config)

    # 绘制填充多边形
    # OpenCV fillPoly 需要 shape (N, 1, 2) 格式
    pts = pixel_coords.reshape((-1, 1, 2))
    cv2.fillPoly(bev_map, [pts], color=int(label))


def compute_bev_semantic_map_ads(
    obj_labels: npt.NDArray[np.float64],
    config: ADSGTConfig = None,
    valid_mask: Optional[npt.NDArray[np.bool_]] = None,
    static_obj_feat: Optional[npt.NDArray] = None,
    static_obj_mask: Optional[npt.NDArray] = None,
    curb_feat: Optional[npt.NDArray] = None,
    curb_mask: Optional[npt.NDArray] = None,
) -> npt.NDArray[np.uint8]:
    """
    从 ADS 数据计算 BEV Semantic Map
    
    类别定义:
        0: background
        1: static_objects (barriers, cones等)
        2: vehicles
        3: pedestrians
    
    Args:
        obj_labels: shape (N, 11) - 动态物体
        config: 配置
        valid_mask: shape (N,) - 有效物体的 mask (可选)
        static_obj_feat: shape (M, P, 2) - 静态物体轮廓点
        static_obj_mask: shape (M, ...) - 静态物体 mask
        curb_feat: shape (K, P, 2) - 路缘点
        curb_mask: shape (K, ...) - 路缘 mask
        
    Returns:
        bev_semantic_map: shape (128, 256) - 语义分割图
    """
    if config is None:
        config = ADSGTConfig()
    
    bev_semantic_map = np.zeros(config.bev_semantic_frame, dtype=np.uint8)
    
    # 1. 绘制静态物体 (如果有)
    if static_obj_feat is not None and static_obj_mask is not None:
        _draw_static_objects(bev_semantic_map, static_obj_feat, static_obj_mask, 
                           BEVClassIndex.STATIC_OBJECTS, config)
    
    # 2. 绘制路缘 (作为静态物体的一部分)
    if curb_feat is not None and curb_mask is not None:
        _draw_curb_lines(bev_semantic_map, curb_feat, curb_mask, 
                        BEVClassIndex.STATIC_OBJECTS, config)
    
    # 3. 绘制动态物体 (车辆和行人)
    for i, obj in enumerate(obj_labels):
        # 检查是否无效
        if _is_invalid_obj_label_row(obj):
            continue
        
        if valid_mask is not None and not valid_mask[i]:
            continue
        
        x = obj[ADSObjectIndex.X]
        y = obj[ADSObjectIndex.Y]
        heading = obj[ADSObjectIndex.HEADING]
        length = obj[ADSObjectIndex.LX]
        width = obj[ADSObjectIndex.LY]
        obj_class = int(obj[ADSObjectIndex.LABEL])
        
        # 检查是否在范围内
        if not (config.lidar_min_x <= x <= config.lidar_max_x and
                config.lidar_min_y <= y <= config.lidar_max_y):
            continue
        
        # 确定 BEV 类别 (根据 ADS FusionClassification)
        if obj_class in [1, 2, 3, 4, 5, 6]:  # 各类车辆
            bev_label = BEVClassIndex.VEHICLES
        elif obj_class in [7, 8, 9]:  # 行人和骑车人
            bev_label = BEVClassIndex.PEDESTRIANS
        else:  # 0 (unknown), 10-11 (无人车辆) -> static
            bev_label = BEVClassIndex.STATIC_OBJECTS
        
        _draw_box_on_bev(bev_semantic_map, x, y, heading, length, width, bev_label, config)
    
    return bev_semantic_map


def _draw_static_objects(
    bev_map: npt.NDArray[np.uint8],
    static_feat: npt.NDArray,
    static_mask: npt.NDArray,
    label: int,
    config: ADSGTConfig
) -> None:
    """
    绘制静态物体轮廓（车道线等）

    Args:
        static_feat: shape (N, P, 2) - N个静态物体，每个有P个点，格式 [x, y]
    """
    # 处理 mask (取最后一维)
    if static_mask.ndim >= 2:
        valid = static_mask[..., -1].astype(bool)
    else:
        valid = static_mask.astype(bool)

    for i in range(static_feat.shape[0]):
        if not valid[i]:
            continue

        points = static_feat[i]  # (P, 2)

        # 过滤无效点
        valid_points = points[np.linalg.norm(points, axis=-1) > 0.01]
        if len(valid_points) < 2:
            continue

        # 转换为像素坐标 [u, v] 格式
        pixel_coords = _coords_to_pixel(valid_points, config)

        # 绘制多边形线条
        pts = pixel_coords.reshape((-1, 1, 2))
        cv2.polylines(bev_map, [pts], isClosed=False, color=int(label), thickness=2)


def _draw_curb_lines(
    bev_map: npt.NDArray[np.uint8],
    curb_feat: npt.NDArray,
    curb_mask: npt.NDArray,
    label: int,
    config: ADSGTConfig
) -> None:
    """
    绘制路缘线
    """
    # 处理 mask (取最后一维)
    if curb_mask.ndim >= 2:
        valid = curb_mask[..., -1].astype(bool)
    else:
        valid = curb_mask.astype(bool)

    for i in range(curb_feat.shape[0]):
        if not valid[i]:
            continue

        points = curb_feat[i]  # (P, 2)

        # 过滤无效点
        valid_points = points[np.linalg.norm(points, axis=-1) > 0.01]
        if len(valid_points) < 2:
            continue

        # 转换为像素坐标 [u, v] 格式
        pixel_coords = _coords_to_pixel(valid_points, config)

        # 绘制线条
        pts = pixel_coords.reshape((-1, 1, 2))
        cv2.polylines(bev_map, [pts], isClosed=False, color=int(label), thickness=1)


# ============ 简化版：仅使用 obj_labels ============

def compute_bev_semantic_map_ads_simple(
    obj_labels: npt.NDArray[np.float64],
    config: ADSGTConfig = None,
) -> npt.NDArray[np.uint8]:
    """
    从 ADS obj_label 计算 BEV Semantic Map (简化版，仅使用动态物体)
    
    类别定义:
        0: background
        1: static_objects (其他类型)
        2: vehicles
        3: pedestrians
    
    Args:
        obj_labels: shape (N, 11) - [x, y, z, lx, ly, lz, heading, label, state, vx, vy]
        config: 配置
        
    Returns:
        bev_semantic_map: shape (128, 256) - 语义分割图
    """
    if config is None:
        config = ADSGTConfig()
    
    bev_semantic_map = np.zeros(config.bev_semantic_frame, dtype=np.uint8)
    
    # 按类别分类物体 (先绘制静态物体，再绘制车辆，最后绘制行人)
    # 根据 ADS FusionClassification:
    # 1-6: 车辆类, 7-9: 行人/骑车人, 10-11: 无人车辆/自行车, 0: unknown
    static_objs = []
    vehicles = []
    pedestrians = []
    
    for obj in obj_labels:
        # 检查是否无效
        if _is_invalid_obj_label_row(obj):
            continue
        
        x = obj[ADSObjectIndex.X]
        y = obj[ADSObjectIndex.Y]
        
        # 检查是否在范围内
        if not (config.lidar_min_x <= x <= config.lidar_max_x and
                config.lidar_min_y <= y <= config.lidar_max_y):
            continue
        
        obj_class = int(obj[ADSObjectIndex.LABEL])
        
        # 根据 ADS 类别映射
        if obj_class in [1, 2, 3, 4, 5, 6]:  # 各类车辆
            vehicles.append(obj)
        elif obj_class in [7, 8, 9]:  # 行人和骑车人
            pedestrians.append(obj)
        else:  # 0 (unknown), 10-11 (无人车辆) -> static/other
            static_objs.append(obj)
    
    # 按顺序绘制: static -> vehicle -> pedestrian
    for obj in static_objs:
        x, y, heading = obj[ADSObjectIndex.X], obj[ADSObjectIndex.Y], obj[ADSObjectIndex.HEADING]
        length, width = obj[ADSObjectIndex.LX], obj[ADSObjectIndex.LY]
        _draw_box_on_bev(bev_semantic_map, x, y, heading, length, width, 
                        BEVClassIndex.STATIC_OBJECTS, config)
    
    for obj in vehicles:
        x, y, heading = obj[ADSObjectIndex.X], obj[ADSObjectIndex.Y], obj[ADSObjectIndex.HEADING]
        length, width = obj[ADSObjectIndex.LX], obj[ADSObjectIndex.LY]
        _draw_box_on_bev(bev_semantic_map, x, y, heading, length, width, 
                        BEVClassIndex.VEHICLES, config)
    
    for obj in pedestrians:
        x, y, heading = obj[ADSObjectIndex.X], obj[ADSObjectIndex.Y], obj[ADSObjectIndex.HEADING]
        length, width = obj[ADSObjectIndex.LX], obj[ADSObjectIndex.LY]
        _draw_box_on_bev(bev_semantic_map, x, y, heading, length, width, 
                        BEVClassIndex.PEDESTRIANS, config)
    
    return bev_semantic_map


# ============ 便捷函数 ============

def process_ads_frame(
    obj_labels: npt.NDArray[np.float64],
    config: ADSGTConfig = None,
    static_obj_feat: Optional[npt.NDArray] = None,
    static_obj_mask: Optional[npt.NDArray] = None,
    curb_feat: Optional[npt.NDArray] = None,
    curb_mask: Optional[npt.NDArray] = None,
) -> dict:
    """
    处理单帧 ADS 数据，生成 agent targets 和 BEV 语义图
    
    Args:
        obj_labels: shape (N, 11) - 动态物体标签
        config: 配置
        static_obj_feat: 静态物体特征 (可选)
        static_obj_mask: 静态物体mask (可选)
        curb_feat: 路缘特征 (可选)
        curb_mask: 路缘mask (可选)
        
    Returns:
        dict: 包含 agent_states, agent_labels, bev_semantic_map
    """
    if config is None:
        config = ADSGTConfig()
    
    # Agent targets
    agent_states, agent_labels = compute_agent_targets_ads(obj_labels, config)
    
    # BEV semantic map
    if static_obj_feat is not None:
        bev_semantic_map = compute_bev_semantic_map_ads(
            obj_labels, config, None,
            static_obj_feat, static_obj_mask,
            curb_feat, curb_mask
        )
    else:
        bev_semantic_map = compute_bev_semantic_map_ads_simple(obj_labels, config)
    
    return {
        'agent_states': agent_states,       # shape (num_bounding_boxes, 5)
        'agent_labels': agent_labels,       # shape (num_bounding_boxes,) int64
        'bev_semantic_map': bev_semantic_map,  # shape (128, 256)
    }


# ============ 可视化工具 ============

# BEV 类别对应的颜色 (BGR 格式)
BEV_CLASS_COLORS = {
    BEVClassIndex.BACKGROUND: (40, 40, 40),       # 深灰色背景
    BEVClassIndex.STATIC_OBJECTS: (128, 128, 128), # 灰色 - 静态物体/车道线
    BEVClassIndex.VEHICLES: (0, 165, 255),         # 橙色 - 车辆
    BEVClassIndex.PEDESTRIANS: (0, 255, 0),        # 绿色 - 行人
}

# Agent 类别对应的颜色 (BGR 格式)
AGENT_CLASS_COLORS = {
    AgentClassIndex.EMPTY: (100, 100, 100),
    AgentClassIndex.VEHICLE: (0, 165, 255),
    AgentClassIndex.PEDESTRIAN: (0, 255, 0),
    AgentClassIndex.OTHER: (255, 0, 255),
}


def visualize_bev_semantic_map(
    bev_map: npt.NDArray[np.uint8],
    agent_states: Optional[npt.NDArray] = None,
    agent_labels: Optional[npt.NDArray] = None,
    config: Optional[ADSGTConfig] = None,
    save_path: Optional[str] = None,
    title: str = "BEV Semantic Map",
) -> npt.NDArray[np.uint8]:
    """
    可视化 BEV Semantic Map
    
    Args:
        bev_map: shape (H, W) - 语义分割图，值为 0-3
        agent_states: shape (N, 5) - [x, y, heading, length, width] (可选)
        agent_labels: shape (N,) - agent 类别标签 (可选)
        config: ADSGTConfig (可选)
        save_path: 保存路径 (可选)
        title: 图像标题
        
    Returns:
        vis_image: shape (H, W, 3) - BGR 可视化图像
    """
    if config is None:
        config = ADSGTConfig()
    
    H, W = bev_map.shape
    
    # 创建彩色可视化图
    vis_image = np.zeros((H, W, 3), dtype=np.uint8)
    
    # 根据类别填充颜色
    for cls_idx, color in BEV_CLASS_COLORS.items():
        mask = (bev_map == cls_idx)
        vis_image[mask] = color
    
    # 绘制网格线 (每 10m 一条)
    pixels_per_10m = int(10.0 / config.bev_pixel_size)
    
    # 垂直线 (Y 方向)
    center_x = W // 2
    for i in range(-5, 6):
        x = center_x + i * pixels_per_10m
        if 0 <= x < W:
            cv2.line(vis_image, (x, 0), (x, H-1), (60, 60, 60), 1)
    
    # 水平线 (X 方向) - 从底部往上
    for i in range(0, 10):
        y = H - 1 - i * pixels_per_10m
        if 0 <= y < H:
            cv2.line(vis_image, (0, y), (W-1, y), (60, 60, 60), 1)
    
    # 绘制自车位置 (底部中心)
    ego_x = W // 2
    ego_y = H - 1
    cv2.circle(vis_image, (ego_x, ego_y), 5, (0, 0, 255), -1)  # 红色圆点
    cv2.arrowedLine(vis_image, (ego_x, ego_y), (ego_x, ego_y - 20), (0, 0, 255), 2)  # 红色箭头
    
    # 添加图例
    legend_y = 20
    for cls_idx, color in BEV_CLASS_COLORS.items():
        label_names = {0: "Background", 1: "Static/Lane", 2: "Vehicle", 3: "Pedestrian"}
        cv2.rectangle(vis_image, (10, legend_y - 12), (25, legend_y + 3), color, -1)
        cv2.putText(vis_image, label_names[cls_idx], (30, legend_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        legend_y += 18
    
    # 添加标题
    cv2.putText(vis_image, title, (W // 2 - 60, 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 保存
    if save_path is not None:
        cv2.imwrite(save_path, vis_image)
        print(f"BEV visualization saved to: {save_path}")
    
    return vis_image


def visualize_bev_debug(
    cur_pkl_data: Dict[str, Any],
    obj_label: npt.NDArray,
    config: Optional[ADSGTConfig] = None,
    save_dir: str = "/tmp",
    sample_id: str = "debug",
) -> str:
    """
    调试用：从 pkl 数据生成完整的 BEV 可视化
    
    Args:
        cur_pkl_data: 原始 pkl 数据字典
        obj_label: 转换后的 obj_label (N, 11)
        config: 配置
        save_dir: 保存目录
        sample_id: 样本 ID (用于文件名)
        
    Returns:
        save_path: 保存的文件路径
    """
    import os
    
    if config is None:
        config = ADSGTConfig()
    
    # 获取静态物体数据
    static_obj_feat = cur_pkl_data.get("static_obj_feat", None)
    static_obj_mask = cur_pkl_data.get("static_obj_mask", None)
    
    if static_obj_feat is not None:
        if hasattr(static_obj_feat, 'numpy'):
            static_obj_feat = static_obj_feat.numpy()
        static_obj_feat = np.array(static_obj_feat)
    
    if static_obj_mask is not None:
        if hasattr(static_obj_mask, 'numpy'):
            static_obj_mask = static_obj_mask.numpy()
        static_obj_mask = np.array(static_obj_mask)
    
    # 生成 BEV map
    bev_map = compute_bev_semantic_map_ads(
        obj_labels=obj_label,
        config=config,
        static_obj_feat=static_obj_feat,
        static_obj_mask=static_obj_mask,
    )
    
    # 生成 Agent targets
    agent_states, agent_labels = compute_agent_targets_ads(obj_label, config)
    
    # 可视化
    save_path = os.path.join(save_dir, f"bev_debug_{sample_id}.png")
    visualize_bev_semantic_map(
        bev_map=bev_map,
        agent_states=agent_states,
        agent_labels=agent_labels,
        config=config,
        save_path=save_path,
        title=f"BEV Debug: {sample_id}",
    )
    
    # 打印统计信息
    print(f"\n{'='*50}")
    print(f"BEV Debug: {sample_id}")
    print(f"{'='*50}")
    print(f"obj_label shape: {obj_label.shape}")
    print(f"static_obj_feat shape: {static_obj_feat.shape if static_obj_feat is not None else 'None'}")
    print(f"BEV map shape: {bev_map.shape}")
    
    unique, counts = np.unique(bev_map, return_counts=True)
    label_names = {0: "background", 1: "static", 2: "vehicle", 3: "pedestrian"}
    print("BEV Label Distribution:")
    for label, count in zip(unique, counts):
        name = label_names.get(int(label), f"label_{label}")
        total = bev_map.size
        print(f"  {int(label)} ({name}): {count} ({100*count/total:.2f}%)")
    
    print(f"\nAgent states shape: {agent_states.shape}")
    non_empty = (agent_labels != 0).sum()
    print(f"Non-empty agents: {non_empty}/{len(agent_labels)}")
    
    print(f"\nVisualization saved to: {save_path}")
    print(f"{'='*50}\n")
    
    return save_path


if __name__ == "__main__":
    """测试代码"""
    # 创建模拟数据
    obj_labels = np.array([
        # [x, y, z, lx, ly, lz, heading, label, state, vx, vy]
        [10.0, 2.0, 0.0, 4.5, 2.0, 1.5, 0.1, 2, 0, 5.0, 0.0],   # vehicle (CAR)
        [15.0, -3.0, 0.0, 4.0, 1.8, 1.5, -0.2, 2, 0, 3.0, 0.0],  # vehicle (CAR)
        [8.0, 5.0, 0.0, 0.5, 0.5, 1.7, 0.0, 7, 0, 1.0, 0.5],    # pedestrian
        [-1000, -1000, -1000, 0, 0, 0, 0, 0, 0, 0, 0],          # invalid (padding)
    ])
    
    config = ADSGTConfig()
    
    print("=" * 60)
    print("ADS GT Utils 测试")
    print("=" * 60)
    
    # 测试 agent targets
    agent_states, agent_labels = compute_agent_targets_ads(obj_labels, config)
    print(f"\nAgent States shape: {agent_states.shape}")
    print(f"Agent Labels shape: {agent_labels.shape}")
    print(f"Agent Labels: {agent_labels[:5]}")
    
    # 测试 BEV semantic map
    bev_map = compute_bev_semantic_map_ads_simple(obj_labels, config)
    print(f"\nBEV Map shape: {bev_map.shape}")
    
    unique, counts = np.unique(bev_map, return_counts=True)
    label_names = {0: "background", 1: "static", 2: "vehicle", 3: "pedestrian"}
    print("BEV Label Distribution:")
    for label, count in zip(unique, counts):
        name = label_names.get(int(label), f"label_{label}")
        total = bev_map.size
        print(f"  {int(label)} ({name}): {count} ({100*count/total:.2f}%)")
    
    # 保存可视化
    vis_image = visualize_bev_semantic_map(bev_map, agent_states, agent_labels, config)
    cv2.imwrite('/tmp/bev_semantic_map_ads_test.png', vis_image)
    print("\nVisualization saved to /tmp/bev_semantic_map_ads_test.png")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

