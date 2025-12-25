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
    """BEV 语义图类别定义 (5类，含车道线)"""
    BACKGROUND = 0
    LANE_LINES = 1       # 车道线
    STATIC_OBJECTS = 2   # 静态物体 (barriers, cones等)
    VEHICLES = 3         # 车辆
    PEDESTRIANS = 4      # 行人


@dataclass
class ADSGTConfig:
    """ADS GT生成配置"""
    # 范围 (米) - ego-centric，ego在中心
    # 默认: 前后各32m，左右各32m
    lidar_min_x: float = -32.0
    lidar_max_x: float = 32.0
    lidar_min_y: float = -32.0
    lidar_max_y: float = 32.0

    # Agent detection
    num_bounding_boxes: int = 20

    # FOV 限制 (用于只有front view的情况)
    use_fov_filter: bool = False  # 默认关闭，因为ego在中间需要看四周
    fov_angle_deg: float = 60.0  # 总FOV角度 (左右各30度)

    # BEV训练尺寸 (用于训练的语义图)
    bev_pixel_width: int = 256
    bev_pixel_height: int = 256  # 改为正方形，前后对称
    bev_pixel_size: float = 0.25  # 每像素对应的米数

    # 可视化分辨率 (高分辨率用于可视化)
    vis_scale: int = 4  # 可视化放大倍数

    # BEV类别数
    num_bev_classes: int = 5  # background + lane + static + vehicle + pedestrian

    # 车道线绘制参数
    lane_line_thickness: int = 2  # 训练用的线宽
    lane_line_vis_thickness: int = 3  # 可视化用的线宽

    @property
    def bev_semantic_frame(self) -> Tuple[int, int]:
        """训练用的BEV尺寸"""
        return (self.bev_pixel_height, self.bev_pixel_width)

    @property
    def bev_vis_frame(self) -> Tuple[int, int]:
        """可视化用的高分辨率BEV尺寸"""
        return (self.bev_pixel_height * self.vis_scale,
                self.bev_pixel_width * self.vis_scale)

    @property
    def bev_radius(self) -> float:
        values = [self.lidar_min_x, self.lidar_max_x, self.lidar_min_y, self.lidar_max_y]
        return max([abs(value) for value in values])

    @property
    def fov_half_angle_rad(self) -> float:
        return math.radians(self.fov_angle_deg / 2.0)

    @property
    def ego_pixel_x(self) -> int:
        """Ego在BEV图中的X像素位置 (ego在中心)"""
        # x=0 对应的像素位置
        return int(-self.lidar_min_x / self.bev_pixel_size)

    @property
    def ego_pixel_y(self) -> int:
        """Ego在BEV图中的Y像素位置 (ego在中心)"""
        # y=0 对应的像素位置
        return int(-self.lidar_min_y / self.bev_pixel_size)


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

def _world_to_pixel(x: float, y: float, config: ADSGTConfig, scale: int = 1) -> Tuple[int, int]:
    """
    将世界坐标转换为 BEV 图像像素坐标 (ego在中心)

    坐标系定义:
    - 世界坐标: x正向前，y正向左
    - 图像坐标: row向下增加，col向右增加
    - ego在图像中心

    Args:
        x, y: 世界坐标 (米)
        config: 配置
        scale: 分辨率放大倍数 (用于高分辨率可视化)

    Returns:
        (col, row): OpenCV格式的像素坐标
    """
    H = config.bev_pixel_height * scale
    W = config.bev_pixel_width * scale
    pixel_size = config.bev_pixel_size / scale

    # ego在图像中心
    # x正向前 -> 图像向上 -> row减小
    # y正向左 -> 图像向左 -> col减小
    row = int(H / 2 - x / pixel_size)
    col = int(W / 2 - y / pixel_size)

    return (col, row)  # OpenCV格式 (x, y) = (col, row)


def _world_to_pixel_array(coords: npt.NDArray, config: ADSGTConfig, scale: int = 1) -> npt.NDArray[np.int32]:
    """
    批量将世界坐标转换为像素坐标

    Args:
        coords: shape (N, 2) - [x, y] 世界坐标
        config: 配置
        scale: 分辨率放大倍数

    Returns:
        pixel_coords: shape (N, 1, 2) - OpenCV格式 [col, row]
    """
    H = config.bev_pixel_height * scale
    W = config.bev_pixel_width * scale
    pixel_size = config.bev_pixel_size / scale

    x = coords[:, 0]
    y = coords[:, 1]

    row = (H / 2 - x / pixel_size).astype(np.int32)
    col = (W / 2 - y / pixel_size).astype(np.int32)

    # OpenCV需要 (N, 1, 2) 格式，每个点是 [col, row]
    pixel_coords = np.stack([col, row], axis=-1).reshape(-1, 1, 2)
    return pixel_coords


def _draw_box_on_bev(
    bev_map: npt.NDArray[np.uint8],
    x: float, y: float, heading: float,
    length: float, width: float,
    label: int,
    config: ADSGTConfig,
    scale: int = 1
) -> None:
    """
    在 BEV 图上绘制一个旋转的 bounding box

    Args:
        bev_map: BEV语义图 (会被原地修改)
        x, y: 物体中心的世界坐标
        heading: 航向角 (弧度)
        length: 长度 (x方向)
        width: 宽度 (y方向)
        label: 类别标签
        config: 配置
        scale: 分辨率放大倍数
    """
    # 计算box的四个角点 (相对于物体中心的偏移)
    half_l = length / 2
    half_w = width / 2

    # 四个角点 (物体坐标系)
    corners_local = np.array([
        [ half_l,  half_w],  # 前左
        [ half_l, -half_w],  # 前右
        [-half_l, -half_w],  # 后右
        [-half_l,  half_w],  # 后左
    ])

    # 旋转矩阵
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    rot_matrix = np.array([[cos_h, -sin_h], [sin_h, cos_h]])

    # 旋转并平移到世界坐标
    corners_world = corners_local @ rot_matrix.T
    corners_world[:, 0] += x
    corners_world[:, 1] += y

    # 转换为像素坐标
    pixel_coords = _world_to_pixel_array(corners_world, config, scale)

    # 绘制填充多边形
    cv2.fillPoly(bev_map, [pixel_coords], color=int(label))


def compute_bev_semantic_map_ads(
    obj_labels: npt.NDArray[np.float64],
    config: ADSGTConfig = None,
    valid_mask: Optional[npt.NDArray[np.bool_]] = None,
    static_obj_feat: Optional[npt.NDArray] = None,
    static_obj_mask: Optional[npt.NDArray] = None,
    curb_feat: Optional[npt.NDArray] = None,
    curb_mask: Optional[npt.NDArray] = None,
    lane_feat: Optional[npt.NDArray] = None,
    lane_mask: Optional[npt.NDArray] = None,
) -> npt.NDArray[np.uint8]:
    """
    从 ADS 数据计算 BEV Semantic Map (ego在中心)

    类别定义:
        0: background
        1: lane_lines (车道线)
        2: static_objects (barriers, cones等)
        3: vehicles
        4: pedestrians

    Args:
        obj_labels: shape (N, 11) - 动态物体
        config: 配置
        valid_mask: shape (N,) - 有效物体的 mask (可选)
        static_obj_feat: shape (M, P, 2) - 静态物体轮廓点
        static_obj_mask: shape (M, ...) - 静态物体 mask
        curb_feat: shape (K, P, 2) - 路缘点
        curb_mask: shape (K, ...) - 路缘 mask
        lane_feat: shape (L, P, 2) - 车道线点
        lane_mask: shape (L, ...) - 车道线 mask

    Returns:
        bev_semantic_map: shape (H, W) - 语义分割图
    """
    if config is None:
        config = ADSGTConfig()

    bev_semantic_map = np.zeros(config.bev_semantic_frame, dtype=np.uint8)

    # 1. 绘制车道线 (最底层)
    if lane_feat is not None and lane_mask is not None:
        _draw_lane_lines(bev_semantic_map, lane_feat, lane_mask,
                         BEVClassIndex.LANE_LINES, config)

    # 2. 绘制路缘
    if curb_feat is not None and curb_mask is not None:
        _draw_polylines(bev_semantic_map, curb_feat, curb_mask,
                        BEVClassIndex.STATIC_OBJECTS, config,
                        thickness=config.lane_line_thickness)

    # 3. 绘制静态物体
    if static_obj_feat is not None and static_obj_mask is not None:
        _draw_static_objects(bev_semantic_map, static_obj_feat, static_obj_mask,
                             BEVClassIndex.STATIC_OBJECTS, config)

    # 4. 绘制动态物体 (车辆和行人，优先级最高)
    for i, obj in enumerate(obj_labels):
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


def _draw_polylines(
    bev_map: npt.NDArray[np.uint8],
    feat: npt.NDArray,
    mask: npt.NDArray,
    label: int,
    config: ADSGTConfig,
    thickness: int = 2,
    scale: int = 1,
    closed: bool = False
) -> None:
    """
    通用的多段线绘制函数

    Args:
        bev_map: BEV语义图
        feat: shape (N, P, 2) - N条线，每条有P个点 [x, y]
        mask: shape (N, ...) - 有效性mask
        label: 类别标签
        config: 配置
        thickness: 线宽
        scale: 分辨率放大倍数
        closed: 是否闭合
    """
    # 处理 mask
    if mask.ndim >= 2:
        valid = mask[..., -1].astype(bool)
    else:
        valid = mask.astype(bool)

    for i in range(feat.shape[0]):
        if i >= len(valid) or not valid[i]:
            continue

        points = feat[i]  # (P, 2)

        # 过滤无效点 (距离太小的点视为padding)
        valid_mask = np.linalg.norm(points, axis=-1) > 0.01
        valid_points = points[valid_mask]
        if len(valid_points) < 2:
            continue

        # 转换为像素坐标
        pixel_coords = _world_to_pixel_array(valid_points, config, scale)

        # 绘制线条
        cv2.polylines(bev_map, [pixel_coords], isClosed=closed,
                      color=int(label), thickness=thickness)


def _draw_lane_lines(
    bev_map: npt.NDArray[np.uint8],
    lane_feat: npt.NDArray,
    lane_mask: npt.NDArray,
    label: int,
    config: ADSGTConfig,
    scale: int = 1
) -> None:
    """
    绘制车道线

    Args:
        lane_feat: shape (N, P, 2) - N条车道线
        lane_mask: shape (N, ...) - 有效性mask
    """
    thickness = config.lane_line_thickness * scale
    _draw_polylines(bev_map, lane_feat, lane_mask, label, config,
                    thickness=thickness, scale=scale, closed=False)


def _draw_static_objects(
    bev_map: npt.NDArray[np.uint8],
    static_feat: npt.NDArray,
    static_mask: npt.NDArray,
    label: int,
    config: ADSGTConfig,
    scale: int = 1
) -> None:
    """
    绘制静态物体轮廓 (作为填充多边形)

    Args:
        static_feat: shape (N, P, 2) - N个静态物体，每个有P个点
    """
    # 处理 mask
    if static_mask.ndim >= 2:
        valid = static_mask[..., -1].astype(bool)
    else:
        valid = static_mask.astype(bool)

    for i in range(static_feat.shape[0]):
        if i >= len(valid) or not valid[i]:
            continue

        points = static_feat[i]  # (P, 2)

        # 过滤无效点
        valid_mask = np.linalg.norm(points, axis=-1) > 0.01
        valid_points = points[valid_mask]
        if len(valid_points) < 3:  # 多边形至少需要3个点
            continue

        # 转换为像素坐标
        pixel_coords = _world_to_pixel_array(valid_points, config, scale)

        # 绘制填充多边形
        cv2.fillPoly(bev_map, [pixel_coords], color=int(label))


# ============ 简化版：仅使用 obj_labels ============

def compute_bev_semantic_map_ads_simple(
    obj_labels: npt.NDArray[np.float64],
    config: ADSGTConfig = None,
) -> npt.NDArray[np.uint8]:
    """
    从 ADS obj_label 计算 BEV Semantic Map (简化版，仅使用动态物体)

    类别定义:
        0: background
        1: lane_lines (简化版无此类)
        2: static_objects (其他类型)
        3: vehicles
        4: pedestrians

    Args:
        obj_labels: shape (N, 11) - [x, y, z, lx, ly, lz, heading, label, state, vx, vy]
        config: 配置

    Returns:
        bev_semantic_map: shape (H, W) - 语义分割图
    """
    if config is None:
        config = ADSGTConfig()

    bev_semantic_map = np.zeros(config.bev_semantic_frame, dtype=np.uint8)

    # 按类别分类物体 (先绘制静态物体，再绘制车辆，最后绘制行人)
    static_objs = []
    vehicles = []
    pedestrians = []

    for obj in obj_labels:
        if _is_invalid_obj_label_row(obj):
            continue

        x = obj[ADSObjectIndex.X]
        y = obj[ADSObjectIndex.Y]

        # 检查是否在范围内
        if not (config.lidar_min_x <= x <= config.lidar_max_x and
                config.lidar_min_y <= y <= config.lidar_max_y):
            continue

        obj_class = int(obj[ADSObjectIndex.LABEL])

        if obj_class in [1, 2, 3, 4, 5, 6]:
            vehicles.append(obj)
        elif obj_class in [7, 8, 9]:
            pedestrians.append(obj)
        else:
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
    lane_feat: Optional[npt.NDArray] = None,
    lane_mask: Optional[npt.NDArray] = None,
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
        lane_feat: 车道线特征 (可选)
        lane_mask: 车道线mask (可选)

    Returns:
        dict: 包含 agent_states, agent_labels, bev_semantic_map
    """
    if config is None:
        config = ADSGTConfig()

    # Agent targets
    agent_states, agent_labels = compute_agent_targets_ads(obj_labels, config)

    # BEV semantic map
    if static_obj_feat is not None or lane_feat is not None:
        bev_semantic_map = compute_bev_semantic_map_ads(
            obj_labels, config, None,
            static_obj_feat, static_obj_mask,
            curb_feat, curb_mask,
            lane_feat, lane_mask
        )
    else:
        bev_semantic_map = compute_bev_semantic_map_ads_simple(obj_labels, config)

    return {
        'agent_states': agent_states,       # shape (num_bounding_boxes, 5)
        'agent_labels': agent_labels,       # shape (num_bounding_boxes,) int64
        'bev_semantic_map': bev_semantic_map,  # shape (H, W)
    }


# ============ 可视化工具 ============

# BEV 类别对应的颜色 (BGR 格式) - 更美观的配色
BEV_CLASS_COLORS = {
    BEVClassIndex.BACKGROUND: (30, 30, 30),        # 深灰背景
    BEVClassIndex.LANE_LINES: (255, 255, 255),     # 白色 - 车道线
    BEVClassIndex.STATIC_OBJECTS: (100, 100, 100), # 灰色 - 静态物体
    BEVClassIndex.VEHICLES: (0, 165, 255),          # 橙色 - 车辆
    BEVClassIndex.PEDESTRIANS: (0, 255, 0),         # 绿色 - 行人
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
    high_res: bool = True,
) -> npt.NDArray[np.uint8]:
    """
    可视化 BEV Semantic Map (ego在中心，支持高分辨率)

    Args:
        bev_map: shape (H, W) - 语义分割图
        agent_states: shape (N, 5) - [x, y, heading, length, width] (可选)
        agent_labels: shape (N,) - agent 类别标签 (可选)
        config: ADSGTConfig (可选)
        save_path: 保存路径 (可选)
        title: 图像标题
        high_res: 是否使用高分辨率 (默认True)

    Returns:
        vis_image: shape (H, W, 3) - BGR 可视化图像
    """
    if config is None:
        config = ADSGTConfig()

    H_orig, W_orig = bev_map.shape
    scale = config.vis_scale if high_res else 1

    # 放大 bev_map 用于可视化
    if scale > 1:
        vis_bev = cv2.resize(bev_map, (W_orig * scale, H_orig * scale),
                             interpolation=cv2.INTER_NEAREST)
    else:
        vis_bev = bev_map.copy()

    H, W = vis_bev.shape

    # 创建彩色可视化图
    vis_image = np.zeros((H, W, 3), dtype=np.uint8)

    # 根据类别填充颜色
    for cls_idx, color in BEV_CLASS_COLORS.items():
        mask = (vis_bev == cls_idx)
        vis_image[mask] = color

    # 计算 ego 中心位置 (图像中心)
    ego_col = W // 2
    ego_row = H // 2

    # 绘制网格线 (每 10m 一条)
    pixel_size_vis = config.bev_pixel_size / scale
    pixels_per_10m = int(10.0 / pixel_size_vis)

    # 垂直线 (Y 方向，左右)
    for i in range(-10, 11):
        col = ego_col + i * pixels_per_10m
        if 0 <= col < W:
            cv2.line(vis_image, (col, 0), (col, H-1), (50, 50, 50), 1)

    # 水平线 (X 方向，前后)
    for i in range(-10, 11):
        row = ego_row + i * pixels_per_10m
        if 0 <= row < H:
            cv2.line(vis_image, (0, row), (W-1, row), (50, 50, 50), 1)

    # 绘制坐标轴
    # X轴 (前后方向) - 红色
    cv2.arrowedLine(vis_image, (ego_col, ego_row), (ego_col, ego_row - pixels_per_10m * 2),
                    (0, 0, 255), 2, tipLength=0.15)
    # Y轴 (左右方向) - 绿色
    cv2.arrowedLine(vis_image, (ego_col, ego_row), (ego_col - pixels_per_10m * 2, ego_row),
                    (0, 255, 0), 2, tipLength=0.15)

    # 绘制 Agent bounding boxes (如果有)
    if agent_states is not None and agent_labels is not None:
        for state, label in zip(agent_states, agent_labels):
            if label == AgentClassIndex.EMPTY:
                continue

            x, y, heading, length, width = state[:5]

            # 跳过无效的 agent
            if x == 0 and y == 0:
                continue

            # 转换到像素坐标 (使用可视化坐标)
            col, row = _world_to_pixel(x, y, config, scale)

            # 绘制旋转矩形
            half_l = length / pixel_size_vis / 2
            half_w = width / pixel_size_vis / 2

            # 四个角点 (物体坐标系)
            corners_local = np.array([
                [ half_l,  half_w],
                [ half_l, -half_w],
                [-half_l, -half_w],
                [-half_l,  half_w],
            ])

            # 旋转
            cos_h = np.cos(heading)
            sin_h = np.sin(heading)

            # 旋转并转换为像素坐标
            # 注意: 图像坐标系中，y轴向下，需要调整
            rotated = np.zeros_like(corners_local)
            for i, (dl, dw) in enumerate(corners_local):
                # 旋转
                dx = dl * cos_h - dw * sin_h
                dy = dl * sin_h + dw * cos_h
                # 转换到像素 (x正向上->row减, y正向左->col减)
                rotated[i, 0] = col - dy / pixel_size_vis * config.bev_pixel_size
                rotated[i, 1] = row - dx / pixel_size_vis * config.bev_pixel_size

            pts = rotated.astype(np.int32).reshape((-1, 1, 2))

            color = AGENT_CLASS_COLORS.get(int(label), (255, 255, 255))
            cv2.polylines(vis_image, [pts], isClosed=True, color=color, thickness=2)

    # 绘制自车 (ego) - 在中心
    ego_length_px = int(4.5 / pixel_size_vis)  # 假设车长4.5m
    ego_width_px = int(2.0 / pixel_size_vis)   # 假设车宽2.0m
    ego_pts = np.array([
        [ego_col - ego_width_px // 2, ego_row - ego_length_px // 2],  # 前左
        [ego_col + ego_width_px // 2, ego_row - ego_length_px // 2],  # 前右
        [ego_col + ego_width_px // 2, ego_row + ego_length_px // 2],  # 后右
        [ego_col - ego_width_px // 2, ego_row + ego_length_px // 2],  # 后左
    ], dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(vis_image, [ego_pts], isClosed=True, color=(0, 0, 255), thickness=3)
    cv2.circle(vis_image, (ego_col, ego_row), 6, (0, 0, 255), -1)

    # 添加图例
    legend_x = 15
    legend_y = 30
    label_names = {
        BEVClassIndex.BACKGROUND: "Background",
        BEVClassIndex.LANE_LINES: "Lane Lines",
        BEVClassIndex.STATIC_OBJECTS: "Static Obj",
        BEVClassIndex.VEHICLES: "Vehicle",
        BEVClassIndex.PEDESTRIANS: "Pedestrian",
    }
    for cls_idx, color in BEV_CLASS_COLORS.items():
        cv2.rectangle(vis_image, (legend_x, legend_y - 15), (legend_x + 20, legend_y + 5), color, -1)
        cv2.putText(vis_image, label_names.get(cls_idx, "Unknown"), (legend_x + 28, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        legend_y += 25

    # 添加距离标注
    for dist in [10, 20, 30]:
        dist_px = int(dist / pixel_size_vis)
        # 前方标注
        cv2.putText(vis_image, f"{dist}m", (ego_col + 5, ego_row - dist_px),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # 添加标题
    cv2.putText(vis_image, title, (W // 2 - 80, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 添加范围信息
    range_text = f"Range: [{config.lidar_min_x}, {config.lidar_max_x}]m x [{config.lidar_min_y}, {config.lidar_max_y}]m"
    cv2.putText(vis_image, range_text, (10, H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

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

