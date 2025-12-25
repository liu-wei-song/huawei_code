#!/usr/bin/env python3
"""
Agent/BEV GT 生成和可视化测试脚本

用于验证 ADS 数据的 Agent Detection 和 BEV Semantic Map 生成是否正确。
使用模拟数据，不需要真实的 pkl 文件。

运行方式:
    cd /mnt/nvme0n1p1/weisong.liu/ross_qwen
    python huawei_code/test_agent_bev_visualization.py
"""

import os
import sys
import numpy as np
import cv2

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huawei_code.transfuser_gt_utils_ads import (
    ADSGTConfig,
    compute_agent_targets_ads,
    compute_bev_semantic_map_ads,
    compute_bev_semantic_map_ads_simple,
    visualize_bev_semantic_map,
    BEVClassIndex,
    AgentClassIndex,
)


def create_synthetic_obj_labels():
    """
    创建合成的 obj_labels 数据用于测试
    
    obj_label 格式 (N, 11):
        [x, y, z, length, width, height, heading, class_id, state, vx, vy]
    
    ADS FusionClassification:
        0: UNKNOWN
        1: OTHER
        2: CAR
        3: BUS
        4: TRUCK
        5: BICYCLE
        6: TRICYCLE
        7: PEDESTRIAN
        8: CONE
        9: BICYCLE_WITHOUT_DRIVER
        10: WALL
        11: MISC
    """
    obj_labels = []
    
    # ===== 场景 1: 前方车辆 =====
    # 前方 10m, 偏右 2m 的轿车
    obj_labels.append([10.0, 2.0, 0.0, 4.5, 2.0, 1.5, 0.0, 2, 0, 5.0, 0.0])
    
    # 前方 20m, 偏左 -3m 的轿车
    obj_labels.append([20.0, -3.0, 0.0, 4.0, 1.8, 1.5, 0.1, 2, 0, 3.0, 0.0])
    
    # 前方 15m 的卡车
    obj_labels.append([15.0, 0.0, 0.0, 8.0, 2.5, 3.0, 0.0, 4, 0, 2.0, 0.0])
    
    # ===== 场景 2: 行人 =====
    # 右侧人行道的行人
    obj_labels.append([8.0, 5.0, 0.0, 0.5, 0.5, 1.7, 0.5, 7, 0, 1.0, 0.5])
    
    # 左侧的行人
    obj_labels.append([12.0, -6.0, 0.0, 0.5, 0.5, 1.7, -0.3, 7, 0, 0.8, -0.2])
    
    # ===== 场景 3: 骑车人 =====
    # 骑自行车的人
    obj_labels.append([18.0, 4.0, 0.0, 1.8, 0.8, 1.8, 0.2, 5, 0, 4.0, 0.3])
    
    # 三轮车
    obj_labels.append([25.0, -2.0, 0.0, 2.5, 1.2, 1.6, 0.0, 6, 0, 3.5, 0.0])
    
    # ===== 场景 4: 静态障碍物 =====
    # 锥桶
    obj_labels.append([5.0, 1.0, 0.0, 0.3, 0.3, 0.5, 0.0, 8, 0, 0.0, 0.0])
    obj_labels.append([5.5, 1.5, 0.0, 0.3, 0.3, 0.5, 0.0, 8, 0, 0.0, 0.0])
    
    # ===== 场景 5: 后方车辆 =====
    # 后方的车
    obj_labels.append([-8.0, 1.0, 0.0, 4.2, 1.9, 1.5, 3.14, 2, 0, -2.0, 0.0])
    
    # ===== 无效数据 (padding) =====
    # 这些应该被过滤掉
    obj_labels.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0])  # x,y 都是 0
    obj_labels.append([-1000, -1000, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0])  # padding
    
    return np.array(obj_labels, dtype=np.float32)


def create_synthetic_static_obj_feat():
    """
    创建合成的静态物体特征 (车道线等)
    
    static_obj_feat 格式: (N_polylines, N_points, 2) - 每个多段线的点坐标
    """
    # 模拟车道线
    polylines = []
    
    # 左车道线
    left_lane = np.array([
        [0, -4], [10, -4], [20, -4], [30, -4], [40, -4], [50, -4]
    ], dtype=np.float32)
    polylines.append(left_lane)
    
    # 中间车道线 (虚线)
    center_lane = np.array([
        [0, 0], [10, 0], [20, 0], [30, 0], [40, 0], [50, 0]
    ], dtype=np.float32)
    polylines.append(center_lane)
    
    # 右车道线
    right_lane = np.array([
        [0, 4], [10, 4], [20, 4], [30, 4], [40, 4], [50, 4]
    ], dtype=np.float32)
    polylines.append(right_lane)
    
    # 路沿
    left_curb = np.array([
        [0, -8], [10, -8], [20, -8], [30, -8], [40, -8], [50, -8]
    ], dtype=np.float32)
    polylines.append(left_curb)
    
    right_curb = np.array([
        [0, 8], [10, 8], [20, 8], [30, 8], [40, 8], [50, 8]
    ], dtype=np.float32)
    polylines.append(right_curb)
    
    # Padding 到相同长度
    max_points = max(len(p) for p in polylines)
    padded = []
    for p in polylines:
        if len(p) < max_points:
            pad = np.zeros((max_points - len(p), 2), dtype=np.float32)
            p = np.concatenate([p, pad], axis=0)
        padded.append(p)
    
    static_obj_feat = np.stack(padded, axis=0)  # (N, max_points, 2)
    static_obj_mask = np.ones((len(polylines), max_points), dtype=bool)
    
    return static_obj_feat, static_obj_mask


def test_agent_targets():
    """测试 Agent Targets 生成"""
    print("\n" + "="*60)
    print("测试 Agent Targets 生成")
    print("="*60)
    
    obj_labels = create_synthetic_obj_labels()
    config = ADSGTConfig(num_bounding_boxes=20)
    
    agent_states, agent_labels = compute_agent_targets_ads(obj_labels, config)
    
    print(f"\nInput obj_labels shape: {obj_labels.shape}")
    print(f"Output agent_states shape: {agent_states.shape}")
    print(f"Output agent_labels shape: {agent_labels.shape}")
    
    # 统计类别分布
    label_names = {
        0: "empty",
        1: "vehicle",
        2: "pedestrian",
        3: "other"
    }
    
    print("\nAgent Label Distribution:")
    unique, counts = np.unique(agent_labels, return_counts=True)
    for label, count in zip(unique, counts):
        name = label_names.get(int(label), f"unknown_{label}")
        print(f"  {int(label)} ({name}): {count}")
    
    # 打印有效 agent 的详细信息
    print("\nValid Agents (label != 0):")
    valid_mask = agent_labels != 0
    valid_indices = np.where(valid_mask)[0]
    for i in valid_indices[:10]:  # 最多显示10个
        state = agent_states[i]
        label = agent_labels[i]
        label_name = label_names.get(int(label), "?")
        print(f"  Agent {i}: label={int(label)} ({label_name}), "
              f"x={state[0]:.2f}, y={state[1]:.2f}, "
              f"l={state[2]:.2f}, w={state[3]:.2f}, h={state[4]:.2f}")
    
    return agent_states, agent_labels


def test_bev_semantic_map():
    """测试 BEV Semantic Map 生成"""
    print("\n" + "="*60)
    print("测试 BEV Semantic Map 生成")
    print("="*60)
    
    obj_labels = create_synthetic_obj_labels()
    static_obj_feat, static_obj_mask = create_synthetic_static_obj_feat()
    
    config = ADSGTConfig(
        bev_pixel_width=256,
        bev_pixel_height=128,
        lidar_min_x=-32.0,
        lidar_max_x=32.0,
        lidar_min_y=-16.0,
        lidar_max_y=16.0,
    )
    
    print(f"\nInput obj_labels shape: {obj_labels.shape}")
    print(f"Input static_obj_feat shape: {static_obj_feat.shape}")
    print(f"Config: {config.bev_pixel_width}x{config.bev_pixel_height}, "
          f"x=({config.lidar_min_x}, {config.lidar_max_x}), y=({config.lidar_min_y}, {config.lidar_max_y})")
    
    # 使用简单版本（不带静态物体）
    bev_map_simple = compute_bev_semantic_map_ads_simple(obj_labels, config)
    
    # 使用完整版本（带静态物体）
    bev_map_full = compute_bev_semantic_map_ads(
        obj_labels=obj_labels,
        config=config,
        static_obj_feat=static_obj_feat,
        static_obj_mask=static_obj_mask,
    )
    
    print(f"\nBEV Map (simple) shape: {bev_map_simple.shape}")
    print(f"BEV Map (full) shape: {bev_map_full.shape}")
    
    # 统计类别分布
    label_names = {
        0: "background",
        1: "static",
        2: "vehicle",
        3: "pedestrian"
    }
    
    print("\nBEV Label Distribution (simple):")
    unique, counts = np.unique(bev_map_simple, return_counts=True)
    total = bev_map_simple.size
    for label, count in zip(unique, counts):
        name = label_names.get(int(label), f"unknown_{label}")
        print(f"  {int(label)} ({name}): {count} ({100*count/total:.2f}%)")
    
    print("\nBEV Label Distribution (full):")
    unique, counts = np.unique(bev_map_full, return_counts=True)
    total = bev_map_full.size
    for label, count in zip(unique, counts):
        name = label_names.get(int(label), f"unknown_{label}")
        print(f"  {int(label)} ({name}): {count} ({100*count/total:.2f}%)")
    
    return bev_map_simple, bev_map_full


def test_visualization(save_dir="/tmp/bev_test"):
    """测试可视化功能"""
    print("\n" + "="*60)
    print("测试可视化功能")
    print("="*60)
    
    os.makedirs(save_dir, exist_ok=True)
    
    obj_labels = create_synthetic_obj_labels()
    static_obj_feat, static_obj_mask = create_synthetic_static_obj_feat()
    
    config = ADSGTConfig(
        bev_pixel_width=256,
        bev_pixel_height=128,
        lidar_min_x=-32.0,
        lidar_max_x=32.0,
        lidar_min_y=-16.0,
        lidar_max_y=16.0,
        num_bounding_boxes=20,
    )
    
    # 生成 GT
    agent_states, agent_labels = compute_agent_targets_ads(obj_labels, config)
    bev_map = compute_bev_semantic_map_ads(
        obj_labels=obj_labels,
        config=config,
        static_obj_feat=static_obj_feat,
        static_obj_mask=static_obj_mask,
    )
    
    # 可视化
    vis_image = visualize_bev_semantic_map(
        bev_map=bev_map,
        agent_states=agent_states,
        agent_labels=agent_labels,
        config=config,
        save_path=None,
        title="Synthetic Test Scene",
    )
    
    # 保存
    save_path = os.path.join(save_dir, "bev_synthetic_test.png")
    cv2.imwrite(save_path, vis_image)
    print(f"\nVisualization saved to: {save_path}")
    
    # 创建一个更丰富的场景
    print("\n创建复杂场景...")
    complex_obj_labels = create_complex_scene()
    agent_states2, agent_labels2 = compute_agent_targets_ads(complex_obj_labels, config)
    bev_map2 = compute_bev_semantic_map_ads(
        obj_labels=complex_obj_labels,
        config=config,
        static_obj_feat=static_obj_feat,
        static_obj_mask=static_obj_mask,
    )
    
    vis_image2 = visualize_bev_semantic_map(
        bev_map=bev_map2,
        agent_states=agent_states2,
        agent_labels=agent_labels2,
        config=config,
        save_path=None,
        title="Complex Scene",
    )
    
    save_path2 = os.path.join(save_dir, "bev_complex_scene.png")
    cv2.imwrite(save_path2, vis_image2)
    print(f"Complex scene saved to: {save_path2}")
    
    return save_path, save_path2


def create_complex_scene():
    """创建一个更复杂的场景"""
    obj_labels = []
    
    # 多辆车形成车流
    for i in range(5):
        x = 10 + i * 8
        y = np.random.uniform(-2, 2)
        heading = np.random.uniform(-0.1, 0.1)
        obj_labels.append([x, y, 0, 4.5, 2.0, 1.5, heading, 2, 0, 3.0, 0.0])
    
    # 对向车流
    for i in range(3):
        x = 15 + i * 10
        y = -5 + np.random.uniform(-0.5, 0.5)
        heading = 3.14 + np.random.uniform(-0.1, 0.1)
        obj_labels.append([x, y, 0, 4.5, 2.0, 1.5, heading, 2, 0, -3.0, 0.0])
    
    # 多个行人在人行道
    for i in range(4):
        x = 8 + i * 5 + np.random.uniform(-1, 1)
        y = 6 + np.random.uniform(-0.5, 0.5)
        heading = np.random.uniform(-1, 1)
        obj_labels.append([x, y, 0, 0.5, 0.5, 1.7, heading, 7, 0, 1.0, 0.5])
    
    # 卡车和公交
    obj_labels.append([30, 0, 0, 10.0, 2.5, 3.5, 0.0, 4, 0, 2.0, 0.0])  # 卡车
    obj_labels.append([45, -4, 0, 12.0, 2.8, 3.2, 0.1, 3, 0, 2.5, 0.0])  # 公交
    
    # 骑车人
    obj_labels.append([12, 5, 0, 1.8, 0.8, 1.8, 0.3, 5, 0, 4.0, 0.5])
    obj_labels.append([20, 5.5, 0, 1.8, 0.8, 1.8, 0.2, 5, 0, 3.5, 0.3])
    
    # 锥桶组
    for i in range(4):
        obj_labels.append([5 + i * 0.8, 2.5 + i * 0.3, 0, 0.3, 0.3, 0.5, 0, 8, 0, 0, 0])
    
    return np.array(obj_labels, dtype=np.float32)


def main():
    """主测试函数"""
    print("="*60)
    print("ADS Agent/BEV GT 生成测试")
    print("="*60)
    
    # 1. 测试 Agent Targets
    agent_states, agent_labels = test_agent_targets()
    
    # 2. 测试 BEV Semantic Map
    bev_simple, bev_full = test_bev_semantic_map()
    
    # 3. 测试可视化
    save_path1, save_path2 = test_visualization()
    
    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)
    print(f"\n可视化结果保存在:")
    print(f"  1. {save_path1}")
    print(f"  2. {save_path2}")
    print("\n请查看生成的图像验证 Agent/BEV GT 是否正确。")
    print("="*60)


if __name__ == "__main__":
    main()

