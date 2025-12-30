"""
测试 static_obj_feat 格式适配
验证 (N, 20, 3) 和 (N, 20) mask 格式
"""

import numpy as np
import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from transfuser_gt_utils_ads import (
    ADSGTConfig,
    compute_bev_semantic_map_ads,
    visualize_bev_semantic_map,
)

def test_static_obj_feat_format():
    """测试 (N, 20, 3) 格式的 static_obj_feat"""
    
    print("=" * 60)
    print("测试 static_obj_feat 格式适配")
    print("=" * 60)
    
    # 1. 创建模拟数据
    # static_obj_feat: (50, 20, 3) - 50条线，每条20个点，每个点3维
    N, P = 50, 20
    static_obj_feat = np.zeros((N, P, 3), dtype=np.float32)
    static_obj_mask = np.zeros((N, P), dtype=np.float32)
    
    # 填充一些有效的车道线数据（带曲线）
    for i in range(7):  # 前7条线是有效的
        # 生成带曲线的车道线
        x_coords = np.linspace(-20, 20, P)
        
        # 添加弯曲效果（正弦曲线）
        if i < 3:  # 左侧车道线
            y_offset = (i - 1) * 3.5 - 7  # 左侧，间隔3.5米
            y_coords = y_offset + 0.5 * np.sin(x_coords / 10)  # 轻微弯曲
        elif i < 5:  # 中间车道线  
            y_offset = (i - 3.5) * 3.5
            y_coords = y_offset + 0.3 * np.sin(x_coords / 10)
        else:  # 右侧车道线
            y_offset = (i - 4) * 3.5 + 7
            y_coords = y_offset + 0.5 * np.sin(x_coords / 10)
        
        z_coords = np.zeros(P)  # 第三维无意义
        
        static_obj_feat[i, :, 0] = x_coords
        static_obj_feat[i, :, 1] = y_coords
        static_obj_feat[i, :, 2] = z_coords  # 这一维会被忽略
        
        # mask: 大部分点有效
        static_obj_mask[i, :18] = 1.0
    
    # 2. 创建更丰富的动态物体场景
    obj_labels = np.array([
        # [x, y, z, lx, ly, lz, heading, label, state, vx, vy]
        # 前方车辆
        [8.0, -3.5, 0.0, 4.5, 2.0, 1.5, 0.0, 2, 0, 5.0, 0.0],    # vehicle 同车道
        [15.0, 0.0, 0.0, 4.2, 1.9, 1.5, 0.05, 2, 0, 3.0, 0.0],   # vehicle 前方
        [12.0, 3.5, 0.0, 4.0, 1.8, 1.5, -0.1, 2, 0, 4.0, 0.0],   # vehicle 左车道
        # 后方车辆
        [-6.0, -3.5, 0.0, 4.3, 2.0, 1.5, 0.0, 2, 0, 6.0, 0.0],   # vehicle 后方
        [-10.0, 0.0, 0.0, 4.5, 2.0, 1.5, 0.0, 2, 0, 5.0, 0.0],   # vehicle 后方中间
        # 行人
        [5.0, 8.0, 0.0, 0.5, 0.5, 1.7, 0.0, 7, 0, 1.0, 0.5],     # pedestrian 路边
        [-3.0, -8.0, 0.0, 0.5, 0.5, 1.7, 0.0, 7, 0, 0.5, 0.0],   # pedestrian 路边
        [10.0, 7.0, 0.0, 0.6, 0.6, 1.8, 0.0, 7, 0, 0.8, 0.2],    # pedestrian
    ])
    
    # 3. 配置（更高分辨率）
    config = ADSGTConfig(
        bev_pixel_width=256,
        bev_pixel_height=256,
        lidar_min_x=-32.0,
        lidar_max_x=32.0,
        lidar_min_y=-32.0,
        lidar_max_y=32.0,
        vis_scale=8,  # ✨ 更高分辨率 2048x2048
    )
    
    # 4. 生成 BEV 语义图
    print(f"\nInput shapes:")
    print(f"  static_obj_feat: {static_obj_feat.shape}")
    print(f"  static_obj_mask: {static_obj_mask.shape}")
    print(f"  obj_labels: {obj_labels.shape}")
    
    try:
        bev_map = compute_bev_semantic_map_ads(
            obj_labels=obj_labels,
            config=config,
            static_obj_feat=static_obj_feat,
            static_obj_mask=static_obj_mask,
        )
        
        print(f"\nOutput:")
        print(f"  BEV map shape: {bev_map.shape}")
        
        # 统计分布
        unique, counts = np.unique(bev_map, return_counts=True)
        label_names = {
            0: "background",
            1: "lane_lines",
            2: "static",
            3: "vehicle",
            4: "pedestrian"
        }
        print("\n  BEV Label Distribution:")
        for label, count in zip(unique, counts):
            name = label_names.get(int(label), f"label_{label}")
            pct = 100 * count / bev_map.size
            print(f"    {int(label)} ({name:12s}): {count:6d} ({pct:5.2f}%)")
        
        # 5. 可视化
        save_path = "/tmp/test_static_obj_feat_format.png"
        visualize_bev_semantic_map(
            bev_map=bev_map,
            config=config,
            save_path=save_path,
            title="Test: static_obj_feat (50,20,3)",
            high_res=True,
            show_grid=True,
            show_legend=True,
            show_distance=True,
            show_range_text=True,
        )
        
        print(f"\n✅ 测试成功！")
        print(f"   可视化已保存到: {save_path}")
        
    except Exception as e:
        print(f"\n❌ 测试失败！")
        print(f"   错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    return True


if __name__ == "__main__":
    success = test_static_obj_feat_format()
    sys.exit(0 if success else 1)

