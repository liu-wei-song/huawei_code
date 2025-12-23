"""
ADS Dataset for Qwen2.5-VL with Agent Detection and BEV Semantic Map support.

基于 LazySupervisedHuawei2VAROSSMOEDataset_Multiview4 扩展:
- 支持 <agent> 和 <map> 特殊token
- 动态生成 agent_states_gt, agent_labels_gt, bev_semantic_map_gt
- 支持4类BEV (无centerline)

BEV 类别定义 (4类):
    0: background
    1: static_objects
    2: vehicles
    3: pedestrians

Agent 类别定义 (4类):
    0: empty
    1: vehicle
    2: pedestrian
    3: other
"""

import os
import sys
import copy
import time
import random
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Sequence
from collections.abc import Sequence as SequenceType

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import transformers
from transformers import AutoProcessor

# ADS 相关导入
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from adsData import (
    ADSData,
    LazySupervisedHuawei2VAROSSMOEDataset_Multiview4,
    preprocess_qwen_2_visual_vla_sources,
    rank0_print,
    COMMAND_MAP,
    IGNORE_INDEX as ADS_IGNORE_INDEX,
)
from qwenvl.dataset.rope2d import get_rope_index_25, get_rope_index_2
from qwenvl.utils.token_utils import prepare_action_tokenizer_mapping

# GT 生成工具
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools', 'pickle_gen'))
try:
    from transfuser_gt_utils_ads import (
        ADSGTConfig,
        compute_agent_targets_ads,
        compute_bev_semantic_map_ads,
        visualize_bev_semantic_map,
    )
except ImportError:
    # 如果导入失败，提供 fallback
    print("Warning: transfuser_gt_utils_ads not found, using inline implementation")
    ADSGTConfig = None
    compute_bev_semantic_map_ads = None
    visualize_bev_semantic_map = None

import moxing
import torch.nn.functional as F
import torch.distributed as dist

from .command_utils.behavior_features import calc_ego_behavior, EgoBehaviorType
from .tbt_extractor import (
    compute_tbt_sd_info_from_bag,
    merge_tbt_features,
    find_next_tbt,
    get_combo_tbt_from_pkl,
    get_lane_tbt_from_pkl,
)
from .tbt_formatter import get_tbt_text

moxing.file.shift('os', 'mox')

IGNORE_INDEX = -100

# ============ 特殊Token定义 ============
AGENT_TOKEN = "<agent>"
MAP_TOKEN = "<map>"


def setup_agent_bev_tokens(tokenizer) -> Dict[str, int]:
    """
    添加 <agent> 和 <map> 特殊token到tokenizer
    
    Returns:
        dict: 包含 agent_token_id 和 map_token_id
    """
    special_tokens = {
        "additional_special_tokens": [AGENT_TOKEN, MAP_TOKEN]
    }
    
    # 检查是否已存在
    existing_special = tokenizer.additional_special_tokens if hasattr(tokenizer, 'additional_special_tokens') else []
    tokens_to_add = [t for t in [AGENT_TOKEN, MAP_TOKEN] if t not in existing_special]
    
    if tokens_to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": existing_special + tokens_to_add})
        rank0_print(f"Added special tokens: {tokens_to_add}")
    
    agent_token_id = tokenizer.convert_tokens_to_ids(AGENT_TOKEN)
    map_token_id = tokenizer.convert_tokens_to_ids(MAP_TOKEN)
    
    return {
        "agent_token_id": agent_token_id,
        "map_token_id": map_token_id,
    }


class LazySupervisedADSAgentBEVDataset(LazySupervisedHuawei2VAROSSMOEDataset_Multiview4):
    """
    ADS Dataset with Agent Detection and BEV Semantic Map support.
    
    继承自 LazySupervisedHuawei2VAROSSMOEDataset_Multiview4，添加:
    - <agent> token 用于 agent detection
    - <map> token 用于 BEV semantic map prediction
    """
    
    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args):
        # 调用父类初始化 (复用所有基础功能)
        super().__init__(tokenizer=tokenizer, data_args=data_args)
        
        # Setup agent/map special tokens
        token_info = setup_agent_bev_tokens(tokenizer)
        self.agent_token_id = token_info["agent_token_id"]
        self.map_token_id = token_info["map_token_id"]
        
        # Agent/Map query configuration
        self.num_agent_queries = getattr(data_args, "num_agent_queries", 20)
        self.num_map_queries = getattr(data_args, "num_map_queries", 64)
        
        rank0_print(f"Agent/Map tokens: agent_id={self.agent_token_id}, map_id={self.map_token_id}")
        rank0_print(f"Query counts: agent={self.num_agent_queries}, map={self.num_map_queries}")
        
        # Control which tokens to add
        self.enable_agent_tokens = getattr(data_args, "enable_agent_tokens", True)
        self.enable_bev_tokens = getattr(data_args, "enable_bev_tokens", True)
        
        # obj_label may be (T, N, 11). Default to last frame if time dimension exists.
        self.obj_label_time_index = getattr(data_args, "obj_label_time_index", -1)
        # object_feat may be (N, T, 22). Default to follow obj_label_time_index unless specified.
        self.object_feat_time_index = getattr(data_args, "object_feat_time_index", self.obj_label_time_index)

        # BEV config (keep consistent with model/config)
        self.bev_pixel_width = getattr(data_args, "bev_pixel_width", 256)
        self.bev_pixel_height = getattr(data_args, "bev_pixel_height", 128)
        self.bev_pixel_size = getattr(data_args, "bev_pixel_size", 0.25)

        # GT Config
        self.gt_config = ADSGTConfig(
            num_bounding_boxes=self.num_agent_queries,
            num_bev_classes=4,  # ADS 使用4类 BEV
            bev_pixel_width=self.bev_pixel_width,
            bev_pixel_height=self.bev_pixel_height,
            bev_pixel_size=self.bev_pixel_size,
            # Default to front BEV (0~32m) like NavSim; can override via args if needed.
            lidar_min_x=getattr(data_args, "bev_lidar_min_x", 0.0),
            lidar_max_x=getattr(data_args, "bev_lidar_max_x", 32.0),
            lidar_min_y=getattr(data_args, "bev_lidar_min_y", -32.0),
            lidar_max_y=getattr(data_args, "bev_lidar_max_y", 32.0),
            # Multi-view: default disable azimuth FOV cropping; can be enabled via args.
            use_fov_filter=getattr(data_args, "agent_use_fov_filter", False),
            fov_angle_deg=getattr(data_args, "agent_fov_angle_deg", 60.0),
        ) if ADSGTConfig else None
        
        rank0_print(f"ADS AgentBEV Dataset: agent_tokens={self.enable_agent_tokens}, bev_tokens={self.enable_bev_tokens}")

    @property
    def lengths(self):
        # 覆盖父类方法，添加 agent/map tokens 的长度估算
        length_list = []
        for sample in self.data:
            img_tokens = 2 * 128 * 4  # 2 frames, 4 views
            action_tokens = (int(1.0 * self.action_hz) + int(4.0 * self.action_hz))
            agent_tokens = self.num_agent_queries if self.enable_agent_tokens else 0
            map_tokens = self.num_map_queries if self.enable_bev_tokens else 0
            text_tokens = 200
            length_list.append(text_tokens + img_tokens + action_tokens + agent_tokens + map_tokens)
        return length_list
    
    # ---------- Agent/BEV 相关方法 ----------
    def build_prompt_with_queries(self, base_prompt: str) -> str:
        """
        构建包含 <agent> 和 <map> token 的 prompt
        """
        tokens = base_prompt
        if self.enable_agent_tokens:
            # IMPORTANT: ensure tokens are separable for tokenizer (avoid '<agent><agent>...' concatenation)
            tokens += " " + " ".join([AGENT_TOKEN] * self.num_agent_queries)
        if self.enable_bev_tokens:
            tokens += " " + " ".join([MAP_TOKEN] * self.num_map_queries)
        return tokens
    
    def _build_conversation_for_pairs(self, pairs_prompts: List[Tuple[str, str]]):
        """构建对话，只在当前帧添加 agent/map tokens"""
        conv = []
        for pre_p, cur_p in pairs_prompts:
            # 当前帧添加 agent/map tokens
            cur_p_with_queries = self.build_prompt_with_queries(cur_p)
            conv.extend([
                {"from": "user", "value": f"<image><image><image><image>{pre_p}"},
                {"from": "assistant", "value": ""},
                {"from": "user", "value": f"<image><image><image><image>{cur_p_with_queries}"},
                {"from": "assistant", "value": ""},
            ])
        return [conv]

    def _convert_object_feat_to_obj_label(
        self, 
        object_feat: Any,
        object_pred_mask: Optional[Any] = None,
    ) -> Optional[np.ndarray]:
        """
        Convert ADS `object_feat` to `obj_label` format (N, 11):
            [x, y, z, lx, ly, lz, heading, category, state, vx, vy]

        According to `huawei_code/ads_data.md`:
            object_feat shape: (N, T, 21) 或 (N, T, 22)
            特征顺序 (自车坐标系):
                0: object_pose_ego.x
                1: object_pose_ego.y
                2: object_pose_ego.heading
                3: object_fusion_class (FusionClassification 枚举)
                4: object_length
                5: object_width
                6: speed_limit
                7: object_vel (速度大小)
                8: object_vel_orien (速度方向)
                9: object_yaw_rate
                10: object_is_static
                11: object_left_turn_light
                12: object_right_turn_light
                13-20: object_box_pts[0-3] (x, y) 四个角点
                
        Args:
            object_feat: shape (N, T, 21) 或 (N, 21)
            object_pred_mask: shape (N,) 或 (N, T) - 有效物体 mask (可选)
        """
        if object_feat is None:
            return None

        # to numpy
        if isinstance(object_feat, torch.Tensor):
            feat = object_feat.detach().cpu().numpy()
        else:
            feat = np.array(object_feat)

        # object_feat shape: (N, T, feat_dim)
        if feat.ndim == 3:
            # 取指定时间步 (默认最后一帧，即当前帧)
            t = int(self.object_feat_time_index)
            if t < 0:
                t = feat.shape[1] + t
            if 0 <= t < feat.shape[1]:
                feat = feat[:, t, :]
            else:
                feat = feat[:, -1, :]
        elif feat.ndim == 2:
            # (N, feat_dim) 已经是单帧
            pass
        else:
            return None

        if feat.shape[-1] < 9:
            return None

        N = feat.shape[0]
        
        # 特征索引 (根据 ads_data.md)
        IDX_X = 0
        IDX_Y = 1
        IDX_HEADING = 2
        IDX_CLASS = 3       # FusionClassification 枚举值
        IDX_LENGTH = 4
        IDX_WIDTH = 5
        IDX_VEL = 7         # 速度大小
        IDX_VEL_ORIEN = 8   # 速度方向

        x = feat[:, IDX_X]
        y = feat[:, IDX_Y]
        heading = feat[:, IDX_HEADING]
        cls = feat[:, IDX_CLASS]
        length = feat[:, IDX_LENGTH]
        width = feat[:, IDX_WIDTH]
        speed = feat[:, IDX_VEL]
        vel_orien = feat[:, IDX_VEL_ORIEN]

        # 计算速度分量
        vx = speed * np.cos(vel_orien)
        vy = speed * np.sin(vel_orien)

        # ========== 有效性过滤 ==========
        # 1. 位置过滤: x ≈ 0 && y ≈ 0 是 padding
        pos_valid = ~((np.abs(x) < 1e-6) & (np.abs(y) < 1e-6))
        
        # 2. object_pred_mask 过滤 (如果提供)
        if object_pred_mask is not None:
            if isinstance(object_pred_mask, torch.Tensor):
                pred_mask = object_pred_mask.detach().cpu().numpy()
            else:
                pred_mask = np.array(object_pred_mask)
            
            # 处理多维 mask: 取最后一维或展平
            if pred_mask.ndim == 2:
                # (N, T) -> 取最后一帧
                pred_mask = pred_mask[:, -1]
            elif pred_mask.ndim > 2:
                pred_mask = pred_mask.reshape(N, -1)[:, -1]
            
            # 确保长度匹配
            if len(pred_mask) == N:
                pred_mask = pred_mask.astype(bool)
            else:
                pred_mask = np.ones(N, dtype=bool)
        else:
            pred_mask = np.ones(N, dtype=bool)
        
        # 综合过滤
        valid_mask = pos_valid & pred_mask

        # Build obj_label: [x, y, z, lx, ly, lz, heading, category, state, vx, vy]
        obj_label = np.zeros((N, 11), dtype=np.float32)
        obj_label[:, 0] = x
        obj_label[:, 1] = y
        obj_label[:, 2] = 0.0       # z (高度中心，ADS 数据未提供)
        obj_label[:, 3] = length    # lx (长度)
        obj_label[:, 4] = width     # ly (宽度)
        obj_label[:, 5] = 1.5       # lz (高度，默认 1.5m)
        obj_label[:, 6] = heading
        obj_label[:, 7] = cls       # FusionClassification 值
        obj_label[:, 8] = 0.0       # state (ADS 数据未提供)
        obj_label[:, 9] = vx
        obj_label[:, 10] = vy

        # 只保留有效物体
        obj_label = obj_label[valid_mask]

        return obj_label
    
    def compute_agent_bev_gt(
        self, 
        obj_label: np.ndarray,
        static_obj_feat: Optional[np.ndarray] = None,
        static_obj_mask: Optional[np.ndarray] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从 obj_label 和静态物体数据计算 agent 和 BEV GT
        
        Args:
            obj_label: shape (N, 11) - ADS 物体标签
            static_obj_feat: shape (M, P, 2) - 静态物体/车道线轮廓点 (可选)
            static_obj_mask: shape (M,) or (M, ...) - 静态物体 mask (可选)
            
        Returns:
            agent_states: shape (num_agent_queries, 5)
            agent_labels: shape (num_agent_queries,)
            bev_semantic_map: shape (bev_pixel_height, bev_pixel_width)
        """
        if self.gt_config is None:
            # Fallback: 返回空的 GT
            agent_states = torch.zeros(self.num_agent_queries, 5, dtype=torch.float32)
            agent_labels = torch.zeros(self.num_agent_queries, dtype=torch.long)
            bev_semantic_map = torch.zeros(self.bev_pixel_height, self.bev_pixel_width, dtype=torch.long)
            return agent_states, agent_labels, bev_semantic_map
        
        # Agent targets (动态物体)
        agent_states_np, agent_labels_np = compute_agent_targets_ads(obj_label, self.gt_config)
        
        # BEV semantic map (包含静态物体)
        bev_semantic_map_np = compute_bev_semantic_map_ads(
            obj_labels=obj_label,
            config=self.gt_config,
            valid_mask=None,  # obj_label 已经过滤过无效数据
            static_obj_feat=static_obj_feat,
            static_obj_mask=static_obj_mask,
            curb_feat=None,   # 用户说不考虑 curb
            curb_mask=None,
        )
        
        agent_states = torch.tensor(agent_states_np, dtype=torch.float32)
        agent_labels = torch.tensor(agent_labels_np, dtype=torch.long)
        bev_semantic_map = torch.tensor(bev_semantic_map_np, dtype=torch.long)
        
        return agent_states, agent_labels, bev_semantic_map
    
    def get_item(self, cur_idx, scene_len, scene):
        """
        覆盖父类方法，添加 Agent/BEV GT 处理
        """
        # 调用父类方法获取基础 data_dict
        data_dict = super().get_item(cur_idx, scene_len, scene)
        
        # ============ Agent/BEV GT ============
        # 从 cur_pkl_data['object_feat'] 获取动态物体数据
        # object_feat 格式: shape (N, T, 21) 或 (N, T, 22)
        # 特征顺序 (根据 ads_data.md):
        #   0: x, 1: y, 2: heading, 3: fusion_class, 4: length, 5: width,
        #   6: speed_limit, 7: vel, 8: vel_orien, 9: yaw_rate, 10: is_static, ...
        
        # 获取 pkl 数据
        cur_img_path, cur_pkl_path, cur_command_pkl_path = scene[cur_idx]
        cur_pkl_data, _ = self.get_pkl(cur_pkl_path)
        
        cur_obj_label = None
        static_obj_feat = None
        static_obj_mask = None
        
        if cur_pkl_data is not None:
            # 1. 动态物体: object_feat -> obj_label (同时使用 object_pred_mask 过滤)
            raw_object_feat = cur_pkl_data.get("object_feat", None)
            raw_object_pred_mask = cur_pkl_data.get("object_pred_mask", None)
            if raw_object_feat is not None:
                converted = self._convert_object_feat_to_obj_label(
                    object_feat=raw_object_feat,
                    object_pred_mask=raw_object_pred_mask,
                )
                if converted is not None and converted.ndim == 2 and converted.shape[1] == 11:
                    cur_obj_label = converted
            
            # 2. 静态物体 (车道线等): static_obj_feat
            raw_static_feat = cur_pkl_data.get("static_obj_feat", None)
            raw_static_mask = cur_pkl_data.get("static_obj_mask", None)
            if raw_static_feat is not None:
                if isinstance(raw_static_feat, torch.Tensor):
                    static_obj_feat = raw_static_feat.detach().cpu().numpy()
                else:
                    static_obj_feat = np.array(raw_static_feat)
            if raw_static_mask is not None:
                if isinstance(raw_static_mask, torch.Tensor):
                    static_obj_mask = raw_static_mask.detach().cpu().numpy()
                else:
                    static_obj_mask = np.array(raw_static_mask)
        
        if cur_obj_label is not None and len(cur_obj_label) > 0:
            agent_states, agent_labels, bev_semantic_map = self.compute_agent_bev_gt(
                obj_label=cur_obj_label,
                static_obj_feat=static_obj_feat,
                static_obj_mask=static_obj_mask,
            )
        else:
            # Fallback: 使用空值
            agent_states = torch.zeros(self.num_agent_queries, 5, dtype=torch.float32)
            agent_labels = torch.zeros(self.num_agent_queries, dtype=torch.long)
            bev_semantic_map = torch.zeros(self.bev_pixel_height, self.bev_pixel_width, dtype=torch.long)
        
        # Add agent/map token masks
        input_ids = data_dict["vlm_input_ids"][0]  # [L]
        
        if self.enable_agent_tokens:
            agent_mask = (input_ids == self.agent_token_id)  # [L]
            data_dict["agent_token_masks"] = agent_mask.unsqueeze(0)  # [1, L]
            data_dict["agent_states_gt"] = agent_states  # [num_queries, 5]
            data_dict["agent_labels_gt"] = agent_labels  # [num_queries]
            num_agent_found = int(agent_mask.sum().item())
            if num_agent_found != self.num_agent_queries:
                rank0_print(f"[WARN][ADSAgentBEV] agent token count mismatch: expected {self.num_agent_queries}, got {num_agent_found}")
        
        if self.enable_bev_tokens:
            map_mask = (input_ids == self.map_token_id)  # [L]
            data_dict["map_token_masks"] = map_mask.unsqueeze(0)  # [1, L]
            data_dict["bev_semantic_map_gt"] = bev_semantic_map  # [H, W]
            num_map_found = int(map_mask.sum().item())
            if num_map_found != self.num_map_queries:
                rank0_print(f"[WARN][ADSAgentBEV] map token count mismatch: expected {self.num_map_queries}, got {num_map_found}")
        
        return data_dict

    def debug_visualize_bev(
        self, 
        idx: int, 
        save_dir: str = "/tmp/bev_debug",
        frame_idx: Optional[int] = None,
        verbose: bool = True,
    ) -> str:
        """
        调试用：可视化指定索引的 BEV semantic map
        
        Args:
            idx: 数据集索引
            save_dir: 保存目录
            frame_idx: 指定帧索引 (None 则自动选择)
            verbose: 是否打印详细信息
            
        Returns:
            save_path: 保存的文件路径
            
        Usage (在调试时直接调用):
            # 方法 1: 在 dataset 上调用
            dataset = LazySupervisedADSAgentBEVDataset(...)
            dataset.debug_visualize_bev(idx=0)
            
            # 方法 2: 批量可视化
            for i in range(5):
                dataset.debug_visualize_bev(idx=i, save_dir="/tmp/bev_debug")
        """
        import os
        import cv2
        
        os.makedirs(save_dir, exist_ok=True)
        
        # 获取数据
        scene = self.data[idx]
        scene_len = len(scene)
        
        # 采样一个有效的帧
        if frame_idx is None:
            one_sec = int(round(self.frames_per_second * 1.0))
            step = max(1, int(round(self.frames_per_second / max(1, self.action_hz))))
            hist_pre = int(round(1.5 * self.action_hz))
            start_idx = one_sec + step * hist_pre
            cur_idx = min(start_idx, scene_len - 1)
        else:
            cur_idx = min(frame_idx, scene_len - 1)
        
        cur_img_path, cur_pkl_path, cur_command_pkl_path = scene[cur_idx]
        cur_pkl_data, _ = self.get_pkl(cur_pkl_path)
        
        if cur_pkl_data is None:
            raise ValueError(f"Failed to load pkl data for idx={idx}")
        
        # ========== 打印 pkl 数据结构 ==========
        if verbose:
            print(f"\n{'='*60}")
            print(f"Debug BEV: idx={idx}, frame={cur_idx}")
            print(f"{'='*60}")
            print(f"pkl path: {cur_pkl_path}")
            print(f"pkl keys ({len(cur_pkl_data)}): {list(cur_pkl_data.keys())[:15]}...")
            
            # 关键字段
            key_fields = ["object_feat", "object_pred_mask", "static_obj_feat", "static_obj_mask"]
            print("\nKey fields:")
            for key in key_fields:
                if key in cur_pkl_data:
                    val = cur_pkl_data[key]
                    if hasattr(val, 'shape'):
                        print(f"  {key}: shape={val.shape}, dtype={getattr(val, 'dtype', 'N/A')}")
                    else:
                        print(f"  {key}: type={type(val)}")
                else:
                    print(f"  {key}: NOT FOUND")
        
        # ========== 转换 object_feat -> obj_label ==========
        raw_object_feat = cur_pkl_data.get("object_feat", None)
        raw_object_pred_mask = cur_pkl_data.get("object_pred_mask", None)
        
        if raw_object_feat is None:
            raise ValueError(f"No object_feat in pkl for idx={idx}")
        
        obj_label = self._convert_object_feat_to_obj_label(
            object_feat=raw_object_feat,
            object_pred_mask=raw_object_pred_mask,
        )
        
        if obj_label is None or len(obj_label) == 0:
            print(f"[WARN] No valid objects after filtering for idx={idx}")
            obj_label = np.zeros((0, 11), dtype=np.float32)
        
        # ========== 获取静态物体 ==========
        static_obj_feat = cur_pkl_data.get("static_obj_feat", None)
        static_obj_mask = cur_pkl_data.get("static_obj_mask", None)
        
        if static_obj_feat is not None:
            if isinstance(static_obj_feat, torch.Tensor):
                static_obj_feat = static_obj_feat.detach().cpu().numpy()
            else:
                static_obj_feat = np.array(static_obj_feat)
                
        if static_obj_mask is not None:
            if isinstance(static_obj_mask, torch.Tensor):
                static_obj_mask = static_obj_mask.detach().cpu().numpy()
            else:
                static_obj_mask = np.array(static_obj_mask)
        
        # ========== 生成 BEV map ==========
        bev_map = compute_bev_semantic_map_ads(
            obj_labels=obj_label,
            config=self.gt_config,
            static_obj_feat=static_obj_feat,
            static_obj_mask=static_obj_mask,
        )
        
        # ========== 生成 Agent targets ==========
        from transfuser_gt_utils_ads import compute_agent_targets_ads
        agent_states, agent_labels = compute_agent_targets_ads(obj_label, self.gt_config)
        
        # ========== 可视化 ==========
        vis_image = visualize_bev_semantic_map(
            bev_map=bev_map,
            agent_states=agent_states,
            agent_labels=agent_labels,
            config=self.gt_config,
        )
        
        # 保存
        sample_id = f"idx{idx}_frame{cur_idx}"
        save_path = os.path.join(save_dir, f"bev_debug_{sample_id}.png")
        cv2.imwrite(save_path, vis_image)
        
        # ========== 打印统计 ==========
        if verbose:
            print(f"\nobj_label shape: {obj_label.shape}")
            print(f"static_obj_feat shape: {static_obj_feat.shape if static_obj_feat is not None else 'None'}")
            print(f"BEV map shape: {bev_map.shape}")
            
            unique, counts = np.unique(bev_map, return_counts=True)
            label_names = {0: "background", 1: "static/lane", 2: "vehicle", 3: "pedestrian"}
            print("\nBEV Label Distribution:")
            for label, count in zip(unique, counts):
                name = label_names.get(int(label), f"label_{label}")
                total = bev_map.size
                print(f"  {int(label)} ({name}): {count} ({100*count/total:.2f}%)")
            
            non_empty = (agent_labels != 0).sum()
            print(f"\nAgent detection: {non_empty}/{len(agent_labels)} non-empty")
            print(f"\nSaved to: {save_path}")
        
        return save_path
    
    def debug_visualize_batch(
        self,
        start_idx: int = 0,
        num_samples: int = 5,
        save_dir: str = "/tmp/bev_debug",
    ) -> List[str]:
        """
        批量可视化多个样本
        
        Usage:
            dataset.debug_visualize_batch(start_idx=0, num_samples=10)
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        saved_paths = []
        for i in range(num_samples):
            idx = start_idx + i
            if idx >= len(self.data):
                print(f"Index {idx} out of range, stopping.")
                break
            
            try:
                save_path = self.debug_visualize_bev(
                    idx=idx,
                    save_dir=save_dir,
                    verbose=(i == 0),  # 只在第一个样本打印详细信息
                )
                saved_paths.append(save_path)
                print(f"[{i+1}/{num_samples}] Saved: {save_path}")
            except Exception as e:
                print(f"[{i+1}/{num_samples}] Error at idx={idx}: {e}")
        
        print(f"\nDone! {len(saved_paths)} visualizations saved to: {save_dir}")
        return saved_paths


@dataclass
class DataCollatorForADSAgentBEVDataset:
    """
    Collate examples for ADS Agent/BEV supervised fine-tuning.
    """
    tokenizer: transformers.PreTrainedTokenizer
    
    def pad_and_cat(self, tensor_list):
        max_length = max(tensor.shape[-1] for tensor in tensor_list)
        padded_tensors = []
        for tensor in tensor_list:
            pad_length = max_length - tensor.shape[-1]
            padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), value=0)
            padded_tensors.append(padded_tensor)
        return torch.cat(padded_tensors, dim=1)
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # VLM inputs
        vlm_input_ids, vlm_labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("vlm_input_ids", "vlm_labels", "position_ids")
        )
        vlm_input_ids = [ids.squeeze(0) for ids in vlm_input_ids]
        vlm_labels = [ids.squeeze(0) for ids in vlm_labels]
        vlm_input_ids = torch.nn.utils.rnn.pad_sequence(
            vlm_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        vlm_labels = torch.nn.utils.rnn.pad_sequence(
            vlm_labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = self.pad_and_cat(position_ids)
        max_len = self.tokenizer.model_max_length
        vlm_input_ids = vlm_input_ids[:, :max_len]
        vlm_labels = vlm_labels[:, :max_len]
        position_ids = position_ids[:, :, :max_len]
        
        batch = dict(
            vlm_input_ids=vlm_input_ids,
            vlm_labels=vlm_labels,
            vlm_attention_mask=vlm_input_ids.ne(self.tokenizer.pad_token_id),
            position_ids=position_ids,
        )
        
        # Images
        images = [instance["pixel_values"] for instance in instances if "pixel_values" in instance]
        if images:
            batch["pixel_values"] = torch.cat(images, dim=0)
            grid_thw = [instance["image_grid_thw"] for instance in instances if "image_grid_thw" in instance]
            batch["image_grid_thw"] = torch.cat(grid_thw, dim=0) if grid_thw else None
        
        # Raw VAE
        raw_pixel_values_vae = [instance["raw_pixel_values_vae"] for instance in instances if "raw_pixel_values_vae" in instance]
        frame_image_counts = [instance["frame_image_counts"] for instance in instances if "frame_image_counts" in instance]
        if raw_pixel_values_vae:
            batch["raw_pixel_values_vae"] = torch.cat(raw_pixel_values_vae, dim=1)
        if frame_image_counts:
            batch["frame_image_counts"] = torch.stack(frame_image_counts, dim=0)
        
        # Agent/Map masks - pad to same length
        max_vlm_len = vlm_input_ids.shape[1]
        
        agent_masks = [instance.get("agent_token_masks") for instance in instances]
        map_masks = [instance.get("map_token_masks") for instance in instances]
        
        if agent_masks[0] is not None:
            padded_agent_masks = []
            for mask in agent_masks:
                L = mask.shape[-1]
                if L < max_vlm_len:
                    mask = torch.nn.functional.pad(mask, (0, max_vlm_len - L), value=False)
                mask = mask[:, :max_vlm_len]
                padded_agent_masks.append(mask.squeeze(0))
            batch["agent_token_masks"] = torch.stack(padded_agent_masks, dim=0)
        
        if map_masks[0] is not None:
            padded_map_masks = []
            for mask in map_masks:
                L = mask.shape[-1]
                if L < max_vlm_len:
                    mask = torch.nn.functional.pad(mask, (0, max_vlm_len - L), value=False)
                mask = mask[:, :max_vlm_len]
                padded_map_masks.append(mask.squeeze(0))
            batch["map_token_masks"] = torch.stack(padded_map_masks, dim=0)
        
        # Agent/BEV GT
        agent_states_gt = [instance.get("agent_states_gt") for instance in instances]
        agent_labels_gt = [instance.get("agent_labels_gt") for instance in instances]
        bev_semantic_map_gt = [instance.get("bev_semantic_map_gt") for instance in instances]
        
        if agent_states_gt[0] is not None:
            batch["agent_states_gt"] = torch.stack(agent_states_gt, dim=0)
        if agent_labels_gt[0] is not None:
            batch["agent_labels_gt"] = torch.stack(agent_labels_gt, dim=0)
        if bev_semantic_map_gt[0] is not None:
            batch["bev_semantic_map_gt"] = torch.stack(bev_semantic_map_gt, dim=0)
        
        # Action expert inputs
        input_ids_stack = [instance["input_ids"] for instance in instances]
        labels_stack = [instance["labels"] for instance in instances]
        pre_action_stack = [instance["pre_action"] for instance in instances]
        action_stack = [instance["action"] for instance in instances]
        
        batch["input_ids"] = torch.stack(input_ids_stack, dim=0)
        batch["labels"] = torch.stack(labels_stack, dim=0)
        batch["pre_action"] = torch.stack(pre_action_stack, dim=0)
        batch["action"] = torch.stack(action_stack, dim=0)
        
        # Image/action masks for ROSS
        image_token_masks = [instance.get("image_token_masks") for instance in instances if instance.get("image_token_masks") is not None]
        action_future_masks = [instance.get("action_future_masks") for instance in instances if instance.get("action_future_masks") is not None]
        
        if image_token_masks and action_future_masks:
            max_len = max(max(m.shape[-1] for m in image_token_masks), max(m.shape[-1] for m in action_future_masks))
            padded_img = []
            padded_act = []
            for img_m, act_m in zip(image_token_masks, action_future_masks):
                img_m = torch.nn.functional.pad(img_m, (0, max_len - img_m.shape[-1]), value=0)
                act_m = torch.nn.functional.pad(act_m, (0, max_len - act_m.shape[-1]), value=0)
                padded_img.append(img_m)
                padded_act.append(act_m)
            batch["image_token_masks"] = torch.cat(padded_img, dim=0)
            batch["action_future_masks"] = torch.cat(padded_act, dim=0)
        
        return batch


def make_supervised_data_module_ads_vla_agent_bev(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args
) -> Dict:
    """
    Make ADS VLA dataset with Agent Detection and BEV Semantic Map support.
    """
    train_dataset = LazySupervisedADSAgentBEVDataset(tokenizer=tokenizer, data_args=data_args)
    data_collator = DataCollatorForADSAgentBEVDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

