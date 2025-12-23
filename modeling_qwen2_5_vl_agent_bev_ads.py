"""
ADS 专用的 Qwen2.5-VL Agent Detection 和 BEV Semantic Map 模型

与 NavSim 版本的区别：
1. BEV 类别数为 4 (无 centerline): background, static, vehicle, pedestrian
2. Agent 类别数为 4: empty, vehicle, pedestrian, other

基于 modeling_qwen2_5_vl_agent_bev.py 修改
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Any
from dataclasses import dataclass

from transformers.utils import ModelOutput

# 从原版导入基础类
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reference.transformers.src.transformers.models.qwen2_5_vl.modeling_qwen2_5_vl_agent_bev import (
    Qwen2_5_VLConfigAgentBEV,
    Qwen2_5_VLAgentBEVOutput,
    AgentHead,
    BEVSemanticHead,
    compute_agent_loss,
    Qwen2_5_VLForConditionalGenerationAgentBEV,
)


# ============ ADS 专用配置 ============

class Qwen2_5_VLConfigAgentBEV_ADS(Qwen2_5_VLConfigAgentBEV):
    """
    ADS 专用配置
    
    主要区别：
    - num_bev_classes = 4 (无 centerline)
    - BEV 类别: 0=background, 1=static, 2=vehicle, 3=pedestrian
    """
    
    model_type = "qwen2_5_vl_agent_bev_ads"
    
    def __init__(
        self,
        # Agent 配置
        num_agent_queries: int = 20,
        num_agent_classes: int = 4,  # empty, vehicle, pedestrian, other
        agent_d_ffn: int = 256,
        
        # BEV 配置 (ADS 专用)
        num_map_queries: int = 64,
        num_bev_classes: int = 4,  # background, static, vehicle, pedestrian
        bev_output_size: Tuple[int, int] = (128, 256),
        
        # Loss 权重
        agent_class_weight: float = 1.0,
        agent_box_weight: float = 1.0,
        bev_weight: float = 1.0,
        
        # 特殊 token ID
        agent_token_id: Optional[int] = None,
        map_token_id: Optional[int] = None,
        
        # Loss 开关
        enable_agent_loss: bool = True,
        enable_bev_loss: bool = True,
        
        **kwargs
    ):
        # 调用父类初始化，但覆盖 BEV 类别数
        super().__init__(
            num_agent_queries=num_agent_queries,
            num_agent_classes=num_agent_classes,
            agent_d_ffn=agent_d_ffn,
            num_map_queries=num_map_queries,
            num_bev_classes=num_bev_classes,  # ADS: 4 类
            bev_output_size=bev_output_size,
            agent_class_weight=agent_class_weight,
            agent_box_weight=agent_box_weight,
            bev_weight=bev_weight,
            agent_token_id=agent_token_id,
            map_token_id=map_token_id,
            enable_agent_loss=enable_agent_loss,
            enable_bev_loss=enable_bev_loss,
            **kwargs
        )
        
        # ADS 专用属性
        self.ads_bev_class_names = {
            0: "background",
            1: "static_objects",
            2: "vehicles",
            3: "pedestrians",
        }


# ============ ADS 专用模型 ============

class Qwen2_5_VLForConditionalGenerationAgentBEV_ADS(Qwen2_5_VLForConditionalGenerationAgentBEV):
    """
    ADS 专用的 Agent Detection + BEV Semantic Map 模型
    
    继承自 NavSim 版本，主要区别：
    - 使用 4 类 BEV (无 centerline)
    - 类别权重可能需要调整
    """
    
    config_class = Qwen2_5_VLConfigAgentBEV_ADS
    
    def __init__(self, config: Qwen2_5_VLConfigAgentBEV_ADS):
        super().__init__(config)
        
        # BEV 类别权重 (针对 ADS 数据调整)
        # 由于 ADS 没有 centerline，类别分布可能不同
        self.register_buffer(
            "bev_class_weights",
            torch.tensor([0.1, 1.0, 1.0, 2.0], dtype=torch.float32)  # 行人较少，给更大权重
        )
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        # Agent/BEV specific inputs
        agent_token_masks: Optional[torch.Tensor] = None,
        map_token_masks: Optional[torch.Tensor] = None,
        agent_states_gt: Optional[torch.Tensor] = None,
        agent_labels_gt: Optional[torch.Tensor] = None,
        bev_semantic_map_gt: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Qwen2_5_VLAgentBEVOutput:
        """
        Forward pass (与父类相同，但 BEV loss 使用加权交叉熵)
        """
        # 调用父类 forward 获取基础输出
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
            second_per_grid_ts=second_per_grid_ts,
            logits_to_keep=logits_to_keep,
            agent_token_masks=agent_token_masks,
            map_token_masks=map_token_masks,
            agent_states_gt=agent_states_gt,
            agent_labels_gt=agent_labels_gt,
            bev_semantic_map_gt=bev_semantic_map_gt,
            **kwargs,
        )
        
        # 如果需要，可以在这里添加 ADS 特定的后处理
        # 例如：使用加权 BEV loss
        
        return outputs
    
    def compute_weighted_bev_loss(
        self,
        bev_pred: torch.Tensor,
        bev_gt: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算加权 BEV 分割 loss
        
        Args:
            bev_pred: [B, C, H, W] 预测
            bev_gt: [B, H, W] GT
            
        Returns:
            loss: 加权交叉熵损失
        """
        weight = self.bev_class_weights.to(bev_pred.device)
        loss = F.cross_entropy(bev_pred, bev_gt.long(), weight=weight)
        return loss


# ============ 与 MOE 模型集成 ============

class Qwen2_5_VLAgentBEV_MOE_ADS:
    """
    将 Agent/BEV 功能集成到 MOE 模型中
    
    使用方式：
    1. 在 MOE 模型初始化时创建 Agent/BEV heads
    2. 在 forward 中处理 agent/map tokens
    3. 计算额外的 loss
    """
    
    @staticmethod
    def add_agent_bev_heads(
        model,
        hidden_size: int,
        num_agent_queries: int = 20,
        num_map_queries: int = 64,
        num_agent_classes: int = 4,
        num_bev_classes: int = 4,
        agent_d_ffn: int = 256,
        bev_output_size: Tuple[int, int] = (128, 256),
    ):
        """
        向现有模型添加 Agent 和 BEV heads
        
        Args:
            model: 现有的 MOE 模型
            hidden_size: 模型隐藏层维度
        """
        # Agent query embeddings
        model.agent_query_embedding = nn.Embedding(num_agent_queries, hidden_size)
        model.map_query_embedding = nn.Embedding(num_map_queries, hidden_size)
        
        # Prediction heads
        model.agent_head = AgentHead(
            num_agents=num_agent_queries,
            d_ffn=agent_d_ffn,
            d_model=hidden_size,
            num_classes=num_agent_classes,
        )
        
        model.bev_head = BEVSemanticHead(
            d_model=hidden_size,
            num_map_queries=num_map_queries,
            num_classes=num_bev_classes,
            output_size=bev_output_size,
        )
        
        # 保存配置
        model.num_agent_queries = num_agent_queries
        model.num_map_queries = num_map_queries
        model.num_agent_classes = num_agent_classes
        model.num_bev_classes = num_bev_classes
        
        return model
    
    @staticmethod
    def process_agent_bev_in_forward(
        model,
        hidden_states: torch.Tensor,
        agent_token_masks: Optional[torch.Tensor],
        map_token_masks: Optional[torch.Tensor],
        agent_states_gt: Optional[torch.Tensor],
        agent_labels_gt: Optional[torch.Tensor],
        bev_semantic_map_gt: Optional[torch.Tensor],
        enable_agent_loss: bool = True,
        enable_bev_loss: bool = True,
        agent_class_weight: float = 1.0,
        agent_box_weight: float = 1.0,
        bev_weight: float = 1.0,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], dict, dict]:
        """
        在 forward 中处理 Agent/BEV 预测
        
        Args:
            model: 包含 agent/bev heads 的模型
            hidden_states: [B, L, C] transformer 输出
            ...
            
        Returns:
            agent_loss: Agent detection loss
            bev_loss: BEV segmentation loss
            agent_pred: Agent predictions
            bev_pred: BEV predictions
        """
        batch_size = hidden_states.shape[0]
        agent_loss = None
        bev_loss = None
        agent_pred = None
        bev_pred = None
        logs = {}
        
        # Agent detection
        if agent_token_masks is not None and agent_token_masks.any():
            # Extract agent features
            agent_hidden = hidden_states[agent_token_masks]
            agent_hidden = agent_hidden.view(batch_size, model.num_agent_queries, -1)
            
            # Predict
            agent_pred = model.agent_head(agent_hidden)
            
            # Compute loss
            if agent_states_gt is not None and agent_labels_gt is not None and enable_agent_loss:
                ce_loss, l1_loss = compute_agent_loss(
                    predictions=agent_pred,
                    targets={"agent_states": agent_states_gt, "agent_labels": agent_labels_gt},
                    config=model,
                )
                agent_loss = agent_class_weight * ce_loss + agent_box_weight * l1_loss
                logs["agent_ce_loss"] = float(ce_loss.detach().cpu())
                logs["agent_l1_loss"] = float(l1_loss.detach().cpu())
        
        # BEV semantic map
        if map_token_masks is not None and map_token_masks.any():
            # Extract map features
            map_hidden = hidden_states[map_token_masks]
            map_hidden = map_hidden.view(batch_size, model.num_map_queries, -1)
            
            # Predict
            bev_pred = model.bev_head(map_hidden)
            
            # Compute loss
            if bev_semantic_map_gt is not None and enable_bev_loss:
                bev_loss = bev_weight * F.cross_entropy(bev_pred, bev_semantic_map_gt.long())
                logs["bev_loss"] = float(bev_loss.detach().cpu())
        
        return agent_loss, bev_loss, agent_pred, bev_pred, logs


# ============ 导出 ============

__all__ = [
    "Qwen2_5_VLConfigAgentBEV_ADS",
    "Qwen2_5_VLForConditionalGenerationAgentBEV_ADS",
    "Qwen2_5_VLAgentBEV_MOE_ADS",
]

