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

class AgentBEVMixin:
    """
    Agent/BEV 功能的 Mixin 类
    
    用于将 Agent Detection 和 BEV Semantic Map 功能集成到 MOE 模型中。
    
    关键设计：
    - Agent/BEV tokens 在 VLM 主干上处理（感知任务）
    - 从 current_vlm_h 中提取 agent/map tokens 的 hidden states
    - 与 Action Expert 分开处理
    
    使用方式：
    1. MOE 模型继承此 Mixin
    2. 在 __init__ 中调用 _init_agent_bev_heads()
    3. 在 forward 中调用 _compute_agent_bev_outputs()
    """
    
    def _init_agent_bev_heads(
        self,
        hidden_size: int,
        num_agent_queries: int = 20,
        num_map_queries: int = 64,
        num_agent_classes: int = 4,
        num_bev_classes: int = 4,
        agent_d_ffn: int = 256,
        bev_output_size: Tuple[int, int] = (128, 256),
        enable_agent_loss: bool = True,
        enable_bev_loss: bool = True,
    ):
        """
        初始化 Agent 和 BEV prediction heads
        
        Args:
            hidden_size: VLM 隐藏层维度 (应与 qwen_ross.config.hidden_size 一致)
        """
        # 保存配置
        self.num_agent_queries = num_agent_queries
        self.num_map_queries = num_map_queries
        self.num_agent_classes = num_agent_classes
        self.num_bev_classes = num_bev_classes
        self.enable_agent_loss = enable_agent_loss
        self.enable_bev_loss = enable_bev_loss
        
        # Prediction heads
        self.agent_head = AgentHead(
            num_agents=num_agent_queries,
            d_ffn=agent_d_ffn,
            d_model=hidden_size,
            num_classes=num_agent_classes,
        )
        
        self.bev_head = BEVSemanticHead(
            d_model=hidden_size,
            num_map_queries=num_map_queries,
            num_classes=num_bev_classes,
            output_size=bev_output_size,
        )
        
        # BEV 类别权重 (针对 ADS 数据调整)
        self.register_buffer(
            "bev_class_weights",
            torch.tensor([0.1, 1.0, 1.0, 2.0], dtype=torch.float32)
        )
    
    def _compute_agent_bev_outputs(
        self,
        vlm_hidden_states: torch.Tensor,
        agent_token_masks: Optional[torch.Tensor],
        map_token_masks: Optional[torch.Tensor],
        agent_states_gt: Optional[torch.Tensor] = None,
        agent_labels_gt: Optional[torch.Tensor] = None,
        bev_semantic_map_gt: Optional[torch.Tensor] = None,
        agent_class_weight: float = 1.0,
        agent_box_weight: float = 1.0,
        bev_weight: float = 1.0,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], dict, dict, dict]:
        """
        从 VLM hidden states 中提取 Agent/BEV 特征并计算预测和损失
        
        关键：此函数处理的是 current_vlm_h（VLM 主干输出），
        而不是 current_action_h（Action Expert 输出）
        
        Args:
            vlm_hidden_states: [B, L_vlm, C] VLM 主干的输出 (current_vlm_h after norm)
            agent_token_masks: [B, L_vlm] boolean mask，指示哪些位置是 <agent> tokens
            map_token_masks: [B, L_vlm] boolean mask，指示哪些位置是 <map> tokens
            agent_states_gt: [B, num_agents, 7] Agent GT states
            agent_labels_gt: [B, num_agents] Agent GT labels
            bev_semantic_map_gt: [B, H, W] BEV GT semantic map
            
        Returns:
            agent_loss: Agent detection loss (or None)
            bev_loss: BEV segmentation loss (or None)
            agent_pred: Agent predictions dict
            bev_pred: BEV predictions tensor [B, C, H, W]
            logs: Loss 日志
        """
        batch_size = vlm_hidden_states.shape[0]
        hidden_size = vlm_hidden_states.shape[-1]
        device = vlm_hidden_states.device
        
        agent_loss = None
        bev_loss = None
        agent_pred = {}
        bev_pred = None
        logs = {}
        
        # ===== Agent Detection =====
        # Agent tokens 在 vlm_input_ids 中，经过 shared_layers 处理后
        # 需要从 vlm_hidden_states 中提取对应位置的 hidden states
        if agent_token_masks is not None and self.enable_agent_loss:
            # 确保 mask 维度正确
            if agent_token_masks.dim() == 2:
                # [B, L] -> 提取每个样本的 agent tokens
                agent_hidden_list = []
                for b in range(batch_size):
                    mask_b = agent_token_masks[b]  # [L]
                    if mask_b.any():
                        # 提取该样本的 agent hidden states
                        agent_h = vlm_hidden_states[b][mask_b]  # [num_agent_tokens, C]
                        # 截断或填充到 num_agent_queries
                        if agent_h.shape[0] > self.num_agent_queries:
                            agent_h = agent_h[:self.num_agent_queries]
                        elif agent_h.shape[0] < self.num_agent_queries:
                            pad_len = self.num_agent_queries - agent_h.shape[0]
                            agent_h = torch.cat([
                                agent_h,
                                torch.zeros(pad_len, hidden_size, device=device, dtype=agent_h.dtype)
                            ], dim=0)
                        agent_hidden_list.append(agent_h)
                    else:
                        # 无 agent tokens，填零
                        agent_hidden_list.append(
                            torch.zeros(self.num_agent_queries, hidden_size, device=device, dtype=vlm_hidden_states.dtype)
                        )
                
                agent_hidden = torch.stack(agent_hidden_list, dim=0)  # [B, num_agents, C]
            else:
                # 假设已经是 [B, num_agents, C]
                agent_hidden = vlm_hidden_states
            
            # Agent head 预测
            agent_pred = self.agent_head(agent_hidden)
            
            # 计算 loss
            if agent_states_gt is not None and agent_labels_gt is not None:
                ce_loss, l1_loss = compute_agent_loss(
                    predictions=agent_pred,
                    targets={"agent_states": agent_states_gt, "agent_labels": agent_labels_gt},
                    config=self,
                )
                agent_loss = agent_class_weight * ce_loss + agent_box_weight * l1_loss
                logs["agent_ce_loss"] = float(ce_loss.detach().cpu())
                logs["agent_l1_loss"] = float(l1_loss.detach().cpu())
                logs["agent_loss"] = float(agent_loss.detach().cpu())
        
        # ===== BEV Semantic Map =====
        if map_token_masks is not None and self.enable_bev_loss:
            # 与 Agent 类似，提取 map tokens 的 hidden states
            if map_token_masks.dim() == 2:
                map_hidden_list = []
                for b in range(batch_size):
                    mask_b = map_token_masks[b]  # [L]
                    if mask_b.any():
                        map_h = vlm_hidden_states[b][mask_b]  # [num_map_tokens, C]
                        if map_h.shape[0] > self.num_map_queries:
                            map_h = map_h[:self.num_map_queries]
                        elif map_h.shape[0] < self.num_map_queries:
                            pad_len = self.num_map_queries - map_h.shape[0]
                            map_h = torch.cat([
                                map_h,
                                torch.zeros(pad_len, hidden_size, device=device, dtype=map_h.dtype)
                            ], dim=0)
                        map_hidden_list.append(map_h)
                    else:
                        map_hidden_list.append(
                            torch.zeros(self.num_map_queries, hidden_size, device=device, dtype=vlm_hidden_states.dtype)
                        )
                
                map_hidden = torch.stack(map_hidden_list, dim=0)  # [B, num_map_queries, C]
            else:
                map_hidden = vlm_hidden_states
            
            # BEV head 预测
            bev_pred = self.bev_head(map_hidden)  # [B, num_classes, H, W]
            
            # 计算 loss
            if bev_semantic_map_gt is not None:
                weight = self.bev_class_weights.to(device)
                bev_loss = bev_weight * F.cross_entropy(bev_pred, bev_semantic_map_gt.long(), weight=weight)
                logs["bev_loss"] = float(bev_loss.detach().cpu())
        
        return agent_loss, bev_loss, agent_pred, bev_pred, logs


# ============ 使用示例：集成到 MOE 模型 ============
"""
在 qwen_moe.py 中集成 Agent/BEV 的方式：

1. 修改 Qwen2_5_VLForConditionalGenerationROSS_MOE 继承 AgentBEVMixin：

    class Qwen2_5_VLForConditionalGenerationROSS_MOE_AgentBEV(
        Qwen2_5_VLForConditionalGenerationROSS_MOE, 
        AgentBEVMixin
    ):
        def __init__(self, config, action_config, ckpt_path=None, data_type=torch.float32):
            super().__init__(config, action_config, ckpt_path, data_type)
            
            # 初始化 Agent/BEV heads
            self._init_agent_bev_heads(
                hidden_size=self.qwen_ross.config.hidden_size,  # VLM hidden size
                num_agent_queries=20,
                num_map_queries=64,
                num_agent_classes=4,
                num_bev_classes=4,
            )

2. 在 forward 中添加 Agent/BEV 处理（在 shared_layers 处理完成后）：

    def forward(self, ..., agent_token_masks=None, map_token_masks=None, 
                agent_states_gt=None, agent_labels_gt=None, bev_semantic_map_gt=None):
        ...
        # 经过所有 shared_layers 处理
        for layer_idx in range(num_layers):
            current_vlm_h, current_action_h = shared_layer(...)
        
        # VLM 输出 norm
        vlm_final = self.qwen_ross.model.language_model.norm(current_vlm_h)
        
        # ===== 添加 Agent/BEV 处理 =====
        agent_loss, bev_loss, agent_pred, bev_pred, logs = self._compute_agent_bev_outputs(
            vlm_hidden_states=vlm_final,  # 使用 VLM 主干输出
            agent_token_masks=agent_token_masks,
            map_token_masks=map_token_masks,
            agent_states_gt=agent_states_gt,
            agent_labels_gt=agent_labels_gt,
            bev_semantic_map_gt=bev_semantic_map_gt,
        )
        
        # 将 Agent/BEV loss 加入总 loss
        if agent_loss is not None:
            loss = loss + agent_loss
        if bev_loss is not None:
            loss = loss + bev_loss
        
        # Action Expert 继续处理动作预测
        h_final = self.action_expert.norm(current_action_h)
        logits = self.action_lm_head(h_final)
        ...

3. 数据准备：
   - agent_token_masks: [B, L_vlm] 指示 vlm_input_ids 中哪些是 <agent> tokens
   - map_token_masks: [B, L_vlm] 指示 vlm_input_ids 中哪些是 <map> tokens
   - 这些 mask 由 data_qwen_vla_agent_bev_ads.py 的 DataCollator 提供
"""


# ============ 导出 ============

__all__ = [
    "Qwen2_5_VLConfigAgentBEV_ADS",
    "Qwen2_5_VLForConditionalGenerationAgentBEV_ADS",
    "AgentBEVMixin",
]

