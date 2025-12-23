"""
Qwen2.5-VL with Agent Detection and BEV Semantic Map support.

This module extends the base Qwen2.5-VL model to support:
- Agent Detection via learnable <agent> tokens
- BEV Semantic Map prediction via learnable <map> tokens

The embedding replacement mechanism follows the same pattern as image token replacement
in the original Qwen2.5-VL model (using masked_scatter).
"""

import math
from typing import Optional, Union, Dict, Any, Tuple, List
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLCausalLMOutputWithPast,
)
from .configuration_qwen2_5_vl import Qwen2_5_VLConfig
from ...utils import ModelOutput


# ============ Data Classes ============

@dataclass
class Qwen2_5_VLAgentBEVOutput(ModelOutput):
    """Output class for Agent/BEV model."""
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    # Agent/BEV specific
    agent_states: Optional[torch.FloatTensor] = None
    agent_labels: Optional[torch.FloatTensor] = None
    bev_semantic_map: Optional[torch.FloatTensor] = None
    # Loss components for logging
    lm_loss: Optional[torch.FloatTensor] = None
    agent_loss: Optional[torch.FloatTensor] = None
    bev_loss: Optional[torch.FloatTensor] = None


# ============ Config ============

class Qwen2_5_VLConfigAgentBEV(Qwen2_5_VLConfig):
    """Extended config with Agent/BEV support."""
    
    def __init__(
        self,
        # Agent/Map token IDs (set by tokenizer)
        agent_token_id: int = None,
        map_token_id: int = None,
        # Query counts
        num_agent_queries: int = 20,
        num_map_queries: int = 64,
        # Agent head config
        agent_d_ffn: int = 1024,
        num_agent_classes: int = 4,  # empty, vehicle, pedestrian, other
        # BEV head config
        num_bev_classes: int = 7,
        bev_output_size: Tuple[int, int] = (128, 256),
        # Loss weights
        agent_class_weight: float = 10.0,
        agent_box_weight: float = 1.0,
        bev_weight: float = 10.0,
        # Enable flags
        enable_agent_loss: bool = True,
        enable_bev_loss: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.agent_token_id = agent_token_id
        self.map_token_id = map_token_id
        self.num_agent_queries = num_agent_queries
        self.num_map_queries = num_map_queries
        self.agent_d_ffn = agent_d_ffn
        self.num_agent_classes = num_agent_classes
        self.num_bev_classes = num_bev_classes
        self.bev_output_size = bev_output_size
        self.agent_class_weight = agent_class_weight
        self.agent_box_weight = agent_box_weight
        self.bev_weight = bev_weight
        self.enable_agent_loss = enable_agent_loss
        self.enable_bev_loss = enable_bev_loss


# ============ Prediction Heads ============

class BoundingBox2DIndex:
    """Index for 2D bounding box."""
    X = 0
    Y = 1
    HEADING = 2
    LENGTH = 3
    WIDTH = 4
    
    @classmethod
    def size(cls):
        return 5
    
    @classmethod
    def point_slice(cls):
        return slice(cls.X, cls.Y + 1)


class AgentHead(nn.Module):
    """Agent detection head for bounding box prediction."""
    
    def __init__(
        self,
        num_agents: int,
        d_ffn: int,
        d_model: int,
        num_classes: int = 4,
    ):
        super().__init__()
        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn
        self._num_classes = num_classes
        
        # State prediction: [x, y, heading, length, width]
        self._mlp_states = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, BoundingBox2DIndex.size()),
        )
        
        # Class prediction: num_classes logits
        self._mlp_label = nn.Linear(d_model, num_classes)
    
    def forward(self, agent_queries: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            agent_queries: [B, num_agents, d_model]
        Returns:
            agent_states: [B, num_agents, 5]
            agent_labels: [B, num_agents, num_classes]
        """
        raw_states = self._mlp_states(agent_queries)
        
        # Position range limit (±32m for x, y)
        xy = raw_states[..., :2].tanh() * 32
        # Heading range limit (±π)
        theta = raw_states[..., 2:3].tanh() * math.pi
        # Size positive constraint
        size = F.softplus(raw_states[..., 3:5])
        
        agent_states = torch.cat([xy, theta, size], dim=-1)
        agent_labels = self._mlp_label(agent_queries)
        
        return {"agent_states": agent_states, "agent_labels": agent_labels}


class BEVSemanticHead(nn.Module):
    """BEV semantic segmentation head."""
    
    def __init__(
        self, 
        d_model: int, 
        num_map_queries: int,
        num_classes: int = 7,
        output_size: Tuple[int, int] = (128, 256),
    ):
        super().__init__()
        self.d_model = d_model
        self.num_map_queries = num_map_queries
        self.num_classes = num_classes
        self.output_size = output_size
        
        # Calculate intermediate feature size
        # We'll reshape map_queries into a 2D feature map
        # num_map_queries should be h*w where h and w are intermediate sizes
        self.feat_h = 8
        self.feat_w = 8
        assert num_map_queries == self.feat_h * self.feat_w, \
            f"num_map_queries ({num_map_queries}) must equal feat_h*feat_w ({self.feat_h*self.feat_w})"
        
        # Project to feature channels
        self.proj = nn.Linear(d_model, 256)
        
        # Upsample and predict
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1),
            nn.Upsample(size=output_size, mode='bilinear', align_corners=False),
        )
    
    def forward(self, map_queries: torch.Tensor) -> torch.Tensor:
        """
        Args:
            map_queries: [B, num_map_queries, d_model]
        Returns:
            bev_semantic_map: [B, num_classes, H, W]
        """
        B = map_queries.shape[0]
        
        # Project and reshape to 2D
        x = self.proj(map_queries)  # [B, num_map_queries, 256]
        x = x.view(B, self.feat_h, self.feat_w, -1)  # [B, h, w, 256]
        x = x.permute(0, 3, 1, 2)  # [B, 256, h, w]
        
        # Decode to full resolution
        bev_map = self.decoder(x)  # [B, num_classes, H, W]
        
        return bev_map


# ============ Loss Functions ============

def focal_loss(
    pred_logits: torch.Tensor,
    gt_labels: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """Multi-class Focal Loss."""
    ce_loss = F.cross_entropy(pred_logits, gt_labels, reduction='none')
    pred_probs = pred_logits.softmax(dim=-1)
    target_probs = pred_probs.gather(dim=1, index=gt_labels.unsqueeze(1)).squeeze(1)
    
    focal_weight = alpha * (1 - target_probs) ** gamma
    loss = focal_weight * ce_loss
    
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


@torch.no_grad()
def _get_focal_cost(gt_labels: torch.Tensor, pred_logits: torch.Tensor) -> torch.Tensor:
    """Calculate Focal loss cost matrix for Hungarian matching."""
    B, N_pred, num_classes = pred_logits.shape
    N_gt = gt_labels.shape[1]
    
    pred_probs = pred_logits.softmax(dim=-1)  # [B, N_pred, num_classes]
    
    # Expand for broadcasting
    gt_labels_expanded = gt_labels[:, None, :].expand(B, N_pred, N_gt)  # [B, N_pred, N_gt]
    pred_probs_expanded = pred_probs[:, :, None, :].expand(B, N_pred, N_gt, num_classes)
    
    # Gather target class probabilities
    target_probs = torch.gather(
        pred_probs_expanded, 
        dim=3, 
        index=gt_labels_expanded.unsqueeze(-1)
    ).squeeze(-1)  # [B, N_pred, N_gt]
    
    # Focal loss cost
    focal_cost = -0.25 * (1 - target_probs) ** 2.0 * torch.log(target_probs.clamp(min=1e-8))
    
    return focal_cost


@torch.no_grad()
def _get_l1_cost(
    gt_states: torch.Tensor,
    pred_states: torch.Tensor,
    gt_valid: torch.Tensor
) -> torch.Tensor:
    """Calculate L1 cost matrix for Hungarian matching."""
    gt_xy = gt_states[:, :, None, :2].detach()
    pred_xy = pred_states[:, None, :, :2].detach()
    l1_cost = (gt_xy - pred_xy).abs().sum(dim=-1)  # [B, N_gt, N_pred]
    
    l1_cost = gt_valid[..., None].float() * l1_cost
    l1_cost = l1_cost.permute(0, 2, 1)  # [B, N_pred, N_gt]
    
    return l1_cost


def _get_src_permutation_idx(indices):
    """Get batch and source indices from matching results."""
    batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx


def compute_agent_loss(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    config: Qwen2_5_VLConfigAgentBEV,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Hungarian matching loss for agent detection.
    
    Args:
        predictions: {"agent_states": [B, N_pred, 5], "agent_labels": [B, N_pred, num_classes]}
        targets: {"agent_states": [B, N_gt, 5], "agent_labels": [B, N_gt]}
        config: model config
    
    Returns:
        ce_loss: classification loss
        l1_loss: box regression loss
    """
    pred_states = predictions["agent_states"]
    pred_logits = predictions["agent_labels"]
    gt_states = targets["agent_states"]
    gt_labels = targets["agent_labels"]
    
    B, N_pred = pred_states.shape[:2]
    
    # Valid mask: non-empty agents (label > 0)
    gt_valid = gt_labels > 0
    num_gt = gt_valid.sum()
    num_gt = num_gt if num_gt > 0 else num_gt + 1
    
    # Cost matrices
    focal_cost = _get_focal_cost(gt_labels, pred_logits)
    l1_cost = _get_l1_cost(gt_states, pred_states, gt_valid)
    
    cost = config.agent_class_weight * focal_cost + config.agent_box_weight * l1_cost
    cost = cost.cpu()
    
    # Hungarian matching
    indices = [linear_sum_assignment(c) for c in cost]
    matching = [
        (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
        for i, j in indices
    ]
    idx = _get_src_permutation_idx(matching)
    
    # Get matched predictions and GT
    pred_states_idx = pred_states[idx]
    gt_states_idx = torch.cat([t[i] for t, (_, i) in zip(gt_states, indices)], dim=0)
    
    pred_logits_idx = pred_logits[idx]
    gt_labels_idx = torch.cat([t[i] for t, (_, i) in zip(gt_labels, indices)], dim=0).long()
    
    # L1 loss (only for valid agents)
    valid_mask = (gt_labels_idx > 0).float()
    
    # Position loss
    l1_pos = F.l1_loss(pred_states_idx[..., :2], gt_states_idx[..., :2], reduction="none").sum(-1)
    
    # Heading loss with angle wrap
    pred_theta = pred_states_idx[..., 2]
    gt_theta = gt_states_idx[..., 2]
    dtheta = torch.atan2(torch.sin(pred_theta - gt_theta), torch.cos(pred_theta - gt_theta)).abs()
    
    # Size loss
    l1_size = F.l1_loss(pred_states_idx[..., 3:5], gt_states_idx[..., 3:5], reduction="none").sum(-1)
    
    l1_total = l1_pos + 10.0 * dtheta + l1_size
    l1_loss = (l1_total * valid_mask).sum() / num_gt
    
    # Focal loss for classification
    ce_loss = focal_loss(pred_logits_idx, gt_labels_idx, reduction='mean')
    
    return ce_loss, l1_loss


# ============ Main Model ============

class Qwen2_5_VLForConditionalGenerationAgentBEV(Qwen2_5_VLForConditionalGeneration):
    """
    Qwen2.5-VL model with Agent Detection and BEV Semantic Map support.
    
    This model extends the base Qwen2.5-VL to:
    1. Replace <agent> tokens with learnable query embeddings
    2. Replace <map> tokens with learnable query embeddings
    3. Extract features at these positions from hidden states
    4. Predict agent states/labels and BEV semantic map
    5. Compute auxiliary losses
    """
    
    config_class = Qwen2_5_VLConfigAgentBEV
    
    def __init__(self, config: Qwen2_5_VLConfigAgentBEV):
        super().__init__(config)
        
        # Get hidden size from text config
        hidden_size = config.text_config.hidden_size
        
        # Learnable query embeddings
        self.agent_query_embedding = nn.Embedding(
            config.num_agent_queries, hidden_size
        )
        self.map_query_embedding = nn.Embedding(
            config.num_map_queries, hidden_size
        )
        
        # Prediction heads
        self.agent_head = AgentHead(
            num_agents=config.num_agent_queries,
            d_ffn=config.agent_d_ffn,
            d_model=hidden_size,
            num_classes=config.num_agent_classes,
        )
        
        self.bev_head = BEVSemanticHead(
            d_model=hidden_size,
            num_map_queries=config.num_map_queries,
            num_classes=config.num_bev_classes,
            output_size=config.bev_output_size,
        )
        
        # Loss logging
        self._last_logs = {}
        
        self.post_init()
    
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        # Agent/BEV specific inputs
        agent_token_masks: Optional[torch.Tensor] = None,
        map_token_masks: Optional[torch.Tensor] = None,
        agent_states_gt: Optional[torch.Tensor] = None,
        agent_labels_gt: Optional[torch.Tensor] = None,
        bev_semantic_map_gt: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # Call super (Qwen2_5_VLForConditionalGeneration)
        # Note: super() will handle pixel_values etc.
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            **kwargs,
        )

        # Pass through Agent/BEV inputs during prefill
        model_inputs.update({
            "agent_token_masks": agent_token_masks,
            "map_token_masks": map_token_masks,
            "agent_states_gt": agent_states_gt,
            "agent_labels_gt": agent_labels_gt,
            "bev_semantic_map_gt": bev_semantic_map_gt,
        })

        # If in decode phase (cache_position != 0), clear these inputs
        # to prevent shape mismatch or misuse
        if cache_position[0] != 0:
            model_inputs["agent_token_masks"] = None
            model_inputs["map_token_masks"] = None
            model_inputs["agent_states_gt"] = None
            model_inputs["agent_labels_gt"] = None
            model_inputs["bev_semantic_map_gt"] = None
            
        return model_inputs

    def get_agent_map_embeddings(
        self, 
        input_ids: torch.LongTensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get agent and map token masks and embeddings.
        
        Returns:
            agent_mask: [B, L] boolean mask
            map_mask: [B, L] boolean mask
            agent_embeds: flattened embeddings for masked_scatter
            map_embeds: flattened embeddings for masked_scatter
        """
        device = input_ids.device
        
        # Create masks
        agent_mask = (input_ids == self.config.agent_token_id)
        map_mask = (input_ids == self.config.map_token_id)
        
        # Get embeddings and expand for batch
        # masked_scatter expects flat embeddings matching total mask count
        agent_embeds = self.agent_query_embedding.weight  # [num_agent, hidden]
        map_embeds = self.map_query_embedding.weight      # [num_map, hidden]
        
        # Repeat for batch size
        agent_embeds = agent_embeds.unsqueeze(0).expand(batch_size, -1, -1)  # [B, num_agent, hidden]
        map_embeds = map_embeds.unsqueeze(0).expand(batch_size, -1, -1)      # [B, num_map, hidden]
        
        # Flatten for masked_scatter (which expects 1D tensor of values)
        agent_embeds_flat = agent_embeds.reshape(-1, agent_embeds.shape[-1])  # [B*num_agent, hidden]
        map_embeds_flat = map_embeds.reshape(-1, map_embeds.shape[-1])        # [B*num_map, hidden]
        
        return agent_mask, map_mask, agent_embeds_flat, map_embeds_flat
    
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
        agent_token_masks: Optional[torch.Tensor] = None,  # [B, L] from dataset
        map_token_masks: Optional[torch.Tensor] = None,    # [B, L] from dataset
        agent_states_gt: Optional[torch.Tensor] = None,    # [B, num_agent, 5]
        agent_labels_gt: Optional[torch.Tensor] = None,    # [B, num_agent]
        bev_semantic_map_gt: Optional[torch.Tensor] = None,  # [B, H, W]
        **kwargs,
    ) -> Qwen2_5_VLAgentBEVOutput:
        """
        Forward pass with agent detection and BEV semantic map prediction.
        """
        # Force output hidden states for extraction
        output_hidden_states = True
        
        # Get batch size
        if input_ids is not None:
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")
        
        # ============ Step 1: Get base embeddings ============
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
        
        # ============ Step 2: Replace image tokens (original logic) ============
        if pixel_values is not None:
            image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        
        # ============ Step 3: Replace <agent> and <map> tokens ============
        # Use masks from dataset if provided, otherwise compute from input_ids
        if agent_token_masks is not None:
            agent_mask = agent_token_masks.squeeze(0) if agent_token_masks.dim() == 3 else agent_token_masks
        else:
            agent_mask = (input_ids == self.config.agent_token_id) if self.config.agent_token_id is not None else None
        
        if map_token_masks is not None:
            map_mask = map_token_masks.squeeze(0) if map_token_masks.dim() == 3 else map_token_masks
        else:
            map_mask = (input_ids == self.config.map_token_id) if self.config.map_token_id is not None else None
        
        # Only replace tokens if mask has True values
        if agent_mask is not None and agent_mask.any():
            agent_embeds_flat = self.agent_query_embedding.weight.repeat(batch_size, 1).reshape(-1, self.config.hidden_size)
            agent_mask_expanded = agent_mask.unsqueeze(-1).expand_as(inputs_embeds)
            agent_embeds_flat = agent_embeds_flat.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(agent_mask_expanded, agent_embeds_flat)
        
        if map_mask is not None and map_mask.any():
            map_embeds_flat = self.map_query_embedding.weight.repeat(batch_size, 1).reshape(-1, self.config.hidden_size)
            map_mask_expanded = map_mask.unsqueeze(-1).expand_as(inputs_embeds)
            map_embeds_flat = map_embeds_flat.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(map_mask_expanded, map_embeds_flat)
        
        # ============ Step 4: Call transformer ============
        outputs = self.model(
            input_ids=None,  # Use inputs_embeds instead
            pixel_values=None,  # Already processed
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )
        
        hidden_states = outputs.last_hidden_state  # [B, L, C]
        
        # ============ Step 5: Compute LM logits and loss ============
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        
        lm_loss = None
        if labels is not None:
            lm_loss = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size, **kwargs
            )
        
        # ============ Step 6: Extract agent/map features and predict ============
        agent_pred = None
        bev_pred = None
        agent_loss = None
        bev_loss = None
        
        if agent_mask is not None and agent_mask.any():
            # Extract agent features
            agent_hidden = hidden_states[agent_mask]  # [total_agent_tokens, C]
            agent_hidden = agent_hidden.view(batch_size, self.config.num_agent_queries, -1)
            
            # Predict
            agent_pred = self.agent_head(agent_hidden)
            
            # Compute loss if GT provided
            if agent_states_gt is not None and agent_labels_gt is not None and self.config.enable_agent_loss:
                ce_loss, l1_loss = compute_agent_loss(
                    predictions=agent_pred,
                    targets={"agent_states": agent_states_gt, "agent_labels": agent_labels_gt},
                    config=self.config,
                )
                agent_loss = self.config.agent_class_weight * ce_loss + self.config.agent_box_weight * l1_loss
                
                self._last_logs["agent_ce_loss"] = float(ce_loss.detach().cpu())
                self._last_logs["agent_l1_loss"] = float(l1_loss.detach().cpu())
            
        if map_mask is not None and map_mask.any():
            # Extract map features
            map_hidden = hidden_states[map_mask]  # [total_map_tokens, C]
            map_hidden = map_hidden.view(batch_size, self.config.num_map_queries, -1)
            
            # Predict
            bev_pred = self.bev_head(map_hidden)
            
            # Compute loss if GT provided
            if bev_semantic_map_gt is not None and self.config.enable_bev_loss:
                bev_loss = F.cross_entropy(bev_pred, bev_semantic_map_gt.long())
                self._last_logs["bev_loss"] = float(bev_loss.detach().cpu())
        
        # ============ Step 7: Combine losses ============
        total_loss = None
        if lm_loss is not None:
            total_loss = lm_loss
            self._last_logs["lm_loss"] = float(lm_loss.detach().cpu())
        
        if agent_loss is not None:
            total_loss = agent_loss if total_loss is None else total_loss + agent_loss
        
        if bev_loss is not None:
            bev_loss_weighted = self.config.bev_weight * bev_loss
            total_loss = bev_loss_weighted if total_loss is None else total_loss + bev_loss_weighted
        
        if total_loss is not None:
            self._last_logs["total_loss"] = float(total_loss.detach().cpu())
        
        return Qwen2_5_VLAgentBEVOutput(
            loss=total_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
            agent_states=agent_pred["agent_states"] if agent_pred else None,
            agent_labels=agent_pred["agent_labels"] if agent_pred else None,
            bev_semantic_map=bev_pred,
            lm_loss=lm_loss,
            agent_loss=agent_loss,
            bev_loss=bev_loss,
        )


__all__ = [
    "Qwen2_5_VLConfigAgentBEV",
    "Qwen2_5_VLForConditionalGenerationAgentBEV",
    "Qwen2_5_VLAgentBEVOutput",
    "AgentHead",
    "BEVSemanticHead",
]
