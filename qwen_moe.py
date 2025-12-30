"""
Qwen2.5-VL with ROSS (Raw-pixel Observation State Separation) support.
This module extends the base Qwen2.5-VL model to support extracting hidden states
for image and action tokens based on provided spans.
"""
import os
if os.getenv('USING_ASCEND_910B') == "1":
    import torch_npu
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from torch.distributions import Beta, Uniform
from typing import Optional, Union, List, Tuple, Dict, Any
from dataclasses import dataclass, asdict, fields
from einops import rearrange
from diffusers import AutoencoderKL
import math
import time
from transformers.generation.utils import GenerationMixin

from .modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLCausalLMOutputWithPast,
    Qwen2_5_VLPreTrainedModel
)
from .configuration_qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLTextConfig
from .modeling_qwen2_5_vl import Qwen2_5_VLTextModel
from ...utils import ModelOutput
from .modeling_ross.denoiser_sd import RossStableDiffusionXOmni
from transformers import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

## 直接copy
@dataclass
class Qwen2_5_VLROSSOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    image_hidden_states: Optional[torch.FloatTensor] = None
    action_hidden_states: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None


class Qwen2_5_VLConfigROSS(Qwen2_5_VLConfig):
    """Extended config with ROSS support."""
    
    def __init__(
        self,
        enable_ross: bool = False,
        extract_image_hidden: bool = True,
        extract_action_hidden: bool = True,
        sd_model_path: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.enable_ross = enable_ross
        self.extract_image_hidden = extract_image_hidden
        self.extract_action_hidden = extract_action_hidden
        self.sd_model_path = sd_model_path

class Qwen2_5_VLConfigROSSMOE_ACTIONEXPERT(Qwen2_5_VLTextConfig):
    """Extended config with ROSS support."""
    
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)    
        self.intermediate_size = 3584
        self.hidden_size = 896
        self.head_dim = 128

class Qwen2_5_VLConfigROSSMOE(PretrainedConfig):
    """Extended config with ROSS support."""
    
    def __init__(
        self,
        training_args = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        if training_args is not None:
            self.action_loss_weight = getattr(training_args, "action_loss_weight", 1.0)
            self.vlm_action_loss_weight = getattr(training_args, "vlm_action_loss_weight", 0.0)
            self.vlm_ross_loss_weight = getattr(training_args, "vlm_ross_loss_weight", 0.0)
        else:
            # 推理阶段不传 training_args 时使用安全默认值
            self.action_loss_weight = 1.0
            self.vlm_action_loss_weight = 0.0
            self.vlm_ross_loss_weight = 0.0        

class Qwen2_5_VLForConditionalGenerationROSS(Qwen2_5_VLForConditionalGeneration):

    config_class = Qwen2_5_VLConfigROSS
    
    def __init__(self, config: Qwen2_5_VLConfigROSS):
        super().__init__(config)
        self.enable_ross = getattr(config, 'enable_ross', True)
        self.extract_image_hidden = getattr(config, 'extract_image_hidden', True)
        self.extract_action_hidden = getattr(config, 'extract_action_hidden', True)
        self.denoiser = RossStableDiffusionXOmni(
            unet_path=getattr(config, 'sd_model_path', 'pretrained_models/stable-diffusion-v1-5/unet'),
            z_channel=getattr(config, 'hidden_size', 3584),
            mlp_depth=2,
            n_patches=180,
        )
        # 优先使用更安全的 safetensors 格式，如果不可用则使用 pickle 格式
        vae_path = getattr(config, 'sd_model_path', 'pretrained_models/stable-diffusion-v1-5/unet').replace('/unet', '/vae')
        try:
            # 尝试不使用 pickle 加载 (safetensors)
            self.vae = AutoencoderKL.from_pretrained(vae_path, use_safetensors=True)
        except:
            # 如果失败，则使用 pickle 格式
            self.vae = AutoencoderKL.from_pretrained(vae_path, allow_pickle=True)
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.vae_shift_factor = self.vae.config.shift_factor if self.vae.config.shift_factor is not None else 0.
        self.vae_scaling_factor = self.vae.config.scaling_factor if self.vae.config.scaling_factor is not None else 1.
        
    
    def extract_hidden_with_masks(
        self,
        hidden: torch.Tensor,  # [B, L, C]
        image_masks: Optional[torch.Tensor] = None,  # [B, T, N, L] boolean
        action_masks: Optional[torch.Tensor] = None,  # [B, T, L] boolean
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Extract hidden states using structured boolean masks.
        
        Args:
            hidden: [B, L, C] hidden states
            image_masks: [B, T, N, L] boolean mask for image tokens (T time steps, N images per step)
            action_masks: [B, T, L] boolean mask for action tokens (T time steps)
            
        Returns:
            image_hidden: [B, T, N, L_img, C] extracted image hidden states
            action_hidden: [B, T, L_act, C] extracted action hidden states
        """
        B, L, C = hidden.shape
        image_hidden = None
        action_hidden = None
        
        # Extract image hidden states
        if image_masks is not None:
            # image_masks: [B, T, N, L]
            B_mask, T, N, L_mask = image_masks.shape
            
            # Ensure mask length matches hidden length
            if L_mask != L:
                if L_mask > L:
                    image_masks = image_masks[..., :L]
                    L_mask = L
                else:
                    # Pad mask to match hidden length
                    pad_len = L - L_mask
                    image_masks = torch.nn.functional.pad(image_masks, (0, pad_len), value=False)
                    L_mask = L
            
            # Extract tokens for each time step and image
            image_hidden_list = []
            for t in range(T):
                time_step_images = []
                for n in range(N):
                    batch_images = []
                    for b in range(B):
                        mask = image_masks[b, t, n]  # [L]
                        if mask.any():
                            # Extract tokens where mask is True
                            extracted = hidden[b][mask]  # [num_tokens, C]
                            batch_images.append(extracted)
                        else:
                            # No tokens for this image, use empty tensor
                            batch_images.append(torch.zeros(0, C, device=hidden.device, dtype=hidden.dtype))
                    
                    # Pad to same length within each (t, n)
                    max_tokens = max(img.shape[0] for img in batch_images) if batch_images else 0
                    if max_tokens > 0:
                        padded_batch = []
                        for img in batch_images:
                            if img.shape[0] < max_tokens:
                                pad_len = max_tokens - img.shape[0]
                                img = torch.cat([img, torch.zeros(pad_len, C, device=img.device, dtype=img.dtype)], dim=0)
                            padded_batch.append(img)
                        time_step_images.append(torch.stack(padded_batch, dim=0))  # [B, L_img, C]
                    else:
                        time_step_images.append(torch.zeros(B, 0, C, device=hidden.device, dtype=hidden.dtype))
                
                # Stack images for this time step: [N, B, L_img, C]
                if time_step_images:
                    time_tensor = torch.stack(time_step_images, dim=0)  # [N, B, L_img, C]
                    time_tensor = time_tensor.permute(1, 0, 2, 3)  # [B, N, L_img, C]
                    image_hidden_list.append(time_tensor)
            
            if image_hidden_list:
                # 先在时间维统一 L_img 再堆叠
                max_l_img = max(t.size(2) for t in image_hidden_list)  # [B, N, L_img, C]
                padded_time = []
                for t_tensor in image_hidden_list:
                    cur_l = t_tensor.size(2)
                    if cur_l < max_l_img:
                        pad_len = max_l_img - cur_l
                        # 仅在 L 维（倒数第二维）右侧补零
                        t_tensor = torch.nn.functional.pad(t_tensor, (0, 0, 0, pad_len), value=0)
                    padded_time.append(t_tensor)
                # Stack time steps: [B, T, N, L_img, C]
                image_hidden = torch.stack(padded_time, dim=1)
        
        # Extract action hidden states
        if action_masks is not None:
            # action_masks: [B, T, L]
            B_mask, T, L_mask = action_masks.shape
            
            # Ensure mask length matches hidden length
            if L_mask != L:
                if L_mask > L:
                    action_masks = action_masks[..., :L]
                    L_mask = L
                else:
                    # Pad mask to match hidden length
                    pad_len = L - L_mask
                    action_masks = torch.nn.functional.pad(action_masks, (0, pad_len), value=False)
                    L_mask = L
            
            # Extract tokens for each time step
            action_hidden_list = []
            for t in range(T):
                batch_actions = []
                for b in range(B):
                    mask = action_masks[b, t]  # [L]
                    if mask.any():
                        # Extract tokens where mask is True
                        extracted = hidden[b][mask]  # [num_tokens, C]
                        batch_actions.append(extracted)
                    else:
                        # No tokens for this action, use empty tensor
                        batch_actions.append(torch.zeros(0, C, device=hidden.device, dtype=hidden.dtype))
                
                # Pad to same length within each time step
                max_tokens = max(act.shape[0] for act in batch_actions) if batch_actions else 0
                if max_tokens > 0:
                    padded_batch = []
                    for act in batch_actions:
                        if act.shape[0] < max_tokens:
                            pad_len = max_tokens - act.shape[0]
                            act = torch.cat([act, torch.zeros(pad_len, C, device=act.device, dtype=act.dtype)], dim=0)
                        padded_batch.append(act)
                    action_hidden_list.append(torch.stack(padded_batch, dim=0))  # [B, L_act, C]
                else:
                    action_hidden_list.append(torch.zeros(B, 0, C, device=hidden.device, dtype=hidden.dtype))
            
            if action_hidden_list:
                # 先在时间维统一 L_act 再堆叠
                max_l_act = max(t.size(1) for t in action_hidden_list)  # [B, L_act, C]
                padded_time = []
                for t_tensor in action_hidden_list:
                    cur_l = t_tensor.size(1)
                    if cur_l < max_l_act:
                        pad_len = max_l_act - cur_l
                        # 仅在 L 维（倒数第二维）右侧补零
                        t_tensor = torch.nn.functional.pad(t_tensor, (0, 0, 0, pad_len), value=0)
                    padded_time.append(t_tensor)
                # Stack time steps: [B, T, L_act, C]
                action_hidden = torch.stack(padded_time, dim=1)
        
        return image_hidden, action_hidden
    
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
        # ROSS-specific inputs
        raw_pixel_values_vae: Optional[torch.Tensor] = None,  # [B, T, N, C, H, W]
        image_token_masks: Optional[torch.Tensor] = None,  # [B, T, N, L] boolean mask
        action_future_masks: Optional[torch.Tensor] = None,  # [B, T, L] boolean mask
        **kwargs,
    ) -> Union[Qwen2_5_VLCausalLMOutputWithPast, Qwen2_5_VLROSSOutput]:

        output_hidden_states = True
        
        # Call parent forward

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
            **kwargs,
        )
        
        # Extract hidden states if requested
        image_hidden = None
        action_hidden = None
        last_hidden = None
        
        last_hidden = outputs.hidden_states[-1]  # [B, L, C]
        
        # 如果 collator 已经提供了按 batch 拼接且长度对齐的张量，直接使用；
        # 否则在这里进行一次性填充并拼接为张量。
        def _pad_and_cat_list(mask_list: list, target_len: int) -> torch.Tensor:
            padded_list = []
            for m in mask_list:
                pad = target_len - m.shape[-1]
                if pad > 0:
                    m = torch.nn.functional.pad(m, (0, pad), mode='constant', value=0)
                padded_list.append(m)
            return torch.cat(padded_list, dim=0)

        if image_token_masks is not None:
            if isinstance(image_token_masks, (list, tuple)):
                img_max_len = max(m.shape[-1] for m in image_token_masks)
            else:
                img_max_len = image_token_masks.shape[-1]
        else:
            img_max_len = 0

        if action_future_masks is not None:
            if isinstance(action_future_masks, (list, tuple)):
                act_max_len = max(m.shape[-1] for m in action_future_masks)
            else:
                act_max_len = action_future_masks.shape[-1]
        else:
            act_max_len = 0

        target_length = max(img_max_len, act_max_len)

        if image_token_masks is not None and isinstance(image_token_masks, (list, tuple)):
            image_token_masks = _pad_and_cat_list(image_token_masks, target_length)
        # 若为张量则保持不变

        if action_future_masks is not None and isinstance(action_future_masks, (list, tuple)):
            action_future_masks = _pad_and_cat_list(action_future_masks, target_length)
        # 若为张量则保持不变
            
        # Extract hidden states using masks
        if (self.extract_image_hidden or self.extract_action_hidden) and (image_token_masks is not None or action_future_masks is not None):
            image_hidden, action_hidden = self.extract_hidden_with_masks(
                last_hidden,
                image_masks=image_token_masks if self.extract_image_hidden else None,
                action_masks=action_future_masks if self.extract_action_hidden else None,
            )

        ### conditions from t to **predict** t+1
        assert image_hidden.shape[1] == 2, "Currently only support 2 images for each sample"
        assert action_hidden.shape[1] == 2, "Currently only support 2 images for each sample"
        assert raw_pixel_values_vae.shape[0] == 2, "Currently only support 2 images for each sample"

        image_hidden = image_hidden[:, 0].squeeze()             # [bsz, seq_len, dim]
        action_hidden = action_hidden.mean(2)[:, 0].squeeze()   # [bsz, dim]
        raw_pixel_values_vae = raw_pixel_values_vae[1]          # [bsz, 3, h, w]

        ### conditions from t to **reconstruct** t
        # image_hidden = image_hidden.flatten(0, 1).squeeze()                                 # [bsz * t, seq_len, dim]
        # action_hidden = action_hidden.mean(2).flatten(0, 1)                                 # [bsz * t, dim]
        # raw_pixel_values_vae = raw_pixel_values_vae.permute(1, 0, 2, 3, 4).flatten(0, 1)    # [bsz * t, 3, h, w]

        raw_pixel_values_vae = torch.nn.functional.interpolate(raw_pixel_values_vae, size=(280, 504), mode='bilinear', align_corners=False)

        with torch.no_grad():
            posterior = self.vae.encode(raw_pixel_values_vae).latent_dist
            z_q = (posterior.sample() - self.vae_shift_factor) * self.vae_scaling_factor

        with torch.amp.autocast('cuda', dtype=torch.float32):
            action_hidden = self.denoiser.ln_pre_a(action_hidden)
            image_hidden = self.denoiser.ln_pre(image_hidden)
            _, n, _, _ = image_hidden.shape
            action_hidden = action_hidden.repeat_interleave(n, dim=0)
            # image_hidden = image_hidden + self.denoiser.pos_embed
            image_hidden = rearrange(image_hidden, 'b n (h w) c -> b n c h w', h=10, w=18)
            image_hidden = rearrange(image_hidden, 'b n c h w -> (b n) c h w')
            ross_loss = self.denoiser(z=image_hidden.float(), target=z_q.float(), z_a=action_hidden.float())


        self._last_logs = {
            "action_loss": float(outputs.loss.detach().cpu()),
            "ross_loss": float(ross_loss.mean().detach().cpu()),
        } 

        outputs.loss = outputs.loss + ross_loss.mean()
        
        # Return ROSS output
        return Qwen2_5_VLROSSOutput(
            loss=outputs.loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas if hasattr(outputs, 'rope_deltas') else None,
            image_hidden_states=image_hidden,
            action_hidden_states=action_hidden,
            last_hidden_state=last_hidden,
        )
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Load pretrained model (no extra ROSS-side wiring here)."""
        return super().from_pretrained(*args, **kwargs)

class Qwen2_5_VLForConditionalGenerationROSS_MOE(PreTrainedModel, GenerationMixin):
    def __init__(self, config, action_config, ckpt_path=None, data_type=torch.float32):
        super().__init__(config)

        self.data_type = data_type
        
        # 立即加载ROSS子模型
        self.qwen_ross, loading_info = Qwen2_5_VLForConditionalGenerationROSS.from_pretrained(
            ckpt_path,
            torch_dtype=data_type,
            trust_remote_code=True,
            output_loading_info=True,
        )
        print("Missing keys for Qwen2_5_VLForConditionalGenerationROSS:", loading_info["missing_keys"])
        print("Unexpected keys for Qwen2_5_VLForConditionalGenerationROSS:", loading_info["unexpected_keys"])
        print("Mismatched sizes for Qwen2_5_VLForConditionalGenerationROSS:", loading_info.get("mismatched_keys", "N/A"))

        self.config = config
        self.action_expert = Qwen2_5_VLTextModel(action_config)  # 基础类
        
        self.qwen_ross = self.qwen_ross.to(self.data_type)
        self.action_expert = self.action_expert.to(self.data_type)

        # LM 头（Action hidden -> vocab�?
        self.action_lm_head = nn.Linear(action_config.hidden_size, self.qwen_ross.config.vocab_size, bias=False).to(self.data_type)

        # state projector：将 [pre_action, cmd] -> 1 �?state token
        pre_action_dim = getattr(config, "action_dim", 3)
        self.pre_action_frames = getattr(config, "pre_action_frames", 3)
        # state_input_dim = self.pre_action_frames * pre_action_dim + 4  # 4 = cmd one-hot
        state_input_dim = self.pre_action_frames * pre_action_dim  # remove cmd one-hot
        self.state_projector = nn.Sequential(
            nn.Linear(state_input_dim, action_config.hidden_size),
            nn.SiLU(),
            nn.Linear(action_config.hidden_size, action_config.hidden_size),
        ).to(self.data_type)

        # 投影：将 action_expert �?token-emb �?vlm_hidden -> action_hidden（若两者维度不同）
        self.action_embed_projector = nn.Sequential(
            nn.Linear(self.qwen_ross.config.hidden_size, action_config.hidden_size),
            nn.SiLU(),
            nn.Linear(action_config.hidden_size, action_config.hidden_size),
        ).to(self.data_type)

        # 然后在 shared_layers 中使用这些固定引用
        self.shared_layers = [
            QwenPi0SharedLayer(vlm_layer=vlm_layer, action_layer=action_layer)
            for vlm_layer, action_layer in zip(self.qwen_ross.model.language_model.layers, self.action_expert.layers)
        ]

        # 训练配置 / 特殊 token
        self.action_loss_weight = getattr(config, "action_loss_weight", 1.0)
        self.boa_token_id = getattr(config, "boa_token_id", 151666)
        self.eoa_token_id = getattr(config, "eoa_token_id", 151665)
        self.pad_token_id = getattr(self.qwen_ross.config, "pad_token_id", None) or 151643

        # VLM loss weighting
        self.vlm_action_loss_weight = getattr(config, "vlm_action_loss_weight", 0.0)
        self.vlm_ross_loss_weight = getattr(config, "vlm_ross_loss_weight", 0.0)

        # Gradient checkpointing configuration
        self.gradient_checkpointing = False
        self.supports_gradient_checkpointing = True
        _skip_keys_device_placement = "past_key_values"
        _supports_flash_attn = True
        _supports_sdpa = True

        _can_compile_fullgraph = True
        _supports_attention_backend = True

        # Initialize weights and apply final processing
        self.post_init()

        # 解绑权重并重新初始化
        if self.qwen_ross.lm_head.weight.data_ptr() == self.qwen_ross.model.language_model.embed_tokens.weight.data_ptr():
            original_weight = self.qwen_ross.lm_head.weight.clone()  # 复制一份
            self.qwen_ross.lm_head.weight = nn.Parameter(original_weight)  # 强制新内存地址
            print("LM head 已解绑并重新初始化")

        self.loss_dict = {}

    def create_causal_style_attention_mask(self, vlm_seq_len, action_seq_len, vlm_attention_mask_original,
                                           input_ids, batch_size, device, dtype, boa_token_id=151666,
                                           eoa_token_id=151665, action_is_causal=False,
                                           visualize=False, save_path="attention_mask_causal_vis.png",
                                           action_attention_mask_1d: Optional[torch.Tensor] = None):
        """
        Create Causal-style attention mask for combined sequence: [VLM Causal] [Action Expert]
        
        Attention Rules (Causal变体):
        - VLM全部 (Text/Image + Pre Action + Future Action): 全部causal attention（只能看到当前和之前的token），不能看到Action Expert
        - Action Expert: 双向或Causal attention，能看到Text/Image + Pre Action，不能看到Future Action (Future Action is defined as the content from the second BOA token onwards)
        
        序列组织
        - VLM全部: [0, vlm_seq_len) - 全部causal
        - Action Expert: [vlm_seq_len, vlm_seq_len + action_seq_len) - 双向或causal，但只能看到VLM中第二个BOA之前的内容
        
        Args:
            action_is_causal: If True, the action expert part of the mask will be causal.
            visualize: 是否生成可视化图片
            save_path: 可视化图片保存路径
        """
        eoa_token_id = self.eoa_token_id
        boa_token_id = self.boa_token_id
        total_seq_len = vlm_seq_len + action_seq_len

        # Initialize combined mask (additive, so 0 means attend, -inf means no attend)
        combined_mask = torch.full(
            (batch_size, 1, total_seq_len, total_seq_len),
            torch.finfo(dtype).min / 2,
            device=device,
            dtype=dtype
        )

        # 必须有vlm_attention_mask_original且为2D
        if vlm_attention_mask_original is None:
            raise ValueError("vlm_attention_mask_original is required for Causal-style attention")
        if vlm_attention_mask_original.dim() != 2:
            raise ValueError(f"vlm_attention_mask_original must be 2D, got {vlm_attention_mask_original.dim()}D")

        vlm_ids = input_ids[:, :vlm_seq_len]

        # 检查每个样本是否包含BOA和EOA token
        is_boa = (vlm_ids == boa_token_id)
        boa_counts = is_boa.sum(dim=1)
        assert torch.all(boa_counts == 2), f"Expected 2 BOA tokens per sample, but found counts: {boa_counts}"

        # 找到第二个boa的位置
        cumsum_boa = torch.cumsum(is_boa.float(), dim=1)
        is_second_boa = (cumsum_boa == 2) & is_boa
        second_boa_positions = torch.argmax(is_second_boa.float(), dim=1)  # [batch_size]

        # 1. VLM全部内部: causal attention + padding
        vlm_valid = vlm_attention_mask_original.bool()  # [B, VLM_S] - 使用原始padding mask
        vlm_seq_idx = torch.arange(vlm_seq_len, device=device).unsqueeze(0).unsqueeze(0)  # [1, 1, VLM_S]
        vlm_causal = vlm_seq_idx.transpose(1, 2) >= vlm_seq_idx  # [1, VLM_S, VLM_S] - causal mask
        vlm_attend = vlm_valid.unsqueeze(2) & vlm_valid.unsqueeze(1) & vlm_causal  # [B, VLM_S, VLM_S]
        combined_mask[:, 0, :vlm_seq_len, :vlm_seq_len] = torch.where(
            vlm_attend,
            torch.tensor(0.0, device=device, dtype=dtype),
            combined_mask[:, 0, :vlm_seq_len, :vlm_seq_len]
        )

        # 2. Action Expert 内部: 双向或causal注意
        action_expert_start = vlm_seq_len
        action_expert_end = vlm_seq_len + action_seq_len
        if action_seq_len > 0:
            if action_is_causal:
                action_seq_idx = torch.arange(action_seq_len, device=device).unsqueeze(0).unsqueeze(0)  # [1, 1, A_S]
                action_causal_mask = action_seq_idx.transpose(1, 2) >= action_seq_idx  # [1, A_S, A_S]
                action_mask_quadrant = torch.full(
                    (batch_size, 1, action_seq_len, action_seq_len), torch.finfo(dtype).min / 2, device=device,
                    dtype=dtype
                )
                action_mask_quadrant = torch.where(
                    action_causal_mask,
                    torch.tensor(0.0, device=device, dtype=dtype),
                    action_mask_quadrant
                )
                combined_mask[:, :, action_expert_start:action_expert_end,
                action_expert_start:action_expert_end] = action_mask_quadrant
            else:
                combined_mask[:, :, action_expert_start:action_expert_end, action_expert_start:action_expert_end] = 0.0

        # 3. Action Expert -> VLM (只能看到第二个BOA之前的部分
        if action_seq_len > 0:
            seq_indices = torch.arange(vlm_seq_len, device=device).unsqueeze(0)  # [1, VLM_S]

            # Action Expert can see VLM content before the second BOA token.
            vlm_visible_to_action_expert_mask = seq_indices < second_boa_positions.unsqueeze(1)  # [B, VLM_S]

            # Also respect original padding
            vlm_visible_to_action_expert_mask = vlm_visible_to_action_expert_mask & vlm_attention_mask_original.bool()

            # Expand to fit the mask quadrant shape [B, 1, A_S, VLM_S]
            vlm_visible_to_action_expert_mask = vlm_visible_to_action_expert_mask.unsqueeze(1).expand(-1,
                                                                                                      action_seq_len,
                                                                                                      -1)
            vlm_visible_to_action_expert_mask = vlm_visible_to_action_expert_mask.unsqueeze(1)

            combined_mask[:, :, action_expert_start:action_expert_end, :vlm_seq_len] = torch.where(
                vlm_visible_to_action_expert_mask,
                torch.tensor(0.0, device=device, dtype=dtype),
                torch.tensor(torch.finfo(dtype).min / 2, device=device, dtype=dtype)
            )

        # 4.（可选）屏蔽 Action 段中的 PAD（仅 Query 屏蔽）
        # 说明：causal 情况下 PAD 在尾部，Query 本就不能看到后续 PAD Key，
        #       故只需对 Query 做行屏蔽，避免无效计算；列屏蔽可省略。
        if action_seq_len > 0 and action_attention_mask_1d is not None:
            act_valid = action_attention_mask_1d.bool()
            row_valid = act_valid.unsqueeze(1).unsqueeze(-1)  # [B,1,A_S,1]
            combined_mask[:, :, action_expert_start:action_expert_end, :] = torch.where(
                row_valid,
                combined_mask[:, :, action_expert_start:action_expert_end, :],
                torch.tensor(torch.finfo(dtype).min / 2, device=device, dtype=dtype)
            )

        return combined_mask

    def build_multimodal_inputs_embeds(
        self, 
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        inputs_embeds: torch.LongTensor,
        image_grid_thw: torch.LongTensor,
        **kwargs
    ):
        if pixel_values is not None:
            # torch.cuda.synchronize()
            # start_time = time.perf_counter()

            image_embeds = self.qwen_ross.model.get_image_features(pixel_values, image_grid_thw)

            # end_time = time.perf_counter()
            # run_time = (end_time - start_time) * 1000
            # print(f"infer for image_embeds: {run_time:.1f}ms") 

            image_embeds = torch.cat(image_embeds, dim=0)
            image_mask, _ = self.qwen_ross.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        return inputs_embeds

    def forward(
            self,
            pre_action: torch.Tensor,
            # cmd: Optional[torch.Tensor],
            action: Optional[torch.Tensor] = None,
            vlm_input_ids: torch.LongTensor = None,
            vlm_labels: torch.LongTensor = None,
            input_ids: torch.LongTensor = None,
            action_input_ids: torch.LongTensor = None,
            vlm_attention_mask: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
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
            # ROSS-specific inputs
            raw_pixel_values_vae: Optional[torch.Tensor] = None,  # [B, T, N, C, H, W]
            image_token_masks: Optional[torch.Tensor] = None,  # [B, T, N, L] boolean mask
            action_future_masks: Optional[torch.Tensor] = None,  # [B, T, L] boolean mask
            frame_image_counts: Optional[torch.Tensor] = None,
            return_dict: Optional[bool] = None,
            # 可选：外部已算好的缓存
            vlm_k_rope_cache: Optional[List[torch.Tensor]] = None,
            vlm_v_cache: Optional[List[torch.Tensor]] = None,
            vlm_seq_len: Optional[int] = None,
            token=None,
            grpo_sample=False,
    ) -> Union[Qwen2_5_VLCausalLMOutputWithPast, Qwen2_5_VLROSSOutput]:
        self.loss_dict.clear()
        use_cache = False  # Explicitly setting use_cache to False for this training-focused forward pass
        return_dict = return_dict if return_dict is not None else True

        # self.qwen_ross.to(input_ids.device)
        # self.action_expert.to(input_ids.device)

        if self.training and not grpo_sample:
            if input_ids is None:
                raise ValueError("action_input_ids is required (starts with <boa>) in training")
            if pre_action is None:
                raise ValueError("pre_action and cmd must be provided to build the state token in training")
            if vlm_input_ids is None:
                raise ValueError("Training requires vlm_input_ids / vlm_attention_mask / pre_action / cmd")

            B = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
            device = input_ids.device if input_ids is not None else inputs_embeds.device

            # Initial VLM embeddings
            if inputs_embeds is None:
                vlm_inputs_embeds = self.qwen_ross.model.language_model.embed_tokens(vlm_input_ids)
            current_vlm_h = self.build_multimodal_inputs_embeds(vlm_input_ids, pixel_values, vlm_inputs_embeds, image_grid_thw)

            vlm_seq_len = current_vlm_h.shape[1]

            # Build state + action embeddings
            # state_in = torch.cat([pre_action.view(B, -1), cmd], dim=1).to(self.data_type)
            state_in = pre_action.view(B, -1).to(self.data_type)
            state_tok = self.state_projector(state_in).unsqueeze(1)  # [B,1,H_a]

            action_emb = self.qwen_ross.model.language_model.embed_tokens(input_ids)  # [B,T,H_vlm]
            action_emb = self.action_embed_projector(action_emb)  # [B,T,H_a]
            current_action_h = torch.cat([state_tok, action_emb], dim=1)  # [B,1+T,H_a]
            action_len_with_state = current_action_h.size(1)

            num_layers = len(self.qwen_ross.model.language_model.layers) # 28 layers
            # create position embeddings to be shared across the decoder layers
            vlm_position_embeddings = self.qwen_ross.model.language_model.rotary_emb(vlm_inputs_embeds, position_ids)

            # Valid action mask with state token
            action_valid_wo_state = (input_ids != self.pad_token_id)
            action_valid_with_state = torch.cat(
                [torch.ones(B, 1, dtype=torch.bool, device=device), action_valid_wo_state], dim=1)

            combined_attention_mask_4d = self.create_causal_style_attention_mask(vlm_seq_len=vlm_seq_len, action_seq_len=action_len_with_state,
                                                                                vlm_attention_mask_original=vlm_attention_mask, input_ids=vlm_input_ids, 
                                                                                batch_size=B, device=device, dtype=self.data_type, action_is_causal=True,
                                                                                action_attention_mask_1d=action_valid_with_state)

            # Check for gradient checkpointing
            if self.gradient_checkpointing and self.training:
                if use_cache:
                    logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...")
                    use_cache = False

            # Layer-wise processing with gradient checkpointing support
            for layer_idx in range(num_layers):
                shared_layer = self.shared_layers[layer_idx]
                shared_layer.vlm_layer.to(device)
                shared_layer.action_layer.to(device)

                if self.gradient_checkpointing and self.training:
                    # Use gradient checkpointing with proper nn.Module
                    current_vlm_h, current_action_h = self._gradient_checkpointing_func(
                        shared_layer.__call__,
                        current_vlm_h,
                        current_action_h,
                        vlm_position_embeddings,
                        combined_attention_mask_4d,
                        vlm_seq_len,
                        action_len_with_state,
                        B,
                    )
                else:
                    # Normal forward pass without checkpointing
                    current_vlm_h, current_action_h = shared_layer(
                        current_vlm_h,
                        current_action_h,
                        vlm_position_embeddings,
                        combined_attention_mask_4d,
                        vlm_seq_len,
                        action_len_with_state,
                        B,
                    )

            h_final = self.action_expert.norm(current_action_h)
            logits = self.action_lm_head(h_final)
            loss = None
            if labels is not None:
                # Next-token objective via shifting logits; include state-><boa>
                logits_for_loss = logits[:, :-1, :].contiguous()  # [B, T, V]
                targets = labels.contiguous()  # [B, T]
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(
                    logits_for_loss.view(-1, logits_for_loss.size(-1)),
                    targets.view(-1)
                )
                loss = loss * getattr(self, "action_loss_weight", 1.0)

            # loss for VLM and ROSS
            if self.vlm_action_loss_weight > 0.0:
                vlm_final = self.qwen_ross.model.language_model.norm(current_vlm_h)
                logits = self.qwen_ross.lm_head(vlm_final) # 与qwen_ross.lm_head共享内存
                logits = logits.float()
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = vlm_labels[..., 1:].contiguous()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                vlm_action_loss = loss_fct(shift_logits, shift_labels)
                loss = loss + self.vlm_action_loss_weight * vlm_action_loss

            if self.vlm_ross_loss_weight > 0.0:
                ross_loss = self.compute_ross_loss_and_update_logs(vlm_final, image_token_masks, action_future_masks, raw_pixel_values_vae)
                loss = loss + self.vlm_ross_loss_weight * ross_loss

            if not return_dict:
                return logits, None, None, loss
            return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None)
        else:
            if input_ids is None:
                raise ValueError("action_input_ids（Action tokens，从 <boa> 开始）不能为空")
            if pre_action is None or cmd is None:
                raise ValueError("需要提供pre_action and cmd 以构成state token")
            if (vlm_k_rope_cache is None) or (vlm_v_cache is None) or (vlm_seq_len is None):
                if vlm_input_ids is None:
                    raise ValueError("缺少 VLM 上下文：请提供vlm_input_ids 及其 mask")
                # 训练态：带梯度；评估/推理：无梯度
                no_grad = not self.training
                vlm_k_rope_cache, vlm_v_cache, vlm_seq_len = self._precompute_vlm_kv(
                    vlm_input_ids=vlm_input_ids,
                    vlm_attention_mask=vlm_attention_mask,
                    vlm_position_ids=vlm_position_ids,
                    no_grad=no_grad,
                )

            B = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
            device = input_ids.device if input_ids is not None else inputs_embeds.device

            # Build state + action embeddings
            state_in = torch.cat([pre_action.view(B, -1), cmd], dim=1).to(self.data_type)
            state_tok = self.state_projector(state_in).unsqueeze(1)  # [B,1,H_a]

            action_emb = self.qwen_ross.model.language_model.embed_tokens(input_ids)  # [B,T,H_vlm]
            action_emb = self.action_embed_projector(action_emb)  # [B,T,H_a]
            h = torch.cat([state_tok, action_emb], dim=1)  # [B,1+T,H_a]
            action_len_with_state = h.size(1)

            # Valid action mask with state token
            action_valid_wo_state = (input_ids != self.pad_token_id)
            action_valid_with_state = torch.cat(
                [torch.ones(B, 1, dtype=torch.bool, device=device), action_valid_wo_state], dim=1)

            combined_attention_mask_4d = self.create_causal_style_attention_mask(vlm_seq_len=vlm_seq_len, action_seq_len=action_len_with_state,
                                                                                vlm_attention_mask_original=vlm_attention_mask, input_ids=vlm_input_ids, 
                                                                                batch_size=B, device=device, dtype=self.data_type, action_is_causal=True,
                                                                                action_attention_mask_1d=action_valid_with_state)
            action_attention_mask = combined_attention_mask_4d[:, :, vlm_seq_len:, :]

            # --- 过共享层：Action-Q attends [VLM-KV(cache), Action-KV] ---
            # torch.cuda.synchronize()
            # start_time = time.perf_counter()
            for li, layer in enumerate(self.shared_layers):
                h = layer.forward_with_cache(
                    h,
                    action_attention_mask=action_attention_mask,
                    vlm_seq_len=vlm_seq_len,
                    action_seq_len=action_len_with_state,
                    batch_size=B,
                    vlm_k_rope_cached=vlm_k_rope_cache[li],
                    vlm_v_cached=vlm_v_cache[li],
                    action_query_valid_1d=action_valid_with_state,
                )

            # --- logits & loss ---
            h = self.action_expert.norm(h)
            logits = self.action_lm_head(h)  # [B,1+T,V]
            # torch.cuda.synchronize()
            # end_time = time.perf_counter()
            # run_time = (end_time - start_time) * 1000
            # print(f"infer for one token: {run_time:.1f}ms")

            loss = None
            if labels is not None:
                # 典型 CausalLM：丢弃 state 位置对应的 logits，确保输出与 labels 对齐
                # 要求 labels 形状为 [B, T]（与 input_ids 完全对齐），在需要忽略的位置（如特殊标记 [STATE]、padding 位置等）将 labels 设为 -100，以忽略这些位置的 loss 计算
                logits_for_loss = logits[:, :-1, :].contiguous()  # [B,T,V]
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(logits_for_loss.view(-1, logits_for_loss.size(-1)), labels.view(-1)) * getattr(self,
                                                                                                               "action_loss_weight",
                                                                                                               1.0)

            if not return_dict:
                return logits, None, None, loss
            return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None)

    def compute_ross_loss_and_update_logs(
        self,
        vlm_final,
        image_token_masks,
        action_future_masks,
        raw_pixel_values_vae,
    ):
        # Extract last hidden state
        last_hidden = vlm_final  # [B, L, C]

        # Utility function to pad and concatenate list of masks
        def _pad_and_cat_list(mask_list: list, target_len: int) -> torch.Tensor:
            padded_list = []
            for m in mask_list:
                pad = target_len - m.shape[-1]
                if pad > 0:
                    m = torch.nn.functional.pad(m, (0, pad), mode='constant', value=0)
                padded_list.append(m)
            return torch.cat(padded_list, dim=0)

        # Determine target length from masks
        img_max_len = 0
        if image_token_masks is not None:
            if isinstance(image_token_masks, (list, tuple)):
                img_max_len = max(m.shape[-1] for m in image_token_masks)
            else:
                img_max_len = image_token_masks.shape[-1]

        act_max_len = 0
        if action_future_masks is not None:
            if isinstance(action_future_masks, (list, tuple)):
                act_max_len = max(m.shape[-1] for m in action_future_masks)
            else:
                act_max_len = action_future_masks.shape[-1]

        target_length = max(img_max_len, act_max_len)

        # Pad and concatenate masks if they are lists
        if image_token_masks is not None and isinstance(image_token_masks, (list, tuple)):
            image_token_masks = _pad_and_cat_list(image_token_masks, target_length)

        if action_future_masks is not None and isinstance(action_future_masks, (list, tuple)):
            action_future_masks = _pad_and_cat_list(action_future_masks, target_length)

        # Extract hidden states using masks
        image_hidden = None
        action_hidden = None

        if (self.qwen_ross.extract_image_hidden or self.qwen_ross.extract_action_hidden) and (
            image_token_masks is not None or action_future_masks is not None
        ):
            image_hidden, action_hidden = self.qwen_ross.extract_hidden_with_masks(
                last_hidden,
                image_masks=image_token_masks if self.qwen_ross.extract_image_hidden else None,
                action_masks=action_future_masks if self.qwen_ross.extract_action_hidden else None,
            )

        # Sanity checks
        assert image_hidden is not None, "image_hidden should not be None"
        assert action_hidden is not None, "action_hidden should not be None"

        assert image_hidden.shape[1] == 2, "Currently only support 2 images per sample"
        assert action_hidden.shape[1] == 2, "Currently only support 2 images per sample"
        assert raw_pixel_values_vae.shape[0] == 2, "Currently only support 2 images per sample"

        # Select the second image for reconstruction (index 1)
        # And process hidden states
        image_hidden = image_hidden[:, 0].squeeze(dim=1)             # [bsz, seq_len, dim]
        action_hidden = action_hidden.mean(2)[:, 0].squeeze(dim=1)   # [bsz, dim]
        raw_pixel_values_vae = raw_pixel_values_vae[1]          # [bsz, 3, h, w]

        # Resize input to match expected resolution for VAE
        raw_pixel_values_vae = torch.nn.functional.interpolate(
            raw_pixel_values_vae,
            size=(280, 504),
            mode='bilinear',
            align_corners=False
        )

        # Encode with VAE to get posterior
        with torch.no_grad():
            posterior = self.qwen_ross.vae.encode(raw_pixel_values_vae).latent_dist
            z_q = (posterior.sample() - self.qwen_ross.vae_shift_factor) * self.qwen_ross.vae_scaling_factor

        # Denoising
        with torch.amp.autocast('cuda', dtype=torch.float32):
            action_hidden = self.qwen_ross.denoiser.ln_pre_a(action_hidden)
            image_hidden = self.qwen_ross.denoiser.ln_pre(image_hidden)
            # image_hidden = rearrange(image_hidden, 'b (h w) c -> b c h w', h=10, w=18)
            # support multiview
            _, n, _, _ = image_hidden.shape
            action_hidden = action_hidden.repeat_interleave(n, dim=0)
            image_hidden = rearrange(image_hidden, 'b n (h w) c -> b n c h w', h=10, w=18)
            image_hidden = rearrange(image_hidden, 'b n c h w -> (b n) c h w')            

            ross_loss = self.qwen_ross.denoiser(z=image_hidden.float(), target=z_q.float(), z_a=action_hidden.float())

        return ross_loss.mean()

    # ---- 便捷封装：直接生成动作序列
    @torch.no_grad()
    def generate_actions(
            self,
            vlm_input_ids: torch.LongTensor,
            vlm_attention_mask: Optional[torch.Tensor] = None,
            vlm_position_ids: Optional[torch.LongTensor] = None,
            pre_action: Optional[torch.Tensor] = None,
            cmd: Optional[torch.Tensor] = None,
            max_new_tokens: int = 10,
            do_sample: bool = True,
            top_p: float = 1.0,
            temperature: float = 1.0,
            pad_token_id: Optional[int] = None,
            eos_token_id: Optional[int] = None,
            logits_processor: Optional[list] = None,
            return_scores: bool = False,
            vlm_k_rope_cache: Optional[List[torch.Tensor]] = None,
            vlm_v_cache: Optional[List[torch.Tensor]] = None,
            vlm_seq_len: Optional[int] = None,
            pixel_values: Optional[torch.Tensor] = None,
            image_grid_thw: Optional[torch.Tensor] = None,
            frame_idx: int = 0,
    ) -> Union[torch.LongTensor, "GenerateDecoderOnlyOutput"]:
        device = vlm_input_ids.device
        B = vlm_input_ids.size(0)
        bos = torch.full((B, 1), self.boa_token_id, dtype=torch.long, device=device)

        # extend_sequence = False
        # num_init_tokens = 2
        # num_repeats = 100
        # if extend_sequence and vlm_input_ids.size(1) >= num_init_tokens:
        #     # Store original inputs
        #     orig_input_ids = vlm_input_ids
        #     orig_attention_mask = vlm_attention_mask
        #     orig_position_ids = vlm_position_ids
            
        #     # Get the first 'num_init_tokens' tokens to repeat
        #     init_tokens = vlm_input_ids[:, :num_init_tokens]
            
        #     # Create the repeated block (shape will be [B, num_init_tokens * num_repeats])
        #     repeated_block = init_tokens.repeat(1, num_repeats)
            
        #     # Concatenate with original input
        #     vlm_input_ids = torch.cat([vlm_input_ids, repeated_block], dim=1)
            
        #     # Similarly handle attention mask if present
        #     if vlm_attention_mask is not None:
        #         init_mask = vlm_attention_mask[:, :num_init_tokens]
        #         repeated_mask = init_mask.repeat(1, num_repeats)
        #         vlm_attention_mask = torch.cat([vlm_attention_mask, repeated_mask], dim=1)
            
        #     # Handle position IDs - 修改部分开始
        #     if vlm_position_ids is None:
        #         seq_len = vlm_input_ids.size(1)
        #         # 创建三维位置ID [B, 3, L]
        #         pos_base = torch.arange(seq_len, device=device).unsqueeze(0).unsqueeze(0)
        #         vlm_position_ids = pos_base.expand(B, 3, -1)
        #     else:
        #         # 获取前num_init_tokens的位置 [B, 3, num_init_tokens]
        #         init_positions = vlm_position_ids[:, :, :num_init_tokens]
        #         # 获取最后一个位置的偏移量 [B, 3, 1]
        #         last_pos = vlm_position_ids[:, :, -1:]
        #         offset = (last_pos + 1) - init_positions[:, :, :1]
                
        #         # 创建重复块的位置ID
        #         position_blocks = []
        #         current_offset = 0
        #         for _ in range(num_repeats):
        #             # 每个重复块增加相应的offset
        #             block = init_positions + current_offset
        #             position_blocks.append(block)
        #             current_offset += offset
                
        #         # 拼接所有块 [B, 3, num_init_tokens*num_repeats]
        #         repeated_positions = torch.cat(position_blocks, dim=-1)
        #         # 合并到原始位置IDs [B, 3, orig_len + num_init_tokens*num_repeats]
        #         vlm_position_ids = torch.cat([vlm_position_ids, repeated_positions], dim=-1)
            
        #     # 确保新的position_ids长度与输入一致
        #     assert vlm_position_ids.shape[-1] == vlm_input_ids.shape[-1]

        # Fast-path：无梯度缓存（可复用传入缓存)
        # torch.cuda.synchronize()
        # start_time = time.perf_counter()
        if vlm_k_rope_cache is None or vlm_v_cache is None or vlm_seq_len is None:
            k_cache, v_cache, vlm_seq_len = self._precompute_vlm_kv(
                vlm_input_ids=vlm_input_ids,
                vlm_attention_mask=vlm_attention_mask,
                vlm_position_ids=vlm_position_ids,
                no_grad=True,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                frame_idx=frame_idx
            )
        else:
            k_cache, v_cache = vlm_k_rope_cache, vlm_v_cache
        # torch.cuda.synchronize()
        # end_time = time.perf_counter()
        # run_time = (end_time - start_time) * 1000
        # print(f"infer for self._precompute_vlm_kv: {run_time:.1f}ms")

        bos_attention_mask = torch.ones_like(bos, dtype=torch.long, device=device)

        gen_res = self.generate(
            input_ids=bos,
            attention_mask=bos_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=max(temperature, 1e-5) if do_sample else 1.0,
            eos_token_id=self.eoa_token_id,
            pad_token_id=self.config.pad_token_id,
            logits_processor=logits_processor,
            return_dict_in_generate=return_scores,
            output_scores=return_scores,
            vlm_input_ids=vlm_input_ids,
            vlm_attention_mask=vlm_attention_mask,
            position_ids=vlm_position_ids,
            pre_action=pre_action,
            # cmd=cmd,
            vlm_k_rope_cache=k_cache,
            vlm_v_cache=v_cache,
            vlm_seq_len=vlm_seq_len,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

        return gen_res

    # ---- HF .generate() 会先调这个方法打包额外参数（只在首步构建 VLM 缓存) ---
    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            **kwargs,
    ):
        # 兼容 action_input_ids 作为 generate() 的输入名
        if kwargs.get("action_input_ids", None) is not None and input_ids is None:
            input_ids = kwargs.pop("action_input_ids")
        # 首步：无缓存时，启用 no_grad 与快路径预计算（仅推理）
        if kwargs.get("vlm_k_rope_cache", None) is None or kwargs.get("vlm_v_cache", None) is None:
            vlm_input_ids = kwargs.get("vlm_input_ids", None)
            if vlm_input_ids is None:
                raise ValueError("generate() 需要传 vlm_input_ids / vlm_attention_mask / pre_action / cmd")
            k_cache, v_cache, vlm_seq_len = self._precompute_vlm_kv(
                vlm_input_ids=vlm_input_ids,
                vlm_attention_mask=kwargs.get("vlm_attention_mask", None),
                vlm_position_ids=kwargs.get("vlm_position_ids", None),
                no_grad=True,
            )
            kwargs["vlm_k_rope_cache"] = k_cache
            kwargs["vlm_v_cache"] = v_cache
            kwargs["vlm_seq_len"] = vlm_seq_len
        # 我们不使用past_key_values（Action 侧每步重算，已足够快)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds,
            **kwargs,
        }

    # ---------------- 内部：VLM KV 预计算（可控是否带梯度 & 可选 checkpoint----------------
    def _precompute_vlm_kv(
            self,
            vlm_input_ids: torch.LongTensor,
            vlm_attention_mask: Optional[torch.Tensor] = None,
            vlm_position_ids: Optional[torch.LongTensor] = None,
            no_grad: bool = True,
            pixel_values: Optional[torch.LongTensor] = None,
            image_grid_thw: Optional[torch.LongTensor] = None,
            frame_idx: int=0,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], int]:
        """
        返回
            vlm_k_rope_cache, vlm_v_cache, vlm_seq_len
            形状：list[num_layers]，每个元素[B, n_kv_heads, S_vlm, head_dim]
        """
        ctx = torch.no_grad() if no_grad else contextlib.nullcontext()
        with ctx:
            device = vlm_input_ids.device
            B, S = vlm_input_ids.shape

            vlm_inputs_embeds = self.qwen_ross.model.language_model.embed_tokens(vlm_input_ids)
            h = self.build_multimodal_inputs_embeds(vlm_input_ids, pixel_values, vlm_inputs_embeds, image_grid_thw)    

            if vlm_position_ids is None:
                vlm_position_ids = torch.arange(S, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)

            attn_4d = _prepare_4d_causal_attention_mask(vlm_attention_mask, (B, S), h, 0)

            vlm_k_list: List[torch.Tensor] = []
            vlm_v_list: List[torch.Tensor] = []

            # 逐层构建缓存；如在训练并开启了 GC，对 layer(h) 前向进行 checkpoint
            for layer in self.qwen_ross.model.language_model.layers:
                normed = layer.input_layernorm(h)
                n_heads = layer.self_attn.num_heads
                kv_heads = layer.self_attn.num_key_value_heads
                head_dim = layer.self_attn.head_dim

                q = layer.self_attn.q_proj(normed).view(B, S, n_heads, head_dim).transpose(1, 2)
                k = layer.self_attn.k_proj(normed).view(B, S, kv_heads, head_dim).transpose(1, 2)
                v = layer.self_attn.v_proj(normed).view(B, S, kv_heads, head_dim).transpose(1, 2)

                position_embeds = self.qwen_ross.model.language_model.rotary_emb(h, vlm_position_ids)
                cos, sin = position_embeds
                _, k_rope = apply_multimodal_rotary_pos_emb(
                    q, k, cos, sin, layer.self_attn.rope_scaling["mrope_section"]
                )

                vlm_k_list.append(k_rope)
                vlm_v_list.append(v)

                h = layer(h, attention_mask=attn_4d, position_ids=vlm_position_ids, position_embeddings=position_embeds)[0]

        return vlm_k_list, vlm_v_list, S

    def get_input_embeddings(self):
        return self.qwen_ross.model.language_model.embed_tokens

    def set_input_embeddings(self, value):
        self.qwen_ross.model.language_model.embed_tokens = value

    def get_output_embeddings(self):
        return self.qwen_ross.lm_head # 与qwen_ross.lm_head共享内存

    def set_output_embeddings(self, new_embeddings):
        self.qwen_ross.model.language_model.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.qwen_ross.model.language_model = decoder

    def get_decoder(self):
        return self.qwen_ross.model.language_model

    def freeze_vlm(self):
        """Freeze VLM parameters, training only the Action Expert parts."""
        for param in self.qwen_ross.model.language_model.parameters():
            param.requires_grad = False
        # Keep action expert and its head trainable
        for param in self.action_expert.parameters():
            param.requires_grad = True
        for param in self.action_lm_head.parameters():
            param.requires_grad = True

    def freeze_vlm_half(self):
        num_layers = len(self.qwen_ross.model.language_model.layers)
        half_point = num_layers // 2  # 分界点：前半部分冻结，后半部分解冻

        # 冻结下半部分（前 half_point 层）
        for i in range(half_point):
            for param in self.qwen_ross.model.language_model.layers[i].parameters():
                param.requires_grad = False

        # 解冻上半部分（后 half_point 层）
        for i in range(half_point, num_layers):
            for param in self.qwen_ross.model.language_model.layers[i].parameters():
                param.requires_grad = True

        # Keep action expert and its head trainable
        for param in self.action_expert.parameters():
            param.requires_grad = True
        for param in self.action_lm_head.parameters():
            param.requires_grad = True

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        # 未使�?past；为兼容 BeamSearch 接口，返回原�?
        return past_key_values

    # ---------------- HF generate 需要的接口 ----------------
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: Optional[dict] = None, **kwargs):
        super().gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self, **kwargs):
        super().gradient_checkpointing_disable()
        self.gradient_checkpointing = False

class QwenPi0SharedLayer:
    """
    A dedicated helper class for shared VLM-Action Expert layer processing.
    It is not an nn.Module to avoid duplicated parameter registration.
    This is compatible with gradient checkpointing.
    """

    def __init__(self, vlm_layer, action_layer):
        self.vlm_layer = vlm_layer
        self.action_layer = action_layer

    def __call__(
            self,
            current_vlm_h: torch.Tensor,
            current_action_h: torch.Tensor,
            vlm_position_embeddings: torch.Tensor,
            combined_attention_mask_4d: torch.Tensor,
            vlm_seq_len: int,
            action_seq_len: int,
            batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a single layer of both VLM and Action Expert with shared attention.
        """
        return self.forward(
            current_vlm_h,
            current_action_h,
            vlm_position_embeddings,
            combined_attention_mask_4d,
            vlm_seq_len,
            action_seq_len,
            batch_size
        )    

    def forward(
            self,
            current_vlm_h: torch.Tensor,
            current_action_h: torch.Tensor,
            vlm_position_embeddings: torch.Tensor,
            combined_attention_mask_4d: torch.Tensor,
            vlm_seq_len: int,
            action_seq_len: int,
            batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a single layer of both VLM and Action Expert with shared attention.
        """
        _num_heads = self.vlm_layer.self_attn.num_heads
        _head_dim = self.vlm_layer.self_attn.head_dim
        _num_key_value_heads = self.vlm_layer.self_attn.num_key_value_heads
        _num_key_value_groups = self.vlm_layer.self_attn.num_key_value_groups
        _hidden_size_attn_output = _num_heads * _head_dim

        residual_vlm = current_vlm_h
        residual_action = current_action_h

        normed_vlm_h = self.vlm_layer.input_layernorm(current_vlm_h)
        normed_action_h = self.action_layer.input_layernorm(current_action_h)

        q_vlm = self.vlm_layer.self_attn.q_proj(normed_vlm_h)
        k_vlm = self.vlm_layer.self_attn.k_proj(normed_vlm_h)
        v_vlm = self.vlm_layer.self_attn.v_proj(normed_vlm_h)

        q_action = self.action_layer.self_attn.q_proj(normed_action_h)
        k_action = self.action_layer.self_attn.k_proj(normed_action_h)
        v_action = self.action_layer.self_attn.v_proj(normed_action_h)

        q_vlm = q_vlm.view(batch_size, vlm_seq_len, _num_heads, _head_dim).transpose(1, 2)
        k_vlm = k_vlm.view(batch_size, vlm_seq_len, _num_key_value_heads, _head_dim).transpose(1, 2)
        v_vlm = v_vlm.view(batch_size, vlm_seq_len, _num_key_value_heads, _head_dim).transpose(1, 2)

        q_action = q_action.view(batch_size, action_seq_len, _num_heads, _head_dim).transpose(1, 2)
        k_action = k_action.view(batch_size, action_seq_len, _num_key_value_heads, _head_dim).transpose(1, 2)
        v_action = v_action.view(batch_size, action_seq_len, _num_key_value_heads, _head_dim).transpose(1, 2)

        # Apply RoPE only to VLM part, not to action expert
        cos, sin = vlm_position_embeddings
        q_vlm_rope, k_vlm_rope = apply_multimodal_rotary_pos_emb(
            q_vlm, k_vlm, cos, sin, self.vlm_layer.self_attn.rope_scaling["mrope_section"]
        )

        # Action expert does not use RoPE - keep original q_action, k_action  [2400, 2408] -> [1600, 1608]
        combined_q = torch.cat([q_vlm_rope, q_action], dim=2)
        combined_k = torch.cat([k_vlm_rope, k_action], dim=2)
        combined_v = torch.cat([v_vlm, v_action], dim=2)

        combined_k_repeated = repeat_kv(combined_k, _num_key_value_groups)
        combined_v_repeated = repeat_kv(combined_v, _num_key_value_groups)

        combined_q = combined_q.contiguous()
        combined_k_repeated = combined_k_repeated.contiguous()
        combined_v_repeated = combined_v_repeated.contiguous()


        attn_output_combined = torch.nn.functional.scaled_dot_product_attention(
            combined_q,
            combined_k_repeated,
            combined_v_repeated,
            attn_mask=combined_attention_mask_4d,
            dropout_p=0.0,
            is_causal=False,
        )            

        attn_output_combined = attn_output_combined.transpose(1, 2).contiguous()
        total_seq_len = vlm_seq_len + action_seq_len
        attn_output_combined = attn_output_combined.reshape(batch_size, total_seq_len, -1)

        attn_out_vlm = attn_output_combined[:, :vlm_seq_len, :]
        attn_out_action = attn_output_combined[:, vlm_seq_len:, :]

        attn_out_vlm_proj = self.vlm_layer.self_attn.o_proj(attn_out_vlm)
        current_vlm_h = residual_vlm + attn_out_vlm_proj

        residual_vlm_mlp = current_vlm_h
        normed_vlm_for_mlp = self.vlm_layer.post_attention_layernorm(current_vlm_h)
        mlp_out_vlm = self.vlm_layer.mlp(normed_vlm_for_mlp)
        current_vlm_h = residual_vlm_mlp + mlp_out_vlm

        attn_out_action_proj = self.action_layer.self_attn.o_proj(attn_out_action)
        current_action_h = residual_action + attn_out_action_proj

        residual_action_mlp = current_action_h
        normed_action_for_mlp = self.action_layer.post_attention_layernorm(current_action_h)
        mlp_out_action = self.action_layer.mlp(normed_action_for_mlp)
        current_action_h = residual_action_mlp + mlp_out_action

        return current_vlm_h, current_action_h

    def forward_with_cache(
            self,
            current_action_h: torch.Tensor,
            action_attention_mask: torch.Tensor,
            vlm_seq_len: int,
            action_seq_len: int,
            batch_size: int,
            vlm_k_rope_cached: torch.Tensor,
            vlm_v_cached: torch.Tensor,
            action_query_valid_1d: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Process a single layer for the Action Expert using cached VLM K/V.
        In this optimized path, only the Action Expert's Query is computed and used
        to attend to the combined VLM (cached) and Action (real-time) K/V pairs.
        """
        _num_heads = self.vlm_layer.self_attn.num_heads
        _head_dim = self.vlm_layer.self_attn.head_dim
        _num_key_value_heads = self.vlm_layer.self_attn.num_key_value_heads
        _num_key_value_groups = self.vlm_layer.self_attn.num_key_value_groups
        _hidden_size_attn_output = _num_heads * _head_dim

        residual_action = current_action_h
        normed_action_h = self.action_layer.input_layernorm(current_action_h)

        q_action = self.action_layer.self_attn.q_proj(normed_action_h)
        k_action = self.action_layer.self_attn.k_proj(normed_action_h)
        v_action = self.action_layer.self_attn.v_proj(normed_action_h)

        q_action = q_action.view(batch_size, action_seq_len, _num_heads, _head_dim).transpose(1, 2)
        k_action = k_action.view(batch_size, action_seq_len, _num_key_value_heads, _head_dim).transpose(1, 2)
        v_action = v_action.view(batch_size, action_seq_len, _num_key_value_heads, _head_dim).transpose(1, 2)

        combined_k = torch.cat([vlm_k_rope_cached, k_action], dim=2)
        combined_v = torch.cat([vlm_v_cached, v_action], dim=2)

        combined_k_repeated = repeat_kv(combined_k, _num_key_value_groups)
        combined_v_repeated = repeat_kv(combined_v, _num_key_value_groups)

        q_action = q_action.contiguous()
        combined_k_repeated = combined_k_repeated.contiguous()
        combined_v_repeated = combined_v_repeated.contiguous()

        attn_output_action = torch.nn.functional.scaled_dot_product_attention(
            q_action,
            combined_k_repeated,
            combined_v_repeated,
            attn_mask=action_attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )            

        attn_output_action = attn_output_action.transpose(1, 2).contiguous()
        attn_output_action = attn_output_action.reshape(batch_size, action_seq_len, _hidden_size_attn_output)

        # Zero-out outputs for padded action query positions to avoid numerical artifacts
        if action_query_valid_1d is not None:
            qv = action_query_valid_1d.to(attn_output_action.dtype).unsqueeze(-1)  # [B, A, 1]
            attn_output_action = attn_output_action * qv

        # Process Action path
        attn_out_action_proj = self.action_layer.self_attn.o_proj(attn_output_action)
        current_action_h = residual_action + attn_out_action_proj

        residual_action_mlp = current_action_h
        normed_action_for_mlp = self.action_layer.post_attention_layernorm(current_action_h)
        mlp_out_action = self.action_layer.mlp(normed_action_for_mlp)
        current_action_h = residual_action_mlp + mlp_out_action

        return current_action_h

def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Applies Rotary Position Embedding with Multimodal Sections to the query and key tensors (https://qwenlm.github.io/blog/qwen2-vl/).

    Explanation:
        Multimodal 3D rotary position embedding is an extension to 1D rotary position embedding. The input embedding
        sequence contains vision (images / videos) embedding and text embedding or just contains text embedding. For
        vision embedding part, we apply rotary position embedding on temporal, height and width dimension separately.
        Here we split the channel dimension to 3 chunks for the temporal, height and width rotary position embedding.
        For text embedding part, we just apply 1D rotary position embedding. The three rotary position index (temporal,
        height and width) of text embedding is always the same, so the text embedding rotary position embedding has no
        difference with modern LLMs.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        mrope_section(`List(int)`):
            Multimodal rope section is for channel dimension of temporal, height and width in rope calculation.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
    
# ============ BEV Semantic Head ============

class BEVSemanticHead(nn.Module):
    """BEV semantic segmentation head."""
    
    def __init__(
        self, 
        d_model: int, 
        num_map_queries: int = 64,
        num_classes: int = 5,
        output_size: Tuple[int, int] = (256, 256),
    ):
        super().__init__()
        self.d_model = d_model
        self.num_map_queries = num_map_queries
        self.num_classes = num_classes
        self.output_size = output_size
        
        # Calculate intermediate feature size
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
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 8->16
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 16->32
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 32->64
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),  # 64->128
            nn.Conv2d(32, num_classes, kernel_size=1),
            nn.Upsample(size=output_size, mode='bilinear', align_corners=False),  # 128->256
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


# ============ BEV Config ============

class Qwen2_5_VLConfigROSSMOE_BEV(PretrainedConfig):
    """Extended config with ROSS + BEV support."""
    
    def __init__(
        self,
        training_args=None,
        # BEV specific
        num_map_queries: int = 64,
        num_bev_classes: int = 5,
        bev_output_size: Tuple[int, int] = (256, 256),
        bev_loss_weight: float = 10.0,
        enable_bev_loss: bool = True,
        map_token_id: Optional[int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        if training_args is not None:
            self.action_loss_weight = getattr(training_args, "action_loss_weight", 1.0)
            self.vlm_action_loss_weight = getattr(training_args, "vlm_action_loss_weight", 0.0)
            self.vlm_ross_loss_weight = getattr(training_args, "vlm_ross_loss_weight", 0.0)
        else:
            self.action_loss_weight = 1.0
            self.vlm_action_loss_weight = 0.0
            self.vlm_ross_loss_weight = 0.0
        
        # BEV config
        self.num_map_queries = num_map_queries
        self.num_bev_classes = num_bev_classes
        self.bev_output_size = bev_output_size
        self.bev_loss_weight = bev_loss_weight
        self.enable_bev_loss = enable_bev_loss
        self.map_token_id = map_token_id


# ============ BEV Model ============

class Qwen2_5_VLForConditionalGenerationROSS_MOE_BEV(Qwen2_5_VLForConditionalGenerationROSS_MOE):
    """
    MOE model with BEV semantic map prediction support.
    Extends Qwen2_5_VLForConditionalGenerationROSS_MOE with:
    - <map> token embedding replacement
    - BEV semantic head
    - BEV loss computation
    """
    
    def __init__(self, config, action_config, ckpt_path=None, data_type=torch.float32):
        # Initialize parent MOE model
        super().__init__(config, action_config, ckpt_path, data_type)
        
        # BEV specific config
        self.num_map_queries = getattr(config, "num_map_queries", 64)
        self.num_bev_classes = getattr(config, "num_bev_classes", 5)
        self.bev_output_size = getattr(config, "bev_output_size", (256, 256))
        self.bev_loss_weight = getattr(config, "bev_loss_weight", 10.0)
        self.enable_bev_loss = getattr(config, "enable_bev_loss", True)
        self.map_token_id = getattr(config, "map_token_id", None)
        
        # Get hidden size from VLM
        vlm_hidden_size = self.qwen_ross.config.hidden_size
        
        # Learnable map query embeddings
        self.map_query_embedding = nn.Embedding(
            self.num_map_queries, vlm_hidden_size
        ).to(data_type)
        
        # BEV prediction head
        self.bev_head = BEVSemanticHead(
            d_model=vlm_hidden_size,
            num_map_queries=self.num_map_queries,
            num_classes=self.num_bev_classes,
            output_size=self.bev_output_size,
        ).to(data_type)
        
        print(f"[BEV] Initialized with {self.num_map_queries} queries, {self.num_bev_classes} classes, output {self.bev_output_size}")
    
    def forward(
            self,
            pre_action: torch.Tensor,
            action: Optional[torch.Tensor] = None,
            vlm_input_ids: torch.LongTensor = None,
            vlm_labels: torch.LongTensor = None,
            input_ids: torch.LongTensor = None,
            action_input_ids: torch.LongTensor = None,
            vlm_attention_mask: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
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
            # ROSS-specific inputs
            raw_pixel_values_vae: Optional[torch.Tensor] = None,
            image_token_masks: Optional[torch.Tensor] = None,
            action_future_masks: Optional[torch.Tensor] = None,
            frame_image_counts: Optional[torch.Tensor] = None,
            return_dict: Optional[bool] = None,
            vlm_k_rope_cache: Optional[List[torch.Tensor]] = None,
            vlm_v_cache: Optional[List[torch.Tensor]] = None,
            vlm_seq_len: Optional[int] = None,
            token=None,
            grpo_sample=False,
            # BEV-specific inputs (placeholders)
            map_token_masks: Optional[torch.Tensor] = None,        # [B, L] boolean
            bev_semantic_map_gt: Optional[torch.Tensor] = None,    # [B, H, W] long
    ) -> Union[Qwen2_5_VLCausalLMOutputWithPast, Qwen2_5_VLROSSOutput]:
        
        self.loss_dict.clear()
        use_cache = False
        return_dict = return_dict if return_dict is not None else True

        if self.training and not grpo_sample:
            if input_ids is None:
                raise ValueError("action_input_ids is required (starts with <boa>) in training")
            if pre_action is None:
                raise ValueError("pre_action must be provided to build the state token in training")
            if vlm_input_ids is None:
                raise ValueError("Training requires vlm_input_ids / vlm_attention_mask / pre_action")

            B = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]
            device = input_ids.device if input_ids is not None else inputs_embeds.device

            # Initial VLM embeddings
            if inputs_embeds is None:
                vlm_inputs_embeds = self.qwen_ross.model.language_model.embed_tokens(vlm_input_ids)
            
            # Build multimodal embeddings (replace image tokens)
            current_vlm_h = self.build_multimodal_inputs_embeds(vlm_input_ids, pixel_values, vlm_inputs_embeds, image_grid_thw)
            
            # ===== BEV: Replace <map> tokens with learnable embeddings =====
            if map_token_masks is not None and map_token_masks.any():
                # map_token_masks: [B, L]
                map_embeds = self.map_query_embedding.weight  # [num_map_queries, H]
                map_embeds = map_embeds.unsqueeze(0).expand(B, -1, -1)  # [B, num_map_queries, H]
                map_embeds_flat = map_embeds.reshape(-1, map_embeds.shape[-1])  # [B*num_map_queries, H]
                
                # Use masked_scatter to replace
                map_mask_expanded = map_token_masks.unsqueeze(-1).expand_as(current_vlm_h)
                current_vlm_h = current_vlm_h.masked_scatter(map_mask_expanded, map_embeds_flat.to(current_vlm_h.dtype))

            vlm_seq_len_local = current_vlm_h.shape[1]

            # Build state + action embeddings
            state_in = pre_action.view(B, -1).to(self.data_type)
            state_tok = self.state_projector(state_in).unsqueeze(1)

            action_emb = self.qwen_ross.model.language_model.embed_tokens(input_ids)
            action_emb = self.action_embed_projector(action_emb)
            current_action_h = torch.cat([state_tok, action_emb], dim=1)
            action_len_with_state = current_action_h.size(1)

            num_layers = len(self.qwen_ross.model.language_model.layers)
            vlm_position_embeddings = self.qwen_ross.model.language_model.rotary_emb(vlm_inputs_embeds, position_ids)

            action_valid_wo_state = (input_ids != self.pad_token_id)
            action_valid_with_state = torch.cat(
                [torch.ones(B, 1, dtype=torch.bool, device=device), action_valid_wo_state], dim=1)

            combined_attention_mask_4d = self.create_causal_style_attention_mask(
                vlm_seq_len=vlm_seq_len_local, action_seq_len=action_len_with_state,
                vlm_attention_mask_original=vlm_attention_mask, input_ids=vlm_input_ids, 
                batch_size=B, device=device, dtype=self.data_type, action_is_causal=True,
                action_attention_mask_1d=action_valid_with_state)

            # Layer-wise processing
            for layer_idx in range(num_layers):
                shared_layer = self.shared_layers[layer_idx]
                shared_layer.vlm_layer.to(device)
                shared_layer.action_layer.to(device)

                if self.gradient_checkpointing and self.training:
                    current_vlm_h, current_action_h = self._gradient_checkpointing_func(
                        shared_layer.__call__,
                        current_vlm_h, current_action_h, vlm_position_embeddings,
                        combined_attention_mask_4d, vlm_seq_len_local, action_len_with_state, B,
                    )
                else:
                    current_vlm_h, current_action_h = shared_layer(
                        current_vlm_h, current_action_h, vlm_position_embeddings,
                        combined_attention_mask_4d, vlm_seq_len_local, action_len_with_state, B,
                    )

            # Action loss
            h_final = self.action_expert.norm(current_action_h)
            logits = self.action_lm_head(h_final)
            loss = None
            if labels is not None:
                logits_for_loss = logits[:, :-1, :].contiguous()
                targets = labels.contiguous()
                loss_fct = CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(
                    logits_for_loss.view(-1, logits_for_loss.size(-1)),
                    targets.view(-1)
                )
                loss = loss * getattr(self, "action_loss_weight", 1.0)
                self.loss_dict["action_loss"] = float(loss.detach().cpu())

            # VLM action loss
            if self.vlm_action_loss_weight > 0.0:
                vlm_final = self.qwen_ross.model.language_model.norm(current_vlm_h)
                vlm_logits = self.qwen_ross.lm_head(vlm_final)
                vlm_logits = vlm_logits.float()
                shift_logits = vlm_logits[..., :-1, :].contiguous()
                shift_labels = vlm_labels[..., 1:].contiguous()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                shift_labels = shift_labels.to(shift_logits.device)
                vlm_action_loss = loss_fct(shift_logits, shift_labels)
                loss = loss + self.vlm_action_loss_weight * vlm_action_loss
                self.loss_dict["vlm_action_loss"] = float(vlm_action_loss.detach().cpu())

            # ROSS loss
            if self.vlm_ross_loss_weight > 0.0:
                if 'vlm_final' not in locals():
                    vlm_final = self.qwen_ross.model.language_model.norm(current_vlm_h)
                ross_loss = self.compute_ross_loss_and_update_logs(vlm_final, image_token_masks, action_future_masks, raw_pixel_values_vae)
                loss = loss + self.vlm_ross_loss_weight * ross_loss
                self.loss_dict["ross_loss"] = float(ross_loss.detach().cpu())

            # ===== BEV Loss =====
            bev_pred = None
            if map_token_masks is not None and map_token_masks.any():
                # Extract map features from VLM hidden states
                if 'vlm_final' not in locals():
                    vlm_final = self.qwen_ross.model.language_model.norm(current_vlm_h)
                
                # Extract map hidden states
                map_hidden = vlm_final[map_token_masks]  # [B*num_map_queries, H]
                map_hidden = map_hidden.view(B, self.num_map_queries, -1)  # [B, num_map_queries, H]
                
                # BEV prediction
                bev_pred = self.bev_head(map_hidden)  # [B, num_classes, H, W]
                
                # Compute BEV loss only if GT is provided
                if bev_semantic_map_gt is not None and self.enable_bev_loss:
                    bev_loss = F.cross_entropy(bev_pred, bev_semantic_map_gt.long())
                    loss = loss + self.bev_loss_weight * bev_loss
                    self.loss_dict["bev_loss"] = float(bev_loss.detach().cpu())

            if not return_dict:
                return logits, None, None, loss
            return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None)
        else:
            # Inference path - call parent's inference logic
            return super().forward(
                pre_action=pre_action, action=action, vlm_input_ids=vlm_input_ids,
                vlm_labels=vlm_labels, input_ids=input_ids, action_input_ids=action_input_ids,
                vlm_attention_mask=vlm_attention_mask, attention_mask=attention_mask,
                position_ids=position_ids, inputs_embeds=inputs_embeds, labels=labels,
                use_cache=use_cache, output_attentions=output_attentions,
                output_hidden_states=output_hidden_states, pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos, image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw, rope_deltas=rope_deltas,
                cache_position=cache_position, second_per_grid_ts=second_per_grid_ts,
                logits_to_keep=logits_to_keep, raw_pixel_values_vae=raw_pixel_values_vae,
                image_token_masks=image_token_masks, action_future_masks=action_future_masks,
                frame_image_counts=frame_image_counts, return_dict=return_dict,
                vlm_k_rope_cache=vlm_k_rope_cache, vlm_v_cache=vlm_v_cache,
                vlm_seq_len=vlm_seq_len, token=token, grpo_sample=grpo_sample,
            )


__all__ = [
    "Qwen2_5_VLConfigROSS",
    "Qwen2_5_VLForConditionalGenerationROSS",
    "Qwen2_5_VLROSSOutput",
    "Qwen2RossMoeConfigBuilder",
    "Qwen2_5_VLForConditionalGenerationROSS_MOE",
    "QwenPi0SharedLayer",
    "Qwen2_5_VLConfigROSSMOE_ACTIONEXPERT",
    "Qwen2_5_VLConfigROSSMOE",
    # BEV
    "BEVSemanticHead",
    "Qwen2_5_VLConfigROSSMOE_BEV",
    "Qwen2_5_VLForConditionalGenerationROSS_MOE_BEV",
]
