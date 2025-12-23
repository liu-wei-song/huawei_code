#!/usr/bin/env python3
"""
ADS Agent Detection + BEV Semantic Map 训练脚本

基于 train_qwen_vla.py 结构，支持:
- Agent Detection (4类: empty, vehicle, pedestrian, other)
- BEV Semantic Map (4类: background, static, vehicle, pedestrian)
- 可选的 ROSS 重建损失

使用方式:
    python huawei_code/train_agent_bev_ads_example.py \
        --model_name_or_path Qwen/Qwen2.5-VL-7B \
        --data_path /path/to/ads/data \
        --output_dir ./outputs/agent_bev_ads \
        --dataset_type ads_agent_bev
"""
import os
import sys
import logging
import pathlib
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    Qwen2VLImageProcessor,
)

# Add paths to import Qwen-VL components
qwen_vl_path = Path(__file__).parent.parent / "reference" / "Qwen2.5-VL" / "qwen-vl-finetune"
sys.path.append(str(qwen_vl_path))

# Ensure trainer monkey patches are applied
import qwenvl.train.trainer  # noqa: F401
from qwenvl.train.trainer import replace_qwen2_vl_attention_class  # noqa: F401

from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig

# Import Custom components
from ross_trainer import RossTrainer as Trainer
from reference.transformers.src.transformers.models.qwen2_5_vl.modeling_qwen2_5_vl_ross import (
    Qwen2_5_VLConfigROSS, 
    Qwen2_5_VLForConditionalGenerationROSS,
)

import transformers.trainer as _trainer
from transformers.utils import import_utils as _iu

def _skip_check():
    return None
 
_iu.check_torch_load_is_safe = _skip_check
_trainer.check_torch_load_is_safe = _skip_check

# Import VLA token utilities
sys.path.append(str(qwen_vl_path / "qwenvl" / "utils"))
from token_utils import check_and_add_vla_tokens

# Import ADS Agent BEV dataset
from huawei_code.data_qwen_vla_agent_bev_ads import (
    make_supervised_data_module_ads_vla_agent_bev,
)

# Import arguments (使用与 train_qwen_vla.py 相同的参数结构)
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)

logger = logging.getLogger(__name__)
local_rank = None


def rank0_print(*args):
    if local_rank == 0 or local_rank is None:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def setup_ads_agent_bev_model_and_tokenizer(model_args, training_args):
    """
    Setup model and tokenizer for ADS Agent BEV training.
    Similar to setup_vla_model_and_tokenizer in train_qwen_vla.py.
    """
    # Model type selection
    model_type = getattr(model_args, "model_type", "qwen2.5vl")
    
    # Currently support baseline and ROSS variants
    ARCH_BY_MODEL_TYPE = {
        "qwen2.5vl": "Qwen2_5_VLForConditionalGeneration",
        "qwen2.5vl_ross": "Qwen2_5_VLForConditionalGenerationROSS",
        "qwen2.5vl_agent_bev": "Qwen2_5_VLForConditionalGenerationAgentBEV_ADS",
    }
    
    if model_type not in ARCH_BY_MODEL_TYPE:
        raise ValueError(f"Unsupported model_type: {model_type}. Supported: {list(ARCH_BY_MODEL_TYPE.keys())}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    
    # Build config and model based on type
    if model_type == "qwen2.5vl_ross":
        base_cfg = Qwen2_5_VLConfig.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=True
        )
        
        if "enable_ross" in base_cfg.to_dict().keys():
            rank0_print("Loading ROSS model without manual VAE/UNet initialization")
            ross_cfg = Qwen2_5_VLConfigROSS(**base_cfg.to_dict())
            model = Qwen2_5_VLForConditionalGenerationROSS.from_pretrained(
                model_args.model_name_or_path,
                config=ross_cfg,
                cache_dir=training_args.cache_dir,
                attn_implementation="flash_attention_2",
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                trust_remote_code=True,
            )
        else:
            rank0_print("Loading ROSS model with manual VAE/UNet initialization")
            ross_cfg = Qwen2_5_VLConfigROSS(
                **base_cfg.to_dict(),
                enable_ross=True,
                extract_image_hidden=True,
                extract_action_hidden=True,
                sd_model_path=model_args.sd_model_path,
            )
            model = Qwen2_5_VLForConditionalGenerationROSS.from_pretrained(
                model_args.model_name_or_path,
                config=ross_cfg,
                cache_dir=training_args.cache_dir,
                attn_implementation="flash_attention_2",
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                trust_remote_code=True,
            )
            # Manually load VAE and UNet
            from diffusers import AutoencoderKL
            from reference.transformers.src.transformers.models.qwen2_5_vl.modeling_ross.unet_2d_condition import UNet2DConditionModel
            model.vae = AutoencoderKL.from_pretrained(
                model_args.sd_model_path.replace("/unet", "/vae"), 
                torch_dtype=model.dtype
            )
            model.vae.eval()
            model.vae.requires_grad_(False)
            model.denoiser.unet = UNet2DConditionModel.from_pretrained(
                model_args.sd_model_path, 
                torch_dtype=model.dtype
            )
            model.denoiser.unet.train()
            model.denoiser.unet.requires_grad_(True)
            
    elif model_type == "qwen2.5vl_agent_bev":
        # Load Agent BEV specific model
        from huawei_code.modeling_qwen2_5_vl_agent_bev_ads import (
            Qwen2_5_VLConfigAgentBEV_ADS,
            Qwen2_5_VLForConditionalGenerationAgentBEV_ADS,
        )
        
        base_cfg = Qwen2_5_VLConfig.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=True
        )
        
        # Create Agent BEV config with ADS-specific settings
        agent_bev_cfg = Qwen2_5_VLConfigAgentBEV_ADS(
            **base_cfg.to_dict(),
            num_agent_queries=getattr(model_args, 'num_agent_queries', 20),
            num_map_queries=getattr(model_args, 'num_map_queries', 64),
            num_agent_classes=4,  # ADS: empty, vehicle, pedestrian, other
            num_bev_classes=4,    # ADS: background, static, vehicle, pedestrian
            enable_agent_loss=getattr(model_args, 'enable_agent_loss', True),
            enable_bev_loss=getattr(model_args, 'enable_bev_loss', True),
        )
        
        model = Qwen2_5_VLForConditionalGenerationAgentBEV_ADS.from_pretrained(
            model_args.model_name_or_path,
            config=agent_bev_cfg,
            cache_dir=training_args.cache_dir,
            attn_implementation="flash_attention_2",
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            trust_remote_code=True,
        )
        
    else:
        # Baseline Qwen2.5-VL
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation="flash_attention_2",
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            trust_remote_code=True,
        )
    
    # Add VLA tokens
    tokenizer, model, _ = check_and_add_vla_tokens(tokenizer, model)
    
    rank0_print(f"Loaded model type: {model_type} from {model_args.model_name_or_path}")
    
    # Reload VAE/UNet if needed (for ROSS after token embedding resize)
    if model_type == "qwen2.5vl_ross" and "enable_ross" not in base_cfg.to_dict().keys():
        from diffusers import AutoencoderKL
        from reference.transformers.src.transformers.models.qwen2_5_vl.modeling_ross.unet_2d_condition import UNet2DConditionModel
        model.vae = AutoencoderKL.from_pretrained(
            model_args.sd_model_path.replace("/unet", "/vae"), 
            torch_dtype=model.dtype
        )
        model.vae.eval()
        model.vae.requires_grad_(False)
        model.denoiser.unet = UNet2DConditionModel.from_pretrained(
            model_args.sd_model_path, 
            torch_dtype=model.dtype
        )
        model.denoiser.unet.train()
        model.denoiser.unet.requires_grad_(True)
    
    # Load image processor
    if model_type.startswith("qwen2.5vl"):
        image_processor = AutoProcessor.from_pretrained(model_args.model_name_or_path).image_processor
    else:
        image_processor = Qwen2VLImageProcessor.from_pretrained(model_args.model_name_or_path)
    
    # Set tokenizer parameters
    tokenizer.model_max_length = training_args.model_max_length
    tokenizer.padding_side = "right"
    
    # Configure model for training
    model.config.use_cache = False
    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    
    # Configure which parts of the model to train
    # Vision encoder
    if getattr(model_args, 'tune_mm_vision', True):
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
        rank0_print("Vision encoder: TRAINABLE")
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False
        rank0_print("Vision encoder: FROZEN")
    
    # Vision-language connector (merger)
    if getattr(model_args, 'tune_mm_mlp', True):
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
        rank0_print("Vision-language merger: TRAINABLE")
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False
        rank0_print("Vision-language merger: FROZEN")
    
    # Language model
    if getattr(model_args, 'tune_mm_llm', True):
        for n, p in model.model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
        rank0_print("Language model: TRAINABLE")
    else:
        for n, p in model.model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False
        rank0_print("Language model: FROZEN")
    
    # ROSS specific
    if model_type == "qwen2.5vl_ross":
        for n, p in model.denoiser.named_parameters():
            p.requires_grad = True
        rank0_print("Denoiser: TRAINABLE")
        for n, p in model.vae.named_parameters():
            p.requires_grad = False
        rank0_print("VAE: FROZEN")
    
    # Agent BEV specific heads
    if model_type == "qwen2.5vl_agent_bev":
        if hasattr(model, 'agent_head'):
            for n, p in model.agent_head.named_parameters():
                p.requires_grad = True
            rank0_print("Agent Head: TRAINABLE")
        if hasattr(model, 'bev_head'):
            for n, p in model.bev_head.named_parameters():
                p.requires_grad = True
            rank0_print("BEV Head: TRAINABLE")
    
    # Print parameter statistics
    def count(m):
        tot = sum(p.numel() for p in m.parameters())
        trn = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return tot, trn

    tot, trn = count(model)
    if local_rank in (0, None):
        print(f"[PARAM] total={tot/1e6:.1f}M, trainable={trn/1e6:.1f}M")

    # Rough memory estimate
    bytes_per_param = 2   # bf16
    adam_state = 16       # Adam m,v
    master_fp32 = 4       # FP32 master
    rough = trn * (bytes_per_param + adam_state + master_fp32) / (1024**3)
    print(f"[EST] optimizer+master upper bound ~= {rough:.1f} GiB")

    return model, tokenizer, image_processor, model_type


def setup_ads_data_args(data_args, image_processor, model_type):
    """Setup data arguments for ADS Agent BEV training."""
    # Standard Qwen-VL data args
    data_args.image_processor = image_processor
    data_args.model_type = model_type
    
    if hasattr(data_args, 'max_pixels'):
        data_args.image_processor.max_pixels = getattr(data_args, 'max_pixels', 1280*28*28)
    if hasattr(data_args, 'min_pixels'):
        data_args.image_processor.min_pixels = getattr(data_args, 'min_pixels', 256*28*28)
        
    data_args.image_processor.size = {
        "longest_edge": data_args.max_pixels,
        "shortest_edge": data_args.min_pixels
    }
    
    # VLA-specific args
    data_args.use_actions = getattr(data_args, 'use_actions', True)
    data_args.actions_format = getattr(data_args, 'actions_format', 'fast')
    data_args.action_tokenizer_path = getattr(data_args, 'action_tokenizer_path', None)
    data_args.action_dim = getattr(data_args, 'action_dim', 3)
    
    # Agent BEV specific args
    data_args.enable_agent_tokens = getattr(data_args, 'enable_agent_tokens', True)
    data_args.enable_bev_tokens = getattr(data_args, 'enable_bev_tokens', True)
    data_args.num_agent_queries = getattr(data_args, 'num_agent_queries', 20)
    data_args.num_map_queries = getattr(data_args, 'num_map_queries', 64)
    data_args.bev_pixel_width = getattr(data_args, 'bev_pixel_width', 256)
    data_args.bev_pixel_height = getattr(data_args, 'bev_pixel_height', 128)
    data_args.bev_pixel_size = getattr(data_args, 'bev_pixel_size', 0.25)
    
    rank0_print(f"ADS Agent BEV Data Config:")
    rank0_print(f"  - use_actions: {data_args.use_actions}")
    rank0_print(f"  - enable_agent_tokens: {data_args.enable_agent_tokens}")
    rank0_print(f"  - enable_bev_tokens: {data_args.enable_bev_tokens}")
    rank0_print(f"  - num_agent_queries: {data_args.num_agent_queries}")
    rank0_print(f"  - num_map_queries: {data_args.num_map_queries}")
    rank0_print(f"  - bev_size: {data_args.bev_pixel_height} x {data_args.bev_pixel_width}")
    
    return data_args


def train():
    global local_rank
    
    # Parse arguments
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Set random seeds
    seed = training_args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    rank0_print("=" * 60)
    rank0_print("ADS Agent Detection + BEV Semantic Map Training")
    rank0_print("=" * 60)
    
    # Setup model and tokenizer
    model, tokenizer, image_processor, model_type = setup_ads_agent_bev_model_and_tokenizer(
        model_args, training_args
    )
    
    # Setup data
    data_args = setup_ads_data_args(data_args, image_processor, model_type)

    # Apply flash-attn varlen + causal mask override when requested
    if getattr(data_args, "data_flatten", False):
        replace_qwen2_vl_attention_class()
    
    # Create data module based on dataset type
    dataset_type = getattr(data_args, 'dataset_type', 'ads_agent_bev')
    
    if dataset_type == "ads_agent_bev":
        data_module = make_supervised_data_module_ads_vla_agent_bev(
            tokenizer=tokenizer, 
            data_args=data_args
        )
        rank0_print("Using ADS Agent BEV data format")
    else:
        # Fallback to standard ADS data modules
        from huawei_code.adsData import (
            make_supervised_data_module_huawei2_vla_ross_multiview4_5kw,
            make_supervised_data_module_huawei2_vla_ross_moe_multiview4_5kw,
        )
        if dataset_type == "huawei2va_ross_multiview4":
            data_module = make_supervised_data_module_huawei2_vla_ross_multiview4_5kw(
                tokenizer=tokenizer, 
                data_args=data_args
            )
        elif dataset_type == "huawei2va_ross_moe_multiview4":
            data_module = make_supervised_data_module_huawei2_vla_ross_moe_multiview4_5kw(
                tokenizer=tokenizer, 
                data_args=data_args
            )
        else:
            raise ValueError(f"Unsupported dataset_type: {dataset_type}")
        rank0_print(f"Using {dataset_type} data format")
    
    # Create trainer (use standard RossTrainer, no custom trainer)
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        **data_module
    )
    
    # Print dataloader info
    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        rank0_print(f"Dataloader length: {len(trainer.get_train_dataloader())}")
        rank0_print(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
        print("len(train_dataset) =", len(data_module["train_dataset"]))
        print("per_device_train_batch_size =", training_args.per_device_train_batch_size)
        print("TrainingArguments.world_size =", training_args.world_size)
        print("torch.distributed.get_world_size() =", torch.distributed.get_world_size())
    
    # Start training
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        rank0_print("Checkpoint found, resuming training...")
        trainer.train(resume_from_checkpoint=True)
    else:
        rank0_print("Starting training from scratch...")
        trainer.train()
    
    # Save final model
    trainer.save_state()
    
    # Save image processor
    if hasattr(data_args, 'image_processor'):
        data_args.image_processor.save_pretrained(training_args.output_dir)
    
    # Save tokenizer
    tokenizer.save_pretrained(training_args.output_dir)
    
    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    rank0_print("=" * 60)
    rank0_print("Training completed successfully!")
    rank0_print(f"Model saved to: {training_args.output_dir}")
    rank0_print("=" * 60)


if __name__ == "__main__":
    train()


# ============ 使用示例 ============

"""
# 示例 1: ADS Agent BEV 联合训练
python huawei_code/train_agent_bev_ads_example.py \
    --model_name_or_path Qwen/Qwen2.5-VL-7B \
    --model_type qwen2.5vl_agent_bev \
    --dataset_type ads_agent_bev \
    --data_path /path/to/ads/data \
    --output_dir ./outputs/agent_bev_ads \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 3 \
    --bf16 True

# 示例 2: 使用 ROSS 模型
python huawei_code/train_agent_bev_ads_example.py \
    --model_name_or_path Qwen/Qwen2.5-VL-7B \
    --model_type qwen2.5vl_ross \
    --dataset_type huawei2va_ross_multiview4 \
    --data_path /path/to/ads/data \
    --sd_model_path /path/to/stable-diffusion/unet \
    --output_dir ./outputs/ross_ads \
    --learning_rate 2e-5 \
    --bf16 True

# 示例 3: 冻结视觉编码器
python huawei_code/train_agent_bev_ads_example.py \
    --model_name_or_path Qwen/Qwen2.5-VL-7B \
    --model_type qwen2.5vl_agent_bev \
    --dataset_type ads_agent_bev \
    --data_path /path/to/ads/data \
    --output_dir ./outputs/agent_bev_frozen_vision \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --learning_rate 2e-5 \
    --gradient_checkpointing True
"""
