#!/usr/bin/env python3
"""
Unified training script for Qwen-VL VLA model
Combines Qwen-VL's multimodal capabilities with VLA action prediction
"""
import os
import sys
import logging
import pathlib
import pickle
from pathlib import Path
import torch
import transformers
from typing import Dict

# Add paths to import Qwen-VL components
qwen_vl_path = Path(__file__).parent.parent / "reference" / "Qwen2.5-VL" / "qwen-vl-finetune"
# qwen_vl_path = "/home/ma-user/work/lws/repo/VLA-Qwen/reference/Qwen2.5-VL/qwen-vl-finetune"
sys.path.append(str(qwen_vl_path))

transformers_path = Path(__file__).parent.parent / "reference"
sys.path.append(str(transformers_path))

# Ensure trainer monkey patches (optimizer grouping, print helpers) are applied
import qwenvl.train.trainer  # noqa: F401

# Optional utilities from trainer
from qwenvl.train.trainer import replace_qwen2_vl_attention_class  # noqa: F401

# Import Qwen-VL components
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    # Qwen2_5_VLForConditionalGenerationROSS,
    AutoTokenizer,
    AutoProcessor,
    Qwen2VLImageProcessor,
    # Trainer,
)
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig

# Import Custom components
# from reference.transformers.src.transformers.trainer import Trainer
from ross_trainer import RossTrainer as Trainer 
from reference.transformers.src.transformers.models.qwen2_5_vl.modeling_qwen2_5_vl_ross import (
    Qwen2_5_VLConfigROSS, 
    Qwen2_5_VLForConditionalGenerationROSS,
    Qwen2_5_VLForConditionalGenerationROSS_MOE,
    Qwen2_5_VLConfigROSSMOE_ACTIONEXPERT,
    Qwen2_5_VLConfigROSSMOE
)

import transformers.trainer as _trainer
from transformers.utils import import_utils as _iu

def _skip_check():
    return None

_iu.check_torch_load_is_safe = _skip_check
_trainer.check_torch_load_is_safe = _skip_check

from qwenvl.dataset.data_qwen_vla import (
    make_supervised_data_module_huawei2_vla_ross_multiview4_5s,
    make_supervised_data_module_huawei2_vla_ross_moe,
    make_supervised_data_module_huawei2_vla_ross_moe_multiview4,
    make_supervised_data_module_huawei2_vla_ross_moe_multiview4_5s
)

from adsdata.adsData import (
    make_supervised_data_module_huawei2_vla_ross_multiview4_5kw,
    make_supervised_data_module_huawei2_vla_ross_moe_multiview4_5kw
)

from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)

# Import VLA-specific components
sys.path.append(str(qwen_vl_path / "qwenvl" / "utils"))
from token_utils import smart_load_model_and_tokenizer, prepare_action_tokenizer_mapping, check_and_add_vla_tokens
from safetensors.torch import load_file

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
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

def setup_vla_model_and_tokenizer(model_args, training_args):
    """
    Complete VLA model setup: load model, configure training parameters, and set up tokenizer
    """
    
    # Select model class by model_type (persistent scheme for ROSS)
    model_type = getattr(model_args, "model_type", "qwen2.5vl")
    ARCH_BY_MODEL_TYPE = {
        "qwen2.5vl": Qwen2_5_VLForConditionalGeneration,
        "qwen2.5vl_ross": Qwen2_5_VLForConditionalGenerationROSS,
        "qwen2.5vl_ross_moe": Qwen2_5_VLForConditionalGenerationROSS_MOE,
    }
    if model_type not in ARCH_BY_MODEL_TYPE:
        raise ValueError(f"Unsupported model_type: {model_type}")
    model_class = ARCH_BY_MODEL_TYPE[model_type]

    qwen_addr = '/preset_model/data/external/personal/t00865236/llm4drive_utils/LLM_config/Qwen25VL'
    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    # Build config: base -> (optional) ROSS wrapper
    if model_type == "qwen2.5vl_ross_moe":
        base_cfg = Qwen2_5_VLConfig.from_pretrained(
            qwen_addr, trust_remote_code=True
        )
        action_cfg = Qwen2_5_VLConfigROSSMOE_ACTIONEXPERT(
            **base_cfg.to_dict()
        )
        moe_cfg = Qwen2_5_VLConfigROSSMOE(
            **base_cfg.to_dict(),
            training_args=training_args
        )
        model = Qwen2_5_VLForConditionalGenerationROSS_MOE(
            config=moe_cfg,               
            action_config=action_cfg, 
            ckpt_path=model_args.model_name_or_path,
            data_type=torch.bfloat16
        ).cuda()
        # model = Qwen2_5_VLForConditionalGenerationROSS_MOE(
        #     config=moe_cfg,               
        #     action_config=action_cfg, 
        #     ckpt_path='/home/ma-user/work/lws/repo/VLA-Qwen/logs/reload_vae_qwen25vl_fast_ross_7000w_30k',
        #     data_type=torch.bfloat16
        # )

        # model_weight = load_file('/home/ma-user/work/wxm/debug2/checkpoint-2000/model.safetensors')
        # model.load_state_dict(model_weight)
    elif model_type == "qwen2.5vl_ross":
        base_cfg = Qwen2_5_VLConfig.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=True
        )
        ross_cfg = Qwen2_5_VLConfigROSS(
            **base_cfg.to_dict(),
            enable_ross=True,
            extract_image_hidden=True,
            extract_action_hidden=True,
            sd_model_path=model_args.sd_model_path,
        )
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            config=ross_cfg,
            cache_dir=training_args.cache_dir,
            attn_implementation="flash_attention_2",
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            trust_remote_code=True,
        )

        from diffusers import AutoencoderKL
        from reference.transformers.src.transformers.models.qwen2_5_vl.modeling_ross.unet_2d_condition import UNet2DConditionModel
        model.vae = AutoencoderKL.from_pretrained(model_args.sd_model_path.replace("/unet", "/vae"), torch_dtype=model.dtype)
        model.vae.eval()
        model.vae.requires_grad_(False)
        model.denoiser.unet = UNet2DConditionModel.from_pretrained(model_args.sd_model_path, torch_dtype=model.dtype)
        model.denoiser.unet.train()
        model.denoiser.unet.requires_grad_(True)     

    else:
        # Use smart loader for baseline
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation="flash_attention_2",
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            trust_remote_code=True,
        )
    
    tokenizer, model, _ = check_and_add_vla_tokens(tokenizer, model)
    model.boa_token_id = tokenizer.encode(tokenizer.boa_token)[0]
    model.eoa_token_id = tokenizer.encode(tokenizer.eoa_token)[0]
    model.pad_token_id = tokenizer.encode(tokenizer.pad_token)[0]
    
    rank0_print(f"Loaded {model_class.__name__} from {model_args.model_name_or_path}")

    
    # Load image processor
    if model_type.startswith("qwen2.5vl"):
        image_processor = AutoProcessor.from_pretrained(qwen_addr).image_processor
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
    if model_type == "qwen2.5vl_ross_moe":
        if getattr(model_args, 'tune_mm_vision', True):
            for n, p in model.qwen_ross.visual.named_parameters():
                p.requires_grad = True
            rank0_print("Vision encoder: TRAINABLE")
        else:
            for n, p in model.qwen_ross.visual.named_parameters():
                p.requires_grad = False
            rank0_print("Vision encoder: FROZEN")
        
        # Vision-language connector (merger)
        if getattr(model_args, 'tune_mm_mlp', True):
            for n, p in model.qwen_ross.visual.merger.named_parameters():
                p.requires_grad = True
            rank0_print("Vision-language merger: TRAINABLE")
        else:
            for n, p in model.qwen_ross.visual.merger.named_parameters():
                p.requires_grad = False
            rank0_print("Vision-language merger: FROZEN")
        
        # Language model
        if getattr(model_args, 'tune_mm_llm', True):
            for n, p in model.qwen_ross.model.language_model.named_parameters():
                p.requires_grad = True
            model.qwen_ross.lm_head.requires_grad = True # 与qwen_ross.lm_head共享内存
            rank0_print("Language model: TRAINABLE")
        else:
            for n, p in model.qwen_ross.model.language_model.named_parameters():
                p.requires_grad = False
            model.qwen_ross.lm_head.requires_grad = False # 与qwen_ross.lm_head共享内存
            rank0_print("Language model: FROZEN")
        
        if model_type == "qwen2.5vl_ross" or "qwen2.5vl_ross_moe":
            for n, p in model.qwen_ross.denoiser.named_parameters():
                p.requires_grad = True
            rank0_print("Denoiser: TRAINABLE")
    else:
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
        
        if model_type == "qwen2.5vl_ross":
            for n, p in model.denoiser.named_parameters():
                p.requires_grad = True
            rank0_print("Denoiser: TRAINABLE")
    
    def count(m):
        tot = sum(p.numel() for p in m.parameters())
        trn = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return tot, trn

    tot, trn = count(model)
    if local_rank in (0, None):
        print(f"[PARAM] total={tot/1e6:.1f}M, trainable={trn/1e6:.1f}M")

    # 粗估初始化上界（单卡、ZeRO-3 之前）
    bytes_per_param = 2   # bf16 权重
    adam_state = 16       # Adam m,v 各 8B
    master_fp32 = 4       # FP32 master
    rough = trn*(bytes_per_param+adam_state+master_fp32)/(1024**3)
    print(f"[EST] optimizer+master upper bound ~= {rough:.1f} GiB")

    return model, tokenizer, image_processor, model_type

def setup_vla_data_args(data_args, image_processor, model_type):
    """Setup data arguments for VLA training"""
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
    
    # VLA-specific args (with defaults)
    data_args.use_actions = getattr(data_args, 'use_actions', True)
    data_args.actions_format = getattr(data_args, 'actions_format', 'fast')
    data_args.action_tokenizer_path = getattr(data_args, 'action_tokenizer_path', None)
    data_args.action_dim = getattr(data_args, 'action_dim', 3)  # steering, acceleration, braking
    
    # Driving-specific args
    data_args.use_previous_actions = getattr(data_args, 'use_previous_actions', False)
    data_args.cur_frame_idx = getattr(data_args, 'cur_frame_idx', 3)
    
    rank0_print(f"VLA Data Config:")
    rank0_print(f"  - use_actions: {data_args.use_actions}")
    rank0_print(f"  - actions_format: {data_args.actions_format}")
    rank0_print(f"  - action_tokenizer_path: {data_args.action_tokenizer_path}")
    rank0_print(f"  - action_dim: {data_args.action_dim}")
    
    return data_args

def train():
    global local_rank
    
    # Parse arguments
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    rank0_print("="*50)
    rank0_print("Qwen-VL VLA Training Setup")
    rank0_print("="*50)
    
    # Setup model and tokenizer (includes all training configuration)
    model, tokenizer, image_processor, model_type = setup_vla_model_and_tokenizer(model_args, training_args)
    
    # Setup data
    data_args = setup_vla_data_args(data_args, image_processor, model_type)

    # Apply flash-attn varlen + causal mask override when requested (keeps parity with reference trainer)
    if getattr(data_args, "data_flatten", False):
        replace_qwen2_vl_attention_class()
    
    if data_args.dataset_type == "huawei2va_ross_moe":
        data_module = make_supervised_data_module_huawei2_vla_ross_moe(tokenizer=tokenizer, data_args=data_args)
    elif data_args.dataset_type == "huawei2va_ross_moe_multiview4":
        data_module = make_supervised_data_module_huawei2_vla_ross_moe_multiview4(tokenizer=tokenizer, data_args=data_args)
    elif data_args.dataset_type == "huawei2va_ross_moe_multiview4_5s":
        data_module = make_supervised_data_module_huawei2_vla_ross_moe_multiview4_5s(tokenizer=tokenizer, data_args=data_args)
    elif data_args.dataset_type == "huawei2va_ross_multiview4_5s":
        data_module = make_supervised_data_module_huawei2_vla_ross_multiview4_5s(tokenizer=tokenizer, data_args=data_args)
    elif data_args.dataset_type == "huawei2va_ross_multiview4_5kw":
        data_module = make_supervised_data_module_huawei2_vla_ross_multiview4_5kw(tokenizer=tokenizer, data_args=data_args)
    elif data_args.dataset_type == "huawei2va_ross_moe_5kw":
        data_module = make_supervised_data_module_huawei2_vla_ross_moe_multiview4_5kw(tokenizer=tokenizer, data_args=data_args)
    else:
        raise ValueError(f"Unsupport dataset_type {data_args.dataset_type}")
    rank0_print("Using standard VLA data format")
    
        
    # Create trainer
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        **data_module
    )
    
    # Print dataloader len
    if torch.distributed.get_rank() == 0:
        rank0_print(f"dataloader len:")
        rank0_print(f"{len(trainer.get_train_dataloader())}")
        rank0_print(f"{training_args.gradient_accumulation_steps}")
        print("len(train_dataset) =", len(data_module["train_dataset"]))
        print("per_device_train_batch_size =", training_args.per_device_train_batch_size)
        print("TrainingArguments.world_size =", training_args.world_size)
        print("torch.distributed.get_world_size() =", torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1)
    
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
    
    # Save tokenizer (with new VLA tokens)
    tokenizer.save_pretrained(training_args.output_dir)
    
    model.config.use_cache = True
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    rank0_print("="*50)
    rank0_print("Training completed successfully!")
    rank0_print(f"Model saved to: {training_args.output_dir}")
    rank0_print("="*50)

if __name__ == "__main__":
    train()
