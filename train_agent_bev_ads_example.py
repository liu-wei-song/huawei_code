"""
ADS Agent Detection + BEV Semantic Map 训练示例

该脚本展示如何将 Agent/BEV 监督集成到 Huawei ADS 训练流程中。

使用方式:
    python train_agent_bev_ads_example.py \
        --model_name_or_path Qwen/Qwen2.5-VL-7B \
        --data_path /path/to/ads/data \
        --output_dir ./outputs/agent_bev_ads \
        --enable_agent_loss True \
        --enable_bev_loss True

BEV 类别 (4类):
    0: background
    1: static_objects (障碍物、curb等)
    2: vehicles
    3: pedestrians

Agent 类别 (4类):
    0: empty (无目标)
    1: vehicle
    2: pedestrian
    3: other
"""

import os
import sys
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import torch
import transformers
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
)

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入 ADS 专用模块
from huawei_code.data_qwen_vla_agent_bev_ads import (
    LazySupervisedHuawei2VAROSSAgentBEVDataset,
    DataCollatorForAgentBEVDataset,
    make_supervised_data_module_huawei2_vla_agent_bev,
)
from huawei_code.modeling_qwen2_5_vl_agent_bev_ads import (
    Qwen2_5_VLConfigAgentBEV_ADS,
    Qwen2_5_VLForConditionalGenerationAgentBEV_ADS,
)

logger = logging.getLogger(__name__)


# ============ 参数定义 ============

@dataclass
class ModelArguments:
    """模型相关参数"""
    model_name_or_path: str = field(
        default="Qwen/Qwen2.5-VL-7B",
        metadata={"help": "预训练模型路径或 HuggingFace Hub 名称"},
    )
    freeze_vision_encoder: bool = field(
        default=True,
        metadata={"help": "是否冻结视觉编码器"},
    )
    freeze_llm_backbone: bool = field(
        default=False,
        metadata={"help": "是否冻结 LLM backbone"},
    )


@dataclass
class DataArguments:
    """数据相关参数"""
    data_path: str = field(
        default=None,
        metadata={"help": "训练数据路径"},
    )
    image_folder: Optional[str] = field(
        default=None,
        metadata={"help": "图像文件夹路径"},
    )
    
    # BEV 配置
    bev_pixel_width: int = field(
        default=256,
        metadata={"help": "BEV 图像宽度 (像素)"},
    )
    bev_pixel_height: int = field(
        default=128,
        metadata={"help": "BEV 图像高度 (像素)"},
    )
    bev_pixel_size: float = field(
        default=0.25,
        metadata={"help": "BEV 像素尺寸 (米/像素)"},
    )
    
    # Agent/Map Query 配置
    num_agent_queries: int = field(
        default=20,
        metadata={"help": "Agent query 数量"},
    )
    num_map_queries: int = field(
        default=64,
        metadata={"help": "Map query 数量"},
    )
    
    # Token 开关
    enable_agent_tokens: bool = field(
        default=True,
        metadata={"help": "是否启用 agent tokens"},
    )
    enable_bev_tokens: bool = field(
        default=True,
        metadata={"help": "是否启用 bev tokens"},
    )
    
    # 继承自父类的参数
    action_hz: int = field(default=5, metadata={"help": "动作采样频率"})
    norm_path: str = field(default="", metadata={"help": "归一化参数路径"})
    action_type: str = field(default="delta_traj", metadata={"help": "动作类型"})


@dataclass
class AgentBEVArguments:
    """Agent/BEV 损失相关参数"""
    enable_agent_loss: bool = field(
        default=True,
        metadata={"help": "是否启用 agent detection loss"},
    )
    enable_bev_loss: bool = field(
        default=True,
        metadata={"help": "是否启用 BEV semantic map loss"},
    )
    agent_class_weight: float = field(
        default=1.0,
        metadata={"help": "Agent 分类 loss 权重"},
    )
    agent_box_weight: float = field(
        default=1.0,
        metadata={"help": "Agent 边界框 loss 权重"},
    )
    bev_weight: float = field(
        default=1.0,
        metadata={"help": "BEV 分割 loss 权重"},
    )


# ============ 训练器 ============

class AgentBEVTrainer(Trainer):
    """
    自定义 Trainer，支持 Agent/BEV 多任务训练
    """
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        计算总损失 = LM loss + Agent loss + BEV loss
        """
        # 提取 Agent/BEV 相关输入
        agent_token_masks = inputs.pop("agent_token_masks", None)
        map_token_masks = inputs.pop("map_token_masks", None)
        agent_states_gt = inputs.pop("agent_states_gt", None)
        agent_labels_gt = inputs.pop("agent_labels_gt", None)
        bev_semantic_map_gt = inputs.pop("bev_semantic_map_gt", None)
        
        # Forward
        outputs = model(
            **inputs,
            agent_token_masks=agent_token_masks,
            map_token_masks=map_token_masks,
            agent_states_gt=agent_states_gt,
            agent_labels_gt=agent_labels_gt,
            bev_semantic_map_gt=bev_semantic_map_gt,
        )
        
        # 总损失
        loss = outputs.loss
        
        # 记录分项损失
        if self.state.global_step % 10 == 0:
            if hasattr(outputs, 'agent_loss') and outputs.agent_loss is not None:
                logger.info(f"Step {self.state.global_step}: agent_loss = {outputs.agent_loss.item():.4f}")
            if hasattr(outputs, 'bev_loss') and outputs.bev_loss is not None:
                logger.info(f"Step {self.state.global_step}: bev_loss = {outputs.bev_loss.item():.4f}")
        
        return (loss, outputs) if return_outputs else loss


# ============ 主函数 ============

def main():
    # 解析参数
    parser = HfArgumentParser((
        ModelArguments,
        DataArguments,
        AgentBEVArguments,
        TrainingArguments,
    ))
    model_args, data_args, agent_bev_args, training_args = parser.parse_args_into_dataclasses()
    
    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    
    logger.info("=" * 60)
    logger.info("ADS Agent Detection + BEV Semantic Map Training")
    logger.info("=" * 60)
    logger.info(f"Model: {model_args.model_name_or_path}")
    logger.info(f"Data: {data_args.data_path}")
    logger.info(f"Agent queries: {data_args.num_agent_queries}")
    logger.info(f"Map queries: {data_args.num_map_queries}")
    logger.info(f"BEV size: {data_args.bev_pixel_height} x {data_args.bev_pixel_width}")
    logger.info(f"Enable agent loss: {agent_bev_args.enable_agent_loss}")
    logger.info(f"Enable BEV loss: {agent_bev_args.enable_bev_loss}")
    logger.info("=" * 60)
    
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        padding_side="right",
    )
    
    # 加载 processor
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    
    # 创建模型配置
    config = Qwen2_5_VLConfigAgentBEV_ADS.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
    )
    
    # 更新配置
    config.num_agent_queries = data_args.num_agent_queries
    config.num_map_queries = data_args.num_map_queries
    config.num_agent_classes = 4  # ADS: empty, vehicle, pedestrian, other
    config.num_bev_classes = 4    # ADS: background, static, vehicle, pedestrian
    config.bev_output_size = (data_args.bev_pixel_height, data_args.bev_pixel_width)
    config.enable_agent_loss = agent_bev_args.enable_agent_loss
    config.enable_bev_loss = agent_bev_args.enable_bev_loss
    config.agent_class_weight = agent_bev_args.agent_class_weight
    config.agent_box_weight = agent_bev_args.agent_box_weight
    config.bev_weight = agent_bev_args.bev_weight
    
    # 加载模型
    logger.info("Loading model...")
    model = Qwen2_5_VLForConditionalGenerationAgentBEV_ADS.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    
    # 冻结部分参数
    if model_args.freeze_vision_encoder:
        logger.info("Freezing vision encoder...")
        for name, param in model.named_parameters():
            if "visual" in name:
                param.requires_grad = False
    
    if model_args.freeze_llm_backbone:
        logger.info("Freezing LLM backbone...")
        for name, param in model.named_parameters():
            if "model.layers" in name:
                param.requires_grad = False
    
    # 统计可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    # 创建数据集和 collator
    logger.info("Creating dataset...")
    data_module = make_supervised_data_module_huawei2_vla_agent_bev(
        tokenizer=tokenizer,
        data_args=data_args,
    )
    
    # 创建 Trainer
    trainer = AgentBEVTrainer(
        model=model,
        args=training_args,
        train_dataset=data_module["train_dataset"],
        eval_dataset=data_module["eval_dataset"],
        data_collator=data_module["data_collator"],
        tokenizer=tokenizer,
    )
    
    # 开始训练
    logger.info("Starting training...")
    trainer.train()
    
    # 保存模型
    logger.info(f"Saving model to {training_args.output_dir}")
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()


# ============ 使用示例 ============

"""
# 示例 1: 仅 Agent Detection
python train_agent_bev_ads_example.py \
    --model_name_or_path Qwen/Qwen2.5-VL-7B \
    --data_path /path/to/ads/data \
    --output_dir ./outputs/agent_only \
    --enable_agent_loss True \
    --enable_bev_loss False \
    --num_agent_queries 20 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 3

# 示例 2: 仅 BEV Semantic Map
python train_agent_bev_ads_example.py \
    --model_name_or_path Qwen/Qwen2.5-VL-7B \
    --data_path /path/to/ads/data \
    --output_dir ./outputs/bev_only \
    --enable_agent_loss False \
    --enable_bev_loss True \
    --num_map_queries 64 \
    --bev_pixel_width 256 \
    --bev_pixel_height 128

# 示例 3: 联合训练
python train_agent_bev_ads_example.py \
    --model_name_or_path Qwen/Qwen2.5-VL-7B \
    --data_path /path/to/ads/data \
    --output_dir ./outputs/agent_bev_joint \
    --enable_agent_loss True \
    --enable_bev_loss True \
    --agent_class_weight 1.0 \
    --agent_box_weight 1.0 \
    --bev_weight 0.5 \
    --freeze_vision_encoder True \
    --learning_rate 2e-5 \
    --bf16 True \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16
"""

