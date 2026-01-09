from collections import defaultdict
import numpy as np

from reference.transformers.src.transformers.trainer import Trainer
import torch
import torch.nn as nn

class RossTrainer(Trainer):
    """
    训练时在每个 optimizer step 输出一次你在 forward 里写的 _last_logs 的均值。
    如需每个 micro-step 都输出，见标注处。
    支持 BEV 预测结果的 TensorBoard 可视化。
    """
    def __init__(self, *args, bev_log_steps=100, **kwargs):
        super().__init__(*args, **kwargs)
        self._acc_sum = defaultdict(float)
        self._acc_n = 0
        self._micro_in_accum = 0  # 当前累积里的 micro-step 计数
        # BEV visualization
        self.bev_log_steps = bev_log_steps
        self._pending_bev_pred = None
        self._pending_bev_gt = None

    def _log_mean_and_reset(self):
        if self._acc_n == 0:
            return
        mean_logs = {k: (s / float(self._acc_n)) for k, s in self._acc_sum.items()}

        # 仅主进程写
        acc = getattr(self, "accelerator", None)
        is_main = acc.is_main_process if acc is not None else self.is_world_process_zero()
        if is_main:
            self.log(mean_logs)

        self._acc_sum.clear()
        self._acc_n = 0

    def _get_tb_writer(self):
        """获取 TensorBoard writer"""
        for callback in self.callback_handler.callbacks:
            if hasattr(callback, 'tb_writer') and callback.tb_writer is not None:
                return callback.tb_writer
        return None

    def _colorize_bev(self, bev_map, num_classes=5):
        """将 BEV 类别图转为 RGB 彩色图"""
        # 颜色映射 (支持 3/5/7 类)
        if num_classes == 3:
            colors = np.array([
                [0, 0, 0],       # 0: background (黑色)
                [255, 255, 0],   # 1: centerline (黄色)
                [255, 0, 0],     # 2: objects (红色)
            ], dtype=np.uint8)
        elif num_classes == 5:
            colors = np.array([
                [0, 0, 0],       # 0: background (黑色)
                [255, 255, 0],   # 1: centerline (黄色)
                [128, 128, 128], # 2: static (灰色)
                [255, 0, 0],     # 3: vehicle (红色)
                [0, 0, 255],     # 4: pedestrian (蓝色)
            ], dtype=np.uint8)
        else:
            # 7 类或其他：使用 colormap 自动生成
            colors = np.array([
                [0, 0, 0],       # 0: background
                [255, 255, 0],   # 1: centerline
                [128, 128, 128], # 2: static
                [255, 0, 0],     # 3: vehicle
                [0, 0, 255],     # 4: pedestrian
                [0, 255, 0],     # 5: green
                [255, 0, 255],   # 6: magenta
            ], dtype=np.uint8)

        if isinstance(bev_map, torch.Tensor):
            bev_map = bev_map.numpy()

        bev_map = np.clip(bev_map.astype(np.int32), 0, len(colors) - 1)
        rgb = colors[bev_map]  # [H, W, 3]
        rgb = rgb.transpose(2, 0, 1)  # [3, H, W]
        return torch.from_numpy(rgb).float() / 255.0

    def _log_bev_images(self):
        """记录 BEV 图像到 TensorBoard"""
        tb_writer = self._get_tb_writer()
        if tb_writer is None:
            return

        step = self.state.global_step
        # 获取类别数
        actual_model = self.model.module if hasattr(self.model, 'module') else self.model
        num_classes = getattr(actual_model, 'num_bev_classes', 5)

        if self._pending_bev_pred is not None:
            pred_class = self._pending_bev_pred.argmax(dim=0)
            pred_rgb = self._colorize_bev(pred_class, num_classes)
            tb_writer.add_image("bev/prediction", pred_rgb, step)

        if self._pending_bev_gt is not None:
            gt_rgb = self._colorize_bev(self._pending_bev_gt, num_classes)
            tb_writer.add_image("bev/ground_truth", gt_rgb, step)

        self._pending_bev_pred = None
        self._pending_bev_gt = None
        tb_writer.flush()

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)

        # 累积 forward 写入的 _last_logs (或 loss_dict)
        actual_model = model.module if hasattr(model, 'module') else model
        logs = getattr(actual_model, "loss_dict", None) or getattr(actual_model, "_last_logs", None)
        if logs:
            for k, v in logs.items():
                if k == "bev_pred":
                    self._pending_bev_pred = v
                elif k == "bev_gt":
                    self._pending_bev_gt = v
                else:
                    try:
                        self._acc_sum[k] += float(v)
                    except Exception:
                        pass
            self._acc_n += 1

        # 统计当前累积内的 micro-step，并在到达 optimizer step 时输出
        self._micro_in_accum += 1
        gas = max(1, int(self.args.gradient_accumulation_steps))
        if self._micro_in_accum % gas == 0:
            self._log_mean_and_reset()
            # 每 bev_log_steps 步记录 BEV 图像
            if self.state.global_step > 0 and self.state.global_step % self.bev_log_steps == 0:
                self._log_bev_images()

        return loss
