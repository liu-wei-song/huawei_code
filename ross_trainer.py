from collections import defaultdict

from reference.transformers.src.transformers.trainer import Trainer
import torch
import torch.nn as nn

class RossTrainer(Trainer):
    """
    训练时在每个 optimizer step 输出一次你在 forward 里写的 _last_logs 的均值。
    如需每个 micro-step 都输出，见标注处。
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._acc_sum = defaultdict(float)
        self._acc_n = 0
        self._micro_in_accum = 0  # 当前累积里的 micro-step 计数

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

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)

        # 累积 forward 写入的 _last_logs
        actual_model = model.module if hasattr(model, 'module') else model
        logs = getattr(actual_model, "_last_logs", None)
        if logs:
            for k, v in logs.items():
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

        return loss
