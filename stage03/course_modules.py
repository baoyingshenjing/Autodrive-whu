"""MMEngine hooks that make a training result auditable for this course."""
from __future__ import annotations

import time
from pathlib import Path

import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

from course_common import save_json


@HOOKS.register_module(force=True)
class HeadOnlyGradientAuditHook(Hook):
    """Freeze non-head layers and prove gradients caused parameter changes."""
    priority = 'VERY_HIGH'

    def __init__(self, audit_file: str):
        self.audit_file = Path(audit_file)
        self.before = {}
        self.gradient_tensors = 0
        self.gradient_events = 0
        self.max_abs_gradient = 0.0
        self.nonfinite_gradient_events = 0
        self.started = 0.0
        self.handles = []
        self.loss_history = []

    def before_train(self, runner) -> None:
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith('pts_bbox_head.')
        trainable = [(name, p) for name, p in model.named_parameters()
                     if p.requires_grad]
        if not trainable:
            raise RuntimeError('没有可训练的 pts_bbox_head 参数，拒绝伪训练。')
        self.before = {name: p.detach().float().cpu().clone()
                       for name, p in trainable}
        self.started = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        seen = set()
        def hook(name):
            def _record(grad):
                self.gradient_events += 1
                if name not in seen:
                    seen.add(name)
                    self.gradient_tensors += 1
                if grad is not None and grad.numel():
                    finite = torch.isfinite(grad)
                    if not bool(finite.all()):
                        self.nonfinite_gradient_events += 1
                    if bool(finite.any()):
                        self.max_abs_gradient = max(
                            self.max_abs_gradient,
                            float(grad.detach()[finite].abs().max().cpu()))
            return _record
        for name, parameter in trainable:
            self.handles.append(parameter.register_hook(hook(name)))
        runner.logger.info('AUDIT: only pts_bbox_head is trainable; gradient hooks registered.')

    def after_train_iter(self, runner, batch_idx: int, data_batch=None,
                         outputs=None) -> None:
        loss = runner.message_hub.get_scalar('train/loss')
        if loss is not None:
            value = float(loss.current())
            runner.message_hub.update_info('course_last_loss', value)
            self.loss_history.append(dict(iteration=int(runner.iter), loss=value))

    def after_train(self, runner) -> None:
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        changed = 0
        max_change = 0.0
        all_finite = True
        for name, before in self.before.items():
            after = dict(model.named_parameters())[name].detach().float().cpu()
            diff = (after - before).abs()
            change = float(diff.max()) if diff.numel() else 0.0
            max_change = max(max_change, change)
            changed += int(change > 0.0)
            all_finite = all_finite and bool(torch.isfinite(after).all())
        for handle in self.handles:
            handle.remove()
        allocated = reserved = 0
        if torch.cuda.is_available():
            allocated = int(torch.cuda.max_memory_allocated())
            reserved = int(torch.cuda.max_memory_reserved())
        status = ('PARAMETERS_UPDATED' if changed > 0 and self.gradient_tensors > 0
                  and self.max_abs_gradient > 0 and all_finite else 'FAILED_AUDIT')
        audit = dict(
            validation_status=status,
            trainable_scope='pts_bbox_head only',
            changed_parameter_tensors=changed,
            total_trainable_parameter_tensors=len(self.before),
            gradient_parameter_tensors=self.gradient_tensors,
            gradient_events=self.gradient_events,
            nonfinite_scaled_gradient_events=self.nonfinite_gradient_events,
            max_abs_gradient=self.max_abs_gradient,
            max_abs_parameter_change=max_change,
            all_trainable_parameters_finite=all_finite,
            wall_seconds=round(time.perf_counter() - self.started, 3),
            peak_gpu_allocated_bytes=allocated,
            peak_gpu_reserved_bytes=reserved,
            last_loss=runner.message_hub.get_info('course_last_loss'))
        save_json(self.audit_file, audit)
        save_json(self.audit_file.parent / 'loss_history.json',
                  dict(points=self.loss_history))
        runner.logger.info(f'AUDIT: {status}; changed={changed}; max_change={max_change:.3e}')
        if status != 'PARAMETERS_UPDATED':
            raise RuntimeError(f'梯度/参数更新审计失败：{audit}')
