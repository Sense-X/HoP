# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner.hooks import HOOKS, Hook
from mmdet3d.core.hook.utils import is_parallel

__all__ = ['SequentialControlHook', 'HisInfoControlHook']


@HOOKS.register_module()
class SequentialControlHook(Hook):
    """ """

    def __init__(self, temporal_start_epoch=1):
        super().__init__()
        self.temporal_start_epoch=temporal_start_epoch

    def set_temporal_flag(self, runner, flag):
        if is_parallel(runner.model.module):
            runner.model.module.module.with_prev=flag
        else:
            runner.model.module.with_prev = flag

    def before_run(self, runner):
        self.set_temporal_flag(runner, False)

    def before_train_epoch(self, runner):
        if runner.epoch > self.temporal_start_epoch:
            self.set_temporal_flag(runner, True)

@HOOKS.register_module()
class HisInfoControlHook(Hook):
    """ """

    def __init__(self, hisinfo_start_epoch=-1):
        super().__init__()
        self.his_info_start_epoch=hisinfo_start_epoch

    def set_temporal_flag(self, runner, flag):
        if is_parallel(runner.model.module):
            runner.model.module.module.use_his_info=flag
        else:
            runner.model.module.use_his_info = flag

    def before_run(self, runner):
        self.set_temporal_flag(runner, False)

    def before_train_epoch(self, runner):
        if runner.epoch > self.his_info_start_epoch:
            print(f"Open His Info")
            self.set_temporal_flag(runner, True)