# Copyright (c) OpenMMLab. All rights reserved.
# modified from megvii-bevdepth.
import math
import os
from copy import deepcopy
import json
import pdb
import shutil

import torch
from mmcv.runner import load_state_dict
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks import HOOKS, Hook
import torch.distributed as dist
from mmdet3d.core.hook.utils import is_parallel
import mmcv

@HOOKS.register_module()
class ISSaveHook(Hook):
    """EMAHook used in BEVDepth.

    Modified from https://github.com/Megvii-Base
    Detection/BEVDepth/blob/main/callbacks/ema.py.
    """

    def __init__(self, save_path=None):
        super().__init__()
        self.save_path = save_path
        self.cnt = 0

    # def before_run(self, runner):
    #     from torch.nn.modules.batchnorm import SyncBatchNorm

    #     bn_model_list = list()
    #     bn_model_dist_group_list = list()
    #     for model_ref in runner.model.modules():
    #         if isinstance(model_ref, SyncBatchNorm):
    #             bn_model_list.append(model_ref)
    #             bn_model_dist_group_list.append(model_ref.process_group)
    #             model_ref.process_group = None
    #     runner.ema_model = ModelEMA(runner.model, self.decay)

    #     for bn_model, dist_group in zip(bn_model_list,
    #                                     bn_model_dist_group_list):
    #         bn_model.process_group = dist_group
    #     runner.ema_model.updates = self.init_updates

    #     if self.resume is not None:
    #         runner.logger.info(f'resume ema checkpoint from {self.resume}')
    #         cpt = torch.load(self.resume, map_location='cpu')
    #         load_state_dict(runner.ema_model.ema, cpt['state_dict'])
    #         runner.ema_model.updates = cpt['updates']

    # def after_train_iter(self, runner):
    #     # print(dist.get_rank())
    #     # pdb.set_trace()
    #     self.cnt += 1
    #     if self.cnt % 2==0:
    #         self.save_is_dict(runner)
    # #         # pdb.set_trace()



    def after_train_epoch(self, runner):
        self.save_is_dict(runner)


    def save_is_dict(self, runner):
        
        # if is_parallel(runner.model.module):
        #     runner.model.module.module.pts_bbox_head.save_epoch=runner.epoch
        # else:
        #     runner.model.module.pts_bbox_head.save_epoch=runner.epoch
        if is_parallel(runner.model.module):
            save_dict = runner.model.module.module.pts_bbox_head.match_dict
        else:
            save_dict = runner.model.module.pts_bbox_head.match_dict

        for i in range(len(save_dict)):
            path = os.path.join(self.save_path, f'dec_{i}', f'rank_{dist.get_rank()}')
            if not os.path.exists(path):
                os.makedirs(path)

            mmcv.dump(save_dict[str(i)], os.path.join(path,
                                            f'match_epoch{runner.epoch}.json')) 
            dist.barrier()

        if is_parallel(runner.model.module):
            runner.model.module.module.pts_bbox_head.match_dict = {}
            for i in range(6):
                runner.model.module.module.pts_bbox_head.match_dict[str(i)] = {}
        else:
            runner.model.module.pts_bbox_head.match_dict = {}
            for i in range(6):
                runner.model.module.pts_bbox_head.match_dict[str(i)] = {}
