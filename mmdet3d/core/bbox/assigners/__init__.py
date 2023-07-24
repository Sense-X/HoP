# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.core.bbox import AssignResult, BaseAssigner, MaxIoUAssigner
from .atss3d_assigner import ATSS3DAssigner

__all__ = ['BaseAssigner', 'MaxIoUAssigner', 'AssignResult', 'ATSS3DAssigner']
