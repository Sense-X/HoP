# Copyright (c) OpenMMLab. All rights reserved.
import json
import warnings

from mmcv.runner import DefaultOptimizerConstructor, get_dist_info
from mmcv.runner.optimizer.builder import OPTIMIZER_BUILDERS, OPTIMIZERS
from mmdet.utils import get_root_logger


def get_layer_id_for_cb(var_name, max_layer_id):
    """Get the layer id to set the different learning rates.

    Args:
        var_name (str): The key of the model.
        num_max_layer (int): Maximum number of backbone layers.

    Returns:
        int: Returns the layer id of the key.
    """

    if 'absolute_pos_embed' in var_name and 'backbone' in var_name:
        return 0
    elif 'patch_embed' in var_name and 'backbone' in var_name:
        return 0
    elif 'stages' in var_name and 'backbone' in var_name:
        num_layers = [2, 2, 18, 2]
        num_layers = [2, 6, 24, 2]
        stage_id = int(var_name.split('.')[4])
        if 'downsample' in var_name:
            return sum(num_layers[:stage_id + 1]) + 1
        block_id = int(var_name.split('.')[6])
        return sum(num_layers[:stage_id]) + block_id + 1
    else:
        return max_layer_id + 1

def get_layer_id_for_swin(var_name, max_layer_id):
    """Get the layer id to set the different learning rates.

    Args:
        var_name (str): The key of the model.
        num_max_layer (int): Maximum number of backbone layers.

    Returns:
        int: Returns the layer id of the key.
    """

    if var_name in ('backbone.absolute_pos_embed'):
        return 0
    elif var_name.startswith('backbone.patch_embed'):
        return 0
    elif var_name.startswith('backbone.stages'):
        num_layers = [2, 2, 18, 2]
        num_layers = [2, 6, 24, 2]
        stage_id = int(var_name.split('.')[2])
        if 'downsample' in var_name:
            return sum(num_layers[:stage_id + 1]) + 1
        block_id = int(var_name.split('.')[4])
        return sum(num_layers[:stage_id]) + block_id + 1
    else:
        return max_layer_id + 1

def get_layer_id_for_codet(var_name, max_layer_id):
    """Get the layer id to set the different learning rates.

    Args:
        var_name (str): The key of the model.
        num_max_layer (int): Maximum number of backbone layers.

    Returns:
        int: Returns the layer id of the key.
    """

    if var_name in ('backbone.absolute_pos_embed'):
        return 0
    elif var_name.startswith('backbone.patch_embed'):
        return 0
    elif var_name.startswith('backbone.stages'):
        num_layers = [2, 2, 18, 2]
        num_layers = [2, 6, 24, 2]
        stage_id = int(var_name.split('.')[2])
        if 'downsample' in var_name:
            return sum(num_layers[:stage_id + 1]) + 1
        block_id = int(var_name.split('.')[4])
        return sum(num_layers[:stage_id]) + block_id + 1
    elif 'neck' in var_name or 'sampling_offsets' in var_name or 'reference_points' in var_name:
        return 35
    elif 'transformer.encoder.layers.0' in var_name:
        return 36
    elif 'transformer.encoder.layers.1' in var_name:
        return 37
    elif 'transformer.encoder.layers.2' in var_name:
        return 38
    elif 'transformer.encoder.layers.3' in var_name:
        return 39           
    elif 'transformer.encoder.layers.4' in var_name:
        return 40
    elif 'transformer.encoder.layers.5' in var_name:
        return 41
    elif 'transformer.decoder.layers.0' in var_name:
        return 42
    elif 'transformer.decoder.layers.1' in var_name:
        return 43
    elif 'transformer.decoder.layers.2' in var_name:
        return 44
    elif 'transformer.decoder.layers.3' in var_name:
        return 45
    elif 'transformer.decoder.layers.4' in var_name:
        return 46
    elif 'transformer.decoder.layers.5' in var_name:
        return 47
    else:
        return max_layer_id + 1

def get_layer_id_for_convnext(var_name, max_layer_id):
    """Get the layer id to set the different learning rates in ``layer_wise``
    decay_type.

    Args:
        var_name (str): The key of the model.
        max_layer_id (int): Maximum number of backbone layers.

    Returns:
        int: The id number corresponding to different learning rate in
        ``LearningRateDecayOptimizerConstructor``.
    """

    if var_name in ('backbone.cls_token', 'backbone.mask_token',
                    'backbone.pos_embed'):
        return 0
    elif var_name.startswith('backbone.downsample_layers'):
        stage_id = int(var_name.split('.')[2])
        if stage_id == 0:
            layer_id = 0
        elif stage_id == 1:
            layer_id = 2
        elif stage_id == 2:
            layer_id = 3
        elif stage_id == 3:
            layer_id = max_layer_id
        return layer_id
    elif var_name.startswith('backbone.stages'):
        stage_id = int(var_name.split('.')[2])
        block_id = int(var_name.split('.')[3])
        if stage_id == 0:
            layer_id = 1
        elif stage_id == 1:
            layer_id = 2
        elif stage_id == 2:
            layer_id = 3 + block_id // 3
        elif stage_id == 3:
            layer_id = max_layer_id
        return layer_id
    else:
        return max_layer_id + 1


def get_stage_id_for_convnext(var_name, max_stage_id):
    """Get the stage id to set the different learning rates in ``stage_wise``
    decay_type.

    Args:
        var_name (str): The key of the model.
        max_stage_id (int): Maximum number of backbone layers.

    Returns:
        int: The id number corresponding to different learning rate in
        ``LearningRateDecayOptimizerConstructor``.
    """

    if var_name in ('backbone.cls_token', 'backbone.mask_token',
                    'backbone.pos_embed'):
        return 0
    elif var_name.startswith('backbone.downsample_layers'):
        return 0
    elif var_name.startswith('backbone.stages'):
        stage_id = int(var_name.split('.')[2])
        return stage_id + 1
    else:
        return max_stage_id - 1


def get_layer_id_for_vit(var_name, max_layer_id):
    """Get the layer id to set the different learning rates.

    Args:
        var_name (str): The key of the model.
        num_max_layer (int): Maximum number of backbone layers.

    Returns:
        int: Returns the layer id of the key.
    """

    if var_name in ('img_backbone.cls_token', 'img_backbone.mask_token',
                    'img_backbone.pos_embed'):
        return 0
    elif var_name.startswith('img_backbone.patch_embed'):
        return 0
    elif var_name.startswith('img_backbone.blocks'):
        layer_id = int(var_name.split('.')[2])
        return layer_id + 1
    else:
        return max_layer_id - 1


def get_layer_id_for_cbvit(var_name, max_layer_id):
    """Get the layer id to set the different learning rates.

    Args:
        var_name (str): The key of the model.
        num_max_layer (int): Maximum number of backbone layers.

    Returns:
        int: Returns the layer id of the key.
    """

    if 'backbone' in var_name:
        if 'cls_token' in var_name or 'mask_token' in var_name or 'pos_embed' in var_name:
            return 0
        elif 'patch_embed' in var_name:
            return 0
        elif 'blocks' in var_name:
            layer_id = int(var_name.split('.')[4])
            return layer_id + 1
    return max_layer_id - 1


@OPTIMIZER_BUILDERS.register_module()
class LearningRateDecayOptimizerConstructor(DefaultOptimizerConstructor):
    """Different learning rates are set for different layers of backbone.

    Note: Currently, this optimizer constructor is built for ConvNeXt,
    BEiT and MAE.
    """

    def add_params(self, params, module, **kwargs):
        """Add all parameters of module to the params list.

        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.

        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
        """
        logger = get_root_logger()

        parameter_groups = {}
        logger.info(f'self.paramwise_cfg is {self.paramwise_cfg}')
        num_layers = self.paramwise_cfg.get('num_layers') + 2
        decay_rate = self.paramwise_cfg.get('decay_rate')
        decay_type = self.paramwise_cfg.get('decay_type', 'layer_wise')
        logger.info('Build LearningRateDecayOptimizerConstructor  '
                    f'{decay_type} {decay_rate} - {num_layers}')
        weight_decay = self.base_wd
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith('.bias') or name in (
                    'pos_embed', 'cls_token') or 'relative_position_bias_table' in name or 'rel_pos' in name:
                group_name = 'no_decay'
                this_weight_decay = 0.
            else:
                group_name = 'decay'
                this_weight_decay = weight_decay
            if 'layer_wise' in decay_type:
                if 'CBXMNet' in module.img_backbone.__class__.__name__:
                    layer_id = get_layer_id_for_cb(
                        name, self.paramwise_cfg.get('num_layers'))
                    logger.info(f'set param {name} as id {layer_id}')
                elif 'Composite' in module.__class__.__name__ and 'XMNet' in module.img_backbone.__class__.__name__:
                    layer_id = get_layer_id_for_codet(
                        name, self.paramwise_cfg.get('num_layers'))
                    logger.info(f'set param {name} as id {layer_id}')                
                elif 'Swin' in module.img_backbone.__class__.__name__:
                    layer_id = get_layer_id_for_swin(
                        name, self.paramwise_cfg.get('num_layers'))
                    logger.info(f'set param {name} as id {layer_id}')
                elif 'XMNet' in module.img_backbone.__class__.__name__:
                    layer_id = get_layer_id_for_swin(
                        name, self.paramwise_cfg.get('num_layers'))
                    logger.info(f'set param {name} as id {layer_id}')
                elif 'ConvNeXt' in module.img_backbone.__class__.__name__:
                    layer_id = get_layer_id_for_convnext(
                        name, self.paramwise_cfg.get('num_layers'))
                    logger.info(f'set param {name} as id {layer_id}')
                elif 'CBViT' in module.img_backbone.__class__.__name__:
                    layer_id = get_layer_id_for_cbvit(name, num_layers)
                    logger.info(f'set param {name} as id {layer_id}')
                elif 'BEiT' in module.img_backbone.__class__.__name__ or \
                     'ViT' in module.img_backbone.__class__.__name__:
                    layer_id = get_layer_id_for_vit(name, num_layers)
                    logger.info(f'set param {name} as id {layer_id}')
                else:
                    raise NotImplementedError()
            elif decay_type == 'stage_wise':
                if 'ConvNeXt' in module.img_backbone.__class__.__name__:
                    layer_id = get_stage_id_for_convnext(name, num_layers)
                    logger.info(f'set param {name} as id {layer_id}')
                else:
                    raise NotImplementedError()
            group_name = f'layer_{layer_id}_{group_name}'

            if group_name not in parameter_groups:
                scale = decay_rate**(num_layers - layer_id - 1)
                '''
                if 'Composite' in module.__class__.__name__:
                    if not 'layer_35' in group_name:
                        scale = scale*0.1
                '''
                parameter_groups[group_name] = {
                    'weight_decay': this_weight_decay,
                    'params': [],
                    'param_names': [],
                    'lr_scale': scale,
                    'group_name': group_name,
                    'lr': scale * self.base_lr,
                }

            parameter_groups[group_name]['params'].append(param)
            parameter_groups[group_name]['param_names'].append(name)
        rank, _ = get_dist_info()
        if rank == 0:
            to_display = {}
            for key in parameter_groups:
                to_display[key] = {
                    'param_names': parameter_groups[key]['param_names'],
                    'lr_scale': parameter_groups[key]['lr_scale'],
                    'lr': parameter_groups[key]['lr'],
                    'weight_decay': parameter_groups[key]['weight_decay'],
                }
            logger.info(f'Param groups = {json.dumps(to_display, indent=2)}')
        params.extend(parameter_groups.values())


@OPTIMIZER_BUILDERS.register_module()
class LayerDecayOptimizerConstructor(LearningRateDecayOptimizerConstructor):
    """Different learning rates are set for different layers of backbone.

    Note: Currently, this optimizer constructor is built for BEiT,
    and it will be deprecated.
    Please use ``LearningRateDecayOptimizerConstructor`` instead.
    """

    def __init__(self, optimizer_cfg, paramwise_cfg):
        warnings.warn('DeprecationWarning: Original '
                      'LayerDecayOptimizerConstructor of BEiT '
                      'will be deprecated. Please use '
                      'LearningRateDecayOptimizerConstructor instead, '
                      'and set decay_type = layer_wise in paramwise_cfg.')
        paramwise_cfg.update({'decay_type': 'layer_wise'})
        warnings.warn('DeprecationWarning: Layer_decay_rate will '
                      'be deleted, please use decay_rate instead.')
        paramwise_cfg['decay_rate'] = paramwise_cfg.pop('layer_decay_rate')
        super(LayerDecayOptimizerConstructor,
              self).__init__(optimizer_cfg, paramwise_cfg)
