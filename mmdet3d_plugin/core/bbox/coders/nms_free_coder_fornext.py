import torch

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS
from mmdet3d_plugin.core.bbox.util import denormalize_bbox
import numpy as np


@BBOX_CODERS.register_module()
class NMSFreeCoderForNext(BaseBBoxCoder):
    """Bbox coder for NMS-free detector.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range,
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 num_classes=10,
                 distance_weight=False,
                 size_thresh=5,
                 dis_sigma=35):
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes
        self.distance_weight = distance_weight
        self.size_thresh = size_thresh
        self.dis_sigma = dis_sigma

    def encode(self):

        pass

    def decode_single(self, cls_scores, bbox_preds, center=None, decode_num=None):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        if decode_num is None:
            max_num = self.max_num 
        else:
            max_num = decode_num

        cls_scores = cls_scores.sigmoid()

        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        bbox_preds = bbox_preds[bbox_index]
       
        final_box_preds = denormalize_bbox(bbox_preds, self.pc_range)   
        final_scores = scores 
        final_preds = labels 

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
            tmp_score = self.score_threshold
            while thresh_mask.sum() == 0:
                tmp_score *= 0.9
                if tmp_score < 0.01:
                    thresh_mask = final_scores > -1
                    break
                thresh_mask = final_scores >= tmp_score

        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(
                self.post_center_range, device=scores.device)
            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]

            labels = final_preds[mask]
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels,
                'index': bbox_index
            }

        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')
        return predictions_dict

    def decode(self, preds_dicts, img_metas=None, list_index=None, decode_num=None):
        """Decode bboxes.
        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
            center (Tensor, shape [3,1]): x,y,z of sweep ego in key ego frame 
        Returns:
            list[dict]: Decoded boxes.
        """
        all_cls_scores = preds_dicts['all_cls_scores'][-1]
        all_bbox_preds = preds_dicts['all_bbox_preds'][-1]

        
        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(self.decode_single(all_cls_scores[i], all_bbox_preds[i],decode_num=decode_num))
        return predictions_list

    # def distance_reweight(self, cls_scores, bbox_preds, center):
    #     center = center.to(bbox_preds).squeeze() # [3]
    #     pred_coord = torch.cat([bbox_preds[:, :2], bbox_preds[:, 4:5]], dim=1) # [N,3]

    #     w = bbox_preds[:, 2:3]
    #     l = bbox_preds[:, 3:4]
    #     h = bbox_preds[:, 5:6]
    #     w = w.exp() 
    #     l = l.exp() 
    #     h = h.exp() 

    #     size = torch.cat([w, l, h], dim=-1)
    #     dis2center = torch.sum(torch.square(pred_coord-center), dim=-1)

    #     size_decay = 0.3 + torch.sigmoid(size.sum(dim=-1) - self.size_thresh)
    #     size_decay = torch.min(size_decay, torch.ones_like(size_decay))
    #     dis_decay = torch.exp(-dis2center/(2*(self.dis_sigma**2)))
    #     score_decay = size_decay * dis_decay
    #     cls_scores = cls_scores * score_decay.unsqueeze(1)
    #     return cls_scores
