# Copyright (c) Phigent Robotics. All rights reserved.
import argparse
import json
import os
import pickle

import cv2
import numpy as np
from pyquaternion.quaternion import Quaternion
import mmcv
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes as LB
from petrel_client.client import Client 
from PIL import Image
import io
import pdb
import os, shutil
from mmdet3d.core.visualizer.image_vis import draw_lidar_bbox3d_on_img

color_map_nusc = {  # RGB.

    "animal": (70, 130, 180),  # Steelblue

    "pedestrian": (230, 0, 0),  # Blue
    "car": (0, 158, 255),  # Orange
    'traffic_cone': (47, 79, 79), 
    'construction_vehicle': (255, 127, 80), # coral
    'truck': (255, 15, 71),  # Tomato,
    'motorcycle': (255, 61, 200),  # Red
    'bicycle': (220, 20, 140),  # Crimson
    'bus':(255, 69, 0),
    'barrier': (112, 128, 144),
    'trailer': (255, 180, 0),  # Darkorange

    "human.pedestrian.child": (135, 206, 235),  # Skyblue,
    "human.pedestrian.construction_worker": (100, 149, 237),  # Cornflowerblue
    "human.pedestrian.personal_mobility": (219, 112, 147),  # Palevioletred
    "human.pedestrian.police_officer": (0, 0, 128),  # Navy,
    "human.pedestrian.stroller": (240, 128, 128),  # Lightcoral
    "human.pedestrian.wheelchair": (138, 43, 226),  # Blueviolet
    "movable_object.barrier": (112, 128, 144),  # Slategrey
    "movable_object.debris": (210, 105, 30),  # Chocolate
    "movable_object.pushable_pullable": (105, 105, 105),  # Dimgrey
    "static_object.bicycle_rack": (188, 143, 143),  # Rosybrown
    "vehicle.bus.rigid": (255, 69, 0),  # Orangered

    "vehicle.construction": (233, 150, 70),  # Darksalmon
    "vehicle.emergency.ambulance": (255, 83, 0),
    "vehicle.emergency.police": (255, 215, 0),  # Gold
    "vehicle.trailer": (255, 140, 0),  # Darkorange
    # "vehicle.": 
    "flat.driveable_surface": (0, 207, 191),  # nuTonomy green
    "flat.other": (175, 0, 75),
    "flat.sidewalk": (75, 0, 75),
    "flat.terrain": (112, 180, 60),
    "static.manmade": (222, 184, 135),  # Burlywood
    "static.other": (255, 228, 196),  # Bisque
    "static.vegetation": (0, 175, 0),  # Green
    "vehicle.ego": (255, 240, 245)
}

class RC:
    def __init__(self) -> None:
        self.file_client_args = dict(
                                backend='petrel',
                                enable_mc=True,
                                path_mapping=dict({
                                    './data/nuscenes/': 'openmmlab:s3://openmmlab/datasets/detection3d/nuscenes/',
                                    'data/nuscenes/': 'openmmlab:s3://openmmlab/datasets/detection3d/nuscenes/'
                                }))
        self.client = mmcv.FileClient(**self.file_client_args)
        # self.client = Client(enable_multi_cluster=True)
    
    def load_pil_image(self, filename, color_type='color'):
        ''' Adapt for petrel. Origin Implementation: img = Image.open(filename)
        copy from LoadMultiViewImageFromFiles
        Image.open() default is RGB files
        Validated 
        '''
        if self.file_client_args is None or self.file_client_args['backend'] == 'disk':
            load_fun = mmcv.load

        elif self.file_client_args['backend'] == 'petrel':
            def petrel_load_image(name, color_type):
                img_bytes = self.file_client.get(name)
                return mmcv.imfrombytes(img_bytes, flag=color_type, channel_order='rgb',backend='pillow')    
            load_fun = petrel_load_image

        else:
            raise NotImplementedError(f'File client args is {self.file_client_args}')
        
        img_array = load_fun(filename, color_type)
        img_pil = Image.fromarray(img_array.astype(np.uint8), mode='RGB') 
        return img_pil

    def load_image(self, filename, color_type='unchanged'):
        if self.file_client_args is None or self.file_client_args['backend'] == 'disk':
            load_fun = mmcv.load

        elif self.file_client_args['backend'] == 'petrel':
            self.file_client = mmcv.FileClient(**self.file_client_args)
            def petrel_load_image(name, color_type):
                img_bytes = self.file_client.get(name)
                return mmcv.imfrombytes(img_bytes, flag=color_type)    
            load_fun = petrel_load_image

        else:
            raise NotImplementedError(f'File client args is {self.file_client_args}')

        return load_fun(filename, color_type)        
        # return  np.stack(
        #             [load_fun(name, color_type) for name in filename], axis=-1)


def check_point_in_img(points, height, width):
    valid = np.logical_and(points[:, 0] >= 0, points[:, 1] >= 0)
    # valid = np.logical_and(
    #     valid, np.logical_and(points[:, 0] < width, points[:, 1] < height))
    # valid = np.ones((points.shape[0]),dtype=np.bool)
    return valid


def depth2color(depth):
    gray = max(0, min((depth + 2.5) / 3.0, 1.0))
    max_lumi = 200
    colors = np.array(
        [[max_lumi, 0, max_lumi], [max_lumi, 0, 0], [max_lumi, max_lumi, 0],
         [0, max_lumi, 0], [0, max_lumi, max_lumi], [0, 0, max_lumi]],
        dtype=np.float32)
    if gray == 1:
        return tuple(colors[-1].tolist())
    num_rank = len(colors) - 1
    rank = np.floor(gray * num_rank).astype(np.int)
    diff = (gray - rank / num_rank) * num_rank
    return tuple(
        (colors[rank] + (colors[rank + 1] - colors[rank]) * diff).tolist())

def get_lidar2img(camrera_info):
    camera2lidar = np.eye(4, dtype=np.float32)
    camera2lidar[:3, :3] = camrera_info['sensor2lidar_rotation']
    camera2lidar[:3, 3] = camrera_info['sensor2lidar_translation']
    lidar2camera = np.linalg.inv(camera2lidar)
    camera2img = np.eye(4, dtype=np.float32)
    camera2img[:camrera_info['cam_intrinsic'].shape[0],
                :camrera_info['cam_intrinsic'].shape[1]] = camrera_info['cam_intrinsic']
    # pdb.set_trace()
    return  camera2img@lidar2camera

def lidar2img(points_lidar, camrera_info):
    points_lidar_homogeneous = \
        np.concatenate([points_lidar,
                        np.ones((points_lidar.shape[0], 1),
                                dtype=points_lidar.dtype)], axis=1)
    camera2lidar = np.eye(4, dtype=np.float32)
    camera2lidar[:3, :3] = camrera_info['sensor2lidar_rotation']
    camera2lidar[:3, 3] = camrera_info['sensor2lidar_translation']
    lidar2camera = np.linalg.inv(camera2lidar)
    points_camera_homogeneous = points_lidar_homogeneous @ lidar2camera.T
    points_camera = points_camera_homogeneous[:, :3]
    valid = np.ones((points_camera.shape[0]), dtype=bool)
    # valid = np.logical_and(points_camera[:, -1] > 0.5, valid)
    points_camera = points_camera / points_camera[:, 2:3]
    camera2img = camrera_info['cam_intrinsic']
    points_img = points_camera @ camera2img.T
    points_img = points_img[:, :2]
    return points_img, valid


def get_lidar2global(infos):
    lidar2ego = np.eye(4, dtype=np.float32)
    lidar2ego[:3, :3] = Quaternion(infos['lidar2ego_rotation']).rotation_matrix
    lidar2ego[:3, 3] = infos['lidar2ego_translation']
    ego2global = np.eye(4, dtype=np.float32)
    ego2global[:3, :3] = Quaternion(
        infos['ego2global_rotation']).rotation_matrix
    ego2global[:3, 3] = infos['ego2global_translation']
    return ego2global @ lidar2ego


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize the predicted '
                                     'result of nuScenes')
    parser.add_argument(
        'res', help='Path to the predicted result in json format')
    parser.add_argument(
        '--show-range',
        type=int,
        default=50,
        help='Range of visualization in BEV')
    parser.add_argument(
        '--canva-size', type=int, default=1000, help='Size of canva in pixel')
    parser.add_argument(
        '--vis-frames',
        type=int,
        default=500,
        help='Number of frames for visualization')
    parser.add_argument(
        '--scale-factor',
        type=int,
        default=2,
        help='Trade-off between image-view and bev in size of '
        'the visualized canvas')
    parser.add_argument(
        '--vis-thred',
        type=float,
        default=0.25,
        help='Threshold the predicted results')
    parser.add_argument('--draw-gt', action='store_true')
    parser.add_argument(
        '--version',
        type=str,
        default='val',
        help='Version of nuScenes dataset')
    parser.add_argument(
        '--root_path',
        type=str,
        default='./data/nuscenes',
        help='Path to nuScenes dataset')
    parser.add_argument(
        '--save_path',
        type=str,
        default='./vis',
        help='Path to save visualization results')
    parser.add_argument(
        '--format',
        type=str,
        default='image',
        choices=['video', 'image'],
        help='The desired format of the visualization result')
    parser.add_argument(
        '--fps', type=int, default=20, help='Frame rate of video')
    parser.add_argument(
        '--video-prefix', type=str, default='vis', help='name of video')
    parser.add_argument(
        '--split-gt-in-view', type=bool, default=True, help='split in view of gt & pred')
        
    args = parser.parse_args()
    return args


color_map = {0: (255, 0, 0), 1: (29, 155, 205)} # 0(blue): gt,1(yellow): pred
# (34, 180, 238)

def main():
    args = parse_args()
    # load predicted results
    res = json.load(open(args.res, 'r'))
    # load dataset information
    info_path = \
        args.root_path + '/bevdetv2-nuscenes_infos_%s.pkl' % args.version
    dataset = pickle.load(open(info_path, 'rb'))
    # prepare save path and medium
    vis_dir = args.save_path
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    print('saving visualized result to %s' % vis_dir)
    scale_factor = args.scale_factor
    canva_size = args.canva_size
    show_range = args.show_range
    if args.format == 'video':
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        vout = cv2.VideoWriter(
            os.path.join(vis_dir, '%s.mp4' % args.video_prefix), fourcc,
            args.fps, (int(1600 / scale_factor * 3),
                       int(900 / scale_factor * 2 + canva_size)))

    draw_boxes_indexes_bev = [(0, 1), (1, 2), (2, 3), (3, 0)]
    draw_boxes_indexes_img_view = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5),
                                   (5, 6), (6, 7), (7, 4), (0, 4), (1, 5),
                                   (2, 6), (3, 7)]
    views = [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ]
    print('start visualizing results')
    for cnt, infos in enumerate(
            dataset['infos'][5:5*min(args.vis_frames, len(dataset['infos'])):5]):
        if infos['token'] != 'f274b48b3f8245669f97556d66ce468b':
            continue
        if cnt % 10 == 0:
            print('%d/%d' % (cnt, min(args.vis_frames, len(dataset['infos']))))
        # collect instances
        pred_res = res['results'][infos['token']]
        pred_boxes = [
            pred_res[rid]['translation'] + pred_res[rid]['size'] + [
                Quaternion(pred_res[rid]['rotation']).yaw_pitch_roll[0] +
                np.pi / 2
            ] for rid in range(len(pred_res))
        ]
        scores = [
            pred_res[rid]['detection_score'] for rid in range(len(pred_res))
        ]
        labels = [
            pred_res[rid]['detection_name'] for rid in range(len(pred_res))
        ]
        # filter according to thresh
        pred_boxes_array = np.array(pred_boxes)
        scores_array = np.array(scores)
        labels_array = np.array(labels)

        valid_idx = (scores_array > args.vis_thred)
        valid_box, valid_score, valid_labels = pred_boxes_array[valid_idx], \
                                                scores_array[valid_idx], \
                                                    labels_array[valid_idx]

        pred_boxes = valid_box.tolist()
        scores = valid_score.tolist()
        pred_labels = valid_labels.tolist()

        if len(pred_boxes) == 0:
            corners_lidar = np.zeros((0, 3), dtype=np.float32)
            corners_lidar_pred = np.zeros((0, 3), dtype=np.float32)
        else:
            pred_boxes = np.array(pred_boxes, dtype=np.float32)
            pred_boxes = LB(pred_boxes, origin=(0.5, 0.5, 0.0))
            corners_global = pred_boxes.corners.numpy().reshape(-1, 3)
            corners_global = np.concatenate(
                [corners_global,
                 np.ones([corners_global.shape[0], 1])],
                axis=1)
            l2g = get_lidar2global(infos)
            corners_lidar = corners_global @ np.linalg.inv(l2g).T
            corners_lidar_pred = corners_lidar[:, :3]
        pred_flag = np.ones((corners_lidar_pred.shape[0] // 8, ), dtype=np.bool)

        if args.draw_gt:
            gt_boxes = infos['gt_boxes']
            gt_boxes[:, -1] = gt_boxes[:, -1] + np.pi / 2
            width = gt_boxes[:, 4].copy()
            gt_boxes[:, 4] = gt_boxes[:, 3]
            gt_boxes[:, 3] = width
            corners_lidar_gt = \
                LB(infos['gt_boxes'],
                   origin=(0.5, 0.5, 0.5)).corners.numpy().reshape(-1, 3)
            corners_lidar = np.concatenate([corners_lidar_pred, corners_lidar_gt],
                                           axis=0)
            gt_flag = np.ones((corners_lidar_gt.shape[0] // 8), dtype=np.bool)
            pred_flag = np.concatenate(
                [pred_flag, np.logical_not(gt_flag)], axis=0)
            scores = scores + [0 for _ in range(infos['gt_boxes'].shape[0])]
            labels = pred_labels + infos['gt_names'].tolist()
        scores = np.array(scores, dtype=np.float32)
        sort_ids = np.argsort(scores)

        # image view
        # def draw_on_view(corners_lidar, color_op='split', start_idx=0):
        #     imgs = []
        #     for view in views:
        #         ###
        #         img = rc.load_image(infos['cams'][view]['data_path'])
        #         # img = cv2.imread(infos['cams'][view]['data_path'])
        #         # draw instances
        #         corners_img, valid = lidar2img(corners_lidar, infos['cams'][view])
        #         valid = np.logical_and(
        #             valid,
        #             check_point_in_img(corners_img, img.shape[0], img.shape[1]))
        #         valid = valid.reshape(-1, 8)
        #         corners_img = corners_img.reshape(-1, 8, 2).astype(np.int)
        #         for aid in range(valid.shape[0]):
        #             for index in draw_boxes_indexes_img_view:
        #                 if valid[aid, index[0]] and valid[aid, index[1]]:
        #                     if color_op == 'both':
        #                         color = color_map[int(pred_flag[aid])]
        #                     elif color_op == 'split':
        #                         try:
        #                             if labels[start_idx+aid] not in set(color_map_nusc.keys()):
        #                                 print(labels[start_idx+aid])
        #                                 color = (255,0,255)
        #                             else:
        #                                 color = color_map_nusc[labels[start_idx+aid]]
        #                         except:
        #                             pdb.set_trace()
        #                             print(start_idx, aid, valid.shape[0], labels.shape)
        #                             color = (255,0,255)
        #                     cv2.line(
        #                         img,
        #                         tuple(corners_img[aid, index[0]]),
        #                         tuple(corners_img[aid, index[1]]),
        #                         color=color,
        #                         thickness=scale_factor)
        #         imgs.append(img)
        #     return imgs
        def draw_on_view(corners_lidar, color_op='split', start_idx=0, bbo3d=None, type='gt'):
            imgs = []
            for view in views:
                ###
                img = rc.load_image(infos['cams'][view]['data_path'])
                # img = cv2.imread(infos['cams'][view]['data_path'])
                # draw instances
                lidar2img = get_lidar2img(infos['cams'][view])
                if type == 'pred':
                    lidar2img = lidar2img @ np.linalg.inv(l2g)
                
                for aid in range(bbo3d.tensor.shape[0]):
                    color = color_map_nusc[labels[start_idx+aid]]
                    img = draw_lidar_bbox3d_on_img(
                        bbo3d[aid], img, lidar2img, None, color=color, thickness=3)
                    # for index in draw_boxes_indexes_img_view:
                    #     if valid[aid, index[0]] and valid[aid, index[1]]:
                    #         if color_op == 'both':
                    #             color = color_map[int(pred_flag[aid])]
                    #         elif color_op == 'split':
                    #             try:
                    #                 if labels[start_idx+aid] not in set(color_map_nusc.keys()):
                    #                     print(labels[start_idx+aid])
                    #                     color = (255,0,255)
                    #                 else:
                    #                     color = color_map_nusc[labels[start_idx+aid]]
                    #             except:
                    #                 pdb.set_trace()
                    #                 print(start_idx, aid, valid.shape[0], labels.shape)
                    #                 color = (255,0,255)
                    #         cv2.line(
                    #             img,
                    #             tuple(corners_img[aid, index[0]]),
                    #             tuple(corners_img[aid, index[1]]),
                    #             color=color,
                    #             thickness=scale_factor)
                imgs.append(img)
            return imgs

        if not args.split_gt_in_view:
            imgs = draw_on_view(corners_lidar, 'both')
        else:
            if corners_lidar_pred.shape[0] != 0:
                pred_imgs = draw_on_view(corners_lidar_pred, 'split', 0, pred_boxes, 'pred')
            gt_imgs = draw_on_view(corners_lidar_gt, 'split', len(pred_labels), 
                                    LB(infos['gt_boxes'],origin=(0.5, 0.5, 0.5)), 'gt')


        # bird-eye-view
        # canvas = np.zeros((int(canva_size), int(canva_size), 3),
        #                   dtype=np.uint8)
        canvas = np.ones((int(canva_size), int(canva_size), 3),
                          dtype=np.uint8) * 255

        pts_bytes = rc.file_client.get(infos['lidar_path'])
        lidar_points = np.frombuffer(pts_bytes, dtype=np.float32)
        lidar_points = lidar_points.copy()

        # lidar_points = np.fromfile(infos['lidar_path'], dtype=np.float32)
        lidar_points = lidar_points.reshape(-1, 5)[:, :3]
        lidar_points[:, 1] = -lidar_points[:, 1]
        lidar_points[:, :2] = \
            (lidar_points[:, :2] + show_range) / show_range / 2.0 * canva_size
        for p in lidar_points:
            if check_point_in_img(
                    p.reshape(1, 3), canvas.shape[1], canvas.shape[0])[0]:
                color = depth2color(p[2])
                cv2.circle(
                    canvas, (int(p[0]), int(p[1])),
                    radius=0,
                    color=color,
                    thickness=1)

        # draw instances
        corners_lidar = corners_lidar.reshape(-1, 8, 3)
        corners_lidar[:, :, 1] = -corners_lidar[:, :, 1]
        bottom_corners_bev = corners_lidar[:, [0, 3, 7, 4], :2]
        bottom_corners_bev = \
            (bottom_corners_bev + show_range) / show_range / 2.0 * canva_size
        bottom_corners_bev = np.round(bottom_corners_bev).astype(np.int32)
        center_bev = corners_lidar[:, [0, 3, 7, 4], :2].mean(axis=1)
        head_bev = corners_lidar[:, [0, 4], :2].mean(axis=1)
        canter_canvas = \
            (center_bev + show_range) / show_range / 2.0 * canva_size
        center_canvas = canter_canvas.astype(np.int32)
        head_canvas = (head_bev + show_range) / show_range / 2.0 * canva_size
        head_canvas = head_canvas.astype(np.int32)

        for rid in sort_ids:
            score = scores[rid]
            if score < args.vis_thred and pred_flag[rid]:
                continue
            score = min(score * 2.0, 1.0) if pred_flag[rid] else 1.0
            color = color_map[int(pred_flag[rid])]
            for index in draw_boxes_indexes_bev:
                cv2.line(
                    canvas,
                    tuple(bottom_corners_bev[rid, index[0]]),
                    tuple(bottom_corners_bev[rid, index[1]]),
                    # [color[0] * score, color[1] * score, color[2] * score],
                    [color[0], color[1], color[2]],
                    thickness=1,
                    lineType=cv2.LINE_AA)
            cv2.line(
                canvas,
                tuple(center_canvas[rid]),
                tuple(head_canvas[rid]),
                # [color[0] * score, color[1] * score, color[2] * score],
                [color[0], color[1], color[2]],
                1,
                lineType=cv2.LINE_AA)

    #     # fuse image-view and bev
    #     img = np.zeros((900 * 2 + canva_size * scale_factor, 1600 * 3, 3),
    #                    dtype=np.uint8)
    #     img[:900, :, :] = np.concatenate(imgs[:3], axis=1)
    #     img_back = np.concatenate(
    #         [imgs[3][:, ::-1, :], imgs[4][:, ::-1, :], imgs[5][:, ::-1, :]],
    #         axis=1)
    #     img[900 + canva_size * scale_factor:, :, :] = img_back
    #     img = cv2.resize(img, (int(1600 / scale_factor * 3),
    #                            int(900 / scale_factor * 2 + canva_size)))
    #     w_begin = int((1600 * 3 / scale_factor - canva_size) // 2)
    #     img[int(900 / scale_factor):int(900 / scale_factor) + canva_size,
    #         w_begin:w_begin + canva_size, :] = canvas

    #     if args.format == 'image':
    #         cv2.imwrite(os.path.join(vis_dir, '%s.jpg' % infos['token']), img)
    #     elif args.format == 'video':
    #         vout.write(img)
    # if args.format == 'video':
    #     vout.release()
        save_path = os.path.join(vis_dir, f"{infos['token']}")
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        os.makedirs(save_path)
        if not args.split_gt_in_view:
            for idx, img in enumerate(imgs):
                cv2.imwrite(os.path.join(save_path, f'{idx}.png'), img)
        else:
            for idx, img in enumerate(gt_imgs):
                cv2.imwrite(os.path.join(save_path, f'gt_{idx}.png'), img)
            for idx, img in enumerate(pred_imgs):
                cv2.imwrite(os.path.join(save_path, f'pred_{idx}.png'), img)
        cv2.imwrite(os.path.join(save_path, f'bev.png'), canvas)


if __name__ == '__main__':
    rc = RC()
    main()
