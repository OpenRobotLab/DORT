import lap
import numpy as np
import torch
from addict import Dict
from mmcv.runner import force_fp32
from mmdet.core import bbox_overlaps
from mmtrack.models.trackers import SortTracker
from mmtrack.models.builder import TRACKERS
from mmtrack.models.trackers import BaseTracker
from motmetrics.lap import linear_sum_assignment
from mmdet3d.core import bbox_overlaps_nearest_3d, bbox_overlaps_3d
from .utils import mmdet3d_to_nusc_bbox, nusc_to_mmdet3d_bbox, \
                        lidar_nusc_bbox_to_global

@TRACKERS.register_module()
class CustomSORTTracker(SortTracker):
    """Tracker for OC-SORT.

    Args:
        obj_score_thrs (float): Detection score threshold for matching objects.
            Defaults to 0.3.
        init_track_thr (float): Detection score threshold for initializing a
            new tracklet. Defaults to 0.7.
        weight_iou_with_det_scores (bool): Whether using detection scores to
            weight IOU which is used for matching. Defaults to True.
        match_iou_thr (float): IOU distance threshold for matching between two
            frames. Defaults to 0.3.
        num_tentatives (int, optional): Number of continuous frames to confirm
            a track. Defaults to 3.
        vel_consist_weight (float): Weight of the velocity consistency term in
            association (OCM term in the paper).
        vel_delta_t (int): The difference of time step for calculating of the
            velocity direction of tracklets.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """
    def __init__(self,
                 obj_score_thr=0.3,
                 init_track_thr=0.7,
                #  weight_iou_with_det_scores=True,
                 matching_threshold=0.3,
                 num_tentatives=3,
                #  vel_consist_weight=0.2,
                #  vel_delta_t=3,
                 overlap_mode = "align_bev",
                 dist_mode = "iou",
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg, **kwargs)
        self.obj_score_thr = obj_score_thr
        self.init_track_thr = init_track_thr
        self.dist_mode = dist_mode

        # self.weight_iou_with_det_scores = weight_iou_with_det_scores
        self.matching_threshold = matching_threshold
        # self.vel_consist_weight = vel_consist_weight
        # self.vel_delta_t = vel_delta_t

        self.num_tentatives = num_tentatives
        self.overlap_mode = overlap_mode
        if overlap_mode == "align_bev":
            self.num_bbox_var = 4
            self.num_loc_var = 2
        elif overlap_mode == "bev":
            self.num_bbox_var = 5
            self.num_loc_var = 2
        elif overlap_mode == "3d":
            self.num_bbox_var = 7
            self.num_loc_var = 3 

        

    @property
    def unconfirmed_ids(self):
        """Unconfirmed ids in the tracker."""
        ids = [id for id, track in self.tracks.items() if track.tentative]
        return ids

    @property
    def confirmed_ids(self):
        """Confirmed ids in the tracker."""
        ids = [id for id, track in self.tracks.items() if not track.tentative]
        return ids

    def init_track(self, id, obj):
        """Initialize a track."""
        BaseTracker.init_track(self, id, obj)
        
        # TODO
        if self.tracks[id].frame_ids[-1] == 0:
            self.tracks[id].tentative = False
        else:
            self.tracks[id].tentative = True
        
        # TODO convert the bounding boxes to xyxy

        bbox = (self.tracks[id].bboxes)
        # bbox =  (self.tracks[id].bboxes[-1])  # size = (1, 4)
        # assert bbox.ndim == 2 and bbox.shape[0] == 1
        # bbox = bbox.squeeze(0).cpu().numpy()

        self.tracks[id].kf = self.kf.initiate(
            bbox)
        # track.obs maintains the history associated detections to this track
        self.tracks[id].obs = []
        bbox_id = self.memo_items.index('bboxes')
        self.tracks[id].obs.append(obj[bbox_id])
        # a placefolder to save mean/covariance before losing tracking it
        # parameters to save: mean, covariance, measurement
        self.tracks[id].tracked = True
        self.tracks[id].saved_attr = Dict()
        self.tracks[id].velocity = torch.tensor(
            (-1, -1)).to(obj[bbox_id].device)  # placeholder

    def update_track(self, id, obj):
        """Update a track."""
        BaseTracker.update_track(self, id, obj)
        if self.tracks[id].tentative:
            if len(self.tracks[id]['bboxes']) >= self.num_tentatives:
                self.tracks[id].tentative = False
        # bbox = bbox_xyxy_to_cxcyah(self.tracks[id].bboxes[-1])  # size = (1, 4)
        # assert bbox.ndim == 2 and bbox.shape[0] == 1
        # bbox = bbox.squeeze(0).cpu().numpy()
        # bbox.tensor = bbox.tensor.cpu()
        bbox = self.tracks[id]['bboxes'][-1]
        self.tracks[id].kf = self.kf.update(
            self.tracks[id].kf, bbox)
        self.tracks[id].tracked = True
        bbox_id = self.memo_items.index('bboxes')
        self.tracks[id].obs.append(obj[bbox_id])
        # bbox1 = self.k_step_observation(self.tracks[id])
        # bbox2 = obj[bbox_id]
        # self.tracks[id].velocity = self.vel_direction(bbox1, bbox2).to(
        #     obj[bbox_id].device)

    def align_bboxes(self, bboxes, img_metas):
        # 1. OUTPUT TO NUSC bboxes
        # 2. lidar to ego to global
        # 3. nusc bbox to mmdet3d bboxes
        bboxes = mmdet3d_to_nusc_bbox(bboxes)
        bboxes = lidar_nusc_bbox_to_global(bboxes, img_metas)
        bboxes = nusc_to_mmdet3d_bbox(bboxes)
        return bboxes    

    @force_fp32(apply_to=('img', ))
    def track(self,
              img_metas,
              model,
              bboxes,
              bbox_confidences,
              labels,
              frame_id,
              rescale=False,
              **kwargs):
        """Tracking forward function.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            model (nn.Module): MOT model.
            bboxes (Tensor): of shape (N, 5).
            labels (Tensor): of shape (N, ).
            frame_id (int): The id of current frame, 0-index.
            rescale (bool, optional): If True, the bounding boxes should be
                rescaled to fit the original scale of the image. Defaults to
                False.

        Returns:
            tuple: Tracking results.
        """
        if not hasattr(self, 'kf'):
            self.kf = model.motion
        valid_inds = bbox_confidences > self.obj_score_thr
        bboxes.tensor = bboxes.tensor[valid_inds]
        labels = labels[valid_inds]
        bbox_confidences = bbox_confidences[valid_inds]
        ori_bboxes = bboxes
        bboxes = self.align_bboxes(bboxes, img_metas[0])
        if self.empty or len(bboxes) == 0:
            num_new_tracks = len(bboxes)
            ids = torch.arange(
                self.num_tracks,
                self.num_tracks + num_new_tracks,
                dtype=torch.long)
            self.num_tracks += num_new_tracks
            # if self.with_reid:
            #     embeds = model.reid.simple_test(
            #         self.crop_imgs(reid_img, img_metas, bboxes[:, :4].clone(),
            #                        rescale))
        else:
            ids = torch.full((len(bboxes), ), -1, dtype=torch.long)
            # motion
            if model.with_motion:
                self.tracks, costs = model.motion.track(
                    self.tracks, bboxes)

            active_ids = self.confirmed_ids
            active_ids = [
                id for id in self.ids if id not in ids
                and self.tracks[id].frame_ids[-1] == frame_id - 1
            ]
            if len(active_ids) > 0:
                active_dets = torch.nonzero(ids == -1).squeeze(1)
                track_bboxes = self.get('bboxes', active_ids)
                association_matrix = self.get_association_matrix(
                    track_bboxes, bboxes[active_dets]).cpu().numpy()
                # dists = 1 - association_matrix
                row, col = linear_sum_assignment(association_matrix)
                for r, c in zip(row, col):
                    dist = association_matrix[r, c]
                    if dist > self.matching_threshold:
                        ids[active_dets[c]] = active_ids[r]

            new_track_inds = ids == -1
            ids[new_track_inds] = torch.arange(
                self.num_tracks,
                self.num_tracks + new_track_inds.sum(),
                dtype=torch.long)
            self.num_tracks += new_track_inds.sum()
        self.update(
            ids=ids,
            bboxes=bboxes.tensor,
            bbox_confidences=bbox_confidences,
            labels=labels,
            embeds=None,
            frame_ids=frame_id)
        return ori_bboxes, bbox_confidences, labels, ids

    def get_association_matrix(self, source_bboxes, target_bboxes):
        if self.dist_mode == "iou":
            source_bboxes = source_bboxes[:,:7]
            target_bboxes = target_bboxes.tensor[:,:7]

            iou = bbox_overlaps_3d(source_bboxes, target_bboxes)
            iou = 1-iou
            return iou
        elif self.dist_mode == "l1":
            source_loc = source_bboxes[:,:3]
            target_loc = target_bboxes.tensor[:,:3]
            dist = (source_loc[:, None] - target_loc[None, :]).abs()
            dist = dist.mean(dim=-1)
            # dist = dist / 10
            return dist
