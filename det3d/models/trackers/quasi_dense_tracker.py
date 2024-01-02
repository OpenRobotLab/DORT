import torch
import torch.nn.functional as F
from mmdet.core import bbox_overlaps
from mmdet3d.core import bbox_overlaps_nearest_3d
from mmtrack.models.builder import TRACKERS
from mmtrack.models.trackers import BaseTracker, QuasiDenseTracker

# def calcular_bbox_overlaps(bboxes1, bboxes2, overlap_func, mode="bev"):
    # if mode == 

@TRACKERS.register_module()
class CustomQuasiDenseTracker(QuasiDenseTracker):
    """Tracker for Quasi-Dense Tracking.

    Args:
        init_score_thr (float): The cls_score threshold to
            initialize a new tracklet. Defaults to 0.8.
        obj_score_thr (float): The cls_score threshold to
            update a tracked tracklet. Defaults to 0.5.
        match_score_thr (float): The match threshold. Defaults to 0.5.
        memo_tracklet_frames (int): The most frames in a tracklet memory.
            Defaults to 10.
        memo_backdrop_frames (int): The most frames in the backdrops.
            Defaults to 1.
        memo_momentum (float): The momentum value for embeds updating.
            Defaults to 0.8.
        nms_conf_thr (float): The nms threshold for confidence.
            Defaults to 0.5.
        nms_backdrop_iou_thr (float): The nms threshold for backdrop IoU.
            Defaults to 0.3.
        nms_class_iou_thr (float): The nms threshold for class IoU.
            Defaults to 0.7.
        with_cats (bool): Whether to track with the same category.
            Defaults to True.
        match_metric (str): The match metric. Defaults to 'bisoftmax'.
        support 3 version:
            1. 3D, 2. BEV, 3. Fov
    """
    def __init__(self,
                 init_score_thr=0.8,
                 obj_score_thr=0.5,
                 match_score_thr=0.5,
                 memo_tracklet_frames=10,
                 memo_backdrop_frames=1,
                 memo_momentum=0.8,
                 nms_conf_thr=0.5,
                 nms_backdrop_iou_thr=0.3,
                 nms_class_iou_thr=0.7,
                 with_cats=True,
                 match_metric='bisoftmax',
                 track_space="bev",

                 **kwargs):
        BaseTracker.__init__(self, **kwargs)
        assert 0 <= memo_momentum <= 1.0
        assert memo_tracklet_frames >= 0
        assert memo_backdrop_frames >= 0
        self.init_score_thr = init_score_thr
        self.obj_score_thr = obj_score_thr
        self.match_score_thr = match_score_thr
        self.memo_tracklet_frames = memo_tracklet_frames
        self.memo_backdrop_frames = memo_backdrop_frames
        self.memo_momentum = memo_momentum
        self.nms_conf_thr = nms_conf_thr
        self.nms_backdrop_iou_thr = nms_backdrop_iou_thr
        self.nms_class_iou_thr = nms_class_iou_thr
        self.with_cats = with_cats
        assert match_metric in ['bisoftmax', 'softmax', 'cosine']
        self.match_metric = match_metric

        self.num_tracks = 0
        self.tracks = dict()
        self.backdrops = []
        self.track_space = track_space
        if self.track_space == "bev":
            self.iou_overlaps_func = bbox_overlaps_nearest_3d


    def reset(self):
        """Reset the buffer of the tracker."""
        self.num_tracks = 0
        self.tracks = dict()
        self.backdrops = []

    def update(self, ids, bboxes, bbox_confidences, embeds, labels, frame_id):
        """Tracking forward function.

        Args:
            ids (Tensor): of shape(N, ).
            bboxes: A 3D bounding box instances.
            bbox_confidences (Tensor): of shape(N, ).
            embeds (Tensor): of shape (N, 256).
            labels (Tensor): of shape (N, ).
            # frame_id (int): The id of current frame, 0-index.
        """
        tracklet_inds = ids > -1

        for id, bbox, bbox_confidence, embed, label in zip(ids[tracklet_inds],
                                          bboxes.tensor[tracklet_inds],
                                          bbox_confidences[tracklet_inds],
                                          embeds[tracklet_inds],
                                          labels[tracklet_inds]):
            id = int(id)
            # update the tracked ones and initialize new tracks
            try:
                if id in self.tracks.keys():
                    velocity = (bbox - self.tracks[id]['bbox']) / (
                        frame_id - self.tracks[id]['last_frame'])
                    self.tracks[id]['bbox'] = bbox
                    self.tracks[id]['embed'] = (
                        1 - self.memo_momentum
                    ) * self.tracks[id]['embed'] + self.memo_momentum * embed
                    # self.tracks[id]['last_frame'] = frame_id
                    self.tracks[id]['label'] = label
                    self.tracks[id]['velocity'] = (
                        self.tracks[id]['velocity'] * self.tracks[id]['acc_frame']
                        + velocity) / (self.tracks[id]['acc_frame'] + 1)
                    self.tracks[id]['acc_frame'] += 1
            except:
                import pdb; pdb.set_trace()
            else:
                self.tracks[id] = dict(
                    bbox=bbox,
                    bbox_confidence=bbox_confidence,
                    embed=embed,
                    label=label,
                    last_frame=frame_id,
                    velocity=torch.zeros_like(bbox),
                    acc_frame=0)
        import pdb; pdb.set_trace()
        # backdrop update according to IoU
        backdrop_inds = torch.nonzero(ids == -1, as_tuple=False).squeeze(1)
        ious = self.iou_overlaps_func(bboxes[backdrop_inds].tensor, bboxes.tensor)
        import pdb; pdb.set_trace()

        # ious = bbox_overlaps(bboxes[backdrop_inds, :4], bboxes[:, :4])
        for i, ind in enumerate(backdrop_inds):
            if (ious[i, :ind] > self.nms_backdrop_iou_thr).any():
                backdrop_inds[i] = -1
        backdrop_inds = backdrop_inds[backdrop_inds > -1]
        # old backdrops would be removed at first
        self.backdrops.insert(
            0,
            dict(bboxes=bboxes[backdrop_inds],
                 bbox_confidences=bbox_confidences[backdrop_inds],
                 embeds=embeds[backdrop_inds],
                 labels=labels[backdrop_inds]))

        # pop memo
        import pdb; pdb.set_trace()
        invalid_ids = []
        for k, v in self.tracks.items():
            if frame_id - v['last_frame'] >= self.memo_tracklet_frames:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracks.pop(invalid_id)

        if len(self.backdrops) > self.memo_backdrop_frames:
            self.backdrops.pop()

    @property
    def memo(self):
        """Get tracks memory."""
        memo_embeds = []
        memo_ids = []
        memo_bboxes = []
        memo_bbox_confidences = []
        memo_labels = []
        # velocity of tracks
        memo_vs = []
        # get tracks
        for k, v in self.tracks.items():
            memo_bboxes.append(v['bbox'][None, :])
            memo_bbox_confidences.append(v['bbox_confidence'][None, :])
            memo_embeds.append(v['embed'][None, :])
            memo_ids.append(k)
            memo_labels.append(v['label'].view(1, 1))
            memo_vs.append(v['velocity'][None, :])
        memo_ids = torch.tensor(memo_ids, dtype=torch.long).view(1, -1)
        # get backdrops
        for backdrop in self.backdrops:
            backdrop_ids = torch.full((1, backdrop['embeds'].size(0)),
                                      -1,
                                      dtype=torch.long)
            backdrop_vs = torch.zeros_like(backdrop['bboxes'])
            memo_bboxes.append(backdrop['bboxes'])
            memo_bbox_confidences.append(backdrop['bbox_confidences'])
            memo_embeds.append(backdrop['embeds'])
            memo_ids = torch.cat([memo_ids, backdrop_ids], dim=1)
            memo_labels.append(backdrop['labels'][:, None])
            memo_vs.append(backdrop_vs)

        memo_bboxes = torch.cat(memo_bboxes, dim=0)
        memo_bbox_confidences = torch.cat(memo_bbox_confidences, dim=0)
        memo_embeds = torch.cat(memo_embeds, dim=0)
        memo_labels = torch.cat(memo_labels, dim=0).squeeze(1)
        memo_vs = torch.cat(memo_vs, dim=0)
        return memo_bboxes, memo_bbox_confidences, \
                memo_labels, memo_embeds, memo_ids.squeeze(
            0), memo_vs

    def track(self, img_metas,
                    feats, 
                    model, 
                    bboxes, 
                    bbox_confidences,
                    labels,
                    frame_id):
        """Tracking forward function.

        Args:
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            feats (tuple): Backbone features of the input image. 
                if bev or fov: input 4D shape features. 
                if 3d: input 5D shape features. 
            model (nn.Module): The forward model.
            bboxes A 3D bounding boxes.
            bbox_confidences (Tensor): of shape (N, ).
                use bbox_confidences to avoid the conflict with scores.
            labels (Tensor): of shape (N, ).
            frame_id (int): The id of current frame, 0-index.

        Returns:
            list: Tracking results.
        """
        # return zero bboxes if there is no track targets
        # get labels from bboxes
        if len(bboxes) == 0:
            ids = torch.zeros_like(labels)
            return bboxes, bbox_confidences, labels, ids
        # get track feats

        track_bboxes = bboxes
        
        #TODO check the feature module.
        # Ignore the scale mechanism 
        track_feats = model.track_head.extract_bbox_feats(
            feats, [track_bboxes], img_metas)
        # sort according to the object_score
        _, inds = bbox_confidences.sort(descending=True)

        bbox_confidences = bbox_confidences[inds]
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        embeds = track_feats[inds, :]

        # duplicate removal for potential backdrops and cross classes
        valids = bboxes.tensor.new_ones((bboxes.tensor.size(0)))
        ious = self.iou_overlaps_func(bboxes.tensor, bboxes.tensor)
        for i in range(1, bboxes.tensor.size(0)):
            thr = self.nms_backdrop_iou_thr if bbox_confidences[
                i] < self.obj_score_thr else self.nms_class_iou_thr
            if (ious[i, :i] > thr).any():
                valids[i] = 0
        valids = valids == 1
        bboxes.tensor = bboxes.tensor[valids, :]
        bbox_confidences = bbox_confidences[valids]
        labels = labels[valids]
        embeds = embeds[valids, :]

        # init ids container
        ids = torch.full((bboxes.tensor.size(0), ), -1, dtype=torch.long)

        # match if buffer is not empty
        if bboxes.tensor.size(0) > 0 and not self.empty:

            # TODO what is memo_vs
            (memo_bboxes, memo_bbox_confidences, memo_labels, memo_embeds, memo_ids,
             memo_vs) = self.memo

            if self.match_metric == 'bisoftmax':
                feats = torch.mm(embeds, memo_embeds.t())
                d2t_scores = feats.softmax(dim=1)
                t2d_scores = feats.softmax(dim=0)
                scores = (d2t_scores + t2d_scores) / 2
            elif self.match_metric == 'softmax':
                feats = torch.mm(embeds, memo_embeds.t())
                scores = feats.softmax(dim=1)
            elif self.match_metric == 'cosine':
                scores = torch.mm(F.normalize(embeds, p=2, dim=1),
                                  F.normalize(memo_embeds, p=2, dim=1).t())
            else:
                raise NotImplementedError
            # track with the same category
            if self.with_cats:
                cat_same = labels.view(-1, 1) == memo_labels.view(1, -1)
                scores *= cat_same.float().to(scores.device)
            # track according to scores
            for i in range(bboxes.tensor.size(0)):
                conf, memo_ind = torch.max(scores[i, :], dim=0)
                id = memo_ids[memo_ind]
                if conf > self.match_score_thr:
                    if id > -1:
                        # keep bboxes with high object score
                        # and remove background bboxes
                        if bbox_confidences[i] > self.obj_score_thr:
                            ids[i] = id
                            scores[:i, memo_ind] = 0
                            scores[i + 1:, memo_ind] = 0
                        else:
                            if conf > self.nms_conf_thr:
                                ids[i] = -2
        # initialize new tracks
        import pdb; pdb.set_trace()
        new_inds = (ids == -1) & (bbox_confidences > self.init_score_thr).cpu()
        import pdb; pdb.set_trace()
        num_news = new_inds.sum()
        ids[new_inds] = torch.arange(self.num_tracks,
                                     self.num_tracks + num_news,
                                     dtype=torch.long)
        import pdb; pdb.set_trace()
        self.num_tracks += num_news

        self.update(ids, bboxes, bbox_confidences, embeds, labels, frame_id)

        return bboxes, bbox_confidences, labels, ids
