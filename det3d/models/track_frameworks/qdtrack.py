import torch
from mmdet3d.models import build_detector, build_head
# from mmtrack.core import outs2results, results2outs
from mmtrack.models.builder import build_tracker
from mmtrack.models.mot import BaseMultiObjectTracker, QDTrack
from mmdet.models import DETECTORS # use detectors for registory.
from mmcv.runner import auto_fp16
from mmdet3d.core import bbox3d2result
from mmdet3d.models.dense_heads import CenterHead


@DETECTORS.register_module()
class CustomQDTrack(QDTrack):
    def __init__(self,
                 keep_classes=None,
                 detector=None,
                 track_head=None,
                 tracker=None,
                 freeze_detector=False,
                 get_feats='bev',
                 pretrained=None,
                 train_cfg=None,
                 test_cfg=None,
                 *args,
                 **kwargs):
        BaseMultiObjectTracker.__init__(self, *args, **kwargs)
        self.pretrained = pretrained
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if detector is not None:
            self.detector = build_detector(detector)

        if track_head is not None:
            self.track_head = build_head(track_head)

        if tracker is not None:
            self.tracker = build_tracker(tracker)

        self.freeze_detector = freeze_detector
        if self.freeze_detector:
            self.freeze_module('detector')

        self.get_feats = get_feats
        self.frame_id = -1
        self.keep_classes = keep_classes

    def forward_train(self,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      gt_match_indices,
                      ref_img_metas,
                      ref_gt_bboxes_3d,
                      ref_gt_labels_3d,
                      ref_gt_match_indices,
                      img=None,
                      ref_img=None,
                      img_inputs=None,
                      ref_img_inputs=None,
                      gt_bboxes_ignore=None,
                      ref_gt_bboxes_ignore=None,
                      **kwargs):
        """Forward function during training.

         Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            gt_bboxes (list[Tensor]): Ground truth bboxes of the image,
                each item has a shape (num_gts, 4).
            gt_labels (list[Tensor]): Ground truth labels of all images.
                each has a shape (num_gts,).
            gt_match_indices (list(Tensor)): Mapping from gt_instance_ids to
                ref_gt_instance_ids of the same tracklet in a pair of images.
            ref_img (Tensor): of shape (N, C, H, W) encoding input reference
                images. Typically these should be mean centered and std scaled.
            ref_img_metas (list[dict]): list of reference image info dict where
                each dict has: 'img_shape', 'scale_factor', 'flip', and may
                also contain 'filename', 'ori_shape', 'pad_shape',
                and 'img_norm_cfg'.
            ref_gt_bboxes (list[Tensor]): Ground truth bboxes of the
                reference image, each item has a shape (num_gts, 4).
            ref_gt_labels (list[Tensor]): Ground truth labels of all
                reference images, each has a shape (num_gts,).
            gt_masks (list[Tensor]) : Masks for each bbox, has a shape
                (num_gts, h , w).
            gt_bboxes_ignore (list[Tensor], None): Ground truth bboxes to be
                ignored, each item has a shape (num_ignored_gts, 4).
            ref_gt_bboxes_ignore (list[Tensor], None): Ground truth bboxes
                of reference images to be ignored,
                each item has a shape (num_ignored_gts, 4).
            ref_gt_masks (list[Tensor]) : Masks for each reference bbox,
                has a shape (num_gts, h , w).

        Returns:
            dict[str : Tensor]: All losses.
        """
        import pdb; pdb.set_trace()
        losses = {}
        det_loss, x_3d, x_bev, x_fov, x = self.detector.forward_train(img_metas,
                                               img, 
                                               img_inputs, 
                                               gt_bboxes_3d, 
                                               gt_labels_3d,
                                               return_output=True)
        losses.update(det_loss)
        proposals = self.get_bboxes(x, img_metas, self.detector.bbox_head)

        ref_x_3d, ref_x_bev, ref_x_fov = self.extract_feat(ref_img, ref_img_metas)
        ref_x = self.bbox_head(ref_x_bev)
        ref_proposals = self.get_bboxes(
                ref_x, ref_img_metas, self.detector.bbox_head)
        if self.get_feats == "bev":
            x = x_bev
            ref_x = ref_x_bev
        track_losses = self.track_head.forward_train(
            x, img_metas, proposals, gt_bboxes_3d, gt_labels_3d,
            gt_match_indices, ref_x, ref_img_metas, ref_proposals,
            ref_gt_bboxes_3d, ref_gt_labels_3d, gt_bboxes_ignore)
        
        losses.update(track_losses)
        return losses
        
    
    def get_bboxes(self, x, img_metas, bbox_head):
        if isinstance(bbox_head, CenterHead):
            bbox_list = bbox_head.get_bboxes(*x, img_metas)
        else:
            bbox_list = bbox_head.get_bboxes(x, img_metas, rescale=False)
        bbox_results = [
            bbox3d2result(det_bboxes, det_scores, det_labels)
            for det_bboxes, det_scores, det_labels in bbox_list
        ]
        return bbox_results

    # should remove it in the future. Use for balance with bevdet
    @auto_fp16(apply_to=('img', ))
    def forward(self, img_metas,
                      img=None,
                      img_inputs=None,
                      return_loss=True, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.onnx_export(img[0], img_metas[0])
        if return_loss:
            import pdb; pdb.set_trace()
            return self.forward_train(img_metas,
                                      img=img,
                                      img_inputs=img_inputs,
                                      **kwargs)
        else:
            return self.forward_test(img_metas, img=img, img_inputs=img_inputs, **kwargs)
    def forward_test(self,
                    img_metas,
                    img=None, 
                    img_inputs=None, 
                    **kwargs):
        # currently the model do not support test time augmentation
        if not isinstance(img, list):
            img = [img]
            img_metas = [img_metas]
        return self.simple_test(img_metas[0], img[0], img_inputs=img_inputs)


    def simple_test(self,
                    img_metas,
                    img=None,
                    img_inputs=None,
                    rescale=False):
        """Test forward.

         Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            rescale (bool): whether to rescale the bboxes.

        Returns:
            dict[str : Tensor]: Track results.
        """
        if img is None:
            img = img_inputs
        is_first_frame = img_metas[0].get('is_first_frame', True)

        if is_first_frame == True:
            self.tracker.reset()
            self.frame_id = -1
        
        self.frame_id += 1
        bbox_results, feats = self.detector.simple_test(
            img_metas, img, img_inputs=img_inputs, get_feats=self.get_feats)
        bbox_results = bbox_results[0]
        # det_bboxes = torch.tensor
        bboxes = bbox_results['boxes_3d']
        scores = bbox_results['scores_3d']
        labels = bbox_results['labels_3d']

        if self.keep_classes is not None:
            mask = sum(labels == i for i in self.keep_classes).bool()
            bboxes.tensor = bboxes.tensor[mask]
            scores = scores[mask]
            labels = labels[mask]

        # TODO check how to extract features and handle the data conversion issue.
        bboxes, scores, labels, ids = self.tracker.track(img_metas,
                                                        feats=feats,
                                                        model=self,
                                                        bboxes=bboxes,
                                                        bbox_confidences=scores,
                                                        labels=labels,
                                                        frame_id=self.frame_id)
        return [dict(
                boxes_3d=bboxes, 
                scores_3d=scores, 
                labels_3d=labels, 
                ids=ids)]

