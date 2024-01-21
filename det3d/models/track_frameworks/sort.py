
import torch
from mmdet3d.models import build_detector, build_head

from mmtrack.core import outs2results, results2outs
from mmdet.models import DETECTORS # use detectors for registory.
from mmtrack.models.builder import build_motion, build_tracker
from mmtrack.models.mot.base import BaseMultiObjectTracker

@DETECTORS.register_module()
class CustomSORT(BaseMultiObjectTracker):
    """OCSORT in 3D task.
    """

    def __init__(self,
                keep_classes=None,
                detector=None,
                tracker=None,
                motion=None,
                get_feats='bev',
                init_cfg=None,
                pretrained=None,
                train_cfg=None,
                test_cfg=None,
                *args,
                **kwargs):
        
        super().__init__(init_cfg)
        self.pretrained = pretrained
        self.train_cfg = train_cfg
        self.test_cf = test_cfg

        self.frame_id = -1
        self.keep_classes = keep_classes
        if detector is not None:
            self.detector = build_detector(detector)
        self.detector.init_weights()
        if motion is not None:
            self.motion = build_motion(motion)

        if tracker is not None:
            self.tracker = build_tracker(tracker)

        self.get_feats = get_feats


    
    def forward_train(self, *args, **kwargs):
            """ Forward function during training"""
            return self.detector.forward_train(*args, **kwargs)


    # should remove it in the future. Use for balance with bevdet
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
                    rescale=False,
                    **kwargs):
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

        is_first_frame = img_metas[0].get('is_first_frame', -1)

        if is_first_frame == True:
            self.tracker.reset()
            self.frame_id = -1
        self.frame_id += 1

        bbox_results = self.detector.simple_test(
            img_metas, img, img_inputs=img_inputs)
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

        track_bboxes, track_scores, track_labels, track_ids = self.tracker.track(
                                                            img_metas=img_metas,
                                                            model=self,
                                                            bboxes=bboxes,
                                                            bbox_confidences=scores,
                                                            labels=labels, 
                                                            frame_id=self.frame_id,
                                                            rescale=rescale,
                                                            **kwargs)
        
        return [dict(
            boxes_3d=track_bboxes,
            scores_3d=track_scores,
            labels_3d=track_labels,
            ids=track_ids)]
        

