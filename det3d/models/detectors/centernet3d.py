import torch
from mmdet.models import DETECTORS, build_backbone, build_head, build_neck

from mmdet3d.core import bbox3d2result

from mmdet.models.detectors import BaseDetector

from mmdet3d.core import bbox3d2result

#from mmdet3d.core.utils.mask import mask_background_region

import os.path as osp

@DETECTORS.register_module()
class CenterNet3D(BaseDetector):
    def __init__(self,
            backbone,
            neck,
            bbox_head,
            two_stage_infer=True,
            train_cfg=None,
            nocs_head=None,
            two_stage_head=None,
            test_cfg=None,
            pretrained=None,
            init_cfg=None):
        '''
        Args:

        '''
        super().__init__(init_cfg)
        if pretrained:
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone) # backbone from DLA 34

        self.neck = build_neck(neck) # identity module

        self.bbox_head = build_head(bbox_head) # centernet head
        self.pretrained=pretrained

        if nocs_head is not None:
            self.pred_nocs=True
            self.nocs_head = build_head(nocs_head)
        else:
            self.nocs_head = None
            self.pred_nocs = False


        self.two_stage_infer = two_stage_infer
        self.two_stage_head = build_head(two_stage_head) if two_stage_head is not None else None


    def extract_feat(self, img, img_metas, mode):
        batch_size = img.shape[0]
        N, V, C, H, W = img.shape
        # img = img.reshape([-1] + list(img.shape)[2:])
        img = img.reshape(-1, C, H, W)
        x = self.backbone(img)
        x = self.neck(x)
        if isinstance(x, tuple):
            x = [x[0]]

        # features_2d = self.head()
        #  = self.bbox_head.forward(x[-1], img_metas)
        return x

    def forward_train(self, img, img_metas, gt_bboxes_3d, gt_labels_3d, **kwargs):
        # get
        x = self.extract_feat(img, img_metas, "train")
        # if "dense_depth" in kwargs:
        #     dense_depth = kwargs["dense_depth"]
        # else:
        #     dense_depth = None
        losses, preds = self.bbox_head.forward_train(
            x, img_metas, gt_bboxes_3d, gt_labels_3d, **kwargs)
        if self.pred_nocs:
            losses_ncos, preds_nocs = self.nocs_head.forward_train(
                x, img_metas, **kwargs)
            preds.update(preds_nocs)
            preds["normalize_nocs"] = self.nocs_head.normalize_nocs
            losses.update(losses_ncos)

        if self.two_stage_head is not None:
            preds["features"] = x
            gt_bboxes = kwargs["gt_bboxes"]
            bbox_list = self.bbox_head.get_bboxes(preds, img_metas)
            losses_two_stage, preds = self.two_stage_head.forward_train(
                                bbox_list, preds, img, img_metas,
                                gt_bboxes_3d, gt_labels_3d, gt_bboxes)
            losses.update(losses_two_stage)

        return losses


    def forward_dummy(self, img):

        if img.dim() == 4: # fix the situation of multiview inputs
            img = img.unsqueeze(0)
        x = self.extract_feat(img, None, 'test')
        outs = self.bbox_head.forward(x)
        return outs

    def forward_semi_train(self, img, img_metas, **kwargs):
        x = self.extract_feat(img, img_metas, "train")

        losses, preds = self.bbox_head.forward_semi_train(
                                    x, img_metas, **kwargs)
        return losses

    def forward_test(self, img, img_metas, **kwargs):
        if isinstance(img_metas[0]['ori_shape'], tuple):
            return self.simple_test(img, img_metas, **kwargs)
        else:
            return self.aug_test(img, img_metas, **kwargs)

    def simple_test(self, img, img_metas, **kwargs):
        N, V, C, H, W = img.shape
        if V == 2:
            right_img = img[:,1:]
            img = img[:,:1]
        x = self.extract_feat(img, img_metas, "test")
        preds = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(preds, img_metas, img=img)
        preds["features"] = x
        if V == 2:
            right_x = self.extract_feat(right_img, img_metas, "test")
            right_preds = self.bbox_head(right_x)
            for key, item in right_preds.items():
                preds["right_" + key] = item
            preds["right_features"] = right_x
            kwargs["right_img"] = right_img

        if self.nocs_head is not None:
            nocs = self.nocs_head(x)
            preds.update(nocs)
            preds["normalize_nocs"] = self.nocs_head.normalize_nocs

        if self.pred_nocs and self.nocs_head.vis_nocs is True:
            # img_idx = osp.basename(img_metas[0]['filename']).replace(".png", "")
            intrinsic = torch.tensor(img_metas[0]['cam2img'][0])
            extrinsic = torch.tensor(img_metas[0]['lidar2cam'][0])

            # bbox = bbox_list[]
            self.nocs_head.vis_nocs_func(
                bbox_list[0][0], nocs, img[0], img_metas[0], intrinsic, extrinsic,
                scores=bbox_list[0][1])

        if self.two_stage_head is not None and self.two_stage_infer:

        # if self.refined_by_nocs_head is not None:

            # intrinsic = torch.tensor(img_metas[0]['cam2img'][0])
            # extrinsic = torch.tensor(img_metas[0]['lidar2cam'][0])

            bbox_list = self.two_stage_head.simple_test(
                                    bbox_list, preds, img, img_metas, **kwargs)





        bbox_results = [
            bbox3d2result(det_bboxes, det_scores, det_labels)
            for det_bboxes, det_scores, det_labels in bbox_list
        ]

            # self.nocs_head
        return bbox_results


    def aug_test(self, img, img_metas, rescale=False):
        # only assume batch size = 1
        feats = self.extract_feat(img, img_metas, 'test')

        # only support aug_test for one sample
        outs_list = self.bbox_head(feats[-1])
        for key, item in outs_list.items():
            if item is None:
                continue
            new_item = []
            for i in range(img.shape[1]):
                item_idx = item[i]
                if img_metas[0]['flip'][i] == True:
                    item_idx = torch.flip(item_idx, dims=[2])
                    if key == "offset":
                        item_idx[0] *= -1
                else:
                    continue
                new_item.append(item_idx)
            item = torch.stack(new_item, dim=0).mean(0, keepdim=True)
            outs_list[key] = item

        img_metas[0]['lidar2cam'] = img_metas[0]['lidar2cam'][0]
        bbox_list = self.bbox_head.get_bboxes(outs_list, img_metas)

        bbox_results = [
            bbox3d2result(det_bboxes, det_scores, det_labels)
            for det_bboxes, det_scores, det_labels in bbox_list
        ]
        return bbox_results



    def show_results(self, *args, **kwargs):
        pass


    # note that mmdet contains the parse loss module, where the key do not contain
    # will not be treated as loss for backgrou
    def semi_train_step(self, data, optimizer):
        """The iteration step during semi-supervised training
            data (dict): The output of dataloader (without annotation)
            optimizer (:obj:'torch.optim.Optimizer' | dict): The optimizer of
            runner is passed to ``train.step()''. This argument is unused and
            reserved.
        """
        import pdb; pdb.set_trace()




    def train_step(self, data, optimizer, semi_data=None, semi=False):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a \
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                logger.
                - ``num_samples`` indicates the batch size (when the model is \
                DDP, it means the batch size on each GPU), which is used for \
                averaging the logs.
        """
        if semi is False:
            losses = self(**data)
            loss, log_vars = self._parse_losses(losses)

            outputs = dict(
                loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

            return outputs
        else:
            losses = self(**data)

            semi_losses = self.forward_semi_train(**semi_data)
            losses.update(semi_losses)

            # losses = self.forward_semi_train(**data)
            assert len(losses) > 0
            loss, log_vars = self._parse_losses(losses)

            outputs = dict(
                loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

            return outputs
