import torch
from mmdet.models import DETECTORS, build_backbone, build_head, build_neck
from mmdet.models.detectors import BaseDetector

from mmdet3d.core import bbox3d2result
from mmdet3d.core.utils.mask import mask_background_region


@DETECTORS.register_module()
class ImVoxelNetInteract(BaseDetector):
    def __init__(self,
                 backbone,
                 neck,
                #  neck_before_lift,
                 neck_3d,
                 bbox_head,
                 n_voxels,
                 voxel_size,
                 neck_before_lift=dict(type="IdentityNeck"),
                 neck_bev=None,
                 head_2d=None,
                 head_2d_aux=None,
                 train_cfg=None,
                 test_cfg=None,
                 mask_background=False,
                 multiple_volume_prob = [],
                 multiple_volume_prob_aux = [],
                 pretrained=None):
        super().__init__()
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.neck_3d = build_neck(neck_3d)
        self.neck_before_lift = build_neck(neck_before_lift)

        if neck_bev is not None:
            self.neck_bev = build_neck(neck_bev)
        else: 
            self.neck_bev = neck_bev
        # self.neck_before
        # else:
            # self.neck_bev = None
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.bbox_head.voxel_size = voxel_size
        self.head_2d = build_head(head_2d) if head_2d is not None else None
        self.head_2d_aux = build_head(head_2d_aux) if head_2d_aux is not None else None 
        self.n_voxels = n_voxels
        self.voxel_size = voxel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)
        self.mask_background = mask_background
        self.multiple_volume_prob = multiple_volume_prob
        self.multiple_volume_prob_aux = multiple_volume_prob_aux

    def init_weights(self, pretrained=None):
        super().init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.neck.init_weights()
        self.neck_3d.init_weights()
        self.bbox_head.init_weights()
        if self.head_2d is not None:
            self.head_2d.init_weights()

    def extract_feat(self, img, img_metas, mode, gt_bboxes=None):
        batch_size = img.shape[0]
        img = img.reshape([-1] + list(img.shape)[2:])
        x = self.backbone(img)


        x = self.neck(x)[0]

        if self.mask_background:
            x = mask_background_region(x, img_metas, gt_bboxes)

        xs = self.neck_before_lift([x])
        xs = [x.reshape([batch_size, -1] + list(x.shape[1:])) for x in xs]
        
        x_3d, valids = self.neck_3d(xs, img_metas, img.shape)

        if self.neck_bev is not None:
            # xs = [xs[i].squeeze(1) for i in range(len(xs))]
            # x_bev = self.neck_bev(x_3d, xs, img_meta)
            x_bev = self.neck_bev(x_3d)
        else:
            x_bev = x_3d
        if self.head_2d_aux is not None:
            return x_bev, None, depth_prob, depth_prob_aux   # (z, y)
        
        
        return x_bev, valids, None

    def forward_train(self, img, img_metas, gt_bboxes_3d, gt_labels_3d, **kwargs):

        if self.mask_background:
            x, valids, features_2d = self.extract_feat(img, img_metas, 'train', gt_bboxes = kwargs["gt_bboxes"])
        else:
            if self.head_2d_aux is not None:
                x, valids, features_2d, features_2d_aux = self.extract_feat(img, img_metas, 'train')
            else: 
                x, valids, features_2d = self.extract_feat(img, img_metas, 'train')


        losses = self.bbox_head.forward_train(x, valids.float(), img_metas, gt_bboxes_3d, gt_labels_3d)

        if self.head_2d is not None:
            losses.update(self.head_2d.loss(features_2d, img_metas, kwargs))
        if self.head_2d_aux is not None:
            losses.update(self.head_2d_aux.loss(features_2d_aux, img_metas, kwargs))
        return losses

    def forward_test(self, img, img_metas, **kwargs):
        # not supporting aug_test for now
        return self.simple_test(img, img_metas, **kwargs)

    def simple_test(self, img, img_metas, **kwargs):
        if self.mask_background is True:
            x, valids, features_2d = self.extract_feat(img, img_metas, 'test',
                                             gt_bboxes = kwargs["gt_bboxes"])
        else:
            if self.head_2d_aux is not None:
                x, valids, features_2d, features_2d_aux = self.extract_feat(img, img_metas, 'test')
            else:
                x, valids, features_2d = self.extract_feat(img, img_metas, 'test')
                
        x = self.bbox_head(x)
        if isinstance(x, dict):
            bbox_list = self.bbox_head.get_bboxes(x, valids.float(), img_metas)
        else:
            # import pdb
            # pdb.set_trace()
            bbox_list = self.bbox_head.get_bboxes(*x, valids.float(), img_metas) #TODO
        bbox_results = [
            bbox3d2result(det_bboxes, det_scores, det_labels)
            for det_bboxes, det_scores, det_labels in bbox_list
        ]
        # if self.head_2d is not None:
        #     angles, layouts = self.head_2d.get_bboxes(*features_2d, img_metas)
        #     for i in range(len(img)):
        #         bbox_results[i]['angles'] = angles[i]
        #         bbox_results[i]['layout'] = layouts[i]
        return bbox_results

    def aug_test(self, imgs, img_metas):
        pass

    def show_results(self, *args, **kwargs):
        pass

    @staticmethod
    def _compute_projection(img_meta, stride, angles):
        projection = []
        intrinsic = torch.tensor(img_meta['lidar2img']['intrinsic'][:3, :3])
        ratio = img_meta['ori_shape'][0] / (img_meta['img_shape'][0] / stride)
        intrinsic[:2] /= ratio
        # use predicted pitch and roll for SUNRGBDTotal test
        if angles is not None:
            extrinsics = []
            for angle in angles:
                extrinsics.append(get_extrinsics(angle).to(intrinsic.device))
        else:
            extrinsics = map(torch.tensor, img_meta['lidar2img']['extrinsic'])
        for extrinsic in extrinsics:
            projection.append(intrinsic @ extrinsic[:3])
        return torch.stack(projection)


@torch.no_grad()
def get_points(n_voxels, voxel_size, origin):
    points = torch.stack(torch.meshgrid([
        torch.arange(n_voxels[0]), 
        torch.arange(n_voxels[1]), 
        torch.arange(n_voxels[2]) 
    ]))
    new_origin = origin - n_voxels / 2. * voxel_size
    points = points * voxel_size.view(3, 1, 1, 1) + new_origin.view(3, 1, 1, 1)
    return points



def get_valid(features, points, projection):
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    points_2d_3 = torch.bmm(projection, points)
    x = (points_2d_3[:, 0] / (points_2d_3[:, 2] + 1e-10)) # 1. add round
    y = (points_2d_3[:, 1] / (points_2d_3[:, 2] + 1e-10))
    z = points_2d_3[:, 2]
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0) # 2. add valid
    valid = valid.view(n_images, 1, n_x_voxels, n_y_voxels, n_z_voxels)
    return  valid

# modify from https://github.com/magicleap/Atlas/blob/master/atlas/model.py
def backproject(features, points, projection):
    n_images, n_channels, height, width = features.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    points_2d_3 = torch.bmm(projection, points)
    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).round().long() # 1. add round
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).round().long()
    z = points_2d_3[:, 2]
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0) # 2. add valid
    volume = torch.zeros((n_images, n_channels, points.shape[-1]), device=features.device)
    x = (x.float() / features.shape[-1]) * 2 - 1
    y = (y.float() / features.shape[-2]) * 2 - 1
    grid = torch.stack([x, y], dim=-1)
    grid = grid.expand(n_images, -1, -1)
    volume = torch.nn.functional.grid_sample(features, grid.unsqueeze(1), align_corners=True, padding_mode="zeros").squeeze(2)
    #
    # for i in range(n_images):
        # volume[i, :, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]
    volume = volume.view(n_images, n_channels, n_x_voxels, n_y_voxels, n_z_voxels)
    valid = valid.view(n_images, 1, n_x_voxels, n_y_voxels, n_z_voxels)
    return volume, valid


# modify from https://github.com/magicleap/Atlas/blob/master/atlas/model.py
def backproject_3d(depth_prob, head_2d, points, projection, mode="z"):
    n_images, n_channels, height, width = depth_prob.shape
    n_x_voxels, n_y_voxels, n_z_voxels = points.shape[-3:]
    points = points.view(1, 3, -1).expand(n_images, 3, -1)
    points = torch.cat((points, torch.ones_like(points[:, :1])), dim=1)
    points_2d_3 = torch.bmm(projection, points)
    x = (points_2d_3[:, 0] / points_2d_3[:, 2]).float() # 1. add round
    y = (points_2d_3[:, 1] / points_2d_3[:, 2]).float()
    z = points_2d_3[:, 2]
    valid = (x >= 0) & (y >= 0) & (x < width) & (y < height) & (z > 0) # 2. add valid

    volume_prob = torch.zeros((n_images, n_channels, points.shape[-1]), device=depth_prob.device)


    x = (x.float() / depth_prob.shape[-1]) * 2 - 1
    y = (y.float() / depth_prob.shape[-2]) * 2 - 1
    if mode == "z":
        indices = head_2d.bin_depths(z, target=False)
    elif mode == "y":
        indices = head_2d.bin_depths(y, target=False)

    # import pdb; pdb.set_trace()
    indices = (indices / head_2d.depth_interval_size) * 2 -1
    grid = torch.stack([indices, x, y], dim=-1)
    grid = grid.expand(n_images, -1, -1)
    # import pdb; pdb.set_trace()
    volume_prob = torch.nn.functional.grid_sample(depth_prob[:,None], grid.unsqueeze(1)[:,None], align_corners=True, padding_mode="zeros").squeeze(2)
    #
    # for i in range(n_images):
        # volume[i, :, valid[i]] = features[i, :, y[i, valid[i]], x[i, valid[i]]]
    volume_prob = volume_prob.view(n_images, 1, n_x_voxels, n_y_voxels, n_z_voxels)
    valid = valid.view(n_images, 1, n_x_voxels, n_y_voxels, n_z_voxels)
    return volume_prob, valid



# for SUNRGBDTotal test
def get_extrinsics(angles):
    yaw = angles.new_zeros(())
    pitch, roll = angles
    r = angles.new_zeros((3, 3))
    r[0, 0] = torch.cos(yaw) * torch.cos(pitch)
    r[0, 1] = torch.sin(yaw) * torch.sin(roll) - torch.cos(yaw) * torch.cos(roll) * torch.sin(pitch)
    r[0, 2] = torch.cos(roll) * torch.sin(yaw) + torch.cos(yaw) * torch.sin(pitch) * torch.sin(roll)
    r[1, 0] = torch.sin(pitch)
    r[1, 1] = torch.cos(pitch) * torch.cos(roll)
    r[1, 2] = -torch.cos(pitch) * torch.sin(roll)
    r[2, 0] = -torch.cos(pitch) * torch.sin(yaw)
    r[2, 1] = torch.cos(yaw) * torch.sin(roll) + torch.cos(roll) * torch.sin(yaw) * torch.sin(pitch)
    r[2, 2] = torch.cos(yaw) * torch.cos(roll) - torch.sin(yaw) * torch.sin(pitch) * torch.sin(roll)

    # follow Total3DUnderstanding
    t = angles.new_tensor([[0., 0., 1.], [0., -1., 0.], [-1., 0., 0.]])
    r = t @ r.T
    # follow DepthInstance3DBoxes
    r = r[:, [2, 0, 1]]
    r[2] *= -1
    extrinsic = angles.new_zeros((4, 4))
    extrinsic[:3, :3] = r
    extrinsic[3, 3] = 1.
    return extrinsic

