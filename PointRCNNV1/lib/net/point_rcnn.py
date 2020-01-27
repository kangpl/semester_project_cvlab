import torch
import torch.nn as nn
from lib.net.rpn import RPN
from lib.net.rcnn_net import RCNNNet
from lib.net.pspnet import pspnet as PSPNet 
from lib.config import cfg


class PointRCNN(nn.Module):
    def __init__(self, num_classes, use_xyz=True, mode='TRAIN'):
        super().__init__()

        assert cfg.RPN.ENABLED or cfg.RCNN.ENABLED

        if cfg.PSP.ENABLED:
            self.psp = PSPNet(n_classes = 19)

        if cfg.RPN.ENABLED:
            self.rpn = RPN(use_xyz=use_xyz, mode=mode)

        if cfg.RCNN.ENABLED:
            rcnn_input_channels = 640  # channels of 128 rpn features + 512 img_features
            if cfg.RCNN.BACKBONE == 'pointnet':
                self.rcnn_net = RCNNNet(num_classes=num_classes, input_channels=rcnn_input_channels, use_xyz=use_xyz)
            elif cfg.RCNN.BACKBONE == 'pointsift':
                pass 
            else:
                raise NotImplementedError

    def forward(self, input_data):
        if cfg.PSP.ENABLED:
            if cfg.PSP.FIXED:
                self.psp.eval()
                with torch.no_grad():
                    psp_output = self.psp(input_data['img']) 
        # print("this is psp_output shape: ", psp_output.shape, psp_output.device, psp_output.dtype)
        pts_img = input_data['pts_img']
        x_ind = pts_img[:,:,0]
        y_ind = pts_img[:,:,1]
        # print('y_ind:', y_ind.shape, y_ind.device, y_ind.dtype, y_ind.min(), y_ind.max())
        # print('x_ind:', x_ind.shape, x_ind.device, x_ind.dtype, x_ind.min(), x_ind.max())
        img_features = torch.stack([psp_output[i, :, y_ind[i], x_ind[i]] for i in range(psp_output.shape[0])], dim=0)

        if cfg.RPN.ENABLED:
            rpn_input_info = {'pts_input': input_data['pts_input'], 'img_features': img_features} #B,C,N
            # rpn_input_info = input_data
            output = {}
            # rpn inference
            with torch.set_grad_enabled((not cfg.RPN.FIXED) and self.training):
                if cfg.RPN.FIXED:
                    self.rpn.eval()
                rpn_output = self.rpn(rpn_input_info)
                output.update(rpn_output)
            # print("rpn_output['rpn_cls']: ", rpn_output['rpn_cls'].shape, rpn_output['rpn_cls'].device, rpn_output['rpn_cls'].requires_grad)
            # print("rpn_output['rpn_reg']: ", rpn_output['rpn_reg'].shape, rpn_output['rpn_reg'].device, rpn_output['rpn_reg'].requires_grad)
            # print("rpn_output['backbone_xyz']: ", rpn_output['backbone_xyz'].shape, rpn_output['backbone_xyz'].device, rpn_output['backbone_xyz'].requires_grad)
            # print("rpn_output['backbone_features']: ", rpn_output['backbone_features'].shape, rpn_output['backbone_features'].device, rpn_output['backbone_features'].requires_grad)

            # rcnn inference
            if cfg.RCNN.ENABLED:
                with torch.no_grad():
                    rpn_cls, rpn_reg = rpn_output['rpn_cls'], rpn_output['rpn_reg']
                    backbone_xyz, backbone_features = rpn_output['backbone_xyz'], rpn_output['backbone_features']

                    # print("rpn_output['rpn_cls']: ", rpn_output['rpn_cls'].shape, rpn_output['rpn_cls'].device, rpn_output['rpn_cls'].requires_grad)
                    # print("rpn_output['rpn_reg']: ", rpn_output['rpn_reg'].shape, rpn_output['rpn_reg'].device, rpn_output['rpn_reg'].requires_grad)
                    # print("rpn_output['backbone_xyz']: ", rpn_output['backbone_xyz'].shape, rpn_output['backbone_xyz'].device, rpn_output['backbone_xyz'].requires_grad)
                    # print("rpn_output['backbone_features']: ", rpn_output['backbone_features'].shape, rpn_output['backbone_features'].device, rpn_output['backbone_features'].requires_grad)

                    rpn_scores_raw = rpn_cls[:, :, 0]
                    rpn_scores_norm = torch.sigmoid(rpn_scores_raw)
                    seg_mask = (rpn_scores_norm > cfg.RPN.SCORE_THRESH).float()
                    pts_depth = torch.norm(backbone_xyz, p=2, dim=2)

                    # proposal layer
                    rois, roi_scores_raw = self.rpn.proposal_layer(rpn_scores_raw, rpn_reg, backbone_xyz)  # (B, M, 7)

                    output['rois'] = rois
                    output['roi_scores_raw'] = roi_scores_raw
                    output['seg_result'] = seg_mask

                rcnn_input_info = {'rpn_xyz': backbone_xyz,
                                   'rpn_features': backbone_features.permute((0, 2, 1)), #(B N C)
                                   'seg_mask': seg_mask,
                                   'roi_boxes3d': rois,
                                   'pts_depth': pts_depth}
                if self.training:
                    rcnn_input_info['gt_boxes3d'] = input_data['gt_boxes3d']

                rcnn_output = self.rcnn_net(rcnn_input_info)
                output.update(rcnn_output)

        elif cfg.RCNN.ENABLED:
            output = self.rcnn_net(input_data)
        else:
            raise NotImplementedError

        return output
