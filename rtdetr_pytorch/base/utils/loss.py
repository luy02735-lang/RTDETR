#1

import torch
import torch.nn as nn
import torch.nn.functional as F
import dill as pickle

from .ops import HungarianMatcher


class DETRLoss(nn.Module):
    """
    DETR (DEtection TRansformer) Loss class. This class calculates and returns the different loss components for the
    DETR object detection model. It computes classification loss, bounding box loss, GIoU loss, and optionally auxiliary
    losses.

    Attributes:
        nc (int): The number of classes.
        loss_gain (dict): Coefficients for different loss components.
        aux_loss (bool): Whether to compute auxiliary losses.
        use_fl (bool): Use FocalLoss or not.
        use_vfl (bool): Use VarifocalLoss or not.
        use_uni_match (bool): Whether to use a fixed layer to assign labels for the auxiliary branch.
        uni_match_ind (int): The fixed indices of a layer to use if `use_uni_match` is True.
        matcher (HungarianMatcher): Object to compute matching cost and indices.
        fl (FocalLoss or None): Focal Loss object if `use_fl` is True, otherwise None.
        vfl (VarifocalLoss or None): Varifocal Loss object if `use_vfl` is True, otherwise None.
        device (torch.device): Device on which tensors are stored.
    """

    def __init__(self,
                 nc=80,
                 loss_gain=None,
                 aux_loss=True,
                 use_fl=True,
                 use_vfl=False,
                 use_sl=False, # SlideLoss
                 use_emasl=False, # EMASlideLoss
                 use_svfl=False, # SlideVarifocalLoss
                 use_emasvfl=False, # EMASlideVarifocalLoss
                 use_mal=False, # CVPR2025-DEIM MAL
                 use_uni_match=False,
                 uni_match_ind=0):
        """
        DETR loss function.

        Args:
            nc (int): The number of classes.
            loss_gain (dict): The coefficient of loss.
            aux_loss (bool): If 'aux_loss = True', loss at each decoder layer are to be used.
            use_vfl (bool): Use VarifocalLoss or not.
            use_uni_match (bool): Whether to use a fixed layer to assign labels for auxiliary branch.
            uni_match_ind (int): The fixed indices of a layer.
        """
        super().__init__()

        if loss_gain is None:
            loss_gain = {'class': 1, 'bbox': 5, 'giou': 2, 'no_object': 0.1, 'mask': 1, 'dice': 1}
        self.nc = nc
        self.matcher = HungarianMatcher(cost_gain={'class': 2, 'bbox': 5, 'giou': 2})
        self.loss_gain = loss_gain
        self.aux_loss = aux_loss
        self.fl = FocalLoss() if use_fl else None
        self.vfl = VarifocalLoss() if use_vfl else None
        self.sl = SlideLoss(nn.BCEWithLogitsLoss(reduction='none')) if use_sl else None
        self.emasl = EMASlideLoss(nn.BCEWithLogitsLoss(reduction='none')) if use_emasl else None
        self.svfl = SlideVarifocalLoss() if use_svfl else None
        self.emasvfl = EMASlideVarifocalLoss() if use_emasvfl else None
        self.mal = MALoss() if use_mal else None

        self.use_uni_match = use_uni_match
        self.uni_match_ind = uni_match_ind
        self.device = None
        
        # for nwd loss
        self.nwd_loss = False
        self.iou_ratio = 0.5
        
        # for wise-iou loss
        self.use_wiseiou = False
        if self.use_wiseiou:
            self.wiou_loss = WiseIouLoss(ltype='WIoU', monotonous=False, inner_iou=False, focaler_iou=False)

    def _get_loss_class(self, pred_scores, targets, gt_scores, num_gts, postfix=''):
        """Computes the classification loss based on predictions, target values, and ground truth scores."""
        # Logits: [b, query, num_classes], gt_class: list[[n, 1]]
        name_class = f'loss_class{postfix}'
        bs, nq = pred_scores.shape[:2]
        # one_hot = F.one_hot(targets, self.nc + 1)[..., :-1]  # (bs, num_queries, num_classes)
        one_hot = torch.zeros((bs, nq, self.nc + 1), dtype=torch.int64, device=targets.device)
        one_hot.scatter_(2, targets.unsqueeze(-1), 1)
        one_hot = one_hot[..., :-1]
        gt_scores = gt_scores.view(bs, nq, 1) * one_hot

        if self.sl or self.emasl:
            if num_gts > 0:
                auto_iou = (gt_scores[gt_scores > 0]).mean()
            else:
                auto_iou = -1
            if self.sl:
                loss_cls = self.sl(pred_scores, gt_scores, auto_iou).mean(1).sum()
            else:
                loss_cls = self.emasl(pred_scores, gt_scores, auto_iou).mean(1).sum()
        elif self.svfl or self.emasvfl:
            if num_gts > 0:
                auto_iou = (gt_scores[gt_scores > 0]).mean()
            else:
                auto_iou = -1
            if num_gts:
                if self.svfl:
                    loss_cls = self.svfl(pred_scores, gt_scores, one_hot, auto_iou)
                else:
                    loss_cls = self.emasvfl(pred_scores, gt_scores, one_hot, auto_iou)
            else:
                loss_cls = self.fl(pred_scores, one_hot.float())
            loss_cls /= max(num_gts, 1) / nq
        elif self.fl:
            if num_gts:
                if self.vfl:
                    loss_cls = self.vfl(pred_scores, gt_scores, one_hot)
                elif self.mal:
                    loss_cls = self.mal(pred_scores, gt_scores, one_hot)
            else:
                loss_cls = self.fl(pred_scores, one_hot.float())
            loss_cls /= max(num_gts, 1) / nq
        else:
            loss_cls = nn.BCEWithLogitsLoss(reduction='none')(pred_scores, gt_scores).mean(1).sum()  # YOLO CLS loss

        return {name_class: loss_cls.squeeze() * self.loss_gain['class']}
            nwd = wasserstein_loss(pred_bboxes, gt_bboxes)
            loss[name_giou] = self.iou_ratio * (loss[name_giou].sum() / len(gt_bboxes)) + (1.0 - self.iou_ratio) * ((1.0 - nwd).sum() / len(gt_bboxes))
        else:
            loss[name_giou] = loss[name_giou].sum() / len(gt_bboxes)
        loss[name_giou] = self.loss_gain['giou'] * loss[name_giou]
        return {k: v.squeeze() for k, v in loss.items()}

    # This function is for future RT-DETR Segment models
    # def _get_loss_mask(self, masks, gt_mask, match_indices, postfix=''):
    #     # masks: [b, query, h, w], gt_mask: list[[n, H, W]]
    #     name_mask = f'loss_mask{postfix}'
    #     name_dice = f'loss_dice{postfix}'
    #
    #     loss = {}
    #     if sum(len(a) for a in gt_mask) == 0:
    #         loss[name_mask] = torch.tensor(0., device=self.device)
    #         loss[name_dice] = torch.tensor(0., device=self.device)
    #         return loss
    #
    #     num_gts = len(gt_mask)
    #     src_masks, target_masks = self._get_assigned_bboxes(masks, gt_mask, match_indices)
    #     src_masks = F.interpolate(src_masks.unsqueeze(0), size=target_masks.shape[-2:], mode='bilinear')[0]
    #     # TODO: torch does not have `sigmoid_focal_loss`, but it's not urgent since we don't use mask branch for now.
    #     loss[name_mask] = self.loss_gain['mask'] * F.sigmoid_focal_loss(src_masks, target_masks,
    #                                                                     torch.tensor([num_gts], dtype=torch.float32))
    #     loss[name_dice] = self.loss_gain['dice'] * self._dice_loss(src_masks, target_masks, num_gts)
    #     return loss

    # This function is for future RT-DETR Segment models
    # @staticmethod
    # def _dice_loss(inputs, targets, num_gts):
    #     inputs = F.sigmoid(inputs).flatten(1)
    #     targets = targets.flatten(1)
    #     numerator = 2 * (inputs * targets).sum(1)
    #     denominator = inputs.sum(-1) + targets.sum(-1)
    #     loss = 1 - (numerator + 1) / (denominator + 1)
    #     return loss.sum() / num_gts


    def forward(self, pred_bboxes, pred_scores, batch, postfix='', **kwargs):
        """
        Args:
            pred_bboxes (torch.Tensor): [l, b, query, 4]
            pred_scores (torch.Tensor): [l, b, query, num_classes]
            batch (dict): A dict includes:
                gt_cls (torch.Tensor) with shape [num_gts, ],
                gt_bboxes (torch.Tensor): [num_gts, 4],
                gt_groups (List(int)): a list of batch size length includes the number of gts of each image.
            postfix (str): postfix of loss name.
        """
        self.device = pred_bboxes.device
        match_indices = kwargs.get('match_indices', None)
        gt_cls, gt_bboxes, gt_groups = batch['cls'], batch['bboxes'], batch['gt_groups']

        total_loss = self._get_loss(pred_bboxes[-1],
                                    pred_scores[-1],
                                    gt_bboxes,
                                    gt_cls,
                                    gt_groups,
                                    postfix=postfix,
                                    match_indices=match_indices)

        if self.aux_loss:
            total_loss.update(
                self._get_loss_aux(pred_bboxes[:-1], pred_scores[:-1], gt_bboxes, gt_cls, gt_groups, match_indices,
                                   postfix))

        return total_loss


class RTDETRDetectionLoss(DETRLoss):
    """
    Real-Time DeepTracker (RT-DETR) Detection Loss class that extends the DETRLoss.

    This class computes the detection loss for the RT-DETR model, which includes the standard detection loss as well as
    an additional denoising training loss when provided with denoising metadata.
    """

    def forward(self, preds, batch, dn_bboxes=None, dn_scores=None, dn_meta=None):
        """
        Forward pass to compute the detection loss.

        Args:
            preds (tuple): Predicted bounding boxes and scores.
            batch (dict): Batch data containing ground truth information.
            dn_bboxes (torch.Tensor, optional): Denoising bounding boxes. Default is None.
            dn_scores (torch.Tensor, optional): Denoising scores. Default is None.
            dn_meta (dict, optional): Metadata for denoising. Default is None.

        Returns:
            (dict): Dictionary containing the total loss and, if applicable, the denoising loss.
        """
        pred_bboxes, pred_scores = preds
        total_loss = super().forward(pred_bboxes, pred_scores, batch)

        # Check for denoising metadata to compute denoising training loss
        if dn_meta is not None:
            dn_pos_idx, dn_num_group = dn_meta['dn_pos_idx'], dn_meta['dn_num_group']
            assert len(batch['gt_groups']) == len(dn_pos_idx)

            # Get the match indices for denoising
            match_indices = self.get_dn_match_indices(dn_pos_idx, dn_num_group, batch['gt_groups'])

            # Compute the denoising training loss
            dn_loss = super().forward(dn_bboxes, dn_scores, batch, postfix='_dn', match_indices=match_indices)
            total_loss.update(dn_loss)
        else:
            # If no denoising metadata is provided, set denoising loss to zero
            total_loss.update({f'{k}_dn': torch.tensor(0., device=self.device) for k in total_loss.keys()})

        return total_loss

    @staticmethod
    def get_dn_match_indices(dn_pos_idx, dn_num_group, gt_groups):
        """
        Get the match indices for denoising.

        Args:
            dn_pos_idx (List[torch.Tensor]): List of tensors containing positive indices for denoising.
            dn_num_group (int): Number of denoising groups.
            gt_groups (List[int]): List of integers representing the number of ground truths for each image.

        Returns:
            (List[tuple]): List of tuples containing matched indices for denoising.
        """
        dn_match_indices = []
        idx_groups = torch.as_tensor([0, *gt_groups[:-1]]).cumsum_(0)
        for i, num_gt in enumerate(gt_groups):
            if num_gt > 0:
                gt_idx = torch.arange(end=num_gt, dtype=torch.long) + idx_groups[i]
                gt_idx = gt_idx.repeat(dn_num_group)
                assert len(dn_pos_idx[i]) == len(gt_idx), 'Expected the same length, '
                f'but got {len(dn_pos_idx[i])} and {len(gt_idx)} respectively.'
                dn_match_indices.append((dn_pos_idx[i], gt_idx))
            else:
                dn_match_indices.append((torch.zeros([0], dtype=torch.long), torch.zeros([0], dtype=torch.long)))
        return dn_match_indices
