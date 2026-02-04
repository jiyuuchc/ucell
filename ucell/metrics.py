import numpy as np

from skimage.measure import regionprops
from .utils import clean_up_mask

def box_intersection(boxes_a, boxes_b):
    """Compute pairwise intersection areas between boxes.

    Args:
      boxes_a: [..., N, 2d]
      boxes_b: [..., M, 2d]

    Returns:
      its: [..., N, M] representing pairwise intersections.
    """
    minimum = np.minimum
    maximum = np.maximum
    boxes_a = np.array(boxes_a)
    boxes_b = np.array(boxes_b)

    ndim = boxes_a.shape[-1] // 2
    assert ndim * 2 == boxes_a.shape[-1]
    assert ndim * 2 == boxes_b.shape[-1]

    min_vals_1 = boxes_a[..., None, :ndim]  # [..., N, 1, d]
    max_vals_1 = boxes_a[..., None, ndim:]
    min_vals_2 = boxes_b[..., None, :, :ndim]  # [..., 1, M, d]
    max_vals_2 = boxes_b[..., None, :, ndim:]

    min_max = minimum(max_vals_1, max_vals_2)  # [..., N, M, d]
    max_min = maximum(min_vals_1, min_vals_2)

    intersects = maximum(0, min_max - max_min)  # [..., N, M, d]

    return intersects.prod(axis=-1)


def mask_intersection(rp_a, rp_b):
    def get_its(d0, d1):
        y1a, x1a, y2a, x2a = d0.bbox
        y1b, x1b, y2b, x2b = d1.bbox
        y1, x1 = max(y1a, y1b), max(x1a, x1b)
        y2, x2 = min(y2a, y2b), min(x2a, x2b)

        ovp = d0.image_filled[y1-y1a:y2-y1a, x1-x1a:x2-x1a] & d1.image_filled[y1-y1b:y2-y1b, x1-x1b:x2-x1b]
        return np.count_nonzero( ovp )

    to_boxes = lambda dets: np.array([rp.bbox for rp in dets]).reshape(-1, 4)

    boxes_f = to_boxes(rp_a)
    boxes_f1 = to_boxes(rp_b)

    its = box_intersection(boxes_f, boxes_f1)

    ids = np.where(its > 0)
    its[ids] = [get_its(rp_a[pid], rp_b[gid]) for pid, gid in zip(*ids)]

    return its


def mask_intersection_3d(rp_a, rp_b):
    def get_its(d0, d1):
        z1a, y1a, x1a, z2a, y2a, x2a = d0.bbox
        z1b, y1b, x1b, z2b, y2b, x2b = d1.bbox
        z1, y1, x1 = max(z1a, z1b), max(y1a, y1b), max(x1a, x1b)
        z2, y2, x2 = min(z2a, z2b), min(y2a, y2b), min(x2a, x2b)

        ovp = d0.image[z1-z1a:z2-z1a, y1-y1a:y2-y1a, x1-x1a:x2-x1a] & d1.image[z1-z1b:z2-z1b, y1-y1b:y2-y1b, x1-x1b:x2-x1b]
        return np.count_nonzero( ovp )

    to_boxes = lambda dets: np.array([rp.bbox for rp in dets]).reshape(-1, 6)

    boxes_f = to_boxes(rp_a)
    boxes_f1 = to_boxes(rp_b)

    its = box_intersection(boxes_f, boxes_f1)

    ids = np.where(its > 0)
    its[ids] = [get_its(rp_a[pid], rp_b[gid]) for pid, gid in zip(*ids)]

    return its

class LabelMetrics:
    """Compute various metrics based on labels"""
    def __init__(self):
        self.pred_areas = []
        self.gt_areas = []
        self.pred_scores = []
        self.gt_scores = []
        self.ious = []

    def _update(self, pred_its, pred_areas, gt_areas):
        n_pred, n_gt = pred_its.shape

        if n_gt > 0:
            pred_best = pred_its.max(axis=1)
            pred_best_matches = pred_its.argmax(axis=1)

            assert (pred_best <= pred_areas).all()
            assert (pred_best <= gt_areas[pred_best_matches]).all()

            pred_dice = pred_best * 2 / (pred_areas + gt_areas[pred_best_matches])
            pred_ious = pred_best / (pred_areas + gt_areas[pred_best_matches] - pred_best)

        else:
            pred_dice = np.zeros([n_pred])
            pred_ious = np.zeros([n_pred])

        if n_pred > 0:
            gt_best = pred_its.max(axis=0)
            gt_best_matches = pred_its.argmax(axis=0)

            assert (gt_best <= gt_areas).all()
            assert (gt_best <= pred_areas[gt_best_matches]).all()

            gt_dice = gt_best * 2 / (gt_areas + pred_areas[gt_best_matches])
        
        else:
            gt_dice = np.zeros([n_gt])

        return pred_dice, gt_dice, pred_ious


    def update(self, pred_mask, gt_mask):
        pred_rps = regionprops(pred_mask)
        gt_rps = regionprops(gt_mask)

        mask_its = mask_intersection(pred_rps, gt_rps)
        pred_areas = np.array([rp.area_filled for rp in pred_rps]).reshape(-1)
        gt_areas = np.array([rp.area_filled for rp in gt_rps]).reshape(-1)

        pred_scores, gt_scores, ious = self._update(mask_its, pred_areas, gt_areas)

        self.pred_areas.append(pred_areas)
        self.gt_areas.append(gt_areas)
        self.pred_scores.append(pred_scores)
        self.gt_scores.append(gt_scores)
        self.ious.append(ious)


    def _compute(self, gt_areas, pred_areas, gt_scores, pred_scores, ious,  iou_threshold):
        n_gts = len(gt_areas)
        n_preds = len(pred_areas)
        n_tps = np.count_nonzero(np.array(ious) >= iou_threshold)

        if n_preds == 0:
            pred_dice = 0
        else:
            pred_dice = (pred_areas / pred_areas.sum() * pred_scores).sum()

        if n_gts == 0:
            gt_dice = 0
        else:
            gt_dice = (gt_areas / gt_areas.sum() * gt_scores).sum()

        dice = (pred_dice + gt_dice) / 2

        return dict(
            n_preds = n_preds,
            n_gts = n_gts,
            n_tps = n_tps,
            accuracy = n_tps / n_preds if n_preds > 0 else float('nan'),
            recall = n_tps / n_gts if n_gts > 0 else float('nan'),
            f1 = (2 * n_tps) / (n_preds + n_gts) if n_tps > 0 else 0,
            instance_dice = dice,
            ap = n_tps /(n_gts + n_preds - n_tps) if n_gts + n_preds > 0 else float('nan'),
        )
    
    def compute(self, iou_threshold=.5, micros=False):
        if len(self.pred_areas) == 0:
            return None
        # micro stats
        if micros:
            micros = []
            for k in range(len(self.pred_areas)):
                micros.append(self._compute(
                    self.gt_areas[k],
                    self.pred_areas[k], 
                    self.gt_scores[k],
                    self.pred_scores[k],
                    self.ious[k],
                    iou_threshold,
                ))

        # macro stats
        pred_areas = np.concatenate(self.pred_areas)
        pred_scores = np.concatenate(self.pred_scores)
        gt_areas = np.concatenate(self.gt_areas)
        gt_scores = np.concatenate(self.gt_scores)
        ious = np.concatenate(self.ious)

        macros = self._compute(gt_areas, pred_areas, gt_scores, pred_scores, ious, iou_threshold)

        if micros:
            return macros, micros
        else:
            return macros

