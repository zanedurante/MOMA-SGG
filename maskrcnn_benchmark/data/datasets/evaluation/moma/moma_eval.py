import logging
import tempfile
import os
import torch
from collections import OrderedDict
from tqdm import tqdm

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou

from maskrcnn_benchmark.data.datasets.evaluation.sg.sg_tsv_eval import do_sg_evaluation, do_att_evaluation

from scene_graph_benchmark.scene_parser import SceneParserOutputs

#import wandb
import pdb
from collections import defaultdict

def do_moma_evaluation(
    dataset,
    predictions,
    box_only,
    output_folder,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
    eval_attributes,
):
    if isinstance(predictions[0], SceneParserOutputs):
        if eval_attributes:
            return do_moma_attribute_evaluation(
                dataset,
                predictions,
                box_only,
                output_folder,
                iou_types,
                expected_results,
                expected_results_sigma_tol,
            )
        else:
            return do_moma_graph_evaluation(
                dataset,
                predictions,
                box_only,
                output_folder,
                iou_types,
                expected_results,
                expected_results_sigma_tol,
            )

    else:
        return do_moma_object_evaluation(
            dataset,
            predictions,
            box_only,
            output_folder,
            iou_types,
            expected_results,
            expected_results_sigma_tol,
        )


def do_moma_attribute_evaluation(
    dataset,
    predictions,
    box_only,
    output_folder,
    iou_types,
    expected_results,
    expected_results_sigma_tol):

    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    return do_att_evaluation(dataset, predictions, output_folder, logger)



def do_moma_graph_evaluation(
    dataset,
    predictions,
    box_only,
    output_folder,
    iou_types,
    expected_results,
    expected_results_sigma_tol,

):
  #  obj_preds = {k: v.predictions for k,v in predictions.items()}
  #  do_moma_object_evaluation(
  #      dataset,
  #      obj_preds,
  #      box_only,
  #      output_folder,
  #      iou_types,
  #      expected_results,
  #      expected_results_sigma_tol
  #  )
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    return do_sg_evaluation(dataset, predictions, output_folder, logger)
    

def do_moma_object_evaluation(
    dataset,
    predictions,
    box_only,
    output_folder,
    iou_types,
    expected_results,
    expected_results_sigma_tol,
):
    print("OUTPUT TO:", output_folder)
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    if box_only:
        logger.info("Evaluating bbox proposals")
        res = COCOResults("box_proposal")
        for limit in [100, 1000]:
            stats = evaluate_box_proposals(
                predictions, dataset, limit=limit
            )
            key = "AR{}@{:d}".format(suffix, limit)
            res.results["box_proposal"][key] = stats["ar"].item()
        logger.info(res)
        if output_folder:
            torch.save(res, os.path.join(output_folder, "box_proposals.pth"))
        return
    logger.info("Preparing results for COCO format")
    coco_results = {}
    if "bbox" in iou_types:
        logger.info("Preparing bbox results")
        coco_results["bbox"] = prepare_for_coco_detection(predictions, dataset)
    if coco_results["bbox"] == []: # End early, since there are no predictions
        logger.info("WARNING: Skipping evaluation, no bounding box predictions!")
        return 0, 0
    
    results = COCOResults(*iou_types)
    logger.info("Evaluating predictions")
    for iou_type in iou_types:
        with tempfile.NamedTemporaryFile() as f:
            file_path = f.name
            if output_folder:
                file_path = os.path.join(output_folder, iou_type + ".json")

            anns = []
            if dataset.debug:
                dataset_dict = dataset.dataset_dict[len(dataset):]
            else:
                assert len(dataset.dataset_dict) == len(predictions)
                dataset_dict = dataset.dataset_dict
            for image_id, entry in enumerate(dataset_dict):
                labels = entry["labels"]
                boxes = entry["bboxes"]
                for cls, box in zip(labels, boxes):
                    anns.append({
                        'area': (box[3] - box[1] + 1) * (box[2] - box[0] + 1),
                        'bbox': [box[0], box[1], box[2] - box[0] + 1, box[3] - box[1] + 1], # xywh
                        'category_id': cls,
                        'id': len(anns),
                        'image_id': image_id,
                        'iscrowd': 0,
                    })
            fauxcoco = COCO()
            fauxcoco.dataset = {
                    'info': {'description': 'use coco script for moma detection evaluation'},
                    'images': [{'id': i} for i in range(len(dataset_dict))],
                    'categories': [
                        {'supercategory': 'person', 'id': i, 'name': name} 
                        for i, name in enumerate(dataset.classes) if name != '__background__'
                        ],
                    'annotations': anns,
                }
            fauxcoco.createIndex()
            
            res = evaluate_predictions_on_moma(
                fauxcoco, coco_results[iou_type], file_path, iou_type
            )
            results.update(res)
    logger.info(results)
    if output_folder:
        torch.save(results, os.path.join(output_folder, "moma_results.pth"))

    return results, coco_results


def prepare_for_coco_detection(predictions, dataset):
    # assert isinstance(dataset, COCODataset)
    coco_results = []
    for image_id, prediction in predictions.items():
        #original_id = dataset.id_to_img_map[image_id]
        if len(prediction) == 0:
            continue

        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))
        prediction = prediction.convert("xywh")

        boxes = prediction.bbox.tolist()
        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()

        coco_results.extend(
            [
                {
                    "image_id": image_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results

# inspired from Detectron
def evaluate_box_proposals(
    predictions, dataset, thresholds=None, limit=None
):
    """Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official COCO API recall evaluation code. However,
    it produces slightly different results.
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values

    gt_overlaps = []
    num_pos = 0

    for image_id, prediction in predictions.items():
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        prediction = prediction.resize((image_width, image_height))

        # sort predictions in descending order
        # TODO maybe remove this and make it explicit in the documentation
        inds = prediction.get_field("pred_scores").sort(descending=True)[1]
        prediction = prediction[inds]

        gt_boxes = dataset.dataset_dict[image_id]["bboxes"]
        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
        gt_boxes = BoxList(gt_boxes, (image_width, image_height), mode="xywh").convert(
            "xyxy"
        )

        if len(gt_boxes) == 0:
            continue

        num_pos += len(gt_boxes)

        if len(gt_boxes) == 0:
            continue

        if len(prediction) == 0:
            continue

        if limit is not None and len(prediction) > limit:
            prediction = prediction[:limit]

        overlaps = boxlist_iou(prediction, gt_boxes)

        _gt_overlaps = torch.zeros(len(gt_boxes))
        for j in range(min(len(prediction), len(gt_boxes))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        # append recorded iou coverage level
        gt_overlaps.append(_gt_overlaps)
    gt_overlaps = torch.cat(gt_overlaps, dim=0)
    gt_overlaps, _ = torch.sort(gt_overlaps)

    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    return {
        "ar": ar,
        "recalls": recalls,
        "thresholds": thresholds,
        "gt_overlaps": gt_overlaps,
        "num_pos": num_pos,
    }


def evaluate_predictions_on_moma(
    coco_gt, coco_results, json_result_file, iou_type="bbox"
):
    import json

    with open(json_result_file, "w") as f:
        json.dump(coco_results, f)

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    
    if coco_results == []:
        coco_results = [{"image_id":0}] # COCO checks coco_results[0]
    coco_dt = coco_gt.loadRes(coco_results)

    # coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval


class COCOResults(object):
    METRICS = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "box_proposal": [
            "AR@100",
            "ARs@100",
            "ARm@100",
            "ARl@100",
            "AR@1000",
            "ARs@1000",
            "ARm@1000",
            "ARl@1000",
        ],
    }

    def __init__(self, *iou_types):
        allowed_types = ("box_proposal", "bbox")
        assert all(iou_type in allowed_types for iou_type in iou_types)
        results = OrderedDict()
        for iou_type in iou_types:
            results[iou_type] = OrderedDict(
                [(metric, -1) for metric in COCOResults.METRICS[iou_type]]
            )
        self.results = results

    def update(self, coco_eval):
        if coco_eval is None:
            return
        from pycocotools.cocoeval import COCOeval

        assert isinstance(coco_eval, COCOeval)
        s = coco_eval.stats
        iou_type = coco_eval.params.iouType
        res = self.results[iou_type]
        metrics = COCOResults.METRICS[iou_type]
        for idx, metric in enumerate(metrics):
            res[metric] = s[idx]

    def __repr__(self):
        # TODO make it pretty
        return repr(self.results) 