# Copyright (c) 2021 Microsoft Corporation. Licensed under the MIT license. 
import os
import json
from collections import defaultdict
import base64

import numpy as np
import torch
from maskrcnn_benchmark.structures.tsv_file_ops import tsv_reader
from .evaluator import BasicSceneGraphEvaluator
from .box import bbox_overlaps
import pdb

import pickle


def do_att_evaluation(dataset, predictions, output_folder, logger, iou_thresh=0.5):
    prediction_file = os.path.join(output_folder, 'predictions_forcebox.tsv')

    gt_dicts = prepare_vrd_groundtruths(dataset)
    if hasattr(dataset, 'labelmap'):
        predict_dicts = prepare_vrd_predictions(prediction_file, dataset.labelmap)
    else:
        predict_dicts = prepare_vrd_predictions_no_labelmap(prediction_file, dataset)
    all_rec = []
    for classindex in range(dataset.num_att_classes):
        class_recs = {}
        image_ids = []
        confidence = []
        BB = []
        npos = 0
        for image_key, gt_boxlist in gt_dicts.items():
            sg_prediction = predict_dicts[str(image_key)]
            pred_bbox = sg_prediction["bboxes"].numpy()
            gt_bbox = gt_boxlist.bbox.numpy()
            if len(sg_prediction) == 0:
                sg_prediction = {'bboxes':torch.as_tensor([]), 'bbox_scores':torch.tensor([]), 'bbox_labels':torch.tensor([]),
                                 'relation_pairs':torch.tensor([]), 'relation_scores':torch.tensor([]), 'relation_scores_all':torch.tensor([]),
                                 'relation_labels':torch.tensor([]), 'attr_labels': torch.tensor([]), 'attr_scores': torch.tensor([])}

            gt_label = gt_boxlist.get_field("attributes").numpy()
            pred_label = sg_prediction["attr_labels"].numpy()
            pred_score = sg_prediction["attr_scores"].numpy()

            gt_mask_l = np.array([classindex in i for i in gt_label])
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_difficult_l = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            det = [False] * gt_bbox_l.shape[0]
            npos = npos + sum(~gt_difficult_l)
            class_recs[image_key] = {'bbox': gt_bbox_l,
                                     'difficult': gt_difficult_l,
                                     'det': det}

            pred_mask_l = np.logical_and(pred_label == classindex, np.not_equal(pred_score, 0.0)).nonzero()
            pred_bbox_l = pred_bbox[pred_mask_l[0]]
            pred_score_l = pred_score[pred_mask_l]

            for bbox_tmp, score_tmp in zip(pred_bbox_l, pred_score_l):
                image_ids.append(image_key)
                confidence.append(float(score_tmp))
                BB.append([float(z) for z in bbox_tmp])

        if npos == 0:
            # No ground truth examples
            # return 0
            continue
        #
        if len(confidence) == 0:
            # No detection examples
            # return 0
            all_rec.append(0)
            continue

        confidence = np.array(confidence)
        BB = np.array(BB)

        # sort by confidence

        sorted_ind = np.argsort(-confidence)# descending
        sorted_scores = -np.sort(-confidence)
        BB = BB[sorted_ind, :]
        image_ids = [image_ids[x] for x in sorted_ind]

        nd = len(image_ids)
        tp = np.zeros(nd)
        fp = np.zeros(nd)

        for d in range(nd):
            R = class_recs[image_ids[d]]
            bb = BB[d, :].astype(float)
            ovmax = -np.inf
            BBGT = R['bbox'].astype(float)

            if BBGT.size > 0:
                # compute overlaps
                # intersection
                ixmin = np.maximum(BBGT[:, 0], bb[0])
                iymin = np.maximum(BBGT[:, 1], bb[1])
                ixmax = np.minimum(BBGT[:, 2], bb[2])
                iymax = np.minimum(BBGT[:, 3], bb[3])
                iw = np.maximum(ixmax - ixmin + 1., 0.)
                ih = np.maximum(iymax - iymin + 1., 0.)
                inters = iw * ih

                # union
                uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                       (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                       (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

                overlaps = inters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > iou_thresh:
                if not R['difficult'][jmax]:
                    if not R['det'][jmax]:
                        tp[d] = 1.
                        R['det'][jmax] = 1
                    else:
                        fp[d] = 1.
            else:
                fp[d] = 1.

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / float(npos)
        all_rec.append(rec[-1])
        # prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        # print("prec", prec)
    all_rec = np.array(all_rec)
    print("Average recall among all classes is", np.mean(all_rec))


def do_sg_evaluation(dataset, predictions, output_folder, logger):
    """
    scene graph generation evaluation
    """
    prediction_file = os.path.join(output_folder, 'predictions.tsv')
    use_preds = True
    gt_dicts = prepare_vrd_groundtruths(dataset, predictions, use_preds) # Updated function to take predictions to allow for smaller numbers of predictions.
    
    if hasattr(dataset, 'labelmap'):
        predict_dicts = prepare_vrd_predictions(prediction_file, dataset.labelmap)
    else:
        predict_dicts = prepare_vrd_predictions_no_labelmap(prediction_file, dataset)

    evaluator = BasicSceneGraphEvaluator.all_modes(multiple_preds=False)

    top_Ns = [20, 50, 100]
    mode = "sgdet"
    result_dict = {}
    danfei_metric = {}
    rowan_metric = {}

    result_dict[mode + '_recall'] = {20: [], 50: [], 100: []}
    for image_key, gt_boxlist in gt_dicts.items():
        sg_prediction = predict_dicts[str(image_key)]
        if len(sg_prediction)==0:
            sg_prediction = {'bboxes':torch.as_tensor([]) , 'bbox_scores':torch.tensor([]), 'bbox_labels':torch.tensor([]), 'relation_pairs':torch.tensor([]), 'relation_scores':torch.tensor([]), 'relation_scores_all':torch.tensor([]), 'relation_labels':torch.tensor([])}

        gt_entry = {
            'gt_classes': gt_boxlist.get_field("labels").numpy(),
            'gt_relations': gt_boxlist.get_field("relation_labels").numpy().astype(int),
            'gt_boxes': gt_boxlist.bbox.numpy(),
        }

        if len(sg_prediction['relation_pairs'].numpy())==0:
            pred_entry = {
                'pred_boxes': np.array([]),
                'pred_classes': np.array([]),
                'obj_scores': np.array([]),
                'pred_rel_inds': np.array([]),
                'rel_scores': np.array([]),
            }
        else:
            obj_scores = sg_prediction['bbox_scores'].numpy()
            all_rels = sg_prediction['relation_pairs'].numpy()
            fp_pred = sg_prediction['relation_scores_all'].numpy()

            scores = np.column_stack((
                obj_scores[all_rels[:, 0]],
                obj_scores[all_rels[:, 1]],
                fp_pred[:, 1:].max(1)
            )).prod(1)
            sorted_inds = np.argsort(-scores)
            sorted_inds = sorted_inds[scores[sorted_inds] > 0]  # [:100]

            pred_entry = {
                'pred_boxes': sg_prediction['bboxes'].numpy(),
                'pred_classes': sg_prediction['bbox_labels'].numpy(),
                'obj_scores': sg_prediction['bbox_scores'].numpy(),
                'pred_rel_inds': all_rels[sorted_inds],
                'rel_scores': fp_pred[sorted_inds],
            }

        evaluator[mode].evaluate_scene_graph_entry(
            gt_entry,
            pred_entry,
        )
       # pdb.set_trace()
        evaluate(gt_boxlist.get_field("labels"), gt_boxlist.bbox, gt_boxlist.get_field("pred_labels"),
                    sg_prediction['bboxes'], sg_prediction['bbox_scores'], sg_prediction['bbox_labels'],
                    sg_prediction['relation_pairs'], sg_prediction['relation_scores_all'],
                    top_Ns, result_dict, mode)

    evaluator[mode].print_stats(logger)
    rowan_nums = {mode+str(key): np.mean(np.array(val))
                    for key, val in evaluator[mode].result_dict[mode + '_recall'].items()}
    rowan_metric.update(rowan_nums)

    logger.warning('=====================' + mode + '(IMP)' + '=========================')
    logger.warning("{}-recall@20: {}".format(mode, np.mean(np.array(result_dict[mode + '_recall'][20]))))
    logger.warning("{}-recall@50: {}".format(mode, np.mean(np.array(result_dict[mode + '_recall'][50]))))
    logger.warning("{}-recall@100: {}".format(mode, np.mean(np.array(result_dict[mode + '_recall'][100]))))
    danfei_nums = {mode+str(key): np.mean(np.array(val)) for key, val in result_dict[mode + '_recall'].items()}
    danfei_metric.update(danfei_nums)

    return {"danfei_metric": danfei_metric, "rowan_metric": rowan_metric}


def evaluate(gt_classes, gt_boxes, gt_rels,
             obj_rois, obj_scores, obj_labels,
             rel_inds, rel_scores,
             top_Ns, result_dict,
             mode, iou_thresh=0.5):
    #pdb.set_trace()
    gt_classes = gt_classes.cpu()
    gt_boxes = gt_boxes.cpu()
    gt_rels = gt_rels.cpu()

    obj_rois = obj_rois.cpu()
    obj_scores = obj_scores.cpu()
    obj_labels = obj_labels.cpu()
    rel_inds = rel_inds.cpu()
    rel_scores = rel_scores.cpu()

    if gt_rels.ne(0).sum() == 0:
        return (None, None)
    
    if len(rel_inds) == 0:
        for k in result_dict[mode + '_recall']:
            result_dict[mode + '_recall'][k].append(0)
        return (None, None)

    rel_sum = ((gt_rels.sum(1) > 0).int() + (gt_rels.sum(0) > 0).int())
    ix_w_rel = rel_sum.nonzero(as_tuple=False).numpy().squeeze()

    # label = (((gt_rel_label.sum(1) == 0).int() + (gt_rel_label.sum(0) == 0).int()) == 2)
    # change_ix = label.nonzero(as_tuple=False)

    gt_boxes = gt_boxes.numpy()
    num_gt_boxes = gt_boxes.shape[0]
    gt_relations = gt_rels.nonzero(as_tuple=False).numpy()
    gt_classes = gt_classes.view(-1, 1).numpy()

    gt_rels_view = gt_rels.contiguous().view(-1)
    gt_pred_labels = gt_rels_view[gt_rels_view.nonzero(as_tuple=False).squeeze()].contiguous().view(-1, 1).numpy()

    num_gt_relations = gt_relations.shape[0]
    if num_gt_relations == 0:
        return (None, None)
    gt_class_scores = np.ones(num_gt_boxes)
    gt_predicate_scores = np.ones(num_gt_relations)
    gt_triplets, gt_triplet_boxes, _ = _triplet(gt_pred_labels,
                                                gt_relations,
                                                gt_classes,
                                                gt_boxes,
                                                gt_predicate_scores,
                                                gt_class_scores)

    # pred
    box_preds = obj_rois.numpy()
    num_boxes = box_preds.shape[0]

    predicate_preds = rel_scores.numpy()

    # no bg
    predicate_preds = predicate_preds[:, 1:]
    predicates = np.argmax(predicate_preds, 1).ravel() + 1
    predicate_scores = predicate_preds.max(axis=1).ravel()

    relations = rel_inds.numpy()

    # if relations.shape[0] != num_boxes * (num_boxes - 1):

    # assert(relations.shape[0] == num_boxes * (num_boxes - 1))
    assert (predicates.shape[0] == relations.shape[0])
    num_relations = relations.shape[0]
    if mode == 'predcls':
        # if predicate classification task
        # use ground truth bounding boxes
        assert (num_boxes == num_gt_boxes)
        classes = gt_classes
        class_scores = gt_class_scores
        boxes = gt_boxes
    elif mode == 'sgcls':
        assert (num_boxes == num_gt_boxes)
        # if scene graph classification task
        # use gt boxes, but predicted classes
        classes = obj_labels.numpy()  # np.argmax(class_preds, 1)
        class_scores = obj_scores.numpy()
        boxes = gt_boxes
    elif mode == 'sgdet' or mode == 'sgdet+':
        # if scene graph detection task
        # use preicted boxes and predicted classes
        classes = obj_labels.numpy()  # np.argmax(class_preds, 1)
        class_scores = obj_scores.numpy()  # class_preds.max(axis=1)
        # boxes = []
        # for i, c in enumerate(classes):
        #     boxes.append(box_preds[i, c*4:(c+1)*4])
        # boxes = np.vstack(boxes)
        boxes = box_preds
    else:
        raise NotImplementedError('Incorrect Mode! %s' % mode)

    pred_triplets, pred_triplet_boxes, relation_scores = \
        _triplet(predicates, relations, classes, boxes,
                 predicate_scores, class_scores, is_pred=False)
    sorted_inds = np.argsort(relation_scores)[::-1]
    sorted_inds_obj = np.argsort(class_scores)[::-1]
    # compute recall

    for k in result_dict[mode + '_recall']:
        this_k = min(k, num_relations)
        keep_inds = sorted_inds[:this_k]
        keep_inds_obj = sorted_inds_obj[:this_k]

        # triplets_valid = _relation_recall_triplet(gt_triplets,
        #                           pred_triplets[keep_inds,:],
        #                           gt_triplet_boxes,
        #                           pred_triplet_boxes[keep_inds,:],
        #                           iou_thresh)

        recall = _relation_recall(gt_triplets,
                                  pred_triplets[keep_inds, :],
                                  gt_triplet_boxes,
                                  pred_triplet_boxes[keep_inds, :],
                                  iou_thresh)
        num_gt = gt_triplets.shape[0]

        result_dict[mode + '_recall'][k].append(recall / num_gt)
        # result_dict[mode + '_triplets'][k].append(triplets_valid)

    # for visualization
    return pred_triplets[sorted_inds, :], pred_triplet_boxes[sorted_inds, :]


def _triplet(predicates, relations, classes, boxes,
             predicate_scores, class_scores, is_pred=False):
    # format predictions into triplets

    # compute the overlaps between boxes
    if is_pred:
        overlaps = bbox_overlaps(torch.from_numpy(boxes).contiguous(), torch.from_numpy(boxes).contiguous())

    assert (predicates.shape[0] == relations.shape[0])
    num_relations = relations.shape[0]
    triplets = np.zeros([num_relations, 3]).astype(np.int32)
    triplet_boxes = np.zeros([num_relations, 8]).astype(np.int32)
    triplet_scores = np.zeros([num_relations]).astype(np.float32)
    for i in range(num_relations):
        triplets[i, 1] = predicates[i]
        sub_i, obj_i = relations[i, :2]
        triplets[i, 0] = classes[sub_i]
        triplets[i, 2] = classes[obj_i]
        triplet_boxes[i, :4] = boxes[sub_i, :]
        triplet_boxes[i, 4:] = boxes[obj_i, :]
        # compute triplet score
        score = class_scores[sub_i]
        score *= class_scores[obj_i]

        if is_pred:
            if overlaps[sub_i, obj_i] == 0:
                score *= 0
            else:
                score *= predicate_scores[i]
        else:
            score *= predicate_scores[i]

        triplet_scores[i] = score
    return triplets, triplet_boxes, triplet_scores


def _relation_recall(gt_triplets, pred_triplets,
                     gt_boxes, pred_boxes, iou_thresh):
    # compute the R@K metric for a set of predicted triplets
    num_gt = gt_triplets.shape[0]
    num_correct_pred_gt = 0
    for gt, gt_box in zip(gt_triplets, gt_boxes):
        keep = np.zeros(pred_triplets.shape[0]).astype(bool)
        for i, pred in enumerate(pred_triplets):
            if gt[0] == pred[0] and gt[1] == pred[1] and gt[2] == pred[2]:
                keep[i] = True
        if not np.any(keep):
            continue
        boxes = pred_boxes[keep, :]
        sub_iou = iou(gt_box[:4], boxes[:, :4])
        obj_iou = iou(gt_box[4:], boxes[:, 4:])
        inds = np.intersect1d(np.where(sub_iou >= iou_thresh)[0],
                              np.where(obj_iou >= iou_thresh)[0])
        if inds.size > 0:
            num_correct_pred_gt += 1
    return float(num_correct_pred_gt)


def _relation_recall_triplet(gt_triplets, pred_triplets,
                             gt_boxes, pred_boxes, iou_thresh):
    # compute the R@K metric for a set of predicted triplets
    num_gt = gt_triplets.shape[0]
    num_correct_pred_gt = 0
    triplets_valid = []
    boxes_valid = []
    for gt, gt_box in zip(gt_triplets, gt_boxes):
        keep = np.zeros(pred_triplets.shape[0]).astype(bool)
        for i, pred in enumerate(pred_triplets):
            if gt[0] == pred[0] and gt[1] == pred[1] and gt[2] == pred[2]:
                keep[i] = True
        if not np.any(keep):
            continue
        boxes = pred_boxes[keep, :]
        triplets = pred_triplets[keep, :]
        sub_iou = iou(gt_box[:4], boxes[:, :4])
        obj_iou = iou(gt_box[4:], boxes[:, 4:])
        inds = np.intersect1d(np.where(sub_iou >= iou_thresh)[0],
                              np.where(obj_iou >= iou_thresh)[0])
        if inds.size > 0:
            triplets_valid.append(triplets[inds[0]])
            boxes_valid.append(boxes[inds[0]])
            num_correct_pred_gt += 1
    return triplets_valid, boxes_valid


def _object_recall(gt_triplets, pred_triplets,
                   gt_boxes, pred_boxes, iou_thresh):
    # compute the R@K metric for a set of predicted triplets
    num_gt = gt_triplets.shape[0]
    num_correct_pred_gt = 0
    for gt, gt_box in zip(gt_triplets, gt_boxes):
        keep = np.zeros(pred_triplets.shape[0]).astype(bool)
        for i, pred in enumerate(pred_triplets):
            if gt[0] == pred[0]:
                keep[i] = True
        if not np.any(keep):
            continue
        boxes = pred_boxes[keep, :]
        box_iou = iou(gt_box[:4], boxes[:, :4])
        inds = np.where(box_iou >= iou_thresh)[0]
        if inds.size > 0:
            num_correct_pred_gt += 1
    return float(num_correct_pred_gt)


def _predicate_recall(gt_triplets, pred_triplets,
                      gt_boxes, pred_boxes, iou_thresh):
    # compute the R@K metric for a set of predicted triplets
    num_gt = gt_triplets.shape[0]
    num_correct_pred_gt = 0
    for gt, gt_box in zip(gt_triplets, gt_boxes):
        keep = np.zeros(pred_triplets.shape[0]).astype(bool)
        for i, pred in enumerate(pred_triplets):
            if gt[1] == pred[1]:
                keep[i] = True
        if not np.any(keep):
            continue
        boxes = pred_boxes[keep, :]
        sub_iou = iou(gt_box[:4], boxes[:, :4])
        obj_iou = iou(gt_box[4:], boxes[:, 4:])
        inds = np.intersect1d(np.where(sub_iou >= iou_thresh)[0],
                              np.where(obj_iou >= iou_thresh)[0])
        if inds.size > 0:
            num_correct_pred_gt += 1
    return float(num_correct_pred_gt)


def iou(gt_box, pred_boxes):
    # computer Intersection-over-Union between two sets of boxes
    ixmin = np.maximum(gt_box[0], pred_boxes[:, 0])
    iymin = np.maximum(gt_box[1], pred_boxes[:, 1])
    ixmax = np.minimum(gt_box[2], pred_boxes[:, 2])
    iymax = np.minimum(gt_box[3], pred_boxes[:, 3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih
    # union
    uni = ((gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) +
           (pred_boxes[:, 2] - pred_boxes[:, 0] + 1.) *
           (pred_boxes[:, 3] - pred_boxes[:, 1] + 1.) - inters)
    overlaps = inters / uni
    return overlaps

def prepare_vrd_predictions(pred_tsv_file, labelmap):
    predictions_dict = defaultdict(dict)
    for row in tsv_reader(pred_tsv_file):
        img_key = row[0]
        predictions = json.loads(row[1])
        bboxes = []
        bbox_scores = []
        bbox_labels = []
        for obj in predictions['objects']:
            bboxes.append(obj['rect'])
            bbox_scores.append(obj['conf'])
            bbox_labels.append(labelmap['class_to_ind'][obj['class']])
        idx_pairs = []
        relation_scores = []
        relation_scores_all = []
        relation_labels = []
        for triplet in predictions['relations']:
            idx_pairs.append([triplet['subj_id'], triplet['obj_id']])
            relation_scores.append(triplet['conf'])
            relation_scores_all.append(np.frombuffer(base64.b64decode(triplet['scores_all']), np.float32))
            relation_labels.append(labelmap['relation_to_ind'][triplet['class']])

        predictions_dict[img_key] = {'bboxes':torch.as_tensor(bboxes).reshape(-1, 4) , 'bbox_scores':torch.tensor(bbox_scores), 'bbox_labels':torch.tensor(bbox_labels), 'relation_pairs':torch.tensor(idx_pairs), 'relation_scores':torch.tensor(relation_scores), 'relation_scores_all':torch.tensor(relation_scores_all), 'relation_labels':torch.tensor(relation_labels)}
    return predictions_dict


def prepare_vrd_predictions_no_labelmap(pred_tsv_file, dataset, has_attr=False):
    #pdb.set_trace()
    predictions_dict = defaultdict(dict)
    for row in tsv_reader(pred_tsv_file):
        img_key = row[0]
        predictions = json.loads(row[1])
        bboxes = []
        bbox_scores = []
        bbox_labels = []
        for obj in predictions['objects']:
            bboxes.append(obj['rect'])
            bbox_scores.append(obj['conf'])
            bbox_labels.append(dataset.class_to_ind[obj['class']])
        idx_pairs = []
        relation_scores = []
        relation_scores_all = []
        relation_labels = []
        for triplet in predictions['relations']:
            idx_pairs.append([triplet['subj_id'], triplet['obj_id']])
            relation_scores.append(triplet['conf'])
            relation_scores_all.append(np.frombuffer(base64.b64decode(triplet['scores_all']), np.float32))
            relation_labels.append(dataset.relation_to_ind[triplet['class']])
        if has_attr:
            for attribute in predictions['attributes']: # Currently, attributes are stored in the object detection boxes in extra_fields['attr_labels']
                attr_labels = attribute["attr_labels"]
                attr_scores = attribute["attr_scores"]
        
        pred_dict = {'bboxes':torch.as_tensor(bboxes).reshape(-1, 4) , 'bbox_scores':torch.tensor(bbox_scores),
                     'bbox_labels':torch.tensor(bbox_labels), 'relation_pairs':torch.tensor(idx_pairs),
                     'relation_scores':torch.tensor(relation_scores), 'relation_scores_all':torch.tensor(relation_scores_all),
                     'relation_labels':torch.tensor(relation_labels)}
        if has_attr:
            pred_dict['attr_labels'] = torch.tensor(attr_labels) 
            pred_dict['attr_scores'] = torch.tensor(attr_scores)
        
        predictions_dict[img_key] = pred_dict
    return predictions_dict


def prepare_vrd_groundtruths(dataset, preds, use_preds=False):
    gt_dict = defaultdict(dict)
    idx_gen = range(len(dataset))
    if use_preds:
        idx_gen = preds.keys()
    for idx in idx_gen:
        img_key = dataset.get_img_key(idx)
        gt_boxlist = dataset.get_groundtruth(idx)
        gt_dict[img_key] = gt_boxlist
    return gt_dict