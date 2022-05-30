import os
import sys
import torch
import h5py
import json
from PIL import Image
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import random
import os.path as op

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.data.datasets.vg_tsv import _box_filter
from .momaapi import MOMA
import os
BOX_SCALE = 1024  # Scale at which we have the boxes


class MOMADataset(torch.utils.data.Dataset):
    
    def __init__(self, split, moma_path='default val set in paths_catalog.py', num_instances_threshold=50, transforms=None,
                filter_empty_rels=True, num_im=-1, num_val_im=5000,
                filter_duplicate_rels=True, filter_non_overlap=True, flip_aug=False, custom_eval=False, custom_path='', 
                debug=True, no_human_classes=True, relation_on=True, attribute_on=True, freq_prior_file='/home/ssd/data/moma/moma.freq_prior.npy'):
        """
        Torch dataset for MOMA
        Parameters:
            split: Must be train, test, or val
            img_dir: folder containing all vg images
            roidb_file:  HDF5 containing the GT boxes, classes, and relationships
            dict_file: JSON Contains mapping of classes/relationships to words
            image_file: HDF5 containing image filenames
            filter_empty_rels: True if we filter out images without relationships between
                             boxes. One might want to set this to false if training a detector.
            filter_duplicate_rels: Whenever we see a duplicate relationship we'll sample instead
            num_im: Number of images in the entire dataset. -1 for all images.
            num_val_im: Number of images in the validation set (must be less than num_im
               unless num_im is -1.)
        """
        
        # for debug
        # num_im = 10000
        # num_val_im = 4
        assert split in {'train', 'val', 'test'}
        print("getting split:", split)
        #if debug:
        #    self.moma = MOMA(moma_path, load_val=True, toy=True)
        #    self.actor_classes = self.moma.get_cnames(concept='actor')
        #    self.object_classes = self.moma.get_cnames(concept='object')
        #else:
        self.debug = debug
        self.moma = MOMA(moma_path, load_val=True) # Save moma API object
        self.actor_classes = self.moma.get_cnames(kind='actor', threshold=num_instances_threshold, split='either') # Only train on actors and objects from val set
        self.object_classes = self.moma.get_cnames(kind='object', threshold=num_instances_threshold, split='either') # Ensure there are at least 50 examples to include

        if 'crowd' in self.actor_classes:
            self.actor_classes.remove('crowd')
        
        self.has_human_classes = True
        if no_human_classes:
            self.has_human_classes = False
            self.real_human_classes = self.actor_classes
            self.actor_classes = ['person']
            
        self.classes = self.actor_classes + self.object_classes
        self.class_to_ind = {}
        if no_human_classes:
            self.class_to_ind['person'] = 0
        else:
            for idx, actor in enumerate(self.actor_classes):
                self.class_to_ind[actor] = idx
        for idx, object in enumerate(self.object_classes):
            self.class_to_ind[object] = len(self.actor_classes) + idx
            
        self.ind_to_class = {v: k for k, v in self.class_to_ind.items()}

        print("Object detection on", len(self.classes), "classes")
        print("Class list:", self.classes)

        self.flip_aug = flip_aug
        self.split = split
        self.filter_non_overlap = filter_non_overlap and self.split == 'train'
        self.filter_duplicate_rels = filter_duplicate_rels and self.split == 'train'
        self.transforms = transforms

        #self.ind_to_classes, self.ind_to_predicates, self.ind_to_attributes = load_info() # contiguous 151, 51 containing __background__
        #self.categories = {i : self.ind_to_classes[i] for i in range(len(self.ind_to_classes))}

        self.custom_eval = custom_eval
        assert not self.custom_eval # MOMA Does not support custom eval currently

        self.relation_on = relation_on
        self.attribute_on = attribute_on
        self.rel_id_map = {}
        self.rel_id_map['__no_relation__'] = 0
        self.att_id_map = {}
        self.att_id_map['__no_attribute__'] = 0

        if self.relation_on:
            for ta in self.moma.taxonomy['ta']:
                self.rel_id_map[ta[0]] = len(self.rel_id_map)
            for rel in self.moma.taxonomy['rel']:
                self.rel_id_map[rel[0]] = len(self.rel_id_map)
            self.num_rel_classes = len(self.rel_id_map)
            self.ind_to_relation = {v: k for k, v in self.rel_id_map.items()}
        if self.attribute_on:
            for ia in self.moma.taxonomy['ia']:
                self.att_id_map[ia[0]] = len(self.att_id_map)
            for att in self.moma.taxonomy['att']:
                self.att_id_map[att[0]] = len(self.att_id_map)
            self.num_att_classes = len(self.att_id_map)
            self.ind_to_attribute = {v: k for k, v in self.att_id_map.items()}

        print("Relation prediction on", len(self.rel_id_map), "classes")
        # print("Relation list:", self.rel_id_map.keys())
        print("Attribute prediction on", len(self.att_id_map), "classes")
        # print("Attribute list:", self.att_id_map.keys())

        # TODO: Implement everything (include relationships) via self.create_dataset()
        self.dataset_dict = self.create_dataset()
        print("DATASET HAS:", self.__len__(), "examples")

        if self.relation_on and self.split == 'train' and not op.exists(freq_prior_file):
            print("Computing frequency prior matrix...")
            fg_matrix, bg_matrix = self._get_freq_prior()
            prob_matrix = fg_matrix.astype(np.float32)
            prob_matrix[:, :, 0] = bg_matrix
            prob_matrix[:, :, 0] += 1
            prob_matrix /= np.sum(prob_matrix, 2)[:, :, None]
            np.save(freq_prior_file, prob_matrix)

    def _get_freq_prior(self, must_overlap=False):
        fg_matrix = np.zeros((
            len(self.classes),
            len(self.classes),
            self.num_rel_classes
        ), dtype=np.int64)

        bg_matrix = np.zeros((
            len(self.classes),
            len(self.classes),
        ), dtype=np.int64)

        for ex_ind in range(self.__len__()):
            target = self.get_groundtruth(ex_ind)
            gt_classes = target.get_field('labels').numpy()
            gt_relations = target.get_field('relation_labels').numpy()
            gt_boxes = target.bbox

            # For the foreground, we'll just look at everything
            try:
                o1o2 = gt_classes[gt_relations[:, :2]]
                for (o1, o2), gtr in zip(o1o2, gt_relations[:, 2]):
                    fg_matrix[o1, o2, gtr] += 1

                # For the background, get all of the things that overlap.
                o1o2_total = gt_classes[np.array(
                    _box_filter(gt_boxes, must_overlap=must_overlap), dtype=int)]
                for (o1, o2) in o1o2_total:
                    bg_matrix[o1, o2] += 1
            except IndexError as e:
                assert len(gt_relations) == 0

            if ex_ind % 20 == 0:
                print("processing {}/{}".format(ex_ind, self.__len__()))

        return fg_matrix, bg_matrix

    def get_groundtruth(self, index):
        # similar to __getitem__ but without transform
        filename = self.dataset_dict[index]["file_name"]
        boxes = self.dataset_dict[index]["bboxes"]
        labels = torch.Tensor(self.dataset_dict[index]["labels"]).type(torch.int64)
        img = Image.open(filename).convert("RGB")

        boxlist = BoxList(boxes, img.size, mode="xyxy")
        boxlist.add_field("labels", labels)
        boxlist.add_field("relation_labels", self.dataset_dict[index]["pred_triplets"])
        boxlist.add_field("pred_labels", self.dataset_dict[index]["pred_matrix"])
        boxlist.add_field("attributes", self.dataset_dict[index]["attributes"])

        return boxlist


    def __len__(self):
        if self.debug:
            return 40 # For faster debugging
        return len(self.dataset_dict)

    def __getitem__(self, index):
        #if self.split == 'train':
        #    while(random.random() > self.img_info[index]['anti_prop']):
        #        index = int(random.random() * len(self.filenames))

        filename = self.dataset_dict[index]["file_name"]
        height = self.dataset_dict[index]["height"]
        width = self.dataset_dict[index]["width"]
        boxes = self.dataset_dict[index]["bboxes"]
        labels = torch.Tensor(self.dataset_dict[index]["labels"]).type(torch.int64)

        img = Image.open(filename).convert("RGB")
        if img.size[0] != width or img.size[1] != height:
            print("File:", filename, "does not match metadata!")
            print('='*20, ' ERROR index ', str(index), ' ', str(img.size), ' ', str(width), ' ', str(height), ' ', '='*20)


        #flip_img = (random.random() > 0.5) and self.flip_aug and (self.split == 'train')
        boxlist = BoxList(boxes, img.size, mode="xyxy")
        # add the labels to the boxlist
        boxlist.add_field("labels", labels)

        assert len(boxes) == len(labels)

        #if flip_img:
        #    img = img.transpose(method=Image.FLIP_LEFT_RIGHT)

        if self.transforms:
            img, boxlist = self.transforms(img, boxlist)

        if self.relation_on:
            boxlist.add_field("relation_labels", self.dataset_dict[index]["pred_triplets"])
            boxlist.add_field("pred_labels",  self.dataset_dict[index]["pred_matrix"])
            self.contrastive_loss_target_transform(boxlist)

        if self.attribute_on:
            boxlist.add_field("attributes", self.dataset_dict[index]["attributes"])

        return img, boxlist, index


    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        return {"height": self.dataset_dict[idx]["height"], "width": self.dataset_dict[idx]["width"]}
    
    def get_img_key(self, idx):
        return idx
    
    def create_dataset(self):
        ids_hoi = self.moma.get_ids_hoi(split=self.split)
        ids_hoi = sorted(ids_hoi) # Added for reproducability
        #ids_hoi = ids_hoi[120:] # Debugging


        anns_hoi = self.moma.get_anns_hoi(ids_hoi)
        image_paths = self.moma.get_paths(ids_hoi=ids_hoi)
        dataset_dicts = []

        # hoi --> act --> metadata
        for ann_hoi, image_path in zip(anns_hoi, image_paths):

            record = {}
            record["file_name"] = image_path
            record["image_id"] = ann_hoi.id

            act_id = self.moma.get_ids_act(ids_hoi=[ann_hoi.id])
            metadatum = self.moma.get_metadata(act_id)[0]

            record["height"]= metadatum.height
            record["width"] = metadatum.width

            obj_labels = []
            obj_bbs = []
            obj_id_map = {}

            for actor in ann_hoi.actors:
                bbox = actor.bbox
                if self.has_human_classes:
                    id = actor.id
                    actor_cname = actor.cname
                else: # If no human classes only has generic "person" class 
                    # id = '0'
                    id = actor.id
                    actor_cname = self.actor_classes[0]
                if actor_cname in self.classes:
                    class_id = self.classes.index(actor_cname)
                    obj_labels.append(class_id)
                    obj_bbs.append([bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height])
                    obj_id_map[id] = len(obj_id_map)

            for object in ann_hoi.objects:
                bbox = object.bbox
                id = object.id
                object_cname = object.cname
                object_cid = object.cid + len(self.actor_classes)
                if object_cname in self.classes:
                    class_id = self.classes.index(object_cname)
                    obj_labels.append(class_id)
                    obj_bbs.append([bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height])
                    obj_id_map[id] = len(obj_id_map)

            # print(obj_labels)
            record["labels"] = obj_labels
            record["bboxes"] = obj_bbs

            if len(obj_bbs) == 0:
                continue # maskrcnn benchmark can only train on images with bounding boxes

            # relation triplets
            relation_triplets = []
            # relation matrices
            relations = torch.zeros([len(obj_labels), len(obj_labels)], dtype=torch.int64)
            attributes = [[] for _ in range(len(obj_labels))] #[[], [], [], []]
            for ia in ann_hoi.ias:
                src = ia.id_src
                if src in obj_id_map:
                    src_id = obj_id_map[src]
                    predicate = self.att_id_map[ia.cname]
                    attributes[src_id].append(predicate)

            for att in ann_hoi.atts:
                src = att.id_src
                if src in obj_id_map:
                    src_id = obj_id_map[src]
                    predicate = self.att_id_map[att.cname]
                    attributes[src_id].append(predicate)

            for att in attributes:
                pad = 5 - len(att)
                for _ in range(pad):
                    att.append(0)
            record["attributes"] = torch.tensor(attributes)

            for ta in ann_hoi.tas:
                src = ta.id_src
                trg = ta.id_trg
                if src in obj_id_map and trg in obj_id_map:
                    src_id = obj_id_map[src]
                    trg_id = obj_id_map[trg]
                    predicate = self.rel_id_map[ta.cname]
                    relations[src_id, trg_id] = predicate
                    relation_triplets.append([src_id, trg_id, predicate])

            for rel in ann_hoi.rels:
                src = rel.id_src
                trg = rel.id_trg
                if src in obj_id_map and trg in obj_id_map:
                    src_id = obj_id_map[src]
                    trg_id = obj_id_map[trg]
                    predicate = self.rel_id_map[rel.cname]
                    relations[src_id, trg_id] = predicate
                    relation_triplets.append([src_id, trg_id, predicate])
            
            if len(relation_triplets) == 0:
                continue # scene_graph_benchmark can only train on relationships that exist (have bounding boxes)

            # print(relation_triplets)
            relation_triplets = torch.tensor(relation_triplets)
            record["pred_triplets"] = relation_triplets
            record["pred_matrix"] = relations

            dataset_dicts.append(record)
        
        # Create relation_to_ind --> not sure if this is right!
        self.relation_to_ind = self.rel_id_map # For compatibility with repo
        self.attribute_to_ind = self.att_id_map
        
        return dataset_dicts

    def contrastive_loss_target_transform(self, target):
        # add relationship annotations
        relation_triplets = target.get_field("relation_labels")
        sbj_gt_boxes = np.zeros((len(relation_triplets), 4), dtype=np.float32)
        obj_gt_boxes = np.zeros((len(relation_triplets), 4), dtype=np.float32)
        sbj_gt_classes_minus_1 = np.zeros(len(relation_triplets), dtype=np.int32)
        obj_gt_classes_minus_1 = np.zeros(len(relation_triplets), dtype=np.int32)
        prd_gt_classes_minus_1 = np.zeros(len(relation_triplets), dtype=np.int32)
        for ix, rel in enumerate(relation_triplets):
            # sbj
            sbj_gt_box = target.bbox[rel[0]]
            sbj_gt_boxes[ix] = sbj_gt_box
            sbj_gt_classes_minus_1[ix] = target.get_field('labels')[rel[0]]
            # obj
            obj_gt_box = target.bbox[rel[1]]
            obj_gt_boxes[ix] = obj_gt_box
            obj_gt_classes_minus_1[ix] = target.get_field('labels')[rel[1]]
            # prd
            prd_gt_classes_minus_1[ix] = rel[2] - 1  # excludes first one

        target.add_field('sbj_gt_boxes', torch.from_numpy(sbj_gt_boxes))
        target.add_field('obj_gt_boxes', torch.from_numpy(obj_gt_boxes))
        target.add_field('sbj_gt_classes_minus_1', torch.from_numpy(sbj_gt_classes_minus_1))
        target.add_field('obj_gt_classes_minus_1', torch.from_numpy(obj_gt_classes_minus_1))
        target.add_field('prd_gt_classes_minus_1', torch.from_numpy(prd_gt_classes_minus_1))

        # misc
        num_obj_classes = len(self.classes)
        num_prd_classes = len(self.rel_id_map) - 1

        sbj_gt_overlaps = np.zeros(
            (len(relation_triplets), num_obj_classes), dtype=np.float32)
        for ix in range(len(relation_triplets)):
            sbj_cls = sbj_gt_classes_minus_1[ix]
            sbj_gt_overlaps[ix, sbj_cls] = 1.0
        # sbj_gt_overlaps = scipy.sparse.csr_matrix(sbj_gt_overlaps)
        target.add_field('sbj_gt_overlaps', torch.from_numpy(sbj_gt_overlaps))

        obj_gt_overlaps = np.zeros(
            (len(relation_triplets), num_obj_classes), dtype=np.float32)
        for ix in range(len(relation_triplets)):
            obj_cls = obj_gt_classes_minus_1[ix]
            obj_gt_overlaps[ix, obj_cls] = 1.0
        # obj_gt_overlaps = scipy.sparse.csr_matrix(obj_gt_overlaps)
        target.add_field('obj_gt_overlaps', torch.from_numpy(obj_gt_overlaps))

        prd_gt_overlaps = np.zeros(
            (len(relation_triplets), num_prd_classes), dtype=np.float32)
        pair_to_gt_ind_map = np.zeros(
            (len(relation_triplets)), dtype=np.int32)
        for ix in range(len(relation_triplets)):
            prd_cls = prd_gt_classes_minus_1[ix]
            prd_gt_overlaps[ix, prd_cls] = 1.0
            pair_to_gt_ind_map[ix] = ix
        # prd_gt_overlaps = scipy.sparse.csr_matrix(prd_gt_overlaps)
        target.add_field('prd_gt_overlaps', torch.from_numpy(prd_gt_overlaps))
        target.add_field('pair_to_gt_ind_map', torch.from_numpy(pair_to_gt_ind_map))
    # Already defined above...
    #def get_img_info(self, index):
    #    height = self.dataset_dict[index]["height"]
    #    width = self.dataset_dict[index]["width"]
    #    hw_dict = {"height": int(height), "width": int(width)}
    #    return hw_dict



