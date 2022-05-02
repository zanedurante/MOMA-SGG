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

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from momaapi import MOMA
import os
BOX_SCALE = 1024  # Scale at which we have the boxes

from torch.utils.data import DataLoader

class MOMADataset(torch.utils.data.Dataset):

    def __init__(self, split, moma_path='default val set in paths_catalog.py', num_instances_threshold=50, transforms=None,
                filter_empty_rels=True, num_im=-1, num_val_im=5000,
                filter_duplicate_rels=True, filter_non_overlap=True, flip_aug=False, custom_eval=False, custom_path='', 
                debug=False, no_human_classes=True, relation_on=False):
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
        if debug:
            self.moma = MOMA(moma_path, load_val=True, toy=True)
            self.actor_classes = self.moma.get_cnames(concept='actor')
            self.object_classes = self.moma.get_cnames(concept='object')
        else:
            self.moma = MOMA(moma_path, load_val=True) # Save moma API object
            self.actor_classes = self.moma.get_cnames(concept='actor', threshold=num_instances_threshold, split='either') # Only train on actors and objects from val set
            self.object_classes = self.moma.get_cnames(concept='object', threshold=num_instances_threshold, split='either') # Ensure there are at least 50 examples to include

        if 'crowd' in self.actor_classes:
            self.actor_classes.remove('crowd')
        
        self.has_human_classes = True
        if no_human_classes:
            self.has_human_classes = False
            self.real_human_classes = self.actor_classes
            self.actor_classes = ['person']
            
        self.classes = self.actor_classes + self.object_classes

        print("Object detection on", len(self.classes), "classes")

        self.flip_aug = flip_aug
        self.split = split
        self.filter_non_overlap = filter_non_overlap and self.split == 'train'
        self.filter_duplicate_rels = filter_duplicate_rels and self.split == 'train'
        self.transforms = transforms

        #self.ind_to_classes, self.ind_to_predicates, self.ind_to_attributes = load_info() # contiguous 151, 51 containing __background__
        #self.categories = {i : self.ind_to_classes[i] for i in range(len(self.ind_to_classes))}

        self.custom_eval = custom_eval
        assert not self.custom_eval # MOMA Does not support custom eval currently

        # TODO: Implement everything (include relationships) via self.create_dataset()
        self.dataset_dict = self.create_dataset()
        print("DATASET HAS", len(self.dataset_dict), "examples")

        self.relation_on = relation_on


    def __len__(self):
        return len(self.dataset_dict)

    def __getitem__(self, index):
        #if self.split == 'train':
        #    while(random.random() > self.img_info[index]['anti_prop']):
        #        index = int(random.random() * len(self.filenames))

        filename = self.dataset_dict[index]["file_name"]
        height = self.dataset_dict[index]["height"]
        width = self.dataset_dict[index]["width"]
        boxes = self.dataset_dict[index]["bboxes"]
        labels = torch.Tensor(self.dataset_dict[index]["labels"])

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
            return self.dataset_dict[index]["pred_triplets"], self.dataset_dict[index]["pred_matrix"], index

        return img, boxlist, index


    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        return {"height": self.dataset_dict[idx]["height"], "width": self.dataset_dict[idx]["width"]}

    def create_dataset(self):
        ids_hoi = self.moma.get_ids_hoi(split=self.split)
        ids_hoi = sorted(ids_hoi) # Added for reproducability
        #ids_hoi = ids_hoi[120:] # Debugging
        anns_hoi = self.moma.get_anns_hoi(ids_hoi)
        image_paths = self.moma.get_paths(ids_hoi=ids_hoi)
        print("Number of examples:", len(ids_hoi))
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
            count = 0

            for actor in ann_hoi.actors:
                bbox = actor.bbox
                if self.has_human_classes:
                    id = actor.id
                    actor_cname = actor.cname
                else: # If no human classes only has generic "person" class 
                    id = '0'
                    actor_cname = self.actor_classes[0]
                if actor_cname in self.classes:
                    class_id = self.classes.index(actor_cname)
                    obj_labels.append(class_id)
                    obj_bbs.append([bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height])
                if id not in obj_id_map:
                    obj_id_map[id] = count
                    count += 1

            for object in ann_hoi.objects:
                bbox = object.bbox
                id = object.id
                object_cname = object.cname
                object_cid = object.cid + len(self.actor_classes)
                if object_cname in self.classes:
                    class_id = self.classes.index(object_cname)
                    obj_labels.append(class_id)
                    obj_bbs.append([bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height])
                obj_id_map[id] = count
                count += 1

            record["labels"] = obj_labels
            record["bboxes"] = obj_bbs

            if len(obj_bbs) == 0:
                continue # maskrcnn benchmark can only train on images with bounding boxes

            # relation triplets
            ia_triplets = []
            ta_triplets = []
            att_triplets = []
            rel_triplets = []

            # relation matrices
            obj_num = len(obj_id_map)
            ia_matrix = torch.zeros([obj_num, obj_num], dtype=torch.int64)
            ta_matrix = torch.zeros([obj_num, obj_num], dtype=torch.int64)
            att_matrix = torch.zeros([obj_num, obj_num], dtype=torch.int64)
            rel_matrix = torch.zeros([obj_num, obj_num], dtype=torch.int64)

            for ia in ann_hoi.ias:
                src = '0' if ia.id_src.isalpha() and not self.has_human_classes else ia.id_src
                ia_triplets.append((src, src, ia.cid))
                ia_matrix[obj_id_map[src], obj_id_map[src]] = ia.cid
            for ta in ann_hoi.tas:
                src = '0' if ta.id_src.isalpha() and not self.has_human_classes else ta.id_src
                trg = '0' if ta.id_trg.isalpha() and not self.has_human_classes else ta.id_trg
                ta_triplets.append((src, trg, ta.cid))
                ta_matrix[obj_id_map[src], obj_id_map[trg]] = ta.cid
            for att in ann_hoi.atts:
                src = '0' if att.id_src.isalpha() and not self.has_human_classes else att.id_src
                att_triplets.append((src, src, att.cid))
                att_matrix[obj_id_map[src], obj_id_map[src]] = att.cid
            for rel in ann_hoi.rels:
                src = '0' if rel.id_src.isalpha() and not self.has_human_classes else rel.id_src
                trg = '0' if rel.id_trg.isalpha() and not self.has_human_classes else rel.id_trg
                rel_triplets.append((src, trg, rel.cid))
                rel_matrix[obj_id_map[src], obj_id_map[trg]] = rel.cid

            record["pred_triplets"] = {
                "ia": ia_triplets,
                "ta": ta_triplets,
                "att": att_triplets,
                "rel": rel_triplets
            }

            record["pred_matrix"] = {
                "ia": ia_matrix,
                "ta": ta_matrix,
                "att": att_matrix,
                "rel": rel_matrix
            }

            dataset_dicts.append(record)

        return dataset_dicts