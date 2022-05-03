# set debug=True in MOMADataset constructor in scene_graph_benchmark/maskrcnn_benchmark/data/datasets/moma_dataset.py
python -m pdb tools/train_net.py --config-file "configs/moma/e2e_faster_rcnn_R_101_FPN_1x.yaml" MODEL.ROI_BOX_HEAD.NUM_CLASSES 261
