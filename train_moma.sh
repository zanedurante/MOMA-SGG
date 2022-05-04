export NGPUS=4
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "configs/moma/e2e_faster_rcnn_R_101_FPN_1x.yaml" DATASETS.TRAIN ("moma_train",) DATASETS.TEST ("moma_val",)
