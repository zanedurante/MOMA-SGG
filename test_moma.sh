export NGPUS=4
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_net.py --config-file "configs/moma/test_e2e_faster_rcnn_R_101_FPN_1x.yaml"
#