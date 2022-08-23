export NGPUS=4
export CUDA_VISIBLE_DEVICES=4,5,6,7
python -m torch.distributed.launch --nproc_per_node=$NGPUS --master_port 1235 tools/test_net.py --config-file "configs/moma/test_e2e_faster_rcnn_R_101_FPN_1x.yaml"
#python tools/test_net.py --config-file "configs/moma/test_e2e_faster_rcnn_R_101_FPN_1x.yaml"