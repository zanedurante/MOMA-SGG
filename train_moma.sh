export NGPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "configs/moma/67_e2e_faster_rcnn_R_101_FPN_1x.yaml"
#export CUDA_LAUNCH_BLOCKING=1
#python tools/train_net.py --config-file "configs/moma/e2e_faster_rcnn_R_101_FPN_1x.yaml"