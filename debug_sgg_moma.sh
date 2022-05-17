# For debugging with fewer examples (for faster iteration), 
# set debug=True in MOMADataset constructor in scene_graph_benchmark/maskrcnn_benchmark/data/datasets/moma_dataset.py
export NGPUS=4
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --master_port 1238 --nproc_per_node=$NGPUS tools/train_sg_net.py --config-file "sgg_configs/moma/debug_rel_danfeiX_FPN101_reldn_moma.yaml" 