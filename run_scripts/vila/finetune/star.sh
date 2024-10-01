# parameters
result_dir="/scratch_xijun/code/Video/SeViLA/lavis/output/VLAP/STAR/QA/"

exp_name='vlap_4f'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 train.py \
--cfg-path lavis/projects/vila/train/star.yaml \
--options run.output_dir=${result_dir}${exp_name}
