# parameters
result_dir="/scratch_xijun/code/Video/SeViLA/lavis/output/BLIP2/NextQA/QA/"

exp_name='blip2_nextqa_instruct_32f_to_4f_dist_temp01_ol_decode'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node=8 --master_port 12347 train.py \
--cfg-path lavis/projects/blip2/train/nextqa_instruct_32f_to_4f_dist_ol.yaml \
--options run.output_dir=${result_dir}${exp_name} \
run.batch_size_train=1 \
run.batch_size_eval=1 \
run.init_lr=1e-6 \
run.accum_grad_iters=8

