# parameters
result_dir="/scratch_xijun/code/Video/SeViLA/lavis/output/BLIP2/NextQA/QA/"

exp_name='nextqa_ft_sevila_ttp_5e-5'
ckpt='sevila_checkpoints/sevila_pretrained.pth'
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.run --nproc_per_node=4 --master_port=12345 train.py \
--cfg-path lavis/projects/sevila/train/nextqa_ttp.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.frame_num=4 \
datasets.nextqa.vis_processor.train.n_frms=32 \
datasets.nextqa.vis_processor.eval.n_frms=32 \
run.batch_size_train=2 \
run.batch_size_eval=2 \
run.init_lr=5e-5 \
run.max_epoch=15 \
run.warmup_steps=1000 \
run.accum_grad_iters=8 \
model.task='qvh_freeze_loc_train_qa_with_loc_train_qa_vid' \
model.finetuned=${ckpt} \
run.task='videoqa'
