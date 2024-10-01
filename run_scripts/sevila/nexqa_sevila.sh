# parameters
result_dir="/scratch_xijun/code/Video/SeViLA/lavis/output/BLIP2/NextQA/QA/"

exp_name='qvh_pretraining'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 train.py \
--cfg-path lavis/projects/sevila/train/qvh.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.frame_num=4 \
datasets.qvh.vis_processor.train.n_frms=4 \
datasets.qvh.vis_processor.eval.n_frms=75 \
run.batch_size_train=16 \
run.batch_size_eval=4 \
run.init_lr=3e-5 \
run.max_epoch=80 \
run.warmup_steps=1000 \
run.accum_grad_iters=1 \
run.task='moment_retrieval'





exp_name='nextqa_ft'
ckpt='sevila_checkpoints/sevila_pretrained.pth'
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 train.py \
--cfg-path lavis/projects/sevila/train/nextqa.yaml \
--options run.output_dir=${result_dir}${exp_name} \
model.frame_num=4 \
datasets.nextqa.vis_processor.train.n_frms=32 \
datasets.nextqa.vis_processor.eval.n_frms=32 \
run.batch_size_train=8 \
run.batch_size_eval=8 \
run.init_lr=3e-5 \
run.max_epoch=10 \
run.warmup_steps=1000 \
run.accum_grad_iters=2 \
model.task='qvh_freeze_loc_train_qa_with_loc_train_qa_vid' \
model.finetuned=${ckpt} \
run.task='videoqa'