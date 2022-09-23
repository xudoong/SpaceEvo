python -m torch.distributed.launch --nproc_per_node=2 train_block_kd.py\
    --dataset_path ${IMAGENET_PATH}\
    --output_path tmp_demo/block_kd\
    --superspace mobilew\
    --inplace_distill_from_teacher\
    --num_epochs 5\
    --hw_list 160 192 224\
    --stage_list stage1_6