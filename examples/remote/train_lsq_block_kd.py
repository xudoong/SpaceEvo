import subprocess
import random
import string
import os
import argparse

from target import target_dict


template = \
"""
description: {job_name}

{target}

code:
  # upload the code
  local_dir: $CONFIG_DIR/../../

storage:
  teamdrive:
    storage_account_name: hexnas
    container_name: teamdrive
    mount_dir: /mnt/data

jobs:
{jobs}
"""


job_template = \
"""- name: {job_name}
  sku: G4
  command:
  - python -m torch.distributed.launch --nproc_per_node=4 train_block_kd.py 
    --superspace {superspace}
    --output_path /mnt/data/EdgeDL/quantized_nas/block_kd/{superspace}/
    --inplace_distill_from_teacher
    --num_epochs 5
    --stage_list {stage_list}
    --hw_list 160 192 224
    --dataset_path ./dataset
  - python -m torch.distributed.launch --nproc_per_node=4 lsq_block_kd.py
    --superspace {superspace}
    --output_path /mnt/data/EdgeDL/quantized_nas/block_kd/{superspace}/
    --inplace_distill_from_teacher
    --num_epochs 1
    --stage_list {stage_list}
    --hw_list 160 192 224
    --dataset_path ./dataset
    --teacher_checkpoint_path /mnt/data/EdgeDL/quantized_nas/block_kd/teacher_checkpoint/efficientnet_b5/checkpoint.pth
    {half_batch_size}
  submit_args: 
    env:
      DEBUG: {debug}
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('func', choices=['submit', 'debug'], help='submit job or local debug')
    parser.add_argument('--target', default='sing_octo', choices=list(target_dict.keys()), help='where to submit')
    parser.add_argument('--stage_list', nargs='+')
    parser.add_argument('--half_batch_size', action='store_true')
    parser.set_defaults(half_batch_size=True)
    parser.add_argument('--superspace', choices=['onnx', 'onnx_v1', 'onnxw'], default='onnxw')
    args = parser.parse_args()

    # legal check
    for stage_name in args.stage_list:
      assert stage_name in [f'stage{i}_{j}' for i in range(1, 7) for j in range(7)], f'invaild stage_name: {stage_name}'
    stage_list_str = '_'.join(args.stage_list)

    if args.func == 'submit':
        mode = 1
    else:
        mode = 0
    
    half_batch_size_str = ''
    if args.half_batch_size:
      half_batch_size_str = '--train_batch_size 32 --learning_rate_list 0.00125 0.00125 0.00125 0.00125 0.00125 0.00125'

    job_name = f'block_kd_v2-{args.superspace}-{stage_list_str}'
    jobs = job_template.format(
        job_name=job_name, debug=mode, stage_list=" ".join(args.stage_list), half_batch_size=half_batch_size_str,
        superspace=args.superspace
    )
    description = f'{job_name}'

    # ======================================================================================================
    # Don't need to modify following code
    result = template.format(
        job_name=job_name,
        jobs=jobs,
        target=target_dict[args.target], 
    )   
    print(result)

    tmp_name = ''.join(random.choices(string.ascii_lowercase, k=6)) + job_name
    tmp_name = os.path.join(os.path.dirname(__file__), '.tmp', tmp_name)
    os.makedirs(os.path.dirname(tmp_name), exist_ok=True)
    with open(tmp_name, "w") as fout:
        fout.write(result)

    if mode == 0:
        subprocess.run(["amlt", "run", "-t", "local", "--use-sudo", tmp_name, "--devices", "all"])
    else:
        # subprocess.run(f'amlt run -d {description} {tmp_name} {job_name}', shell=True)
        subprocess.run(["amlt", "run", '-r', "-d", description, tmp_name, job_name])

if __name__ == "__main__":
    main()