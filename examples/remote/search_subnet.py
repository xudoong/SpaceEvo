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
  local_dir: $CONFIG_DIR/../../

storage:
  teamdrive:
    storage_account_name: hexnas
    container_name: teamdrive
    mount_dir: /mnt/data


jobs:
{jobs}
"""

job_template_one_node = \
"""- name: {job_name}
  sku: G{num_gpus}
  command:
  - python -m torch.distributed.launch --nproc_per_node {num_gpus} search_subnet.py
    --superspace {superspace}
    --supernet_choice {supernet_choice}
    --dataset_path ./dataset
    --output_path /mnt/data/EdgeDL/quantized_nas/search_subnet
    --checkpoint_path /mnt/data/EdgeDL/quantized_nas/supernet_training
    --latency_constraint {latency_constraint}
    --latency_delta 5
    --batch_size {batch_size}
    --num_calib_batches {num_calib_batches}
    {use_testset}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('func', choices=['submit', 'debug'], help='submit job or local debug')
    parser.add_argument('--target', default='sing_octo', choices=list(target_dict.keys()), help='where to submit')
    parser.add_argument('--superspace', type=str, choices=['mobile_v1', 'mobile_v1_res', 'onnx', 'onnxw', 'mobilew'])
    parser.add_argument('--supernet_choice', type=str)
    parser.add_argument('--latency_constraint', required=True, type=int)
    parser.add_argument('--num_gpus', type=int, default=4, choices=[1, 2, 4, 8, 16, 32])
    parser.add_argument('--use_testset', action='store_true')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_calib_batches', type=int, default=40)
    args = parser.parse_args()

    if args.func == 'submit':
        mode = 1
    else:
        mode = 0

    job_template = job_template_one_node

    job_name = 'search_subnet-' + args.superspace + '-' + args.supernet_choice + f'-{args.latency_constraint}ms' + ('-testset' if args.use_testset else '')
    jobs = job_template.format(
        job_name=job_name, 
        superspace=args.superspace,
        supernet_choice=args.supernet_choice,
        latency_constraint=args.latency_constraint,
        num_gpus=args.num_gpus,
        use_testset= '--use_testset' if args.use_testset else '',
        batch_size=args.batch_size,
        num_calib_batches=args.num_calib_batches
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
    tmp_name = os.path.join("./.tmp", tmp_name)
    with open(tmp_name, "w") as fout:
        fout.write(result)

    if mode == 0:
        subprocess.run(["amlt", "run", "-t", "local", "--use-sudo", tmp_name, "--devices", "all"])
    else:
        # subprocess.run(f'amlt run -d {description} {tmp_name} {job_name}', shell=True)
        subprocess.run(["amlt", "run", '-r', "-d", description, tmp_name, job_name])

if __name__ == "__main__":
    main()