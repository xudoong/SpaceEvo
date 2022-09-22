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

job_template_one_node = \
"""- name: {job_name}
  sku: G{num_gpus}
  command:
  - python -m torch.distributed.launch --nproc_per_node {num_gpus} train_supernet.py 
    --config-file supernet_training_configs/train.yaml
    --superspace {superspace}
    --supernet_choice {supernet_choice}
    --batch_size_per_gpu {fp32_batch_size}
    --resume
  - python -m torch.distributed.launch --nproc_per_node {num_gpus} train_supernet.py 
    --config-file supernet_training_configs/train.yaml
    --superspace {superspace}
    --supernet_choice {supernet_choice}
    --batch_size_per_gpu {int8_batch_size}
    --quant_mode
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('func', choices=['submit', 'debug'], help='submit job or local debug')
    parser.add_argument('--target', default='itp_rr1', choices=list(target_dict.keys()), help='where to submit')
    parser.add_argument('--superspace', type=str, choices=['mobile_v1', 'mobile_v1_res', 'onnx', 'onnxw', 'mobilew'], required=True)
    parser.add_argument('--supernet_choice', type=str)
    parser.add_argument('--num_gpus', type=int, default=8, choices=[4, 8, 16, 32])
    parser.add_argument('--fp32_batch_size', type=int, default=64, choices=[32, 48, 64, 72, 96, 128])
    parser.add_argument('--int8_batch_size', type=int, default=32, choices=[32, 48, 64, 72, 96, 128])
    args = parser.parse_args()

    if args.func == 'submit':
        mode = 1
    else:
        mode = 0

    job_template = job_template_one_node

    job_name = 'supernet_training-' + args.superspace + '-' + args.supernet_choice
    jobs = job_template.format(
        job_name=job_name, 
        superspace=args.superspace,
        supernet_choice=args.supernet_choice,
        num_gpus=args.num_gpus,
        fp32_batch_size=args.fp32_batch_size,
        int8_batch_size=args.int8_batch_size
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