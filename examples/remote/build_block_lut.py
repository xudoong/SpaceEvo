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

storage:
  teamdrive:
    storage_account_name: hexnas
    container_name: teamdrive
    mount_dir: /mnt/data
    local_dir: $CONFIG_DIR/../../../faketeamdrive/


jobs:
{jobs}
"""


job_template = \
"""- name: {job_name}
  sku: G1
  command:
  - python eval_block_kd.py
    --superspace {superspace}
    --platform {platform}
    --output_path /mnt/data/EdgeDL/quantized_nas/block_kd/lut
    --stage_list {stage_list} {width_window_filter}
    --hw_list 160 192 224
    --dataset_path ./dataset
    --checkpoint_path /mnt/data/EdgeDL/quantized_nas/block_kd
    --teacher_checkpoint_path /mnt/data/EdgeDL/quantized_nas/block_kd/teacher_checkpoint/efficientnet_b5/checkpoint.pth
    --debug
  submit_args: 
    env:
      DEBUG: {debug}
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('func', choices=['submit', 'debug'], help='submit job or local debug')
    parser.add_argument('--target', default='sing_octo', choices=list(target_dict.keys()), help='where to submit')
    parser.add_argument('--stage_list', nargs='+')
    parser.add_argument('--width_window_filter', nargs='*')
    parser.add_argument('--superspace', choices=['onnx', 'onnx_v1', 'onnxw', 'mobilew'], default='onnxw')
    args = parser.parse_args()

    if 'onnx' in args.superspace:
      platform = 'onnx_lut'
    else:
      platform = 'tflite27_cpu_int8'

    # legal check
    for stage_name in args.stage_list:
      assert stage_name in [f'stage{i}_{j}' for i in range(1, 7) for j in range(7)], f'invaild stage_name: {stage_name}'
    stage_list_str = '_'.join(args.stage_list)

    if args.func == 'submit':
        mode = 1
    else:
        mode = 0

    if args.width_window_filter:
      width_window_filter_args = '--width_window_filter ' + ' '.join([v for v in args.width_window_filter])
    else:
      width_window_filter_args = ''
    
    job_name = f'build_block_kd_v2_lut-{args.superspace}-{stage_list_str}'
    if args.width_window_filter:
      job_name += '-w' + ''.join(args.width_window_filter)
    jobs = job_template.format(
        job_name=job_name, 
        debug=mode, 
        stage_list=" ".join(args.stage_list),
        superspace=args.superspace,
        platform=platform,
        width_window_filter=width_window_filter_args
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