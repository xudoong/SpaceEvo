#! /bin/bash

TASK=$1

function eval_supernet() {
    # eval min & max subnet fp32 accuracy
    python -m torch.distributed.launch --nproc_per_node=2 eval_supernet.py --superspace onnx --supernet_choice 111211 --resume results/teamdrive/supernet_training --batch_size_per_gpu 32
    # eval min & max subnet int8 accuracy
    python -m torch.distributed.launch --nproc_per_node=2 eval_supernet.py --superspace onnx --supernet_choice 111211 --resume results/teamdrive/supernet_training --batch_size_per_gpu 32 --quant_mode
    # eval subnets fp32 accuracy
    python -m torch.distributed.launch --nproc_per_node=2 eval_supernet.py --superspace onnx --supernet_choice 111211 --resume results/teamdrive/supernet_training --batch_size_per_gpu 32 --subnet_choice d1#1#2#2#2#2#2#2_k3#3#3#3#3#3#3#5#3#3#3#3#5#5_w16#16#32#32#64#48#96#80#128#128#208#192#352#384_e0#0.5#4.0#4.0#4.0#4.0#6.0#6.0#4.0#4.0#6.0#4.0#4.0#4.0_r192 d1#1#2#2#2#2#4#2_k3#3#5#3#3#3#3#5#3#3#3#3#3#3#5#3_w16#16#32#32#64#48#80#80#144#144#192#224#256#208#352#432_e0#0.5#4.0#4.0#4.0#6.0#8.0#6.0#8.0#4.0#4.0#6.0#6.0#4.0#8.0#6.0_r192
}

function eval_subnet() {
    # fp32 accuracy
    python -m torch.distributed.launch --nproc_per_node=2 eval_supernet.py --superspace mobile_v1_res --supernet_choice 032220 --resume results/teamdrive/supernet_training --subnet_choice  d1#1#2#3#2#3#5#2_k3#3#7#3#3#3#3#7#3#5#5#3#7#5#5#7#5#5#3_w32#16#32#24#48#48#56#88#96#120#96#128#256#256#256#192#192#400#400_e0#1.0#0#0#1.0#1.0#1.0#3.0#6.0#6.0#3.0#3.0#8.0#8.0#8.0#3.0#3.0#0#0_r192

    # int8 accuracy
    python -m torch.distributed.launch --nproc_per_node=2 eval_supernet.py --superspace mobile_v1_res --supernet_choice 032220 --resume results/teamdrive/supernet_training --subnet_choice  d1#1#2#3#2#3#5#2_k3#3#7#3#3#3#3#7#3#5#5#3#7#5#5#7#5#5#3_w32#16#32#24#48#48#56#88#96#120#96#128#256#256#256#192#192#400#400_e0#1.0#0#0#1.0#1.0#1.0#3.0#6.0#6.0#3.0#3.0#8.0#8.0#8.0#3.0#3.0#0#0_r192 --quant_mode

}
function benchmark_onnx_subnet() {
    echo 'please write input to onnx_tools/input.csv'
    python onnx_tools/export_onnx.py --skip_weights
    taskset $CPU39_MASK python onnx_tools/benchmark.py
}

function benchmark_tflite_subnet() {
    echo 'please write input to tflite_tools/input.csv'
    python tflite_tools/benchmark.py
}

$TASK ""