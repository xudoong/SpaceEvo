## Usage
The whole pipeline contains the following procedure: 
1. search space search based on block_kd
   1. train and lsq+ block, see *train_block_kd.py* and *lsq_block_kd.py*
   2. build block lut, see *eval_block_kd_v2.py*
   3. search, see *search_block_kd_v2.py*
2. supernet training and lsq+, see *train_supernet.py*
3. subnet search, see *search_subnet.py*
4. eval supernet and subnet (latency and accuracy)
   1. eval accuracy and predict latency, see *eval_supernet.py*
   2. benchmark latency on real device: see *onnx_tools/* and *tflite_tools/*

Assume all of this project's checkpoints and results are stored in ${ROOT_DIR}. (In my experiment, most jobs run in itp/sing, and the results are stored in teamdrive. `ROOT_DIR=EdgeDL/quantized_nas`.) Directory layout will be
```
${ROOT_DIR}
|   block_kd/teacher_checkpoint/efficientnet_b5/checkpoint.pth
|
└---block_kd/lut
|   |   mobilew/*.csv
|   |    onnxw/*.csv
|   |    <other hyperspaces>/*.csv
|
└---block_kd/mobilew # store block_kd checkpoints of mobilew
|   |   stage1_0/
|   |   .../
|   |   stage6_6/
|
└---block_kd/onnxw $ store block_kd checkpoints of onnxw
|   |
|
└---block_kd/search # search block_kd results # (search runs locally on M20. this directory does not exist on teamdrive)
|   |   mobilew/
|   |   onnxw/
|
└---supernet_training
|   |   mobilew-312120-155501-align0
|   |       |   checkpoint.pth
|   |       |   lsq.pth
|   |       |   ...
|   |   onnxw_-121121-023230-align0
|
└---search_subnet
    |   onnx-111211/*.log
    |   ...
```
### search space search

#### train block kd
First you need to train and LSQ+ QAT all blocks in the hyperspace (superspace). Blocks are independent, thus can be trained in parallel. You can specify the block id in argument `--stage_list`.
```
# first train block in fp32 mode
python -m torch.distributed.launch --nproc_per_node=4 train_block_kd.py \
--superspace <superspace> \
--output_path ${ROOT_DIR}/block_kd/<superspace> \
--inplace_distill_from_teacher \
--num_epochs 5 \
--stage_list stage1_0 stage1_2
--hw_list 160 192 224 \
--dataset_path ./dataset

# then lsq+ in quant mode
python -m torch.distributed.launch --nproc_per_node=4 lsq_block_kd.py \
--superspace <superspace> \
--output_path ${ROOT_DIR}/block_kd/<superspace> \
--inplace_distill_from_teacher \
--num_epochs 1 \
--stage_list stage1_0 stage1_2 \
--hw_list 160 192 224 \
--dataset_path ./dataset \
--teacher_checkpoint_path ${ROOT_DIR}/block_kd/teacher_checkpoint/efficientnet_b5/checkpoint.pth \
--train_batch_size 32 \
--learning_rate_list 0.00125 0.00125 0.00125 0.00125 0.00125 0.00125
```
In the above example, we train 2 blocks: stage1_0 and stage1_2. `<superspace>`can be chosen in [mobilew | onnxw | ...].

#### build block lut
```
python eval_block_kd_v2.py \
--superspace [mobilew|onnxw] \
--platform [tflite27_cpu_int8|onnx_lut] \
--output_path ${ROOT_DIR}/block_kd/lut \
--stage_list stage3_0 stage3_1 \
--width_window_filter 0 1 2 \
--hw_list 160 192 224 \
--dataset_path ./dataset \
--checkpoint_path ${ROOT_DIR}/block_kd \
--teacher_checkpoint_path ${ROOT_DIR}/block_kd/teacher_checkpoint/efficientnet_b5/checkpoint.pth \
--debug
```
`--stage_list` and `--width_window_filter` specify the blocks and the width window candidates to build lut. The above scripts will build 6 luts: stage3_0_0, stage3_0_1, stage3_0_2, stage3_1_0, stage3_1_1, stage3_1_2. When `--debug` argment is set, evalution is performed only on 10 batches. We found after a few batches, the loss becomes stable, so we set `--debug` flag when building block lut to speed up this process.

Each line in the output lut csv file represents a stage sampled from the dynamic stage. There are 6 items in a line, whose meanings are
| sub-stage-config | input shape | nsr-loss | FLOPS(M) | Params(M) | pred int8 latency(ms) |
|---------------|-------------|----------|----------|-----------|-----------------------|
|5#32#8_3#40#3|1x24x112x112|0.2734|118.7395|0.0489|11.7578|

If a sub-stage has 2 blocks (depth=2) and each block has (kernel_size, width, expand_ratio) = (ki, wi, ei), then it can be encoded as *k1#w1#e1_k2#w2_e2*. 

The built LUT stores in `data/block_lut`.

If nn-meter gets update, then the latency term in lut needs update accordingly. You can do this by running `replace_block_lut_latency.py`
```
# first move the original <lut-dir> to <lut-dir>.bak
mv data/block_lut/mobilew data/block_lut/mobilew.bak

# re-predict latency in parallel.
python replace_block_lut_latency.py --superspace mobilew --pool_size 20

# remove the old lut
rm -r data/block_lut/mobilew.bak
``` 
#### search space search
Search space search is vary fast because no netural network forward is needed. Subnets are sampled from the previously built lut. Thus search space search runs locally. All other training and searching processes run in the cluster.
```
# search on hyperspace mobilew with latency constraint {15 20}, latency_loss_t 0.08 and latency_loss_a 0.01
python search_block_kd_v2.py --superspace mobilew --latency_constraint 15 20 --platform tflite27_cpu_int8 --latency_loss_t 0.08 --latency_loss_a 0.01 --lut_path ${ROOT_DIR}/block_kd/lut --output_dir ${ROOT_DIR}/block_kd/search

# search on hyperspace onnxw with latency constraint 10
python search_block_kd_v2.py --superspace onnxw --latency_constraint 10 --platform onnx_lut --latency_loss_t 0.08 --latency_loss_a 0.01 --lut_path ${ROOT_DIR}/block_kd/lut --output_dir ${ROOT_DIR}/block_kd/search
```
You can also get the quality score of specific search spaces.
```
python search_block_kd_v2.py --superspace onnxw --latency_constraint 15 --platform onnx_lut --latency_loss_t 0.08 --supernet_choices 111111_000000 222222_000000
```

### supernet training
```
# first train 360 epochs in fp32 mode
python -m torch.distributed.launch --nproc_per_node 8 train_supernet.py \
--config-file supernet_training_configs/train.yaml \
--superspace mobilew \
--supernet_choice 123214-012321 \
--batch_size_per_gpu 64 \
--resume
# then lsq+ for 50 epochs
python -m torch.distributed.launch --nproc_per_node 8 train_supernet.py \
--config-file supernet_training_configs/train.yaml \
--superspace mobilew \
--supernet_choice 123214-012321 \
--batch_size_per_gpu 32 \
--quant_mode
```
A supernet can be encoded as `<hyperspace>-<block_type_choices>` (search space v1) or `<hyperspace>-<block_type_choices>-<width_window_choices>` (search space v2), e.g., mobile_v1_res-032220 (v1) and mobilew-111211-123211 (v2). The above script trains and QAT supernet mobilew-123214-012321. `Supernet.build_from_str` method builds a supernet torch model from an encoding.

### search target subnet
```
python -m torch.distributed.launch --nproc_per_node 4 search_subnet.py \
--superspace onnxw \
--supernet_choice 121122-133333 \
--dataset_path ./dataset \
--output_path ${ROOT_DIR}/search_subnet \
--checkpoint_path ${ROOT_DIR}/supernet_training \
--latency_constraint 15 \
--latency_delta 2 \
--batch_size 32 \
--num_calib_batches 20
```
Before searching, make sure the supernet's checkpoint after qat exists. In the above example, the target checkpoint path is `${ROOT_DIR}/supernet_training/onnxw-121122-133333-align0/lsq.pth`. Also the code needs nn-meter installed and registered.
The valid latency range is specified by `--latency_constraint c` and `--latency_delta d`. The range is [c-d, c].  

### evaluate subnet and predict latency
A subnet can be encoded as the depth, width, kernel_size, expand_ratio and resolution choices from the supernet, e.g.,  d1#1#2#4#5#4#6#2_k3#3#5#5#3#5#5#3#5#3#3#3#5#3#3#5#5#5#5#3#3#5#5#5#5_w32#32#32#48#48#48#64#64#96#112#96#112#80#144#144#144#128#240#256#256#256#256#256#432#432_e0#0.5#8.0#6.0#8.0#6.0#8.0#4.0#6.0#6.0#4.0#4.0#4.0#6.0#8.0#4.0#6.0#6.0#6.0#6.0#8.0#8.0#4.0#8.0#8.0_r224.
```
# evalute the fp32 accuracy of a list of subnets in a supernet
python eval_supernet.py --superspace onnxw --supernet_choice 121122-133333 --mode acc --resume ${ROOT_DIR}/supernet_training --dataset_dir ./dataset --subnet_choice <subnet_choice1> <subnet_choice2> <...> 

# evalute the int8 accuracy of a list of subnets in a supernet
python eval_supernet.py --superspace onnxw --supernet_choice 121122-133333 --mode acc --resume ${ROOT_DIR}/supernet_training --dataset_dir ./dataset --subnet_choice <subnet_choice1> <subnet_choice2> <...> --quant_mode

# also you can run this code with torch ddp to speed up evaluation: python -m torch.distributed.launch --nproc_per_node 4 eval_supernet.py ...

# predict the latency of a list of subnets
python eval_supernet.py --superspace onnxw --supernet_choice 121122-133333 --mode lat --subnet_choice <subnet_choice1> <subnet_choice2> <...> 
```

### benchmark subnet latency
```
##### benchmark onnx latency #####
# 0. login to srgws08
ssh lzhani@10.150.242.126 (passwd: 123456)
conda activate benchmark
cd /data/v-xudongwang/benchmark_tools/experiments/D0323_evolve_space

# 1. write subnet encoding to onnx_tools/input.csv. Each line represents a subnet: <superspace>,<supernet_choice>,<subnet_choice>

# 2. export onnx
python onnx_tools/export_onnx.py --skip_weights

# 3. benchmark
python onnx_tools/benchmark.py


##### benchmark tflite latency #####
# 0. login in to 10.172.141.20
ssh v-xudongwang@10.172.141.20 (passwd: v-xudongwang)
conda activate benchmark
cd /data/v-xudongwang/benchmark_tools/experiments/D0323_evolve_space

# 1. write subnet encoding to tflite_tools/input.csv. Each line represents a subnet: <superspace>,<supernet_choice>,<subnet_choice>

# 2. export and benchmark
python tflite_tools/benchmark.py

```
## Examples
*examples/local* direcory constrains example scripts to run the jobs locally. *examples/remote* directory contains python codes to submit jobs to itp/sing (these files are based on *itp/\*.py*, by which I submit jobs).

## LSQ+ Implementation
LSQ+ quantization is implemented in *modules/modeling/ops/lsq_plus.py*. The main components are
* function `quantize_activation(activation, scale, num_bits, beta, is_training)` fake quantize (quantize and then de-quantize) the activation using parameter *scale* and *beta*.
* function `quantize_weight(weight, scale, num_bits, is_training)` fake quantize the weight using *scale* (no offset parameter is needed because the weight quantization is symmetric).
* class `QBase` is the base class for a quantized OP. It initialize three parameters *activation_scale*, *activation_beta*, *weight_scale* and provides quantization parameters initial methods and fake quantize method. There are two initial methods: min_max_initial and lsq_initial. We use the first one, which is simple.
* function `set_quant_mode(model: nn.Module)` sets a torch model with lsq+ op to int8 mode.

A quantized op can be implemented by inheriting `QBase` and `nn.Module`, see *modules/modeling.ops/op.py*, which contains `QConv` and `QLinear`.

Because the normal training flow is first training in fp32 mode and then qat in int8 mode, models are initialized into fp32 mode, by setting the `nbits_w` and `nbits_a` attributes in `QBase` to 32. In forward pass, lsq+ op in fp32 mode behaves the same as the normal torch module. To change a model to int8 mode, all `nbits_w` and `nbits_a` attributes are needed to change to 8, which can be done by function `set_quant_mode`.

## About nn-meter
We use nn-meter to predict the int8 latency when building block lut or searching subnets. nn-meter is installed in conda environment *sing* in M20. The repo locates in `/data/data0/v-xudongwang/nn-meter-working/nn-Meter`. For itp/sing jobs, they will first install nn-meter in teamdrive `EdgeDL/quantized_nas/nn-meter-teamdrive/nn-Meter`. 

So please make sure these two nn-meter package stay the same. 

An inconvenient thing is that the `package_location` term in `tflite_int8_predictor/meta.yaml` needs to point to the exact package path, which is not equal between M20 and itp/sing. So I first copy `/data/data0/v-xudongwang/nn-meter-working` to `/data/data0/v-xudongwang/nn-meter-teamdrive`, change the `package_location` term and then upload it to teamdrive.