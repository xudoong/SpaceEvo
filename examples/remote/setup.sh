#### install requirements
pip install -r requirements.txt


#### setup nn-meter
echo "*** Setting Up nn-meter ***"
nn_meter_remote_dir=/mnt/data/EdgeDL/quantized_nas/nn-meter-teamdrive
nn_meter_local_dir=${AMLT_CODE_DIR}/nn-meter-teamdrive

cp -r ${nn_meter_remote_dir} ${AMLT_CODE_DIR}
pip install -e ${nn_meter_local_dir}/nn-Meter

# For sing 
/home/aiscuser/.local/bin/nn-meter register --predictor ${nn_meter_local_dir}/tflite_int8_predictor/meta.yaml
# For itp
nn-meter register --predictor ${nn_meter_local_dir}/tflite_int8_predictor/meta.yaml
echo "***      Done           ***"


#### copy imagenet to local directory
echo "Copying Imagenet Dataset"
date
mkdir /tmp/code/dataset
cp -r /mnt/data/EdgeDL/imagenet2012/tar /tmp/code/dataset
tar -xvf dataset/tar/ILSVRC2012_img_val.tar -C dataset > /dev/null
tar -xvf dataset/tar/ILSVRC2012_img_train.tar -C dataset > /dev/null
rm -r /tmp/code/dataset/tar
date 
echo "Done"
