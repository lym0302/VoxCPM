data_name=$1
#log_file=logs/train_${data_name}.log

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python data_process/change_yaml.py --data_name $data_name --output conf/voxcpm_v2/voxcpm_finetune_all_hindi.yaml

cat conf/voxcpm_v2/voxcpm_finetune_all_hindi.yaml

echo "Start training with data: $data_name"
python scripts/train_voxcpm_finetune.py --config_path conf/voxcpm_v2/voxcpm_finetune_all_hindi.yaml
