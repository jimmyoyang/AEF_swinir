export CUDA_VISIBLE_DEVICES=${1:-0}
python main.py --cfg_path configs/config_swinir.yaml