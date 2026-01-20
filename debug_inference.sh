python inference.py \
    --config configs/config_swinir.yaml \
    --ckpt training_logs/your_run/model_best.pth \
    --input ./debug_sample \
    --output ./inference_results/single_sample_debug
