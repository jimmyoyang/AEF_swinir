python inference.py \
    --config configs/config_swinir.yaml \
    --ckpt training_logs/your_run/model_best.pth \
    --input ./data/processed_data/test \
    --output ./inference_results/test_set_predictions
