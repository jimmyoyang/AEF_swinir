python inference.py \
    --config configs/config_swinir.yaml \
    --ckpt /mnt/lm_data_afs/wangzining/charles/AEF_swinir/training_logs/integration_tests/integration_test/2026-01-27_12-42-29/ckpts/model_best.pth \
    --input ./data/processed_data/test \
    --output ./inference_results/test_set_predictions
