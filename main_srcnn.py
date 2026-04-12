#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path

from omegaconf import OmegaConf

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

project_root = Path(__file__).parent.resolve()
sys.path.append(str(project_root))

from trainer_srcnn import TrainerSRCNN
from inference_srcnn import run_inference_srcnn


def _find_latest_run_dir(save_dir):
    save_dir = Path(save_dir)
    if not save_dir.exists():
        return None
    run_dirs = [d for d in save_dir.iterdir() if d.is_dir()]
    if not run_dirs:
        return None
    return max(run_dirs, key=os.path.getmtime)


def _select_best_ckpt(save_dir, preferred_run_dir=None):
    """Select checkpoint with priority: model_best.pth > latest model_*.pth."""
    run_dir = preferred_run_dir or _find_latest_run_dir(save_dir)
    if run_dir is None:
        return None, None

    ckpt_dir = Path(run_dir) / "ckpts"
    if not ckpt_dir.exists():
        return None, run_dir

    best_ckpt = ckpt_dir / "model_best.pth"
    if best_ckpt.exists():
        return str(best_ckpt), run_dir

    ckpt_files = list(ckpt_dir.glob("model_*.pth"))
    if not ckpt_files:
        return None, run_dir

    try:
        latest_ckpt = max(ckpt_files, key=lambda p: int(p.stem.split("_")[-1]))
    except Exception:
        latest_ckpt = max(ckpt_files, key=os.path.getmtime)
    return str(latest_ckpt), run_dir


def _default_test_input_dir(configs):
    """Infer test root dir that contains LR/HR from config.data.test.params.lr_dir."""
    try:
        lr_dir = Path(configs.data.test.params.lr_dir)
        return str(lr_dir.parent)
    except Exception:
        return None


def _default_test_output_dir(configs, run_dir, ckpt_path):
    base = Path(run_dir) if run_dir is not None else Path(configs.train.save_dir)
    ckpt_name = Path(ckpt_path).stem if ckpt_path else "auto"
    return str(base / "test_outputs" / f"srcnn_{ckpt_name}")


def _resolve_resume_ckpt(args, configs):
    """Resolve checkpoint path for resuming SRCNN training."""
    resume_path = None

    if args.ckpt_path:
        resume_path = args.ckpt_path
        print(f"[main_srcnn] Resume from --ckpt_path: {resume_path}")
        return resume_path

    if not args.resume:
        return None

    exp_dir = Path(configs.train.save_dir)
    if not exp_dir.exists() or not any(exp_dir.iterdir()):
        print(f"[main_srcnn] --resume set but {exp_dir} is empty/missing. Start from scratch.")
        return None

    try:
        latest_run_dir = max([d for d in exp_dir.iterdir() if d.is_dir()], key=os.path.getmtime)
    except ValueError:
        print(f"[main_srcnn] --resume set but no run dirs in {exp_dir}. Start from scratch.")
        return None

    ckpt_dir = latest_run_dir / "ckpts"
    if not ckpt_dir.exists():
        print(f"[main_srcnn] --resume set but ckpt dir missing: {ckpt_dir}. Start from scratch.")
        return None

    ckpt_files = list(ckpt_dir.glob("model_*.pth"))
    if not ckpt_files:
        print(f"[main_srcnn] --resume set but no model_*.pth in {ckpt_dir}. Start from scratch.")
        return None

    try:
        latest_ckpt = max(ckpt_files, key=lambda p: int(p.stem.split("_")[-1]))
    except Exception:
        latest_ckpt = max(ckpt_files, key=os.path.getmtime)

    resume_path = str(latest_ckpt)
    print(f"[main_srcnn] Auto-resume checkpoint: {resume_path}")
    return resume_path


def run_training(args, configs):
    print("\n" + "=" * 80)
    print("[main_srcnn] SRCNN Training Mode")
    print("=" * 80)

    resume_path = _resolve_resume_ckpt(args, configs)

    # Keep both fields for compatibility with different trainer versions.
    configs.resume_path = resume_path
    configs.resume = resume_path

    trainer = TrainerSRCNN(configs)
    trainer.train()


def run_testing(args, configs):
    print("\n" + "=" * 80)
    print("[main_srcnn] SRCNN Test/Inference Mode")
    print("=" * 80)

    latest_run_dir = _find_latest_run_dir(configs.train.save_dir)
    ckpt_path = args.ckpt_path
    if not ckpt_path:
        ckpt_path, latest_run_dir = _select_best_ckpt(configs.train.save_dir, preferred_run_dir=latest_run_dir)
        if ckpt_path:
            print(f"[main_srcnn] Auto-selected checkpoint: {ckpt_path}")

    # 没有可用权重时，自动先训练/续训，再进行测试
    if not ckpt_path:
        print("[main_srcnn] No checkpoint found. Start training/resume automatically...")
        args.resume = True
        run_training(args, configs)
        ckpt_path, latest_run_dir = _select_best_ckpt(configs.train.save_dir)
        if not ckpt_path:
            print("[main_srcnn] ERROR: training finished but still no checkpoint found.")
            sys.exit(1)
        print(f"[main_srcnn] Use newly generated checkpoint: {ckpt_path}")

    input_dir = args.input_dir or _default_test_input_dir(configs)
    if not input_dir:
        print("[main_srcnn] ERROR: cannot infer test input dir. Please pass --input_dir.")
        sys.exit(1)

    output_dir = args.output_dir or _default_test_output_dir(configs, latest_run_dir, ckpt_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 给 inference_srcnn 传递最终解析后的参数
    args.ckpt_path = ckpt_path
    args.input_dir = input_dir
    args.output_dir = output_dir

    print(f"[main_srcnn] Test input_dir: {args.input_dir}")
    print(f"[main_srcnn] Test output_dir: {args.output_dir}")

    run_inference_srcnn(args, configs)


def run_train_test(args, configs):
    """One-shot mode: train (or resume) then test with auto-selected best checkpoint."""
    run_training(args, configs)
    run_testing(args, configs)


def main():
    parser = argparse.ArgumentParser(description="Dedicated SRCNN entrypoint")
    parser.add_argument("--cfg_path", type=str, default="configs/config_srcnn.yaml", help="Path to config file")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test", "train_test"], help="Execution mode")
    parser.add_argument("--save_dir", type=str, default=None, help="Override configs.train.save_dir")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint under save_dir")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Checkpoint for resume/test")
    parser.add_argument("--input_dir", type=str, default=None, help="[test] input data dir (contains LR/HR)")
    parser.add_argument("--output_dir", type=str, default=None, help="[test] output directory")

    args = parser.parse_args()
    configs = OmegaConf.load(args.cfg_path)

    if args.save_dir:
        configs.train.save_dir = args.save_dir

    model_target = str(configs.model.target).lower()
    if "srcnn" not in model_target:
        print(f"[main_srcnn] WARNING: model target may not be SRCNN: {configs.model.target}")

    if args.mode == "train":
        run_training(args, configs)
    elif args.mode == "test":
        run_testing(args, configs)
    else:
        run_train_test(args, configs)


if __name__ == "__main__":
    main()
