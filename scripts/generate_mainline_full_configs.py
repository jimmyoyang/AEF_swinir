#!/usr/bin/env python3
"""
Generate full-only mainline post-ablation configs.

Goal:
- Keep legacy v2/v3 configs untouched
- Create new full configs that use 4c or 4d as the base backbone/loss path
- Preserve postprocessor variants (scheme1/3/6, plus6, stage, gated)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

V2_SOURCES = [
    "configs/ablation_v2/ablation_v2_5a_nopost_bestpre_full.yaml",
    "configs/ablation_v2/ablation_v2_5b_scheme1_gumbel_full.yaml",
    "configs/ablation_v2/ablation_v2_5c_scheme3_dual_branch_full.yaml",
    "configs/ablation_v2/ablation_v2_5d_scheme6_matrix_full.yaml",
]

V3_SOURCES = [
    "configs/ablation_v3/ablation_v3_5a_nopost_bestpre_full.yaml",
    "configs/ablation_v3/ablation_v3_5e_scheme1_plus6_full.yaml",
    "configs/ablation_v3/ablation_v3_5f_scheme3_plus6_full.yaml",
    "configs/ablation_v3/ablation_v3_5e_scheme1_plus6_stage_full.yaml",
    "configs/ablation_v3/ablation_v3_5f_scheme3_plus6_stage_full.yaml",
    "configs/ablation_v3/ablation_v3_5e_scheme1_plus6_gated_lightweight_full.yaml",
    "configs/ablation_v3/ablation_v3_5f_scheme3_plus6_gated_lightweight_full.yaml",
    # stage_gated only has quick in legacy; derive a full variant from quick templates.
    "configs/ablation_v3/ablation_v3_5e_scheme1_plus6_stage_gated_quick.yaml",
    "configs/ablation_v3/ablation_v3_5f_scheme3_plus6_stage_gated_quick.yaml",
]


def _replace_once(text: str, pattern: str, repl: str) -> str:
    out, n = re.subn(pattern, repl, text, count=1, flags=re.MULTILINE)
    if n == 0:
        raise ValueError(f"Pattern not found: {pattern}")
    return out


def _replace_all(text: str, pattern: str, repl: str) -> str:
    return re.sub(pattern, repl, text, flags=re.MULTILINE)


def _ensure_train_key(text: str, key: str, value: str) -> str:
    line_re = re.compile(rf"^\s*{re.escape(key)}\s*:\s*.*$", re.MULTILINE)
    if line_re.search(text):
        return line_re.sub(f"  {key}: {value}", text)

    anchor = re.search(r"^\s*local_logging\s*:\s*.*$", text, flags=re.MULTILINE)
    if anchor:
        idx = anchor.end()
        return text[:idx] + f"\n  {key}: {value}" + text[idx:]

    train_block = re.search(r"^train:\n", text, flags=re.MULTILINE)
    if not train_block:
        raise ValueError("Could not find train block")
    insert_at = train_block.end()
    return text[:insert_at] + f"  {key}: {value}\n" + text[insert_at:]


def _make_output_name(base: str, src_name: str) -> str:
    name = src_name.replace("ablation_v2_", f"ablation_{base}_v2_")
    name = name.replace("ablation_v3_", f"ablation_{base}_v3_")
    name = name.replace("_quick.yaml", "_full.yaml")
    return name


def _transform(text: str, base: str, exp_bucket: str, out_name: str) -> str:
    if base not in {"4c", "4d"}:
        raise ValueError(f"Unsupported base: {base}")

    # Keep postprocessor namespace compatibility:
    # - v2 configs use post_v2 type names (e.g., gumbel_routing_v2)
    # - v3 configs use composed type names (e.g., scheme1_plus6)
    # so model target must remain the matching wrapper.
    if exp_bucket == "v2":
        model_target = "models.network_swinir_post_v2.SwinIRPostV2"
    elif exp_bucket == "v3":
        model_target = "models.network_swinir_post_v3.SwinIRPostV3"
    else:
        model_target = "models.network_swinir.SwinIR"
    text = _replace_once(text, r"^model:\n\s*target:\s*.*$", f"model:\n  target: {model_target}")

    # keep mainline representation settings
    text = _replace_all(text, r"^\s*use_pos_emb\s*:\s*.*$", "    use_pos_emb: false")
    text = _replace_all(text, r"^\s*use_learnable_pos_emb\s*:\s*.*$", "    use_learnable_pos_emb: false")

    # trainer path
    if base == "4c":
        trainer_target = "trainer_mask_ablation.TrainerAlphaSRMaskAblation"
    else:
        trainer_target = "trainer_mask_ablation.TrainerAlphaSRMaskTemporalLossAblation"

    text = _replace_once(text, r"^trainer:\n\s*target:\s*.*$", f"trainer:\n  target: {trainer_target}")

    # force full profile
    text = _replace_all(text, r"^(\s*)iterations\s*:\s*\d+\s*$", r"\1iterations: 5000")
    text = _replace_all(text, r"^(\s*)num_workers\s*:\s*\d+\s*$", r"\1num_workers: 4")
    text = _replace_all(text, r"^(\s*)log_freq\s*:\s*\[.*\]\s*$", r"\1log_freq: [100, 200, 200]")
    text = _replace_all(text, r"^(\s*)save_freq\s*:\s*\d+\s*$", r"\1save_freq: 1000")
    text = _replace_all(text, r"^(\s*)val_freq\s*:\s*\d+\s*$", r"\1val_freq: 50")

    # stage_gated quick -> full-like schedule
    text = _replace_all(text, r"^(\s*)tau_steps\s*:\s*\d+\s*$", r"\1tau_steps: 5000")
    text = _replace_all(text, r"^(\s*)tau_end\s*:\s*[0-9.]+\s*$", r"\1tau_end: 0.5")
    text = _replace_all(text, r"^(\s*)warmup_steps\s*:\s*\d+\s*$", r"\1warmup_steps: 1200")
    text = _replace_all(text, r"^(\s*)ramp_steps\s*:\s*\d+\s*$", r"\1ramp_steps: 600")

    # train save_dir
    out_stem = out_name.replace(".yaml", "")
    save_dir = f"./training_logs/experiments_mainline_{base}_{exp_bucket}/{out_stem}"
    text = _replace_all(text, r"^(\s*)save_dir\s*:\s*.*$", f"\\1save_dir: \"{save_dir}\"")

    # base-specific train controls
    if base == "4c":
        text = _ensure_train_key(text, "indicating_mask_reduce", '"prob_or"')
    else:
        text = _ensure_train_key(text, "temporal_loss_reduce", '"weighted_mean"')

    return text


def _emit_set(base: str, exp_bucket: str, sources: list[str], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = []
    for src_rel in sources:
        src = ROOT / src_rel
        src_name = src.name
        out_name = _make_output_name(base, src_name)
        out = out_dir / out_name

        raw = src.read_text(encoding="utf-8")
        new_text = _transform(raw, base=base, exp_bucket=exp_bucket, out_name=out_name)
        out.write_text(new_text, encoding="utf-8")
        generated.append(out)
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate mainline full ablation configs")
    parser.add_argument("--base", choices=["4c", "4d", "both"], default="both")
    args = parser.parse_args()

    all_generated: list[Path] = []
    bases = ["4c", "4d"] if args.base == "both" else [args.base]

    for base in bases:
        all_generated += _emit_set(
            base=base,
            exp_bucket="v2",
            sources=V2_SOURCES,
            out_dir=ROOT / f"configs/ablation_mainline_{base}_v2",
        )
        all_generated += _emit_set(
            base=base,
            exp_bucket="v3",
            sources=V3_SOURCES,
            out_dir=ROOT / f"configs/ablation_mainline_{base}_v3",
        )

    print(f"Generated {len(all_generated)} config files")
    for p in sorted(all_generated):
        print(p.relative_to(ROOT))


if __name__ == "__main__":
    main()
