#!/usr/bin/env python
# -*- coding:utf-8 -*-

import csv

from trainer_post_v2_mask_ablation import TrainerAlphaSRPostV2MaskAblation


class TrainerAlphaSRPostV3MaskAblation(TrainerAlphaSRPostV2MaskAblation):
    """V3 trainer with best-preprocessing mask reduction (prob_or).

    Supports dynamic postprocessor stat fields in csv logging.
    """

    def __init__(self, configs):
        super().__init__(configs)
        self._post_rows = []
        self._post_fields = ["iter", "stage", "post_tau", "routing_entropy", "active_groups"]

    def _collect_post_stats(self, stage):
        if self.rank != 0:
            return
        model = self._unwrap_model()
        if not hasattr(model, "get_postprocessor_stats"):
            return

        stats = model.get_postprocessor_stats() or {}
        if not stats:
            return

        row = {
            "iter": int(getattr(self, "current_iters", 0)),
            "stage": stage,
        }
        for k, v in stats.items():
            if isinstance(v, (int, float)):
                row[k] = float(v)

        for key in row.keys():
            if key not in self._post_fields:
                self._post_fields.append(key)

        self._post_rows.append(row)
        with open(self._post_stats_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._post_fields)
            writer.writeheader()
            writer.writerows(self._post_rows)


class TrainerAlphaSRPostV3StageMaskAblation(TrainerAlphaSRPostV3MaskAblation):
    """V3 staged-training trainer.

    Training behavior is controlled by model postprocessor staged settings.
    """

    pass
