#!/usr/bin/env python
# -*- coding:utf-8 -*-

from models.network_swinir import SwinIR as SwinIRBase
from models.post_v3 import build_postprocessor_v3, ContextAwarePostprocessorWrapper


class SwinIRPostV3(SwinIRBase):
    """Parallel v3 model wrapper.

    Reuses original SwinIR body and attaches composed postprocessor (1+6 / 3+6).
    """

    def __init__(
        self,
        use_spectral_postprocessor=False,
        spectral_postprocessor_type="noop",
        spectral_postprocessor_params=None,
        **kwargs,
    ):
        super().__init__(
            use_spectral_postprocessor=False,
            spectral_postprocessor_type=spectral_postprocessor_type,
            spectral_postprocessor_params=spectral_postprocessor_params,
            **kwargs,
        )

        self.use_spectral_postprocessor = use_spectral_postprocessor
        self.spectral_postprocessor = None
        if self.use_spectral_postprocessor:
            pp_params = spectral_postprocessor_params or {}
            if "n_channels" not in pp_params:
                pp_params["n_channels"] = kwargs.get("out_channels", 64)
            post_v3 = build_postprocessor_v3(spectral_postprocessor_type, pp_params)
            self.spectral_postprocessor = ContextAwarePostprocessorWrapper(post_v3)

    def set_runtime_context(self, context):
        if self.spectral_postprocessor is not None:
            self.spectral_postprocessor.set_runtime_context(context)

    def get_postprocessor_stats(self):
        if self.spectral_postprocessor is None:
            return {}
        return self.spectral_postprocessor.get_last_stats()

    def get_postprocessor_extra_losses(self):
        if self.spectral_postprocessor is None:
            return {}
        return self.spectral_postprocessor.get_last_extra_losses()
