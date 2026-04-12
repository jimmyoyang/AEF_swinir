#!/usr/bin/env python
# -*- coding:utf-8 -*-

from models.post_v2.asymmetric_dual_branch import AsymmetricDualBranchPostProcessorV2
from models.post_v2.gumbel import GumbelRoutingPostProcessorV2
from models.post_v2.matrix_reshape_soef import MatrixReshapeSOEFPostProcessorV2
from models.post_v2.noop import NoOpPostProcessor
from models.post_v3.composed import SequentialComposedPostProcessorV3


def _build_scheme1_plus6(params):
    params = params or {}
    p1 = dict(params.get("scheme1", {}))
    p6 = dict(params.get("scheme6", {}))
    compose = dict(params.get("compose", {}))

    first = GumbelRoutingPostProcessorV2(**p1)
    second = MatrixReshapeSOEFPostProcessorV2(**p6)
    return SequentialComposedPostProcessorV3(first_module=first, second_module=second, **compose)


def _build_scheme3_plus6(params):
    params = params or {}
    p3 = dict(params.get("scheme3", {}))
    p6 = dict(params.get("scheme6", {}))
    compose = dict(params.get("compose", {}))

    first = AsymmetricDualBranchPostProcessorV2(**p3)
    second = MatrixReshapeSOEFPostProcessorV2(**p6)
    return SequentialComposedPostProcessorV3(first_module=first, second_module=second, **compose)


def build_postprocessor_v3(pp_type, pp_params):
    pp_type = (pp_type or "").lower()
    pp_params = pp_params or {}

    if pp_type in ["none", "noop", "identity"]:
        return NoOpPostProcessor(**pp_params)
    if pp_type in ["scheme1_plus6", "1plus6", "gumbel_plus_soef_v3"]:
        return _build_scheme1_plus6(pp_params)
    if pp_type in ["scheme3_plus6", "3plus6", "adbr_plus_soef_v3"]:
        return _build_scheme3_plus6(pp_params)

    raise ValueError(f"Unknown postprocessor v3 type: {pp_type}")
