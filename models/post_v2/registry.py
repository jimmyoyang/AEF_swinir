#!/usr/bin/env python
# -*- coding:utf-8 -*-

from models.post_v2.gumbel import GumbelRoutingPostProcessorV2
from models.post_v2.asymmetric_dual_branch import AsymmetricDualBranchPostProcessorV2
from models.post_v2.matrix_reshape_soef import MatrixReshapeSOEFPostProcessorV2
from models.post_v2.noop import NoOpPostProcessor


def build_postprocessor_v2(pp_type, pp_params):
    pp_type = (pp_type or "").lower()
    pp_params = pp_params or {}

    if pp_type in ["none", "noop", "identity"]:
        return NoOpPostProcessor(**pp_params)
    if pp_type in ["scheme1", "gumbel", "gumbel_routing", "gumbel_routing_v2"]:
        return GumbelRoutingPostProcessorV2(**pp_params)
    if pp_type in ["scheme3", "asymmetric_dual_branch", "adbr", "asymmetric_dual_branch_v2"]:
        return AsymmetricDualBranchPostProcessorV2(**pp_params)
    if pp_type in ["scheme6", "matrix_reshape", "soef", "matrix_reshape_soef_v2"]:
        return MatrixReshapeSOEFPostProcessorV2(**pp_params)
    raise ValueError(f"Unknown postprocessor v2 type: {pp_type}")
