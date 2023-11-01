// Copyright (C) 2023-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/fetch_utils.cl"

KERNEL(rpe_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    const __global INPUT1_TYPE* sin,
    const __global INPUT2_TYPE* cos,
    __global OUTPUT_TYPE* output)
{
    /* TO BE FILLED */
}