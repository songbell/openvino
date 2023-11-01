// Copyright (C) 2023-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rpe_kernel_selector.h"
#include "rpe_kernel_ref.h"

namespace kernel_selector {

rpe_kernel_selector::rpe_kernel_selector() { Attach<RPEKernelRef>(); }

KernelsData rpe_kernel_selector::GetBestKernels(const Params& params, const optional_params& options) const {
    return GetNaiveBestKernel(params, options, KernelType::RPE);
}
}  // namespace kernel_selector