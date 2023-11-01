// Copyright (C) 2023-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "rpe_kernel_ref.h"
#include "kernel_selector_utils.h"
#include <string>
#include <vector>

namespace kernel_selector {
ParamsKey RPEKernelRef::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F16);

    k.EnableInputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfyx);

    k.EnableTensorOffset();
    k.EnableTensorPitches();
    k.EnableBatching();
    return k;
}

CommonDispatchData RPEKernelRef::SetDefault(const rpe_params& params) const {
    CommonDispatchData dispatchData;

    /* FIXME: even for ref implementation, we can parallelize f-axis */
    dispatchData.gws = {1, 1, 1};
    dispatchData.lws = dispatchData.gws;

    return dispatchData;
}

JitConstants RPEKernelRef::GetJitConstants(const rpe_params& params) const {
    JitConstants jit = MakeBaseParamsJitConstants(params);

    return jit;
}

bool RPEKernelRef::Validate(const Params& p, const optional_params& o) const {
    if (p.GetType() != KernelType::RPE || o.GetType() != KernelType::RPE) {
        return false;
    }

    /* FIXME: fill here to allow SD-2.1 only */

    return true;
}

KernelsData RPEKernelRef::GetKernelsData(const Params& params, const optional_params& options) const {
    KernelData kd = KernelData::Default<rpe_params>(params);
    rpe_params& newParams = *static_cast<rpe_params*>(kd.params.get());

    if (!Validate(params, options)) {
        return {};
    }

    auto dispatchData = SetDefault(newParams);
    auto entry_point = GetEntryPoint(kernelName, newParams.layerID, params, options);
    auto cldnn_jit = GetJitConstants(newParams);
    auto jit = CreateJit(kernelName, cldnn_jit, entry_point);

    auto& kernel = kd.kernels[0];

    FillCLKernelData(kernel, dispatchData, params.engineInfo, kernelName, jit, entry_point,
                     EXE_MODE_DEFAULT, false, false, 3, GetFusedPrimitiveInputsCount(params));

    return { kd };}

}  // namespace kernel_selector