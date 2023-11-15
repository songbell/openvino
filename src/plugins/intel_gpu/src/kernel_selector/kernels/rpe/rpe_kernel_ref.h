// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "kernel_base_opencl.h"

namespace kernel_selector {
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// rpe_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct rpe_params : public base_params {
    rpe_params() : base_params(KernelType::RPE) {}
    int64_t axis;
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// rpe_optional_params
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
struct rpe_optional_params : optional_params {
    rpe_optional_params() : optional_params(KernelType::RPE) {}
};

class RPEKernelRef : public KernelBaseOpenCL {
public:
    RPEKernelRef() : KernelBaseOpenCL("rpe_ref") {}
    virtual ~RPEKernelRef() {}
    virtual JitConstants GetJitConstants(const rpe_params& params) const;
    virtual CommonDispatchData SetDefault(const rpe_params& params) const;
    KernelsData GetKernelsData(const Params& params, const optional_params& options) const override;
    ParamsKey GetSupportedKey() const override;
    std::vector<FusedOpType> GetSupportedFusedOps() const override {
        return { };
    }

protected:
    bool Validate(const Params& p, const optional_params& o) const override;
};
}  // namespace kernel_selector