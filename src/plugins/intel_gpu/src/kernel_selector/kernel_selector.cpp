// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "kernel_base.h"
#include "kernel_selector_common.h"
#include "kernel_selector.h"
#include "kernel_selector_params.h"
#include <type_traits>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <tuple>
#include <set>
#include <iostream>
#include "intel_gpu/runtime/debug_configuration.hpp"

// #define ENABLE_ENV
// #define ENABLE_ENV_PRINT

#ifdef ENABLE_ENV_PRINT
#define ENV_PRINTF(...) printf(__VA_ARGS__)
#else
#define ENV_PRINTF(...)
#endif  // ENABLE_ENV_PRINT

#define ENABLE_OFFLINE_TUNING_CACHE 1

namespace kernel_selector {

AutoTuner kernel_selector_base::autoTuner;

#ifdef ENABLE_ENV
std::string strip(const std::string str) {
    size_t start = str.find_first_not_of(' ');
    size_t end = str.find_last_not_of(' ');
    if (start == std::string::npos || end == std::string::npos) {
        return "";
    }

    return str.substr(start, end - start + 1);
}

static void AddToForceMap(ForceList& force_list, bool force_or_deny, const char* env_str) {
    std::stringstream ss;
    ss.str(GetStringEnv(env_str));

    ENV_PRINTF("ENV: %s = %s\n", env_str, ss.str().c_str());

    std::string val;
    while (std::getline(ss, val, ',')) {
        std::string kernel_name = strip(val);
        if (!kernel_name.empty()) {
            force_list[kernel_name] = force_or_deny;
        }
    }
}
#endif

kernel_selector_base::kernel_selector_base() {
#ifdef ENABLE_ENV
    AddToForceMap(forceKernels, true, "CL_DNN_FORCE_KERNELS");
    AddToForceMap(forceKernels, false, "CL_DNN_DENY_KERNELS");
#endif
}

KernelData kernel_selector_base::get_best_kernel(const Params& params, const optional_params& options) const {
    auto kernels = GetBestKernels(params, options);
    OPENVINO_ASSERT(!kernels.empty(), "[GPU] Couldn't find a suitable kernel for ", params.layerID, " params raw string: ", params.to_cache_string_v2());
    return kernels[0];
}


KernelsData kernel_selector_base::GetNaiveBestKernel(const KernelList& all_impls, const Params& params, const optional_params& options) const {
    KernelsData kernelsData;
    std::string kernelName;

    for (const auto& implementation : all_impls) {
        // TODO: Unify this check with the Validate virtual method. Make
        // sure that the method is called here only, not in all the
        // GetKernelsData implementations.
        try {
            KernelsData kds = implementation->GetKernelsData(params, options);

            if (kds.size() && kds[0].kernels.size()) {
                kernelsData = kds;
                kernelName = implementation->GetName();
                if (!params.is_shape_agnostic) {
                    for (size_t k = 0; k < kds[0].kernels.size(); ++k) {
                        auto gws = kds[0].kernels[k].params.workGroups.global;
                        kernelsData[0].kernels[k].skip_execution = (std::accumulate(gws.begin(),
                                                                                    gws.end(),
                                                                                    static_cast<size_t>(1),
                                                                                    std::multiplies<size_t>()) == 0);
                    }
                }
                break;
            }
        } catch (std::runtime_error& ex) {
            // we have to handle it in order to avoid exception in KernelSelector as much we can
            kernelName = (implementation != nullptr)? implementation->GetName() : "[impl is null]";
            GPU_DEBUG_TRACE << "layerID: " << params.layerID << " kernel: " << kernelName << " - " << ex.what() << std::endl;
        }
    }

    // TODO: find a better place to located this assignment
    if (kernelsData.size()) {
        kernelsData[0].kernelName = kernelName;
        kernelsData[0].kernels[0].params.layerID = params.layerID;
    }

    return kernelsData;
}
KernelsData kernel_selector_base::GetNaiveBestKernel(const Params& params, const optional_params& options, KernelType kType) const {
    return GetNaiveBestKernel(GetAllImplementations(params, options, kType), params, options);
}

KernelsData kernel_selector_base::GetAutoTuneBestKernel(const Params& params, const optional_params& options, KernelType kType) const {
    KernelsData kernelsData;
    std::string kernelName;

    auto allImplementations = GetAllImplementations(params, options, kType);
    auto kernel_params = static_cast<const base_params&>(params);
    bool int8_kernel = kernel_params.inputs[0].GetDType() == Datatype::INT8 || kernel_params.inputs[0].GetDType() == Datatype::UINT8;
    std::tuple<std::string, int> cachedKernelConfig;
    if (!int8_kernel) {  // Try to load kernel/config from offline cache
        cachedKernelConfig = autoTuner.LoadKernelOffline(params);
    }
    bool hashFoundInCache = !std::get<0>(cachedKernelConfig).empty();

    if (hashFoundInCache) {
        std::string cachedkernelName = std::get<0>(cachedKernelConfig);
        int autoTuneIndex = std::get<1>(cachedKernelConfig);

        for (const auto& implementation : allImplementations) {
            // TODO: make sure kernel names are unique.
            if (implementation->GetName().compare(cachedkernelName) == 0) {
                KernelsData kds = implementation->GetTunedKernelsDataByIndex(params, options, autoTuneIndex);
                if (kds.size() && kds[0].kernels.size()) {
                    kernelsData = kds;
                    kernelsData[0].kernelName = cachedkernelName;
                    kernelsData[0].kernels[0].params.layerID = params.layerID;
                }
                break;
            }
        }

        if (!kernelsData.empty()) {
            return kernelsData;
        }
    }

    return GetNaiveBestKernel(allImplementations, params, options);
}

KernelList kernel_selector_base::GetAllImplementations(const Params& params, const optional_params& options, KernelType kType) const {
    using PriorityPair = std::pair<KernelsPriority, std::shared_ptr<KernelBase>>;
    auto comparePriority = [](const PriorityPair& firstImpl, const PriorityPair& secondImpl) {
        return firstImpl.first < secondImpl.first;
    };

    std::multiset<PriorityPair, decltype(comparePriority)> sortedImpls(comparePriority);
    KernelList result;

    auto device_features_key = params.engineInfo.get_supported_device_features_key();

    if (params.GetType() == kType && options.GetType() == kType) {
        ParamsKey requireKey = params.GetParamsKey().Merge(options.GetSupportedKey());
        bool forceImplementation = !params.forceImplementation.empty();
        for (auto& impl : implementations) {
            const ParamsKey implKey = impl->GetSupportedKey();
            if (!implKey.Support(requireKey))
                continue;

            auto required_device_features_key = impl->get_required_device_features_key(params, options);
            if (!device_features_key.supports(required_device_features_key))
                continue;

            if (forceImplementation && params.forceImplementation != impl->GetName())
                continue;
            sortedImpls.emplace(impl->GetKernelsPriority(params, options), impl);
        }

        std::transform(
            sortedImpls.begin(),
            sortedImpls.end(),
            std::back_inserter(result),
            [](const PriorityPair& impl) {
                return std::move(impl.second);
            });
    }

    return result;
}

}  // namespace kernel_selector
