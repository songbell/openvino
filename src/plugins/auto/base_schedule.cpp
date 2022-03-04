// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <mutex>
#include <string>
#include <vector>
#include <memory>
#include <utility>
#include <map>
#include <unordered_map>

#include "ie_icore.hpp"
#include "ie_metric_helpers.hpp"
#include <ie_plugin_config.hpp>
#include "base_schedule.hpp"
#include "async_infer_request.hpp"
#include "plugin.hpp"

#include "ngraph/opsets/opset1.hpp"
#include "transformations/utils/utils.hpp"
#include "utils/log_util.hpp"

#include "itt.hpp"
// ------------------------------MultiDeviceExecutableNetwork----------------------------
namespace MultiDevicePlugin {
using namespace InferenceEngine;

Schedule::Schedule(InferenceEngine::SoExecutableNetworkInternal exenetwork) {
    _soExeNetwork = exenetwork;
}

InferenceEngine::IInferRequestInternal::Ptr Schedule::CreateInferRequestImpl(
    const std::vector<std::shared_ptr<const ov::Node>>& inputs,
    const std::vector<std::shared_ptr<const ov::Node>>& outputs) {
    return _soExeNetwork->CreateInferRequest();
}

InferenceEngine::IInferRequestInternal::Ptr Schedule::CreateInferRequestImpl(
    InferenceEngine::InputsDataMap networkInputs,
    InferenceEngine::OutputsDataMap networkOutputs) {
    return _soExeNetwork->CreateInferRequest();
}

IInferRequestInternal::Ptr Schedule::CreateInferRequest() {
    return _soExeNetwork->CreateInferRequest();
}
} // namespace MultiDevicePlugin