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
#include "auto_schedule.hpp"
#include "async_infer_request.hpp"
#include "plugin.hpp"

#include "ngraph/opsets/opset1.hpp"
#include "transformations/utils/utils.hpp"
#include "utils/log_util.hpp"

#include "itt.hpp"
// ------------------------------MultiDeviceExecutableNetwork----------------------------
namespace MultiDevicePlugin {
using namespace InferenceEngine;

InferenceEngine::IInferRequestInternal::Ptr AutoSchedule::CreateInferRequestImpl(
    const std::vector<std::shared_ptr<const ov::Node>>& inputs,
    const std::vector<std::shared_ptr<const ov::Node>>& outputs) {
    InferenceEngine::SoIInferRequestInternal request_to_share_blobs_with;
    return std::make_shared<MultiDeviceInferRequest>(inputs, outputs, request_to_share_blobs_with);
}

InferenceEngine::IInferRequestInternal::Ptr AutoSchedule::CreateInferRequestImpl(
    InferenceEngine::InputsDataMap networkInputs,
    InferenceEngine::OutputsDataMap networkOutputs) {
    InferenceEngine::SoIInferRequestInternal request_to_share_blobs_with;
    return std::make_shared<MultiDeviceInferRequest>(networkInputs, networkOutputs, request_to_share_blobs_with);
}

IInferRequestInternal::Ptr AutoSchedule::CreateInferRequest() {
    InferenceEngine::IExecutableNetworkInternal::Ptr exenetwork = GetExecutableNetworkInternal();
    auto syncRequestImpl = CreateInferRequestImpl(exenetwork->getInputs(), exenetwork->getOutputs());
    syncRequestImpl->setPointerToExecutableNetworkInternal(std::static_pointer_cast<MultiDeviceExecutableNetwork>(exenetwork));
    return std::make_shared<MultiDeviceAsyncInferRequest>(std::static_pointer_cast<MultiDeviceInferRequest>(syncRequestImpl),
                                                          std::static_pointer_cast<MultiDeviceExecutableNetwork>(exenetwork)->_needPerfCounters,
                                                          std::static_pointer_cast<MultiDeviceExecutableNetwork>(exenetwork),
                                                          GetCallbackExe());
}
} // namespace MultiDevicePlugin