// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <atomic>
#include <mutex>
#include <queue>
#include <unordered_map>
#include <map>
#include <vector>
#include <utility>
#include <memory>
#include <string>
#include <cpp_interfaces/interface/ie_iinfer_request_internal.hpp>
#include "multi_device_exec_network.hpp"

#ifdef  MULTIUNITTEST
#define MOCKTESTMACRO virtual
#define MultiDevicePlugin MockMultiDevicePlugin
#else
#define MOCKTESTMACRO
#endif

namespace MultiDevicePlugin {

class MultiDeviceInferRequest : public InferenceEngine::IInferRequestInternal {
public:
    using Ptr = std::shared_ptr<MultiDeviceInferRequest>;
    explicit MultiDeviceInferRequest(const InferenceEngine::InputsDataMap&  networkInputs,
                                     const InferenceEngine::OutputsDataMap& networkOutputs,
                                     const InferenceEngine::SoIInferRequestInternal & request_to_share_blobs_with,
                                     bool needRecycle = false);
    explicit MultiDeviceInferRequest(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                                     const std::vector<std::shared_ptr<const ov::Node>>& outputs,
                                     const InferenceEngine::SoIInferRequestInternal & request_to_share_blobs_with,
                                     bool needRecycle = false);
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> GetPerformanceCounts() const override;
    void InferImpl() override;
    // Multi-Device impl specific: sets the data (blobs from the device-less requests to the specific device request)
    void SetBlobsToAnotherRequest(const InferenceEngine::SoIInferRequestInternal& req);
    const InferenceEngine::SoIInferRequestInternal& GetBlobSharedRequest() { return _requestToShareBlobWith; }
    bool NeedRecycle() { return _needRecycleManually; }

private:
    void CreateInferRequest(const InferenceEngine::SoIInferRequestInternal& request_to_share_blobs_with);
    bool _needRecycleManually;
    const InferenceEngine::SoIInferRequestInternal _requestToShareBlobWith;
};

}  // namespace MultiDevicePlugin
