// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <atomic>
#include <mutex>
#include <unordered_map>
#include <map>
#include <vector>
#include <string>

#include "cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp"
#include "threading/ie_thread_safe_containers.hpp"
#include "threading/ie_itask_executor.hpp"
#include "threading/ie_executor_manager.hpp"
#include "ie_icore.hpp"
#include <ie_performance_hints.hpp>
#include "openvino/runtime/properties.hpp"

#ifdef  MULTIUNITTEST
#define MOCKTESTMACRO virtual
#define MultiDevicePlugin MockMultiDevicePlugin
#else
#define MOCKTESTMACRO
#endif

namespace MultiDevicePlugin {

class MultiDeviceInferencePlugin;
class Schedule : public std::enable_shared_from_this<Schedule>  {
public:
    using Ptr = std::shared_ptr<Schedule>;
    Schedule() {}
    Schedule(InferenceEngine::SoExecutableNetworkInternal& exenetwork) { _soExeNetwork = exenetwork; }

    virtual InferenceEngine::IInferRequestInternal::Ptr CreateInferRequest() {
        return _soExeNetwork->CreateInferRequest();
    }

    void SetCallBackExecutor(const InferenceEngine::ITaskExecutor::Ptr& callbackexe) {_callbackExecutor = callbackexe; }
    InferenceEngine::ITaskExecutor::Ptr& GetCallbackExe() { return _callbackExecutor; }
    void SetExecutableNetworkInternal(const InferenceEngine::IExecutableNetworkInternal::Ptr& exeNetwork) { _exeNetwork = exeNetwork; }
    InferenceEngine::IExecutableNetworkInternal::Ptr& GetExecutableNetworkInternal() { return _exeNetwork; }
    virtual ~Schedule() = default;
private:
    InferenceEngine::IExecutableNetworkInternal::Ptr _exeNetwork;
    InferenceEngine::ITaskExecutor::Ptr _callbackExecutor;
    InferenceEngine::SoExecutableNetworkInternal _soExeNetwork;
};

}  // namespace MultiDevicePlugin
