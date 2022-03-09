// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <string>
#include <map>

#include "auto_schedule.hpp"
#include "base_executable_network.hpp"

#ifdef  MULTIUNITTEST
#define MOCKTESTMACRO virtual
#define MultiDevicePlugin MockMultiDevicePlugin
#else
#define MOCKTESTMACRO
#endif

namespace MultiDevicePlugin {
class AutoExecutableNetwork : public BaseExecutableNetwork {
    friend IInferPtr AutoSchedule::CreateInferRequest();
public:
    using Ptr = std::shared_ptr<AutoExecutableNetwork>;

    explicit AutoExecutableNetwork(AutoContext::Ptr& context,
        const AutoSchedule::Ptr& schedule);

    void SetConfig(const std::map<std::string, IE::Parameter>& config) override;
    IE::Parameter GetConfig(const std::string& name) const override;
    IE::Parameter GetMetric(const std::string& name) const override;
    std::shared_ptr<IE::RemoteContext> GetContext() const override;
    virtual ~AutoExecutableNetwork() = default;

private:
    AutoContext::Ptr _autoContext;
    AutoSchedule::Ptr _autoSchedule;
};
}  // namespace MultiDevicePlugin
