// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>
#include <tuple>
#include <utility>

#include <ie_metric_helpers.hpp>
#include <cpp/ie_cnn_network.h>
#include <cpp_interfaces/interface/ie_iexecutable_network_internal.hpp>
#include <legacy/ie_util_internal.hpp>

#include <vpu/vpu_plugin_config.hpp>
#include <vpu/parsed_config.hpp>
#include <vpu/frontend/frontend.hpp>
#include <vpu/utils/profiling.hpp>
#include <vpu/utils/error.hpp>
#include <vpu/ngraph/query_network.hpp>

#include <vpu/configuration/options/log_level.hpp>
#include <vpu/configuration/options/copy_optimization.hpp>

#include "myriad_plugin.h"

using namespace InferenceEngine;
using namespace InferenceEngine::PluginConfigParams;
using namespace InferenceEngine::VPUConfigParams;
using namespace vpu::MyriadPlugin;


IExecutableNetworkInternal::Ptr Engine::LoadExeNetworkImpl(
        const CNNNetwork& network,
        const std::map<std::string, std::string>& config) {
    VPU_PROFILE(LoadExeNetworkImpl);

    auto executableNetworkConfiguration = _parsedConfig;
    executableNetworkConfiguration.from(config);
    executableNetworkConfiguration.validate();

    return std::make_shared<ExecutableNetwork>(network, _mvnc, _devicePool, executableNetworkConfiguration, GetCore());
}

void Engine::SetConfig(const std::map<std::string, std::string> &config) {
    _parsedConfig.from(config);

    // TODO: remove once all options are migrated
    for (const auto& entry : config) {
        _config[entry.first] = entry.second;
    }

#ifndef NDEBUG
    if (const auto envVar = std::getenv("IE_VPU_LOG_LEVEL")) {
        _parsedConfig.set(LogLevelOption::key(), envVar);
    }
#endif
}

Parameter Engine::GetConfig(const std::string& name, const std::map<std::string, Parameter>& options) const {
    // TODO: remove once all options are migrated
    const auto& supportedKeys = _metrics->SupportedConfigKeys();
    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(supportedKeys.count(name) == 1 || _parsedConfig.supports(name), "Unsupported configuration key: {}", name);

    Parameter result;
    if (_parsedConfig.supports(name)) {
        result = _parsedConfig.asParameter(name);
    } else if (_config.count(name)) {
        // TODO: remove once all options are migrated
        result = _config.at(name);
    }

    return result;
}

QueryNetworkResult Engine::QueryNetwork(
        const CNNNetwork& network,
        const std::map<std::string, std::string>& config) const {
    VPU_PROFILE(QueryNetwork);
    QueryNetworkResult res;

    auto parsedConfigCopy = _parsedConfig;
    parsedConfigCopy.from(config);

    const auto deviceName = parsedConfigCopy.deviceName();
    if (!deviceName.empty()) {
        const auto deviceIDs = GetMetric(METRIC_KEY(AVAILABLE_DEVICES), {}).as<std::vector<std::string>>();
        VPU_THROW_UNLESS(!(std::find(deviceIDs.begin(), deviceIDs.end(), deviceName) == deviceIDs.end()), "Myriad device: {} not found.", deviceName);
    }

    const auto log = std::make_shared<Logger>(
            "GraphCompiler",
            _parsedConfig.get<LogLevelOption>(),
            defaultOutput(parsedConfigCopy.compilerLogFilePath()));

    const auto supportedLayers = getSupportedLayers(
            network,
            parsedConfigCopy.platform(),
            parsedConfigCopy,
            log,
            GetCore());

    if (auto function = network.getFunction()) {
        auto clonedNetwork = cloneNetwork(network);
        auto convertedNetwork = vpu::FrontEnd::convertNetwork(clonedNetwork);

        res = getQueryNetwork(convertedNetwork, function, GetName(), supportedLayers);
    } else {
        for (const auto& layerName : supportedLayers) {
            res.supportedLayersMap.insert({ layerName, GetName() });
        }
    }

    return res;
}

Engine::Engine(std::shared_ptr<IMvnc> mvnc) :
        _mvnc(std::move(mvnc)),
        _metrics(std::make_shared<MyriadMetrics>()) {
    VPU_THROW_UNLESS(_mvnc, "mvnc is null");

    _pluginName = "MYRIAD";

    // TODO: remove once all options are migrated
IE_SUPPRESS_DEPRECATED_START
    _config = {
        { MYRIAD_ENABLE_HW_ACCELERATION, CONFIG_VALUE(YES) },
        { MYRIAD_ENABLE_RECEIVING_TENSOR_TIME, CONFIG_VALUE(NO) },
        { MYRIAD_CUSTOM_LAYERS, "" },
        { MYRIAD_ENABLE_FORCE_RESET, CONFIG_VALUE(NO) },
        { MYRIAD_THROUGHPUT_STREAMS, "-1" },

        // Deprecated
        { KEY_VPU_HW_STAGES_OPTIMIZATION, CONFIG_VALUE(YES) },
        { KEY_VPU_PRINT_RECEIVE_TENSOR_TIME, CONFIG_VALUE(NO) },
        { KEY_VPU_CUSTOM_LAYERS, "" },
        { KEY_VPU_MYRIAD_FORCE_RESET, CONFIG_VALUE(NO) },
        { KEY_VPU_MYRIAD_PLATFORM, "" },

        { KEY_EXCLUSIVE_ASYNC_REQUESTS, CONFIG_VALUE(NO) },
        { KEY_PERF_COUNT, CONFIG_VALUE(NO) },
        { KEY_CONFIG_FILE, "" },
        { KEY_DEVICE_ID, "" },
    };
IE_SUPPRESS_DEPRECATED_END

    _parsedConfig.registerOption<LogLevelOption>();
    _parsedConfig.registerOption<CopyOptimizationOption>();

IE_SUPPRESS_DEPRECATED_START
    _parsedConfig.registerDeprecatedOption<LogLevelOption>(VPU_CONFIG_KEY(LOG_LEVEL));
IE_SUPPRESS_DEPRECATED_END
}

InferenceEngine::IExecutableNetworkInternal::Ptr Engine::ImportNetwork(
        std::istream& model,
        const std::map<std::string, std::string>& config) {
    VPU_PROFILE(ImportNetwork);

    auto executableNetworkConfiguration = _parsedConfig;
    executableNetworkConfiguration.fromAtRuntime(config);
    executableNetworkConfiguration.validate();

    const auto executableNetwork = std::make_shared<ExecutableNetwork>(model, _mvnc, _devicePool, executableNetworkConfiguration, GetCore());
    executableNetwork->SetPointerToPlugin(shared_from_this());
    return executableNetwork;
}

InferenceEngine::Parameter Engine::GetMetric(const std::string& name,
                                     const std::map<std::string, InferenceEngine::Parameter> & options) const {
    const auto mvnc = _mvnc;
    const auto metrics = _metrics;
    const auto devicePool = _devicePool;
    const auto getSpecifiedDeviceName = [&mvnc, &metrics, &devicePool, &options]() {
        if (options.count(KEY_DEVICE_ID)) {
            return options.at(KEY_DEVICE_ID).as<std::string>();
        }

        const auto availableDevices = metrics->AvailableDevicesNames(mvnc, devicePool);
        VPU_THROW_UNLESS(!availableDevices.empty(), "No devices available.");
        VPU_THROW_UNLESS(availableDevices.size() == 1, "KEY_DEVICE_ID is undefined.");

        return availableDevices.front();
    };
    const auto getDeviceByName = [&devicePool](const std::string& deviceName) {
        const auto deviceIt = std::find_if(
                devicePool.begin(), devicePool.end(), [&deviceName](DevicePtr device) {
                    return device->_name == deviceName;
                });
        if (deviceIt == devicePool.end()) {
            return DevicePtr();
        }
        return *deviceIt;
    };

    if (name == METRIC_KEY(AVAILABLE_DEVICES)) {
        IE_SET_METRIC_RETURN(AVAILABLE_DEVICES, _metrics->AvailableDevicesNames(_mvnc, _devicePool));
    } else if (name == METRIC_KEY(FULL_DEVICE_NAME)) {
        IE_SET_METRIC_RETURN(FULL_DEVICE_NAME, _metrics->FullName(getSpecifiedDeviceName()));
    } else if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        const auto& supportedMetrics = _metrics->SupportedMetrics();
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, std::vector<std::string>{supportedMetrics.cbegin(), supportedMetrics.cend()});
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        // TODO: remove once all options are migrated
        auto supportedConfigKeys = _metrics->SupportedConfigKeys();
        const auto& publicKeys = _parsedConfig.getPublicKeys();
        supportedConfigKeys.insert(publicKeys.cbegin(), publicKeys.cend());
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, std::vector<std::string>{supportedConfigKeys.cbegin(), supportedConfigKeys.cend()});
    } else if (name == METRIC_KEY(OPTIMIZATION_CAPABILITIES)) {
        const auto& optimizationCapabilities = _metrics->OptimizationCapabilities();
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, std::vector<std::string>{optimizationCapabilities.cbegin(), optimizationCapabilities.cend()});
    } else if (name == METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS)) {
        IE_SET_METRIC_RETURN(RANGE_FOR_ASYNC_INFER_REQUESTS, _metrics->RangeForAsyncInferRequests(_config));
    } else if (name == METRIC_KEY(DEVICE_ARCHITECTURE)) {
        IE_SET_METRIC_RETURN(DEVICE_ARCHITECTURE, _metrics->DeviceArchitecture(options));
    } else if (name == METRIC_KEY(IMPORT_EXPORT_SUPPORT)) {
        IE_SET_METRIC_RETURN(IMPORT_EXPORT_SUPPORT, true);
    } else if (name == METRIC_KEY(DEVICE_THERMAL)) {
        const auto& device = getDeviceByName(getSpecifiedDeviceName());
        if (device != nullptr) {
            IE_SET_METRIC_RETURN(DEVICE_THERMAL, _metrics->DevicesThermal(device));
        } else {
            return Parameter();
        }
    }
    IE_THROW(NotImplemented);
}
