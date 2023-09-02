// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/compiled_model/properties.hpp"
#include "ie_system_conf.h"
#include "openvino/runtime/properties.hpp"

using namespace ov::test::behavior;

namespace {

const std::vector<ov::AnyMap> inproperties = {
    {ov::num_streams(-100)},
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests,
                         OVClassCompiledModelPropertiesIncorrectTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI, "AUTO:CPU"),
                                            ::testing::ValuesIn(inproperties)),
                         OVClassCompiledModelPropertiesIncorrectTests::getTestCaseName);

#if (defined(__APPLE__) || defined(_WIN32))
auto default_affinity = [] {
    auto numaNodes = InferenceEngine::getAvailableNUMANodes();
    auto coreTypes = InferenceEngine::getAvailableCoresTypes();
    if (coreTypes.size() > 1) {
        return ov::Affinity::HYBRID_AWARE;
    } else if (numaNodes.size() > 1) {
        return ov::Affinity::NUMA;
    } else {
        return ov::Affinity::NONE;
    }
}();
#else
auto default_affinity = [] {
    auto coreTypes = InferenceEngine::getAvailableCoresTypes();
    if (coreTypes.size() > 1) {
        return ov::Affinity::HYBRID_AWARE;
    } else {
        return ov::Affinity::CORE;
    }
}();
#endif

const std::vector<ov::AnyMap> default_properties = {
    {ov::affinity(default_affinity)},
};

const std::vector<ov::AnyMap> multi_properties = {
    {ov::device::priorities(ov::test::utils::DEVICE_CPU), ov::num_streams(ov::streams::AUTO)},
    {ov::device::priorities(ov::test::utils::DEVICE_CPU),
     {InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS,
      InferenceEngine::PluginConfigParams::CPU_THROUGHPUT_AUTO}},
};

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests,
                         OVClassCompiledModelPropertiesTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(multi_properties)),
                         OVClassCompiledModelPropertiesTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_OVCompiledModelIncorrectDevice, OVCompiledModelIncorrectDevice, ::testing::Values("CPU"));


const std::vector<ov::AnyMap> auto_multi_device_properties = {
    {ov::device::priorities(ov::test::utils::DEVICE_CPU), ov::device::properties("CPU", ov::num_streams(4))},
    {ov::device::priorities(ov::test::utils::DEVICE_CPU),
     ov::device::properties("CPU", ov::num_streams(4), ov::enable_profiling(true))},
    {ov::device::priorities(ov::test::utils::DEVICE_CPU),
     ov::device::properties(ov::AnyMap{{"CPU", ov::AnyMap{{ov::num_streams(4), ov::enable_profiling(true)}}}})}};

INSTANTIATE_TEST_SUITE_P(smoke_AutoMultiSetAndCompileModelBehaviorTestsNoThrow,
                         OVClassCompiledModelPropertiesTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO,
                                                              ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(auto_multi_device_properties)),
                         OVClassCompiledModelPropertiesTests::getTestCaseName);

const std::vector<ov::AnyMap> configsWithSecondaryProperties = {
    {ov::device::properties("CPU", ov::num_streams(4))},
    {ov::device::properties("CPU",
                            ov::num_streams(4),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
    {ov::device::properties("CPU",
                            ov::num_streams(4),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)),
     ov::device::properties("GPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY))}};

const std::vector<ov::AnyMap> multiConfigsWithSecondaryProperties = {
    {ov::device::priorities(ov::test::utils::DEVICE_CPU),
     ov::device::properties("CPU",
                            ov::num_streams(4),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
    {ov::device::priorities(ov::test::utils::DEVICE_CPU),
     ov::device::properties("CPU",
                            ov::num_streams(4),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)),
     ov::device::properties("GPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY))}};

const std::vector<ov::AnyMap> autoConfigsWithSecondaryProperties = {
    {ov::device::priorities(ov::test::utils::DEVICE_CPU),
     ov::device::properties("AUTO",
                            ov::enable_profiling(false),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
    {ov::device::priorities(ov::test::utils::DEVICE_CPU),
     ov::device::properties("CPU",
                            ov::num_streams(4),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
    {ov::device::priorities(ov::test::utils::DEVICE_CPU),
     ov::device::properties("CPU",
                            ov::num_streams(4),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)),
     ov::device::properties("GPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY))},
    {ov::device::priorities(ov::test::utils::DEVICE_CPU),
     ov::device::properties("AUTO",
                            ov::enable_profiling(false),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)),
     ov::device::properties("CPU",
                            ov::num_streams(4),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT))},
    {ov::device::priorities(ov::test::utils::DEVICE_GPU),
     ov::device::properties("AUTO",
                            ov::enable_profiling(false),
                            ov::device::priorities(ov::test::utils::DEVICE_CPU),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)),
     ov::device::properties("CPU",
                            ov::num_streams(4),
                            ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)),
     ov::device::properties("GPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY))}};

// IE Class Load network
INSTANTIATE_TEST_SUITE_P(smoke_CPUOVClassCompileModelWithCorrectPropertiesTest,
                         OVClassCompileModelWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values("AUTO:CPU", "MULTI:CPU"),
                                            ::testing::ValuesIn(configsWithSecondaryProperties)));

INSTANTIATE_TEST_SUITE_P(smoke_Multi_OVClassCompileModelWithCorrectPropertiesTest,
                         OVClassCompileModelWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values("MULTI"),
                                            ::testing::ValuesIn(multiConfigsWithSecondaryProperties)));

INSTANTIATE_TEST_SUITE_P(smoke_AUTO_OVClassCompileModelWithCorrectPropertiesTest,
                         OVClassCompileModelWithCorrectPropertiesTest,
                         ::testing::Combine(::testing::Values("AUTO"),
                                            ::testing::ValuesIn(autoConfigsWithSecondaryProperties)));

const std::vector<std::pair<ov::AnyMap, std::string>> automultiExeDeviceConfigs = {
    std::make_pair(ov::AnyMap{{ov::device::priorities(ov::test::utils::DEVICE_CPU)}}, "CPU")};

INSTANTIATE_TEST_SUITE_P(smoke_AutoMultiCompileModelBehaviorTests,
                         OVCompileModelGetExecutionDeviceTests,
                         ::testing::Combine(::testing::Values(ov::test::utils::DEVICE_AUTO,
                                                              ov::test::utils::DEVICE_MULTI),
                                            ::testing::ValuesIn(automultiExeDeviceConfigs)),
                         OVCompileModelGetExecutionDeviceTests::getTestCaseName);

const std::vector<ov::AnyMap> multiDevicePriorityConfigs = {
        {ov::device::priorities(ov::test::utils::DEVICE_CPU)}};

INSTANTIATE_TEST_SUITE_P(smoke_OVClassCompiledModelGetPropertyTest,
                         OVClassCompiledModelGetPropertyTest_DEVICE_PRIORITY,
                         ::testing::Combine(::testing::Values("MULTI", "AUTO"),
                                            ::testing::ValuesIn(multiDevicePriorityConfigs)),
                         OVClassCompiledModelGetPropertyTest_DEVICE_PRIORITY::getTestCaseName);

const std::vector<ov::AnyMap> multiModelPriorityConfigs = {
        {ov::hint::model_priority(ov::hint::Priority::HIGH)},
        {ov::hint::model_priority(ov::hint::Priority::MEDIUM)},
        {ov::hint::model_priority(ov::hint::Priority::LOW)},
        {ov::hint::model_priority(ov::hint::Priority::DEFAULT)}};

INSTANTIATE_TEST_SUITE_P(smoke_OVClassCompiledModelGetPropertyTest,
                         OVClassCompiledModelGetPropertyTest_MODEL_PRIORITY,
                         ::testing::Combine(::testing::Values("AUTO:CPU"),
                                            ::testing::ValuesIn(multiModelPriorityConfigs)));
}  // namespace
