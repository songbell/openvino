// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>

#include "behavior/ov_infer_request/infer_consistency.hpp"

using namespace ov::test::behavior;

namespace {
// for deviceConfigs, the deviceConfigs[0] is target device which need to be tested.
// deviceConfigs[1], deviceConfigs[2],deviceConfigs[n] are the devices which will
// be compared with target device, the result of target should be in one of the compared
// device.
using Configs = std::vector<std::pair<std::string, ov::AnyMap>>;

auto configs = []() {
    return std::vector<Configs>{{{CommonTestUtils::DEVICE_GPU, {}}, {CommonTestUtils::DEVICE_GPU, {}}}};
};

auto AutoConfigs = []() {
    return std::vector<Configs>{{{CommonTestUtils::DEVICE_AUTO + std::string(":") + CommonTestUtils::DEVICE_GPU,
                                  {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)}},
                                 {CommonTestUtils::DEVICE_GPU, {}}},
                                {{CommonTestUtils::DEVICE_AUTO + std::string(":") + CommonTestUtils::DEVICE_GPU,
                                  {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)}},
                                 {CommonTestUtils::DEVICE_GPU, {}}},
                                {{CommonTestUtils::DEVICE_AUTO + std::string(":") + CommonTestUtils::DEVICE_GPU,
                                  {ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)}},
                                 {CommonTestUtils::DEVICE_GPU, {}}},
                                {{CommonTestUtils::DEVICE_AUTO + std::string(":") + CommonTestUtils::DEVICE_GPU + "," +
                                      CommonTestUtils::DEVICE_CPU,
                                  {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)}},
                                 {CommonTestUtils::DEVICE_GPU, {}},
                                 {CommonTestUtils::DEVICE_CPU, {}}},
                                {{CommonTestUtils::DEVICE_AUTO + std::string(":") + CommonTestUtils::DEVICE_GPU + "," +
                                      CommonTestUtils::DEVICE_CPU,
                                  {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)}},
                                 {CommonTestUtils::DEVICE_GPU, {}},
                                 {CommonTestUtils::DEVICE_CPU, {}}},
                                {{CommonTestUtils::DEVICE_AUTO + std::string(":") + CommonTestUtils::DEVICE_GPU + "," +
                                      CommonTestUtils::DEVICE_CPU,
                                  {ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)}},
                                 {CommonTestUtils::DEVICE_GPU, {}},
                                 {CommonTestUtils::DEVICE_CPU, {}}},
                                {{CommonTestUtils::DEVICE_AUTO + std::string(":") + CommonTestUtils::DEVICE_CPU + "," +
                                      CommonTestUtils::DEVICE_GPU,
                                  {ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT)}},
                                 {CommonTestUtils::DEVICE_CPU, {}},
                                 {CommonTestUtils::DEVICE_GPU, {}}},
                                {{CommonTestUtils::DEVICE_AUTO + std::string(":") + CommonTestUtils::DEVICE_GPU + "," +
                                      CommonTestUtils::DEVICE_CPU,
                                  {ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT),
                                   ov::intel_auto::device_bind_buffer(true)}},
                                 {CommonTestUtils::DEVICE_GPU, {}},
                                 {CommonTestUtils::DEVICE_CPU, {}}},
                                {{CommonTestUtils::DEVICE_AUTO + std::string(":") + CommonTestUtils::DEVICE_CPU + "," +
                                      CommonTestUtils::DEVICE_GPU,
                                  {ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT),
                                   ov::intel_auto::device_bind_buffer(true)}},
                                 {CommonTestUtils::DEVICE_CPU, {}},
                                 {CommonTestUtils::DEVICE_GPU, {}}}};
};

auto MultiConfigs = []() {
    return std::vector<Configs>{{{CommonTestUtils::DEVICE_MULTI + std::string(":") + CommonTestUtils::DEVICE_GPU,
                                  {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)}},
                                 {CommonTestUtils::DEVICE_GPU, {}}},
                                {{CommonTestUtils::DEVICE_MULTI + std::string(":") + CommonTestUtils::DEVICE_GPU,
                                  {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)}},
                                 {CommonTestUtils::DEVICE_GPU, {}}},
                                {{CommonTestUtils::DEVICE_MULTI + std::string(":") + CommonTestUtils::DEVICE_GPU + "," +
                                      CommonTestUtils::DEVICE_CPU,
                                  {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)}},
                                 {CommonTestUtils::DEVICE_GPU, {}},
                                 {CommonTestUtils::DEVICE_CPU, {}}},
                                {{CommonTestUtils::DEVICE_MULTI + std::string(":") + CommonTestUtils::DEVICE_GPU + "," +
                                      CommonTestUtils::DEVICE_CPU,
                                  {ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT)}},
                                 {CommonTestUtils::DEVICE_GPU, {}},
                                 {CommonTestUtils::DEVICE_CPU, {}}}};
};

auto MultiBindConfigs = []() {
    return std::vector<Configs>{{{CommonTestUtils::DEVICE_MULTI + std::string(":") + CommonTestUtils::DEVICE_GPU + "," +
                                      CommonTestUtils::DEVICE_CPU,
                                  {ov::intel_auto::device_bind_buffer(true)}},
                                 {CommonTestUtils::DEVICE_GPU, {}},
                                 {CommonTestUtils::DEVICE_CPU, {}}}};
};

auto AutoBindConfigs = []() {
    return std::vector<Configs>{{{CommonTestUtils::DEVICE_AUTO + std::string(":") + CommonTestUtils::DEVICE_GPU + "," +
                                      CommonTestUtils::DEVICE_CPU,
                                  {ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT),
                                   ov::intel_auto::device_bind_buffer(true)}},
                                 {CommonTestUtils::DEVICE_GPU, {}},
                                 {CommonTestUtils::DEVICE_CPU, {}}},
                                {{CommonTestUtils::DEVICE_AUTO + std::string(":") + CommonTestUtils::DEVICE_CPU + "," +
                                      CommonTestUtils::DEVICE_GPU,
                                  {ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT),
                                   ov::intel_auto::device_bind_buffer(true)}},
                                 {CommonTestUtils::DEVICE_CPU, {}},
                                 {CommonTestUtils::DEVICE_GPU, {}}}};
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTests, OVInferConsistencyTest,
    ::testing::Combine(
        ::testing::Values(10),// inferRequest num
        ::testing::Values(50),// infer counts
        ::testing::ValuesIn(configs())),
    OVInferConsistencyTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, OVInferConsistencyTest,
    ::testing::Combine(
        ::testing::Values(10),// inferRequest num
        ::testing::Values(50),// infer counts
        ::testing::ValuesIn(AutoConfigs())),
    OVInferConsistencyTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, OVInferConsistencyTest,
    ::testing::Combine(
        ::testing::Values(10),// inferRequest num
        ::testing::Values(50),// infer counts
        ::testing::ValuesIn(MultiConfigs())),
    OVInferConsistencyTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_Bind_BehaviorTests, OVInferConsistencyTest,
    ::testing::Combine(
        ::testing::Values(0),// inferRequest num, will use optimal request number if set 0
        ::testing::Values(100),// infer counts
        ::testing::ValuesIn(AutoBindConfigs())),
    OVInferConsistencyTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Multi_Bind_BehaviorTests, OVInferConsistencyTest,
    ::testing::Combine(
        ::testing::Values(0),// inferRequest num, will use optimal request number if set 0
        ::testing::Values(100),// infer counts
        ::testing::ValuesIn(MultiBindConfigs())),
    OVInferConsistencyTest::getTestCaseName);

}  // namespace