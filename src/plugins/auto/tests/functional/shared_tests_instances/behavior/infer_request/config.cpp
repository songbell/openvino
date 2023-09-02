// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "behavior/infer_request/config.hpp"

using namespace BehaviorTestsDefinitions;
namespace {
    const std::vector<std::map<std::string, std::string>> multiConfigs = {
            {{ MULTI_CONFIG_KEY(DEVICE_PRIORITIES) , ov::test::utils::DEVICE_CPU}},
            {{ MULTI_CONFIG_KEY(DEVICE_PRIORITIES) , ov::test::utils::DEVICE_GPU}}
    };

    const std::vector<std::map<std::string, std::string>> MultiInConfigs = {
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , ov::test::utils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS,
                        InferenceEngine::PluginConfigParams::CPU_THROUGHPUT_AUTO}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , ov::test::utils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS,
                        InferenceEngine::PluginConfigParams::CPU_THROUGHPUT_NUMA}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , ov::test::utils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_CPU_THROUGHPUT_STREAMS, "8"}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , ov::test::utils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_CPU_BIND_THREAD, InferenceEngine::PluginConfigParams::NO}},
            {{InferenceEngine::MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES , ov::test::utils::DEVICE_CPU},
             {InferenceEngine::PluginConfigParams::KEY_CPU_BIND_THREAD, InferenceEngine::PluginConfigParams::YES}}
    };

    INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, InferRequestConfigTest,
                            ::testing::Combine(
                                    ::testing::Values(1u),
                                    ::testing::Values(ov::test::utils::DEVICE_MULTI),
                                    ::testing::ValuesIn(multiConfigs)),
                             InferRequestConfigTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests_, InferRequestConfigTest,
                             ::testing::Combine(
                                     ::testing::Values(1u),
                                     ::testing::Values(ov::test::utils::DEVICE_MULTI),
                                     ::testing::ValuesIn(MultiInConfigs)),
                                     InferRequestConfigTest::getTestCaseName);


}  // namespace
