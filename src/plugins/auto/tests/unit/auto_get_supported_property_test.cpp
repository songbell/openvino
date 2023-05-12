// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "include/mock_engine.hpp"

TEST_P(CoreIntegrationTest, getCorrectSupportProperty) {
    auto result =  mockplugin->get_core()->get_supported_property("MOCK_PLUGIN", {ov::hint::allow_auto_batching(true)});

    if (result.find(ov::hint::allow_auto_batching.name()) == result.end())
        std::cout << "error" << std::endl;
}

const std::vector<pluginConfigParams> testConfigs = {pluginConfigParams {{ov::hint::allow_auto_batching(true)}}
                                              };
INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests,
                         CoreIntegrationTest,
                        ::testing::ValuesIn(testConfigs),
                         CoreIntegrationTest::getTestCaseName);