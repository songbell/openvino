// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "behavior/ov_infer_request/multithreading.hpp"

using namespace ov::test::behavior;

namespace {
const std::vector<ov::AnyMap> Multiconfigs = {
        {ov::device::priorities(ov::test::utils::DEVICE_TEMPLATE)}
};

INSTANTIATE_TEST_SUITE_P(smoke_Multi_BehaviorTests, OVInferRequestMultithreadingTests,
                        ::testing::Combine(
                                ::testing::Values(ov::test::utils::DEVICE_MULTI),
                                ::testing::ValuesIn(Multiconfigs)),
                            OVInferRequestMultithreadingTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Auto_BehaviorTests, OVInferRequestMultithreadingTests,
                        ::testing::Combine(
                                ::testing::Values(ov::test::utils::DEVICE_AUTO),
                                ::testing::ValuesIn(Multiconfigs)),
                            OVInferRequestMultithreadingTests::getTestCaseName);

}  // namespace
