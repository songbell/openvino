// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/reshape.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
};

INSTANTIATE_TEST_CASE_P(
    smoke_ReshapeCheckDynBatch, ReshapeLayerTestRevise,
    ::testing::Combine(
        ::testing::Values(true), ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({30, 30, 30, 30})),
        ::testing::Values(std::vector<int64_t>({30, 30, 30, 30})),
        ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE),
        ::testing::Values(std::map<std::string, std::string>({}))),
    ReshapeLayerTestRevise::getTestCaseName);

INSTANTIATE_TEST_CASE_P(
    smoke_ReshapeCheck, ReshapeLayerTestRevise,
    ::testing::Combine(
        ::testing::Values(true), ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({10, 10, 10, 10})),
        ::testing::Values(std::vector<int64_t>({10, 0, 100})),
        ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE),
        ::testing::Values(std::map<std::string, std::string>({}))),
    ReshapeLayerTestRevise::getTestCaseName);

INSTANTIATE_TEST_CASE_P(
    smoke_ReshapeCheckNegative, ReshapeLayerTestRevise,
    ::testing::Combine(
        ::testing::Values(true), ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({10, 10, 10, 10})),
        ::testing::Values(std::vector<int64_t>({10, -1, 100})),
        ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE),
        ::testing::Values(std::map<std::string, std::string>({}))),
    ReshapeLayerTestRevise::getTestCaseName);
}  // namespace
