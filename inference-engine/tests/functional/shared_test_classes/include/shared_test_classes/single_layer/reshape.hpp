// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

namespace LayerTestsDefinitions {

//TODO: remove this alias when ticket 57975 is done - ticket: 57976
typedef std::tuple<bool,                               // SpecialZero
                   InferenceEngine::Precision,         // Network precision
                   InferenceEngine::Precision,         // Input precision
                   InferenceEngine::Precision,         // Output precision
                   InferenceEngine::Layout,            // Input layout
                   InferenceEngine::Layout,            // Output layout
                   std::vector<size_t>,                // Input shapes
                   std::vector<size_t>,                // OutForm Shapes
                   std::string,                        // Device name
                   std::map<std::string, std::string>  // Config
                   >
    reshapeParams;

//TODO: remove this class when ticket 57975 is done - ticket: 57976
class ReshapeLayerTest : public testing::WithParamInterface<reshapeParams>,
                         virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(
        testing::TestParamInfo<reshapeParams> obj);

protected:
    void SetUp() override;
};


//TODO: use this alias in all dependencies - ticket: 57975
typedef std::tuple<bool,                               // SpecialZero
                   InferenceEngine::Precision,         // Network precision
                   InferenceEngine::Precision,         // Input precision
                   InferenceEngine::Precision,         // Output precision
                   InferenceEngine::Layout,            // Input layout
                   InferenceEngine::Layout,            // Output layout
                   std::vector<size_t>,                // Input shapes
                   std::vector<int64_t>,               // OutForm Shapes
                   std::string,                        // Device name
                   std::map<std::string, std::string>  // Config
                   >
    reshapeParamsRevise;

//TODO: use this class in all dependencies - ticket: 57975
class ReshapeLayerTestRevise
    : public testing::WithParamInterface<reshapeParamsRevise>,
      virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(
        testing::TestParamInfo<reshapeParamsRevise> obj);

protected:
    void SetUp() override;
};
}  // namespace LayerTestsDefinitions
