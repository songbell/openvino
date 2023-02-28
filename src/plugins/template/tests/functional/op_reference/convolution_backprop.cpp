// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "openvino/op/convolution.hpp"
#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct ConvolutionBackpropParams {
    template <class IT>
    ConvolutionBackpropParams(const PartialShape& inputShape, const PartialShape& filterShape, const PartialShape& outputShape,
                      const element::Type& iType,
                      const std::vector<IT>& iValues, const std::vector<IT>& filterValues, const std::vector<IT>& oValues,
                      const Strides& strides, const CoordinateDiff& padBegin, const CoordinateDiff& padEnd, const Strides& dialations)
        : inputShape(inputShape),
          filterShape(filterShape),
          outputShape(outputShape),
          inType(iType),
          filterType(iType),
          outType(iType),
          inputData(CreateTensor(iType, iValues)),
          filterData(CreateTensor(iType, filterValues)),
          refData(CreateTensor(iType, oValues)),
          strides(strides),
          padBegin(padBegin),
          padEnd(padEnd),
          dialations(dialations) {}

    template <class IT>
    ConvolutionBackpropParams(const PartialShape& inputShape, const PartialShape& filterShape, const PartialShape& outputShape,
                      const element::Type& iType,
                      const std::vector<IT>& iValues, const std::vector<IT>& filterValues, const std::vector<IT>& oValues,
                      const Strides& strides, const CoordinateDiff& padBegin, const CoordinateDiff& padEnd, const Strides& dialations,
                      const CoordinateDiff& outPadding)
        : inputShape(inputShape),
          filterShape(filterShape),
          outputShape(outputShape),
          inType(iType),
          filterType(iType),
          outType(iType),
          inputData(CreateTensor(iType, iValues)),
          filterData(CreateTensor(iType, filterValues)),
          refData(CreateTensor(iType, oValues)),
          strides(strides),
          padBegin(padBegin),
          padEnd(padEnd),
          dialations(dialations),
          outPadding(outPadding) {}

    PartialShape inputShape;
    PartialShape filterShape;
    PartialShape outputShape;
    ov::element::Type inType;
    ov::element::Type filterType;
    ov::element::Type outType;
    ov::Tensor inputData;
    ov::Tensor filterData;
    ov::Tensor refData;
    ov::Strides strides;
    ov::CoordinateDiff padBegin;
    ov::CoordinateDiff padEnd;
    ov::Strides dialations;
    ov::CoordinateDiff outPadding;
};

class ReferenceConvolutionBackpropLayerTest : public testing::TestWithParam<ConvolutionBackpropParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.inputData, params.filterData};
        refOutData = {params.refData};
    }
    static std::string getTestCaseName(const testing::TestParamInfo<ConvolutionBackpropParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "inputShape=" << param.inputShape << "_";
        result << "filterShape=" << param.filterShape << "_";
        result << "outputShape=" << param.outputShape << "_";
        result << "iType=" << param.inType << "_";
        result << "oType=" << param.outType << "_";
        result << "strides=" << param.strides << "_";
        result << "padBegin=" << param.padBegin << "_";
        result << "padEnd=" << param.padEnd << "_";
        result << "dialations=" << param.dialations << "_";
        result << "outPadding=" << param.outPadding;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const ConvolutionBackpropParams& params) {
        const op::PadType auto_pad{op::PadType::EXPLICIT};

        const auto in = std::make_shared<op::v0::Parameter>(params.inType, params.inputShape);
        const auto filter = std::make_shared<op::v0::Parameter>(params.inType, params.filterShape);
        const auto ConvolutionBackprop = std::make_shared<op::v1::ConvolutionBackpropData>(in,
                                                                       filter,
                                                                       params.strides,
                                                                       params.padBegin,
                                                                       params.padEnd,
                                                                       params.dialations,
                                                                       auto_pad,
                                                                       params.outPadding);

        return std::make_shared<ov::Model>(NodeVector {ConvolutionBackprop}, ParameterVector {in, filter});
    }
};

TEST_P(ReferenceConvolutionBackpropLayerTest, CompareWithRefs) {
    Exec();
}

template <element::Type_t IN_ET>
std::vector<ConvolutionBackpropParams> generateConvolutionBackpropFloatParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<ConvolutionBackpropParams> convolutionBackpropParams {
// --------------------- 1D ConvolutionBackprop ------------------------------------------
        ConvolutionBackpropParams(PartialShape {1, 1, 4},
                          PartialShape {1, 1, 3},
                          PartialShape {1, 1, 6},
                          IN_ET,
                          std::vector<T>{5, 6, 7, 2},
                          std::vector<T>{2, 0, 1},
                          std::vector<T>{10, 12, 19, 10, 7, 2},
                          {1},
                          {0},
                          {0},
                          {1},
                          {0}),
        ConvolutionBackpropParams(PartialShape {1, 1, 4},
                          PartialShape {1, 1, 3},
                          PartialShape {1, 1, 4},
                          IN_ET,
                          std::vector<T>{5, 6, 7, 2},
                          std::vector<T>{2, 0, 1},
                          std::vector<T>{12, 19, 10, 7},
                          {1},
                          {1},
                          {1},
                          {1},
                          {0}),
        ConvolutionBackpropParams(PartialShape {1, 1, 2},
                          PartialShape {1, 1, 3},
                          PartialShape {1, 1, 5},
                          IN_ET,
                          std::vector<T>{5, 7},
                          std::vector<T>{2, 0, 1},
                          std::vector<T>{10, 0, 19, 0, 7},
                          {2},
                          {0},
                          {0},
                          {1},
                          {0}),
        ConvolutionBackpropParams(PartialShape {1, 1, 4},
                          PartialShape {1, 1, 3},
                          PartialShape {1, 1, 5},
                          IN_ET,
                          std::vector<T>{5, 6, 7, 2},
                          std::vector<T>{2, 0, 1},
                          std::vector<T>{12, 19, 10, 7, 2},
                          {1},
                          {1},
                          {1},
                          {1},
                          {1}),
        ConvolutionBackpropParams(PartialShape {1, 1, 3},
                          PartialShape {1, 1, 3},
                          PartialShape {1, 1, 7},
                          IN_ET,
                          std::vector<T>{8, 5, 1},
                          std::vector<T>{2, 0, 1},
                          std::vector<T>{16, 10, 2, 0, 8, 5, 1},
                          {1},
                          {0},
                          {0},
                          {2},
                          {0}),
        ConvolutionBackpropParams(PartialShape {1, 1, 4},
                          PartialShape {1, 1, 3},
                          PartialShape {1, 1, 7},
                          IN_ET,
                          std::vector<T>{3, 9, 1, 2},
                          std::vector<T>{2, 0, 1},
                          std::vector<T>{18, 0, 5, 0, 13, 0, 1},
                          {2},
                          {2},
                          {2},
                          {2},
                          {0}),
        ConvolutionBackpropParams(PartialShape {1, 1, 2},
                          PartialShape {1, 2, 3},
                          PartialShape {1, 2, 4},
                          IN_ET,
                          std::vector<T>{10, 3},
                          std::vector<T>{
                                    // channel 1
                                    2, 0, 1,
                                    // channel 2
                                    1, 0, 2},
                          std::vector<T>{
                                    // channel 1
                                    20, 6, 10, 3,
                                    // channel 2
                                    10, 3, 20, 6},
                          {1},
                          {0},
                          {0},
                          {1},
                          {0}),
        ConvolutionBackpropParams(PartialShape {1, 2, 2},
                          PartialShape {2, 1, 3},
                          PartialShape {1, 1, 4},
                          IN_ET,
                          std::vector<T>{
                                    // channel 1
                                    4, 7,
                                    // channel 2
                                    5, 5},
                          std::vector<T>{
                                    // filter 1
                                    2, 0, 1,
                                    // filter 2
                                    1, 0, 2},
                          std::vector<T>{13, 19, 14, 17},
                          {1},
                          {0},
                          {0},
                          {1},
                          {0}),
        ConvolutionBackpropParams(PartialShape {2, 1, 2},
                          PartialShape {1, 1, 3},
                          PartialShape {2, 1, 4},
                          IN_ET,
                          std::vector<T>{
                                    // batch 1
                                    1, 3,
                                    // batch 2
                                    2, 2},
                          std::vector<T>{2, 0, 1},
                          std::vector<T>{
                                    // batch 1
                                    2, 6, 1, 3,
                                    // batch 2
                                    4, 4, 2, 2},
                          {1},
                          {0},
                          {0},
                          {1},
                          {0}),
// --------------------- 2D ConvolutionBackprop ------------------------------------------
        ConvolutionBackpropParams(PartialShape {1, 1, 2, 2},
                          PartialShape {1, 1, 3, 3},
                          PartialShape {1, 1, 4, 4},
                          IN_ET,
                          std::vector<T>{
                                    1, 3,
                                    7, 5},
                          std::vector<T>{
                                    1, 2, 3,
                                    0, 1, 0,
                                    3, 2, 1},
                          std::vector<T>{
                                    1, 5, 9, 9,
                                    7, 20, 34, 15,
                                    3, 18, 12, 3,
                                    21, 29, 17, 5},
                          {1, 1},
                          {0, 0},
                          {0, 0},
                          {1, 1},
                          {0, 0}),
        ConvolutionBackpropParams(PartialShape {1, 1, 2, 2},
                          PartialShape {1, 1, 3, 3},
                          PartialShape {1, 1, 3, 3},
                          IN_ET,
                          std::vector<T>{
                                    1, 3,
                                    7, 5},
                          std::vector<T>{
                                    1, 2, 3,
                                    1, 1, 1,
                                    3, 2, 1},
                          std::vector<T>{
                                    23, 35, 18,
                                    23, 19, 8,
                                    29, 17, 5},
                          {1, 1},
                          {1, 1},
                          {1, 1},
                          {1, 1},
                          {1, 1}),
        ConvolutionBackpropParams(PartialShape {1, 1, 4, 4},
                          PartialShape {1, 1, 3, 3},
                          PartialShape {1, 1, 4, 4},
                          IN_ET,
                          std::vector<T>{
                                    1, 3, 5, 7,
                                    7, 5, 3, 1,
                                    2, 4, 6, 8,
                                    8, 6, 4, 2},
                          std::vector<T>{
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2},
                          std::vector<T>{
                                    20, 37, 27, 18,
                                    22, 40, 60, 52,
                                    41, 69, 49, 31,
                                    18, 26, 34, 22},
                          {1, 1},
                          {1, 1},
                          {1, 1},
                          {1, 1},
                          {0, 0}),
        ConvolutionBackpropParams(PartialShape {1, 1, 2, 2},
                          PartialShape {1, 1, 3, 3},
                          PartialShape {1, 1, 5, 5},
                          IN_ET,
                          std::vector<T>{
                                    2, 5,
                                    4, 3},
                          std::vector<T>{
                                    1, 2, 3,
                                    1, 1, 1,
                                    3, 2, 1},
                          std::vector<T>{
                                    2, 4, 11, 10, 15,
                                    2, 2, 7, 5, 5,
                                    10, 12, 32, 16, 14,
                                    4, 4, 7, 3, 3,
                                    12, 8, 13, 6, 3},
                          {2, 2},
                          {0, 0},
                          {0, 0},
                          {1, 1},
                          {0, 0}),
        ConvolutionBackpropParams(PartialShape {1, 1, 2, 2},
                          PartialShape {1, 1, 3, 3},
                          PartialShape {1, 1, 6, 6},
                          IN_ET,
                          std::vector<T>{
                                    2, 3,
                                    4, 3},
                          std::vector<T>{
                                    1, 2, 3,
                                    1, 1, 1,
                                    3, 2, 1},
                          std::vector<T>{
                                    2, 3, 4, 6, 6, 9,
                                    4, 3, 8, 6, 12, 9,
                                    2, 3, 2, 3, 2, 3,
                                    4, 3, 4, 3, 4, 3,
                                    6, 9, 4, 6, 2, 3,
                                    12, 9, 8, 6, 4, 3},
                          {1, 1},
                          {0, 0},
                          {0, 0},
                          {2, 2},
                          {0, 0}),
        ConvolutionBackpropParams(PartialShape {1, 1, 3, 3},
                          PartialShape {1, 1, 3, 3},
                          PartialShape {1, 1, 5, 5},
                          IN_ET,
                          std::vector<T>{
                                    1, 3, 5,
                                    7, 5, 3,
                                    2, 4, 6},
                          std::vector<T>{
                                    1, 2, 3,
                                    1, 1, 1,
                                    3, 2, 1},
                          std::vector<T>{
                                    23, 0, 43, 0, 29,
                                    0, 0, 0, 0, 0,
                                    31, 0, 57, 0, 45,
                                    0, 0, 0, 0, 0,
                                    35, 0, 38, 0, 21},
                          {2, 2},
                          {2, 2},
                          {2, 2},
                          {2, 2},
                          {0, 0}),
        ConvolutionBackpropParams(PartialShape {1, 1, 2, 2},
                          PartialShape {1, 2, 3, 3},
                          PartialShape {1, 2, 4, 4},
                          IN_ET,
                          std::vector<T>{
                                    1, 3,
                                    7, 5},
                          std::vector<T>{
                                    // channel 1
                                    5, 3, 5,
                                    1, 3, 1,
                                    4, 2, 4,
                                    // channel 2
                                    -5, 3, 5,
                                    1, -3, 1,
                                    4, 2, -4},
                          std::vector<T>{
                                    // channel 1
                                    5, 18, 14, 15,
                                    36, 52, 60, 28,
                                    11, 40, 32, 17,
                                    28, 34, 38, 20,
                                    // channel 2
                                    -5, -12, 14, 15,
                                    -34, -4, 42, 28,
                                    11, -2, -6, -7,
                                    28, 34, -18, -20},
                          {1, 1},
                          {0, 0},
                          {0, 0},
                          {1, 1},
                          {0, 0}),
        ConvolutionBackpropParams(PartialShape {1, 2, 2, 2},
                          PartialShape {2, 1, 3, 3},
                          PartialShape {1, 1, 4, 4},
                          IN_ET,
                          std::vector<T>{
                                    // channel 1
                                    1, 3,
                                    7, 5,
                                    // channel 2
                                    2, 4,
                                    8, 6},
                          std::vector<T>{
                                    // channel 1
                                    5, 3, 5,
                                    1, 3, 1,
                                    4, 2, 4,
                                    // channel 2
                                   -5, 3, 5,
                                    1, -3, 1,
                                    4, 2, -4},
                          std::vector<T>{
                                    -5, 4, 36, 35,
                                     -2, 44, 108, 62,
                                     27, 42, 22, 7,
                                     60, 74, 18, -4},
                          {1, 1},
                          {0, 0},
                          {0, 0},
                          {1, 1},
                          {0, 0}),
        ConvolutionBackpropParams(PartialShape {1, 2, 1, 1},
                          PartialShape {2, 2, 2, 2},
                          PartialShape {1, 2, 2, 2},
                          IN_ET,
                          std::vector<T>{
                                    // channel 1
                                    2,
                                    // channel 2
                                    3},
                          std::vector<T>{
                                    // batch 0
                                    // channel 1
                                    5, 3,
                                    1, 3,
                                    // channel 2
                                   -5, 3,
                                    1, -3,
                                    // batch 1
                                    // channel 1
                                    5, 3,
                                    1, 3,
                                    // channel 2
                                   -5, 3,
                                    1, -3},
                          std::vector<T>{
                                    25, 15, 5, 15, -25, 15, 5, -15},
                          {1, 1},
                          {0, 0},
                          {0, 0},
                          {1, 1},
                          {0, 0}),
        ConvolutionBackpropParams(PartialShape {2, 1, 2, 2},
                          PartialShape {1, 1, 3, 3},
                          PartialShape {2, 1, 4, 4},
                          IN_ET,
                          std::vector<T>{
                                    // batch 1
                                    1, 3,
                                    1, 3,
                                    // batch 2
                                    -1, 3,
                                    1, 3},
                          std::vector<T>{
                                    -5, 3, 5,
                                    1, -3, 1,
                                    4, 2, -4},
                          std::vector<T>{
                                    // batch 1
                                    -5, -12, 14, 15,
                                    -4, -12, 6, 18,
                                    5, 14, -6, -9,
                                    4, 14, 2, -12,
                                    // batch 2
                                    5, -18, 4, 15,
                                    -6, -6, 4, 18,
                                    -3, 10, 2, -9,
                                    4, 14, 2, -12},
                          {1, 1},
                          {0, 0},
                          {0, 0},
                          {1, 1},
                          {0, 0}),
// --------------------- 3D convolution ------------------------------------------
        ConvolutionBackpropParams(PartialShape {1, 1, 2, 2, 2},
                          PartialShape {1, 1, 3, 3, 3},
                          PartialShape {1, 1, 4, 4, 4},
                          IN_ET,
                          std::vector<T>{
                                    // depth: 1
                                    15, 3,
                                    21, 10,
                                    // depth: 2
                                    10, 13,
                                    11, 17},
                          std::vector<T>{
                                    // depth: 1
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 2
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 3
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2},
                          std::vector<T>{
                                    // depth: 1
                                    15, 33, 51, 9,
                                    21, 67, 86, 30,
                                    30, 42, 43, 6,
                                    42, 41, 52, 20,
                                    // depth: 2
                                    25, 66, 107, 48,
                                    32, 116, 166, 81,
                                    50, 89, 93, 32,
                                    64, 86, 91, 54,
                                    // depth: 3
                                    25, 66, 107, 48,
                                    32, 116, 166, 81,
                                    50, 89, 93, 32,
                                    64, 86, 91, 54,
                                    // depth: 4
                                    10, 33, 56, 39,
                                    11, 49, 80, 51,
                                    20, 47, 50, 26,
                                    22, 45, 39, 34},
                          {1, 1, 1},
                          {0, 0, 0},
                          {0, 0, 0},
                          {1, 1, 1},
                          {0, 0, 0}),
        ConvolutionBackpropParams(PartialShape {1, 1, 2, 2, 2},
                          PartialShape {1, 1, 3, 3, 3},
                          PartialShape {1, 1, 3, 3, 3},
                          IN_ET,
                          std::vector<T>{
                                    // depth: 1
                                    15, 3,
                                    21, 10,
                                    // depth: 2
                                    10, 13,
                                    11, 17},
                          std::vector<T>{
                                    // depth: 1
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 2
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 3
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2},
                          std::vector<T>{
                                    // depth: 1
                                    116, 166, 81,
                                    89, 93, 32,
                                    86, 91, 54,
                                    // depth: 2
                                    116, 166, 81,
                                    89, 93, 32,
                                    86, 91, 54,
                                    // depth: 3
                                    49, 80, 51,
                                    47, 50, 26,
                                    45, 39, 34},
                          {1, 1, 1},
                          {1, 1, 1},
                          {1, 1, 1},
                          {1, 1, 1},
                          {1, 1, 1}),
        ConvolutionBackpropParams(PartialShape {1, 1, 4, 4, 4},
                          PartialShape {1, 1, 3, 3, 3},
                          PartialShape {1, 1, 4, 4, 4},
                          IN_ET,
                          std::vector<T>{
                                    // depth: 1
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3,
                                    // depth: 2
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3,
                                    // depth: 3
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3,
                                    // depth: 4
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3},
                          std::vector<T>{
                                    // depth: 1
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 2
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 3
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2},
                          std::vector<T>{
                                    // depth: 1
                                     12, 30, 36, 24,
                                     26, 42, 42, 30,
                                     34, 56, 54, 50,
                                     14, 18, 24, 16,
                                     // depth: 2
                                     18, 45, 54, 36,
                                     39, 63, 63, 45,
                                     51, 84, 81, 75,
                                     21, 27, 36, 24,
                                     // depth: 3
                                     18, 45, 54, 36,
                                     39, 63, 63, 45,
                                     51, 84, 81, 75,
                                     21, 27, 36, 24,
                                     // depth: 4
                                     12, 30, 36, 24,
                                     26, 42, 42, 30,
                                     34, 56, 54, 50,
                                     14, 18, 24, 16},
                          {1, 1, 1},
                          {1, 1, 1},
                          {1, 1, 1},
                          {1, 1, 1},
                          {0, 0, 0}),
        ConvolutionBackpropParams(PartialShape {1, 1, 2, 2, 2},
                          PartialShape {1, 1, 3, 3, 3},
                          PartialShape {1, 1, 5, 5, 5},
                          IN_ET,
                          std::vector<T>{
                                    // depth: 1
                                    15, 3,
                                    21, 10,
                                    // depth: 2
                                    10, 13,
                                    11, 17},
                          std::vector<T>{
                                    // depth: 1
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 2
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 3
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2},
                          std::vector<T>{
                                    // depth: 1
                                    15, 30, 48, 6, 9,
                                    0, 15, 0, 3, 0,
                                    51, 57, 109, 23, 36,
                                    0, 21, 0, 10, 0,
                                    42, 21, 62, 10, 20,
                                    // depth: 2
                                    15, 30, 48, 6, 9,
                                    0, 15, 0, 3, 0,
                                    51, 57, 109, 23, 36,
                                    0, 21, 0, 10, 0,
                                    42, 21, 62, 10, 20,
                                    // depth: 3
                                    25, 50, 91, 32, 48,
                                    0, 25, 0, 16, 0,
                                    82, 89, 205, 70, 113,
                                    0, 32, 0, 27, 0,
                                    64, 32, 118, 27, 54,
                                    // depth: 4
                                    10, 20, 43, 26, 39,
                                    0, 10, 0, 13, 0,
                                    31, 32, 96, 47, 77,
                                    0, 11, 0, 17, 0,
                                    22, 11, 56, 17, 34,
                                    // depth: 5
                                    10, 20, 43, 26, 39,
                                    0, 10, 0, 13, 0,
                                    31, 32, 96, 47, 77,
                                    0, 11, 0, 17, 0,
                                    22, 11, 56, 17, 34},
                          {2, 2, 2},
                          {0, 0, 0},
                          {0, 0, 0},
                          {1, 1, 1},
                          {0, 0, 0}),
        ConvolutionBackpropParams(PartialShape {1, 1, 4, 4, 4},
                          PartialShape {1, 1, 3, 3, 3},
                          PartialShape {1, 1, 7, 7, 7},
                          IN_ET,
                          std::vector<T>{
                                    // depth: 1
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3,
                                    // depth: 2
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3,
                                    // depth: 3
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3,
                                    // depth: 4
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3},
                          std::vector<T>{
                                    // depth: 1
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 2
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 3
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2},
                          std::vector<T>{
                                    // depth: 1
                                    12, 0, 30, 0, 36, 0, 24,
                                    0, 0, 0, 0, 0, 0, 0,
                                    26, 0, 42, 0, 42, 0, 30,
                                    0, 0, 0, 0, 0, 0, 0,
                                    34, 0, 56, 0, 54, 0, 50,
                                    0, 0, 0, 0, 0, 0, 0,
                                    14, 0, 18, 0, 24, 0, 16,
                                    // depth: 2
                                    0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0,
                                    // depth: 3
                                    18, 0, 45, 0, 54, 0, 36,
                                    0, 0, 0, 0, 0, 0, 0,
                                    39, 0, 63, 0, 63, 0, 45,
                                    0, 0, 0, 0, 0, 0, 0,
                                    51, 0, 84, 0, 81, 0, 75,
                                    0, 0, 0, 0, 0, 0, 0,
                                    21, 0, 27, 0, 36, 0, 24,
                                    // depth: 4
                                    0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0,
                                    // depth: 5
                                    18, 0, 45, 0, 54, 0, 36,
                                    0, 0, 0, 0, 0, 0, 0,
                                    39, 0, 63, 0, 63, 0, 45,
                                    0, 0, 0, 0, 0, 0, 0,
                                    51, 0, 84, 0, 81, 0, 75,
                                    0, 0, 0, 0, 0, 0, 0,
                                    21, 0, 27, 0, 36, 0, 24,
                                    // depth: 6
                                    0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0,
                                    // depth: 7
                                    12, 0, 30, 0, 36, 0, 24,
                                    0, 0, 0, 0, 0, 0, 0,
                                    26, 0, 42, 0, 42, 0, 30,
                                    0, 0, 0, 0, 0, 0, 0,
                                    34, 0, 56, 0, 54, 0, 50,
                                    0, 0, 0, 0, 0, 0, 0,
                                    14, 0, 18, 0, 24, 0, 16},
                          {2, 2, 2},
                          {2, 2, 2},
                          {2, 2, 2},
                          {2, 2, 2},
                          {0, 0, 0}),
        ConvolutionBackpropParams(PartialShape {1, 1, 2, 2, 2},
                          PartialShape {1, 2, 3, 3, 3},
                          PartialShape {1, 2, 4, 4, 4},
                          IN_ET,
                          std::vector<T>{
                                    // depth: 1
                                    1, 8,
                                    1, 3,
                                    // depth: 2
                                    1, 7,
                                    3, 8},
                          std::vector<T>{
                                    // -- channel 1 --
                                    // depth: 1
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 2
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 3
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // -- channel 2 --
                                    // depth: 1
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 2
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 3
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2},
                          std::vector<T>{
                                    // -- channel 1 --
                                    // depth: 1
                                    1, 10, 19, 24,
                                    1, 6, 17, 9,
                                    2, 18, 13, 16,
                                    2, 7, 5, 6,
                                    // depth: 2
                                    2, 19, 36, 45,
                                    4, 21, 49, 33,
                                    4, 36, 30, 30,
                                    8, 26, 19, 22,
                                    // depth: 3
                                    2, 19, 36, 45,
                                    4, 21, 49, 33,
                                    4, 36, 30, 30,
                                    8, 26, 19, 22,
                                    // depth: 4
                                    1, 9, 17, 21,
                                    3, 15, 32, 24,
                                    2, 18, 17, 14,
                                    6, 19, 14, 16,
                                    // -- channel 2 --
                                    // depth: 1
                                    1, 10, 19, 24,
                                    1, 6, 17, 9,
                                    2, 18, 13, 16,
                                    2, 7, 5, 6,
                                    // depth: 2
                                    2, 19, 36, 45,
                                    4, 21, 49, 33,
                                    4, 36, 30, 30,
                                    8, 26, 19, 22,
                                    // depth: 3
                                    2, 19, 36, 45,
                                    4, 21, 49, 33,
                                    4, 36, 30, 30,
                                    8, 26, 19, 22,
                                    // depth: 4
                                    1, 9, 17, 21,
                                    3, 15, 32, 24,
                                    2, 18, 17, 14,
                                    6, 19, 14, 16},
                          {1, 1, 1},
                          {0, 0, 0},
                          {0, 0, 0},
                          {1, 1, 1},
                          {0, 0, 0}),
        ConvolutionBackpropParams(PartialShape {1, 2, 2, 2, 2},
                          PartialShape {2, 1, 3, 3, 3},
                          PartialShape {1, 1, 4, 4, 4},
                          IN_ET,
                          std::vector<T>{
                                    // -- in 1 --
                                    // depth: 1
                                    1, 3,
                                    2, 5,
                                    // depth: 2
                                    1, 0,
                                    3, 6,
                                    // -- in 2 --
                                    // depth: 1
                                    1, 3,
                                    2, 5,
                                    // depth: 2
                                    3, 0,
                                    1, 8},
                          std::vector<T>{
                                    // -- filter 1 --
                                    // depth: 1
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 2
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 3
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // -- filter 2 --
                                    // depth: 1
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 2
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 3
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2},
                          std::vector<T>{
                                    // depth: 1
                                     2, 10, 18, 18,
                                     4, 20, 38, 30,
                                     4, 18, 20, 12,
                                     8, 24, 18, 20,
                                     // depth: 2
                                     6, 18, 30, 18,
                                     8, 46, 78, 72,
                                     12, 26, 42, 12,
                                     16, 56, 40, 48,
                                     // depth: 3
                                     6, 18, 30, 18,
                                     8, 46, 78, 72,
                                     12, 26, 42, 12,
                                     16, 56, 40, 48,
                                     // depth: 4
                                     4, 8, 12, 0,
                                     4, 26, 40, 42,
                                     8, 8, 22, 0,
                                     8, 32, 22, 28},
                          {1, 1, 1},
                          {0, 0, 0},
                          {0, 0, 0},
                          {1, 1, 1},
                          {0, 0, 0}),
        ConvolutionBackpropParams(PartialShape {2, 1, 2, 2, 2},
                          PartialShape {1, 1, 3, 3, 3},
                          PartialShape {2, 1, 4, 4, 4},
                          IN_ET,
                          std::vector<T>{
                                    // -- batch 1 --
                                    // depth: 1
                                    1, 3,
                                    2, 5,
                                    // depth: 2
                                    1, 0,
                                    6, 4,
                                    // -- batch 2 --
                                    // depth: 1
                                    1, 5,
                                    2, 8,
                                    // depth: 2
                                    2, 1,
                                    0, 5},
                          std::vector<T>{
                                    // depth: 1
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 2
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 3
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2},
                          std::vector<T>{
                                    // -- batch 1 --
                                     // depth: 1
                                     1, 5, 9, 9,
                                     2, 10, 19, 15,
                                     2, 9, 10, 6,
                                     4, 12, 9, 10,
                                     // depth: 2
                                     2, 7, 12, 9,
                                     8, 27, 45, 27,
                                     4, 16, 16, 6,
                                     16, 26, 25, 18,
                                     // depth: 3
                                     2, 7, 12, 9,
                                     8, 27, 45, 27,
                                     4, 16, 16, 6,
                                     16, 26, 25, 18,
                                     // depth: 4
                                     1, 2, 3, 0,
                                     6, 17, 26, 12,
                                     2, 7, 6, 0,
                                     12, 14, 16, 8,
                                     // -- batch 2 --
                                     // depth: 1
                                     1, 7, 13, 15,
                                     2, 13, 27, 24,
                                     2, 13, 15, 10,
                                     4, 18, 12, 16,
                                     // depth: 2
                                     3, 12, 21, 18,
                                     2, 20, 38, 39,
                                     6, 17, 25, 12,
                                     4, 28, 17, 26,
                                     // depth: 3
                                     3, 12, 21, 18,
                                     2, 20, 38, 39,
                                     6, 17, 25, 12,
                                     4, 28, 17, 26,
                                     // depth: 4
                                     2, 5, 8, 3,
                                     0, 7, 11, 15,
                                     4, 4, 10, 2,
                                     0, 10, 5, 10},
                          {1, 1, 1},
                          {0, 0, 0},
                          {0, 0, 0},
                          {1, 1, 1},
                          {0, 0, 0})
    };
    return convolutionBackpropParams;
}

template <element::Type_t IN_ET>
std::vector<ConvolutionBackpropParams> generateConvolutionBackpropUintParams() {
    using T = typename element_type_traits<IN_ET>::value_type;

    std::vector<ConvolutionBackpropParams> convolutionBackpropParams {
// --------------------- 1D ConvolutionBackprop ------------------------------------------
        ConvolutionBackpropParams(PartialShape {1, 1, 4},
                          PartialShape {1, 1, 3},
                          PartialShape {1, 1, 6},
                          IN_ET,
                          std::vector<T>{5, 6, 7, 2},
                          std::vector<T>{2, 0, 1},
                          std::vector<T>{10, 12, 19, 10, 7, 2},
                          {1},
                          {0},
                          {0},
                          {1},
                          {0}),
        ConvolutionBackpropParams(PartialShape {1, 1, 4},
                          PartialShape {1, 1, 3},
                          PartialShape {1, 1, 4},
                          IN_ET,
                          std::vector<T>{5, 6, 7, 2},
                          std::vector<T>{2, 0, 1},
                          std::vector<T>{12, 19, 10, 7},
                          {1},
                          {1},
                          {1},
                          {1},
                          {0}),
        ConvolutionBackpropParams(PartialShape {1, 1, 2},
                          PartialShape {1, 1, 3},
                          PartialShape {1, 1, 5},
                          IN_ET,
                          std::vector<T>{5, 7},
                          std::vector<T>{2, 0, 1},
                          std::vector<T>{10, 0, 19, 0, 7},
                          {2},
                          {0},
                          {0},
                          {1},
                          {0}),
        ConvolutionBackpropParams(PartialShape {1, 1, 4},
                          PartialShape {1, 1, 3},
                          PartialShape {1, 1, 5},
                          IN_ET,
                          std::vector<T>{5, 6, 7, 2},
                          std::vector<T>{2, 0, 1},
                          std::vector<T>{12, 19, 10, 7, 2},
                          {1},
                          {1},
                          {1},
                          {1},
                          {1}),
        ConvolutionBackpropParams(PartialShape {1, 1, 3},
                          PartialShape {1, 1, 3},
                          PartialShape {1, 1, 7},
                          IN_ET,
                          std::vector<T>{8, 5, 1},
                          std::vector<T>{2, 0, 1},
                          std::vector<T>{16, 10, 2, 0, 8, 5, 1},
                          {1},
                          {0},
                          {0},
                          {2},
                          {0}),
        ConvolutionBackpropParams(PartialShape {1, 1, 4},
                          PartialShape {1, 1, 3},
                          PartialShape {1, 1, 7},
                          IN_ET,
                          std::vector<T>{3, 9, 1, 2},
                          std::vector<T>{2, 0, 1},
                          std::vector<T>{18, 0, 5, 0, 13, 0, 1},
                          {2},
                          {2},
                          {2},
                          {2},
                          {0}),
        ConvolutionBackpropParams(PartialShape {1, 1, 2},
                          PartialShape {1, 2, 3},
                          PartialShape {1, 2, 4},
                          IN_ET,
                          std::vector<T>{10, 3},
                          std::vector<T>{
                                    // channel 1
                                    2, 0, 1,
                                    // channel 2
                                    1, 0, 2},
                          std::vector<T>{
                                    // channel 1
                                    20, 6, 10, 3,
                                    // channel 2
                                    10, 3, 20, 6},
                          {1},
                          {0},
                          {0},
                          {1},
                          {0}),
        ConvolutionBackpropParams(PartialShape {1, 2, 2},
                          PartialShape {2, 1, 3},
                          PartialShape {1, 1, 4},
                          IN_ET,
                          std::vector<T>{
                                    // channel 1
                                    4, 7,
                                    // channel 2
                                    5, 5},
                          std::vector<T>{
                                    // filter 1
                                    2, 0, 1,
                                    // filter 2
                                    1, 0, 2},
                          std::vector<T>{13, 19, 14, 17},
                          {1},
                          {0},
                          {0},
                          {1},
                          {0}),
        ConvolutionBackpropParams(PartialShape {2, 1, 2},
                          PartialShape {1, 1, 3},
                          PartialShape {2, 1, 4},
                          IN_ET,
                          std::vector<T>{
                                    // batch 1
                                    1, 3,
                                    // batch 2
                                    2, 2},
                          std::vector<T>{2, 0, 1},
                          std::vector<T>{
                                    // batch 1
                                    2, 6, 1, 3,
                                    // batch 2
                                    4, 4, 2, 2},
                          {1},
                          {0},
                          {0},
                          {1},
                          {0}),
// --------------------- 2D ConvolutionBackprop ------------------------------------------
        ConvolutionBackpropParams(PartialShape {1, 1, 2, 2},
                          PartialShape {1, 1, 3, 3},
                          PartialShape {1, 1, 4, 4},
                          IN_ET,
                          std::vector<T>{
                                    1, 3,
                                    7, 5},
                          std::vector<T>{
                                    1, 2, 3,
                                    0, 1, 0,
                                    3, 2, 1},
                          std::vector<T>{
                                    1, 5, 9, 9,
                                    7, 20, 34, 15,
                                    3, 18, 12, 3,
                                    21, 29, 17, 5},
                          {1, 1},
                          {0, 0},
                          {0, 0},
                          {1, 1},
                          {0, 0}),
        ConvolutionBackpropParams(PartialShape {1, 1, 2, 2},
                          PartialShape {1, 1, 3, 3},
                          PartialShape {1, 1, 3, 3},
                          IN_ET,
                          std::vector<T>{
                                    1, 3,
                                    7, 5},
                          std::vector<T>{
                                    1, 2, 3,
                                    1, 1, 1,
                                    3, 2, 1},
                          std::vector<T>{
                                    23, 35, 18,
                                    23, 19, 8,
                                    29, 17, 5},
                          {1, 1},
                          {1, 1},
                          {1, 1},
                          {1, 1},
                          {1, 1}),
        ConvolutionBackpropParams(PartialShape {1, 1, 4, 4},
                          PartialShape {1, 1, 3, 3},
                          PartialShape {1, 1, 4, 4},
                          IN_ET,
                          std::vector<T>{
                                    1, 3, 5, 7,
                                    7, 5, 3, 1,
                                    2, 4, 6, 8,
                                    8, 6, 4, 2},
                          std::vector<T>{
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2},
                          std::vector<T>{
                                    20, 37, 27, 18,
                                    22, 40, 60, 52,
                                    41, 69, 49, 31,
                                    18, 26, 34, 22},
                          {1, 1},
                          {1, 1},
                          {1, 1},
                          {1, 1},
                          {0, 0}),
        ConvolutionBackpropParams(PartialShape {1, 1, 2, 2},
                          PartialShape {1, 1, 3, 3},
                          PartialShape {1, 1, 5, 5},
                          IN_ET,
                          std::vector<T>{
                                    2, 5,
                                    4, 3},
                          std::vector<T>{
                                    1, 2, 3,
                                    1, 1, 1,
                                    3, 2, 1},
                          std::vector<T>{
                                    2, 4, 11, 10, 15,
                                    2, 2, 7, 5, 5,
                                    10, 12, 32, 16, 14,
                                    4, 4, 7, 3, 3,
                                    12, 8, 13, 6, 3},
                          {2, 2},
                          {0, 0},
                          {0, 0},
                          {1, 1},
                          {0, 0}),
        ConvolutionBackpropParams(PartialShape {1, 1, 2, 2},
                          PartialShape {1, 1, 3, 3},
                          PartialShape {1, 1, 6, 6},
                          IN_ET,
                          std::vector<T>{
                                    2, 3,
                                    4, 3},
                          std::vector<T>{
                                    1, 2, 3,
                                    1, 1, 1,
                                    3, 2, 1},
                          std::vector<T>{
                                    2, 3, 4, 6, 6, 9,
                                    4, 3, 8, 6, 12, 9,
                                    2, 3, 2, 3, 2, 3,
                                    4, 3, 4, 3, 4, 3,
                                    6, 9, 4, 6, 2, 3,
                                    12, 9, 8, 6, 4, 3},
                          {1, 1},
                          {0, 0},
                          {0, 0},
                          {2, 2},
                          {0, 0}),
        ConvolutionBackpropParams(PartialShape {1, 1, 3, 3},
                          PartialShape {1, 1, 3, 3},
                          PartialShape {1, 1, 5, 5},
                          IN_ET,
                          std::vector<T>{
                                    1, 3, 5,
                                    7, 5, 3,
                                    2, 4, 6},
                          std::vector<T>{
                                    1, 2, 3,
                                    1, 1, 1,
                                    3, 2, 1},
                          std::vector<T>{
                                    23, 0, 43, 0, 29,
                                    0, 0, 0, 0, 0,
                                    31, 0, 57, 0, 45,
                                    0, 0, 0, 0, 0,
                                    35, 0, 38, 0, 21},
                          {2, 2},
                          {2, 2},
                          {2, 2},
                          {2, 2},
                          {0, 0}),
// --------------------- 3D convolution ------------------------------------------
        ConvolutionBackpropParams(PartialShape {1, 1, 4, 4, 4},
                          PartialShape {1, 1, 3, 3, 3},
                          PartialShape {1, 1, 4, 4, 4},
                          IN_ET,
                          std::vector<T>{
                                    // depth: 1
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3,
                                    // depth: 2
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3,
                                    // depth: 3
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3,
                                    // depth: 4
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3},
                          std::vector<T>{
                                    // depth: 1
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 2
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 3
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2},
                          std::vector<T>{
                                    // depth: 1
                                     12, 30, 36, 24,
                                     26, 42, 42, 30,
                                     34, 56, 54, 50,
                                     14, 18, 24, 16,
                                     // depth: 2
                                     18, 45, 54, 36,
                                     39, 63, 63, 45,
                                     51, 84, 81, 75,
                                     21, 27, 36, 24,
                                     // depth: 3
                                     18, 45, 54, 36,
                                     39, 63, 63, 45,
                                     51, 84, 81, 75,
                                     21, 27, 36, 24,
                                     // depth: 4
                                     12, 30, 36, 24,
                                     26, 42, 42, 30,
                                     34, 56, 54, 50,
                                     14, 18, 24, 16},
                          {1, 1, 1},
                          {1, 1, 1},
                          {1, 1, 1},
                          {1, 1, 1},
                          {0, 0, 0}),
        ConvolutionBackpropParams(PartialShape {1, 1, 4, 4, 4},
                          PartialShape {1, 1, 3, 3, 3},
                          PartialShape {1, 1, 7, 7, 7},
                          IN_ET,
                          std::vector<T>{
                                    // depth: 1
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3,
                                    // depth: 2
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3,
                                    // depth: 3
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3,
                                    // depth: 4
                                    1, 3, 2, 1,
                                    1, 3, 3, 1,
                                    2, 1, 1, 3,
                                    3, 2, 3, 3},
                          std::vector<T>{
                                    // depth: 1
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 2
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 3
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2},
                          std::vector<T>{
                                    // depth: 1
                                    12, 0, 30, 0, 36, 0, 24,
                                    0, 0, 0, 0, 0, 0, 0,
                                    26, 0, 42, 0, 42, 0, 30,
                                    0, 0, 0, 0, 0, 0, 0,
                                    34, 0, 56, 0, 54, 0, 50,
                                    0, 0, 0, 0, 0, 0, 0,
                                    14, 0, 18, 0, 24, 0, 16,
                                    // depth: 2
                                    0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0,
                                    // depth: 3
                                    18, 0, 45, 0, 54, 0, 36,
                                    0, 0, 0, 0, 0, 0, 0,
                                    39, 0, 63, 0, 63, 0, 45,
                                    0, 0, 0, 0, 0, 0, 0,
                                    51, 0, 84, 0, 81, 0, 75,
                                    0, 0, 0, 0, 0, 0, 0,
                                    21, 0, 27, 0, 36, 0, 24,
                                    // depth: 4
                                    0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0,
                                    // depth: 5
                                    18, 0, 45, 0, 54, 0, 36,
                                    0, 0, 0, 0, 0, 0, 0,
                                    39, 0, 63, 0, 63, 0, 45,
                                    0, 0, 0, 0, 0, 0, 0,
                                    51, 0, 84, 0, 81, 0, 75,
                                    0, 0, 0, 0, 0, 0, 0,
                                    21, 0, 27, 0, 36, 0, 24,
                                    // depth: 6
                                    0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0, 0,
                                    // depth: 7
                                    12, 0, 30, 0, 36, 0, 24,
                                    0, 0, 0, 0, 0, 0, 0,
                                    26, 0, 42, 0, 42, 0, 30,
                                    0, 0, 0, 0, 0, 0, 0,
                                    34, 0, 56, 0, 54, 0, 50,
                                    0, 0, 0, 0, 0, 0, 0,
                                    14, 0, 18, 0, 24, 0, 16},
                          {2, 2, 2},
                          {2, 2, 2},
                          {2, 2, 2},
                          {2, 2, 2},
                          {0, 0, 0}),
        ConvolutionBackpropParams(PartialShape {1, 1, 2, 2, 2},
                          PartialShape {1, 2, 3, 3, 3},
                          PartialShape {1, 2, 4, 4, 4},
                          IN_ET,
                          std::vector<T>{
                                    // depth: 1
                                    1, 8,
                                    1, 3,
                                    // depth: 2
                                    1, 7,
                                    3, 8},
                          std::vector<T>{
                                    // -- channel 1 --
                                    // depth: 1
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 2
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 3
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // -- channel 2 --
                                    // depth: 1
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 2
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 3
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2},
                          std::vector<T>{
                                    // -- channel 1 --
                                    // depth: 1
                                    1, 10, 19, 24,
                                    1, 6, 17, 9,
                                    2, 18, 13, 16,
                                    2, 7, 5, 6,
                                    // depth: 2
                                    2, 19, 36, 45,
                                    4, 21, 49, 33,
                                    4, 36, 30, 30,
                                    8, 26, 19, 22,
                                    // depth: 3
                                    2, 19, 36, 45,
                                    4, 21, 49, 33,
                                    4, 36, 30, 30,
                                    8, 26, 19, 22,
                                    // depth: 4
                                    1, 9, 17, 21,
                                    3, 15, 32, 24,
                                    2, 18, 17, 14,
                                    6, 19, 14, 16,
                                    // -- channel 2 --
                                    // depth: 1
                                    1, 10, 19, 24,
                                    1, 6, 17, 9,
                                    2, 18, 13, 16,
                                    2, 7, 5, 6,
                                    // depth: 2
                                    2, 19, 36, 45,
                                    4, 21, 49, 33,
                                    4, 36, 30, 30,
                                    8, 26, 19, 22,
                                    // depth: 3
                                    2, 19, 36, 45,
                                    4, 21, 49, 33,
                                    4, 36, 30, 30,
                                    8, 26, 19, 22,
                                    // depth: 4
                                    1, 9, 17, 21,
                                    3, 15, 32, 24,
                                    2, 18, 17, 14,
                                    6, 19, 14, 16},
                          {1, 1, 1},
                          {0, 0, 0},
                          {0, 0, 0},
                          {1, 1, 1},
                          {0, 0, 0}),
        ConvolutionBackpropParams(PartialShape {1, 2, 2, 2, 2},
                          PartialShape {2, 1, 3, 3, 3},
                          PartialShape {1, 1, 4, 4, 4},
                          IN_ET,
                          std::vector<T>{
                                    // -- in 1 --
                                    // depth: 1
                                    1, 3,
                                    2, 5,
                                    // depth: 2
                                    1, 0,
                                    3, 6,
                                    // -- in 2 --
                                    // depth: 1
                                    1, 3,
                                    2, 5,
                                    // depth: 2
                                    3, 0,
                                    1, 8},
                          std::vector<T>{
                                    // -- filter 1 --
                                    // depth: 1
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 2
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 3
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // -- filter 2 --
                                    // depth: 1
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 2
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 3
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2},
                          std::vector<T>{
                                    // depth: 1
                                     2, 10, 18, 18,
                                     4, 20, 38, 30,
                                     4, 18, 20, 12,
                                     8, 24, 18, 20,
                                     // depth: 2
                                     6, 18, 30, 18,
                                     8, 46, 78, 72,
                                     12, 26, 42, 12,
                                     16, 56, 40, 48,
                                     // depth: 3
                                     6, 18, 30, 18,
                                     8, 46, 78, 72,
                                     12, 26, 42, 12,
                                     16, 56, 40, 48,
                                     // depth: 4
                                     4, 8, 12, 0,
                                     4, 26, 40, 42,
                                     8, 8, 22, 0,
                                     8, 32, 22, 28},
                          {1, 1, 1},
                          {0, 0, 0},
                          {0, 0, 0},
                          {1, 1, 1},
                          {0, 0, 0}),
        ConvolutionBackpropParams(PartialShape {2, 1, 2, 2, 2},
                          PartialShape {1, 1, 3, 3, 3},
                          PartialShape {2, 1, 4, 4, 4},
                          IN_ET,
                          std::vector<T>{
                                    // -- batch 1 --
                                    // depth: 1
                                    1, 3,
                                    2, 5,
                                    // depth: 2
                                    1, 0,
                                    6, 4,
                                    // -- batch 2 --
                                    // depth: 1
                                    1, 5,
                                    2, 8,
                                    // depth: 2
                                    2, 1,
                                    0, 5},
                          std::vector<T>{
                                    // depth: 1
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 2
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2,
                                    // depth: 3
                                    1, 2, 3,
                                    0, 1, 0,
                                    2, 1, 2},
                          std::vector<T>{
                                    // -- batch 1 --
                                     // depth: 1
                                     1, 5, 9, 9,
                                     2, 10, 19, 15,
                                     2, 9, 10, 6,
                                     4, 12, 9, 10,
                                     // depth: 2
                                     2, 7, 12, 9,
                                     8, 27, 45, 27,
                                     4, 16, 16, 6,
                                     16, 26, 25, 18,
                                     // depth: 3
                                     2, 7, 12, 9,
                                     8, 27, 45, 27,
                                     4, 16, 16, 6,
                                     16, 26, 25, 18,
                                     // depth: 4
                                     1, 2, 3, 0,
                                     6, 17, 26, 12,
                                     2, 7, 6, 0,
                                     12, 14, 16, 8,
                                     // -- batch 2 --
                                     // depth: 1
                                     1, 7, 13, 15,
                                     2, 13, 27, 24,
                                     2, 13, 15, 10,
                                     4, 18, 12, 16,
                                     // depth: 2
                                     3, 12, 21, 18,
                                     2, 20, 38, 39,
                                     6, 17, 25, 12,
                                     4, 28, 17, 26,
                                     // depth: 3
                                     3, 12, 21, 18,
                                     2, 20, 38, 39,
                                     6, 17, 25, 12,
                                     4, 28, 17, 26,
                                     // depth: 4
                                     2, 5, 8, 3,
                                     0, 7, 11, 15,
                                     4, 4, 10, 2,
                                     0, 10, 5, 10},
                          {1, 1, 1},
                          {0, 0, 0},
                          {0, 0, 0},
                          {1, 1, 1},
                          {0, 0, 0})
    };
    return convolutionBackpropParams;
}

std::vector<ConvolutionBackpropParams> generateConvolutionBackpropCombinedParams() {
    const std::vector<std::vector<ConvolutionBackpropParams>> convolutionBackpropTypeParams {
        generateConvolutionBackpropFloatParams<element::Type_t::f64>(),
        generateConvolutionBackpropFloatParams<element::Type_t::f32>(),
        generateConvolutionBackpropFloatParams<element::Type_t::f16>(),
        generateConvolutionBackpropFloatParams<element::Type_t::bf16>(),
        generateConvolutionBackpropFloatParams<element::Type_t::i64>(),
        generateConvolutionBackpropFloatParams<element::Type_t::i32>(),
        generateConvolutionBackpropFloatParams<element::Type_t::i16>(),
        generateConvolutionBackpropUintParams<element::Type_t::i8>(),
        generateConvolutionBackpropUintParams<element::Type_t::u64>(),
        generateConvolutionBackpropUintParams<element::Type_t::u32>(),
        generateConvolutionBackpropUintParams<element::Type_t::u16>(),
        generateConvolutionBackpropUintParams<element::Type_t::u8>(),
        };
    std::vector<ConvolutionBackpropParams> combinedParams;

    for (const auto& params : convolutionBackpropTypeParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionBackprop_With_Hardcoded_Refs, ReferenceConvolutionBackpropLayerTest,
    testing::ValuesIn(generateConvolutionBackpropCombinedParams()), ReferenceConvolutionBackpropLayerTest::getTestCaseName);

} // namespace
