// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <functional_test_utils/skip_tests_config.hpp>

#include "openvino/op/pad.hpp"
#include "openvino/op/constant.hpp"
#include "base_reference_test.hpp"

using namespace reference_tests;
using namespace ov;

namespace {
struct PadParams {
    PadParams(
        const Tensor& inputData, const Tensor& padsBegin, const Tensor& padsEnd,
        const Tensor& expectedOutput, op::PadMode padMode, const Tensor& constantValue,
        const std::string& testcaseName = "") :
        inputData(inputData), padsBegin(padsBegin), padsEnd(padsEnd),
        expectedOutput(expectedOutput), padMode(padMode),
        useConstValue{true}, constantValue(constantValue),
        testcaseName(testcaseName) {}

    PadParams(
        const Tensor& inputData, const Tensor& padsBegin, const Tensor& padsEnd,
        const Tensor& expectedOutput, op::PadMode padMode,
        const std::string& testcaseName = "") :
        inputData(inputData), padsBegin(padsBegin), padsEnd(padsEnd),
        expectedOutput(expectedOutput), padMode(padMode),
        testcaseName(testcaseName) {}

    Tensor inputData;
    Tensor padsBegin;
    Tensor padsEnd;
    Tensor expectedOutput;
    op::PadMode padMode;
    bool useConstValue{false};
    Tensor constantValue;
    std::string testcaseName;
};

class ReferencePadTest : public testing::TestWithParam<PadParams>, public CommonReferenceTest {
public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        auto params = GetParam();
        function = CreateFunction(params);
        inputData = {params.inputData.data};
        refOutData = {params.expectedOutput.data};
    }

    static std::string getTestCaseName(const testing::TestParamInfo<PadParams>& obj) {
        auto param = obj.param;
        std::ostringstream result;
        result << "iType=" << param.inputData.type;
        result << "_iShape=" << param.inputData.shape;
        result << "_pbType=" << param.padsBegin.type;
        result << "_pbShape=" << param.padsBegin.shape;
        result << "_peType=" << param.padsEnd.type;
        result << "_peShape=" << param.padsEnd.shape;
        result << "_oType=" << param.expectedOutput.type;
        result << "_oShape=" << param.expectedOutput.shape;
        result << "_=" << param.testcaseName;
        return result.str();
    }

private:
    static std::shared_ptr<Model> CreateFunction(const PadParams& params) {
        const auto data = std::make_shared<op::v0::Parameter>(params.inputData.type,
                                                              params.inputData.shape);
        const auto padsBegin = op::v0::Constant::create(params.padsBegin.type,
                                                        params.padsBegin.shape,
                                                        params.padsBegin.data.data());
        const auto padsEnd = op::v0::Constant::create(params.padsEnd.type,
                                                      params.padsEnd.shape,
                                                      params.padsEnd.data.data());
        const auto f = [&] {
            if (params.useConstValue) {
                // pad_value should be used only in CONSTANT mode
                const auto padVal = op::v0::Constant::create(params.constantValue.type,
                                                             params.constantValue.shape,
                                                             params.constantValue.data.data());
                return std::make_shared<Model>(std::make_shared<op::v1::Pad>(data,
                                                                                padsBegin,
                                                                                padsEnd,
                                                                                padVal,
                                                                                params.padMode),
                                                  ParameterVector{data});
            }

            return std::make_shared<Model>(std::make_shared<op::v1::Pad>(data,
                                                                            padsBegin,
                                                                            padsEnd,
                                                                            params.padMode),
                                              ParameterVector{data});
        }();
        return f;
    }
};

TEST_P(ReferencePadTest, CompareWithRefs) {
    Exec();
}

class ReferencePadTestParamsTooLarge : public ReferencePadTest {};

TEST_P(ReferencePadTestParamsTooLarge, CompareWithRefs) {
    EXPECT_ANY_THROW(Exec());
}

class ReferencePadTestParamsOk : public ReferencePadTest {};

TEST_P(ReferencePadTestParamsOk, CompareWithRefs) {
    EXPECT_NO_THROW(Exec());
}

class ReferencePadTestNonConstPadsBeginPadsEndPadVal : public ReferencePadTest {
public:
    void SetUp() override {
        SKIP_IF_CURRENT_TEST_IS_DISABLED();
        auto params = GetParam();
        function = CreateFunction(params);
        if (params.useConstValue)
            inputData = {params.inputData.data, params.padsBegin.data, params.padsEnd.data, params.constantValue.data};
        else
            inputData = {params.inputData.data, params.padsBegin.data, params.padsEnd.data};
        refOutData = {params.expectedOutput.data};
    }

private:
    static std::shared_ptr<Model> CreateFunction(const PadParams& params) {
        const auto data = std::make_shared<op::v0::Parameter>(params.inputData.type,
                                                              params.inputData.shape);
        const auto padsBegin = std::make_shared<op::v0::Parameter>(params.padsBegin.type,
                                                                   params.padsBegin.shape);
        const auto padsEnd = std::make_shared<op::v0::Parameter>(params.padsEnd.type,
                                                                 params.padsEnd.shape);
        const auto f = [&] {
            if (params.useConstValue) {
                // pad_value should be used only in CONSTANT mode
                const auto padVal = std::make_shared<op::v0::Parameter>(params.constantValue.type,
                                                                        params.constantValue.shape);
                return std::make_shared<Model>(std::make_shared<op::v1::Pad>(data,
                                                                                padsBegin,
                                                                                padsEnd,
                                                                                padVal,
                                                                                params.padMode),
                                                  ParameterVector{data, padsBegin, padsEnd, padVal});
            }

            return std::make_shared<Model>(std::make_shared<op::v1::Pad>(data,
                                                                            padsBegin,
                                                                            padsEnd,
                                                                            params.padMode),
                                              ParameterVector{data, padsBegin, padsEnd});
        }();
        return f;
    }
};

TEST_P(ReferencePadTestNonConstPadsBeginPadsEndPadVal, CompareWithRefs) {
    Exec();
}

class ReferencePadTestNonConstPadsBeginPadsEndPadValTooLarge : public ReferencePadTestNonConstPadsBeginPadsEndPadVal {};

TEST_P(ReferencePadTestNonConstPadsBeginPadsEndPadValTooLarge, CompareWithRefs) {
    EXPECT_ANY_THROW(Exec());
}

class ReferencePadTestNonConstPadsBeginPadsEndPadValParamsOk : public ReferencePadTestNonConstPadsBeginPadsEndPadVal {};

TEST_P(ReferencePadTestNonConstPadsBeginPadsEndPadValParamsOk, CompareWithRefs) {
    EXPECT_NO_THROW(Exec());
}

template <element::Type_t ET, element::Type_t ET_INT>
std::vector<PadParams> generateParams() {
    using T = typename element_type_traits<ET>::value_type;
    using T_INT = typename element_type_traits<ET_INT>::value_type;
    std::vector<PadParams> params {
        PadParams(
            Tensor(ET, {6}, std::vector<T>{
                1, 2, 3, 4, 5, 6,
            }),
            Tensor(ET_INT, {1}, std::vector<T_INT>{4}),
            Tensor(ET_INT, {1}, std::vector<T_INT>{5}),
            Tensor(ET, {15}, std::vector<T>{
                2112, 2112, 2112, 2112, 1, 2, 3, 4, 5, 6, 2112, 2112, 2112, 2112, 2112,
            }),
            op::PadMode::CONSTANT,
            Tensor(ET, {}, std::vector<T>{2112}),
            "pad_1d_constant_const_value_provided_0"),
        PadParams(
            Tensor(ET, {6}, std::vector<T>{
                1, 2, 3, 4, 5, 6,
            }),
            Tensor(ET_INT, {1}, std::vector<T_INT>{4}),
            Tensor(ET_INT, {1}, std::vector<T_INT>{0}),
            Tensor(ET, {10}, std::vector<T>{
                2112, 2112, 2112, 2112, 1, 2, 3, 4, 5, 6,
            }),
            op::PadMode::CONSTANT,
            Tensor(ET, {}, std::vector<T>{2112}),
            "pad_1d_constant_const_value_provided_1"),
        PadParams(
            Tensor(ET, {6}, std::vector<T>{
                1, 2, 3, 4, 5, 6,
            }),
            Tensor(ET_INT, {1}, std::vector<T_INT>{0}),
            Tensor(ET_INT, {1}, std::vector<T_INT>{3}),
            Tensor(ET, {9}, std::vector<T>{
                1, 2, 3, 4, 5, 6, 2112, 2112, 2112,
            }),
            op::PadMode::CONSTANT,
            Tensor(ET, {}, std::vector<T>{2112}),
            "pad_1d_constant_const_value_provided_2"),

        PadParams(
            Tensor(ET, {6}, std::vector<T>{
                1, 2, 3, 4, 5, 6,
            }),
            Tensor(ET_INT, {1}, std::vector<T_INT>{4}),
            Tensor(ET_INT, {1}, std::vector<T_INT>{5}),
            Tensor(ET, {15}, std::vector<T>{
                0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0,
            }),
            op::PadMode::CONSTANT,
            "pad_1d_constant_use_default_const_0"),
        PadParams(
            Tensor(ET, {6}, std::vector<T>{
                1, 2, 3, 4, 5, 6,
            }),
            Tensor(ET_INT, {1}, std::vector<T_INT>{4}),
            Tensor(ET_INT, {1}, std::vector<T_INT>{0}),
            Tensor(ET, {10}, std::vector<T>{
                0, 0, 0, 0, 1, 2, 3, 4, 5, 6,
            }),
            op::PadMode::CONSTANT,
            "pad_1d_constant_use_default_const_1"),
        PadParams(
            Tensor(ET, {6}, std::vector<T>{
                1, 2, 3, 4, 5, 6,
            }),
            Tensor(ET_INT, {1}, std::vector<T_INT>{0}),
            Tensor(ET_INT, {1}, std::vector<T_INT>{3}),
            Tensor(ET, {9}, std::vector<T>{
                1, 2, 3, 4, 5, 6, 0, 0, 0,
            }),
            op::PadMode::CONSTANT,
            "pad_1d_constant_use_default_const_2"),

        PadParams(
            Tensor(ET, {2, 2}, std::vector<T>{
                1, 2, 3, 4,
            }),
            Tensor(ET_INT, {2}, std::vector<T_INT>{1, 2}),
            Tensor(ET_INT, {2}, std::vector<T_INT>{3, 4}),
            Tensor(ET, {6, 8}, std::vector<T>{
                2112, 2112, 2112, 2112, 2112, 2112, 2112, 2112,
                2112, 2112, 1,    2,    2112, 2112, 2112, 2112,
                2112, 2112, 3,    4,    2112, 2112, 2112, 2112,
                2112, 2112, 2112, 2112, 2112, 2112, 2112, 2112,
                2112, 2112, 2112, 2112, 2112, 2112, 2112, 2112,
                2112, 2112, 2112, 2112, 2112, 2112, 2112, 2112,
            }),
            op::PadMode::CONSTANT,
            Tensor(ET, {}, std::vector<T>{2112}),
            "pad_2d_constant_const_value_provided_0"),
        PadParams(
            Tensor(ET, {2, 2}, std::vector<T>{
                1, 2, 3, 4,
            }),
            Tensor(ET_INT, {2}, std::vector<T_INT>{1, 2}),
            Tensor(ET_INT, {2}, std::vector<T_INT>{0, 0}),
            Tensor(ET, {3, 4}, std::vector<T>{
                2112, 2112, 2112, 2112,
                2112, 2112, 1,    2,
                2112, 2112, 3,    4,
            }),
            op::PadMode::CONSTANT,
            Tensor(ET, {}, std::vector<T>{2112}),
            "pad_2d_constant_const_value_provided_1"),
        PadParams(
            Tensor(ET, {2, 2}, std::vector<T>{
                1, 2, 3, 4,
            }),
            Tensor(ET_INT, {2}, std::vector<T_INT>{0, 0}),
            Tensor(ET_INT, {2}, std::vector<T_INT>{1, 2}),
            Tensor(ET, {3, 4}, std::vector<T>{
                1,    2,    2112, 2112,
                3,    4,    2112, 2112,
                2112, 2112, 2112, 2112,
            }),
            op::PadMode::CONSTANT,
            Tensor(ET, {}, std::vector<T>{2112}),
            "pad_2d_constant_const_value_provided_2"),

        PadParams(
            Tensor(ET, {2, 2}, std::vector<T>{
                1, 2, 3, 4,
            }),
            Tensor(ET_INT, {2}, std::vector<T_INT>{1, 2}),
            Tensor(ET_INT, {2}, std::vector<T_INT>{3, 4}),
            Tensor(ET, {6, 8}, std::vector<T>{
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 1, 2, 0, 0, 0, 0,
                0, 0, 3, 4, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
            }),
            op::PadMode::CONSTANT,
            "pad_2d_constant_use_default_const_0"),
        PadParams(
            Tensor(ET, {2, 2}, std::vector<T>{
                1, 2, 3, 4,
            }),
            Tensor(ET_INT, {2}, std::vector<T_INT>{1, 2}),
            Tensor(ET_INT, {2}, std::vector<T_INT>{0, 0}),
            Tensor(ET, {3, 4}, std::vector<T>{
                0, 0, 0, 0,
                0, 0, 1, 2,
                0, 0, 3, 4,
            }),
            op::PadMode::CONSTANT,
            "pad_2d_constant_use_default_const_1"),
        PadParams(
            Tensor(ET, {2, 2}, std::vector<T>{
                1, 2, 3, 4,
            }),
            Tensor(ET_INT, {2}, std::vector<T_INT>{0, 0}),
            Tensor(ET_INT, {2}, std::vector<T_INT>{1, 2}),
            Tensor(ET, {3, 4}, std::vector<T>{
                1, 2, 0, 0,
                3, 4, 0, 0,
                0, 0, 0, 0,
            }),
            op::PadMode::CONSTANT,
            "pad_2d_constant_use_default_const_2"),

        PadParams(
            Tensor(ET, {6}, std::vector<T>{
                1, 2, 3, 4, 5, 6,
            }),
            Tensor(ET_INT, {1}, std::vector<T_INT>{2}),
            Tensor(ET_INT, {1}, std::vector<T_INT>{3}),
            Tensor(ET, {11}, std::vector<T>{
                1, 1, 1, 2, 3, 4, 5, 6, 6, 6, 6,
            }),
            op::PadMode::EDGE,
            "pad_1d_edge_0"),
        PadParams(
            Tensor(ET, {6}, std::vector<T>{
                1, 2, 3, 4, 5, 6,
            }),
            Tensor(ET_INT, {1}, std::vector<T_INT>{1}),
            Tensor(ET_INT, {1}, std::vector<T_INT>{0}),
            Tensor(ET, {7}, std::vector<T>{
                1, 1, 2, 3, 4, 5, 6,
            }),
            op::PadMode::EDGE,
            "pad_1d_edge_1"),
        PadParams(
            Tensor(ET, {6}, std::vector<T>{
                1, 2, 3, 4, 5, 6,
            }),
            Tensor(ET_INT, {1}, std::vector<T_INT>{0}),
            Tensor(ET_INT, {1}, std::vector<T_INT>{2}),
            Tensor(ET, {8}, std::vector<T>{
                1, 2, 3, 4, 5, 6, 6, 6,
            }),
            op::PadMode::EDGE,
            "pad_1d_edge_2"),

        PadParams(
            Tensor(ET, {2, 2}, std::vector<T>{
                1, 2, 3, 4,
            }),
            Tensor(ET_INT, {2}, std::vector<T_INT>{1, 2}),
            Tensor(ET_INT, {2}, std::vector<T_INT>{2, 1}),
            Tensor(ET, {5, 5}, std::vector<T>{
                1, 1, 1, 2, 2,
                1, 1, 1, 2, 2,
                3, 3, 3, 4, 4,
                3, 3, 3, 4, 4,
                3, 3, 3, 4, 4,
            }),
            op::PadMode::EDGE,
            "pad_2d_edge_0"),
        PadParams(
            Tensor(ET, {2, 2}, std::vector<T>{
                1, 2, 3, 4,
            }),
            Tensor(ET_INT, {2}, std::vector<T_INT>{1, 2}),
            Tensor(ET_INT, {2}, std::vector<T_INT>{0, 0}),
            Tensor(ET, {3, 4}, std::vector<T>{
                1, 1, 1, 2,
                1, 1, 1, 2,
                3, 3, 3, 4,
            }),
            op::PadMode::EDGE,
            "pad_2d_edge_1"),
        PadParams(
            Tensor(ET, {2, 2}, std::vector<T>{
                1, 2, 3, 4,
            }),
            Tensor(ET_INT, {2}, std::vector<T_INT>{0, 0}),
            Tensor(ET_INT, {2}, std::vector<T_INT>{2, 1}),
            Tensor(ET, {4, 3}, std::vector<T>{
                1, 2, 2,
                3, 4, 4,
                3, 4, 4,
                3, 4, 4,
            }),
            op::PadMode::EDGE,
            "pad_2d_edge_2"),

        PadParams(
            Tensor(ET, {6}, std::vector<T>{
                1, 2, 3, 4, 5, 6,
            }),
            Tensor(ET_INT, {1}, std::vector<T_INT>{2}),
            Tensor(ET_INT, {1}, std::vector<T_INT>{3}),
            Tensor(ET, {11}, std::vector<T>{
                3, 2, 1, 2, 3, 4, 5, 6, 5, 4, 3,
            }),
            op::PadMode::REFLECT,
            "pad_1d_reflect_0"),
        PadParams(
            Tensor(ET, {6}, std::vector<T>{
                1, 2, 3, 4, 5, 6,
            }),
            Tensor(ET_INT, {1}, std::vector<T_INT>{1}),
            Tensor(ET_INT, {1}, std::vector<T_INT>{0}),
            Tensor(ET, {7}, std::vector<T>{
                2, 1, 2, 3, 4, 5, 6,
            }),
            op::PadMode::REFLECT,
            "pad_1d_reflect_1"),
        PadParams(
            Tensor(ET, {6}, std::vector<T>{
                1, 2, 3, 4, 5, 6,
            }),
            Tensor(ET_INT, {1}, std::vector<T_INT>{0}),
            Tensor(ET_INT, {1}, std::vector<T_INT>{2}),
            Tensor(ET, {8}, std::vector<T>{
                1, 2, 3, 4, 5, 6, 5, 4,
            }),
            op::PadMode::REFLECT,
            "pad_1d_reflect_2"),

        PadParams(
            Tensor(ET, {3, 3}, std::vector<T>{
                1, 2, 3, 4, 5, 6, 7, 8, 9,
            }),
            Tensor(ET_INT, {2}, std::vector<T_INT>{1, 2}),
            Tensor(ET_INT, {2}, std::vector<T_INT>{2, 1}),
            Tensor(ET, {6, 6}, std::vector<T>{
                6, 5, 4, 5, 6, 5,
                3, 2, 1, 2, 3, 2,
                6, 5, 4, 5, 6, 5,
                9, 8, 7, 8, 9, 8,
                6, 5, 4, 5, 6, 5,
                3, 2, 1, 2, 3, 2,
            }),
            op::PadMode::REFLECT,
            "pad_2d_reflect_0"),
        PadParams(
            Tensor(ET, {3, 3}, std::vector<T>{
                1, 2, 3, 4, 5, 6, 7, 8, 9,
            }),
            Tensor(ET_INT, {2}, std::vector<T_INT>{1, 2}),
            Tensor(ET_INT, {2}, std::vector<T_INT>{0, 0}),
            Tensor(ET, {4, 5}, std::vector<T>{
                6, 5, 4, 5, 6,
                3, 2, 1, 2, 3,
                6, 5, 4, 5, 6,
                9, 8, 7, 8, 9,
            }),
            op::PadMode::REFLECT,
            "pad_2d_reflect_1"),
        PadParams(
            Tensor(ET, {3, 3}, std::vector<T>{
                1, 2, 3, 4, 5, 6, 7, 8, 9,
            }),
            Tensor(ET_INT, {2}, std::vector<T_INT>{0, 0}),
            Tensor(ET_INT, {2}, std::vector<T_INT>{2, 1}),
            Tensor(ET, {5, 4}, std::vector<T>{
                1, 2, 3, 2,
                4, 5, 6, 5,
                7, 8, 9, 8,
                4, 5, 6, 5,
                1, 2, 3, 2,
            }),
            op::PadMode::REFLECT,
            "pad_2d_reflect_2"),

        PadParams(
            Tensor(ET, {6}, std::vector<T>{
                1, 2, 3, 4, 5, 6,
            }),
            Tensor(ET_INT, {1}, std::vector<T_INT>{2}),
            Tensor(ET_INT, {1}, std::vector<T_INT>{3}),
            Tensor(ET, {11}, std::vector<T>{
                2, 1, 1, 2, 3, 4, 5, 6, 6, 5, 4,
            }),
            op::PadMode::SYMMETRIC,
            "pad_1d_symmetric_0"),
        PadParams(
            Tensor(ET, {6}, std::vector<T>{
                1, 2, 3, 4, 5, 6,
            }),
            Tensor(ET_INT, {1}, std::vector<T_INT>{1}),
            Tensor(ET_INT, {1}, std::vector<T_INT>{0}),
            Tensor(ET, {7}, std::vector<T>{
                1, 1, 2, 3, 4, 5, 6,
            }),
            op::PadMode::SYMMETRIC,
            "pad_1d_symmetric_1"),
        PadParams(
            Tensor(ET, {6}, std::vector<T>{
                1, 2, 3, 4, 5, 6,
            }),
            Tensor(ET_INT, {1}, std::vector<T_INT>{0}),
            Tensor(ET_INT, {1}, std::vector<T_INT>{2}),
            Tensor(ET, {8}, std::vector<T>{
                1, 2, 3, 4, 5, 6, 6, 5,
            }),
            op::PadMode::SYMMETRIC,
            "pad_1d_symmetric_2"),

        PadParams(
            Tensor(ET, {3, 3}, std::vector<T>{
                1, 2, 3, 4, 5, 6, 7, 8, 9,
            }),
            Tensor(ET_INT, {2}, std::vector<T_INT>{1, 2}),
            Tensor(ET_INT, {2}, std::vector<T_INT>{2, 1}),
            Tensor(ET, {6, 6}, std::vector<T>{
                2, 1, 1, 2, 3, 3,
                2, 1, 1, 2, 3, 3,
                5, 4, 4, 5, 6, 6,
                8, 7, 7, 8, 9, 9,
                8, 7, 7, 8, 9, 9,
                5, 4, 4, 5, 6, 6,
            }),
            op::PadMode::SYMMETRIC,
            "pad_2d_symmetric_0"),
        PadParams(
            Tensor(ET, {3, 3}, std::vector<T>{
                1, 2, 3, 4, 5, 6, 7, 8, 9,
            }),
            Tensor(ET_INT, {2}, std::vector<T_INT>{1, 2}),
            Tensor(ET_INT, {2}, std::vector<T_INT>{0, 0}),
            Tensor(ET, {4, 5}, std::vector<T>{
                2, 1, 1, 2, 3,
                2, 1, 1, 2, 3,
                5, 4, 4, 5, 6,
                8, 7, 7, 8, 9,
            }),
            op::PadMode::SYMMETRIC,
            "pad_2d_symmetric_1"),
        PadParams(
            Tensor(ET, {3, 3}, std::vector<T>{
                1, 2, 3, 4, 5, 6, 7, 8, 9,
            }),
            Tensor(ET_INT, {2}, std::vector<T_INT>{0, 0}),
            Tensor(ET_INT, {2}, std::vector<T_INT>{2, 1}),
            Tensor(ET, {5, 4}, std::vector<T>{
                1, 2, 3, 3,
                4, 5, 6, 6,
                7, 8, 9, 9,
                7, 8, 9, 9,
                4, 5, 6, 6,
            }),
            op::PadMode::SYMMETRIC,
            "pad_2d_symmetric"),

        PadParams(
            Tensor(ET, {6}, std::vector<T>{
                1, 2, 3, 4, 5, 6,
            }),
            Tensor(ET_INT, {1}, std::vector<T_INT>{4}),
            Tensor(ET_INT, {1}, std::vector<T_INT>{5}),
            Tensor(ET, {15}, std::vector<T>{
                2112, 2112, 2112, 2112, 1, 2, 3, 4, 5, 6, 2112, 2112, 2112, 2112, 2112,
            }),
            op::PadMode::CONSTANT,
            Tensor(ET, {}, std::vector<T>{2112}),
            "pad_exterior_1d"),

        PadParams(
            Tensor(ET, {6}, std::vector<T>{
                1, 2, 3, 4, 5, 6,
            }),
            Tensor(ET_INT, {1}, std::vector<T_INT>{4}),
            Tensor(ET_INT, {1}, std::vector<T_INT>{-2}),
            Tensor(ET, {8}, std::vector<T>{
                2112, 2112, 2112, 2112, 1, 2, 3, 4,
            }),
            op::PadMode::CONSTANT,
            Tensor(ET, {}, std::vector<T>{2112}),
            "pad_negative_exterior_1d"),

        PadParams(
            Tensor(ET, {6}, std::vector<T>{
                1, 2, 3, 4, 5, 6,
            }),
            Tensor(ET_INT, {1}, std::vector<T_INT>{4}),
            Tensor(ET_INT, {1}, std::vector<T_INT>{-7}),
            Tensor(ET, {3}, std::vector<T>{
                2112, 2112, 2112,
            }),
            op::PadMode::CONSTANT,
            Tensor(ET, {}, std::vector<T>{2112}),
            "pad_negative_exterior_1d_check_limits"),

        PadParams(
            Tensor(ET, {6}, std::vector<T>{
                1, 2, 3, 4, 5, 6,
            }),
            Tensor(ET_INT, {1}, std::vector<T_INT>{2}),
            Tensor(ET_INT, {1}, std::vector<T_INT>{3}),
            Tensor(ET, {11}, std::vector<T>{
                1, 1, 1, 2, 3, 4, 5, 6, 6, 6, 6,
            }),
            op::PadMode::EDGE,
            Tensor(ET, {}, std::vector<T>{2112}),
            "pad_edge_1d"),

        PadParams(
            Tensor(ET, {6}, std::vector<T>{
                1, 2, 3, 4, 5, 6,
            }),
            Tensor(ET_INT, {1}, std::vector<T_INT>{2}),
            Tensor(ET_INT, {1}, std::vector<T_INT>{-3}),
            Tensor(ET, {5}, std::vector<T>{
                1, 1, 1, 2, 3,
            }),
            op::PadMode::EDGE,
            Tensor(ET, {}, std::vector<T>{2112}),
            "pad_edge_1d_top_neg"),

        PadParams(
            Tensor(ET, {6}, std::vector<T>{
                1, 2, 3, 4, 5, 6,
            }),
            Tensor(ET_INT, {1}, std::vector<T_INT>{2}),
            Tensor(ET_INT, {1}, std::vector<T_INT>{-7}),
            Tensor(ET, {1}, std::vector<T>{
                1,
            }),
            op::PadMode::EDGE,
            Tensor(ET, {}, std::vector<T>{2112}),
            "pad_edge_1d_top_neg_bigger_than_tensor"),

        PadParams(
            Tensor(ET, {6}, std::vector<T>{
                1, 2, 3, 4, 5, 6,
            }),
            Tensor(ET_INT, {1}, std::vector<T_INT>{-2}),
            Tensor(ET_INT, {1}, std::vector<T_INT>{3}),
            Tensor(ET, {7}, std::vector<T>{
                3, 4, 5, 6, 6, 6, 6,
            }),
            op::PadMode::EDGE,
            Tensor(ET, {}, std::vector<T>{2112}),
            "pad_edge_1d_bottom_neg"),

        PadParams(
            Tensor(ET, {6}, std::vector<T>{
                1, 2, 3, 4, 5, 6,
            }),
            Tensor(ET_INT, {1}, std::vector<T_INT>{-7}),
            Tensor(ET_INT, {1}, std::vector<T_INT>{3}),
            Tensor(ET, {2}, std::vector<T>{
                6, 6,
            }),
            op::PadMode::EDGE,
            Tensor(ET, {}, std::vector<T>{2112}),
            "pad_edge_1d_bottom_neg_bigger_than_tensor"),

        PadParams(
            Tensor(ET, {3, 4}, std::vector<T>{
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            }),
            Tensor(ET_INT, {2}, std::vector<T_INT>{2, 3}),
            Tensor(ET_INT, {2}, std::vector<T_INT>{1, 2}),
            Tensor(ET, {6, 9}, std::vector<T>{
                1, 1, 1, 1, 2,  3,  4,  4,  4,
                1, 1, 1, 1, 2,  3,  4,  4,  4,
                1, 1, 1, 1, 2,  3,  4,  4,  4,
                5, 5, 5, 5, 6,  7,  8,  8,  8,
                9, 9, 9, 9, 10, 11, 12, 12, 12,
                9, 9, 9, 9, 10, 11, 12, 12, 12,
            }),
            op::PadMode::EDGE,
            Tensor(ET, {}, std::vector<T>{2112}),
            "pad_edge_2d"),

        PadParams(
            Tensor(ET, {3, 4}, std::vector<T>{
                1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            }),
            Tensor(ET_INT, {2}, std::vector<T_INT>{2, -1}),
            Tensor(ET_INT, {2}, std::vector<T_INT>{1, 2}),
            Tensor(ET, {6, 5}, std::vector<T>{
                2,  3,  4,  4,  4,
                2,  3,  4,  4,  4,
                2,  3,  4,  4,  4,
                6,  7,  8,  8,  8,
                10, 11, 12, 12, 12,
                10, 11, 12, 12, 12,
            }),
            op::PadMode::EDGE,
            Tensor(ET, {}, std::vector<T>{2112}),
            "pad_edge_2d_with_neg"),

        PadParams(
            Tensor(ET, {6}, std::vector<T>{
                1, 2, 3, 4, 5, 6,
            }),
            Tensor(ET_INT, {1}, std::vector<T_INT>{2}),
            Tensor(ET_INT, {1}, std::vector<T_INT>{3}),
            Tensor(ET, {11}, std::vector<T>{
                3, 2, 1, 2, 3, 4, 5, 6, 5, 4, 3,
            }),
            op::PadMode::REFLECT,
            Tensor(ET, {}, std::vector<T>{2112}),
            "pad_reflect_1d"),

        PadParams(
            Tensor(ET, {6}, std::vector<T>{
                1, 2, 3, 4, 5, 6,
            }),
            Tensor(ET_INT, {1}, std::vector<T_INT>{2}),
            Tensor(ET_INT, {1}, std::vector<T_INT>{-3}),
            Tensor(ET, {5}, std::vector<T>{
                3, 2, 1, 2, 3,
            }),
            op::PadMode::REFLECT,
            Tensor(ET, {}, std::vector<T>{2112}),
            "pad_reflect_1d_top_neg"),

        PadParams(
            Tensor(ET, {6}, std::vector<T>{
                1, 2, 3, 4, 5, 6,
            }),
            Tensor(ET_INT, {1}, std::vector<T_INT>{2}),
            Tensor(ET_INT, {1}, std::vector<T_INT>{-7}),
            Tensor(ET, {1}, std::vector<T>{
                3,
            }),
            op::PadMode::REFLECT,
            Tensor(ET, {}, std::vector<T>{2112}),
            "pad_reflect_1d_top_neg_bigger_than_tensor"),

        PadParams(
            Tensor(ET, {6}, std::vector<T>{
                1, 2, 3, 4, 5, 6,
            }),
            Tensor(ET_INT, {1}, std::vector<T_INT>{-2}),
            Tensor(ET_INT, {1}, std::vector<T_INT>{3}),
            Tensor(ET, {7}, std::vector<T>{
                3, 4, 5, 6, 5, 4, 3,
            }),
            op::PadMode::REFLECT,
            Tensor(ET, {}, std::vector<T>{2112}),
            "pad_reflect_1d_bottom_neg"),

        PadParams(
            Tensor(ET, {6}, std::vector<T>{
                1, 2, 3, 4, 5, 6,
            }),
            Tensor(ET_INT, {1}, std::vector<T_INT>{-7}),
            Tensor(ET_INT, {1}, std::vector<T_INT>{3}),
            Tensor(ET, {2}, std::vector<T>{
                4, 3,
            }),
            op::PadMode::REFLECT,
            Tensor(ET, {}, std::vector<T>{2112}),
            "pad_reflect_1d_bottom_neg_bigger_than_tensor"),

        PadParams(
            Tensor(ET, {3, 4}, std::vector<T>{
                1, 2,  3,  4,
                5, 6,  7,  8,
                9, 10, 11, 12,
            }),
            Tensor(ET_INT, {2}, std::vector<T_INT>{2, 3}),
            Tensor(ET_INT, {2}, std::vector<T_INT>{1, 2}),
            Tensor(ET, {6, 9}, std::vector<T>{
                12, 11, 10, 9, 10, 11, 12, 11, 10,
                8,  7,  6,  5, 6,  7,  8,  7,  6,
                4,  3,  2,  1, 2,  3,  4,  3,  2,
                8,  7,  6,  5, 6,  7,  8,  7,  6,
                12, 11, 10, 9, 10, 11, 12, 11, 10,
                8,  7,  6,  5, 6,  7,  8,  7,  6,
            }),
            op::PadMode::REFLECT,
            Tensor(ET, {}, std::vector<T>{2112}),
            "pad_reflect_2d"),

        PadParams(
            Tensor(ET, {3, 4}, std::vector<T>{
                1, 2,  3,  4,
                5, 6,  7,  8,
                9, 10, 11, 12,
            }),
            Tensor(ET_INT, {2}, std::vector<T_INT>{2, -1}),
            Tensor(ET_INT, {2}, std::vector<T_INT>{1, 2}),
            Tensor(ET, {6, 5}, std::vector<T>{
                10, 11, 12, 11, 10,
                6,  7,  8,  7,  6,
                2,  3,  4,  3,  2,
                6,  7,  8,  7,  6,
                10, 11, 12, 11, 10,
                6,  7,  8,  7,  6,
            }),
            op::PadMode::REFLECT,
            Tensor(ET, {}, std::vector<T>{2112}),
            "pad_reflect_2d_with_neg"),

        PadParams(
            Tensor(ET, {2, 3}, std::vector<T>{
                1, 2, 3,
                4, 5, 6,
            }),
            Tensor(ET_INT, {2}, std::vector<T_INT>{1, -1}),
            Tensor(ET_INT, {2}, std::vector<T_INT>{2, 0}),
            Tensor(ET, {5, 2}, std::vector<T>{
                9, 9,
                2, 3,
                5, 6,
                9, 9,
                9, 9,
            }),
            op::PadMode::CONSTANT,
            Tensor(ET, {}, std::vector<T>{9}),
            "pad_negative_exterior_2d"),

        PadParams(
            Tensor(ET, {3, 3}, std::vector<T>{
                1, 2, 3,
                4, 5, 6,
            }),
            Tensor(ET_INT, {2}, std::vector<T_INT>{-1, -1}),
            Tensor(ET_INT, {2}, std::vector<T_INT>{-1, -1}),
            Tensor(ET, {1, 1}, std::vector<T>{5}),
            op::PadMode::CONSTANT,
            Tensor(ET, {}, std::vector<T>{9}),
            "pad_negative_exterior_2d_all_negative"),

        PadParams(
            Tensor(ET, {0, 0}, std::vector<T>{}),
            Tensor(ET_INT, {2}, std::vector<T_INT>{2, 3}),
            Tensor(ET_INT, {2}, std::vector<T_INT>{3, 2}),
            Tensor(ET, {5, 5}, std::vector<T>{
                2112, 2112, 2112, 2112, 2112,
                2112, 2112, 2112, 2112, 2112,
                2112, 2112, 2112, 2112, 2112,
                2112, 2112, 2112, 2112, 2112,
                2112, 2112, 2112, 2112, 2112,
            }),
            op::PadMode::CONSTANT,
            Tensor(ET, {}, std::vector<T>{2112}),
            "pad_exterior_2d_0x0"),

        PadParams(
            Tensor(ET, {0, 3}, std::vector<T>{}),
            Tensor(ET_INT, {2}, std::vector<T_INT>{2, 1}),
            Tensor(ET_INT, {2}, std::vector<T_INT>{3, 1}),
            Tensor(ET, {5, 5}, std::vector<T>{
                2112, 2112, 2112, 2112, 2112,
                2112, 2112, 2112, 2112, 2112,
                2112, 2112, 2112, 2112, 2112,
                2112, 2112, 2112, 2112, 2112,
                2112, 2112, 2112, 2112, 2112,
            }),
            op::PadMode::CONSTANT,
            Tensor(ET, {}, std::vector<T>{2112}),
            "pad_exterior_2d_0x3"),

        PadParams(
            Tensor(ET, {3, 0}, std::vector<T>{}),
            Tensor(ET_INT, {2}, std::vector<T_INT>{1, 3}),
            Tensor(ET_INT, {2}, std::vector<T_INT>{1, 2}),
            Tensor(ET, {5, 5}, std::vector<T>{
                2112, 2112, 2112, 2112, 2112,
                2112, 2112, 2112, 2112, 2112,
                2112, 2112, 2112, 2112, 2112,
                2112, 2112, 2112, 2112, 2112,
                2112, 2112, 2112, 2112, 2112,
            }),
            op::PadMode::CONSTANT,
            Tensor(ET, {}, std::vector<T>{2112}),
            "pad_exterior_2d_3x0"),

        PadParams(
            Tensor(ET, {2, 2, 4, 4}, std::vector<T>{
                0, 1, 0, 2,
                0, 3, 2, 0,
                2, 0, 0, 0,
                0, 2, 1, 0,

                0, 0, 0, 2,
                0, 2, 3, 0,
                2, 0, 1, 0,
                2, 0, 0, 0,

                0, 2, 1, 1,
                0, 0, 2, 0,
                0, 0, 1, 2,
                0, 0, 0, 0,

                2, 1, 0, 0,
                0, 2, 0, 0,
                1, 1, 2, 0,
                1, 0, 0, 0,
            }),
            Tensor(ET_INT, {4}, std::vector<T_INT>{0, 0, 0, 0}),
            Tensor(ET_INT, {4}, std::vector<T_INT>{0, 0, 2, 2}),
            Tensor(ET, {2, 2, 6, 6}, std::vector<T>{
                0,  1,  0,  2,  42, 42,
                0,  3,  2,  0,  42, 42,
                2,  0,  0,  0,  42, 42,
                0,  2,  1,  0,  42, 42,
                42, 42, 42, 42, 42, 42,
                42, 42, 42, 42, 42, 42,

                0,  0,  0,  2,  42, 42,
                0,  2,  3,  0,  42, 42,
                2,  0,  1,  0,  42, 42,
                2,  0,  0,  0,  42, 42,
                42, 42, 42, 42, 42, 42,
                42, 42, 42, 42, 42, 42,

                0,  2,  1,  1,  42, 42,
                0,  0,  2,  0,  42, 42,
                0,  0,  1,  2,  42, 42,
                0,  0,  0,  0,  42, 42,
                42, 42, 42, 42, 42, 42,
                42, 42, 42, 42, 42, 42,

                2,  1,  0,  0,  42, 42,
                0,  2,  0,  0,  42, 42,
                1,  1,  2,  0,  42, 42,
                1,  0,  0,  0,  42, 42,
                42, 42, 42, 42, 42, 42,
                42, 42, 42, 42, 42, 42,
            }),
            op::PadMode::CONSTANT,
            Tensor(ET, {}, std::vector<T>{42}),
            "pad_2channel_2image_asym"),

        PadParams(
            Tensor(ET, {2, 3}, std::vector<T>{
                1, 2, 3,
                4, 5, 6,
            }),
            Tensor(ET_INT, {2}, std::vector<T_INT>{1, 2}),
            Tensor(ET_INT, {2}, std::vector<T_INT>{1, 2}),
            Tensor(ET, {4, 7}, std::vector<T>{
                2, 1, 1, 2, 3, 3, 2,
                2, 1, 1, 2, 3, 3, 2,
                5, 4, 4, 5, 6, 6, 5,
                5, 4, 4, 5, 6, 6, 5,
            }),
            op::PadMode::SYMMETRIC,
            Tensor(ET, {}, std::vector<T>{2112}),
            "pad_symmetric"),
    };
    return params;
}

template <element::Type_t ET, element::Type_t ET_INT>
std::vector<PadParams> generateParamsFloatValue() {
    using T = typename element_type_traits<ET>::value_type;
    using T_INT = typename element_type_traits<ET_INT>::value_type;
    std::vector<PadParams> params {
        PadParams(
            Tensor(ET, {1, 2, 2, 2}, std::vector<T>{
                0.0f, 0.0f,
                0.0f, 0.0f,
                0.0f, 0.0f,
                0.0f, 0.0f,
            }),
            Tensor(ET_INT, {4}, std::vector<T_INT>{0, 0, 1, 1}),
            Tensor(ET_INT, {4}, std::vector<T_INT>{0, 0, 1, 1}),
            Tensor(ET, {1, 2, 4, 4}, std::vector<T>{
                42.0f, 42.0f, 42.0f, 42.0f,
                42.0f, 0.0f, 0.0f, 42.0f,
                42.0f, 0.0f, 0.0f, 42.0f,
                42.0f, 42.0f, 42.0f, 42.0f,
                42.0f, 42.0f, 42.0f, 42.0f,
                42.0f, 0.0f, 0.0f, 42.0f,
                42.0f, 0.0f, 0.0f, 42.0f,
                42.0f, 42.0f, 42.0f, 42.0f,
            }),
            op::PadMode::CONSTANT,
            Tensor(ET, {}, std::vector<T>{42}),
            "pad_exterior_4d_1x2x2x2"),

        PadParams(
            Tensor(ET, {1, 3, 2, 2}, std::vector<T>{
                0.0f, 0.0f,
                0.0f, 0.0f,
                1.0f, 1.0f,
                1.0f, 1.0f,
                2.0f, 2.0f,
                2.0f, 2.0f,
            }),
            Tensor(ET_INT, {4}, std::vector<T_INT>{0, -1, 1, 1}),
            Tensor(ET_INT, {4}, std::vector<T_INT>{0, -1, 1, 1}),
            Tensor(ET, {1, 1, 4, 4}, std::vector<T>{
                42.0f, 42.0f, 42.0f, 42.0f,
                42.0f, 1.0f,  1.0f,  42.0f,
                42.0f, 1.0f,  1.0f,  42.0f,
                42.0f, 42.0f, 42.0f, 42.0f,
            }),
            op::PadMode::CONSTANT,
            Tensor(ET, {}, std::vector<T>{42}),
            "pad_negative_exterior_4d"),
    };
    return params;
}

std::vector<PadParams> generateCombinedParams() {
    const std::vector<std::vector<PadParams>> generatedParams {
        generateParams<element::Type_t::i16, element::Type_t::i32>(),
        generateParams<element::Type_t::i32, element::Type_t::i32>(),
        generateParams<element::Type_t::i64, element::Type_t::i32>(),
        generateParams<element::Type_t::u16, element::Type_t::i32>(),
        generateParams<element::Type_t::u32, element::Type_t::i32>(),
        generateParams<element::Type_t::u64, element::Type_t::i32>(),
        generateParams<element::Type_t::bf16, element::Type_t::i32>(),
        generateParams<element::Type_t::f16, element::Type_t::i32>(),
        generateParams<element::Type_t::f32, element::Type_t::i32>(),
        generateParams<element::Type_t::f64, element::Type_t::i32>(),
        generateParams<element::Type_t::i16, element::Type_t::i64>(),
        generateParams<element::Type_t::i32, element::Type_t::i64>(),
        generateParams<element::Type_t::i64, element::Type_t::i64>(),
        generateParams<element::Type_t::u16, element::Type_t::i64>(),
        generateParams<element::Type_t::u32, element::Type_t::i64>(),
        generateParams<element::Type_t::u64, element::Type_t::i64>(),
        generateParams<element::Type_t::bf16, element::Type_t::i64>(),
        generateParams<element::Type_t::f16, element::Type_t::i64>(),
        generateParams<element::Type_t::f32, element::Type_t::i64>(),
        generateParams<element::Type_t::f64, element::Type_t::i64>(),
        generateParamsFloatValue<element::Type_t::bf16, element::Type_t::i64>(),
        generateParamsFloatValue<element::Type_t::f16, element::Type_t::i64>(),
        generateParamsFloatValue<element::Type_t::f32, element::Type_t::i64>(),
        generateParamsFloatValue<element::Type_t::f64, element::Type_t::i64>(),
    };
    std::vector<PadParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Pad_With_Hardcoded_Refs, ReferencePadTest,
    testing::ValuesIn(generateCombinedParams()), ReferencePadTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Pad_With_Hardcoded_Refs, ReferencePadTestNonConstPadsBeginPadsEndPadVal,
    testing::ValuesIn(generateCombinedParams()), ReferencePadTest::getTestCaseName);

template <element::Type_t ET, element::Type_t ET_INT>
std::vector<PadParams> generateParamsTooLarge() {
    using T = typename element_type_traits<ET>::value_type;
    using T_INT = typename element_type_traits<ET_INT>::value_type;
    std::vector<PadParams> params {
        PadParams(
            Tensor(ET, {2, 2}, std::vector<T>{
                1, 2, 4, 5,
            }),
            Tensor(ET_INT, {2}, std::vector<T_INT>{0, 3}),
            Tensor(ET_INT, {2}, std::vector<T_INT>{0, 0}),
            Tensor(ET, {2, 5}, std::vector<T>{
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            }),
            op::PadMode::SYMMETRIC,
            "pad_to_large_symmetric_padding"),

        PadParams(
            Tensor(ET, {2, 2}, std::vector<T>{
                1, 2, 4, 5,
            }),
            Tensor(ET_INT, {2}, std::vector<T_INT>{0, 2}),
            Tensor(ET_INT, {2}, std::vector<T_INT>{0, 0}),
            Tensor(ET, {2, 4}, std::vector<T>{
                0, 0, 0, 0, 0, 0, 0, 0,
            }),
            op::PadMode::REFLECT,
            "pad_to_large_reflect_padding"),
    };
    return params;
}

std::vector<PadParams> generateCombinedParamsTooLarge() {
    const std::vector<std::vector<PadParams>> generatedParams {
        generateParamsTooLarge<element::Type_t::f32, element::Type_t::i64>(),
    };
    std::vector<PadParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Pad_With_Hardcoded_Refs, ReferencePadTestParamsTooLarge,
    testing::ValuesIn(generateCombinedParamsTooLarge()), ReferencePadTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Pad_With_Hardcoded_Refs, ReferencePadTestNonConstPadsBeginPadsEndPadValTooLarge,
    testing::ValuesIn(generateCombinedParamsTooLarge()), ReferencePadTest::getTestCaseName);

template <element::Type_t ET, element::Type_t ET_INT>
std::vector<PadParams> generateParamsOk() {
    using T = typename element_type_traits<ET>::value_type;
    using T_INT = typename element_type_traits<ET_INT>::value_type;
    std::vector<PadParams> params {
        PadParams(
            Tensor(ET, {2, 2}, std::vector<T>{
                1, 2, 4, 5,
            }),
            Tensor(ET_INT, {2}, std::vector<T_INT>{0, 2}),
            Tensor(ET_INT, {2}, std::vector<T_INT>{0, 0}),
            Tensor(ET, {2, 4}, std::vector<T>{
                2, 1, 1, 2, 5, 4, 4, 5,
            }),
            op::PadMode::SYMMETRIC,
            "pad_ok_symmetric_padding"),

        PadParams(
            Tensor(ET, {2, 2}, std::vector<T>{
                1, 2, 4, 5,
            }),
            Tensor(ET_INT, {2}, std::vector<T_INT>{0, 1}),
            Tensor(ET_INT, {2}, std::vector<T_INT>{0, 0}),
            Tensor(ET, {2, 3}, std::vector<T>{
                2, 1, 2, 5, 4, 5,
            }),
            op::PadMode::REFLECT,
            "pad_ok_reflect_padding"),
    };
    return params;
}

std::vector<PadParams> generateCombinedParamsOk() {
    const std::vector<std::vector<PadParams>> generatedParams {
        generateParamsOk<element::Type_t::f32, element::Type_t::i64>(),
    };
    std::vector<PadParams> combinedParams;

    for (const auto& params : generatedParams) {
        combinedParams.insert(combinedParams.end(), params.begin(), params.end());
    }
    return combinedParams;
}

INSTANTIATE_TEST_SUITE_P(smoke_Pad_With_Hardcoded_Refs, ReferencePadTestParamsOk,
    testing::ValuesIn(generateCombinedParamsOk()), ReferencePadTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Pad_With_Hardcoded_Refs, ReferencePadTestNonConstPadsBeginPadsEndPadValParamsOk,
    testing::ValuesIn(generateCombinedParamsOk()), ReferencePadTest::getTestCaseName);
} // namespace
