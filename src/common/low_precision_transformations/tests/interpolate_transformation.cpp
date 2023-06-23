// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <sstream>
#include <memory>

#include <gtest/gtest.h>

#include <ngraph/opsets/opset4.hpp>
#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include "low_precision/interpolate.hpp"

#include "common_test_utils/ngraph_test_utils.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "simple_low_precision_transformer.hpp"
#include "lpt_ngraph_functions/interpolate_function.hpp"

using namespace testing;
using namespace ngraph::pass;
using namespace ngraph;
using namespace ngraph::builder::subgraph;

class interpAttributes {
public:
    ngraph::AxisSet axes;
    std::string mode;
    bool align_corners;
    bool antialias;
    std::vector<size_t> pads_begin;
    std::vector<size_t> pads_end;

    interpAttributes() = default;

    interpAttributes(const ngraph::AxisSet& axes,
        const std::string& mode,
        const bool& align_corners,
        const bool& antialias,
        const std::vector<size_t>& pads_begin,
        const std::vector<size_t>& pads_end) :
        axes(axes), mode(mode), align_corners(align_corners),
        antialias(antialias), pads_begin(pads_begin), pads_end(pads_end) {}
};

class interp4Attributes {
public:
    op::v4::Interpolate::InterpolateMode mode;
    op::v4::Interpolate::CoordinateTransformMode coordinate_transformation_mode;
    std::vector<size_t> pads_begin;
    std::vector<size_t> pads_end;

    interp4Attributes() = default;

    interp4Attributes(const op::v4::Interpolate::InterpolateMode mode,
        const op::v4::Interpolate::CoordinateTransformMode coordinate_transformation_mode,
        const std::vector<size_t>& pads_begin,
        const std::vector<size_t>& pads_end) :
        mode(mode), coordinate_transformation_mode(coordinate_transformation_mode),
        pads_begin(pads_begin), pads_end(pads_end) {}
};

class InterpolateTransformationTestValues {
public:
    class Actual {
    public:
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
    };

    class Expected {
    public:
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantizationBefore;
        ngraph::element::Type precisionAfterOperation;
        ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
    };

    ngraph::PartialShape inputShape;
    ngraph::Shape outputShape;
    ngraph::Shape scalesShape;
    TestTransformationParams params;
    interpAttributes interpAttrs;
    interp4Attributes interp4Attrs;
    int opset_version;
    Actual actual;
    Expected expected;
};

class InterpolateTransformation : public LayerTransformation, public testing::WithParamInterface<InterpolateTransformationTestValues> {
public:
    void SetUp() override {
        const InterpolateTransformationTestValues testValues = GetParam();

        if (testValues.opset_version == 1) {
            ngraph::op::InterpolateAttrs interpAttrs;
            interpAttrs.axes = testValues.interpAttrs.axes;
            interpAttrs.mode = testValues.interpAttrs.mode;
            interpAttrs.align_corners = testValues.interpAttrs.align_corners;
            interpAttrs.antialias = testValues.interpAttrs.antialias;
            interpAttrs.pads_begin = testValues.interpAttrs.pads_begin;
            interpAttrs.pads_end = testValues.interpAttrs.pads_end;

            actualFunction = ngraph::builder::subgraph::InterpolateFunction::getOriginal(
                testValues.inputShape,
                testValues.outputShape,
                interpAttrs,
                testValues.actual.precisionBeforeDequantization,
                testValues.actual.dequantization);

            SimpleLowPrecisionTransformer transformer;
            transformer.add<ngraph::pass::low_precision::InterpolateTransformation, ov::op::v0::Interpolate>(testValues.params);
            transformer.transform(actualFunction);

            referenceFunction = ngraph::builder::subgraph::InterpolateFunction::getReference(
                testValues.inputShape,
                testValues.outputShape,
                interpAttrs,
                testValues.expected.precisionBeforeDequantization,
                testValues.expected.dequantizationBefore,
                testValues.expected.precisionAfterOperation,
                testValues.expected.dequantizationAfter);
        } else if (testValues.opset_version == 4) {
            ngraph::op::v4::Interpolate::InterpolateAttrs interp4Attrs;
            interp4Attrs.mode = testValues.interp4Attrs.mode;
            interp4Attrs.coordinate_transformation_mode = testValues.interp4Attrs.coordinate_transformation_mode;
            interp4Attrs.pads_begin = testValues.interp4Attrs.pads_begin;
            interp4Attrs.pads_end = testValues.interp4Attrs.pads_end;

            actualFunction = ngraph::builder::subgraph::InterpolateFunction::getOriginal(
                testValues.inputShape,
                testValues.outputShape,
                testValues.scalesShape,
                interp4Attrs,
                testValues.actual.precisionBeforeDequantization,
                testValues.actual.dequantization);

            SimpleLowPrecisionTransformer transformer;
            transformer.add<ngraph::pass::low_precision::InterpolateTransformation, ngraph::opset4::Interpolate>(testValues.params);
            transformer.transform(actualFunction);

            referenceFunction = ngraph::builder::subgraph::InterpolateFunction::getReference(
                testValues.inputShape,
                testValues.outputShape,
                testValues.scalesShape,
                interp4Attrs,
                testValues.expected.precisionBeforeDequantization,
                testValues.expected.dequantizationBefore,
                testValues.expected.precisionAfterOperation,
                testValues.expected.dequantizationAfter);
        }
    }

    static std::string getTestCaseName(testing::TestParamInfo<InterpolateTransformationTestValues> obj) {
        const InterpolateTransformationTestValues testValues = obj.param;

        std::ostringstream result;
        if (testValues.opset_version == 1) {
            result <<
            testValues.inputShape << "_" <<
            testValues.outputShape << "_" <<
            testValues.opset_version << "_" <<
            testValues.interpAttrs.align_corners << "_" <<
            testValues.interpAttrs.antialias << "_" <<
            testValues.interpAttrs.axes << "_" <<
            testValues.interpAttrs.mode << "_" <<
            testValues.interpAttrs.pads_begin << "_" <<
            testValues.interpAttrs.pads_end << "_" <<
            testValues.actual.precisionBeforeDequantization << "_" <<
            testValues.actual.dequantization << "_" <<
            testValues.expected.dequantizationBefore;
        } else if (testValues.opset_version == 4) {
            result <<
            testValues.inputShape << "_" <<
            testValues.outputShape << "_" <<
            testValues.opset_version << "_" <<
            testValues.interp4Attrs.mode << "_" <<
            testValues.interp4Attrs.coordinate_transformation_mode << "_" <<
            testValues.interp4Attrs.pads_begin << "_" <<
            testValues.interp4Attrs.pads_end << "_" <<
            testValues.actual.precisionBeforeDequantization << "_" <<
            testValues.actual.dequantization << "_" <<
            testValues.expected.dequantizationBefore;
        }
        return result.str();
    }
};

TEST_P(InterpolateTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();
    auto res = compare_functions(actualFunction, referenceFunction, true, true);
    ASSERT_TRUE(res.first) << res.second;

    ASSERT_TRUE(LayerTransformation::allNamesAreUnique(actualFunction)) << "Not all names are unique";
}

const std::vector<InterpolateTransformationTestValues> testValues {
    // opset1
    // nearest mode - move dequantization
    {
        { 1, 4, 16, 16 },
        ngraph::Shape{ 1, 4, 32, 32 },
        ngraph::Shape{},
        LayerTransformation::createParamsU8I8(),
        interpAttributes(
            ngraph::AxisSet{2, 3},
            "nearest",
            false,
            false,
            {0},
            {0}),
        interp4Attributes(),
        1,
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {-0.32f}, {0.1f}}
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {{ngraph::element::f32}, {-0.32f}, {0.1f}}
        }
    },

    // nearest mode - move dequantization
    {
        { -1, -1, -1, -1 },
        ngraph::Shape{ 1, 4, 32, 32 },
        ngraph::Shape{},
        LayerTransformation::createParamsU8I8(),
        interpAttributes(
            ngraph::AxisSet{2, 3},
            "nearest",
            false,
            false,
            {0},
            {0}),
        interp4Attributes(),
        1,
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {-0.32f}, {0.1f}}
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {{ngraph::element::f32}, {-0.32f}, {0.1f}}
        }
    },

    // per-channel dequantization
    {
        { -1, 4, -1, -1 },
        ngraph::Shape{ 1, 4, 32, 32 },
        ngraph::Shape{},
        LayerTransformation::createParamsU8I8(),
        interpAttributes(
            ngraph::AxisSet{2, 3},
            "nearest",
            false,
            false,
            {0},
            {0}),
        interp4Attributes(),
        1,
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {{-0.32f, -0.31f, 0.31f, 0.32f}}, {{0.1f, 0.2f, 0.3f, 0.4f}}}
        },
        {
            ngraph::element::u8,
            {{}, {}, {}},
            ngraph::element::u8,
            {{ngraph::element::f32}, {{-0.32f, -0.31f, 0.31f, 0.32f}}, {{0.1f, 0.2f, 0.3f, 0.4f}}}
        }
    },

    // per-channel dequantization and dynamic channels
    {
        { -1, -1, -1, -1 },
        ngraph::Shape{ 1, 4, 32, 32 },
        ngraph::Shape{},
        LayerTransformation::createParamsU8I8(),
        interpAttributes(
            ngraph::AxisSet{2, 3},
            "nearest",
            false,
            false,
            {0},
            {0}),
        interp4Attributes(),
        1,
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {{-0.32f, -0.31f, 0.31f, 0.32f}}, {{0.1f, 0.2f, 0.3f, 0.4f}}}
        },
        {
            ngraph::element::u8,
            {},
            ngraph::element::u8,
            {{ngraph::element::f32}, {{-0.32f, -0.31f, 0.31f, 0.32f}}, {{0.1f, 0.2f, 0.3f, 0.4f}}},
        }
    },

    // mode is not nearest - not transformed
    {
        { 1, 4, 16, 16 },
        ngraph::Shape{ 1, 4, 32, 32 },
        ngraph::Shape{},
        LayerTransformation::createParamsU8I8(),
        interpAttributes(
            ngraph::AxisSet{2, 3},
            "linear",
            false,
            false,
            {0},
            {0}),
        interp4Attributes(),
        1,
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {-0.32f}, {0.1f}}
        },
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {-0.32f}, {0.1f}},
            ngraph::element::u8,
            {{}, {}, {}}
        }
    },

    // AxisSet is not {2,3} - not transformed
    {
        { 1, 4, 16, 16 },
        ngraph::Shape{ 1, 8, 32, 32 },
        ngraph::Shape{},
        LayerTransformation::createParamsU8I8(),
        interpAttributes(
            ngraph::AxisSet{1, 2, 3},
            "nearest",
            false,
            false,
            {0},
            {0}),
        interp4Attributes(),
        1,
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {-0.32f}, {0.1f}}
        },
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {-0.32f}, {0.1f}},
            ngraph::element::u8,
            {{}, {}, {}}
        }
    },

    // align_corners set to true - not transformed
    {
        { 1, 4, 16, 16 },
        ngraph::Shape{ 1, 4, 32, 32 },
        ngraph::Shape{},
        LayerTransformation::createParamsU8I8(),
        interpAttributes(
            ngraph::AxisSet{2, 3},
            "nearest",
            true,
            false,
            {0},
            {0}),
        interp4Attributes(),
        1,
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {-0.32f}, {0.1f}}
        },
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {-0.32f}, {0.1f}},
            ngraph::element::u8,
            {{}, {}, {}}
        }
    },

    // have pads - not transformed
    {
        { 1, 4, 16, 16 },
        ngraph::Shape{ 1, 4, 32, 32 },
        ngraph::Shape{},
        LayerTransformation::createParamsU8I8(),
        interpAttributes(
            ngraph::AxisSet{2, 3},
            "nearest",
            false,
            false,
            {1},
            {1}),
        interp4Attributes(),
        1,
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {-0.32f}, {0.1f}}
        },
        {
            ngraph::element::u8,
            {{ngraph::element::f32}, {-0.32f}, {0.1f}},
            ngraph::element::u8,
            {{}, {}, {}}
        }
    },

    // v4::Interpolate
    // nearest mode - move dequantization
    {
        { 1, 4, 16, 16 },
        ngraph::Shape{ 1, 4, 32, 32 },
        ngraph::Shape{ 1, 1, 2, 2 },
        LayerTransformation::createParamsU8I8(),
        interpAttributes(),
        interp4Attributes(
            ngraph::op::v4::Interpolate::InterpolateMode::NEAREST,
            ngraph::op::v4::Interpolate::CoordinateTransformMode::HALF_PIXEL,
            {0, 0, 0, 0},
            {0, 0, 0, 0}),
        4,
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {-0.32f}, {0.1f}}
        },
        {
            ngraph::element::i8,
            {{}, {}, {}},
            ngraph::element::i8,
            {{ngraph::element::f32}, {-0.32f}, {0.1f}}
        }
    },

    // nearest mode - move dequantization
    {
        { -1, -1, -1, -1 },
        ngraph::Shape{ 1, 4, 32, 32 },
        ngraph::Shape{ 1, 1, 2, 2 },
        LayerTransformation::createParamsU8I8(),
        interpAttributes(),
        interp4Attributes(
            ngraph::op::v4::Interpolate::InterpolateMode::NEAREST,
            ngraph::op::v4::Interpolate::CoordinateTransformMode::HALF_PIXEL,
            {0, 0, 0, 0},
            {0, 0, 0, 0}),
        4,
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {-0.32f}, {0.1f}}
        },
        {
            ngraph::element::i8,
            {{}, {}, {}},
            ngraph::element::i8,
            {{ngraph::element::f32}, {-0.32f}, {0.1f}}
        }
    },

    // per-channel dequantization
    {
        { -1, 4, -1, -1 },
        ngraph::Shape{ 1, 4, 32, 32 },
        ngraph::Shape{ 1, 1, 2, 2 },
        LayerTransformation::createParamsU8I8(),
        interpAttributes(),
        interp4Attributes(
            ngraph::op::v4::Interpolate::InterpolateMode::NEAREST,
            ngraph::op::v4::Interpolate::CoordinateTransformMode::HALF_PIXEL,
            {0, 0, 0, 0},
            {0, 0, 0, 0}),
        4,
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {{-0.32f, -0.31f, 0.31f, 0.32f}}, {{0.1f, 0.2f, 0.3f, 0.4f}}}
        },
        {
            ngraph::element::i8,
            {{}, {}, {}},
            ngraph::element::i8,
            {{ngraph::element::f32}, {{-0.32f, -0.31f, 0.31f, 0.32f}}, {{0.1f, 0.2f, 0.3f, 0.4f}}}
        }
    },

    // per-channel dequantization and dynamic channels
    {
        { -1, -1, -1, -1 },
        ngraph::Shape{ 1, 4, 32, 32 },
        ngraph::Shape{ 1, 1, 2, 2 },
        LayerTransformation::createParamsU8I8(),
        interpAttributes(),
        interp4Attributes(
            ngraph::op::v4::Interpolate::InterpolateMode::NEAREST,
            ngraph::op::v4::Interpolate::CoordinateTransformMode::HALF_PIXEL,
            {0, 0, 0, 0},
            {0, 0, 0, 0}),
        4,
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {{-0.32f, -0.31f, 0.31f, 0.32f}}, {{0.1f, 0.2f, 0.3f, 0.4f}}}
        },
        {
            ngraph::element::i8,
            {},
            ngraph::element::i8,
            {{ngraph::element::f32}, {{-0.32f, -0.31f, 0.31f, 0.32f}}, {{0.1f, 0.2f, 0.3f, 0.4f}}},
        }
    },

    // mode is not nearest - not transformed
    {
        { 1, 4, 16, 16 },
        ngraph::Shape{ 1, 4, 32, 32 },
        ngraph::Shape{ 1, 1, 2, 2 },
        LayerTransformation::createParamsU8I8(),
        interpAttributes(),
        interp4Attributes(
            ngraph::op::v4::Interpolate::InterpolateMode::LINEAR_ONNX,
            ngraph::op::v4::Interpolate::CoordinateTransformMode::HALF_PIXEL,
            {0, 0, 0, 0},
            {0, 0, 0, 0}),
        4,
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {-0.32f}, {0.1f}}
        },
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {-0.32f}, {0.1f}},
            ngraph::element::i8,
            {{}, {}, {}}
        }
    },

    // align_corners set to true - not transformed
    {
        { 1, 4, 16, 16 },
        ngraph::Shape{ 1, 4, 32, 32 },
        ngraph::Shape{ 1, 1, 2, 2 },
        LayerTransformation::createParamsU8I8(),
        interpAttributes(),
        interp4Attributes(
            ngraph::op::v4::Interpolate::InterpolateMode::NEAREST,
            ngraph::op::v4::Interpolate::CoordinateTransformMode::ALIGN_CORNERS,
            {0, 0, 0, 0},
            {0, 0, 0, 0}),
        4,
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {-0.32f}, {0.1f}}
        },
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {-0.32f}, {0.1f}},
            ngraph::element::i8,
            {{}, {}, {}}
        }
    },

    // have pads - not transformed
    {
        { 1, 4, 16, 16 },
        ngraph::Shape{ 1, 4, 32, 32 },
        ngraph::Shape{ 1, 1, 2, 2 },
        LayerTransformation::createParamsU8I8(),
        interpAttributes(),
        interp4Attributes(
            ngraph::op::v4::Interpolate::InterpolateMode::NEAREST,
            ngraph::op::v4::Interpolate::CoordinateTransformMode::HALF_PIXEL,
            {0, 0, 0, 1},
            {0, 0, 1, 0}),
        4,
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {-0.32f}, {0.1f}}
        },
        {
            ngraph::element::i8,
            {{ngraph::element::f32}, {-0.32f}, {0.1f}},
            ngraph::element::i8,
            {{}, {}, {}}
        }
    },
};

INSTANTIATE_TEST_SUITE_P(
    smoke_LPT,
    InterpolateTransformation,
    ::testing::ValuesIn(testValues),
    InterpolateTransformation::getTestCaseName);
