// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/max_roi_pool.hpp"

#include "openvino/frontend/exception.hpp"
#include "openvino/op/roi_pooling.hpp"

using namespace ov::op;

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector max_roi_pool(const Node& node) {
    const auto& inputs = node.get_ng_inputs();
    const auto X = inputs.at(0);
    const auto rois = inputs.at(1);

    FRONT_END_GENERAL_CHECK(X.get_element_type() == element::f16 || X.get_element_type() == element::f32 ||
                                X.get_element_type() == element::f64,
                            "MaxRoiPool operator only supports float16, float32 and float64 datatypes.");

    const auto pooled_shape = node.get_attribute_value<std::vector<size_t>>("pooled_shape");
    const auto spatial_scale = node.get_attribute_value<float>("spatial_scale", 1.0);

    return {std::make_shared<v0::ROIPooling>(X, rois, Shape(pooled_shape), spatial_scale, "max")};
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
