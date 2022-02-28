// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "default_opset.hpp"
#include "ngraph/node.hpp"
#include "onnx_import/core/node.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
inline OutputVector where(const Node& node) {
    OutputVector ng_inputs{node.get_ng_inputs()};

    return {std::make_shared<default_opset::Select>(ng_inputs.at(0), ng_inputs.at(1), ng_inputs.at(2))};
}
}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
