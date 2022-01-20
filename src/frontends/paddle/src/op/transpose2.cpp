// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <node_context.hpp>

#include "openvino/opsets/opset6.hpp"

namespace ov {
namespace frontend {
namespace paddle {
namespace op {
NamedOutputs transpose2(const NodeContext& node) {
    auto data = node.get_ng_input("X");
    auto perm = node.get_attribute<std::vector<int>>("axis");
    auto input_order = ov::opset6::Constant::create(ov::element::i64, {perm.size()}, perm);
    return node.default_single_output_mapping({std::make_shared<ov::opset6::Transpose>(data, input_order)}, {"Out"});
}

}  // namespace op
}  // namespace paddle
}  // namespace frontend
}  // namespace ov
