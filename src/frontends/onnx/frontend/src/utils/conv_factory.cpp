// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils/conv_factory.hpp"

#include "exceptions.hpp"
#include "onnx_import/core/null_node.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "ov_models/ov_builders/reshape.hpp"
#include "utils/conv_factory.hpp"
#include "utils/convpool.hpp"
#include "utils/reshape.hpp"

namespace ngraph {
namespace onnx_import {
namespace conv_factory {
std::shared_ptr<ov::op::Op> make_ng_convolution(const Output<ov::Node>& data,
                                                const Output<ov::Node>& filters,
                                                const ov::Strides& strides,
                                                const ov::Strides& dilations,
                                                const ov::CoordinateDiff& padding_below,
                                                const ov::CoordinateDiff& padding_above,
                                                int64_t groups,
                                                const ov::op::PadType& auto_pad) {
    if (groups > 1) {
        const auto reshaped_filters = convpool::get_reshaped_filters(filters, groups);

        return std::make_shared<ov::op::v1::GroupConvolution>(data,
                                                              reshaped_filters,
                                                              strides,
                                                              padding_below,
                                                              padding_above,
                                                              dilations,
                                                              auto_pad);
    } else {
        return std::make_shared<ov::op::v1::Convolution>(data,
                                                         filters,
                                                         strides,
                                                         padding_below,
                                                         padding_above,
                                                         dilations,
                                                         auto_pad);
    }
}
}  // namespace conv_factory
}  // namespace onnx_import
}  // namespace ngraph
