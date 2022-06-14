// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/reduce_min.hpp"
#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include "low_precision/network_helper.hpp"
#include "itt.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

ReduceMinTransformation::ReduceMinTransformation(const Params& params) : ReduceBaseTransformation(params) {
    MATCHER_SCOPE(ReduceMinTransformation);
    auto matcher = pattern::wrap_type<opset1::ReduceMin>({ pattern::wrap_type<opset1::Multiply>(), pattern::wrap_type<opset1::Constant>() });

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        auto op = m.get_match_root();
        if (transformation_callback(op)) {
            return false;
        }
        MATCHER_SCOPE_ENABLE(ReduceMinTransformation);
        return transform(*context, m);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(matcher, matcher_name);
    this->register_matcher(m, callback);
}

bool ReduceMinTransformation::canBeTransformed(const TransformationContext& context, std::shared_ptr<Node> reduce) const {
    if (!ov::is_type<opset1::ReduceMin>(reduce)) {
        return false;
    }

    if (!ReduceBaseTransformation::canBeTransformed(context, reduce)) {
        return false;
    }

    const auto dequantization = NetworkHelper::getDequantization(reduce, defaultPrecisions);
    const std::vector<float> scales = ov::as_type_ptr<opset1::Constant>(dequantization.multiplyConstant)->cast_vector<float>();
    if (std::any_of(scales.begin(), scales.end(), [](const float value) { return value < 0.0; })) {
        return false;
    }

    return true;
}

bool ReduceMinTransformation::isPrecisionPreserved(std::shared_ptr<Node> reduce) const noexcept {
    return true;
}

bool ReduceMinTransformation::getUpdatePrecision(const std::shared_ptr<Node>& reduce) const {
    return true;
}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
