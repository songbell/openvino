// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <utility>
#include <vector>
#include <memory>

#include "openvino/core/any.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/properties.hpp"

#include "base/ov_behavior_test_utils.hpp"

using namespace ::testing;

namespace ov {
namespace test {
namespace intel_gpu {
void name_node_and_output(const std::shared_ptr<Node>& op, const std::string& name);
void name_node_and_output(const std::shared_ptr<Node>& op, const std::string& name) {
    op->set_friendly_name(name);
    op->output(0).set_names({name});
}

static std::shared_ptr<ov::Model> get_simple_model_with_rpe() {
    auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), 1, Dimension::dynamic()}); // (B, seq_len*1*, Dim)
    name_node_and_output(data, "source");
    auto sin = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), Dimension::dynamic()}); // (seq_len_max, head_dim)
    name_node_and_output(sin, "sin");
    auto cos = std::make_shared<ov::op::v0::Parameter>(element::f32, PartialShape{Dimension::dynamic(), Dimension::dynamic()}); // (seq_len_max, head_dim)
    name_node_and_output(cos, "cos");
    auto axis = ov::op::v0::Constant::create(element::i64, {}, {-1});
    auto split_lengths = ov::op::v0::Constant::create(element::i64, {2}, {10, 10});
    auto split = std::make_shared<ov::op::v1::VariadicSplit>(data, axis, split_lengths);

    auto minus_one = ov::op::v0::Constant::create(element::f32, {}, {-1});
    auto negate = std::make_shared<ov::op::v1::Multiply>(split->output(1), minus_one);

    auto concat = std::make_shared<ov::op::v0::Concat>(OutputVector{negate, split->output(0)}, -1);

    auto mul_sin = std::make_shared<ov::op::v1::Multiply>(concat, sin);
    auto mul_cos = std::make_shared<ov::op::v1::Multiply>(data, cos);
    auto add = std::make_shared<ov::op::v1::Add>(mul_cos, mul_sin);
    name_node_and_output(add, "rpe");

    auto model = std::make_shared<Model>(NodeVector{add}, ParameterVector{data, sin, cos});

    return model;
}

TEST(RPE, CanTransformAndCompile) {
    ov::Core core;
    auto model = get_simple_model_with_rpe();
    auto compiled_model = core.compile_model(model, ov::test::utils::DEVICE_GPU);

    auto runtime_graph = compiled_model.get_runtime_model();

    auto ops = runtime_graph->get_ordered_ops();
    /*ASSERT_EQ(ops.size(), 3);
    ASSERT_STREQ(ops[0]->get_rt_info()[ov::exec_model_info::LAYER_TYPE].as<std::string>().c_str(), "Input");
    ASSERT_STREQ(ops[1]->get_rt_info()[ov::exec_model_info::LAYER_TYPE].as<std::string>().c_str(), "CustomGPUPrimitive");
    ASSERT_STREQ(ops[2]->get_rt_info()[ov::exec_model_info::LAYER_TYPE].as<std::string>().c_str(), "Result");*/
}

} // namespace intel_gpu
} // namespace test
} // namespace ov
