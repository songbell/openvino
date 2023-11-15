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
#include "common_test_utils/ov_tensor_utils.hpp"
#include "shared_test_classes/base/utils/compare_results.hpp"

using namespace ::testing;

namespace ov {
namespace test {
namespace intel_gpu {

class mhaOp : public ov::op::Op {
public:
    OPENVINO_OP("mhaOp");

    mhaOp() = default;

    mhaOp(const OutputVector& inputs) : Op({inputs}) {
        constructor_validate_and_infer_types();
    }

    void validate_and_infer_types() override {
        set_output_size(1);
        const auto &A_shape = get_input_partial_shape(0), B_shape = get_input_partial_shape(1), C_shape = get_input_partial_shape(2);
        std::vector<ov::PartialShape> input_shapes = {A_shape, B_shape, C_shape};
        // fake shape infer
        // std::vector<ov::PartialShape> output_shapes = shape_infer(this, input_shapes);
        set_output_type(0, get_input_element_type(0), ov::PartialShape{1, 10, 9216, 64});
    }

    bool visit_attributes(ov::AttributeVisitor& visitor) override {
        return true;
    }

    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& inputs) const override {
        return std::make_shared<mhaOp>(inputs);
    }

    bool has_evaluate() const override {
        return true;
    }

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override {
        auto in = inputs[0];
        auto out = outputs[0];
        out.set_shape(in.get_shape());
        for (size_t i = 0; i < out.get_size(); i++) {
            out.data<float>()[i] = in.data<float>()[i];
        }
        return true;
    }
};

static std::shared_ptr<ov::Model> make_llm_mha_pattern(ov::Dimension batch = ov::Dimension::dynamic(),
                                                            ov::Dimension n_heads = ov::Dimension::dynamic(),
                                                            ov::Dimension n_features = ov::Dimension::dynamic(),
                                                            ov::element::Type_t element_type = ov::element::f16) {
    ov::PartialShape q_size = {batch, n_heads, 9216, n_features};
    ov::PartialShape k_size = {batch, n_heads, 9216, n_features};
    ov::PartialShape v_size = {batch, n_heads, 9216, n_features};

    auto q_mat = std::make_shared<ov::op::v0::Parameter>(element_type, q_size);
    q_mat->set_friendly_name("q_matrix");
    auto k_mat = std::make_shared<ov::op::v0::Parameter>(element_type, k_size);
    k_mat->set_friendly_name("k_matrix");
    auto v_mat = std::make_shared<ov::op::v0::Parameter>(element_type, v_size);
    v_mat->set_friendly_name("v_matrix");

    auto transpose_const = ov::op::v0::Constant::create(ov::element::i32, {k_size.size()}, {0, 1, 3, 2});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(k_mat, transpose_const);
    auto matmul = std::make_shared<ov::op::v0::MatMul>(q_mat, transpose, false, false);

    auto softmax = std::make_shared<ov::op::v8::Softmax>(matmul);

    auto matmul_v = std::make_shared<ov::op::v0::MatMul>(softmax, v_mat, false, false);
    auto matmul_out = std::make_shared<ov::op::v0::Result>(matmul_v);
    matmul_out->set_friendly_name("matmul_out");

    ov::ParameterVector params{q_mat, k_mat, v_mat};
    ov::ResultVector results{matmul_out};
    return std::make_shared<ov::Model>(results, params, "LLM-MHA-pattern");
}

static std::shared_ptr<ov::Model> make_llm_mha_node(ov::Dimension batch = ov::Dimension::dynamic(),
                                                            ov::Dimension n_heads = ov::Dimension::dynamic(),
                                                            ov::Dimension n_features = ov::Dimension::dynamic(),
                                                            ov::element::Type_t element_type = ov::element::f16) {
    ov::PartialShape q_size = {batch, n_heads, 9216, n_features};
    ov::PartialShape k_size = {batch, n_heads, 9216, n_features};
    ov::PartialShape v_size = {batch, n_heads, 9216, n_features};

    auto q_mat = std::make_shared<ov::op::v0::Parameter>(element_type, q_size);
    q_mat->set_friendly_name("q_matrix");
    auto k_mat = std::make_shared<ov::op::v0::Parameter>(element_type, k_size);
    k_mat->set_friendly_name("k_matrix");
    auto v_mat = std::make_shared<ov::op::v0::Parameter>(element_type, v_size);
    v_mat->set_friendly_name("v_matrix");
    auto transpose_const = ov::op::v0::Constant::create(ov::element::i32, {k_size.size()}, {0, 1, 3, 2});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(k_mat, transpose_const);
    auto mha = std::make_shared<ov::test::intel_gpu::mhaOp>(ov::OutputVector{q_mat, transpose, v_mat});

    auto matmul_out = std::make_shared<ov::op::v0::Result>(mha);
    matmul_out->set_friendly_name("matmul_out");

    ov::ParameterVector params{q_mat, k_mat, v_mat};
    ov::ResultVector results{matmul_out};
    return std::make_shared<ov::Model>(results, params, "LLM-MHA-node");
}

TEST(mhaOP, validate_result) {
    ov::Core core;
    const size_t n_heads = 10;
    const size_t n_features = 64;
    auto model = make_llm_mha_pattern(1, n_heads, n_features);
    ov::serialize(model, "mha_pattern.xml");
    auto model_2 = make_llm_mha_node(1, n_heads, n_features);
    ov::serialize(model_2, "mha_node.xml");
    ov::AnyMap config = {{"CONFIG_FILE", TEST_MHA_OP_CONFIG_PATH}};
    auto compiled_model = core.compile_model(model_2, ov::test::utils::DEVICE_GPU, config);

    auto runtime_graph = compiled_model.get_runtime_model();

    ov::serialize(runtime_graph, "mha_runtime.xml");

    // for output validate
    auto compare_tensors = [&model](const std::vector<ov::Tensor> expected, const std::vector<ov::Tensor>& actual) {
            ASSERT_EQ(expected.size(), actual.size());
            ASSERT_EQ(expected.size(), model->get_results().size());
            auto compareMap = ov::test::utils::getCompareMap();
            const auto& results = model->get_results();
            for (size_t j = 0; j < results.size(); j++) {
                const auto result = results[j];
                for (size_t i = 0; i < result->get_input_size(); ++i) {
                    std::shared_ptr<ov::Node> inputNode = result->get_input_node_shared_ptr(i);
                    if (std::dynamic_pointer_cast<ov::op::v0::Convert>(inputNode)) {
                        std::shared_ptr<ov::Node> nextNodePtr = inputNode->get_input_node_shared_ptr(0);
                        if (!ngraph::is_type<ov::op::v0::Result>(nextNodePtr)) {
                            inputNode = nextNodePtr;
                        }
                    }
                    auto it = compareMap.find(inputNode->get_type_info());
                    ASSERT_NE(it, compareMap.end());
                    it->second(inputNode, i, expected[j], actual[j], 1e-4f, 1e-4f);
                }
            }
    };

    auto input0 = model_2->get_parameters().at(0);
    auto input1 = model_2->get_parameters().at(1);
    auto input2 = model_2->get_parameters().at(2);
    auto output0 = model_2->get_results().at(0);

    auto infer_request = compiled_model.create_infer_request();

    {
        auto q_data = ov::test::utils::create_and_fill_tensor(ov::element::f16, ov::Shape{1, 10, 9216, 64});
        auto k_data = ov::test::utils::create_and_fill_tensor(ov::element::f16, ov::Shape{1, 10, 9216, 64});
        auto v_data = ov::test::utils::create_and_fill_tensor(ov::element::f16, ov::Shape{1, 10, 9216, 64});

        auto results = ngraph::helpers::interpretFunction(model, {{input0, q_data}, {input1, k_data}, {input2, v_data}});

        infer_request.set_tensor(input0, q_data);
        infer_request.set_tensor(input1, k_data);
        infer_request.set_tensor(input2, v_data);

        infer_request.infer();

        compare_tensors(results, {infer_request.get_tensor(output0)});
    }
}

} // namespace intel_gpu
} // namespace test
} // namespace ov
