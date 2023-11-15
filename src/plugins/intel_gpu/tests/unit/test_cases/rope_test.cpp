// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/rpe.hpp>
#include <intel_gpu/runtime/memory.hpp>
#include <intel_gpu/graph/topology.hpp>
#include <intel_gpu/graph/network.hpp>

#include <cstddef>

using namespace cldnn;
using namespace ::tests;

void test_simple_input();

void test_simple_input() {
    // # N = 2
    // # b = 1
    // # d = 3
    // q = np.array([5, 2, 1,
    //               1, 2, 5]).reshape([1, 1, 1, 2, 3]) # b x N x d

    // k = np.array([1, 5,
    //               2, 2,
    //               5, 1]).reshape([1, 1, 1, 3, 2]) # b x d x N

    // v = np.array([5, 0, 5,
    //               0, 5, 5]).reshape([1, 1, 1, 2, 3]) # b x N x d
    // return (q, k, v)

    // =simple r:
    // [[[[[5.62675810e-07 4.99999944e+00 5.00000000e+00]
    //     [4.99999944e+00 5.62675810e-07 5.00000000e+00]]]]]
    auto& engine = get_test_engine();

    auto input1 = engine.allocate_memory({ data_types::f16, format::bfyx, tensor{ 1, 1, 3, 2 } }); // query
    auto input2 = engine.allocate_memory({ data_types::f16, format::bfyx, tensor{ 1, 1, 2, 3 } }); // pre calculated sin
    auto input3 = engine.allocate_memory({ data_types::f16, format::bfyx, tensor{ 1, 1, 3, 2 } }); // pre calculated cos

    set_values(input1, {
        FLOAT16(5.0f), FLOAT16(2.0f), FLOAT16(1.0f),
        FLOAT16(3.0f), FLOAT16(7.0f), FLOAT16(9.0f),
    });

    set_values(input2, {
        FLOAT16(2.0f), FLOAT16(13.0f),
        FLOAT16(4.0f), FLOAT16(6.0f),
        FLOAT16(10.0f), FLOAT16(3.0f),
    });

    set_values(input3, {
        FLOAT16(30.0f), FLOAT16(0.0f), FLOAT16(35.0f),
        FLOAT16(0.0f), FLOAT16(45.0f), FLOAT16(55.0f),
    });

    topology topology;
    topology.add(input_layout("input_q_or_k", input1->get_layout()));
    topology.add(input_layout("sin_tab", input2->get_layout()));
    topology.add(input_layout("cos_tab", input3->get_layout()));
    topology.add(
        rpe("rpe", input_info("input_q_or_k"), input_info("sin_tab"), input_info("cos_tab"))
    );

    cldnn::network::ptr network = get_network(engine, topology, get_test_default_config(engine), get_test_stream_ptr(), false);

    network->set_input_data("input_q_or_k", input1);
    network->set_input_data("sin_tab", input2);
    network->set_input_data("cos_tab", input3);

    auto outputs = network->execute();

    auto output = outputs.at("rpe").get_memory();
    cldnn::mem_lock<uint16_t> output_ptr(output, get_test_stream());

    std::vector<float> expected_results = {
        0.f, 45.f, 55.f,
        30.f, 0.f, 35.f
    };

    for (size_t i = 0; i < expected_results.size(); ++i) {
        ASSERT_NEAR(expected_results[i], half_to_float(output_ptr[i]), 1e-3);
    }
}

TEST(rope_gpu_fp16, simple_input) {
    test_simple_input(false);
}