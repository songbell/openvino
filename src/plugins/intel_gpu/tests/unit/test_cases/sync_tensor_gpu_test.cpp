// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "random_generator.hpp"
#include "concatenation_inst.h"
#include "permute_inst.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/convolution.hpp>
#include <intel_gpu/primitives/data.hpp>
#include "intel_gpu/primitives/fully_connected.hpp"
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/sync_tensor.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <thread>
#include <type_traits>
#include <fstream>

using namespace cldnn;
using namespace ::tests;

TEST(concat_gpu, split_fc) {
    auto& engine = get_test_engine();
    const int32_t input_f = 3, input_b = 1,    // size of the whole input buffer
                  weight_b = 4, weight_f = 3;  // size of the whole weights buffer

    auto input_prim = engine.allocate_memory({ data_types::f32, format::yxfb, { input_b, input_f, 1, 1 } });
    auto weights_prim = engine.allocate_memory({ data_types::f32, format::bfyx, { weight_b, weight_f, 1, 1 } });
    auto engine2 = create_test_engine("1");
    set_values(input_prim, { -0.5f, 2.0f, 0.5f });
    set_values(weights_prim, { 1.5f, 1.0f, 0.5f, -1.0f, 0.0f, 0.5f, 0.5f, -0.5f, -2.0f, -0.5f, 1.0f, 1.5f });

    auto input = input_layout("input", input_prim->get_layout());
    auto w_data = data("weights", weights_prim);
    auto fc = fully_connected("fc_prim", input_info("input"), "weights");
    auto synctensor = sync_tensor("broadcast_tensor", input_info("fc_prim"));
    auto mem = engine2->allocate_memory({ data_types::f32, format::bfyx, { 1, 4, 1, 1 } }); // place holder
    auto sync_data = data("sync_data", mem, true);
    auto concat = concatenation("concat",
                          { input_info("broadcast_tensor"), input_info("sync_data")},
                          1,
                          data_types::f32,
                          padding{ { 0,0,0,0 }, 0 });
    topology topology;
    topology.add(input);
    topology.add(w_data);
    topology.add(fc);
    topology.add(synctensor);
    topology.add(sync_data);
    topology.add(concat);
    network network(engine, topology, get_test_default_config(engine));
}

