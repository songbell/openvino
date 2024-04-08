// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "program_helpers.h"
#include "pass_manager.h"

#include "fully_connected_inst.h"
#include "sync_tensor_inst.h"

#include <vector>
#include <map>

using namespace cldnn;
static void enable_tensor_parallel(fully_connected_node& fc_node, program& p) {
    auto fc_prim = fc_node.get_primitive();
}

void add_tensor_parallel_opt::run(program& p) {
    bool recalc_processing_order = false;
    auto itr = p.get_processing_order().begin();
    while (itr != p.get_processing_order().end()) {
        auto &node = (*itr++);
        if (node->is_type<fully_connected>()) {
            enable_tensor_parallel(node->as<fully_connected>(), p);
        }
    }
    // Need to update processing order to handle cases when peer node processing number is greater
    // than fused node one
    if (recalc_processing_order)
        p.get_processing_order().calc_processing_order(p);
}