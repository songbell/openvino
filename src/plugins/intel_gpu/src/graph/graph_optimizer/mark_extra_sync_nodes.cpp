// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass_manager.h"
#include "sync_tensor_inst.h"
#include "intel_gpu/graph/program.hpp"

using namespace cldnn;

void mark_extra_sync_nodes::run(program& p) {
    for (const auto& node : p.get_processing_order()) {
        for (auto& iter : node->get_dependencies()) {
            if (iter.first->is_type<sync_tensor>())
                node->set_extra_sync_needed(true);
        }
    }
}
