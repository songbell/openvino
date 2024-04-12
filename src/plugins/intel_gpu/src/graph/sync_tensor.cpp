// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sync_tensor_inst.h>
#include "primitive_type_base.h"
#include <sstream>
#include <json_object.h>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(sync_tensor)

sync_tensor_inst::typed_primitive_inst(network& network, const sync_tensor_node& node) :
    parent(network, node, !node.can_be_optimized() && (node.get_output_layout().is_static() || node.get_output_layout().has_upper_bound())) {
}

layout sync_tensor_inst::calc_output_layout(const sync_tensor_node& node, kernel_impl_params const& impl_param) {
    return {};
}

std::string sync_tensor_inst::to_string(const sync_tensor_node& node) {
    auto node_info = node.desc_to_json();
    std::stringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

void sync_tensor_inst::on_execute() {
    update_output_memory();
}

void sync_tensor_inst::update_output_memory() {
    if (!can_be_optimized())
        return;

    _outputs = {input_memory_ptr()}; // to be optimized further
    _mem_allocated = false;
}
} // namespace cldnn
