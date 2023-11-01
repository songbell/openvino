// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sstream>

#include "json_object.h"
#include "primitive_type_base.h"
#include "rpe_inst.h"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(rpe)

layout rpe_inst::calc_output_layout(const rpe_node& node, kernel_impl_params const& impl_param) {
    // TBD
    return impl_param.get_input_layout();
}

std::string rpe_inst::to_string(const rpe_node& node) {
    auto node_info = node.desc_to_json();
    json_composite rpe_info;
    for (size_t i = 0; i < node.get_inputs_count(); i++) {
        rpe_info.add("input_" + std::to_string(i), node.input(i).id());
    }
    rpe_info.add("axis", node.get_primitive()->axis);
    node_info->add("rpe info", rpe_info);
    std::ostringstream primitive_description;
    node_info->dump(primitive_description);
    return primitive_description.str();
}

}  // namespace cldnn
