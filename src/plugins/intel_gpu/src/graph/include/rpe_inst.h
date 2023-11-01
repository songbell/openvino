// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/rpe.hpp"
#include "primitive_inst.h"

namespace cldnn {

using rpe_node = typed_program_node<rpe>;

template <>
class typed_primitive_inst<rpe> : public typed_primitive_inst_base<rpe> {
public:
    using parent = typed_primitive_inst_base<rpe>;
    using parent::parent;

    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(rpe_node const& /*node*/, const kernel_impl_params& impl_param) {
        return forward_input0_shape<ShapeType>(impl_param);
    }
    static layout calc_output_layout(const rpe_node& node, kernel_impl_params const& impl_param);
    static std::string to_string(const rpe_node& node);
};

using rpe_inst = typed_primitive_inst<rpe>;

}  // namespace cldnn
