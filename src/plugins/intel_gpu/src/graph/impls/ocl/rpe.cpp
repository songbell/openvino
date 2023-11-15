// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "rpe_inst.h"
#include "rpe/rpe_kernel_ref.h"
#include "rpe/rpe_kernel_selector.h"

namespace cldnn {
namespace ocl {

struct rpe_impl : typed_primitive_impl_ocl<rpe> {
    using parent = typed_primitive_impl_ocl<rpe>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::rpe_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::rpe_params, kernel_selector::rpe_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::rpe_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<rpe_impl>(*this);
    }

    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<rpe>();
        auto params = get_default_params<kernel_selector::rpe_params>(impl_param);
        auto optional_params = get_default_optional_params<kernel_selector::rpe_optional_params>(impl_param.get_program());

        params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(1)));
        params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(2)));
        params.axis = primitive->axis;

        return {params, optional_params};
    }
};

namespace detail {

attach_rpe_impl::attach_rpe_impl() {
    auto types = {data_types::f16};
    auto formats = {
        format::bfyx
    };
    std::set<std::tuple<data_types, format::type>> keys;
    for (const auto& t : types) {
        for (const auto& f : formats) {
            keys.emplace(t, f);
        }
    }
    implementation_map<rpe>::add(impl_types::ocl, typed_primitive_impl_ocl<rpe>::create<rpe_impl>, keys);
}

}  // namespace detail

}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::rpe_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::rpe)
