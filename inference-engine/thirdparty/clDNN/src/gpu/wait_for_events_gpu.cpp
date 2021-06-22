// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_inst.h"
#include "data_inst.h"
#include "prior_box_inst.h"
#include "input_layout_inst.h"
#include "implementation_map.h"
#include "register_gpu.hpp"

#include "network_impl.h"
#include <vector>

namespace cldnn {
namespace gpu {

class wait_for_events_gpu : public primitive_impl {
public:
    explicit wait_for_events_gpu(const program_node& /*node*/) {}

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<wait_for_events_gpu>(*this);
    }

    void init_kernels() override {}
    void set_arguments(primitive_inst& /*instance*/) override {}

    event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& instance) override {
        auto& stream = instance.get_network().get_stream();
        return stream.enqueue_marker(events);
    }

    bool validate(const primitive_inst&) const override { return true; }

    static primitive_impl* create_data(const data_node& data) { return new wait_for_events_gpu(data); }

    static primitive_impl* create_input_layout(const input_layout_node& input) {
        return new wait_for_events_gpu(input);
    }

    static primitive_impl* create_prior_box(const prior_box_node& prior_box) {
        // This primitive is being executed on CPU during network compilation.
        return new wait_for_events_gpu(prior_box);
    }
};

namespace detail {

attach_data_gpu::attach_data_gpu() {
    implementation_map<data>::add({ {engine_types::ocl, wait_for_events_gpu::create_data} });
}

attach_input_layout_gpu::attach_input_layout_gpu() {
    implementation_map<input_layout>::add({{engine_types::ocl, wait_for_events_gpu::create_input_layout}});
}

attach_prior_box_gpu::attach_prior_box_gpu() {
    implementation_map<prior_box>::add({{engine_types::ocl, wait_for_events_gpu::create_prior_box}});
}

}  // namespace detail
}  // namespace gpu
}  // namespace cldnn
