// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <vector>

#include "primitive.hpp"

namespace cldnn {

/// @brief rpe-7 primitive.
struct rpe : primitive_base<rpe> {
    CLDNN_DECLARE_PRIMITIVE(rpe)

    rpe() : primitive_base("", {}) {}

    /// @brief Constructs rpe primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param axis Tensor which specifies the number of places by which the elements are shifted.
    rpe(const primitive_id& id,
        const std::vector<input_info>& input,
        const int64_t axis,
        const padding& output_padding = padding())
        : primitive_base(id, {input}, {output_padding}), axis(axis) {}

    int64_t axis = 0;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, axis);
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const rpe>(rhs);

        return axis == rhs_casted.axis;
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<rpe>::save(ob);
        ob << axis;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<rpe>::load(ib);
        ib >> axis;
    }
};

}  // namespace cldnn
