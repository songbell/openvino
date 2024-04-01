// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

namespace cldnn {

/// @brief
/// @details
struct mha : public primitive_base<mha> {
    CLDNN_DECLARE_PRIMITIVE(mha)

    mha() : primitive_base("", {}) {}

    /// @brief Constructs mha primitive.
    /// @param id This primitive id.
    /// @param inputq, inputk, inputv inputs of MHA.
    mha(const primitive_id& id,
                   const input_info& inputq,
                   const input_info& inputk,
                   const input_info& inputv,
                   const padding& output_padding = padding())
        : primitive_base(id, {inputq, inputk, inputv},
        {output_padding})
        {}

    size_t hash() const override {
        size_t seed = primitive::hash();
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;
        return true;
    }

    /* We don't have any argument to serialize at this moment. */
    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<mha>::save(ob);
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<mha>::load(ib);
    }
};
}  // namespace cldnn
