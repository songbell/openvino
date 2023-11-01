// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/constant.hpp"

#include "intel_gpu/primitives/rpe.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program_builder.hpp"
#include <ov_ops/rotary_positional_embeddings.hpp>

namespace ov {
namespace intel_gpu {

namespace {

void CreateRPEOp(ProgramBuilder& p, const std::shared_ptr<ov::op::internal::RPE>& op) {
    // TBD
    validate_inputs_count(op, {3});
}

}  // namespace

REGISTER_FACTORY_IMPL(internal, RPE);

}  // namespace intel_gpu
}  // namespace ov
