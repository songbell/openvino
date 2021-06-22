// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "graph_transformer.hpp"

#include <vpu/model/base.hpp>

namespace vpu {

CompiledGraph::Ptr compileModel(
        const Model& model,
        ncDevicePlatform_t platform,
        const PluginConfiguration& config,
        const Logger::Ptr& log);

}  // namespace vpu
