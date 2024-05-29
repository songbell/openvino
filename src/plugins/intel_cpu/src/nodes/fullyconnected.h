// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>

#include <memory>
#include <oneapi/dnnl/dnnl.hpp>
#include <string>
#include <vector>

#include "cpu_memory.h"
#include "nodes/executors/executor_factory.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "nodes/executors/fullyconnected_config.hpp"
#include "post_ops.hpp"
#include "openvino/runtime/threading/cpu_message.hpp"

namespace ov {
namespace intel_cpu {
namespace node {

class FullyConnected : public Node {
public:
    FullyConnected(const std::shared_ptr<ov::Node>& op, const GraphContext::CPtr context);

    void getSupportedDescriptors() override{};
    void execute(dnnl::stream strm) override;
    bool created() const override;

    bool canBeInPlace() const override {
        return false;
    }

    int getFusingAxis() const override {
        return getOutputShapeAtPort(0).getRank() == 3 ? 2 : 1;
    }

    const std::vector<impl_desc_type>& getDefaultImplPriority() override;

    size_t descInputNumbers() override {
        return static_cast<size_t>(getOriginalInputsNumber());
    }

    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;

    ov::element::Type getRuntimePrecision() const override;

    bool canFuse(const NodePtr& node) const override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    void prepareParams() override;
    void executeDynamicImpl(dnnl::stream strm) override;
    bool canBeExecutedInInt8() const override;
    void keepWeightsNonTransposed(bool weightsNonTransposed) {
        this->attrs.weightsNonTransposed = weightsNonTransposed;
    }

    void fuseDecompressionMultiply(const MemoryCPtr& memory);
    void fuseDecompressionSubtract(const MemoryCPtr& memory);

    MemoryPtr split_h(const MemoryPtr src, int dim, int w_rank, int w_size, bool need_fill=true);
    MemoryPtr split_v(const MemoryPtr src, int dim, int w_rank, int w_size, bool need_fill=true);
    void allreduce(void *send_buf, void *recv_buf, size_t count, ov::element::Type dtype);

protected:
    void toNumaNodeImpl(int numaID) override;

private:
    static const size_t DATA_ID = 0;
    static const size_t WEIGHTS_ID = 1;
    static const size_t BIAS_ID = 2;

    ExecutorPtr createExecutor();
    void fuseDecompressionConstant(const MemoryCPtr& memory, MemoryCPtr& decompressionValuesPtr);

    FCAttrs attrs;
    PostOps postOps;
    MemoryArgs memory;
    ExecutorFactoryPtr<FCAttrs, node::FullyConnected> factory;
    ExecutorPtr executor = nullptr;
    std::string errorPrefix;
    int w_rank = -1;
    int w_size = -1;
    std::shared_ptr<ov::threading::MessageManager> message = nullptr;
    /*
     * 1: allreduce     : split src and wgt, element-add dst
     * 2: allgather_h   : split src and wgt, concat in horizontal direction
     * 3: allgather_v   : split src(batch size > 1 is required.), concat in vertical direction
    */
    int tp_mode;
    MemoryPtr cached_splited_weight;
    MemoryPtr cached_splited_bias;
    MemoryPtr cached_scale = nullptr;
    MemoryPtr cached_zeropoint= nullptr;
};

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
