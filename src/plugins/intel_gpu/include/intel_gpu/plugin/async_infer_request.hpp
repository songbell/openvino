// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/iasync_infer_request.hpp"
#include "intel_gpu/plugin/sync_infer_request.hpp"
#include <string>
#include <map>

namespace ov {
namespace intel_gpu {

class AsyncInferRequest : public ov::IAsyncInferRequest {
public:
    using Parent = ov::IAsyncInferRequest;
    AsyncInferRequest(const std::shared_ptr<SyncInferRequest>& infer_request,
                      const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor,
                      const std::shared_ptr<ov::threading::ITaskExecutor>& wait_executor,
                      const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor);

    ~AsyncInferRequest() override;
    void setSubInfer(bool sub_infer) {
        m_sub_infers = sub_infer;
    }
    void start_async() override;
    bool m_sub_infers = false;

private:
    std::shared_ptr<SyncInferRequest> m_infer_request;
    std::shared_ptr<ov::threading::ITaskExecutor> m_wait_executor;
};

}  // namespace intel_gpu
}  // namespace ov
