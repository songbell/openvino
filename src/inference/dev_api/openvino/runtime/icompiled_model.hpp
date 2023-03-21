// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief OpenVINO Runtime CompiledModel interface
 * @file openvino/runtime/icompiled_model.hpp
 */

#pragma once

#include <memory>
#include <ostream>
#include <vector>

#include "openvino/core/node_output.hpp"
#include "openvino/runtime/common.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/remote_context.hpp"
#include "openvino/runtime/threading/cpu_streams_executor.hpp"
#include "openvino/runtime/threading/itask_executor.hpp"

namespace InferenceEngine {
class ICompiledModelWrapper;
}  // namespace InferenceEngine

namespace ov {

class CoreImpl;
class IPlugin;
class IExecutableNetworkWrapper;
class IAsyncInferRequest;

/**
 * @brief OpenVINO ICompiledModel interface
 * @ingroup ov_dev_api_compiled_model_api
 */
class OPENVINO_RUNTIME_API ICompiledModel : public std::enable_shared_from_this<ICompiledModel> {
public:
    /**
     * @brief Constructor for ICompiledModel interface
     *
     * @param model OpenVINO model representation
     *
     * @param plugin Pointer to plugin
     *
     * @param task_executor Task executor (CPUStreamsExecutor by default)
     *
     * @param callback_executor Callback executor (CPUStreamsExecutor by default)
     */
    ICompiledModel(
        const std::shared_ptr<const ov::Model>& model,
        const std::shared_ptr<const ov::IPlugin>& plugin,
        const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor =
            std::make_shared<ov::threading::CPUStreamsExecutor>(ov::threading::IStreamsExecutor::Config{"Default"}),
        const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor =
            std::make_shared<ov::threading::CPUStreamsExecutor>(ov::threading::IStreamsExecutor::Config{"Callback"}));

    /**
     * @brief Constructor for ICompiledModel interface with remote context
     *
     * @param model OpenVINO model representation
     *
     * @param plugin Pointer to plugin
     *
     * @param context Remote context
     *
     * @param task_executor Task executor (CPUStreamsExecutor by default)
     *
     * @param callback_executor Callback executor (CPUStreamsExecutor by default)
     */
    ICompiledModel(
        const std::shared_ptr<const ov::Model>& model,
        const std::shared_ptr<const ov::IPlugin>& plugin,
        const ov::RemoteContext& context,
        const std::shared_ptr<ov::threading::ITaskExecutor>& task_executor =
            std::make_shared<ov::threading::CPUStreamsExecutor>(ov::threading::IStreamsExecutor::Config{"Default"}),
        const std::shared_ptr<ov::threading::ITaskExecutor>& callback_executor =
            std::make_shared<ov::threading::CPUStreamsExecutor>(ov::threading::IStreamsExecutor::Config{"Callback"}));

    /**
     * @brief Gets all outputs from compiled model
     *
     * @return model outputs
     */
    const std::vector<ov::Output<const ov::Node>>& outputs() const;

    /**
     * @brief Gets all inputs from compiled model
     *
     * @return model inputs
     */
    const std::vector<ov::Output<const ov::Node>>& inputs() const;

    /**
     * @brief Create infer request
     *
     * @return Asynchronous infer request interface
     */
    virtual std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const;

    /**
     * @brief Export compiled model to stream
     *
     * @param model output stream
     */
    virtual void export_model(std::ostream& model) const = 0;

    /**
     * @brief Returns runtime model
     *
     * @return OpenVINO Model which represents runtime graph
     */
    virtual std::shared_ptr<const ov::Model> get_runtime_model() const = 0;

    /**
     * @brief Allows to set propertu
     *
     * @param properties new plugin properties
     */
    virtual void set_property(const ov::AnyMap& properties) = 0;

    /**
     * @brief Returns property
     *
     * @param name Property name
     *
     * @return Property value
     */
    virtual ov::Any get_property(const std::string& name) const = 0;

    /**
     * @brief Creates device specific remote context
     *
     * @return OpenVINO RemoteContext
     */
    std::shared_ptr<ov::IRemoteContext> get_context() const;

private:
    std::shared_ptr<const ov::IPlugin> m_plugin;
    std::vector<ov::Output<const ov::Node>> m_inputs;
    std::vector<ov::Output<const ov::Node>> m_outputs;
    ov::RemoteContext m_context;

    std::shared_ptr<ov::threading::ITaskExecutor> m_task_executor = nullptr;      //!< Holds a task executor
    std::shared_ptr<ov::threading::ITaskExecutor> m_callback_executor = nullptr;  //!< Holds a callback executor

    friend ov::CoreImpl;
    friend ov::IExecutableNetworkWrapper;
    friend InferenceEngine::ICompiledModelWrapper;

    // FIXME: Remove after removing IE API
    std::vector<std::shared_ptr<const ov::Node>> _parameters;
    std::vector<std::shared_ptr<const ov::Node>> _results;

protected:
    /**
     * @brief Method creates infer request implementation
     *
     * @return Sync infer request
     */
    virtual std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const = 0;

    /**
     * @brief Default implementation of create async inter request method
     *
     * @tparam AsyncInferRequestType Async infer request type. ov::IAsyncInferRequest by
     * default
     *
     * @return Asynchronous infer request
     */
    template <typename AsyncInferRequestType = ov::IAsyncInferRequest>
    std::shared_ptr<ov::IAsyncInferRequest> create_async_infer_request() const {
        auto syncRequestImpl = create_sync_infer_request();
        return std::make_shared<AsyncInferRequestType>(syncRequestImpl, m_task_executor, m_callback_executor);
    }

    /**
     * @brief Returns pointer to the plugin
     *
     * @return OpenVINO Plugin interface
     */
    const std::shared_ptr<const ov::IPlugin>& get_plugin() const;
    const std::shared_ptr<ov::threading::ITaskExecutor> get_task_executor() const;
    const std::shared_ptr<ov::threading::ITaskExecutor> get_callback_executor() const;
};

}  // namespace ov
