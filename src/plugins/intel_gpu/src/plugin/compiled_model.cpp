// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_attention.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/runtime/internal_properties.hpp"

#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "intel_gpu/plugin/graph.hpp"
#include "intel_gpu/plugin/compiled_model.hpp"
#include "intel_gpu/plugin/async_infer_request.hpp"
#include "openvino/runtime/threading/cpu_message.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "plugin/transformations/pa_tensor_parallel.hpp"
#include "plugin/transformations/fc_all_reduce.hpp"
#include "plugin/transformations/remaining_fc_parallel.hpp"
#include <sys/types.h>
#include "openvino/pass/visualize_tree.hpp"
#include "plugin/transformations/fc_horizontal_fusion.hpp"
namespace ov {
namespace intel_gpu {

namespace {
std::shared_ptr<ov::threading::ITaskExecutor> create_task_executor(const std::shared_ptr<const ov::IPlugin>& plugin,
                                                                   const ExecutionConfig& config) {
    if (config.get_property(ov::internal::exclusive_async_requests)) {
        // exclusive_async_requests essentially disables the streams (and hence should be checked first) => aligned with
        // the CPU behavior
        return plugin->get_executor_manager()->get_executor("GPU");
    } else if (config.get_property(ov::hint::enable_cpu_pinning)) {
        return std::make_shared<ov::threading::CPUStreamsExecutor>(
            ov::threading::IStreamsExecutor::Config{"Intel GPU plugin executor",
                                                    config.get_property(ov::num_streams),
                                                    1,
                                                    ov::hint::SchedulingCoreType::PCORE_ONLY,
                                                    true});
    } else if (config.enableSubStreams) {
        return std::make_shared<ov::threading::CPUStreamsExecutor>(
            ov::threading::IStreamsExecutor::Config{"Intel GPU plugin executor", config.get_property(ov::num_streams)});
    } else {
        if (config.subStreamExecConfig.get_name() != "StreamsExecutor") {
            ov::threading::IStreamsExecutor::Config executor_confg = std::move(config.subStreamExecConfig);
            return std::make_shared<ov::threading::CPUStreamsExecutor>(executor_confg);
        } else {
            return std::make_shared<ov::threading::CPUStreamsExecutor>(
                ov::threading::IStreamsExecutor::Config{"Intel GPU plugin executor", config.get_property(ov::num_streams)});
        }
    }
}
}  // namespace

CompiledModel::CompiledModel(std::shared_ptr<ov::Model> model,
                             const std::shared_ptr<const ov::IPlugin>& plugin,
                             RemoteContextImpl::Ptr context,
                             const ExecutionConfig& config,
                             const std::shared_ptr<SubMemoryManager> sub_memory_manager)
    : ov::ICompiledModel(model,
                         plugin,
                         context,
                         create_task_executor(plugin, config),
                         nullptr)
    // : ov::ICompiledModel::ICompiledModel(model, plugin)
    , m_context(context)
    , m_config(config)
    , m_wait_executor(std::make_shared<ov::threading::CPUStreamsExecutor>(ov::threading::IStreamsExecutor::Config{"Intel GPU plugin wait executor"}))
    , m_model_name(model->get_friendly_name())
    , m_inputs(ov::ICompiledModel::inputs())
    , m_outputs(ov::ICompiledModel::outputs())
    , m_loaded_from_cache(false) {
    auto graph_base = std::make_shared<Graph>(model, m_context, m_config, 0, sub_memory_manager);
    for (uint16_t n = 0; n < m_config.get_property(ov::num_streams); n++) {
        auto graph = n == 0 ? graph_base : std::make_shared<Graph>(graph_base, n);
        m_graphs.push_back(graph);
    }
    if (m_config.enableSubStreams) {
        std::vector<ExecutionConfig> configs_for_tp;
        configs_for_tp.resize(m_config.get_context_for_tp().size());
        m_has_sub_compiled_models = true;
        auto message = ov::threading::message_manager();
        auto tp_compile_executor = get_plugin()->get_executor_manager()->get_idle_cpu_streams_executor(ov::threading::IStreamsExecutor::Config{
                "async compile executor for TP",
                static_cast<int>(std::thread::hardware_concurrency()) /* max possible #streams*/,
                0});
        m_sub_memory_manager = std::make_shared<SubMemoryManager>(m_config.get_context_for_tp().size());
        message->set_num_sub_streams(m_config.get_context_for_tp().size());
        std::vector<std::shared_ptr<ov::ICompiledModel>> sub_models;
        std::vector<ov::threading::Task> sub_tasks;
        for (size_t i = 0; i < m_config.get_context_for_tp().size(); i++) {
            auto compile_tp_model = [&](size_t i) {
                configs_for_tp[i] = m_config;
                configs_for_tp[i].enableSubStreams = false;
                auto streamExecutorConfig = ov::threading::IStreamsExecutor::Config{"GPUStreamsExecutor",
                                                                 1,
                                                                 0,
                                                                 ov::hint::SchedulingCoreType::ANY_CORE,
                                                                 false,
                                                                 false,
                                                                 {},
                                                                 configs_for_tp[i].streamsRankTable[i]};
                configs_for_tp[i].subStreamExecConfig = std::move(streamExecutorConfig);
                auto model_clone = model->clone();
                //ov::serialize(model_clone, "./model_clone_original.xml");
                //ov::serialize(model_clone, "./model_pa_o.xml", "./model_pa_o.bin");
                ov::pass::Manager manager;

                const char* env = getenv("OV_TP_ALLREDUCE_TEST");
                if (env) {
                    std::cout << "Notice: GPU TP allReduce test only!" << std::endl;
                    manager.register_pass<FCALLReduce>(m_config.get_context_for_tp().size(), i);
                    manager.run_passes(model_clone);
                } else {
                    bool has_pa_op = false;
                    for (const auto& op : model_clone->get_ops()) {
                        if (std::dynamic_pointer_cast<ov::op::PagedAttentionExtension>(op)) {
                            has_pa_op = true;
                            break;
                        }
                    }

                    if (has_pa_op) {
                        std::map<size_t, ov::PartialShape> shapes;
                        const auto& params = model_clone->get_parameters();
                        for (size_t input_id = 0; input_id < params.size(); input_id++) {
                            const auto& param = params[input_id];
                            shapes[input_id] = param->get_output_partial_shape(0);
                            if (param->get_friendly_name().find("_cache") != std::string::npos) {
                                auto head_num = shapes[input_id][1];
                                shapes[input_id][1] = head_num / config.get_context_for_tp().size();
                            }
                        }
                        model_clone->reshape(shapes);
                        manager.register_pass<ov::intel_gpu::PATensorParallelFusion>(config.get_context_for_tp().size(),
                                                                                     i);
                    }
                    manager.register_pass<ov::intel_gpu::RemainFCParallelFusion>(config.get_context_for_tp().size(), i);
                    manager.run_passes(model_clone);
                }
                //ov::serialize(model_clone, "integrated_vllm_pa_" + std::to_string(i) + ".xml");
                m_sub_compiled_models.push_back(std::make_shared<CompiledModel>(
                    model_clone, plugin, m_config.get_context_for_tp()[i].as<RemoteContextImpl::Ptr>(), configs_for_tp[i], m_sub_memory_manager));
                GPU_DEBUG_TRACE_DETAIL << "sub models for TP created, rank " << configs_for_tp[i].streamsRankTable[i][0] << std::endl;
            };
            sub_tasks.push_back(std::bind(compile_tp_model, i));
        }
        for (auto & iter : sub_tasks)
            tp_compile_executor->run_and_wait({std::move(iter)});

        // clear up the tp executor for async compile
        get_plugin()->get_executor_manager()->clear("async compile executor for TP");
        tp_compile_executor.reset();
    }
}

CompiledModel::CompiledModel(cldnn::BinaryInputBuffer& ib,
                             const std::shared_ptr<const ov::IPlugin>& plugin,
                             RemoteContextImpl::Ptr context,
                             const ExecutionConfig& config,
                             const bool loaded_from_cache)
    : ov::ICompiledModel(nullptr,
                         plugin,
                         context,
                         create_task_executor(plugin, config),
                         nullptr)
    , m_context(context)
    , m_config(config)
    , m_wait_executor(std::make_shared<ov::threading::CPUStreamsExecutor>(ov::threading::IStreamsExecutor::Config{"Intel GPU plugin wait executor"}))
    , m_model_name("")
    , m_loaded_from_cache(loaded_from_cache) {
    {
        size_t num_params;
        ib >> num_params;

        for (size_t idx = 0; idx < num_params; ++idx) {
            std::string param_name;
            ib >> param_name;
            ov::element::Type param_element_type;
            std::string str_element_type;
            ib >> str_element_type;
            std::stringstream oss(str_element_type);
            oss >> param_element_type;
            ov::PartialShape param_shape;
            ib >> param_shape;
            std::unordered_set<std::string> param_names;
            size_t num_names;
            ib >> num_names;
            for (size_t i = 0; i < num_names; ++i) {
                std::string name;
                ib >> name;
                param_names.emplace(name);
            }

            auto new_param = std::make_shared<ov::op::v0::Parameter>(param_element_type, param_shape);
            new_param->set_friendly_name(param_name);
            new_param->set_element_type(param_element_type);
            new_param->output(0).get_tensor().set_names(param_names);
            new_param->validate_and_infer_types();
            m_inputs.push_back(new_param->output(0));
        }
    }

    {
        size_t num_results;
        ib >> num_results;

        for (size_t idx = 0; idx < num_results; ++idx) {
            ov::element::Type fake_element_type;
            std::string str_element_type;
            ib >> str_element_type;
            std::stringstream oss(str_element_type);
            oss >> fake_element_type;

            ov::PartialShape fake_shape;
            ib >> fake_shape;

            std::string fake_name;
            ib >> fake_name;

            std::string param_name;
            ib >> param_name;

            std::unordered_set<std::string> param_names;
            size_t num_names;
            ib >> num_names;
            for (size_t i = 0; i < num_names; ++i) {
                std::string name;
                ib >> name;
                param_names.emplace(name);
            }

            auto fake_param = std::make_shared<ov::op::v0::Parameter>(fake_element_type, fake_shape);
            fake_param->set_friendly_name(fake_name);
            fake_param->validate_and_infer_types();

            auto new_result = std::make_shared<ov::op::v0::Result>(fake_param);
            new_result->set_friendly_name(param_name);
            new_result->output(0).get_tensor().set_names(param_names);
            new_result->validate_and_infer_types();
            m_outputs.push_back(new_result->output(0));
        }
    }

    auto graph_base = std::make_shared<Graph>(ib, context, m_config, 0);
    for (uint16_t n = 0; n < m_config.get_property(ov::num_streams); n++) {
        auto graph = n == 0 ? graph_base : std::make_shared<Graph>(graph_base, n);
        m_graphs.push_back(graph);
    }
}

std::shared_ptr<ov::IAsyncInferRequest> CompiledModel::create_infer_request() const {
    auto sync_request = create_sync_infer_request();
    auto async_infer_request = std::make_shared<AsyncInferRequest>(std::static_pointer_cast<SyncInferRequest>(sync_request),
                                                                   get_task_executor(),
                                                                   m_wait_executor,
                                                                   get_callback_executor());
     if (m_has_sub_compiled_models) {
        std::vector<std::shared_ptr<IAsyncInferRequest>> requests;
        for (auto model : m_sub_compiled_models) {
            requests.push_back(model->create_infer_request());
        }
        async_infer_request->setSubInferRequest(requests);
        async_infer_request->setSubInfer(true);
    }
    return async_infer_request;
}

// Cache blob format:
//     [ is_dynamic flag ]
//     [ ov::Node::Input/ ov::Node::Output ]
//     [ ov::intel_gpu::Graph ]
void CompiledModel::export_model(std::ostream& model) const {
    if (m_config.get_property(ov::cache_mode) == ov::CacheMode::OPTIMIZE_SIZE)
        return;

    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "CompiledModel::export_model");
    OPENVINO_ASSERT(!m_graphs.empty(), "[GPU] Model not loaded");

    cldnn::BinaryOutputBuffer ob(model);

    // Inputs
    {
        const auto& params = inputs();
        ob << params.size();

        for (const auto& param : params) {
            std::stringstream ss;
            ss << param.get_element_type();

            ob << param.get_node()->get_friendly_name();
            ob << ss.str();
            ob << param.get_partial_shape();
            ob << param.get_names().size();
            for (const auto& name : param.get_names()) {
                ob << name;
            }
        }
    }

    // Outputs
    {
        const auto& results = outputs();
        ob << results.size();

        for (const auto& param : results) {
            std::stringstream ss;
            ss << param.get_element_type();

            ob << ss.str();
            ob << param.get_partial_shape();
            ob << param.get_node()->get_input_node_ptr(0)->get_friendly_name();
            ob << param.get_node()->get_friendly_name();
            ob << param.get_names().size();
            for (const auto& name : param.get_names()) {
                ob << name;
            }
        }
    }

    get_graph(0)->export_model(ob);
}

CompiledModel::Ptr CompiledModel::get_tp_compiled_model() const {
    auto messenger = ov::threading::message_manager();
    for (auto& iter : m_sub_compiled_models) {
        if (iter->get_context()->get_device_name() == get_context()->get_device_name())
            return std::dynamic_pointer_cast<CompiledModel>(iter);
    }
    return nullptr;
}

std::shared_ptr<const ov::Model> CompiledModel::get_runtime_model() const {
    if (m_config.enableSubStreams) {
       return get_tp_compiled_model()->get_runtime_model();
    }
    return get_graph(0)->get_runtime_model();
}

const std::vector<std::shared_ptr<Graph>>& CompiledModel::get_graphs() const {
    if (m_config.enableSubStreams) {
        return get_tp_compiled_model()->get_graphs();
    }
    return m_graphs;
}

std::shared_ptr<Graph> CompiledModel::get_graph(size_t n) const {
    if (m_config.enableSubStreams) {
        return get_tp_compiled_model()->get_graph(n);
    }
    OPENVINO_ASSERT(m_graphs.size() >= n, "[GPU] Invalid graph idx: ", n, ". Only ", m_graphs.size(), " were created");
    return m_graphs[n];
}

ov::Any CompiledModel::get_property(const std::string& name) const {
    if (name == ov::supported_properties) {
        return decltype(ov::supported_properties)::value_type {
            // Metrics
            ov::PropertyName{ov::supported_properties.name(), PropertyMutability::RO},
            ov::PropertyName{ov::model_name.name(), PropertyMutability::RO},
            ov::PropertyName{ov::optimal_number_of_infer_requests.name(), PropertyMutability::RO},

            // Configs
            ov::PropertyName{ov::enable_profiling.name(), PropertyMutability::RO},
            ov::PropertyName{ov::hint::enable_cpu_pinning.name(), PropertyMutability::RO},
            ov::PropertyName{ov::hint::model_priority.name(), PropertyMutability::RO},
            ov::PropertyName{ov::intel_gpu::hint::host_task_priority.name(), PropertyMutability::RO},
            ov::PropertyName{ov::intel_gpu::hint::queue_priority.name(), PropertyMutability::RO},
            ov::PropertyName{ov::intel_gpu::hint::queue_throttle.name(), PropertyMutability::RO},
            ov::PropertyName{ov::intel_gpu::enable_loop_unrolling.name(), PropertyMutability::RO},
            ov::PropertyName{ov::intel_gpu::disable_winograd_convolution.name(), PropertyMutability::RO},
            ov::PropertyName{ov::cache_dir.name(), PropertyMutability::RO},
            ov::PropertyName{ov::cache_mode.name(), PropertyMutability::RO},
            ov::PropertyName{ov::hint::performance_mode.name(), PropertyMutability::RO},
            ov::PropertyName{ov::hint::execution_mode.name(), PropertyMutability::RO},
            ov::PropertyName{ov::compilation_num_threads.name(), PropertyMutability::RO},
            ov::PropertyName{ov::num_streams.name(), PropertyMutability::RO},
            ov::PropertyName{ov::hint::num_requests.name(), PropertyMutability::RO},
            ov::PropertyName{ov::hint::inference_precision.name(), PropertyMutability::RO},
            ov::PropertyName{ov::hint::dynamic_quantization_group_size.name(), PropertyMutability::RO},
            ov::PropertyName{ov::device::id.name(), PropertyMutability::RO},
            ov::PropertyName{ov::execution_devices.name(), PropertyMutability::RO},
        };
    } else if (name == ov::model_name) {
        return decltype(ov::model_name)::value_type {m_model_name};
    } else if (name == ov::loaded_from_cache) {
        return decltype(ov::loaded_from_cache)::value_type {m_loaded_from_cache};
    } else if (name == ov::optimal_number_of_infer_requests) {
        unsigned int nr = m_config.get_property(ov::num_streams);
        if (m_config.get_property(ov::hint::performance_mode) != ov::hint::PerformanceMode::LATENCY)
            nr *= 2;
        return decltype(ov::optimal_number_of_infer_requests)::value_type {nr};
    } else if (name == ov::execution_devices) {
        return decltype(ov::execution_devices)::value_type{m_context->get_device_name()};
    }

    return m_config.get_property(name);
}

std::shared_ptr<ov::ISyncInferRequest> CompiledModel::create_sync_infer_request() const {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "CompiledModel::create_sync_infer_request");
    OPENVINO_ASSERT(!m_graphs.empty(), "[GPU] Model not loaded");

    for (auto& graph : m_graphs) {
        OPENVINO_ASSERT(graph != nullptr, "[GPU] Model not loaded: graph is nullptr");
        if (!m_config.enableSubStreams)
            OPENVINO_ASSERT(graph->is_loaded(), "[GPU] Model not loaded: invalid graph");
    }
    return std::make_shared<SyncInferRequest>(std::static_pointer_cast<const CompiledModel>(shared_from_this()));
}

}  // namespace intel_gpu
}  // namespace ov
