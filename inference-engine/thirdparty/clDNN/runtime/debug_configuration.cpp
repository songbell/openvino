// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/debug_configuration.hpp"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <memory>
#include <vector>
#include <sstream>

namespace cldnn {

const char *debug_configuration::prefix = "GPU_Debug: ";

// Default policy is that dump_configuration will override other configuration from IE.

#ifdef GPU_DEBUG_CONFIG

template<typename T>
void print_option(std::string option_name, T option_value) {
    GPU_DEBUG_COUT << "Config " << option_name << " = " << option_value << std::endl;
}

static std::string to_upper_case(const std::string& var) {
    std::stringstream s;

    for (size_t i = 0; i < var.size(); i++) {
        if (std::isupper(var[i])) {
            if (i != 0) {
                s << "_";
            }
            s << var[i];
        } else {
            s << static_cast<char>(std::toupper(var[i]));
        }
    }

    return s.str();
}

static std::vector<std::string> get_possible_option_names(const std::string& var, std::vector<std::string> allowed_option_prefixes) {
    std::vector<std::string> result;

    for (auto& prefix : allowed_option_prefixes) {
        result.push_back(prefix + var);
        result.push_back(prefix + to_upper_case(var));
    }

    return result;
}

template <typename T>
T convert_to(const std::string &str) {
    std::istringstream ss(str);
    T res;
    ss >> res;
    return res;
}

template <>
std::string convert_to(const std::string &str) {
    return str;
}

template<typename T>
void get_debug_env_var(const std::string &var, T &val, std::vector<std::string> allowed_option_prefixes) {
    bool found = false;
    for (auto o : get_possible_option_names(var, allowed_option_prefixes)) {
        if (const auto env_var = std::getenv(o.c_str())) {
            val = convert_to<T>(env_var);
            found = true;
        }
    }

    if (found) {
        print_option(var, val);
    }
}

template<typename T>
void get_gpu_debug_env_var(const std::string &var, T &val) {
    return get_debug_env_var(var, val, {"OV_GPU_"});
}

template<typename T>
void get_common_debug_env_var(const std::string &var, T &val) {
    // The list below should be prioritized from lowest to highest prefix priority
    // If an option is set several times with different prefixes, version with the highest priority will be actually used.
    // This may allow to enable global option with some value and override this value for GPU plugin
    // For example: OV_GPU_Verbose=2 OV_Verbose=1 ./my_app => this->verbose == 2
    // In that case we enable Verbose (with level = 1) for all OV components that support this option, but for GPU plugin we increase verbose level to 2
    std::vector<std::string> allowed_option_prefixes = {
        "OV_",
        "OV_GPU_"
    };

    return get_debug_env_var(var, val, allowed_option_prefixes);
}

static void print_help_messages() {
    std::vector<std::pair<std::string, std::string>> message_list;
    message_list.emplace_back("OV_GPU_Help", "Print help messages");
    message_list.emplace_back("OV_GPU_Verbose", "Verbose execution");
    message_list.emplace_back("OV_GPU_PrintMultiKernelPerf", "Print execution time of each kernel in multi-kernel primitimive");
    message_list.emplace_back("OV_GPU_DisableUsm", "Disable usm usage");
    message_list.emplace_back("OV_GPU_DisableOnednn", "Disable onednn for discrete GPU (no effect for integrated GPU)");
    message_list.emplace_back("OV_GPU_DumpGraphs", "Dump optimized graph");
    message_list.emplace_back("OV_GPU_DumpSources", "Dump opencl sources");
    message_list.emplace_back("OV_GPU_DumpLayersPath", "Enable dumping intermediate buffers and set the dest path");
    message_list.emplace_back("OV_GPU_DumpLayers", "Dump intermediate buffers of specified layers only, separated by space");
    message_list.emplace_back("OV_GPU_DumpLayersDstOnly", "Dump only output of layers");
    message_list.emplace_back("OV_GPU_DumpLayersLimitBatch", "Limit the size of batch to dump");
    message_list.emplace_back("OV_GPU_DryRunPath", "Dry run and serialize execution graph into the specified path");
    message_list.emplace_back("OV_GPU_BaseBatchForMemEstimation", "Base batch size to be used in memory estimation");

    auto max_name_length_item = std::max_element(message_list.begin(), message_list.end(),
        [](std::pair<std::string, std::string>& a, std::pair<std::string, std::string>& b){
            return a.first.size() < b.first.size();
    });
    int name_width = static_cast<int>(max_name_length_item->first.size()) + 2;

    GPU_DEBUG_COUT << "Supported environment variables for debugging" << std::endl;
    for (auto& p : message_list) {
        GPU_DEBUG_COUT << " - " << std::left << std::setw(name_width) << p.first + "  " << p.second << std::endl;
    }
}

#endif

debug_configuration::debug_configuration()
        : help(0)
        , verbose(0)
        , print_multi_kernel_perf(0)
        , disable_usm(0)
        , dump_graphs(std::string())
        , dump_sources(std::string())
        , dump_layers_path(std::string())
        , dump_layers_dst_only(0)
        , dry_run_path(std::string())
        , disable_onednn(0)
        , dump_layers_limit_batch(std::numeric_limits<int>::max())
        , base_batch_for_memory_estimation(-1) {
#ifdef GPU_DEBUG_CONFIG
    get_gpu_debug_env_var("Help", help);
    get_common_debug_env_var("Verbose", verbose);
    get_gpu_debug_env_var("PrintMultiKernelPerf", print_multi_kernel_perf);
    get_gpu_debug_env_var("DisableUsm", disable_usm);
    get_gpu_debug_env_var("DumpGraphs", dump_graphs);
    get_gpu_debug_env_var("DumpSources", dump_sources);
    get_gpu_debug_env_var("DumpLayersPath", dump_layers_path);
    get_gpu_debug_env_var("DumpLayersDstOnly", dump_layers_dst_only);
    get_gpu_debug_env_var("DumpLayersLimitBatch", dump_layers_limit_batch);
    get_gpu_debug_env_var("DisableOnednn", disable_onednn);
    get_gpu_debug_env_var("DryRunPath", dry_run_path);
    get_gpu_debug_env_var("BaseBatchForMemEstimation", base_batch_for_memory_estimation);
    std::string dump_layers_str;
    get_gpu_debug_env_var("DumpLayers", dump_layers_str);

    if (help > 0) {
        print_help_messages();
        exit(0);
    }

    if (dump_layers_str.length() > 0) {
        dump_layers_str = " " + dump_layers_str + " "; // Insert delimiter for easier parsing when used
        std::stringstream ss(dump_layers_str);
        std::string layer;
        while (ss >> layer) {
            dump_layers.push_back(layer);
        }
    }
#endif
}

const debug_configuration *debug_configuration::get_instance() {
    static std::unique_ptr<debug_configuration> instance(nullptr);
#ifdef GPU_DEBUG_CONFIG
    static std::mutex _m;
    std::lock_guard<std::mutex> lock(_m);
    if (nullptr == instance)
        instance.reset(new debug_configuration());
    return instance.get();
#else
    return nullptr;
#endif
}

bool debug_configuration::is_dumped_layer(const std::string& layerName) const {
#ifdef GPU_DEBUG_CONFIG
    if (dump_layers.empty()) return true;
    auto iter = std::find_if(dump_layers.begin(), dump_layers.end(), [&](const std::string& dl){
        return (layerName.find(dl) != std::string::npos);
    });
    return (iter != dump_layers.end());
#else
    return false;
#endif
}
} // namespace cldnn
