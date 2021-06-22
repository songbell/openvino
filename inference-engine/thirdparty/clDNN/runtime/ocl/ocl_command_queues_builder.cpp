// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ocl_command_queues_builder.hpp"
#include "cldnn/runtime/error_handler.hpp"
#include <string>

namespace cldnn {
namespace ocl {

command_queues_builder::command_queues_builder()
    : _profiling(false),
      _out_of_order(false),
      _priority_mode(priority_mode_types::disabled),
      _throttle_mode(throttle_mode_types::disabled) {}

cl_command_queue_properties command_queues_builder::get_properties() {
    cl_command_queue_properties ret =
        ((_profiling ? CL_QUEUE_PROFILING_ENABLE : 0) | (_out_of_order ? CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE : 0));
    return ret;
}

ocl_queue_type command_queues_builder::build(const cl::Context& context, const cl::Device& device) {
    auto properties = get_properties();

    ocl_queue_type queue;

    if (_priority_mode == priority_mode_types::disabled && _throttle_mode == throttle_mode_types::disabled) {
        queue = ocl_queue_type(context, device, properties);
    }

    unsigned cl_queue_priority_value = CL_QUEUE_PRIORITY_MED_KHR;

    switch (_priority_mode) {
        case priority_mode_types::high:
            cl_queue_priority_value = CL_QUEUE_PRIORITY_HIGH_KHR;
            break;
        case priority_mode_types::low:
            cl_queue_priority_value = CL_QUEUE_PRIORITY_LOW_KHR;
            break;
        default:
            break;
    }

    unsigned cl_queue_throttle_value = CL_QUEUE_THROTTLE_MED_KHR;

    switch (_throttle_mode) {
        case throttle_mode_types::high:
            cl_queue_throttle_value = CL_QUEUE_THROTTLE_HIGH_KHR;
            break;
        case throttle_mode_types::low:
            cl_queue_throttle_value = CL_QUEUE_THROTTLE_LOW_KHR;
            break;
        default:
            break;
    }

    cl_int error_code = CL_SUCCESS;

    if (_priority_mode != priority_mode_types::disabled && _throttle_mode != throttle_mode_types::disabled) {
        cl_queue_properties properties_low[] = {CL_QUEUE_PRIORITY_KHR,
                                                cl_queue_priority_value,
                                                CL_QUEUE_THROTTLE_KHR,
                                                cl_queue_throttle_value,
                                                CL_QUEUE_PROPERTIES,
                                                properties,
                                                0};

        queue = ocl_queue_type(clCreateCommandQueueWithProperties(context.get(), device.get(), properties_low, &error_code));
    } else if (_priority_mode != priority_mode_types::disabled) {
        cl_queue_properties properties_low[] = {CL_QUEUE_PRIORITY_KHR,
                                                cl_queue_priority_value,
                                                CL_QUEUE_PROPERTIES,
                                                properties,
                                                0};

        queue = ocl_queue_type(clCreateCommandQueueWithProperties(context.get(), device.get(), properties_low, &error_code));
    } else if (_throttle_mode != throttle_mode_types::disabled) {
        cl_queue_properties properties_low[] = {CL_QUEUE_THROTTLE_KHR,
                                                cl_queue_throttle_value,
                                                CL_QUEUE_PROPERTIES,
                                                properties,
                                                0};

        queue = ocl_queue_type(clCreateCommandQueueWithProperties(context.get(), device.get(), properties_low, &error_code));
    }

    if (error_code != CL_SUCCESS) {
        CLDNN_ERROR_MESSAGE("Command queues builders",
                            "clCreateCommandQueueWithPropertiesINTEL error " + std::to_string(error_code));
    }

    return queue;
}

void command_queues_builder::set_priority_mode(priority_mode_types priority, bool extension_support) {
    if (priority != priority_mode_types::disabled && !extension_support) {
        CLDNN_ERROR_MESSAGE("Command queues builders - priority_mode",
                            std::string("The param priority_mode is set in engine_configuration, ")
                            .append("but cl_khr_priority_hints or cl_khr_create_command_queue ")
                            .append("is not supported by current OpenCL implementation."));
    }
    _priority_mode = priority;
}

void command_queues_builder::set_throttle_mode(throttle_mode_types throttle, bool extension_support) {
    if (throttle != throttle_mode_types::disabled && !extension_support) {
        CLDNN_ERROR_MESSAGE("Command queues builders - throttle_mode",
                            std::string("The param throttle_mode is set in engine_configuration, ")
                            .append("but cl_khr_throttle_hints is not supported by current OpenCL implementation."));
    }
    _throttle_mode = throttle;
}
}  // namespace ocl
}  // namespace cldnn
