// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

namespace cldnn
{
    struct program_node;
    struct program_impl;
    // This class is intended to allow using private methods from program_impl within tests_core_internal project.
    // Once needed, more methods wrapper should be added here.
    class program_impl_wrapper
    {
    public:
        static void add_connection(program_impl& p, program_node& prev, program_node& next)
        {
            p.add_connection(prev, next);
        }
        template <class Pass, typename... Args>
        static void apply_opt_pass(program_impl& p, Args&&... args)
        {
            p.apply_opt_pass<Pass>(std::forward<Args>(args)...);
        }
        static void run_graph_compilation(program_impl& p)
        {
            p.run_graph_compilation();
        }
        static void compile(program_impl& p)
        {
            p.compile();
        }
        static void build(program_impl& p)
        {
            program_impl_wrapper::run_graph_compilation(p);
            program_impl_wrapper::compile(p);
            program_impl_wrapper::init_kernels(p);
        }
        static void init_kernels(program_impl& p)
        {
            p.init_kernels();
        }
        static void prepare_memory_dependencies(program_impl& p)
        {
            p.prepare_memory_dependencies();
        }
    };

}
