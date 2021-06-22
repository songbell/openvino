// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "profiling.hpp"

#include <list>
#include <mutex>
#include <utility>
#include <utility>
#include <functional>

namespace cldnn {
struct user_event;

/// @brief user-defined event handler callback.
using event_handler = std::function<void(void*)>;

struct event {
public:
    using ptr = std::shared_ptr<event>;
    event() = default;

    void wait();
    bool is_set();
    virtual bool is_valid() const { return _attached; }
    virtual void reset() {
        _attached = false;
        _set = false;
        _profiling_captured = false;
        _profiling_info.clear();
    }

    // returns true if handler has been successfully added
    bool add_event_handler(event_handler handler, void* data);

    std::vector<instrumentation::profiling_interval> get_profiling_info();

private:
    std::mutex _handlers_mutex;
    std::list<std::pair<event_handler, void*>> _handlers;

    bool _profiling_captured = false;
    std::list<instrumentation::profiling_interval> _profiling_info;

protected:
    bool _set = false;
    bool _attached =
        false;  // because ocl event can be attached later, we need mechanism to check if such event was attached
    void call_handlers();

    virtual void wait_impl() = 0;
    virtual bool is_set_impl() = 0;
    virtual bool add_event_handler_impl(event_handler, void*) { return true; }

    // returns whether profiling info has been captures successfully and there's no need to call this impl a second time
    // when user requests to get profling info
    virtual bool get_profiling_info_impl(std::list<instrumentation::profiling_interval>&) { return true; }
};

struct user_event : virtual public event {
public:
    explicit user_event(bool set = false) { _set = set; }

    void set() {
        if (_set)
            return;
        _set = true;
        set_impl();
        call_handlers();
    }

private:
    virtual void set_impl() = 0;
};

}  // namespace cldnn
