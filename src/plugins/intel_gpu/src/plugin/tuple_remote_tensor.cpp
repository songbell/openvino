// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/tuple_remote_tensor.hpp"

#include <memory>

#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/plugin.hpp"
#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/plugin/remote_tensor.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "intel_gpu/runtime/memory_caps.hpp"
namespace ov {
namespace intel_gpu {

TupleRemoteTensorImpl::TupleRemoteTensorImpl(std::shared_ptr<TupleRemoteContextImpl> context, std::vector<ov::SoPtr<ov::IRemoteTensor>> tensors)
    : m_context(context)
    , m_tensors(tensors) {
    for (auto& tensor : m_tensors) {
        auto remote_tensor = std::dynamic_pointer_cast<RemoteTensorImpl>(tensor._ptr);
        m_remote_tensors.emplace_back(remote_tensor);
    }
}

TupleRemoteTensorImpl::~TupleRemoteTensorImpl() {
    deallocate();
}

ov::SoPtr<ov::IRemoteTensor> TupleRemoteTensorImpl::get_tensor(int index) const {
    return m_tensors[index];
}

const ov::element::Type& TupleRemoteTensorImpl::get_element_type() const {
    return m_tensors[0]->get_element_type();
}

const ov::Shape& TupleRemoteTensorImpl::get_shape() const {
    return m_tensors[0]->get_shape();
}

const ov::Strides& TupleRemoteTensorImpl::get_strides() const {
    return m_tensors[0]->get_strides();
}

const AnyMap& TupleRemoteTensorImpl::get_properties() const {
    return m_tensors[0]->get_properties();
}

void TupleRemoteTensorImpl::set_shape(ov::Shape shape) {
    for (auto& tensor : m_tensors) {
        tensor->set_shape(shape);
    }
}

bool TupleRemoteTensorImpl::deallocate() noexcept {
    bool deallocate = true;
    for (auto& tensor : m_remote_tensors) {
        deallocate &= tensor->deallocate();
    }
    return deallocate;
}

bool TupleRemoteTensorImpl::is_allocated() const noexcept {
    bool is_allocated = true;
    for (auto& tensor : m_remote_tensors) {
        is_allocated &= tensor->is_allocated();
    }
    return is_allocated;
}

void TupleRemoteTensorImpl::allocate() {
    for (auto& tensor : m_remote_tensors) {
        tensor->allocate();
    }
}

const std::string& TupleRemoteTensorImpl::get_device_name() const {
    return m_context->get_device_name();
}

void TupleRemoteTensorImpl::set_memory(cldnn::memory::ptr memory, size_t actual_size) {
    for (auto& tensor : m_remote_tensors) {
        tensor->set_memory(memory, actual_size);
    }
}

std::shared_ptr<TupleRemoteContextImpl> TupleRemoteTensorImpl::get_context() const {
    return m_context;
}

}  // namespace intel_gpu
}  // namespace ov
