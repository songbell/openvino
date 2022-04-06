// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pyopenvino/core/tensor.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "openvino/runtime/tensor.hpp"
#include "pyopenvino/core/common.hpp"

namespace py = pybind11;

void regclass_Tensor(py::module m) {
    py::class_<ov::Tensor, std::shared_ptr<ov::Tensor>> cls(m, "Tensor");
    cls.doc() = "openvino.runtime.Tensor holding either copy of memory or shared host memory.";

    cls.def(py::init([](py::array& array, bool shared_memory) {
                return Common::tensor_from_numpy(array, shared_memory);
            }),
            py::arg("array"),
            py::arg("shared_memory") = false,
            R"(
                Tensor's special constructor.

                :param array: Array to create tensor from.
                :type array: numpy.array
                :param shared_memory: If `True`, this Tensor memory is being shared with a host,
                                      that means the responsibility of keeping host memory is
                                      on the side of a user. Any action performed on the host
                                      memory is reflected on this Tensor's memory!
                                      If `False`, data is being copied to this Tensor.
                                      Requires data to be C_CONTIGUOUS if `True`.
                :type shared_memory: bool
            )");

    cls.def(py::init([](py::array& array, const ov::Shape& shape, const ov::element::Type& ov_type) {
                return Common::tensor_from_pointer(array, shape, ov_type);
            }),
            py::arg("array"),
            py::arg("shape"),
            py::arg("type") = ov::element::undefined,
            R"(
                Another Tensor's special constructor.

                Represents array in the memory with given shape and element type.
                It's recommended to use this constructor only for wrapping array's
                memory with the specific openvino element type parameter.

                :param array: C_CONTIGUOUS numpy array which will be wrapped in
                              openvino.runtime.Tensor with given parameters (shape
                              and element_type). Array's memory is being shared with
                              a host, that means the responsibility of keeping host memory is
                              on the side of a user. Any action performed on the host
                              memory will be reflected on this Tensor's memory!
                :type array: numpy.array
                :param shape: Shape of the new tensor.
                :type shape: openvino.runtime.Shape
                :param type: Element type
                :type type: openvino.runtime.Type

                :Example:
                .. code-block:: python

                    import openvino.runtime as ov
                    import numpy as np

                    arr = np.array(shape=(100), dtype=np.uint8)
                    t = ov.Tensor(arr, ov.Shape([100, 8]), ov.Type.u1)
            )");

    cls.def(py::init([](py::array& array, const std::vector<size_t> shape, const ov::element::Type& ov_type) {
                return Common::tensor_from_pointer(array, shape, ov_type);
            }),
            py::arg("array"),
            py::arg("shape"),
            py::arg("type") = ov::element::undefined,
            R"(
                 Another Tensor's special constructor.

                Represents array in the memory with given shape and element type.
                It's recommended to use this constructor only for wrapping array's
                memory with the specific openvino element type parameter.

                :param array: C_CONTIGUOUS numpy array which will be wrapped in
                              openvino.runtime.Tensor with given parameters (shape
                              and element_type). Array's memory is being shared with
                              a host, that means the responsibility of keeping host memory is
                              on the side of a user. Any action performed on the host
                              memory will be reflected on this Tensor's memory!
                :type array: numpy.array
                :param shape: Shape of the new tensor.
                :type shape: list or tuple
                :param type: Element type.
                :type type: openvino.runtime.Type

                :Example:
                .. code-block:: python

                    import openvino.runtime as ov
                    import numpy as np

                    arr = np.array(shape=(100), dtype=np.uint8)
                    t = ov.Tensor(arr, [100, 8], ov.Type.u1)
            )");

    cls.def(py::init<const ov::element::Type, const ov::Shape>(), py::arg("type"), py::arg("shape"));

    cls.def(py::init<const ov::element::Type, const std::vector<size_t>>(), py::arg("type"), py::arg("shape"));

    cls.def(py::init([](py::dtype& np_dtype, std::vector<size_t>& shape) {
                return ov::Tensor(Common::dtype_to_ov_type().at(py::str(np_dtype)), shape);
            }),
            py::arg("type"),
            py::arg("shape"));

    cls.def(py::init([](py::object& np_literal, std::vector<size_t>& shape) {
                return ov::Tensor(Common::dtype_to_ov_type().at(py::str(py::dtype::from_args(np_literal))), shape);
            }),
            py::arg("type"),
            py::arg("shape"));

    cls.def(py::init([](py::dtype& np_dtype, const ov::Shape& shape) {
                return ov::Tensor(Common::dtype_to_ov_type().at(py::str(np_dtype)), shape);
            }),
            py::arg("type"),
            py::arg("shape"));

    cls.def(py::init([](py::object& np_literal, const ov::Shape& shape) {
                return ov::Tensor(Common::dtype_to_ov_type().at(py::str(py::dtype::from_args(np_literal))), shape);
            }),
            py::arg("type"),
            py::arg("shape"));

    cls.def(py::init<ov::Tensor, ov::Coordinate, ov::Coordinate>(), py::arg("other"), py::arg("begin"), py::arg("end"));

    cls.def(py::init<ov::Tensor, std::vector<size_t>, std::vector<size_t>>(),
            py::arg("other"),
            py::arg("begin"),
            py::arg("end"));

    cls.def("get_element_type",
            &ov::Tensor::get_element_type,
            R"(
            Gets Tensor's element type.

            :rtype: openvino.runtime.Type
            )");

    cls.def_property_readonly("element_type",
                              &ov::Tensor::get_element_type,
                              R"(
                                Tensor's element type.

                                :rtype: openvino.runtime.Type
                              )");

    cls.def("get_size",
            &ov::Tensor::get_size,
            R"(
            Gets Tensor's size as total number of elements.

            :rtype: int
            )");

    cls.def_property_readonly("size",
                              &ov::Tensor::get_size,
                              R"(
                                Tensor's size as total number of elements.

                                :rtype: int
                              )");

    cls.def("get_byte_size",
            &ov::Tensor::get_byte_size,
            R"(
            Gets Tensor's size in bytes.

            :rtype: int
            )");

    cls.def_property_readonly("byte_size",
                              &ov::Tensor::get_byte_size,
                              R"(
                                Tensor's size in bytes.

                                :rtype: int
                              )");

    cls.def("get_strides",
            &ov::Tensor::get_strides,
            R"(
            Gets Tensor's strides in bytes.

            :rtype: openvino.runtime.Strides
            )");

    cls.def_property_readonly("strides",
                              &ov::Tensor::get_strides,
                              R"(
                                Tensor's strides in bytes.

                                :rtype: openvino.runtime.Strides
                              )");

    cls.def_property_readonly(
        "data",
        [](ov::Tensor& self) {
            auto ov_type = self.get_element_type();
            auto dtype = Common::ov_type_to_dtype().at(ov_type);
            if (ov_type.bitwidth() < 8) {
                return py::array(dtype, self.get_byte_size(), self.data(), py::cast(self));
            }
            return py::array(dtype, self.get_shape(), self.get_strides(), self.data(), py::cast(self));
        },
        R"(
            Access to Tensor's data.

            Returns numpy array with corresponding shape and dtype.
            For tensors with openvino specific element type, such as u1, u4 or i4
            it returns linear array, with uint8 / int8 numpy dtype.

            :rtype: numpy.array
        )");

    cls.def("get_shape",
            &ov::Tensor::get_shape,
            R"(
            Gets Tensor's shape.

            :rtype: openvino.runtime.Shape
            )");

    cls.def("set_shape",
            &ov::Tensor::set_shape,
            R"(
            Sets Tensor's shape.
            )");

    cls.def(
        "set_shape",
        [](ov::Tensor& self, std::vector<size_t>& shape) {
            self.set_shape(shape);
        },
        R"(
            Sets Tensor's shape.
        )");

    cls.def_property("shape",
                     &ov::Tensor::get_shape,
                     &ov::Tensor::set_shape,
                     R"(
                        Tensor's shape get/set.
                     )");

    cls.def_property(
        "shape",
        &ov::Tensor::get_shape,
        [](ov::Tensor& self, std::vector<size_t>& shape) {
            self.set_shape(shape);
        },
        R"(
            Tensor's shape get/set.
        )");

    cls.def("__repr__", [](const ov::Tensor& self) {
        std::stringstream ss;

        ss << "shape" << self.get_shape() << " type: " << self.get_element_type();

        return "<Tensor: " + ss.str() + ">";
    });
}
