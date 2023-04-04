// Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../api/cpp/Bolt.h"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(bolt, m)
{
    m.doc() = "bolt python class";

    m.attr("__version__") = "1.5.0";

    py::class_<Bolt>(m, "Bolt")
        .def(py::init<>())
        .def("convert", &Bolt::convert, py::arg("modelDirectory"), py::arg("modelName"),
            py::arg("precision"))
        .def("set_num_threads", &Bolt::set_num_threads, py::arg("threads"))
        .def("load", &Bolt::load, py::arg("boltModelPath"), py::arg("affinity"))
        .def("get_input_info", &Bolt::get_input_info)
        .def("get_output_info", &Bolt::get_output_info)
        .def("infer", &Bolt::infer);
}
