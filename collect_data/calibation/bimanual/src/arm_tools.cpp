#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "arx_r5_src/interfaces/InterfaceToolsPy.hpp"

namespace py = pybind11;

PYBIND11_MODULE(arx_r5_python_tools, m) {
    py::class_<arx::r5::InterfacesToolsPy>(m, "InterfacesToolsPy")
        .def(py::init<int>()) 
        .def("forward_kinematics_rpy", &arx::r5::InterfacesToolsPy::forward_kinematics_rpy);
}