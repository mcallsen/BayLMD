#include <iostream>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/iostream.h>
#include <pybind11/stl.h>

#include "forcefields.h"

class ForcefieldMock {
    public:
    std::vector<double> parameters;

    ForcefieldMock(const std::string & file_name, bool parallel) {
        std::cout << "File name: " << file_name << ", parallel: " << parallel << std::endl;
    }

    auto get_forces(const std::vector<double> & displacements) -> std::vector<double> {
        return get_phi(displacements);
    }

    auto get_phi(const std::vector<double> & displacements) -> std::vector<double> {
        std::vector<double> result { displacements };
        for (size_t i = 0; i < result.size(); i++) {
            result[i] += 1.0;
        }
        return result;
    }
};


// NOTE: here are a couple of additional call policies for the methods/properties:
//
//     (1) py::keep_alive<2, 0>(), (2) py::keep_alive<0, 1>()
//
//         Tells Pybind11 to keep the return value alive while the argument exists (1) or the temporary object while the return value exists (2)
//         Which apparently might be necessary if the returnvalue is used in a long chain.
//
//     py::call_guard<py::gil_scoped_acquire, py::gil_scoped_release>()
//
//         If further down the line the C++ code looks at python state (which I think we are not doing) than the GIL has to be acquired and released.
//
//     py::return_value_policy::copy 
//
//         By default py::return_value_policy::move is used, which should be faster. However, if problems occur, we might have to change that.


PYBIND11_MODULE(forceModelPy, m) {
    pybind11::class_<Forcefields::Forcefield>(m, "ForceModel")
        .def(pybind11::init<const std::string &, bool, size_t>(), pybind11::call_guard<pybind11::scoped_ostream_redirect, pybind11::scoped_estream_redirect>())
        .def_readwrite("n_components", & Forcefields::Forcefield::n_components)
        .def_readwrite("n_parameters", & Forcefields::Forcefield::n_parameters)
        .def_readwrite("parameters", & Forcefields::Forcefield::parameters)
        .def("get_fit_matrix", & Forcefields::Forcefield::get_fit_matrix, "Get AC corresponding to the displacements.")
        //.def("get_correlation_matrix", & Forcefields::Forcefield::get_correlation_matrix, "Get the A matrix corresponding to the displacements.")
        .def("get_forces", & Forcefields::Forcefield::get_forces, "Get the Forces corresponding to the displacements.")
        .def("set_parameter_mask", & Forcefields::Forcefield::set_parameter_mask, "Set the parameters.")
	    .def("write", & Forcefields::Forcefield::write, "Write the forcemodel to a file.");;

    pybind11::class_<ForcefieldMock>(m, "ForcefieldMock")
        .def(pybind11::init<const std::string &, bool>(), pybind11::call_guard<pybind11::scoped_ostream_redirect, pybind11::scoped_estream_redirect>())
        .def_readwrite("parameters", & ForcefieldMock::parameters)
        .def("get_phi", & ForcefieldMock::get_phi, "Get the fit matrix corresponding to the displacements.")
        .def("get_forces", & ForcefieldMock::get_forces, "Get the Forces corresponding to the displacements.");;
}
