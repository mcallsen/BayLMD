#pragma once

#include <string>
#include <vector>

namespace Settings {
    struct Settings {
        std::string structure_file { "POSCAR.vasp" };
        std::string supercell_file { "SPOSCAR.vasp" };
        std::string forcefield_file { "ml_forcefield.dat" };
        std::vector<double> cutoffs { 5.70, 3.53, 3.53, 3.53, 3.53 };
    };
}