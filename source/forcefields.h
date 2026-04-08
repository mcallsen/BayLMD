#pragma once

#include <vector>
#include "mpi.h"

namespace Forcefields {
    class Forcefield {
        public:

        Forcefield() = default;
        Forcefield(const std::string & file_name, bool parallel, size_t group);

        auto get_forces(const std::vector<double> & displacements) -> std::vector<double>;
        auto get_fit_matrix(const std::vector<double> & displacements) -> std::vector<double>;

        auto set_parameter_mask(const std::vector<size_t> & indices) -> void;

        // Write the ForceModel to an output file stream.
        auto write(const std::string & file_name) const -> void; 

        sparse::Matrix<double> tensors;
        std::vector<Orbit> orbits;

        // Vector holding the flat version of all rotation matrices of all rotation matrices.
        std::vector<int> rotations;

        // row and column indices for each non zero entry in matrix. Required for
        // storing the FitMatrix as scipy sparse matrix.
        std::vector<size_t> rows;
        std::vector<size_t> columns;

        std::vector<double> parameters;

        size_t n_symmetries {0};
        size_t n_components {0};
        size_t n_tensors {0};
        size_t n_parameters {0};

        private:
        auto calculate_phi(const std::vector<double> & displacements)  -> void;
	    auto calculate_a_matrix(const std::vector<double> & displacements) -> void;
        auto rotate_displacements(const std::vector<double> & displacements) -> void;

        mpi::communicator _communicator;

        mpi::Parallelisation _parallel_phi;
        mpi::Parallelisation _parallel_forces;

        // Temporary storage to avoid allocating large arrays at every MD step.
	    std::vector<double> forces;
	    std::vector<double> phi_matrix;
	    std::vector<double> a_matrix;
	    std::vector<double> rotated_displacements;

        std::vector<bool> _is_zero_at;
        std::vector<bool> _is_zero_at_cache;

        size_t my_start = {0};
        size_t my_end = {0};
	    size_t my_rows = {0};
        size_t my_group = {0};
    };
}
