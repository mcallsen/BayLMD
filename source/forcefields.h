#pragma once

#include <string>
#include <vector>

#include "extensions.h"
#include "math.h"
#include "mpi.h"
#include "sparse.h"

namespace Forcefields {
    class Cluster {
        public:
        Cluster(const std::vector<size_t> & vector, size_t index): indices(vector), rotation_index(index) {}

        auto operator <=> (const Cluster & other) const = default;
        auto operator == (const Cluster & other) const -> bool = default;

        std::vector<size_t> indices {};
        size_t rotation_index {0};
    };

    class Orbit {
        public:
        Orbit(size_t o, size_t i, size_t p): order(o), dimension(Math::power(extensions::as<size_t>(3), o)), tensor_index(i), prefactor(p) {}

        auto operator <=> (const Orbit & other) const = default;
        auto operator == (const Orbit & other) const -> bool = default;

        std::vector<Cluster> clusters {};

        size_t order {0};
        size_t dimension {0};
        size_t tensor_index {0};
        size_t prefactor {1};
    };

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
