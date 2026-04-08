#pragma once

#include <string>
#include <vector>

#include "structure.h"
#include "dense.h"

namespace Symmetry {
    class SymmetryOperation {
        public:
        bool is_identity;
        bool is_rotation_identity;

        Dense::Vector<double> translation;
        Dense::Matrix<int> rotation;
        Dense::Matrix<int> rotation_cartesian;
        Dense::Matrix<int> inverse_cartesian;

        SymmetryOperation() = default;
        SymmetryOperation(std::vector<double> &v, std::vector<std::vector<int>> &m);

        auto operator == (const SymmetryOperation & other) const noexcept -> bool = default;

        const std::string as_string() const;
        const Dense::Vector<double> apply (const Dense::Vector<double> &v) const;
        auto rotate(Dense::Vector<int> const & vector) const -> Dense::Vector<int>;
        const Dense::Vector<double> translate (const Dense::Vector<double> &v) const;
    };

    auto operator << (std::ostream & os, const SymmetryOperation & symmetry) -> std::ostream &;

    class SymmetryData {
        public:
        int number;
        std::string symbol;

        structures::Structure primitive;

        std::vector<SymmetryOperation> primitive_operations;
        std::vector<Dense::Vector<double>> translations;
        std::vector<SymmetryOperation> operations;

        std::vector<Dense::Matrix<int>> rotation_matrices;

        std::vector<std::vector<std::vector<size_t>>> permutations;

        std::vector<size_t> map_to_primitive;
        std::vector<size_t> primitive_indices;
        std::vector<size_t> sublattice_indices;
        std::vector<size_t> primitive_to_sublattice;

        //std::vector<std::vector<int>> index_to_ijkl;

        SymmetryData() = default;
        SymmetryData(structures::Structure & structure);

        void find_symmetry_operations(std::vector<std::vector<double>> &translationsTemp, std::vector<std::vector<std::vector<int>>> &rotationsTemp, size_t number);
    };
}