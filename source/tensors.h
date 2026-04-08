#pragma once

#include <vector>

#include "dense.h"
#include "orbits.h"
#include "numbers.h"
#include "sparse.h"

namespace Tensor {
    // Get the Constraint matrix for the index constraint.
    auto index_constraint(const sparse::Matrix<Numbers::Rational> & tensors) -> sparse::Matrix<Numbers::Rational>;

    // Get the Constraint matrix for a spacegroup constraint.
    auto spacegroup_constraint(size_t dimension, const Orbits::EigenSymmetry & symmetry) -> sparse::Matrix<Numbers::Rational>;

    // Get a list of equivalent tensor indices.
    auto get_tensor_basis(size_t order, const std::vector<size_t> &multiplicities) -> sparse::Matrix<Numbers::Rational>;

    // Convert a tensor subscript to an index in the flattened array according to asc() (row major).
    auto subscript_to_index_asc(const std::vector<size_t> & subscript, size_t dimension = 3) -> size_t;

    // Convert a tensor subscript to an index in the flattened array according to vec() (column major).
    auto subscript_to_index_vec(const std::vector<size_t> & subscript, size_t dimension = 3) -> size_t;

    // Convert the index in the flattened array to a tensor subscript according to inverse asc() (row major).
    auto index_to_subscript_asc(size_t index, size_t order, size_t dimension = 3) -> std::vector<size_t>;

    // Convert the index in the flattened array to a tensor subscript according to inverse vec() (column major).
    auto index_to_subscript_vec(size_t index, size_t order, size_t dimension = 3) -> std::vector<size_t>;

    auto generate_subscripts(size_t order) -> std::vector<std::vector<size_t>>;

    auto symmetrize_subscript(const std::vector<size_t> &subscript, const std::vector<size_t> &multiplicities) -> std::vector<size_t>;

    auto transformation_matrix(size_t size, const Dense::Matrix<int> &rotation, const std::vector<size_t> &permutation) -> sparse::Matrix<Numbers::Rational>;
    auto tensor_transform(const sparse::Matrix<Numbers::Rational> &matrix, const Orbits::EigenSymmetry &eigenSymmetry) -> sparse::Matrix<Numbers::Rational>;
}
