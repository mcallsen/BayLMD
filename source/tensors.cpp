#include "tensors.h"

#include <vector>

#include "extensions.h"
#include "linalg.h"
#include "math.h"
#include "dense.h"
#include "orbits.h"
#include "numbers.h"
#include "sparse.h"

using Numbers::Rational;

namespace Tensor {
    auto index_constraint(const sparse::Matrix<Rational> & tensors) -> sparse::Matrix<Rational> {
        //At this point tensors basically is a list of equivalent tensor components
        size_t dimension = tensors[0].dimension;
        sparse::Matrix<Rational> constraints(0, dimension);
        for (const auto & tensor: tensors){
            if (tensor.size() == 1) {
                // This is not a repeated index.
                continue;
            }

            size_t origin = tensor.indices[0];
            for (size_t index = 1; index < tensor.size(); index++) {
                // we found a new constraint
                sparse::Vector<Rational> vector(dimension);
                vector.insert(origin, extensions::as<Rational>(1));
                vector.insert(tensor.indices[index], extensions::as<Rational>(-1));

                constraints.push_back(vector);
            }
        }

        return constraints;
    }

    // Generate a basis for an order-Tensor with certain symmetric sub-tensors according to multiplicities.
    auto get_tensor_basis(size_t order, const std::vector<size_t> & multiplicities) -> sparse::Matrix<Rational> {
        size_t dimension = Math::power(extensions::as<size_t>(3), order);

        //Matrix::SparseMatrix<Math::RationalNumber> tensorBasis;
        sparse::Matrix<Rational> tensor_basis(0, dimension);
        std::map<size_t, size_t> basis_indices;

        // Get all possible combinations of elements with length order, which are all subscripts for this tensor.
        auto subscripts = generate_subscripts(order);
        for (const auto & subscript: subscripts) {

            auto symmetrised = symmetrize_subscript(subscript, multiplicities);
            size_t reduced_index = subscript_to_index_asc(symmetrised);

            if (!extensions::contains(basis_indices, reduced_index)) {
                // Found a new independent basis tensor.
                basis_indices[reduced_index] = tensor_basis.size();
                tensor_basis.push_back(sparse::Vector<Rational>(dimension));
            }
            
            size_t index = subscript_to_index_asc(subscript);
            size_t tensor_index = basis_indices[reduced_index];

            // Assumption: if the above works as intended, the multi-Indices should be sorted.
            tensor_basis[tensor_index].insert(index, extensions::as<Rational>(1));
        }

        return tensor_basis;
    }

    // Create the constraint matrix for the space-group symmetry constraint according eqn. (15.1) in Zhou et al, 
    // PRB 100, 184308 (2019). i.e. this is [Gamma(s) - 1] * phi for a given s.
    auto spacegroup_constraint(size_t dimension, const Orbits::EigenSymmetry & symmetry) -> sparse::Matrix<Rational> {
        sparse::Matrix<Rational> constraint(dimension, dimension);

        auto gamma = transformation_matrix(dimension, symmetry.rotation, symmetry.permutation);
        auto identity = sparse::identity<Rational>(dimension);

        for (size_t i = 0; i < gamma.size(); i++) {
            constraint[i] = gamma[i] - identity[i];
        }

        return constraint;
    }

// Conversions between tensor subscript and flattend index notation. CSLD uses the asc() operator
// While HiPhive uses vec(). The Kronecker-product of rotation matrices will be different, the 
// conversion has to be consistent everywhere. 

    auto subscript_to_index_asc(const std::vector<size_t> & subscript, size_t dimension) -> size_t {
        size_t index = 0;
        size_t factor = Math::power(dimension, subscript.size() - 1);
        for (auto i: subscript) {
            index += factor * i;
            factor /= dimension;
        }
        return index;
    }

    auto subscript_to_index_vec(const std::vector<size_t> & subscript, size_t dimension) -> size_t {
        size_t index = 0;
        size_t factor = 1;
        for (auto i: subscript) {
            index += factor * i;
            factor *= dimension;
        }
        return index; 
    }

    auto index_to_subscript_asc(size_t index, size_t order, size_t dimension) -> std::vector<size_t> {
        std::vector<size_t> subscript(order);
        while (order > 0) {
            order--;
            subscript[order] = index % dimension;
            index /= dimension;
        }
        return subscript;
    }

    auto index_to_subscript_vec(size_t index, size_t order, size_t dimension) -> std::vector<size_t> {
        std::vector<size_t> subscript;
        while (order > 0) {
            order--;
            subscript.push_back(index % dimension);
            index /= dimension;
        }
        return subscript;
    }

    // Generate a range of subscripts.
    auto generate_subscripts(size_t order) -> std::vector<std::vector<size_t>> {
        std::vector<size_t> elements {0, 1, 2};
        std::vector<std::vector<size_t>> combinations;
        std::vector<size_t> current;
        extensions::product(current, combinations, elements, order);
        return combinations;
    }

    auto symmetrize_subscript(const std::vector<size_t> & subscript, const std::vector<size_t> & multiplicities) -> std::vector<size_t> {
        // If subscripts are equivalent due to repeated atom indices, return the same subscript. This is done by sorting
        // each slice of the subscript with length equal to the multiplicity of that atoms index. For exampla a third order
        // cluster with two repeated indices j and k : [i, k, j] -> [i, j, k].
        std::vector<size_t> result;

        size_t current = 0;
        for (auto multiplicity: multiplicities) {
            // Get the slice of the subscript that corresponds to repeated indices.
            std::vector<size_t> slice; 
            for (size_t i = current; i < current + multiplicity; i++) {
                slice.push_back(subscript[i]);
            }

            // Sort the slice. This way in every equivalent subscript these indices will be in ascending order.
            std::sort(slice.begin(), slice.end());

            result.insert(result.end(), slice.begin(), slice.end());

            current += multiplicity;
        }
        return result;
    }

    // Construct the matrix that transforms a tensor according to a given eigensymmetry. specifically this is the
    // Kronecker-product [R x ... x R] of order n. Technically it should not matter whether we are using vec or asc.
    // The only difference would be the order of the matrices.
    auto transformation_matrix(size_t size, const Dense::Matrix<int> & rotation, const std::vector<size_t> &permutation) -> sparse::Matrix<Rational> {
        size_t order = permutation.size();
        sparse::Matrix<Rational> result(size, size);

        for (size_t i = 0; i < size; i++) {
            auto outer = index_to_subscript_asc(i, order);
            for (size_t j = 0; j < size; j++) {
                auto inner = index_to_subscript_asc(j, order);
                int tmp = 1;
                for (size_t k = 0; k < order; k++) {
                    tmp *= rotation[outer[k]][inner[permutation[k]]];
                }

                if (tmp == 0) continue;
                result.insert_quick(i, j, extensions::as<Rational>(tmp)); 
            }
        }

        return result;
    }

    // Transform a matrix of tensors according to a given EigenSymmetry, which can involve transposing and rotating the tensors.
    auto tensor_transform(const sparse::Matrix<Rational> & matrix, const Orbits::EigenSymmetry & symmetry) -> sparse::Matrix<Rational> {
        sparse::Matrix<Rational> result(matrix);
        auto transformation = transformation_matrix(matrix.columns, symmetry.rotation, symmetry.permutation);
        for (size_t i = 0; i < matrix.size(); i++) {
            result[i] = sparse::times_vector(transformation, result[i]);
        }
        return result;
    }

}
