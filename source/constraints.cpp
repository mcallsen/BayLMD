#include "constraints.h"

#include "linalg.h"
#include "math.h"
#include "orbits.h"
#include "numbers.h"
#include "sparse.h"
#include "tensors.h"

using Numbers::Rational;

namespace Constraints {

// Sumrule.
    auto Sumrule::add_cluster(const Orbits::EigenSymmetry & symmetry, size_t orbit_index) -> void {
        symmetries.push_back(symmetry);
        orbit_indices.push_back(orbit_index);
    }

    auto Sumrule::constraint(size_t dimension, size_t number) const -> sparse::Matrix<Numbers::Rational> {
        sparse::Matrix<Numbers::Rational> matrix(dimension, dimension * number);
        for (size_t i = 0; i < size(); i++) {
            auto & operation = symmetries[i];
            auto transform = Tensor::transformation_matrix(dimension, operation.rotation, operation.permutation);

            // ATTENTION: The interface has changed! not sure this is correct.:q
            sparse::add_with_offset(matrix, transform, sparse::Offset { 0, orbit_indices[i] * dimension });
        }
        return matrix;
    }

    auto operator << (std::ostream & os, const Sumrule & sumrule) -> std::ostream & {
        os << sumrule.cluster.size() + 1 << "   ";
        for (const auto & site: sumrule.cluster.sites) {
            os << site << "";
        }
        os << "    " << sumrule.size();
        return os;
    }

// Constraint.

    void Constraint::iterate(sparse::Matrix<Numbers::Rational> &constraint) {
        //std::cout << "B Matrix: " << constraint.rows << " " << constraint.columns << std::endl;
        //std::cout << constraint << std::endl;
        constraint = sparse::matrix_product<Rational>(constraint, matrix);
        //std::cout << "BxC Matrix: " << constraint.rows << " " << constraint.columns << std::endl;
        //std::cout << constraint.ToString() << std::endl;


        // if BxC is zero, we do not have any constraints left, then C' will just be the identity.
        if (sparse::is_empty(constraint)) return;

        // Find the Nullspace of the constraint matrix.
        constraint = Math::compute_nullspace(constraint);

        //std::cout << "Nullspace: " << constraint.rows << " " << constraint.columns << std::endl;
        //std::cout << constraint << std::endl;

        // There are no remaining independent eigentensors.
        if (sparse::is_empty(constraint)) {
            is_empty = true;
            return;
        }

        matrix = sparse::matrix_product<Rational>(matrix, sparse::transpose(constraint));   
    }

    IndexConstraint::IndexConstraint(Orbits::Orbit & orb): orbit(orb) {
        matrix = sparse::transpose(orbit.tensors);
    }

    SpaceGroupConstraint::SpaceGroupConstraint(Orbits::Orbit &orb): orbit(orb) {
        matrix = sparse::transpose(orbit.tensors);
    }

    TranslationalConstraint::TranslationalConstraint(size_t number, std::vector<Orbits::Orbit> &orbs): orbits(orbs) {
        if (orbs.size() > 0)
            dimension = Math::power((size_t) 3, orbits[0].cluster.size());

        matrix = sparse::Matrix<Rational>(number, dimension * orbits.size());

        size_t count = 0;
        for (size_t index = 0; index < orbits.size(); index++) {
            auto & orbit = orbits[index];

            if (orbit.tensors.size() < 1) continue;

            sparse::add_with_offset(matrix, orbit.tensors, sparse::Offset { count, index * dimension });
            count += orbit.tensors.size();
            orbit.tensors_old = orbit.tensors;
            orbit.tensors = sparse::Matrix<Rational>(0, dimension);
            orbit.tensor_indices = std::vector<size_t>();
        }
        matrix = sparse::transpose(matrix);
    }

    auto apply_index_constraint(Orbits::Orbit & orbit) -> void {
        IndexConstraint constraint(orbit);

        //std::cout << "Index constraint: " << orbit.cluster << std::endl;

        auto & cluster = orbit.cluster;

        if (cluster.size() == 3) {
            //std::cout << "Index constraint: " << cluster << std::endl;
        }

        if (!cluster.is_proper()) {
            // we are dealing with an improper cluster, so we first have to apply
            // the symmetry constraints arising from the exchange of the repeated indices.
            auto tensors = Tensor::get_tensor_basis(cluster.size(), cluster.multiplicities);
            auto matrix = Tensor::index_constraint(tensors);

            //if (cluster.size() == 3) {
            //    std::cout << "Constraint: " << matrix.rows << " " << matrix.columns << std::endl;
            //    std::cout << matrix << std::endl;
            //}

            constraint.iterate(matrix);
            
        }
        orbit.tensors = sparse::transpose(constraint.matrix);
        //std::cout << "Index constraint (final): " << orbit.tensors.rows << "  " << orbit.tensors.columns << std::endl;
        //std::cout << orbit.tensors << std::endl; 
    }

    auto apply_spacegroup_constraint(Orbits::Orbit & orbit) -> void {
        SpaceGroupConstraint constraint(orbit);
        size_t dimension = constraint.matrix.size();

        //if (orbit.cluster.size() == 3) {
        //    std::cout << "Spacegroup constraint: " << orbit.cluster << std::endl;
        //}

        for (const auto & isotropy: orbit.isotropies) {
            // Apply the constraint corresponding to each spacegroup operation
            // in the isotropy group of this orbit.
            auto matrix = Tensor::spacegroup_constraint(dimension, isotropy);
            if (sparse::is_empty(matrix)) continue;

            //if (orbit.cluster.size() == 3) {
            //    std::cout << isotropy << std::endl;
            //    std::cout << matrix << std::endl;
            //}

            constraint.iterate(matrix);
            if (constraint.is_empty) break;
        }

        if (constraint.is_empty) {
            // There are no independent eigen tensors remaining.
            orbit.tensors = sparse::Matrix<Rational>(0, dimension);
            return;
        }

        // Assign the tensors back to the orbits.
        orbit.tensors = sparse::transpose(constraint.matrix);
        //std::cout << "Spacegroup constraint (final): " << orbit.tensors.rows << "  " << orbit.tensors.columns << std::endl;
        //std::cout << orbit.tensors << std::endl; 
    }

    auto apply_translational_constraint(size_t number, std::vector<Orbits::Orbit> & orbits, const std::vector<Sumrule> & collection) -> size_t {
        TranslationalConstraint constraint(number, orbits);
        std::cout << "Translational constraint (" << orbits[0].cluster.size() << "): " << constraint.matrix.columns << " initial components." << std::endl;

        size_t count = 0;
        for (const auto & sum_rule: collection) {
            size_t previous = constraint.matrix.columns;
            //std::cout << "C Matrix: " << constraint.matrix.rows << " " << constraint.matrix.columns << std::endl;
            //std::cout << constraint.matrix.ToString() << std::endl;
            auto matrix = sum_rule.constraint(constraint.dimension, orbits.size());
            //std::cout << "B Matrix: " << constraintMatrix.rows << " " << constraintMatrix.columns << std::endl;
            //std::cout << constraintMatrix.ToString() << std::endl;
            constraint.iterate(matrix);

            std::cout << "    Sumrule " << count << ": Reduced " << previous << " to " << constraint.matrix.columns << " components." << std::endl;
            count++; 
        }

        // Set the block of the C Matrix for each orbit.
        for (size_t i = 0; i < orbits.size(); i++) {
            orbits[i].c_matrix = sparse::Matrix<Numbers::Rational>(0, constraint.matrix.columns);
            for (size_t j = 0; j < constraint.dimension; j++) {
                orbits[i].c_matrix.push_back(constraint.matrix[i * constraint.dimension + j]);
            }
        }

        // redistribute the tensors to the orbits.
        auto matrix = sparse::transpose(constraint.matrix);
        for (size_t i = 0; i < matrix.size(); i++) {
            auto tensors = sparse::split(matrix[i], orbits.size());
            for (size_t j = 0; j < tensors.size(); j++) {
                orbits[j].add_tensor(i, tensors[j]);
            }
        }
        std::cout << std::endl;

        return matrix.rows;
    }
}