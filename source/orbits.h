#pragma once

#include <ostream>
#include <tuple>
#include <vector>

#include "clusters.h"
#include "groups.h" 
#include "dense.h"
#include "numbers.h"
#include "sparse.h"
#include "symmetry.h"


namespace Orbits {
    class EigenSymmetry {
        public:
        EigenSymmetry() = default;
        EigenSymmetry(size_t i, const Dense::Matrix<int> & r, const Dense::Matrix<int> & inv, const std::vector<size_t> & per):
            index(i), permutation(per), rotation(r), inverse_rotation(inv) {}
        EigenSymmetry(size_t i, const Symmetry::SymmetryOperation & operation, const std::vector<size_t> & p): 
            EigenSymmetry(i, operation.rotation_cartesian, operation.inverse_cartesian, p) {}

        auto operator == (const EigenSymmetry & other) const -> bool { return index == other.index && permutation == other.permutation; }

        auto inverse() const -> EigenSymmetry;

        auto is_identity() const -> bool;

        size_t index {0};
        size_t primitive_index {0};

        std::vector<size_t> permutation;

        Dense::Matrix<int> rotation;
        Dense::Matrix<int> inverse_rotation;
    };

    auto operator << (std::ostream & os, const EigenSymmetry & symmetry) -> std::ostream &;

    // One Fiber of clusters inside an orbit that are equivalent by primitive translations.
    // The representing cluster should be in the first copy of the primitive cell inside a super cell.
    class Fiber {
        public:
        Fiber(const Clusters::Cluster & clus, const EigenSymmetry & operation): cluster(clus), symmetry(operation) {}

        auto operator == (const Fiber & other) const noexcept -> bool = default;

        auto rotation() const -> EigenSymmetry;
        auto inverse_rotation() const -> EigenSymmetry;

        // The cluster and symmetry operation representing this fiber.
        Clusters::Cluster cluster;
        EigenSymmetry symmetry;

        // The eigentensors in the correct orientation for this fiber.
        sparse::Matrix<Numbers::Rational> tensors;
    };

    auto operator << (std::ostream & os, const Fiber & symmetry) -> std::ostream &;

    class Orbit {
        public:
        Orbit(const Clusters::Cluster & clus, const Groups::SpaceGroup & space_group);

        // std::vector interface for the fibers.
        auto operator [] (size_t index) -> Fiber & { return fibers[index]; }
        auto operator [] (size_t index) const -> const Fiber & { return fibers[index]; }

        auto operator == (const Orbit & other) const noexcept -> bool = default;

        auto begin() noexcept { return fibers.begin(); }
        auto begin() const noexcept { return fibers.begin(); }
        auto cbegin() const noexcept { return fibers.cbegin(); }

        auto end() noexcept { return fibers.end(); }
        auto end() const noexcept { return fibers.end(); }
        auto cend() const noexcept { return fibers.cend(); } 

        auto size() const noexcept -> size_t { return fibers.size(); }

        auto identify_cluster(const Clusters::Cluster & clus) const -> std::optional<EigenSymmetry>;

        auto add_tensor(size_t index, const sparse::Vector<Numbers::Rational> & tensor) -> void;
        auto rotate_tensors() -> void;

        // The cluster representing this orbit.
        Clusters::Cluster cluster;

        // The two independent collections of symmetry operations of this cluster:
        //
        //     isotropies: symmetry operations that map the representing cluster onto itself.
        //     fibers: The translation symmetry equivalence classes in this orbit.
        //
        // By Lagranges theorem isotropies.size() * fibers.size() = number of symmorphic transformations.
        std::vector<EigenSymmetry> isotropies;
        std::vector<Fiber> fibers;

        // The independent force constant tensors and their indices within this order.
        sparse::Matrix<Numbers::Rational> tensors;
        sparse::Matrix<Numbers::Rational> tensors_old;

        size_t row_offset;
        sparse::Matrix<Numbers::Rational> c_matrix;

        std::vector<size_t> tensor_indices;
    };

    auto operator << (std::ostream & os, const Orbit & symmetry) -> std::ostream &;

    auto identity(size_t order) -> EigenSymmetry;
    auto pure_rotation(const EigenSymmetry & other) -> EigenSymmetry;
}