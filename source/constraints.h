#pragma once

#include <ostream>
#include <vector>

#include "clusters.h"
#include "orbits.h"
#include "numbers.h"
#include "sparse.h"

namespace Constraints {
    class Sumrule {
        public:
        std::vector<Orbits::EigenSymmetry> symmetries;
        std::vector<size_t> orbit_indices;

        Clusters::Cluster cluster;

        Sumrule() = default;
        explicit Sumrule(const Clusters::Cluster & clus) noexcept : cluster(clus) {};

        auto operator == (const Sumrule & other) const -> bool = default;

        auto size() const -> size_t { return symmetries.size(); }

        auto add_cluster(const Orbits::EigenSymmetry & symmetry, size_t orbit_index) -> void;
        auto constraint(size_t dimension, size_t number) const -> sparse::Matrix<Numbers::Rational>;
    };

    auto operator << (std::ostream & os, const Sumrule & sumrule) -> std::ostream &;

    class Constraint {
        public:
        sparse::Matrix<Numbers::Rational> matrix;
        bool is_empty = false;

        auto iterate(sparse::Matrix<Numbers::Rational> & constraint) -> void;
    };

    class IndexConstraint: public Constraint {
        public:
        Orbits::Orbit & orbit;
        explicit IndexConstraint(Orbits::Orbit & orbit);
    };

    class SpaceGroupConstraint: public Constraint {
        public:
        Orbits::Orbit & orbit;
        explicit SpaceGroupConstraint(Orbits::Orbit & orbit);
    };

    class TranslationalConstraint: public Constraint {
        public:
        std::vector<Orbits::Orbit> & orbits;
        size_t dimension {0};

        TranslationalConstraint(size_t number, std::vector<Orbits::Orbit> & orbits);
    };

    auto apply_index_constraint(Orbits::Orbit & orbit) -> void;
    auto apply_spacegroup_constraint(Orbits::Orbit & orbit) -> void;
    auto apply_translational_constraint(size_t number, std::vector<Orbits::Orbit> & orbits, const std::vector<Sumrule> & collection) -> size_t;
}