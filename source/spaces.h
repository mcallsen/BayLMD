#pragma once

#include <set>
#include <vector>

#include "clusters.h"
#include "constraints.h"
#include "groups.h"
#include "orbits.h"
#include "settings.h"
#include "structure.h"
#include "symmetry.h"

namespace Spaces {

    // The subspace of a clusterspace for a given order n.
    class Subspace {
        public:
        double cutoff {0.0};

        std::vector<std::vector<structures::Site>> neighbour_sites {};

        size_t order {0};
        size_t first_tensor_index {0};

        // Clusters, Orbits and sumRules constituting the cluster expansion.
        // TODO: I am not sure whether clusters is just a temporary in builder.
        std::vector<Clusters::Cluster> clusters;
        std::vector<Orbits::Orbit> orbits;
        std::vector<Constraints::Sumrule> sum_rules;

        Subspace() = default;
        Subspace(size_t o, double c) :  cutoff(c), order(o) {}

        // std::vector interface for the Orbits.
        auto operator [] (size_t index) -> Orbits::Orbit & { return orbits[index]; }
        auto operator [] (size_t index) const -> const Orbits::Orbit & { return orbits[index]; }

        auto begin() noexcept { return orbits.begin(); }
        auto begin() const noexcept { return orbits.begin(); }
        auto cbegin() const noexcept { return orbits.cbegin(); }

        auto end() noexcept { return orbits.end(); }
        auto end() const noexcept { return orbits.end(); }
        auto cend() const noexcept { return orbits.cend(); } 

        auto size() const noexcept -> size_t { return orbits.size(); }

        auto print_orbits() const -> void;

        auto create_neighbour_tables(const structures::Structure & structure) -> void;
        auto create_sumrules(const Subspace & other, const structures::Structure & structure) -> void;

        auto get_neighbour_sites(const structures::Site & site) const -> const std::vector<structures::Site> &;
        auto get_neighbour_clusters(const Clusters::Cluster & cluster, const structures::Structure & structure) const -> std::vector<Clusters::Cluster>;
    };

    auto operator << (std::ostream & os, Subspace const & space) -> std::ostream &;
    
    class ClusterSpace {
        public:
        explicit ClusterSpace(const Settings::Settings & settings);

        // std::vector interface for the Subspaces.
        auto operator [] (size_t index) -> Subspace & { return spaces[index]; }
        auto operator [] (size_t index) const -> const Subspace & { return spaces[index]; }

        auto begin() noexcept { return spaces.begin(); }
        auto begin() const noexcept { return spaces.begin(); }
        auto cbegin() const noexcept { return spaces.cbegin(); }

        auto end() noexcept { return spaces.end(); }
        auto end() const noexcept { return spaces.end(); }
        auto cend() const noexcept { return spaces.cend(); } 

        auto size() const noexcept -> size_t { return spaces.size(); }

        auto create_neighbour_list() -> void;
        auto print_orbits() const -> void;

        structures::Structure structure {};

        // The atomic structure of the system and its corresponding symmetry data.
        Symmetry::SymmetryData symmetry_data;

        // Total number of indpendent tensors.
        size_t ntensors {0};

        // Subspaces for each order.
        std::vector<Subspace> spaces;

        // Maps for the total Spacegroup S and its translation subgroup T.
        Groups::SpaceGroup space_group;
    };
}