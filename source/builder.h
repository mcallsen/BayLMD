#pragma once

#include <string>
#include <vector>

#include "clusters.h"
#include "orbits.h"
#include "spaces.h"
#include "settings.h"

namespace Builder {
    auto build_cluster_space(Settings::Settings const & settings) -> Spaces::ClusterSpace; 

    // Finding clusters.
    auto find_clusters(Spaces::Subspace const & space, Spaces::ClusterSpace const & cluster_space) -> std::vector<Clusters::Cluster>;
    auto find_proper_clusters(Spaces::Subspace const & space, Spaces::ClusterSpace const & cluster_space) -> std::vector<Clusters::Cluster>;
    auto find_improper_clusters(Spaces::Subspace const & space, Spaces::ClusterSpace const & cluster_space) -> std::vector<Clusters::Cluster>;
    auto remove_duplicates(std::vector<Clusters::Cluster> const & clusters, Groups::SpaceGroup const & space_group) -> std::vector<Clusters::Cluster>;

    auto create_orbits(std::vector<Clusters::Cluster> const & clusters, Groups::SpaceGroup const & space_group) -> std::vector<Orbits::Orbit>;
    auto symmetrize_tensors(Spaces::ClusterSpace & cluster_space) -> void;
}