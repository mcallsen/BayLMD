#pragma once

#include <compare>
#include <ostream>
#include <optional>
#include <string>
#include <vector>
#include <utility>

#include "dense.h"
#include "structure.h"

namespace Clusters {
    class Cluster {
        using Site = structures::Site;
        public:
        Cluster() = default;

        // Constructor for atoms in the primitive cell.
        explicit Cluster(const Site & site) noexcept;

        Cluster(const std::vector<Site> & vec) noexcept;
        Cluster(const std::vector<Site> & vec, const structures::Structure & structure);

        auto operator <=> (const Cluster & other) const noexcept -> std::weak_ordering; 
        auto operator == (const Cluster & other) const noexcept -> bool;

        auto front() const -> const Site & { return sites.front(); }
        auto back() const -> const Site & { return sites.back(); }

        auto size() const -> size_t { return sites.size(); }
        auto size_unique() const -> size_t { return sites_unique.size(); } 

        auto add_site(const Site & site) -> void;
        auto update_data() -> void;

        auto is_proper() const -> bool;
        auto is_primitive() const -> bool;

        double radius {0.0};

        size_t prefactor {1};

        bool has_primitive_site {false};

        std::vector<Site> sites {};
        std::vector<Site> sites_sorted {};
        std::vector<Site> sites_unique {};

        std::vector<size_t> multiplicities {};

        private:
        auto sort_by_index() -> void;
    };

    auto operator << (std::ostream & os, const Cluster & cluster) -> std::ostream &;
    auto center_sites(const Cluster & cluster) -> Cluster;

    auto get_radius(const std::vector<structures::Site> & sites, const structures::Structure & structure) -> double;
    auto geometrical_center(const Cluster & cluster, const structures::Structure & structure) -> Dense::Vector<double>;

    auto get_indices(const Cluster & Cluster) -> std::vector<size_t>;

    // Check whether two clusters are equivalent by translation
    auto obviously_inequivalent(const Cluster & left, const Cluster & right) -> bool;

    // Check whether two clusters are equivalent by translation
    auto equivalent_by_translation(const Cluster & left, const Cluster & right) -> std::optional<Cluster>;

    class ClusterFilter {
        public:
        virtual auto is_valid(const Cluster & cluster) const -> bool = 0;
    };

    // This particular filter checks whether each individual pair in the combination fulfills the criterion.
    // Equivalent to the behaviour of Hiphive and CSLD.
    class DistanceFilter: public ClusterFilter {
        public:
        DistanceFilter(structures::Structure & cs, double distance);

        auto is_valid(const Cluster & cluster) const -> bool override;

        private:
        double cutoff;
        structures::Structure & structure;
    };

    class AtomCenteredFilter: public ClusterFilter {
        public:
        auto is_valid(const Cluster & cluster) const -> bool override;
    };

    class SphereFilter: public ClusterFilter {
        public:
        double cutoff;

        SphereFilter(double distance);

        auto is_valid(const Cluster & cluster) const -> bool override;
    };
}