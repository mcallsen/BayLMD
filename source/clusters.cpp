#include "clusters.h"

#include <algorithm>
#include <compare>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <string>
#include <vector>

#include "common.h"
#include "extensions.h"
#include "math.h"
#include "string.h"
#include "structure.h"
#include "dense.h"

namespace Clusters {
    Cluster::Cluster(const Site & site) noexcept {
        add_site(site);
        update_data();
    }

    Cluster::Cluster(const std::vector<Site> & vector) noexcept {
        for (const auto & site: vector) {
            add_site(site);
        }
        update_data();
    }

    Cluster::Cluster(const std::vector<Site> & vec, const structures::Structure & structure) : Cluster(vec) {
        radius = get_radius(sites, structure);
    }

    // Default order of clusters. For some unknown reason this is important
    // for the application of the accoustic sumrules.
    //
    // (1) order, (a) < (aa) < (aaa).
    // (2) number of unique sites, (aa) < (ab).
    // (3) radius of the cluster.
    // (4) -factorial, (aaab) < (aabb).
    //
    // NOTE: The <=> of the radius is only a partial ordering. We are ignoring NaN here. 
    auto Cluster::operator <=> (const Cluster & other) const noexcept -> std::weak_ordering {
        if (auto cmp = size() <=> other.size(); cmp != 0) { return cmp; }
        if (auto cmp = size_unique() <=> other.size_unique(); cmp != 0) { return cmp; }
        if (!Math::epsilon_equal(radius, other.radius)) {
            auto cmp = radius <=> other.radius;
            if (cmp < 0) return std::weak_ordering::less;
            if (cmp > 0) return std::weak_ordering::greater;
        }
        //if (auto cmp = -prefactor <=> -other.prefactor; cmp != 0) { return cmp; }
        //return sites <=> other.sites;


        // This is suprising. Consider making prefactor negative by default.
        return -prefactor <=> -other.prefactor;
    }

    auto Cluster::operator == (const Cluster & other) const noexcept -> bool {
        if (size() != other.size()) { return false; }
        
        // In principle Clusters should be sorted after setup. NOTE: use Site '==', which only compares indices.
        return sites_sorted == other.sites_sorted;
    }

    auto Cluster::update_data() -> void {
        prefactor = 1; 
        for (const auto & number: multiplicities) {
            prefactor *= Math::factorial(number);
        }

        sort_by_index();

        // Check whether any of the sites in this cluster are primitive.
        has_primitive_site = std::any_of(sites.begin(), sites.end(), [](const Site & site) { return site.is_primitive(); });
    }

    auto Cluster::add_site(structures::Site const & site) -> void {
        // Check whether this site is already in sites_unique. If there was a way to not sort a set that 
        // would be the better data_structure for this.
        auto unique_index = extensions::get_index_of(sites_unique, site);

        if (!unique_index) {
            // We are adding a new atom to the cluster, increase the geometrial order by 1.
            sites_unique.push_back(site);
            multiplicities.push_back(0);
            unique_index = sites_unique.size() - 1;
        }

        multiplicities[unique_index.value()] += 1;
        sites.push_back(site);
    }

    auto Cluster::sort_by_index() -> void {
        sites_sorted = sites;

        // sorting the sites using <=> of Site.
        std::stable_sort(sites_sorted.begin(), sites_sorted.end());
    }

    auto Cluster::is_proper() const -> bool {
        return size() == size_unique();
    }

    auto Cluster::is_primitive() const -> bool {
        return front().is_primitive();
    }

    auto operator << (std::ostream & os, const Cluster & cluster) -> std::ostream & {
        for (size_t i = 0; i < cluster.sites_unique.size(); i++)
        {
            os << "  ";
            size_t multiplicity = cluster.multiplicities[i];
            if (multiplicity > 1) {
                os << multiplicity << "*";
            } 
            os << cluster.sites_unique[i];
        }
        return os;
    }

    auto get_radius(const std::vector<structures::Site> & sites, const structures::Structure & structure) -> double {
        // Compute the CSLD style radius. This is the largest distance between any two atoms.
        double current = 0.0;
        for (const auto & site: sites) {
            for (const auto & other: sites) {
                double distance = structure.get_distance(site, other);
                if (distance > current) {
                    current = distance;
                }
            }
        }

        return current;
    }

    auto geometrical_center(const Cluster & cluster, const structures::Structure & structure) -> Dense::Vector<double> {
        auto & origin = structure[cluster.front().index];

        if (cluster.size() == 1) return origin.position;

        std::vector<Dense::Vector<double>> positions;
        positions.push_back(origin.position);

        for (size_t i = 1; i < cluster.sites_unique.size(); i++) {
            auto & other = structure[cluster.sites_unique[i].index];

            // find the offset for the closest periodic image of other w.r.t. origin.
            auto difference = Dense::wrap(other.position - origin.position, Dense::Interval<double> {-0.5, 0.5});
            positions.push_back(origin.position + difference);
        }

        return Dense::mean(positions);
    }

    // NOTE: consider refactoring code that calls this function.
    auto get_indices(const Cluster & cluster) -> std::vector<size_t> {
        std::vector<size_t> indices;
        for (const auto & site: cluster.sites) {
            indices.push_back(site.index);
        }
        return indices;
    }

    auto translate_cluster(const Cluster & cluster, const Dense::Vector<int> & translation) -> Cluster {
        auto sites = cluster.sites;
        for (size_t i = 0; i < cluster.size(); i++) {
            sites[i].indices += translation;
        }
        Cluster result { sites };
        result.radius = cluster.radius;
        return result;
    }

    auto center_sites(const Cluster & cluster) -> Cluster {
        Dense::Vector<double> average(3, 0);
        for (const auto & site: cluster.sites) {
            average += Dense::convert(site.indices);
        }
        average *= -1.0 / cluster.size();
        return translate_cluster(cluster, Dense::convert(average));
    }

    auto obviously_inequivalent(const Cluster & left, const Cluster & right) -> bool {
        if (left.size() != right.size())
            return true;
        if (left.size_unique() != right.size_unique())
            return true;
        if (!Math::epsilon_equal(left.radius, right.radius))
            return true;
        if (left.prefactor != right.prefactor)
            return true;
        
        return false;
    }

    auto equivalent_by_translation(const Cluster & left, const Cluster & right) -> std::optional<Cluster> {
        if (left.size() != right.size())
            return {};
        
        if (left.sites_sorted == right.sites_sorted)
            return right;

        auto site = left.front();
        for (auto const & other: right.sites) {
            // Only sites with the same l can be equivalent by lattice translations.
            if (site.index != other.index)
                continue;

            auto cluster = translate_cluster(right, site.indices - other.indices);
            if (left.sites_sorted == cluster.sites_sorted)
                return cluster;    
        }
        
        return {};
    }

// ClusterFilter.
    DistanceFilter::DistanceFilter(structures::Structure & structure, double distance) : cutoff(distance), structure(structure) {}

    auto DistanceFilter::is_valid(const Clusters::Cluster & cluster) const -> bool {
        if (cluster.size() == 1) {
            return true;
        }
        
        // This recomputes the CSLD style radius. Probably for now cluster.radius > cutoff should be sufficient.
        for (const auto & site: cluster.sites) {
            for (const auto & other: cluster.sites) {
                double distance = structure.get_distance(site, other);
                if (distance > cutoff) {
                    return false;
                }
            }
        }
        return true;
    }

    auto AtomCenteredFilter::is_valid(const Clusters::Cluster &cluster) const -> bool {
        // This filter checks whether a cluster is centered around one atom.
        // This is true by default, since clusters are constructed from the list of all
        // neighbours of an atom in the primitive cell.

        return true;
    }

    SphereFilter::SphereFilter(double radius): cutoff(radius) {}

    auto SphereFilter::is_valid(const Clusters::Cluster &cluster) const -> bool {
        // This filter checks whether a cluster is within a sphere of radius cutoff.
        return cluster.radius <= cutoff;
    }
}
