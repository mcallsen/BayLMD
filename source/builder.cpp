#include "builder.h"

#include <algorithm>
#include <iterator>
#include <map>
#include <numeric>
#include <set>
#include <vector>

#include "clusters.h"
#include "common.h"
#include "spaces.h"
#include "constraints.h"
#include "extensions.h"
#include "groups.h"
#include "orbits.h"
#include "reader.h"
#include "settings.h"
#include "sparse.h"
#include "string.h"

namespace Builder {
    Spaces::ClusterSpace build_cluster_space(Settings::Settings const & settings) {
        Spaces::ClusterSpace cluster_space { settings };

        cluster_space.create_neighbour_list();


        String::section_header("Cluster Space");  
        std::cout << "Order  Clusters  Orbits  Sumrules" << std::endl;

        Spaces::Subspace previous;
        for (auto & space: cluster_space) {
            space.clusters = find_clusters(space, cluster_space);
            space.orbits = create_orbits(space.clusters, cluster_space.space_group);
            space.create_sumrules(previous, cluster_space.structure);

            std::cout << space << std::endl; 

            // Symmetrise tensors.
            previous = space;
        }

        std::cout << std::endl;

        /*
        std::cout << std::endl << "Number of clusters per order:" << std::endl;
        for (const auto & space: cluster_space) {
            std::cout << space.clusters.size() << " ";
        }

        std::cout << std::endl << "Number of orbits per order:" << std::endl;
        for (const auto & space: cluster_space) {
            std::cout << space.orbits.size() << " ";
        }
        */

        // Apply the translation symmetry constraints.

        symmetrize_tensors(cluster_space);

        // Print a table with all Orbits.
        cluster_space.print_orbits();

        return cluster_space;
    }

    auto find_clusters(Spaces::Subspace const & space, Spaces::ClusterSpace const & cluster_space) -> std::vector<Clusters::Cluster> {
        std::vector<Clusters::Cluster> clusters;

        auto const & structure = cluster_space.structure;
        if (space.order == 1) {
            // The clusters of first order are just the primitive indices.
            for (auto const index: structure.primitive_indices) {
                clusters.push_back(Clusters::Cluster(structure[index].site));
            }
            clusters = remove_duplicates(clusters, cluster_space.space_group);
            return clusters;
        }

        clusters = find_proper_clusters(space, cluster_space);
        auto improper = find_improper_clusters(space, cluster_space);

        // add the improper clusters.
        clusters.insert(clusters.end(), improper.begin(), improper.end());

        // Sort the clusters using the builtin <=>
        std::stable_sort(clusters.begin(), clusters.end());
            
        return clusters;
    }

    auto find_proper_clusters(Spaces::Subspace const & space, Spaces::ClusterSpace const & cluster_space) -> std::vector<Clusters::Cluster> {
        std::vector<Clusters::Cluster> clusters;

        // Setup the clusterFilters. NOTE: Currently the Radius of the cluster is computed as the largest distance between any two
        // atoms in the cluster, so that the SphereFilter is equivalent to the DistanceFilter.
        Clusters::SphereFilter filter { space.cutoff };

        auto const & structure = cluster_space.structure;
        auto & previous = cluster_space[space.order - 2].clusters;

        // Use the clusters of order - 1 as basis for this orders clusters.
        for (auto const & cluster: previous) {
            if (cluster.radius > space.cutoff) {
                // Only clusters within the cutoff radius can be a basis for a new proper cluster.
                continue;
            }

            if (!cluster.is_proper()) {
                // Only proper clusters can be a basis for a new proper cluster.
                continue;
            }

            // By definition the first atom in a cluster will be one of the primitve indices.
            // And all atoms that could possiblly be added to the cluster by staying within the cutoff 
            // must be in the list of that atoms neighbours. This is only true for the cluster
            // filter ensuring that none of the bond-lengths in a cluster exceeds the cut-off!
            for (const auto & other: space.get_neighbour_clusters(cluster, structure)) {

                //std::cout << "c: " << cluster << " o: " << other << std::endl;
                if (other.back().index < cluster.back().index) {
                    // We are trying to add a site belonging to a lower orbit.
                    continue;
                }

                if (!other.is_proper())
                    continue;

                if (filter.is_valid(other))
                    clusters.push_back(other);
            }
        }
        
        // sort the clusters.
        std::stable_sort(clusters.begin(), clusters.end());

        //std::cout << std::endl;
        //for (const auto & cluster: clusters) {
        //    std::cout << "Found proper: " << cluster << std::endl;
        //}

        clusters = remove_duplicates(clusters, cluster_space.space_group);

        //std::cout << std::endl;
        //for (const auto & cluster: clusters) {
        //    std::cout << "Found proper: " << cluster << std::endl;
        //}

        return clusters;
    }

    auto compute_multiplicities(std::vector<size_t> const & indices, size_t length) -> std::vector<size_t> {
        if (indices.size() == 0) 
            return std::vector<size_t> { length };

        std::vector<size_t> multiplicties { indices[0] };
        for (size_t index = 0; index < indices.size() - 1; index++) {
            multiplicties.push_back(indices[index + 1] - indices[index]);
        }
        multiplicties.push_back(length - indices.back());
        return multiplicties;
    }

    auto find_improper_clusters(Spaces::Subspace const & space, Spaces::ClusterSpace const & cluster_space) -> std::vector<Clusters::Cluster> {
        std::vector<Clusters::Cluster> clusters;

        for (auto const & other: cluster_space.spaces) {

            if (other.order >= space.order) 
                break;

            // Foreach cluster of the previous order duplicate each unique site in
            // order to obtain the improper clusters. For some reason this has to be done
            // in reverse order.
            for (auto const & cluster: other.clusters) {

                // only proper clusters of a lower order would lead to new improper clusters. Duplicating indices of
                // an improper cluster would yield duplicates of clusters we already have.
                if (!cluster.is_proper())
                    continue;

                // duplicating a site will not change the radius of the cluster.
                if (cluster.radius > space.cutoff)
                    continue;

                // Get all possible combinations of numbers in [1, space.order - 1] of length cluster.size() - 1.
                // These correspond to the starting indices for each of the cluster.size() - 1 sites in the improper
                // cluster. The starting index for the first site is always 1. The indices are in lexicographical order.
                // In the case that space.order - cluster.size() = 1 this corresponds to duplicating sites from right to left.
                auto combinations = extensions::combination(extensions::range<size_t>(1, space.order), cluster.size() - 1);

                for (auto const & combination: combinations) {
                    // The multiplicities of the sites can be computed from the differnce of the starting indices.
                    auto multiplicities { compute_multiplicities(combination, space.order) };

                    // Get the new sites vector including the duplicated site.
                    auto sites { extensions::repeat(cluster.sites_unique, multiplicities) };

                    clusters.push_back(Clusters::Cluster(sites));

                    // Duplicating one of the sites does not alter the radius of the cluster.
                    clusters.back().radius = cluster.radius;
                }
            }
        }

        return remove_duplicates(clusters, cluster_space.space_group);
    }

    auto remove_duplicates(std::vector<Clusters::Cluster> const & clusters, Groups::SpaceGroup const & space_group) -> std::vector<Clusters::Cluster> {
        std::vector<Clusters::Cluster> unique_clusters;
        for (auto const & cluster: clusters) {
            // Check whether any of the unique clusters is symmetrically equivalent to the cluster.
            bool is_equivalent = std::any_of(unique_clusters.begin(), unique_clusters.end(), [&](Clusters::Cluster const & other){ return space_group.equivalent(other, cluster); });
            if (!is_equivalent)
                unique_clusters.push_back(cluster);
        }
        return unique_clusters;
    }

    auto create_orbits(std::vector<Clusters::Cluster> const & clusters, Groups::SpaceGroup const & space_group) -> std::vector<Orbits::Orbit> {
        std::vector<Orbits::Orbit> orbits;
        for (const auto & cluster: clusters) {
            orbits.push_back(Orbits::Orbit(cluster, space_group));
        }
        return orbits;
    }

    // Finding the independet Tensor components for a given orbit and order is equivalent to first
    // creating a basis for a rank order tensor with 3^order components and then subsequently 
    // reducing the basis by applying the following constraints:
    //
    // (1) Some components are symmetric due to repeated atom indices in the prototype cluster,
    //     e.g. for the orbit {aab} the Tensor components obey phi_ijk = phi_jik.
    // (2) Space group symmetry.
    // (3) Translational symmetry (i.e. Acoustic sum rules).
    //
    // The latter involves tensor components of different orbits.
    //
    auto symmetrize_tensors(Spaces::ClusterSpace & cluster_space) -> void {
        std::vector<size_t> counts_spacegroup;
        std::vector<size_t> counts_translational;

        String::section_header("Applying Constraints");

        // size_t row_offset = 6; // First order FCT.
        size_t row_offset = 0; 

        for (auto & space: cluster_space) {
            if (space.order == 1) {
                // The contribution of first order terms to the total energy expansion
                // vanishes due to translational symmetry. Only for completeness.
                for (auto & orbit: space) {
                    orbit.tensors = sparse::Matrix<Numbers::Rational>(0, 3);
                }

                space.first_tensor_index = 0;
            }
            else {
                size_t count = 0;
            
                // Loop over all orbits of the current order.
                for (auto & orbit: space) {
                    // Apply the constraints regarding the repetition of indices.
                    Constraints::apply_index_constraint(orbit);

                    // Apply all constraints regarding the space group symmetry (2).
                    Constraints::apply_spacegroup_constraint(orbit);
                    orbit.tensor_indices = extensions::range(count, count + orbit.tensors.size());

                    count += orbit.tensors.size();
                    orbit.row_offset = row_offset;
                    row_offset += orbit.tensors.columns;
                }
                
                counts_spacegroup.push_back(count);

                space.first_tensor_index = cluster_space.ntensors;

                // Apply the translational symmetry constraint (3). This one relates tensors
                // of different orbits with the same order.
                count = Constraints::apply_translational_constraint(count, space.orbits, space.sum_rules);

                cluster_space.ntensors += count;
                counts_translational.push_back(count);

                // Finally, for each fiber in each orbit the eigentensors need to be
                // rotated according to their eigensymmetry.
                for (auto & orbit: space) {
                    orbit.rotate_tensors();
                }
            }
        }
        std::cout << "Number of independent Tensor components per order (Index + Spacegroup constraint):" << std::endl;
        std::cout << String::format_vector(counts_spacegroup, String::flat_format) << std::endl << std::endl;

        std::cout << "Number of independent Tensor components per order (Translational constraint):" << std::endl;
        std::cout << String::format_vector(counts_translational, String::flat_format) << std::endl << std::endl;
    }
}