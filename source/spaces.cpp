#include "spaces.h"

#include <iostream>
#include <iomanip>
#include <vector>

#include "clusters.h"
#include "common.h"
#include "groups.h"
#include "orbits.h"
#include "reader.h"
#include "settings.h"
#include "structure.h"
#include "string.h"
#include "symmetry.h"

namespace Spaces {
// Subspace.

    auto Subspace::print_orbits() const -> void {
        for (auto const & orbit: orbits) {
            std::cout << orbit << std::endl; // << "    Fibers:" << std::endl;
            /*
            for (auto const & fiber: orbit) {
                std::cout << "    " << fiber << std::endl;
            }

            std::cout << "    Isotropies:" << std::endl;
            for (auto const & isotropy: orbit.isotropies) {
                std::cout << "    " << isotropy.index << "    ";
                String::format_vector(std::cout, isotropy.permutation, String::list_format);
                std::cout << std::endl;
            }
            */
        }
        std::cout << std::endl;
    }

    auto find_periodic_image(Dense::Vector<int> const & vector) -> Dense::Vector<int> {
        Dense::Vector<double> result { extensions::as<Dense::Vector<double>>(vector) };
        for (size_t i = 0; i < 3; i++) {
            double limit = common::super_cell_indices[i] / 2.0;
            result[i] = Dense::wrap_value(result[i], Dense::Interval<double> {-limit, limit});
        }
        return extensions::as<Dense::Vector<int>>(result);
    }

    // TODO: this requires some major rework.
    auto Subspace::create_neighbour_tables(structures::Structure const & structure) -> void {

        if (order == 1)
            return;

        // Find supercell indices large enough to contain all possible neighbours within the cutoff.
        std::vector<size_t> super_cell;
        for (size_t i = 0; i < 3; i++) {
            super_cell.push_back(std::ceil(1.15 * cutoff / Dense::norm(structure.unit_cell[i])));
        }

        //std::cout << "Supercell for order " << order << ": [";
        //std::cout << super_cell[0] << " " << super_cell[1] << " " << super_cell[2] << "]" << std::endl;

        // Get all possible combinations of ijkl indices that might be compatible with the cutoff.
        auto combinations = extensions::tensor_product(super_cell);
        for (auto const & atom: structure) {

            //std::cout << "Finding neighbours for: " << atom << std::endl;

            std::vector<structures::Site> neighbours;
            for (size_t l = 0; l < structure.size(); l++) {
                for (auto const & combination: combinations) {
                    structures::Site neighbour { combination, l };
                    if (structure.get_distance(atom.site, neighbour) < cutoff) {
                        //std::cout << neighbour << std::endl;
                        neighbours.push_back(neighbour);
                    } 
                }
            }
            
            // The neighour_sites need to be ordered by site instead of their atom index.
            // NOTE: this should use the <=> of Site.
            std::stable_sort(neighbours.begin(), neighbours.end());
            neighbour_sites.push_back(neighbours);
        }
    }

    auto Subspace::create_sumrules(const Subspace & other, const structures::Structure & structure) -> void {
        Clusters::SphereFilter filter(other.cutoff);
        for (const auto & other_orbit: other.orbits) {
            auto & other_cluster = other_orbit.cluster;

            if (!filter.is_valid(other_cluster))
                continue; 
                
            //std::cout << "Identified sum rule for: " << other_cluster << std::endl;

            Constraints::Sumrule sum_rule { other_cluster };

            for (const auto & cluster: get_neighbour_clusters(other_cluster, structure)) {
                size_t orbit_index = 0;
                bool found = false;
                for (const auto & orbit: orbits) {

                    for (size_t i = 0; i < orbit.fibers.size(); i++) {
                        auto & fiber = orbit.fibers[i];
                        auto translated = Clusters::equivalent_by_translation(fiber.cluster, cluster);
                        if (translated.has_value()) {
                            auto symmetry = fiber.symmetry;
                            symmetry.permutation = extensions::find_smallest_permutation(fiber.cluster.sites, translated.value().sites);
                            sum_rule.add_cluster(symmetry, orbit_index);
                            found = true;

                            //std::cout << "Identified: ['" << cluster << "', " << orbit_index << ", " << symmetry.index << ", [";
                            //for (const auto & value: symmetry.permutation) {
                            //   std::cout << value << ",";
                            //}
                            //std::cout << ";]]" << std::endl;
                            //std::cout << "            " << fiber.cluster << std::endl;

                            break;
                        } 
                    }

                    /*
                    auto symmetry = orbit.identify_cluster(cluster);
                    if (symmetry.has_value()) {
                        sum_rule.add_cluster(symmetry.value(), orbit_index);
                        found = true;
                    }
                    */

                    if (found)
                        break;

                    orbit_index += 1;
                }
            }

            //std::cout << "Sumrule has " << sum_rule.size() << " entries." << std::endl;
            if (sum_rule.size() > 0) {
                //std::cout << "Adding sumrule." << std::endl;
                //sum_rules.push_back(sum_rule);
            }
        }
    }

    auto Subspace::get_neighbour_sites(structures::Site const & site) const -> const std::vector<structures::Site> & {
        return neighbour_sites[site.index];
    }

    auto Subspace::get_neighbour_clusters(const Clusters::Cluster & cluster, const structures::Structure & structure) const -> std::vector<Clusters::Cluster> {
        std::vector<Clusters::Cluster> clusters;
        // Get all nearest neighbor sites for the first atom in the cluster.
        auto const & sites = get_neighbour_sites(cluster.front());
        for (auto const & site: sites) {
            auto new_sites = cluster.sites;
            new_sites.push_back(site);
            clusters.push_back(Clusters::Cluster(new_sites, structure));
        }
        return clusters;
    }

    auto operator << (std::ostream & os, Subspace const & space) -> std::ostream & {
        os << std::setw(4) << space.order << "    "
            << std::setw(4) << space.clusters.size() << "    "
            << std::setw(4) << space.orbits.size() << "    "
            << std::setw(4) << space.sum_rules.size(); 
        return os;
    }

// ClusterSpace
    ClusterSpace::ClusterSpace(Settings::Settings const & settings) {
        structure = reader::read_poscar(settings.structure_file);

        String::section_header("Primitive Structure");
        std::cout << settings.structure_file << " read succesfully." << std::endl;
        std::cout << structure.to_string() << std::endl;

        symmetry_data = Symmetry::SymmetryData(structure);

        // Create the 1st order space explicitely.
        spaces.push_back(Subspace { 1, 0.0 });
        symmetry_data.permutations.resize(1);

        // Setup the lookup table for permutations and their inverse.
        for (size_t order = 0; order < settings.cutoffs.size(); order++) {
            symmetry_data.permutations.push_back(extensions::permutations_of_length(order + 2));
            spaces.push_back(Subspace {order + 2, settings.cutoffs[order]});
        }

        // Get the maps that transform the atom indices for each symmetry Operation.
        space_group = Groups::SpaceGroup(structure, symmetry_data.operations);

        String::section_header("Symmetry Information");

        // Output Spacegroup, number of symmetry operations and the indices of atoms in the primitive cell.
        std::cout << "Space group: " << symmetry_data.symbol << " (" << symmetry_data.number << ")" << std::endl
            << "Number of Symmetry Operations: " << std::setw(3) << symmetry_data.operations.size() << std::endl;
        std::cout << std::endl;
    }

    auto ClusterSpace::create_neighbour_list() -> void {
        // Get a list of all possible sites.
        for (auto & space: spaces) {
            space.create_neighbour_tables(structure);
        }
    }

    auto ClusterSpace::print_orbits() const -> void {
        String::section_header("Orbits");
        std::cout << "Order  Radius  |I|  |S\\I|  Tensors  Sites" << std::endl;
        for (auto const & space: spaces) {
            space.print_orbits();
        }
    }
}