#include "orbits.h"

#include <algorithm>
#include <functional>
#include <iomanip>
#include <ostream>
#include <tuple>
#include <vector>

#include "clusters.h"
#include "extensions.h" 
#include "groups.h"
#include "math.h"
#include "numbers.h"
#include "dense.h"
#include "sparse.h"
#include "string.h"
#include "tensors.h"
#include "symmetry.h"

namespace Orbits {

// Eigensymmetry

    EigenSymmetry EigenSymmetry::inverse() const {
        EigenSymmetry symmetry(index, inverse_rotation, rotation, permutation);
        symmetry.primitive_index = primitive_index;
        return symmetry;
    }

    auto EigenSymmetry::is_identity() const -> bool {
        return primitive_index == 0;;
    }

    auto operator << (std::ostream & os, const EigenSymmetry & symmetry) -> std::ostream & {
        os << "Indices: " << symmetry.index << " " << symmetry.primitive_index << std::endl;
        os << "Rotation:" << std::endl;
        String::format_matrix(os, symmetry.rotation.to_vector());
        os << std::endl;
        os << "Permutation: ";
        String::format_vector(os, symmetry.permutation, String::list_format);
        os << std::endl;
        return os;
    }

// Fiber.

    auto Fiber::rotation() const -> EigenSymmetry{
        return pure_rotation(symmetry);
    }

    auto Fiber::inverse_rotation() const -> EigenSymmetry {
        return pure_rotation(symmetry.inverse());
    }

    auto operator << (std::ostream & os, const Fiber & fiber) -> std::ostream & {
        os << fiber.symmetry.index << "   " << fiber.cluster;
        return os;
    }

// Orbit.

    Orbit::Orbit(const Clusters::Cluster & clus, const Groups::SpaceGroup & space_group): cluster(clus) {
        // Because the identity is both in I and S/I add the first fiber by hand.
        fibers.push_back(Fiber(cluster, identity(clus.size())));

        //std::cout << "Creating orbit: " << std::endl;

        // loop over all symmetry operations g in S/T.
        for (size_t i = 0; i < space_group.size(); i++) {
            auto & element = space_group[i];
            auto image = space_group.transform(i, cluster);

            // Shift the cluster's sites to the center of mass.
            image = Clusters::center_sites(image);

            auto translated = Clusters::equivalent_by_translation(cluster, image);

            if (translated.has_value()) {
                //std::cout << "isotropy: " << i << "    " << cluster << "  " << image << "    ";
                //for (auto value: extensions::find_smallest_permutation(cluster.sites, translated.value().sites)) {
                //    std::cout << value << " ";
                //}
                //std::cout << std::endl;
                // Clusters are identical up to translation. Add the symmetry operation
                // to the Isotropy group.
                auto symmetry = EigenSymmetry(i, element.operation, extensions::find_smallest_permutation(cluster.sites, translated.value().sites));
                isotropies.push_back(symmetry);
            }
            else {                
                // Check whether the shifted cluster is equivalent by translation to any of the fibers in this orbit.
                auto is_equivalent = std::any_of(fibers.begin(), fibers.end(), [&](Fiber const & fiber){ return Clusters::equivalent_by_translation(fiber.cluster, image).has_value(); });

                if (!is_equivalent) {
                    //std::cout << "Fiber: " << i << "    " << image << std::endl;
                    //std::cout << element.operation << std::endl;
                    fibers.push_back(Fiber(image, EigenSymmetry(i, element.operation, extensions::range<size_t>(0, clus.size()))));
                }
            }
        }

        //std::cout << "    " << isotropies.size() << "  " << fibers.size() << std::endl;

        tensors = sparse::identity<Numbers::Rational>(Math::power<size_t>(3, cluster.size()));
    }

    auto Orbit::identify_cluster(const Clusters::Cluster & clus) const -> std::optional<EigenSymmetry> {
        //return std::any_of(fibers.begin(), fibers.end(), [&](const Fiber & fiber){ return fiber.contains(clus); });
        for (size_t i = 0; i < fibers.size(); i++) {
            auto & fiber = fibers[i];
            auto translated = Clusters::equivalent_by_translation(fiber.cluster, clus);
            if (translated.has_value()) {
                auto symmetry = fiber.symmetry;
                symmetry.permutation = extensions::find_smallest_permutation(fiber.cluster.sites, translated.value().sites);
                return symmetry;
            }  
        }
        
        return {};
    }

    auto Orbit::add_tensor(size_t index, const sparse::Vector<Numbers::Rational> & tensor) -> void {
        if (sparse::is_empty(tensor)) return;
        tensor_indices.push_back(index);
        tensors.push_back(tensor);
    }

    auto Orbit::rotate_tensors() -> void {
        for (auto & fiber: fibers) {
            fiber.tensors = Tensor::tensor_transform(tensors, fiber.inverse_rotation());
        }
    }

    auto operator << (std::ostream & os, const Orbit & orbit) -> std::ostream & {
        os << std::setw(3) << orbit.cluster.size() << "    "
            << std::fixed << std::setw(6) << std::setprecision(3) << orbit.cluster.radius << " "
            << std::setw(3) << orbit.isotropies.size() << "   "
            << std::setw(3) << orbit.fibers.size() << "  " 
            << std::setw(6) << orbit.tensors.size() << "  "
            << orbit.cluster;
        return os;
    }

// Additional helpers.

    auto identity(size_t order) -> EigenSymmetry{
        return EigenSymmetry(0, Dense::identity<int>(3), Dense::identity<int>(3), extensions::range<size_t>(0, order));
    }

    auto pure_rotation(const EigenSymmetry &other) -> EigenSymmetry {
        size_t order = other.permutation.size();
        return EigenSymmetry(other.index, other.rotation, other.inverse_rotation, extensions::range<size_t>(0, order));
    }
}