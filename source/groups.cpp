#include "groups.h"

#include <functional>
#include <map>
#include <vector>

#include "common.h"
#include "structure.h"
#include "clusters.h"
#include "extensions.h"
#include "symmetry.h"
#include "dense.h"

namespace Groups {
    SpacegroupOperation::SpacegroupOperation(const structures::Structure & structure, const Symmetry::SymmetryOperation & op): operation(op) {
        // Compute the offsets for all primitive atoms.
        for (size_t index = 0; index < structure.size(); index++) {
            // Transform the position of the atom.
            auto position = operation.apply(structure[index].position);
            auto site = structure.fractional_to_ijkl(position);
            offsets.push_back(site);
        }
    }

    auto SpacegroupOperation::transform(Site const & site) const -> Site {
        auto & offset = offsets[site.index];
        auto indices = operation.rotate(site.indices) + offset.indices;
        return Site(indices.values, offset.index);
    }

    auto SpacegroupOperation::transform(const Clusters::Cluster & cluster, const structures::Structure & structure) const -> Clusters::Cluster {
        std::vector<Site> sites;
        for (auto const & site: cluster.sites) {
            sites.push_back(transform(site));
        }
        Clusters::Cluster transformed(sites);
        transformed.radius = cluster.radius;
        return transformed;
    }

    auto SpacegroupOperation::equivalent(const Clusters::Cluster & left, const Clusters::Cluster & right, const structures::Structure & structure) const -> bool {
        auto image = transform(left, structure);
        //auto tmp = Clusters::equivalent_by_translation(image, right).has_value();
        //std::cout << left << " " << right << " " << image << "    " << tmp << std::endl;
        return Clusters::equivalent_by_translation(image, right).has_value();
        //return tmp;
    }

    Translation::Translation(const structures::Structure & structure, const Dense::Vector<double> & vector) {
        for (const auto & atom: structure) {
            // Transform the position of the atom.
            auto position = Dense::wrap(atom.position + vector, Dense::Interval<double> {0.0, 1.0});

            // Find the index of the mapped position.
            auto other = structure.get_atom_at(position);
            index_map[atom.get_index()] = other.value().get_index();
        }
    }

    SpaceGroup::SpaceGroup(const structures::Structure & structure, const std::vector<Symmetry::SymmetryOperation> & operations) : structure(structure) {
        for (const auto & operation: operations) {
            elements.push_back(SpacegroupOperation(structure, operation));
        }
    }

    auto Translation::transform(const std::vector<size_t> & indices) const -> std::vector<size_t> {
        return extensions::apply_map(indices, index_map);
    }

    Clusters::Cluster SpaceGroup::transform(size_t index, const Clusters::Cluster & cluster) const {
        return elements[index].transform(cluster, structure);
    }

    auto SpaceGroup::equivalent(const Clusters::Cluster & left, const Clusters::Cluster & other) const -> bool {
        if (Clusters::obviously_inequivalent(left, other)) return false;
        return std::any_of(elements.begin(), elements.end(), [&](const SpacegroupOperation & element){ return element.equivalent(left, other, structure); });
    }

    TranslationGroup::TranslationGroup(const structures::Structure & structure, const std::vector<Dense::Vector<double>> & vectors) {
        for (size_t i = 0; i < vectors.size(); i++) {
            elements.push_back(Translation { structure, vectors[i] });
        }
    }

    auto TranslationGroup::transform(const std::vector<size_t> & indices) const -> std::vector<std::vector<size_t>> {
        std::vector<std::vector<size_t>> result;
        for (const auto & element: elements) {
            result.push_back(element.transform(indices));
        }
        return result;
    }

    ExpressionMap::ExpressionMap(size_t order, size_t offset) {
        currentIndex = offset;
        maps = std::vector<std::map<std::pair<size_t, size_t>, size_t>>(order - 2);
    }

    std::pair<size_t, size_t> ExpressionMap::reduce_vector(const std::vector<size_t> & vector) {
        size_t order = vector.size() - 2;
        if (order == 0) return std::pair<size_t, size_t>{vector[0], vector[1]};

        // remove the last element from the vector.
        // auto subVector = exl::SubVector(vector, vector.size() - 1);
        auto sub_vector(vector);
        sub_vector.pop_back();
        auto tmp = reduce_vector(sub_vector);

        // It is assumed that Expressions for all subVectors are already in indices.
        return std::pair<size_t, size_t> { maps[order - 1][tmp], vector.back() };
    }

    std::map<std::pair<size_t, size_t>, size_t>::iterator ExpressionMap::insert(const std::pair<size_t, size_t> &pair, size_t order) {
        auto & umap = maps[order - 2];
        auto it = umap.insert(std::pair<std::pair<size_t, size_t>, size_t> (pair, currentIndex));
        if (it.second) {
            // A new value has been inserted increment the current index.
            currentIndex += 1;
        }
        return it.first;
    }

    size_t ExpressionMap::size(size_t order) {
        return maps[order - 2].size();
    }
}