#pragma once

#include <functional>
#include <map>
#include <vector>

#include "structure.h"
#include "clusters.h" 
#include "symmetry.h"

namespace Groups {

    template<typename T>
    class Group {
        public:
        T & operator [] (size_t index) { return elements[index]; }
        const T & operator [] (size_t index) const { return elements[index]; }

        auto operator == (const Group & other) const noexcept -> bool = default;

        constexpr auto begin() noexcept { return elements.begin(); }
        constexpr auto begin() const noexcept { return elements.begin(); }
        constexpr auto cbegin() const noexcept { return elements.cbegin(); }

        constexpr auto end() noexcept { return elements.end(); }
        constexpr auto end() const noexcept { return elements.end(); }
        constexpr auto cend() const noexcept { return elements.cend(); } 

        constexpr auto size() const noexcept -> size_t { return elements.size(); }

        protected:
        std::vector<T> elements;
    };

    // Represents a space group operation (R, t).
    class SpacegroupOperation {
        using Site = structures::Site;
        public:
        Symmetry::SymmetryOperation operation;

        SpacegroupOperation() = default;
        SpacegroupOperation(structures::Structure const & structure, Symmetry::SymmetryOperation const & operation);

        auto operator == (SpacegroupOperation const & other) const noexcept -> bool = default;

        auto transform(Clusters::Cluster const & cluster, const structures::Structure & structure) const -> Clusters::Cluster;
        auto equivalent(Clusters::Cluster const & left, Clusters::Cluster const & other, structures::Structure const & structure) const -> bool;

        private:
        auto transform(Site const & site) const -> Site;

        std::vector<Site> offsets;
    };

    class Translation {
        public:
        Dense::Vector<double> translation;
        std::map<size_t, size_t> index_map;

        Translation() = default;
        Translation(structures::Structure const & structure, Dense::Vector<double> const & vector);

        auto operator == (Translation const & other) const noexcept -> bool = default;        

        auto transform(std::vector<size_t> const & indices) const -> std::vector<size_t>;

    };

    class SpaceGroup: public Group<SpacegroupOperation> {
        public:
        SpaceGroup() = default;
        SpaceGroup(const structures::Structure & atoms, const std::vector<Symmetry::SymmetryOperation> & operations);

        auto transform(size_t index, const Clusters::Cluster & cluster) const -> Clusters::Cluster;

        // std::optional<std::vector<size_t>> equivalent(const Clusters::Cluster &left, const Clusters::Cluster &right, size_t start = 0, size_t stride = 1);
        bool equivalent(const Clusters::Cluster & left, const Clusters::Cluster & other) const;

        private:
        structures::Structure structure;
    };

    class TranslationGroup: public Group<Translation> {
        public:
        TranslationGroup() = default;
        TranslationGroup(const structures::Structure & structure, const std::vector<Dense::Vector<double>> & vectors);

        auto transform(const std::vector<size_t> & indices) const -> std::vector<std::vector<size_t>>; 
    };

    // Holds maps for all orders between pairs of indices and their position in an array.
    class ExpressionMap {
        public:
        // Next free index in the displacement array. 
        size_t currentIndex;

        // Vector holding the maps for each order.
        std::vector<std::map<std::pair<size_t, size_t>, size_t>> maps;

        ExpressionMap(size_t order, size_t offset);

        // Reduce a vector of atom/coordinate indices to a pair of indices in the displacement vector.
        std::pair<size_t, size_t> reduce_vector(const std::vector<size_t> &indices);
        std::map<std::pair<size_t, size_t>, size_t>::iterator insert(const std::pair<size_t, size_t> &pair, size_t order);

        size_t size(size_t order);
    };
}