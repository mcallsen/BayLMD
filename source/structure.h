
#pragma once

#include <compare>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "dense.h"

namespace structures {
    class UnitCell {
        public:
        Dense::Matrix<double> cell;
        Dense::Matrix<double> transposed;
        Dense::Matrix<double> reciprocal;

        UnitCell() = default;
        UnitCell(const Dense::Matrix<double> & matrix);

        // std::vector interface for the lattice vectors.
        auto operator [] (size_t index) -> Dense::Vector<double> & { return cell[index]; }
        auto operator [] (size_t index) const -> const Dense::Vector<double> & { return cell[index]; }

        auto operator == (const UnitCell & other) const noexcept -> bool = default;

        auto to_cartesian(const Dense::Vector<double> & vector) const -> Dense::Vector<double>;
        auto to_scaled(const Dense::Vector<double> & vector) const -> Dense::Vector<double>;

        auto to_string() const -> std::string;
    };

    class Site {
        public:
        Site() = default;

        // Constructor to be used with a single index in the primitive cell.
        explicit Site(size_t ind) noexcept : Site({0, 0, 0}, ind) {};

        // Constructor for Generic Sites.
        Site(std::vector<int> const & vector, size_t ind) noexcept : indices(vector), index(ind) {};

        auto operator <=> (const Site & other) const noexcept -> std::strong_ordering;
        auto operator == (const Site & other) const noexcept -> bool = default;

        auto is_primitive() const -> bool;

        // The supercell indices will be initialised only in some places. Use std::optional to detect uninitialised usage.
        Dense::Vector<int> indices {}; 

        size_t index {0};               // index in the supercell.
    };

    auto operator << (std::ostream & os, const Site & site) -> std::ostream &;

    class AtomType {
        public:
        AtomType() = default;
        explicit AtomType(size_t n);
 
        auto operator == (const AtomType & other) const noexcept -> bool;

        std::string symbol {};  // Atomic symbol of this type.
        
        size_t number {0};      // Atomic number of this type.
        size_t count {0};       // Number of atoms of this type. 
    };

    class Atom {
        public:
        Atom() = default;
        Atom(const Dense::Vector<double> & pos, const Site & s) noexcept : position(pos), site(s) {}

        Dense::Vector<double> position {};  // position in scaled coordinates.
        Site site {};                       // site of this atom.
        size_t atom_type {0};               // ID of this atoms type.
        size_t index {0};                   // atom index. Atom should not know about its index in the structure.
                                            // However it makes comparing atoms way more convenient.

        auto operator == (const Atom & other) const noexcept -> bool;

        auto get_index() const -> size_t { return index; }
        auto get_primitive() const -> size_t { return site.index; }
    };

    auto operator << (std::ostream & os, const Atom & atom) -> std::ostream &;

    class Structure {
        public:
        UnitCell unit_cell;

        Structure() = default;
        Structure(const std::vector<std::vector<double>> &c, const double scale);

        // std::vector interface for the Atoms.
        auto operator [] (size_t index) -> Atom & { return atoms[index]; }
        auto operator [] (size_t index) const -> const Atom & { return atoms[index]; }

        auto operator == (const Structure & other) const noexcept -> bool = default;

        auto begin() noexcept { return atoms.begin(); }
        auto begin() const noexcept { return atoms.begin(); }
        auto cbegin() const noexcept { return atoms.cbegin(); }

        auto end() noexcept { return atoms.end(); }
        auto end() const noexcept { return atoms.end(); }
        auto cend() const noexcept { return atoms.cend(); } 

        auto size() const noexcept -> const size_t { return atoms.size(); }

        auto push_back(Atom const & atom) -> void { atoms.push_back(atom); }
        auto push_back(Atom && atom) -> void { atoms.push_back(std::forward<Atom>(atom)); }

        // 'Will-move-from parameter' because I could not figure out how to pass an rvalue otherwise.
        auto add_atom(Dense::Vector<double> position, const Site & site, size_t atom_type, bool is_cartesian) -> void;
        //auto add_atom(Dense::Vector<double> && vector, Site && site, size_t atom_type, bool is_cartesian) -> void;

        // Precalculate distances and site indices.
        auto precalculate_tables(const std::vector<size_t> & indices, const std::vector<size_t> & map_to_primitive) -> void;

        auto get_atom_at(const Dense::Vector<double> & position) const -> std::optional<Atom>;
        auto get_atom_type(size_t index) const -> const AtomType &;
        auto get_distance(const Site & left, const Site & right) const -> double;

        // Transform between ijkl and fractional coordinates assuming that structure is the primitive structure.
        auto ijkl_to_fractional(Site const & site) const -> Dense::Vector<double>;
        auto ijkls_to_fractional(std::vector<Site> const & sites) const -> std::vector<Dense::Vector<double>>;
        auto fractional_to_ijkl(Dense::Vector<double> const & position) const -> Site;

        auto to_string() const -> std::string;

        std::vector<size_t> primitive_indices;      // indices of the primitive atoms.

        private:
        auto get_equivalent_atom(const Atom & atom) const -> const Atom &;
        
        std::vector<Atom> atoms;                            // All atoms in the unit cell.
        std::vector<AtomType> atom_types;                   // All atom types.

        std::vector<std::vector<double>> distance_matrix;   // Distance between a pair of atoms i,j. 
    };

    auto make_primitive_structure(double c[3][3], double positions[][3], int t[], const std::vector<size_t> & map_to_primitive, bool is_cartesian) -> Structure;

    class StructureTransformer {
        public:
        Dense::Matrix<double> matrix;
        std::vector<Dense::Vector<double>> positions;

        StructureTransformer(const Structure & from, const Structure & to);

        auto get_index(const Dense::Vector<double> & coordinates) const -> std::optional<size_t>;
        auto get_indices(const std::vector<Dense::Vector<double>> & vectors) const -> std::vector<size_t>;
    };


    // Transform rotation matrices between fractional and cartesian coordinates.
    auto transform_rotation(const Dense::Matrix<int> & rotation, const UnitCell & unit_cell, bool is_scaled) -> Dense::Matrix<int>;
    auto get_distance(const Atom & left, const Atom & right, const UnitCell &unit_cell) -> double;
    auto get_distances(const Atom & origin, const std::vector<Atom> & positions, const UnitCell & unit_cell) -> std::vector<double>;
    auto wrap_differences(const Dense::Matrix<double> & left, const Dense::Matrix<double> & right) -> Dense::Matrix<double>;


}