#include "structure.h"

#include <cmath>
#include <compare>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "common.h"
#include "extensions.h"
#include "math.h"
#include "dense.h"
#include "string.h"

namespace structures {

    UnitCell::UnitCell(const Dense::Matrix<double> &matrix) : cell(matrix), transposed(Dense::Transpose(matrix)) {
        reciprocal = Dense::invert<double>(transposed);
    }

    auto UnitCell::to_cartesian (const Dense::Vector<double> & vector) const -> Dense::Vector<double> {
        return transposed * vector;
    }

    auto UnitCell::to_scaled (const Dense::Vector<double> & vector) const -> Dense::Vector<double> {
        return reciprocal * vector;
    }

    auto UnitCell::to_string() const -> std::string {
        std::stringstream ss;
        ss << std::endl << "Unit Cell:" << std::endl;
        for (auto const & vector: cell) {
            for (auto const & value: vector) {
                ss << std::fixed << std::setw(19) << std::setprecision(12) << value << " ";
            }
            ss << std::endl;
        }
        ss << std::endl << "Reciprocal Cell:" << std::endl;
        for (auto const & vector: reciprocal) {
            for (auto const & value: vector) {
                ss << std::fixed << std::setw(19) << std::setprecision(12) << value << " ";
            }
            ss << std::endl;
        }
        return ss.str();
    }

// Site.

    auto Site::operator <=> (const Site & other) const noexcept -> std::strong_ordering {
        if (auto cmp = index <=> other.index; cmp != 0) {
            return cmp;
        }
        return indices <=> other.indices;
    }

    auto Site::is_primitive() const -> bool {
        return std::all_of(indices.begin(), indices.end(), [](int x){ return x == 0; });
    }

    auto operator << (std::ostream & os, const Site & site) -> std::ostream & {
        os << "[";
        for (auto const index: site.indices) {
            os << index << ", ";
        }
        os << site.index << "]";
        return os;
    }

    AtomType::AtomType(size_t n) : symbol(common::get_atomic_symbol(n)), number(n) {}

    auto AtomType::operator == (const AtomType & other) const noexcept -> bool {
        return number == other.number;
    }

    auto Atom::operator == (const Atom & other) const noexcept -> bool {
        return get_index() == other.get_index();
    }

    auto operator << (std::ostream & os, const Atom & atom) -> std::ostream & {
        os << " ";
        for (auto const coord: atom.position) {
            os << coord << " ";
        }
        os << "   " << atom.site;
        return os;
    }

    Structure::Structure(const std::vector<std::vector<double>> &c, const double scale) {
        unit_cell = scale * Dense::Matrix<double>(c);
    }

    auto Structure::precalculate_tables(const std::vector<size_t> & indices, const std::vector<size_t> & map_to_primitive) -> void {
        primitive_indices = indices;
        for (auto & atom: atoms) {
            // Set the primtive index of this atom.
            atom.site.index = map_to_primitive[atom.get_index()];

            // Find an equivalent atom in the primitive cell as reference point.
            const auto & other = get_equivalent_atom(atom);

            //std::cout << atom.get_index() << " " << atom.position[0] << " " << atom.position[1] << " " << atom.position[2] << "    ";
            //std::cout << other.get_index() << " " << other.position[0] << " " << other.position[1] << " " << other.position[2] << std::endl;

            if (atom != other) {
                // Find the site indices relative to the primitive atom.
                auto difference = Dense::wrap(atom.position - other.position, Dense::Interval<double> {-0.5, 0.5});
                //std::cout << difference[0] << " " << difference[1] << " " << difference[2] << "    ";
                std::vector<int> site_indices {0, 0, 0};
                for (size_t i = 0; i < site_indices.size(); i++) {
                    site_indices[i] += extensions::as<int>(std::round(difference[i] * common::super_cell_indices[i]));
                }
                atom.site.indices = site_indices;
                //std::cout << atom.site << std::endl;
            }

            // calculate the distances between the atom pairs.
            distance_matrix.push_back(get_distances(atom, atoms, unit_cell));
        }
    }

    void Structure::add_atom(Dense::Vector<double> position, const Site & site, size_t atomic_number, bool is_cartesian) {
        // Convert to scaled coordinates if necessary.
        if (is_cartesian) {
            position = unit_cell.to_scaled(position);
        }     

        // Update the atom_types.
        auto index = extensions::get_index_of(atom_types, [=](const AtomType & atom_type){ return atom_type.number == atomic_number; });
        if (!index) {
            atom_types.push_back(AtomType(atomic_number));
            index = atom_types.size() - 1;
        }

        Atom atom(position, site);
        atom.index = atoms.size();

        atom.atom_type = index.value();
        atom_types[atom.atom_type].count += 1;

        // move the atom into the storage. Definite last use of atom
        atoms.push_back(std::move(atom));   
    }

    auto Structure::get_atom_at(const Dense::Vector<double> & position) const -> std::optional<Atom> {
        for (const auto & atom: atoms) {
            if (Math::approximately_zero(Dense::distance(atom.position, position))) {
                return atom;
            }
        }

        std::cout << "Atoms::position_to_index: Index not found." << std::endl;
        std::cout << String::format_vector(position.values) << std::endl;
        return {};
    }

    auto Structure::get_atom_type(size_t index) const -> const AtomType & {
        return atom_types[index];
    }

    auto Structure::get_distance(const Site & left, const Site & right) const -> double {
        auto difference = ijkl_to_fractional(left) - ijkl_to_fractional(right);
        return Dense::norm(unit_cell.to_cartesian(difference));
    }

    auto Structure::ijkl_to_fractional(Site const & site) const -> Dense::Vector<double> {
        return Dense::convert(site.indices) + atoms[site.index].position;
    }

    auto Structure::ijkls_to_fractional(std::vector<Site> const & sites) const -> std::vector<Dense::Vector<double>> {
        std::vector<Dense::Vector<double>> result;
        for (const auto & site: sites) {
            result.push_back(ijkl_to_fractional(site));
        }
        return result;
    }

    auto Structure::fractional_to_ijkl(Dense::Vector<double> const & position) const -> Site {
        // Find the periodic image in the primitive cell.
        auto other = get_atom_at(Dense::wrap(position, Dense::Interval<double> {0.0, 1.0}));
        auto indices = Dense::convert(position - other.value().position);
        return structures::Site(indices.values, other.value().get_primitive());
    }

    auto Structure::to_string() const -> std::string {
        std::stringstream ss;
        ss << unit_cell.to_string() << std::endl
            << "Number of atoms: " << atoms.size() << std::endl
            << "Found " << atom_types.size() << " atom types: ";
        for (const auto & atom_type: atom_types) {
            ss << atom_type.symbol << " ";
        }
        ss << std::endl;
        return ss.str();    
    }

    auto Structure::get_equivalent_atom(const Atom & atom) const -> const Atom & {
        return atoms[primitive_indices[atom.get_primitive()]];
    }

    auto make_primitive_structure(double cell[3][3], double p[][3], int t[], const std::vector<size_t> & map_to_primitive, bool is_cartesian) -> Structure {
        Structure structure(extensions::to_vector<double>(cell, 3), 1.0);
        size_t natoms = map_to_primitive.size();       
        for (size_t index = 0; index < natoms; index++) {
            //std::span<double> position { p[index] };
            // In the primtive cell index == primitive_index.
            Dense::Vector<double> position { extensions::to_vector<double>(p[index], 3) };
            structure.add_atom(position, Site(index), t[index], is_cartesian);
        }
        structure.precalculate_tables(map_to_primitive, map_to_primitive);
        return structure; 
    }

    StructureTransformer::StructureTransformer(const Structure & from, const Structure & to) {
        matrix = Dense::matrix_product(to.unit_cell.reciprocal, from.unit_cell.transposed);
        for (const auto & atom: to) {
            positions.push_back(atom.position);
        }
    }

    auto StructureTransformer::get_index(const Dense::Vector<double> & vector) const -> std::optional<size_t> {
        // transform the vector to the supercell.
        auto position = Dense::wrap(Dense::times_vector(matrix, vector), Dense::Interval<double> {0.0, 1.0});
        for (size_t index = 0; index < positions.size(); index++) {
            if (Math::approximately_zero(Dense::distance(position, positions[index])))
                return index;
        }
        std::cout << "index not found!" << std::endl;
        return {};
    }

    auto StructureTransformer::get_indices(const std::vector<Dense::Vector<double>> & vectors) const -> std::vector<size_t> {
        std::vector<size_t> indices;
        for (const auto & vector: vectors) {
           indices.push_back(get_index(vector).value());
        }
        return indices;
    }

    auto transform_rotation(const Dense::Matrix<int> & rotation, const UnitCell & unit_cell, bool is_scaled) -> Dense::Matrix<int> {
        // Transform a rotation from fractional to cartesian coordinates and vice versa.
        auto tmp = Dense::convert<int, double>(rotation);
        if (is_scaled) {
            tmp = unit_cell.transposed * tmp * unit_cell.reciprocal;
        }
        else {
            tmp = unit_cell.reciprocal * tmp * unit_cell.transposed;
        }
        return Dense::convert<double, int>(tmp);
    }

    double get_distance(const Atom & left, const Atom & right, const UnitCell & unit_cell) {
        Dense::Vector<double> difference = Dense::wrap(right.position - left.position, Dense::Interval<double> {-0.5, 0.5});
        difference = unit_cell.to_cartesian(difference);
        return Dense::norm(difference);
    }

    std::vector<double> get_distances(const Atom & origin, const std::vector<Atom> & atoms, const UnitCell & unit_cell) {
        std::vector<double> distances;
        for (const auto & other: atoms) {
            distances.push_back(get_distance(origin, other, unit_cell));
        }
        return distances;
    }

    Dense::Matrix<double> wrap_differences(const Dense::Matrix<double> &left, const Dense::Matrix<double> &right) {
        Dense::Matrix<double> wrapped;
        for (size_t i = 0; i < left.size(); i++) {
            wrapped.push_back(Dense::wrap(left[i] - right[i], Dense::Interval<double> {-0.5, 0.5}));
        }
        return wrapped;
    }
}