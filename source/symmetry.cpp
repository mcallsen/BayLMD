#include "symmetry.h"

#include "spglib.h"

#include <functional>
#include <numeric>
#include <string>
#include <span>

#include "common.h"
#include "extensions.h"
#include "dense.h"
#include "reader.h"
#include "string.h"
#include "structure.h"

namespace Symmetry {
    SymmetryData::SymmetryData(structures::Structure & structure) {
        int natoms = structure.size();
        int types[natoms];
        double lattice[3][3];
        double positions[natoms][3];

        // Create temporary arrays for the spglib C-interface.
        // TODO: it should be possible to do this in a better way.
        for (auto & atom: structure) {
            auto index = atom.get_index();
            for (size_t i = 0; i < 3; i++) {
                positions[index][i] = atom.position[i];
            }
            types[index] = structure.get_atom_type(atom.atom_type).number;

            //std::cout << positions[index][0] << " " << positions[index][1] << " " << positions[index][2] << " " << types[index] << std::endl;
        }

        // spglib uses the transpose of the unit cell.
        extensions::to_array(structure.unit_cell.transposed.to_vector(), lattice);

        auto data_set = spg_get_dataset(lattice, positions, types, natoms, 1e-5);

        number = data_set->spacegroup_number;
        symbol = data_set->international_symbol;

        // Get all symmetry operations.
        auto translations_temp = extensions::to_vector(data_set->translations, data_set->n_operations);
        auto rotations_temp = extensions::to_vector(data_set->rotations, data_set->n_operations);

        // store a mapping from the original to the primitive cell
        auto test = extensions::to_vector(data_set->mapping_to_primitive, natoms);
        auto vec = extensions::convert_vector<size_t, int>(test);
        
        map_to_primitive = vec;

        // Store the sublattice indices of all atoms in the supercell.
        test = extensions::to_vector(data_set->equivalent_atoms, natoms);
        sublattice_indices = extensions::convert_vector<size_t, int>(test);

        // Remove all duplicates from the mapping to obtain just the indices of the 'primitive' atoms.
        std::sort(vec.begin(), vec.end());
        vec.erase(std::unique(vec.begin(), vec.end()), vec.end());

        primitive_indices = extensions::get_indices_of(map_to_primitive, vec);

        // Store the sublattice indices of all atoms in the supercell.
        test = extensions::to_vector(data_set->equivalent_atoms, natoms);
        sublattice_indices = extensions::convert_vector<size_t, int>(test);

        for (auto index: primitive_indices) {
            primitive_to_sublattice.push_back(sublattice_indices[index]);
        }

        // Number of translations = natoms / n_primitive
        size_t number_of_translations = natoms / primitive_indices.size();

        // Until issues with spglib for non-rectangular cells are solved read the symmetry operationsfrom a file.
        //read_symmetry_operations(rotations_temp, translations_temp);

        find_symmetry_operations(translations_temp, rotations_temp, number_of_translations);

        // Convert the rotation matrices to cartesian coordinates.
        for (auto & operation: operations) {
            operation.rotation_cartesian = structures::transform_rotation(operation.rotation, structure.unit_cell, true);
            operation.inverse_cartesian = Dense::invert<int>(operation.rotation_cartesian);
        }

/*
        map_to_primitive.resize(natoms);

        // Find a map between the supercell and the "primitive" cell. One constraint is that the supercell needs to be an
        // integer multiple of the "primitive" cell, which means that when this is not the case we use the "standard" unit-cell
        // in spglib terms. Find the primitive structure. 
        std::vector<bool> found_mask(natoms, false);
        size_t count = 0;
        for (size_t i = 0; i < natoms; i++) {
            if (found_mask[i])
                continue;

            std::cout << "atom: " << i << ", found: " << found_mask[i] << std::endl;

            primitive_indices.push_back(i);
            auto position = structure[i].position;
            for (auto const & translation: translations) {
                auto other = Dense::wrap(translation.translate(position), 0.0, 1.0);
                auto index = structure.get_atom_at(other).value().get_index();

                std::cout << "translation: ";
                String::format_vector(std::cout, translation.translation.to_vector());
                std::cout << ", other: " << index << ", found: " << found_mask[index] << std::endl;

                if (found_mask[index])
                    continue;

                map_to_primitive[index] = count;
                found_mask[index] = true;
            }
            count += 1;
        }
        
        // NOTE this is actually not used anywhere.
        natoms = spg_standardize_cell(lattice, positions, types, natoms, 1, 1, 1e-5);
        primitive = structures::make_primitive_structure(lattice, positions, types, extensions::range<size_t>(0, natoms), false);
*/

        // Setup the internal tables of the structure.
        structure.precalculate_tables(primitive_indices, map_to_primitive);

        //Debug::print_vector(primitive_to_sublattice, String::list_format);

        // 'dataset' is a C object with allocated pointers so needs to be properly deallocated.
        spg_free_dataset(data_set);
    }

    SymmetryOperation::SymmetryOperation(std::vector<double> &v, std::vector<std::vector<int>> &m) {
        translation = Dense::Vector<double>(v);
        rotation = Dense::Matrix<int>(m);

        is_rotation_identity = Dense::is_identity(rotation);
        is_identity = translation.is_zero() && is_rotation_identity; 
    }

    const std::string SymmetryOperation::as_string() const{
        std::stringstream ss;
        ss << "Rotation Matrix:" << std::endl;
        String::format_matrix(ss, rotation.to_vector());
        ss << std::endl;
        ss << std::endl << "Translation Vector" << std::endl;
        String::format_vector(ss, translation.to_vector());
        ss << std::endl;
        return ss.str();
    }

    const Dense::Vector<double> SymmetryOperation::apply (const Dense::Vector<double> &v) const {
        return Dense::times_vector(rotation, v) + translation;
    }

    auto SymmetryOperation::rotate(Dense::Vector<int> const & vector) const -> Dense::Vector<int> {
        return Dense::times_vector(rotation, vector);
    }

    const Dense::Vector<double> SymmetryOperation::translate (const Dense::Vector<double> &v) const {
        return v + translation;
    }

    void SymmetryData::find_symmetry_operations(std::vector<std::vector<double>> &translationsTemp, std::vector<std::vector<std::vector<int>>> &rotationsTemp, size_t number) {
        
        size_t n_primitive = translationsTemp.size() / number;
        for (size_t i = 0; i < translationsTemp.size(); i++) {
            SymmetryOperation op(translationsTemp[i], rotationsTemp[i]);

            if (i % n_primitive == 0) {
                translations.push_back(op.translation);
            }

            operations.push_back(op);

            if (i < n_primitive) {
                //operations.push_back(op);
                primitive_operations.push_back(op);
                rotation_matrices.push_back(op.rotation);
            } 
        }
    }

    auto operator << (std::ostream & os, const SymmetryOperation & symmetry) -> std::ostream & {
        os << "Rotation:" << std::endl;
        for (size_t i = 0; i < 3; i++) {
            os << symmetry.rotation[i][0] << " " << symmetry.rotation[i][1] << " " << symmetry.rotation[i][2] << "    ";
            os << symmetry.rotation_cartesian[i][0] << " " << symmetry.rotation_cartesian[i][1] << " " << symmetry.rotation_cartesian[i][2] << std::endl;
        }
        os << "Translation:" << std::endl;
        os << symmetry.translation[0] << " " << symmetry.translation[1] << " " << symmetry.translation[2] << std::endl;
        return os;
    }
}