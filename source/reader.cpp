#include "reader.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "common.h"
#include "extensions.h"
#include "parameter.h"
#include "settings.h"
#include "sparse.h"
#include "structure.h"
#include "string.h"

namespace reader {
auto read_input(const std::string & file_name) -> Settings::Settings {
    Reader input { file_name };
    auto settings = Settings::Settings();
    for (auto const & line: input.get_lines()) {
        if (line.category != Line::Category::Assignment) 
            continue;

        Expression expression(line);
        if (expression.left == "cutoffs") {
            settings.cutoffs = String::parse_vector<double>(expression.right);
            continue;
        }
        if (expression.left == "structure_file") {
            settings.structure_file = expression.right;
            continue;
        }
        if (expression.left == "supercell_file") {
            settings.supercell_file = expression.right;
            continue;
        }
        if (expression.left == "forcefield_file") {
            settings.forcefield_file = expression.right;
            continue;
        }
    }
    return settings;
}

structures::Structure read_poscar(const std::string & file_name) {
    Reader poscar(file_name);
    Line line;

    structures::Structure structure = read_poscar_header(poscar);   
 
    // VASP5 format also includes atom types
    auto atomic_symbols = read_vector<std::string>(poscar);

    // number of atoms per type
    auto atoms_per_type = read_vector<size_t>(poscar);

    // list of atom type for each atom
    auto atom_types = extensions::repeat(atomic_symbols, atoms_per_type);
    size_t natoms = atom_types.size();

    line = poscar.get_line();
    if (line.value.starts_with('S')) {
        // 'Selective Dynamics' read another line
        line = poscar.get_line();
    }

    bool is_cartesian = line.value.starts_with('C');

    // read positions
    auto positions = read_block<double>(poscar, natoms);

    for (size_t i = 0; i < natoms; i++) {
        structure.add_atom(Dense::Vector<double>(positions[i]), structures::Site(0), common::atomic_numbers[atom_types[i]], is_cartesian);
    }

    poscar.close();
    return structure;
}

structures::Structure read_poscar_header(Reader & input) {
    Line line;

    // Comment Line
    input.skip_line();

    // Scale factor
    double scale = read_vector<double>(input)[0];

    // Unit cell vectors
    std::vector<std::vector<double>> cell = read_block<double>(input, 3);
    structures::Structure structure(cell, scale);

    return structure;   
}

auto read_parameters(Reader & input) -> std::vector<double> {
    size_t n = read_tag<size_t>(input, "Parameters");
    std::cout << "Reading parameters:  " << n << " parameters." << std::endl;
    return read_vector<double>(input);
}

auto read_forcemodel(Reader & input) -> sparse::Matrix<Parameter::Parameter> {
    auto numbers = read_vector_tag<size_t>(input, "Matrix");
    size_t rows = numbers[0];
    size_t columns = numbers[1];

    std::cout << "Reading matrix:      " << rows << " rows, " << columns << " columns." << std::endl;

    sparse::Matrix<Parameter::Parameter> matrix(rows, columns);

    size_t row, count;
    Expression expression;

    for (size_t j = 0; j < rows; j++) {
        numbers = read_vector<size_t>(input);
        row = 3 * numbers[0] + numbers[1];
        count = numbers[2];
        for (size_t i = 0; i < count; i++) {
            auto expression = read_expression<size_t, std::string>(input);

            Parameter::Parameter parameter(expression.second);
            matrix.insert(row, expression.first, parameter);
        }
    }
    return matrix;
}

auto read_symmetry_operations(std::vector<std::vector<std::vector<int>>> & rotations, std::vector<std::vector<double>> & trainslations) -> void {
    Reader input("symops.dat");

    std::cout << "Reading symmetry operations: ";

    // Read the number of rotations and translations.
    auto numbers = read_vector<size_t>(input);

    std::cout << numbers[0] << " rotations, " << numbers[1] << " translations." << std::endl;

    // Read the rotation matrices.
    for (size_t i = 0; i < numbers[0]; i++) {
        auto rotation = read_block<int>(input, 3);
        input.skip_line();
        for (size_t j = 0; j < numbers[1]; j++) {
            std::cout << "rotations progress " << i << " " << j << " " << i + j * numbers[0] << std::endl;
            rotations[i + j * numbers[0]] = rotation;
        } 
    }

    // Read the translation vectors.
    for (size_t i = 0; i < numbers[1]; i++) {
        auto translation = read_vector<double>(input);
        for (size_t j = 0; j < numbers[0]; j++) {
            std::cout << "translations progress " << i << " " << j << " " << j + i * numbers[0] << std::endl;
            trainslations[i * numbers[0] + j] = translation;
        } 
    }

    input.close();

    std::cout << "Finished reading symmetries." << std::endl;
}

auto read_expressions(Reader & input) -> std::vector<std::vector<Parameter::Expression>>{
    size_t max_order = read_tag<size_t>(input, "Expressions");
    std::cout << "Reading expressions: "<< max_order << " orders." << std::endl;
    std::vector<std::vector<Parameter::Expression>> expressions(max_order);

    Line line;
    for (size_t order = 0; order < max_order; order++) {
        line = input.get_line();
        auto n = String::parse<size_t>(line.value).value();
        for (size_t i = 0; i < n; i++) {
            line = input.get_line();
            expressions[order].push_back(Parameter::Expression(line.value));
        }
    }
    return expressions;
}
}