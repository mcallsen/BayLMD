#pragma once

#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include "parameter.h"
#include "settings.h"
#include "sparse.h"
#include "structure.h"
#include "string.h"

namespace reader {
// Readers for basic YAML-like format.

struct Line {
    enum class Category {Empty, Comment, Assignment, Value, Vector};
    static auto determine_category(const std::string & line) -> Category {
        // Determine the line category.
        if (line.starts_with('#')) {
            return Category::Comment;
        }

        // Assignment.
        if (String::contains(line, ":")) {
            return Category::Assignment; 
        }
            
        
        // Single value or vector.
        if (auto vector = String::split(line, " "); vector.size() > 0) {
            return vector.size() == 1 ? Category::Value: Category::Vector;
        }

        return Category::Empty;
    }

    size_t number {0};
    size_t indentation {0};

    std::string value {};

    Category category { Category::Empty };

    Line() = default;
    Line(const std::string & string, size_t line_number): number(line_number) {
        indentation = string.find_first_not_of(" \t");
        value = String::trim(string);
        category = determine_category(value);
    };
};

struct Expression {
    std::string left {};
    std::string right {};
    size_t level {0};

    Expression() = default;
    Expression(const Line & line, const std::string & delimiter = ":")  {
        auto tokens = String::split(line.value, delimiter);
        if (tokens.size() == 2) {
            std::for_each(tokens.begin(), tokens.end(), String::trim);
            left = tokens[0]; 
            right = tokens[1];
            level = line.indentation / 4; 
        }
    }
};

// Sequential file reader. Currently does not check for end of file.
class Reader {
    public:
    std::ifstream file_stream;
    size_t line_number {0};
    std::string line {};

    Reader(const std::string & file_name): file_stream(file_name) {};

    auto get_line() -> Line {
        skip_line();
        return Line(line, line_number - 1);
    }

    auto get_lines() -> std::vector<Line> {
        std::vector<Line> lines;
        // From cpp reference example. Not sure that I 100% understand this type of for loop.
        size_t number = 0;
        for (std::string line; std::getline(file_stream, line);) {
            lines.push_back(Line(line, number));
            number += 1;
        }
        return lines;
    }

    auto skip_line() -> void {
        std::getline(file_stream, line);
        line_number += 1;
    }

    auto close() -> void { file_stream.close(); }
};

template<typename T>
auto parse_value(const Line & line) -> std::optional<T> {
    if (line.category != Line::Category::Value)
        return {};
    return String::parse<T>(line.value);
}

template<typename T>
auto parse_vector(const Line & line) -> std::optional<std::vector<T>> {
    if (line.category != Line::Category::Vector)
        return {};
    return String::parse_vector<T>(line.value);
}

template<typename TLeft, typename TRight>
auto parse_assignment(const Line & line) -> std::optional<std::pair<TLeft, TRight>> {
    if (line.category != Line::Category::Assignment)
        return {};
    Expression expression(line);
    return std::pair(String::parse<TLeft>(expression.left).value(), String::parse<TRight>(expression.right).value());
}

// Convenience functions for reading and parsing. Might be redundant.

template<typename TLeft, typename TRight>
auto read_expression(Reader & input) -> std::pair<TLeft, TRight> {
    return parse_assignment<TLeft, TRight>(input.get_line()).value();
}

template <typename T>
auto read_tag(Reader & input, const std::string & tag) -> T {
    auto values = parse_assignment<std::string, T>(input.get_line()).value();
    return values.second;
}

template <typename T>
auto read_vector_tag(Reader & input, const std::string & tag) -> std::vector<T> {
    Expression expression(input.get_line());
    return String::parse_vector<T>(expression.right);
}

// Helpers for reading POSCAR like files.

template <typename T>
auto read_vector(Reader & input) -> std::vector<T> {
    return parse_vector<T>(input.get_line()).value();
}

template <typename T>
auto read_block(Reader & input, size_t size) -> std::vector<std::vector<T>> {
    std::vector<std::vector<T>> values;
    for (size_t i = 0; i < size; i++) {   
        values.push_back(read_vector<T>(input));
    }
    return values;
}

// Read input file.
auto read_input(const std::string & file_name) -> Settings::Settings;

// Read a VASP POSCAR format file into an 'Atoms' Object.
auto read_poscar(const std::string & file_name) -> structures::Structure;

// Read the header of a poscar file.
auto read_poscar_header(Reader & input) -> structures::Structure;

// Read the parameters of a forcemodel from a file.
auto read_parameters(Reader & input) -> std::vector<double>;

// Read the sensing matrix for a ForceModel from a file.
auto read_forcemodel(Reader & input) -> sparse::Matrix<Parameter::Parameter>;

// Read the expressions representing the products of displacements from a file.
auto read_expressions(Reader & input) -> std::vector<std::vector<Parameter::Expression>>;

auto read_symmetry_operations(std::vector<std::vector<std::vector<int>>> & rotations, std::vector<std::vector<double>> & trainslations) -> void;
}