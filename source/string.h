#pragma once

#include <array>
#include <algorithm>
#include <charconv>
#include <iostream>
#include <optional>
#include <string>
#include <sstream>
#include <vector>

namespace String {
// Helper functions for pretty printing.
inline auto section_header(const std::string_view & message) -> void {
    std::cout << std::endl << "    " << message << std::endl << std::endl;
}

// Some helper functions useful until the corresponding string_view versions become more commonly available.
inline auto starts_with(const std::string & string, std::string_view what) -> bool {
    return std::string_view(string).starts_with(what);
}

inline auto contains(const std::string & string, auto const & value) -> bool {
    return string.find(value) != string.npos;
}

// Parsing from string.

inline auto trim(const std::string & string) -> std::string {
    std::string whitespace = " \t";
    auto start = string.find_first_not_of(whitespace);
    if (start == string.npos)
        return ""; // no content

    auto end = string.find_last_not_of(whitespace);
    auto range = end - start + 1;

    return string.substr(start, range);
}

template<typename T> 
    requires std::convertible_to<T, std::string> || std::same_as<T, char>
auto split(const std::string & input, const T & delimiter) -> std::vector<std::string> 
{   
    std::vector<std::string> result;
    if (!input.empty()) {
        size_t start = input.find_first_not_of(delimiter);
        size_t end = start;

        while (start != input.npos) {
            end = input.find_first_of(delimiter, start);
            result.push_back(input.substr(start, end - start));
            start = input.find_first_not_of(delimiter, end);
        }
    }

    return result;
}

// Parse a string to an optional<T> using std::from_chars. 
// NOTE: apparently only integers are supported so far.
// NOTE: It does not appear to work on Clang either.
template<typename T>
auto parse(const std::string & input) -> std::optional<T> {
    // TODO: This should throw an exaption when parsing fails.
    T value;
    // C++23 version:
    // if (std::from_chars(input.begin(), input.end(), value).ec == std::errc{}) {

    // C++20 version does not work on Clang.
    // Leading whitespace is not ignored so find the first non-whitespace char.
    //size_t start = input.find_first_not_of(" "); 
    //if (std::from_chars(input.data() + start, input.data() + input.size(), value).ec == std::errc{}) {
    //    return value;
    //}

    //auto ss = std::stringstream(std::string(input));
    std::stringstream(input) >> value;
    return value;

    //return std::nullopt;
}

// Specialisation because std::from_chars does not convert to std::string. Because this is a full specialisation, usually the
// template should be omitted. But because the type cannot be deduced from the argument, a type parameter is required.
template<>
inline auto parse<std::string>(const std::string & input) -> std::optional<std::string> { return std::string(input); }

template <typename T>
auto parse_vector(const std::vector<std::string> & words) -> std::vector<T> {
    std::vector<T> result;
    for (const auto & word: words) {
        if (auto parsed = parse<T>(word)) result.push_back(parsed.value());
    }
    return result;
}

template<typename T>
auto parse_vector(const std::string & input) -> std::vector<T> {
    return parse_vector<T>(split(input, " "));
}

// Formating vectors.

template<typename T>
auto operator << (std::ostream & os, const std::vector<T> & vector) -> std::ostream & {
    for (size_t i = 0; i < vector.size() - 1; i++) {
        os << vector[i] << " ";
    }
    os << vector.back();
    return os;
}

struct FormatString {
    std::string prefix = "(";
    std::string postfix = ")";
    std::string separator = ", "; 
};

const FormatString list_format {"[", "]", " "};
const FormatString flat_format {"", "", " "};
const FormatString vector_format {};
const FormatString matrix_format {"(", ")", ",\n "};

template<typename T>
auto format_vector(std::ostream & os, const std::vector<T> & vector, const FormatString & format = vector_format) -> void {
    os << format.prefix;
    for (size_t i = 0; i < vector.size() - 1; i++) {
        os << vector[i] << format.separator;
    }
    os << vector.back() << format.postfix;
}

template<typename T>
auto format_vector(const std::vector<T> & vector, const String::FormatString & format = String::vector_format) -> std::string {
    std::stringstream ss;
    format_vector(ss, vector, format);
    return ss.str();
}

template<typename T>
auto format_matrix(std::ostream & os, const std::vector<std::vector<T>> & vector, const FormatString & format = matrix_format) {
    os << format.prefix;
    for (size_t i = 0; i < vector.size() - 1; i++) {
        format_vector(os, vector[i]);
        os << format.separator;
    }
    os << vector.back() << vector_format.postfix << format.postfix;
}  
}