#pragma once

#include <compare>
#include <cstdio>
#include <string>
#include <vector>

namespace Parameter {
    // An expression representing the product of two displacements.
    class Expression {
        public:
        Expression(size_t l, size_t r, size_t i): left(l), right(r), index(i) {}
        Expression(const std::pair<size_t, size_t> & pair, size_t i): Expression(pair.first, pair.second, i) {}

        // Constructor meant for the reader.
        Expression(const std::string & s);

        // Calculate the product of left and right and insert it in the correct
        // position of the array.
        void Evaluate(std::vector<double> &vec) const;

        // index for the left side of the expression referrencing either a single
        // displacement or the result of another expression.
        size_t left {0};

        // index for the right side of the expression representing a single displacement.
        size_t right {0};

        // the index in the displacement vector, where the result of this expression will be stored.
        size_t index {0};
    };

    // One entry in the sensing matrix representing the sum of products of displacements
    // cooresponding to a single force constant tensor.
    class Parameter {
        public:
        std::vector<double> prefactors {};

        // indices of the expressions in the overall displacement vector.
        std::vector<size_t> indices {};

        Parameter() = default;
        Parameter(double factor, size_t index);

        // Explicit conversion for the sparse::matrix interface.
        explicit Parameter(int i) {}

        // Constructor meant for the reader.
        Parameter(const std::string & s);

        auto operator == (const Parameter & other) const -> bool = default;

        // Arithmetic operators required for the SparseVector interface.
        auto operator += (const Parameter & other) -> Parameter &;
        auto operator -= (const Parameter & other) -> Parameter &;

        auto size() const -> size_t { return indices.size(); }

        auto Evaluate(const std::vector<double> & vector) const -> double;

        auto ToString() const -> std::string;

        // Add another vector and factor to this parameter. The Parameter will be reduced automatically.
        auto Add(double factor, size_t index) -> void;

        // Push a factor and a list of indices to the back of this parameter without checking
        // whether they are already contianed in this. Meant to be used by the reader.
        auto push_back(double factor, size_t index) -> void;
    };

    Parameter operator + (const Parameter &left, const Parameter &right);
    Parameter operator - (const Parameter &left, const Parameter &right);    
}
