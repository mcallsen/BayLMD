#pragma once

#include <algorithm>
#include <concepts>
#include <functional>
#include <iostream>
#include <ostream>
#include <vector>

#include "extensions.h"
#include "numbers.h"

#include "string.h"

namespace sparse {
    template<template<typename T> typename Vector, typename T, typename BinaryOperation>
    auto zip_transform(Vector<T> const & a, Vector<T> const & b, BinaryOperation const & operation) -> Vector<T> {
        size_t i = 0; 
        size_t j = 0;
        Vector<T> result(a.dimension);
        while (i < a.size() && j < b.size()) {
            if (a.indices[i] > b.indices[j]) {
                result.push_back(b.indices[j], operation(Numbers::zero<T>, b.values[j]));
                j++;
            }
            else if (a.indices[i] < b.indices[j]) {
                result.push_back(a.indices[i], a.values[i]);
                i++;
            }
            else {
                result.insert_quick(a.indices[i], operation(a.values[i], b.values[j]));
                i++; 
                j++;
            }
        }

        while (i < a.size()) {
            result.push_back(a.indices[i], a.values[i]);
            i++;
        }

        while (j < b.size()) {
            result.push_back(b.indices[j], operation(Numbers::zero<T>, b.values[j]));
            j++;
        }      
        return result;
    }

    template<typename T> 
    class Vector {
        public:
        // Internal storage.
        std::vector<T> values {};
        std::vector<size_t> indices {};

        // The total dimension of the sparse vector >> size().
        size_t dimension {0};

        Vector() = default;
        explicit Vector(size_t d): dimension(d) {}

        Vector(std::vector<T> const & vector) {
            dimension = vector.size();
            for (size_t i = 0; i < vector.size(); i++) {
                insert_quick(i, vector[i]);
            }
        }

        auto operator <=> (Vector const & other) const noexcept = default;
        auto operator == (Vector const & other) const noexcept -> bool = default;

        constexpr auto begin() noexcept { return values.begin(); }
        constexpr auto begin() const noexcept { return values.begin(); }
        constexpr auto cbegin() const noexcept { return values.cbegin(); }

        constexpr auto end() noexcept { return values.end(); }
        constexpr auto end() const noexcept { return values.end(); }
        constexpr auto cend() const noexcept { return values.cend(); }

        constexpr auto first_value() noexcept { return values.front(); }
        constexpr auto first_value() const noexcept { return values.front(); }

        constexpr auto first_index() const noexcept{
            // Return the first index with non zero value.
            if (indices.size() == 0) return dimension;
            return indices.front();
        }

        constexpr auto first_index() noexcept {
            // Return the first index with non zero value.
            if (indices.size() == 0) return dimension;
            return indices.front();
        }              

        constexpr auto size() const noexcept -> size_t { return indices.size(); }

        auto operator += (Vector const & other) -> Vector & {
            Vector result = zip_transform<Vector, T>(*this, other, std::plus<T>{});
            indices = result.indices; 
            values = result.values;
            return *this;
        }

        auto operator -= (Vector const & other) -> Vector & {
            Vector result = zip_transform<Vector, T>(*this, other, std::minus<T>{});
            indices = result.indices; 
            values = result.values;
            return *this;
        }

        auto operator *= (T const & factor) -> Vector &  {
            if (factor == 0) {
                indices = std::vector<size_t> {};
                values = std::vector<T> {}; 
            } else {
                for (auto & value: values) {
                    value *= factor;
                }
            }
            return *this;
        }

        auto operator /= (T const & factor) -> Vector &  {
            for (auto & value: values) {
                value /= factor;
            }
            return *this;
        }

        auto operator [] (size_t i) -> T & {
            auto index = extensions::get_index_of(indices, i);
            if (index) {
                return values[index.value()];
            }
            return Numbers::zero<T>;
        }

        auto operator [] (size_t i) const -> T const & {
            auto index = extensions::get_index_of(indices, i);
            if (index) {
                return values[index.value()];
            }
            return Numbers::zero<T>;
        }

        auto insert(size_t i, T value) {
            if (value == Numbers::zero<T>) return;
            auto index = extensions::get_index_of(indices, i);
            if (index) {
                values[index.value()] = value; 
            }
            // This is a new entry. Add it to the vector.
            push_back(i, value);
        }

        // Insert assuming that position index is empty.
        auto insert_quick(size_t index, T value) {
            if (value == Numbers::zero<T>) return;
            push_back(index, value);
        }

        // Append a value at the end of the SparseVector. This is intended for
        // quick inserts in case that the indices are ordered and unique.
        auto push_back(size_t index, T value) {
            indices.push_back(index);
            values.push_back(value);
        }

        auto as_vector() const -> std::vector<T> {
            std::vector<T> vector(dimension, Numbers::zero<T>);
            for (size_t i = 0; i < indices.size(); i++) {
                vector[indices[i]] = values[i];
            }
            return vector;
        }
    };

    template<typename T>
    auto operator + (Vector<T> const & left, Vector<T> const & right) -> Vector<T> {
        return zip_transform<Vector, T>(left, right, std::plus<T>{});
    }

    template<typename T>
    auto operator - (Vector<T> const & left, Vector<T> const & right) -> Vector<T> {
        return zip_transform<Vector, T>(left, right, std::minus<T>{});
    }

    template<typename T>
    auto operator * (T scalar, Vector<T> const & vector) -> Vector<T> {
        if (scalar == 0) 
            return Vector<T> { vector.dimension };
        Vector<T> result(vector);
        for (auto & component: result) {
            component *= scalar;
        }
        return result;
    }

    template<typename T>
    auto operator << (std::ostream & os, Vector<T> const & vector) -> std::ostream & {
        if (vector.size() == 0) return os;

        String::operator<<(os, vector.indices) << "    ";
        String::operator<<(os, vector.values);

        return os;
    }

    template<typename T>
    auto scalar_product(Vector<T> const & left, Vector<T> const & right) -> T {
        T result(0);
        size_t i = 0; size_t j = 0;
        while (i < left.indices.size() && j < right.indices.size()) {
            if (left.indices[i] < right.indices[j]) i++;
            else if(left.indices[i] > right.indices[j]) j++; 
            else {
                // Only terms with matching indices contribute to the scalar product.
                result += left.values[i] * right.values[j];
                i++; j++;
            }
        }
        return result;
    }

    // Check whether the vector is zero at a given index. Because we have to check,
    // whether the index is actually contained in the vector this can be relatively slow.
    template<typename T>
    auto is_zero_at(Vector<T> const & vector, size_t index) -> bool {
        return !extensions::get_index_of(vector.indices, index);
    }

    template <typename T>
    auto sort(Vector<T> & vector) {
        // Sort the indices and values simultaneously.
        extensions::sort_simultaneously(vector.indices, vector.values, std::less<size_t>{});
    }

    template<typename T>
    auto shift(Vector<T> const & vector, size_t offset) -> Vector<T> {
        Vector<T> shifted(vector);
        shifted.dimension += offset;
        for (size_t i = 0; i < shifted.size(); i++) {
            shifted.indices[i] += offset;
        }
        return shifted;
    }

    // TODO Split and slice should be replaced by std::partition or something like that.
    template<typename T>
    auto slice(Vector<T> const & vector, size_t lower, size_t upper) -> Vector<T> {
        Vector<T> slice(upper - lower);
        for (size_t i = 0; i < vector.size(); i++) {
            size_t index = vector.indices[i];
            if (index >= upper) break;
            if (index < lower) continue;
            slice.push_back(index - lower, vector.values[i]);
        }
        return slice;
    }

    template<typename T>
    auto split(Vector<T> const & vector, size_t number) -> std::vector<Vector<T>> {
        std::vector<Vector<T>> vectors;
        size_t length = vector.dimension / number;
        for (size_t i = 0; i < number; i++) {
            vectors.push_back(slice(vector, i * length, (i + 1) * length));
        }
        return vectors;
    }

    template<typename T>
    auto is_empty(Vector<T> const & vector) -> bool {
        return vector.size() == 0;
    }

    template<typename T>
    class Matrix {
        public:
        std::vector<Vector<T>> matrix {};

        size_t rows {0};
        size_t columns {0};

        Matrix() = default;
        Matrix(size_t m, size_t n): rows(m), columns(n) {
            for (size_t index = 0; index < m; index++) {
                matrix.push_back(Vector<T>(n));
            }
        }

        Matrix(std::vector<Vector<T>> const & m): rows(m.size()) {
            if (rows > 0) { 
                columns = m[0].dimension; 
            } else { 
                columns = 0; 
            }
            matrix = m;
        }

        Matrix(std::vector<std::vector<T>> const & m): rows(m.size()) {
            if (rows > 0) { 
                columns = m[0].size();
                for (const auto & row: m) {
                    matrix.push_back(Vector<T>(row));
                } 
            } else { 
                columns = 0; 
            }
        }

        Matrix & operator += (Matrix const & other) {
            for (size_t i = 0; i < rows; i++) {
                matrix[i] += other[i];
            }
            return *this;
        }

        Matrix & operator -= (Matrix const & other) {
            for (size_t i = 0; i < rows; i++) {
                matrix[i] -= other[i];
            }
            return *this;
        }

        Matrix & operator *= (T const & factor) {
            for (auto & row: matrix) { 
                row *= factor; 
            }
            return *this;
        }

        auto operator [] (size_t row) -> Vector<T> & { return matrix[row]; }
        auto operator [] (size_t row) const -> const Vector<T> & { return matrix[row]; }

        auto operator <=> (Matrix const & other) const noexcept = default;
        auto operator == (Matrix const & other) const noexcept -> bool = default;

        auto begin() noexcept { return matrix.begin(); }
        auto begin() const noexcept { return matrix.begin(); }
        auto cbegin() const noexcept { return matrix.cbegin(); }

        auto end() noexcept { return matrix.end(); }
        auto end() const noexcept { return matrix.end(); }
        auto cend() const noexcept { return matrix.cend(); } 

        auto size() const noexcept -> size_t { return matrix.size(); }

        auto push_back(Vector<T> const & vector) {
            rows += 1;
            if (vector.dimension > columns)
                columns = vector.dimension;
            matrix.push_back(vector);
        }

        auto push_back(Vector<T> && vector) {
            rows += 1;
            if (vector.dimension > columns)
                columns = vector.dimension;
            matrix.push_back(std::move(vector));
        }

        // General insert. potentially slow because it will check first whether there
        // already is a value in the provided row and column.
        auto insert(size_t row, size_t column, T value) { 
            matrix[row].insert(column, value);
        }

        // Use quick insert if Values are inserted preserving the order.
        auto insert_quick(size_t row, size_t column, T value) {
            matrix[row].insert_quick(column, value);
        }

        // Return the Matrix as 2D std::vector.
        auto as_vector() const -> std::vector<std::vector<T>> {
            std::vector<std::vector<T>> result;
            for (auto const & vector: matrix) {
                result.push_back(vector.as_vector());
            }
            return result;
        }
    };

    template<typename T>
    auto operator << (std::ostream & os, Matrix<T> const & matrix) -> std::ostream & {
        for (size_t i = 0; i < matrix.rows; i++) {
            os << i << ": " << matrix[i] << std::endl;
        }
        return os;
    }

// Non-member functions for Sparse::Matrix.

    // Apply the matrix to a vector
    template<typename T>
    auto times_vector(Matrix<T> const & matrix, Vector<T> const & vec) -> Vector<T> {
        Vector<T> result(matrix.size());
        for (size_t index = 0; index < matrix.size(); index++) {
            // Since the indices are unique, we can use SparseVector.QuickInsert.
            result.insert_quick(index, sparse::scalar_product(matrix[index], vec));
        }
        return result;
    }

    // Get the transpose of a sparse matrix.
    template<typename T>
    auto transpose(Matrix<T> const & matrix) -> Matrix<T> {
        // Get the transpose of a SparseMatrix. Because the transpose does not
        // preserve the order of the SparseVectors the matrix will be sorted.
        Matrix<T> result(matrix.columns, matrix.rows);
        for (size_t i = 0; i < matrix.size(); i++) {
            auto & vector = matrix[i];
            for (size_t j = 0; j < vector.size(); j++) {
                // Since the indices are unique and the values non zero, we
                // can just use SparseVector.push_back.
                result[vector.indices[j]].push_back(i, vector.values[j]);
            }
        }
        sort(result);

        return result;
    }

    // Compute the matrix product of two sparse matrices.
    template<typename T>
    auto matrix_product(Matrix<T> const & left, Matrix<T> const & right) -> Matrix<T> {
        // TODO: this should be GSL::Expects()
        if (left.columns != right.rows) {
            std::cout << "Sparse::matrix_product: Multiplying incompatible matrices. " << left.columns << " " << right.rows << std::endl; 
        }
        
        Matrix<T> product(left.rows, right.columns);
        auto tr = transpose(right);
        for (size_t i = 0; i < left.size(); i++) {
            for (size_t j = 0; j < tr.size(); j++) {
                // Since the values are inserted in order, we can use QuickInsert.
                product.insert_quick(i, j, sparse::scalar_product<T>(left[i], tr[j]));
            }
        }
        return product;
    }

    // Sort all the SparseVectors in this SparseMatrix.
    template<typename T>
    auto sort(Matrix<T> & matrix) {
        for (auto & row: matrix) { 
            sort(row); 
        } 
    }

    // Sort the rows of the matrix by the index of the first non-zero element.
    template<typename T>
    auto sort_by_first_index(Matrix<T> & matrix, size_t start = 0) {
        std::stable_sort(matrix.begin() + start, matrix.end(), [](Vector<T> const & left, Vector<T> const & right) 
            { 
                //if (left.pivot() != right.pivot()) 
                    return left.first_index() < right.first_index();
                //return left.reverse_pivot() < right.reverse_pivot(); 
            });
    }

    template<typename T>
    auto swap_rows(Matrix<T> & matrix, size_t index, size_t other) -> void {
        auto tmp = matrix[index];
        matrix[index] = matrix[other];
        matrix[other] = tmp;
    } 

    // Helper struct for row, column offsets.
    struct Offset { 
        size_t row {0}; 
        size_t column {0}; 
    };

    // Add or insert a block given by another sparseMatrix. It is assumed that M is
    // big enough to accomodate other. In general, does not preserve the ordering of the vectors.
    template<typename T>
    auto add_with_offset(Matrix<T> & matrix, Matrix<T> const & other, Offset const & offset = {0, 0} ) {
        // Ensure(matrix.rows > other.rows + offset.rows && matrix.columns > other.columns + offset.columns)
        for (size_t index = 0; index < other.size(); index++) {
            matrix[index + offset.row] += shift(other[index], offset.column);
            //add_with_offset<T>(matrix, other[index], index + offset.row, offset.column);
        }
    }

    /*
    // Add or Insert a single row given by a Sparsevector with a given offset.
    template<typename T>
    auto add_with_offset(Matrix<T> & matrix, const Vector<T> &vector, size_t index, size_t offset = 0) {
        matrix[index] += shift<T>(vector, offset);
    }
    */

    // Return Id(n).
    template<typename T>
    auto identity(size_t n) -> Matrix<T> {
        Matrix<T> identity(n, n);
        for (size_t index = 0; index < n; index++) {
            identity[index].push_back(index, Numbers::one<T>);
        }
        return identity;
    }

    // Return a vector with all the non-zero values.
    template<typename T>
    auto values(Matrix<T> const & matrix) -> std::vector<T> {
        std::vector<T> result;
        for (auto const & vector: matrix) {
            result.insert(result.end(), vector.values.begin(), vector.values.end());
        }
        return result;
    }

    template<typename T>
    auto is_empty(Matrix<T> const & matrix) -> bool {
        return std::all_of(matrix.begin(), matrix.end(), [](Vector<T> const & vector){ return is_empty(vector); });
    }   
}