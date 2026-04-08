#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "extensions.h"
#include "math.h"
#include "numbers.h"

namespace Dense {
    template<Numbers::Number T>
    class Vector {
        public:
        std::vector<T> values;

        Vector() = default;
        Vector(const std::vector<T> & vector) : values(vector) {};

        // Fill constructor.
        Vector(size_t n, T const & value) { 
            values = std::vector<T>(n, value); 
        }

        template<Numbers::Number S> requires (!std::is_same_v<S, T>)
        explicit operator Vector<S> () const {
            return convert(*this);
        } 

        auto operator += (Vector const & other) -> Vector & { 
            for (size_t i = 0; i < values.size(); i++) {
                values[i] += other.values[i];
            }
            return *this; 
        }

        auto operator -= (Vector const & other) -> Vector & { 
            for (size_t i = 0; i < values.size(); i++) {
                values[i] -= other.values[i];
            }
            return *this; 
        }

        auto operator *= (T const & value) -> Vector & {
            for (auto & component: values) {
                component *= value;
            } 
            return *this; 
        }

        constexpr auto operator[] (size_t i) -> T &  { return values[i]; }
        constexpr auto operator[] (size_t i) const -> T const & { return values[i]; }

        constexpr auto operator <=> (Vector const & other) const noexcept = default;
        constexpr auto operator == (Vector const & other) const noexcept -> bool = default;

        constexpr auto begin() noexcept { return values.begin(); }
        constexpr auto begin() const noexcept { return values.begin(); }
        constexpr auto cbegin() const noexcept { return values.cbegin(); }

        constexpr auto end() noexcept { return values.end(); }
        constexpr auto end() const noexcept { return values.end(); }
        constexpr auto cend() const noexcept { return values.cend(); } 

        constexpr auto size() const noexcept -> size_t { return values.size(); }

        constexpr auto push_back(T const & value) -> void { values.push_back(value); }
        constexpr auto push_back(T && value) -> void { values.push_back(std::forward<T>(value)); }


        auto to_vector() -> std::vector<T> & { return values; }
        auto to_vector() const -> std::vector<T> const & { return values; }

        auto zero_up_to(size_t index) const -> bool {
            return std::all_of(values.begin(), values.begin() + index, [] (T v) { return v == Numbers::zero<T>; });
        }

        auto is_zero() const -> bool {
            size_t maxIndex = values.size() - 1;
            return zero_up_to(maxIndex);
        }
    };

    template<Numbers::Number T, typename BinaryOperation>
    auto zip_transform(std::vector<T> const & left, std::vector<T> const & right, BinaryOperation const & operation) -> std::vector<T> {
        std::vector<T> result(left.size());
        for (size_t i = 0; i < left.size(); i++) {
            result[i] = operation(left[i], right[i]);
        }
        return result;  
    }

    template<Numbers::Number T>
    auto operator + (Vector<T> const & left, Vector<T> const & right) -> Vector<T> {
        return zip_transform(left.values, right.values, std::plus<T> {});
    }

    template<Numbers::Number T>
    auto operator - (Vector<T> const & left, Vector<T> const & right) -> Vector<T> {
        return zip_transform(left.values, right.values, std::minus<T> {});
    }

    template<Numbers::Number T>
    auto operator * (T const & scalar, Vector<T> const & vector) -> Vector<T> {
        Vector<T> result { vector };
        result *= scalar;
        return result;
    }

    template<Numbers::Number T>
    auto scalar_product(Vector<T> const & left, Vector<T> const & right) -> T {
        //if (left.size() != right.size()) {
        //    std::cout << "Scalar product of vectors with unequal size: " << left.size() << " " << right.size() << std::endl;
        //}
        return std::inner_product(left.begin(), left.end(), right.begin(), Numbers::zero<T>);
    }

    template<Numbers::Number T>
    auto norm(Vector<T> const & vector) -> double {
        T result { scalar_product(vector, vector) };
        return std::sqrt(extensions::as<double>(result));
    }

    template<Numbers::Number T>
    auto distance(Vector<T> const & left, Vector<T> const & right) -> double {
        return norm(right - left);
    }

    inline auto convert(Vector<int> const & vector)  -> Vector<double> {
        Vector<double> result;
        for (const auto value: vector) {
            result.push_back(extensions::as<double>(value));
        }
        return result;
    }

    inline auto convert(Vector<double> const & vector) -> Vector<int> {
        Vector<int> result;
        for (const auto value: vector) {
            result.push_back(extensions::as<int>(std::round(value)));
        }
        return result;
    }

    template<Numbers::Number T>
    struct Interval {
        T lower { Numbers::zero<T> };
        T upper { Numbers::one<T> };

        auto contains(T value) const -> bool { return (lower <= value) && (value < upper); }
        auto size() const -> T { return upper - lower; }
    };

    template<Numbers::Number T>
    auto wrap_value(T const & value, Interval<T> const & interval) -> T  {
        // Check for early return if value is already in the interval.
        //if (interval.contains(value)) return value;

        T width = interval.size();

        // Transform value to [0, 1].
        T fractional { (value - interval.lower) / width };

        // Split number into integral and fractional part.
        T integral;
        fractional = std::modf(fractional, &integral);

        // Due to floating point errors, check both edge cases explicitely.
        if (Math::approximately_zero(fractional) || Math::epsilon_equal(fractional, 1.0)) return interval.lower;

        // Transform back to [lower, upper]
        if (fractional < 0) return interval.upper + fractional * width;
        return interval.lower + fractional * width;
    } 

    template<Numbers::Number T>
    auto wrap(Vector<T> const & vector, Interval<T> const & interval) -> Vector<T> {
        Vector<T> wrapped;
        for (auto const & element: vector) {
            wrapped.push_back(wrap_value(element, interval));
        }
        return wrapped;
    }

    template<Numbers::Number T>
    auto mean(std::vector<Vector<T>> const & vectors) -> Vector<T> {
        size_t size = vectors[0].size();
        Vector<T> average { std::accumulate(vectors.begin(), vectors.end(), Vector<T>(size, Numbers::zero<T>)) };
        return (1 / extensions::as<T>(vectors.size())) * average;
    }

// Matrix

template<Numbers::Number T> 
    class Matrix {
        public:
        std::vector<Dense::Vector<T>> matrix;

        Matrix() = default;
        explicit Matrix(std::vector<Dense::Vector<T>> const & A) noexcept : matrix(A) {}
        explicit Matrix(std::vector<std::vector<T>> const & A) noexcept {
            for (auto const & vector: A) {
                push_back(vector);
            }
        }

        // Fill constructor
        Matrix(size_t rows, size_t columns, T const & value) noexcept {
            matrix = std::vector<Dense::Vector<T>>(rows, Dense::Vector<T>(columns, value));
        }

        auto operator[] (size_t index) -> Dense::Vector<T> & { return matrix[index]; }
        auto operator[] (size_t index) const -> Dense::Vector<T> const & { return matrix[index]; }

        auto operator == (Matrix const & other) const -> bool = default;

        constexpr auto begin() noexcept { return matrix.begin(); }
        constexpr auto begin() const noexcept { return matrix.begin(); }
        constexpr auto cbegin() const noexcept { return matrix.cbegin(); }

        constexpr auto end() noexcept { return matrix.end(); }
        constexpr auto end() const noexcept { return matrix.end(); }
        constexpr auto cend() const noexcept { return matrix.cend(); } 

        constexpr auto size() const noexcept -> size_t { return matrix.size(); }

        constexpr auto push_back(Vector<T> const & vector) -> void { matrix.push_back(vector); }
        constexpr auto push_back(Vector<T> && vector) -> void { matrix.push_back(std::move(vector)); }

        auto operator += (Matrix const & other) -> Matrix & {
            for (size_t i = 0; i < matrix.size(); i++) {
                matrix[i] += other[i];
            }
            return *this;
        }

        auto operator -= (Matrix const & other) -> Matrix & {
            for (size_t i = 0; i < matrix.size(); i++) {
                matrix[i] -= other[i];
            }
            return *this;
        }

        auto operator *= (T const & value) -> Matrix & {
            for (auto & vector: matrix) 
                vector *= value;
            return *this;
        }

        auto to_vector() const -> std::vector<std::vector<T>> const {
            std::vector<std::vector<T>> result;
            for (auto const & vector: matrix) {
                result.push_back(vector.to_vector());
            }
            return result;
        }
    };

    template<Numbers::Number T>
    auto Transpose(Matrix<T> const & matrix) -> Matrix<T> {
        size_t columns { matrix[0].size() };
        Matrix transpose { columns, matrix.size(), extensions::as<T>(0) };
        for (size_t row = 0; row < matrix.size(); row++) {
            for (size_t column = 0; column < columns; column++) {
                transpose[column][row] = matrix[row][column];
            } 
        }
        return transpose; 
    }

    template<Numbers::Number T>
    auto is_identity(Matrix<T> const & matrix) -> bool {
        for (size_t row = 0; row < matrix.size(); row++) {
            for (size_t column = 0; column < matrix[0].size(); column++) {
                T element { matrix[row][column] };
                if ( row == column && element != Numbers::one<T>) return false;
                if ( row != column && element != Numbers::zero<T>) return false;
            }
        }
        return true;  
    }

    template<Numbers::Number T>
    auto flatten(Matrix<T> const & matrix) -> std::vector<T> const {
        std::vector<T> result;
        for (auto const & vector: matrix) {
            result.insert(result.end(), vector.begin(), vector.end());
        }
        return result;
    }

    template<Numbers::Number T> 
    auto matrix_product(Matrix<T> const & left, Matrix<T> const & right) -> Matrix<T> {
        // Assuming that B is already transpose.
        size_t rows = left.size();
        size_t columns = right.size();

        Matrix<T> result {rows, columns, extensions::as<T>(0) };
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < columns; j++) {
                result[i][j] = Dense::scalar_product(left[i], right[j]);
            }
        }
        return result;
    }

    template<Numbers::Number T>
    auto operator + (Matrix<T> const & left, Matrix<T> const & right) -> Matrix<T> {
        Matrix<T> result { left };
        return result += right;     
    }

    template<Numbers::Number T>
    auto operator - (Matrix<T> const & left, Matrix<T> const & right) -> Matrix<T> {
        Matrix<T> result { left };
        return result -= right;     
    }

    template<Numbers::Number T>
    auto operator * (Matrix<T> const & left, Matrix<T> const & right) -> Matrix<T> {
        // Matrix product (m x n) left * (n x p) right. This is not double checked.
        // But it will fail with a Runtime error.
        return matrix_product<T>(left, Transpose(right));     
    }

    template<Numbers::Number T>
    auto operator * (Matrix<T> const & matrix, Vector<T> const & vector) {
        Vector<T> result;
        for (auto const & row: matrix) {
            result.push_back(Dense::scalar_product(row, vector));
        }
        return result;
    }

    template<Numbers::Number T>
    auto operator * (T const & scalar, Matrix<T> const & matrix) -> Matrix<T> {
        Matrix<T> result { matrix };
        result *= scalar;
        return result;
    }

    template<Numbers::Number T>
    auto square_matrix(size_t dimension) -> Matrix<T> {
        return Matrix<T>(dimension, dimension, Numbers::zero<T>); 
    }

    template<Numbers::Number T>
    auto identity(size_t dimension) -> Matrix<T> {
        Matrix<T> result { square_matrix<T>(dimension) };
        for (size_t index = 0; index < dimension; index++) {
            result[index][index] = Numbers::one<T>;
        }
        return result;
    }

    // TODO: this one is a bit weird, but would require a more complicated template for the product operator.
    template<typename TMatrix, typename TVector>
    inline auto times_vector(Matrix<TMatrix> const & matrix, Dense::Vector<TVector> const & vector) -> Dense::Vector<TVector> {
        Dense::Vector<TVector> result { vector.size(), Numbers::zero<TVector> };
        for (size_t i = 0; i < vector.size(); i++){
            for (size_t j = 0; j < vector.size(); j++) {
                result[i] += matrix[i][j] * vector[j];
            } 
        }
        return result; 
    }

    template<Numbers::Number From, Numbers::Number To> requires(!std::is_reference_v<From> && !std::is_reference_v<To>) 
    auto convert(const Matrix<From> matrix) -> Matrix<To> {
        Matrix<To> result;
        for (const auto & vector: matrix) {
            result.push_back(convert(vector));
        }
        return result;
    }

    template<Numbers::Number T>
    auto invert(Matrix<T> const & m) -> Matrix<T> {
        auto inverse = square_matrix<T>(3);
        T determinant { Numbers::zero<T> };

        // computes the inverse of a matrix m
        determinant = m[0][0] * (m[1][1] * m[2][2] - m[2][1] * m[1][2]) -
                      m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                      m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

        determinant = Numbers::one<T> / determinant;

        inverse[0][0] = (m[1][1] * m[2][2] - m[2][1] * m[1][2]) * determinant;
        inverse[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * determinant;
        inverse[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * determinant;
        inverse[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * determinant;
        inverse[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * determinant;
        inverse[1][2] = (m[1][0] * m[0][2] - m[0][0] * m[1][2]) * determinant;
        inverse[2][0] = (m[1][0] * m[2][1] - m[2][0] * m[1][1]) * determinant;
        inverse[2][1] = (m[2][0] * m[0][1] - m[0][0] * m[2][1]) * determinant;
        inverse[2][2] = (m[0][0] * m[1][1] - m[1][0] * m[0][1]) * determinant;

        return inverse;
    }
}

