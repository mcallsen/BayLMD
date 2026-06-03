#pragma once

#include <algorithm>
#include <concepts>
#include <functional>
#include <map>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace extensions {

// Useful concepts

    template<typename T>
    concept arithmetic = std::integral<T> || std::floating_point<T>;

     // Attempts to express this using std::constructible_from always led to ambigious templates.
    template<typename To, typename From>
    concept constructible_from = !std::same_as<To, From> && requires (From x) { To { x }; }; 

// Generic conversion using 'as'

    // For same type conversion forward the parameter. 
    template<typename T>
    constexpr auto as(T const & x) -> decltype(auto) { return x; }

    // Generic conversion.
    template<typename To, typename From>
    constexpr auto as(From const & x) -> To requires constructible_from<To, From> { return To { x }; }

    template<typename To, typename From> requires std::derived_from<From, To> // std::is_base_of_v<To, From>
    auto as(From && x) -> To && { return std::forward<From>(x); }

    template<class T>
    auto ignore_unused_variable(const T &) {}

// Concept to allow function templates like:
// void print(signature<auto (int) -> int> auto fn, int x) { ... }

    template<class Fn, class Fs>
    struct has_signature;

    template<class Fn, class R, class... Args>
    struct has_signature<Fn, auto (Args...) -> R> : public std::is_invocable_r<R, Fn, Args...> {};

    template<class Fn, class C, class R, class... Args>
    struct has_signature<Fn, auto (C::*)(Args...) -> R>  : public std::is_invocable_r<R, Fn, C&, Args...> {};

    template<class Fn, class Fs>
    constexpr bool has_signature_v = has_signature<Fn, Fs>::value;

    template<class Fn, class Fs>
    concept signature = has_signature_v<Fn, Fs>;

// Monadic functions for std::optional

    template<typename S, typename T>
    constexpr auto and_then(std::optional<S> const & value, std::function<std::optional<T>(S const &)> function) -> std::optional<T> {
        if (value) {
            return function(value.value());
        }
        return std::nullopt;
    }

    template<typename T>
    constexpr auto or_else(std::optional<T> const & value, std::function<std::optional<T>()> function) -> std::optional<T> {
        if (value) {
            return value;
        }
        return function();
    }  

    template<typename S, typename T>
    constexpr auto transform(std::optional<S> const & value, std::function<T(S const &)> function) -> std::optional<T> {
        if (value) {
            return function(value.value());
        }
        return std::nullopt;
    }

// Functions for permutations and sorting.

    // Get a range of unsigned integers between start and end.
    template<std::integral T>
    auto range(T start, T end) -> std::vector<T> {
        std::vector<T> range(end - start);
        std::iota(range.begin(), range.end(), start);
        return range;
    }

    // Get all permutations for a given length.
    inline auto permutations_of_length(size_t length) -> std::vector<std::vector<size_t>> {
        std::vector<std::vector<size_t>> permutations;
        auto permutation = range<size_t>(0, length);
        do {
            permutations.push_back(permutation);
        } while (std::next_permutation(permutation.begin(), permutation.end()));
        return permutations;
    }

    // Find the permutation of indices that has to be applied to vec in order to sort vec w.r.t. to compare.
    // Since in general the order of equivalent objects might matter, use std::stable_sort.
    template<typename T, typename Compare>
    auto sort_permutation(const std::vector<T> & vector, const Compare & compare, size_t start = 0) -> std::vector<size_t> {
        std::vector<size_t> permutation = range<size_t>(0, vector.size());
        std::stable_sort(permutation.begin() + start, permutation.end(), [&](size_t i, size_t j) { return compare(vector[i], vector[j]); });
        return permutation;
    }

    // Apply a permutation of indices to a vector.
    template<typename T>
    auto apply_permutation(const std::vector<T> & vector, const std::vector<size_t> & permutation, size_t start = 0) -> std::vector<T> {
        std::vector<T> sorted = vector;
        std::transform(permutation.begin() + start, permutation.end(), sorted.begin() + start, [&](size_t i) { return vector[i]; });
        return sorted;
    }

    // Sort a and b simultaneously with respect to compare.
    template<typename T, typename U,  typename Compare>
    auto sort_simultaneously(std::vector<T> & a, std::vector<U> & b, const Compare & compare, size_t start = 0) {
        auto permutation = sort_permutation(a, compare, start);
        a = apply_permutation(a, permutation, start);
        b = apply_permutation(b, permutation, start);
    }

/*
    // Find the permutation transforming a into b.
    template<typename T>
    auto find_permutation(const std::vector<T> &a, const std::vector<T> &b) -> std::vector<size_t> {
        auto compare = std::less<T>{};
        return apply_permutation(sort_permutation(a, compare), sort_permutation(b, compare));
    }
*/

    // Get all the indices of a specific element in a container.
    template<typename T>
    auto all_indices(std::vector<T> const & container, T const & element) -> std::vector<size_t> {
        std::vector<size_t> indices;
        for (size_t index = 0; index < container.size(); index++) {
            if (container[index] == element) 
                indices.push_back(index);
        }
        return indices;
    }

    // Find the lowest permutation in lexicographical order transforming a into b.
    template<typename T>
    auto find_smallest_permutation (const std::vector<T> & origin, const std::vector<T> & final) -> std::vector<size_t> {
        std::vector<size_t> permutation(final.size());
        std::vector<std::vector<size_t>> indices(final.size());

        // get all possible indices for each element of the final vector.
        for (size_t index = 0; index < final.size(); index++) {
            indices[index] = all_indices<T>(origin, final[index]);
        }

        for (size_t i = 0; i < final.size(); i++) {
            // Take the first of all possible indices.

            size_t value = indices[i][0];
            permutation[i] = value;

            if (indices[i].size() == 1) continue; 
            
            // remove the index we have just chosen from all other possible indices.
            for (size_t j = i + 1; j < final.size(); j++) {
                std::erase(indices[j], value);
                /*
                std::vector<size_t> current;
                for (auto index: indices[j]) {
                    if (index != value) current.push_back(index);
                }
                indices[j] = current;
                */
            }
        }

        return permutation;
    }

// Extensions for containers. C++23 will have builtin Contains.

    // Special case for span. Probably not used anywhere.
    template<class T, std::size_t N, std::size_t M> [[nodiscard]]
    constexpr auto contains(std::span<T, N> span, std::span<T, M> sub) -> bool {
        return std::search(span.begin(), span.end(), sub.begin(), sub.end()) != span.end();
    }

    // Special case for maps.
    template<typename S, typename T>
    auto contains(const std::map<S, T> map, S key) -> bool {
        auto it = map.find(key);
        return it != map.end();
    }

    // Generic default for everyting else.
    auto contains(auto const & range, auto const & value) -> bool {
        return std::find(range.begin(), range.end(), value) != range.end();
    }

/*

    // check wether a list of type T contains an item of type T.
    // Out of scope variables ('item') used in a lambda need to be captured by value [=]
    // or by reference [&].
    template<typename T>
    auto contains(const std::vector<T> & vector, T item) -> bool {
        return any_of(vector.begin(), vector.end(), [=](T i){return i == item;});
    }

*/

    // Get the index of an item in a list.
    template<typename T> 
    auto get_index_of(const std::vector<T> & vector, const T & item) -> std::optional<size_t> {
        auto it = std::find(vector.begin(), vector.end(), item);
        if(it != vector.end()) {
            return std::distance(vector.begin(), it);
        }
        return std::nullopt;
    }

    // Get the index of an item in a list.
    template<typename T, typename UnaryPredicate> 
    auto get_index_of(const std::vector<T> & vector, const UnaryPredicate & predicate) -> std::optional<size_t> {
        auto it = std::find_if(vector.begin(), vector.end(), predicate);
        if(it != vector.end()) {
            return std::distance(vector.begin(), it);
        }
        return std::nullopt;
    }

    // Get the indices of all itmes in elements in a list.
    template<typename T>
    auto get_indices_of(const std::vector<T> & vector, const std::vector<T> & elements) -> std::vector<T> {
        std::vector<T> indices;
        for (const auto & element: elements) {
            auto index = get_index_of(vector, element);
            if (index) {
                indices.push_back(index.value());
            }
        }
        return indices;
    }

    // Repeat element vector[i] counts[i] times. NOTE: this should be implementable using std::ranges::fold
    template<typename T>
    auto repeat(std::vector<T> const & container, std::vector<size_t> const & counts) -> std::vector<T> {
        // Expects vector.size() == counts.size()
        std::vector<T> result;
        for (size_t index = 0; index < counts.size(); index++) {
            for (size_t j = 0; j < counts[index]; j++) {
                result.push_back(container[index]);
            } 
        }
        return result;
    }

    // Get the counts of all items in a sorted list.
    template<typename T>
    auto multiplicities(std::span<T const> vector) -> std::vector<size_t> {
        std::unordered_map<T, size_t> counts;
        for (auto const & item: vector){
            counts[item] += 1;
        }

        std::vector<size_t> result;
        for (auto it = counts.begin(); it != counts.end(); it++) {
            result.push_back(it.second);
        }

        return result;
    }

// Getting slices of vectors. C++23 std::span<T> should provide builtin functions that simplify this.

    template<typename T>
    auto sub_vectors(const std::vector<T> & vector) -> std::vector<std::vector<T>> {
        // initialise a vector with n copies of the input vector.
        std::vector<std::vector<T>> sub_vectors(vector.size(), vector);

        // remove the ith element from each vector.
        for (size_t index = 0; index < vector.size(); index++) {
            auto & vec = sub_vectors[index]; 
            vec.erase(vec.begin() + index);
        }
        return sub_vectors;
    }

// Conversion functions for C-style arrays.

    template<typename T> 
    auto to_array(const std::vector<T> &v, T arr[]) {
        for (size_t i = 0; i < v.size(); i++) {
            arr[i] = v[i];
        }
    }

    template<typename T>
    auto to_array(const std::vector<std::vector<T>> &v, T arr[][3]) {
        size_t N = v.size();
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < 3; j++) {
                arr[i][j] = v[i][j];
            }
        }
    }

    template<typename T>
    auto to_vector(const T arr[], size_t size) -> std::vector<T> {
        return std::vector<T>(arr, arr + size);
    }

    template<typename T>
    auto to_vector(const T arr[][3], size_t size) -> std::vector<std::vector<T>> {
        std::vector<std::vector<T>> v;
        for (size_t i = 0; i < size; i++) {
            v.push_back(to_vector<T>(arr[i], 3));
        }
        return v;
    }

    template<typename T>
    auto to_vector(const T arr[][3][3], size_t size) -> std::vector<std::vector<std::vector<T>>> {
        std::vector<std::vector<std::vector<T>>> v;
        for (size_t i = 0; i < size; i++) {
            v.push_back(to_vector<T>(arr[i], 3));
        }
        return v;
    }

    template<typename To, std::convertible_to<To> From> 
    auto convert_vector(const std::vector<From> &vector) -> std::vector<To> {
        std::vector<To> result;
        for (const auto & item: vector) {
            result.push_back(as<To>(item));
        }
        return result;
    }

    template<typename T>
    auto Free(T** arr, size_t size) {
        for (size_t i = 0; i < size; i++) {
            delete [] arr[i];
        }
        delete [] arr;
    }

// Extensions for maps.

    template<typename TFirst, typename TSecond>
    auto flip_pair(std::pair<TFirst, TSecond> const & pair) -> std::pair<TSecond, TFirst> {
        return std::pair<TSecond, TFirst>(pair.second, pair.first);
    }

    template<typename TKey, typename TValue> 
    auto invert_map(std::map<TKey, TValue> const & map) -> std::map<TValue, TKey> {
        std::map<TValue, TKey> result;
        std::transform(map.begin(), map.end(), std::inserter(result, result.begin()), flip_pair<TKey, TValue>);
        return result; 
    }

    template<typename TKey, typename TValue>
    auto apply_map(std::vector<TKey> const & vector, std::map<TKey, TValue> const & map) -> std::vector<TValue> {
        std::vector<TValue> result;
        for (auto const & item: vector) {
            result.push_back(map.at(item));
        }
        return result;
    }

// Combinatorics

    // Maybe std::cartesian_product?
    inline auto tensor_product(std::vector<size_t> const & indices) -> std::vector<std::vector<int>> {
        std::vector<std::vector<int>> combinations;
        for (auto i: range<int>(-indices[0], indices[0] + 1)) {
            for (auto j: range<int>(-indices[1], indices[1] + 1)) {
                for (auto k: range<int>(-indices[2], indices[2] + 1)) {
                    combinations.push_back({i, j, k});
                }
            }
        }
        return combinations;
    }

    // Find all combinations of elements with length.
    template<typename T>
    auto combination(std::vector<T> const & elements, size_t length) -> std::vector<std::vector<T>> {
        std::vector<std::vector<T>> combination;

        std::string mask(length, 1); // N leading 1s.
        mask.resize(elements.size(), 0); // N - K trailing 0s.

        do {
            std::vector<T> temp;
            for (size_t index = 0; index < elements.size(); index++) {
                if (mask[index])
                    temp.push_back(elements[index]);
            }
            combination.push_back(temp);
            
        } while(std::prev_permutation(mask.begin(), mask.end()));

        return combination;
    }

    // Find all combinations of elements with a certain length and store them in combinations.
    template<typename T>
    auto product(std::vector<T> & current, std::vector<std::vector<T>> & combinations, const std::vector<T> & elements, size_t length) {
        if (current.size() == length) {
            // Reached the required length. end the recursion.
            combinations.push_back(current);
            return;
        }
        
        for (auto element: elements) {
            std::vector<T> next = current;
            next.push_back(element);
            product(next, combinations, elements, length);
        }
    }
}