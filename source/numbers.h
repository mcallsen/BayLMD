#pragma once

#include <compare>
#include <concepts>
#include <ostream>
#include <string>

#include "extensions.h"

namespace Numbers {
    template<typename T>
    // The operators +, -, *, and / for a number are assumed to follow the usual mathematical rules
    // axiom(T a, T b) { a + b == b + a; a - a == 0; a * (b + c) == a * b + a * c; /*...*/ }
    concept Number = std::equality_comparable<T> &&
                     std::three_way_comparable<T> &&
                     requires(T a, T b) {
        {a + b} -> std::convertible_to<T>;
        {a - b} -> std::convertible_to<T>;
        {a * b} -> std::convertible_to<T>;
        {a / b} -> std::convertible_to<T>;
        T { 0 }; // additive invariant.
        T { 1 }; // multiplicative invariant
    };

    template<typename T> T zero = extensions::as<T>(0);
    template<typename T> T one = extensions::as<T>(1);

    class Rational {
        public:
        int numerator {0};
        int denominator {1};

        //RationalNumber() = default;
        Rational(int i): numerator(i) {};
        Rational(int i, int j);

        // Constructor meant for the reader.
        explicit Rational(const std::string &s);

        // NOTE: conversion and comparison as doubles would be safer but only provides a partial order.
        // The int version is, while technically correct, prone to int overflow.
        auto operator <=> (const Rational & other) const { return numerator * other.denominator <=> other.numerator * denominator; }
        auto operator == (const Rational & other) const -> bool = default;

        auto operator += (const Rational & number) -> Rational &;
        auto operator -= (const Rational & number) -> Rational &;
        auto operator *= (const Rational & number) -> Rational &;
        auto operator /= (const Rational & number) -> Rational &;

        // Conversion to double
        explicit operator double() const;

        auto simplify() -> void;

        auto is_zero() const -> bool;
        auto is_none() const -> bool;
    };

    auto operator + (const Rational & left, const Rational & right) -> Rational;
    auto operator - (const Rational & left, const Rational & right) -> Rational;
    auto operator * (const Rational & left, const Rational & right) -> Rational;
    auto operator * (int left, const Rational & right) -> Rational;
    auto operator / (const Rational & left, const Rational & right) -> Rational;

    auto operator << (std::ostream & os, const Rational & number) -> std::ostream &;

    auto inverse(const Rational & number) -> Rational;
}