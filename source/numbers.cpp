#include "numbers.h"

#include <iostream>
#include <ostream>
#include <numeric>
#include <string>

#include "extensions.h"
#include "math.h"
#include "string.h"

namespace Numbers {
    Rational::Rational(int i, int j): numerator(i), denominator(j) { simplify(); }
    Rational::Rational(const std::string &s) {
        auto tokens = String::split(s, '/');

        // Checking the size of tokens is a quick hack to avoid dealing with 0 
        if (tokens.size() > 1) {
            numerator =  String::parse<int>(tokens[0]).value();
            denominator = String::parse<int>(tokens[1]).value();
        }
    }

    auto Rational::operator += (const Rational & number) -> Rational & {
        numerator = numerator * number.denominator + number.numerator * denominator;
        denominator = denominator * number.denominator;
        simplify();
        return *this;
    }

    auto Rational::operator -= (const Rational & number) -> Rational & {
        numerator = numerator * number.denominator - number.numerator * denominator;
        denominator = denominator * number.denominator;
        simplify();
        return *this;
    }

    auto Rational::operator *= (const Rational & number) -> Rational & {
        numerator *= number.numerator;
        denominator *= number.denominator;
        simplify();
        return *this;
    }

    auto Rational::operator /= (const Rational &number) -> Rational & {
        numerator *= number.denominator;
        denominator *= number.numerator;
        simplify();
        return *this;
    }

    Rational::operator double() const {
        return extensions::as<double>(numerator) / extensions::as<double>(denominator);
    }

    auto Rational::simplify() -> void {
        //std::cout << "Simplifying: " << numerator << " " << denominator << std::endl;
        // For simplicity the sign should only be on the numerator.
        if (denominator < 0) {
            numerator *= -1;
            denominator *= -1;
        }

        if (denominator == 1) return;
        if (denominator == 0) {
            std::cout << "found 0 denominator" << std::endl;
            numerator = 1; 
            return;
        }

        int divisor = std::gcd(abs(numerator), denominator);
        if (divisor == 0) {
            std::cout << "gcd returned 0 divisor " << numerator << "  " << denominator << std::endl;
        }
        
        numerator = numerator / divisor;
        denominator = denominator / divisor;

        //std::cout << "Simplified: " << numerator << " " << denominator << std::endl << std::endl;
    }

    auto Rational::is_zero() const -> bool{
        return numerator == 0 && denominator != 0;
    }

    auto Rational::is_none() const -> bool{
        return denominator == 0;
    }

    auto operator + (const Rational & left, const Rational & right) -> Rational {
        return Rational(left.numerator * right.denominator + right.numerator * left.denominator, left.denominator * right.denominator);
    }

    auto operator - (const Rational & left, const Rational & right) -> Rational {
        return Rational(left.numerator * right.denominator - right.numerator * left.denominator, left.denominator * right.denominator);
    }

    auto operator * (const Rational & left, const Rational & right) -> Rational {
        return Rational(left.numerator * right.numerator, left.denominator * right.denominator);
    }

    auto operator * (int left, const Rational & right) -> Rational {
        return Rational(left * right.numerator, right.denominator);
    }

    auto operator / (const Rational & left, const Rational & right) -> Rational {
        return Rational(left.numerator * right.denominator, left.denominator * right.numerator);
    }

    auto operator << (std::ostream & os, const Rational & number) -> std::ostream & {
        os << number.numerator << "/" << number.denominator;
        return os;
    }

    auto inverse(const Rational & number) -> Rational {
        if (number.is_zero()) {
            std::cout << "Inverting 0" << std::endl;
        }
        return Rational(number.denominator, number.numerator);
    }
}

