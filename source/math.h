#pragma once

#include <cmath>
#include <concepts>

namespace Math {
    template<std::unsigned_integral T>
    constexpr auto power(T x, T p) -> T {
        if (p == 0) return 1;
        if (p == 1) return x;
        
        auto tmp = power(x, p/2);
        if (p % 2 == 0) return tmp * tmp;
        else return x * tmp * tmp;
    };

    template<std::unsigned_integral T>
    constexpr auto factorial(T n) -> T {
        if (n == 0 || n == 1) { return 1; }
        return n * factorial(n - 1);
    };

    template<std::unsigned_integral T>
    constexpr auto logarithm(T number, T base) -> T {
        // Assumption: number is an integer power of base.
        T exponent = 0;
        while (number > 1) {
            number /= base;
            exponent += 1;
        }
        return exponent;
    };

    constexpr auto abs(std::integral auto a) -> std::integral auto {
        if (a < 0) return -1 * a;
        return a;
    };

    constexpr auto epsilon_equal(double left, double right, double epsilon = 0.000001) -> bool {
        return fabs(left - right) <= epsilon;
    }

    constexpr auto epsilon_less_equal(double const left, double const right, double const epsilon = 0.000001) -> bool {
        return epsilon_equal(left, right, epsilon) || left < right;
    }

    constexpr auto approximately_zero(double const value, double const epsilon = 0.000001) -> bool {
        return epsilon_equal(value, 0.0, epsilon);
    }

    template<typename T>
    constexpr auto signum(T const & value) noexcept -> int {
        return (T(0) < value) ? 1 : (value < T(0)) ? -1 : 0;
    }

    constexpr auto levi_civita_symbol(size_t const i, size_t const j, size_t const k) -> int {
        return signum((i - j) * (j - k) * (k - i));
    }
}