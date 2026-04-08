#pragma once

namespace Dense {

    // Arguments of vector expressions
    template<class ta_a>
    class vecarg {
        const ta_a& Argv;

        public:
        inline vecarg(const ta_a& A): Argv(A) {}
        inline const double Evaluate(const int i) const { return Argv.Evaluate(i); }
    };

    template <>
    class vecarg<const double> {
        const double& Argv;

        public:
        inline vecarg(const double A): Argv(A) {}
        inline const double Evaluate(const int i) const { return Argv; }
    };

    template <>
    class vecarg<const int> {
        const int& Argv;

        public:
        inline vecarg(const int A): Argv(A) {}
        inline const double Evaluate(const int i) const { return (double)Argv; }
    };

    // Vector Expressions
    template<class ta_a, class ta_b, class ta_eval>
    class vecexp_2 {
        const vecarg<ta_a> Arg1;
        const vecarg<ta_b> Arg2;

        public:
        inline vecexp_2(const ta_a A1, const ta_b A2): Arg1(A1), Arg2(A2) {}
        inline const double Evaluate(const int i) const { ta_eval::Evaluate(i, Arg1, Arg2); }
    };

    template<class ta_a, class ta_eval>
    class vecexp_1 {
        const vecarg<ta_a> Arg1;

        public:
        inline vecexp_1(const ta_a A1): Arg1(A1) {}
        inline const double Evaluate(const int i) const { ta_eval::Evaluate(i, Arg1.Evaluate(i)); }
    };
}