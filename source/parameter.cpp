#include "parameter.h"

#include <sstream>
#include <string>
#include <vector>

#include "extensions.h"
#include "string.h"

namespace Parameter {

    Expression::Expression(const std::string & s) {
        auto tokens = String::split(s, ' ');
        left = String::parse<size_t>(tokens[0]).value();
        right = String::parse<size_t>(tokens[1]).value();
        index = String::parse<size_t>(tokens[2]).value();
    } 

    void Expression::Evaluate(std::vector<double> &vector) const{
        vector[index] = vector[left] * vector[right];
    }

    Parameter::Parameter(double factor, size_t index) {
        Add(factor, index);
    }

    Parameter::Parameter(const std::string & s) {
        auto tokens = String::split(s, ';');
        for (const auto & token: tokens) {
            auto items = String::split(token, ',');
            push_back(String::parse<double>(items[0]).value(), String::parse<size_t>(items[1]).value());
        }
    }

    Parameter& Parameter::operator += (const Parameter &other) {
        for (size_t i = 0; i < other.size(); i++) {
            Add(other.prefactors[i], other.indices[i]);
        }
        return *this;
    }

    Parameter& Parameter::operator -= (const Parameter &other) {
        for (size_t i = 0; i < other.size(); i++) {
            Add(-1 * other.prefactors[i], other.indices[i]);
        }
        return *this;
    }

    auto Parameter::Evaluate(const std::vector<double> & vector) const -> double {
        double result = 0;
        for (size_t i = 0; i < indices.size(); i++) {
            result += prefactors[i] * vector[indices[i]];
        }
        
        return result;
    }

    std::string Parameter::ToString() const {
        std::stringstream ss;
        ss << prefactors[0] << ", " << indices[0]; 
        for (size_t i = 1; i < size(); i++) {
            ss << "; " << prefactors[i] << ", " << indices[i];   
        } 
        return ss.str();
    }

    void Parameter::Add(double factor, size_t index) {
        auto i = extensions::get_index_of(indices, index);
        if (!i) {
            // Adding a new vector.
            push_back(factor, index);
            return;
        }
        prefactors[i.value()] += factor;
    }

    void Parameter::push_back(double factor, size_t index) {
        prefactors.push_back(factor);
        indices.push_back(index);
    }

    Parameter operator + (const Parameter &left, const Parameter &right) {
        Parameter result(left);
        return result += right;
    }

    Parameter operator - (const Parameter &left, const Parameter &right){
        Parameter result(left);
        return result -= right;
    }
}
