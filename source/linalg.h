#pragma once

#include "sparse.h"

namespace Math {
    // This is a probably slightly faster version, which does not have to deal
    // with the augment part. 
    template<typename T>
    auto reduced_rowechelon_form(sparse::Matrix<T> & matrix) {
        size_t rows = matrix.rows;
        size_t columns = matrix.columns;
         
        size_t row = 0;
        size_t column = 0;

        while (row < rows && column < columns) {
            
            // sort the rows below the current one by their pivot.
            sparse::sort_by_first_index(matrix, row);

            auto & current = matrix[row];
            if (sparse::is_empty(current)) {
                // There are no further non zero rows, we can break early.
                break;
            }

            // Advance to the first non zero element.
            while (column < current.first_index()) {
                column++;    
            }

            // scale the pivot of the current column. to 1.
            current /= current.first_value();

            for (size_t index = 0; index < rows; index++) {
                if (index != row) {
                    auto & other = matrix[index];
                    if (!sparse::is_zero_at(other, column)) {
                        other -= other[column] * current;
                    }
                }
            }
            
            row++; column++;
        }
    }

    // Compute the NullSpace of matrix of rational numbers by gaussian elimination.
    template<typename T>
    auto compute_nullspace(sparse::Matrix<T> & matrix) -> sparse::Matrix<T> {
        // Get M into Row echelon form.

        //std::cout << "Linalg::NullSpace (start): " << matrix.rows << "  " << matrix.columns << std::endl;
        //std::cout << matrix << std::endl;

        reduced_rowechelon_form<T>(matrix);

        //std::cout << "Linalg::NullSpace (RREF): " << matrix.rows << "  " << matrix.columns << std::endl;
        //std::cout << matrix << std::endl;

        size_t columns = matrix.columns;

        // Find the columns that do not have a pivot and the pivots.
        std::vector<bool> column_has_pivot(columns, false);
        std::for_each(matrix.begin(), matrix.end(), [&](auto const & vector){ 
            if (!sparse::is_empty(vector)) {
                column_has_pivot[vector.first_index()] = true;
            } 
        });

        // Find the indices of columns containing basis vectors.
        std::vector<size_t> basis_columns;
        std::vector<size_t> pivot_columns;

        for (size_t index = 0; index < columns; index++) {
            if (column_has_pivot[index]) {
                pivot_columns.push_back(index);
            } else {
                basis_columns.push_back(index);
            }
        }

        // Create the empty nullspace.       
        sparse::Matrix<T> nullspace(basis_columns.size(), columns);

        // There are no degrees of freedom, return an empty matrix.
        if (basis_columns.size() == 0) return nullspace;

        //std::cout << "Indices: ";
        //for (auto index: basis_columns) {
        //    std::cout << index << " ";
        //}
        //std::cout << std::endl;

        //std::cout << "Offsets: ";
        //for (auto index: pivot_columns) {
        //    std::cout << index << " ";
       // }
        //std::cout << std::endl;        

        for (size_t i = 0; i < basis_columns.size(); i++) {
            auto index = basis_columns[i];
            nullspace.insert_quick(i, index, extensions::as<T>(1));
            for (size_t row = 0; row < pivot_columns.size(); row++) {
                nullspace.insert_quick(i, pivot_columns[row], -1 * matrix[row][index]);
            }
        }
    
        // The above construction messes up the order of the elements in the basis vectors
        // so they need to be sorted before we proceed.
        sparse::sort(nullspace);

        return nullspace;
    }
}