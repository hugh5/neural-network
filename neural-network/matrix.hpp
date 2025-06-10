//
//  matrix.hpp
//  neural-network
//
//  Created by Hugh Drummond on 10/4/2024.
//

#ifndef matrix_hpp
#define matrix_hpp

#include <vector>
#include <iostream>
#include <iomanip>
#include <random>
#include <stdexcept>

class Matrix {
private:
    std::vector<std::vector<double>> data;
    size_t rows;
    size_t cols;

public:
    // Constructors
    Matrix() : rows(0), cols(0) {
        data.resize(0);
    }
    Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
        data.resize(rows, std::vector<double>(cols, 0.0));
    }
    Matrix(const std::vector<std::vector<double>>& values) {
        data = values;
        rows = values.size();
        cols = values.empty() ? 0 : values[0].size();
    }

    // Accessors
    size_t numRows() const {
        return rows;
    }

    size_t numCols() const {
        return cols;
    }

    // Matrix methods
    Matrix transpose() const {
        Matrix result(cols, rows);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(j, i) = data[i][j];
            }
        }
        return result;
    }
    
    Matrix hadamard(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument(
                "Matrix dimensions must match for Hadamard product"
            );
        }
        Matrix result(rows, cols);
        
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = data[i][j] * other(i, j);
            }
        }
        return result;
    }
    
    Matrix apply(double (*func)(double)) const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = func(data[i][j]);
            }
        }
        return result;
    }
    // Utility
    void randomize(double min = -1.0, double max = 1.0) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(min, max);
        
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                data[i][j] = dis(gen);
            }
        }
    }
    
    std::vector<double> toVector() const {
        if (cols != 1) {
            throw std::invalid_argument(
                "Can only convert single-column matrix to vector"
            );
        }
        std::vector<double> result(rows);
        for (size_t i = 0; i < rows; ++i) {
            result[i] = data[i][0];
        }
        return result;
    }

    // Operator overloads
    double& operator()(size_t row, size_t col) {
        if (row >= rows || col >= cols || row < 0 || col < 0) {
            throw std::out_of_range("Matrix index out of range");
        }
        return data[row][col];
    }
    const double& operator()(size_t row, size_t col) const {
        if (row >= rows || col >= cols || row < 0 || col < 0) {
            throw std::out_of_range("Matrix index out of range");
        }
        return data[row][col];
    }
    Matrix operator+(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument(
                "ERROR: Matrix dimensions do not match for addition."
            );
        }
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = data[i][j] + other(i, j);
            }
        }
        return result;
    }
    Matrix operator-(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument(
                "ERROR: Matrix dimensions do not match for addition."
            );
        }
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = data[i][j] - other(i, j);
            }
        }
        return result;
    }
    Matrix operator*(double scalar) const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result(i, j) = data[i][j] * scalar;
            }
        }
        return result;
    }
    Matrix operator*(const Matrix& other) const {
        if (cols != other.rows) {
            throw std::invalid_argument(
                "ERROR: Matrix dimensions do not match for multiplication."
            );
        }
        Matrix result(rows, other.cols);
        
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < other.cols; ++j) {
                for (size_t k = 0; k < cols; ++k) {
                    result(i, j) += data[i][k] * other(k, j);
                }
            }
        }
        return result;
    }
    
    friend std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
        if (matrix.numRows() == 0 && matrix.numCols() == 0) {
            os << "{Empty Matrix}" << std::endl;
            goto end;
        }
        os << std::setprecision(6) << std::fixed;
        for (int i = 0; i < matrix.rows; ++i) {
            os << "[";
            for (int j = 0; j < matrix.cols; ++j) {
                os << std::fixed << std::setprecision(4) << matrix.data[i][j];
                if (j < matrix.cols - 1) os << ", ";
            }
            os << "]\n";
        }
        end:
        return os;
    }

};

#endif /* matrix_hpp */
