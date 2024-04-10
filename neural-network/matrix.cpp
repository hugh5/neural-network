//
//  matrix.cpp
//  neural-network
//
//  Created by Hugh Drummond on 10/4/2024.
//

#include "matrix.hpp"

#include <iostream>
#include <iomanip>
#include <cstdlib>

Matrix::Matrix() : rows(0), cols(0) {}

Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols) {
    data.resize(rows, std::vector<double>(cols, 0.0));
}

Matrix::Matrix(int rows, int cols, double val) : rows(rows), cols(cols) {
    data.resize(rows, std::vector<double>(cols, val));
}

Matrix::Matrix(const std::vector<std::vector<double>>& values) {
    data = values;
    rows = (int) values.size();
    cols = (int) ((rows > 0) ? values[0].size() : 0);
}

int Matrix::numRows() const {
    return rows;
}

int Matrix::numCols() const {
    return cols;
}

std::vector<std::vector<double>> Matrix::getData() const {
    return data;
}

Matrix Matrix::transpose() const {
    Matrix result(cols, rows);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result(j, i) = data[i][j];
        }
    }
    return result;
}

void Matrix::randomize() {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            data[i][j] = (double) rand() / RAND_MAX; // Random number between 0 and 1
        }
    }
}

void Matrix::scalarAdd(double scalar) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            data[i][j] += scalar;
        }
    }
}

void Matrix::scalarMultiply(double scalar) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            data[i][j] *= scalar;
        }
    }
}

void Matrix::add(const Matrix& other) {
    if (rows != other.rows || cols != other.cols) {
        std::cerr << "ERROR: Matrix dimensions do not match for addition." << std::endl;
        return;
    }
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            data[i][j] += other(i, j);
        }
    }
}

void Matrix::subtract(const Matrix& other) {
    if (rows != other.rows || cols != other.cols) {
        std::cerr << "ERROR: Matrix dimensions do not match for subtraction." << std::endl;
        return;
    }
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            data[i][j] -= other(i, j);
        }
    }
}

void Matrix::multiply(const Matrix& other) {
    if (cols != other.rows) {
        std::cerr << "ERROR: Matrix dimensions do not match for multiplication." << std::endl;
        return;
    }

    Matrix result(rows, other.cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < other.cols; ++j) {
            double sum = 0;
            for (int k = 0; k < cols; ++k) {
                sum += data[i][k] * other(k, j);
            }
            result(i, j) = sum;
        }
    }

    *this = result;
}

Matrix Matrix::multiply(const Matrix& a, const Matrix& b) {
    Matrix result = a;
    result.multiply(b);
    return result;
}

double& Matrix::operator()(int row, int col) {
    return data[row][col];
}

const double& Matrix::operator()(int row, int col) const {
    return data[row][col];
}

Matrix Matrix::operator+(const Matrix& other) const {
    Matrix result = *this;
    result.add(other);
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    Matrix result = *this;
    result.subtract(other);
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    Matrix result = *this;
    result.multiply(other);
    return result;
}

Matrix Matrix::operator*(double scalar) const {
    Matrix result = *this;
    result.scalarMultiply(scalar);
    return result;
}

std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
    os << std::setprecision(6) << std::fixed;
    for (int i = 0; i < matrix.numRows(); ++i) {
        os << "| ";
        for (int j = 0; j < matrix.numCols(); ++j) {
            os << matrix(i, j) << " ";
        }
        os << "|" << std::endl;
    }
    return os;
}
