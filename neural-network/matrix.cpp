//
//  matrix.cpp
//  neural-network
//
//  Created by Hugh Drummond on 10/4/2024.
//

#include "matrix.hpp"


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

double Matrix::generateBinomial() {
    // Set up random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
//    std::uniform_real_distribution<double> dis(0.0, 1.0);
    std::uniform_real_distribution<double> dis(0.0, 2.0);

    // Generate two independent standard normal random numbers
    double u1 = dis(gen);
//    double u2 = dis(gen);

    // Box-Muller transform
//    double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);

//    return z0;
    return u1 - 1;
}

std::string Matrix::dimension() const {
    return std::format("({} x {})", rows, cols);
}

std::vector<std::vector<double>> Matrix::getData() const {
    return data;
}

void Matrix::randomize() {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            data[i][j] = generateBinomial();
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

Matrix Matrix::transpose() const {
    Matrix result(cols, rows);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result(j, i) = data[i][j];
        }
    }
    return result;
}

Matrix Matrix::add(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        std::cerr << "ERROR: Matrix dimensions do not match for addition." << std::endl;
        return Matrix();
    }
    Matrix result(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result(i, j) = data[i][j] + other(i, j);
        }
    }
    return result;
}

Matrix Matrix::subtract(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        std::cerr << "ERROR: Matrix dimensions do not match for subtraction." << std::endl;
        return Matrix();
    }
    Matrix result(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result(i, j) = data[i][j] - other(i, j);
        }
    }
    return result;
}

Matrix Matrix::multiply(const Matrix& other) const {
    if (cols != other.rows) {
        std::cerr << "ERROR: Matrix dimensions do not match for multiplication." << std::endl;
        std::cerr << std::format("Multiplying {} with {}", dimension(), other.dimension()) << std::endl;
        return Matrix();
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
    return result;
}

Matrix Matrix::elemntWiseMultiply(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        std::cerr << "ERROR: Matrix dimensions do not match for multiplication." << std::endl;
        std::cerr << std::format("Element wise multiplying {} with {}", dimension(), other.dimension()) << std::endl;
        return Matrix();
    }
    Matrix result(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result(i, j) = data[i][j] * other(i, j);;
        }
    }
    return result;
    
}

void Matrix::appendColum(double value) {
    for (int i = 0; i < rows; ++i) {
        data[i].push_back(value);
    }
    cols++;
}

void Matrix::appendRow(double value) {
    data.push_back(std::vector<double>(cols, value));
    rows++;
}

std::vector<double> Matrix::popRow() {
    std::vector<double> ret = data.back();
    data.pop_back();
    rows--;
    return ret;
}

void Matrix::map(std::function<double(double)> func) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            data[i][j] = func(data[i][j]);
        }
    }
}

double& Matrix::operator()(int row, int col) {
    return data[row][col];
}

const double& Matrix::operator()(int row, int col) const {
    return data[row][col];
}

Matrix Matrix::operator+(const Matrix& other) const {
    Matrix result = (*this).add(other);
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    Matrix result = (*this).subtract(other);
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    Matrix result = (*this).multiply(other);
    return result;
}

std::ostream& operator<<(std::ostream& os, const Matrix& matrix) {
    if (matrix.numRows() == 0 && matrix.numCols() == 0) {
        os << "{Empty Matrix}" << std::endl;
        goto end;
    }
    os << std::setprecision(6) << std::fixed;
    for (int i = 0; i < matrix.numRows(); ++i) {
        os << "| ";
        for (int j = 0; j < matrix.numCols(); ++j) {
            os << matrix(i, j) << " ";
        }
        os << "|" << std::endl;
    }
    end:
    return os;
}
