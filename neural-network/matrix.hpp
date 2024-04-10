//
//  matrix.hpp
//  neural-network
//
//  Created by Hugh Drummond on 10/4/2024.
//

#ifndef matrix_hpp
#define matrix_hpp

#include <iostream>
#include <vector>

class Matrix {
private:
    std::vector<std::vector<double>> data;
    int rows;
    int cols;

public:
    // Constructors
    Matrix();
    Matrix(int rows, int cols);
    Matrix(int rows, int cols, double val);
    Matrix(const std::vector<std::vector<double>>& values);

    // Accessors
    int numRows() const;
    int numCols() const;
    std::vector<std::vector<double>> getData() const;

    // Basic operations
    Matrix transpose() const;
    void randomize();
    void scalarAdd(double scalar);
    void scalarMultiply(double scalar);
    void add(const Matrix& other);
    void subtract(const Matrix& other);
    void multiply(const Matrix& other);

    // Static methods
    static Matrix multiply(const Matrix& a, const Matrix& b);

    // Operator overloads
    double& operator()(int row, int col);
    const double& operator()(int row, int col) const;
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix operator*(double scalar) const;
    friend std::ostream& operator<<(std::ostream& os, const Matrix& matrix);
};

#endif /* matrix_hpp */
