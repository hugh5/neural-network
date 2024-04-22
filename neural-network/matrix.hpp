//
//  matrix.hpp
//  neural-network
//
//  Created by Hugh Drummond on 10/4/2024.
//

#ifndef matrix_hpp
#define matrix_hpp

#include <iostream>
#include <random>
#include <vector>
#include <functional>
#include <format>
#include <iomanip>

class Matrix {
private:
    std::vector<std::vector<double>> data;
    int rows;
    int cols;
    
    static double generateBinomial();

public:
    // Constructors
    Matrix();
    Matrix(int rows, int cols);
    Matrix(int rows, int cols, double val);
    Matrix(const std::vector<std::vector<double>>& values);

    // Accessors
    int numRows() const;
    int numCols() const;
    std::string dimension() const;
    std::vector<std::vector<double>> getData() const;

    // Basic operations
    void randomize();
    void scalarAdd(double scalar);
    void scalarMultiply(double scalar);
    Matrix transpose() const;
    Matrix add(const Matrix& other) const;
    Matrix subtract(const Matrix& other) const;
    Matrix multiply(const Matrix& other) const;
    Matrix elemntWiseMultiply(const Matrix& other) const;
    void appendColum(double value);
    void appendRow(double value);
    std::vector<double> popRow();
    void map(std::function<double(double)> func);

    // Operator overloads
    double& operator()(int row, int col);
    const double& operator()(int row, int col) const;
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;

    friend std::ostream& operator<<(std::ostream& os, const Matrix& matrix);
};

#endif /* matrix_hpp */
