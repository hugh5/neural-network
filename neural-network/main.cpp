//
//  main.cpp
//  neural-network
//
//  Created by Hugh Drummond on 10/4/2024.
//

#include <iostream>
#include <math.h>
#include "matrix.hpp"

int main(int argc, const char * argv[]) {
    Matrix m1(1, 3, 5);
    Matrix m2(1, 3, 7);
    m2 = m2.transpose();
    m1.multiply(m2);

    std::cout << m1 << std::endl;

    return 0;
}
