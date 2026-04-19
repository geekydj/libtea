/*
    Read this website for how we are using namespace here 
    https://www.learncpp.com/cpp-tutorial/user-defined-namespaces-and-the-scope-resolution-operator/
*/

#pragma once
#include <vector>

namespace libtea::ml{
    
    // The Vector and Matrix are Aliases names of the std::vector<double> and std::vector<std::vector<double>>

    // The Vector here is a variable name which we are going to be using in libtea::ml for creating an object vector which takes value of double
    using Vector = std::vector<double>;
    
    // This Matrix variable is a vectorized element which stores other vectorized elements
    using Matrix = std::vector<std::vector<double>>;

    class Estimator{
        public :
            virtual void fit(const Matrix& X , const Vector& y) = 0 ;
            virtual Vector predict(const Matrix& X) = 0;
            virtual ~Estimator() = default; 
                // Read https://www.learncpp.com/cpp-tutorial/virtual-destructors-virtual-assignment-and-overriding-virtualization/
    };

}

