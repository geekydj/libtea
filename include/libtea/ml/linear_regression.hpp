#pragma once
#include <vector>

namespace libtea::ml {

using Vector = std::vector<double>;
using Matrix = std::vector<std::vector<double>>;

class LinearRegression {
private:
    Vector weights;
    double bias;

    double lr;
    int epochs;

public:
    LinearRegression(double learning_rate = 0.01, int iterations = 1000);

    void fit(const Matrix& X, const Vector& y);
    Vector predict(const Matrix& X);

private:
    double dot(const Vector& a, const Vector& b);
};

}