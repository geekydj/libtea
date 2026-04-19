#pragma once
#include "libtea/ml/base.hpp"

namespace libtea::ml {

class LinearRegression : public Estimator {
private:
    Vector weights;
    double bias = 0.0;
    double lr;
    int epochs;

    Vector mean;
    Vector stddev;

    double dot(const Vector& a, const Vector& b);

public:
    LinearRegression(double learning_rate = 0.01, int iterations = 1000)
        : lr(learning_rate), epochs(iterations), bias(0.0) {}

    void fit(const Matrix& X, const Vector& y) override;
    Vector predict(const Matrix& X) override;
    Vector getWeights() const;
};

} // namespace libtea::ml