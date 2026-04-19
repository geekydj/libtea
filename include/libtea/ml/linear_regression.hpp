#pragma once

#include "libtea/ml/base.hpp"

namespace libtea::ml {

class LinearRegression : public Estimator {
private:
    Vector weights;

    Vector mean;
    Vector stddev;

    Matrix normalize(const Matrix& X) const;

public:
    void fit(const Matrix& X, const Vector& y) override;
    Vector predict(const Matrix& X) override;

    Vector getWeights() const;
};

}