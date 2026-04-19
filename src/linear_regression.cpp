#include "libtea/ml/linear_regression.hpp"
#include <cmath>

namespace libtea::ml {

// ---------------- helpers ----------------

static Vector compute_mean(const Matrix& X) {
    Vector mean(X[0].size(), 0.0);

    for (auto &row : X)
        for (size_t j = 0; j < row.size(); j++)
            mean[j] += row[j];

    for (auto &m : mean)
        m /= X.size();

    return mean;
}

static Vector compute_std(const Matrix& X, const Vector& mean) {
    Vector std(X[0].size(), 0.0);

    for (auto &row : X)
        for (size_t j = 0; j < row.size(); j++)
            std[j] += (row[j] - mean[j]) * (row[j] - mean[j]);

    for (auto &s : std)
        s = std::sqrt(s / X.size() + 1e-8);

    return std;
}

// ---------------- normalization ----------------

Matrix LinearRegression::normalize(const Matrix& X) const {
    Matrix out = X;

    for (auto &row : out)
        for (size_t j = 0; j < row.size(); j++)
            row[j] = (row[j] - mean[j]) / stddev[j];

    return out;
}

// ---------------- training ----------------

void LinearRegression::fit(const Matrix& X, const Vector& y) {
    if (X.empty()) return;

    size_t d = X[0].size();

    mean = compute_mean(X);
    stddev = compute_std(X, mean);

    Matrix Xn = normalize(X);

    weights.assign(d, 0.0);

    double lr = 0.01;
    int epochs = 1000;

    for (int e = 0; e < epochs; e++) {
        Vector grad(d, 0.0);

        for (size_t i = 0; i < Xn.size(); i++) {
            double pred = 0.0;

            for (size_t j = 0; j < d; j++)
                pred += Xn[i][j] * weights[j];

            double error = pred - y[i];

            for (size_t j = 0; j < d; j++)
                grad[j] += error * Xn[i][j];
        }

        for (size_t j = 0; j < d; j++)
            weights[j] -= lr * grad[j] / Xn.size();
    }
}

// ---------------- prediction ----------------

Vector LinearRegression::predict(const Matrix& X) {
    Matrix Xn = normalize(X);

    Vector res;
    res.reserve(X.size());

    for (auto &row : Xn) {
        double val = 0.0;

        for (size_t j = 0; j < weights.size(); j++)
            val += row[j] * weights[j];

        res.push_back(val);
    }

    return res;
}

Vector LinearRegression::getWeights() const {
    return weights;
}

}