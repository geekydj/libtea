#include "libtea/ml/linear_regression.hpp"
#include <stdexcept>

namespace libtea::ml {

LinearRegression::LinearRegression(double learning_rate, int iterations)
    : lr(learning_rate), epochs(iterations), bias(0.0) {}

double LinearRegression::dot(const Vector& a, const Vector& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++)
        sum += a[i] * b[i];
    return sum;
}

void LinearRegression::fit(const Matrix& X, const Vector& y) {

    if (X.empty()) return;

    int n_samples = X.size();
    int n_features = X[0].size();

    weights.assign(n_features, 0.0);
    bias = 0.0;

    for (int epoch = 0; epoch < epochs; epoch++) {

        Vector dw(n_features, 0.0);
        double db = 0.0;

        for (int i = 0; i < n_samples; i++) {

            double pred = dot(X[i], weights) + bias;
            double error = pred - y[i];

            for (int j = 0; j < n_features; j++) {
                dw[j] += error * X[i][j];
            }

            db += error;
        }

        for (int j = 0; j < n_features; j++) {
            weights[j] -= lr * (dw[j] / n_samples);
        }

        bias -= lr * (db / n_samples);
    }
}

Vector LinearRegression::predict(const Matrix& X) {

    Vector result;

    for (const auto& row : X) {
        double pred = dot(row, weights) + bias;
        result.push_back(pred);
    }

    return result;
}

}