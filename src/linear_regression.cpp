#include "libtea/ml/linear_regression.hpp"
#include <cmath>

namespace libtea::ml {

double LinearRegression::dot(const Vector& a, const Vector& b) {
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); i++)
        sum += a[i] * b[i];
    return sum;
}

void LinearRegression::fit(const Matrix& X, const Vector& y) {
    if (X.empty()) return;

    int n = X.size();
    int d = X[0].size();

    mean.assign(d, 0.0);
    stddev.assign(d, 0.0);

    for (auto& row : X)
        for (int j = 0; j < d; j++)
            mean[j] += row[j];
    for (int j = 0; j < d; j++)
        mean[j] /= n;

    for (auto& row : X)
        for (int j = 0; j < d; j++)
            stddev[j] += (row[j] - mean[j]) * (row[j] - mean[j]);
    for (int j = 0; j < d; j++)
        stddev[j] = std::sqrt(stddev[j] / n + 1e-8);

    Matrix Xn = X;
    for (auto& row : Xn)
        for (int j = 0; j < d; j++)
            row[j] = (row[j] - mean[j]) / stddev[j];

    weights.assign(d, 0.0);
    bias = 0.0;

    for (int epoch = 0; epoch < epochs; epoch++) {
        Vector dw(d, 0.0);
        double db = 0.0;

        for (int i = 0; i < n; i++) {
            double pred = dot(Xn[i], weights) + bias;
            double error = pred - y[i];

            for (int j = 0; j < d; j++)
                dw[j] += error * Xn[i][j];
            db += error;
        }

        for (int j = 0; j < d; j++)
            weights[j] -= lr * (dw[j] / n);
        bias -= lr * (db / n);
    }
}

Vector LinearRegression::predict(const Matrix& X) {
    Matrix Xn = X;
    for (auto& row : Xn)
        for (size_t j = 0; j < weights.size(); j++)
            row[j] = (row[j] - mean[j]) / stddev[j];

    Vector result;
    for (auto& row : Xn)
        result.push_back(dot(row, weights) + bias);

    return result;
}

Vector LinearRegression::getWeights() const {
    return weights;
}

} // namespace libtea::ml