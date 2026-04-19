#include "libtea/ml/metrics.hpp"
#include <cmath>

namespace libtea::ml {

double mae(const Vector& y_true, const Vector& y_pred) {
    double sum = 0.0;

    for (size_t i = 0; i < y_true.size(); i++)
        sum += std::abs(y_true[i] - y_pred[i]);

    return sum / y_true.size();
}

double mse(const Vector& y_true, const Vector& y_pred) {
    double sum = 0.0;

    for (size_t i = 0; i < y_true.size(); i++) {
        double d = y_true[i] - y_pred[i];
        sum += d * d;
    }

    return sum / y_true.size();
}

}