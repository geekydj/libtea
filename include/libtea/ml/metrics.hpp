#pragma once

#include "libtea/ml/base.hpp"

namespace libtea::ml {

double mae(const Vector& y_true, const Vector& y_pred);
double mse(const Vector& y_true, const Vector& y_pred);

}