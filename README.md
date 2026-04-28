# libtea

`libtea` is a C++17 machine learning library with a small, readable API.
It currently provides a full regression workflow:

1. split data into train and test sets,
2. train a linear regression model,
3. generate predictions,
4. evaluate results with common regression metrics.

This README is written as a practical guide so you can understand what each part does and how to use or extend it.

## Table of Contents

- [What libtea Includes](#what-libtea-includes)
- [How the Library Is Organized](#how-the-library-is-organized)
- [Requirements](#requirements)
- [Build and Run](#build-and-run)
- [Install](#install)
- [Usage Walkthrough](#usage-walkthrough)
- [API Reference](#api-reference)
- [How Training Works Internally](#how-training-works-internally)
- [Data Assumptions and Notes](#data-assumptions-and-notes)
- [UML Documentation](#uml-documentation)
- [Project Structure](#project-structure)
- [Extending libtea](#extending-libtea)

## What libtea Includes

`libtea::ml` currently contains:

- **Core data aliases**
  - `Vector` for 1D numeric data (`std::vector<double>`)
  - `Matrix` for 2D numeric data (`std::vector<std::vector<double>>`)
- **Estimator interface**
  - Shared contract for ML models (`fit` and `predict`)
- **LinearRegression model**
  - Learns weights and bias with gradient descent
  - Stores normalization statistics from training
- **Data split utility**
  - `train_test_split` with configurable test ratio and random seed
- **Regression metrics**
  - `mae`, `mse`, `r2_score`

## How the Library Is Organized

The project follows a simple C++ library layout:

- Public headers in `include/libtea/ml/`
- Model implementation in `src/`
- Runnable example in `examples/`
- Build/install/package configuration in `CMakeLists.txt` and `cmake/`
- UML source and rendered diagrams in `uml/`

The build system exports CMake package files so downstream projects can consume `libtea` with `find_package`.

## Requirements

- CMake `3.14` or newer
- A C++17-compatible compiler

## Build and Run

From repository root:

```bash
cmake -S . -B build
cmake --build build
./build/example
```

On Windows with Visual Studio generator, run the executable from the generated build output directory (for example `build/Debug/example.exe`).

## Install

To install library targets, headers, and package config files:

```bash
cmake -S . -B build
cmake --build build
cmake --install build
```

Install exports include:

- `libteaTargets.cmake`
- `libteaConfig.cmake`
- `libteaConfigVersion.cmake`

## Usage Walkthrough

A complete usage example is available in `examples/basic_usage.cpp`.
Typical flow:

1. Define input features `X` and target values `y`.
2. Call `train_test_split(X, y, test_size, random_seed)`.
3. Create `LinearRegression model(learning_rate, iterations)`.
4. Train with `model.fit(split.X_train, split.y_train)`.
5. Predict with `model.predict(split.X_test)`.
6. Evaluate using:
   - `mae(split.y_test, y_pred)`
   - `mse(split.y_test, y_pred)`
   - `r2_score(split.y_test, y_pred)`

## API Reference

### Namespace

- `libtea::ml`

### Core Types

- `using Vector = std::vector<double>;`
- `using Matrix = std::vector<std::vector<double>>;`

### Estimator Interface

```cpp
class Estimator {
public:
    virtual void fit(const Matrix& X, const Vector& y) = 0;
    virtual Vector predict(const Matrix& X) = 0;
    virtual ~Estimator() = default;
};
```

### LinearRegression

```cpp
class LinearRegression : public Estimator {
public:
    LinearRegression(double learning_rate = 0.01, int iterations = 1000);
    void fit(const Matrix& X, const Vector& y) override;
    Vector predict(const Matrix& X) override;
    Vector getWeights() const;
};
```

Constructor parameters:

- `learning_rate`: gradient descent step size
- `iterations`: number of training epochs

### Data Splitting

```cpp
struct SplitData {
    Matrix X_train, X_test;
    Vector y_train, y_test;
};

SplitData train_test_split(
    const Matrix& X,
    const Vector& y,
    double test_size = 0.2,
    int random_seed = 42
);
```

### Metrics

- `double mae(const Vector& y_true, const Vector& y_pred);`
- `double mse(const Vector& y_true, const Vector& y_pred);`
- `double r2_score(const Vector& y_true, const Vector& y_pred);`

## How Training Works Internally

`LinearRegression::fit` performs these steps:

1. Return early if input matrix is empty.
2. Compute per-feature mean.
3. Compute per-feature standard deviation.
4. Normalize training features.
5. Initialize weights and bias to zero.
6. For each epoch:
   - compute prediction error for each sample,
   - accumulate gradients (`dw`, `db`),
   - update weights and bias using learning rate.

`LinearRegression::predict`:

1. Normalizes incoming features using stored training mean/stddev.
2. Computes output using `dot(features, weights) + bias`.

Important behavior: normalization statistics are learned in `fit` and reused in `predict`.

## Data Assumptions and Notes

- Inputs are numeric (`double`) and dense.
- All rows in `Matrix` should have the same number of columns.
- `fit` must be called before `predict`.
- Metrics expect vectors of matching lengths.
- `train_test_split` shuffles indices using `std::mt19937` and the provided seed.

## UML Documentation

UML source files are in `uml/`:

- `package_overview.puml`
- `class_diagram.puml`
- `sequence_basic_usage.puml`
- `activity_linear_regression_fit.puml`
- `activity_linear_regression_predict.puml`

Rendered diagrams are in `uml/out/`.

To regenerate all SVG diagrams:

```bash
java -jar tools/plantuml.jar -tsvg -o out uml/*.puml
```

## Project Structure

```text
libtea/
  include/libtea/ml/
    base.hpp
    linear_regression.hpp
    metrics.hpp
    train_test_split.hpp
  src/
    linear_regression.cpp
  examples/
    basic_usage.cpp
  cmake/
    libteaConfig.cmake.in
  uml/
    *.puml
    out/*.svg
  CMakeLists.txt
  um-theme.puml
```

## Extending libtea

To add a new model:

1. Create a new class in `include/libtea/ml/` implementing `Estimator`.
2. Add implementation in `src/`.
3. Register the source file in `CMakeLists.txt`.
4. Add a minimal example under `examples/`.
5. Add/update UML diagrams to reflect the new architecture.
