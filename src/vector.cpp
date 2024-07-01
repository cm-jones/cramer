// SPDX-License-Identifier: GPL-3.0-or-later

#include "vector.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace cramer {

template <typename T>
Vector<T>::Vector() : data() {}

template <typename T>
Vector<T>::Vector(size_t size) : data(size) {}

template <typename T>
Vector<T>::Vector(size_t size, const T& value) : data(size, value) {}

template <typename T>
Vector<T>::Vector(const std::vector<T>& values) : data(values) {}

template <typename T>
size_t Vector<T>::size() const {
    return data.size();
}

template <typename T>
T& Vector<T>::operator[](size_t index) {
    return data[index];
}

template <typename T>
const T& Vector<T>::operator[](size_t index) const {
    return data[index];
}

template <typename T>
Vector<T> Vector<T>::operator+(const Vector<T>& other) const {
    if (this->size() != other.size()) {
        throw std::invalid_argument(
            "Vectors must have the same size for addition");
    }
    Vector<T> result(this->size());
    for (size_t i = 0; i < this->size(); ++i) {
        result[i] = data[i] + other[i];
    }
    return result;
}

template <typename T>
Vector<T>& Vector<T>::operator+=(const Vector<T>& other) {
    if (this->size() != other.size()) {
        throw std::invalid_argument(
            "Vectors must have the same size for addition");
    }
    for (size_t i = 0; i < this->size(); ++i) {
        data[i] += other[i];
    }
    return *this;
}

template <typename T>
Vector<T> Vector<T>::operator-(const Vector<T>& other) const {
    if (this->size() != other.size()) {
        throw std::invalid_argument(
            "Vectors must have the same size for subtraction");
    }
    Vector<T> result(this->size());
    for (size_t i = 0; i < this->size(); ++i) {
        result[i] = data[i] - other[i];
    }
    return result;
}

template <typename T>
Vector<T>& Vector<T>::operator-=(const Vector<T>& other) {
    if (this->size() != other.size()) {
        throw std::invalid_argument(
            "Vectors must have the same size for subtraction");
    }
    for (size_t i = 0; i < this->size(); ++i) {
        data[i] -= other[i];
    }
    return *this;
}

template <typename T>
Vector<T> Vector<T>::operator*(const T& scalar) const {
    Vector<T> result(this->size());
    for (size_t i = 0; i < this->size(); ++i) {
        result[i] = data[i] * scalar;
    }
    return result;
}

template <typename T>
Vector<T>& Vector<T>::operator*=(const T& scalar) {
    for (size_t i = 0; i < this->size(); ++i) {
        data[i] *= scalar;
    }
    return *this;
}

template <typename T>
T Vector<T>::dot(const Vector<T>& other) const {
    if (this->size() != other.size()) {
        throw std::invalid_argument(
            "Vectors must have the same size for dot product");
    }
    return std::inner_product(data.begin(), data.end(), other.data.begin(),
                              T());
}

template <typename T>
T Vector<T>::norm() const {
    return std::sqrt(
        std::inner_product(data.begin(), data.end(), data.begin(), T()));
}

template <typename T>
Vector<T> Vector<T>::normalize() const {
    T magnitude = this->norm();
    if (magnitude == T()) {
        throw std::runtime_error("Cannot normalize zero vector");
    }
    return *this * (T(1) / magnitude);
}

template <typename T>
Vector<T> Vector<T>::cross(const Vector<T>& other) const {
    if (this->size() != 3 || other.size() != 3) {
        throw std::invalid_argument(
            "Cross product is only defined for 3D vectors");
    }
    Vector<T> result(3);
    result[0] = data[1] * other[2] - data[2] * other[1];
    result[1] = data[2] * other[0] - data[0] * other[2];
    result[2] = data[0] * other[1] - data[1] * other[0];
    return result;
}

template <typename T>
T Vector<T>::angle(const Vector<T>& other) const {
    T magnitude1 = this->norm();
    T magnitude2 = other.norm();
    if (magnitude1 == T() || magnitude2 == T()) {
        throw std::runtime_error("Cannot compute angle with zero vector");
    }
    return std::acos(dot(other) / (magnitude1 * magnitude2));
}

template <typename T>
Vector<T> Vector<T>::project(const Vector<T>& onto) const {
    T scaleFactor = dot(onto) / onto.dot(onto);
    return onto * scaleFactor;
}

template <typename T>
Vector<T> Vector<T>::reject(const Vector<T>& from) const {
    return *this - project(from);
}

template <typename T>
Vector<T> Vector<T>::reflect(const Vector<T>& normal) const {
    return *this - normal * (T(2) * dot(normal));
}

template <typename T>
bool Vector<T>::operator==(const Vector<T>& other) const {
    return data == other.data;
}

template <typename T>
bool Vector<T>::operator!=(const Vector<T>& other) const {
    return !(*this == other);
}

template <typename T>
Vector<T> Vector<T>::elementwise_multiply(const Vector<T>& other) const {
    if (this->size() != other.size()) {
        throw std::invalid_argument(
            "Vectors must have the same size for element-wise multiplication");
    }
    Vector<T> result(this->size());
    for (size_t i = 0; i < this->size(); ++i) {
        result[i] = data[i] * other[i];
    }
    return result;
}

template <typename T>
Vector<T> Vector<T>::elementwise_divide(const Vector<T>& other) const {
    if (this->size() != other.size()) {
        throw std::invalid_argument(
            "Vectors must have the same size for element-wise division");
    }
    Vector<T> result(this->size());
    for (size_t i = 0; i < this->size(); ++i) {
        if (other[i] == T()) {
            throw std::runtime_error(
                "Division by zero in element-wise division");
        }
        result[i] = data[i] / other[i];
    }
    return result;
}

template <typename T>
T Vector<T>::sum() const {
    return std::accumulate(data.begin(), data.end(), T());
}

template <typename T>
T Vector<T>::product() const {
    return std::accumulate(data.begin(), data.end(), T(1),
                           std::multiplies<T>());
}

// Specialization for complex types
template <typename T>
struct is_complex : std::false_type {};

template <typename T>
struct is_complex<std::complex<T>> : std::true_type {};

template <typename T>
T Vector<T>::min() const {
    if (data.empty()) {
        throw std::runtime_error("Cannot find minimum of empty vector");
    }
    if constexpr (is_complex<T>::value) {
        return *std::min_element(
            data.begin(), data.end(),
            [](const T& a, const T& b) { return std::abs(a) < std::abs(b); });
    } else {
        return *std::min_element(data.begin(), data.end());
    }
}

template <typename T>
T Vector<T>::max() const {
    if (data.empty()) {
        throw std::runtime_error("Cannot find maximum of empty vector");
    }
    if constexpr (is_complex<T>::value) {
        return *std::max_element(
            data.begin(), data.end(),
            [](const T& a, const T& b) { return std::abs(a) < std::abs(b); });
    } else {
        return *std::max_element(data.begin(), data.end());
    }
}

template <typename T>
Vector<T> Vector<T>::abs() const {
    Vector<T> result(this->size());
    std::transform(data.begin(), data.end(), result.data.begin(),
                   [](const T& x) { return std::abs(x); });
    return result;
}

template <typename T>
Vector<T> Vector<T>::pow(T exponent) const {
    Vector<T> result(this->size());
    std::transform(data.begin(), data.end(), result.data.begin(),
                   [exponent](const T& x) { return std::pow(x, exponent); });
    return result;
}

template <typename T>
Vector<T> Vector<T>::sqrt() const {
    Vector<T> result(this->size());
    for (size_t i = 0; i < this->size(); ++i) {
        if constexpr (is_complex<T>::value) {
            result[i] = std::sqrt(data[i]);
        } else {
            if (data[i] < T()) {
                throw std::runtime_error(
                    "Cannot compute square root of negative number");
            }
            result[i] = std::sqrt(data[i]);
        }
    }
    return result;
}

template <typename T>
Vector<T> Vector<T>::exp() const {
    Vector<T> result(this->size());
    std::transform(data.begin(), data.end(), result.data.begin(),
                   [](const T& x) { return std::exp(x); });
    return result;
}

template <typename T>
Vector<T> Vector<T>::log() const {
    Vector<T> result(this->size());
    for (size_t i = 0; i < this->size(); ++i) {
        if constexpr (is_complex<T>::value) {
            result[i] = std::log(data[i]);
        } else {
            if (data[i] <= T()) {
                throw std::runtime_error(
                    "Cannot compute logarithm of non-positive number");
            }
            result[i] = std::log(data[i]);
        }
    }
    return result;
}

// Explicit instantiations
template class Vector<int>;
template class Vector<float>;
template class Vector<double>;
template class Vector<std::complex<float>>;
template class Vector<std::complex<double>>;

}  // namespace cramer
