/*
 * This file is part of libdsc.
 *
 * libdsc is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 *
 * libdsc is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * libdsc. If not, see <https://www.gnu.org/licenses/>.
 */

#include <vector.h>

#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>

namespace cramer {

template <typename T>
Vector<T>::Vector() {}

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
    if (size() != other.size()) {
        throw std::invalid_argument(
            "Vectors must have the same size for addition.");
    }

    Vector<T> result(size());
    for (size_t i = 0; i < size(); ++i) {
        result[i] = data[i] + other[i];
    }
    return result;
}

template <typename T>
Vector<T>& Vector<T>::operator+=(const Vector<T>& other) {
    if (size() != other.size()) {
        throw std::invalid_argument(
            "Vectors must have the same size for addition.");
    }

    for (size_t i = 0; i < size(); ++i) {
        data[i] += other[i];
    }
    return *this;
}

template <typename T>
Vector<T> Vector<T>::operator-(const Vector<T>& other) const {
    if (size() != other.size()) {
        throw std::invalid_argument(
            "Vectors must have the same size for subtraction.");
    }

    Vector<T> result(size());
    for (size_t i = 0; i < size(); ++i) {
        result[i] = data[i] - other[i];
    }
    return result;
}

template <typename T>
Vector<T>& Vector<T>::operator-=(const Vector<T>& other) {
    if (size() != other.size()) {
        throw std::invalid_argument(
            "Vectors must have the same size for subtraction.");
    }

    for (size_t i = 0; i < size(); ++i) {
        data[i] -= other[i];
    }
    return *this;
}

template <typename T>
Vector<T> Vector<T>::operator*(const T& scalar) const {
    Vector<T> result(size());
    for (size_t i = 0; i < size(); ++i) {
        result[i] = data[i] * scalar;
    }
    return result;
}

template <typename T>
Vector<T>& Vector<T>::operator*=(const T& scalar) {
    for (size_t i = 0; i < size(); ++i) {
        data[i] *= scalar;
    }
    return *this;
}

template <typename T>
T Vector<T>::dot(const Vector<T>& other) const {
    if (size() != other.size()) {
        throw std::invalid_argument(
            "Vectors must have the same size for dot product.");
    }

    T result = 0;
    for (size_t i = 0; i < size(); ++i) {
        result += data[i] * other[i];
    }
    return result;
}

template <typename T>
T Vector<T>::norm() const {
    return std::sqrt(dot(*this));
}

template <typename T>
Vector<T> Vector<T>::normalize() const {
    T n = norm();
    if (n == 0) {
        throw std::runtime_error("Cannot normalize a zero vector.");
    }
    return *this * (1 / n);
}

template <typename T>
Vector<T> Vector<T>::cross(const Vector<T>& other) const {
    if (size() != 3 || other.size() != 3) {
        throw std::invalid_argument("Cross product is only defined for 3D vectors.");
    }

    Vector<T> result(3);
    result[0] = data[1] * other[2] - data[2] * other[1];
    result[1] = data[2] * other[0] - data[0] * other[2];
    result[2] = data[0] * other[1] - data[1] * other[0];
    return result;
}

template <typename T>
T Vector<T>::angle(const Vector<T>& other) const {
    T dot_product = dot(other);
    T magnitudes = norm() * other.norm();
    
    if (magnitudes == 0) {
        throw std::runtime_error("Cannot calculate angle with a zero vector.");
    }
    
    return std::acos(dot_product / magnitudes);
}

template <typename T>
Vector<T> Vector<T>::project(const Vector<T>& onto) const {
    T scalar = dot(onto) / onto.dot(onto);
    return onto * scalar;
}

template <typename T>
Vector<T> Vector<T>::reject(const Vector<T>& from) const {
    return *this - project(from);
}

template <typename T>
Vector<T> Vector<T>::reflect(const Vector<T>& normal) const {
    return *this - normal * (2 * dot(normal));
}

template <typename T>
Vector<T> Vector<T>::lerp(const Vector<T>& other, T t) const {
    return *this * (1 - t) + other * t;
}

template <typename T>
bool Vector<T>::operator==(const Vector<T>& other) const {
    if (size() != other.size()) {
        return false;
    }
    for (size_t i = 0; i < size(); ++i) {
        if (data[i] != other[i]) {
            return false;
        }
    }
    return true;
}

template <typename T>
bool Vector<T>::operator!=(const Vector<T>& other) const {
    return !(*this == other);
}

template <typename T>
Vector<T> Vector<T>::elementwise_multiply(const Vector<T>& other) const {
    if (size() != other.size()) {
        throw std::invalid_argument("Vectors must have the same size for elementwise multiplication.");
    }

    Vector<T> result(size());
    for (size_t i = 0; i < size(); ++i) {
        result[i] = data[i] * other[i];
    }
    return result;
}

template <typename T>
Vector<T> Vector<T>::elementwise_divide(const Vector<T>& other) const {
    if (size() != other.size()) {
        throw std::invalid_argument("Vectors must have the same size for elementwise division.");
    }

    Vector<T> result(size());
    for (size_t i = 0; i < size(); ++i) {
        if (other[i] == 0) {
            throw std::runtime_error("Division by zero in elementwise division.");
        }
        result[i] = data[i] / other[i];
    }
    return result;
}

template <typename T>
T Vector<T>::sum() const {
    return std::accumulate(data.begin(), data.end(), T(0));
}

template <typename T>
T Vector<T>::product() const {
    return std::accumulate(data.begin(), data.end(), T(1), std::multiplies<T>());
}

template <typename T>
T Vector<T>::min() const {
    if (size() == 0) {
        throw std::runtime_error("Cannot find minimum of an empty vector.");
    }
    return *std::min_element(data.begin(), data.end());
}

template <typename T>
T Vector<T>::max() const {
    if (size() == 0) {
        throw std::runtime_error("Cannot find maximum of an empty vector.");
    }
    return *std::max_element(data.begin(), data.end());
}

template <typename T>
Vector<T> Vector<T>::abs() const {
    Vector<T> result(size());
    std::transform(data.begin(), data.end(), result.data.begin(), [](const T& x) { return std::abs(x); });
    return result;
}

template <typename T>
Vector<T> Vector<T>::pow(T exponent) const {
    Vector<T> result(size());
    std::transform(data.begin(), data.end(), result.data.begin(), 
                   [exponent](const T& x) { return std::pow(x, exponent); });
    return result;
}

template <typename T>
Vector<T> Vector<T>::sqrt() const {
    Vector<T> result(size());
    std::transform(data.begin(), data.end(), result.data.begin(), 
                   [](const T& x) { 
                       if (x < 0) throw std::runtime_error("Cannot calculate square root of negative number.");
                       return std::sqrt(x); 
                   });
    return result;
}

template <typename T>
Vector<T> Vector<T>::exp() const {
    Vector<T> result(size());
    std::transform(data.begin(), data.end(), result.data.begin(), [](const T& x) { return std::exp(x); });
    return result;
}

template <typename T>
Vector<T> Vector<T>::log() const {
    Vector<T> result(size());
    std::transform(data.begin(), data.end(), result.data.begin(), 
                   [](const T& x) { 
                       if (x <= 0) throw std::runtime_error("Cannot calculate logarithm of non-positive number.");
                       return std::log(x); 
                   });
    return result;
}

}  // namespace cramer

// Explicit template instantiations
template class cramer::Vector<float>;
template class cramer::Vector<double>;
