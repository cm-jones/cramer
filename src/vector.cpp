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

#include "../include/vector.h"

#include <cmath>
#include <stdexcept>

namespace numcpp {

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

}  // namespace numcpp
