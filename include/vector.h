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

#ifndef LINCPP_VECTOR_H
#define LINCPP_VECTOR_H

#include <vector>

namespace numcpp {

/**
 * @brief Represents a vector of elements of type T.
 *
 * @tparam T The type of elements stored in the vector.
 */
template <typename T>
class Vector {
private:
    std::vector<T> data; /**< The underlying container storing the vector elements. */

public:
    /**
     * @brief Default constructor. Creates an empty vector.
     */
    Vector();

    /**
     * @brief Constructor that creates a vector of a specified size.
     *
     * @param size The size of the vector.
     */
    Vector(size_t size);

    /**
     * @brief Constructor that creates a vector of a specified size and initializes all elements with a given value.
     *
     * @param size The size of the vector.
     * @param value The value to initialize all elements with.
     */
    Vector(size_t size, const T& value);

    /**
     * @brief Constructor that creates a vector from an initializer list or another vector.
     *
     * @param values The initializer list or vector to create the vector from.
     */
    Vector(const std::vector<T>& values);

    /**
     * @brief Returns the size of the vector.
     *
     * @return The size of the vector.
     */
    size_t size() const;

    /**
     * @brief Overloads the [] operator to access elements of the vector.
     *
     * @param index The index of the element to access.
     * @return A reference to the element at the specified index.
     */
    T& operator[](size_t index);

    /**
     * @brief Overloads the [] operator to access elements of the vector (const version).
     *
     * @param index The index of the element to access.
     * @return A const reference to the element at the specified index.
     */
    const T& operator[](size_t index) const;

    /**
     * @brief Overloads the + operator to perform vector addition.
     *
     * @param other The vector to add to the current vector.
     * @return A new vector that is the result of adding the current vector and the other vector.
     */
    Vector<T> operator+(const Vector<T>& other) const;

    /**
     * @brief Overloads the += operator to perform vector addition and assignment.
     *
     * @param other The vector to add to the current vector.
     * @return A reference to the updated current vector.
     */
    Vector<T>& operator+=(const Vector<T>& other);

    /**
     * @brief Overloads the - operator to perform vector subtraction.
     *
     * @param other The vector to subtract from the current vector.
     * @return A new vector that is the result of subtracting the other vector from the current vector.
     */
    Vector<T> operator-(const Vector<T>& other) const;

    /**
     * @brief Overloads the -= operator to perform vector subtraction and assignment.
     *
     * @param other The vector to subtract from the current vector.
     * @return A reference to the updated current vector.
     */
    Vector<T>& operator-=(const Vector<T>& other);

    /**
     * @brief Calculates the dot product of the current vector and another vector.
     *
     * @param other The vector to calculate the dot product with.
     * @return The dot product of the current vector and the other vector.
     */
    T dot(const Vector<T>& other) const;

    // Other vector operations and methods
    // ...
};

}  // namespace numcpp

#endif  // LINCPP_VECTOR_H
