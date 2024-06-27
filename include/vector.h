// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <vector>

namespace cramer {

/**
 * @brief Represents a vector of elements of type T.
 *
 * @tparam T The type of elements stored in the vector.
 */
template <typename T>
class Vector {
   private:
    std::vector<T>
        data; /**< The underlying container storing the vector elements. */

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
     * @brief Constructor that creates a vector of a specified size and
     * initializes all elements with a given value.
     *
     * @param size The size of the vector.
     * @param value The value to initialize all elements with.
     */
    Vector(size_t size, const T& value);

    /**
     * @brief Constructor that creates a vector from an initializer list or
     * another vector.
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
     * @brief Overloads the [] operator to access elements of the vector (const
     * version).
     *
     * @param index The index of the element to access.
     * @return A const reference to the element at the specified index.
     */
    const T& operator[](size_t index) const;

    /**
     * @brief Overloads the + operator to perform vector addition.
     *
     * @param other The vector to add to the current vector.
     * @return A new vector that is the result of adding the current vector and
     * the other vector.
     */
    Vector<T> operator+(const Vector<T>& other) const;

    /**
     * @brief Overloads the += operator to perform vector addition and
     * assignment.
     *
     * @param other The vector to add to the current vector.
     * @return A reference to the updated current vector.
     */
    Vector<T>& operator+=(const Vector<T>& other);

    /**
     * @brief Overloads the - operator to perform vector subtraction.
     *
     * @param other The vector to subtract from the current vector.
     * @return A new vector that is the result of subtracting the other vector
     * from the current vector.
     */
    Vector<T> operator-(const Vector<T>& other) const;

    /**
     * @brief Overloads the -= operator to perform vector subtraction and
     * assignment.
     *
     * @param other The vector to subtract from the current vector.
     * @return A reference to the updated current vector.
     */
    Vector<T>& operator-=(const Vector<T>& other);

    /**
     * @brief Overloads the * operator to perform scalar multiplication.
     *
     * @param scalar The scalar to multiply the vector by.
     * @return A new vector that is the result of multiplying the current vector by the scalar.
     */
    Vector<T> operator*(const T& scalar) const;

    /**
     * @brief Overloads the *= operator to perform scalar multiplication and assignment.
     *
     * @param scalar The scalar to multiply the vector by.
     * @return A reference to the updated current vector.
     */
    Vector<T>& operator*=(const T& scalar);

    /**
     * @brief Calculates the dot product of the current vector and another
     * vector.
     *
     * @param other The vector to calculate the dot product with.
     * @return The dot product of the current vector and the other vector.
     */
    T dot(const Vector<T>& other) const;

    /**
     * @brief Calculates the Euclidean norm (magnitude) of the vector.
     *
     * @return The Euclidean norm of the vector.
     */
    T norm() const;

    /**
     * @brief Returns a normalized version of the vector.
     *
     * @return A new vector that is the normalized version of the current vector.
     * @throw std::runtime_error if the vector is a zero vector.
     */
    Vector<T> normalize() const;

    /**
     * @brief Calculates the cross product of the current vector and another vector.
     *
     * @param other The vector to calculate the cross product with.
     * @return A new vector that is the cross product of the current vector and the other vector.
     * @throw std::invalid_argument if either vector is not 3-dimensional.
     */
    Vector<T> cross(const Vector<T>& other) const;

    /**
     * @brief Calculates the angle between the current vector and another vector.
     *
     * @param other The vector to calculate the angle with.
     * @return The angle between the vectors in radians.
     * @throw std::runtime_error if either vector is a zero vector.
     */
    T angle(const Vector<T>& other) const;

    /**
     * @brief Projects the current vector onto another vector.
     *
     * @param onto The vector to project onto.
     * @return A new vector that is the projection of the current vector onto the other vector.
     */
    Vector<T> project(const Vector<T>& onto) const;

    /**
     * @brief Calculates the rejection of the current vector from another vector.
     *
     * @param from The vector to reject from.
     * @return A new vector that is the rejection of the current vector from the other vector.
     */
    Vector<T> reject(const Vector<T>& from) const;

    /**
     * @brief Reflects the current vector about a normal vector.
     *
     * @param normal The normal vector to reflect about.
     * @return A new vector that is the reflection of the current vector.
     */
    Vector<T> reflect(const Vector<T>& normal) const;

    /**
     * @brief Overloads the == operator to check for vector equality.
     *
     * @param other The vector to compare with.
     * @return true if the vectors are equal, false otherwise.
     */
    bool operator==(const Vector<T>& other) const;

    /**
     * @brief Overloads the != operator to check for vector inequality.
     *
     * @param other The vector to compare with.
     * @return true if the vectors are not equal, false otherwise.
     */
    bool operator!=(const Vector<T>& other) const;

    /**
     * @brief Performs element-wise multiplication of two vectors.
     *
     * @param other The vector to multiply element-wise with.
     * @return A new vector that is the result of element-wise multiplication.
     * @throw std::invalid_argument if the vectors have different sizes.
     */
    Vector<T> elementwise_multiply(const Vector<T>& other) const;

    /**
     * @brief Performs element-wise division of two vectors.
     *
     * @param other The vector to divide element-wise by.
     * @return A new vector that is the result of element-wise division.
     * @throw std::invalid_argument if the vectors have different sizes.
     * @throw std::runtime_error if division by zero occurs.
     */
    Vector<T> elementwise_divide(const Vector<T>& other) const;

    /**
     * @brief Calculates the sum of all elements in the vector.
     *
     * @return The sum of all elements.
     */
    T sum() const;

    /**
     * @brief Calculates the product of all elements in the vector.
     *
     * @return The product of all elements.
     */
    T product() const;

    /**
     * @brief Finds the minimum element in the vector.
     *
     * @return The minimum element.
     * @throw std::runtime_error if the vector is empty.
     */
    T min() const;

    /**
     * @brief Finds the maximum element in the vector.
     *
     * @return The maximum element.
     * @throw std::runtime_error if the vector is empty.
     */
    T max() const;

    /**
     * @brief Computes the absolute value of each element in the vector.
     *
     * @return A new vector with the absolute values of the current vector's elements.
     */
    Vector<T> abs() const;

    /**
     * @brief Raises each element of the vector to the specified power.
     *
     * @param exponent The power to raise each element to.
     * @return A new vector with each element raised to the specified power.
     */
    Vector<T> pow(T exponent) const;

    /**
     * @brief Computes the square root of each element in the vector.
     *
     * @return A new vector with the square root of each element of the current vector.
     * @throw std::runtime_error if any element is negative.
     */
    Vector<T> sqrt() const;

    /**
     * @brief Computes the exponential of each element in the vector.
     *
     * @return A new vector with the exponential of each element of the current vector.
     */
    Vector<T> exp() const;

    /**
     * @brief Computes the natural logarithm of each element in the vector.
     *
     * @return A new vector with the natural logarithm of each element of the current vector.
     * @throw std::runtime_error if any element is non-positive.
     */
    Vector<T> log() const;
};

}  // namespace cramer
