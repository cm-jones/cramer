# `vector.h`

## Vector Class

The `Vector` class represents a vector of elements of type `T`. It provides various constructors, operators, and member functions for vector manipulation and computation.

### Constructors

- `Vector()`: Default constructor. Creates an empty vector.
- `Vector(size_t size)`: Creates a vector of a specified size.
- `Vector(size_t size, const T& value)`: Creates a vector of a specified size and initializes all elements with a given value.
- `Vector(const std::vector<T>& values)`: Creates a vector from an initializer list or another vector.

### Member Functions

- `size_t size() const`: Returns the size of the vector.
- `T& operator[](size_t index)`: Overloads the `[]` operator to access elements of the vector.
- `const T& operator[](size_t index) const`: Overloads the `[]` operator to access elements of the vector (const version).
- `Vector<T> operator+(const Vector<T>& other) const`: Overloads the `+` operator to perform vector addition.
- `Vector<T>& operator+=(const Vector<T>& other)`: Overloads the `+=` operator to perform vector addition and assignment.
- `Vector<T> operator-(const Vector<T>& other) const`: Overloads the `-` operator to perform vector subtraction.
- `Vector<T>& operator-=(const Vector<T>& other)`: Overloads the `-=` operator to perform vector subtraction and assignment.
- `T dot(const Vector<T>& other) const`: Calculates the dot product of the current vector and another vector.
