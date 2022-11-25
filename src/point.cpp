#include "openposert/point.hpp"

#include <ostream>

#include "openposert/macros.hpp"
#include "spdlog/spdlog.h"
#include "utils.h"

namespace openposert {

template <typename T>
Point<T>::Point(const T x_, const T y_) : x{x_}, y{y_} {}

template <typename T>
Point<T>::Point(const Point<T>& point) {
  x = point.x;
  y = point.y;
}

template <typename T>
Point<T>& Point<T>::operator=(const Point<T>& point) {
  x = point.x;
  y = point.y;
  // Return
  return *this;
}

template <typename T>
Point<T>::Point(Point<T>&& point) {
  x = point.x;
  y = point.y;
}

template <typename T>
Point<T>& Point<T>::operator=(Point<T>&& point) {
  x = point.x;
  y = point.y;
  // Return
  return *this;
}

template <typename T>
std::string Point<T>::toString() const {
  return '[' + std::to_string(x) + ", " + std::to_string(y) + ']';
}

template <typename T>
Point<T>& Point<T>::operator+=(const Point<T>& point) {
  x += point.x;
  y += point.y;
  // Return
  return *this;
}

template <typename T>
Point<T> Point<T>::operator+(const Point<T>& point) const {
  return Point<T>{T(x + point.x), T(y + point.y)};
}

template <typename T>
Point<T>& Point<T>::operator+=(const T value) {
  x += value;
  y += value;
  // Return
  return *this;
}

template <typename T>
Point<T> Point<T>::operator+(const T value) const {
  return Point<T>{T(x + value), T(y + value)};
}

template <typename T>
Point<T>& Point<T>::operator-=(const Point<T>& point) {
  x -= point.x;
  y -= point.y;
  // Return
  return *this;
}

template <typename T>
Point<T> Point<T>::operator-(const Point<T>& point) const {
  return Point<T>{T(x - point.x), T(y - point.y)};
}

template <typename T>
Point<T>& Point<T>::operator-=(const T value) {
  x -= value;
  y -= value;
  // Return
  return *this;
}

template <typename T>
Point<T> Point<T>::operator-(const T value) const {
  return Point<T>{T(x - value), T(y - value)};
}

template <typename T>
Point<T>& Point<T>::operator*=(const T value) {
  x *= value;
  y *= value;
  // Return
  return *this;
}

template <typename T>
Point<T> Point<T>::operator*(const T value) const {
  return Point<T>{T(x * value), T(y * value)};
}

template <typename T>
Point<T>& Point<T>::operator/=(const T value) {
  x /= value;
  y /= value;
  // Return
  return *this;
}

template <typename T>
Point<T> Point<T>::operator/(const T value) const {
  return Point<T>{T(x / value), T(y / value)};
}

COMPILE_TEMPLATE_BASIC_TYPES_STRUCT(Point);
}  // namespace openposert
