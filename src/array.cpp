#include "openposert/array.hpp"

#include <numeric>   // std::accumulate
#include <typeinfo>  // typeid

#include "openposert/macros.hpp"
#include "spdlog/spdlog.h"
#include "utils.h"

namespace openposert {

// Note: std::shared_ptr not (fully) supported for array pointers:
// http://stackoverflow.com/questions/8947579/
// Solutions:
// 1) Using boost::shared_ptr from <boost/shared_ptr.hpp>: Very easy but
// requires Boost. 2) Using std::unique_ptr from <memory>: Same behaviour than
// 1, but only `unique`. 3) Using std::shared_ptr from <memory>: Harder to use,
// but benefits of 1 & 2. Solutions to its problems:
//     a) Accessing elements:
//        https://stackoverflow.com/questions/30780262/accessing-array-of-shared-ptr
//     b) Default delete:
//        https://stackoverflow.com/questions/13061979/shared-ptr-to-an-array-should-it-be-used

template <typename T>
Array<T>::Array(const int size) {
  reset(size);
}

template <typename T>
Array<T>::Array(const std::vector<int>& sizes) {
  reset(sizes);
}

template <typename T>
Array<T>::Array(const int size, const T value) {
  reset(size, value);
}

template <typename T>
Array<T>::Array(const std::vector<int>& sizes, const T value) {
  reset(sizes, value);
}

template <typename T>
Array<T>::Array(const int size, T* const dataPtr) {
  if (size > 0)
    resetAuxiliary(std::vector<int>{size}, dataPtr);
  else
    spdlog::error("Size cannot be less than 1.", __LINE__, __FUNCTION__,
                  __FILE__);
}

template <typename T>
Array<T>::Array(const std::vector<int>& sizes, T* const dataPtr) {
  if (!sizes.empty())
    resetAuxiliary(sizes, dataPtr);
  else
    spdlog::error("Size cannot be empty or less than 1.", __LINE__,
                  __FUNCTION__, __FILE__);
}

template <typename T>
Array<T>::Array(const Array<T>& array)
    : mSize{array.mSize},
      mVolume{array.mVolume},
      spData{array.spData},
      pData{array.pData} {}

template <typename T>
Array<T>& Array<T>::operator=(const Array<T>& array) {
  mSize = array.mSize;
  mVolume = array.mVolume;
  spData = array.spData;
  pData = array.pData;
  // Return
  return *this;
}

template <typename T>
Array<T>::Array(Array<T>&& array) : mSize{array.mSize}, mVolume{array.mVolume} {
  std::swap(spData, array.spData);
  std::swap(pData, array.pData);
}

template <typename T>
Array<T>& Array<T>::operator=(Array<T>&& array) {
  mSize = array.mSize;
  mVolume = array.mVolume;
  std::swap(spData, array.spData);
  std::swap(pData, array.pData);
  // Return
  return *this;
}

template <typename T>
Array<T> Array<T>::clone() const {
  // Constructor
  Array<T> array{mSize};
  // Clone data
  // Equivalent: std::copy(spData.get(), spData.get() + mVolume,
  // array.spData.get());
  std::copy(pData, pData + mVolume, array.pData);
  // Return
  return std::move(array);
}

template <typename T>
void Array<T>::reset(const int size) {
  if (size > 0)
    reset(std::vector<int>{size});
  else
    reset(std::vector<int>{});
}

template <typename T>
void Array<T>::reset(const std::vector<int>& sizes) {
  resetAuxiliary(sizes);
}

template <typename T>
void Array<T>::reset(const int sizes, const T value) {
  reset(sizes);
  setTo(value);
}

template <typename T>
void Array<T>::reset(const std::vector<int>& sizes, const T value) {
  reset(sizes);
  setTo(value);
}

template <typename T>
void Array<T>::setTo(const T value) {
  if (mVolume > 0) {
    for (auto i = 0u; i < mVolume; ++i) operator[](i) = value;
  }
}

template <typename T>
int Array<T>::getSize(const int index) const {
  // Matlab style:
  // If empty -> return 0
  // If index >= # dimensions -> return 1
  if ((unsigned int)index < mSize.size() && 0 <= index) return mSize[index];
  // Long version:
  // else if (mSize.empty())
  //     return 0;
  // else // if mSize.size() <= (unsigned int)index
  //     return 1;
  // Equivalent to:
  else
    return (!mSize.empty());
}

template <typename T>
std::string Array<T>::printSize() const {
  auto counter = 0u;
  std::string sizeString = "[ ";
  for (const auto& i : mSize) {
    sizeString += std::to_string(i);
    if (++counter < mSize.size()) sizeString += " x ";
  }
  sizeString += " ]";
  return sizeString;
}

template <typename T>
size_t Array<T>::getVolume(const int indexA, const int indexB) const {
  if (indexA < indexB) {
    if (0 <= indexA && (unsigned int)indexB <
                           mSize.size())  // 0 <= indexA < indexB < mSize.size()
      return std::accumulate(mSize.begin() + indexA, mSize.begin() + indexB + 1,
                             1ul, std::multiplies<size_t>());
    else {
      spdlog::error("Indexes out of dimension.", __LINE__, __FUNCTION__,
                    __FILE__);
      return 0;
    }
  } else if (indexA == indexB)
    return mSize.at(indexA);
  else  // if (indexA > indexB)
  {
    spdlog::error("indexA > indexB.", __LINE__, __FUNCTION__, __FILE__);
    return 0;
  }
}

template <typename T>
std::vector<int> Array<T>::getStride() const {
  std::vector<int> strides(mSize.size());
  if (!strides.empty()) {
    strides.back() = sizeof(T);
    for (auto i = (int)strides.size() - 2; i > -1; i--)
      strides[i] = strides[i + 1] * mSize[i + 1];
  }
  return strides;
}

template <typename T>
int Array<T>::getStride(const int index) const {
  return getStride()[index];
}

template <typename T>
const std::string Array<T>::toString() const {
  // Initial value
  std::string string{"Array<T>::toString():\n"};
  // Add each element
  for (auto i = 0u; i < mVolume; ++i) {
    // Adding element separated by a space
    string += std::to_string(pData[i]) + " ";
    // Introduce an enter for each dimension change
    // If comented, all values will be printed in the same line
    auto multiplier = 1;
    for (auto dimension = (int)(mSize.size() - 1u);
         dimension > 0 &&
         (int(i / multiplier) % getSize(dimension) == getSize(dimension) - 1);
         dimension--) {
      string += "\n";
      multiplier *= getSize(dimension);
    }
  }
  // Return string
  return string;
}

template <typename T>
int Array<T>::getIndex(const std::vector<int>& indexes) const {
  auto index = 0;
  auto accumulated = 1;
  for (auto i = (int)indexes.size() - 1; i >= 0; i--) {
    index += accumulated * indexes[i];
    accumulated *= mSize[i];
  }
  return index;
}

template <typename T>
int Array<T>::getIndexAndCheck(const std::vector<int>& indexes) const {
  if (indexes.size() != mSize.size())
    spdlog::error("Requested indexes size is different than Array size.",
                  __LINE__, __FUNCTION__, __FILE__);
  return getIndex(indexes);
}

template <typename T>
T& Array<T>::commonAt(const int index) const {
  if (0 <= index && (size_t)index < mVolume)
    return pData[index];  // spData.get()[index]
  else {
    spdlog::error("Index out of bounds: 0 <= index && index < mVolume",
                  __LINE__, __FUNCTION__, __FILE__);
    return pData[0];  // spData.get()[0]
  }
}

template <typename T>
void Array<T>::resetAuxiliary(const std::vector<int>& sizes, T* const dataPtr) {
  if (!sizes.empty()) {
    // New size & volume
    mSize = sizes;
    mVolume = {std::accumulate(sizes.begin(), sizes.end(), 1ul,
                               std::multiplies<size_t>())};
    // Prepare shared_ptr
    if (dataPtr == nullptr) {
      spData.reset(new T[mVolume], std::default_delete<T[]>());
      pData = spData.get();
    } else {
      spData.reset();
      pData = dataPtr;
    }
  } else {
    mSize = {};
    mVolume = 0ul;
    spData.reset();
    pData = nullptr;
  }
}

// Instantiate a class with all the basic types

COMPILE_TEMPLATE_BASIC_TYPES_CLASS(Array);

}  // namespace openposert
