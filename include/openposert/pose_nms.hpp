#pragma once

#include <array>

#include "openposert/point.hpp"

namespace openposert {

template <typename T>
void nmsCpu(T* targetPtr, int* kernelPtr, const T* const sourcePtr,
            const T threshold, const std::array<int, 4>& targetSize,
            const std::array<int, 4>& sourceSize, const Point<T>& offset);

// Windows: Cuda functions do not include OP_API
template <typename T>
void nmsGpu(T* targetPtr, int* kernelPtr, const T* const sourcePtr,
            const T threshold, const std::array<int, 4>& targetSize,
            const std::array<int, 4>& sourceSize, const Point<T>& offset);

// Windows: OpenCL functions do not include OP_API
template <typename T>
void nmsOcl(T* targetPtr, uint8_t* kernelGpuPtr, uint8_t* kernelCpuPtr,
            const T* const sourcePtr, const T threshold,
            const std::array<int, 4>& targetSize,
            const std::array<int, 4>& sourceSize, const Point<T>& offset,
            const int gpuID = 0);

}  // namespace openposert
