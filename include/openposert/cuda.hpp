#pragma once

#include <array>
#include <iostream>
#include <utility>  // std::pair
#include <vector>

namespace openposert {

const auto CUDA_NUM_THREADS = 512u;

void cudaCheck(const int line = -1, const std::string& function = "",
               const std::string& file = "");

int getCudaGpuNumber();

inline unsigned int getNumberCudaBlocks(
    const unsigned int totalRequired,
    const unsigned int numberCudaThreads = CUDA_NUM_THREADS) {
  return (totalRequired + numberCudaThreads - 1) / numberCudaThreads;
}
template <typename T>
void reorderAndNormalize(T* targetPtr, const unsigned char* const srcPtr,
                         const int width, const int height, const int channels);

template <typename T>
void uCharImageCast(unsigned char* targetPtr, const T* const srcPtr,
                    const int volume);

}  // namespace openposert
