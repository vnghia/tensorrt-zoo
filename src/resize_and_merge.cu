#include "openposert/cuda.cuh"
#include "openposert/cuda.hpp"
#include "openposert/resize_and_merge.hpp"
#include "spdlog/spdlog.h"
#include "utils.h"

namespace openposert {

const auto THREADS_PER_BLOCK = 256u;
const auto THREADS_PER_BLOCK_1D = 16u;

template <typename T>
__global__ void fillKernel(T *targetPtr, const T *const sourcePtr,
                           const int N) {
  const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (x < N)
    targetPtr[x] = sourcePtr[x];
}

template <typename T>
__global__ void resizeKernel(T *targetPtr, const T *const sourcePtr,
                             const int widthSource, const int heightSource,
                             const int widthTarget, const int heightTarget) {
  const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
  const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
  const auto channel = (blockIdx.z * blockDim.z) + threadIdx.z;
  if (x < widthTarget && y < heightTarget) {
    const auto sourceArea = widthSource * heightSource;
    const auto targetArea = widthTarget * heightTarget;
    const T xSource = (x + T(0.5f)) * widthSource / T(widthTarget) - T(0.5f);
    const T ySource = (y + T(0.5f)) * heightSource / T(heightTarget) - T(0.5f);
    const T *const sourcePtrChannel = sourcePtr + channel * sourceArea;
    targetPtr[channel * targetArea + y * widthTarget + x] =
        bicubicInterpolate(sourcePtrChannel, xSource, ySource, widthSource,
                           heightSource, widthSource);
  }
}

template <typename T>
__global__ void
resizeAndPadKernel(T *targetPtr, const T *const sourcePtr,
                   const int widthSource, const int heightSource,
                   const int widthTarget, const int heightTarget,
                   const float rescaleFactor) {
  const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
  const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
  const auto channel = (blockIdx.z * blockDim.z) + threadIdx.z;
  if (x < widthTarget && y < heightTarget) {
    const auto targetArea = widthTarget * heightTarget;
    if (x < widthSource * rescaleFactor && y < heightSource * rescaleFactor) {
      const auto sourceArea = widthSource * heightSource;
      const T xSource = (x + T(0.5f)) / T(rescaleFactor) - T(0.5f);
      const T ySource = (y + T(0.5f)) / T(rescaleFactor) - T(0.5f);
      const T *const sourcePtrChannel = sourcePtr + channel * sourceArea;
      targetPtr[channel * targetArea + y * widthTarget + x] =
          bicubicInterpolate(sourcePtrChannel, xSource, ySource, widthSource,
                             heightSource, widthSource);
    } else
      targetPtr[channel * targetArea + y * widthTarget + x] = 0;
  }
}

template <typename T>
__global__ void
resizeAndPadKernel(T *targetPtr, const unsigned char *const sourcePtr,
                   const int widthSource, const int heightSource,
                   const int widthTarget, const int heightTarget,
                   const float rescaleFactor) {
  const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
  const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
  const auto channel = (blockIdx.z * blockDim.z) + threadIdx.z;
  if (x < widthTarget && y < heightTarget) {
    const auto targetArea = widthTarget * heightTarget;
    if (x < widthSource * rescaleFactor && y < heightSource * rescaleFactor) {
      const auto sourceArea = widthSource * heightSource;
      const T xSource = (x + T(0.5f)) / T(rescaleFactor) - T(0.5f);
      const T ySource = (y + T(0.5f)) / T(rescaleFactor) - T(0.5f);
      const unsigned char *sourcePtrChannel = sourcePtr + channel * sourceArea;
      targetPtr[channel * targetArea + y * widthTarget + x] =
          bicubicInterpolate(sourcePtrChannel, xSource, ySource, widthSource,
                             heightSource, widthSource);
    } else
      targetPtr[channel * targetArea + y * widthTarget + x] = 0;
  }
}

template <typename T>
__global__ void
resize8TimesKernel(T *targetPtr, const T *const sourcePtr,
                   const int widthSource, const int heightSource,
                   const int widthTarget, const int heightTarget,
                   const unsigned int rescaleFactor) {
  const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
  const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
  const auto channel = (blockIdx.z * blockDim.z) + threadIdx.z;

  if (x < widthTarget && y < heightTarget) {
    // Normal resize
    // Note: The first blockIdx of each dimension behaves differently, so
    // applying old version in those
    if (blockIdx.x < 1 || blockIdx.y < 1)
    // Actually it is only required for the first 4, but then I would have not
    // loaded the shared memory if ((blockIdx.x < 1 || blockIdx.y < 1) &&
    // (threadIdx.x < 4 || threadIdx.y < 4))
    {
      const auto sourceArea = widthSource * heightSource;
      const auto targetArea = widthTarget * heightTarget;
      const T xSource = (x + T(0.5f)) / T(rescaleFactor) - T(0.5f);
      const T ySource = (y + T(0.5f)) / T(rescaleFactor) - T(0.5f);
      const T *const sourcePtrChannel = sourcePtr + channel * sourceArea;
      targetPtr[channel * targetArea + y * widthTarget + x] =
          bicubicInterpolate(sourcePtrChannel, xSource, ySource, widthSource,
                             heightSource, widthSource);
      return;
    }

    // Load shared memory
    // If resize >= 5, then #threads per block >= # elements of shared memory
    const auto sharedSize = 25; // (4+1)^2
    __shared__ T sourcePtrShared[sharedSize];
    const auto sharedLoadId = threadIdx.x + rescaleFactor * threadIdx.y;
    if (sharedLoadId < sharedSize) {
      // Idea: Find minimum possible x and y
      const auto minTargetX = blockIdx.x * rescaleFactor;
      const auto minSourceXFloat =
          (minTargetX + T(0.5f)) / T(rescaleFactor) - T(0.5f);
      const auto minSourceXInt = int(floor(minSourceXFloat)) - 1;
      const auto minTargetY = blockIdx.y * rescaleFactor;
      const auto minSourceYFloat =
          (minTargetY + T(0.5f)) / T(rescaleFactor) - T(0.5f);
      const auto minSourceYInt = int(floor(minSourceYFloat)) - 1;
      // Get current x and y
      const auto xClean = fastTruncateCuda(
          minSourceXInt + int(sharedLoadId % 5), 0, widthSource - 1);
      const auto yClean = fastTruncateCuda(
          minSourceYInt + int(sharedLoadId / 5), 0, heightSource - 1);
      // Load into shared memory
      const auto sourceIndex =
          (channel * heightSource + yClean) * widthSource + xClean;
      sourcePtrShared[sharedLoadId] = sourcePtr[sourceIndex];
    }
    __syncthreads();

    // Apply resize
    const auto targetArea = widthTarget * heightTarget;
    const T xSource = (x + T(0.5f)) / T(rescaleFactor) - T(0.5f);
    const T ySource = (y + T(0.5f)) / T(rescaleFactor) - T(0.5f);
    targetPtr[channel * targetArea + y * widthTarget + x] =
        bicubicInterpolate8Times(sourcePtrShared, xSource, ySource, widthSource,
                                 heightSource, threadIdx.x, threadIdx.y);
  }
}

template <typename T>
__global__ void resizeAndAddAndAverageKernel(
    T *targetPtr, const int counter, const T *const scaleWidths,
    const T *const scaleHeights, const int *const widthSources,
    const int *const heightSources, const int widthTarget,
    const int heightTarget, const T *const sourcePtr0,
    const T *const sourcePtr1, const T *const sourcePtr2,
    const T *const sourcePtr3, const T *const sourcePtr4,
    const T *const sourcePtr5, const T *const sourcePtr6,
    const T *const sourcePtr7) {
  const auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
  const auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
  const auto channel = (blockIdx.z * blockDim.z) + threadIdx.z;
  // For each pixel
  if (x < widthTarget && y < heightTarget) {
    // Local variable for higher speed
    T interpolated = T(0.f);
    // For each input source pointer
    for (auto i = 0; i < counter; ++i) {
      const auto sourceArea = widthSources[i] * heightSources[i];
      const T xSource = (x + T(0.5f)) / scaleWidths[i] - T(0.5f);
      const T ySource = (y + T(0.5f)) / scaleHeights[i] - T(0.5f);
      const T *const sourcePtr = (i == 0   ? sourcePtr0
                                  : i == 1 ? sourcePtr1
                                  : i == 2 ? sourcePtr2
                                  : i == 3 ? sourcePtr3
                                  : i == 4 ? sourcePtr4
                                  : i == 5 ? sourcePtr5
                                  : i == 6 ? sourcePtr6
                                           : sourcePtr7);
      const T *const sourcePtrChannel = sourcePtr + channel * sourceArea;
      interpolated += bicubicInterpolate(sourcePtrChannel, xSource, ySource,
                                         widthSources[i], heightSources[i],
                                         widthSources[i]);
    }
    // Save into memory
    const auto targetArea = widthTarget * heightTarget;
    targetPtr[channel * targetArea + y * widthTarget + x] =
        interpolated / T(counter);
  }
}

template <typename T>
void resizeAndMergeGpu(T *targetPtr, const std::vector<const T *> &sourcePtrs,
                       const std::array<int, 4> &targetSize,
                       const std::vector<std::array<int, 4>> &sourceSizes,
                       const std::vector<T> &scaleInputToNetInputs) {
  // Sanity checks
  if (sourceSizes.empty())
    spdlog::error("sourceSizes cannot be empty.", __LINE__, __FUNCTION__,
                  __FILE__);
  if (sourcePtrs.size() != sourceSizes.size() ||
      sourceSizes.size() != scaleInputToNetInputs.size())
    spdlog::error("Size(sourcePtrs) must match size(sourceSizes) and "
                  "size(scaleInputToNetInputs). Currently: " +
                      std::to_string(sourcePtrs.size()) + " vs. " +
                      std::to_string(sourceSizes.size()) + " vs. " +
                      std::to_string(scaleInputToNetInputs.size()) + ".",
                  __LINE__, __FUNCTION__, __FILE__);

  // Parameters
  const auto channels = targetSize[1];
  const auto heightTarget = targetSize[2];
  const auto widthTarget = targetSize[3];
  const dim3 threadsPerBlock{THREADS_PER_BLOCK_1D, THREADS_PER_BLOCK_1D};
  const dim3 numBlocks{getNumberCudaBlocks(widthTarget, threadsPerBlock.x),
                       getNumberCudaBlocks(heightTarget, threadsPerBlock.y)};
  const auto &sourceSize = sourceSizes[0];
  const auto heightSource = sourceSize[2];
  const auto widthSource = sourceSize[3];

  // No multi-scale merging or no merging required
  if (sourceSizes.size() == 1) {
    const auto num = sourceSize[0];
    if (targetSize[0] > 1 || num == 1) {
      if (widthTarget / widthSource == 1 && heightTarget / heightSource == 1) {
        const auto N = widthTarget * heightTarget * num * channels;
        const dim3 threadsPerBlock{THREADS_PER_BLOCK};
        const dim3 numBlocks{getNumberCudaBlocks(N, threadsPerBlock.x)};
        fillKernel<<<numBlocks, threadsPerBlock>>>(targetPtr, sourcePtrs.at(0),
                                                   N);
      } else {
        if (widthTarget / widthSource != 8 || heightTarget / heightSource != 8)
          spdlog::error(
              "Kernel only implemented for 8x resize. Notify us if this "
              "error appears.",
              __LINE__, __FUNCTION__, __FILE__);
        const auto rescaleFactor =
            (unsigned int)std::ceil(heightTarget / (float)(heightSource));
        const dim3 threadsPerBlock{rescaleFactor, rescaleFactor, 1};
        const dim3 numBlocks{
            getNumberCudaBlocks(widthTarget, threadsPerBlock.x),
            getNumberCudaBlocks(heightTarget, threadsPerBlock.y),
            getNumberCudaBlocks(num * channels, threadsPerBlock.z)};
        resize8TimesKernel<<<numBlocks, threadsPerBlock>>>(
            targetPtr, sourcePtrs.at(0), widthSource, heightSource, widthTarget,
            heightTarget, rescaleFactor);
      }
    }
    // Old inefficient multi-scale merging
    else
      spdlog::error("It should never reache this point. Notify us otherwise.",
                    __LINE__, __FUNCTION__, __FILE__);
  }
  // Multi-scaling merging
  else {
    const auto scaleToMainScaleWidth = widthTarget / T(widthSource);
    const auto scaleToMainScaleHeight = heightTarget / T(heightSource);

    if (sourcePtrs.size() > 8)
      spdlog::error(
          "More than 8 scales are not implemented (yet). Notify us to "
          "implement it.",
          __LINE__, __FUNCTION__, __FILE__);
    const dim3 threadsPerBlock{THREADS_PER_BLOCK_1D, THREADS_PER_BLOCK_1D, 1};
    const dim3 numBlocks{getNumberCudaBlocks(widthTarget, threadsPerBlock.x),
                         getNumberCudaBlocks(heightTarget, threadsPerBlock.y),
                         getNumberCudaBlocks(channels, threadsPerBlock.z)};
    // Fill auxiliary params
    std::vector<int> widthSourcesCpu(sourceSizes.size());
    std::vector<int> heightSourcesCpu(sourceSizes.size());
    std::vector<T> scaleWidthsCpu(sourceSizes.size());
    std::vector<T> scaleHeightsCpu(sourceSizes.size());
    for (auto i = 0u; i < sourceSizes.size(); ++i) {
      const auto &currentSize = sourceSizes.at(i);
      heightSourcesCpu[i] = currentSize[2];
      widthSourcesCpu[i] = currentSize[3];
      const auto scaleInputToNet =
          scaleInputToNetInputs[i] / scaleInputToNetInputs[0];
      scaleWidthsCpu[i] = scaleToMainScaleWidth / scaleInputToNet;
      scaleHeightsCpu[i] = scaleToMainScaleHeight / scaleInputToNet;
    }
    // GPU params
    int *widthSources;
    cudaMalloc((void **)&widthSources, sizeof(int) * sourceSizes.size());
    cudaMemcpy(widthSources, widthSourcesCpu.data(),
               sizeof(int) * sourceSizes.size(), cudaMemcpyHostToDevice);
    int *heightSources;
    cudaMalloc((void **)&heightSources, sizeof(int) * sourceSizes.size());
    cudaMemcpy(heightSources, heightSourcesCpu.data(),
               sizeof(int) * sourceSizes.size(), cudaMemcpyHostToDevice);
    T *scaleWidths;
    cudaMalloc((void **)&scaleWidths, sizeof(T) * sourceSizes.size());
    cudaMemcpy(scaleWidths, scaleWidthsCpu.data(),
               sizeof(T) * sourceSizes.size(), cudaMemcpyHostToDevice);
    T *scaleHeights;
    cudaMalloc((void **)&scaleHeights, sizeof(T) * sourceSizes.size());
    cudaMemcpy(scaleHeights, scaleHeightsCpu.data(),
               sizeof(T) * sourceSizes.size(), cudaMemcpyHostToDevice);
    // Resize each channel, add all, and get average
    resizeAndAddAndAverageKernel<<<numBlocks, threadsPerBlock>>>(
        targetPtr, (int)sourceSizes.size(), scaleWidths, scaleHeights,
        widthSources, heightSources, widthTarget, heightTarget, sourcePtrs[0],
        sourcePtrs[1], sourcePtrs[2], sourcePtrs[3], sourcePtrs[4],
        sourcePtrs[5], sourcePtrs[6], sourcePtrs[7]);
    // Free memory
    if (widthSources != nullptr)
      cudaFree(widthSources);
    if (heightSources != nullptr)
      cudaFree(heightSources);
    if (scaleWidths != nullptr)
      cudaFree(scaleWidths);
    if (scaleHeights != nullptr)
      cudaFree(scaleHeights);
  }
}

template <typename T>
void resizeAndPadRbgGpu(T *targetPtr, const T *const srcPtr,
                        const int widthSource, const int heightSource,
                        const int widthTarget, const int heightTarget,
                        const T scaleFactor) {
  const auto channels = 3;
  const dim3 threadsPerBlock{THREADS_PER_BLOCK_1D, THREADS_PER_BLOCK_1D, 1};
  const dim3 numBlocks{getNumberCudaBlocks(widthTarget, threadsPerBlock.x),
                       getNumberCudaBlocks(heightTarget, threadsPerBlock.y),
                       getNumberCudaBlocks(channels, threadsPerBlock.z)};
  resizeAndPadKernel<<<numBlocks, threadsPerBlock>>>(
      targetPtr, srcPtr, widthSource, heightSource, widthTarget, heightTarget,
      scaleFactor);
}

template <typename T>
void resizeAndPadRbgGpu(T *targetPtr, const unsigned char *const srcPtr,
                        const int widthSource, const int heightSource,
                        const int widthTarget, const int heightTarget,
                        const T scaleFactor)

{
  const auto channels = 3;
  const dim3 threadsPerBlock{THREADS_PER_BLOCK_1D, THREADS_PER_BLOCK_1D, 1};
  const dim3 numBlocks{getNumberCudaBlocks(widthTarget, threadsPerBlock.x),
                       getNumberCudaBlocks(heightTarget, threadsPerBlock.y),
                       getNumberCudaBlocks(channels, threadsPerBlock.z)};
  resizeAndPadKernel<<<numBlocks, threadsPerBlock>>>(
      targetPtr, srcPtr, widthSource, heightSource, widthTarget, heightTarget,
      scaleFactor);
}

template __global__ void
resizeKernel(float *targetPtr, const float *const sourcePtr,
             const int widthSource, const int heightSource,
             const int widthTarget, const int heightTarget);

template void
resizeAndMergeGpu(float *targetPtr,
                  const std::vector<const float *> &sourcePtrs,
                  const std::array<int, 4> &targetSize,
                  const std::vector<std::array<int, 4>> &sourceSizes,
                  const std::vector<float> &scaleInputToNetInputs);
template void
resizeAndMergeGpu(double *targetPtr,
                  const std::vector<const double *> &sourcePtrs,
                  const std::array<int, 4> &targetSize,
                  const std::vector<std::array<int, 4>> &sourceSizes,
                  const std::vector<double> &scaleInputToNetInputs);

template void resizeAndPadRbgGpu(float *targetPtr, const float *const srcPtr,
                                 const int widthSource, const int heightSource,
                                 const int widthTarget, const int heightTarget,
                                 const float scaleFactor);
template void resizeAndPadRbgGpu(double *targetPtr, const double *const srcPtr,
                                 const int widthSource, const int heightSource,
                                 const int widthTarget, const int heightTarget,
                                 const double scaleFactor);

template void resizeAndPadRbgGpu(float *targetPtr,
                                 const unsigned char *const srcPtr,
                                 const int widthSource, const int heightSource,
                                 const int widthTarget, const int heightTarget,
                                 const float scaleFactor);
template void resizeAndPadRbgGpu(double *targetPtr,
                                 const unsigned char *const srcPtr,
                                 const int widthSource, const int heightSource,
                                 const int widthTarget, const int heightTarget,
                                 const double scaleFactor);

} // namespace openposert