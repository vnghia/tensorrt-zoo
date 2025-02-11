#include "Trt.h"
#include "openposert/body_part_connector.hpp"
#include "openposert/cuda.cuh"
#include "openposert/cuda.hpp"
#include "openposert/open_pose.hpp"
#include "openposert/point.hpp"
#include "openposert/pose_nms.hpp"
#include "openposert/resize_and_merge.hpp"
#include "spdlog/spdlog.h"

#include "time.h"
#include <cassert>
#include <cstring>
#include <memory>

#include <NvInfer.h>
#include <array>

#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)                                                    \
  {                                                                            \
    cudaError_t error_code = callstr;                                          \
    if (error_code != cudaSuccess) {                                           \
      std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":"    \
                << __LINE__ << std::endl;                                      \
      exit(0);                                                                 \
    }                                                                          \
  }
#endif

namespace openposert {

inline unsigned int getElementSize(nvinfer1::DataType t) {
  switch (t) {
  case nvinfer1::DataType::kINT32:
    return 4;
  case nvinfer1::DataType::kFLOAT:
    return 4;
  case nvinfer1::DataType::kHALF:
    return 2;
  case nvinfer1::DataType::kINT8:
    return 1;
  default:
    throw std::runtime_error("Invalid DataType.");
  }
}

inline void *safeCudaMalloc(size_t memSize) {
  void *deviceMem;
  CUDA_CHECK(cudaMalloc(&deviceMem, memSize));
  if (deviceMem == nullptr) {
    std::cerr << "Out of memory" << std::endl;
    exit(1);
  }
  return deviceMem;
}

OpenPose::OpenPose(const std::string &saveEngine) {
  mNet = new Trt();
  mNet->DeserializeEngine(saveEngine);
  MallocExtraMemory();
}
// TODO: release resource
OpenPose::~OpenPose() {
  if (mNet != nullptr) {
    delete mNet;
    mNet = nullptr;
  }
  if (mpKernelCpu != nullptr) {
    delete mpKernelCpu;
    mpKernelCpu = nullptr;
    cudaFree(mpKernelGpu);
  }
  if (mpPeaksCpu != nullptr) {
    delete mpPeaksCpu;
    mpPeaksCpu = nullptr;
    cudaFree(mpPeaksGpu);
  }
  if (mpResizeMapCpu != nullptr) {
    delete mpResizeMapCpu;
    mpResizeMapCpu = nullptr;
    cudaFree(mpResizeMapGpu);
  }
}

__global__ void Normalize(float *inputData) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  inputData[index] = inputData[index] / 255.f - 0.5f;
}
void OpenPose::DoInference(std::vector<float> &inputData,
                           std::vector<float> &result) {
  mNet->CopyFromHostToDevice(inputData, 0);
  int numBlocks = (mInputSize / getElementSize(mInputDataType) + 512 - 1) / 512;
  Normalize<<<numBlocks, 512, 0>>>((float *)mpInputGpu);
  mNet->Forward();
  std::vector<float> net_output;
  net_output.resize(78 * 60 * 80);
  mNet->CopyFromDeviceToHost(net_output, 1);
  memcpy((void *)mpHeatMapCpu, (void *)(net_output.data()), mHeatMapSize);

  if (mResizeScale > 1) {
    int widthSouce = mHeatMapDims.d[2];
    int heightSource = mHeatMapDims.d[1];
    int widthTarget = mResizeMapDims.d[2];
    int heightTarget = mResizeMapDims.d[1];
    const dim3 threadsPerBlock{16, 16, 1};
    const dim3 numBlocks{
        getNumberCudaBlocks(widthTarget, threadsPerBlock.x),
        getNumberCudaBlocks(heightTarget, threadsPerBlock.y),
        getNumberCudaBlocks(mResizeMapDims.d[0], threadsPerBlock.z)};
    resizeKernel<<<numBlocks, threadsPerBlock>>>(
        (float *)mpResizeMapGpu, (float *)mpHeatMapGpu, widthSouce,
        heightSource, widthTarget, heightTarget);
    CUDA_CHECK(cudaMemcpy(mpResizeMapCpu, mpResizeMapGpu, mResizeMapSize,
                          cudaMemcpyDeviceToHost));
  }

  // pose nms
  std::array<int, 4> targetSize2{mBatchSize, mNumPeaks, mMaxPerson,
                                 mPeaksVector};
  std::array<int, 4> sourceSize2{mBatchSize, mResizeMapDims.d[0],
                                 mResizeMapDims.d[1], mResizeMapDims.d[2]};
  Point<float> offset = Point<float>(0.5, 0.5);
  nmsGpu((float *)mpPeaksGpu, (int *)mpKernelGpu, (float *)mpResizeMapGpu,
         mThreshold, targetSize2, sourceSize2, offset);
  CUDA_CHECK(cudaMemcpyAsync(mpPeaksCpu, mpPeaksGpu, mPeaksSize,
                             cudaMemcpyDeviceToHost, 0));

  // bodypart connect
  Array<float> poseKeypoints;
  Array<float> poseScores;
  Point<int> resizeMapSize =
      Point<int>(mResizeMapDims.d[2], mResizeMapDims.d[1]);
  connectBodyPartsCpu(poseKeypoints, poseScores, mpResizeMapCpu, mpPeaksCpu,
                      resizeMapSize, mMaxPerson, mInterMinAboveThreshold,
                      mInterThreshold, mMinSubsetCnt, mMinSubsetScore, 1.f);

  result.resize(poseKeypoints.getVolume());
  // std::cout << "number of person: " << poseKeypoints.getVolume()/75 <<
  // std::endl;
  for (int i = 0; i < poseKeypoints.getVolume(); ++i) {
    if ((i + 1) % 3 == 0) {
      result[i] = poseKeypoints[i];
    } else {
      result[i] = poseKeypoints[i] * (8 / mResizeScale);
    }
  }
}

void OpenPose::MallocExtraMemory() {
  mBatchSize = 1;

  mpInputGpu = mNet->GetBindingPtr(0);
  mInputDataType = mNet->GetBindingDataType(0);
  nvinfer1::Dims inputDims = mNet->GetBindingDims(0);

  int offset = inputDims.nbDims == 4 ? 1 : 0;

  mInputDims = nvinfer1::Dims3(inputDims.d[offset], inputDims.d[offset + 1],
                               inputDims.d[offset + 2]);
  mInputSize = mNet->GetBindingSize(0);

  std::cout << "=====>malloc extra memory for openpose..." << std::endl;

  mpHeatMapGpu = mNet->GetBindingPtr(1);
  nvinfer1::Dims heatMapDims = mNet->GetBindingDims(1);
  std::cout << "heatmap Dims" << heatMapDims.nbDims << std::endl;
  mHeatMapDims =
      nvinfer1::Dims3(heatMapDims.d[offset], heatMapDims.d[offset + 1],
                      heatMapDims.d[offset + 2]);
  std::cout << "heatmap size: " << mBatchSize << " " << mHeatMapDims.d[0] << " "
            << mHeatMapDims.d[1] << " " << mHeatMapDims.d[2] << std::endl;
  mHeatMapSize = mBatchSize * mHeatMapDims.d[0] * mHeatMapDims.d[1] *
                 mHeatMapDims.d[2] * getElementSize(mInputDataType);
  mpHeatMapCpu = new float[mHeatMapSize / getElementSize(mInputDataType)];
  std::cout << "allocate heatmap host and divice memory done" << std::endl;

  // malloc resieze memory
  mResizeMapDims =
      nvinfer1::Dims3(mHeatMapDims.d[0], int(mHeatMapDims.d[1] * mResizeScale),
                      int(mHeatMapDims.d[2] * mResizeScale));
  mResizeMapSize = mBatchSize * mResizeMapDims.d[0] * mResizeMapDims.d[1] *
                   mResizeMapDims.d[2] * getElementSize(mInputDataType);
  if (mResizeScale > 1) {
    mpResizeMapGpu = safeCudaMalloc(mResizeMapSize);
    mpResizeMapCpu = new float[mResizeMapSize / getElementSize(mInputDataType)];
  } else {
    mpResizeMapGpu = mpHeatMapGpu;
    mpResizeMapCpu = mpHeatMapCpu;
  }

  std::cout << "resize map size: " << mBatchSize << " " << mResizeMapDims.d[0]
            << " " << mResizeMapDims.d[1] << " " << mResizeMapDims.d[2]
            << std::endl;

  // malloc kernel memory
  mKernelDims = nvinfer1::Dims3(mResizeMapDims.d[0], mResizeMapDims.d[1],
                                mResizeMapDims.d[2]);
  mKernelSize = mBatchSize * mKernelDims.d[0] * mKernelDims.d[1] *
                mKernelDims.d[2] * sizeof(int);
  mpKernelGpu = safeCudaMalloc(mKernelSize);
  mpKernelCpu = new int[mBatchSize * mKernelDims.d[0] * mKernelDims.d[1] *
                        mKernelDims.d[2]];
  std::cout << "kernel size: " << mBatchSize << " " << mKernelDims.d[0] << " "
            << mKernelDims.d[1] << " " << mKernelDims.d[2] << std::endl;
  std::cout << "allocate kernel host and device memory done" << std::endl;

  // malloc peaks memory
  mPeaksDims = nvinfer1::Dims3(mNumPeaks, mMaxPerson, mPeaksVector);
  mPeaksSize = mPeaksDims.d[0] * mPeaksDims.d[1] * mPeaksDims.d[2] *
               getElementSize(mInputDataType);
  mpPeaksGpu = safeCudaMalloc(mPeaksSize);
  mpPeaksCpu = new float[mPeaksDims.d[0] * mPeaksDims.d[1] * mPeaksDims.d[2]];
  std::cout << "peaks size: " << mBatchSize << " " << mPeaksDims.d[0] << " "
            << mPeaksDims.d[1] << " " << mPeaksDims.d[2] << std::endl;
  std::cout << "allocate peaks host and device memory done" << std::endl;
  std::cout << "=====> malloc extra memory done" << std::endl;
}

} // namespace openposert
