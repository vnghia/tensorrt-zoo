#pragma once

#include <string>
#include <vector>

#include "Trt.h"

namespace openposert {

class OpenPose {
 public:
  /**
   * @prototxt: NOTE: set input height and width in prototxt,
   * @calibratorData: create an empty instance, not support int8 now.
   * @maxBatchSize: set to 1.
   */
  OpenPose(const std::string& saveEngine);

  ~OpenPose();

  /**
   * @inputData: 1 * 3 * 480 * 640, or your favorite size, make sure modify it
   * in prototxt.
   * @result: output keypoint, (x1,y1,score1, x2,y2,score2 ... x25, y25,
   * scrore25) for one person and so on.
   */
  void DoInference(std::vector<float>& inputData, std::vector<float>& result);

 private:
  void MallocExtraMemory();

  Trt* mNet;

  int mBatchSize;

  // input's device memory
  void* mpInputGpu;
  // input size, count in byte
  int64_t mInputSize;
  // input datatype
  nvinfer1::DataType mInputDataType;
  // input dims
  nvinfer1::Dims3 mInputDims;

  void* mpHeatMapGpu;
  float* mpHeatMapCpu;
  int64_t mHeatMapSize;
  nvinfer1::Dims3 mHeatMapDims;

  const float mResizeScale = 4;  // resize 8x
  void* mpResizeMapGpu;
  float* mpResizeMapCpu;
  int64_t mResizeMapSize;
  nvinfer1::Dims3 mResizeMapDims;

  void* mpKernelGpu;
  int* mpKernelCpu;
  int64_t mKernelSize;
  nvinfer1::Dims3 mKernelDims;

  void* mpPeaksGpu;
  float* mpPeaksCpu;
  int64_t mPeaksSize;
  const int mNumPeaks = 25;
  int mMaxPerson = 128;
  const int mPeaksVector = 3;
  nvinfer1::Dims3 mPeaksDims;

  // nms parameters
  const float mThreshold = 0.05f;
  const float mNMSoffset = 0.5f;

  // body part connect parameters
  float mInterMinAboveThreshold = 0.95f;
  float mInterThreshold = 0.05f;
  int mMinSubsetCnt = 3;
  float mMinSubsetScore = 0.4f;
  float mScaleFactor = 8.f;
};

}  // namespace openposert
