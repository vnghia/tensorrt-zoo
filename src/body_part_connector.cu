#include "openposert/body_part_connector.hpp"
#include "openposert/cuda.hpp"
#include "openposert/fast_math.hpp"
#include "spdlog/spdlog.h"
#include "utils.h"

#include <set>

namespace openposert {

template <typename T> inline __device__ int intRoundGPU(const T a) {
  return int(a + T(0.5));
}

template <typename T>
inline __device__ T process(const T *bodyPartA, const T *bodyPartB,
                            const T *mapX, const T *mapY,
                            const int heatmapWidth, const int heatmapHeight,
                            const T interThreshold,
                            const T interMinAboveThreshold,
                            const T defaultNmsThreshold) {
  const auto vectorAToBX = bodyPartB[0] - bodyPartA[0];
  const auto vectorAToBY = bodyPartB[1] - bodyPartA[1];
  const auto vectorAToBMax = max(abs(vectorAToBX), abs(vectorAToBY));
  const auto numberPointsInLine =
      max(5, min(25, intRoundGPU(sqrt(5 * vectorAToBMax))));
  const auto vectorNorm =
      T(sqrt(vectorAToBX * vectorAToBX + vectorAToBY * vectorAToBY));

  if (vectorNorm > 1e-6) {
    const auto sX = bodyPartA[0];
    const auto sY = bodyPartA[1];
    const auto vectorAToBNormX = vectorAToBX / vectorNorm;
    const auto vectorAToBNormY = vectorAToBY / vectorNorm;

    auto sum = T(0.);
    auto count = 0;
    const auto vectorAToBXInLine = vectorAToBX / numberPointsInLine;
    const auto vectorAToBYInLine = vectorAToBY / numberPointsInLine;
    for (auto lm = 0; lm < numberPointsInLine; ++lm) {
      const auto mX =
          min(heatmapWidth - 1, intRoundGPU(sX + lm * vectorAToBXInLine));
      const auto mY =
          min(heatmapHeight - 1, intRoundGPU(sY + lm * vectorAToBYInLine));
      const auto idx = mY * heatmapWidth + mX;
      const auto score =
          (vectorAToBNormX * mapX[idx] + vectorAToBNormY * mapY[idx]);
      if (score > interThreshold) {
        sum += score;
        count++;
      }
    }

    // Return PAF score
    if (count / T(numberPointsInLine) > interMinAboveThreshold)
      return sum / count;
    else {
      // Ideally, if distanceAB = 0, PAF is 0 between A and B, provoking a false
      // negative To fix it, we consider PAF-connected keypoints very close to
      // have a minimum PAF score, such that:
      //     1. It will consider very close keypoints (where the PAF is 0)
      //     2. But it will not automatically connect them (case PAF score = 1),
      //     or real PAF might got
      //        missing
      const auto l2Dist =
          sqrtf(vectorAToBX * vectorAToBX + vectorAToBY * vectorAToBY);
      const auto threshold = sqrtf(heatmapWidth * heatmapHeight) /
                             150; // 3.3 for 368x656, 6.6 for 2x resolution
      if (l2Dist < threshold)
        return T(
            defaultNmsThreshold +
            1e-6); // Without 1e-6 will not work because I use strict greater
    }
  }
  return -1;
}

template <typename T>
__global__ void
pafScoreKernel(T *pairScoresPtr, const T *const heatMapPtr,
               const T *const peaksPtr,
               const unsigned int *const bodyPartPairsPtr,
               const unsigned int *const mapIdxPtr, const unsigned int maxPeaks,
               const int numberBodyPartPairs, const int heatmapWidth,
               const int heatmapHeight, const T interThreshold,
               const T interMinAboveThreshold, const T defaultNmsThreshold) {
  const auto peakB = (blockIdx.x * blockDim.x) + threadIdx.x;
  const auto peakA = (blockIdx.y * blockDim.y) + threadIdx.y;
  const auto pairIndex = (blockIdx.z * blockDim.z) + threadIdx.z;

  if (peakA < maxPeaks && peakB < maxPeaks)
  // if (pairIndex < numberBodyPartPairs && peakA < maxPeaks && peakB <
  // maxPeaks)
  {
    const auto baseIndex = 2 * pairIndex;
    const auto partA = bodyPartPairsPtr[baseIndex];
    const auto partB = bodyPartPairsPtr[baseIndex + 1];

    const T numberPeaksA = peaksPtr[3 * partA * (maxPeaks + 1)];
    const T numberPeaksB = peaksPtr[3 * partB * (maxPeaks + 1)];

    const auto outputIndex = (pairIndex * maxPeaks + peakA) * maxPeaks + peakB;
    if (peakA < numberPeaksA && peakB < numberPeaksB) {
      const auto mapIdxX = mapIdxPtr[baseIndex];
      const auto mapIdxY = mapIdxPtr[baseIndex + 1];

      const T *const bodyPartA =
          peaksPtr + (3 * (partA * (maxPeaks + 1) + peakA + 1));
      const T *const bodyPartB =
          peaksPtr + (3 * (partB * (maxPeaks + 1) + peakB + 1));
      const T *const mapX = heatMapPtr + mapIdxX * heatmapWidth * heatmapHeight;
      const T *const mapY = heatMapPtr + mapIdxY * heatmapWidth * heatmapHeight;
      pairScoresPtr[outputIndex] =
          process(bodyPartA, bodyPartB, mapX, mapY, heatmapWidth, heatmapHeight,
                  interThreshold, interMinAboveThreshold, defaultNmsThreshold);
    } else
      pairScoresPtr[outputIndex] = -1;
  }
}

template <typename T>
void connectBodyPartsGpu(Array<T> &poseKeypoints, Array<T> &poseScores,
                         const T *const heatMapGpuPtr, const T *const peaksPtr,
                         const Point<int> &heatMapSize, const int maxPeaks,
                         const T interMinAboveThreshold, const T interThreshold,
                         const int minSubsetCnt, const T minSubsetScore,
                         const T scaleFactor, const bool maximizePositives,
                         Array<T> pairScoresCpu, T *pairScoresGpuPtr,
                         const unsigned int *const bodyPartPairsGpuPtr,
                         const unsigned int *const mapIdxGpuPtr,
                         const T *const peaksGpuPtr,
                         const T defaultNmsThreshold) {
  // Parts Connection
  const auto &bodyPartPairs = std::vector<unsigned int>{
      1,  8,  1, 2,  1,  5,  2,  3,  3,  4,  5,  6,  6,  7,  8,  9,  9,  10,
      10, 11, 8, 12, 12, 13, 13, 14, 1,  0,  0,  15, 15, 17, 0,  16, 16, 18,
      2,  17, 5, 18, 14, 19, 19, 20, 14, 21, 11, 22, 22, 23, 11, 24};
  const auto numberBodyParts = 25;
  const auto numberBodyPartPairs = (unsigned int)(bodyPartPairs.size() / 2);
  const auto totalComputations = pairScoresCpu.getVolume();

  if (numberBodyParts == 0)
    spdlog::error(
        "Invalid value of numberBodyParts, it must be positive, not " +
            std::to_string(numberBodyParts),
        __LINE__, __FUNCTION__, __FILE__);
  if (bodyPartPairsGpuPtr == nullptr || mapIdxGpuPtr == nullptr)
    spdlog::error("The pointers bodyPartPairsGpuPtr and mapIdxGpuPtr cannot be "
                  "nullptr.",
                  __LINE__, __FUNCTION__, __FILE__);

  const dim3 THREADS_PER_BLOCK{128, 1, 1};
  const dim3 numBlocks{
      getNumberCudaBlocks(maxPeaks, THREADS_PER_BLOCK.x),
      getNumberCudaBlocks(maxPeaks, THREADS_PER_BLOCK.y),
      getNumberCudaBlocks(numberBodyPartPairs, THREADS_PER_BLOCK.z)};
  pafScoreKernel<<<numBlocks, THREADS_PER_BLOCK>>>(
      pairScoresGpuPtr, heatMapGpuPtr, peaksGpuPtr, bodyPartPairsGpuPtr,
      mapIdxGpuPtr, maxPeaks, (int)numberBodyPartPairs, heatMapSize.x,
      heatMapSize.y, interThreshold, interMinAboveThreshold,
      defaultNmsThreshold);
  // pairScoresCpu <-- pairScoresGpu
  cudaMemcpy(pairScoresCpu.getPtr(), pairScoresGpuPtr,
             totalComputations * sizeof(T), cudaMemcpyDeviceToHost);
  // OP_PROFILE_END(timeNormalize2, 1e3, REPS);

  // Get pair connections and their scores
  const auto pairConnections = pafPtrIntoVector(
      pairScoresCpu, peaksPtr, maxPeaks, bodyPartPairs, numberBodyPartPairs);
  auto peopleVector = pafVectorIntoPeopleVector(
      pairConnections, peaksPtr, maxPeaks, bodyPartPairs, numberBodyParts);
  // // Old code: Get pair connections and their scores
  // // std::vector<std::pair<std::vector<int>, double>> refers to:
  // //     - std::vector<int>: [body parts locations, #body parts found]
  // //     - double: person subset score
  // const T* const tNullptr = nullptr;
  // const auto peopleVector = createPeopleVector(
  //     tNullptr, peaksPtr, poseModel, heatMapSize, maxPeaks, interThreshold,
  //     interMinAboveThreshold, bodyPartPairs, numberBodyParts,
  //     numberBodyPartPairs, pairScoresCpu);
  // Delete people below the following thresholds:
  // a) minSubsetCnt: removed if less than minSubsetCnt body parts
  // b) minSubsetScore: removed if global score smaller than this
  // c) maxPeaks (POSE_MAX_PEOPLE): keep first maxPeaks people above
  // thresholds
  int numberPeople;
  std::vector<int> validSubsetIndexes;
  // validSubsetIndexes.reserve(fastMin((size_t)maxPeaks,
  // peopleVector.size()));
  validSubsetIndexes.reserve(peopleVector.size());
  removePeopleBelowThresholdsAndFillFaces(
      validSubsetIndexes, numberPeople, peopleVector, numberBodyParts,
      minSubsetCnt, minSubsetScore, maximizePositives, peaksPtr);
  // Fill and return poseKeypoints
  peopleVectorToPeopleArray(poseKeypoints, poseScores, scaleFactor,
                            peopleVector, validSubsetIndexes, peaksPtr,
                            numberPeople, numberBodyParts, numberBodyPartPairs);
}

template void connectBodyPartsGpu(
    Array<float> &poseKeypoints, Array<float> &poseScores,
    const float *const heatMapGpuPtr, const float *const peaksPtr,
    const Point<int> &heatMapSize, const int maxPeaks,
    const float interMinAboveThreshold, const float interThreshold,
    const int minSubsetCnt, const float minSubsetScore, const float scaleFactor,
    const bool maximizePositives, Array<float> pairScoresCpu,
    float *pairScoresGpuPtr, const unsigned int *const bodyPartPairsGpuPtr,
    const unsigned int *const mapIdxGpuPtr, const float *const peaksGpuPtr,
    const float defaultNmsThreshold);
template void connectBodyPartsGpu(
    Array<double> &poseKeypoints, Array<double> &poseScores,
    const double *const heatMapGpuPtr, const double *const peaksPtr,
    const Point<int> &heatMapSize, const int maxPeaks,
    const double interMinAboveThreshold, const double interThreshold,
    const int minSubsetCnt, const double minSubsetScore,
    const double scaleFactor, const bool maximizePositives,
    Array<double> pairScoresCpu, double *pairScoresGpuPtr,
    const unsigned int *const bodyPartPairsGpuPtr,
    const unsigned int *const mapIdxGpuPtr, const double *const peaksGpuPtr,
    const double defaultNmsThreshold);

// ============================ cpu ==============================
template <typename T>
inline T getScoreAB(const int i, const int j, const T *const candidateAPtr,
                    const T *const candidateBPtr, const T *const mapX,
                    const T *const mapY, const Point<int> &heatMapSize,
                    const T interThreshold, const T interMinAboveThreshold) {

  // candidatePtr 某一个关键点的peaks 128x3
  const auto vectorAToBX = candidateBPtr[3 * j] - candidateAPtr[3 * i];
  const auto vectorAToBY = candidateBPtr[3 * j + 1] - candidateAPtr[3 * i + 1];
  const auto vectorAToBMax =
      fastMax(std::abs(vectorAToBX), std::abs(vectorAToBY));
  const auto numberPointsInLine =
      fastMax(5, fastMin(25, positiveIntRound(std::sqrt(5 * vectorAToBMax))));
  const auto vectorNorm =
      T(std::sqrt(vectorAToBX * vectorAToBX + vectorAToBY * vectorAToBY));
  if (vectorNorm > 1e-6) {
    const auto sX = candidateAPtr[3 * i];
    const auto sY = candidateAPtr[3 * i + 1];
    // std::cout << "sX: " << sX << " sY " << sY << std::endl;
    const auto vectorAToBNormX = vectorAToBX / vectorNorm;
    const auto vectorAToBNormY = vectorAToBY / vectorNorm;

    auto sum = T(0);
    auto count = 0u;
    const auto vectorAToBXInLine = vectorAToBX / numberPointsInLine;
    const auto vectorAToBYInLine = vectorAToBY / numberPointsInLine;
    for (auto lm = 0; lm < numberPointsInLine; ++lm) {
      const auto mX =
          fastMax(0, fastMin(heatMapSize.x - 1,
                             positiveIntRound(sX + lm * vectorAToBXInLine)));
      const auto mY =
          fastMax(0, fastMin(heatMapSize.y - 1,
                             positiveIntRound(sY + lm * vectorAToBYInLine)));
      const auto idx = mY * heatMapSize.x + mX;
      const auto score =
          (vectorAToBNormX * mapX[idx] + vectorAToBNormY * mapY[idx]);

      if (score > interThreshold) {
        sum += score;
        count++;
      }
    }

    if (count / T(numberPointsInLine) > interMinAboveThreshold) {
      return sum / count;
    }
  }
  return T(0);
}

template <typename T>
void getKeypointCounter(
    int &personCounter,
    const std::vector<std::pair<std::vector<int>, T>> &peopleVector,
    const unsigned int part, const int partFirst, const int partLast,
    const int minimum) {
  // Count keypoints
  auto keypointCounter = 0;
  for (auto i = partFirst; i < partLast; ++i)
    keypointCounter += (peopleVector[part].first.at(i) > 0);
  // If enough keypoints --> subtract them and keep them at least as big as
  // minimum
  if (keypointCounter > minimum)
    personCounter +=
        minimum -
        keypointCounter; // personCounter = non-considered keypoints + minimum
}

template <typename T>
std::vector<std::pair<std::vector<int>, T>>
createPeopleVector(const T *const heatMapPtr, const T *const peaksPtr,
                   const Point<int> &heatMapSize, const int maxPeaks,
                   const T interThreshold, const T interMinAboveThreshold,
                   const std::vector<unsigned int> &bodyPartPairs,
                   const unsigned int numberBodyParts,
                   const unsigned int numberBodyPartPairs,
                   const Array<T> &pairScores) {
  // std::vector<std::pair<std::vector<int>, double>> refers to:
  //     - std::vector<int>: [body parts locations, #body parts found]
  //     - double: person subset score
  std::vector<std::pair<std::vector<int>, T>> peopleVector;
  const auto &mapIdx = std::vector<unsigned int>{
      0,  1,  14, 15, 22, 23, 16, 17, 18, 19, 24, 25, 26, 27, 6,  7,  2,  3,
      4,  5,  8,  9,  10, 11, 12, 13, 30, 31, 32, 33, 36, 37, 34, 35, 38, 39,
      20, 21, 28, 29, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51};
  const auto numberBodyPartsAndBkg = numberBodyParts + 1;
  const auto vectorSize = numberBodyParts + 1;
  const auto peaksOffset = 3 * (maxPeaks + 1);
  const auto heatMapOffset = heatMapSize.area();
  // Iterate over it PAF connection, e.g., neck-nose, neck-Lshoulder, etc.
  for (auto pairIndex = 0u; pairIndex < numberBodyPartPairs; ++pairIndex) {
    const auto bodyPartA = bodyPartPairs[2 * pairIndex];
    const auto bodyPartB = bodyPartPairs[2 * pairIndex + 1];
    const auto *candidateAPtr = peaksPtr + bodyPartA * peaksOffset;
    const auto *candidateBPtr = peaksPtr + bodyPartB * peaksOffset;
    const auto numberPeaksA = positiveIntRound(candidateAPtr[0]);
    const auto numberPeaksB = positiveIntRound(candidateBPtr[0]);
    // std::cout << "--------------" << std::endl;
    // std::cout << "bodyPartA: " << bodyPartA << " bodyPartB: " << bodyPartB
    // << std::endl; std::cout << "numberPeaksA: " << numberPeaksA << "
    // numberPeaksB: " << numberPeaksB << std::endl;

    // E.g., neck-nose connection. If one of them is empty (e.g., no noses
    // detected) Add the non-empty elements into the peopleVector
    if (numberPeaksA == 0 || numberPeaksB == 0) {
      // E.g., neck-nose connection. If no necks, add all noses
      // Change w.r.t. other
      if (numberPeaksA == 0) // numberPeaksB == 0 or not
      {
        // Non-MPI
        if (numberBodyParts != 15) {
          for (auto i = 1; i <= numberPeaksB; ++i) {
            bool found = false;
            for (const auto &personVector : peopleVector) {
              const auto off = (int)bodyPartB * peaksOffset + i * 3 + 2;
              if (personVector.first[bodyPartB] == off) {
                found = true;
                break;
              }
            }
            // Add new personVector with this element
            if (!found) {
              std::vector<int> rowVector(vectorSize, 0);
              // Store the index
              rowVector[bodyPartB] = bodyPartB * peaksOffset + i * 3 + 2;
              // Last number in each row is the parts number of that person
              rowVector.back() = 1;
              const auto personScore = candidateBPtr[i * 3 + 2];
              // Second last number in each row is the total score
              peopleVector.emplace_back(std::make_pair(rowVector, personScore));
            }
          }
        }
        // MPI
        else {
          for (auto i = 1; i <= numberPeaksB; ++i) {
            std::vector<int> rowVector(vectorSize, 0);
            // Store the index
            rowVector[bodyPartB] = bodyPartB * peaksOffset + i * 3 + 2;
            // Last number in each row is the parts number of that person
            rowVector.back() = 1;
            // Second last number in each row is the total score
            const auto personScore = candidateBPtr[i * 3 + 2];
            peopleVector.emplace_back(std::make_pair(rowVector, personScore));
          }
        }
      }
      // E.g., neck-nose connection. If no noses, add all necks
      else // if (numberPeaksA != 0 && numberPeaksB == 0)
      {
        // Non-MPI
        if (numberBodyParts != 15) {
          for (auto i = 1; i <= numberPeaksA; ++i) {
            bool found = false;
            const auto indexA = bodyPartA;
            for (const auto &personVector : peopleVector) {
              const auto off = (int)bodyPartA * peaksOffset + i * 3 + 2;
              if (personVector.first[indexA] == off) {
                found = true;
                break;
              }
            }
            if (!found) {
              std::vector<int> rowVector(vectorSize, 0);
              // Store the index
              rowVector[bodyPartA] = bodyPartA * peaksOffset + i * 3 + 2;
              // Last number in each row is the parts number of that person
              rowVector.back() = 1;
              // Second last number in each row is the total score
              const auto personScore = candidateAPtr[i * 3 + 2];
              peopleVector.emplace_back(std::make_pair(rowVector, personScore));
            }
          }
        }
        // MPI
        else {
          for (auto i = 1; i <= numberPeaksA; ++i) {
            std::vector<int> rowVector(vectorSize, 0);
            // Store the index
            rowVector[bodyPartA] = bodyPartA * peaksOffset + i * 3 + 2;
            // Last number in each row is the parts number of that person
            rowVector.back() = 1;
            // Second last number in each row is the total score
            const auto personScore = candidateAPtr[i * 3 + 2];
            peopleVector.emplace_back(std::make_pair(rowVector, personScore));
          }
        }
      }
    }
    // E.g., neck-nose connection. If necks and noses, look for maximums
    else // if (numberPeaksA != 0 && numberPeaksB != 0)
    {
      // (score, indexA, indexB). Inverted order for easy std::sort
      std::vector<std::tuple<double, int, int>> allABConnections;
      // Note: Problem of this function, if no right PAF between A and B, both
      // elements are discarded. However, they should be added indepently, not
      // discarded
      if (heatMapPtr != nullptr) {
        const auto *mapX =
            heatMapPtr +
            (numberBodyPartsAndBkg + mapIdx[2 * pairIndex]) * heatMapOffset;
        const auto *mapY =
            heatMapPtr +
            (numberBodyPartsAndBkg + mapIdx[2 * pairIndex + 1]) * heatMapOffset;
        // E.g., neck-nose connection. For each neck
        for (auto i = 1; i <= numberPeaksA; ++i) {
          // E.g., neck-nose connection. For each nose
          for (auto j = 1; j <= numberPeaksB; ++j) {
            // Initial PAF
            auto scoreAB =
                getScoreAB(i, j, candidateAPtr, candidateBPtr, mapX, mapY,
                           heatMapSize, interThreshold, interMinAboveThreshold);

            // E.g., neck-nose connection. If possible PAF between neck i,
            // nose j --> add parts score + connection score std::cout <<
            // "score of " << i << " and " << j << " : " << scoreAB <<
            // std::endl;
            if (scoreAB > 1e-6)
              allABConnections.emplace_back(std::make_tuple(scoreAB, i, j));
          }
        }
      } else if (!pairScores.empty()) {
        const auto firstIndex =
            (int)pairIndex * pairScores.getSize(1) * pairScores.getSize(2);
        // E.g., neck-nose connection. For each neck
        for (auto i = 0; i < numberPeaksA; ++i) {
          const auto iIndex = firstIndex + i * pairScores.getSize(2);
          // E.g., neck-nose connection. For each nose
          for (auto j = 0; j < numberPeaksB; ++j) {
            const auto scoreAB = pairScores[iIndex + j];

            // E.g., neck-nose connection. If possible PAF between neck i,
            // nose j --> add parts score + connection score
            if (scoreAB > 1e-6)
              // +1 because peaksPtr starts with counter
              allABConnections.emplace_back(
                  std::make_tuple(scoreAB, i + 1, j + 1));
          }
        }
      } else
        spdlog::error("Error. Should not reach here.", __LINE__, __FUNCTION__,
                      __FILE__);

      // select the top minAB connection, assuming that each part occur only
      // once sort rows in descending order based on parts + connection score
      if (!allABConnections.empty())
        std::sort(allABConnections.begin(), allABConnections.end(),
                  std::greater<std::tuple<double, int, int>>());

      std::vector<std::tuple<int, int, double>> abConnections; // (x, y, score)
      {
        const auto minAB = fastMin(numberPeaksA, numberPeaksB);
        std::vector<int> occurA(numberPeaksA, 0);
        std::vector<int> occurB(numberPeaksB, 0);
        auto counter = 0;
        for (const auto &aBConnection : allABConnections) {
          const auto score = std::get<0>(aBConnection);
          const auto indexA = std::get<1>(aBConnection);
          const auto indexB = std::get<2>(aBConnection);
          if (!occurA[indexA - 1] && !occurB[indexB - 1]) {
            abConnections.emplace_back(std::make_tuple(
                bodyPartA * peaksOffset + indexA * 3 + 2,
                bodyPartB * peaksOffset + indexB * 3 + 2, score));
            counter++;
            if (counter == minAB)
              break;
            occurA[indexA - 1] = 1;
            occurB[indexB - 1] = 1;
          }
        }
      }

      // Cluster all the body part candidates into peopleVector based on the
      // part connection
      if (!abConnections.empty()) {
        // initialize first body part connection 15&16
        if (pairIndex == 0) {
          for (const auto &abConnection : abConnections) {
            std::vector<int> rowVector(numberBodyParts + 3, 0);
            const auto indexA = std::get<0>(abConnection);
            const auto indexB = std::get<1>(abConnection);
            const auto score = std::get<2>(abConnection);
            rowVector[bodyPartPairs[0]] = indexA;
            rowVector[bodyPartPairs[1]] = indexB;
            rowVector.back() = 2;
            // add the score of parts and the connection
            const auto personScore =
                peaksPtr[indexA] + peaksPtr[indexB] + score;
            peopleVector.emplace_back(std::make_pair(rowVector, personScore));
          }
        }
        // Add ears connections (in case person is looking to opposite
        // direction to camera) Note: This has some issues:
        //     - It does not prevent repeating the same keypoint in different
        //     people
        //     - Assuming I have nose,eye,ear as 1 person subset, and whole
        //     arm as another one, it
        //       will not merge them both
        else if ((numberBodyParts == 18 &&
                  (pairIndex == 17 || pairIndex == 18)) ||
                 ((numberBodyParts == 19 || (numberBodyParts == 25) ||
                   numberBodyParts == 59 || numberBodyParts == 65) &&
                  (pairIndex == 18 || pairIndex == 19))) {
          for (const auto &abConnection : abConnections) {
            const auto indexA = std::get<0>(abConnection);
            const auto indexB = std::get<1>(abConnection);
            for (auto &personVector : peopleVector) {
              auto &personVectorA = personVector.first[bodyPartA];
              auto &personVectorB = personVector.first[bodyPartB];
              if (personVectorA == indexA && personVectorB == 0) {
                personVectorB = indexB;
                // // This seems to harm acc 0.1% for BODY_25
                // personVector.first.back()++;
              } else if (personVectorB == indexB && personVectorA == 0) {
                personVectorA = indexA;
                // // This seems to harm acc 0.1% for BODY_25
                // personVector.first.back()++;
              }
            }
          }
        } else {
          // A is already in the peopleVector, find its connection B
          for (const auto &abConnection : abConnections) {
            const auto indexA = std::get<0>(abConnection);
            const auto indexB = std::get<1>(abConnection);
            const auto score = T(std::get<2>(abConnection));
            bool found = false;
            for (auto &personVector : peopleVector) {
              // Found partA in a peopleVector, add partB to same one.
              if (personVector.first[bodyPartA] == indexA) {
                personVector.first[bodyPartB] = indexB;
                personVector.first.back()++;
                personVector.second += peaksPtr[indexB] + score;
                found = true;
                break;
              }
            }
            // Not found partA in peopleVector, add new peopleVector element
            if (!found) {
              std::vector<int> rowVector(vectorSize, 0);
              rowVector[bodyPartA] = indexA;
              rowVector[bodyPartB] = indexB;
              rowVector.back() = 2;
              const auto personScore =
                  peaksPtr[indexA] + peaksPtr[indexB] + score;
              peopleVector.emplace_back(std::make_pair(rowVector, personScore));
            }
          }
        }
      }
    }
  }
  return peopleVector;
}

template <typename T>
std::vector<std::tuple<T, T, int, int, int>>
pafPtrIntoVector(const Array<T> &pairScores, const T *const peaksPtr,
                 const int maxPeaks,
                 const std::vector<unsigned int> &bodyPartPairs,
                 const unsigned int numberBodyPartPairs) {
  // Result is a std::vector<std::tuple<double, double, int, int, int>> with:
  // (totalScore, PAFscore, pairIndex, indexA, indexB)
  // totalScore is first to simplify later sorting
  std::vector<std::tuple<T, T, int, int, int>> pairConnections;

  // Get all PAF pairs in a single std::vector
  const auto peaksOffset = 3 * (maxPeaks + 1);
  for (auto pairIndex = 0u; pairIndex < numberBodyPartPairs; ++pairIndex) {
    const auto bodyPartA = bodyPartPairs[2 * pairIndex];
    const auto bodyPartB = bodyPartPairs[2 * pairIndex + 1];
    const auto *candidateAPtr = peaksPtr + bodyPartA * peaksOffset;
    const auto *candidateBPtr = peaksPtr + bodyPartB * peaksOffset;
    const auto numberPeaksA = positiveIntRound(candidateAPtr[0]);
    const auto numberPeaksB = positiveIntRound(candidateBPtr[0]);
    const auto firstIndex =
        (int)pairIndex * pairScores.getSize(1) * pairScores.getSize(2);
    // E.g., neck-nose connection. For each neck
    for (auto indexA = 0; indexA < numberPeaksA; ++indexA) {
      const auto iIndex = firstIndex + indexA * pairScores.getSize(2);
      // E.g., neck-nose connection. For each nose
      for (auto indexB = 0; indexB < numberPeaksB; ++indexB) {
        const auto scoreAB = pairScores[iIndex + indexB];

        // E.g., neck-nose connection. If possible PAF between neck indexA,
        // nose indexB --> add parts score + connection score
        if (scoreAB > 1e-6) {
          // totalScore - Only used for sorting
          // // Original totalScore
          // const auto totalScore = scoreAB;
          // Improved totalScore
          // Improved to avoid too much weight in the PAF between 2 elements,
          // adding some weight on their confidence (avoid connecting high
          // PAFs on very low-confident keypoints)
          const auto indexScoreA =
              bodyPartA * peaksOffset + (indexA + 1) * 3 + 2;
          const auto indexScoreB =
              bodyPartB * peaksOffset + (indexB + 1) * 3 + 2;
          const auto totalScore = scoreAB + T(0.1) * peaksPtr[indexScoreA] +
                                  T(0.1) * peaksPtr[indexScoreB];
          // +1 because peaksPtr starts with counter
          pairConnections.emplace_back(std::make_tuple(
              totalScore, scoreAB, pairIndex, indexA + 1, indexB + 1));
        }
      }
    }
  }

  // Sort rows in descending order based on its first element (`totalScore`)
  if (!pairConnections.empty())
    std::sort(pairConnections.begin(), pairConnections.end(),
              std::greater<std::tuple<double, double, int, int, int>>());

  // Return result
  return pairConnections;
}

template <typename T>
std::vector<std::pair<std::vector<int>, T>> pafVectorIntoPeopleVector(
    const std::vector<std::tuple<T, T, int, int, int>> &pairConnections,
    const T *const peaksPtr, const int maxPeaks,
    const std::vector<unsigned int> &bodyPartPairs,
    const unsigned int numberBodyParts) {
  // std::vector<std::pair<std::vector<int>, double>> refers to:
  //     - std::vector<int>: [body parts locations, #body parts found]
  //     - double: person subset score
  std::vector<std::pair<std::vector<int>, T>> peopleVector;
  const auto vectorSize = numberBodyParts + 1;
  const auto peaksOffset = (maxPeaks + 1);
  // Save which body parts have been already assigned
  std::vector<int> personAssigned(numberBodyParts * maxPeaks, -1);
  std::set<int, std::greater<int>> indexesToRemoveSortedSet;
  // Iterate over each PAF pair connection detected
  // E.g., neck1-nose2, neck5-Lshoulder0, etc.
  for (const auto &pairConnection : pairConnections) {
    // Read pairConnection
    // // Total score - only required for previous sort
    // const auto totalScore = std::get<0>(pairConnection);
    const auto pafScore = std::get<1>(pairConnection);
    const auto pairIndex = std::get<2>(pairConnection);
    const auto indexA = std::get<3>(pairConnection);
    const auto indexB = std::get<4>(pairConnection);
    // Derived data
    const auto bodyPartA = bodyPartPairs[2 * pairIndex];
    const auto bodyPartB = bodyPartPairs[2 * pairIndex + 1];

    const auto indexScoreA = (bodyPartA * peaksOffset + indexA) * 3 + 2;
    const auto indexScoreB = (bodyPartB * peaksOffset + indexB) * 3 + 2;
    // -1 because indexA and indexB are 1-based
    auto &aAssigned = personAssigned[bodyPartA * maxPeaks + indexA - 1];
    auto &bAssigned = personAssigned[bodyPartB * maxPeaks + indexB - 1];

    // Different cases:
    //     1. A & B not assigned yet: Create new person
    //     2. A assigned but not B: Add B to person with A (if no another B
    //     there)
    //     3. B assigned but not A: Add A to person with B (if no another A
    //     there)
    //     4. A & B already assigned to same person (circular/redundant PAF):
    //     Update person score
    //     5. A & B already assigned to different people: Merge people if
    //     keypoint intersection is null
    // 1. A & B not assigned yet: Create new person
    if (aAssigned < 0 && bAssigned < 0) {
      // Keypoint indexes
      std::vector<int> rowVector(vectorSize, 0);
      rowVector[bodyPartA] = indexScoreA;
      rowVector[bodyPartB] = indexScoreB;
      // Number keypoints
      rowVector.back() = 2;
      // Score
      const auto personScore =
          peaksPtr[indexScoreA] + peaksPtr[indexScoreB] + pafScore;
      // Set associated personAssigned as assigned
      aAssigned = (int)peopleVector.size();
      bAssigned = aAssigned;
      // Create new personVector
      peopleVector.emplace_back(std::make_pair(rowVector, personScore));
    }
    // 2. A assigned but not B: Add B to person with A (if no another B there)
    // or
    // 3. B assigned but not A: Add A to person with B (if no another A there)
    else if ((aAssigned >= 0 && bAssigned < 0) ||
             (aAssigned < 0 && bAssigned >= 0)) {
      // Assign person1 to one where xAssigned >= 0
      const auto assigned1 = (aAssigned >= 0 ? aAssigned : bAssigned);
      auto &assigned2 = (aAssigned >= 0 ? bAssigned : aAssigned);
      const auto bodyPart2 = (aAssigned >= 0 ? bodyPartB : bodyPartA);
      const auto indexScore2 = (aAssigned >= 0 ? indexScoreB : indexScoreA);
      // Person index
      auto &personVector = peopleVector[assigned1];

      // If person with 1 does not have a 2 yet
      if (personVector.first[bodyPart2] == 0) {
        // Update keypoint indexes
        personVector.first[bodyPart2] = indexScore2;
        // Update number keypoints
        personVector.first.back()++;
        // Update score
        personVector.second += peaksPtr[indexScore2] + pafScore;
        // Set associated personAssigned as assigned
        assigned2 = assigned1;
      }
      // Otherwise, ignore this B because the previous one came from a higher
      // PAF-confident score
    }
    // 4. A & B already assigned to same person (circular/redundant PAF):
    // Update person score
    else if (aAssigned >= 0 && bAssigned >= 0 && aAssigned == bAssigned)
      peopleVector[aAssigned].second += pafScore;
    // 5. A & B already assigned to different people: Merge people if keypoint
    // intersection is null I.e., that the keypoints in person A and B do not
    // overlap
    else if (aAssigned >= 0 && bAssigned >= 0 && aAssigned != bAssigned) {
      // Assign person1 to the one with lowest index for 2 reasons:
      //     1. Speed up: Removing an element from std::vector is cheaper for
      //     latest elements
      //     2. Avoid harder index update: Updated elements in person1ssigned
      //     would depend on
      //        whether person1 > person2 or not: element = aAssigned -
      //        (person2 > person1 ? 1 : 0)
      const auto assigned1 = (aAssigned < bAssigned ? aAssigned : bAssigned);
      const auto assigned2 = (aAssigned < bAssigned ? bAssigned : aAssigned);
      auto &person1 = peopleVector[assigned1].first;
      const auto &person2 = peopleVector[assigned2].first;
      // Check if complementary
      // Defining found keypoint indexes in personA as kA, and analogously kB
      // Complementary if and only if kA intersection kB = empty. I.e., no
      // common keypoints
      bool complementary = true;
      for (auto part = 0u; part < numberBodyParts; ++part) {
        if (person1[part] > 0 && person2[part] > 0) {
          complementary = false;
          break;
        }
      }
      // If complementary, merge both people into 1
      if (complementary) {
        // Update keypoint indexes
        for (auto part = 0u; part < numberBodyParts; ++part)
          if (person1[part] == 0)
            person1[part] = person2[part];
        // Update number keypoints
        person1.back() += person2.back();
        // Update score
        peopleVector[assigned1].second +=
            peopleVector[assigned2].second + pafScore;
        // Erase the non-merged person
        // peopleVector.erase(peopleVector.begin()+assigned2); // x2 slower
        // when removing on-the-fly
        indexesToRemoveSortedSet.emplace(
            assigned2); // Add into set so we can remove them all at once
        // Update associated personAssigned (person indexes have changed)
        for (auto &element : personAssigned) {
          if (element == assigned2)
            element = assigned1;
          // No need because I will only remove them at the very end
          // else if (element > assigned2)
          //     element--;
        }
      }
    }
  }
  // Remove unused people
  for (const auto &index : indexesToRemoveSortedSet)
    peopleVector.erase(peopleVector.begin() + index);
  // Return result
  return peopleVector;
}

template <typename T>
void removePeopleBelowThresholdsAndFillFaces(
    std::vector<int> &validSubsetIndexes, int &numberPeople,
    std::vector<std::pair<std::vector<int>, T>> &peopleVector,
    const unsigned int numberBodyParts, const int minSubsetCnt,
    const T minSubsetScore, const bool maximizePositives,
    const T *const peaksPtr)
// const int minSubsetCnt, const T minSubsetScore, const int maxPeaks, const
// bool maximizePositives)
{
  // Delete people below the following thresholds:
  // a) minSubsetCnt: removed if less than minSubsetCnt body parts
  // b) minSubsetScore: removed if global score smaller than this
  // c) maxPeaks (POSE_MAX_PEOPLE): keep first maxPeaks people above
  // thresholds -> Not required
  numberPeople = 0;
  validSubsetIndexes.clear();
  // validSubsetIndexes.reserve(fastMin((size_t)maxPeaks,
  // peopleVector.size())); // maxPeaks is not required
  validSubsetIndexes.reserve(peopleVector.size());
  // Face valid sets
  std::vector<int> faceValidSubsetIndexes;
  faceValidSubsetIndexes.reserve(peopleVector.size());
  // Face invalid sets
  std::vector<int> faceInvalidSubsetIndexes;
  faceInvalidSubsetIndexes.reserve(peopleVector.size());
  // For each person candidate
  for (auto person = 0u; person < peopleVector.size(); ++person) {
    auto personCounter = peopleVector[person].first.back();
    // Analog for hand/face keypoints
    if (numberBodyParts >= 135) {
      // No consider face keypoints for personCounter
      const auto currentCounter = personCounter;
      getKeypointCounter(personCounter, peopleVector, person, 65, 135, 1);
      const auto newCounter = personCounter;
      if (personCounter == 1) {
        faceInvalidSubsetIndexes.emplace_back(person);
        continue;
      }
      // If body is still valid and facial points were removed, then add to
      // valid faces
      else if (currentCounter != newCounter)
        faceValidSubsetIndexes.emplace_back(person);
      // No consider right hand keypoints for personCounter
      getKeypointCounter(personCounter, peopleVector, person, 45, 65, 1);
      // No consider left hand keypoints for personCounter
      getKeypointCounter(personCounter, peopleVector, person, 25, 45, 1);
    }
    // Foot keypoints do not affect personCounter (too many false positives,
    // same foot usually appears as both left and right keypoints)
    // Pros: Removed tons of false positives
    // Cons: Standalone leg will never be recorded
    // Solution: No consider foot keypoints for that
    if (!maximizePositives && (numberBodyParts == 25 || numberBodyParts > 70)) {
      const auto currentCounter = personCounter;
      getKeypointCounter(personCounter, peopleVector, person, 19, 25, 0);
      const auto newCounter = personCounter;
      // Problem: Same leg/foot keypoints are considered for both left and
      // right keypoints. Solution: Remove legs that are duplicated and that
      // do not have upper torso Result: Slight increase in COCO mAP and
      // decrease in mAR + reducing a lot false positives!
      if (newCounter != currentCounter && newCounter <= 4)
        continue;
    }
    // Add only valid people
    const auto personScore = peopleVector[person].second;
    if (personCounter >= minSubsetCnt &&
        (personScore / personCounter) >= minSubsetScore) {
      numberPeople++;
      validSubsetIndexes.emplace_back(person);
      // // This is not required, it is OK if there are more people. No more
      // GPU memory used. if (numberPeople == maxPeaks)
      //     break;
    }
    // Sanity check
    else if ((personCounter < 1 && numberBodyParts != 25 &&
              numberBodyParts < 70) ||
             personCounter < 0)
      spdlog::error("Bad personCounter (" + std::to_string(personCounter) +
                        "). Bug in this"
                        " function if this happens.",
                    __LINE__, __FUNCTION__, __FILE__);
  }
  // If no people found --> Repeat with maximizePositives = true
  // Result: Increased COCO mAP because we catch more foot-only images
  if (numberPeople == 0 && !maximizePositives) {
    removePeopleBelowThresholdsAndFillFaces(
        validSubsetIndexes, numberPeople, peopleVector, numberBodyParts,
        minSubsetCnt, minSubsetScore, true, peaksPtr);
    // // Debugging
    // if (numberPeople > 0)
    //     log("Found " + std::to_string(numberPeople) + " people in second
    //     iteration");
  }
}

template <typename T>
void peopleVectorToPeopleArray(
    Array<T> &poseKeypoints, Array<T> &poseScores, const T scaleFactor,
    const std::vector<std::pair<std::vector<int>, T>> &peopleVector,
    const std::vector<int> &validSubsetIndexes, const T *const peaksPtr,
    const int numberPeople, const unsigned int numberBodyParts,
    const unsigned int numberBodyPartPairs) {
  // Allocate memory (initialized to 0)
  if (numberPeople > 0) {
    // Initialized to 0 for non-found keypoints in people
    poseKeypoints.reset({numberPeople, (int)numberBodyParts, 3}, 0.f);
    poseScores.reset(numberPeople);
  }
  // No people --> Empty Arrays
  else {
    poseKeypoints.reset();
    poseScores.reset();
  }
  // Fill people keypoints
  const auto oneOverNumberBodyPartsAndPAFs =
      1 / T(numberBodyParts + numberBodyPartPairs);
  // For each person
  for (auto person = 0u; person < validSubsetIndexes.size(); ++person) {
    const auto &personPair = peopleVector[validSubsetIndexes[person]];
    const auto &personVector = personPair.first;
    // For each body part
    for (auto bodyPart = 0u; bodyPart < numberBodyParts; ++bodyPart) {
      const auto baseOffset = (person * numberBodyParts + bodyPart) * 3;
      const auto bodyPartIndex = personVector[bodyPart];
      if (bodyPartIndex > 0) {
        poseKeypoints[baseOffset] = peaksPtr[bodyPartIndex - 2] * scaleFactor;
        poseKeypoints[baseOffset + 1] =
            peaksPtr[bodyPartIndex - 1] * scaleFactor;
        poseKeypoints[baseOffset + 2] = peaksPtr[bodyPartIndex];
      }
    }
    poseScores[person] = personPair.second * oneOverNumberBodyPartsAndPAFs;
  }
}

template <typename T>
void connectBodyPartsCpu(Array<T> &poseKeypoints, Array<T> &poseScores,
                         const T *const heatMapPtr, const T *const peaksPtr,
                         const Point<int> &heatMapSize, const int maxPeaks,
                         const T interMinAboveThreshold, const T interThreshold,
                         const int minSubsetCnt, const T minSubsetScore,
                         const T scaleFactor, const bool maximizePositives) {
  const auto &bodyPartPairs = std::vector<unsigned int>{
      1,  8,  1, 2,  1,  5,  2,  3,  3,  4,  5,  6,  6,  7,  8,  9,  9,  10,
      10, 11, 8, 12, 12, 13, 13, 14, 1,  0,  0,  15, 15, 17, 0,  16, 16, 18,
      2,  17, 5, 18, 14, 19, 19, 20, 14, 21, 11, 22, 22, 23, 11, 24};
  const auto numberBodyParts = 25;
  const auto numberBodyPartPairs = (unsigned int)(bodyPartPairs.size() / 2);
  if (numberBodyParts == 0)
    spdlog::error(
        "Invalid value of numberBodyParts, it must be positive, not " +
            std::to_string(numberBodyParts),
        __LINE__, __FUNCTION__, __FILE__);

  auto peopleVector =
      createPeopleVector(heatMapPtr, peaksPtr, heatMapSize, maxPeaks - 1,
                         interThreshold, interMinAboveThreshold, bodyPartPairs,
                         numberBodyParts, numberBodyPartPairs);

  // Delete people below the following thresholds:
  // a) minSubsetCnt: removed if less than minSubsetCnt body parts
  // b) minSubsetScore: removed if global score smaller than this
  // c) maxPeaks (POSE_MAX_PEOPLE): keep first maxPeaks people above
  // thresholds
  int numberPeople;
  std::vector<int> validSubsetIndexes;

  validSubsetIndexes.reserve(peopleVector.size());
  removePeopleBelowThresholdsAndFillFaces(
      validSubsetIndexes, numberPeople, peopleVector, numberBodyParts,
      minSubsetCnt, minSubsetScore, maximizePositives, peaksPtr);
  // Fill and return poseKeypoints
  peopleVectorToPeopleArray(poseKeypoints, poseScores, scaleFactor,
                            peopleVector, validSubsetIndexes, peaksPtr,
                            numberPeople, numberBodyParts, numberBodyPartPairs);
}

template void
connectBodyPartsCpu(Array<float> &poseKeypoints, Array<float> &poseScores,
                    const float *const heatMapPtr, const float *const peaksPtr,
                    const Point<int> &heatMapSize, const int maxPeaks,
                    const float interMinAboveThreshold,
                    const float interThreshold, const int minSubsetCnt,
                    const float minSubsetScore, const float scaleFactor,
                    const bool maximizePositives);
template void
connectBodyPartsCpu(Array<double> &poseKeypoints, Array<double> &poseScores,
                    const double *const heatMapPtr,
                    const double *const peaksPtr, const Point<int> &heatMapSize,
                    const int maxPeaks, const double interMinAboveThreshold,
                    const double interThreshold, const int minSubsetCnt,
                    const double minSubsetScore, const double scaleFactor,
                    const bool maximizePositives);

template std::vector<std::pair<std::vector<int>, float>> createPeopleVector(
    const float *const heatMapPtr, const float *const peaksPtr,
    const Point<int> &heatMapSize, const int maxPeaks,
    const float interThreshold, const float interMinAboveThreshold,
    const std::vector<unsigned int> &bodyPartPairs,
    const unsigned int numberBodyParts, const unsigned int numberBodyPartPairs,
    const Array<float> &precomputedPAFs);
template std::vector<std::pair<std::vector<int>, double>> createPeopleVector(
    const double *const heatMapPtr, const double *const peaksPtr,
    const Point<int> &heatMapSize, const int maxPeaks,
    const double interThreshold, const double interMinAboveThreshold,
    const std::vector<unsigned int> &bodyPartPairs,
    const unsigned int numberBodyParts, const unsigned int numberBodyPartPairs,
    const Array<double> &precomputedPAFs);

template void removePeopleBelowThresholdsAndFillFaces(
    std::vector<int> &validSubsetIndexes, int &numberPeople,
    std::vector<std::pair<std::vector<int>, float>> &peopleVector,
    const unsigned int numberBodyParts, const int minSubsetCnt,
    const float minSubsetScore, const bool maximizePositives,
    const float *const peaksPtr);
template void removePeopleBelowThresholdsAndFillFaces(
    std::vector<int> &validSubsetIndexes, int &numberPeople,
    std::vector<std::pair<std::vector<int>, double>> &peopleVector,
    const unsigned int numberBodyParts, const int minSubsetCnt,
    const double minSubsetScore, const bool maximizePositives,
    const double *const peaksPtr);

template void peopleVectorToPeopleArray(
    Array<float> &poseKeypoints, Array<float> &poseScores,
    const float scaleFactor,
    const std::vector<std::pair<std::vector<int>, float>> &peopleVector,
    const std::vector<int> &validSubsetIndexes, const float *const peaksPtr,
    const int numberPeople, const unsigned int numberBodyParts,
    const unsigned int numberBodyPartPairs);
template void peopleVectorToPeopleArray(
    Array<double> &poseKeypoints, Array<double> &poseScores,
    const double scaleFactor,
    const std::vector<std::pair<std::vector<int>, double>> &peopleVector,
    const std::vector<int> &validSubsetIndexes, const double *const peaksPtr,
    const int numberPeople, const unsigned int numberBodyParts,
    const unsigned int numberBodyPartPairs);

template std::vector<std::tuple<float, float, int, int, int>>
pafPtrIntoVector(const Array<float> &pairScores, const float *const peaksPtr,
                 const int maxPeaks,
                 const std::vector<unsigned int> &bodyPartPairs,
                 const unsigned int numberBodyPartPairs);
template std::vector<std::tuple<double, double, int, int, int>>
pafPtrIntoVector(const Array<double> &pairScores, const double *const peaksPtr,
                 const int maxPeaks,
                 const std::vector<unsigned int> &bodyPartPairs,
                 const unsigned int numberBodyPartPairs);

template std::vector<std::pair<std::vector<int>, float>>
pafVectorIntoPeopleVector(
    const std::vector<std::tuple<float, float, int, int, int>> &pairConnections,
    const float *const peaksPtr, const int maxPeaks,
    const std::vector<unsigned int> &bodyPartPairs,
    const unsigned int numberBodyParts);
template std::vector<std::pair<std::vector<int>, double>>
pafVectorIntoPeopleVector(
    const std::vector<std::tuple<double, double, int, int, int>>
        &pairConnections,
    const double *const peaksPtr, const int maxPeaks,
    const std::vector<unsigned int> &bodyPartPairs,
    const unsigned int numberBodyParts);

} // namespace openposert
