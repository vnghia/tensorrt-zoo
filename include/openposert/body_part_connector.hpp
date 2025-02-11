#pragma once

#include <iostream>
#include <tuple>
#include <vector>

#include "openposert/array.hpp"
#include "openposert/point.hpp"

namespace openposert {

template <typename T>
void connectBodyPartsCpu(Array<T>& poseKeypoints, Array<T>& poseScores,
                         const T* const heatMapPtr, const T* const peaksPtr,
                         const Point<int>& heatMapSize, const int maxPeaks,
                         const T interMinAboveThreshold, const T interThreshold,
                         const int minSubsetCnt, const T minSubsetScore,
                         const T scaleFactor = 1.f,
                         const bool maximizePositives = false);

// Windows: Cuda functions do not include OP_API
template <typename T>
void connectBodyPartsGpu(Array<T>& poseKeypoints, Array<T>& poseScores,
                         const T* const heatMapGpuPtr, const T* const peaksPtr,
                         const Point<int>& heatMapSize, const int maxPeaks,
                         const T interMinAboveThreshold, const T interThreshold,
                         const int minSubsetCnt, const T minSubsetScore,
                         const T scaleFactor, const bool maximizePositives,
                         Array<T> pairScoresCpu, T* pairScoresGpuPtr,
                         const unsigned int* const bodyPartPairsGpuPtr,
                         const unsigned int* const mapIdxGpuPtr,
                         const T* const peaksGpuPtr,
                         const T defaultNmsThreshold);

// Private functions used by the 2 above functions
template <typename T>
std::vector<std::pair<std::vector<int>, T>> createPeopleVector(
    const T* const heatMapPtr, const T* const peaksPtr,
    const Point<int>& heatMapSize, const int maxPeaks, const T interThreshold,
    const T interMinAboveThreshold,
    const std::vector<unsigned int>& bodyPartPairs,
    const unsigned int numberBodyParts, const unsigned int numberBodyPartPairs,
    const Array<T>& precomputedPAFs = Array<T>());

template <typename T>
void removePeopleBelowThresholdsAndFillFaces(
    std::vector<int>& validSubsetIndexes, int& numberPeople,
    std::vector<std::pair<std::vector<int>, T>>& subsets,
    const unsigned int numberBodyParts, const int minSubsetCnt,
    const T minSubsetScore, const bool maximizePositives,
    const T* const peaksPtr);

template <typename T>
void peopleVectorToPeopleArray(
    Array<T>& poseKeypoints, Array<T>& poseScores, const T scaleFactor,
    const std::vector<std::pair<std::vector<int>, T>>& subsets,
    const std::vector<int>& validSubsetIndexes, const T* const peaksPtr,
    const int numberPeople, const unsigned int numberBodyParts,
    const unsigned int numberBodyPartPairs);

template <typename T>
std::vector<std::tuple<T, T, int, int, int>> pafPtrIntoVector(
    const Array<T>& pairScores, const T* const peaksPtr, const int maxPeaks,
    const std::vector<unsigned int>& bodyPartPairs,
    const unsigned int numberBodyPartPairs);

template <typename T>
std::vector<std::pair<std::vector<int>, T>> pafVectorIntoPeopleVector(
    const std::vector<std::tuple<T, T, int, int, int>>& pairScores,
    const T* const peaksPtr, const int maxPeaks,
    const std::vector<unsigned int>& bodyPartPairs,
    const unsigned int numberBodyParts);

}  // namespace openposert
