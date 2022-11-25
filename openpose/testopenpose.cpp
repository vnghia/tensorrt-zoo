/*
 * @Description: In User Settings Edit
 * @Author: your name
 * @Date: 2019-08-21 09:45:10
 * @LastEditTime: 2019-12-04 19:27:27
 * @LastEditors: zerollzeng
 */
#include <string>
#include <vector>

#include "OpenPose.hpp"
#include "opencv2/opencv.hpp"
#include "time.h"

int main(int argc, char** argv) {
  cv::Mat img = cv::imread(argv[1]);
  if (img.empty()) {
    std::cout << "error: can not read image" << std::endl;
    return -1;
  }
  int W = std::atoi(argv[2]);
  int H = std::atoi(argv[3]);
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
  cv::resize(img, img, cv::Size(W, H));

  int N = 1;
  int C = 3;
  std::vector<float> inputData;
  inputData.resize(N * C * H * W);

  unsigned char* data = img.data;
  for (int n = 0; n < N; n++) {
    for (int c = 0; c < 3; c++) {
      for (int i = 0; i < W * H; i++) {
        inputData[i + c * W * H + n * 3 * H * W] = (float)data[i * 3 + c];
      }
    }
  }
  std::vector<float> result;

  OpenPose* openpose = new OpenPose(argv[4]);

  clock_t start = clock();
  openpose->DoInference(inputData, result);
  clock_t end = clock();
  std::cout << "inference Time : "
            << ((double)(end - start) / CLOCKS_PER_SEC) * 1000 << " ms"
            << std::endl;

  cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
  for (size_t i = 0; i < result.size() / 3; i++) {
    cv::circle(img, cv::Point(result[i * 3], result[i * 3 + 1]), 2,
               cv::Scalar(0, 255, 0), -1);
  }
  cv::imwrite("result.jpg", img);

  return 0;
}
