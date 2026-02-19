/*
  Jenny Nguyen
  Parker Cai
  February 16, 2026
  CS5330 - Project 3: Real-time 2-D Object Recognition

  Header file with function declarations
*/

#ifndef OR2D_H
#define OR2D_H

#include <opencv2/opencv.hpp>
#include <vector>

cv::Mat thresholdImage(const cv::Mat& input, int threshValue = -1);

cv::Mat cleanupBinary(const cv::Mat& binary);
cv::Mat erode(const cv::Mat& src);
cv::Mat dilate(const cv::Mat& src);


#endif // OR2D_H