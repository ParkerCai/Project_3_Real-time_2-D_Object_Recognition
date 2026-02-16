/*
  Jenny Nguyen
  February 16, 2026
  CS5330 - Project 3: Real-time 2-D Object Recognition
   
*/

#ifndef OR2D_H
#define OR2D_H

#include <opencv2/opencv.hpp>
#include <vector>

// Main threshold function with ISODATA if threshValue is -1
cv::Mat thresholdImage(const cv::Mat& input, int threshValue = -1);

// Simple version with fixed threshold
cv::Mat simpleThreshold(const cv::Mat& input, int threshValue);


#endif // OR2D_H