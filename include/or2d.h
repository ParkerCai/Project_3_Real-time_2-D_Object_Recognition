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

// Region info struct for storing segmentation results
struct RegionInfo {
  int label;
  cv::Point2f centroid;
  cv::Rect bbox;
  int area;
  cv::Vec3b color;
};

/**
  @brief Segment binary image into regions using connected components analysis.
  @param binary input binary image (CV_8U, single channel)
  @param regions output vector of RegionInfo structs for each detected region
  @param minSize minimum area (in pixels) for a region to be considered valid (default 400 px = 20x20)
  @param maxRegions maximum number of regions to keep based on area (default 3)
  @return Color-coded image with bounding boxes and centroids drawn for each detected region
*/
cv::Mat segmentRegions(const cv::Mat& binary,
  std::vector<RegionInfo>& regions,
  int minSize = 400, // min: 20x20 pixels area
  int maxRegions = 3); // max: 3 objects in the frame to recognize

#endif // OR2D_H