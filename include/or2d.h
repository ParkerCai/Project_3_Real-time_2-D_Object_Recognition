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

// Region info struct for storing segmentation results and features
struct RegionInfo {
  int label;
  cv::Point2f centroid;
  cv::Rect bbox; // axis-aligned bounding box
  int area;
  cv::Vec3b color;
  float theta; // angle of axis of least central moment (radians)
  cv::RotatedRect orientedBBox; // oriented bounding box (OBB) along the primary axis
  float bboxRatio; // min(w,h) : max(w,h) of the OBB
  float percentFilled; // = region area / OBB area
  double huMoments[7]; // 7 moments from Hu moment invariants
  std::vector<double> featureVector; // assembled feature vector for classification
};

/**
  @brief Segment binary image into regions using connected components analysis.
  @param binary input binary image (CV_8U, single channel)
  @param regions output vector of RegionInfo structs for each detected region
  @param labelMap output label map (CV_32S) from connected components (region ID per pixel)
  @param minSize minimum area (in pixels) for a region to be considered valid (default 400 px = 20x20)
  @param maxRegions maximum number of regions to keep based on area (default 3)
  @return Color-coded image with bounding boxes and centroids drawn for each detected region
*/
cv::Mat segmentRegions(const cv::Mat& binary,
  std::vector<RegionInfo>& regions,
  cv::Mat& labelMap,
  int minSize = 400, // min: 20x20 pixels area
  int maxRegions = 3); // max: 3 objects in the frame to recognize

#endif // OR2D_H