/*
  Parker Cai
  February 19, 2026
  CS5330 - Project 3: Real-time 2-D Object Recognition

  Connected components analysis and region segmentation
*/

#include "or2d.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cmath>
#include <unordered_map>


// Simple hard-coded pastel color palette for region visualization (BGR format)
static cv::Vec3b colorForLabel(int label) {
  static const cv::Vec3b COLORS[] = {
    {204, 197, 102}, // pastel cyan     #66c5cc
    {113, 207, 246}, // pastel yellow   #f6cf71
    {116, 156, 248}, // pastel orange   #f89c74
    {242, 176, 220}, // pastel lavender #dcb0f2
    { 95, 197, 135}, // pastel green    #87c55f
    {243, 185, 158}, // pastel blue     #9eb9f3
    {177, 136, 254}, // pastel pink     #fe88b1
    {116, 219, 201}, // pastel lime     #c9db74
    {164, 224, 139}, // pastel mint     #8be0a4
  };
  return COLORS[label % 9];
}


/*
  Build a color-coded region image without any overlays (no bbox, no centroid).
  Used as a clean base for the features display mode to draw features on top of the colored regions.
*/
cv::Mat colorizeRegions(const cv::Mat& labelMap, const std::vector<RegionInfo>& regions) {
  // Assign color to the region pixels (single pass)
  // build a label:color map for faster color lookup
  std::unordered_map<int, cv::Vec3b> labelColor;
  for (const auto& region : regions) {
    labelColor[region.label] = region.color;
  }
  cv::Mat result = cv::Mat::zeros(labelMap.size(), CV_8UC3); // Color-coded display image
  // Iterate through the label image and assign colors
  for (int r = 0; r < labelMap.rows; r++) {
    for (int c = 0; c < labelMap.cols; c++) {
      int label = labelMap.at<int>(r, c);
      auto it = labelColor.find(label);
      if (it != labelColor.end()) {
        result.at<cv::Vec3b>(r, c) = it->second; // assign color based on label color map
      }
    }
  }
  return result;
}


/*
  Segment binary image into regions using connected components analysis.

  Steps:
    1. Run OpenCV's connectedComponentsWithStats to get labels, stats, and centroids
    2. Filter out small regions and the ones touching the borders
    3. Sort remaining regions by area and keep only top maxRegions (3 largest)
    4. Assign colors based on label and create a color-coded result image
    5. Draw bounding boxes and centroids on the result image
*/
cv::Mat segmentRegions(
  const cv::Mat& binary,
  std::vector<RegionInfo>& regions,
  cv::Mat& labelMap,
  int minSize,
  int maxRegions) {
  // Clear output regions vector
  regions.clear();

  // Run OpenCV's connected components with stats
  cv::Mat stats, centroids;
  int numLabels = cv::connectedComponentsWithStats(binary, labelMap, stats, centroids, 4, CV_32S, cv::CCL_DEFAULT);
  // CCL_DEFAULT (Spaghetti4C) — Bolelli et al. 2021. Two-pass union-find for merging labels + DAG-based decision trees 
  // CCL_WU (SAUF) — Wu et al. 2009. Classic two-pass with array-based union-find. 

  // Build candidate region list
  std::vector<RegionInfo> candidates;
  // (skip label 0 = background)
  for (int i = 1; i < numLabels; i++) {
    int area = stats.at<int>(i, cv::CC_STAT_AREA);

    // Ignore regions smaller than minSize (default 20x20 pixels = 400 px area)
    if (area < minSize) {
      continue;
    }

    int left = stats.at<int>(i, cv::CC_STAT_LEFT);
    int top = stats.at<int>(i, cv::CC_STAT_TOP);
    int width = stats.at<int>(i, cv::CC_STAT_WIDTH);
    int height = stats.at<int>(i, cv::CC_STAT_HEIGHT);

    // Skip regions touching the image border
    if (left == 0 || top == 0 ||
      left + width >= binary.cols ||
      top + height >= binary.rows) {
      continue;
    }

    // Get centroid
    cv::Point2f centroid;
    centroid.x = centroids.at<double>(i, 0);
    centroid.y = centroids.at<double>(i, 1);

    // Add candidate region info to the list
    RegionInfo candidate;
    candidate.label = i;
    candidate.centroid = centroid;
    candidate.bbox = cv::Rect(left, top, width, height);
    candidate.area = area;
    candidate.color = { 0, 0, 0 };  // init to black, will assign color later

    candidates.push_back(candidate);
  }

  // Sort candidates by area in descending order
  std::sort(candidates.begin(), candidates.end(),
    // lambda comparator for sorting by area
    [](const RegionInfo& a, const RegionInfo& b) {
      return a.area > b.area;
    });

  // Filter out only top maxRegions (default 3) largest regions
  if (candidates.size() > (size_t)maxRegions) {
    candidates.resize(maxRegions);
  }

  // Assign colors with centroid matching against previous frame regions
  // use static variables to persist between calls
  static std::vector<RegionInfo> prevRegions;
  static int nextColorIdx = 0;
  // max allowed centroid match distance squared: dx^2 + dy^2 < 50^2 pixels
  float maxMatchDist = 50.0f;
  std::vector<uchar> prevUsed(prevRegions.size(), 0); // to track which previous regions have been matched

  for (RegionInfo& candidate : candidates) {
    int bestIdx = -1;
    float bestDistSq = maxMatchDist * maxMatchDist;

    for (int i = 0; i < static_cast<int>(prevRegions.size()); i++) {
      // Skip already matched previous regions
      if (prevUsed[i]) {
        continue;
      }
      // Compute squared distance between centroids
      const float dx = candidate.centroid.x - prevRegions[i].centroid.x;
      const float dy = candidate.centroid.y - prevRegions[i].centroid.y;
      const float distSq = dx * dx + dy * dy;
      // Check if this previous region is a better match
      if (distSq < bestDistSq) {
        bestDistSq = distSq;
        bestIdx = i;
      }
    }
    // Assign color 
    if (bestIdx != -1) { // found a match, reuse color
      candidate.color = prevRegions[bestIdx].color;
      prevUsed[bestIdx] = 1; // mark this previous region as matched
    }
    else { // no match, assign new color based on label
      candidate.color = colorForLabel(nextColorIdx++);
    }

    regions.push_back(candidate);
  }
  prevRegions = regions;

  // Build color-coded region image, then draw AABB + centroids on top
  cv::Mat result = colorizeRegions(labelMap, regions);

  // Draw bounding boxes and centroids
  for (const auto& region : regions) {
    // draw bounding box
    cv::rectangle(result, region.bbox, cv::Scalar(255, 255, 255), 2);
    // draw centroid as a white circle with radius 5
    cv::circle(result, cv::Point(static_cast<int>(region.centroid.x),
      static_cast<int>(region.centroid.y)),
      5, cv::Scalar(255, 255, 255), -1);
  }

  return result;
}
