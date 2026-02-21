/*
  Parker Cai
  Jenny Nguyen
  February 19, 2026
  CS5330 - Project 3: Real-time 2-D Object Recognition

  Compute features for each segmented region for classification.
*/

#include "or2d.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>


/*
  Compute features for a single region using region-based analysis.

  Steps:
    1. Extracts a binary mask for the region (labelMap == region.label)
    2. Computes moments via cv::moments() on the binary mask
    3. Derives the principal axis angle (axis of least central moment)
    4. Projects all region pixels onto the principal and perpendicular axes
        to compute the oriented bounding box (OBB) region-based
    5. Computes percent filled = region area / OBB area
    6. Computes OBB aspect ratio = min(w,h) / max(w,h)  (always in [0,1])
    7. Computes Hu moment invariants via cv::HuMoments()
    8. Assembles a feature vector: {percentFilled, bboxRatio, log|hu0|, log|hu1|}
*/
void computeRegionFeatures(const cv::Mat& labelMap, RegionInfo& region) {
  // 1. Extract binary mask for this region from the label map
  cv::Mat mask = cv::Mat::zeros(labelMap.size(), CV_8U);
  for (int r = 0; r < labelMap.rows; r++) {
    for (int c = 0; c < labelMap.cols; c++) {
      if (labelMap.at<int>(r, c) == region.label) {
        mask.at<uchar>(r, c) = 255;
      }
    }
  }

  // 2. Compute moments on the binary mask
  //    cv::moments(mask, true) computes:
  //      Spatial moments:  m00, m10, m01, m20, m11, m02, m30, m21, m12, m03
  //      Central moments:  mu20, mu11, mu02, mu30, mu21, mu12, mu03
  //      Normalized central moments: nu20, nu11, nu02, nu30, nu21, nu12, nu03
  //    binaryImage=true treats any nonzero pixel as 1 (no weighting by intensity).
  cv::Moments m = cv::moments(mask, true);

  // 3. Axis of least central moment (principal orientation)
  //    theta = 0.5 * atan2(2 * mu11, mu20 - mu02)
  //    This angle minimizes the moment of inertia about the axis through the centroid.
  region.theta = 0.5f * static_cast<float>(std::atan2(2.0 * m.mu11, m.mu20 - m.mu02));

  // 4. Oriented Bounding Box (OBB) via region pixel projection
  //    Project each region pixel onto the principal axis (u) and perpendicular axis (v)
  //    relative to the centroid. Track min/max of u and v to get the OBB extents.
  float cosT = std::cos(region.theta);
  float sinT = std::sin(region.theta);
  float uMin = std::numeric_limits<float>::max();
  float uMax = std::numeric_limits<float>::lowest();
  float vMin = std::numeric_limits<float>::max();
  float vMax = std::numeric_limits<float>::lowest();

  // Iterate through the mask pixels to find the OBB extents in the rotated coordinate frame
  for (int r = 0; r < mask.rows; r++) {
    for (int c = 0; c < mask.cols; c++) {
      if (mask.at<uchar>(r, c) == 0) continue;
      float dx = c - region.centroid.x;
      float dy = r - region.centroid.y;
      // Project onto principal axis (u) and perpendicular (v)
      float u = dx * cosT + dy * sinT;
      float v = -dx * sinT + dy * cosT;
      uMin = std::min(uMin, u);
      uMax = std::max(uMax, u);
      vMin = std::min(vMin, v);
      vMax = std::max(vMax, v);
    }
  }

  float obbWidth = uMax - uMin;
  float obbHeight = vMax - vMin;

  // Construct cv::RotatedRect from centroid, size, and angle (degrees)
  // The center of the OBB may be slightly offset from the centroid if the region
  // is not symmetric, so compute it from the midpoint of the projections.
  float obbCenterU = (uMin + uMax) / 2.0f;
  float obbCenterV = (vMin + vMax) / 2.0f;
  float obbCx = region.centroid.x + obbCenterU * cosT - obbCenterV * sinT;
  float obbCy = region.centroid.y + obbCenterU * sinT + obbCenterV * cosT;

  // cv::RotatedRect angle is in degrees; theta is in radians
  region.orientedBBox = cv::RotatedRect(
    cv::Point2f(obbCx, obbCy),
    cv::Size2f(obbWidth, obbHeight),
    region.theta * 180.0f / static_cast<float>(CV_PI)
  );

  // 5. Percent filled = region area / OBB area
  //    Measures how much of the OBB is occupied by the region.
  //    Invariant to translation, scale, and rotation.
  float obbArea = obbWidth * obbHeight;
  region.percentFilled = (obbArea > 0) ? (static_cast<float>(region.area) / obbArea) : 0.0f;

  // 6. OBB aspect ratio = min(w,h) / max(w,h), always in [0, 1]
  //    A square gives 1.0; a long thin shape gives near 0.
  //    Invariant to translation, scale, and rotation.
  float minDim = std::min(obbWidth, obbHeight);
  float maxDim = std::max(obbWidth, obbHeight);
  region.bboxRatio = (maxDim > 0) ? (minDim / maxDim) : 0.0f;

  // 7. Hu moment invariants
  //    cv::HuMoments() computes 7 moments from the normalized central moments.
  //    hu[0] through hu[6] are invariant to translation, scale, and rotation.
  //    hu[6] also changes sign under reflection (skew invariant).
  cv::HuMoments(m, region.huMoments);

  // 8. Assemble feature vector
  //    Using log10(|hu[i]|) for manageable magnitude values.
  //    Feature vector: {percentFilled, bboxRatio, log|hu0|, log|hu1|}
  //    This gives 4 features; can be extended for later tasks.
  region.featureVector.clear();
  region.featureVector.push_back(static_cast<double>(region.percentFilled));
  region.featureVector.push_back(static_cast<double>(region.bboxRatio));
  for (int i = 0; i < 2; i++) {
    double absHu = std::abs(region.huMoments[i]);
    // Use log10 for scale, with a floor to avoid log(0)
    double logHu = (absHu > 1e-10) ? std::log10(absHu) : -10.0;
    region.featureVector.push_back(logHu);
  }
}


/*
  Draw feature overlays on an image for all regions.

    - The oriented bounding box (OBB) as a white polygon
    - The principal axis (axis of least central moment) as a blue line
    - Feature values (percent filled, aspect ratio) as text near the centroid
*/
void drawFeatures(cv::Mat& image, const std::vector<RegionInfo>& regions) {
  for (const auto& region : regions) {
    // Draw oriented bounding box (OBB)
    cv::Point2f corners[4];
    region.orientedBBox.points(corners);
    for (int i = 0; i < 4; i++) {
      cv::line(image, corners[i], corners[(i + 1) % 4],
        cv::Scalar(255, 255, 255), 2); // white
    }

    // Draw principal axis
    // Extend a line through the centroid along the principal axis direction
    // Use half the OBB length along the primary axis for line extent
    float halfLen = region.orientedBBox.size.width / 2.0f;
    float cosT = std::cos(region.theta);
    float sinT = std::sin(region.theta);
    cv::Point2f p1(region.centroid.x - halfLen * cosT,
      region.centroid.y - halfLen * sinT);
    cv::Point2f p2(region.centroid.x + halfLen * cosT,
      region.centroid.y + halfLen * sinT);
    cv::line(image, p1, p2, cv::Scalar(255, 0, 0), 2); // blue

    // Draw centroid dot
    cv::circle(image, cv::Point(static_cast<int>(region.centroid.x),
      static_cast<int>(region.centroid.y)),
      4, cv::Scalar(0, 0, 255), -1); // red

    // Draw feature text above the topmost OBB corner
    float topY = corners[0].y;
    for (int i = 1; i < 4; i++) topY = std::min(topY, corners[i].y);

    int textX = std::clamp((int)region.centroid.x - 60, 5, image.cols - 200);
    int textY = std::clamp((int)topY - 25, 20, image.rows - 10);

    cv::putText(image, std::format("Fill: {:.1f}%", region.percentFilled * 100.0f),
      cv::Point(textX, textY), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 255), 1);
    cv::putText(image, std::format("Ratio: {:.2f}", region.bboxRatio),
      cv::Point(textX, textY + 18), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 255), 1);
  }
}
