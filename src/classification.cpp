/*
  Jenny Nguyen
  Parker Cai
  February 19, 2026
  CS5330 - Project 3: Real-time 2-D Object Recognition

  Classification- Identify objects from the training data
  Uses nearest neighbor classification with a distance-based accuracy score.
*/

#include "or2d.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <cmath>
#include <limits>

constexpr double INF = std::numeric_limits<double>::infinity();
constexpr float INF_F = std::numeric_limits<float>::infinity();

/*
  Calculate standard deviations for each feature
  Need this for scaled euclidean distance
*/
std::vector<double> computeStdDevs(const std::vector<std::vector<double>>& features) {
  if (features.empty()) {
    return std::vector<double>();
  }

  int num_dims = features[0].size();
  std::vector<double> means(num_dims, 0.0);
  std::vector<double> stds(num_dims, 0.0);

  // get means
  for (const auto& fvec : features) {
    for (int i = 0; i < num_dims; i++) {
      means[i] += fvec[i];
    }
  }
  for (int i = 0; i < num_dims; i++) {
    means[i] /= features.size();
  }

  // get std devs
  for (const auto& fvec : features) {
    for (int i = 0; i < num_dims; i++) {
      double diff = fvec[i] - means[i];
      stds[i] += diff * diff;
    }
  }
  for (int i = 0; i < num_dims; i++) {
    stds[i] = std::sqrt(stds[i] / features.size());
    if (stds[i] < 0.0001) stds[i] = 1.0;  // avoid divide by zero
  }

  return stds;
}

//Scaled euclidean distance
double scaledEuclideanDistance(
  const std::vector<double>& f1,
  const std::vector<double>& f2,
  const std::vector<double>& stds) {
  if (f1.size() != f2.size()) return INF;

  double sum = 0.0;
  for (size_t i = 0; i < f1.size(); i++) {
    double diff = (f1[i] - f2[i]) / stds[i];
    sum += diff * diff;
  }

  return std::sqrt(sum);
}

// Use nearest neighbor to find closest match 
std::string classifyObject(const std::vector<double>& query,
  const std::vector<std::string>& train_labels,
  const std::vector<std::vector<double>>& train_features,
  double& acccuracy) {
  if (train_labels.empty()) {
    acccuracy = 0.0;
    return "unknown";
  }

  // get standard deviations
  std::vector<double> stds = computeStdDevs(train_features);

  // find nearest neighbor
  double min_dist = INF;
  int best_idx = -1;

  for (size_t i = 0; i < train_features.size(); i++) {
    double dist = scaledEuclideanDistance(query, train_features[i], stds);
    if (dist < min_dist) {
      min_dist = dist;
      best_idx = i;
    }
  }

  if (best_idx == -1) {
    acccuracy = 0.0;
    return "unknown";
  }

  // accuracy based on distance
  acccuracy = 1.0 / (1.0 + min_dist);

  return train_labels[best_idx];
}

// Classify all regions and draw labels on image
void classifyAndLabel(cv::Mat& image,
  std::vector<RegionInfo>& regions,
  const std::vector<std::string>& train_labels,
  const std::vector<std::vector<double>>& train_features) {
  if (train_labels.empty()) {
    cv::putText(image, "No training data", cv::Point(10, 60),
      cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    return;
  }

  for (auto& region : regions) {
    double acc;
    std::string label = classifyObject(region.featureVector,
      train_labels,
      train_features,
      acc);

    // Position label below the OBB
    cv::Point2f corners[4];
    region.orientedBBox.points(corners);
    float botY = corners[0].y;
    for (int i = 1; i < 4; i++) botY = std::max(botY, corners[i].y);

    int x = std::clamp((int)region.centroid.x - 40, 5, image.cols - 200);
    int y = std::clamp((int)botY + 25, 25, image.rows - 10);

    cv::putText(image, label, cv::Point(x, y),
      cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);

    // draw accuracy as percentage
    int acc_pct = (int)(acc * 100);
    std::string acc_text = std::to_string(acc_pct) + "%";
    cv::putText(image, acc_text, cv::Point(x, y + 22),
      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
  }
}


/*
  Sum of Squared Difference (SSD)
  - Compute with L-2 norm: Euclidean distance
    (but we don't care about the sqrt due to monotonicity)
  - SSD or SSE formula: d(a, b) = Σ (aᵢ - bᵢ)²

  Input:
    featuresA - first feature vector (std::vector<float>)
    featuresB - second feature vector (std::vector<float>)

  Output:
    float - SSD value
*/
float sumOfSquaredDifference(const std::vector<float>& featuresA,
  const std::vector<float>& featuresB) {
  // Check for size mismatch
  if (featuresA.size() != featuresB.size()) return INF_F;

  float sum = 0.0f;

  for (size_t i = 0; i < featuresA.size(); i++) {
    // x*x is faster than pow(x, 2)
    sum += (featuresA[i] - featuresB[i]) * (featuresA[i] - featuresB[i]);
  }

  return sum;
}


/*
  One-shot classification using CNN embeddings using SSD as distance metric.
  Uses float for native DNN precision.
*/
std::string classifyObjectCNN(const std::vector<float>& query,
  const std::vector<std::string>& train_labels,
  const std::vector<std::vector<float>>& train_features,
  float& accuracy) {
  if (train_labels.empty() || query.empty()) {
    accuracy = 0.0f;
    return "unknown";
  }

  float min_dist = INF_F;
  int best_idx = -1;

  for (size_t i = 0; i < train_features.size(); i++) {
    float dist = sumOfSquaredDifference(query, train_features[i]);
    if (dist < min_dist) {
      min_dist = dist;
      best_idx = i;
    }
  }

  if (best_idx == -1) {
    accuracy = 0.0f;
    return "unknown";
  }

  // Normalize SSD by dimensionality to get per-dimension average distance
  float avg_dist = min_dist / query.size();
  // then convert to confidence score
  accuracy = 1.0f / (1.0f + avg_dist);

  return train_labels[best_idx];
}


/*
  Classify all regions using CNN embeddings and draw labels on image.
  Uses cyan text to distinguish from hand-built feature classification (yellow).
*/
void classifyAndLabelCNN(cv::Mat& image,
  std::vector<RegionInfo>& regions,
  const std::vector<std::string>& train_labels,
  const std::vector<std::vector<float>>& train_features) {
  if (train_labels.empty()) {
    cv::putText(image, "No CNN training data", cv::Point(10, 60),
      cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    return;
  }

  for (auto& region : regions) {
    if (region.embeddingVector.empty()) continue;

    float acc;
    std::string label = classifyObjectCNN(region.embeddingVector,
      train_labels,
      train_features,
      acc);

    // Position label below the OBB
    cv::Point2f corners[4];
    region.orientedBBox.points(corners);
    float botY = corners[0].y;
    for (int i = 1; i < 4; i++) botY = std::max(botY, corners[i].y);

    int x = std::clamp((int)region.centroid.x - 40, 5, image.cols - 200);
    int y = std::clamp((int)botY + 25, 25, image.rows - 10);

    cv::putText(image, label, cv::Point(x, y),
      cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);

    cv::putText(image, "[CNN]", cv::Point(x, y + 25),
      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);

    int acc_pct = (int)(acc * 100);
    std::string acc_text = std::to_string(acc_pct) + "%";
    cv::putText(image, acc_text, cv::Point(x + 50, y + 25),
      cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 0), 1);
  }
}
