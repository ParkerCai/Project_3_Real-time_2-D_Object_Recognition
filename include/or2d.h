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
  cv::Rect bbox; // axis-aligned bounding box (AABB)
  int area;
  cv::Vec3b color;
  float theta; // angle of axis of least central moment (radians)
  cv::RotatedRect orientedBBox; // oriented bounding box (OBB) along the primary axis
  float bboxRatio; // min(w,h) : max(w,h) of the OBB
  float percentFilled; // = region area / OBB area
  double huMoments[7]; // 7 moments from Hu moment invariants
  std::vector<double> featureVector; // assembled feature vector for classification
  // OBB projection extents (for CNN embedding image prep)
  float uMin = 0, uMax = 0; // primary axis extents
  float vMin = 0, vMax = 0; // secondary axis extents
  // CNN embedding vector (512-d from ResNet18, use float for native DNN precision)
  std::vector<float> embeddingVector;
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

/**
  @brief Compute features for a single region using region-based analysis.
  Computes principal axis, oriented bounding box, percent filled, aspect ratio,
  Hu moments, and assembles a feature vector.
  @param labelMap integer label map (CV_32S) from connected components
  @param region RegionInfo struct to populate with computed features
*/
/**
  @brief Build a color-coded region image without overlays (no AABB, no centroid).
  @param labelMap integer label map (CV_32S) from connected components
  @param regions vector of RegionInfo structs with assigned colors
  @return Clean color-coded image with only region fills
*/
cv::Mat colorizeRegions(const cv::Mat& labelMap, const std::vector<RegionInfo>& regions);

void computeRegionFeatures(const cv::Mat& labelMap, RegionInfo& region);

/**
  @brief Draw feature overlays (OBB, principal axis, feature text) on an image.
  @param image image to draw on (modified in-place)
  @param regions vector of RegionInfo structs with computed features
*/
void drawFeatures(cv::Mat& image, const std::vector<RegionInfo>& regions);

// Training
void saveTrainingExample(const std::string& filename,
  const std::string& label,
  const std::vector<double>& features);
// Overload for CNN embedding data (float)
void saveTrainingExample(const std::string& filename,
  const std::string& label,
  const std::vector<float>& features);

int loadTrainingData(const std::string& filename,
  std::vector<std::string>& labels,
  std::vector<std::vector<double>>& features);
// Overload for CNN embedding data (float)  
int loadTrainingData(const std::string& filename,
  std::vector<std::string>& labels,
  std::vector<std::vector<float>>& features);

void initializeDatabase(const std::string& filename);


// Classification (hand-built features)
std::vector<double> computeStdDevs(const std::vector<std::vector<double>>& features);

double scaledEuclideanDistance(const std::vector<double>& f1,
  const std::vector<double>& f2,
  const std::vector<double>& stddevs);

std::string classifyObject(const std::vector<double>& query,
  const std::vector<std::string>& train_labels,
  const std::vector<std::vector<double>>& train_features,
  double& accuracy);

void classifyAndLabel(cv::Mat& image,
  std::vector<RegionInfo>& regions,
  const std::vector<std::string>& train_labels,
  const std::vector<std::vector<double>>& train_features);


// Classification (CNN embedding - one-shot, uses float for native DNN precision)
float sumOfSquaredDifference(const std::vector<float>& featuresA,
  const std::vector<float>& featuresB);

std::string classifyObjectCNN(const std::vector<float>& query,
  const std::vector<std::string>& train_labels,
  const std::vector<std::vector<float>>& train_features,
  float& accuracy);

void classifyAndLabelCNN(cv::Mat& image,
  std::vector<RegionInfo>& regions,
  const std::vector<std::string>& train_labels,
  const std::vector<std::vector<float>>& train_features);


// Confusion matrix 
struct ConfusionMatrix {
  std::vector<std::string> classes;
  std::map<std::string, int> class_index;
  std::vector<std::vector<int>> matrix;
};

void addClassToMatrix(ConfusionMatrix& cm, const std::string& name);
void addResultToMatrix(ConfusionMatrix& cm, const std::string& true_label, const std::string& predicted);
void printConfusionMatrix(ConfusionMatrix& cm);
void saveConfusionMatrix(ConfusionMatrix& cm, const std::string& filename);

#endif // OR2D_H