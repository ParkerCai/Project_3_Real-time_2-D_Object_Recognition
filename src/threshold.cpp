/*
  Jenny Nguyen
  Parker Cai
  February 16, 2026
  CS5330 - Project 3: Real-time 2-D Object Recognition

  Thresholding implementation build from scratch
*/

#include "or2d.h"
#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

/*
  Threshold the input image to separate objects from background

  If threshValue is -1, calculates threshold automatically
  otherwise uses the value you give it

  Input: grayscale or color image
  Output: binary image (white objects, black background)
*/
Mat thresholdImage(const Mat& input, int threshValue) {
  Mat gray, binary;

  // convert to grayscale if needed
  if (input.channels() == 3) {
    cvtColor(input, gray, COLOR_BGR2GRAY);
  }
  else {
    gray = input.clone();
  }

  // reduce noise
  GaussianBlur(gray, gray, Size(5, 5), 0);

  // auto threshold if not given
  if (threshValue < 0) {
    vector<float> pixels;

    // use average pixel value
    for (int r = 0; r < gray.rows; r += 4) {
      for (int c = 0; c < gray.cols; c += 4) {
        pixels.push_back((float)gray.at<uchar>(r, c));
      }
    }

    // run k-means to find object and background clusters
    if (pixels.size() > 0) {
      // convert vector to Mat for k-means
      Mat pixelMat(pixels.size(), 1, CV_32F);
      for (size_t i = 0; i < pixels.size(); i++) {
        pixelMat.at<float>(i) = pixels[i];
      }

      // k-means with k=2 
      Mat labels, centers;
      kmeans(pixelMat, 2, labels,
        TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0),
        3, KMEANS_PP_CENTERS, centers);
      threshValue = (centers.at<float>(0) + centers.at<float>(1)) / 2;
    }
    else {
      threshValue = 128;
    }
  }

  // manual thresholding loop
  binary = Mat::zeros(gray.size(), CV_8U);

  for (int r = 0; r < gray.rows; r++) {
    for (int c = 0; c < gray.cols; c++) {
      // compare pixel value to threshold
      if (gray.at<uchar>(r, c) < threshValue) {
        binary.at<uchar>(r, c) = 255;
      }
      else {
        binary.at<uchar>(r, c) = 0;
      }
    }
  }

  return binary;
}