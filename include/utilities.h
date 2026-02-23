/*
  ATTRIBUTION: This code is based on utilities provided by Prof. Bruce A. Maxwell

  Modified by:
  Parker Cai
  Jenny Nguyen
  February 21, 2026

  Set of utility functions for computing features and embeddings
*/

#ifndef UTILITIES_H
#define UTILITIES_H

#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"

/**
  @brief Compute embedding vector from image using pre-trained ResNet18 network.
  
  @param src input image (thresholded and cleaned up, 8UC3 format)
  @param embedding output embedding vector from the network
  @param net pre-trained ResNet 18 network loaded from ONNX model
  @param debug if 1, display the image and embedding; if 0, run silently
  @return 0 on success
*/
int getEmbedding(cv::Mat &src, cv::Mat &embedding, cv::dnn::Net &net, int debug);

/**
  @brief Extract and rotate object region for embedding computation.
  
  Extracts the region from the original image and rotates it so the primary 
  axis is pointing right, preparing it for embedding computation.
  
  @param frame original image
  @param embimage output extracted ROI
  @param cx x-coordinate of the region centroid
  @param cy y-coordinate of the region centroid
  @param theta orientation of the primary axis (in radians)
  @param minE1 minimum projection along primary axis (should be negative)
  @param maxE1 maximum projection along primary axis (should be positive)
  @param minE2 minimum projection along secondary axis (should be negative)
  @param maxE2 maximum projection along secondary axis (should be positive)
  @param debug if 1, display intermediate images; if 0, run silently
*/
void prepEmbeddingImage(cv::Mat &frame, cv::Mat &embimage, 
                        int cx, int cy, float theta, 
                        float minE1, float maxE1, 
                        float minE2, float maxE2, int debug);

#endif // UTILITIES_H
