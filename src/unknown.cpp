/*
  Jenny Nguyen
  Parker Cai
  February 20, 2026
  CS5330 - Project 3: Real-time 2-D Object Recognition

  Extension: Auto-learn unknown objects
*/

#include "or2d.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <print>

/*
  Check if object is unknown
  
  If confidence is low, probably don't recognize it
*/
bool isUnknownObject(const std::vector<double>& query,
                     const std::vector<std::string>& train_labels,
                     const std::vector<std::vector<double>>& train_features,
                     double threshold) {
    if(train_labels.empty()) {
        return true;
    }
    
    double conf;
    classifyObject(query, train_labels, train_features, conf);
    
    // low confidence = unknown
    return conf < threshold;
}

//Classify but return "UNKNOWN" if confidence too low
std::string classifyWithUnknown(const std::vector<double>& query,
                                const std::vector<std::string>& train_labels,
                                const std::vector<std::vector<double>>& train_features,
                                double& confidence,
                                double threshold) {
    if(train_labels.empty()) {
        confidence = 0.0;
        return "UNKNOWN";
    }
    
    std::string label = classifyObject(query, train_labels, train_features, confidence);
    
    if(confidence < threshold) {
        return "UNKNOWN";
    }
    
    return label;
}

// Draw labels - red for unknown, yellow for known
void classifyAndLabelWithUnknown(cv::Mat& image,
                                 std::vector<RegionInfo>& regions,
                                 const std::vector<std::string>& train_labels,
                                 const std::vector<std::vector<double>>& train_features,
                                 double threshold) {
    if(train_labels.empty()) {
        cv::putText(image, "No training data", cv::Point(10, 60),
                   cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
        return;
    }
    
    for(auto& region : regions) {
        double conf;
        std::string label = classifyWithUnknown(region.featureVector, 
                                                train_labels, 
                                                train_features,
                                                conf,
                                                threshold);
        
        int x = (int)region.centroid.x - 40;
        int y = (int)region.centroid.y - 50;
        if(x < 5) x = 5;
        if(y < 25) y = 25;
        
        // pick color
        cv::Scalar color;
        if(label == "UNKNOWN") {
            color = cv::Scalar(0, 0, 255);  // red
        } else {
            color = cv::Scalar(255, 255, 0);  // yellow
        }
        
        // draw label
        cv::putText(image, label, cv::Point(x, y),
                   cv::FONT_HERSHEY_SIMPLEX, 0.8, color, 2);
        
        // draw confidence
        int pct = (int)(conf * 100);
        cv::putText(image, std::to_string(pct) + "%", cv::Point(x, y + 20),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }
}