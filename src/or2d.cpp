/*
  Jenny Nguyen
  Parker Cai
  February 16, 2026
  CS5330 - Project 3: Real-time 2-D Object Recognition

  Main program for object recognition.
  Captures video from webcam and applies thresholding/cleanup for 2D object recognition.
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <print>
#include <chrono>
#include "or2d.h"


/*
  Use the chrono time library to get the current time
  returns the current time in seconds as a double
*/
double getTime() {
  using namespace std::chrono;
  auto now = high_resolution_clock::now();
  auto now_in_seconds = duration_cast<duration<double>>(now.time_since_epoch()).count();
  return now_in_seconds;
}


/*
  Helper function to show the cli controls menu of the program.
*/
void showHelp() {
  std::println("Controls:");
  std::println("  q - quit");
  std::println("  s - save image");
  std::println("  a - toggle auto/manual");
  std::println("  t - toggle training mode");
  std::println("  n - save training sample");
  std::println("  + - increase threshold");
  std::println("  - - decrease threshold");
  std::println("  h - help");
  std::println("  0 - show original");
  std::println("  1 - show threshold");
  std::println("  2 - show cleaned");
  std::println("  3 - show segmented regions");
  std::println("  4 - show features (OBB + axis)");
  std::println("========================");
  std::println("");
}


/*
  Main function: capture video from webcam and perform object recognition.
*/
int main(int argc, char** argv) {
  // get camera number
  int camNum = 0;
  if (argc > 1) {
    camNum = atoi(argv[1]);
  }

  // open the camera
  cv::VideoCapture cap(camNum);
  if (!cap.isOpened()) {
    std::println("Can't open camera");
    return -1;
  }

  cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
  cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

  // get some properties of the image
  cv::Size refS((int)cap.get(cv::CAP_PROP_FRAME_WIDTH),
    (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT));
  std::println("Expected size: {} {}", refS.width, refS.height);

  std::println("Camera opened successfully");
  showHelp();

  // create windows
  cv::namedWindow("Original", cv::WINDOW_NORMAL);
  cv::namedWindow("Result", cv::WINDOW_NORMAL);
  cv::resizeWindow("Original", refS.width, refS.height);
  cv::resizeWindow("Result", refS.width, refS.height);

  // variables for the program
  bool auto_mode = true;
  int manual_thresh = 120;
  int display_mode = 2; // 0=original, 1=threshold, 2=cleaned, 3=segmented
  cv::Mat frame;
  std::vector<RegionInfo> regions;
  cv::Mat segmented, labelMap;

  // training mode variables
  bool training_mode = false;
  std::string db_filename = "data/objects_db.csv";

  // Load existing training data
  std::vector<std::string> train_labels;
  std::vector<std::vector<double>> train_features;
  loadTrainingData(db_filename, train_labels, train_features);

  // main loop
  while (true) {
    // grab frame from camera, treat as a stream
    cap >> frame;
    if (frame.empty()) {
      std::println("frame is empty");
      break;
    }

    // display the original frame with mode text overlay
    cv::Mat display = frame.clone();
    std::string text = auto_mode ? "Auto" : "Manual=" + std::to_string(manual_thresh);
    cv::putText(display, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    cv::imshow("Original", display);


    // show training mode status
    if(training_mode) {
      text += " | TRAINING MODE";
      cv::putText(display, "Press 'n' to save example", cv::Point(10, 60), 
                 cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
    }
    
    cv::putText(display, text, cv::Point(10, 30), 
               cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    cv::imshow("Original", display);

    // do the processing
    cv::Mat thresh, cleaned;
    if (auto_mode) {
      thresh = thresholdImage(frame);
    }
    else { // manual threshold mode
      thresh = thresholdImage(frame, manual_thresh);
    }
    cleaned = cleanupBinary(thresh);

    // Segment regions for multi-object recognition
    segmented = segmentRegions(cleaned, regions, labelMap);

    // Compute features for each region
    for (auto& region : regions) {
      computeRegionFeatures(labelMap, region);
    }

   

    // show result based on display mode
    cv::Mat show;
    std::string label;
    switch (display_mode) {
      case 0:
        cv::cvtColor(frame, show, cv::COLOR_BGR2GRAY);
        cv::cvtColor(show, show, cv::COLOR_GRAY2BGR);
        label = "Original";
        break;
      case 1:
        cv::cvtColor(thresh, show, cv::COLOR_GRAY2BGR);
        label = "Threshold";
        break;
      case 2:
        cv::cvtColor(cleaned, show, cv::COLOR_GRAY2BGR);
        label = "Cleaned";
        break;
      case 3:
        show = segmented.clone();
        label = "Segmented";
        break;
      case 4:
        show = colorizeRegions(labelMap, regions);
        drawFeatures(show, regions);
        label = "Features";
        break;
      default:
        cv::cvtColor(cleaned, show, cv::COLOR_GRAY2BGR);
        label = "Cleaned";
        break;
    }
    // overlay label text on the result
    cv::putText(show, label, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    cv::imshow("Result", show);

    // handle keypresses cases
    char key = cv::waitKey(30);
    switch (key) {
      case 'q':
        std::println("Quitting...");
        break;
      case 'h':
        showHelp();
        break;
      case 't':
        training_mode = !training_mode;
        std::println("Training mode: {}", training_mode ? "ON" : "OFF");
        break;
        
      case 'n':
        if(training_mode) {
          if(regions.empty()) {
            std::println("No object detected!");
          } else {
            RegionInfo& obj = regions[0];
            
            std::println("Enter object name: ");
            std::string obj_name;
            std::getline(std::cin, obj_name);
            
            if(!obj_name.empty()) {
              saveTrainingExample(db_filename, obj_name, obj.featureVector);
              loadTrainingData(db_filename, train_labels, train_features);
            }
          }
        } else {
          std::println("Not in training mode. Press 't' to toggle training mode.");
      }
      break;
      case '0':
        display_mode = 0;
        std::println("Showing original");
        break;
      case '1':
        display_mode = 1;
        std::println("Showing threshold");
        break;
      case '2':
        display_mode = 2;
        std::println("Showing cleaned");
        break;
      case '3':
        display_mode = 3;
        std::println("Showing segmented regions");
        break;
      case '4':
        display_mode = 4;
        std::println("Showing features");
        break;
      case 'a':
        auto_mode = !auto_mode;
        std::println("Mode: {}", auto_mode ? "Auto" : "Manual");
        break;
      case '+':
        manual_thresh = std::min(manual_thresh + 5, 255);
        std::println("Threshold: {}", manual_thresh);
        break;
      case '-':
        manual_thresh = std::max(manual_thresh - 5, 0);
        std::println("Threshold: {}", manual_thresh);
        break;
      case 's': {
        // save the original, threshold, cleaned, and segmented images with timestamped filenames
        std::string timestamp = std::to_string(getTime());
        cv::imwrite(timestamp + "_original" + ".jpg", frame);
        cv::imwrite(timestamp + "_threshold" + ".jpg", thresh);
        cv::imwrite(timestamp + "_cleaned" + ".jpg", cleaned);
        cv::imwrite(timestamp + "_segmented" + ".jpg", segmented);
        // save features overlay image
        cv::Mat featImg = colorizeRegions(labelMap, regions);
        drawFeatures(featImg, regions);
        cv::imwrite(timestamp + "_features" + ".jpg", featImg);
        // loop through feature vectors and print to console
        for (size_t i = 0; i < regions.size(); i++) {
          std::print("Region {}: [", i);
          for (size_t j = 0; j < regions[i].featureVector.size(); j++) {
            std::print("{:.4f}{}", regions[i].featureVector[j],
              (j < regions[i].featureVector.size() - 1) ? ", " : "");
          }
          std::println("]");
        }
        std::println("Saved frame_{}", timestamp);
        break;
      }
    }

    if (key == 'q') break;
  }

  // cleanup and exit
  cap.release();
  cv::destroyAllWindows();
  return 0;
}
