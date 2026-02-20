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
  std::println("  + - increase threshold");
  std::println("  - - decrease threshold");
  std::println("  h - help");
  std::println("  0 - show original");
  std::println("  1 - show threshold");
  std::println("  2 - show cleaned");
  std::println("  3 - show segmented regions");
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
