/*
  Jenny Nguyen
  February 16, 2026
  CS5330 - Project 3: Real-time 2-D Object Recognition
  
  Main program for object recognition
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include "or2d.h"

using namespace cv;
using namespace std;

void showHelp() {
    cout << "Controls:" << endl;
    cout << "  q - quit" << endl;
    cout << "  s - save image" << endl;
    cout << "  a - toggle auto/manual" << endl; 
    cout << "  + - increase threshold" << endl;
    cout << "  - - decrease threshold" << endl;
    cout << "  h - help" << endl;
    cout << "  1 - show original" << endl;
    cout << "  2 - show threshold" << endl;
    cout << "  3 - show cleaned" << endl;
    cout << "========================\n" << endl;
}

int main(int argc, char** argv) {
    // get camera number
    int camNum = 0;
    if(argc>1){
        camNum=atoi(argv[1]);
    }
    
    // open the camera
    VideoCapture cap(camNum);
    if(!cap.isOpened()) {
        cout << "Can't open camera" << endl;
        return -1;
    }
    
    cap.set(CAP_PROP_FRAME_WIDTH, 640);
    cap.set(CAP_PROP_FRAME_HEIGHT, 480);
    
    cout << "Camera opened successfully" << endl;
    showHelp();
    
    // create windows
    namedWindow("Original");
    namedWindow("Result");
    
    // variables for the program
    bool auto_mode = true;
    int manual_thresh = 120; 
    int display_mode = 3; // 1=original, 2=threshold, 3=cleaned
    int saveNum = 0;
    Mat frame;
    
    // main loop
    while(true) {
        // grab frame from camera
        cap >> frame;
        if(frame.empty()) {
            break;
        }
        
        // display the original frame
        Mat display = frame.clone();
        string text = auto_mode ? "Auto" : "Manual=" + to_string(manual_thresh);
        putText(display, text, Point(10,30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,255,0), 2);
        imshow("Original", display);
        
        // do the processing
        Mat thresh, cleaned;
        if(auto_mode) {
            thresh = thresholdImage(frame);
        } else {
            // manual threshold mode
            thresh = thresholdImage(frame, manual_thresh);
        }
        cleaned = cleanupBinary(thresh);
        
       // show result based on mode
        Mat show;
        string label;
        if(display_mode == 1) {
            cvtColor(frame, show, COLOR_BGR2GRAY);
            cvtColor(show, show, COLOR_GRAY2BGR);
            label = "Original";
        } else if(display_mode == 2) {
            cvtColor(thresh, show, COLOR_GRAY2BGR);
            label = "Threshold";
        } else {
            cvtColor(cleaned, show, COLOR_GRAY2BGR);
            label = "Cleaned";
        }
        putText(show, label, Point(10,30), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,255,0), 2);
        imshow("Result", show);
        
        // check for key presses
        int key = waitKey(30);
        
        if(key == 'q') {
            cout << "Quitting..." << endl;
            break;
        }
        else if(key == 'h') {
            showHelp();
        }
        else if(key == '1') {
            display_mode = 1;
            cout << "Showing original" << endl;
        }
        else if(key == '2') {
            display_mode = 2;
            cout << "Showing threshold" << endl;
        }
        else if(key == '3') {
            display_mode = 3;
            cout << "Showing cleaned" << endl;
        }
        else if(key == 'a') {
            auto_mode = !auto_mode;
            if(auto_mode) {
                cout << "Mode: Auto" << endl;
            } else {
                cout << "Mode: Manual" << endl;
            }
        }
        else if(key == '+') {
            manual_thresh = manual_thresh + 5;
            if(manual_thresh > 255) {
                manual_thresh = 255;
            }
            cout << "Threshold: " << manual_thresh << endl;
        }
        else if(key == '-') {
            manual_thresh = manual_thresh - 5;
            if(manual_thresh < 0) {
                manual_thresh = 0;
            }
            cout << "Threshold: " << manual_thresh << endl;
        }
        else if(key == 's') {
            // save both images
            saveNum++;
            imwrite("orig_" + to_string(saveNum) + ".jpg", frame);
            imwrite("thresh_" + to_string(saveNum) + ".jpg", thresh);
            imwrite("clean_" + to_string(saveNum) + ".jpg", cleaned);
            cout << "Saved set " << saveNum << endl;
        }
    }
    
    // cleanup
    cap.release();
    destroyAllWindows();
    
    return 0;
}