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
    namedWindow("Threshold");
    
    // variables for the program
    bool auto_mode = true;
    int manual_thresh = 120; 
    int saveNum = 0;
    Mat frame;
    
    // main loop
    while(true) {
        // grab frame from camera
        cap >> frame;
        if(frame.empty()) {
            // cout << "empty frame" << endl;
            break;
        }
        
        // display the original frame
        Mat display;
        frame.copyTo(display);
        string mode_text;
        if(auto_mode) {
            mode_text = "Auto";
        } else {
            mode_text = "Manual=" + to_string(manual_thresh);
        }
        putText(display, mode_text, Point(10, 30), 
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,255,0), 2);
        imshow("Original", display);
        
        // thresholding
        Mat thresh;
        if(auto_mode) {
            thresh = thresholdImage(frame);
        } else {
            // manual threshold mode
            thresh = thresholdImage(frame, manual_thresh);
        }
        
        // show the thresholded image
        Mat thresh_display;
        cvtColor(thresh, thresh_display, COLOR_GRAY2BGR);
        putText(thresh_display, "Binary", Point(10, 30),
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,255,0), 2);
        imshow("Threshold", thresh_display);
        
        // check for key presses
        int key = waitKey(30);
        
        if(key == 'q') {
            cout << "Quitting..." << endl;
            break;
        }
        else if(key == 'h') {
            showHelp();
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
            string filename1 = "orig_" + to_string(saveNum) + ".jpg";
            string filename2 = "thresh_" + to_string(saveNum) + ".jpg";
            imwrite(filename1, frame);
            imwrite(filename2, thresh);
            cout << "Saved images: " << filename1 << " and " << filename2 << endl;
        }
    }
    
    // cleanup
    cap.release();
    destroyAllWindows();
    
    return 0;
}