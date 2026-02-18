/*
  Jenny Nguyen
  February 16, 2026
  CS5330 - Project 3: Real-time 2-D Object Recognition

  Morphological filtering - built from scratch to remove noise and fill holes in binary images
*/

#include "or2d.h"
#include <opencv2/opencv.hpp>

using namespace cv;

/*
  Erosion - makes white regions smaller
  
  Goes through each pixel and checks neighbors
  Only keeps pixel white if ALL neighbors are white
  
  Input: binary image
  Output: eroded image
*/
Mat erode(const Mat& src) {
    Mat result = Mat::zeros(src.size(), CV_8U);
    
    // loop through image (skip border pixels)
    for(int r = 1; r < src.rows - 1; r++) {
        for(int c = 1; c < src.cols - 1; c++) {
            
            // check all 8 neighbors around this pixel
            bool keep_pixel = true;
            
            for(int i = -1; i <= 1; i++) {
                for(int j = -1; j <= 1; j++) {
                    // if any neighbor is black, don't keep this pixel
                    if(src.at<uchar>(r + i, c + j) == 0) {
                        keep_pixel = false;
                        break;
                    }
                }
                if(!keep_pixel) break;
            }
            
            // only set white if all neighbors were white
            if(keep_pixel) {
                result.at<uchar>(r, c) = 255;
            }
        }
    }
    
    return result;
}

/*
  Dilation - makes white regions bigger
  
  If any neighbor is white, make this pixel white
  
  Input: binary image
  Output: dilated image
*/
Mat dilate(const Mat& src) {
    Mat result = Mat::zeros(src.size(), CV_8U);
    
    for(int r = 1; r < src.rows - 1; r++) {
        for(int c = 1; c < src.cols - 1; c++) {
            
            // check if any neighbor is white
            bool found_white = false;
            
            for(int i = -1; i <= 1; i++) {
                for(int j = -1; j <= 1; j++) {
                    if(src.at<uchar>(r + i, c + j) == 255) {
                        found_white = true;
                        break;
                    }
                }
                if(found_white) break;
            }
            
            if(found_white) {
                result.at<uchar>(r, c) = 255;
            }
        }
    }
    
    return result;
}

/*
  Main cleanup function
  
  Does opening then closing:
  - Opening = erode then dilate (gets rid of noise)
  - Closing = dilate then erode (fills holes)
  
  Input: messy binary image
  Output: cleaned up version
*/
Mat cleanupBinary(const Mat& binary) {
    Mat temp, cleaned;
    
    // opening - remove noise spots
    temp = erode(binary);
    temp = dilate(temp);
    
    // closing - fill holes
    temp = dilate(temp);
    cleaned = erode(temp);
    
    return cleaned;
}