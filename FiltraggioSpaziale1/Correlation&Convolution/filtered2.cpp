#include <iostream>
#include <opencv4/opencv2/opencv.hpp>

using namespace cv;


uchar filtered2(Mat pad, Mat mask, int off_X, int off_Y) {
    float value = 0.0;
    for(int x = 0; x < mask.rows; x++) {
        for(int y = 0; y < mask.cols; y++) {
            value += mask.at<float>(x, y)*pad.at<uchar>(off_X-x, off_Y-y);
        }
    }
    return uchar(value);
}