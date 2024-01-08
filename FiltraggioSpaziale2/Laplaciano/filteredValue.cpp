#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

float filteredValue(const Mat& imageInput, const Mat& mask, int offX, int offY) {
    float value;
    for(int i = 0; i < mask.rows; i++) {
        for(int j = 0; j < mask.cols; j++) {
            value += mask.at<float>(i, j) * imageInput.at<float>(i+offX, j+offY);
        }
    }
    return value;
}