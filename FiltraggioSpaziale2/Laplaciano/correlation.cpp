#include <opencv2/opencv.hpp>
#include <iostream>

#include "filteredValue.cpp"

using namespace cv;
using namespace std;

Mat correlation(const Mat& imageInput, const Mat& mask) {
    Mat imageOutput;
    Mat imagePadded;

    int border = mask.rows/2;

    copyMakeBorder(imageOutput, imagePadded, border, border, border, border, BORDER_REFLECT);

    for(int i = 0; i < imageOutput.rows; i++) {
        for(int j = 0; j < imageOutput.cols; j++) {
            imageOutput.at<float>(i, j) = filteredValue(imagePadded, mask, i, j);
        }
    }

    return imageOutput;
}