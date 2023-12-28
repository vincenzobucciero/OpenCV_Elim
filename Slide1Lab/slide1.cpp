#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/highgui.hpp>

#include <opencv4/opencv2/opencv.hpp>

#include <iostream>
#include <string.h>

using namespace cv;

int main() {
    cv::Mat matrix;

    matrix.create(3, 3, CV_32FC3); //matrice di 3x10 di tipo float a 32 bit su 3 canali
    cv::randu(matrix, 1.0f, 5.0f);
    // matrix = cv::Mat::eye(5, 5, CV_32FC2);

    //printf("Element (7,3) is [%f][%f]\n", matrix.at<cv::Vec2f>(7,3)[0], matrix.at<cv::Vec2f>(7,3)[1]);

    for(int i = 0; i < matrix.rows; ++i) {
        for(int j = 0; j < matrix.cols; ++j) {
            cv::Vec2f element = matrix.at<cv::Vec2f>(i, j);
            std::cout << "Element (" << i << "," << j << ") is [" << element[0] << "][" << element[1] << "]\n";
        }
    }

    cv::Mat matrix2 = -matrix;

    for(int i = 0; i < matrix2.rows; ++i) {
        for(int j = 0; j < matrix2.cols; ++j) {
            cv::Vec2f element = matrix2.at<cv::Vec2f>(i, j);
            std::cout << "Element (" << i << "," << j << ") is [" << element[0] << "][" << element[1] << "]\n";
        }
    }

    return 0;
}