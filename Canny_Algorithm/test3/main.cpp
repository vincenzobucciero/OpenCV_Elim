#include <iostream>
#include <opencv4/opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int lt = 30;
int ht = 100;

void calculateMagAndAngle(const Mat& imageInput, Mat& mag, Mat& ang) {
    Mat dx, dy;
    Sobel(imageInput, dx, CV_32F, 1, 0);
    Sobel(imageInput, dy, CV_32F, 0, 1);
    magnitude(dx, dy, mag);
    phase(dx, dy, ang, true);
}

void nms(const Mat& mag, const Mat& ang, Mat& nmsOut) {
    copyMakeBorder(mag, nmsOut, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0));
    for(int x = 1; x < mag.rows; x++) {
        for(int y = 1; y < mag.cols; y++) {
            float angVal = ang.at<float>(x-1, y-1) > 180 ? ang.at<float>(x, y) - 180 : ang.at<float>(x, y);
            if(angVal >= 0 && angVal <= 22.5 || angVal <= 180 && angVal > 157.5) {
                if(nmsOut.at<float>(x, y) < nmsOut.at<float>(x, y-1) || nmsOut.at<float>(x, y) < nmsOut.at<float>(x, y+1)) {
                    nmsOut.at<float>(x, y) = 0;
                }
            } else if(angVal > 22.5 && angVal <= 67.5) {
                if(nmsOut.at<float>(x, y) < nmsOut.at<float>(x-1, y-1) || nmsOut.at<float>(x, y) < nmsOut.at<float>(x+1, y+1)) {
                    nmsOut.at<float>(x, y) = 0;
                }
            } else if(angVal > 67.5 && angVal <= 112.5) {
                if(nmsOut.at<float>(x, y) < nmsOut.at<float>(x-1, y) || nmsOut.at<float>(x, y) < nmsOut.at<float>(x+1, y)) {
                    nmsOut.at<float>(x, y) = 0;
                }
            } else if(angVal > 112.5 && angVal <= 157.5) {
                if(nmsOut.at<float>(x, y) < nmsOut.at<float>(x+1, y-1) || nmsOut.at<float>(x, y) < nmsOut.at<float>(x-1, y+1)) {
                    nmsOut.at<float>(x, y) = 0;
                }
            }
        }
    }
}

void treshold(Mat& nmsOut, Mat& imageOutput) {
    imageOutput = Mat::zeros(nmsOut.rows-2, nmsOut.cols-2, CV_8U);
    for(int x = 1; x < nmsOut.rows-1; x++) {
        for(int y = 1; y < nmsOut.cols-1; y++) {
            if(nmsOut.at<float>(x, y) > ht)
                imageOutput.at<uchar>(x-1, y-1) = 255;
            else if(nmsOut.at<float>(x, y) < lt)
                imageOutput.at<uchar>(x-1, y-1) = 0;
            else {
                bool strongN = false;
                for(int i = -1; i >= 1; i++) {
                    for(int j = -1; j >= 1; j++) {
                        if(nmsOut.at<float>(x+i, y+j) > ht)
                            strongN = true;
                    }
                }
                if(strongN)
                    imageOutput.at<uchar>(x-1, y-1) = 255;
                else
                    imageOutput.at<uchar>(x-1, y-1) = 0;
            }
        }
    }
}

void cannyTest3(const Mat& imageInput, Mat& imageOutput) {
    Mat blur, mag, ang, nmsOut;
    //gaussiana
    GaussianBlur(imageInput, blur, Size(5, 5), 0, 0);
    //mag&ang
    calculateMagAndAngle(imageInput, mag, ang);
    //nms
    nms(mag, ang, nmsOut);
    //treshold
    treshold(nmsOut, imageOutput);
}

int main(int argc, char**argv) {
    if(argc < 2) {
        cout << "Usage: ./main image_path" << endl;
        exit(0);
    }
    Mat imageInput = imread(argv[1], IMREAD_GRAYSCALE);
    if(imageInput.empty()) {
        cout << "Error, can't open / read image" << endl;
        exit(0);
    }
    imshow("original image input", imageInput);

    Mat myCannyImageOutput;
    cannyTest3(imageInput, myCannyImageOutput);
    imshow("myCanny image output", myCannyImageOutput);

    Mat CannyCVImageOutput;
    Canny(imageInput, CannyCVImageOutput, 30, 100);
    imshow("CannyCV image output", CannyCVImageOutput);

    waitKey(0);

    return 0;
}