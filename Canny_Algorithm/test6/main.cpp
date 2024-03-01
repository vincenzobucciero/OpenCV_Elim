#include <iostream>
#include <opencv4/opencv2/opencv.hpp>

using namespace cv;
using namespace std;

const int ht = 100, lt = 30;

void calcMagAng(const Mat& src, Mat& mag, Mat& ang) {
    Mat dx, dy;
    Sobel(src, dx, CV_32F, 1, 0);
    Sobel(src, dy, CV_32F, 0, 1);
    magnitude(dx, dy, mag);
    phase(dx, dy, ang, true);
}

void nms(Mat& mag, Mat& ang, Mat& nmsOut) {
    copyMakeBorder(mag, nmsOut, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0));
    for(int y = 1; y < mag.rows; y++) {
        for(int x = 1; x < mag.cols; x++) {
            float angVal = ang.at<float>(y-1,x-1) > 180 ? ang.at<float>(y,x)-180 : ang.at<float>(y,x);
            if(angVal >= 0 && angVal <= 22.5 || angVal >= 157.5 && angVal <= 180) {
                if(nmsOut.at<float>(y,x) < nmsOut.at<float>(y-1,x) || nmsOut.at<float>(y,x) < nmsOut.at<float>(y+1,x)) {
                    nmsOut.at<float>(y,x) = 0;
                }
            } else if(angVal > 22.5 && angVal <= 66.5) {
                if(nmsOut.at<float>(y,x) < nmsOut.at<float>(y-1,x-1) || nmsOut.at<float>(y,x) < nmsOut.at<float>(y+1,x+1)) {
                    nmsOut.at<float>(y,x) = 0;
                }
            } else if(angVal > 66.5 && angVal <= 112.5) {
                if(nmsOut.at<float>(y,x) < nmsOut.at<float>(y,x-1) || nmsOut.at<float>(y,x) < nmsOut.at<float>(y,x+1)) {
                    nmsOut.at<float>(y,x) = 0;
                }
            } else if(angVal > 112.5 && angVal <= 157.5) {
                if(nmsOut.at<float>(y,x) < nmsOut.at<float>(y-1,x+1) || nmsOut.at<float>(y,x) < nmsOut.at<float>(y+1,x-1)) {
                    nmsOut.at<float>(y,x) = 0;
                }
            }
        }
    }
}

void myThreshold(const Mat& nmsOut, Mat& dest) {
    dest = Mat::zeros(nmsOut.rows-2, nmsOut.cols-2, CV_8U);
    for(int y = 0; y < nmsOut.rows-1; y++) {
        for(int x = 0; x < nmsOut.cols-1; x++) {
            if(nmsOut.at<float>(y,x) > ht) 
                dest.at<uchar>(y-1,x-1) = 255;
            else if(nmsOut.at<float>(y,x) < lt) 
                dest.at<uchar>(y-1,x-1) = 0;
            else {
                bool strongN = false;
                for(int j = -1; j >= 1; j++) {
                    for(int i = -1; i >= 1; i++) {
                        if(dest.at<uchar>(y+j, x+i) > ht) {
                            strongN = true;
                        }
                    }
                }
                if(strongN)
                    dest.at<uchar>(y-1,x-1) = 255;
                else
                    dest.at<uchar>(y-1,x-1) = 0;
            }
        }
    }
}

void myCanny(const Mat& src, Mat& dest) {
    Mat blur, mag, ang, nmsOut;
    GaussianBlur(src, blur, Size(3,3), 0, 0);
    calcMagAng(blur, mag, ang);
    nms(mag, ang, nmsOut);
    myThreshold(nmsOut, dest);
}

int main(int argc, char**argv) {
    Mat src = imread(argv[1], IMREAD_GRAYSCALE);
    imshow("src", src);
    Mat dest, cannyCV;
    myCanny(src, dest);
    imshow("dest", dest);
    Canny(src, cannyCV, 30, 100);
    imshow("cannyCV", cannyCV);

    waitKey(0);

    return 0;
}