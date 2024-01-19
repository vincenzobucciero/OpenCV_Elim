#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

int ht = 100; 
int lt = 30;

void calcMagAndAng(const Mat& src, Mat& mag, Mat& ang) {
    Mat dx, dy;
    Sobel(src, dx, CV_32F, 1, 0);
    Sobel(src, dy, CV_32F, 0, 1);
    magnitude(dx, dy, mag);
    phase(dx, dy, ang, true);
}

void nonMaximaSuppression(const Mat& mag, const Mat& ang, Mat& nmsOut) {
    copyMakeBorder(mag, nmsOut, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0));
    for(int y = 1; y < mag.rows; y++) {
        for(int x = 1; x < mag.cols; x++) {
            float angVal = ang.at<float>(y-1, x-1) > 180 ? ang.at<float>(y, x) - 180 : ang.at<float>(y, x);
            if(angVal >= 0 && angVal <= 22.5 || angVal <= 180 && angVal > 157.5) {
                if(nmsOut.at<float>(y, x) < nmsOut.at<float>(y, x-1) || nmsOut.at<float>(y, x) < nmsOut.at<float>(y, x+1)) {
                    nmsOut.at<float>(y, x) = 0;
                }
            } else if(angVal > 22.5 && angVal <= 67.5) {
                if(nmsOut.at<float>(y, x) < nmsOut.at<float>(y-1, x-1) || nmsOut.at<float>(y, x) < nmsOut.at<float>(y+1, x+1)) {
                    nmsOut.at<float>(y, x) = 0;
                }
            } else if(angVal > 67.5 && angVal <= 112.5) {
                if(nmsOut.at<float>(y, x) < nmsOut.at<float>(y-1, x) || nmsOut.at<float>(y, x) < nmsOut.at<float>(y+1, x)) {
                    nmsOut.at<float>(y, x) = 0;
                }
            } else if(angVal > 112.5 && angVal <= 157.5) {
                if(nmsOut.at<float>(y, x) < nmsOut.at<float>(y+1, x-1) || nmsOut.at<float>(y, x) < nmsOut.at<float>(y-1, x+1)) {
                    nmsOut.at<float>(y, x) = 0;
                }
            }
        }
    }
}

void treshold(Mat& nmsOut, Mat& dest) {
    dest = Mat::zeros(nmsOut.rows-2, nmsOut.cols-2, CV_8U);
    for(int y = 1; y < nmsOut.rows-1; y++) {
        for(int x = 1; x < nmsOut.cols-1; x++) {
            if(nmsOut.at<float>(y, x) > ht)
                dest.at<uchar>(y-1, x-1) = 255;
            else if(nmsOut.at<float>(y, x) < lt)
                dest.at<uchar>(y-1, x-1) = 0;
            else {
                bool strongN = false;
                for(int i = -1; i >= 1; i++) {
                    for(int j = -1; j >= 1; j++) {
                        if(nmsOut.at<float>(y+i, x+j) > ht)
                            strongN = true;
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

void canny(const Mat& src, Mat& dest) {
    Mat blur, mag, ang, nmsOut;
    GaussianBlur(src, blur, Size(3, 3), 0, 0);
    calcMagAndAng(src, mag, ang);
    nonMaximaSuppression(mag, ang, nmsOut);
    treshold(nmsOut, dest);
}

int main(int argc, char**argv) {
    if(argc < 2) {
        cout << "error, using " << argv[0] << endl;
        exit(0);
    }
    Mat src = imread(argv[1], IMREAD_GRAYSCALE);
    if(src.empty()) {
        cout << "error can't read/open image" << endl;
        exit(0);
    }
    imshow("Original image", src);
    Mat dest;
    canny(src, dest);
    imshow("Canny edge detection result", dest);
    Mat cannyCVDest;
    Canny(src, cannyCVDest, 30, 100),
    imshow("Canny OPENCV edge detection result", cannyCVDest);

    waitKey(0);

    return 0;
}