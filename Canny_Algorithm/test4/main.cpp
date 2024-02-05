#include <iostream>
#include <opencv4/opencv2/opencv.hpp>

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

void nms(const Mat& mag, const Mat& ang, Mat& nmsOut) {
    copyMakeBorder(mag, nmsOut, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0));
    for(int x = 1; x < mag.rows; x++) {
        for(int y = 1; y < mag.cols; y++) {
            float angVal = ang.at<float>(x-1,y-1) > 180 ? ang.at<float>(x,y) - 180 : ang.at<float>(x,y);
            if(angVal >= 0 && angVal <= 22.5 || angVal <= 180 && angVal >= 157.5) {
                if(nmsOut.at<float>(x,y) < nmsOut.at<float>(x,y-1) || nmsOut.at<float>(x,y) < nmsOut.at<float>(x,y+1)) {
                    nmsOut.at<float>(x,y) = 0;
                }
            } else if(angVal > 22.5 && angVal <= 67.5) {
                if(nmsOut.at<float>(x,y) < nmsOut.at<float>(x-1,y-1) || nmsOut.at<float>(x,y) < nmsOut.at<float>(x+1,y+1)) {
                    nmsOut.at<float>(x,y) = 0;
                }
            } else if(angVal > 67.5 && angVal <= 112.5) {
                if(nmsOut.at<float>(x,y) < nmsOut.at<float>(x-1,y) || nmsOut.at<float>(x,y) < nmsOut.at<float>(x+1,y)) {
                    nmsOut.at<float>(x,y) = 0;
                }
            } else if(angVal > 112.5 && angVal <= 157.5) {
                if(nmsOut.at<float>(x,y) < nmsOut.at<float>(x-1,y+1) || nmsOut.at<float>(x,y) < nmsOut.at<float>(x+1,y-1)) {
                    nmsOut.at<float>(x,y) = 0;
                }
            }
        }
    }
}

void threshold(Mat& nmsOut, Mat& dest) {
    dest = Mat::zeros(nmsOut.rows-2, nmsOut.cols-2, CV_8U);
    for(int x = 1; x < nmsOut.rows-1; x++) {
        for(int y = 1; y < nmsOut.cols-1; y++) {
            if(nmsOut.at<float>(x, y) > ht) 
                dest.at<uchar>(x-1, y-1) = 255;
            else if(nmsOut.at<float>(x, y) < lt)
                dest.at<uchar>(x-1, y-1) = 0;
            else {
                bool strongN = false;
                for(int i = -1; i >= 1; i++) {
                    for(int j = -1; j >= 1; j++) {
                        if(nmsOut.at<float>(x+i, y+j) > ht) {
                            strongN = true;
                        }
                    }
                }
                if(strongN)
                    dest.at<uchar>(x-1, y-1) = 255;
                else    
                    dest.at<uchar>(x-1, y-1) = 0;
            }
        }
    }
}

void canny(const Mat& src, Mat& dest) {
    Mat blur, mag, ang, nmsOut;
    GaussianBlur(src, blur, Size(5,5), 0, 0);
    calcMagAndAng(blur, mag, ang);
    nms(mag, ang, nmsOut);
    threshold(nmsOut, dest);
}

int main(int argc, char**argv) {
    if(argc < 2) {
        cout << "Usage: ./main image_path" << endl;
        exit(0);
    }
    Mat src = imread(argv[1], IMREAD_GRAYSCALE);
    if(src.empty()) {
        cout << "Error, can't open / read image" << endl;
        exit(0);
    }
    imshow("original image input", src);

    Mat dest;
    canny(src, dest);
    imshow("myCanny image output", dest);

    waitKey(0);

    return 0;
}