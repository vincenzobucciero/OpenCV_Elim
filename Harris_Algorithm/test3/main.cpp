#include <iostream>
#include <vector>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int th = 120;

void calcGradient(const Mat& src, Mat& c00, Mat& c11, Mat& c10) {
    Mat dx, dy, dxy;
    Sobel(src, dx, CV_32F, 1, 0);
    Sobel(src, dy, CV_32F, 0, 1);
    multiply(dx, dy, dxy);
    pow(dx, 2, dx);
    pow(dy, 2, dy);
    GaussianBlur(dx, c00, Size(3,3), 0, 0);
    GaussianBlur(dy, c11, Size(3,3), 0, 0);
    GaussianBlur(dxy, c10, Size(3,3), 0, 0);
}

void detAndTrack(const Mat& c00, const Mat& c11, const Mat& c10, Mat& det, Mat& track) {
    Mat diag1, diag2;
    multiply(c00, c11, diag1);
    multiply(c10, c10, diag2);
    det = diag1-diag2;
    track = c00+c11;
    pow(track, 2, track);
    track = track*0.04;
}

void calcR(const Mat& det, const Mat& track, Mat& R) {
    R = det - track;
    normalize(R, R, 0, 255, NORM_MINMAX);
}

void detectCorner(const Mat& R, Mat& dest) {
    convertScaleAbs(R, dest);
    cvtColor(dest, dest, COLOR_GRAY2BGR);
    for(int y = 0; y < R.rows; y++) {
        for(int x = 0; x < R.cols; x++) {
            if(R.at<float>(y,x) > th)
                circle(dest, Point(x, y), 4, Scalar(0,0,255));
        }
    }
}

void myHarris(const Mat& src, Mat& dest) {
    Mat c00, c11, c10, blur, det, track, R;
    GaussianBlur(src, blur, Size(3,3), 0, 0);
    calcGradient(src, c00, c11, c10);
    detAndTrack(c00, c11, c10, det, track);
    calcR(det, track, R);
    detectCorner(R, dest);
}

int main(int argc, char**argv) {
    if(argc < 2) {
        cout << "using " << argv[0] << endl;
        exit(0);
    }
    Mat src = imread(argv[1], IMREAD_GRAYSCALE);
    if(src.empty()) {
        cout << "error cant't open/read image" << endl;
    }
    imshow("original image", src);
    Mat dest;
    myHarris(src, dest);
    imshow("MYHARRIS image", dest);

    waitKey(0);

    return 0;
}