#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

vector<float> normHistogram(const Mat& src) {
    vector<float> hist(256, 0.0f);
    for(int y = 0; y < src.rows; y++) {
        for(int x = 0; x < src.rows; x++) {
            hist.at(src.at<uchar>(y,x))++;
        }
    }
    for(int i = 0; i < 256; i++) {
        hist.at(i) /= (src.rows*src.cols);
    }
    return hist;
}

int otsu1k(const Mat& src) {
    Mat blur;
    GaussianBlur(src, blur, Size(3,3), 0, 0);
    vector<float> hist = normHistogram(blur);
    float cumAvg = 0.0f, globAvg = 0.0f, prob = 0.0f, interClassVariance = 0.0f, maxVariance = 0.0f;
    int th = 0;

    for(int i = 0; i < 256; i++) {
        globAvg += (i+1)*hist.at(i);
    }
    for(int k = 0; k < 256; k++) {
        prob += hist.at(k);
        cumAvg += (k+1)*hist.at(k);
        float interClassNum = pow(((globAvg*prob)-cumAvg), 2);
        float interClassDen = prob*(1-prob);
        if(interClassNum != 0)
            interClassVariance = interClassNum/interClassDen;
        else        
            interClassVariance = 0;
        if(interClassVariance > maxVariance) {
            maxVariance = interClassVariance;
            th = k;
        }
    }
    return th;
}

int main(int argc, char**argv) {
    if(argc < 2) {
        cout << "error" << endl;
        exit(0);
    }
    Mat src = imread(argv[1], IMREAD_COLOR);
    if(src.empty()) {
        cout << "can not open/read image" << argv[1] << endl;
        exit(0);
    }
    imshow("original image", src);
    Mat dest;
    threshold(src, dest, otsu1k(src), 255, THRESH_BINARY);
    imshow("otsu1k image", dest);
    waitKey(0);

    return 0;
}