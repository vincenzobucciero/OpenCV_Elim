#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <tuple>

using namespace std;
using namespace cv;

const int rMax = 90, rMin = 25;
const float RADC = CV_PI/180;
const int th = 150;

void createVotingSpace(const Mat& src, Mat& votingSpace) {
    int dim[] = {src.rows, src.cols, rMax-rMin+1};
    votingSpace = Mat(3, dim, CV_8U, Scalar(0));
}

void vote(const Mat& edge, Mat& votingSpace) {
    for(auto y = 0; y < edge.rows; y++) {
        for(auto x = 0; x < edge.cols; x++) {
            if(edge.at<uchar>(y,x) == 255) {
                for(auto r = rMin; r <= rMax; r++) {
                    for(int thetaIndex = 0; thetaIndex <= 360; thetaIndex++) {
                        double theta = thetaIndex*RADC;
                        double sint = sin(theta);
                        double cost = cos(theta);
                        int a = cvRound(x-r*cost);
                        int b = cvRound(y-r*sint);
                        if(a >= 0 && a < edge.cols && b >= 0 && b < edge.rows)
                            votingSpace.at<uchar>(b, a, r-rMin)++;
                    }
                }
            }
        }
    }
}

vector<tuple<int,int,int>> detectCirlce(const Mat& src, Mat& votingSpace) {
    vector<tuple<int,int,int>> detected;
    for(auto b = 0; b < src.rows; b++) {
        for(auto a = 0; a < src.cols; a++) {
            for(auto r = 0; r <= rMax-rMin; r++) {
                if(votingSpace.at<uchar>(b,a,r) > th)
                    detected.push_back(make_tuple(b,a,r+rMin)); 
            } 
        }
    }
    return detected;
}

void drowCircle(vector<tuple<int,int,int>> detected, Mat& dest) {
    for(auto c : detected) {
        circle(dest, Point(get<1>(c), get<0>(c)), get<2>(c), Scalar(0,0,255), 2);
        circle(dest, Point(get<1>(c), get<0>(c)), 1, Scalar(0,0,255));
    }
}

void myHoughCircle(const Mat& src, Mat& dest) {
    Mat blur, edge, votingSpace;
    src.copyTo(dest);
    GaussianBlur(src, blur, Size(3,3), 0, 0);
    cvtColor(blur, blur, COLOR_BGR2GRAY);
    createVotingSpace(src, votingSpace);
    Canny(blur, edge, 50, 150);
    vote(edge, votingSpace);
    //detectCircle&tuple
    vector<tuple<int,int,int>> detected = detectCirlce(src, votingSpace);
    drowCircle(detected, dest);
}

int main(int argc, char**argv) {
    if(argc < 2) {
        cout << "error " << argv[0] << ", using image name" << endl;
        exit(0);
    }
    Mat src = imread(argv[1], IMREAD_COLOR);
    if(src.empty()) {
        cout << "error, can't open / read image" << endl;
        exit(0);
    }
    Mat dest;
    imshow("original image", src);
    myHoughCircle(src, dest);
    imshow("image test w circle", dest);
    waitKey(0);

    return 0;
}