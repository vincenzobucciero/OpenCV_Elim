#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <tuple>

using namespace std;
using namespace cv;

const int rMax = 60;
const int rMin = 25;
const float RADC = CV_PI/180;
const int th = 210;

void createVotingSpace(const Mat& src, Mat& votingSpace) {
    int dim[] = {src.rows, src.cols, rMax-rMin+1};
    votingSpace = Mat(3, dim, CV_8U, Scalar(0));
}

void vote(const Mat& edge, Mat& votingSpace) {
    for(auto y = 0; y < edge.rows; y++) {
        for(auto x = 0; x < edge.cols; x++) {
            if(edge.at<uchar>(y, x) == 255) {
                for(int r = rMin; r <= rMax; r++) {
                    for(int thetaIndex = 0; thetaIndex <= 360; thetaIndex++) {
                        float theta = thetaIndex*RADC;
                        float sint = sin(theta);
                        float cost = cos(theta);
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

vector<tuple<int,int,int>> detectCircle(const Mat& src, const Mat& votingSpace) {
    vector<tuple<int,int,int>> detected;
    for(int b = 0; b < src.rows; b++) {
        for(int a = 0; a < src.cols; a++) {
            for(int r = 0; r <= rMax-rMin; r++) {
                if(votingSpace.at<uchar>(b,a,r) > th)
                    detected.push_back(make_tuple(b,a,r+rMin));
            }
        }
    }
    return detected;
}

void drowCircle(Mat& dest, vector<tuple<int,int,int>> detected) {
    for(auto circles : detected) {
        circle(dest, Point(get<1>(circles), get<0>(circles)), get<2>(circles), Scalar(0,0,255), 2);
        circle(dest, Point(get<1>(circles), get<0>(circles)), 1, Scalar(0,0,255));
    }
}

void houghCircle(const Mat& src, Mat& dest) {
    Mat blur, edge, votingSpace;
    src.copyTo(dest);
    GaussianBlur(src, blur, Size(3,3), 0, 0);
    cvtColor(blur, blur, COLOR_BGR2GRAY);
    createVotingSpace(blur, votingSpace);
    Canny(blur, edge, 30, 100);
    imshow("canny", edge);
    vote(edge, votingSpace);
    vector<tuple<int,int,int>> detected = detectCircle(blur, votingSpace);
    drowCircle(dest, detected);
}

int main(int argc, char const* argv[])
{
    if (argc < 2) {
        cout << "usage: " << argv[0] << " image_name" << endl;
        exit(0);
    }
    Mat src = imread(argv[1], IMREAD_COLOR);
    if(src.empty()) {
        cout << " error " << endl;
        exit(0);
    }
    imshow("src", src);
    Mat dest;
    houghCircle(src,dest);
    imshow("dest",dest);
    waitKey();
    return 0;
}