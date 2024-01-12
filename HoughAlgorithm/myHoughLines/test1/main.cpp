//test1 hough lines

#include <opencv4/opencv2/imgproc/imgproc.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <stdlib.h>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

const float RADC = CV_PI/180;
const int th = 200;

class rect {
        Point begin, end;
    public:
        rect(Point begin, Point end) : begin(begin), end(end) { };
        inline Point getBegin() {return begin;}
        inline Point getEnd() {return end;}
};

void createVotingSpace(const Mat& imageInput, Mat& votingSpace) {
    int dist = hypot(imageInput.rows, imageInput.cols);
    votingSpace = Mat::zeros(2*dist, 181, CV_32F);
}

void vote(const Mat& edge, Mat& votingSpace) {
    int dist = votingSpace.rows/2;
    for(int x = 0; x < edge.rows; x++) {
        for(int y = 0; y < edge.cols; y++) {
            if(edge.at<uchar>(x, y) == 255) {
                for(int thetaIndex = 0; thetaIndex <= 180; thetaIndex++) {
                    float theta = (thetaIndex-90)*RADC;
                    float sin_t = sin(theta);
                    float cos_t = cos(theta);
                    int rhoIndex = cvRound(x*cos_t + y*sin_t) + dist;
                    votingSpace.at<float>(rhoIndex, thetaIndex)++;
                }
            }
        }
    }
}

void detectRect(const Mat& votingSpace, vector<rect>& detected) {
    int dist = votingSpace.rows/2;
    for(int rhoIndex = 0; rhoIndex < votingSpace.rows; rhoIndex++) {
        for(int thetaIndex = 0; thetaIndex <= 180; thetaIndex++) {
            if(votingSpace.at<float>(rhoIndex, thetaIndex) > th) {
                int rho = rhoIndex - dist;
                float theta = (thetaIndex-90)*RADC;
                float sin_t = sin(theta);
                float cos_t = cos(theta);
                int x_0 = rho*sin_t;
                int y_0 = rho*cos_t;
                Point begin, end;
                begin.x = cvRound(x_0-1000*cos_t);
                begin.y = cvRound(y_0-1000*-sin_t);
                end.x = cvRound(x_0+1000*cos_t);
                end.y = cvRound(y_0+1000*-sin_t);

                detected.push_back(rect(begin, end));
            }
        }
    }
}

void drowRect(vector<rect>& detected, Mat& imageOutput) {
    for(auto r : detected) {
        line(imageOutput, r.getBegin(), r.getEnd(), Scalar(0,0,255));
    }
}

void houghLines(const Mat& imageInput, Mat& imageOutput) {
    Mat blur, edge, votingSpace;
    vector<rect> detected;
    //copy
    imageInput.copyTo(imageOutput);
    //gaussblur
    GaussianBlur(imageInput, blur, Size(5, 5), 0, 0);
    //Color
    cvtColor(blur, blur, COLOR_BGR2GRAY);
    //createVotingSpace
    createVotingSpace(blur, votingSpace);
    //canny
    Canny(blur, edge, 120, 140);
    //vote
    vote(edge, votingSpace);
    //detectrect
    detectRect(votingSpace, detected);
    //drowRect
    drowRect(detected, imageOutput);
}

int main(int argc, char**argv) {
    if(argc < 2) {
        cout << "error " << argv[0] << ", using image name" << endl;
        exit(0);
    }
    Mat imageInput = imread(argv[1], IMREAD_COLOR);
    if(imageInput.empty()) {
        cout << "error, can't open / read image" << endl;
        exit(0);
    }
    Mat imageOutput;
    imshow("original image", imageInput);
    houghLines(imageInput, imageOutput);
    imshow("hough lines imageOutput", imageOutput);

    waitKey(0);

    return 0;
}