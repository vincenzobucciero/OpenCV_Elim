#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

const float RADC = CV_PI/180;
const int th = 190;

class rect {
        Point begin, end;
    public:
        rect(Point begin, Point end) : begin(begin), end(end) { };
        inline Point getBegin() { return begin; }
        inline Point getEnd() { return end; }
};

void creatingVotingSpace(const Mat& src, Mat& votingSpace) {
    int dist = hypot(src.rows, src.cols);
    votingSpace = Mat::zeros(2*dist, 181, CV_32F);
}

void vote(const Mat& edge, Mat& votingSpace) {
    int dist = votingSpace.rows/2;
    for(int y = 0; y < edge.rows; y++) {
        for(int x = 0; x < edge.cols; x++) {
            if(edge.at<uchar>(y,x) == 255) {
                for(int thetaIndex = 0; thetaIndex <= 180; thetaIndex++) {
                    float theta = (thetaIndex-90)*RADC;
                    float sint = sin(theta);
                    float cost = cos(theta);
                    int rhoIndex = cvRound(x*cost+y*sint)+dist;
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
                int rho = rhoIndex-dist;
                float theta = (thetaIndex-90)*RADC;
                float sint = sin(theta);
                float cost = cos(theta);
                int x0 = rho*cost;
                int y0 = rho*sint;
                Point begin, end;
                begin.x = x0-1000*-sint;
                begin.y = y0-1000*cost;
                end.x = x0+1000*-sint;
                end.y = y0+1000*cost;

                detected.push_back(rect(begin, end));
            }
        }
    }
}

void drowRect(vector<rect> detected, Mat& dest) {
    for(auto r : detected) {
        line(dest, r.getBegin(), r.getEnd(), Scalar(0,0,255));
    }
}

void myHoughLines(const Mat& src, Mat& dest) {
    Mat blur, edge, votingSpace;
    vector<rect> detected;
    src.copyTo(dest);
    GaussianBlur(src, blur, Size(3,3), 0, 0);
    cvtColor(blur, blur, COLOR_BGR2GRAY);
    creatingVotingSpace(blur, votingSpace);
    Canny(blur, edge, 50, 150);
    vote(edge, votingSpace);
    detectRect(votingSpace, detected);
    drowRect(detected, dest);
}

int main(int argc, char**argv) {
    if(argc < 2) {
        cout << "error, using " << argv[0] << endl;
        exit(0);
    }
    Mat src = imread(argv[1], IMREAD_COLOR);
    if(src.empty()) {
        cout << "error, can't open/read image" << endl;
        exit(0);
    }
    imshow("Original image", src);
    Mat dest;
    myHoughLines(src, dest);
    imshow("image with lines", dest);
    
    waitKey(0);

    return 0;
}