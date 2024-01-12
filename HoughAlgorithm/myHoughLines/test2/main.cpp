#include <iostream>
#include <opencv4/opencv2/imgproc/imgproc.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
#include <vector>

using namespace std;
using namespace cv;

const float RADC = CV_PI/180;
const int th = 150;

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
    for(int y = 0; y < edge.rows; y++) {
        for(int x = 0; x < edge.cols; x++) {
            if(edge.at<uchar>(y, x) == 255) {
                for(int thetaIndex = 0; thetaIndex <= 180; thetaIndex++) {
                    float theta = (thetaIndex-90)*RADC;
                    float sint = sin(theta);
                    float cost = cos(theta);
                    int rhoIndex = cvRound(x*cost + y*sint) + dist;
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
                begin.x = cvRound(x0-1000*-sint);
                begin.y = cvRound(y0-1000*cost);
                end.x = cvRound(x0+1000*-sint);
                end.y = cvRound(y0+1000*cost);

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

void houghLinesTest2(const Mat& imageInput, Mat& imageOutput) {
    Mat blur, edge, votingSpace;
    vector<rect> detected;
    imageInput.copyTo(imageOutput);
    GaussianBlur(imageInput, blur, Size(5, 5), 0, 0);
    cvtColor(blur, blur, COLOR_BGR2GRAY);
    createVotingSpace(blur, votingSpace);
    Canny(blur, edge, 120, 140);
    vote(edge, votingSpace);
    detectRect(votingSpace, detected);
    drowRect(detected, imageOutput);
}

int main(int argc, char**argv) {
    if(argc < 2) {
        cout << "error using, " << argv[0] << "image name" << endl;
        exit(0);
    }
    Mat imageInput = imread(argv[1], IMREAD_COLOR);
    if(imageInput.empty()) {
        cout << "error, can't open / read image" << endl;
        exit(0);
    }
    Mat imageOutput;
    imshow("original image", imageInput);
    houghLinesTest2(imageInput, imageOutput);
    imshow("hough lines imageOutput", imageOutput);

    waitKey(0);

    return 0;

}