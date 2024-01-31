#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <stack>

using namespace std;
using namespace cv;

int th = 200;
float minAreaFactor = 0.01;
int regionNumber = 100;

const Point shiftPoint[8] {
    Point(-1,-1), Point(-1,0), Point(-1,1),
    Point(0,-1),               Point(0,1),
    Point(1,-1), Point(1,0), Point(1,1)
};

void grow(const Mat& src, const Mat& dest, Mat& mask, Point seed) {
    stack<Point> stackPoint;
    stackPoint.push(seed);
    while(!stackPoint.empty()) {
        Point center = stackPoint.top();
        mask.at<uchar>(center) = 1;
        stackPoint.pop();
        for(int i = 0; i < 8; i++) {
            Point estimatingPoint = center+shiftPoint[i];
            if(estimatingPoint.x < 0 || estimatingPoint.y < 0 || estimatingPoint.x >= src.cols-1 || estimatingPoint.y >= src.rows-1) {
                continue;
            } else {
                int delta = int(pow(src.at<Vec3b>(center)[0] - src.at<Vec3b>(estimatingPoint)[0], 2) + 
                                pow(src.at<Vec3b>(center)[1] - src.at<Vec3b>(estimatingPoint)[1], 2) +
                                pow(src.at<Vec3b>(center)[2] - src.at<Vec3b>(estimatingPoint)[2], 2));
                if(dest.at<uchar>(estimatingPoint) == 0 && mask.at<uchar>(estimatingPoint) == 0 && delta < th) {
                    stackPoint.push(estimatingPoint);
                }
            }
        }
    }
}

void regionGrowing(const Mat& src, Mat& dest) {
    dest = Mat::zeros(src.rows, src.cols, CV_8UC1);
    Mat mask = Mat::zeros(src.rows, src.cols, CV_8UC1);
    int minRegionArea = int(minAreaFactor*src.rows*src.cols);
    uchar padding = 1;

    for(int y = 0; y < src.rows; y++) {
        for(int x = 0; x < src.cols; x++) {
            if(dest.at<uchar>(Point(x,y)) == 0) {
                grow(src, dest, mask, Point(x,y));
                if(sum(mask).val[0] > minRegionArea) {
                    imshow("region", mask*255);
                    waitKey(0);
                    dest = dest+mask*padding;
                    if(++padding > regionNumber) {
                        cout <<"errror" << endl;
                        exit(0);
                    }
                } else
                    dest = dest+mask*255;
                    mask = mask - mask;
            }
        }
    }
}

int main(int argc, char**argv) {
    if(argc < 2) {
        cout << "error using " << argv[0] << " ./imageName" << endl;
        exit(0);
    }
    Mat src = imread(argv[1], IMREAD_COLOR);
    if(src.empty()) {
        cout << "error can't open/read image" << endl;
        exit(0);
    }
    imshow("original image", src);
    Mat dest;
    regionGrowing(src, dest);
    imshow("RG image", dest);
    waitKey(0);

    return 0;
}