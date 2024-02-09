#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <stack>

using namespace std;
using namespace cv;

const int th = 200;
const float minAreaFactor = 0.01;
const int regionNumber = 100;

const Point shiftPoint[8] {
    Point(-1,-1), Point(-1,0), Point(-1,1),
    Point(0,-1),               Point(0,1),
    Point(1,-1), Point(1,0), Point(1,1)
};

void grow(const Mat& src, const Mat& dest, Mat& mask, Point seed) {
    stack<Point> front;
    front.push(seed);
    while(!front.empty()) {
        Point center = front.top();
        mask.at<uchar>(center) = 1;
        front.pop();
        for(int i = 0; i < 8; i++) {
            Point neigh = center+shiftPoint[i];
            if(neigh.x < 0 || neigh.y < 0 || neigh.x >= src.cols-1 || neigh.y >= src.rows-1) {
                continue;
            }
            else {
                int delta = cvRound(pow(src.at<Vec3b>(center)[0] - src.at<Vec3b>(neigh)[0], 2) +
                                    pow(src.at<Vec3b>(center)[1] - src.at<Vec3b>(neigh)[1], 2) +
                                    pow(src.at<Vec3b>(center)[2] - src.at<Vec3b>(neigh)[2], 2));
                if(mask.at<uchar>(neigh) == 0 && dest.at<uchar>(neigh) == 0 && delta < th) {
                    front.push(neigh);
                }
            }
        }
    }
}

void regionGrowing(const Mat& src, Mat& dest) {
    dest = Mat::zeros(src.rows, src.cols, CV_8UC1);
    Mat mask = Mat::zeros(src.rows, src.cols, CV_8UC1);
    uchar padding = 1;
    int minAreaRegion = src.rows*src.cols*minAreaFactor;
    for(int y = 0; y < src.rows; y++) {
        for(int x = 0; x < src.cols; x++) {
            if(dest.at<uchar>(Point(x, y)) == 0) {
                grow(src, dest, mask, Point(x, y));
                if(sum(mask).val[0] > minAreaRegion) {
                    imshow("region", mask*255);
                    waitKey(0);
                    dest = dest+mask*padding;
                    if(++padding > regionNumber) {
                        cout << " error " << endl;
                        exit(0);
                    }
                } else
                    dest = dest+mask*255;
                    mask = mask-mask;
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