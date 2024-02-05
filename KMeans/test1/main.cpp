#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdlib.h>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

int clusterNum = 5;
double th = 0.05;

void initializeCluster(const Mat& img, vector<Scalar>& clusterCenters, vector<vector<Point>> ptInCluster) {
    RNG clusterGenerator(getTickCount());
    for(auto c = 0; c < clusterNum; c++) {
        Point cluster;
        cluster.y = clusterGenerator.uniform(0, img.rows);
        cluster.x = clusterGenerator.uniform(0, img.cols);
        Scalar clusterValue = img.at<Vec3b>(cluster);
        clusterCenters.push_back(clusterValue);
        vector<Point> clusterGroup;
        ptInCluster.push_back(clusterGroup);
    }
}

void populateCluster(const Mat& img, vector<Scalar>& clusterCenters, vector<vector<Point>> ptInCluster) {
    for(auto y = 0; y < img.rows; y++) {
        for(auto x = 0; x < img.cols; x++) {
            double minDist = INFINITY;
            
        }
    }
}

void kMeans(const Mat& src, Mat& dest) {
    dest = Mat::zeros(src.rows, src.cols, CV_8UC3);
    vector<Scalar> clusterCenters;
    vector<vector<Point>> clusterGroup;
    initializeCluster(src, clusterCenters, clusterGroup);
    double oldVal = INFINITY;
    double newVal = 0;
    double dist = abs(oldVal-newVal);
    while(dist > th) {
        newVal = 0;
        for(auto id = 0; id < clusterNum; id++) {
            clusterGroup.at(id).clear();
        }
        //populateCluster
        //adjust
    }
    //segment
}

int main(int argc, char**argv) {
    if (argc < 2) {
        cout << "usage: " << argv[0] << " image_name" << endl;
        exit(0);
    }
    Mat src = imread(argv[1], IMREAD_COLOR);
    Mat dest;
    imshow("src", src);
    kMeans(src, dest);
    imshow("dest", dest);
    waitKey();

    return 0;
}