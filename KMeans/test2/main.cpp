#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <ctime>

using namespace std;
using namespace cv;

void createClusterInfo(const Mat& src, int clustersNum, vector<Scalar>& clustersCenters, vector<vector<Point>>& ptInClusters) {
    RNG random(getTickCount());
    for(int k = 0; k < clustersNum; k++) {
        Point pixel;
        pixel.x = random.uniform(0, src.cols);
        pixel.y = random.uniform(0, src.rows);
        Scalar center = src.at<Vec3b>(pixel);
        Scalar centerColor(center.val[0], center.val[1], center.val[2]);
        clustersCenters.push_back(centerColor);
        vector<Point> ptInClustersK;
        ptInClusters.push_back(ptInClustersK);
    }
}

double colorDistance(Scalar pixel, Scalar clusterPixel) {
    double blue = pixel.val[0] - clusterPixel.val[0];
    double green = pixel.val[1] - clusterPixel.val[1];
    double red = pixel.val[2] - clusterPixel.val[2];
    double distance = sqrt(pow(blue,2)+pow(green,2)+pow(red,2));
    return distance;
}

void findKCluster(const Mat& src, int clustersNum, vector<Scalar>& clustersCenters, vector<vector<Point>>& ptInClusters) {
    for(int y = 0; y < src.rows; y++) {
        for(int x = 0; x < src.cols; x++) {
            double minDistance = INFINITY;
            int indexCluster = 0;
            Scalar pixel = src.at<Vec3b>(y,x);
            for(int k = 0; k < clustersNum; k++) {
                double distance = colorDistance(pixel, clustersCenters[k]);
                if(distance < minDistance) {
                    minDistance = distance;
                    indexCluster = k;
                }
            }
            ptInClusters[indexCluster].push_back(Point(x,y));
        }
    }
}

double fixClusterCenters(const Mat& src, int clustersNum, vector<Scalar>& clustersCenters, vector<vector<Point>>& ptInClusters, double& oldValue, double newValue) {
    double diffChange = 0;
    for(int k = 0; k < clustersNum; k++) {
        vector<Point> ptInClusterK = ptInClusters[k];
        double newBlue = 0;
        double newGreen = 0;
        double newRed = 0;
        for(int i = 0; i < ptInClusterK.size(); i++) {
            Scalar pixel = src.at<Vec3b>(ptInClusterK[i]);
            newBlue += pixel.val[0];
            newGreen += pixel.val[1];
            newRed += pixel.val[2];
        }
        newBlue /= ptInClusterK.size();
        newGreen /= ptInClusterK.size();
        newRed /= ptInClusterK.size();
        Scalar center(newBlue, newGreen, newRed);
        newValue += colorDistance(center, clustersCenters[k]);
        clustersCenters[k] = center;
    }
    newValue /= clustersNum;
    diffChange = abs(oldValue-newValue);
    oldValue = newValue;
    return diffChange;
}

void applyClusters(Mat& dest, int clustersNum, vector<Scalar>& clustersCenters, vector<vector<Point>>& ptInClusters) {
    for(int k = 0; k < clustersNum; k++) {
        Scalar pixel = clustersCenters[k];
        vector<Point> ptInClusterK = ptInClusters[k];
        for(int i = 0; i < ptInClusterK.size(); i++) {
            dest.at<Vec3b>(ptInClusterK[i])[0] = pixel.val[0];
            dest.at<Vec3b>(ptInClusterK[i])[1] = pixel.val[1];
            dest.at<Vec3b>(ptInClusterK[i])[2] = pixel.val[2];
        }
    }
}

int main(int argc, char**argv) {
    if(argc < 2) {
        cout << " error " << endl;
        exit(0);
    }
    Mat src = imread(argv[1], IMREAD_COLOR);
    int clustersNum = stoi(argv[2]);
    vector<Scalar> clustersCenters;
    vector<vector<Point>> ptInClusters;
    double oldValue = INFINITY;
    double newValue = 0;
    double diffChange = oldValue-newValue;
    double ht = 0.1;

    createClusterInfo(src, clustersNum, clustersCenters, ptInClusters);

    while(diffChange > ht) {
        newValue = 0;
        for(int k = 0; k < clustersNum; k++) {
            ptInClusters[k].clear();
        }
        findKCluster(src, clustersNum, clustersCenters, ptInClusters);
        diffChange = fixClusterCenters(src, clustersNum, clustersCenters, ptInClusters, oldValue, newValue);
    }
    Mat dest = src.clone();
    applyClusters(dest, clustersNum, clustersCenters, ptInClusters);

    imshow("src", src);
    imshow("dest", dest);
    waitKey(0);

    return 0;
}