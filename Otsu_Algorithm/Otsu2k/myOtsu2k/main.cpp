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
        for(int x = 0; x < src.cols; x++) {
            hist.at(src.at<uchar>(y,x))++;
        }
    }
    for(int i = 0; i < 256; i++) {
        hist.at(i) /= (src.rows*src.cols);
    }

    return hist;
}

vector<int> otsu2k(const Mat& src) {
    Mat blur;
    GaussianBlur(src, blur, Size(3, 3), 0, 0);
    vector<float> hist = normHistogram(blur);
    vector<float> cumAvg(3, 0.0f);  // 3 medie cumulative m1, m2, m3
    vector<float> prob(3, 0.0f);    // 3 probabilità p1, p2, p3
    vector<int> kstar(2, 0);  // 2 soglie 
    float globAvg = 0.0f;
    float interClassVariance = 0.0f;
    float maxVariance = 0.0f;

    for(int i = 0; i < 256; i++) {
        globAvg += (i+1)*hist.at(i);
    }

    for(int i = 0; i < 256-2; i++) {
        prob.at(0) += hist.at(i);
        cumAvg.at(0) += (i+1)*hist.at(i);
        for(int j = i+1; j < 256-1; j++) {
            prob.at(1) += hist.at(j);
            cumAvg.at(1) += (j+1)*hist.at(j);
            for(int k = j+1; k < 256; k++) {
                prob.at(2) += hist.at(k);
                cumAvg.at(2) += (k+1)*hist.at(k);
                for(int w = 0; w < 3; w++) {
                    if(prob.at(w)) {
                        interClassVariance += prob.at(w)*pow(cumAvg.at(w)/prob.at(w)-globAvg, 2);
                    }
                }
                if(interClassVariance > maxVariance) {
                    maxVariance = interClassVariance;
                    kstar.at(0) = i;
                    kstar.at(1) = j;
                }
                interClassVariance = 0.0f;
            }
            prob.at(2) = cumAvg.at(2) = 0.0f;
        } 
        prob.at(1) = cumAvg.at(1) = 0.0f;
    }
    return kstar;
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
    vector<int> thresholds = otsu2k(src);
    double thresholdValue = static_cast<double>(thresholds[0]);
    threshold(src, dest, thresholdValue, 255, THRESH_BINARY);
    imshow("otsu2k image", dest);
    waitKey(0);

    return 0;
}