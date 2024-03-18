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
            hist.at(src.at<uchar>(y, x))++;
        }
    }
    for(int i = 0; i < 256; i++) {
        hist.at(i) /= src.rows*src.cols;
    }
    return hist;
}

vector<int> kStar(const Mat& src) {
    Mat blur;
    GaussianBlur(src, blur, Size(3,3), 0, 0);
    vector<float> hist = normHistogram(blur);

    vector<float> prob(3, 0.0f);
    vector<float> cumAvg(3, 0.0f);
    vector<int> kstar(2, 0);
    float globAvg = 0.0f;
    float maxVariance = 0.0f;
    float sigma = 0.0f;
    
    for(int i = 0; i < 256; i++) {
        globAvg += (i+1)*hist.at(i);
    }

    for(int i = 0; i < 254; i++) {
        prob.at(0) += hist.at(i);
        cumAvg.at(0) += (i+1)*hist.at(i);

        for(int j = i+1; j < 255; j++) {
            prob.at(1) += hist.at(j);
            cumAvg.at(1) += (j+1)*hist.at(j);
            for(int k = j+1; k < 256; k++) {
                prob.at(2) += hist.at(k);
                cumAvg.at(2) += (k+1)*hist.at(k);
                for(int z = 0; z < 3; z++) {
                    if(prob.at(z)) {
                        sigma += prob.at(z)*pow(cumAvg.at(z)/prob.at(z)-globAvg, 2);
                    }
                }
                if(sigma > maxVariance) {
                    maxVariance = sigma;
                    kstar.at(0) = i;
                    kstar.at(1) = j;
                }
                sigma = 0.0f;
            }
            prob.at(2) = cumAvg.at(2) = 0.0f;
        }
        prob.at(1) = cumAvg.at(1) = 0.0f; 
    }
    return kstar;
}

void myThreshold(const Mat& src, Mat& dest) {
    vector<int> kstar = kStar(src);
    dest = Mat::zeros(src.size(), CV_8U);
    for(int y = 0; y < src.rows; y++) {
        for(int x = 0; x < src.cols; x++) {
            if(src.at<uchar>(y, x) > kstar.at(1))
                dest.at<uchar>(y, x) = 255;
            else if(src.at<uchar>(y, x) >= kstar.at(0) && src.at<uchar>(y, x) <= kstar.at(1))
                dest.at<uchar>(y, x) = 127;
            else
                dest.at<uchar>(y, x) = 0;
        }
    }
}

//ciao <3, ho fame 

int main(int argc, char**argv) {
    if(argc < 2) {
        cout << "error, ha ha ha c'è un errore" << endl;
        exit(0);
    }
    Mat src = imread(argv[1], IMREAD_GRAYSCALE);
    if(src.empty()) {
        cout << "can not open/read image" << argv[1] << endl;
        exit(0);
    }
    imshow("original image, bravo, sei originale!", src);
    Mat dest;
    myThreshold(src, dest);
    imshow("otsu2k, ciao mi chiamo otsu2k", dest);

    waitKey(0);

    return 0;
}