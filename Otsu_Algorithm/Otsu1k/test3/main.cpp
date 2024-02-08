#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <vector>

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

vector<float> probability(vector<float> hist) {
    vector<float> prob(256, 0.0f);
    prob.at(0) = hist.at(0);
    for(int i = 1; i < 256; i++) {
        prob.at(i) = prob.at(i-1)+hist.at(i);
    }
    return prob;
}

vector<float> cumulativeAvg(vector<float> hist) {
    vector<float> cumAvg(256, 0.0f);
    cumAvg.at(0) = hist.at(0);
    for(int i = 1; i < 256; i++) {
        cumAvg.at(i) = cumAvg.at(i-1)+i*hist.at(i);
    }
    return cumAvg;
}

float globalAvg(vector<float> hist) {
    float globAvg = hist.at(0);
    for(int i = 1; i < 256; i++) {
        globAvg += i*hist.at(i);
    }
    return globAvg;
}

vector<float> interClassVariance(vector<float> prob, vector<float> cumAvg, float globAvg) {
    vector<float> sigma(256, 0.0f);
    float num, den;
    for(int i = 0; i < 256; i++) {
        num = pow(((globAvg*prob.at(i))-cumAvg.at(i)), 2);
        den = prob.at(i)*(1-prob.at(i));
        sigma.at(i) = den == 0 ? 0 : num/den;
    }
    return sigma;
}

int kstar(vector<float> sigma) {
    float maxVariance = sigma.at(0);
    int k = 0;
    for(int i = 1; i < 256; i++) {
        if(sigma.at(i) > maxVariance) {
            maxVariance = sigma.at(i);
            k = i;
        }
    }
    return k;
}

int otsu1k(const Mat& src) {
    Mat blur;
    GaussianBlur(src, blur, Size(3,3), 0, 0);
    vector<float> hist = normHistogram(blur);
    vector<float> prob = probability(hist);
    vector<float> cumAvg = cumulativeAvg(hist);
    float globAvg = globalAvg(hist);
    vector<float> sigma = interClassVariance(prob, cumAvg, globAvg);
    int k = kstar(sigma);
    return k;
}


int main(int argc, char** argv){

    Mat src, dst;

    src = imread(argv[1], IMREAD_GRAYSCALE);
    if( src.empty() )
        return -1;

    imshow("Orginal Img", src);

    threshold(src, dst, otsu1k(src), 255, THRESH_BINARY);
    
    imshow("Otsu", dst);
    waitKey(0);


    return 0;
}