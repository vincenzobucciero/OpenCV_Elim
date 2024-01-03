#include <opencv4/opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat correlation1D(const Mat& imageInput) {
    Mat imageOutput = Mat::zeros(imageInput.size(), imageInput.type());

    for(int x = 0; x < imageInput.rows; ++x) {
        for(int y = 1; y < imageInput.cols - 1; ++y) {
            float sum = 0.0;
            sum += imageInput.at<float>(x, y-1)*(float)(1.0f/3.0f);
            sum += imageInput.at<float>(x, y)*(float)(1.0f/3.0f);
            sum += imageInput.at<float>(x, y+1)*(float)(1.0f/3.0f);
        
            imageOutput.at<float>(x, y) = sum;
        }
    }

    return imageOutput;
}

int main(int argc, char**argv) {
    //prendo img in input da riga di comando
    if(argc < 2) {
        cout << "Errore -> devi inserire " << argv[0] << " il nome dell immagine" << endl;
        exit(0);
    }
    String imageName = argv[1];
    
    //immagine in input
    Mat imageInput;
    imageInput = imread(imageName, IMREAD_COLOR);

    //controllo l'apertura dell'immagine
    if(imageInput.empty()) {
        cout << "Errore nell'apertura dell'immagine" << endl;
        return -1;
    }

    int dim = 3;
    
    Mat mask = Mat::ones(3, 3, CV_32F)/(dim*dim);

    Mat imageOutput;
    filter2D(imageInput, imageOutput, imageInput.type(), mask);
    Mat imageOutputMyCorrelation = correlation1D(imageInput);

    cv::imshow("Original Image", imageInput);
    cv::imshow("Filter2d Image", imageOutput);
    cv::imshow("Custom myCorrelation Image", imageOutputMyCorrelation);
    cv::waitKey(0);

    return 0;
}

//TODO: another file cpp w correlation and convolution