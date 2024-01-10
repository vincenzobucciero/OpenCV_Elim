#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

//valore di soglia inferiore
int lth = 60;
//valore di soglia superiore
int hth = 80;

void magnitudeAndAngle(const Mat& imageInput, Mat& mag, Mat& ang) {
    Mat dx, dy;
    Sobel(imageInput, dx, CV_32F, 1, 0);
    Sobel(imageInput, dy, CV_32F, 0, 1);

    //calcolo magnitudine
    magnitude(dx, dy, mag);     //mag = abs(dx) + abs(dy);

    //calcolo angolo di fase
    phase(dx, dy, ang, true);
}

void nonMaximaSuppression(const Mat& mag,const Mat& ang, Mat& nmsOut){
    copyMakeBorder(mag,nmsOut,1,1,1,1,BORDER_CONSTANT,Scalar(0));
    for(int y = 1; y < mag.rows; y++){
        for(int x = 1; x < mag.cols; x++){
            float angVal = ang.at<float>(y-1,x-1) > 180 ? ang.at<float>(y,x)-180 : ang.at<float>(y,x);
            if(angVal >=0 && angVal <= 22.5 || angVal <= 180 && angVal > 157.5 ){
                if(nmsOut.at<float>(y,x) < nmsOut.at<float>(y,x - 1) || nmsOut.at<float>(y,x) < nmsOut.at<float>(y,x + 1) ){
                    nmsOut.at<float>(y,x) = 0;
                }
            }
            else if(angVal > 22.5 && angVal <= 67.5){
                if(nmsOut.at<float>(y,x) < nmsOut.at<float>(y+1,x+1) || nmsOut.at<float>(y,x) < nmsOut.at<float>(y-1,x-1) ){
                    nmsOut.at<float>(y,x) = 0;
                }
            } 
            else if(angVal > 67.5 && angVal <= 112.5){
                if(nmsOut.at<float>(y,x) < nmsOut.at<float>(y-1,x) || nmsOut.at<float>(y,x) < nmsOut.at<float>(y+1,x) ){
                    nmsOut.at<float>(y,x) = 0;
                }
            } 
            else if(angVal > 112.5 && angVal <= 157.5 ){
                if(nmsOut.at<float>(y,x) < nmsOut.at<float>(y+1,x-1) || nmsOut.at<float>(y,x) < nmsOut.at<float>(y-1,x+1) ){
                    nmsOut.at<float>(y,x) = 0;
                }
            }  
        }
    }
}

void HThreshold(Mat& nmsOut, Mat& imageOutput){
    imageOutput = Mat::zeros(nmsOut.rows-2,nmsOut.cols-2,CV_8U);
    for(int y = 1; y < nmsOut.rows-1; y++){
        for(int x = 1; x < nmsOut.cols;x++){
            if(nmsOut.at<float>(y,x) > hth)
                imageOutput.at<uchar>(y-1,x-1) = 255;
            else if(nmsOut.at<float>(y,x)< lth) 
                imageOutput.at<uchar>(y-1,x-1) = 0;
            else{
                bool strongN = false;
                for(int j = -1; j >= 1; j++){
                    for(int i = -1; i >= 1; i++){
                        if(nmsOut.at<float>(y + j,x + i) > hth) strongN = true;
                    }
                }
                if(strongN) imageOutput.at<uchar>(y-1,x-1) = 255;
                else imageOutput.at<uchar>(y-1,x-1) = 0;
            }
        }
    }
}

void cannyAlgorithm(const Mat& imageInput, Mat& imageOutput) {
    Mat blur, mag, ang, nmsOut;
    GaussianBlur(imageInput, blur, Size(5,5), 0, 0);
    magnitudeAndAngle(imageInput, mag, ang);
    nonMaximaSuppression(mag, ang, nmsOut);
    HThreshold(nmsOut, imageOutput);
}

int main(int argc, char**argv) {
    if(argc < 2) {
        cout << "Usage: " << argv[0] << " <Image_Path>" << endl;
        exit(0);
    }
    Mat imageInput = imread(argv[1], IMREAD_GRAYSCALE);
    if(imageInput.empty()) {
        cout << "Error reading image" << endl;
        exit(0);
    }

    Mat imageOutput, imageOutputGaussianBlur, imageOutputCannyCV;
    imshow("Original Image", imageInput);
    //GaussianBlur(imageInput, imageOutputGaussianBlur, Size(3, 3), 0, 0);
    //imshow("Image w GaussianBlur", imageOutputGaussianBlur);
    cannyAlgorithm(imageInput,imageOutput);
    imshow("Canny Algorithm", imageOutput);
    
    //confronto con canny di openCV
    Canny(imageInput, imageOutputCannyCV, 30, 100);
    imshow("cvCanny", imageOutputCannyCV);
    waitKey(0);

    return 0;
}