#include <opencv4/opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

Mat

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

    Mat imageOutput;
    
    Mat mask = Mat::ones(3, 3, CV_32F)/(dim*dim);

    filter2D(imageInput, imageOutput, imageInput.type(), mask);

    cv::imshow("Original Image", imageInput);
    cv::waitKey(0);
    cv::imshow("Filter2d Image", imageOutput);
    cv::waitKey(0);

    return 0;
}