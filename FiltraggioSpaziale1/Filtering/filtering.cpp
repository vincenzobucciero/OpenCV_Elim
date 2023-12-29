#include <opencv4/opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    int dim;

    if(argc < 2) {
        cout << "Error, using " << argv[0] << " image name" << endl;
        exit(0);
    }

    String imageInput = argv[1];
    cv::Mat image;

    image = imread(imageInput, IMREAD_COLOR);

    if(image.empty()) {
        cout << "Errore apertura immagine" << endl;
        exit(0);
    }

    cout << "Inserisci dimesione filtro:  ";
    cin >> dim;

    cv::Mat average_filter = cv::Mat::ones(dim, dim, CV_32F)/(float)(dim*dim);

    cv::Mat outputImage1;
    filter2D(image, outputImage1, image.type(), average_filter);
    cv::Mat outputImage2;
    blur(image, outputImage2, Size(dim, dim));
    cv::Mat outputImage3;
    boxFilter(image, outputImage3, image.type(), Size(dim, dim));
    cv::Mat outputImage4;
    medianBlur(image, outputImage4, dim);
    cv::Mat outputImage5;
    GaussianBlur(image, outputImage5, Size(dim, dim), 0, 0);

    cv::imshow("Original Image", image);
    cv::waitKey(0);
    cv::imshow("Filter2d Image", outputImage1);
    cv::waitKey(0);
    cv::imshow("Blur Image", outputImage2);
    cv::waitKey(0);
    cv::imshow("BoxFilter Image", outputImage3);
    cv::waitKey(0);
    cv::imshow("MedianBlur Image", outputImage4);
    cv::waitKey(0);
    cv::imshow("GaussianBlur Image", outputImage5);
    cv::waitKey(0);

    return 0;
}