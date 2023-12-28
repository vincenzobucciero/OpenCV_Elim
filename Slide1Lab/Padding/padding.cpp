/**
 * Realizza una funzione che effettui il padding di un immagine
*/

#include <opencv4/opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

cv::Mat paddingImage(const cv::Mat& inputImage, int top, int bottom, int left, int right, const cv::Scalar& value) {
    cv::Mat outputImage;
    cv::copyMakeBorder(inputImage, outputImage, top, bottom, left, right, cv::BORDER_REFLECT, value);
    return outputImage;
}

int main() {
    cv::Mat image = cv::imread("lena.png", IMREAD_GRAYSCALE);

    //controllo apertura immagine
    if(image.empty()) {
        std::cout << "Impossibile leggere l'immagine" << std::endl;
        return -1;
    }

    int top = 40, bottom = 40, left = 40, right = 40;
    cv::Scalar paddingValue = cv::Scalar(0, 0, 0); //black

    //richiamo funzione
    cv::Mat outputImage = paddingImage(image, top, bottom, left, right, paddingValue);

    // Visualizza l'immagine originale e quella con il padding
    cv::namedWindow("Original Image", cv::WINDOW_NORMAL);
    cv::imshow("Original Image", image);

    cv::namedWindow("Padded Image", cv::WINDOW_NORMAL);
    cv::imshow("Padded Image", outputImage);

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}