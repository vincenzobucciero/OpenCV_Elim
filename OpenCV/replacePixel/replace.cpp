/**
 * Sostituire al valore di ogni pixel il valore medio dei livelli di grigio in un intorno 3x3
*/

#include <opencv4/opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

cv::Mat paddingImage(const cv::Mat& inputImage, int top, int bottom, int left, int right) {
    cv::Mat outputImage;
    cv::copyMakeBorder(inputImage, outputImage, top, bottom, left, right, cv::BORDER_REFLECT);
    return outputImage;
}

int sommaPixel(const cv::Mat& inputImage, int i, int j) {
    int somma = 0;
    somma = inputImage.at<uchar>(i-1, j+1) + inputImage.at<uchar>(i, j+1) + inputImage.at<uchar>(i-1, j) + inputImage.at<uchar>(i, j) + 
                inputImage.at<uchar>(i+1, j) + inputImage.at<uchar>(i-1, j-1) + inputImage.at<uchar>(i, j-1) + inputImage.at<uchar>(i-1, j+1) + inputImage.at<uchar>(i+1, j+1);
    return somma;
}

cv::Mat replaceImagePixel(const cv::Mat& inputImage) {
    /*
    cv::Mat outputImage = inputImage.clone(); //crea copia dell'immagine

    for(int i = 1; i < inputImage.rows - 1; ++i) {
        for(int j = 1; j < inputImage.cols - 1; ++j) {
            cv::Rect roi(j-1, i-1, 3, 3); //roi è la regione di interesse che definiamo
            cv::Mat neighborhood = inputImage(roi);

            // Calcolo della media dei valori di pixel nell'intorno
            double mean = cv::mean(neighborhood)[0];

            // Sostituzione del valore del pixel con la media calcolata
            outputImage.at<uchar>(i, j) = static_cast<uchar>(mean);
        }
    }

    return outputImage;
    */

    int top = 20, bottom = 20, left = 20, right = 20;
    // cv::Scalar paddingValue = cv::Scalar(0, 0, 0); //black

    cv::Mat outputImage = paddingImage(inputImage, top, bottom, left, right);
    for(int i = 1; i < outputImage.rows-1; i++) {
        for(int j = 1; j < outputImage.cols-1; j++) {
            outputImage.at<uchar>(i-1, j-1) = static_cast<uchar>(sommaPixel(outputImage, i, j) / 9);
        }
    }

    return outputImage;
}

int main() {
    cv::Mat image = cv::imread("lena.png", IMREAD_GRAYSCALE);

    //controllo apertura immagine
    if(image.empty()) {
        std::cout << "Impossibile leggere l'immagine" << std::endl;
        return -1;
    }

    // Applica la sostituzione con la media locale
    cv::Mat replacedImage = replaceImagePixel(image);

    cv::imshow("Original Image", image);
    cv::imshow("ReplacePixel Image", replacedImage);

    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}