/**
 * Sostituire al valore di ogni pixel il valore medio dei livelli di grigio in un intorno 3x3
*/

#include <opencv4/opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

cv::Mat replaceImagePixel(const cv::Mat& inputImage) {
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