/**
 * Implementare il filtro di Sobel (gx e gy) utilizzando la funzione di correlazione/convoluzione (o filter2D())
 * 
 * Calcolare la risposta di entrambi i filtri
 * 
 * Calcolare la magnitudo del gradiente (entrambe le formulazioni)
 * 
 * Utilizzare la risposta ottenuta per effettuare lo sharpening di un'immagine
*/

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat sobelFilter(const Mat& imageInput, const Mat& mask) {
    Mat imageOutput;
    filter2D(imageInput, imageOutput, -1, mask);
    return imageOutput;
}

int main(int argc, char**argv) {
    if(argc < 2) {
        cout << "Error, insert name " << argv[0] << endl;
        exit(0);
    }
    Mat imageInput = imread(argv[1], IMREAD_GRAYSCALE);
    if(imageInput.empty()) {
        cout << "Error during reading imageInput" << endl;
        exit(0);
    }

    //definisco i kernel
    Mat filterSobelGx = (Mat_<float>(3,3) << -1.0, -2.0, -1.0,
                                              0.0, 0.0, 0.0,
                                              1.0, 2.0, 1.0);

    Mat filterSobelGy = (Mat_<float>(3,3) << -1.0, 0.0, 1.0,
                                             -2.0, 0.0, 2.0,
                                             -1.0, 0.0, 1.0);

    Mat gradientX = sobelFilter(imageInput, filterSobelGx);
    Mat gradientY = sobelFilter(imageInput, filterSobelGy);

    // Calcolo della magnitudo del gradiente (entrambe le formulazioni)
    Mat magnitude, magnitudeAlt;
    magnitude = abs(gradientX) + abs(gradientY);
    magnitudeAlt = Mat(imageInput.size(), imageInput.type());
    for (int i = 0; i < imageInput.rows; ++i) {
        for (int j = 0; j < imageInput.cols; ++j) {
            float gx = gradientX.at<float>(i, j);
            float gy = gradientY.at<float>(i, j);
            magnitudeAlt.at<uchar>(i, j) = sqrt(gx * gx + gy * gy);
        }
    }

    // Sharpening dell'immagine usando la magnitudo del gradiente
    Mat sharpened;
    addWeighted(imageInput, 1.5, magnitude, -0.5, 0, sharpened);

    // Visualizzazione delle immagini
    imshow("Original", imageInput);
    imshow("Sobel Gx", gradientX);
    imshow("Sobel Gy", gradientY);
    imshow("Magnitude", magnitude);
    imshow("Magnitude Alt", magnitudeAlt);
    imshow("Sharpened", sharpened);

    waitKey(0);

    return 0;
}