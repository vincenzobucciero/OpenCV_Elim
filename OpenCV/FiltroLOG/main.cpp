/**
 * Utilizzando le funzioni di OpenCV implementare il LoG
 * 
 * Implementare una funzione che trovi i punti di zero-crossing
*/

#include <opencv2/opencv.hpp>

using namespace cv;

// Funzione per trovare i punti di zero-crossing
Mat findZeroCrossings(const Mat &logImage) {
    Mat zeroCrossings(logImage.rows, logImage.cols, CV_8U, Scalar(0));

    for (int i = 1; i < logImage.rows - 1; ++i) {
        for (int j = 1; j < logImage.cols - 1; ++j) {
            // Verifica se il segno cambia in almeno una direzione
            if ((logImage.at<float>(i, j) > 0 && logImage.at<float>(i, j + 1) < 0) ||
                (logImage.at<float>(i, j) > 0 && logImage.at<float>(i, j - 1) < 0) ||
                (logImage.at<float>(i, j) > 0 && logImage.at<float>(i + 1, j) < 0) ||
                (logImage.at<float>(i, j) > 0 && logImage.at<float>(i - 1, j) < 0)) {
                zeroCrossings.at<uchar>(i, j) = 255;
            }
        }
    }

    return zeroCrossings;
}

int main(int argc, char**argv) {
    // Carica un'immagine in scala di grigi
    Mat image = imread(argv[1], IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Errore nel caricare l'immagine." << std::endl;
        return -1;
    }

    // Applica il filtro Laplaciano della Gaussiana (LoG)
    Mat logImage;
    GaussianBlur(image, logImage, Size(5, 5), 2.0, 2.0);  // Applica la Gaussiana
    Laplacian(logImage, logImage, CV_32F, 3, 1, 0);         // Applica il Laplaciano

    // Normalizza l'immagine LoG
    normalize(logImage, logImage, 0, 255, NORM_MINMAX);

    // Trova i punti di zero-crossing
    Mat zeroCrossings = findZeroCrossings(logImage);

    // Visualizza l'immagine originale e i punti di zero-crossing
    imshow("Immagine Originale", image);
    imshow("Punti di Zero-Crossing", zeroCrossings);
    waitKey(0);

    return 0;
}


