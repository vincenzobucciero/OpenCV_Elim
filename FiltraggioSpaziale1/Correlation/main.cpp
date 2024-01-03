/**
 * Algoritmo di correlazione, senza uso del filter2d()
*/

#include <opencv4/opencv2/opencv.hpp>
#include <iostream>

#include "correlation.cpp"

using namespace std;
using namespace cv;

/**
* |1 2 1| 
* |2 4 2| -> la somma di questo filtro è 16, quindi dobbiamo dividere ogni valore per 16
* |1 2 1|
*   
*  1/16 -> 1/16. 
*  2/16 -> 1/8. 
*  4/16 -> 1/4. 
*/

int main(int argc, char**argv) {
    if(argc < 2) {
        cout << "Error, using " << argv[0] << " image name" << endl;
        exit(0);
    }

    Mat image = imread(string(argv[1]), IMREAD_GRAYSCALE);

    if(image.empty()) {
        cout << "Errore apertura immagine" << endl;
        exit(0);
    }

    //filtro media
    Mat filtroMedia = (Mat_<float>(3,3) << 1/16.0, 1/8.0, 1/16.0,
									       1/8.0,  1/4.0, 1/8.0,
										   1/16.0, 1/8.0, 1/16.0);
    Mat filter2Dimg;
    filter2D(image, filter2Dimg, image.type(), filtroMedia);

    Mat corrImg = correlation(image, filtroMedia);

    imshow("original", image);
    waitKey(0);
    imshow("filter2D", filter2Dimg);
    waitKey(0);
    imshow("correlazione", corrImg);
    waitKey(0);

    return 0;
}
