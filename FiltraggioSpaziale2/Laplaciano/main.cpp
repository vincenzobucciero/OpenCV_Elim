/**
 * Implementare il Laplaciano con kernel isotropico a 45° e 90° utilizzando la
 * funzione di correlazione/convoluzione (o filter2D())
 * 
 * Per normalizzare i livelli di grigio è possibile usare la funzione normalize()
 * 
 * normalize(src, dst, 0, 255, NORM_MINMAX, CV_8U)
*/

#include <opencv2/opencv.hpp>
#include <iostream>

#include "correlation.cpp"

using namespace cv;
using namespace std;

int main(int argc, char**argv) {
    if(argc < 2) {
        cout << "Insert image name " << argv[0] << endl;
        exit(0);
    }
    Mat imageInput = imread(argv[1], IMREAD_GRAYSCALE);
    if(imageInput.empty()) {
        cout << "Error during reading imageInput" << endl;
        exit(0);
    }

    Mat laplaciano90 = (Mat_<float>(3,3) << 0.0, 1.0, 0.0,
                                            1.0, -4.0, 1.0,
                                            0.0, 1.0, 0.0);
    
    Mat laplaciano45 = (Mat_<float>(3,3) << 1.0, 1.0, 1.0,
                                            1.0, -8.0, 1.0,
                                            1.0, 1.0, 1.0);
    //USO FUNCTION filter2D()
    Mat imageFilter90, imageFilter45;
    filter2D(imageInput, imageFilter90, imageInput.type(), laplaciano90);
    filter2D(imageInput, imageFilter45, imageInput.type(), laplaciano45);

    //USO LAPLACIANO - four argument -> 1 laplacian 90, 3 laplacian 45
    Mat imageLaplacian90, imageLaplacian45;
    Laplacian(imageInput, imageLaplacian90, imageInput.type(), 1);
    Laplacian(imageInput, imageLaplacian45, imageInput.type(), 3);

    /*
    imshow("Original Image", imageInput);
    imshow("Image Filtered with 90 degrees kernel", imageFilter90);
    imshow("Image Filtered with 45 degrees kernel", imageFilter45);
    imshow("Image Laplacian with 90 degrees kernel", imageLaplacian90);
    imshow("Image Laplacian with 45 degrees kernel", imageLaplacian45);
    waitKey(0);
    */

    Mat imageCorrelation90 = correlation(imageInput, laplaciano90);
    imshow("Image Correlated with 90 degrees kernel", imageCorrelation90);
    waitKey(0);

    Mat imageCorrelation45 = correlation(imageInput, laplaciano45);
    imshow("Image Correlated with 45 degrees kernel", imageCorrelation45);
    waitKey(0);

    Mat imageCorrelationNormalized90;
    normalize(imageCorrelation90, imageCorrelationNormalized90, 0, 255, NORM_MINMAX, imageInput.type());
    imshow("Image Correlated Normalized with 90 degrees kernel", imageCorrelationNormalized90);
    waitKey(0);

    Mat imageCorrelationNormalized45;
    normalize(imageCorrelation45, imageCorrelationNormalized45, 0, 255, NORM_MINMAX, imageInput.type());
    imshow("Image Correlated Normalized with 45 degrees kernel", imageCorrelationNormalized45);
    waitKey(0);


    return 0;
}