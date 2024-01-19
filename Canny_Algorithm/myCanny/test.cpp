/**
 * STEP ALGORITMO CANNY
 * 1. CONVOLVERE L'IMMAGINE DI INPUT CON UN FILTRO GAUSSIANO (GaussianBlur)
 * 2. CALCOLARE LA MAGNITUDO E L'ORIENTAZIONE DEL GRADIENTE
 * 3. APPLICARE LA NON MAXIMA SUPPRESSION
 * 4. APPLICARE IL TRESHOLDING CON ISTERESI
*/

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

//valore di soglia inferiore
int lth = 60;
//valore di soglia superiore
int hth = 80;

/**
 * Calcola la magnitudine e l'angolo di fase dei gradienti
 *
 * @param imageInput Immagine di input in scala di grigi
 * @param mag Matrice di output per la magnitudine dei gradienti
 * @param ang Matrice di output per l'angolo di fase dei gradienti
 */
void magnitudeAndAngle(const Mat& imageInput, Mat& mag, Mat& ang) {
    Mat dx, dy;
    // Calcola i gradienti rispetto alle direzioni x e y utilizzando il filtro di Sobel
    Sobel(imageInput, dx, CV_32F, 1, 0);    //derivata rispetto a x
    Sobel(imageInput, dy, CV_32F, 0, 1);    //derivata rispetto a y
    magnitude(dx, dy, mag);     //mag = abs(dx) + abs(dy);
    phase(dx, dy, ang, true);
}

/**
 * Applica la suppressione dei non massimi a una matrice di magnitudine.
 *
 * @param mag Matrice di magnitudine dei gradienti
 * @param ang Matrice di angolo di fase dei gradienti
 * @param nmsOut Matrice di output dopo la suppressione dei non massimi
 */
void nonMaximaSuppression(const Mat& mag,const Mat& ang, Mat& nmsOut){
    //Aggiunge un bordo alla matrice di magnitudine
    copyMakeBorder(mag,nmsOut,1,1,1,1,BORDER_CONSTANT,Scalar(0));
    for(int y = 1; y < mag.rows; y++){
        for(int x = 1; x < mag.cols; x++){

            //Ottiene il valore dell'angolo e lo normalizza nell'intervallo [0, 180)
            float angVal = ang.at<float>(y-1,x-1) > 180 ? ang.at<float>(y,x)-180 : ang.at<float>(y,x);

            //Controlla l'orientazione dell'angolo e esegue la suppressione dei non massimi
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

/**
 * Applica la soglia alle risposte dei non massimi per ottenere l'output finale.
 *
 * @param nmsOut Matrice dei risultati dopo la suppressione dei non massimi
 * @param imageOutput Matrice di output dopo l'applicazione della soglia
 */
void HThreshold(Mat& nmsOut, Mat& imageOutput){
    // Inizializza la matrice di output con zeri
    imageOutput = Mat::zeros(nmsOut.rows-2,nmsOut.cols-2,CV_8U);

    // Itera sulla matrice dei non massimi
    for(int y = 1; y < nmsOut.rows-1; y++){
        for(int x = 1; x < nmsOut.cols;x++){

            // Applica la soglia alta (hth) e bassa (lth)
            if(nmsOut.at<float>(y,x) > hth)
                imageOutput.at<uchar>(y-1,x-1) = 255;
            else if(nmsOut.at<float>(y,x)< lth) 
                imageOutput.at<uchar>(y-1,x-1) = 0;
            else{
                // Controlla se almeno uno dei vicini ha una risposta forte (soglia alta)
                bool strongN = false;
                for(int j = -1; j >= 1; j++){
                    for(int i = -1; i >= 1; i++){
                        if(nmsOut.at<float>(y + j,x + i) > hth) 
                            strongN = true;
                            break;  // Esci dal ciclo interno se viene trovata una risposta forte
                    }
                }
                // Assegna il valore appropriato in base alla presenza di risposta forte nei vicini
                if(strongN) imageOutput.at<uchar>(y-1,x-1) = 255;
                else imageOutput.at<uchar>(y-1,x-1) = 0;
            }
        }
    }
}

/**
 * STEP ALGORITMO CANNY
 * 1. CONVOLVERE L'IMMAGINE DI INPUT CON UN FILTRO GAUSSIANO (GaussianBlur)
 * 2. CALCOLARE LA MAGNITUDO E L'ORIENTAZIONE DEL GRADIENTE
 * 3. APPLICARE LA NON MAXIMA SUPPRESSION
 * 4. APPLICARE IL TRESHOLDING CON ISTERESI
*/
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