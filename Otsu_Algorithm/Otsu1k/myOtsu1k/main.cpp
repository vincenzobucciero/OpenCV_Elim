#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

/**
 * La sogliatura può essere vista come un problema di statistica in cui si vuole minimizzare 
 * l'errore medio dividendo i pixel in due classi.
 * 
 * Il metodo di Otsu è (alternativa alla regola di decisione di Bayes che è complessa e non adatta
 * ad applicazioni pratiche) ottimale nel senso che massimizza la varianza interclasse 
 * (separazione tra valori di intensità delle due classi)
 * 
 * Questo metodo opera esclusivamente sull'istogramma dell'immagine
*/

/**
 * Istogramma Normalizzato
 * 
 * Un istogramma normalizzato consiste nel calcolare il numero dei pixel che assumono 
 * un particolare livello di intensità e dividerlo per il numero totale di pixel (numRighe*numColonne).
 * 
 * Scorriamo con un doppio ciclo for l'immagine, e consideriamo ad ogni iterazione un pixel 
 * che avrà un valore di intensità compreso fra 0 e 255.
 * Utilizziamo questo valore come indice del vettore che rappresenta l'istogramma, così che se è già presente
 * un pixel in quella posizione, significa che ci sono più pixel con la stessa intensità, incrementiamo dunque 
 * quella cella.
 * 
 * Es. se nell’immagine ho 200 pixel la cui intensità è 50, in his[50] avrò valore 200.
 * 
 * Poi dividiamo ogni posizione dell'istogramma per il numeor di pixel.
*/
vector<float> normHistogram(const Mat& src) {
    vector<float> hist(256, 0.0f);  //vettore che rappresenta l'istogramma, è composto da 256 elementi float inizializzati a 0
    for(int y = 0; y < src.rows; y++) {
        for(int x = 0; x < src.cols; x++) {
            hist.at(src.at<uchar>(y,x))++;
        }
    }
    for(int i = 0; i < 256; i++) {
        hist.at(i) /= src.rows*src.cols;
    }
    return hist;
}

int otsu1k(const Mat& src) {
    Mat blur;
    GaussianBlur(src, blur, Szie(3,3), 0, 0);   //cancellazione rumore
    vector<float> histogram = normHistogram(blur);
    //probability
    //cumulativeAvg
    //globalAvg
    //interVariance
    //kstar
}


int main(int argc, char**argv) {
    if(argc < 2) {
        cout << "error" << endl;
        exit(0);
    }
    Mat src = imread(argv[1], IMREAD_COLOR);
    if(src.empty()) {
        cout << "can not open" << argv[1] << endl;
        exit(0);
    }
    imshow("original image", src);
    Mat dest;
    //otsu
    imshow("houghlines image", dest);
    waitKey(0);

    return 0;
}