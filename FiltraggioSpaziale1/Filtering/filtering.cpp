/**
 * Tecnica sviluppata per il dominio delle frequenze che consente di far passare o bloccare alcuni elementi.
 * -> Le variazioni d'intensità dei pixel possono essere:
 *      1. repentine (alta intensità)
 *      2. graduali (bassa intensità)
 * -> Gli effetti del processo di filtraggio sono:
 *      1. attenuazione
 *      2. miglioramento di dettagli
 * Si prendono in considerazione i valori di intensità di un intorno di un pixel (neighborhood)
 * 
 * Per ogni pixel dell’immagine originale, è calcolata l’intensità del pixel corrispondente nell’immagine filtrata.
 * La regola di trasformazione spesso è descritta da una matrice, chiamata filtro (maschera o kernel), 
 * della stessa dimensione dell'intorno.
 * 
 * Se la regola di trasformazione è una funzione lineare delle intensità nell’intorno, 
 * la tecnica è chiamata filtraggio lineare spaziale (altrimenti, non lineare). 
 * 
 * Il pixel nell’immagine filtrata, g(x,y), è ottenuto come combinazione lineare dei pixel nell’immagine originale, 
 * f(), in un intorno di (x, y). La matrice peso, w(), è il filtro o maschera o kernel (spesso è usata una matrice
 * dispari per avere stesso numero di righe (2a+1) e colonne (2b+1) prima e dopo il pixel centrato).
 * 
 * CORRELAZIONE -> Progressivo scorrimento di una maschera sull'immagine e nel calcolo della somma 
 *                 dei prodotti in ogni posizione;
 * CONVOLUZIONE -> Come la correlazione ma il filtro viene ruotato di 180°
 * 
 * Specifica del filtro
 * -> Filtro lineare - Filtro non lineare
 * -> Filtro di smoothing => Utilizzato per sfocare le immagini (blurring) e per la riduzione del rumore (denoising)
 *                              Rimuove i dettagli più piccoli della dim del filtro evidenziando oggetti più grandi;
 *                              Per la riduzione del rumore è possibile utilizzare filtri lineari o non lineari,
 *                                  a seconda del tipo di rumore.
 * -> Filtri di media => Il valore di ogni pixel viene sostituito con la media dei livelli di intensità nell'intorno 
 *                          definito dalla maschera. L'effetto è un'immagine in cui le brusche transizioni di intesità 
 *                          sono ridotte. In quetso modo vengono sfocati anche gli edge (lati)
*/

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