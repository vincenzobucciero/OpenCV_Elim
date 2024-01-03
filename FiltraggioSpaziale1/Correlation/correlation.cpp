/**
 * Algoritmo di correlazione, senza uso del filter2d()
*/

#include <iostream>
#include <opencv4/opencv2/opencv.hpp>
#include "filtered.cpp"

using namespace cv;

/**
 * Per poter appliccare il filtro, dobbiamo aggiungere il padding. 
 * 
 * L’ampiezza del bordo è calcolata come il numero di colonne (o righe) diviso 2, che sarà approssimato per difetto. 
 * Infatti se ho una mask 3x3 dovrò aggiungere 1 riga per ogni lato (3/2=1) dell’immagine, 
 * per una mask 5x5 dovrò aggiungere 2 righe per ogni lato (5/2=2), ecc.
 * 
 * Il doppio ciclo for si occuperà solo di scorrere sugli elementi della matrice che vogliamo restituire.
 * Ad ogni posizione (i,j) richiamerà la funzione filteredValue() il quale restituirà il valore calcolato, 
 * e a cui passeremo la matrice col padding, la maschera, e i due indici 
*/

Mat correlation(const Mat& imageInput, const Mat& mask) {
    Mat imageOutput(imageInput.rows, imageInput.cols, imageInput.type());
    Mat pad;

    int border = imageInput.rows/2;

    copyMakeBorder(imageInput, pad, border, border, border, border, BORDER_DEFAULT);

    for(int x = 0; x < imageOutput.rows; x++) {
        for(int y = 0; y < imageOutput.cols; y++) {
            imageOutput.at<uchar>(x, y) = filtered(pad, mask, x, y);
        }
    }

    return imageOutput;
}