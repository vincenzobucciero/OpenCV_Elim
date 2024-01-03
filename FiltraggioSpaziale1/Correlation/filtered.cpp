#include <iostream>
#include <opencv4/opencv2/opencv.hpp>

using namespace cv;

/**
 * value=0 conterrà il valore finale che verrà restituito. 
 * Sebbene dovremo restituire un valore uchar, le operazioni sono di tipo float (abbiamo numeri con la virgola)
 * 
 * il doppio ciclo for si occuperà di effetturare la somma dei prodotti (elemento per elemento)
 * della maschera con un intorno (pad) dim*dim dell’immagine con il padding. 
 * 
 * Notiamo 2 cose
 *  1. gli elementi di mask sono di tipo float;
 *  2. gli elementi di "pad" hanno un OFFSET, con il quale stabiliamo l’intorno.
 * 
 * Ad es, se stessimo calcolando il pixel(0,2), allora: offx = 0, offy = 2.
 * Ovviamente l’intorno di cui teniamo conto è quello del padding.
 * 
 * N.b stiamo facendo la doppia sommatoria di w(s, t)*f(x+s, y+t)
 * dove w è il filtro/maschera e f e l'immagine
*/

uchar filtered(Mat pad, Mat mask, int off_X, int off_Y) {
    float value = 0.0;
    for(int x = 0; x < mask.rows; x++) {
        for(int y = 0; y < mask.cols; y++) {
            value += mask.at<float>(x, y)*pad.at<uchar>(x+off_X, y+off_Y);
        }
    }
    return uchar(value);
}