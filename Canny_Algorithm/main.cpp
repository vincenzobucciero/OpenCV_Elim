#include <iostream>
#include <opencv2/opencv.hpp>


using namespace cv;
Mat imageInput, dst;
int ht = 100, lt = 30;

void canny(const Mat& imageInput, Mat& dst){
    //Primo passo: Sfocare (smoothing dell'immagine con la Gaussiana)
    Mat imageGaussBlur;
    GaussianBlur(imageInput, imageGaussBlur, Size(3,3), 0, 0);

    //Secondo passo: Calcolare la magnitudo e l'orientamento del gradiente
    //Gradiente
    Mat sobelX, sobelY;
    Sobel(imageGaussBlur, sobelX, imageInput.type(), 1, 0);
    Sobel(imageGaussBlur, sobelY, imageInput.type(), 0, 1);

    //Magnitudo
    Mat mag, orientation;
    //mag = abs(sobelX) + abs(sobelY);
    magnitude(sobelX, sobelY, mag);
    sobelX.convertTo(sobelX, CV_32FC1);
    sobelY.convertTo(sobelY, CV_32FC1);
    phase(sobelX, sobelY, orientation, true);                         //true mi ritorna l'angolo in gradi

    //Terzo passo: Applicare la non maxima suppression
    Mat maxSupp;
    mag.convertTo(maxSupp, CV_32FC1);
    normalize(mag, magNorm, 0, 255, NORM_MINMAX, CV_8UC1);
    
    //maxSupp.setTo(0);
    
    
    //Terzo passo: Applicare la non maxima suppression
    //Mat maxSupp;    // = Mat::zeros(magNorm.rows,magNorm.cols, magNorm.type());
    //magNorm.copyTo(maxSupp);

    //copyMakeBorder(maxSupp, maxSupp, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0));

    /**
     * Per ogni pixel (i,j), che è il pixel centrale che sto considerando nell’intorno, verifico se è 
     * maggiore dei suoi vicini in quella direzione (in tal caso, è un edge forte). 
     * 
     * orizzontale: i vicini sono quelli a sinistra e a destra del pixel centrale, dunque rispetto (i,j-1) e (i, j+1);
    */
    for(int i = 1; i < mag.rows-1; i++){
        for(int j = 1; j < mag.cols-1; j++){
            //si accede all'orientazione dell'intensità del gradiente associata a ciascun pixel
            float ang = orientation.at<float>(i-1,j-1);

            //orizzontale
            if(0 <= ang && ang <= 22.5 || 157.5 < ang && ang <= 180){
                if(mag.at<uchar>(i,j) < mag.at<uchar>(i, j-1) || mag.at<uchar>(i,j) < mag.at<uchar>(i, j+1))
                    maxSupp.at<uchar>(i,j) = 0;
            } 
            //verticale
            else if(67.5 < ang && ang <= 112.5){
                if(mag.at<uchar>(i,j) < mag.at<uchar>(i - 1, j) || mag.at<uchar>(i,j) < mag.at<uchar>(i + 1, j))
                    maxSupp.at<uchar>(i,j) = 0;
            }
            //+45
            else if(22.5 < ang && ang <= 67.5){
                if(mag.at<uchar>(i,j) < mag.at<uchar>(i - 1, j-1) || mag.at<uchar>(i,j) < mag.at<uchar>(i + 1, j + 1))
                    maxSupp.at<uchar>(i,j) = 0;
            }
            //-45
            else if(112.5 < ang && ang <= 157.5){
                if(mag.at<uchar>(i,j) < mag.at<uchar>(i + 1, j - 1) || mag.at<uchar>(i,j) < mag.at<uchar>(i - 1, j + 1))
                    maxSupp.at<uchar>(i,j) = 0;
            }
        }
    }


    //Quarto passo: Applicare il thresholding con isteresi
    for(int i=1; i<maxSupp.rows-1; i++){
		for(int j=1; j<maxSupp.cols-1; j++){
			if(maxSupp.at<uchar>(i,j) > ht)
                //strong edge
				maxSupp.at<uchar>(i,j) = 255;
            else if(maxSupp.at<uchar>(i,j) < lt) 
                //weak edge -> pixel non rilevante come bordo
                maxSupp.at<uchar>(i,j) = 0;
			else if(maxSupp.at<uchar>(i,j) <= ht && maxSupp.at<uchar>(i,j) >= lt){
                //Se il valore del pixel si trova tra lt e ht, viene eseguita una verifica più dettagliata. 
                //Viene esaminato il vicinato 3x3 del pixel corrente. Se almeno uno dei pixel nel vicinato 
                //è sopra il valore di soglia ht, il pixel corrente viene considerato un bordo forte 
                //(impostato a 255), altrimenti viene considerato non rilevante (impostato a 0).
				bool strong_n = false;
				for(int l = -1; l <= 1 && !strong_n ; l++){
					for(int k = -1; k <= 1 && !strong_n ; k++){
						if(maxSupp.at<uchar>(i+l,j+k) > ht) 
                            strong_n = true;
					}
				}
				maxSupp.at<uchar>(i, j) = strong_n ? 255 : 0;
			}
				
        }
	}
    maxSupp.copyTo(dst);
}

void CannyThreshold(int, void*){
	    canny(imageInput,dst);
	    imshow("Canny",dst);
}

int main(int argc, char **argv){

    imageInput = imread(argv[1], IMREAD_GRAYSCALE);
    if(imageInput.empty() )
        return -1;

    imshow("Orginal Image", imageInput);
    waitKey(0);
    
    namedWindow("Canny");
    createTrackbar("Trackbar th", "Canny", &ht, 255, CannyThreshold);
    createTrackbar("Trackbar lh", "Canny", &lt, 255, CannyThreshold);
    CannyThreshold(0,0);

    waitKey(0);

    Mat cannyCV;
    Canny(imageInput, cannyCV, 30, 100);
    imshow("cvCanny", cannyCV);
    waitKey(0);
    
    return 0;
}