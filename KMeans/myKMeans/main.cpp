/**
 * L’algoritmo k-means consiste nei seguenti passi fondamentali:
 * 1. inizializzare i centri di ogni cluster;
 * 2. assegnare ogni pixel al cluster con il centro più vicino.
 *    Per ogni pixel pj calcolare la distanza (euclidea) dai k centri ci, ed assegnare pj al cluster con il centro ci più vicino;
 * 3. aggiornare i centri. Dunque calcolare la media dei pixel in ogni cluster.
 * 
 * Ripetere i punti 2 e 3 finché il centro (media) di ogni cluster non viene più modificata (o comunque è minima).
 * 4. Assegnare ad ogni pixel del cluster i-simo il valore di c[i].
*/
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdlib.h>
#include <iostream>
#include <vector>

using std::cout;
using std::endl;
using std::vector;
using namespace cv;

int clusterNum = 6;
double th = 0.05;

/**
 * Questa funzione inizializza i centroidi dei cluster selezionando casualmente i pixel dell'immagine 
 * e li salva insieme alle informazioni sui punti associati a ciascun centro del cluster. 
 * Questi dati saranno utilizzati successivamente nell'algoritmo K-means per assegnare i punti ai cluster corrispondenti.
 * 
 * RNG random(getTickCount()); 
 * Questa riga inizializza il generatore di numeri casuali utilizzando il valore corrente di tick count come seme. 
 * 
 * Per ogni cluster inizializziamo le coordinate del centroide in modo casuale all'interno dei limiti dell'immagine
 * 
 * Calcoliamo il valore colore del pixel calcolato in modo casuale dall'immagine. 
 * Il valore colore ottenuto viene utilizzato per creare un oggetto Scalar (centerK), che rappresenta il centro del cluster
 * questo centro viene aggiunnti al vettore che conterrà i centroidi di tutti i cluster
 * 
 * Viene creato un vettore per memorizzare i futuri pixel associati a questo centro del cluster
 * Questo vettore viene aggiunti ad un altro vettore che conerrà i vettori di punti associati a ciascun cluster
 * 
*/
void initializeClusters(const Mat& img, vector<Scalar>& clusterCenters, vector<vector<Point>>& ptInCluster){
    RNG clusterGenerator(getTickCount());
    for(auto c = 0; c < clusterNum; c++){
        Point cluster;
        cluster.y = clusterGenerator.uniform(0,img.rows);
        cluster.x = clusterGenerator.uniform(0,img.cols);
        Scalar clusterValue = img.at<Vec3b>(cluster);
        clusterCenters.push_back(clusterValue);
        vector<Point> clusterGroup;
        ptInCluster.push_back(clusterGroup);
    }
}


/**
 * Calcoliamo la distanza tra due pixel in base ai loro valori di colore.
 * La differenza tra i canali dei colori viene calcolata per i canali blu, verde e rosso, 
 * poi utilizziamo la distanza euclidea per ottenere la distanza finale.
 * 
 * Usiamo la distanza per valutare la similarità tra i pixel.
*/
double computeColorDistance(Scalar p, Scalar clusterCenter){
    uchar blueDiff = abs(clusterCenter[0] - p[0]);
    uchar greenDiff = abs(clusterCenter[1] - p[1]);
    uchar redDiff = abs(clusterCenter[2] - p[2]);
    return blueDiff + greenDiff + redDiff;
}

/**
 * Iteriamo su tutti i pixel dell'immagine e, per ciascun pixel, troviamo il cluster più vicino 
 * calcolando la distanza colore tra tale pixel e i pixel dei cluster.
 * 
 * Successivamente il pixel viene associalo al cluster più vicino
*/
void populateCluster(const Mat& img,vector<Scalar>& clusterCenters,vector<vector<Point>>& ptInCluster ){
    for(auto y = 0; y < img.rows; y++){
        for(auto x = 0; x < img.cols; x++){
            double minDist = INFINITY;
            int clusterRep = 0;
            Scalar pixelVal = img.at<Vec3b>(y,x);
            for(auto clusterId = 0; clusterId < clusterNum; clusterId++){
                double dist = computeColorDistance(pixelVal, clusterCenters.at(clusterId));
                if(dist < minDist){
                    minDist = dist;
                    clusterRep = clusterId;
                }
            }
            ptInCluster.at(clusterRep).push_back(Point(x,y));
        }
    }
}

/**
 * Aggiorniamo i centri dei cluster in base alla media dei pixel associati a ciascun cluster, 
 * In particolare:
 *  - Per tutti i pixel di ogni cluster vengono inizializzati i canali colore a 0
 *  - Per ciascun pixel viene preso il loro valore corrispondente nell'immagine fatta la media
 *  - Viene creato un nuovo valore per il centro del cluster utilizzando i valori medi calcolati
 *  - la distanza tra il nuovo centro e il precedente viene calcolata utilizzando computeColorDistance
 *  - dividiamo il nuovo valore per il numero totale di cluster per ottenere il nuovo vaore dei centro medio
 *  - 
*/
double adjustClusterCenters(const Mat& img,vector<Scalar>& clusterCenters,vector<vector<Point>>& ptInCluster, double& oldVal, double newVal){
    for(auto clusterId = 0; clusterId < clusterNum; clusterId++){
        vector<Point> clusterIGroup = ptInCluster.at(clusterId);
        double newBlue = 0.0f;
        double newGreen = 0.0f;
        double newRed = 0.0f;
        for(auto pixel = 0; pixel < clusterIGroup.size(); pixel++){
            Scalar pixelVal = img.at<Vec3b>(clusterIGroup.at(pixel));
            newBlue += pixelVal[0];
            newGreen += pixelVal[1];
            newRed += pixelVal[2];
        }
        newBlue/=clusterIGroup.size();
        newGreen/=clusterIGroup.size();
        newRed/=clusterIGroup.size();
        Scalar newCenter = Scalar(cvRound(newBlue),cvRound(newGreen),cvRound(newRed));
        newVal += computeColorDistance(newCenter, clusterCenters.at(clusterId));
        clusterCenters.at(clusterId) = newCenter;
    }
    newVal/=clusterNum;
    double diffChange = abs(newVal-oldVal);
    oldVal = newVal;
    return diffChange;
}

void segment(Mat& out, const vector<Scalar>& clusterCenter, const vector<vector<Point>>& ptInClusters ){
    for(auto clusterId = 0; clusterId < clusterNum; clusterId++){
        auto clusterGroup = ptInClusters.at(clusterId);
        for(auto clusterPoint : clusterGroup){
            out.at<Vec3b>(clusterPoint)[0] = clusterCenter.at(clusterId)[0];
            out.at<Vec3b>(clusterPoint)[1] = clusterCenter.at(clusterId)[1];
            out.at<Vec3b>(clusterPoint)[2] = clusterCenter.at(clusterId)[2];
        }
    }
}

void kMean(const Mat& src, Mat& dest){
    dest = Mat::zeros(src.size(),CV_8UC3);
    vector<Scalar> clusterCenters;
    vector<vector<Point>> clusterGroup;
    initializeClusters(src,clusterCenters,clusterGroup);
    double newVal = 0;
    double oldVal = INFINITY;
    double dist = abs(oldVal - newVal);

    while(dist > th){
        newVal = 0;
        for(auto id = 0; id < clusterNum; id++){
            clusterGroup.at(id).clear();
        }
        populateCluster(src,clusterCenters,clusterGroup);
        dist = adjustClusterCenters(src,clusterCenters, clusterGroup,oldVal,newVal);
    }
    segment(dest,clusterCenters,clusterGroup);
}

int main(int argc, char const* argv[])
{
    if (argc < 2)
    {
        cout << "usage: " << argv[0] << " image_name" << endl;
        exit(0);
    }

    Mat src = imread(argv[1], IMREAD_COLOR);
    Mat dest;
    if (src.cols > 500 || src.rows > 500) {
        cv::resize(src, src, cv::Size(0, 0), 0.5, 0.5); // resize for speed
    }

    imshow("src", src);
    kMean(src,dest);
    imshow("dest",dest);
    waitKey();

    return 0;
}