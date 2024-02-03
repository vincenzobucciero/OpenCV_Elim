#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <stdlib.h>

using namespace std;
using namespace cv;

int tsize;
double smthreshold; 
int media;

class tNode {
    private:
        Rect region;
        tNode *UL, *UR, *LR, *LL;
        vector<tNode*> merged;
        vector<bool> mergedB = vector<bool>(4, false);
        double mean, stddev;
    public:
        tNode(Rect R) {region = R; UL = nullptr; UR = nullptr; LR = nullptr; LL = nullptr;}

        void addRegion(tNode* R) {merged.push_back(R);}
        Rect &getRegion() {return region;}

        void setUL(tNode *UL) {this->UL = UL;}
        tNode *getUL() {return this->UL;}
        void setUR(tNode *UR) {this->UR = UR;}
        tNode *getUR() {return this->UR;}
        void setLR(tNode *LR) {this->LR = LR;}
        tNode *getLR() {return this->LR;}
        void setLL(tNode *LL) {this->LL = LL;}
        tNode *getLL() {return this->LL;}

        vector<tNode*> &getMerged() {return merged;}

        void setMergedB(int i) {mergedB[i] = true;}
        bool getMergedB(int i) {return mergedB[i];}

        void setMean(double mean) {this->mean = mean;}
        double getMean() {return this->mean;}
        void setStdDev(double stddev) {this->stddev = stddev;}
        double getStdDev() {return this->stddev;}
};

tNode* split(Mat& img, Rect R) {
    tNode *root = new tNode(R);
    Scalar mean, stddev;

    meanStdDev(img(R), mean, stddev);

    root->setMean(mean[0]);
    root->setStdDev(stddev[0]);

    if(R.width > tsize && root->getStdDev() > smthreshold) {
        Rect ul(R.x, R.y, R.height/2, R.width/2);
        root->setUL(split(img, ul));
        Rect ur(R.x, R.y + R.width/2, R.height/2, R.width/2);
        root->setUR(split(img, ur));
        Rect lr(R.x + R.height/2, R.y + R.width/2, R.height/2, R.width/2);
        root->setLR(split(img, lr));
        Rect ll(R.x + R.height/2, R.y, R.height/2, R.width/2);
        root->setLL(split(img, ll));
    }
    rectangle(img, R, Scalar(0));

    return root;
}

void merge(tNode *root) {
    if(root->getRegion().width > tsize && root->getStdDev() > smthreshold) {
        if(root->getUL()->getStdDev() <= smthreshold && root->getUR()->getStdDev() <= smthreshold) {
                root->addRegion(root->getUL());
                root->addRegion(root->getUR());
                root->setMergedB(0);
                root->setMergedB(1);
                if(root->getLR()->getStdDev() <= smthreshold && root->getLL()->getStdDev() <= smthreshold) {
                    root->addRegion(root->getLR());
                    root->addRegion(root->getLL());
                    root->setMergedB(2);
                    root->setMergedB(3);
                } else {
                    merge(root->getLR());
                    merge(root->getLL());
                }
        }
        else if(root->getUR()->getStdDev() <= smthreshold && root->getLR()->getStdDev() <= smthreshold) {
                root->addRegion(root->getUR());
                root->addRegion(root->getLR());
                root->setMergedB(1);
                root->setMergedB(2);
                if(root->getUL()->getStdDev() <= smthreshold && root->getLL()->getStdDev() <= smthreshold) {
                    root->addRegion(root->getUL());
                    root->addRegion(root->getLL());
                    root->setMergedB(0);
                    root->setMergedB(3);
                } else {
                    merge(root->getUL());
                    merge(root->getLL());
                }
        }
        else if(root->getLR()->getStdDev() <= smthreshold && root->getLL()->getStdDev() <= smthreshold) {
                root->addRegion(root->getLR());
                root->addRegion(root->getLL());
                root->setMergedB(2);
                root->setMergedB(3);
                if(root->getUL()->getStdDev() <= smthreshold && root->getUR()->getStdDev() <= smthreshold) {
                    root->addRegion(root->getUL());
                    root->addRegion(root->getUR());
                    root->setMergedB(0);
                    root->setMergedB(1);
                } else {
                    merge(root->getUL());
                    merge(root->getUR());
                }
        }
        else if(root->getUL()->getStdDev() <= smthreshold && root->getLL()->getStdDev() <= smthreshold) {
                root->addRegion(root->getUL());
                root->addRegion(root->getLL());
                root->setMergedB(0);
                root->setMergedB(3);
                if(root->getUR()->getStdDev() <= smthreshold && root->getLR()->getStdDev() <= smthreshold) {
                    root->addRegion(root->getUR());
                    root->addRegion(root->getLR());
                    root->setMergedB(1);
                    root->setMergedB(2);
                } else {
                    merge(root->getUR());
                    merge(root->getLR());
                }
        }
        else {
            merge(root->getUL());
            merge(root->getUR());
            merge(root->getLR());
            merge(root->getLL());
        }
    }
    else {
        root->addRegion(root);
        root->setMergedB(0);
        root->setMergedB(1);
        root->setMergedB(2);
        root->setMergedB(3);
    }
}

void segment(tNode *root, Mat& img) {
    vector<tNode*> tmp = root->getMerged();
    if(!tmp.size()) {
        segment(root->getUL(), img);
        segment(root->getUR(), img);
        segment(root->getLR(), img);
        segment(root->getLL(), img);
    } 
    else {
        double val = 0;
        for(auto x : tmp) {
            val += (int)x->getMean();
        }
        val /= tmp.size();
        for(auto x : tmp) {
            img(x->getRegion()) = (int)val;
        }

        if(tmp.size() > 1) {
            if(!root->getMergedB(0))
                segment(root->getUL(), img);
            if(!root->getMergedB(1))
                segment(root->getUR(), img);
            if(!root->getMergedB(2))
                segment(root->getLR(), img);
            if(!root->getMergedB(3))
                segment(root->getLL(), img);
        }
    }
}

int main(int argc, char**argv) {
    if(argc < 2) {
        cout << "error using " << argv[0] << endl;
        exit(0);
    }
    Mat src = imread(argv[1], IMREAD_GRAYSCALE);
    if(src.empty()) {
        cout << "error can't read/open image" << endl;
        exit(0);
    }
    
    smthreshold = stod(argv[2]);
    tsize = stoi(argv[3]);

    imshow("src", src);

    GaussianBlur(src, src, Size(5, 5), 0, 0);

    int exp = log(min(src.cols, src.rows))/log(2);
    int s = pow(2.0, double(exp));
    Rect square = Rect(0,0,s,s);
    src = src(square).clone();
    
    Mat srcSeg=src.clone();

    tNode *root = split(src, Rect(0,0,src.rows, src.cols));
    merge(root);
    segment(root, srcSeg);

    imshow("src", src);
    imshow("segmentation src", srcSeg);

    waitKey(0);

    return 0;
}