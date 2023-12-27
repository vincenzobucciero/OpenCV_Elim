#include <opencv4/opencv2/core/core.hpp>
#include <opencv4/opencv2/highgui/highgui.hpp>
using namespace cv;
int main()
{
    std::string image_path = "/home/vincenzo/Desktop/Elim/OpenCV_Elim/lena.png";
    Mat img = imread(image_path, IMREAD_COLOR);

    imshow("Display window", img);
    int k = waitKey(0); // Wait for a keystroke in the window
    return 0;
}