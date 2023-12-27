#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgcodecs.hpp>
#include <opencv4/opencv2/highgui.hpp>

#include <iostream>

using namespace cv;
//! [includes]

int main()
{
    //! [imread]
    std::string image_path = samples::findFile("lena.png");
    Mat img = imread(image_path, IMREAD_COLOR);
    //! [imread]

    //! [empty]
    if(img.empty())
    {
        std::cout << "Could not read the image: " << image_path << std::endl;
        return 1;
    }
    //! [empty]

    //! [imshow]
    imshow("Display window", img);
    int k = waitKey(0); // Wait for a keystroke in the window
    //! [imshow]

    //! [imsave]
    if(k == 's')
    {
        imwrite("lena.png", img);
    }
    //! [imsave]

    return 0;
}