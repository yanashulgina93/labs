#include "myFilter.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>

void testMyFilter(const cv::Mat & src)
{
    int threshold1 = 50;
    int threshold2 = 120;
    int ksize = 1;
    cv::Mat filtered(src.rows, src.cols, CV_8UC3);

    myFilter(src, filtered, threshold1, threshold2, ksize);

    cv::imshow("src", src);
    cv::imshow("filtered", filtered);
    cv::waitKey(0);
}

int main(int argc, char** argv)
{
    cv::Mat src = cv::imread(argv[1]);
    if (src.empty())
    {
        std::cout<<"Can not read image "<<argv[1]<<std::endl;
        return -1;
    }

    testMyFilter(src);

    return 0;
}
