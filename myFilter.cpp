#include "myFilter.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stack>

void myCanny(const cv::Mat & grayscale, cv::Mat & edges, int threshold1, int threshold2)
{
    edges = cv::Mat::zeros(grayscale.rows, grayscale.cols, CV_8UC1);

    cv::Mat gradX, gradY;
    cv::Sobel(grayscale, gradX, CV_32FC1, 1, 0);
    cv::Sobel(grayscale, gradY, CV_32FC1, 0, 1);

    cv::Mat gradMagn, gradAngle;
    cv::cartToPolar(gradX, gradY, gradMagn, gradAngle, true);

    std::stack<cv::Point> edgePixels;
    for(int i = 1; i < gradMagn.rows-1; i++)
    {
        float* gradMagn_ptr = gradMagn.ptr<float>(i);
        uchar* edges_row_ptr = edges.ptr<uchar>(i);
        float* gradAngle_ptr = gradAngle.ptr<float>(i);

        int neighbour1;
        int neighbour2;

        for(int j = 1; j < gradMagn.cols-1; j++)
        {
            float curAngle = *(gradAngle_ptr+j);
            float curMagn = *(gradMagn_ptr+j);
            if (curMagn >= threshold2)
            {
                if (curAngle >= 157)
                    curAngle -= 180;
                if (curAngle < 22)//0
                {
                    neighbour1 = j - gradMagn.step;
                    neighbour2 = j + gradMagn.step;
                }
                else if ((curAngle >= 22)&& (curAngle < 67))//45
                {
                    neighbour1 = j - 1 - gradMagn.step;
                    neighbour2 = j + 1 + gradMagn.step;
                }
                else if ((curAngle >= 67) && (curAngle < 112))//90
                {
                    neighbour1 = j - 1;
                    neighbour2 = j + 1;
                }
                else //135
                {
                    neighbour1 = j - 1 + gradMagn.step;
                    neighbour2 = j + 1 - gradMagn.step;
                }

                if ((curMagn > gradMagn_ptr[neighbour1]) && (curMagn > gradMagn_ptr[neighbour2]))
                {
                    edges_row_ptr[j] = 255;
                    edgePixels.push(cv::Point(j,i));
                }

            }
        }
    }

    int radius = 1;
    while(!edgePixels.empty())
    {
        cv::Point p = edgePixels.top();
        edgePixels.pop();
        for(int i = std::max(p.y-radius, 0); i <= std::min(p.y+radius, edges.rows-1); i++)
        {
            float* gradMagn_row_ptr = gradMagn.ptr<float>(i);
            uchar* edges_row_ptr = edges.ptr<uchar>(i);
            for(int j = std::max(p.x-radius, 0); j <= std::min(p.x + radius, edges.cols-1); j++)
            {
                if((gradMagn_row_ptr[j] > threshold1)&&(gradMagn_row_ptr[j] < threshold2)&&(edges_row_ptr[j] != 255))
                {
                    edges_row_ptr[j] = 255;
                    edgePixels.push(cv::Point(j,i));
                }
            }
        }
    }
}

void myFilter(const cv::Mat & src, cv::Mat & dst, int threshold1, int threshold2, int ksize)
{
    dst.create(src.rows, src.cols, CV_8UC3);
    cv::Mat grayscale;
    cv::cvtColor(src, grayscale, CV_BGR2GRAY);
    cv::Mat edges;
    myCanny(grayscale, edges, threshold1, threshold2);

    cv::Mat dist, inv;
    cv::bitwise_not(edges, inv);
    cv::distanceTransform(inv, dist, CV_DIST_C, 3);

    cv::Mat srcIntegral;
    cv::integral(src, srcIntegral, CV_32S);
    int windowSize;
    cv::Rect integralRect;

    for(int r = 0; r < srcIntegral.rows-1; r++)
    {
        for(int c = 0; c < srcIntegral.cols-1; c++)
        {
            windowSize = ksize*dist.at<float>(r,c);
            integralRect.x = c-windowSize;
            integralRect.y = r-windowSize;
            integralRect.width = windowSize*2+1;
            integralRect.height = windowSize*2+1;
            cv::Rect matRect(0,0,srcIntegral.cols-1, srcIntegral.rows-1);
            integralRect = integralRect & matRect;

            cv::Vec3i sum =
                    srcIntegral.at<cv::Vec3i>(integralRect.br())
                    + srcIntegral.at<cv::Vec3i>(integralRect.tl())
                    - srcIntegral.at<cv::Vec3i>(integralRect.br().y, integralRect.x)
                    - srcIntegral.at<cv::Vec3i>(integralRect.y, integralRect.br().x);

            if (integralRect.area() > 0)
                dst.at<cv::Vec3b>(r, c) = sum/integralRect.area();

            else
                dst.at<cv::Vec3b>(r, c) = src.at<cv::Vec3b>(r, c);

        }
    }
}
