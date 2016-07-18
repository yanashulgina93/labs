#include "myFilter.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <stack>

void myCanny(const cv::Mat & grayscale, cv::Mat & edges, int threshold1, int threshold2)
{
    edges=cv::Scalar(0);

    cv::Mat gradX, gradY;
    cv::Sobel(grayscale, gradX, CV_32FC1, 1, 0);
    cv::Sobel(grayscale, gradY, CV_32FC1, 0, 1);

    cv::Mat gradMagn(grayscale.rows, grayscale.cols, CV_32FC1, 0.0);
    cv::Mat gradAngle(grayscale.rows, grayscale.cols, CV_32FC1, 0.0);
    cv::cartToPolar(gradX, gradY, gradMagn, gradAngle, true);

    std::stack<cv::Point> edgePixels;
    for(int i = 1; i < gradMagn.rows-1; i++)
    {
        float* gradMagn_prev_row_ptr = gradMagn.ptr<float>(i-1);
        float* gradMagn_cur_row_ptr = gradMagn.ptr<float>(i);
        float* gradMagn_next_row_ptr = gradMagn.ptr<float>(i+1);
        float* gradAngle_cur_ptr = gradAngle.ptr<float>(i);
        uchar* edges_row_ptr = edges.ptr<uchar>(i);

        float neighbour1Magn;
        float neighbour2Magn;

        for(int j = 1; j < gradMagn.cols-1; j++)
        {
            if (gradMagn_cur_row_ptr[j] >= threshold2)
            {
                if ((gradAngle_cur_ptr[j] < 22) ||
                        ((gradAngle_cur_ptr[j] >= 157)&&(gradAngle_cur_ptr[j] < 202)) ||
                        (gradAngle_cur_ptr[j] >= 337))//0
                {
                    neighbour1Magn = gradMagn_prev_row_ptr[j];
                    neighbour2Magn = gradMagn_next_row_ptr[j];
                }
                else if (((gradAngle_cur_ptr[j] >= 22)&& (gradAngle_cur_ptr[j] < 67)) ||
                         ((gradAngle_cur_ptr[j] >= 202) && (gradAngle_cur_ptr[j] < 247)))//45
                {
                    neighbour1Magn = gradMagn_prev_row_ptr[j-1];
                    neighbour2Magn = gradMagn_next_row_ptr[j+1];
                }
                else if (((gradAngle_cur_ptr[j] >= 67) && (gradAngle_cur_ptr[j] < 112)) ||
                         ((gradAngle_cur_ptr[j] >= 247) && (gradAngle_cur_ptr[j] < 292)))//90
                {
                    neighbour1Magn = gradMagn_cur_row_ptr[j-1];
                    neighbour2Magn = gradMagn_cur_row_ptr[j+1];
                }
                else //135
                {
                    neighbour1Magn = gradMagn_next_row_ptr[j-1];
                    neighbour2Magn = gradMagn_prev_row_ptr[j+1];
                }

                if ((gradMagn_cur_row_ptr[j] > neighbour1Magn) && (gradMagn_cur_row_ptr[j] > neighbour2Magn))
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
    cv::Mat grayscale;
    cv::cvtColor(src, grayscale, CV_BGR2GRAY);
    cv::Mat edges(src.rows, src.cols, CV_8UC1);
    myCanny(grayscale, edges, threshold1, threshold2);

    cv::Mat dist, inv;
    cv::bitwise_not(edges, inv);
    cv::distanceTransform(inv, dist, CV_DIST_C, 3);

    cv::Mat srcIntegral(src.rows, src.cols, CV_32SC3);
    cv::integral(src, srcIntegral, CV_32S);
    int windowSize;
    cv::Rect integralRect;

    for(int r = 1; r < srcIntegral.rows; r++)
    {
        for(int c = 1; c < srcIntegral.cols; c++)
        {
            windowSize = ksize*dist.at<float>(r-1,c-1);
            integralRect.x = c-windowSize-1;
            integralRect.y = r-windowSize-1;
            integralRect.width = windowSize*2+1;
            integralRect.height = windowSize*2+1;
            cv::Rect matRect(0,0,srcIntegral.cols-1, srcIntegral.rows-1);
            integralRect = integralRect & matRect;

            cv::Vec3i sum =
                    srcIntegral.at<cv::Vec3i>(integralRect.br().y, integralRect.br().x)
                    + srcIntegral.at<cv::Vec3i>(integralRect.y, integralRect.x)
                    - srcIntegral.at<cv::Vec3i>(integralRect.br().y, integralRect.x)
                    - srcIntegral.at<cv::Vec3i>(integralRect.y, integralRect.br().x);

            if (integralRect.area() > 0)
                dst.at<cv::Vec3b>(r-1, c-1) = sum/integralRect.area();

            else
                dst.at<cv::Vec3b>(r-1, c-1) = src.at<cv::Vec3b>(r-1, c-1);

        }
    }
}
