#include "myFilter.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "math.h"
#include <stack>
#include <iostream>
#include "highgui.h"

void myCanny(const cv::Mat & grayscale, cv::Mat edges, int threshold1, int threshold2)
{
    cv::Mat gradX, gradY;
    cv::Mat gradMagn(grayscale.rows, grayscale.cols, CV_32FC1, 0.0);
    cv::Mat gradAngle(grayscale.rows, grayscale.cols, CV_32FC1, 0.0);

    cv::Sobel(grayscale, gradX, CV_32FC1, 1, 0);
    cv::Sobel(grayscale, gradY, CV_32FC1, 0, 1);

    float PI = atan(1)*4;

    for(int i = 0; i < gradX.rows; i++)
    {
        for(int j = 0; j < gradX.cols; j++)
        {
            gradMagn.at<float>(i,j) = sqrt((gradX.at<float>(i,j)*gradX.at<float>(i,j)) + (gradY.at<float>(i,j)*gradY.at<float>(i,j)));
            if (gradX.at<float>(i,j) != 0.0)
            {
                gradAngle.at<float>(i,j) = atan2(gradY.at<float>(i,j), gradX.at<float>(i,j)) * 180 / PI;
                if (gradAngle.at<float>(i,j) >= 0)
                {
                    if ((gradAngle.at<float>(i,j) < 22) || (gradAngle.at<float>(i,j) >= 157))
                        gradAngle.at<float>(i,j) = 0.0;
                    if ((gradAngle.at<float>(i,j) >= 22)&& (gradAngle.at<float>(i,j) < 67))
                        gradAngle.at<float>(i,j) = 45.0;
                    if((gradAngle.at<float>(i,j) >= 67) && (gradAngle.at<float>(i,j) < 112))
                        gradAngle.at<float>(i,j) = 90.0;
                    if ((gradAngle.at<float>(i,j) >= 112) && (gradAngle.at<float>(i,j) < 157))
                        gradAngle.at<float>(i,j) = 135;
                }
                else
                {
                    if ((gradAngle.at<float>(i,j) > -22) || (gradAngle.at<float>(i,j) <= -157))
                        gradAngle.at<float>(i,j) = 0.0;
                    if ((gradAngle.at<float>(i,j) <= -22)&& (gradAngle.at<float>(i,j) > -67))
                        gradAngle.at<float>(i,j) = 135.0;
                    if((gradAngle.at<float>(i,j) <= -67) && (gradAngle.at<float>(i,j) > -112))
                        gradAngle.at<float>(i,j) = 90.0;
                    if ((gradAngle.at<float>(i,j) <= -112) && (gradAngle.at<float>(i,j) > -157))
                        gradAngle.at<float>(i,j) = 45;
                }

            }
            else
                gradAngle.at<float>(i,j) = 90.0;
        }
    }

    for(int i = 0; i < gradMagn.rows; i++)
    {
        for(int j = 0; j < gradMagn.cols; j++)
        {
            if (gradMagn.at<float>(i,j) <= threshold1)
                edges.at<uchar>(i,j) = 0;
            else
                edges.at<uchar>(i,j) = grayscale.at<uchar>(i,j);
        }
    }

    std::stack<cv::Point> edgePixels;
    for(int i = 1; i < gradMagn.rows-1; i++)
    {
        for(int j = 1; j < gradMagn.cols-1; j++)
        {
            if (gradMagn.at<float> (i,j) >= threshold2)
            {
                if (gradAngle.at<float>(i,j) == 0)
                {
                    if((gradMagn.at<float>(i,j) > gradMagn.at<float>(i,j-1))&&(gradMagn.at<float>(i,j) > gradMagn.at<float>(i,j+1)))
                    {
                        edges.at<uchar>(i,j) = 255;
                        edgePixels.push(cv::Point(j,i));
                    }
                }
                if (gradAngle.at<float>(i,j) == 45)
                {
                    if ((gradMagn.at<float>(i,j) > gradMagn.at<float>(i+1,j-1)) && (gradMagn.at<float>(i,j) > gradMagn.at<float>(i-1,j+1)))
                    {
                        edges.at<uchar>(i,j) = 255;
                        edgePixels.push(cv::Point(j,i));
                    }
                }
                if (gradAngle.at<float>(i,j) == 90)
                {
                    if((gradMagn.at<float>(i,j) > gradMagn.at<float>(i-1,j))&& (gradMagn.at<float>(i,j) > gradMagn.at<float>(i+1,j)))
                    {
                        edges.at<uchar>(i,j) = 255;
                        edgePixels.push(cv::Point(j,i));
                    }
                }
                if (gradAngle.at<float>(i,j) == 135)
                {
                    if ((gradMagn.at<float>(i,j) > gradMagn.at<float>(i-1,j-1)) && (gradMagn.at<float>(i,j) > gradMagn.at<float>(i+1,j+1)))
                    {
                        edges.at<uchar>(i,j) = 255;
                        edgePixels.push(cv::Point(j,i));
                    }
                }
            }

            else if (edges.at<uchar>(i,j) != 0)
            {
                edges.at<uchar>(i,j) = 100;
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
            for(int j = std::max(p.x-radius, 0); j <= std::min(p.x + radius, edges.cols-1); j++)
            {
                if((gradMagn.at<float>(i,j) > threshold1)&&((gradMagn.at<float>(i,j) < threshold2))
                        && (edges.at<uchar>(i,j) == 100))
                {
                    edges.at<uchar>(i,j) = 255;
                    edgePixels.push(cv::Point(j,i));
                }
            }
        }
    }

    for(int i = 0; i < edges.rows; i++)
    {
        for(int j = 0; j < edges.cols; j++)
        {
            if ((edges.at<uchar>(i,j) > 0)&&(edges.at<uchar>(i,j) < 255))
            {
                edges.at<uchar>(i,j) = 0;
            }

        }
    }
}

void getDistanceMap(const cv::Mat & edges, cv::Mat & Dist)
{
    for(int r = 0; r < Dist.rows; r++)
    {
        for(int c = 0; c < Dist.cols; c++)
        {
            if (edges.at<uchar>(r,c) > 0)
                Dist.at<int>(r,c) = 0;
            else
                Dist.at<int>(r,c) = -1;
        }
    }

    bool findBorder;
    int radius;
    for(int r = 0; r < Dist.rows; r++)
    {
        for(int c = 0; c < Dist.cols; c++)
        {
            if(Dist.at<int>(r,c) == -1)
            {
                findBorder = false;
                radius = 1;
                while(!findBorder)
                {
                    for(int c1 = std::max(c-radius,0); c1 <= std::min(Dist.cols-1, c+radius); c1++)
                    {
                        if((Dist.at<int>(std::max(r-radius, 0), c1)==0) || (Dist.at<int>(std::min(r+radius, Dist.rows-1), c1)==0))
                        {
                            findBorder = true;
                            Dist.at<int>(r,c) = radius;
                            break;
                        }
                    }
                    if(!findBorder)
                    {
                        for(int r1 = std::max(r-radius,0); r1 <= std::min(Dist.rows-1, r+radius); r1++)
                        {
                            if((Dist.at<int>(r1, std::max(c-radius, 0))==0) || (Dist.at<int>(r1, std::min(c+radius, Dist.cols-1))==0))
                            {
                                findBorder = true;
                                Dist.at<int>(r,c) = radius;
                                break;
                            }
                        }
                    }

                    radius++;
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
    cv::Mat Dist(edges.rows, edges.cols, CV_32SC1);
    getDistanceMap(edges, Dist);

    //getDistanceMap verification
    /*
    cv::Mat DistCV, inv;
    cv::imshow("edges", edges);
    cv::bitwise_not(edges, inv);
    cv::imshow("inv", inv);
    cv::waitKey();
    cv::distanceTransform(inv, DistCV, CV_DIST_C, 3);


    bool distOk = true;
    for(int r = 0; r < DistCV.rows; r++)
    {
        for(int c = 0; c < DistCV.cols; c++)
        {
            if (Dist.at<int>(r,c) != (int)(DistCV.at<float>(r,c)))
            {
                distOk = false;
            }
        }
    }
    if (distOk)
        std::cout<<"getDistanceMap verification is ok"<<std::endl;
    else
       std::cout<<"getDistanceMap verification is not ok"<<std::endl;*/

    cv::Mat srcIntegral = src.clone();
    cv::integral(srcIntegral, srcIntegral);
    int windowSize, sum0, sum1, sum2;
    int rTopLeft, cTopLeft, rBottomRight, cBottomRight;
    int neighboursCount;

    for(int r = 1; r < srcIntegral.rows; r++)
    {
        for(int c = 1; c < srcIntegral.cols; c++)
        {
            windowSize = ksize*Dist.at<int>(r-1,c-1);

            rTopLeft = std::max(1, r-windowSize);
            cTopLeft = std::max(1, c-windowSize);
            rBottomRight = std::min(srcIntegral.rows-1, r+windowSize);
            cBottomRight = std::min(srcIntegral.cols-1, c+windowSize);
            neighboursCount = (rBottomRight - rTopLeft + 1)*(cBottomRight - cTopLeft + 1);

            sum0 = srcIntegral.at<cv::Vec3i>(rBottomRight,cBottomRight)[0] + srcIntegral.at<cv::Vec3i>(rTopLeft-1,cTopLeft-1)[0] - srcIntegral.at<cv::Vec3i>(rTopLeft-1,cBottomRight)[0] -srcIntegral.at<cv::Vec3i>(rBottomRight,cTopLeft-1)[0];
            sum1 = srcIntegral.at<cv::Vec3i>(rBottomRight,cBottomRight)[1] + srcIntegral.at<cv::Vec3i>(rTopLeft-1,cTopLeft-1)[1] - srcIntegral.at<cv::Vec3i>(rTopLeft-1,cBottomRight)[1] -srcIntegral.at<cv::Vec3i>(rBottomRight,cTopLeft-1)[1];
            sum2 = srcIntegral.at<cv::Vec3i>(rBottomRight,cBottomRight)[2] + srcIntegral.at<cv::Vec3i>(rTopLeft-1,cTopLeft-1)[2] - srcIntegral.at<cv::Vec3i>(rTopLeft-1,cBottomRight)[2] -srcIntegral.at<cv::Vec3i>(rBottomRight,cTopLeft-1)[2];

            if (neighboursCount > 0)
                dst.at<cv::Vec3b>(r-1, c-1) = cv::Vec3b((uchar)(sum0/neighboursCount), (uchar)(sum1/neighboursCount), (uchar)(sum2/neighboursCount));
            else
                dst.at<cv::Vec3b>(r-1, c-1) = src.at<cv::Vec3b>(r-1, c-1);

        }
    }

    //myFilter verification
    /*bool myFilterOk = true;
    for(int r = 0; r < dst.rows; r++)
    {
        for(int c = 0; c < dst.cols; c++)
        {
            if (edges.at<uchar>(r,c) == 255)
            {
                if(dst.at<cv::Vec3b>(r,c) != src.at<cv::Vec3b>(r,c))
                {
                    myFilterOk = false;
                }
            }

        }
    }
    if (myFilterOk)
        std::cout<<"verification is ok"<<std::endl;
    else
        std::cout<<"verification is not ok"<<std::endl;*/
}
