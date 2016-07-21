#include "lab2.hpp"

cv::Mat frameCopy, markers;
std::vector<cv::Point> objectMarkerPoints;
cv::Point prevPt1(-1, -1);
cv::Point prevPt2(-1, -1);

static void onMouse( int event, int x, int y, int flags, void* )
{
    if( (flags & CV_EVENT_FLAG_LBUTTON) && ( event == CV_EVENT_MOUSEMOVE )&&(x>0)&&(y>0)&&(x<frameCopy.cols)&&(y<frameCopy.rows))
    {
        if(prevPt1 == cv::Point(-1,-1))
            prevPt1 = cv::Point(x,y);
        else
        {
            cv::line(frameCopy, prevPt1, cv::Point(x,y),cv::Scalar(255,255,255),2);
            prevPt1 = cv::Point(x,y);
            markers.at<int>(y,x) = 1;
            objectMarkerPoints.push_back(prevPt1);
        }

        cv::imshow("frame", frameCopy);

    }
    if( (flags & CV_EVENT_FLAG_RBUTTON) &&( event == CV_EVENT_MOUSEMOVE )&&(x>0)&&(y>0)&&(x<frameCopy.cols)&&(y<frameCopy.rows))
    {
        if(prevPt2 == cv::Point(-1,-1))
            prevPt2 = cv::Point(x,y);
        else
        {
            cv::line(frameCopy, prevPt2, cv::Point(x,y), cv::Scalar(0,0,0), 2);
            prevPt2 = cv::Point(x,y);
            markers.at<int>(y,x) = 2;
        }

        cv::imshow("frame", frameCopy);

    }

}

int Lab2(std::string fileName)
{
    cv::VideoCapture cap(fileName);
    if(!cap.isOpened())
    {
        std::cout<<"Can not read video "<<fileName<<std::endl;
        return -1;
    }

    std::vector<cv::Point> trace1;
    std::vector<cv::Point> trace2;

    cv::Mat prevFrame;
    cap >> prevFrame;
    cv::Mat prevGrayscale;
    cv::cvtColor(prevFrame, prevGrayscale, CV_BGR2GRAY);
    cv::Mat frameTrace = prevFrame.clone();
    frameCopy = prevFrame.clone();
    markers = cv::Mat::zeros(prevFrame.rows, prevFrame.cols, CV_32SC1);
    cv::imshow("frame", prevFrame);
    cv::setMouseCallback( "frame", onMouse, 0 );
    cv::waitKey();

    cv::Mat prevFrameCopy = prevFrame.clone();
    cv::Rect tmplRect = cv::boundingRect(cv::Mat(objectMarkerPoints));
    cv::Rect firstRect = tmplRect;
    cv::rectangle(prevFrameCopy,tmplRect,cv::Scalar(0,0,255));

    cv::Point firstCenter = cv::Point(tmplRect.x + (tmplRect.width/2), tmplRect.y + (tmplRect.height/2));
    cv::Point center1 = firstCenter;//center of rect1 (optical flow method)
    trace1.push_back(firstCenter);
    trace2.push_back(firstCenter);

    //correlation method (start)
    cv::Mat tmpl = prevFrame(tmplRect);
    cv::imshow("tmpl", tmpl);
    cv::Mat result;
    cv::matchTemplate(prevFrame, tmpl, result, CV_TM_CCORR_NORMED);
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::Rect rect;
    minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
    rect = cv::Rect(maxLoc, tmpl.size());
    cv::rectangle(prevFrameCopy, rect, cv::Scalar(0, 0, 255), 2);

    //optical flow method (start)
    cv::watershed(prevFrame, markers);
    cv::Mat mask(prevFrame.rows, prevFrame.cols, CV_8UC1);
    for(int i = 0; i < prevFrame.rows; i++)
    {
        for(int j = 0; j < prevFrame.cols; j++)
        {
            if (markers.at<int>(i,j) == 1)
            {
                mask.at<uchar>(i,j) = 255;
            }
            else
            {
                mask.at<uchar>(i,j) = 0;
            }
        }
    }

    std::vector<cv::Point2f> prevObjKeyPoints;
    int maxCorners = 1500;
    double qualityLevel=0.001;
    double minDistance = 10;
    cv::goodFeaturesToTrack(prevGrayscale, prevObjKeyPoints, maxCorners, qualityLevel, minDistance, mask);
    for(int i = 0; i < prevObjKeyPoints.size(); i++)
    {
        cv::circle(prevFrameCopy, prevObjKeyPoints[i], 5, cv::Scalar(0,255,0), -1);
    }

    std::vector<cv::Point2f> prevBackgroundKeyPoints;
    cv::bitwise_not(mask, mask);
    cv::goodFeaturesToTrack(prevGrayscale, prevBackgroundKeyPoints, maxCorners, qualityLevel, minDistance, mask);
    for(int i = 0; i < prevBackgroundKeyPoints.size(); i++)
    {
        cv::circle(prevFrameCopy, prevBackgroundKeyPoints[i], 5, cv::Scalar(255,0,0), -1);
    }

    cv::imshow("prev", prevFrameCopy);


    cv::Mat frame;
    cv::Mat grayscale;
    std::vector<cv::Point2f> objKeyPoints;
    std::vector<cv::Point2f> backgroundKeyPoints;
    std::vector<cv::Point2f>  points;
    std::vector<uchar> statusObj;
    std::vector<uchar> statusBackground;
    cv::Rect rect1;//rect from optical flow method
    rect1.width = firstRect.width;
    rect1.height = firstRect.height;
    cv::Point medianOffset;
    std::vector<int> pointsOffsetX;
    std::vector<int> pointsOffsetY;
    for(;;)
    {
        cap >> frame;
        frameTrace = frame.clone();

        if(!frame.empty())
        {
            frameCopy = frame.clone();

            cv::matchTemplate(frame, tmpl, result, CV_TM_CCORR_NORMED);
            minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
            tmplRect = cv::Rect(maxLoc, tmpl.size());
            trace2.push_back(cv::Point(tmplRect.x + (tmplRect.width/2), tmplRect.y + (tmplRect.height/2)));
            for(int i = 0; i < trace2.size(); i++)
            {
                cv::circle(frameTrace,trace2[i], 5, cv::Scalar(0,0,255), -1 );
            }
            for(int i = 0; i < trace1.size(); i++)
            {
                cv::circle(frameTrace,trace1[i], 5, cv::Scalar(255,255,0), -1 );
            }

            cv::imshow("Trace", frameTrace);
            cv::rectangle(frameCopy, tmplRect, cv::Scalar(0, 0, 255), 2);
            rect1.x = center1.x - (firstRect.width/2);
            rect1.y = center1.y - (firstRect.height/2);

            cv::Rect frameRect(0,0,frame.cols, frame.rows);
            rect1 &= frameRect;
            cv::rectangle(frameCopy, rect1, cv::Scalar(0, 255, 0), 2);
            cv::imshow("cur", frameCopy);

            markers.setTo(0);
            objectMarkerPoints.clear();
            for(int i = 0; i < objKeyPoints.size(); i++)
            {

                markers.at<int>((int)(objKeyPoints[i].y),(int)(objKeyPoints[i].x) ) = 1;
                objectMarkerPoints.push_back(objKeyPoints[i]);

            }


            for(int i = 0; i < backgroundKeyPoints.size(); i++)
            {
                if ((backgroundKeyPoints[i].y > 0) && (backgroundKeyPoints[i].y < markers.rows) && (backgroundKeyPoints[i].x > 0) && (backgroundKeyPoints[i].x < markers.cols))
                    markers.at<int>((int)(backgroundKeyPoints[i].y),(int)(backgroundKeyPoints[i].x) ) = 2;

            }

            for(int i = 0; i < markers.rows; i++)
            {
                for(int j = 0; j < markers.cols; j++)
                {
                    if((markers.at<int>(i,j)==1 ) && (abs(firstCenter.y - i) > firstRect.height/3) && (abs(firstCenter.x - j) > firstRect.width/3))
                    {
                        markers.at<int>(i,j)==2;
                        cv::circle(prevFrameCopy, cv::Point(j,i), 5, cv::Scalar(0,255,255),-1);
                    }
                    if((markers.at<int>(i,j)==2 ) && (abs(firstCenter.y - i) < firstRect.height/2) && (abs(firstCenter.x - j) < firstRect.width/2))
                    {
                        markers.at<int>(i,j)==1;
                        cv::circle(prevFrameCopy, cv::Point(j,i), 5, cv::Scalar(255,0,255),-1);
                    }
                }
            }
            cv::imshow("prev", prevFrameCopy);

            cv::watershed(frame, markers);

            for(int i = 0; i < frame.rows; i++)
            {
                for(int j = 0; j < frame.cols; j++)
                {
                    if (markers.at<int>(i,j) == 1)
                        mask.at<uchar>(i,j) = 255;
                    else
                        mask.at<uchar>(i,j) = 0;
                }
            }

            for(int i = 0; i < prevObjKeyPoints.size(); i++)
            {
                cv::circle(prevFrameCopy, prevObjKeyPoints[i], 5, cv::Scalar(0,255,0), -1);
            }
            for(int i = 0; i < prevObjKeyPoints.size(); i++)
            {
                cv::circle(prevFrameCopy, prevBackgroundKeyPoints[i], 5, cv::Scalar(0,0,255), -1);
            }
            cv::imshow("prev", prevFrameCopy);

            cv::cvtColor(frame, grayscale, CV_BGR2GRAY);
            objKeyPoints.clear();
            backgroundKeyPoints.clear();
            cv::goodFeaturesToTrack(grayscale, objKeyPoints, maxCorners, qualityLevel, minDistance, mask);
            cv::bitwise_not(mask, mask);
            cv::goodFeaturesToTrack(grayscale, backgroundKeyPoints, maxCorners, qualityLevel, minDistance, mask);

            cv::Mat err;
            cv::calcOpticalFlowPyrLK(prevGrayscale, grayscale, prevObjKeyPoints, objKeyPoints, statusObj, err, cv::Size(21,21), 3, cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 30, 0.01), 1, 1e-4);
            points.clear();
            pointsOffsetX.clear();
            pointsOffsetY.clear();
            for(int i = 0; i < statusObj.size(); i++)
            {
                if ((abs(objKeyPoints[i].x - center1.x) < firstRect.width/2) &&(abs(objKeyPoints[i].y - center1.y) < firstRect.height/2)&& (((int)(statusObj[i])) == 1) && (objKeyPoints[i].y > 0) && (objKeyPoints[i].y < markers.rows) && (objKeyPoints[i].x > 0) && (objKeyPoints[i].x < markers.cols))
                {
                    points.push_back(objKeyPoints[i]);
                    cv::circle(frameCopy, objKeyPoints[i], 5, cv::Scalar(0,255,0), -1);
                    pointsOffsetX.push_back(objKeyPoints[i].x - prevObjKeyPoints[i].x);
                    pointsOffsetY.push_back(objKeyPoints[i].y - prevObjKeyPoints[i].y);
                }
                else
                    cv::circle(frameCopy, backgroundKeyPoints[i], 5, cv::Scalar(0,0,0), -1);

            }
            objKeyPoints.clear();
            objKeyPoints = points;

            std::sort(pointsOffsetX.begin(), pointsOffsetX.end());
            std::sort(pointsOffsetY.begin(), pointsOffsetY.end());
            medianOffset.x = pointsOffsetX[pointsOffsetX.size()/2];;
            medianOffset.y = pointsOffsetX[pointsOffsetY.size()/2];
            center1 = center1 + medianOffset;
            trace1.push_back(center1);

            points.clear();
            cv::calcOpticalFlowPyrLK(prevGrayscale,grayscale, prevBackgroundKeyPoints, backgroundKeyPoints, statusBackground, err);

            for(int i = 0; i < statusBackground.size(); i++)
            {
                if ((((int)(statusBackground[i])) == 1) && (backgroundKeyPoints[i].y > 0) && (backgroundKeyPoints[i].y < markers.rows) && (backgroundKeyPoints[i].x > 0) && (backgroundKeyPoints[i].x < markers.cols))
                {
                    points.push_back(backgroundKeyPoints[i]);
                    cv::circle(frameCopy, backgroundKeyPoints[i], 5, cv::Scalar(0,0,255), -1);
                }
                else
                    cv::circle(frameCopy, backgroundKeyPoints[i], 5, cv::Scalar(0,0,0), -1);

            }
            backgroundKeyPoints.clear();
            backgroundKeyPoints = points;

            cv::imshow("cur", frameCopy);
            cv::waitKey();

            prevFrame = frame;
            prevFrameCopy = frame;
            prevBackgroundKeyPoints = backgroundKeyPoints;
            prevObjKeyPoints = objKeyPoints;
        }
        else
            break;
    }

}
