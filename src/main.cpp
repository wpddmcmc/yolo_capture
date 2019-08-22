#include "detecter.hpp"

int main()
{
    cv::VideoCapture cap(0);
    cv::Mat frame;
    DetectProcess detector;
    while(1)
    {
        cap >> frame;
        detector.Detection(frame);
        cv::imshow("view",frame);
        cv::waitKey(10);
    }
    return 1;
}