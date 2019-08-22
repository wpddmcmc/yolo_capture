#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"
extern "C"
{
#include "darknet.h"
}
using namespace cv;
using namespace std;

class DetectProcess{
    public:
    DetectProcess();
    void Detection(cv::Mat &src);
private:
    char *datacfg;
    char *name_list;
    char **names;
    char *cfgfile;
    char *weightfile ;

    float thresh,hier_thresh;
    float nms;

    network *net;

    image **alphabet;
    void Image2Mat(image p,cv::Mat &Img);
    float get_pixel(image m, int x, int y, int c);
    void Mat2Image(cv::Mat RefImg,image *im);
};