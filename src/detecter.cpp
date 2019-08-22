#include "detecter.hpp"

DetectProcess::DetectProcess()
{
	// yolo
    datacfg = "../data/coco.data" ;
    name_list = option_find_str(read_data_cfg(datacfg), "names", "data/names.list");
    names = get_labels(name_list);
    cfgfile = "../data/model/yolov3-tiny.cfg";
    weightfile = "../data/model/yolov3-tiny.weights";

    thresh=.55, hier_thresh=.55;
	nms=.45;

    net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);

    alphabet = load_alphabet_fix("../data/labels");
}

/************************************************* 
    Function:       Mat2Image 
    Description:   	change format: mat->image
	Input:          
					Mat RefImg - mat image need to reformat
					image *im - the output image format image
    Output:         image *im - the output image format image
    Return:         void
    Others:         none
    *************************************************/
void DetectProcess::Mat2Image(Mat RefImg,image *im)
{
	CV_Assert(RefImg.depth() == CV_8U);		//judge if  RefImag is CV_8U
	int h = RefImg.rows;
	int w = RefImg.cols;
	int channels = RefImg.channels();
	*im = make_image(w, h, 3);		//create 3 channels image
	int count = 0;
	switch (channels)
	{
	case 1:
	{
		MatIterator_<unsigned char> it, end;
		for (it = RefImg.begin<unsigned char>(), end = RefImg.end<unsigned char>(); it != end; ++it)
		{
			im->data[count] = im->data[w * h + count] = im->data[w * h * 2 + count] = (float)(*it) / 255.0;

			++count;
		}
		break;
	}

	case 3:
	{
		MatIterator_<Vec3b> it, end;
		for (it = RefImg.begin<Vec3b>(), end = RefImg.end<Vec3b>(); it != end; ++it)
		{
			im->data[count] = (float)(*it)[2] / 255.0;
			im->data[w * h + count] = (float)(*it)[1] / 255.0;
			im->data[w * h * 2 + count] = (float)(*it)[0] / 255.0;

			++count;
		}
		break;
	}

	default:
		printf("Channel number not supported.\n");
		break;
	}
}

/************************************************* 
    Function:       get_pixel 
    Description:   	change format: image->mat
	Input:          
					image m	- image need to get pixel
					int x - width
					int y - height
					int c - channels
    Output:         pixel of input image
    Return:         float
    Others:         none
    *************************************************/
float DetectProcess::get_pixel(image m, int x, int y, int c)
{
    assert(x < m.w && y < m.h && c < m.c);
    return m.data[c*m.h*m.w + y*m.w + x];
}

/************************************************* 
    Function:       image2mat 
    Description:   	change format: image->mat
	Input:          
					image p	- image image need to reformat
					Mat *Img -	the output mat format image
    Output:         Mat *Img - the output immatage format image
    Return:         void
    Others:         none
    *************************************************/
void DetectProcess::Image2Mat(image p,Mat &Img)
{
	IplImage *disp = cvCreateImage(cvSize(p.w,p.h), IPL_DEPTH_8U, p.c);
    image copy = copy_image(p);
    constrain_image(copy);

	int x, y, k;
	if (p.c == 3)
		rgbgr_image(p);

	int step = disp->widthStep;
	for (y = 0; y < p.h; ++y)
	{
		for (x = 0; x < p.w; ++x)
		{
			for (k = 0; k < p.c; ++k)
			{
				disp->imageData[y * step + x * p.c + k] = (unsigned char)(get_pixel(p, x, y, k) * 255);
			}
		}
	}
	if (0)
	{
		int w = 448;
		int h = w * p.h / p.w;
		if (h > 1000)
		{
			h = 1000;
			w = h * p.w / p.h;
		}
		IplImage *buffer = disp;
		disp = cvCreateImage(cvSize(w, h), buffer->depth, buffer->nChannels);
		cvResize(buffer, disp, CV_INTER_LINEAR);
		cvReleaseImage(&buffer);
	}
	
	Img=cvarrToMat(disp);
	free_image(copy);
   	cvReleaseImage(&disp);
}

/************************************************* 
    Function:       detecter 
    Description:   	darknet detect car
	Input:          
					Mat &src - image need to detect
    Output:         Mat &src - image after drawing detect target
    Return:         void
    Others:         none
    *************************************************/
void DetectProcess::Detection(Mat &src)
{  	
	image im;			//net input image
	//format change
	Mat2Image(src,&im);
    float nms=.45;
   	
    layer l = net->layers[(net->n)-1];    
    image sized = letterbox_image(im, net->w, net->h);
    float *X = sized.data;
	network_predict(net, X);

	int nboxes = 0;
	detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
	if (nms)
	{
		do_nms_sort(dets, nboxes, 80, nms);
	}
	int rect_scalar[nboxes][4];	//rect_scalar[i][0] left rect_scalar[i][1] right rect_scalar[i][2] top rect_scalar[i][3] bottom
	//draw_detections(im, dets, nboxes, thresh, names, alphabet,80);
	get_detections(im, dets, nboxes, thresh, names, alphabet,80,rect_scalar);
	vector<Rect> detectBox;
	for(int i=0;i<nboxes;i++)
	{
		detectBox.push_back(Rect(rect_scalar[i][0],rect_scalar[i][2],rect_scalar[i][1]-rect_scalar[i][0],rect_scalar[i][3]-rect_scalar[i][2]));
	}
	Mat result;
	Image2Mat(im,result);
	for(int i=0;i<detectBox.size();i++)
	{
		char position[10];
		sprintf(position,"(%d,%d)",detectBox[i].x+detectBox[i].width/2,detectBox[i].y+detectBox[i].height/2);
		putText(result,position,Point(detectBox[i].tl().x,detectBox[i].tl().y+30),CV_FONT_HERSHEY_PLAIN, 1, Scalar(128, 0, 255), 1);
	}
	result.copyTo(src);
    free_detections(dets, nboxes);
	free_image(sized);
}