#include"opencv\cv.h"
#include"opencv\highgui.h"
#include"opencv2\videoio.hpp"
#include"opencv2\core\core.hpp"
#include "opencv2\opencv.hpp"
#include<iostream>
#include <ctime> 

using namespace cv;
using namespace std;

const int KEY_SPACE = 32;
const int KEY_ESC = 27;

CvHaarClassifierCascade *trained_model; 
CvMemStorage            *storage;

CvSeq* detect_car(IplImage *img);
int isOccupied(IplImage *img, CvSeq *object,int count);

int main(int argc, char** argv)
{
	std::cout << "Using OpenCV " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << "." << CV_SUBMINOR_VERSION << std::endl;

	CvCapture *capture;
	IplImage  *frame;
	int input_resize_percent = 100;
	int count = 0;
	if (argc < 3)
	{
		std::cout << "Usage " << argv[0] << " trained_model.xml video.mp4" << std::endl;
		return 0;
	}

	trained_model = (CvHaarClassifierCascade*)cvLoad(argv[1], 0, 0, 0);
	storage = cvCreateMemStorage(0);
	capture = cvCaptureFromAVI(argv[2]);
	
	double fps = cvGetCaptureProperty(capture,CV_CAP_PROP_FPS);
	cout << "Frames per Second:" << fps << endl;
	assert(trained_model && storage && capture);
	cvNamedWindow("video", 1);

	
	IplImage* frame1 = cvQueryFrame(capture);
	frame = cvCreateImage(cvSize((int)((frame1->width*input_resize_percent) / 100), (int)((frame1->height*input_resize_percent) / 100)), frame1->depth, frame1->nChannels);

	int key = 0;
	do
	{
		frame1 = cvQueryFrame(capture);

		if (!frame1)
			break;

		cvResize(frame1, frame);

		int start_s = clock();
		CvSeq *object=detect_car(frame);
		int checkCount = 4;
		count=isOccupied(frame, object,count);
		if (count > checkCount)
		{
			checkCount = count;
		}
		if(checkCount > 5 && checkCount < 10)
		{
			cout << "About to be occupied"<< endl;
		}
		else if (checkCount >= 10)
		{
			cout << "Car is occupied" << endl;
		}
		cvShowImage("video", frame);
		int stop_s = clock();
		cout << "time: " << (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000 << endl;
		key = cvWaitKey(33);

		if (key == KEY_SPACE)
			key = cvWaitKey(0);

		if (key == KEY_ESC)
			break;

	} while (1);

	cvDestroyAllWindows();
	cvReleaseImage(&frame);
	cvReleaseCapture(&capture);
	cvReleaseHaarClassifierCascade(&trained_model);
	cvReleaseMemStorage(&storage);

	return 0;
}
//------------------------------------------------------------------------------------------------------------

int isOccupied(IplImage *img, CvSeq *object,int count)
{
	for (int i = 0; i < (object ? object->total : 0); i++)
	{
		
		CvRect *r = (CvRect*)cvGetSeqElem(object, i);
		int area = r->width * r->height;
		//cout << "Area is"<<area<<endl;
		if (area >100000)
		{
			cvRectangle(img,
				cvPoint(r->x, r->y),
				cvPoint(r->x + r->width, r->y + r->height),
				CV_RGB(255, 0, 0), 2, 8, 0);
				if (area == 116964)
				{
					count++;
					cout << "Area is" << area << endl;
					return count;
				}
			cout << "Area is" << area << endl;
		}
	}
	return 0;
}

CvSeq* detect_car(IplImage *img)
{
	CvSize img_size = cvGetSize(img);
	CvSeq *object = cvHaarDetectObjects(
		img,
		trained_model,
		storage,
		1.5, //Increase the search range by 50% in each subsequent stage
		1, //2        //------------------MIN NEIGHBOURS
		0, // Puring not required
		cvSize(70,70),//cvSize( 30,30), // ------MINSIZE
		img_size //
	);
	return object;
	
}

