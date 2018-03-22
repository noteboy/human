#include "opencv2/core/core.hpp"  
#include "opencv2/objdetect.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/highgui/highgui.hpp"  
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/video/video.hpp"  
#include "opencv2/ml.hpp"
#include "opencv2/opencv.hpp"
#include <opencv2/objdetect/objdetect.hpp>  
#include <opencv2/imgcodecs.hpp>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <math.h>
#include <iostream>  
#include <vector>
#include <fstream>
#include <cstdlib>
using namespace std;
using namespace cv;
using namespace cv::ml;

//��������
void get_svm_detector(const Ptr< SVM > & svm, vector< float > & hog_detector);
void convert_to_ml(const std::vector< Mat > & train_samples, Mat& trainData);
void load_images(const String & dirname, vector< Mat > & img_lst, bool showImages);
void sample_neg(const vector< Mat > & full_neg_lst, vector< Mat > & neg_lst, const Size & size);
void computeHOGs(const Size wsize, const vector< Mat > & img_lst, vector< Mat > & gradient_lst);
int test_trained_detector(String obj_det_filename, String test_dir, String videofilename);

//��������
void get_svm_detector(const Ptr< SVM >& svm, vector< float > & hog_detector)
{
	// get the support vectors
	Mat sv = svm->getSupportVectors();
	const int sv_total = sv.rows;
	// get the decision function
	Mat alpha, svidx;
	double rho = svm->getDecisionFunction(0, alpha, svidx);

	CV_Assert(alpha.total() == 1 && svidx.total() == 1 && sv_total == 1);	//�����е�����������ʱ�����ش���
	CV_Assert((alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
		(alpha.type() == CV_32F && alpha.at<float>(0) == 1.f));
	CV_Assert(sv.type() == CV_32F);
	hog_detector.clear();

	hog_detector.resize(sv.cols + 1);
	memcpy(&hog_detector[0], sv.ptr(), sv.cols * sizeof(hog_detector[0]));	//memcpyָ����c��c++ʹ�õ��ڴ濽��������memcpy�����Ĺ����Ǵ�Դsrc��ָ���ڴ��ַ����ʼλ�ÿ�ʼ����n���ֽڵ�Ŀ��dest��ָ���ڴ��ַ����ʼλ���С�
	hog_detector[sv.cols] = (float)-rho;
}

/*
* Convert training/testing set to be used by OpenCV Machine Learning algorithms.
* TrainData is a matrix of size (#samples x max(#cols,#rows) per samples), in 32FC1.
* Transposition of samples are made if needed.
*/
void convert_to_ml(const vector< Mat > & train_samples, Mat& trainData)
{
	//--Convert data
	const int rows = (int)train_samples.size();	//��������ѵ����������
	const int cols = (int)std::max(train_samples[0].cols, train_samples[0].rows);	//����ȡ����ͼƬ�п����߶��нϴ����һ��
	Mat tmp(1, cols, CV_32FC1); //< used for transposition if needed
	trainData = Mat(rows, cols, CV_32FC1);

	for (size_t i = 0; i < train_samples.size(); ++i)
	{
		CV_Assert(train_samples[i].cols == 1 || train_samples[i].rows == 1);

		if (train_samples[i].cols == 1)
		{
			transpose(train_samples[i], tmp);
			tmp.copyTo(trainData.row((int)i));
		}
		else if (train_samples[i].rows == 1)
		{
			train_samples[i].copyTo(trainData.row((int)i));
		}
	}
}

void load_images(const String & dirname, vector< Mat > & img_lst, bool showImages = false)
{	//����Ŀ¼�µ�ͼƬ����
	vector< String > files;
	glob(dirname, files);		//����һ��������ƥ���ļ�/Ŀ¼�����顣�����򷵻�false

	for (size_t i = 0; i < files.size(); ++i)
	{
		Mat img = imread(files[i]); // load the image
		if (img.empty())            // invalid image, skip it.
		{
			cout << files[i] << " is invalid!" << endl;
			continue;
		}

		if (showImages)
		{
			imshow("image", img);
			waitKey(1);
		}
		img_lst.push_back(img);//��Imgѹ��img_lst
	}
}

void sample_neg(const vector< Mat > & full_neg_lst, vector< Mat > & neg_lst, const Size & size)
{	//�ú�����ÿһ��������������һ�������64*128�ߴ������������֮ǰ�Ѿ��������ˣ�����main������û��ʹ�øú���
	Rect box;
	box.width = size.width;	//���ڼ�������
	box.height = size.height;	//���ڼ�����߶�

	const int size_x = box.width;
	const int size_y = box.height;

	srand((unsigned int)time(NULL));		//�������������

	for (size_t i = 0; i < full_neg_lst.size(); i++)
	{	//��ÿ�����������вü������ָ��x,y,�ü�һ���ߴ�Ϊ�������С�ĸ�����
		box.x = rand() % (full_neg_lst[i].cols - size_x);
		box.y = rand() % (full_neg_lst[i].rows - size_y);
		Mat roi = full_neg_lst[i](box);
		neg_lst.push_back(roi.clone());
	}
}

void computeHOGs(const Size wsize, const vector< Mat > & img_lst, vector< Mat > & gradient_lst)
{	//����HOG����
	HOGDescriptor hog;
	hog.winSize = wsize;

	Rect r = Rect(0, 0, wsize.width, wsize.height);
	r.x += (img_lst[0].cols - r.width) / 2;	//������ͼƬ�ĳߴ��ȥ������ĳߴ磬�ٳ���2
	r.y += (img_lst[0].rows - r.height) / 2;

	Mat gray;
	vector< float > descriptors;

	for (size_t i = 0; i< img_lst.size(); i++)
	{
		cvtColor(img_lst[i](r), gray, COLOR_BGR2GRAY);
		hog.compute(gray, descriptors, Size(8, 8), Size(0, 0));	//Size(8,8)Ϊ�����ƶ�������
		gradient_lst.push_back(Mat(descriptors).clone());
	}
}

int test_trained_detector(String obj_det_filename, String test_dir, String videofilename)
{	//��videofilenameΪ�գ���ֻ���ͼƬ�е�����
	cout << "Testing trained detector..." << endl;
	HOGDescriptor hog;
	hog.load(obj_det_filename);

	vector< String > files;
	glob(test_dir, files);

	int delay = 0;
	VideoCapture cap;

	if (videofilename != "")
	{
		cap.open(videofilename);
	}

	obj_det_filename = "testing " + obj_det_filename;
	namedWindow(obj_det_filename, WINDOW_NORMAL);

	for (size_t i = 0;; i++)
	{
		Mat img;

		if (cap.isOpened())
		{
			cap >> img;
			delay = 1;
		}
		else if (i < files.size())
		{
			img = imread(files[i]);
		}

		if (img.empty())
		{
			return 0;
		}

		vector< Rect > detections;
		vector< double > foundWeights;

		hog.detectMultiScale(img, detections, foundWeights);
		for (size_t j = 0; j < detections.size(); j++)
		{
			if (foundWeights[j] < 0.5) continue;	//���Ȩֵ��С�ļ�ⴰ��
			Scalar color = Scalar(0, foundWeights[j] * foundWeights[j] * 200, 0);
			rectangle(img, detections[j], color, img.cols / 400 + 1);
		}

		imshow(obj_det_filename, img);

		if (27 == waitKey(delay))
		{
			return 0;
		}
	}
	return 0;
}

int main(int argc, char** argv)
{
	const char* keys =
	{
		"{help h|     | show help message}"
		"{pd    |  D:/INRIAPerson/96X160H96/Train/pos   | path of directory contains possitive images}"
		"{nd    |  D:/negphoto   | path of directory contains negative images}"
		"{td    |  D:/INRIAPerson/Test/pos   | path of directory contains test images}"
		"{tv    |     | test video file name}"
		"{dw    |  64   | width of the detector}"
		"{dh    |  128   | height of the detector}"
		"{d     |false| train twice}"
		"{t     |true| test a trained detector}"
		"{v     |false| visualize training steps}"
		"{fn    |D:/my_detector.yml| file name of trained SVM}"
	};

	CommandLineParser parser(argc, argv, keys);	//�����к�������ȡkeys�е��ַ��� ����key�ĸ�ʽΪ:���� ���| ���� |��ʾ�ַ���

	if (parser.has("help"))
	{
		parser.printMessage();
		exit(0);
	}

	String pos_dir = parser.get< String >("pd");	//������Ŀ¼
	String neg_dir = parser.get< String >("nd");	//������Ŀ¼
	String test_dir = parser.get< String >("td");	//��������Ŀ¼
	String obj_det_filename = parser.get< String >("fn");	//ѵ���õ�SVM������ļ���
	String videofilename = parser.get< String >("tv");	//������Ƶ
	int detector_width = parser.get< int >("dw");	//��������	
	int detector_height = parser.get< int >("dh");	//������߶�
	bool test_detector = parser.get< bool >("t");	//����ѵ���õļ����
	bool train_twice = parser.get< bool >("d");		//ѵ������
	bool visualization = parser.get< bool >("v");	//ѵ�����̿��ӻ�������false����Ȼ��ը)

	if (test_detector)	//��Ϊtrue����Բ��Լ����в���
	{
		test_trained_detector(obj_det_filename, test_dir, videofilename);
		exit(0);
	}

	if (pos_dir.empty() || neg_dir.empty())	//���ǿ�
	{
		parser.printMessage();
		cout << "Wrong number of parameters.\n\n"
			<< "Example command line:\n" << argv[0] << " -pd=/INRIAPerson/96X160H96/Train/pos -nd=/INRIAPerson/neg -td=/INRIAPerson/Test/pos -fn=HOGpedestrian96x160.yml -d\n"
			<< "\nExample command line for testing trained detector:\n" << argv[0] << " -t -dw=96 -dh=160 -fn=HOGpedestrian96x160.yml -td=/INRIAPerson/Test/pos";
		exit(1);
	}

	vector< Mat > pos_lst,	//������ͼƬ����
		full_neg_lst,		//������ͼƬ����
		neg_lst,			//������ĸ�����ͼƬ����
		gradient_lst;		//HOG���������뵽���ݶ���Ϣ����
	vector< int > labels;	//��ǩ����

	clog << "Positive images are being loaded...";
	load_images(pos_dir, pos_lst, visualization);	//����ͼƬ pos�������ĳߴ�Ϊ96*160
	if (pos_lst.size() > 0)
	{
		clog << "...[done]" << endl;
	}
	else
	{
		clog << "no image in " << pos_dir << endl;
		return 1;
	}

	Size pos_image_size = pos_lst[0].size(); //��ߴ����pos_image_size=�������ߴ�

											 //��������������Ƿ������ͬ�ߴ�
	for (size_t i = 0; i < pos_lst.size(); ++i)
	{
		if (pos_lst[i].size() != pos_image_size)
		{
			cout << "All positive images should be same size!" << endl;
			exit(1);
		}
	}

	pos_image_size = pos_image_size / 8 * 8;

	//��pos_image_size�ĳߴ�Ϊ������ĳߴ�
	if (detector_width && detector_height)
	{
		pos_image_size = Size(detector_width, detector_height);
	}

	labels.assign(pos_lst.size(), +1);              //assign()Ϊlabels����pos_lst.size()��С����������+1��� ��ʾΪ������
	const unsigned int old = (unsigned int)labels.size();	//�ɱ�ǩ��С

	clog << "Negative images are being loaded...";
	load_images(neg_dir, neg_lst, false);	//���ظ�����ͼƬ
											//sample_neg(full_neg_lst, neg_lst, pos_image_size);  
	clog << "...[done]" << endl;

	labels.insert(labels.end(), neg_lst.size(), -1);	//��labels������β�����neg_lst.size()��С����������-1��� ��ʾΪ������
	CV_Assert(old < labels.size());		//CV_Assert()���ã�CV_Assert�����������еı��ʽֵΪfalse���򷵻�һ��������Ϣ��

	clog << "Histogram of Gradients are being calculated for positive images...";
	computeHOGs(pos_image_size, pos_lst, gradient_lst);	//����������ͼƬ��HOG����
	clog << "...[done]" << endl;

	clog << "Histogram of Gradients are being calculated for negative images...";
	computeHOGs(pos_image_size, neg_lst, gradient_lst);	//���㸺����ͼƬ��HOG����
	clog << "...[done]" << endl;

	Mat train_data;
	convert_to_ml(gradient_lst, train_data);	//ת��Ϊml�����ѵ��������ʽ

	clog << "Training SVM...";
	Ptr< SVM > svm = SVM::create();
	/* Default values to train SVM */
	svm->setCoef0(0.0);
	svm->setDegree(3);
	svm->setTermCriteria(TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 1e-3));
	svm->setGamma(0);
	svm->setKernel(SVM::LINEAR);	//�������Ժ˺���������sigmoid ��RBF ���������ã���ֵ��0-5��
	svm->setNu(0.5);
	svm->setP(0.1); // for EPSILON_SVR, epsilon in loss function?
	svm->setC(0.01); // From paper, soft classifier
	svm->setType(SVM::EPS_SVR); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task
	svm->train(train_data, ROW_SAMPLE, Mat(labels));
	clog << "...[done]" << endl;

	//ѵ������
	if (train_twice)
	{
		clog << "Testing trained detector on negative images. This may take a few minutes...";
		HOGDescriptor my_hog;
		my_hog.winSize = pos_image_size;

		// Set the trained svm to my_hog
		vector< float > hog_detector;
		get_svm_detector(svm, hog_detector);
		my_hog.setSVMDetector(hog_detector);

		vector< Rect > detections;
		vector< double > foundWeights;

		for (size_t i = 0; i < full_neg_lst.size(); i++)
		{
			my_hog.detectMultiScale(full_neg_lst[i], detections, foundWeights);
			for (size_t j = 0; j < detections.size(); j++)
			{
				Mat detection = full_neg_lst[i](detections[j]).clone();
				resize(detection, detection, pos_image_size);
				neg_lst.push_back(detection);
			}
			if (visualization)
			{
				for (size_t j = 0; j < detections.size(); j++)
				{
					rectangle(full_neg_lst[i], detections[j], Scalar(0, 255, 0), 2);
				}
				imshow("testing trained detector on negative images", full_neg_lst[i]);
				waitKey(5);
			}
		}
		clog << "...[done]" << endl;

		labels.clear();
		labels.assign(pos_lst.size(), +1);
		labels.insert(labels.end(), neg_lst.size(), -1);

		gradient_lst.clear();
		clog << "Histogram of Gradients are being calculated for positive images...";
		computeHOGs(pos_image_size, pos_lst, gradient_lst);
		clog << "...[done]" << endl;

		clog << "Histogram of Gradients are being calculated for negative images...";
		computeHOGs(pos_image_size, neg_lst, gradient_lst);
		clog << "...[done]" << endl;

		clog << "Training SVM again...";
		convert_to_ml(gradient_lst, train_data);
		svm->train(train_data, ROW_SAMPLE, Mat(labels));
		clog << "...[done]" << endl;
	}

	vector< float > hog_detector;	//����hog�����
	get_svm_detector(svm, hog_detector);	//�õ�ѵ���õļ����
	HOGDescriptor hog;
	hog.winSize = pos_image_size;	//���ڴ�С
	hog.setSVMDetector(hog_detector);
	hog.save(obj_det_filename);		//���������

	test_trained_detector(obj_det_filename, test_dir, videofilename);	//���ѵ����

	return 0;
}