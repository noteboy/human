#include <opencv2\highgui\highgui.hpp>  
#include <opencv2\opencv.hpp>  
#include <iostream>  

using namespace std;
using namespace cv;

int main() {
	Mat photo = imread("c:\\users\\liuxiaolong\\desktop\\design\\photo\\result\\singlep3.jpg");
	for (int i = 0; i < photo.cols; i = i + 25)
		line(photo,Point(i,0),Point(i,photo.rows), Scalar(0, 0, 255), 1, 8, 0);

	for (int i = 0; i < photo.rows; i = i +25)
		line(photo, Point(0, i), Point(photo.cols, i), Scalar(0, 0, 255), 1, 8, 0);
	
	imwrite("c:\\users\\liuxiaolong\\desktop\\design\\photo\\result\\asdsa.jpg", photo);
}




//#include <opencv2\highgui\highgui.hpp>  
//#include <opencv2\opencv.hpp>  
//#include <iostream>  
//
//
//using namespace cv;
//using namespace std;
//
//typedef unsigned char byte;
//void gradientgray(Mat &src, Mat &mag);
//int main() {
//
//	Mat src = imread("c:\\users\\liuxiaolong\\desktop\\design\\photo\\singlep3.jpg", 0);//�ԻҶ���ʽ��ȡͼƬ  
//	Mat dst;
//	gradientgray(src, dst);
//	imwrite("c:\\users\\liuxiaolong\\desktop\\design\\photo\\result\\singlep3.jpg", dst); //�����ļ����ݶ�ͼ  
//	imshow("src", src);
//	imshow("dst", dst);
//	waitKey(0);
//
//}
//
//void gradientgray(Mat &src, Mat &mag)
//{
//	const int h = src.rows, w = src.cols;
//	Mat ix(h, w, CV_32S), iy(h, w, CV_32S);
//	//��Ϊ��������ݶ�ֵ���������и�����ֵҲ���ܻ�ܴ󣬹���������Ϊ����  
//
//	// ��ˮƽ�����ݶȣ��������ұ�Ե����  
//	for (int y = 0; y < h; y++) {
//		ix.at<int>(y, 0) = abs(src.at<byte>(y, 1) - src.at<byte>(y, 0)) * 2;
//		for (int x = 1; x < w - 1; x++)
//			ix.at<int>(y, x) = abs(src.at<byte>(y, x + 1) - src.at<byte>(y, x - 1));
//		ix.at<int>(y, w - 1) = abs(src.at<byte>(y, w - 1) - src.at<byte>(y, w - 2)) * 2;
//	}
//	// ��ֱ�����ݶȣ��������ұ�Ե����  
//	for (int x = 0; x < w; x++) {
//		iy.at<int>(0, x) = abs(src.at<byte>(1, x) - src.at<byte>(0, x)) * 2;
//		for (int y = 1; y < h - 1; y++)
//			iy.at<int>(y, x) = abs(src.at<byte>(y + 1, x) - src.at<byte>(y - 1, x));
//		iy.at<int>(h - 1, x) = abs(src.at<byte>(h - 1, x) - src.at<byte>(h - 2, x)) * 2;
//	}
//	/*for (int j = 0; j < h; j++)
//	for (int k = 0; k < w; k++)
//	{
//	mag.at<byte>(j, k) = min(ix.at<int>(j,k) + iy.at<int>(j, k), 255);
//	}*/
//	convertScaleAbs(min(ix + iy, 255), mag); //��仰�������forѭ����ͬ���Ĺ���  
//}



//#include <cv.h>   
//#include <highgui.h>     
//#include <string>   
//#include <iostream>   
//#include <algorithm>   
//#include <iterator>  
//
//#include <stdio.h>  
//#include <string.h>  
//#include <ctype.h>  
//#include <opencv2\opencv.hpp>  
//
//using namespace cv;
//using namespace std;
//
//void help()
//{
//	printf(
//		"\nDemonstrate the use of the HoG descriptor using\n"
//		"  HOGDescriptor::hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());\n"
//		"Usage:\n"
//		"./peopledetect (<image_filename> | <image_list>.txt)\n\n");
//}
//
//int main(int argc, char** argv)
//{
//
//	Mat img;
//	FILE* f = 0;
//	char _filename[1024];
//	char * photopath = "C:\\Users\\liuxiaolong\\Desktop\\design\\photo\\singlep1.jpg";
//
//
//	img = imread(photopath);
//
//	if (img.data)
//	{
//		strcpy(_filename, photopath);
//	}
//	else
//	{
//		f = fopen(photopath, "rt");
//		if (!f)
//		{
//			fprintf(stderr, "ERROR: the specified file could not be loaded\n");
//			return -1;
//		}
//	}
//
//	HOGDescriptor hog;
//	//hog.load("D:\my_detector.yml");
//	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());//�õ������Ĭ��hog����64*128  
//																  //hog.setSVMDetector(HOGDescriptor::getDaimlerPeopleDetector());//����ʹ��Ĭ��ֵ
//																  //HOGDescriptor::getD aimlerPeopleDetector
//	namedWindow("people detector", 1);
//
//	for (;;)
//	{
//		char* filename = _filename;
//		if (f)
//		{
//			if (!fgets(filename, (int)sizeof(_filename) - 2, f))//��ȡһ������
//				break;
//			//while(*filename && isspace(*filename))  
//			//  ++filename;  
//			if (filename[0] == '#')
//				continue;
//			int l = strlen(filename);
//			while (l > 0 && isspace(filename[l - 1]))//�ж��Ƿ�Ϊ�ո�
//				--l;
//			filename[l] = '\0';
//			img = imread(filename);
//		}
//		printf("%s:\n", filename);
//		if (!img.data)
//			continue;
//
//		fflush(stdout);//ǿ�ƽ�������������д�뵽stdout
//		vector<Rect> found, found_filtered;
//		double t = (double)getTickCount();//����ʱ����
//		hog.detectMultiScale(img, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);
//		t = (double)getTickCount() - t;
//		printf("tdetection time = %gms\n", t*1000. / cv::getTickFrequency());
//		size_t i, j;
//		for (i = 0; i < found.size(); i++)
//		{
//			Rect r = found[i];
//			for (j = 0; j < found.size(); j++)
//				if (j != i && (r & found[j]) == r)
//					break;
//			if (j == found.size())
//				found_filtered.push_back(r);//�൱��ѹջ
//		}
//		for (i = 0; i < found_filtered.size(); i++)
//		{
//			Rect r = found_filtered[i];
//			// the HOG detector returns slightly larger rectangles than the real objects.  
//			// so we slightly shrink the rectangles to get a nicer output.  
//			r.x += cvRound(r.width*0.1);//��double�͵���ֵ�����������룬��������int
//			r.width = cvRound(r.width*0.8);
//			r.y += cvRound(r.height*0.07);
//			r.height = cvRound(r.height*0.8);
//			rectangle(img, r.tl(), r.br(), cv::Scalar(0, 255, 0), 3);//������⵽������
//		}
//		imshow("people detector", img);
//		int c = waitKey(0) & 255;
//		if (c == 'q' || c == 'Q' || !f)
//			break;
//	}
//	if (f)
//		fclose(f);
//	return 0;
//}















//#include <iostream>    
//#include <fstream>    
//#include <stdlib.h> //srand()��rand()����    
//#include <time.h> //time()����    
//#include <opencv2/core/core.hpp>    
//#include <opencv2/highgui/highgui.hpp>    
//#include <opencv2/imgproc/imgproc.hpp>    
//#include <opencv2/objdetect/objdetect.hpp>    
//#include <opencv2/ml/ml.hpp>    
//
//using namespace std;
//using namespace cv;
//
//int CropImageCount = 0; //�ü������ĸ�����ͼƬ����    
//
//int main()
//{
//	Mat src;
//	string ImgName;
//
//	char saveName[256];//�ü������ĸ�����ͼƬ�ļ���    
//	ifstream fin("F:/INRIAPerson/Train/neg.lst");//��ԭʼ������ͼƬ�ļ��б�    
//
//												 //һ��һ�ж�ȡ�ļ��б�    
//	while (getline(fin, ImgName))
//	{
//		cout << "����" << ImgName << endl;
//		ImgName = "F:/INRIAPerson/" + ImgName;
//
//		src = imread(ImgName, 1);//��ȡͼƬ    
//								 //cout<<"��"<<src.cols<<"���ߣ�"<<src.rows<<endl;    
//
//								 //ͼƬ��СӦ���������ٰ���һ��64*128�Ĵ���    
//		if (src.cols >= 64 && src.rows >= 128)
//		{
//			srand(time(NULL));//�������������  time(NULL)��ʾ��ǰϵͳʱ��  
//
//							  //��ÿ��ͼƬ���������10��64*128��С�Ĳ������˵ĸ�����    
//			for (int i = 0; i<10; i++)
//			{
//				int x = (rand() % (src.cols - 64)); //���Ͻ�x����    
//				int y = (rand() % (src.rows - 128)); //���Ͻ�y����    
//													 //cout<<x<<","<<y<<endl;    
//				Mat imgROI = src(Rect(x, y, 64, 128));
//				sprintf(saveName, "D:\\negphoto\\noperson%06d.jpg", ++CropImageCount);//���ɲü����ĸ�����ͼƬ���ļ���    
//				imwrite(saveName, imgROI);//�����ļ�    
//			}
//		}
//	}
//
//	system("pause");
//}











