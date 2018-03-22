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
//	Mat src = imread("c:\\users\\liuxiaolong\\desktop\\design\\photo\\singlep3.jpg", 0);//以灰度形式读取图片  
//	Mat dst;
//	gradientgray(src, dst);
//	imwrite("c:\\users\\liuxiaolong\\desktop\\design\\photo\\result\\singlep3.jpg", dst); //保存文件，梯度图  
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
//	//因为计算出的梯度值可能有正有负，且值也可能会很大，故数据类型为整形  
//
//	// 求水平方向梯度，处理左右边缘像素  
//	for (int y = 0; y < h; y++) {
//		ix.at<int>(y, 0) = abs(src.at<byte>(y, 1) - src.at<byte>(y, 0)) * 2;
//		for (int x = 1; x < w - 1; x++)
//			ix.at<int>(y, x) = abs(src.at<byte>(y, x + 1) - src.at<byte>(y, x - 1));
//		ix.at<int>(y, w - 1) = abs(src.at<byte>(y, w - 1) - src.at<byte>(y, w - 2)) * 2;
//	}
//	// 求垂直方向梯度，处理左右边缘像素  
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
//	convertScaleAbs(min(ix + iy, 255), mag); //这句话和上面的for循环是同样的功能  
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
//	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());//得到检测器默认hog像素64*128  
//																  //hog.setSVMDetector(HOGDescriptor::getDaimlerPeopleDetector());//当不使用默认值
//																  //HOGDescriptor::getD aimlerPeopleDetector
//	namedWindow("people detector", 1);
//
//	for (;;)
//	{
//		char* filename = _filename;
//		if (f)
//		{
//			if (!fgets(filename, (int)sizeof(_filename) - 2, f))//读取一行数据
//				break;
//			//while(*filename && isspace(*filename))  
//			//  ++filename;  
//			if (filename[0] == '#')
//				continue;
//			int l = strlen(filename);
//			while (l > 0 && isspace(filename[l - 1]))//判断是否为空格
//				--l;
//			filename[l] = '\0';
//			img = imread(filename);
//		}
//		printf("%s:\n", filename);
//		if (!img.data)
//			continue;
//
//		fflush(stdout);//强制将缓冲区的数据写入到stdout
//		vector<Rect> found, found_filtered;
//		double t = (double)getTickCount();//计算时间用
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
//				found_filtered.push_back(r);//相当于压栈
//		}
//		for (i = 0; i < found_filtered.size(); i++)
//		{
//			Rect r = found_filtered[i];
//			// the HOG detector returns slightly larger rectangles than the real objects.  
//			// so we slightly shrink the rectangles to get a nicer output.  
//			r.x += cvRound(r.width*0.1);//对double型的数值进行四舍五入，返回整数int
//			r.width = cvRound(r.width*0.8);
//			r.y += cvRound(r.height*0.07);
//			r.height = cvRound(r.height*0.8);
//			rectangle(img, r.tl(), r.br(), cv::Scalar(0, 255, 0), 3);//画出检测到的行人
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
//#include <stdlib.h> //srand()和rand()函数    
//#include <time.h> //time()函数    
//#include <opencv2/core/core.hpp>    
//#include <opencv2/highgui/highgui.hpp>    
//#include <opencv2/imgproc/imgproc.hpp>    
//#include <opencv2/objdetect/objdetect.hpp>    
//#include <opencv2/ml/ml.hpp>    
//
//using namespace std;
//using namespace cv;
//
//int CropImageCount = 0; //裁剪出来的负样本图片个数    
//
//int main()
//{
//	Mat src;
//	string ImgName;
//
//	char saveName[256];//裁剪出来的负样本图片文件名    
//	ifstream fin("F:/INRIAPerson/Train/neg.lst");//打开原始负样本图片文件列表    
//
//												 //一行一行读取文件列表    
//	while (getline(fin, ImgName))
//	{
//		cout << "处理：" << ImgName << endl;
//		ImgName = "F:/INRIAPerson/" + ImgName;
//
//		src = imread(ImgName, 1);//读取图片    
//								 //cout<<"宽："<<src.cols<<"，高："<<src.rows<<endl;    
//
//								 //图片大小应该能能至少包含一个64*128的窗口    
//		if (src.cols >= 64 && src.rows >= 128)
//		{
//			srand(time(NULL));//设置随机数种子  time(NULL)表示当前系统时间  
//
//							  //从每张图片中随机采样10个64*128大小的不包含人的负样本    
//			for (int i = 0; i<10; i++)
//			{
//				int x = (rand() % (src.cols - 64)); //左上角x坐标    
//				int y = (rand() % (src.rows - 128)); //左上角y坐标    
//													 //cout<<x<<","<<y<<endl;    
//				Mat imgROI = src(Rect(x, y, 64, 128));
//				sprintf(saveName, "D:\\negphoto\\noperson%06d.jpg", ++CropImageCount);//生成裁剪出的负样本图片的文件名    
//				imwrite(saveName, imgROI);//保存文件    
//			}
//		}
//	}
//
//	system("pause");
//}











