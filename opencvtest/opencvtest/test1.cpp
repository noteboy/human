#include <cv.h>   
#include <highgui.h>     
#include <string>   
#include <iostream>   
#include <algorithm>   
#include <iterator>  

#include <stdio.h>  
#include <string.h>  
#include <ctype.h>  
#include <opencv2\opencv.hpp>  

using namespace cv;
using namespace std;

void help()
{
	printf(
		"\nDemonstrate the use of the HoG descriptor using\n"
		"  HOGDescriptor::hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());\n"
		"Usage:\n"
		"./peopledetect (<image_filename> | <image_list>.txt)\n\n");
}

int main(int argc, char** argv)
{
	
	Mat img;
	FILE* f = 0;
	char _filename[1024];
	char * photopath = "C:\\Users\\liuxiaolong\\Desktop\\design\\photo\\singlep3.jpg";


	img = imread(photopath);

	if (img.data)
	{
		strcpy(_filename, photopath);
	}
	else
	{
		f = fopen(photopath, "rt");
		if (!f)
		{
			fprintf(stderr, "ERROR: the specified file could not be loaded\n");
			return -1;
		}
	}

	HOGDescriptor hog;
	//hog.load("D:\my_detector.yml");
	hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());//得到检测器默认hog像素64*128  
	//hog.setSVMDetector(HOGDescriptor::getDaimlerPeopleDetector());//当不使用默认值
	//HOGDescriptor::getD aimlerPeopleDetector
	namedWindow("people detector", 1);

	for (;;)
	{
		char* filename = _filename;
		if (f)
		{
			if (!fgets(filename, (int)sizeof(_filename) - 2, f))//读取一行数据
				break;
			//while(*filename && isspace(*filename))  
			//  ++filename;  
			if (filename[0] == '#')
				continue;
			int l = strlen(filename);
			while (l > 0 && isspace(filename[l - 1]))//判断是否为空格
				--l;
			filename[l] = '\0';
			img = imread(filename);
		}
		printf("%s:\n", filename);
		if (!img.data)
			continue;

		fflush(stdout);//强制将缓冲区的数据写入到stdout
		vector<Rect> found, found_filtered;
		double t = (double)getTickCount();//计算时间用
		hog.detectMultiScale(img, found, 0, Size(8, 8), Size(32, 32), 1.05, 2);
		t = (double)getTickCount() - t;
		printf("tdetection time = %gms\n", t*1000. / cv::getTickFrequency());
		size_t i, j;
		for (i = 0; i < found.size(); i++)
		{
			Rect r = found[i];
			for (j = 0; j < found.size(); j++)
				if (j != i && (r & found[j]) == r)
					break;
			if (j == found.size())
				found_filtered.push_back(r);//相当于压栈
		}
		for (i = 0; i < found_filtered.size(); i++)
		{
			Rect r = found_filtered[i];
			// the HOG detector returns slightly larger rectangles than the real objects.  
			// so we slightly shrink the rectangles to get a nicer output.  
			r.x += cvRound(r.width*0.1);//对double型的数值进行四舍五入，返回整数int
			r.width = cvRound(r.width*0.8);
			r.y += cvRound(r.height*0.07);
			r.height = cvRound(r.height*0.8);
			rectangle(img, r.tl(), r.br(), cv::Scalar(0, 255, 0), 3);//画出检测到的行人

		    ////////把行人从背景图中分离出来
			Mat imageROI = img(Rect(r.tl().x, r.tl().y, (r.br().x - r.tl().x), (r.br().y - r.tl().y)));///截取检测到的行人，并存储在Mat中
			Rect grabrect;
			grabrect.x = 10;
			grabrect.y = 10;
			grabrect.width = imageROI.cols-20;
			grabrect.height = imageROI.rows-20;
			Mat result; // 分割结果 (4种可能取值)  
			Mat bgModel, fgModel; // 模型(内部使用)  
			grabCut(imageROI, result, grabrect, bgModel, fgModel, 2, GC_INIT_WITH_RECT);
			// 得到可能为前景的像素  
			compare(result, GC_PR_FGD, result, CMP_EQ);
			//imshow("ss", result);
			// 生成输出图像  
			Mat foreground(imageROI.size(), CV_8UC3, cv::Scalar(0, 0, 0));
			imageROI.copyTo(foreground, result); // 不复制背景数据  
			imshow("将行人从背景分离", foreground);
			//imshow("after", imageROI);//显示截取图片部分
			///waitKey(0);
			/////////////////////////////把行人从背景图中分离出来


			/////////////行人上下半身分割
			Mat upbody = foreground(Rect(0,0, foreground.cols-1,(foreground.rows+1)/2-1));
			Mat downbody = foreground(Rect(0,(foreground.rows+1)/2-1, foreground.cols-1, (foreground.rows + 1)/2-1));
			imshow("upbody", upbody);//展示上半身
			imshow("downbody", downbody);//展示下半身
			/////////////行人上下半身分割
			
			/////////////将上下半身分割，获取每个单元的RGB，取其均值，进行颜色分类
			////上半身
			//cvtColor(result, result, COLOR_HSV2RGB);
			double upr = 0, upg = 0, upb = 0;
			double upflag = 0;
			for(int i = 0;i < upbody.rows; i++)
				for (int j = 0; j < upbody.cols; j++) {
					if ((upbody.at<Vec3b>(i, j)[0] != 0) && (upbody.at<Vec3b>(i, j)[1] != 0) && (upbody.at<Vec3b>(i, j)[2] != 0)) {
						upr = upr + upbody.at<Vec3b>(i, j)[0];
						upg = upg + upbody.at<Vec3b>(i, j)[1];
						upb = upb + upbody.at<Vec3b>(i, j)[2];
						upflag++;
					}
				}
			upr = upr / upflag;
			upg = upg / upflag;
			upb = upb / upflag;
			for (int i = 0; i < upbody.rows; i++)
				for (int j = 0; j < upbody.cols; j++) {
					if ((upbody.at<Vec3b>(i, j)[0] != 0) && (upbody.at<Vec3b>(i, j)[1] != 0) && (upbody.at<Vec3b>(i, j)[2] != 0)) {
						upbody.at<Vec3b>(i, j)[0] = upr;
						upbody.at<Vec3b>(i, j)[1] = upg;
						upbody.at<Vec3b>(i, j)[2] = upb;
					}
				}
			imshow("dup",upbody);
			/////下半身
			int downr = 0, downg = 0, downb = 0;
			int downflag = 0;
			for (int i = 0; i < downbody.rows; i++)
				for (int j = 0; j < downbody.cols; j++) {
					if ((downbody.at<Vec3b>(i, j)[0] != 0) && (downbody.at<Vec3b>(i, j)[1] != 0) && (downbody.at<Vec3b>(i, j)[2] != 0)) {
						downflag++;
						downr = downr + downbody.at<Vec3b>(i, j)[0];
						downg = downg + downbody.at<Vec3b>(i, j)[1];
						downb = downb + downbody.at<Vec3b>(i, j)[2];
					}
				}
			downr = downr / downflag;
			downg = downg / downflag;
			downb = downb / downflag;
			for (int i = 0; i < downbody.rows; i++)
				for (int j = 0; j < downbody.cols; j++) {
					if ((downbody.at<Vec3b>(i, j)[0] != 0) && (downbody.at<Vec3b>(i, j)[1] != 0) && (downbody.at<Vec3b>(i, j)[2] != 0)) {
						downbody.at<Vec3b>(i, j)[0] = downr;
						downbody.at<Vec3b>(i, j)[1] = downg;
						downbody.at<Vec3b>(i, j)[2] = downb;
					}
				}
			imshow("ddown", downbody);
			/////////////将上下半身分割，获取每个单元的RGB，取其均值，进行颜色分类



		}
		imshow("people detector", img);
		int c = waitKey(0) & 255;
		if (c == 'q' || c == 'Q' || !f)
			break;
	}
	if (f)
		fclose(f);
	return 0;
}