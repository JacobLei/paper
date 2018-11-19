#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <iostream>
#include <string>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <cmath>

using namespace cv;
using namespace std;

#define MIN_VALUE 1e-8
#define IS_DOUBLE_ZERO(d)  (abs(d) < MIN_VALUE)

// 时间函数
string  getCurrentDate()
{
    time_t nowtime;  
    nowtime = time(NULL); //获取日历时间   
    char tmp[64];   
    strftime(tmp,sizeof(tmp),"%Y-%m-%d-%H-%M-%S",localtime(&nowtime));   
    return tmp;
}

/*
把椭圆数据单独提出来，保存到一个新的mat中去
原理是把区域轮廓在mask中用白线画出来，其余地方都是0，
检测轮廓并填充，然后给原图加一层马赛克只显示想要的部分
*/
void getEllipsePic(Mat gray, RotatedRect rect, Mat &result) {
	Mat mask(gray.rows, gray.cols, CV_8UC1, Scalar(0));
	ellipse(mask, rect, Scalar(255));
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(mask, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	drawContours(mask, contours, 0, Scalar(255), CV_FILLED, 8);
	Mat dstImg = gray.clone();
	Rect rectROI(0, 0, dstImg.cols, dstImg.rows);
	Mat srcROI = dstImg(rectROI);
	Mat dst1;
	srcROI.copyTo(dst1, mask);
	// 生成椭圆的外接矩形，并把外接矩形扣出来
	Rect rectROI2 = boundingRect(contours[0]);
	IplImage iplImage = dst1;
	cvSetImageROI(&iplImage, rectROI2);
	Mat iplPro(&iplImage, true);
	iplPro.copyTo(result);
	cvResetImageROI(&iplImage);
}

void paintEllipse(vector<Point> region, Mat img){
	// 拟合椭圆
	RotatedRect rotatedRect = fitEllipse(region);
	// 画椭圆
	ellipse(img, rotatedRect, Scalar(0, 255, 0));
	Point2f P[4];
	rotatedRect.points(P);
	for (int j = 0; j <= 3; j++)
	{
		// 画外接矩形的四条边
		line(img, P[j], P[(j + 1) % 4], Scalar(0, 0, 255), 1);
	}
	// 画质心
	circle(img, rotatedRect.center, 2, Scalar(0, 255, 0));
}

// 获取一个区域的熵
double  getEntropy(Mat gray, vector<Point> region){
	double entropy = 0.0;
	// 拟合椭圆
	RotatedRect rotatedRect = fitEllipse(region);
	Mat innerResult;
	getEllipsePic(gray, rotatedRect, innerResult);

	int pixel[256] = {0};		// ０-255像素的个数
	int sum = 0;					// 像素点总数
		
	for (int r = 0; r < innerResult.rows; r++) {
		for (int c = 0; c < innerResult.cols; c++) {
			int pex = innerResult.at<uchar>(r, c) % 256;
			pixel[pex]++;
			sum++;
		}
	}

	for(int i=0; i<256; ++i){
		double p = (double)pixel[i]/ (double)sum;
		double tmp =  0.0;
		if( !IS_DOUBLE_ZERO(p))
			tmp = (p) * (log(p) / log(2));
//		cout << "num " << i << " = " << pixel[i] << "  tmp = " << tmp << endl;
		entropy += tmp;
	}
	return entropy * (-1);
}

int main (int argc, char **argv)
{
    Mat image, grayImage;
    string picName = "./srcPic/LenaRGB.bmp";
    image = imread(picName, CV_LOAD_IMAGE_COLOR );
    if (!image.data) {
        cout << "No image data\n";
        return -1;
    }
         
    cvtColor(image, grayImage, CV_RGB2GRAY);	// 灰度化处理
    // 创建MSER类
 	MSER ms;
    vector<vector<Point> > regions;				// 用于组快区域的像素点集
 	ms(grayImage, regions, Mat());				// 计算MSER区域

    vector<long double>  entropys;

    for(int i=0; i<regions.size(); ++i){
    	entropys.push_back(getEntropy(grayImage, regions[i]));
    }

    // 将数据写到csv中
    string retNameCsv = "./res/entropys" + getCurrentDate() + ".csv";
    ofstream outFile;
    outFile.open(retNameCsv, ios::out);

    for(int i=0; i<entropys.size(); ++i){
    	outFile << "The num " << i+1 << " entropys = " << ',' << entropys[i] << endl;
    }
    outFile.close();

    return 0;
}
