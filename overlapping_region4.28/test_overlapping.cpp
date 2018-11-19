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

#define PI 3.1415926
#define NUM 0.0000001

// 时间函数
string  getCurrentDate()
{
    time_t nowtime;  
    nowtime = time(NULL); //获取日历时间   
    char tmp[64];   
    strftime(tmp,sizeof(tmp),"%Y-%m-%d-%H-%M-%S",localtime(&nowtime));   
    return tmp;
}

// 获取计算每个椭圆的面积
double getContoursArea(Mat gray, vector<Point> region) {
	// 拟合椭圆
	RotatedRect a = fitEllipse(region);
	Mat mask(gray.rows, gray.cols, CV_8UC1, Scalar(0));
	ellipse(mask, a, Scalar(255), -1);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(mask, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	//drawContours(mask, contours, -1, Scalar(255), CV_FILLED, 8);
	drawContours(mask, contours, -1, Scalar(255), 0, 8);
	double area = 0;
	if (contours.size() == 1)
		area = abs(contourArea(contours[0], true));
	return area;

}

// 获取椭圆width信息
double getWidth(Mat gray, vector<Point> region){
	RotatedRect a = fitEllipse(region);
	return a.size.width;
}

// 获取椭圆height信息
double getHeight(Mat gray, vector<Point> region){
	RotatedRect a = fitEllipse(region);
	return a.size.height;
}

// 获取椭圆的圆形度信息（计算公式：e=（4π*面积）/(周长*周长）
double getCircularity(Mat gray, vector<Point> region){
	RotatedRect a = fitEllipse(region);
	Mat mask(gray.rows, gray.cols, CV_8UC1, Scalar(0));
	ellipse(mask, a, Scalar(255), -1);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(mask, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	drawContours(mask, contours, -1, Scalar(255), 0, 8);
	double area = 0;
	double ret = 0;
	double length = 0;
	if (contours.size() >= 1){
		area = abs(contourArea(contours[0], true));
		length = abs(arcLength( contours[0], true ));
	}
	if( (length>=-NUM) && (length<=NUM))
		return 0; 
	return (4*PI*area) / (length*length);
	

}

//获取两个椭圆重叠的面积
double getE2llipseArea(Mat gray, vector<Point> region, vector<Point> region2) {
	double earea1 = getContoursArea(gray, region);
	double earea2 = getContoursArea(gray, region2);
	RotatedRect a = fitEllipse(region);
	RotatedRect b = fitEllipse(region2);
	Mat mask(gray.rows, gray.cols, CV_8UC1, Scalar(0));
	// 如果两个椭圆有重叠，则填充后重叠区域会变成一个，在统计这个联通的区域的面积
	ellipse(mask, a, Scalar(255), -1);
	ellipse(mask, b, Scalar(255), -1);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(mask, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
	double area = 0;
	if (contours.size() == 1) {
		area = earea1 + earea2 - abs(contourArea(contours[0], true));
	}
	return area;
}

// 获取整幅影像的特征区域重叠率均值（公式4.28 P79）
double getMeanOfOverlap(Mat gray, vector<vector<Point>> regions){
	int overlapSum = 0;
	double areaSum = 0.0;
	for(int i=0; i <regions.size()-1; i++){		// 计算重叠区域的面积
		double areaI = getContoursArea(gray, regions[i]);
		for(int j=i+1; j<regions.size(); ++j){
			double areaJ = getContoursArea(gray, regions[j]);
			double area = getE2llipseArea(gray, regions[i], regions[j]);
			double r = area / min(areaI, areaJ);		// 公式（4.27）
			areaSum += r;
			if(!(area>=-NUM && area<=NUM))
				overlapSum++;
		} 
	}
	
	if(overlapSum == 0) return 0;		// 表示没有重叠区域
	return areaSum / overlapSum;
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
	
    // 将数据写到csv中
    string retNameCsv = "./res/picInfo" + getCurrentDate() + ".csv";
    ofstream outFile;
    outFile.open(retNameCsv, ios::out | ios::trunc);
    outFile << "num" << ',' << "area" << ',' << "Width" << ',' 
    		<< "Height" << ',' << "w/h" << ',' << "E" << endl;
    cout << "计算中，请等待..." << endl;
    for(int i=0; i<regions.size(); ++i){
    	double ares = getContoursArea(grayImage, regions[i]);
    	double w = getWidth(grayImage, regions[i]);
		double h = getHeight(grayImage, regions[i]);
		double wH = w/h;
		double e = getCircularity(grayImage, regions[i]);
    	outFile << i+1 << ',' << ares << ',' << w << ',' 
    		<< h << ',' << wH << ',' << e << endl;
    }
    
    outFile << "R_ALL" << ',' << getMeanOfOverlap(grayImage, regions) << endl;
    outFile.close();
   	cout << "计算完成...." << endl;
    return 0;
}
