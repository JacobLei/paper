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

// 时间函数
string  getCurrentDate()
{
    time_t nowtime;  
    nowtime = time(NULL); //获取日历时间   
    char tmp[64];   
    strftime(tmp,sizeof(tmp),"%Y-%m-%d-%H-%M-%S",localtime(&nowtime));   
    return tmp;
}

// 用于计算每个椭圆的面积
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
	if (contours.size() >= 1)
		area = abs(contourArea(contours[0], true));
	return area;

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
			if(!(area>=-0.00001 && area<=0.00001))
				overlapSum++;
		} 
	}
	
	if(overlapSum == 0) return 0;		// 表示没有重叠区域
	return areaSum / overlapSum;
}

// 获取两个椭圆和一个矩形重叠的面积
double getE2llipseRectArea(Mat gray, vector<Point> region, vector<Point> region2, Rect rect) {
	double erArea1 = getEllipseRectArea(gray, region, rect);
	double erArea2 = getEllipseRectArea(gray, region2, rect);
	double e2Area = getE2llipseArea(gray, region, region2);
	double area = 0;
	if (erArea1 != 0 && erArea2 != 0 && e2Area != 0) {
		RotatedRect a = fitEllipse(region);
		RotatedRect b = fitEllipse(region2);
		Mat mask(gray.rows, gray.cols, CV_8UC1, Scalar(0));
		// 如果两个椭圆有重叠，则填充后重叠区域会变成一个，在统计这个联通的区域的面积
		ellipse(mask, a, Scalar(255), -1);
		ellipse(mask, b, Scalar(255), -1);
		rectangle(mask, rect.tl(), rect.br(), Scalar(255), -1);
		vector<vector<Point>> contours;
		vector<Vec4i> hierarchy;
		findContours(mask, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);
		if (contours.size() == 1) {
			double aerea = getContoursArea(gray, region);
			double berea = getContoursArea(gray, region2);
			area = erArea1 + erArea2 + e2Area + abs(contourArea(contours[0], true)) - aerea - berea - mlength*mlength;
		}
		return area;
	}
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
 	
	cout << getMeanOfOverlap(grayImage, regions) << endl;	

	
	
    // 将数据写到csv中
/*    string retNameCsv = "./res/overlapping" + getCurrentDate() + ".csv";
    ofstream outFile;
    outFile.open(retNameCsv, ios::out);
    for(int i=0; i<areas.size(); ++i){
    	outFile << "The num." << i+1 << " area is : " << ',' << areas[i] << endl;
    }
    outFile.close();
*/    
    return 0;
}
