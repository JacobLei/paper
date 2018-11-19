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

// 在原图上画椭圆
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
	 
//	for (int i = 0; i < regions.size(); i++) 	// 在原图上用椭圆绘制组块（也可以在灰度图上绘制）[这个是库方法]
//		ellipse(image, fitEllipse(regions[i]), Scalar(255));

	for(int i=0; i <regions.size(); i++){		// 这个是调用的自己写的函数
		paintEllipse(regions[i], image);
	}
	
    imshow("image", image);

    waitKey(0);
    
    string retName =  "./resPic/res" + getCurrentDate() + ".bmp";
    cout << retName << endl;
    imwrite(retName, image);
    return 0;
}
