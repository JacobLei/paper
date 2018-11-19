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

// 傅里叶变换
void DFT(Mat srcImage, Mat &dftResultImage) {
	// 将输入图像扩大至最佳尺寸
	int nRows = getOptimalDFTSize(srcImage.rows);
	int nCols = getOptimalDFTSize(srcImage.cols);
	Mat resultImage;
	// 以最佳尺寸扩充原图，把原图放在左上角，其余部分填充0
	copyMakeBorder(srcImage, resultImage, 0, nRows - srcImage.rows, 0, nCols - srcImage.cols, BORDER_CONSTANT, Scalar::all(0));
	// 为傅里叶变换的结果(实部和虚部)分配空间
	Mat planes[] = {Mat_<float>(resultImage),Mat::zeros(resultImage.size(), CV_32F)};
	Mat completeI;
	// 为扩充后的图像增加一个0通道
	merge(planes, 2, completeI);
	// 进行离散傅里叶变换
	dft(completeI, completeI);
	// 将复数转化为幅度
	split(completeI, planes);
	magnitude(planes[0], planes[1], planes[0]);
	dftResultImage = planes[0];


	// 对数尺度缩放
	dftResultImage += 1;
	log(dftResultImage, dftResultImage);
	
	// 剪切和充分布幅度象限
	dftResultImage = dftResultImage(Rect(0, 0, srcImage.cols, srcImage.rows));
	// 归一化图像
	normalize(dftResultImage, dftResultImage,0,1,CV_MINMAX);
	
	int cx = dftResultImage.cols / 2;
	int cy = dftResultImage.rows / 2;
	Mat tmp;
	// top-left 为每个象限创建roi
	Mat q0(dftResultImage, Rect(0, 0, cx, cy));
	// top-right
	Mat q1(dftResultImage, Rect(cx, 0, cx, cy));
	// bottom-left
	Mat q2(dftResultImage, Rect(0, cy, cx, cy));
	// bottom-right
	Mat q3(dftResultImage, Rect(cx, cy, cx, cy));
	// 交换象限
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);
	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);
}

// 获取一个区域的MSA特征向量
vector<long double> getMSAFeature(Mat gray, vector<Point> region){
	double as[19] = {-1, -1, -1, -1, -1, -0.75, -0.75, -0.75, -0.75, -0.5, -0.5, -0.5, -0.5, -0.25, -0.25, -0.25, 0, 0, 0.25};
 	double bs[19] = {1, 0.75, 0.5, 0.25, 0, 0.75, 0.5, 0.25, 0, 0.75, 0.5, 0.25, 0, 0.5, 0.25, 0, 0.5, 0.25, 0.25};
	vector<long double> des;
		// 拟合椭圆
		RotatedRect rotatedRect = fitEllipse(region);
//		if (rotatedRect.size.height * rotatedRect.size.width > 50) {
			Mat innerResult;
			getEllipsePic(gray, rotatedRect, innerResult);
			// 对原图进行傅里叶变换
			Mat dftResult;
			DFT(innerResult, dftResult);
			
//			for (int j = 0; j < 19; j++) {
//				double alpha = *(as + j);
//				double beta = *(bs + j);
				double alpha = -1;
				double beta = 1;
				double gama = 1 - alpha - beta;
				Size ksize(5, 5);
				Mat ablurMat,bblurMat,gblurMat;
				GaussianBlur(innerResult, ablurMat, ksize, alpha);
				GaussianBlur(innerResult, bblurMat, ksize, beta);
				GaussianBlur(innerResult, gblurMat, ksize, gama);
				Mat adftResult, bdftResult, gdftResult;
				DFT(ablurMat, adftResult);
				DFT(bblurMat, bdftResult);
				DFT(gblurMat, gdftResult);
				Mat f(ablurMat.rows, ablurMat.cols, CV_32FC1, Scalar(0));	// 傅里叶逆变换是alpha、beta、gama变换后的数乘，这里定义的是需要逆变换的Mat矩阵
				
				for (int r = 0; r < ablurMat.rows; r++) {
					for (int c = 0; c < ablurMat.cols; c++) {
						f.at<uchar>(r, c) = adftResult.at<uchar>(r, c) * bdftResult.at<uchar>(r, c) * gdftResult.at<uchar>(r, c);
					}
				}
				// 傅里叶逆变换
				Mat F(innerResult.rows, innerResult.cols, CV_32FC1, Scalar(0));  
     			idft(f, F, DFT_REAL_OUTPUT);  
				// 计算总的乘积和
				long double sum = 0;
				long double value = 0;
				long double temp = dftResult.at<uchar>(dftResult.rows / 2, dftResult.cols / 2); // (0, 0)点
				for (int r = 0; r < dftResult.rows; r++) {
					for (int c = 0; c < dftResult.cols; c++) {
						long double agray = adftResult.at<uchar>(r, c) / temp;
						long double bgray = bdftResult.at<uchar>(r, c) / temp;
						long double ggray = gdftResult.at<uchar>(r, c) / temp;
						long double Fgray = (long double)F.at<uchar>(r, c);
						long double p = agray * bgray * ggray * Fgray;													// 公式　５.23
					cout << "Fgray = " << Fgray << "  agray = " << agray << "  bgray = " << bgray << "  ggray = " << ggray  << " p = " << p << endl;
//						long double 
					}
				}
				value = sum / (dftResult.rows * dftResult.cols);
				//　排除异常值
				if(isnan(value) || isinf(value)){
					return des;
				}
				des.push_back(value);
//				cout << "des = " << value << " temp = " << temp << "   sum = " << sum << endl;
//			}
//		}
		return des;
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

    vector<vector<long double>>  descpters;

    for(int i=0; i<1; ++i){
    	vector<long double> des = getMSAFeature(grayImage, regions[i]);
    	
    }
	
	
    // 将数据写到csv中
/*    string retNameCsv = "./res/overlapping" + getCurrentDate() + ".csv";
    ofstream outFile;
    outFile.open(retNameCsv, ios::out);

    for(int i=0; i<descpters.size(); ++i){
    	outFile << "The num." << i+1 << " descpters is : ";
    	for(int j=0; j<descpters[i].size(); ++j){
    	 outFile << ',' << descpters[i][j] ;
    	}

    	outFile << endl;
   }
    outFile.close();
*/  
    return 0;
}
