#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

//获取仿射变换的图形， 分别为原图，起始角度，结束角度和步长
bool getAffinePicture(string srcPath, int start, int end, int step) {
    double coefficient  = 0.01;
    Mat src = imread(srcPath);
    if (!src.data) {
        cout << "no image " << endl;
        return false;
    }
    int nRow = src.rows;
    int nCol = src.cols;

    //定义仿射变换的二维点数组
    Point2f srcPoint[3], resPoint[3];
    srcPoint[0] = Point2f(0, 0);
    srcPoint[1] = Point2f(nCol - 1, 0);
    srcPoint[2] = Point2f(0, nRow - 1);

    for (int i = start; i <= end; i = i + step) {
        resPoint[0] = Point2f(0, nRow*0.5);
        resPoint[1] = Point2f(nCol*0.85, nRow*coefficient*i);
        resPoint[2] = Point2f(nCol*0.15, nRow*(1 - coefficient*i));
        Mat warp(Size(2, 3), CV_32F);
        Mat result = Mat::zeros(nRow, nCol, src.type());
        warp = getAffineTransform(srcPoint, resPoint);
        warpAffine(src, result, warp, result.size());
        ostringstream osstr;
        cout << " save image" << endl;
        string savePath = "./resPic/LenaRGB_" + to_string(i) + " .bmp";
        osstr << savePath;
        imwrite(osstr.str(), result);
    }
    return true;
}


int main (int argc, char **argv)
{
    Mat image, image_gray;
    string picName = "./srcPic/LenaRGB.bmp";
    getAffinePicture(picName, 5, 20, 5);

    image = imread(picName, CV_LOAD_IMAGE_COLOR );

    return 0;
}
