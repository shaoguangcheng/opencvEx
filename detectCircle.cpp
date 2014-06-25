//opencv headers

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

//c++ headers
#include <iostream>

using namespace std;
using namespace cv;

int DOG(const Mat src,Mat &dst)
{
    if(src.empty())return -1;

    Mat img_1,img_2;

    GaussianBlur(src,img_1,Size(3,3),0,0);
 //   cv::GaussianBlur(src,img_2,cv::Size(3,3),0,0);
    GaussianBlur(img_1,img_2,Size(3,3),0,0);

    dst = img_1 - img_2;
    normalize(dst,dst,255,0,CV_MINMAX);

    return 0;
}

void detectCircle(Mat &img)
{
    if(img.data == NULL){
        cout << "error image date" << endl;
        return;
    }

    imshow("original image",img);

    Mat img_gray,img_binary,img_erode,img_dilate,img_gray_1,img_gauss,img_binary_1,img_erode_1,result;

    DOG(img,img_gauss);
    imshow("gauss image",img_gauss);

    cvtColor(img_gauss,img_gray,CV_RGB2GRAY);
    imshow("gray image",img_gray);

    int ddepth = CV_8UC1;
    img_gray.convertTo(img_gray,ddepth,1.5);
    imshow("inc contrast",img_gray);

//    Laplacian(img_gray,img_lap,ddepth);
//    imshow("lap image",img_lap);

    threshold(img_gray,img_binary,150,255,CV_THRESH_OTSU);
    imshow("binary image",img_binary);

    int erosionSize = 2;
    Mat element = getStructuringElement(MORPH_RECT,Size(erosionSize,erosionSize));
    dilate(img_binary,img_dilate,element);
    imshow("image erode",img_dilate);

    erode(img_dilate,img_erode,element);
    imshow("image dilate",img_erode);

    cvtColor(img,img_gray_1,CV_RGB2GRAY);
    threshold(img_gray_1,img_binary_1,150,255,CV_THRESH_BINARY);
    imshow("binary image 1",img_binary_1);

    Mat element_1 = getStructuringElement(MORPH_RECT,Size(erosionSize*8,erosionSize*2));
    erode(img_binary_1,img_erode_1,element_1);
    imshow("image dilate 1",img_erode_1);

    addWeighted(img_erode,1,img_erode_1,1,0,result);
    imshow("circle",result);

    waitKey();
}

int main(int argc, char* argv[])
{
    system("clear");
    if(argc < 2){
        cout << "program : image" << endl;
        return -1;
    }

    Mat image = imread(string(argv[1]));
    detectCircle(image);


    return 0;
}

