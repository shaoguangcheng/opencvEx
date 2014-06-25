//opencv headers
#include "opencv/highgui.h"
#include "opencv/cv.h"
#include "opencv2/imgproc/imgproc.hpp"

//c++ standard headers
#include <iostream>
#include <algorithm>

//c headers
#include <stdlib.h>

//my useful headers
#include "misc/directory.h"
#include "misc/timeProcess.h"
#include "misc/_io_.h"

using namespace cv;
using namespace std;

//////////////////////////////////////////////////////////////

// show how to access data in Mat format (I demonstrate four ways here)
void lookupPtr(Mat &img, const uchar* const table)
{
    CV_Assert(img.depth() != sizeof(uchar));

    int nRow = img.rows;
    int nCol = img.cols;
    int nChannel = img.channels();

    if(3 == nChannel)
        nCol *= 3;
    if(img.isContinuous()){
        nCol *= nRow;
        nRow = 1;
    }

    uchar* ptr;
    for(int i=0;i<nRow;i++){
        ptr = img.ptr<uchar>(i);
        for(int j=0;j<nCol;j++)
            ptr[j] = table[ptr[j]];
    }
}

void lookupIterator(Mat &img,const uchar* const table)
{
    CV_Assert(img.depth() != sizeof(uchar));

    int nChannel = img.channels();

    switch(nChannel){
        case 1:{
        MatIterator_<uchar> it,start = img.begin<uchar>(),end = img.end<uchar>();
            for(it = start;it!=end;++it)
                *it = table[*it];
            break;
        }
        case 3:{
        MatIterator_<Vec3b> it,start = img.begin<Vec3b>(),end = img.end<Vec3b>();
            for(it = start;it != end;++it){
                (*it)[0] = table[(*it)[0]];
                (*it)[1] = table[(*it)[1]];
                (*it)[2] = table[(*it)[2]];
            }
            break;
        }
        default :
            break;
    }
}

void lookupAt(Mat &img, const uchar* const table)
{
    CV_Assert(img.depth() != sizeof(uchar));

    int nRow = img.rows;
    int nCol = img.cols;
    int nChannel = img.channels();

    if(1 == nChannel){
        for(int i=0;i<nRow;i++)
            for(int j=0;j<nCol;j++)
                img.at<uchar>(i,j) = table[img.at<uchar>(i,j)];
    }
    else if(3 == nChannel){
        Vec3b ptr;
        for(int i=0;i<nRow;i++)
            for(int j=0;j<nCol;j++){
                ptr = img.at<Vec3b>(i,j);
                img.at<Vec3b>(i,j)[0] = table[ptr[0]];
                img.at<Vec3b>(i,j)[1] = table[ptr[1]];
                img.at<Vec3b>(i,j)[2] = table[ptr[2]];
            }
    }
}

void lookupArray(Mat &img, const uchar* const table)
{
    CV_Assert(img.depth() != sizeof(uchar));

    int nRow = img.rows;
    int nCol = img.cols*img.channels();

    uchar* ptr = (uchar*)img.data;
    for(int i=0;i<nRow;i++)
        for(int j=0;j<nCol;j++)
            ptr[i*nCol+j] = table[(int)ptr[i*nCol+j]];

}

void useLUT(Mat& img,const  uchar* const table)
{
    Mat table_(1,256,CV_8UC1);

    uchar* ptr = (uchar*)table_.data;
    for(int i=0;i<256;i++)
        ptr[i] = table[i];

    LUT(img,table_,img);
}

void createLookupTable(uchar* table, const int len)
{
    for(int i=0;i<len;i++)
        table[i] = (i%10)*10;
}

void testAccessSpeed(const string &fileName)
{
    Mat img = imread(fileName,IMREAD_COLOR);
    imshow("origin",img);

    Mat ptrM = img.clone();
    Mat iteratorM = img.clone();
    Mat atM = img.clone();
    Mat arrayM = img.clone();
    Mat lutM = img.clone();

    uchar* table = new uchar [256];
    createLookupTable(table,256);

    double t1 = getTickCount();
    lookupPtr(ptrM,table);
    double t2 = getTickCount();

    lookupIterator(iteratorM,table);
    double t3 = getTickCount();

    lookupAt(atM,table);
    double t4 = getTickCount();

    lookupArray(arrayM,table);
    double t5 = getTickCount();

    useLUT(lutM,table);
    double t6 = getTickCount();

    imshow("ptr",ptrM);
    imshow("at",atM);
    imshow("iterator",iteratorM);
    imshow("array",arrayM);
    imshow("LUT",lutM);
    waitKey();

    const char *cmd = "clear";
    if(0 != system(cmd))
        cerr << "system clear error" << endl;

    cout << "    time consuming comparation    " << endl;
    cout << "--------------------------------------" << endl;
    cout << "     ptr : " << (t2-t1)/getTickFrequency() << endl;
    cout << "      at : " << (t3-t2)/getTickFrequency() << endl;
    cout << "iterator : " << (t4-t3)/getTickFrequency() << endl;
    cout << "   array : " << (t5-t4)/getTickFrequency() << endl;
    cout << "     LUT : " << (t6-t5)/getTickFrequency() << endl;
    cout << "---------------------------------------" << endl;
}

////////////////////////////////////////////////////////////////////////////////////////
// constrast enhancement
// the first version is implement by myself
// the second version is using opencv API to implement this function
// the result shows because the opencv API functions used multi-thread technology. the much faster speed it reached,

void userSharpen(Mat img, const char* const kernel, Mat &dest) //contrast enhancement
{
    CV_Assert(img.depth() != sizeof(uchar));

    int nRow = img.rows;
    int nCol = img.cols;
    int nChannel = img.channels();

    dest.create(nRow,nCol,img.type());

    for(int i=1;i<nRow-1;++i){
        uchar* pPre = img.ptr<uchar>(i-1);
        uchar* pCur = img.ptr<uchar>(i);
        uchar* pNext = img.ptr<uchar>(i+1);
        uchar* p = dest.ptr<uchar>(i);
        for(int j=nChannel;j<nChannel*(nCol-1);++j)
            p[j] = saturate_cast<uchar>(pCur[j]*kernel[4]+pPre[j-nChannel]*kernel[0]
                                         +pPre[j]*kernel[1]
                                         +pPre[j+nChannel]*kernel[2]
                                         +pCur[j-nChannel]*kernel[3]
                                         +pCur[j+nChannel]*kernel[5]
                                         +pNext[j-nChannel]*kernel[6]
                                         +pNext[j]*kernel[7]
                                         +pNext[j+nChannel]*kernel[8]);
    }

    dest.row(0).setTo(Mat::zeros(1,3,img.type()));
    dest.row(nRow-1).setTo(Scalar(0));
    dest.col(0).setTo(Scalar(0));
    dest.col(nCol-1).setTo(Scalar(0));
}

void filterSharpen(Mat img, const Mat &kernel, Mat &dest) //this version is faster 40 times than above version
{
    CV_Assert(img.depth() != sizeof(uchar));

    filter2D(img,dest,img.depth(),kernel);
}

void testSharpen(const string& fileName)
{
    Mat img = imread(fileName,CV_LOAD_IMAGE_COLOR);
    imshow("origin",img);

    Mat lap;
    Laplacian(img,lap,CV_8U,3);
    imshow("1",lap);

    char* kernel1 = new char [9]{0,-1,0,-1,4,-1,0,-1,0}; //c++11 new feature
    Mat kernel2 = (Mat_<char>(3,3) << 0,-1,0,-1,4,-1,0,-1,0);

    Mat uDest, fDest;

    double t1 = getTickCount();
    userSharpen(img,kernel1,uDest);
    double t2 = getTickCount();

    filterSharpen(img,kernel2,fDest);
    double t3 = getTickCount();

    imshow("user sharpen",uDest);
    imshow("filter sharpen",fDest);
    waitKey();

    const char *cmd = "clear";
    if(0 != system(cmd))
        cerr << "system clear error" << endl;

    cout << "    time consuming comparation    " << endl;
    cout << "--------------------------------------" << endl;
    cout << "   user sharpen : " << (t2-t1)/getTickFrequency() << endl;
    cout << " filter sharpen : " << (t3-t2)/getTickFrequency() << endl;
    cout << "--------------------------------------" << endl;
}

 ///////////////////////////////////////////////////////////////////////
//image blend : dest = (1-alpha)*src1 + alpha*src2;
void blendTwoImage(const Mat &src1, const Mat &src2)
{
    assert((src1.rows == src2.rows)&&(src1.cols == src2.cols));

    double alpha = 0.0;
    int delay = 200;

    Mat dest;
    dest.create(src1.rows,src1.cols,src1.type());

    while(alpha <= 1){
        addWeighted(src1,alpha,src2,1-alpha,0.0,dest);

        imshow("image blending",dest);
        waitKey(delay);

        alpha += 0.05;
    }
    waitKey();
}

void testBlend(const string& fileName1,const string &fileName2)
{
    Mat src1 = imread(fileName1,IMREAD_COLOR);
    Mat src2 = imread(fileName2,IMREAD_COLOR);

    int minRow = std::min(src1.rows,src2.rows);
    int minCol = std::min(src1.cols,src2.cols);
    src1 = Mat(src1,Range(1,minRow),Range(1,minCol));
    src2 = Mat(src2,Range(1,minRow),Range(1,minCol));

    blendTwoImage(src1,src2);
}

////////////////////////////////////////////////////////////////////
// random line and text
// test RNG class 

Scalar randomColor(RNG& rng)
{
    int color = (unsigned)rng;
    return Scalar(color,(color>>8)&255,(color>>16)&255); // 取低8wei
}

void randomLine(Mat img)
{
    Point start,end;
    RNG rng(0xFFFFFFFF);
    int nRow = img.rows,nCol = img.cols;
    int key = 0;
    int Thickness = 1,lineType = 8;

    while(key != 27){
        start.x = rng.uniform(0,nCol);
        start.y = rng.uniform(0,nRow);
        end.x = rng.uniform(0,nCol);
        end.y = rng.uniform(0,nRow);

        line(img,start,end,randomColor(rng),Thickness,lineType);
        imshow("random line",img);
        key = waitKey(20);
    }
}

void randomText(Mat img)
{
    Point org;
    int thickness,linetype;
    int fontface,fontscale;
    int nRow = img.rows, nCol = img.cols;

    RNG rng(0xFFFFFFFF);
    string s[] = {"Linux","Windows","C++","python","OpenCV","OpenGL","cheng","shao","guang"};
    int size = sizeof(s)/sizeof(s[0]);
    int key = 0;
    while(key != 27){
        org.x = rng.uniform(0,nCol);
        org.y = rng.uniform(0,nRow);

        fontface = rng.uniform(0,8);
        fontscale = rng.uniform(0,100)*0.02+0.1;
        thickness = rng.uniform(1,3);
        linetype = rng.uniform(0,9);

        putText(img,s[rng.uniform(0,size)],org,fontface,fontscale,randomColor(rng),thickness,linetype);
        imshow("random text",img);
        key = waitKey(20);
    }
}

void displayStringAtCenter(const string& str)
{
    Size size = getTextSize(str,CV_FONT_HERSHEY_COMPLEX,3,4,NULL);
    Mat img(Mat::zeros(size.height*1.5,size.width,CV_8UC3));
    Point org(0,size.height*1.1);
    RNG rng(0xFFFFFFFF);

    int key = 0;
    while(key != 27){
        putText(img,str,org,CV_FONT_HERSHEY_COMPLEX,3,randomColor(rng),3,8);
        imshow("display string",img);
        key = waitKey(20);
    }
}

void testRandom()
{
    int width = 500, height = 500;
    Mat img(Mat::zeros(width,height,CV_8UC3));
//    randomLine(img);
//    randomText(img);
    displayStringAtCenter("Linux");
}

///////////////////////////////////////////////////////////////////////
// image Bright and contrast test

// dest = alpha*src+beta (alpha : contrast, beta bright)
void testBrightContrast(int argc,char* argv[])
{
    if(argc < 4){
        cout << "program [image] [contrast] [bright]" << endl;
        return;
    }
    Mat m = imread(argv[1]);
    double alpha = atof(argv[2]);
    double beta = atof(argv[3]);

    imshow("orgin",m);
    m.convertTo(m,m.type(),alpha,beta);
    imshow("convert to",m);
    waitKey();
}

///////////////////////////////////////////////////////////
//create  image histogram using opencv API

void hist(const Mat& img, Mat& histImg)
{
    if(!img.data)
        return;
    assert(img.channels() == 1);

    int histSize = 256;
    float range[] = {0,256};
    const float *range_[] = {range};

    calcHist(&img,1,0,Mat(),histImg,1,&histSize,range_,true,false);
}

void testHist(int argc,char*argv[])
{
    Mat m = imread(argv[1]);

    vector<Mat> bgr;
    split(m,bgr);

    Mat bHist,gHist,rHist;

    hist(bgr[0],bHist);
    hist(bgr[1],gHist);
    hist(bgr[2],rHist);

    int histSize = bHist.rows;

    int histW = 512,histH = 500;
    double binW = cvRound((double)histW/(double)histSize);

    normalize(bHist,bHist,0,histH,NORM_MINMAX,-1,Mat());
    normalize(gHist,gHist,0,histH,NORM_MINMAX,-1,Mat());
    normalize(rHist,rHist,0,histH,NORM_MINMAX,-1,Mat());

    Mat histImage(histH,histW,CV_8UC3,Scalar(0,0,0));
    for(int i=1;i<histSize;i++){
        line(histImage,Point(binW*(i-1),histH - bHist.at<float>(i-1)),Point(binW*i,histH - bHist.at<float>(i)),Scalar(255,0,0));
        line(histImage,Point(binW*(i-1),histH - gHist.at<float>(i-1)),Point(binW*i,histH - gHist.at<float>(i)),Scalar(0,255,0));
        line(histImage,Point(binW*(i-1),histH - rHist.at<float>(i-1)),Point(binW*i,histH - rHist.at<float>(i)),Scalar(0,0,255));
    }

    imshow("image",m);
    imshow("hist",histImage);
    waitKey(0);
}

/////////////////////////////////////////
//// back projection
//// using calBackProject() API
/////////////////////////////////////////
Mat src,hueSrc,target,hueTarget;
int bins = 13;

void histAndBackProject(int, void*)
{
    Mat hist;
    float range_[] = {0,180};
    const float* range = {range_};
    int histSize = max(bins,2);

    calcHist(&hueSrc,1,0,Mat(),hist,1,&histSize,&range,true,false);
    normalize(hist,hist,0,255,NORM_MINMAX,-1,Mat());

    Mat backProject;
    calcBackProject(&hueTarget,1,0,hist,backProject,&range,1,true);

    imshow("back project",backProject);
    imwrite("backP.jpg",backProject);
}

void testBackProject(int argc, char* argv[])
{
    Mat hsvSrc,hsvTarget,s;

    if(argc != 3){
        cout << "program image1 image2 " << endl;
        return;
    }

    if(!csg::isFileExsit(argv[1]))
        return ;
    if(!csg::isFileExsit(argv[2]))
        return ;

    src = imread(argv[1],CV_LOAD_IMAGE_COLOR);
    if(!src.data){
        cout << "image data error" << endl;
        return;
    }

    target = imread(argv[2],CV_LOAD_IMAGE_COLOR);
    if(!target.data){
        cout << "image data error" << endl;
        return;
    }

    cvtColor(src,hsvSrc,CV_BGR2HSV);
    cvtColor(target,hsvTarget,CV_BGR2HSV);

    const int fromTo[] = {0,0};

    hueSrc.create(hsvSrc.rows,hsvSrc.cols,CV_8UC1);
    hueTarget.create(hsvTarget.rows,hsvTarget.cols,CV_8UC1);
    /**
     *  mixChannels can be replaced by split.
     */
    mixChannels(&hsvSrc,1,&hueSrc,1,fromTo,1);
    mixChannels(&hsvTarget,1,&hueTarget,1,fromTo,1);
    imshow("hue src",hueSrc);
    imshow("hue target",hueTarget);

    const int fromTo_2[] = {1,0};
    s.create(hsvSrc.size(), hsvSrc.depth());
    mixChannels(&hsvSrc,1,&s,1,fromTo_2, 1);
    imshow("s",s);
    imshow("hsv",hsvSrc);
    imwrite("image/soccer_hsv.jpg",hsvSrc);
    imwrite("image/soccer_s.jpg",s);
    imwrite("image/soccer_hue.jpg",hueSrc);

    namedWindow("source image");
    createTrackbar("bins","source image",&bins,180,histAndBackProject);
    histAndBackProject(0,0);

    imshow("source image",src);
    imshow("target image",target);
    waitKey(0);
}

//////////////////////////////////////////////////////////////////
////// implement harris corner detection algorithm by my self and then compare it with the API in opencv
/////////////////////////////////////////////////////////////////
/**
 * @brief myCornerHarris
 * @param src_ input image, only has one channel
 * @param dst all pixels' reponse R = det(M) - k*trace(M)^2 ,M is auto correclation matrix;
 * @param blockSize
 * @param kSize
 * @param k
 * @param borderType
 */
void myCornerHarris(const Mat& src_, Mat& dst, int blockSize, int kSize, double k)
{
    Mat src;
    src_.copyTo(src);
    dst.create(src.size(),CV_32F);

    int depth = src.depth();
    double scale = (double)(1 << ((kSize > 0 ? kSize : 3) - 1)) * blockSize;
    if( depth == CV_8U )
        scale *= 255.;
    scale = 1./scale;

    assert(src.type() == CV_8UC1 || src.type() == CV_32FC1);

    Mat dx,dy;
    Sobel(src,dx,CV_32F,1,0,kSize,scale,0);
    Sobel(src,dy,CV_32F,0,1,kSize,scale,0);

    Size size = src.size();
    Mat cov(size,CV_32FC3);
    int i,j;
    for(i = 0;i < size.height;i++){
        float *covData = (float*)(cov.data + i*cov.step);
        const float *dxData = (const float*)(dx.data + i*dx.step);
        const float *dyData = (const float*)(dy.data + i*dy.step);

        for(j = 0;j < size.width;j++){
            float dx_ = dxData[j];
            float dy_ = dyData[j];

            covData[3*j] = dx_*dx_;
            covData[3*j+1] = dx_*dy_;
            covData[3*j+2] = dy_*dy_;
        }
    }

    // compute the sum of blocksize window
    boxFilter(cov,cov,cov.depth(),Size(blockSize,blockSize),Point(-1,-1),false);

    if(cov.isContinuous() && dst.isContinuous()){
        size.width *= size.height;
        size.height = 1;
    }
    else
        size = dst.size();

    for(i = 0;i < size.height;i++){
        float *dstData = (float*)(dst.data + i*dst.step);
        const float *covData = (const float*)(cov.data + i*cov.step);

        for(j = 0;j < size.width;j++){
            float a = covData[3*j];
            float b = covData[3*j+1];
            float c = covData[3*j+2];

            dstData[j] = a*c - b*b - k*(a+c)*(a+c);
        }
    }
}

void testCornerHarris(int argc, char* argv[]){
    if(argc != 2){
        PRINT "ERROR : program image\n";
        return;
    }

    if(!csg::isFileExsit(argv[1])){
        PRINT "",argv[1],"does not exsit\n";
        return;
    }

    Mat src = imread(argv[1],CV_LOAD_IMAGE_COLOR);
    if(!src.data){
        PRINT "image data error\n";
        return;
    }

    Mat grayImage;
    cvtColor(src,grayImage,CV_BGR2GRAY);

   //compare the difference of my cornerHarris and opencv cornerHarris
    Mat myHarrisCorner, harrisCorner;
    int blockSize = 3;
    int kSize = 3;
    double k = 0.04;
    cornerHarris(grayImage,harrisCorner,blockSize,kSize,k);
    myCornerHarris(grayImage,myHarrisCorner,blockSize,kSize,k);

    double error = 0.0;
    error = norm(myHarrisCorner,harrisCorner,NORM_L2);

    PRINT "error : ",error;
}

/**
 * @brief testVideoCapture test how to use VideoCapture
 */
void testVideoCapture(string name)
{
    VideoCapture video;
    Mat frame;

    video.open(name);
    double nframe = video.get(CV_CAP_PROP_FRAME_COUNT);

    for(int i =0;i<5;i++)
       video.read(frame);
}



int main(int argc,char* argv[])
{
    if(argc < 2){
        cout << "program [image name]" << endl;
        return -1;
    }

    const char *cmd = "clear";
    system(cmd);

//    testAccessSpeed(argv[1]);
//    testSharpen(argv[1]);
//    testBlend(argv[1],argv[2]);
//    testRandom();
//    testBrightContrast(argc,argv);
//    testHist(argc,argv);
    testBackProject(argc,argv);
//    testCornerHarris(argc,argv);

    return 0;
}


