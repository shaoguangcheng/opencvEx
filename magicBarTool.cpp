#include <iostream>
#include <stdlib.h>

//load opencv
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <time.h>

using namespace std;
using namespace cv;

/**
 * This program is written extremely badly.
 * I make so many mistakes when coding it.
 * Have a shallow recognition with OO
 * Be careful when processing the callback function of mouse event
 */

#define TEST
#undef TEST

/**
 * test floodfill function  and how to use trackbar and process mouse event
 */

// This is a complex version

enum colorMode{
    RGB = 0,
    GRAY
};
enum connectivityMode{
    MODE_4 = 4,
    MODE_8=8
};
enum rangeMode{
    FIXED = 1<<16,
    FLOATING = 0
};

Point seed;
connectivityMode connectivity;
rangeMode range;
int loDiff;
int upDiff;
string winName;
Mat imageShow;
Scalar fillColor(255,0,0);

class testFloodFillComplex
{

public:
    testFloodFillComplex(const string &imageFile);
    ~testFloodFillComplex(){}

private:
    void help();
    void keyProcess(int key);

public:
    colorMode color;
    Mat imageOri;
};

void onMouse(int event, int x, int y, int flags, void* userdata)
{
    if(event != EVENT_LBUTTONDBLCLK)
        return;

    seed = Point(x,y);
    int flag = connectivity + range;

//    imageShow.copyTo(imageFill);
    floodFill(imageShow,seed,fillColor,0,Scalar(loDiff,loDiff,loDiff),Scalar(upDiff,upDiff,upDiff),flag);
    imshow(winName,imageShow);
}

testFloodFillComplex::testFloodFillComplex(const string &imageFile)
{
    connectivity = MODE_4;
    color =  RGB;
    range = FIXED;
    loDiff = upDiff = 0;
    fillColor = Scalar(0,255,0);
    seed = Point(50,50);

    imageOri = imread(imageFile);
    if(imageOri.empty()){
        cout << "failed to load image\n"<< endl;
        help();
        exit(-1);
    }

    imageOri.copyTo(imageShow);

    winName = imageFile;
    namedWindow(winName,WINDOW_AUTOSIZE);
    createTrackbar("lof",winName,&loDiff,255);
    createTrackbar("uof",winName,&upDiff,255);
    setMouseCallback(winName,onMouse,0);

    int key;
    while(1){
        imshow(winName,imageShow);
        key = waitKey();
        if(key == 27)
            break;
        keyProcess(key);
    }
}

void testFloodFillComplex::keyProcess(int key)
{
    switch((char)key){
    case 'c':
        if(color == RGB){
            imageOri.copyTo(imageShow);
            cvtColor(imageShow,imageShow,CV_RGB2GRAY);
            color = GRAY;
            cout << "> set color mode : GRAY" << endl;
            break;
        }
        else{
            imageOri.copyTo(imageShow);
            color = RGB;
            cout << "> set color mode : RGB" << endl;
            break;
        }
    case 'f':
        range = FIXED;
        cout << "> set range mode : FIXED" << endl;
        break;
    case 'g':
        range = FLOATING;
        cout << "> set range mode : FLOATING" << endl;
        break;
    case 'r' :
        imageOri.copyTo(imageShow);
        cout << "> reset image" << endl;
        break;
    case '4' :
        connectivity = MODE_4;
        cout << "> set connectivity mode : MODE_4" << endl;
        break;
    case '8' :
        connectivity = MODE_8;
        cout << "> set connectivity mode : MODE_8" << endl;
        break;
    default :
		help();
        break;
    }
}

void testFloodFillComplex::help()
{
    cout << "\n*************This program demonstrates how to use floodfill()"
         <<"function and how to use trackbar and mouse in opencv*************\n";
    cout << "Usage : programname image" << endl;
    cout << "some useful keys : " << endl;
    cout << "     Esc------quit\n"
         <<"      c-------change color mode\n"
         <<"      f-------use fixed(absolute) range\n"
         <<"      r-------reset image\n "
         <<"      g-------use floating(relative) range\n"
         <<"      4-------4 connectivity mode\n"
         <<"      8-------8 connectivity mode\n"
         << endl;
}

// This is a simple version
void testFloodFillSim(int argc,char* argv[])
{
    string imageName = string(argv[1]);
    Mat imageIn = imread(imageName,IMREAD_COLOR);

#ifdef TEST
    imshow("imageIN",imageIn);
    waitKey();
#endif

    Mat imageBlur;
    blur(imageIn,imageBlur,Size(3,3));

#ifdef TEST
    imshow("imageBlur",imageBlur);
    waitKey();
#endif

    int low;
    namedWindow("imageFloodFill");
    createTrackbar("low","imageFloodFill",&low,255,0);

    Point seed;
    Rect rect;
    srand(time(0));
    seed.x = rand()%imageIn.cols;
    seed.y = rand()%imageIn.rows;
    Mat temp;
    for(;;){
        imageBlur.copyTo(temp);
        floodFill(temp,seed,Scalar(255,255,0),&rect,Scalar(low,low,low),Scalar(low,low,low),4);
        imshow("imageFloodFill",temp);
        cout << "low = " << low << endl;
        waitKey(50);
    }

#ifdef TEST
    imshow("imageFloodFill",imageBlur);
    waitKey();
#endif
}


int main(int argc,char* argv[])
{
	if(argc < 2){
		cout << "[Usage] : program image" << endl;
		return -1;
	}
//    testFloodFillSim(argc,argv);
    testFloodFillComplex f(argv[1]);
    return 0;
}

