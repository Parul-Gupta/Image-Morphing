#include <stdio.h>
#include<string>
#include<fstream>
#include<stdlib.h>
#include <opencv2/opencv.hpp>
#include<cmath>
#include<vector>
using namespace cv;
using namespace std;
//Note: The code snippet to check whether a point lies inside a triangle has been taken from geeksforgeeks.
float area(int x1, int y1, int x2, int y2, int x3, int y3);
bool isInside(int x1, int y1, int x2, int y2, int x3, int y3, int x, int y);
void tie(Mat& img,Mat& outimg,int x1,int y1,int x2,int y2,int x3,int y3,int x4,int y4,int X1,int Y1,int X2,int Y2,int X3,int Y3,int X4,int Y4);

int main(){
    	string in="";
    	string out="";
    	cout<<"Enter the initial image path: "<<endl;
    	cin>>in;
    	cout<<"Enter the output image path: "<<endl;
    	cin>>out;
    	string i1="";
    	string o1="";
    	cout<<"Enter the initial image's tie point file-path: "<<endl;
    	cin>>i1;
    	cout<<"Enter the final image's tie point file-path: "<<endl;
    	cin>>o1;
		float n=0;
		cout<<"Enter the number of intermediate images you want: ";
		cin>>n;n++;
		ifstream ifp;
		ifstream ofp;
		ifp.open(i1.c_str(), ios::in);
		ofp.open(o1.c_str(), ios::in);
		vector<int> i_x;
		vector<int> i_y;
		vector<int> o_x;
		vector<int> o_y;
		int x=0,y=0;
		while(ifp>>x>>y){
			i_x.push_back(x);
			i_y.push_back(y);
		}
		ifp.close();
		while(ofp>>x>>y){
			o_x.push_back(x);
			o_y.push_back(y);
		}
		ofp.close();
		Mat inimg=imread(in,IMREAD_GRAYSCALE);
		Mat outimg=imread(out,IMREAD_GRAYSCALE);
		namedWindow("Warp Image", WINDOW_AUTOSIZE );
		imshow("Warp Image",inimg);
		waitKey(10);
		char c='0';
		for(float t=1.0/n;t<1.0;t=t+1.0/n){
			Mat wimg=Mat::zeros(max(inimg.rows,outimg.rows),max(inimg.cols,outimg.cols),CV_8UC1);
			vector<int> w_x;
			vector<int> w_y;
			for(int i=0;i<i_x.size();i++){
				float px=i_x[i]*(1-t)+o_x[i]*t;
				float py=i_y[i]*(1-t)+o_y[i]*t;
				w_x.push_back((int)px);
				w_y.push_back((int)py);
			}
			vector<Point2f> points;
			for(int i=0;i<w_x.size();i++){
				points.push_back(Point2f(w_x[i],w_y[i]));
			}
			Rect rect(0,0,wimg.cols,wimg.rows);
			Subdiv2D subdiv(rect);
			for(vector<Point2f>::iterator it=points.begin();it!=points.end();it++){
				subdiv.insert(*it);
			}
			vector<Vec6f> trianglelist;
			subdiv.getTriangleList(trianglelist);
			for(int i=0;i<trianglelist.size();i++){
				if( !rect.contains(Point(trianglelist[i][0],trianglelist[i][1])) || !rect.contains(Point(trianglelist[i][2],trianglelist[i][3])) || !rect.contains(Point(trianglelist[i][4],trianglelist[i][5])) )continue;
			
				int p=0,q=0,r=0;
				for(p=0;p<w_x.size();p++){
					if(trianglelist[i][0]==w_x[p] && trianglelist[i][1]==w_y[p])break;
				}
				for(q=0;q<w_x.size();q++){
					if(trianglelist[i][2]==w_x[q] && trianglelist[i][3]==w_y[q])break;
				}
				for(r=0;r<w_x.size();r++){
					if(trianglelist[i][4]==w_x[r] && trianglelist[i][5]==w_y[r])break;
				}
				Mat m=Mat::zeros(6,6,CV_32F);
				Mat col=Mat::zeros(6,1,CV_32F);
				Mat col1=Mat::zeros(6,1,CV_32F);
				m.at<float>(0,0)=w_x[p];
				m.at<float>(0,1)=w_y[p];
				m.at<float>(0,2)=1;
				m.at<float>(1,3)=w_x[p];
				m.at<float>(1,4)=w_y[p];
				m.at<float>(1,5)=1;
				m.at<float>(2,0)=w_x[q];
				m.at<float>(2,1)=w_y[q];
				m.at<float>(2,2)=1;
				m.at<float>(3,3)=w_x[q];
				m.at<float>(3,4)=w_y[q];
				m.at<float>(3,5)=1;
				m.at<float>(4,0)=w_x[r];
				m.at<float>(4,1)=w_y[r];
				m.at<float>(4,2)=1;
				m.at<float>(5,3)=w_x[r];
				m.at<float>(5,4)=w_y[r];
				m.at<float>(5,5)=1;
			
				col.at<float>(0,0)=i_x[p];;
				col.at<float>(1,0)=i_y[p];
				col.at<float>(2,0)=i_x[q];
				col.at<float>(3,0)=i_y[q];
				col.at<float>(4,0)=i_x[r];
				col.at<float>(5,0)=i_y[r];
			
				col1.at<float>(0,0)=o_x[p];;
				col1.at<float>(1,0)=o_y[p];
				col1.at<float>(2,0)=o_x[q];
				col1.at<float>(3,0)=o_y[q];
				col1.at<float>(4,0)=o_x[r];
				col1.at<float>(5,0)=o_y[r];
			
				Mat res1=Mat::zeros(6,1,CV_32F);
				res1=m.inv()*col;
				Mat res2=Mat::zeros(6,1,CV_32F);
				res2=m.inv()*col1;
				for(int u=min(w_x[p],min(w_x[q],w_x[r]));u<max(w_x[p],max(w_x[q],w_x[r]));u++){
					for(int v=min(w_y[p],min(w_y[q],w_y[r]));v<max(w_y[p],max(w_y[q],w_y[r]));v++){
						if(isInside(w_x[p],w_y[p],w_x[q],w_y[q],w_x[r],w_y[r],u,v)){
								wimg.at<uchar>(Point(u,v))=(1-t)*inimg.at<uchar>(Point(res1.at<float>(0,0)*u+res1.at<float>(1,0)*v+res1.at<float>(2,0),res1.at<float>(3,0)*u+res1.at<float>(4,0)*v+res1.at<float>(5,0)))+t*outimg.at<uchar>(Point(res2.at<float>(0,0)*u+res2.at<float>(1,0)*v+res2.at<float>(2,0),res2.at<float>(3,0)*u+res2.at<float>(4,0)*v+res2.at<float>(5,0)));
						}
					}
				}
			}
			string filename="img_";
			filename += c;
			filename += ".jpg";
			imwrite(filename,wimg);
			if(c=='9')c='a';
			else if(c>='a' && c<='z')c=c-32;
			else if(c>='A' && c<'Z')c=c+33;
			else c++;
			imshow("Warp Image", wimg);	
			waitKey(10);
		}
		imshow("Warp Image",outimg);
		waitKey(10);
		return 0;
	
}

float area(int x1, int y1, int x2, int y2, int x3, int y3){
   return abs((x1*(y2-y3) + x2*(y3-y1)+ x3*(y1-y2))/2.0);
}

bool isInside(int x1, int y1, int x2, int y2, int x3, int y3, int x, int y){   
   float A = area (x1, y1, x2, y2, x3, y3);
   float A1 = area (x, y, x2, y2, x3, y3);
   float A2 = area (x1, y1, x, y, x3, y3);  
   float A3 = area (x1, y1, x2, y2, x, y);
   return (A == A1 + A2 + A3);
}



















