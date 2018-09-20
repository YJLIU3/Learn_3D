#include<opencv2/opencv.hpp>
using namespace cv;

#include<math.h>
#include<vector>
#include<iostream>
using namespace std;

void Drawfilledcircle(Mat &img, Point center,Scalar color)
{
	circle(img, center, 1, color, -1, 8);
}

void CreatCircle(Mat &Matin, int &r)
{
	int w = 2 * r + 1;
	for (int i = 0; i < w; i++)
	{
		for (int j = 0; j<w; j++)
		{
			if ((i - r)*(i - r) + (j - r)*(j - r)>r*r)
			{
				Matin.at<int>(j, i) = 0;
			}
		}
	}
}

float average(Mat &img1)
{
	float sum = 0;
	float mean_mask = 0;
	for (int i = 0; i < img1.rows; i++)
	{
		for (int j = 0; j < img1.cols; j++)
		{
			sum += img1.at<int>(j, i);
		}
	}
	mean_mask = sum / 317;
	return mean_mask;
}


/**********************Design a conv template***********************/
void setConTemp(int ffsize, Mat ff_A)
{
	int width, height;
	width = height = ffsize;
	Point2f origin((ffsize + 1)*1.0 /2.0, (ffsize + 1)*1.0 /2.0);
	Point2f Cen((ffsize + 1)*1.0 / 2.0, (ffsize + 1)*1.0 / 2.0);

	float dis;
	if (origin.x <= Cen.x && origin.y <= Cen.y)
	{
		dis = sqrt((width - 1 - origin.x)*(width - 1 - origin.x) +
			(height - 1 - origin.y)*(height - 1 - origin.y));
	}
	else if (origin.x <= Cen.x && origin.y>Cen.y)
	{
		dis = sqrt((width - 1 - origin.x)*(width - 1 - origin.x) +
			origin.y*origin.y);

	}
	else if (origin.x>Cen.x && origin.y>Cen.y)
	{
		dis = sqrt(origin.x*origin.x + (origin.y)*(origin.y));
	}
	else
	{
		dis = sqrt(origin.x*origin.x +
			(height - 1 - origin.y)*(height - 1 - origin.y));
	}

	float weight = 1 / dis;
	float dis2;
	for (int i = (ffsize + 1) / 2; i<ffsize; i++)
	{
		for (int j = (ffsize + 1) / 2; j<ffsize; j++)
		{
			dis2 = sqrt((i - origin.x)*(i - origin.x) + (j - origin.y)*(j - origin.y));
			ff_A.at<float>(i, j) = (1-(ff_A.at<float>(i, j) + weight * dis2));
//			cout << 255 - (ff_A.at<char>(i, j) + weight * dis2)<<"\t";
		}
		cout << endl;
	}

	
}

//********************Traversing Pixel get credible image***********************//
void Reduce_C(const Mat& image_A, const Mat& image_B, const Mat& image_C,
			  const Mat& image_D, const Mat& image_U, Mat& outImage, int div)
{
	int nr = image_A.rows;
	int nc = image_A.cols;

	Mat test;
	test.create(image_A.size(), image_A.type());
	outImage.create(image_A.size(), image_A.type());
	if (image_A.isContinuous() && outImage.isContinuous())
	{
		nr = 1;
		nc = nc * image_A.rows*image_A.channels();
	}
	for (int i = 0; i<nr; i++)
	{
		const float* inData_A = image_A.ptr<float>(i);
		const float* inData_B = image_B.ptr<float>(i);
		const float* inData_C = image_C.ptr<float>(i);
		const float* inData_D = image_D.ptr<float>(i);
		const float* inData_U = image_U.ptr<float>(i);
		int* outData = outImage.ptr<int>(i);

		for (int j = 0; j<nc; j++)
		{
			float F_A = *inData_A++ / div * div + div / 2;
			float F_B = *inData_B++ / div * div + div / 2;
			float F_C = *inData_C++ / div * div + div / 2;
			float F_D = *inData_D++ / div * div + div / 2;
			float U = *inData_U++ / div * div + div / 2;

			float s_1 = min(min(F_A, F_B) - U, U - min(F_C, F_D));
			float s_2 = min(U - min(F_A, F_B), min(F_C, F_D) - U);
			float value1 = max(s_1,s_2);
			float value2 = max(s_1*s_1,s_2*s_2);
//			float value = max(value1, 0);
			if (value1 > 1100)
				*outData++ = value1;
			else
				*outData++ = 0;
		}
	}
}

vector<Point2f> Get_local_max(Mat Image_in, int steps, int step1)
{
	int height = Image_in.rows;
	int width = Image_in.cols;
	int step = steps;

	vector<Point2f>Local_Max;
	double maxVal = 0;
	Point MaxLoc;
	int local_max_x, local_max_y;

	vector<Point2f>Local_Max1;
	double maxVal1 = 0;
	Point MaxLoc1;
	int local_max_x1, local_max_y1;

	for (int i = 0; i < height - step; i += step)
	{
		for (int j = 0; j < width - step; j += step)
		{
			Mat PREIWindow = Image_in(Range(i, i + step), Range(j, j + step));
			minMaxLoc(PREIWindow, NULL, &maxVal, NULL, &MaxLoc);
			local_max_x = j + MaxLoc.x;
			local_max_y = i + MaxLoc.y;
			Point2f A(local_max_x, local_max_y);
			if (maxVal >264900)
			{
				cout << maxVal << "\t" << local_max_x << "," << local_max_y << "\t";
			}
			if (maxVal != 0)
			Local_Max.push_back(A);
		}
	}

	for (int i = 0; i<Local_Max.size(); i++)
	{
		if (Local_Max[i].x - step1 > 0 && Local_Max[i].x + step1 < width &&
			Local_Max[i].y - step1 > 0 && Local_Max[i].y + step1 < height)
		{
			Mat PREIwindow2 = Image_in(Range((Local_Max[i].y - step1), (Local_Max[i].y + step1 + 1)),
				Range((Local_Max[i].x - step1), (Local_Max[i].x + step1 + 1)));

			minMaxLoc(PREIwindow2, NULL, &maxVal1, NULL, &MaxLoc1);
			local_max_x1 = Local_Max[i].x;
			local_max_y1 = Local_Max[i].y;

			if (Image_in.at <float>(Local_Max[i].y, Local_Max[i].x) == maxVal1)
			{
				Point2f B(local_max_x1, local_max_y1);
				Local_Max1.push_back(B);
			}
		}
	}

	Mat inp = imread("input.png", CV_8UC1);

	Mat showimage = Mat::zeros(Image_in.rows, Image_in.cols, CV_8UC3);
	cvtColor(inp, showimage, CV_GRAY2BGR);


	for (vector<Point2f>::iterator it = Local_Max1.begin(); it != Local_Max1.end(); ++it)
	{
		Scalar color(0, 0, 255);
		Drawfilledcircle(showimage, *it, color);
	}
	imwrite("step4.jpg", showimage);

	TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 40, 0.1);
	cornerSubPix(inp, Local_Max1, Size(5, 5), Size(-1, -1), criteria);

	for (vector<Point2f>::iterator it = Local_Max1.begin(); it != Local_Max1.end(); ++it)
	{
		Scalar color(255, 0, 0);
		Drawfilledcircle(showimage, *it, color);
	}

	imshow("step4", showimage);
	imwrite("step5.jpg", showimage);
	waitKey(0);
	return Local_Max;
}

int main()
{
	Mat src = imread("input.png", 0);
	Mat dst = src;
//	GaussianBlur(src, src, Size(5, 5), 1);
	medianBlur(src, src, 3);
	src.convertTo(src, CV_32FC1);
	int ff = 19;
	Mat ff_A = Mat::zeros(ff, ff, CV_32FC1);
	Mat ff_B = Mat::zeros(ff, ff, CV_32FC1);
	Mat ff_C = Mat::zeros(ff, ff, CV_32FC1);
	Mat ff_D = Mat::zeros(ff, ff, CV_32FC1);
	setConTemp(ff, ff_A);
	flip(ff_A, ff_B, -1);
	flip(ff_B, ff_C, 0);
	flip(ff_C, ff_D, -1);

	Mat PREI_A = Mat::zeros(src.rows, src.cols, CV_32FC1);
	Mat PREI_B = Mat::zeros(src.rows, src.cols, CV_32FC1);
	Mat PREI_C = Mat::zeros(src.rows, src.cols, CV_32FC1);
	Mat PREI_D = Mat::zeros(src.rows, src.cols, CV_32FC1);

	Mat DREW = Mat::zeros(src.rows, src.cols, CV_8UC3);


	filter2D(src, PREI_A, -1, ff_A);
	filter2D(src, PREI_B, -1, ff_B);
	filter2D(src, PREI_C, -1, ff_C);
	filter2D(src, PREI_D, -1, ff_D);

	Mat U = Mat::zeros(ff, ff, CV_32FC1);
	Mat C = Mat::zeros(ff, ff, CV_32FC1);
	U = 0.25*(PREI_A + PREI_B + PREI_C + PREI_D);

	Reduce_C(PREI_A, PREI_B, PREI_C, PREI_D, U, C, 64);
	vector<Point2f>Candi_fea_Point;
	Candi_fea_Point = Get_local_max(C, 5,3);

	return 0;
}