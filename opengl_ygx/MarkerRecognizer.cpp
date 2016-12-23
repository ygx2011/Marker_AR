#include "MarkerRecognizer.h"
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\calib3d\calib3d.hpp>
#include <math.h>
#include <string>
#include <sstream>

#define ADAPTIVE_THRESH_SIZE 35
#define APPROX_POLY_EPS 0.08
#define MARKER_CELL_SIZE 10
#define MARKER_SIZE (7*MARKER_CELL_SIZE)

using namespace std;
using namespace cv;

//========================================Class Marker=====================================
Marker::Marker()
{
	m_id = -1;
	m_corners.resize(4, Point2f(0.f,0.f));
}

Marker::Marker(int _id, cv::Point2f _c0, cv::Point2f _c1, cv::Point2f _c2, cv::Point2f _c3)
{
	m_id = _id;

	m_corners.reserve(4);
	m_corners.push_back(_c0);
	m_corners.push_back(_c1);
	m_corners.push_back(_c2);
	m_corners.push_back(_c3);
}

void Marker::drawToImage(cv::Mat& image, cv::Scalar color, float thickness)
{
	circle(image, m_corners[0], thickness*2, color, thickness);
	circle(image, m_corners[1], thickness, color, thickness);
	line(image, m_corners[0], m_corners[1], color, thickness, CV_AA);
    line(image, m_corners[1], m_corners[2], color, thickness, CV_AA);
    line(image, m_corners[2], m_corners[3], color, thickness, CV_AA);
    line(image, m_corners[3], m_corners[0], color, thickness, CV_AA);
	
	Point text_point = m_corners[0] + m_corners[2];
	text_point.x /= 2;
	text_point.y /= 2;

	stringstream ss;
	ss << m_id;

	putText(image, ss.str(), text_point, FONT_HERSHEY_SIMPLEX, 0.5, color);
}

void Marker::estimateTransformToCamera(vector<Point3f> corners_3d, cv::Mat& camera_matrix, cv::Mat& dist_coeff, cv::Mat& rmat, cv::Mat& tvec)
{
	Mat rot_vec;
	bool res = solvePnP(corners_3d, m_corners, camera_matrix, dist_coeff, rot_vec, tvec);
	Rodrigues(rot_vec, rmat);
}



//====================================Class MarkerRecognizer================================
MarkerRecognizer::MarkerRecognizer()
{
	//标准Marker坐标，逆时针
	m_marker_coords.push_back(Point2f(0,0));
	m_marker_coords.push_back(Point2f(0, MARKER_SIZE-1));
	m_marker_coords.push_back(Point2f(MARKER_SIZE-1, MARKER_SIZE-1));
	m_marker_coords.push_back(Point2f(MARKER_SIZE-1, 0));
}

int MarkerRecognizer::update(Mat& image_gray, int min_size, int min_side_length)
{
	CV_Assert(!image_gray.empty());
	CV_Assert(image_gray.type() == CV_8UC1);

	Mat img_gray = image_gray.clone();

	//equalizeHist(img_gray, img_gray);

	vector<Marker> possible_markers;
	markerDetect(img_gray, possible_markers, min_size, min_side_length);
	markerRecognize(img_gray, possible_markers, m_markers);
	markerRefine(img_gray, m_markers);

	return m_markers.size();
}

void MarkerRecognizer::markerDetect(Mat& img_gray, vector<Marker>& possible_markers, int min_size, int min_side_length)
{
	Mat img_bin;

	int thresh_size = (min_size/4)*2 + 1;
	adaptiveThreshold(img_gray, img_bin, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, thresh_size, thresh_size/3);
	//threshold(img_gray, img_bin, 125, 255, THRESH_BINARY_INV|THRESH_OTSU);
	morphologyEx(img_bin, img_bin, MORPH_OPEN, Mat());	//use open operator to eliminate small patch

	vector<vector<Point>> all_contours;
	vector<vector<Point>> contours;
	findContours(img_bin, all_contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	for (int i = 0; i < all_contours.size(); ++i)
	{
		if (all_contours[i].size() > min_size)
		{
			contours.push_back(all_contours[i]);
		}
	}

	vector<Point> approx_poly;
	for (int i = 0; i < contours.size(); ++i)
	{
		double eps = contours[i].size()*APPROX_POLY_EPS;
		approxPolyDP(contours[i], approx_poly, eps, true);

		if (approx_poly.size() != 4)
			continue;

		if (!isContourConvex(approx_poly))
			continue;

		//Ensure that the distance between consecutive points is large enough
		float min_side = FLT_MAX;
		for (int j = 0; j < 4; ++j)
		{
			Point side = approx_poly[j] - approx_poly[(j+1)%4];
			min_side = min(min_size, side.dot(side));
		}
		if (min_side < min_side_length*min_side_length)
			continue;

		//Sort the points in anti-clockwise
		Marker marker = Marker(0, approx_poly[0], approx_poly[1], approx_poly[2], approx_poly[3]);
		Point2f v1 = marker.m_corners[1] - marker.m_corners[0];
		Point2f v2 = marker.m_corners[2] - marker.m_corners[0];
		if (v1.cross(v2) > 0)	//由于图像坐标的Y轴向下，所以大于零才代表逆时针
		{
			swap(marker.m_corners[1], marker.m_corners[3]);
		}
		possible_markers.push_back(marker);
	}
}

void MarkerRecognizer::markerRecognize(cv::Mat& img_gray, vector<Marker>& possible_markers, vector<Marker>& final_markers)
{
	final_markers.clear();

	Mat marker_image;
	Mat bit_matrix(5, 5, CV_8UC1);
	for (int i = 0; i < possible_markers.size(); ++i)
	{
		Mat M = getPerspectiveTransform(possible_markers[i].m_corners, m_marker_coords);
		warpPerspective(img_gray, marker_image, M, Size(MARKER_SIZE, MARKER_SIZE));
		threshold(marker_image, marker_image, 125, 255, THRESH_BINARY|THRESH_OTSU); //OTSU determins threshold automatically.

		//A marker must has a whole black border.
		for (int y = 0; y < 7; ++y)
		{
			int inc = (y == 0 || y == 6) ? 1 : 6;
			int cell_y = y*MARKER_CELL_SIZE;

			for (int x = 0; x < 7; x += inc)
			{
				int cell_x = x*MARKER_CELL_SIZE;
				int none_zero_count = countNonZero(marker_image(Rect(cell_x, cell_y, MARKER_CELL_SIZE, MARKER_CELL_SIZE)));
				if (none_zero_count > MARKER_CELL_SIZE*MARKER_CELL_SIZE/4)
					goto __wrongMarker;
			}
		}

		//Decode the marker
		for (int y = 0; y < 5; ++y)
		{
			int cell_y = (y+1)*MARKER_CELL_SIZE;

			for (int x = 0; x < 5; ++x)
			{
				int cell_x = (x+1)*MARKER_CELL_SIZE;
				int none_zero_count = countNonZero(marker_image(Rect(cell_x, cell_y, MARKER_CELL_SIZE, MARKER_CELL_SIZE)));
				if (none_zero_count > MARKER_CELL_SIZE*MARKER_CELL_SIZE/2)
					bit_matrix.at<uchar>(y, x) = 1;
				else
					bit_matrix.at<uchar>(y, x) = 0;
			}
		}

		//Find the right marker orientation
		bool good_marker = false;
		int rotation_idx;	//逆时针旋转的次数
		for (rotation_idx = 0; rotation_idx < 4; ++rotation_idx)
		{
			if (hammingDistance(bit_matrix) == 0)
			{
				good_marker = true;
				break;
			}
			bit_matrix = bitMatrixRotate(bit_matrix);
		}
		if (!good_marker) goto __wrongMarker;

		//Store the final marker
		Marker& final_marker = possible_markers[i];
		final_marker.m_id = bitMatrixToId(bit_matrix);
		std::rotate(final_marker.m_corners.begin(), final_marker.m_corners.begin() + rotation_idx, final_marker.m_corners.end());
		final_markers.push_back(final_marker);

__wrongMarker:
		continue;
	}
}

void MarkerRecognizer::markerRefine(cv::Mat& img_gray, vector<Marker>& final_markers)
{
	for (int i = 0; i < final_markers.size(); ++i)
	{
		vector<Point2f>& corners = final_markers[i].m_corners;
		cornerSubPix(img_gray, corners, Size(5,5), Size(-1,-1), TermCriteria(CV_TERMCRIT_ITER, 30, 0.1));
	}
}

Mat MarkerRecognizer::bitMatrixRotate(cv::Mat& bit_matrix)
{
	//Rotate the bitMatrix by anti-clockwise way
	Mat out = bit_matrix.clone();
	int rows = bit_matrix.rows;
	int cols = bit_matrix.cols;

	for (int i=0; i<rows; ++i)
	{
		for (int j=0; j<cols; j++)
		{
			out.at<uchar>(i,j) = bit_matrix.at<uchar>(cols-j-1, i);
		}
	}
	return out;
}

int MarkerRecognizer::hammingDistance(Mat& bit_matrix)
{
	const int ids[4][5]=
	{
		{1,0,0,0,0},	// 00
		{1,0,1,1,1},	// 01
		{0,1,0,0,1},	// 10
		{0,1,1,1,0}		// 11
	};
  
	int dist=0;

	for (int y=0; y<5; ++y)
	{
		int minSum = INT_MAX; //hamming distance to each possible word
    
		for (int p=0; p<4; ++p)
		{
			int sum=0;
			//now, count
			for (int x=0; x<5; ++x)
			{
				sum += !(bit_matrix.at<uchar>(y, x) == ids[p][x]);
			}
			minSum = min(minSum, sum);
		}
    
		//do the and
		dist += minSum;
	}
  
	return dist;
}

int MarkerRecognizer::bitMatrixToId(Mat& bit_matrix)
{
	int id = 0;
	for (int y=0; y<5; ++y)
	{
		id <<= 1;
		id |= bit_matrix.at<uchar>(y,1);

		id <<= 1;
		id |= bit_matrix.at<uchar>(y,3);
	}
	return id;
}

vector<Marker>& MarkerRecognizer::getMarkers()
{
	return m_markers;
}

void MarkerRecognizer::drawToImage(cv::Mat& image, cv::Scalar color, float thickness)
{
	for (int i = 0; i < m_markers.size(); ++i)
	{
		m_markers[i].drawToImage(image, color, thickness);
	}
}
