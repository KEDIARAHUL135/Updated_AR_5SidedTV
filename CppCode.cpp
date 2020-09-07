
//###########################################################################################
// File Nme		: CppCode.py
// Created By	: Rahul Kedia
// Project		: AR_5SidedTV
// Description	: This file contains the C++ code for the project.
//###########################################################################################
#include <opencv2/aruco.hpp>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include "opencv2/imgcodecs.hpp"
#include <iostream>
#include <stdio.h>
#include <vector>

// Setting size in which all the frames will be resized to get the same size of all frames.
int FrameSize[] = {1024, 576};
// Declaring camera intrinsic parameters variables globally.
double ret;
cv::Mat mtx, dist, rvecs, tvecs;	


// This function reads the camera intrinsic parameters from the file "calibration.yaml"
// and stores them in globally declared variables.
int ReadCalibrationParams()
{
	cv::FileStorage calibration_file("calibration.yaml", cv::FileStorage::READ);
	ret = (double)calibration_file["ret"];
	calibration_file["camera_matrix"] >> mtx;
	calibration_file["distortion_coefficients"] >> dist;
	calibration_file["rotation_vectors"] >> rvecs;
	calibration_file["translation_vectors"] >> tvecs;
	calibration_file.release();	
	return 0;
}


// This function detects the aruco markers in the frame, estimates its pose and find the vertices of the 3D cube.
bool DetectAruco_FindVertices(cv::Mat ArucoVideoFrame, std::vector<int> &IDs, std::vector<cv::Point2f> &BottomVertices, std::vector<cv::Point2f> &TopVertices)
{	
	// BottomVertices and TopVertices variables store the bottom and top 4 vertices of the cube respectively.
	
	std::vector<cv::Point3f> axesPoint;
	axesPoint.push_back(cv::Point3f(0, 0, 0)); 
	axesPoint.push_back(cv::Point3f(0.1, 0, 0));
	axesPoint.push_back(cv::Point3f(0, 0.1, 0));
	axesPoint.push_back(cv::Point3f(0, 0, 0.1));

	// Detecting Markers
	cv::Mat GrayImage = ArucoVideoFrame.clone();
	cv::cvtColor(ArucoVideoFrame, GrayImage, cv::COLOR_BGR2GRAY);
	
	cv::Ptr<cv::aruco::Dictionary> ArucoDict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_50);
	
	std::vector<std::vector<cv::Point2f>> Corners, RejectedCandidates;
	cv::Ptr<cv::aruco::DetectorParameters> Parameters = cv::aruco::DetectorParameters::create();
	cv::aruco::detectMarkers(ArucoVideoFrame, ArucoDict, Corners, IDs, Parameters, RejectedCandidates);

	if (IDs.size() != 4)			// If no aruco marker found
		return false;
	
	// Estimating pose of aruco markers
	std::vector<cv::Vec3d> rvec, tvec;
	cv::aruco::estimatePoseSingleMarkers(Corners, 0.05, mtx, dist, rvec, tvec);

	// Checking if rvec and tvec are found (Pose is estimated)
	if(rvec.size() == 0 || tvec.size() == 0)
		return false;
	
	// Creating a copy to draw axis.
	cv::Mat ArucoVideoFrameCopy = ArucoVideoFrame.clone();

	// Finding cube vertices wrt each aruco marker.
	for (int i = 0 ; i < IDs.size() ; i++)
	{
		// The end points of each axis will be stored in "imagePoints"
		std::vector<cv::Point2f> imagePoints;
		cv::projectPoints(axesPoint, rvec[i], tvec[i], mtx, dist, imagePoints);
		cv::aruco::drawAxis(ArucoVideoFrameCopy, mtx, dist, rvec[i], tvec[i], 0.1);

		// storing bottom and top point of line showing z-axis
		BottomVertices.push_back(imagePoints[0]);
		TopVertices.push_back(imagePoints[3]);
	}
	//cv::imshow("ArucoAxisDisplay", ArucoVideoFrameCopy);		// Uncomment to display axes.
	return true;
}


// This function sets the vertices of cube in order in a single variable "CubeVertices" for reasons discussed in README.
bool SetCubeVertices(std::vector<cv::Point2f> BottomVertices, std::vector<cv::Point2f> TopVertices, std::vector<int> &IDs, std::vector<std::vector<int>> &CubeVertices)
{	
	int ArrangedBottomVertices[4][2], ArrangedTopVertices[4][2];
	
	// Arranging bottom and top vertices in order
	for (int i = 0 ; i < 4 ; i++)
	{
		ArrangedBottomVertices[IDs[i]][0] = int(round(BottomVertices[i].x));
		ArrangedBottomVertices[IDs[i]][1] = int(round(BottomVertices[i].y));
		ArrangedTopVertices[IDs[i]][0] = int(round(TopVertices[i].x));
		ArrangedTopVertices[IDs[i]][1] = int(round(TopVertices[i].y));
	}
	
	// Checking threshold values and appending CubeVertices
	for (int i = 0 ; i < 4 ; i++)
	{
		if ((0 <= ArrangedBottomVertices[i][0]) &&
		   (ArrangedBottomVertices[i][0] < FrameSize[0]) && 
		   (0 <= ArrangedBottomVertices[i][1]) &&
		   (ArrangedBottomVertices[i][1] < FrameSize[1]))
			CubeVertices.push_back(std::vector<int>({*(ArrangedBottomVertices[i]+0), *(ArrangedBottomVertices[i]+1)}));
		else
			return false;
	}
	
	for (int i = 0 ; i < 4 ; i++)
	{
		if ((0 <= ArrangedTopVertices[i][0]) &&
		   (ArrangedTopVertices[i][0] < FrameSize[0]) && 
		   (0 <= ArrangedTopVertices[i][1]) &&
		   (ArrangedTopVertices[i][1] < FrameSize[1]))
			CubeVertices.push_back(std::vector<int>({*(ArrangedTopVertices[i]+0), *(ArrangedTopVertices[i]+1)}));
		else
			return false;
	}

	return true;
}


// This function does a projective transform of a frame wrt the points given in "ArucoPoint". 
// Remaining part of the frame is left black.
cv::Mat ProjectiveTransform(cv::Mat FrameToBeOverlaped, std::vector<cv::Point2f> ArucoPoint)
{
	int Height = FrameToBeOverlaped.rows, Width = FrameToBeOverlaped.cols;
	
	cv::Point2f InitialPoints[4], FinalPoints[4];
	InitialPoints[0] = cv::Point2f(0, 0); 
	InitialPoints[1] = cv::Point2f(Width-1, 0);
	InitialPoints[2] = cv::Point2f(0, Height-1); 
	InitialPoints[3] = cv::Point2f(Width-1, Height-1);
	
	FinalPoints[0] = cv::Point2f(ArucoPoint[0].x, ArucoPoint[0].y); 
	FinalPoints[1] = cv::Point2f(ArucoPoint[1].x, ArucoPoint[1].y);
	FinalPoints[2] = cv::Point2f(ArucoPoint[3].x, ArucoPoint[3].y); 
	FinalPoints[3] = cv::Point2f(ArucoPoint[2].x, ArucoPoint[2].y);
		
	cv::Mat ProjectiveMatrix( 2, 4, CV_32FC1);
	ProjectiveMatrix = cv::Mat::zeros(Height, Width, FrameToBeOverlaped.type());
	ProjectiveMatrix = cv::getPerspectiveTransform(InitialPoints, FinalPoints);
	
	cv::Mat TransformedFrame = FrameToBeOverlaped.clone();
	cv::warpPerspective(FrameToBeOverlaped, TransformedFrame, ProjectiveMatrix, TransformedFrame.size());	

	return TransformedFrame;
}


// This function overlaps the two frames as required.
// ("FrameToBeOverlaped" on "ArucoVideoFrame" at coordinates given in "ArucoPoints")
// Working of this is explained properly in README.
cv::Mat OverlapImage(cv::Mat ArucoVideoFrame, cv::Mat FrameToBeOverlaped, std::vector<cv::Point2f> ArucoPoint)
{
	int Height = FrameToBeOverlaped.rows, Width = FrameToBeOverlaped.cols;
	
	cv::Mat TransformedFrame = ProjectiveTransform(FrameToBeOverlaped, ArucoPoint);
	
	cv::Mat MaskArucoVideoFrame = cv::Mat::zeros(cv::Size(Width, Height), CV_8UC3);
	std::vector<cv::Point> ArucoPointConverted;
	for (std::size_t i = 0 ; i < ArucoPoint.size(); i++)
    	ArucoPointConverted.push_back(cv::Point(ArucoPoint[i].x, ArucoPoint[i].y));
    
	cv::fillConvexPoly(MaskArucoVideoFrame, ArucoPointConverted, cv::Scalar(255, 255, 255), 8);
	cv::bitwise_and(MaskArucoVideoFrame, TransformedFrame, TransformedFrame);
	cv::bitwise_not(MaskArucoVideoFrame, MaskArucoVideoFrame);
	cv::Mat BlackFrameForOverlap = cv::Mat::zeros(cv::Size(Width, Height), CV_8UC3);
	cv::bitwise_and(ArucoVideoFrame, MaskArucoVideoFrame, BlackFrameForOverlap);
	cv::Mat FinalImage = cv::Mat::zeros(cv::Size(Width, Height), CV_8UC3);
	cv::bitwise_or(TransformedFrame, BlackFrameForOverlap, FinalImage);

	return FinalImage;
}


// This function calls the function OverlapImage to overlap the video frames on the sides of virtual cube formed.
// It first sets the vertices for all sides in variable "Vertices" and alsoo determines the order in which frames
// will be overlapped.
bool CallForOverlapping(cv::Mat ArucoVideoFrame, cv::Mat VideoFramesTO[5], std::vector<std::vector<int>> CubeVertices, cv::Mat &FinalFrame)
{
	// This will store the y coordinate of center of the bottom edges of the 4 standing sides of the cube.
	// This is done to determine the order in which sides should be called for overlap.
	int EdgeCentersYCoordinate[4] = {(CubeVertices[0][1] + CubeVertices[1][1])/2,
							         (CubeVertices[1][1] + CubeVertices[2][1])/2,
							  		 (CubeVertices[2][1] + CubeVertices[3][1])/2,
							    	 (CubeVertices[3][1] + CubeVertices[0][1])/2};

	
	int SortedEdgeCenters[4];
	std::copy(std::begin(EdgeCentersYCoordinate), std::end(EdgeCentersYCoordinate), std::begin(SortedEdgeCenters));
	
	std::sort(SortedEdgeCenters, SortedEdgeCenters+4);

	// Determining the order.
	std::vector<int> Order;
	for (int i = 0 ; i < 4 ; i++){
		for (int j = 0 ; j < 4 ; j++){
			if (SortedEdgeCenters[i] == EdgeCentersYCoordinate[j])
			{
				int Count = 0;
				for(std::vector<int>::iterator it = Order.begin(); it != Order.end(); ++it)
					if (*it == j)
						Count ++;
				if (Count == 0)
				{
					Order.push_back(j);
					break;
				}
	}}}
	Order.push_back(4);
	
	// Setting corner vertices of all the faces.
	std::vector<std::vector<cv::Point2f>> Vertices;
	std::vector<cv::Point2f> Temp1, Temp2, Temp3, Temp4, Temp5;
	Temp1.push_back(cv::Point2f(CubeVertices[5][0], CubeVertices[5][1]));
	Temp1.push_back(cv::Point2f(CubeVertices[4][0], CubeVertices[4][1]));
	Temp1.push_back(cv::Point2f(CubeVertices[0][0], CubeVertices[0][1]));
	Temp1.push_back(cv::Point2f(CubeVertices[1][0], CubeVertices[1][1]));
	Vertices.push_back(Temp1);												// Standing face with markers ids 0&1
	Temp2.push_back(cv::Point2f(CubeVertices[6][0], CubeVertices[6][1]));
	Temp2.push_back(cv::Point2f(CubeVertices[5][0], CubeVertices[5][1]));
	Temp2.push_back(cv::Point2f(CubeVertices[1][0], CubeVertices[1][1]));
	Temp2.push_back(cv::Point2f(CubeVertices[2][0], CubeVertices[2][1]));
	Vertices.push_back(Temp2);												// Standing face with markers ids 1&2
	Temp3.push_back(cv::Point2f(CubeVertices[7][0], CubeVertices[7][1]));
	Temp3.push_back(cv::Point2f(CubeVertices[6][0], CubeVertices[6][1]));
	Temp3.push_back(cv::Point2f(CubeVertices[2][0], CubeVertices[2][1]));
	Temp3.push_back(cv::Point2f(CubeVertices[3][0], CubeVertices[3][1]));
	Vertices.push_back(Temp3);												// Standing face with markers ids 2&3
	Temp4.push_back(cv::Point2f(CubeVertices[4][0], CubeVertices[4][1]));
	Temp4.push_back(cv::Point2f(CubeVertices[7][0], CubeVertices[7][1]));
	Temp4.push_back(cv::Point2f(CubeVertices[3][0], CubeVertices[3][1]));
	Temp4.push_back(cv::Point2f(CubeVertices[0][0], CubeVertices[0][1]));
	Vertices.push_back(Temp4);												// Standing face with markers ids 3&0
	Temp5.push_back(cv::Point2f(CubeVertices[4][0], CubeVertices[4][1]));
	Temp5.push_back(cv::Point2f(CubeVertices[5][0], CubeVertices[5][1]));
	Temp5.push_back(cv::Point2f(CubeVertices[6][0], CubeVertices[6][1]));
	Temp5.push_back(cv::Point2f(CubeVertices[7][0], CubeVertices[7][1]));
	Vertices.push_back(Temp5);												// Top face.

	FinalFrame = ArucoVideoFrame.clone();

	for (int i = 0 ; i < 5 ; i++)
		FinalFrame = OverlapImage(FinalFrame, VideoFramesTO[Order[i]], Vertices[Order[i]]);
		
	return true;
}


// Main function. It reads the videos and calls all functions one by one in order to get the 3D box and displays it.
int main()
{
	// Calling function to read camera params from file
	ReadCalibrationParams();
	
	// Reading Videos
	cv::VideoCapture ArucoCap, Video1Cap, Video2Cap, Video3Cap, Video4Cap, Video5Cap;
	ArucoCap.open("Videos/ArucoVideo1.avi");
	Video1Cap.open("Videos/Video1.avi");
	Video2Cap.open("Videos/Video2.avi");
	Video3Cap.open("Videos/Video3.avi");
	Video4Cap.open("Videos/Video4.avi");
	Video5Cap.open("Videos/Video5.avi");

	cv::VideoCapture* CapList[] = {&ArucoCap, &Video1Cap, &Video2Cap, &Video3Cap, &Video4Cap, &Video5Cap};
	int SizeOfCapList = (sizeof(CapList)/sizeof(CapList[0]));

	while (true)
	{
		// Checking if all videos are opened.
		bool Break = false;
		for(int i = 0 ; i < SizeOfCapList ; i++)
			if(!CapList[i]->isOpened()) Break = true;
		if(Break)
		{
			std::cout << "Not able to read video.\n";
			break;
		}

		// Reading all frames.
		cv::Mat FrameList[6], OverlapVideoFrameList[5];
		for(int i = 0 ; i < SizeOfCapList ; i++)
		{
			cv::Mat Frame;
			CapList[i]->read(Frame);
			if(Frame.empty())
			{
				// If video ends, restart it.
				CapList[i]->set(cv::CAP_PROP_POS_FRAMES, 0);
				CapList[i]->read(Frame);
			}
			FrameList[i] = Frame;
		}

		// Resizing all frames to same size.
		for (int i = 0 ; i < (sizeof(FrameList)/sizeof(FrameList[0])) ; i++)
			cv::resize(FrameList[i], FrameList[i], cv::Size(FrameSize[0], FrameSize[1]));

		// Separating Aruco video frame and frames which are to be overlapped.
		cv::Mat ArucoVideoFrame = FrameList[0];
		std::copy(FrameList + 1, FrameList + 6, OverlapVideoFrameList + 0);

		// Detecting Arucos and finding vertices.
		std::vector<int> IDs;
		std::vector<cv::Point2f> BottomVertices, TopVertices;
		bool Ret1 = DetectAruco_FindVertices(ArucoVideoFrame, IDs, BottomVertices, TopVertices);		
		if(!Ret1) continue;					// Checking if all goes well

		// Setting cube vertices
		std::vector<std::vector<int>> CubeVertices;
		bool Ret2 = SetCubeVertices(BottomVertices, TopVertices, IDs, CubeVertices);
		if(!Ret2) continue;					// Checking if all goes well

		// Overlapping frames.
		cv::Mat FinalFrame;
		bool Ret3 = CallForOverlapping(ArucoVideoFrame, OverlapVideoFrameList, CubeVertices, FinalFrame);
		if (!Ret3) continue;					// Checking if all goes well
		
		// Displaying output.
		cv::imshow("FinalFrame", FinalFrame);
		if ((cv::waitKey(1) & 0xFF) == 'q')		// Break if 'q' is pressed.
			break;
	}
	
	// Releasing VideoCapture objects.
	for (int i = 0 ; i < SizeOfCapList ; i++)
		CapList[i]->release();

	return 0;
}
