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


int FrameSize[] = {1024, 576};
double ret;
cv::Mat mtx, dist, rvecs, tvecs;	


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


bool DetectAruco_FindVertices(cv::Mat ArucoVideoFrame, std::vector<int> &IDs, std::vector<cv::Point2f> &BottomVertices, std::vector<cv::Point2f> &TopVertices)
{
	std::vector<cv::Point3f> axesPoint;
	axesPoint.push_back(cv::Point3f(0, 0, 0)); 
	axesPoint.push_back(cv::Point3f(0.1, 0, 0));
	axesPoint.push_back(cv::Point3f(0, 0.1, 0));
	axesPoint.push_back(cv::Point3f(0, 0, 0.1));

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

	// Checking if rvec and tvec are found
	if(rvec.size() == 0 || tvec.size() == 0)
		return false;
	
	cv::Mat ArucoVideoFrameCopy = ArucoVideoFrame.clone();

	for (int i = 0 ; i < IDs.size() ; i++)
	{
		std::vector<cv::Point2f> imagePoints;
		cv::projectPoints(axesPoint, rvec[i], tvec[i], mtx, dist, imagePoints);
		cv::aruco::drawAxis(ArucoVideoFrameCopy, mtx, dist, rvec[i], tvec[i], 0.1);

		// storing bottom and top point of line showing z-axis
		BottomVertices.push_back(imagePoints[0]);
		TopVertices.push_back(imagePoints[3]);
	}
	//cv::imshow("ArucoAxisDisplay", ArucoVideoFrameCopy);
	return true;
}


bool SetCubeVertices(std::vector<cv::Point2f> BottomVertices, std::vector<cv::Point2f> TopVertices, std::vector<int> &IDs, std::vector<std::vector<int>> &CubeVertices)
{	
	int ArrangedBottomVertices[4][2], ArrangedTopVertices[4][2];
	
	for (int i = 0 ; i < 4 ; i++)
	{
		ArrangedBottomVertices[IDs[i]][0] = int(round(BottomVertices[i].x));
		ArrangedBottomVertices[IDs[i]][1] = int(round(BottomVertices[i].y));
		ArrangedTopVertices[IDs[i]][0] = int(round(TopVertices[i].x));
		ArrangedTopVertices[IDs[i]][1] = int(round(TopVertices[i].y));
	}
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


bool CallForOverlapping(cv::Mat ArucoVideoFrame, cv::Mat VideoFramesTO[5], std::vector<std::vector<int>> CubeVertices, cv::Mat &FinalFrame)
{
	int EdgeCentersYCoordinate[4] = {(CubeVertices[0][1] + CubeVertices[1][1])/2,
							         (CubeVertices[1][1] + CubeVertices[2][1])/2,
							  		 (CubeVertices[2][1] + CubeVertices[3][1])/2,
							    	 (CubeVertices[3][1] + CubeVertices[0][1])/2};

	
	int SortedEdgeCenters[4];
	std::copy(std::begin(EdgeCentersYCoordinate), std::end(EdgeCentersYCoordinate), std::begin(SortedEdgeCenters));
	
	std::sort(SortedEdgeCenters, SortedEdgeCenters+4);

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
	

	std::vector<std::vector<cv::Point2f>> Vertices;
	std::vector<cv::Point2f> Temp1, Temp2, Temp3, Temp4, Temp5;
	Temp1.push_back(cv::Point2f(CubeVertices[5][0], CubeVertices[5][1]));
	Temp1.push_back(cv::Point2f(CubeVertices[4][0], CubeVertices[4][1]));
	Temp1.push_back(cv::Point2f(CubeVertices[0][0], CubeVertices[0][1]));
	Temp1.push_back(cv::Point2f(CubeVertices[1][0], CubeVertices[1][1]));
	Vertices.push_back(Temp1);
	Temp2.push_back(cv::Point2f(CubeVertices[6][0], CubeVertices[6][1]));
	Temp2.push_back(cv::Point2f(CubeVertices[5][0], CubeVertices[5][1]));
	Temp2.push_back(cv::Point2f(CubeVertices[1][0], CubeVertices[1][1]));
	Temp2.push_back(cv::Point2f(CubeVertices[2][0], CubeVertices[2][1]));
	Vertices.push_back(Temp2);
	Temp3.push_back(cv::Point2f(CubeVertices[7][0], CubeVertices[7][1]));
	Temp3.push_back(cv::Point2f(CubeVertices[6][0], CubeVertices[6][1]));
	Temp3.push_back(cv::Point2f(CubeVertices[2][0], CubeVertices[2][1]));
	Temp3.push_back(cv::Point2f(CubeVertices[3][0], CubeVertices[3][1]));
	Vertices.push_back(Temp3);
	Temp4.push_back(cv::Point2f(CubeVertices[4][0], CubeVertices[4][1]));
	Temp4.push_back(cv::Point2f(CubeVertices[7][0], CubeVertices[7][1]));
	Temp4.push_back(cv::Point2f(CubeVertices[3][0], CubeVertices[3][1]));
	Temp4.push_back(cv::Point2f(CubeVertices[0][0], CubeVertices[0][1]));
	Vertices.push_back(Temp4);
	Temp5.push_back(cv::Point2f(CubeVertices[4][0], CubeVertices[4][1]));
	Temp5.push_back(cv::Point2f(CubeVertices[5][0], CubeVertices[5][1]));
	Temp5.push_back(cv::Point2f(CubeVertices[6][0], CubeVertices[6][1]));
	Temp5.push_back(cv::Point2f(CubeVertices[7][0], CubeVertices[7][1]));
	Vertices.push_back(Temp5);

	FinalFrame = ArucoVideoFrame.clone();

	for (int i = 0 ; i < 5 ; i++)
		FinalFrame = OverlapImage(FinalFrame, VideoFramesTO[Order[i]], Vertices[Order[i]]);
		
	return true;
}


int main()
{
	ReadCalibrationParams();
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
		bool Break = false;
		for(int i = 0 ; i < SizeOfCapList ; i++)
			if(!CapList[i]->isOpened()) Break = true;
		if(Break)
		{
			std::cout << "Not able to read video.\n";
			break;
		}

		cv::Mat FrameList[6], OverlapVideoFrameList[5];
		for(int i = 0 ; i < SizeOfCapList ; i++)
		{
			cv::Mat Frame;
			CapList[i]->read(Frame);
			if(Frame.empty())
			{
				CapList[i]->set(cv::CAP_PROP_POS_FRAMES, 0);
				CapList[i]->read(Frame);
			}
			FrameList[i] = Frame;
		}

		for (int i = 0 ; i < (sizeof(FrameList)/sizeof(FrameList[0])) ; i++)
			cv::resize(FrameList[i], FrameList[i], cv::Size(FrameSize[0], FrameSize[1]));

		cv::Mat ArucoVideoFrame = FrameList[0];
		std::copy(FrameList + 1, FrameList + 6, OverlapVideoFrameList + 0);

		std::vector<int> IDs;
		std::vector<cv::Point2f> BottomVertices, TopVertices;
		bool Ret1 = DetectAruco_FindVertices(ArucoVideoFrame, IDs, BottomVertices, TopVertices);		
		if(!Ret1) continue;

		std::vector<std::vector<int>> CubeVertices;
		bool Ret2 = SetCubeVertices(BottomVertices, TopVertices, IDs, CubeVertices);
		if(!Ret2) continue;

		cv::Mat FinalFrame;
		bool Ret3 = CallForOverlapping(ArucoVideoFrame, OverlapVideoFrameList, CubeVertices, FinalFrame);
		if (!Ret3) continue;
		
		cv::imshow("FinalFrame", FinalFrame);
		if ((cv::waitKey(1) & 0xFF) == 'q')
			break;
	}
	
	for (int i = 0 ; i < SizeOfCapList ; i++)
		CapList[i]->release();

	return 0;
}