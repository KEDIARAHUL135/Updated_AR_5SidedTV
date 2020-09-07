import cv2
import cv2.aruco as aruco
import numpy as np
import Calibrate as C


def ReadCalibrationParams():
	calibration_file = cv2.FileStorage('calibration.yaml', cv2.FILE_STORAGE_READ)
	ret = calibration_file.getNode('ret').real()
	mtx = calibration_file.getNode('camera_matrix').mat()
	dist = calibration_file.getNode('distortion_coefficients').mat()
	rvecs = calibration_file.getNode('rotation_vectors').mat()
	tvecs = calibration_file.getNode('translation_vectors').mat()
	calibration_file.release()
	
	return ret, mtx, dist, rvecs, tvecs#list(rvecs), list(tvecs)


ret, mtx, dist, rvecs, tvecs = ReadCalibrationParams()
FrameSize = (1024, 576)


def DetectAruco_FindVertices(ArucoVideoFrame):
	BottomVertices = []
	TopVertices = []
	axesPoint = np.float32([[0, 0, 0], 
							[0.25, 0, 0],
							[0, 0.25, 0],
							[0, 0, 0.25]])

	GrayImage = cv2.cvtColor(ArucoVideoFrame,cv2.COLOR_BGR2GRAY)
	ArucoDict = aruco.Dictionary_get(aruco.DICT_6X6_50)
	Parameters = aruco.DetectorParameters_create()
	Corners, IDs, RejectedImgPoints = aruco.detectMarkers(GrayImage, ArucoDict, parameters=Parameters)

	if IDs is None:			# If no aruco marker found
		return None, None, None

	# Estimating pose of aruco markers
	rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(Corners, 0.1, mtx, dist)

	# Checking if rvec and tvec are found
	if (rvec-tvec).any() is None:
		return None, None, None

	ArucoVideoFrameCopy = ArucoVideoFrame.copy()

	for i in range(0, IDs.size):
		imagePoints = cv2.projectPoints(axesPoint, rvec[i], tvec[i], mtx, dist)
		aruco.drawAxis(ArucoVideoFrameCopy, mtx, dist, rvec[i], tvec[i], 0.1)

		# storing bottom and top point of line showing z-axis
		BottomVertices.append([imagePoints[0][0][0][0], imagePoints[0][0][0][1]])
		TopVertices.append([imagePoints[0][3][0][0], imagePoints[0][3][0][1]])

	cv2.imshow("ArucoAxisDisplay", ArucoVideoFrameCopy)
	return BottomVertices, TopVertices, IDs


def SetCubeVertices(BottomVertices, TopVertices, IDs):
	CubeVertices = []
	ArrangedBottomVertices = np.empty([4, 2], dtype=int)
	ArrangedTopVertices = np.empty([4, 2], dtype=int)
	
	for i in range(4):
		ArrangedBottomVertices[IDs[i]] = [int(round(BottomVertices[i][0])), int(round(BottomVertices[i][1]))]
		ArrangedTopVertices[IDs[i]] = [int(round(TopVertices[i][0])), int(round(TopVertices[i][1]))]

	for i in range(4):
		if 0 <= ArrangedBottomVertices[i][0] < FrameSize[0] and 0 <= ArrangedBottomVertices[i][1] < FrameSize[1]:
			CubeVertices.append(ArrangedBottomVertices[i])
		else:
			return None
	for i in range(4):
		if 0 <= ArrangedTopVertices[i][0] < FrameSize[0] and 0 <= ArrangedTopVertices[i][1] < FrameSize[1]:
			CubeVertices.append(ArrangedTopVertices[i])
		else:
			return None

	return np.asarray(CubeVertices)


def ProjectiveTransform(FrameToBeOverlaped, ArucoPoint):
	Height, Width = FrameToBeOverlaped.shape[:2]
	InitialPoints = np.float32([[0, 0], [Width-1, 0], [0, Height-1], [Width-1, Height-1]])
	FinalPoints = np.float32([[ArucoPoint[0][0], ArucoPoint[0][1]], 
							  [ArucoPoint[1][0], ArucoPoint[1][1]], 
							  [ArucoPoint[3][0], ArucoPoint[3][1]], 
							  [ArucoPoint[2][0], ArucoPoint[2][1]]])

	ProjectiveMatrix = cv2.getPerspectiveTransform(InitialPoints, FinalPoints)
	TransformedFrame = cv2.warpPerspective(FrameToBeOverlaped, ProjectiveMatrix, (Width, Height))
	
	return TransformedFrame, FinalPoints


def OverlapImage(ArucoVideoFrame, FrameToBeOverlaped, ArucoPoint):
	ArucoPoint = ArucoPoint.astype(float)

	TransformedFrame, FinalPoints = ProjectiveTransform(FrameToBeOverlaped, ArucoPoint)

	MaskArucoVideoFrame = np.zeros(ArucoVideoFrame.shape, dtype=np.uint8)
	cv2.fillConvexPoly(MaskArucoVideoFrame, ArucoPoint.astype(np.int32), (255, )*ArucoVideoFrame.shape[2])
	TransformedFrame = cv2.bitwise_and(MaskArucoVideoFrame, TransformedFrame)
	MaskArucoVideoFrame = cv2.bitwise_not(MaskArucoVideoFrame)
	BlackFrameForOverlap = cv2.bitwise_and(ArucoVideoFrame, MaskArucoVideoFrame)
	FinalImage = cv2.bitwise_or(TransformedFrame, BlackFrameForOverlap)

	return FinalImage


def CallForOverlapping(ArucoVideoFrame, VideoFramesTO, CubeVertices):
	Vertices = np.zeros([4, 2])
	EdgeCentersYCoordinate = [(CubeVertices[0][1] + CubeVertices[1][1])//2,
							  (CubeVertices[1][1] + CubeVertices[2][1])//2,
							  (CubeVertices[2][1] + CubeVertices[3][1])//2,
							  (CubeVertices[3][1] + CubeVertices[0][1])//2]

	SortedEdgeCenters = EdgeCentersYCoordinate.copy()
	SortedEdgeCenters.sort()

	Order = []
	for Value in SortedEdgeCenters:
		for i in range(4):
			if Value == EdgeCentersYCoordinate[i]:
				if not Order.count(i):
					Order.append(i)
					break
	Order.append(4)
	
	Vertices = [[CubeVertices[5], CubeVertices[4], CubeVertices[0], CubeVertices[1]],
				[CubeVertices[6], CubeVertices[5], CubeVertices[1], CubeVertices[2]],
				[CubeVertices[7], CubeVertices[6], CubeVertices[2], CubeVertices[3]],
				[CubeVertices[4], CubeVertices[7], CubeVertices[3], CubeVertices[0]],
				[CubeVertices[4], CubeVertices[5], CubeVertices[6], CubeVertices[7]]]

	Vertices = np.asarray(Vertices)

	Frame = ArucoVideoFrame.copy()
	for i in range(5):
		Frame = OverlapImage(Frame, VideoFramesTO[Order[i]], Vertices[Order[i]])
	
	return Frame


def main():
	# Dictionary - DICT_6X6_50 is used.
	# Reading video
	CapList = [cv2.VideoCapture(0),#'Videos/ArucoVideo1.avi'),
			   cv2.VideoCapture('Videos/Video1.avi'),
			   cv2.VideoCapture('Videos/Video2.avi'),
			   cv2.VideoCapture('Videos/Video3.avi'),
			   cv2.VideoCapture('Videos/Video4.avi'),
			   cv2.VideoCapture('Videos/Video5.avi')]

	while True:
		Break = False
		for Cap in CapList:
			if not Cap.isOpened():
				Break = True
		if Break:
			print("Not able to read video.")
			break

		FrameList = []
		for i in range(len(CapList)):
			Ret, Frame = CapList[i].read()
			if Ret is False:
				CapList[i].set(cv2.CAP_PROP_POS_FRAMES, 0)
				Ret, Frame = CapList[i].read()
			FrameList.append(Frame)

		for i in range(len(FrameList)):
			FrameList[i] = cv2.resize(FrameList[i], FrameSize)

		ArucoVideoFrame = FrameList[0]
		OverlapVideoFrameList = FrameList[1:]

		BottomVertices, TopVertices, IDs = DetectAruco_FindVertices(ArucoVideoFrame)

		if IDs is None:
			continue
		if len(IDs) != 4:
			continue

		CubeVertices = SetCubeVertices(BottomVertices, TopVertices, IDs)
		if CubeVertices is None:
			continue

		FinalFrame = CallForOverlapping(ArucoVideoFrame, OverlapVideoFrameList, CubeVertices)
	
		cv2.imshow("ArucoVideo", ArucoVideoFrame)
		cv2.imshow("FinalFrame", FinalFrame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	for Cap in CapList:
		Cap.release()

	cv2.destroyAllWindows()
