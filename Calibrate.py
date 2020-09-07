import numpy as np
import cv2
import glob
import cv2.aruco as aruco
import time

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)
ArucoDict = aruco.Dictionary_get(aruco.DICT_6X6_250)
Board = aruco.CharucoBoard_create(10, 7, 1, .8, ArucoDict)
ImageBoard = Board.draw((1024, 576))
cv2.imwrite("chessboard.tiff", ImageBoard)
#cv2.imshow("Board", ImageBoard)
#cv2.waitKey(0)

def WriteImages():
    Cap = cv2.VideoCapture(0)
    ImageCount = 1

    while Cap.isOpened():
        Return, Image = Cap.read()
        if Return is False:
            continue

        cv2.imshow("InputVideo", Image)
        cv2.waitKey(1)

        ImageName = "CalibrateImages/" + str(ImageCount) + ".png"

        Gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
        Corners, IDs, _ = cv2.aruco.detectMarkers(Gray, ArucoDict)
        
        if len(Corners)>0:
            # SUB PIXEL DETECTION
            for Corner in Corners:
                cv2.cornerSubPix(Gray, Corner, 
                                 winSize = (3,3), 
                                 zeroZone = (-1,-1), 
                                 criteria = criteria)
            res2 = cv2.aruco.interpolateCornersCharuco(Corners,IDs,Gray,Board)        
            if res2[1] is not None and res2[2] is not None and len(res2[1])>3:
                cv2.imwrite(ImageName, Image)
                ImageCount += 1

        time.sleep(1)

        if ImageCount >= 25:
            break


def read_chessboards():
    """
    Charuco base pose estimation.
    """
    images = glob.glob('CalibrateImages_PhoneCam/*.jpg')
    #print("POSE ESTIMATION STARTS:")
    allCorners = []
    allIds = []
    decimator = 0
    # SUB PIXEL CORNER DETECTION CRITERION
    
    for im in images:
        #print("=> Processing image {0}".format(im))
        frame = cv2.imread(im)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, ArucoDict)
        
        if len(corners)>0:
            # SUB PIXEL DETECTION
            for corner in corners:
                cv2.cornerSubPix(gray, corner, 
                                 winSize = (3,3), 
                                 zeroZone = (-1,-1), 
                                 criteria = criteria)
            res2 = aruco.interpolateCornersCharuco(corners,ids,gray,Board)        
            if res2[1] is not None and res2[2] is not None and len(res2[1])>3 and decimator%1==0:
                allCorners.append(res2[1])
                allIds.append(res2[2])              
        
        decimator+=1   

    imsize = gray.shape
    return allCorners,allIds,imsize

def calibrate_camera(allCorners,allIds,imsize):   
    """
    Calibrates the camera using the detected corners.
    """
    #print("CAMERA CALIBRATION")
    
    cameraMatrixInit = np.array([[ 1000.,    0., imsize[0]/2.],
                                 [    0., 1000., imsize[1]/2.],
                                 [    0.,    0.,           1.]])

    distCoeffsInit = np.zeros((5,1))
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO) 
    #flags = (cv2.CALIB_RATIONAL_MODEL) 
    (ret, camera_matrix, distortion_coefficients0, 
     rotation_vectors, translation_vectors,
     stdDeviationsIntrinsics, stdDeviationsExtrinsics, 
     perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
                      charucoCorners=allCorners,
                      charucoIds=allIds,
                      board=Board,
                      imageSize=imsize,
                      cameraMatrix=cameraMatrixInit,
                      distCoeffs=distCoeffsInit,
                      flags=flags,
                      criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))

    return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors


def SaveCalibrationParams(ret, camera_matrix, distortion_coefficients, rotation_vectors, translation_vectors):
    rotation_vectors = np.asarray(rotation_vectors)
    translation_vectors = np.asarray(translation_vectors)

    calibration_file = cv2.FileStorage('calibration.yaml', cv2.FILE_STORAGE_WRITE)
    calibration_file.write("ret ", ret)
    calibration_file.write("camera_matrix", camera_matrix)
    calibration_file.write("distortion_coefficients ", distortion_coefficients)
    calibration_file.write("rotation_vectors ", rotation_vectors)
    calibration_file.write("translation_vectors ", translation_vectors)
    calibration_file.release()


def CalibrateMain():
    #WriteImages()

    allCorners,allIds,imsize = read_chessboards()
    ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors = calibrate_camera(allCorners,allIds,imsize)

    #print("{}\n\n{}\n\n{}\n\n{}\n\n{}".format(ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors))

    SaveCalibrationParams(ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors)
    return ret, camera_matrix, distortion_coefficients0, rotation_vectors, translation_vectors
