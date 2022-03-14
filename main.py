import cv2
import cv2.aruco as aruco
import numpy as np
import os

def loadAugmentedImages(path):
    list = os.listdir(path)
    noOfMarkers = len(list)
    print("Total no. of markers = ", noOfMarkers)
    imgAugDict = {}
    for imgPath in list:
        if imgPath == "default.png":
            continue
        key = int(os.path.splitext(imgPath)[0])
        imgAug = cv2.imread(f'{path}/{imgPath}')
        imgAugDict[key] = imgAug
    return imgAugDict

def findArucoMarkers(img, markerSize=6, totalMarkers=250, draw=True):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    key = getattr(aruco, f'DICT_{markerSize}X{markerSize}_{totalMarkers}')
    arucoDict = aruco.Dictionary_get(key)
    arucoParam = aruco.DetectorParameters_create()
    bboxes, ids, rejected = aruco.detectMarkers(imgGray, arucoDict, parameters=arucoParam)
    # print(ids)
    if draw:
        aruco.drawDetectedMarkers(img, bboxes)
    return [bboxes, ids]

def augmentAruco(bbox, id, img, imgAug, drawId=True):
    topLeft = bbox[0][0][0], bbox[0][0][1]
    topRight = bbox[0][1][0], bbox[0][1][1]
    bottomRight = bbox[0][2][0], bbox[0][2][1]
    bottomLeft = bbox[0][3][0], bbox[0][3][1]
    h, w, c = imgAug.shape
    pts1 = np.array([topLeft, topRight, bottomRight, bottomLeft]).astype(int)
    pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    matrix, _ = cv2.findHomography(pts2, pts1)
    imgOut = cv2.warpPerspective(imgAug, matrix, (img.shape[1], img.shape[0]))  # width, height
    cv2.fillConvexPoly(img, pts1, (0, 0, 0))    # black
    imgOut = imgOut + img
    if drawId:
        cv2.putText(imgOut, str(id), pts1[0], cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    return imgOut

def main():
    cap = cv2.VideoCapture(0)
    imgAugDict = loadAugmentedImages("Markers")
    while True:
        success, img = cap.read()
        arucoFound = findArucoMarkers(img)
        if len(arucoFound[0]) != 0:
            for bbox, id in zip(arucoFound[0], arucoFound[1]):
                if int(id) in imgAugDict.keys():
                    imgAug = imgAugDict[int(id)]
                else:
                    imgAug = cv2.imread("Markers/default.png")
                img = augmentAruco(bbox, id, img, imgAug)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()