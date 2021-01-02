import cv2
import os
import time

#####################################################
with open('./capture_configuration.txt') as file:
    for line in file:
        exec(line)
myPath = my_Path
cameraNo = camera_No
cameraBrightness = camera_Brightness
moduleVal = module_Val
minBlur = min_Blur
grayImage = gray_Image
saveData = save_Data
showImage = show_Image
imgWidth = img_Width
imgHeight = img_Height
programmeCloseKey = programme_close_key

#####################################################

global countFolder
cap = cv2.VideoCapture(cameraNo)
cap.set(3, imgWidth)
cap.set(4, imgHeight)
cap.set(10, cameraBrightness)

count = 0
countSave = 0


def saveDataFunc():  # this function automatically creates a new folder everytime this code is run to save data in
    global countFolder
    countFolder = 0
    while os.path.exists(myPath + str(countFolder)):
        countFolder += 1
    os.makedirs(myPath + str(countFolder))


if saveData: saveDataFunc()

while True:

    success, img = cap.read()
    image = img.copy()
    img = cv2.resize(img, (imgWidth, imgHeight))
    if grayImage: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if saveData:
        blur = cv2.Laplacian(img, cv2.CV_64F).var()
        if count % moduleVal == 0 and blur > minBlur:
            nowTime = time.time()
            cv2.imwrite(myPath + str(countFolder) +
                        '/' + str(countSave) + "_" + str(int(blur)) + "_" + str(nowTime) + ".png", img)
            countSave += 1
            cv2.putText(image, str(countSave), (10, 50), cv2.FONT_ITALIC, 1.5, (0, 0, 255), 2)
        count += 1

    if showImage:
        cv2.imshow("Press " + programmeCloseKey + " to exit Cascade image capture", image)

    if cv2.waitKey(1) & 0xFF == ord(programmeCloseKey):
        break

cap.release()
cv2.destroyAllWindows()
