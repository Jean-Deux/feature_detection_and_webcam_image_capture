import cv2
import pickle


#####################################################
with open('./feature_detection_configuration.txt') as file:
    for line in file:
        exec(line)
cameraNo = camera_No
cameraBrightness = camera_Brightness
imgWidth = img_Width
imgHeight = img_Height
pickle_file_path = pickle_file_Path
descriptor_match_threshold = descriptor_match_Threshold
confidence = Confidence
n_orb_features = n_orb_Features
k_val = k_Val
match_colour = match_text_Colour
match_text_position = match_text_Position
match_text_size = match_text_Size
match_text_thick = match_text_thickness
programmeCloseKey = programme_close_key
#####################################################


# capture from webcam
cap = cv2.VideoCapture(cameraNo)
cap.set(3, imgWidth)
cap.set(4, imgHeight)
cap.set(10, cameraBrightness)

# initialise empty list just in case pickle fails
descriptor_list = []
classNames = []

# read the pickle file, rb = read bytes
with open(pickle_file_path, 'rb') as f:
    descriptor_list = pickle.load(f)  # let it redefine the list
    classNames = pickle.load(f)  # just in case the pickle file isn't there

# use OFRB, initialise detector
# change default to 1000 features
orb = cv2.ORB_create(nfeatures=n_orb_features)


# create a function that converts the webcam image to descriptors
# and compares it to the list of descriptors from the image_list
def find_id(web_image, descriptor_list):
    kp2, des2 = orb.detectAndCompute(web_image, None)
    # define the matcher
    bf = cv2.BFMatcher()
    # define a list to store the match value
    match_list = []
    # define a variable that will basically send out the index of the image in image list with the highest match
    final_value = -1  # cannot use 0 as the zeroth element in the list is alr defined
    try:  # have this statement in case descriptor messes up
        for descriptor in descriptor_list:
            matches = bf.knnMatch(descriptor, des2, k=k_val)  # k is the number of values to match
            good = []
            for m, n in matches:
                if m.distance < (descriptor_match_threshold * n.distance):  # 0.75 is a good approx, can tune
                    good.append([m])
            # can use this to determine what image is what based on its match strength
            match_list.append(len(good))  # append the number of good matches to that list
    except:
        pass
    print(match_list)
    if len(match_list) != 0:
        if max(match_list) > confidence:
            final_value = match_list.index(max(match_list))  # find the max value and the index

    return final_value


while True:
    success, img2 = cap.read()
    img_original = img2.copy()
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # convert webcam image to grayscale
    id = find_id(img2, descriptor_list)
    if id != -1:
        cv2.putText(img_original, classNames[id], match_text_position, cv2.FONT_HERSHEY_PLAIN, match_text_size,
                    match_colour, match_text_thick)

    cv2.imshow("Feature Detector Running! Press " + programmeCloseKey + " to exit", img_original)
    if cv2.waitKey(1) & 0xFF == ord(programmeCloseKey):
        break

cap.release()
cv2.destroyAllWindows()
