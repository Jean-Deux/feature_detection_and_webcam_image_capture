import cv2
import os
import pickle

#####################################################
with open('./feature_detection_configuration.txt') as file:
    for line in file:
        exec(line)

resize_width = train_image_width
resize_height = train_image_height
training_images_path = training_images_Path
descriptor_match_threshold = descriptor_match_Threshold
confidence = Confidence
n_orb_features = n_orb_Features
pickle_file_path = pickle_file_Path

#####################################################

# use OFRB, initialise detector
# change default to 1000 features
orb = cv2.ORB_create(nfeatures=n_orb_features)


# CREATE A FUNCTION THAT WILL FIND ALL DESCRIPTORS
def find_des(images):  # pass in a list of images
    des_list = []  # create a list that will store all descriptors
    for image in images:
        kp, des = orb.detectAndCompute(image, None)
        des_list.append(des)
    return des_list


# automatically read images from folders
path = training_images_path

# create lists to store image params
image_list = []  # list to store images
classNames = []  # list to store names of images

# start of OS block
# get images from a folder and append them as a list in image_list as well as their names in classNames
myList = os.listdir(path)
print(myList)
print('total classes detected: ', len(myList))

for cl in myList:
    img_current = cv2.imread(f'{path}/{cl}', 0)  # import image
    # optional process image block will go here
    img_current = cv2.resize(img_current, (resize_width, resize_height))
    # end of optional image processing block
    image_list.append(img_current)  # append the images to the list of images
    classNames.append(os.path.splitext(cl)[0])  # store name without file extension
print(classNames)
# end of OS block

# use ORB to find descriptors in all images
descriptor_list = find_des(image_list)
print(len(descriptor_list))
# end of descriptor block

# start of pickle block to pickle the descriptor_list
# open a file (or create a new one), wb = write bytes
with open(pickle_file_path, 'wb') as f:
    pickle.dump(descriptor_list, f)
    pickle.dump(classNames, f)
# end of pickle block

print('feature detector trained successfully')
