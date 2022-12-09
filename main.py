import face_recognition as fr
import cv2
import numpy as np
import os

path = "./train/"

known_names = []    # list that store the names of the images
known_name_encodings = []    # list that store the respective face encodings.

images = os.listdir(path)   # list containing the names of the entries in the directory given by path
for _ in images:
    image = fr.load_image_file(path + _)    # load the training images
    image_path = path + _
    encoding = fr.face_encodings(image)[0]  # stores the face encoding of training image

    known_name_encodings.append(encoding)
    known_names.append(os.path.splitext(os.path.basename(image_path))[0].capitalize())

print(known_names)

test_image = "./test/test.jpg"
image = cv2.imread(test_image)  # for reading an image
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

face_locations = fr.face_locations(image)   # locates the coordinates (left, bottom, right, top) of every face detected in the testing image
face_encodings = fr.face_encodings(image, face_locations)   # face encoding of each face in testing image

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = fr.compare_faces(known_name_encodings, face_encoding)     # compare encodings of two photos and see if they are the same person
    name = ""

    face_distances = fr.face_distance(known_name_encodings, face_encoding)      # calculate the similarity between the encoding of the test image and that of the train images
    best_match = np.argmin(face_distances)      # returns indices of the min element of the array in a particular axis.

    if matches[best_match]:
        name = known_names[best_match]

    cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.rectangle(image, (left, bottom - 15), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


cv2.imshow("Result", image)     # display an testing image in a window
cv2.imwrite("./output.jpg", image)  # save an image to any storage device. This will save the image according to the specified format in current working directory.
cv2.waitKey(0)  # allows users to display a window until any key is pressed
cv2.destroyAllWindows() # allows users to destroy all windows at any time
