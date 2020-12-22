"""
Use IP camera application in your mobile device and get the IP address, enter that into the main function. 
GUIDE : 
Start running the program.
* Enter SPACEBAR to capture images. 
* Enter ESC to stop capturing
* Program will now display the images captured one by one.
* Mark the corners. (Only 4). Right click to start over. Q to not crop. D to delete.
* Press ESC after marking. Repeat the same for all the images captured.
* Once finished, the cropped versions and will be in the same folder the code is saved in
"""

import os
import numpy
import cv2 as cv2
from urllib.request import urlopen
import configparser
import imutils
from PIL import Image
from threading import Thread

refPt = []
width_img = 0
height_img = 0
resize_ratio = 0
desired_preview_width = 640

p_url = 'http://192.168.86.183:8080/photo.jpg'
v_url = 'http://192.168.86.183:8080/video'

mark_str = "MARK THE CORNERS"


class ThreadedCamera(object):
    def __init__(self, source=""):
        print("Connecting to video feed...")
        self.capture = cv2.VideoCapture(source)
        print("Connection successful.")

        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

        self.status = False
        self.frame = None

    def update(self):
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()

    def grab_frame(self):
        if self.status:
            return self.frame
        return None


def display_im(name, im):
    resized_im = cv2.resize(im, (int(im.shape[1] * resize_ratio), desired_preview_width))
    return cv2.imshow(name, resized_im)


# function that stores the coordinates of the points clicked for marking the edges
def click_event(event, x, y, flags, param):
    global refPt, img
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (int(x / resize_ratio), int(y / resize_ratio)), 25, (255, 0, 0), -1)
        refPt.append([int(x / resize_ratio), int(y / resize_ratio)])
        display_im(mark_str, img)
    elif event == cv2.EVENT_RBUTTONDOWN:
        refPt = []
        img = org.copy()
        display_im(mark_str, org)
    elif event == cv2.EVENT_MBUTTONDOWN:
        refPt = []
        img = org.copy()
        display_im(mark_str, org)


# function that takes the image and outputs the cropped image
def crop(original, im):
    global refPt, img
    display_im(mark_str, org)
    print("Mark the corners. (Only 4). Right click to start over. Q to not crop. D to delete. Press escape when done. ")
    while True:
        cv2.setMouseCallback(mark_str, click_event)
        kc = cv2.waitKey(0)
        # Escape key, attempt crop
        if kc == 27:
            pass
        # Q Key, Don't crop
        elif kc == 113:
            print(original)
            return original.copy()
        # D Key, delete photo
        elif kc == 100:
            print("Press E to confirm.")
            kc2 = cv2.waitKey(0)
            # Not E Key, Cancel deletion
            if kc2 != 101:
                continue
            return None

        else:
            continue
        # print(refPt)
        cv2.destroyAllWindows()
        photo_contour = org.copy()

        try:
            contour = [numpy.array(refPt).astype(int)]
            cv2.drawContours(photo_contour, contour, -1, (0, 255, 0), 6)

            pts1 = numpy.float32(contour)
            pts2 = numpy.float32([[0, 0], [width_img, 0], [width_img, height_img], [0, height_img]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
        except cv2.error:
            print("More or less than four corners detected, retry.")
            refPt = []
            img = org.copy()
            display_im(mark_str, org)
            continue
        photo_cropped = cv2.warpPerspective(original, matrix, (width_img, height_img))
        return photo_cropped


# function that saves the scanned image to the folder
def save_cropped(cropped, count):
    img_name = "Photo_{}_{}.png".format(batch, count)
    cv2.imwrite(img_name, cropped)
    print("{} written!".format(img_name))
    count += 1
    return img_name, count


# function that saves the image from the webcam to the folder
def capture():
    req = urlopen(p_url)

    global width_img, height_img, resize_ratio
    img_name = "raw/image_{}_{}.png".format(batch, conf["config"]["count"])

    arr = numpy.asarray(bytearray(req.read()), dtype=numpy.uint8)
    photo = cv2.imdecode(arr, -1)  # 'Load it as it is'

    cv2.imwrite(img_name, photo)
    print("{} written!".format(img_name))
    # These will be the same for all photos taken, beware.
    width_img = photo.shape[1]
    height_img = photo.shape[0]
    resize_ratio = 1 / (photo.shape[0] / 640)

    display_im("Latest Saved Image", photo)
    conf["config"]["count"] = str(int(conf["config"]["count"]) + 1)
    return img_name


def save_ini():
    with open("conf.ini", "w+") as f:
        conf.write(f)


# main function
if __name__ == "__main__":
    if not os.path.exists('raw'):
        os.makedirs('raw')

    # enter ip here
    conf = configparser.ConfigParser()
    conf.read("conf.ini")
    batch = conf["config"]["batch"]

    streamer = ThreadedCamera(v_url)
    # cap = cv2.VideoCapture(v_url)

    image_list = []

    while True:
        frame = streamer.grab_frame()
        if frame is not None:
            video = cv2.resize(frame, (480, 640))
            cv2.imshow("Camera Preview", video)

        q = cv2.waitKey(1)

        # Press SPACE to capture
        if q == 32:
            image_name = capture()

            image_list.append(image_name)

        # Press ESC to Crop the captured images
        if q == 27:
            cv2.destroyAllWindows()
            count = 0

            for i in image_list:
                refPt = []
                img = cv2.imread(i)

                org = img.copy()
                img = crop(org, img)
                if img is not None:
                    cropName, count = save_cropped(img, count)

            break

    conf["config"]["batch"] = str(int(conf["config"]["batch"]) + 1)
    conf["config"]["count"] = "0"
    save_ini()
