#!/usr/bin/env python2

import time
import os
start = time.time()
import argparse
import cv2
import os
import pickle
import sys
import numpy as np
np.set_printoptions(precision=2)
from sklearn.mixture import GMM
import openface
import threading
from websocket_server import WebsocketServer
DETECTION_WIDTH = 240
DETECTION_HEIGHT = 135
cam_dect_name = None
cam_confidences = None
fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '', 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')
import dlib
detector = dlib.get_frontal_face_detector()


def getRep(bgrImg):
    start = time.time()
    if bgrImg is None:
        raise Exception("Unable to load image/frame")
    rgbImg = cv2.resize(
        cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB), (DETECTION_WIDTH, DETECTION_HEIGHT), interpolation=cv2.INTER_CUBIC)
    gray = cv2.resize(
        cv2.cvtColor(bgrImg, cv2.COLOR_BGR2GRAY), (DETECTION_WIDTH, DETECTION_HEIGHT), interpolation=cv2.INTER_CUBIC)
    if args.verbose:
        print("  + Original size: {}".format(rgbImg.shape))
    if args.verbose:
        print("Loading the image took {} seconds.".format(time.time() - start))
    start = time.time()
    bb = detector(gray, 0)
    if bb is None:
        return None
    if args.verbose:
        print("Face detection took {} seconds.".format(time.time() - start))
    start = time.time()
    alignedFaces = []
    for box in bb:
        alignedFaces.append(
            align.align(
                args.imgDim,
                rgbImg,
                box,
                landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE))
    if alignedFaces is None:
        raise Exception("Unable to align the frame")
    if args.verbose:
        print("Alignment took {} seconds.".format(time.time() - start))
    start = time.time()
    reps = []
    for alignedFace in alignedFaces:
        reps.append(net.forward(alignedFace))
    if args.verbose:
        print("Neural network forward pass took {} seconds.".format(
            time.time() - start))
    # print (reps)
    return (reps,bb)


def infer(img, args):
    with open(args.classifierModel, 'rb') as f:
        if sys.version_info[0] < 3:
                (le, clf) = pickle.load(f)  # le - label and clf - classifer
        else:
                #set_trace()

                (le, clf) = pickle.load(f, encoding='latin1')  # le - label and clf - classifer

    repsAndBBs = getRep(img)
    reps = repsAndBBs[0]
    bbs = repsAndBBs[1]
    persons = []
    confidences = []
    for rep in reps:
        try:
            rep = rep.reshape(1, -1)
        except:
            print ("No Face detected")
            return (None, None)
        start = time.time()
        predictions = clf.predict_proba(rep).ravel()
        # print (predictions)
        maxI = np.argmax(predictions)
        # max2 = np.argsort(predictions)[-3:][::-1][1]
        persons.append(le.inverse_transform(maxI))
        # print (str(le.inverse_transform(max2)) + ": "+str( predictions [max2]))
        # ^ prints the second prediction
        confidences.append(predictions[maxI])
        if args.verbose:
            print("Prediction took {} seconds.".format(time.time() - start))
            pass
        # print("Predict {} with {:.2f} confidence.".format(person.decode('utf-8'), confidence))
        if isinstance(clf, GMM):
            dist = np.linalg.norm(rep - clf.means_[maxI])
            print("  + Distance from the mean: {}".format(dist))
            pass
    return (persons, confidences ,bbs)


def cam_dect():
    global cam_dect_name
    # Capture device. Usually 0 will be webcam and 1 will be usb cam.
    # video_capture = cv2.VideoCapture(args.captureDevice)
    # video_capture.set(3, args.width)
    # video_capture.set(4, args.height)

    confidenceList = []
    # set_trace()

    # create named pipe to
    invasion_subsys_name_pipe = "/tmp/IPC2_Image_pipe"
    try:
        # os.unlink(invasion_subsys_name_pipe)
        os.mkfifo(invasion_subsys_name_pipe)
    except OSError:
        pass
    invasion_subsys_fh = os.open(invasion_subsys_name_pipe, os.O_WRONLY)

    # 7-15-add
    from xmcext import XMCamera
    cp = XMCamera("192.168.0.3", 34567, "admin", "ludian@blq", "")
    cp = XMCamera("192.168.0.6", 34567, "admin", "", "")
    cp.PrintInfo()
    cp.login()
    cp.open()
    time.sleep(2)

    warning_counter = 0
    previous_exist_unknown_person = True
    # 7-15-add

    counter = 0
    while True:
        counter = counter + 1
        # ret, frame = video_capture.read()
        frame = np.asarray(cp.queryframe('array')).reshape(1080, 1920, 3)
        frame = cv2.resize(frame, (960, 540))
        if(counter%2 ==0):
            persons, confidences, bbs = infer(frame, args)
            print("P: " + str(persons) + " C: " + str(confidences))
            # print(str(confidences))
            # print(confidences)
            # if(float(confidences) > 0.85):
            #     print("YES")
            # else:
            #     print("NO")
            cam_dect_name = str(persons)
            try:
                # append with two floating point precision
                confidenceList.append('%.2f' % confidences[0])
            except:
                # If there is no face detected, confidences matrix will be empty.
                # We can simply ignore it.
                pass

            for i, c in enumerate(confidences):
                global cam_confidences
                cam_confidences = c

            exist_unknown_person = False

            # 7-15-23

            send_safe = ""
            send_unsafe = ""
            # Print the person name and conf value on the frame next to the person
            # Also print the bounding box
            for idx, person in enumerate(persons):
                cv2.rectangle(frame, (bbs[idx].left() * 4, bbs[idx].top() * 4),
                              (bbs[idx].right() * 4, bbs[idx].bottom() * 4), (0, 255, 0), 2)
                # cv2.putText(frame, "{}@{:.2f}".format(person, confidences[idx]),
                #             (bbs[idx].left() * 4, bbs[idx].bottom() * 4 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                #             (255, 255, 255), 1)

                if persons[idx] == "_unknown":
                    exist_unknown_person = True
                    send_unsafe += "{} @{:.2f}".format(person,
                                                       confidences[idx]) + "\n"
                else:
                    send_safe += "{} @{:.2f}".format(person,
                                                     confidences[idx]) + "\n"

            if exist_unknown_person and previous_exist_unknown_person:
                warning_counter = warning_counter + 1
            else:
                warning_counter = 0

            previous_exist_unknown_person = exist_unknown_person

            # cv2.imshow('', frame)
            # cv2.waitKey(0)

            os.write(invasion_subsys_fh, frame.tobytes())
        else:
            continue
    cv2.destroyAllWindows()
    cp.close()


class myThread1(threading.Thread):
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter

    # Called for every client connecting (after handshake)
    def new_client(self, client, server):
        print("New client connected and was given id %d" % client['id'])
        server.send_message_to_all("Hey all, a new client has joined us")

    # Called for every client disconnecting
    def client_left(self, client, server):
        print("Client(%d) disconnected" % client['id'])

    # Called when a client sends a message
    def message_received(self, client, server, message):
        global cam_dect_name
        global cam_confidences
        if len(message) > 200:
            message = message[:200] + '..'
        #print("Client(%d) said: %s" % (client['id'], message))
        if (cam_dect_name == "[b'xuliang']"):
            if(cam_confidences >= 0.75):
                server.send_message_to_all("YES")
            else:
                server.send_message_to_all("NO")
        elif (cam_dect_name == "[]"):
            server.send_message_to_all("Normal")
        else:
                server.send_message_to_all("NO")

    def run(self):
        print("开启线程： " + self.name)
        PORT = 9002
        server = WebsocketServer(PORT, '127.0.0.1')
        server.set_fn_new_client(self.new_client)
        server.set_fn_client_left(self.client_left)
        server.set_fn_message_received(self.message_received)
        # 获取锁，用于线程同步
        threadLock.acquire()
        # print_time1(self.name, self.counter, 1)
        # 释放锁，开启下一个线程
        threadLock.release()
        server.run_forever()
        # print ("开启线程： " + self.name)


class myThread2(threading.Thread):
    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter

    def run(self):
        print("开启线程： " + self.name)
        # 获取锁，用于线程同步
        cam_dect()
        #print_time2(self.name, self.counter, 1)
        threadLock.acquire()
        # 释放锁，开启下一个线程
        threadLock.release()


def print_time2(threadName, delay, counter):
    while counter:
        time.sleep(delay)
        print("print_time2:%s: %s" % (threadName, time.ctWime(time.time())))
        counter -= 1


def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    """
    @brief      Overlays a transparant PNG onto another image using CV2

    @param      background_img    The background image
    @param      img_to_overlay_t  The transparent image to overlay (has alpha channel)
    @param      x                 x location to place the top-left corner of our overlay
    @param      y                 y location to place the top-left corner of our overlay
    @param      overlay_size      The size to scale our overlay to (tuple), no scaling if None

    @return     Background image with overlay on top
    """

    bg_img = background_img.copy()

    if overlay_size is not None:
        img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

    # Extract the alpha mask of the RGBA image, convert to RGB
    b, g, r, a = cv2.split(img_to_overlay_t)
    overlay_color = cv2.merge((b, g, r))

    # Apply some simple filtering to remove edge noise
    mask = cv2.medianBlur(a, 5)

    h, w, _ = overlay_color.shape
    roi = bg_img[y:y+h, x:x+w]

    # Black-out the area behind the logo in our original ROI
    img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(),
                              mask=cv2.bitwise_not(mask))

    # Mask out the logo from the logo image.
    img2_fg = cv2.bitwise_and(overlay_color, overlay_color, mask=mask)

    # Update the original image with our new ROI
    bg_img[y:y+h, x:x+w] = cv2.add(img1_bg, img2_fg)

    return bg_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dlibFacePredictor',
        type=str,
        help="Path to dlib's face predictor.",
        default=os.path.join(
            dlibModelDir,
            "shape_predictor_68_face_landmarks.dat"))
    parser.add_argument(
        '--networkModel',
        type=str,
        help="Path to Torch network model.",
        default=os.path.join(
            openfaceModelDir,
            'nn4.small2.v1.t7'))
    parser.add_argument('--imgDim', type=int,
                        help="Default image dimension.", default=96)
    parser.add_argument(
        '--captureDevice',
        type=int,
        default=0,
        help='Capture device. 0 for latop webcam and 1 for usb webcam')
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument(
        '--classifierModel',
        type=str,
        help='The Python pickle representing the classifier. This is NOT the Torch network model, which can be set with --networkModel.',
        default=os.path.join(modelDir,
            'face',
            "IPC2_simple.pkl"))
    args = parser.parse_args()

    align = openface.AlignDlib(args.dlibFacePredictor)
    net = openface.TorchNeuralNet(
        args.networkModel,
        imgDim=args.imgDim,
        cuda=args.cuda)

    threadLock = threading.Lock()
    threads = []

    # 创建新线程
    thread1 = myThread1(1, "Thread-1", 1)
    thread2 = myThread2(2, "Thread-2", 2)

    # 开启新线程
    thread1.start()
    thread2.start()

    # 添加线程到线程列表
    threads.append(thread1)
    threads.append(thread2)

    # 等待所有线程完成
    for t in threads:
        t.join()
    print("退出主线程")
