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

from websocket_server import WebsocketServer

DETECTION_WIDTH = 480
DETECTION_HEIGHT = 270

# Called for every client connecting (after handshake

PORT = 9998
server = WebsocketServer(PORT, "127.0.0.1")

now_push_ip = ""


def new_client(client, server):
    print("New client connected and was given id %d" % client['id'])
    server.send_message_to_all("Hey all, a new client has joined us")

# Called for every client disconnecting


def client_left(client, server):
    print("Client(%d) disconnected" % client['id'])

# Called when a client sends a message


def message_received(client, server, message):
    if len(message) > 200:
        message = message[:200]+'..'
    print("Client(%d) said: %s" % (client['id'], message))

    global now_push_ip
    now_push_ip = message
    print (now_push_ip)


def start_wensocket_server():
    server.set_fn_new_client(new_client)
    server.set_fn_client_left(client_left)
    server.set_fn_message_received(message_received)
    server.run_forever()


import threading
threading.Thread(target=start_wensocket_server, name='LoopThread').start()
print ('websocket server start listening at 9998 port')

# -1 loads with transparency
# warning_icon = cv2.imread('./data/warning_50pix.png', -1)


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


fileDir = os.path.dirname(os.path.realpath(__file__))
modelDir = os.path.join(fileDir, '..', 'models')
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

    return (reps, bb)


def infer(img, args, clf_name):
    with open(clf_name, 'r') as f:
        if sys.version_info[0] < 3:
            (le, clf) = pickle.load(f)
        else:
            (le, clf) = pickle.load(f, encoding='latin1')

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
        maxI = np.argmax(predictions)
        persons.append(le.inverse_transform(maxI))

        confidences.append(predictions[maxI])
        if args.verbose:
            print("Prediction took {} seconds.".format(time.time() - start))
            pass
        # print("Predict {} with {:.2f} confidence.".format(person.decode('utf-8'), confidence))
        if isinstance(clf, GMM):
            dist = np.linalg.norm(rep - clf.means_[maxI])
            print("  + Distance from the mean: {}".format(dist))
            pass
    return (persons, confidences, bbs)


def camera_detect(ip_camera):
    warning_counter = 0
    previous_exist_unknown_person = True
    ip_address = ip_camera.GetIP()
    clf_name = simple + ip_address.split('.')[3] + ".pkl"
    confidenceList = []
    while True:
        frame = np.asarray(ip_camera.queryframe(
            'array')).reshape(1080, 1920, 3)

        persons, confidences, bbs = infer(frame, args, clf_name)
        # if(str(persons) == '[]'):
        #     print (ip_address + " P: " + str(persons) + " C: " + str(confidences))

        try:
            confidenceList.append('%.2f' % confidences[0])
        except:
            pass
        for i, c in enumerate(confidences):
            if c <= args.threshold:
                persons[i] = "_unknown"

        exist_unknown_person = False

        send_safe = ""
        send_unsafe = ""
        for idx, person in enumerate(persons):
            cv2.rectangle(frame, (bbs[idx].left()*4, bbs[idx].top()*4),
                          (bbs[idx].right()*4, bbs[idx].bottom()*4), (0, 255, 0), 2)
            cv2.putText(frame, "{}@{:.2f}".format(person, confidences[idx]),
                        (bbs[idx].left()*4, bbs[idx].bottom()*4+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            if persons[idx] == "_unknown":
                exist_unknown_person = True
                send_unsafe += "{} @{:.2f}".format(person,
                                                   confidences[idx]) + "\n"
            else:
                send_safe += "{} @{:.2f}".format(person,
                                                 confidences[idx]) + "\n"

        if(str(persons) != '[]'):
            print (ip_address+'&' + send_safe+'&'+send_unsafe)
            server.send_message_to_all(send_safe+'&'+send_unsafe)

        if warning_counter > 10:
                # frame = overlay_transparent(frame, warning_icon, 0, 0, (60, 60))
            server.send_message_to_all(ip_address)

        if exist_unknown_person and previous_exist_unknown_person:
            warning_counter = warning_counter + 1
        else:
            warning_counter = 0

        previous_exist_unknown_person = exist_unknown_person

        if now_push_ip == ip_address:
            os.write(invasion_subsys_fh, frame.tobytes())

    ip_camera.close()


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
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument(
        '--classifierModel',
        type=str,
        default = "",
        help='The Python pickle representing the classifier. This is NOT the Torch network model, which can be set with --networkModel.')

    args = parser.parse_args()

    align = openface.AlignDlib(args.dlibFacePredictor)
    net = openface.TorchNeuralNet(
        args.networkModel,
        imgDim=args.imgDim,
        cuda=args.cuda)


    invasion_subsys_name_pipe = "/tmp/invasion_subsys_name_pipe"
    try:
        os.mkfifo(invasion_subsys_name_pipe)
    except OSError:
        pass
    invasion_subsys_fh = os.open(invasion_subsys_name_pipe, os.O_WRONLY)

    # 7-15-add
    from xmcext import XMCamera
    cp = [XMCamera("10.26.0.190", 34567, "admin", "", ""),
          XMCamera("10.26.0.191", 34567, "admin", "", "")]
    #       XMCamera("10.26.0.192", 34567, "admin", "", ""),
    #       XMCamera("10.26.0.193", 34567, "admin", "", "")]

    for camera in cp:
        camera.PrintInfo()
        camera.login()
        camera.open()

    time.sleep(2)

    for camera in cp:
        threading.Thread(target=camera_detect, args=(
            camera,), name=camera.GetIP()).start()


#     while True:
#         frame = np.asarray(cp[0].queryframe('array')).reshape(1080, 1920, 3)

#         persons, confidences, bbs = infer(frame, args)
#         print ("P: " + str(persons) + " C: " + str(confidences))
#         try:
#             confidenceList.append('%.2f' % confidences[0])
#         except:
#             pass

#         for i, c in enumerate(confidences):
#             if c <= args.threshold:  # 0.5 is kept as threshold for known face.
#                 persons[i] = "_unknown"

#         exist_unknown_person = False

# # 7-15-23

#         send_safe = ""
#         send_unsafe = ""
#         # Print the person name and conf value on the frame next to the person
#         # Also print the bounding box
#         for idx, person in enumerate(persons):
#             cv2.rectangle(frame, (bbs[idx].left()*4, bbs[idx].top()*4),
#                           (bbs[idx].right()*4, bbs[idx].bottom()*4), (0, 255, 0), 2)
#             cv2.putText(frame, "{}@{:.2f}".format(person, confidences[idx]),
#                         (bbs[idx].left()*4, bbs[idx].bottom()*4+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#             if persons[idx] == "_unknown":
#                 exist_unknown_person = True
#                 send_unsafe += "{} @{:.2f}".format(person,
#                                                    confidences[idx]) + "\n"
#             else:
#                 send_safe += "{} @{:.2f}".format(person,
#                                                  confidences[idx]) + "\n"

#         if(str(persons) != '[]'):
#             server.send_message_to_all(send_safe+'&'+send_unsafe)

#         # show warning
#         if warning_counter > 10:
#             # frame = overlay_transparent(frame, warning_icon, 0, 0, (60, 60))
#             server.send_message_to_all("danger")

#         if exist_unknown_person and previous_exist_unknown_person:
#             warning_counter = warning_counter + 1
#         else:
#             warning_counter = 0

#         previous_exist_unknown_person = exist_unknown_person

#         # cv2.imshow('', frame)
#         # cv2.waitKey(0)

#         os.write(invasion_subsys_fh, frame.tobytes())

    # cv2.destroyAllWindows()
    # cp.close()
