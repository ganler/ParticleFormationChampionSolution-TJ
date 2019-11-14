import cv2
import pandas as pd
import os
import matplotlib.pyplot as plt


def particle_counter(img, fname):
    shift = 35
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img[:, :, :] = 0
    img[:, :, 0] = gray

    gray = gray[shift:600, shift:265]

    gray = cv2.GaussianBlur(gray, (3, 5), cv2.BORDER_DEFAULT)
    mser = cv2.MSER_create(_delta=3, _min_area=1, _max_area=30)

    regions, boxes = mser.detectRegions(gray)
    remained_boxes = []

    for box in boxes:
        x, y, w, h = box
        ok = True
        for b in remained_boxes:
            x_, y_, w_, h_ = b
            if abs(x - x_ + (w - w_) / 2) + abs(y - y_ + (h - h_) / 2) < 10:
                ok = False
                break
        if ok:
            remained_boxes.append(box)
            cv2.rectangle(img, (x + shift, y + shift), (x + w + shift, y + h + shift), (255, 0, 255), 2)
    # img = img[shift:600, shift:265]
    cv2.imwrite('out/' + fname, img)


if __name__ == '__main__':
    for f in os.listdir('test'):
        particle_counter(cv2.imread('test/' + f), f)

    '''
    shift = 35
    img = cv2.imread('test/b672221d5a1f599d7ae4213aae29fb7b8426fd69.bmp')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = gray[shift:600, shift:265]

    gray = cv2.GaussianBlur(gray, (3, 5), cv2.BORDER_DEFAULT)
    # _, gray = cv2.threshold(gray, 10, 255, 0)

    mser = cv2.MSER_create(_delta=3, _min_area=1, _max_area=25)

    regions, boxes = mser.detectRegions(gray)
    remained_boxes = []

    for box in boxes:
        x, y, w, h = box
        ok = True
        for b in remained_boxes:
            x_, y_, w_, h_ = b
            if abs(x - x_ + (w - w_)/2) + abs(y - y_ + (h - h_)/2) < 10:
                ok = False
                break
        if ok:
            remained_boxes.append(box)
            cv2.rectangle(img, (x+shift, y+shift), (x + w+shift, y + h+shift), (255, 0, 255), 3)

    print(len(remained_boxes))
    plt.imshow(img, 'brg')
    plt.show()

    '''