import os
import cv2
import numpy as np
from PIL import Image

path = '../all data/result_MRI_CT5/'
if not os.path.exists(path):
    os.makedirs(path)

# TNO_5
# pt1=(100, 286)
# pt2=(558, 336)
# width1=150
# hight1 = 100
# width2=150
# hight2=100
# TNO1
# pt1=(250, 94)
# pt2=(400, 260)
# width1=150
# hight1 = 100
# width2=150
# hight2=100
# TNO2
# pt1=(229, 203)
# pt2=(130, 29)
# width1=62
# hight1 = 45
# width2=82
# hight2=60
# TNO3
# pt1=(196, 112)
# pt2=(522, 233)
# width1=62
# hight1 = 45
# width2=62
# hight2=45
# TNO4
# pt1=(250, 118)
# pt2=(265, 212)
# width1=36
# hight1 = 27
# width2=36
# hight2=27
# TNO5
# pt1=(409, 41)
# pt2=(420, 268)
# width1=47
# hight1 = 37
# width2=47
# hight2=37
# RoadScene1
# pt1=(348, 154)
# pt2=(451, 3)
# width1=55
# hight1 = 34
# width2=55
# hight2=34
# RoadScene2
# pt1=(437, 115)
# pt2=(172, 13)
# width1=54
# hight1 = 26
# width2=54
# hight2=26
# RoadScene3
# pt1=(291, 130)
# pt2=(294, 46)
# width1=52
# hight1 = 30
# width2=52
# hight2=30
# RoadScene4
# pt1=(419, 244)
# pt2=(499, 41)
# width1=58
# hight1 = 42
# width2=58
# hight2=42
# RoadScene5
# pt1=(290, 63)
# pt2=(355, 215)
# width1=56
# hight1 = 28
# width2=56
# hight2=28
# MSRS1
# pt1=(140, 107)
# pt2=(57, 417)
# width1=64
# hight1 = 48
# width2=64
# hight2=48
# MSRS2
# pt1=(356, 225)
# pt2=(122, 294)
# width1=64
# hight1 = 48
# width2=64
# hight2=48
# MSRS3
# pt1=(334, 141)
# pt2=(390, 78)
# width1=64
# hight1 = 48
# width2=64
# hight2=48
# MSRS4
# pt1=(100, 200)
# pt2=(158, 357)
# width1=64
# hight1 = 48
# width2=64
# hight2=48
# MSRS5
# pt1=(593, 153)
# pt2=(28, 285)
# width1=64
# hight1 = 48
# width2=64
# hight2=48
# A_TNO1
# pt1=(140, 289)
# pt2=(552, 344)
# width1=76
# hight1 = 57
# width2=76
# hight2=57
# A_MSRS1
# pt1=(146, 105)
# pt2=(44, 415)
# width1=64
# hight1 = 48
# width2=64
# hight2=48
# # A_RoadScene1
# pt1=(434, 118)
# pt2=(160, 12)
# width1=64
# hight1 = 48
# width2=64
# hight2=48
# MRI_CT1
# pt1=(89, 35)
# pt2=(101, 177)
# width1=50
# hight1 = 50
# width2=50
# hight2=50
# MRI_CT2
# pt1=(55, 60)
# pt2=(108, 197)
# width1=50
# hight1 = 50
# width2=50
# hight2=50
# # MRI_CT3
# pt1=(154, 98)
# pt2=(109, 156)
# width1=50
# hight1 = 50
# width2=50
# hight2=50
# MRI_CT4
# pt1=(143, 48)
# pt2=(100, 127)
# width1=50
# hight1 = 50
# width2=50
# hight2=50
# MRI_CT5
pt1=(100, 69)
pt2=(100, 179)
width1=50
hight1 = 50
width2=50
hight2=50
def show_cvimg(im):
    return Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))

def stack_image(path:str):
    image = cv2.imread(path)
    h, w, c = image.shape

    if w % 2 != 0:
        if h % 2 != 0:
            image = cv2.resize(image, (w - 1, h - 1))
        else:
            image = cv2.resize(image, (w - 1, h))
    elif h % 2 != 0:
        image = cv2.resize(image, (w, h - 1))

    # patch1
    pt1_ = (pt1[0] + width1, pt1[1] + hight1)
    cv2.rectangle(image, pt1, pt1_, (0, 0, 255), 2)
    # patch2
    pt2_ = (pt2[0] + width2, pt2[1] + hight2)
    cv2.rectangle(image, pt2, pt2_, (0, 0, 255), 2)

    patch1_ = image[pt1[1] + 2:pt1[1] + hight1 - 2, pt1[0] + 2:pt1[0] + width1 -2, :]
    t1 = patch1_.copy()
    cv2.rectangle(t1, (0, 0), (t1.shape[1]-1, t1.shape[0]-1), (0, 0, 255), 1)
    t1 = cv2.resize(t1, (int(w / 2), int(h / 2)))


    patch2_ = image[pt2[1] + 2:pt2[1] + hight2 - 2, pt2[0] +2:pt2[0] + width2 - 2, :]
    t2 = patch2_.copy()
    cv2.rectangle(t2, (0, 0), (t2.shape[1] - 1, t2.shape[0] - 1), (0, 0, 255), 1)
    t2 = cv2.resize(t2, (int(w / 2), int(h / 2)))

    patch = np.hstack((t1, t2))
    image_stack = np.vstack((image, patch))
    return image_stack


if __name__ == '__main__':
    temp_list = []
    for root, dict, files in os.walk('../all data/MRI_CT5/'):
        for file in files:
            temp_list.append(os.path.join(root, file))
    for i in range(len(temp_list)):
        cv2.imwrite(os.path.join('../all data/result_MRI_CT5/',os.path.basename(temp_list[i])), stack_image(temp_list[i]), [cv2.IMWRITE_PNG_COMPRESSION, 0])

