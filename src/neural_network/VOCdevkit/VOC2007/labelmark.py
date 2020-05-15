import os
import cv2
import xml.dom.minidom
from tqdm import tqdm

image_path = "./JPEGImages/"
annotation_path = "./Annotations/"

files_name = os.listdir(image_path)
for filename_ in tqdm(files_name):
    filename, extension = os.path.splitext(filename_)
    # filename=filename.split('__')[0]
    img_path = image_path + filename + '.jpg'
    xml_path = annotation_path + filename + '.xml'
    img = cv2.imread(img_path)
    if img is None:
        pass
    try:
        dom = xml.dom.minidom.parse(xml_path)
    except:
        # os.remove(img_path)
        continue
    root = dom.documentElement
    objects = dom.getElementsByTagName("object")
    for object in objects:
        bndbox = object.getElementsByTagName('bndbox')[0]
        xmin = bndbox.getElementsByTagName('xmin')[0]
        ymin = bndbox.getElementsByTagName('ymin')[0]
        xmax = bndbox.getElementsByTagName('xmax')[0]
        ymax = bndbox.getElementsByTagName('ymax')[0]
        xmin_data = int(xmin.childNodes[0].data)
        ymin_data = int(ymin.childNodes[0].data)
        xmax_data = int(xmax.childNodes[0].data)
        ymax_data = int(ymax.childNodes[0].data)
        label_name = object.getElementsByTagName('name')[0].childNodes[0].data
        cv2.rectangle(img, (xmin_data, ymin_data), (xmax_data, ymax_data), (55, 255, 155), 1)
        cv2.putText(img, label_name, (int((xmin_data + xmax_data / 2)), int((ymin_data + ymax_data) / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    flag = 0
    flag = cv2.imwrite("./Visualization/{}.jpg".format(filename), img)
    if not (flag):
        print(filename, "error")
print("all done ====================================")
