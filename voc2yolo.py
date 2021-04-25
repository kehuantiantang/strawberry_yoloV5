import xml.etree.ElementTree as ET
import pickle
import os
import os.path as osp
from os import listdir, getcwd
from os.path import join

sets = ['train', 'test']

classes = ['angular_leafspot',
'anthracnose_fruit_rot',
'blossom_blight',
'gray_mold',
'leaf_spot',
'powdery_mildew_fruit',
'powdery_mildew_leaf','anthracnose_runner']

# classes = ['angular_leafspot', 'anthracnose_fruit_rot', 'blossom_blight', 'gray_mold', 'leaf_spot',
#            'powdery_mildew_fruit', 'powdery_mildew_leaf']

def convert(size, box):
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(root, image_id):
    in_file = open(osp.join(root, '%s.xml'% (image_id)) )
    out_file = open(osp.join(root, '%s.txt' % (image_id)), 'w')

    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


root = '/media/HDD3/khtt/dataset/strawberry/data_detaction'
for image_set in sets:
    if not os.path.exists(root):
        os.makedirs(root)

    image_ids = [f.split('.')[0] for f in os.listdir(osp.join(root, image_set)) if f.endswith('jpg')]

    list_file = open(os.path.join(root, '%s.txt'%(image_set)), 'w')
    for image_id in image_ids:
        string = osp.join(root, image_set, '%s.jpg' % image_id) + '\n'
        list_file.write(string)
        convert_annotation(osp.join(root, image_set), image_id)
    list_file.close()