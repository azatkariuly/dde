import os
import cv2

class_dict = {0: 'anomaly', 1: 'normal'}

xmin, ymin, xmax, ymax = 140, 170, 300, 340

for d_pipeline in ['train', 'val']:
    for class_id in class_dict.keys():
        img_dir = os.path.join('dataset', d_pipeline, class_dict[class_id])
        
        for filename in os.listdir(img_dir):
            if filename.endswith('.jpg'):
                frame = cv2.imread(os.path.join(img_dir, filename))
                img_height, img_width, channels = frame.shape
                
                centerx = ((xmax + xmin) / 2) / img_width
                centery = ((ymax + ymin) / 2) / img_height
                boxwidth = (xmax - xmin) / img_width
                boxheight = (ymax - ymin) / img_height
                
                box = [class_id, centerx, centery, boxwidth, boxheight]
                
                txt_filename = filename.replace(".jpg", ".txt")
                
                with open(os.path.join(img_dir, txt_filename), 'w') as file:
                    file.write(" ".join(map(str, box)))