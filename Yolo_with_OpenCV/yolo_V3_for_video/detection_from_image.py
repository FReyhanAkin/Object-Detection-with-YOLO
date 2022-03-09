import cv2
import numpy as np


image_BGR=cv2.imread('#The directory where the image is located')

h, w = image_BGR.shape[:2]

blob = cv2.dnn.blobFromImage(image_BGR, 1 / 255.0, (416, 416), swapRB=True, crop=False) 

with open('#The directory where the COCO data set names file is located') as f:
    labels = [line.strip() for line in f] 
    
network = cv2.dnn.readNetFromDarknet('#The directory where the YOLO v3 configuration file is located', '#The directory where the trained YOLO v3 weights file is located')

layers_names_all=network.getLayerNames()

layers_names_output = \
    [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]
    
probability_minimum = 0.7
threshold = 0.4

colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

network.setInput(blob)

output_from_network=network.forward(layers_names_output)

bounding_boxes = []
confidences = []
class_numbers = []

for result in output_from_network:
    for detected_objects in result:
        
        scores = detected_objects[5:]
        class_current = np.argmax(scores)
        confidence_current = scores[class_current]

        if confidence_current > probability_minimum:
            
            box_current = detected_objects[0:4] * np.array([w, h, w, h])

            x_center, y_center, box_width, box_height = box_current
            x_min = int(x_center - (box_width / 2))
            y_min = int(y_center - (box_height / 2))

            bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
            confidences.append(float(confidence_current))
            class_numbers.append(class_current)
            
results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,probability_minimum, threshold)


if len(results) > 0:
    for i in results.flatten():

        x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
        box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

        colour_box_current = colours[class_numbers[i]].tolist()

        cv2.rectangle(image_BGR, (x_min, y_min), (x_min + box_width, y_min + box_height), colour_box_current, 2)

        text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],confidences[i])

        cv2.putText(image_BGR, text_box_current, (x_min, y_min - 5), cv2.FONT_HERSHEY_COMPLEX, 1, colour_box_current, 2)
        
        
cv2.namedWindow('Result', cv2.WINDOW_NORMAL)
cv2.imshow('Result', image_BGR)
cv2.waitKey(0)
cv2.destroyWindow('Result')