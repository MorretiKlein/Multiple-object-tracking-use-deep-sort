import cv2
import colorsys

def draw_bbox(image, bboxes, CLASSES, show_label=True, show_confidence=True, Text_colors=(255, 255, 0), rectangle_colors='', tracking=False):
    NUM_CLASS = CLASSES

    hsv_tuples = [(1.0 * x / 20, 10., 1.) for x in range(20)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    image_h, image_w, _ = image.shape

    for bbox in bboxes:
        x1,y1,x2,y2 = bbox[:4]
        x1,y1,x2,y2 = int(bbox[0] *640), int(bbox[1] * 640), int(bbox[2] * 640) , int(bbox[3] * 640)
        score = bbox[4]
        class_ind = int(bbox[5])

        bbox_color = rectangle_colors if rectangle_colors != '' else colors[class_ind] 
        bbox_thick = int(0.6 * (image_h + image_w) / 1000)
        
        if bbox_thick < 1:
            bbox_thick = 1
        fontScale = 0.75 * bbox_thick
        image = cv2.rectangle(image, (x1, y1), (x2, y2), bbox_color, bbox_thick * 2)

        if show_label:
            score_str = " {:.2f}".format(score) if show_confidence else ""
            if tracking:
                score_str = " " + str(score)
            
            print(score_str)
            label = "{}".format(NUM_CLASS[class_ind]) + score_str
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale, thickness=bbox_thick)
            image = cv2.rectangle(image, (x1, y1), (x1 + text_width, y1 - text_height - baseline), bbox_color, thickness=cv2.FILLED)
            image = cv2.putText(image, label, (x1, y1 - 4), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        fontScale, Text_colors, bbox_thick, lineType=cv2.LINE_AA)

    return image
