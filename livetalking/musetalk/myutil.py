import numpy as np
import cv2
import copy

def get_image_blending(image,face,face_box,mask_array,crop_box):
    body = image
    x, y, x1, y1 = [int(v) for v in face_box]
    x_s, y_s, x_e, y_e = [int(v) for v in crop_box]

    h, w = body.shape[:2]
    x_s = max(0, min(x_s, w))
    x_e = max(0, min(x_e, w))
    y_s = max(0, min(y_s, h))
    y_e = max(0, min(y_e, h))
    if x_e <= x_s or y_e <= y_s:
        return body

    face_large = copy.deepcopy(body[y_s:y_e, x_s:x_e])
    fh, fw = face_large.shape[:2]

    roi_x0 = max(0, x - x_s)
    roi_y0 = max(0, y - y_s)
    roi_x1 = min(fw, x1 - x_s)
    roi_y1 = min(fh, y1 - y_s)
    if roi_x1 <= roi_x0 or roi_y1 <= roi_y0:
        return body

    target_w = roi_x1 - roi_x0
    target_h = roi_y1 - roi_y0
    face_resized = cv2.resize(face, (target_w, target_h))
    face_large[roi_y0:roi_y1, roi_x0:roi_x1] = face_resized

    mask_image = cv2.cvtColor(mask_array,cv2.COLOR_BGR2GRAY)
    if mask_image.shape[:2] != (y_e - y_s, x_e - x_s):
        mask_image = cv2.resize(mask_image, (x_e - x_s, y_e - y_s))
    mask_image = (mask_image/255).astype(np.float32)

    # mask_not = cv2.bitwise_not(mask_array)
    # prospect_tmp = cv2.bitwise_and(face_large, face_large, mask=mask_array)
    # background_img = body[y_s:y_e, x_s:x_e]
    # background_img = cv2.bitwise_and(background_img, background_img, mask=mask_not)
    # body[y_s:y_e, x_s:x_e] = prospect_tmp + background_img

    #print(mask_image.shape)
    #print(cv2.minMaxLoc(mask_image))

    body[y_s:y_e, x_s:x_e] = cv2.blendLinear(
        face_large,
        body[y_s:y_e, x_s:x_e],
        mask_image,
        1 - mask_image,
    )

    #body.paste(face_large, crop_box[:2], mask_image)
    return body
