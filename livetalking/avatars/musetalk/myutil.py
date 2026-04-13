import numpy as np
import cv2
import copy

def get_image_blending(image,face,face_box,mask_array,crop_box):
    body = image
    x, y, x1, y1 = [int(v) for v in face_box]
    x_s, y_s, x_e, y_e = [int(v) for v in crop_box]

    frame_h, frame_w = body.shape[:2]
    cx1 = max(0, x_s)
    cy1 = max(0, y_s)
    cx2 = min(frame_w, x_e)
    cy2 = min(frame_h, y_e)
    if cx1 >= cx2 or cy1 >= cy2:
        return body

    face_large = copy.deepcopy(body[cy1:cy2, cx1:cx2])

    fx1 = max(x, cx1)
    fy1 = max(y, cy1)
    fx2 = min(x1, cx2)
    fy2 = min(y1, cy2)
    if fx1 < fx2 and fy1 < fy2:
        src_x1 = fx1 - x
        src_y1 = fy1 - y
        src_x2 = src_x1 + (fx2 - fx1)
        src_y2 = src_y1 + (fy2 - fy1)

        dst_x1 = fx1 - cx1
        dst_y1 = fy1 - cy1
        dst_x2 = dst_x1 + (fx2 - fx1)
        dst_y2 = dst_y1 + (fy2 - fy1)
        face_large[dst_y1:dst_y2, dst_x1:dst_x2] = face[src_y1:src_y2, src_x1:src_x2]

    mask_image = cv2.cvtColor(mask_array,cv2.COLOR_BGR2GRAY)
    mask_image = (mask_image/255).astype(np.float32)
    crop_h = cy2 - cy1
    crop_w = cx2 - cx1
    offset_y = cy1 - y_s
    offset_x = cx1 - x_s
    mask_image = mask_image[offset_y:offset_y+crop_h, offset_x:offset_x+crop_w]
    if mask_image.shape[:2] != (crop_h, crop_w):
        mask_image = cv2.resize(mask_image, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)

    # mask_not = cv2.bitwise_not(mask_array)
    # prospect_tmp = cv2.bitwise_and(face_large, face_large, mask=mask_array)
    # background_img = body[y_s:y_e, x_s:x_e]
    # background_img = cv2.bitwise_and(background_img, background_img, mask=mask_not)
    # body[y_s:y_e, x_s:x_e] = prospect_tmp + background_img

    #print(mask_image.shape)
    #print(cv2.minMaxLoc(mask_image))

    body[cy1:cy2, cx1:cx2] = cv2.blendLinear(face_large, body[cy1:cy2, cx1:cx2], mask_image, 1-mask_image)

    #body.paste(face_large, crop_box[:2], mask_image)
    return body
