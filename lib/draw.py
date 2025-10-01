import cv2
GREEN=(0,255,0); RED=(0,0,255); BLUE=(255,0,0); YELLOW=(0,255,255)

def draw_head_pose(frame, bbox_xyxy, nose, ear_left, ear_right):
    x1,y1,x2,y2 = map(int, bbox_xyxy)
    cv2.rectangle(frame, (x1,y1), (x2,y2), GREEN, 2)
    for p,c in [(nose,RED),(ear_left,BLUE),(ear_right,BLUE)]:
        cv2.circle(frame, (int(p[0]),int(p[1])), 4, c, -1, lineType=cv2.LINE_AA)
    ex = int(0.5*(ear_left[0]+ear_right[0])); ey = int(0.5*(ear_left[1]+ear_right[1]))
    cv2.line(frame, (int(nose[0]),int(nose[1])), (ex,ey), YELLOW, 2, lineType=cv2.LINE_AA)
    return frame