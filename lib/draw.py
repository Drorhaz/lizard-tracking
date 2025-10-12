import cv2
GREEN=(0,255,0); RED=(0,0,255); BLUE=(255,0,0); YELLOW=(0,255,255)

def draw_head_pose(frame, bbox_xyxy, nose, ear_left, ear_right, confidence=None):
    """
    Draw head pose detection overlay on frame
    
    Args:
        frame: Input frame (BGR format)
        bbox_xyxy: Bounding box coordinates [x1, y1, x2, y2]
        nose: Nose keypoint coordinates [x, y]
        ear_left: Left ear keypoint coordinates [x, y]  
        ear_right: Right ear keypoint coordinates [x, y]
        confidence: Optional confidence score to display
        
    Returns:
        Frame with drawn overlay
    """
    # Draw bounding box
    x1, y1, x2, y2 = map(int, bbox_xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
    
    # Draw keypoints
    if nose is not None:
        cv2.circle(frame, (int(nose[0]), int(nose[1])), 6, RED, -1, lineType=cv2.LINE_AA)
    
    if ear_left is not None and ear_right is not None:
        cv2.circle(frame, (int(ear_left[0]), int(ear_left[1])), 5, BLUE, -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (int(ear_right[0]), int(ear_right[1])), 5, BLUE, -1, lineType=cv2.LINE_AA)
        
        # Draw line from nose to ear midpoint
        if nose is not None:
            ex = int(0.5 * (ear_left[0] + ear_right[0]))
            ey = int(0.5 * (ear_left[1] + ear_right[1]))
            cv2.line(frame, (int(nose[0]), int(nose[1])), (ex, ey), YELLOW, 2, lineType=cv2.LINE_AA)
    
    # Draw confidence text if provided
    if confidence is not None:
        bbox_txt = f"HEAD {confidence:.3f}"
        # Black outline for better readability
        cv2.putText(frame, bbox_txt, (x1, max(15, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)
        # Green text
        cv2.putText(frame, bbox_txt, (x1, max(15, y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 1, cv2.LINE_AA)
    
    return frame


def draw_head_pose_from_object(frame, head_pose_obj):
    """
    Draw head pose from a head pose object (HeadPose, SimpleDetection, etc.)
    
    Args:
        frame: Input frame (BGR format)
        head_pose_obj: Object with bbox, nose_tip, left_ear, right_ear, confidence attributes
        
    Returns:
        Frame with drawn overlay
    """
    if head_pose_obj is None:
        return frame
        
    # Extract coordinates based on object type
    bbox = getattr(head_pose_obj, 'bbox_xyxy', None) or getattr(head_pose_obj, 'bbox', None)
    nose = getattr(head_pose_obj, 'nose_tip', None) or getattr(head_pose_obj, 'nose', None)
    ear_left = getattr(head_pose_obj, 'left_ear', None) or getattr(head_pose_obj, 'ear_left', None)
    ear_right = getattr(head_pose_obj, 'right_ear', None) or getattr(head_pose_obj, 'ear_right', None)
    confidence = getattr(head_pose_obj, 'confidence', None) or getattr(head_pose_obj, 'conf', None)
    
    if bbox is None:
        return frame
        
    return draw_head_pose(frame, bbox, nose, ear_left, ear_right, confidence)


def draw_behavioral_event(frame, event_type, confidence=None, position=None):
    """
    Draw behavioral event overlay on frame
    
    Args:
        frame: Input frame (BGR format)
        event_type: Type of behavioral event (string)
        confidence: Optional confidence score
        position: Optional position tuple (x, y) for event location
        
    Returns:
        Frame with behavioral event overlay
    """
    # Color mapping for different event types
    event_colors = {
        'APPROACH': (0, 255, 0),    # Green
        'RETREAT': (0, 0, 255),     # Red  
        'STOP': (255, 0, 0),        # Blue
        'CLOSE_TO_TARGET': (0, 255, 255),  # Yellow
        'FAR_FROM_TARGET': (255, 0, 255),  # Magenta
    }
    
    color = event_colors.get(event_type.upper(), (255, 255, 255))  # Default white
    
    # Draw event text at top of frame
    text = f"{event_type.upper()}"
    if confidence is not None:
        text += f" ({confidence:.1%})"
    
    # Draw background rectangle for better readability
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
    cv2.rectangle(frame, (10, 10), (text_size[0] + 20, text_size[1] + 20), (0, 0, 0), -1)
    cv2.putText(frame, text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    
    # Draw position marker if provided
    if position is not None and len(position) >= 2:
        x, y = int(position[0]), int(position[1])
        cv2.circle(frame, (x, y), 10, color, 3)
        cv2.circle(frame, (x, y), 5, (255, 255, 255), -1)
    
    return frame


def draw_no_detection(frame, message="NO DETECTION"):
    """
    Draw 'no detection' overlay on frame
    
    Args:
        frame: Input frame (BGR format)  
        message: Message to display (default: "NO DETECTION")
        
    Returns:
        Frame with no detection overlay
    """
    # Get frame dimensions
    height, width = frame.shape[:2]
    
    # Calculate text size and position
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.0
    thickness = 3
    text_size = cv2.getTextSize(message, font, font_scale, thickness)[0]
    
    # Center the text
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    
    # Draw background rectangle
    rect_margin = 20
    cv2.rectangle(frame, 
                 (text_x - rect_margin, text_y - text_size[1] - rect_margin),
                 (text_x + text_size[0] + rect_margin, text_y + rect_margin),
                 (0, 0, 0), -1)
    
    # Draw text outline for better visibility
    cv2.putText(frame, message, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, message, (text_x, text_y), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
    
    return frame