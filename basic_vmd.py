import cv2
import imutils

if __name__ == "__main__":
    counter = 0
    prev_frame = None
    cap = cv2.VideoCapture("/home/neuronic/EX/People - 6387.mp4")
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file")
        exit()
    
    while True:
        ret, frame = cap.read()
        
        # Check if frame was read successfully
        if not ret:
            print("End of video or failed to read frame")
            break
            
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if counter == 0:
            prev_frame = gray_frame
            counter += 1
        else:
            diff = cv2.absdiff(gray_frame, prev_frame)
            thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            prev_frame = gray_frame
            counter += 1
            
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
