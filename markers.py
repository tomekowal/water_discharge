import cv2
import pickle

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_TRIPLEX
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
markerLength = 1
with open(r'calibration.pkl', 'rb') as calibration:
    cret, mtx, dist, rvec, tvec = pickle.load(calibration)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary)
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, markerLength, mtx, dist)
    #cv2.putText(frame, str(tvec), (10, 600), font, 1.0, (255,255,255), 2, cv2.LINE_AA)
    cv2.aruco.drawDetectedMarkers(frame, corners, ids, 255)
    for i in range(len(corners)):
        frame = cv2.aruco.drawAxis(frame, mtx, dist, rvecs[i], tvecs[i], 1.0)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
