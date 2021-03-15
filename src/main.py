import numpy as np 
import cv2 

webcam = cv2.VideoCapture(0) 

while(1): 
	
	_, imageFrame = webcam.read() 

	hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV) 

	red_lower = np.array([166, 86, 120], np.uint8) 
	red_upper = np.array([186, 106, 200], np.uint8) 
	red_mask1 = cv2.inRange(hsvFrame, red_lower, red_upper) 

	red_lower = np.array([0, 92, 68], np.uint8) 
	red_upper = np.array([10, 112, 148], np.uint8) 
	red_mask2 = cv2.inRange(hsvFrame, red_lower, red_upper) 
	red_mask  = red_mask1+red_mask2

	green_lower = np.array([76, 120, 91], np.uint8) 
	green_upper = np.array([96, 140, 171], np.uint8) 
	green_mask1 = cv2.inRange(hsvFrame, green_lower, green_upper) 

	green_lower = np.array([77, 161, 72], np.uint8) 
	green_upper = np.array([97, 181, 152], np.uint8) 
	green_mask2 = cv2.inRange(hsvFrame, green_lower, green_upper) 
	green_mask  = green_mask1+green_mask2

	# Set range for blue color and 
	# define mask 
	blue_lower = np.array([95, 78, 124], np.uint8)
	blue_upper = np.array([115, 88, 204], np.uint8)
	blue_mask1 = cv2.inRange(hsvFrame, blue_lower, blue_upper)

	blue_lower = np.array([94, 129, 79], np.uint8)
	blue_upper = np.array([114, 149, 159], np.uint8)
	blue_mask2 = cv2.inRange(hsvFrame, blue_lower, blue_upper)

	blue_mask = blue_mask1 + blue_mask2

	# detect orange
	orange_lower = np.array([0, 97, 137], np.uint8)
	orange_upper = np.array([19, 117, 207], np.uint8)
	orange_mask1 = cv2.inRange(hsvFrame, orange_lower, orange_upper)

	orange_lower = np.array([1, 137, 95], np.uint8)
	orange_upper = np.array([21, 157, 175], np.uint8)
	orange_mask2 = cv2.inRange(hsvFrame, orange_lower, orange_upper)

	orange_mask = orange_mask1 + orange_mask2

	yellow_lower = np.array([23, 41, 133], np.uint8)
	yellow_upper = np.array([40, 150, 255], np.uint8)
	yellow_mask1 = cv2.inRange(hsvFrame, orange_lower, orange_upper)

	yellow_lower = np.array([29, 105, 98], np.uint8)
	yellow_upper = np.array([49, 125, 178], np.uint8)
	yellow_mask2 = cv2.inRange(hsvFrame, orange_lower, orange_upper)

	yellow_mask = yellow_mask1 + yellow_mask2

	white_lower = np.array([28, 61, 142], np.uint8)
	white_upper = np.array([48, 81, 222], np.uint8)
	white_mask1 = cv2.inRange(hsvFrame, orange_lower, orange_upper)

	white_lower = np.array([29, 105, 98], np.uint8)
	white_upper = np.array([49, 125, 178], np.uint8)
	white_mask2 = cv2.inRange(hsvFrame, orange_lower, orange_upper)

	white_mask = white_mask1 + white_mask2

	kernal = np.ones((5, 5), "uint8") 
	
	red_mask = cv2.dilate(red_mask, kernal) 
	res_red = cv2.bitwise_and(imageFrame, imageFrame, mask = red_mask) 
	
	green_mask = cv2.dilate(green_mask, kernal) 
	res_green = cv2.bitwise_and(imageFrame, imageFrame, mask = green_mask) 
	
	blue_mask = cv2.dilate(blue_mask, kernal) 
	res_blue = cv2.bitwise_and(imageFrame, imageFrame, mask = blue_mask) 


	orange_mask = cv2.dilate(orange_mask, kernal)
	res_orange = cv2.bitwise_and(imageFrame, imageFrame, mask = orange_mask)


	yellow_mask = cv2.dilate(yellow_mask, kernal)
	res_yellow = cv2.bitwise_and(imageFrame, imageFrame, mask=yellow_mask)

	contours, hierarchy = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
	
	for pic, contour in enumerate(contours): 
		area = cv2.contourArea(contour) 
		if(area > 1000): 
			x, y, w, h = cv2.boundingRect(contour) 
			imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 0, 255), 2) 
			
			cv2.putText(imageFrame, "Red", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))	 

	contours, hierarchy = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
	
	for pic, contour in enumerate(contours): 
		area = cv2.contourArea(contour) 
		if(area > 1000): 
			x, y, w, h = cv2.boundingRect(contour) 
			imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (0, 255, 0), 2) 
			
			cv2.putText(imageFrame, "Green", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0)) 

	contours, hierarchy = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
	for pic, contour in enumerate(contours): 
		area = cv2.contourArea(contour) 
		if(area > 1000): 
			x, y, w, h = cv2.boundingRect(contour) 
			imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (255, 0, 0), 2) 
			
			cv2.putText(imageFrame, "Blue", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0)) 
			
	contours, hierarchy = cv2.findContours(orange_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
	for pic, contour in enumerate(contours): 
		area = cv2.contourArea(contour) 
		if(area > 1000): 
			x, y, w, h = cv2.boundingRect(contour) 
			imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (14, 117, 235), 2) 
			
			cv2.putText(imageFrame, "Orange", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (14, 117, 235)) 

	contours, hierarchy = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
	for pic, contour in enumerate(contours): 
		area = cv2.contourArea(contour) 
		if(area > 1000): 
			x, y, w, h = cv2.boundingRect(contour) 
			imageFrame = cv2.rectangle(imageFrame, (x, y), (x + w, y + h), (12, 237, 245), 2) 
			
			cv2.putText(imageFrame, "Yellow", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (12, 237, 245)) 

	contours, hierarchy = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	for pic, contour in enumerate(contours):
		area = cv2.contourArea(contour)
		if(area > 1000):
			x, y, w, h = cv2.boundingRect(contour)
			imageFrame = cv2.rectangle(
				imageFrame, (x, y), (x + w, y + h), (255, 255, 255), 2)

			cv2.putText(imageFrame, "White", (x, y),cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255))



	cv2.imshow("Multiple Color Detection in Real-TIme", imageFrame) 
	if cv2.waitKey(10) & 0xFF == ord('q'): 
		cap.release() 
		cv2.destroyAllWindows() 
		break

