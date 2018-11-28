#!/usr/bin/env python
import cv2
import numpy as np
from video import create_capture
from common import clock, draw_str
from face_tracker import FaceTracker


if __name__ == '__main__':

	import sys, getopt

	video_src = 0
	
	cascade_fn = "data/haarcascade_frontalface_alt2.xml"
	
	cam = create_capture(video_src, fallback='synth:bg=../cpp/lena.jpg:noise=0.05')

	scale=0.25
	scaleFactor=1.3
	
	tracker = FaceTracker(cascade_fn,scale,scaleFactor)

	while True:

		ret, img = cam.read()

		t = clock()
		
		npoints = tracker.detect(img)
		
		if npoints is not None:
		
			img=tracker.draw_rectangle(img,npoints)					
			
		dt = clock() - t
	
		draw_str(img, (20, 20), 'time: %.1f ms' % (dt*1000))
		cv2.imshow('Rotation Invariant Face Detection', img)
		

		if 0xFF & cv2.waitKey(5) == 27:
			break
	cam.release()

	cv2.destroyAllWindows()
