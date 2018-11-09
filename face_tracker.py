
"""
Created on Fri Nov 9

@author: Anjith George,
email: anjith2006@gmail.com

For details refer to:

1. Dasgupta A, George A, Happy SL, Routray A. A vision-based system for monitoring the loss of attention in automotive drivers. IEEE Transactions on Intelligent Transportation Systems. 2013 Dec;14(4):1825-38.
2. George A, Dasgupta A, Routray A. A framework for fast face and eye detection. arXiv preprint arXiv:1505.03344. 2015 May 13.
"""


import numpy as np
import cv2

class FaceTracker():
	def __init__(self,cascade_fn,scale=1,scaleFactor=1.3,minSize=(30,30)):
		print("cascade_fn",cascade_fn)
		self.prev_angle=0
		self.frames=0
		self.cascade= cv2.CascadeClassifier(cascade_fn)
		self.scale=scale
		self.scaleFactor=scaleFactor
		self.minSize=minSize
		self.prev_points=[]

	def detect(self,frame):
		rects=[]
		acount=0
		dx=30
		angle=self.prev_angle
		maxtimes=360/dx+1
		times=0

		img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		rimg = cv2.resize(img,None,fx=self.scale, fy=self.scale, interpolation = cv2.INTER_LINEAR)
		
		while len(rects)==0 and acount<maxtimes:
			rows,cols = rimg.shape
			times=times+1
			
			M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
			imgw = cv2.warpAffine(rimg,M,(cols,rows))
			rects = self.cascade.detectMultiScale(imgw, scaleFactor=self.scaleFactor, minNeighbors=4, minSize=self.minSize, flags = 2)
			acount=acount+1
			sign=pow(-1,acount)
			self.prev_angle=angle
			angle=angle+(sign*acount*dx)
			angle=angle%360
		if len(rects) == 0:
			return None
		
		
		rects[:,2:] += rects[:,:2]

		points=[]

		try:
			x1, y1, x2, y2 =rects[0]

			height=x2-x1
			width=y2-y1

			points.append((x1,y1))
			points.append((x1,y1+width))
			points.append((x2,y2))

			points.append((x2,y2-width))
		except:
			pass


		self.prev_points=points

		npoints=None


		if len(points)==4:

	
			c=np.array(points)


			iM=cv2.invertAffineTransform(M)


			extra=np.array([0.0,0.0,1.0])

			iM=np.vstack((iM,extra))


			cc=np.array([c],dtype='float')


			conv=cv2.perspectiveTransform(cc,iM)

			npoints=[]

			for vv in conv[0]:
				
				npoints.append((int(vv[0]/self.scale),int(vv[1]/self.scale)))


		return npoints


	def draw_rectangle(self,img,npoints):

		cv2.line(img,npoints[0],npoints[1],(0,255,0),3)
		cv2.line(img,npoints[1],npoints[2],(0,255,0),3)
		cv2.line(img,npoints[2],npoints[3],(0,255,0),3)
		cv2.line(img,npoints[3],npoints[0],(0,0,255),4)

		return img

