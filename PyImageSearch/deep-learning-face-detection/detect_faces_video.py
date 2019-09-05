# USAGE
"""
python detect_faces_video.py --prototxt deploy.prototxt.txt
 --model res10_300x300_ssd_iter_140000.caffemodel
"""

# import the necessary packages
import argparse
import time
import imutils
from imutils.video import VideoStream
import numpy as np
import cv2

def detectFaceVideo(args):
	"Main function"
	# load our serialized model from disk
	print("[INFO] loading model...")
	net = cv2.dnn.readNetFromCaffe(ARGS["prototxt"], args["model"])

	# initialize the video stream and allow the cammera sensor to warmup
	print("[INFO] starting video stream...")
	vid_stream = VideoStream(src=0).start()
	time.sleep(2.0)

	# loop over the frames from the video stream
	while True:
		# grab the frame from the threaded video stream and resize it
		# to have a maximum width of 400 pixels
		frame = vid_stream.read()
		frame = imutils.resize(frame, width=400)

		# grab the frame dimensions and convert it to a blob
		(height, width) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		                             (300, 300), (104.0, 177.0, 123.0))

		# pass the blob through the network and obtain the detections and
		# predictions
		net.setInput(blob)
		detections = net.forward()

		# loop over the detections
		for i in range(0, detections.shape[2]):
			# extract the confidence (i.e., probability) associated with the
			# prediction
			confidence = detections[0, 0, i, 2]

			# filter out weak detections by ensuring the `confidence` is
			# greater than the minimum confidence
			if confidence < ARGS["confidence"]:
				continue

			# compute the (x, y)-coordinates of the bounding box for the
			# object
			box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
			(start_x, start_y, end_x, end_y) = box.astype("int")

			# draw the bounding box of the face along with the associated
			# probability
			text = "{:.2f}%".format(confidence * 100)
			_y = start_y - 10 if start_y - 10 > 10 else start_y + 10
			cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 0, 255), 2)
			cv2.putText(frame, text, (start_x, _y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

		# show the output frame
		cv2.imshow("Frame", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

	# do a bit of cleanup
	cv2.destroyAllWindows()
	vid_stream.stop()

if __name__ == "__main__":
    # execute only if run as a script
	# construct the argument parse and parse the arguments
	AP = argparse.ArgumentParser()
	AP.add_argument("-p", "--prototxt", required=True,
	                help="path to Caffe 'deploy' prototxt file")
	AP.add_argument("-m", "--model", required=True,
	                help="path to Caffe pre-trained model")
	AP.add_argument("-c", "--confidence", type=float, default=0.5,
	                help="minimum probability to filter weak detections")
	ARGS = vars(AP.parse_args())
	detectFaceVideo(ARGS)
