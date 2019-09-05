# USAGE
"""
# python detect_faces.py --image rooster.jpg --prototxt deploy.prototxt.txt
 --model res10_300x300_ssd_iter_140000.caffemodel
"""
# import the necessary packages
import argparse
import numpy as np
import cv2

def detectFacesImage(args):
	"Main function"
	# load our serialized model from disk
	print("[INFO] loading model...")
	net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

	# load the input image and construct an input blob for the image
	# by resizing to a fixed 300x300 pixels and then normalizing it
	image = cv2.imread(args["image"])
	(height, width) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
	                             (300, 300), (104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the detections and
	# predictions
	print("[INFO] computing object detections...")
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with the
		# prediction
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# compute the (x, y)-coordinates of the bounding box for the
			# object
			box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
			(start_x, start_y, end_x, end_y) = box.astype("int")

			# draw the bounding box of the face along with the associated
			# probability
			text = "{:.2f}%".format(confidence * 100)
			_y = start_y - 10 if start_y - 10 > 10 else start_y + 10
			cv2.rectangle(image, (start_x, start_y), (end_x, end_y),
			              (0, 0, 255), 2)
			cv2.putText(image, text, (start_x, _y),
			            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	# show the output image
	cv2.imshow("Output", image)
	cv2.waitKey(0)

if __name__ == "__main__":
    # execute only if run as a script
	# construct the argument parse and parse the arguments
	AP = argparse.ArgumentParser()
	AP.add_argument("-i", "--image", required=True,
	                help="path to input image")
	AP.add_argument("-p", "--prototxt", required=True,
	                help="path to Caffe 'deploy' prototxt file")
	AP.add_argument("-m", "--model", required=True,
	                help="path to Caffe pre-trained model")
	AP.add_argument("-c", "--confidence", type=float, default=0.5,
	                help="minimum probability to filter weak detections")
	ARGS = vars(AP.parse_args())
	detectFacesImage(ARGS)
