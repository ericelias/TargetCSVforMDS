def mdscsv():

	from deepface import DeepFace			### All the modules needed
	import matplotlib.pyplot as plt
	import pandas as pd
	import csv

	models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
	modelname = models[1]		### Facial recognition program to use. Type the number from the list above 
								### that you want to use into models[]

	csvloc = "/Users/EricElias/Desktop/test/test.csv" 			### Place csv path into the quotes
																### A new .csv will be created if there isn't one in the path

	### Input target paths in quotes separated by commas into list (can be any amount of targets)
	imglist = ["/Users/EricElias/Desktop/test/CFD-MF-319-016-N.jpg",
	"/Users/EricElias/Desktop/test/CFD-AF-200-228-N.jpg",
	"/Users/EricElias/Desktop/test/CFD-AF-201-060-N.jpg",
	"/Users/EricElias/Desktop/test/CFD-AF-202-122-N.jpg",
	"/Users/EricElias/Desktop/test/CFD-BF-001-025-N.jpg",
	"/Users/EricElias/Desktop/test/CFD-BF-002-001-N.jpg",
	"/Users/EricElias/Desktop/test/CFD-BF-003-003-N.jpg",
	"/Users/EricElias/Desktop/test/CFD-MF-312-002-N.jpg",
	"/Users/EricElias/Desktop/test/CFD-MF-313-002-N.jpg"]

	targets = [] 									### Creates list with abbreviated pathnames of targets to use in csv file
	for z in imglist:								
	    targets.append(z[-20:-4])					### You'll have to adjust append.z[] based on your filenames

	header = [""] 									### Creates header list from targets to be analyzed
	for i in range(len(targets)):
		header.append(targets[i])

	with open(csvloc, "w") as f:					### Creates headers in csv using the header list
		dw = csv.DictWriter(f, delimiter=',', fieldnames=header)
		dw.writeheader()

	for i in range(len(targets)): 					### Creates similiarity judgements and places into csv
		newrow = [targets[i]]
		for o in range(len(imglist)):
			DeepFace.detectFace(imglist[i])
			DeepFace.detectFace(imglist[o])
			resp = DeepFace.verify(img1_path = imglist[i], img2_path = imglist[o], model_name = modelname)
			if resp['distance'] < 0.001:
				resp['distance'] = 0
			newrow.append(resp['distance'])
		with open(csvloc, "a") as f:
			data = csv.writer(f)
			data.writerow(newrow)
			
	f.close()
mdscsv()
