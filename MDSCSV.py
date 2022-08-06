def mdscsv():

	from deepface import DeepFace			### All the modules needed
	import matplotlib.pyplot as plt
	import pandas as pd
	import csv
	from striprtf.striprtf import rtf_to_text
	
	models = ["VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace", "DeepID", "ArcFace", "Dlib", "SFace"]
	modelname = models[4]		### Facial recognition program to use. Type the number from the list above 
					### that you want to use into models[]

	csvloc = "" 			### Place csv path into the quotes
					### A new .csv will be created if there isn't one in the path

	### Input rtf path that contains paths of target images
	textfilertf = ""


	with open(textfilertf) as infile:
    		content = infile.read()
    		text = rtf_to_text(content)
	newlist = [text]
	imglist = newlist[0].split()

	targets = [] 							### Creates list with abbreviated pathnames of targets to use in csv file
	for z in imglist:								
	    targets.append(z[-14:-4])					### You'll have to adjust append.z[] based on your filenames

	header = [""] 							### Creates header list from targets to be analyzed
	for i in range(len(targets)):
		header.append(targets[i])

	with open(csvloc, "w") as f:					### Creates headers in csv using the header list
		dw = csv.DictWriter(f, delimiter=',', fieldnames=header)
		dw.writeheader()

	for i in range(len(targets)): 					### Creates similiarity judgements and places into csv
		newrow = [targets[i]]
		for o in range(len(imglist)):
			resp = DeepFace.verify(img1_path = imglist[i], img2_path = imglist[o], model_name = modelname)
			if resp['distance'] < 0.001:
				resp['distance'] = 0
			newrow.append(resp['distance'])
		with open(csvloc, "a") as f:
			data = csv.writer(f)
			data.writerow(newrow)	
	f.close()
mdscsv()
