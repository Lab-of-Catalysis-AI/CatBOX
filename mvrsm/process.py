import os
import numpy as np


def read_logs_MVRSM(folder):
	#folder = 'MVRSM/'
	allfiles = os.listdir(folder)
	logfilesMV = [f for f in allfiles if ('.log' in f and 'MVRSM' in f)]
	MVbests = []
	for log in logfilesMV:
		with open(os.path.join(folder,log),'r') as f:
			MVRSMfile = f.readlines()
			MVRSM_best = []
			for i, lines in enumerate(MVRSMfile):
				searchterm = 'Best data point according to the model and predicted value'
				if searchterm in lines:
					#print('Hello', MVRSMfile)
					temp = MVRSMfile[i-1]
					temp = temp.split('] , ')
					temp = temp[1].strip()
					temp = temp.strip('[')
					temp = temp.strip(']')
					MVRSM_best.append(float(temp))
		MVbests.append(MVRSM_best)

	return np.asarray(MVbests).reshape(-1, 1)
	

def read_logs_TPE(file):
	HObests = []
	with open(file,'r') as f:
		best = 10e9
		HOfile = f.readlines()
		HOfile = HOfile[0]
		HOfile = HOfile.split(',')
		HO_ev = []
		for i, lines in enumerate(HOfile):
			searchterm1 = "'result': {'loss': "
			if searchterm1 in lines:
				temp1 = lines
				temp1 = temp1.split(searchterm1)
				temp1 = temp1[1]
				temp1 = float(temp1)
				if temp1 < best:
					best = temp1
					HO_ev.append(temp1)
				else:
					HO_ev.append(best)
	HObests.append(HO_ev)

	return np.asarray(HObests).reshape(-1, 1)
	
def read_logs_RS(file):
	RSbests = []
	with open(file,'r') as f:
		best = 10e9
		RSfile = f.readlines()
		RSfile = RSfile[0]
		RSfile = RSfile.split(',')
		RS_ev = []
		for i, lines in enumerate(RSfile):
			searchterm1 = "'result': {'loss': "
			if searchterm1 in lines:
				temp1 = lines
				temp1 = temp1.split(searchterm1)
				temp1 = temp1[1]
				temp1 = float(temp1)
				if temp1 < best:
					best = temp1
					RS_ev.append(temp1)
				else:
					RS_ev.append(best)
	RSbests.append(RS_ev)

	return np.asarray(RSbests).reshape(-1, 1)