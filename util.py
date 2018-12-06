import numpy as np

def read_textfile(filename):
	with open(filename) as f:
	    polyShape = []
	    for line in f:
	        line = line.split()# to deal with blank 
	        #print(line)
	        if line:            # lines (ie skip them)
	            line = [float(i) for i in line]
	            polyShape.append(line)


	X = np.zeros((len(polyShape),6))
	count = 0
	for i in polyShape:
	    temp=np.array(i)
	    #print(len(temp))
	    X[count,0:len(temp)] = temp[0:len(temp)]
	    count +=1   
	        
	return X        
        