'''
Created on 2014/3/25

@author: bhchen
'''
import sys
input_f = open(sys.argv[1])
pathSplit = sys.argv[1].split('\\')
name = pathSplit[len(pathSplit)-1].split('\.')[0]
folds = int(sys.argv[2])
content = input_f.read()

fixedContent = content.split('@data')[0]
contentForSplit = content.split('@data')[1].strip()
lines = contentForSplit.split('\n')
ips = len(lines)/folds
for j in range(folds):
    output_f = open(sys.argv[3]+'\\'+name+'_'+str(folds)+'-'+str(j+1)+'.arff','w')
    output_f.write(fixedContent+'@data\n')
    if j == folds-1:
        for j in range((folds-1)*ips,len(lines)):
            output_f.write(lines[j]+'\n')
    else:
        for instTmp in range(ips):
            output_f.write(lines[j*ips+instTmp]+'\n')