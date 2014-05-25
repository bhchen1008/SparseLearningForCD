'''
Created on 2014/4/8

@author: bhchen
'''
import sys

input_f = open(sys.argv[1])

pathSplit = sys.argv[1].split('/')
path = pathSplit[0] + '/'
for j in range(1,len(pathSplit)-1):
    path += pathSplit[j] + '/'
    
filename = pathSplit[len(pathSplit)-1].split('.')[0]     

content = input_f.read()

fixedContent = content.split('@data')[0]
        

contentForSplit = content.split('@data')[1].strip()
lines = contentForSplit.split('\n')

dictR = {}
for line in lines:
    key = line.split(',')[0]
    value = line
    if key in dictR:
        dictR[key].append(value)
    else:
        dictR[key] = [value]

   
for key_value in dictR.keys():
    output_f = open(path+'/'+filename+'_split/'+filename+'_'+key_value+'.arff','w')
    
    fixedSplit = fixedContent.split('\n')
    for line in fixedSplit:
        if not('rcdminutes' in line) and not('@relation' in line):
            output_f.write(line+'\n')
        elif '@relation' in line:
            output_f.write(line+'-'+key_value)
        
    output_f.write('@data\n')
    for value in dictR[key_value]:
        valueSplit = value.split(',')
        for j in range(1,len(valueSplit)):
            if j !=1:
                output_f.write(',')
            output_f.write(valueSplit[j])
        output_f.write('\n')
    

