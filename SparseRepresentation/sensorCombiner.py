'''
Created on 2014/4/13

@author: bhchen
'''
import sys,os

path = sys.argv[1]

files = os.listdir(path)
files_2 = sorted(files,key=len)
length = len(path.split("/"))
dir_name = path.split("/")[length-1]
if(os.listdir(path)):
    name = files[1].split("_")[0]
    output_f = open(path+"/"+name+"_"+dir_name+".csv",'w')
else:
    print "There is no file in this path."
    exit(0)

relation = open(path+'/'+files[0]).read().split('@data')[0].strip()

numSet = int(sys.argv[2])
if not(os.path.isdir(path+'/../sensor_combine_'+str(numSet))):    
    os.mkdir(path+'/../sensor_combine_'+str(numSet))
#os.rmdir(path+'/sensor_combine_'+str(numSet))
#os.mkdir(path+'/sensor_combine_'+str(numSet))    
for j in range(1440/numSet):
    output_f = open(path+'/../sensor_combine_'+str(numSet)+'/sensor_'+str(j)+'.0.arff','w')
    output_f.write(relation+'\n@data\n')
    for j in range(numSet):
        file_select = open(path+'/'+files_2[j*numSet+j]) 
        content = file_select.read().split('@data')[1].strip()
        output_f.write(content+'\n')
