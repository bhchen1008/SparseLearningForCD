'''
Created on 2014/6/23

@author: bhchen
'''
from collections import deque
from collections import Counter

import operator

print 'python version:'+

def decideFinalDict(dictWindow):
    most_common,num_most_common = Counter(a).most_common(1)[0]
    num_second_common = Counter(a).most_common(2)[1][1]
    if(num_most_common==num_second_common):
        return 'same_most_common'
    else:
        return most_common

stats = {'a':1000, 'b':3000, 'c': 100}
print 'sorted:'+str(sorted(stats, key=lambda k: stats[k], reverse=True)[0])

print "max_dict:"+max(stats.iteritems(), key=operator.itemgetter(1))[0]


a = deque()
a.append('D1')
a.append('D2')
a.append('D3')
a.append('D4')
a.append('D5')
a.append('D1')
a.append('D1')
a.append('D1')
print a
print a.popleft()
print a
print a.pop()
print a
#a.append('D1')

#print 'D1:'+str(a.count('D1'))
#print 'D2:'+str(a.count('D2'))

finalDict = decideFinalDict(a)
if(finalDict=='same_most_common'):
    print 'same_most_common'
else:
    print 'most_common'+finalDict


#print set(a)
#most_common,num_most_common = Counter(a).most_common(1)[0]
#second_common,num_second_common = Counter(a).most_common(2)[1]
#print 'most_common:'+str(most_common)+',num_most_common:'+str(num_most_common)
#print 'second_common:'+str(second_common)+',num_second_common:'+str(num_second_common)
#if(num_most_common==num_second_common):
#    print 'no_most_common'
#else:
#    print 'most_common:' + most_common
#for i in set(a):
#    print str(i) + ":" + str(a.count(i))