# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 16:27:49 2015

@author: Lanrollend
"""
from xlsx2csv import *
xlsx2csv("dir\training1.xlsx", open("training1.csv", "w+"))

'''import os
 
dirname = your_directory_name
new_ext = ".csv"
fileList = []
for fn in os.listdir(dirname):
    fn1 = os.path.join(dirname, fn)
    if os.path.isfile(fn1):
        fileList.append(fn1)
print "\n".join(fileList)
 
output = []
for fn in fileList:
    dn, fn1 = os.path.split(fn)
    output.append("Old file name: %s\nNew file name: %s\n" %
                  (fn, os.path.join(dn, os.path.splitext(fn1)[0]+new_ext)))
print "\n".join(output)'''