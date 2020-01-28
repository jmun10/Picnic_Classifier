import csv
import os
#runs first
#creates new directories
print("organizing train images")

localpath=r"C:\Users\jesus\Desktop\The Picnic Hackathon 2019 - Copy"

list=open(file="types.txt").readlines()
print(list)
#removes newline chars
for i in range(len(list)):
    if list[i]=="Pineapples, melons & passion fruit":
        break
    list[i]=list[i][:-1]

for i in range(len(list)):
    newdir=localpath+r"/{}".format(list[i])
    os.mkdir(newdir)

print("train images have been organized")