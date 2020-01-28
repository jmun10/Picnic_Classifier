import csv
import os

#runs second
#moves image files to proper location
print("sorting training images")
corruptimages=[]

#Localdir should be modified to user
localDir= r"C:\Users\jesus\Desktop\The Picnic Hackathon 2019 - Copy"
origDir= r"C:\Users\jesus\Desktop\The Picnic Hackathon 2019 - Copy\train"

#use train-jpeg.tsv for new version
#ensure Convert_to_jpeg is ran first

#use train.tsv for non jpeg conversion
with open("train-jpeg.tsv") as tsvfile:
    reader = csv.DictReader(tsvfile, dialect='excel-tab')
    counter=0
    for row in reader:
        #print(row["file"])
        counter+=1

        #keep track of progress
        if counter%1000==0:
            print(str(counter) + " files sorted")

        #skips over missing images
        if counter==3982 or counter==4059 or counter==4893 or counter==5399 or counter==6611:
            counter+=1

        #writes address for file
        #checks for corruption of jpegs
        '''
        if "jpeg" in row["file"]:
            print(row["file"])
            data=open(origDir+ r"/{}".format(row["file"]),"rb")
            if data.readline()[:2]!= b"\xff\xd8":
                corruptimages.append(row["file"])
            data.close()'''





        newDir= localDir + r"/{}".format(row["label"])
        newDir=newDir+r"/{}".format(row["file"])
        oldDir=origDir+r"/{}".format(row["file"])
        #moves files
        os.rename(oldDir, newDir)


print("sorted training images")

print("Corruped files are as follows:")
print(corruptimages)

