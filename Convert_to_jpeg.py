import os
import csv
from PIL import Image
#runs first IF converting to JPEG
print("converting all files to jpegs")

localDir= r"C:\Users\jesus\Desktop\The Picnic Hackathon 2019 - Copy\train"

with open("train.tsv") as tsvfile:
    reader = csv.DictReader(tsvfile, dialect='excel-tab')
    counter=0
    for row in reader:
        counter += 1

        #skip over missing images
        if counter==3982 or counter==4059 or counter==4893 or counter==5399 or counter==6611:
            counter+=1

        #for every png, convert to jpeg
        if "png" in row["file"]:

            imagedir=localDir+r"/{}".format(row["file"])
            newimagedir=localDir+r"/{}".format(row["file"][0:-4]+".jpeg")

            im = Image.open(imagedir)
            rgb_im = im.convert('RGB')
            rgb_im.save(newimagedir)
            os.remove(imagedir)




print("conversion to jpeg done")