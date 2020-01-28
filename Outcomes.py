import csv

seen=[]
with open('train.tsv') as tsvfile:
  reader = csv.DictReader(tsvfile, dialect='excel-tab')
  for row in reader:
      if row['label'] not in seen:
          seen.append(row['label'])
for i in range(len(seen)):
    print(seen[i])
    

