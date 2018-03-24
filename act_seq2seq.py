import pickle
import csv

acts = pickle.load(open('activity_seq.dump','rb'))
f = open('seqpat.csv','wb')
writer = csv.writer(f)

user_id = 0
for act_seq in acts:
    for i, act in enumerate(act_seq):
        writer.writerow([user_id, i, act])
    user_id = user_id + 1

f.close()
