import numpy as np
from collections import Counter
import scipy.io
import pickle

with open('./testdata/test_data.txt') as f:
  docs = []
  for i,line in enumerate(f):
    c = Counter()
    for word in line.strip()[:-2].split(' 1 '):
      c[word] += 1
    docs.append(c)

  W = set()
  for doc in docs:
    W |= set(doc.keys())
  W = list(W)


  X = np.zeros(shape=(len(W), len(docs)), dtype=int)
  for i,doc in enumerate(docs):
    for j,word in enumerate(W):
      X[j,i] = doc[word]

  with open('test_data_dict.pkl', 'wb') as outfile:
    pickle.dump(W, outfile)
  scipy.io.mmwrite('test_data.mm', X)

# Row-indexed RDD: every element of the RDD is a row
