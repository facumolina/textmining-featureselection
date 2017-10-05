import random
import os
import glob
import sys
import errno
import codecs
from nltk.corpus import stopwords
from collections import Counter
from scipy.sparse import coo_matrix
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.cluster import KMeans
from sklearn import preprocessing 

WORDS_SAMPLE_SIZE = 500000 # Words sample size
CORPUS_FILES = 'resources/spanishEtiquetado_10000_15000' # Words files
MIN_FREQUENCY = 10 # Min word frequency to be considered
WINDOWS_SIZE = 2 # Windows size to determine the contexts
WORD_CLASS_INDEX = 2 # Use the POS tag as word class. Change to 3 to use the wordnet senses as word class
CLUSTERS_NUMBER = 40 # Number of clusters of words
WORD_INDEX = 0 # Index of words in lines

def read_words_sample():
  # Read a random sample of WORDS_SAMPLE_SIZE from the corpus files CORPUS_FILES
  print("\nGetting words sample of size",str(WORDS_SAMPLE_SIZE))
  files = glob.glob(CORPUS_FILES)
  words_lines = []

  for text in files:
    try:
      with codecs.open(text, 'r', 'latin1') as f:
        # number of lines from txt file
        for line in f:
          if (len(line.split())==4):
            words_lines.append(line)

    except IOError as exc:
      # Do not fail if a directory is found, just ignore it.
      if exc.errno != errno.EISDIR:
        raise

  random_sample_input = random.sample(words_lines, WORDS_SAMPLE_SIZE)
  return random_sample_input

def frequent_words(words_tagged_sample):
  # Returns the words that appear at least MIN_FREQUENCY times
  print("\nGetting most frequent words")
  
  words = [w.split()[WORD_INDEX] for w in words_tagged_sample] 

  words = [token.lower() for token in words]

  most_frequents = []
  counter = Counter(words)
  for w in counter:
    if (counter[w]>=MIN_FREQUENCY):
      most_frequents.append(w)
  print("Most frequent words calculated. Total:",str(len(most_frequents)))
  return most_frequents

def create_cooccurrence_matrix_and_target(words_tagged,frequent_words):
  # Create coocurrence matrix. Only create columns for those words that are in frequent_words
  print("\nCreating co-occurrence matrix")
  set_all_words={}
  set_freq_words={}
  set_classes={}
  data=[]
  row=[]
  col=[]

  target_data=[]
  target_row = []
  target_col=[]

  all_tokens = [w.split()[WORD_INDEX] for w in words_tagged]
  all_tokens = [token.lower() for token in all_tokens if token.isalpha() and token not in stopwords.words('spanish')]
  all_classes = [w.split()[WORD_CLASS_INDEX] for w in words_tagged]
  for pos,token in enumerate(all_tokens):
    i=set_all_words.setdefault(token,len(set_all_words))
    c=set_classes.setdefault(all_classes[pos],len(set_classes))
    start=max(0,pos-WINDOWS_SIZE)
    end=min(len(all_tokens),pos+WINDOWS_SIZE+1)
    for pos2 in range(start,end):
      if pos2==pos or all_tokens[pos2] not in frequent_words:
        continue
      j=set_freq_words.setdefault(all_tokens[pos2],len(set_freq_words))
      data.append(1.); row.append(i); col.append(j);
      if i not in target_row:
        target_data.append(c); target_row.append(i); target_col.append(0);

  cooccurrence_matrix=coo_matrix((data,(row,col)))
  target_vector = coo_matrix((target_data,(target_row,target_col)))
  print("Vocabulary size:",len(set_all_words))
  print("Matrix shape:",cooccurrence_matrix.shape)
  print()
  print("Target shape: ",target_vector.shape)
  print("Co-occurrence matrix finished")

  return set_all_words,set_freq_words,set_classes,cooccurrence_matrix,target_vector.toarray()

def feature_selection_from_model(data,target):
  # Perform feature selection with a Tree-based estimator
  print("\nPerforming feature selection. Original shape:",data.shape)
  clf = ExtraTreesClassifier()
  data = preprocessing.normalize(data)
  clf = clf.fit(data, target.ravel())
  model = SelectFromModel(clf,prefit=True)
  new_data = model.transform(data)
  print("New shape:",new_data.shape)
  print("Feature selection finished")
  return new_data;

def gen_clusters(vectors):
  # Generate word clusters using the k-means algorithm.
  print("\nClustering started")
  km_model = KMeans(n_clusters=CLUSTERS_NUMBER)
  km_model.fit(vectors)
  print("Clustering finished")
  return km_model

def show_results(vocabulary,model):
  # Show results
  c = Counter(sorted(model.labels_))
  print("\nTotal clusters:",len(c))
  for cluster in c:
    print ("Cluster#",cluster," - Total words:",c[cluster])

  # Show top terms and words per cluster
  print("Top words per cluster:")
  print()

  keysVocab = list(vocabulary.keys())
  for n in range(len(c)):
    print("Cluster %d" % n)
    print("Words:", end='')
    word_indexs = [i for i,x in enumerate(list(model.labels_)) if x == n]
    for i in word_indexs:
      print(' %s' % keysVocab[i], end=',')
    print()
    print()

if __name__ == "__main__":

  fstechnique = sys.argv[1]

  words_tagged_sample = read_words_sample() # Read a words sample

  frequent_words = frequent_words(words_tagged_sample) # Get the most frequent words

  vocabulary, features, classes, vectors, target = create_cooccurrence_matrix_and_target(words_tagged_sample,frequent_words) # Create the co-occurrence matrix

  new_vectors = feature_selection_from_model(vectors,target) # Perform feature selection from model

  km_model = gen_clusters(new_vectors) # Generate clusters

  show_results(vocabulary,km_model)








