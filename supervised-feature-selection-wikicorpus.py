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

CORPUS_FILES = 'resources/spanishEtiquetado_10000_15000' # Words files
MIN_FREQUENCY = 10 # Min word frequency to be considered
WINDOWS_SIZE = 2 # Windows size to determine the contexts
WORD_CLASS_INDEX = 2 # Use the POS tag as word class by default
CLUSTERS_NUMBER = 40 # Number of clusters of words
WORD_INDEX = 0 # Index of words in lines
MAX_SENTECES = 30000 # Sentences to read

def read_words_in_sentences():
  # Read sentences from CORPUS_FILES
  print("\nLoading sentences")
  files = glob.glob(CORPUS_FILES)
  sentences = []

  i = 0;
  for text in files:
    try:
      with codecs.open(text, 'r', 'latin1') as f:
        # number of lines from txt file
        in_sentence = False
        sentence = []
        for line in f:
          if "<doc id" in line or len(line)==1: # Doc start
            if (in_sentence):
              i += 1
              sentences.append(sentence)
            sentence = []
            in_sentence = True
          else:
            if "</doc" in line: # Doc end
              in_sentence = False
            else:
              sentence.append(line)
          if (i==MAX_SENTECES):
            break

    except IOError as exc:
      # Do not fail if a directory is found, just ignore it.
      if exc.errno != errno.EISDIR:
        raise

  print("Total sentences:",len(sentences))
  return sentences

def frequent_words(words_tagged_sentences):
  # Returns the words that appear at least MIN_FREQUENCY times
  print("\nGetting most frequent words")
  
  words = []
  for sentence in words_tagged_sentences:
    for word in sentence:
      words.append(word.split()[WORD_INDEX])

  words = [token.lower() for token in words if token.isalpha()]

  most_frequents = []
  counter = Counter(words)
  for w in counter:
    if (counter[w]>=MIN_FREQUENCY and w not in stopwords.words('spanish')):
      most_frequents.append(w)
  print("Most frequent words calculated. Total:",str(len(most_frequents)))
  return most_frequents

def create_cooccurrence_matrix_and_target(words_tagged_sentences,frequent_words,class_to_use):
  # Create coocurrence matrix. Only create columns for those words that are in frequent_words
  print("\nCreating co-occurrence matrix using the",class_to_use,"as word class")
  set_all_words={}
  set_freq_words={}
  set_classes={}
  data=[]
  row=[]
  col=[]

  target_data=[]
  target_row = []
  target_col=[]

  for sentence in words_tagged_sentences:
    tokens_and_class = [(w.split()[WORD_INDEX],w.split()[WORD_CLASS_INDEX]) for w in sentence]
    tokens_and_class = [(token.lower(),tokenclass) for (token,tokenclass) in tokens_and_class if token.isalpha()]  # All alpha tokens to lowercase
    tokens_and_class = [(token,tokenclass) for (token,tokenclass) in tokens_and_class if token not in stopwords.words('spanish')] # Remove stopwords

    for pos,(token,tokenclass) in enumerate(tokens_and_class):
      i=set_all_words.setdefault(token,len(set_all_words))
      c=set_classes.setdefault(tokenclass,len(set_classes))
      start=max(0,pos-WINDOWS_SIZE)
      end=min(len(tokens_and_class),pos+WINDOWS_SIZE+1)
      for pos2 in range(start,end):
        token_to_compare = tokens_and_class[pos2][0]
        if pos2==pos or token_to_compare not in frequent_words:
          continue
        j=set_freq_words.setdefault(token_to_compare,len(set_freq_words))
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

  class_to_use = sys.argv[1]
  if (class_to_use=="pos"):
    WORD_CLASS_INDEX=2
    MAX_SENTECES=200000
  if (class_to_use=="wordnet-senses"):
    WORD_CLASS_INDEX=3
    MAX_SENTECES = 30000

  sentences = read_words_in_sentences() # Read sentences

  frequent_words = frequent_words(sentences) # Get the most frequent words

  vocabulary, features, classes, vectors, target = create_cooccurrence_matrix_and_target(sentences,frequent_words,class_to_use) # Create the co-occurrence matrix

  new_vectors = feature_selection_from_model(vectors,target) # Perform feature selection from model

  km_model = gen_clusters(new_vectors) # Generate clusters

  show_results(vocabulary,km_model)








