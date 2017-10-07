import os
import glob
import sys
import errno
import codecs
import nltk
from nltk.corpus import stopwords
from collections import Counter
from scipy.sparse import coo_matrix
from sklearn.cluster import KMeans
from sklearn import preprocessing 
from sklearn.decomposition import PCA
from gensim.models import Word2Vec
import numpy

CORPUS_FILE = 'resources/LaVanguardia.txt' # Text to be processed
CLUSTERS_NUMBER = 40 # Number of clusters of words
MIN_FREQUENCY = 10  # Min word frequency to be considered
WINDOWS_SIZE = 2 # Windows size to determine the contexts

def readFile():
  # Read the file TEXT_FILE.
  print("Loading file",CORPUS_FILE)
  f = codecs.open(CORPUS_FILE,'r','latin1')
  content = f.read()
  return content

def tokenize(text):
  # Tokenize and normalize the given text.
  sents = nltk.sent_tokenize(text)

  tokenized_sents = [nltk.word_tokenize(sent) for sent in sents]

  tokenized_sents = [process_tokens(sent) for sent in tokenized_sents]

  return tokenized_sents

def process_tokens(tokens):
  # Process the given list of tokens
  tokens = [token.lower() for token in tokens]  # All tokens to lowercase

  words = [token for token in tokens if token.isalpha()]  # Maintain strings with alphabetic characters

  words = [token for token in words if token not in stopwords.words('spanish')] # Remove stopwords

  wnl = nltk.WordNetLemmatizer()
  lemmatized = [wnl.lemmatize(t) for t in words] # Lemmatization

  return lemmatized

def gen_vectors(normalized_text):
  # Generate word vectors using neural word embeddings
  print("\nGenerating word vectors")
  model = Word2Vec(normalized,size=100,window=5,min_count=5)
  vects = []
  for word in model.wv.vocab:
    vects.append(model.wv[word])

  matrix = numpy.array(vects)
  print("Matrix shape:",matrix.shape)
  print("Vectors generated")
  return model.wv.vocab,matrix

def frequent_words(text):
  # Returns the words that appear at least MIN_FREQUENCY times
  print("\nGetting most frequent words")
  
  words = tokens = nltk.word_tokenize(text)
  words = [token.lower() for token in words]
  wnl = nltk.WordNetLemmatizer()
  words = [wnl.lemmatize(t) for t in words] # Lemmatization

  most_frequents = []
  counter = Counter(words)
  for w in counter:
    if (counter[w]>=MIN_FREQUENCY):
      most_frequents.append(w)
  print("Most frequent words calculated. Total:",str(len(most_frequents)))
  return most_frequents

def create_cooccurrence_matrix(sentences,frequent_words):
  # Create coocurrence matrix. Only create columns for those words that are in frequent_words
  print("\nCreating co-occurrence matrix")
  set_all_words={}
  set_freq_words={}
  data=[]
  row=[]
  col=[]
  for sentence in sentences:
    tokens=sentence
    for pos,token in enumerate(tokens):
      i=set_all_words.setdefault(token,len(set_all_words))
      start=max(0,pos-WINDOWS_SIZE)
      end=min(len(tokens),pos+WINDOWS_SIZE+1)
      for pos2 in range(start,end):
        if pos2==pos or tokens[pos2] not in frequent_words:
          continue
        j=set_freq_words.setdefault(tokens[pos2],len(set_freq_words))
        data.append(1.); row.append(i); col.append(j);
  cooccurrence_matrix=coo_matrix((data,(row,col)))
  print("Vocabulary size:",len(set_all_words))
  print("Matrix shape:",cooccurrence_matrix.shape)
  print("Co-occurrence matrix finished")
  return set_all_words,set_freq_words,cooccurrence_matrix

def unsupervised_fs_pca(vectors):
  print("PCA reduction. Original shape:",vectors.shape)
  pca = PCA(n_components=100)
  vectors = preprocessing.normalize(vectors)
  pca.fit(vectors.todense())
  new_vectors = pca.transform(vectors.todense())
  print("Finished feature selection. Shape:",new_vectors.shape)
  return new_vectors

def gen_clusters(vectors):
  # Generate word clusters using the k-means algorithm.
  print("\nClustering started")
  vectors = preprocessing.normalize(vectors)
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

  keysVocab = list(vocabulary.keys())
  for n in range(len(c)):
    print("Cluster %d" % n)
    print("Words:", end='')
    word_indexs = [i for i,x in enumerate(list(model.labels_)) if x == n]
    for i in word_indexs:
      print(' %s' % keysVocab[i], end=',')
    print()
    print()

  print()


if __name__ == "__main__":

  file_content = readFile() # Read the CORPUS_FILE
  normalized = tokenize(file_content)
  vocabulary = {}
  vectors = []

  unsup_method = sys.argv[1]
  if (unsup_method=="pca"):
    frequent_words = frequent_words(file_content) # Get the most frequent words
    vocabulary, features, vectors = create_cooccurrence_matrix(normalized,frequent_words)
    vectors = unsupervised_fs_pca(vectors)
  if (unsup_method=="embeddings"):
    vocabulary, vectors = gen_vectors(normalized)
    

  km_model = gen_clusters(vectors) # Generate clusters

  show_results(vocabulary,km_model)








