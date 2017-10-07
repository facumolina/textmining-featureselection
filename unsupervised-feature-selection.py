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
from gensim.models import Word2Vec
import numpy

CORPUS_FILE = 'resources/LaVanguardia.txt' # Text to be processed
CLUSTERS_NUMBER = 40 # Number of clusters of words

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

def frequent_words(sentences):
  # Returns the words that appear at least MIN_FREQUENCY times
  print("\nGetting most frequent words")
  
  words = []
  for sentence in sentences:
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

  vocabulary, vectors = gen_vectors(normalized)

  #frequent_words = frequent_words(sentences) # Get the most frequent words

  #vocabulary, features, classes, vectors, target = create_cooccurrence_matrix_and_target(sentences,frequent_words,class_to_use) # Create the co-occurrence matrix

  #new_vectors = feature_selection_from_model(vectors,target) # Perform feature selection from model

  km_model = gen_clusters(vectors) # Generate clusters

  show_results(vocabulary,km_model)








