# textmining-featureselection

Práctico de Feature selection para el curso [Text Mining](https://sites.google.com/view/mdt2017)

## objetivo

Encontrar espacios de menor dimensionalidad que mejoren una tarea de PLN, y después aplicar esos espacios una tarea de clustering. 

## detalles técnicos

Se utilizaron dos corpus: 
* El corpus _resources/tagged.es.tgz_, el [Wikicorpus](http://www.cs.upc.edu/~nlp/wikicorpus/) del español taggeado con POS (Part of Spech) y con los sentidos de Wordnet (para feature selection supervisado)
* El corpus _resources/LaVanguardia.txt.gz_, una recopilación de noticias del diario La Vanguardia (para feature selection no supervisado)

Se utilizaron las siguientes herramientas:
* [nltk](http://www.nltk.org/): 
* [scikit-learn](http://scikit-learn.org/stable/): 
* [gensim](https://radimrehurek.com/gensim/index.html): 

## Feature selection supervisado

### Corpus

Se utilizó el corpus _resources/tagged.es.tgz_, el cuál ya viene separado por tokens y lematizados. 

### Vectorización

* Se construyó la matriz de co-ocurrencia entre palabras en un contexto dado (dos palabras anteriores más las dos palabras siguientes).
* Se redujo la dimensionalidad de la matriz utilizando en las columnas sólo aquellas palabras que superaban un umbral de frecuencia dado. 
* Se obtuvieron los vectores a partir de las filas de la matriz resultante.

### Feature selection y Clustering

La técnica de feature selection que se utilizó fue [Wrapper sobre un classificador](http://scikit-learn.org/stable/modules/feature_selection.html#feature-selection-using-selectfrommodel)

Se utilizó dos veces, una vez usando los tags POS como clase y la otra usando los sentidos de wordnet cómo clases. Luego
se realizó la tarea de clustering sobre el espacio reducido.

**Wrapper sobre un clasificador con POS como clase**

**Wrapper sobre un clasificador con los sentidos de WordNet como clase**

## Feature selection no supervisado

### Corpus

### Vectorización

### Feature selection

### Clustering