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

* Se construyó la matriz de co-ocurrencia entre palabras en un contexto dado.
* Se redujo la dimensionalidad de la matriz utilizando en las columnas sólo aquellas palabras que superaban un umbral de frecuencia dado. 
* Se obtuvieron los vectores a partir de las filas de la matriz resultante.

Para construir la matriz de co-ocurrencias se tuvieron en cuenta los siguientes parámetros:

* **Frecuencia mínima** = 10 (sólo se consideraron las columnas correspondientes a palabras que ocurrieran al menos 10 veces)
* **Tamaño de ventana** = 2 (cantidad de palabras anteriores y siguientes a considerar para formar un contexto. Por ejemplo, en el caso de _word0 **word1** **word2** WORD3 **word4** **word5** word6_, un contexto de WORD3 estaría formado por [_word1, word2, word4, word5_]).

### Feature selection y Clustering

La técnica de feature selection que se utilizó fue [Wrapper sobre un classificador](http://scikit-learn.org/stable/modules/feature_selection.html#feature-selection-using-selectfrommodel) utilizando [ExtraTreeClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.tree.ExtraTreeClassifier.html).

Se utilizó dos veces, una vez usando los tags POS como clase y la otra usando los sentidos de wordnet cómo clases. Luego
se realizó la tarea de clustering sobre el espacio reducido.

**Wrapper sobre un clasificador con POS como clase**

Este caso se puede ejecutar con el siguiente comando:
	_python -i supervised-feature-selection-wikicorpus.py pos_ (asegurar de que el archivo _resources/tagged.es.tgz_ fue descomprimido y que aparece la parte spanishEtiquetado_10000_15000 del corpus)

La matriz original tenía tamaño (106402,17625). Luego de aplicar la técnica de feature selection quedó reducida un tamaño de (106402, 4595).

En el siguiente listado podemos ver algunas palabras de algunos de los clusters más interesantes:

	Cluster 6
	Words: bizantinos, destruida, conquistada, roma, funda, monasterio, fundó, gobernada, fez, fundada, capturan, asedio, constantinopla, grande, gobernante, córdoba, murcia, abbas, gerona, saqueada, amalfi, conquista, arrasada, incendio, saquea, temporalmente, condado, zaragoza, fundación, destruye, fortaleza, apoderan, zar, cerca, reconquista, histórico, somalia, fundando, reims, antigua, sede, sagrada, saldaña, santa, arrasó, lund, arrasa, praga, nació, kazán, fundaron, afueras, sicilia, tiro, castillo, jerusalén, hangzhou, destruyendo, 

	Cluster 9
	Words: emperatriz, madre, princesa, hija, esposa, infanta, doña, berenguela, petronila, profetisa, numerales, hermana, sonur, dóttir, matrimonio, crecida, jin, maría, sublevaron, sobrina, casó, julia, heredera, constanza, ana, afecto, llamarán, margaret, isabel, julie, josefina, urraca, rescatar, panista, chófer, activista, antígona, perecieron, facilitara, teodora, catalina, deméter, delfín, esposo, 

	Cluster 13
	Words: delhi, wohne, ísh, vóne, live, women, anno, quarto, vox, excelso, charles, dutch, zit, jan, nieuwe, winkel, gezien, erlossen, vrede, siele, verlossen, zal, ziel, mijn, shell, military, key, scholars, lumière, berlin, tribüne, ripubblicata, kairuan, oder, eine, vom, dieser, teoria, künstlers, wedderkop, anche, biographische, secondo, kritik, naturstudiums, katalog, zum, festschrift, 

	Cluster 17
	Words: abe, high, king, sex, mars, isbn, an, and, source, ann, war, the, school, looney, wonders, commonwealth, edmund, diet, was, mine, ethnologue, camerunes, affair, npn, how, bugsy, by, music, master, hall, centers, code, art, thinking, between, end, theory, memory, cities, american, life, common, fields, link, system, encyclopaedia, mathematics, darkness, hearts, time, recording, comparative, analysis, 

	Cluster 18
	Words: sucede, papa, vitaliano, dono, agatón, conón, consagra, sisinio, eugenio, valentín, succesión, gwriad, formoso, landon, canoniza, aelfheah, sonrisa, canonizó, canonizado, bula, abnegación, perinde, canonización, tetrarquía, liberio, vigiliosucede, vigilio, sabiniano, severino, sisino, ptm, suced, lupino, cedidas, wojty, excomulgar, luterana, fallecer, entronización, trivializar, papamóvil, personalismo,

	Cluster 19
	Words: grasas, orgánicas, disueltas, provocan, eliminar, reaccionan, trehalosa, abuso, químicas, mescalina, etiología, adicción, tóxicas, favorecido, fluorescencia, reaccionaba, irritantes, ingeridas, histamina, secretoras, engrosada, nocivas, inorgánicas, neón, preferencial, psicodélicas, amenacen, excretadas, ósmosis, estupefacientes, adictivas, excretables, reguladoras, corrosivas, radioactivas, depositan, 

	Cluster 21
	Words: matemático, astrónomo, filósofo, búlgaro, procede, antiguo, historiador, hebreo, moderno, significa, pronuncia, letra, latín, alfabeto, ví, griego, clásico, geógrafo, oráculo, albanés, galego, ellinika, hipo, polibio, transliterado, biógrafo, rojiza, reportada, armenoi, heródoto, jenofonte, acadio, helenístico, cuño, obsoleta, piezoelectricidad, ppecho, estrujar, isógona, isos, gaya, ga, metis,

	Cluster 24
	Words: chiíes, asesinado, deja, expulsados, tratados, podría, derrotada, esclavo, destituido, atacadas, martirio, degollado, decapitado, expulsado, derrotado, considerado, condesa, llega, convertidas, normal, empiece, va, llegaron, llegan, ejecutado, sajón, apresado, destruido, dama, reconocido, dirigente, aceptada, derrocado, ciudadano, calificado, grabada,

	Cluster 28
	Words: griega, control, escritura, volga, político, informático, bugzilla, notificación, basado, legal, financiero, cvs, posicional, ficheros, fonológico, vocálico, operativo, paquetes, ebuilds, desarrollado, windows, implementa, archivos, microsoft, funcionamiento, implementación, compatible, integrado, kernel, subsistema, macintosh, virtual, extendido, fluvial, coordenadas, conectados, óbidos,

	Cluster 29
	Words: localidad, valladolid, españa, capital, munster, palermo, cádiz, sevilla, león, zamora, española, badajoz, burgos, almería, ripoll, palencia, comarca, cáceres, toledo, bari, teruel, provincia, irlandesa, segovia, friulano, ontario, federado, zelanda, cuenca, comunidad,

	Cluster 34
	Words: chií, ávaros, piratas, apodera, arrasadas, saqueos, intestinas, pactó, aéreos, continuos, terroristas, derruida, desconociendo, decayó, chechenos, hijackes, siempren, sabotaje, respaldar, autoinmunes, esporádicos, motivan, arboladas, ocurran, exógenos, angina, resistió, beneficiando, magrebíes, ortodoxia, resistieron, hipotéticos, cardíacos, neodarwinista, preservaba, piráticos, corsarios, sufridos, despoblado, kenpo, delirios, claustrofóbica, péctoris, caribes, bloqueos, paroxísticos, febriles, frenaron, repetirían, devastaciones, desorganizados, peyorativamente, alicantina, nucleados, carod, injustos, braquiosaurio, animaban, alakazam, exeggutor,

	Cluster 37
	Words: islam, cristianismo, católica, bangladesh, cristiana, aztecas, romanche, filosofía, edictos, compartidas, mayoritaria, misticismo, islamismo, predominante, politeísta, protestante, hinduismo, budismo, sunita, judía, azteca, emiratos, espiritualidad, islámica, catolicismo, bután, jurisprudencia, sanciona, divinas, profesó, hebraica, legitimado, pagana, precristiana, orain, profesan, mexica, budista, zoroastrismo, laicidad, mónaco, monegascos, monoteísta, credos, malayos, adivinos, feuerbach, evadía, birmania, luterano, turda, unitarianismo, instaurar, conformista, ofensas, justifique, mundhum, anglicanismo, dionisíaca, taoísta, ateos, ismael, tolerado,  mitológicamente, teotihuacana, vedantista, dharma, vaidika, profesadas, conviertan, tonta, avesta, pendare, profesar, palauana,

	Cluster 39
	Words: brasil, chile, perú, manaos, colombia, jíbaros, christchurch, reunion, paraguay, aguirre, suárez, corumba, asunción, uruguay, brasilia, panamá, carioca, copacabana, inauguran, bogotá, venezuela, silva, clausuran, saskatoon, tam, bundesliga, copresidida, subsecretario, incae, prefeito, françois, favelas, atraso, canberra, gritó, peugeot, balão, araxá, inclinó, glorias, majestuoso, avianca, undécima, concieto, jacta, florianópolis, varig, recife, pertenencen, crioulo, crioulos, mato, epecies, legalización, pantanal, deudores, comerciando, afrobrasileña, akan, igbo, bordearon, guyana, zimbawe, biologizada,

**Wrapper sobre un clasificador con los sentidos de WordNet como clase**


Este caso se puede ejecutar con el siguiente comando:
	_python -i supervised-feature-selection-wikicorpus.py wordnet-senses_ (asegurar de que el archivo _resources/tagged.es.tgz_ fue descomprimido y que aparece la parte spanishEtiquetado_10000_15000 del corpus)

La matriz original tenía tamaño (106402,17625). Luego de aplicar la técnica de feature selection quedó reducida un tamaño de (106402, 4595).

En el siguiente listado podemos ver algunas palabras de algunos de los clusters más interesantes:

	Cluster 6
	Words: bizantinos, destruida, conquistada, roma, funda, monasterio, fundó, gobernada, fez, fundada, capturan, asedio, constantinopla, grande, gobernante, córdoba, murcia, abbas, gerona, saqueada, amalfi, conquista, arrasada, incendio, saquea, temporalmente, condado, zaragoza, fundación, destruye, fortaleza, apoderan, zar, cerca, reconquista, histórico, somalia, fundando, reims, antigua, sede, sagrada, saldaña, santa, arrasó, lund, arrasa, praga, nació, kazán, fundaron, afueras, sicilia, tiro, castillo, jerusalén, hangzhou, destruyendo, 


## Feature selection no supervisado

### Corpus

### Vectorización

### Feature selection

### Clustering