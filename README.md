# textmining-featureselection

Práctico de Feature selection para el curso [Text Mining](https://sites.google.com/view/mdt2017)

## objetivo

Encontrar espacios de menor dimensionalidad que mejoren una tarea de PLN, y después aplicar esos espacios una tarea de clustering. 

## detalles técnicos

Se utilizaron dos corpus: 
* El corpus _resources/tagged.es.tgz_, el [Wikicorpus](http://www.cs.upc.edu/~nlp/wikicorpus/) del español taggeado con POS (Part of Spech) y con los sentidos de Wordnet (para feature selection supervisado)
* El corpus _resources/LaVanguardia.txt.gz_, una recopilación de noticias del diario La Vanguardia (para feature selection no supervisado)

Se utilizaron las siguientes herramientas:
* [nltk](http://www.nltk.org/)
* [scikit-learn](http://scikit-learn.org/stable/)
* [gensim](https://radimrehurek.com/gensim/index.html)

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
se realizó la tarea de clustering sobre el espacio reducido utilizando el algoritmo [K-means](https://en.wikipedia.org/wiki/K-means_clustering) y generando 40 clusters.

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

La matriz original tenía tamaño (37834, 4569). Luego de aplicar la técnica de feature selection quedó reducida un tamaño de (37834, 1439). 

Notar que en este caso se consideraron solo 37834 palabras por razones de memoria. 

En el siguiente listado podemos ver algunas palabras de algunos de los clusters más interesantes:

	Cluster 1
	Words: señor, vizconde, bearn, vizcaya, mercaderes, montmartre, vincula, nacería, celuloide, radicados, residiendo, fahrenheit, imposibles, bonsáis, chapuza, derechas, hankey, inducido, hat, títere, sígueme, baxajaun, ubú, moscas, lachambre,

	Cluster 6
	Words: capturan, king, st, world, war, the, school, hero, querido, mafioso, wonders, edmund, langley, taza, camerunes, affair, npn, by, art, end, american, way, life, link, condecorado, system, encyclopaedia, mathematics, viral, tinieblas, heart, darkness, hearts, time, or, recording, comparative, analysis, deal, get, kitchen, shaping, encargo, america, alcira, columbus, gold, amgot, soldier,

	Cluster 7
	Words: sacro, literatura, llamo, galerías, museo, galería, estudió, abstractas, recuerdan, estatuario, exhiben, primitivo, encargando, franz, compartían, degenerado, dell, contemporanea, mondi, possibili, dramático, empezase, importación, rupestre, inspiradas, resalta, mecenas, educacional, colegiata, lrra, albers, essen, levantino, genuino, almanseño, empezo, denonidado, benidictino, valdehuesa, castellanoleonés, contemplan, epipaleolíticas, esquemático,

	Cluster 12
	Words: última, escalada, destruida, roma, funda, capitular, fundada, capital, gobernante, murcia, conquistar, palermo, amalfi, conquista, sevilla, atacan, fundación, destruye, encomienda, cerca, vikinga, york, reconquista, badajoz, une, burgos, ataca, título, reims, antigua, sede, sagrada, almería, inglesa, arrasó, lund, nació, demografía, kazán, cristiana, allí, fundaron, afueras, llegan,

	Cluster 27
	Words: muere, intenta, niños, accidente, temporada, morir, cae, brazos, piano, cruzado, golpea, cuerno, pez, aprovecha, llena, cenizas, robo, explota, intestinos, pisoteado, convenció, mata, secuestrado, heredar, muñeca, oso, pega, curiosamente, atrapado, secuestra, juguete, cartman, kyle, apropia, token, kenny, butters, unieran, dolares, salgan, hemorroides, come, defeca, excremento,

	Cluster 30
	Words: visigodo, españa, deponer, abd, bizantinos, resultando, arzobispo, italia, bizantinas, abren, muhammad, bera, caudillo, francia, zaragoza, entronizado, simancas, proclama, cardeña, monjes, martirizados, códice, independencia, cataluña, arrasa, pobladas, libra, resto, portugal, celebrada, toros, taurina, ejercicio, comarca, montblanc, toledo, afincado, estamento, valencia, 


	Cluster 35
	Words: español, jurista, poeta, chino, obispo, teólogo, médico, filósofo, poetisa, compositora, m, tang, belasco, escritor, religioso, origen, italiana, años, francés, zar, española, alemán, político, japonés, explorador, germánico, ruso, compositor, escritora, danesa, historiador, músico, italiano, irlandés, francesa, serbio, winchester, lobo, informático, argentino, argentina,

	Cluster 37
	Words: islámico, alí, musulmán, europa, árabe, descubren, grande, enorme, occidental, introduce, erupción, volcánica, otorgado, permitida, sorprende, inteligible, dispersión, moderno, inflexión, cambiado, agudo, perfecto, material, sitios, trabajadores, entero, vendedores, ambientadas, acogida, interpretaciones, mamá, divertido, amazonia, diversidad, comercios, acuarios, selváticas,


## Feature selection no supervisado

### Corpus

Se utilizó el corpus _resources/LaVanguardia.txt.gz_, una recopilación de noticias del diario La Vanguardia.

### vectorización 

Para normalizar las palabras se dividió el texto en sentencias y para cada sentencia, se creó una lista de tokens utilizando nltk. Luego, para cada lista de tokens
* todos los tokens fueron expresados en lowercase,
* se eliminaron los tokens que tenian caracteres no alfabéticos, 
* se eliminaron las _stopwords_ del lenguaje español (palabras muy frecuentes en el lenguaje que aportan poco valor) definidas en nltk,
* y finalmente se utilizó un proceso de lematización de cada palabra (determinar el lemma de una palabra dada).

Para vectorizar las palabras se utilizaron word embeddings. Los vectores de palabras se generaron a partir de una implementación de [word2vec](https://en.wikipedia.org/wiki/Word2vec), que aprende vectores para representar las palabras utilizando redes neuronales. 

### Feature selection

Se utilizaron dos ténicas de feature selection no supervisado: 

**[PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html#sklearn.decomposition.PCA)** (Principal Component Analysis) que permite reducción de dimensionalidad utilizando [Singular Value Decomposition](https://en.wikipedia.org/wiki/Singular-value_decomposition). Parámetros:

* _n_components_: 100 (número de dimensiones del vector)

**Word embeddings** con el modelo [Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html) de gensim con los siguientes parámetros:
* _size_: 100 (número de dimensiones del vector)
* _window_: 5 (distancia máxima entre la palabra a vectorizar y las palabras a su alrededor)
* _min_count_: 5 (las palabras con menos ocurrencias de este valor son ignoradas)

### Clustering

Se utilizó el algoritmo [K-means](https://en.wikipedia.org/wiki/K-means_clustering) para generar 40 clusters. Para usar PCA se puede ejecutar el siguiente comando:
	_python -i unsupervised-feature-selection.py pca_ (asegurar de que el archivo _resources/LaVanguardia.txt.gz_ fue descomprimido y que aparece el corpus LaVanguardia.txt)

Mientras que el caso de word embeddings puede ejecutar con el siguiente comando:
	_python -i unsupervised-feature-selection.py embeddings_ (asegurar de que el archivo _resources/LaVanguardia.txt.gz_ fue descomprimido y que aparece el corpus LaVanguardia.txt)


En el siguiente listado podemos ver algunas palabras de algunos de los clusters más interesantes generados utilizando PCA:

	Cluster 4
	Words: anual, precios, cantidad, equivalente, aplica, automática, sube, indica, pagarán, litros, venta, cubrir, agua, suelo, previsión, pasaje, techo, pérdidas, ciento, mil, abonados, distancia, vender, creciendo, bajada, barato, menor, pescado, aumento, mortalidad, registra, niveles, elevados, datos, promedio, cifra, superior, alcanza, tarjeta, bajará, kilo, habitante, envases,
	
En el siguiente listado podemos ver algunas palabras de algunos de los clusters más interesantes generados utilizando word embeddings:

	Cluster 4
	Words: anual, precios, cantidad, equivalente, aplica, automática, sube, indica, pagarán, litros, venta, cubrir, agua, suelo, previsión, pasaje, techo, pérdidas, ciento, mil, abonados, distancia, vender, creciendo, bajada, barato, menor, pescado, aumento, mortalidad, registra, niveles, elevados, datos, promedio, cifra, superior, alcanza, tarjeta, bajará, kilo, habitante, envases,

	Cluster 10
	Words: ministro, portavoz, piqué, adelantó, aseguró, críticas, nadal, diputado, expresó, recordó, añadió, aprovechó, preguntó, dijo, opinó, respondió, definió, hizo, públicamente, carta, éste, presidente, británico, gabinete, clos, tony, blair, sostiene, reconocido, prensa, miembro, mostró, sorprendido, equipo, popular, votó, socialista, técnico, afirmó, presentara, dimisión, presidido, filas, ex, ruso, contradice, asegura, paeria, sàez, pidió, contestó, socio, alemán, rueda, oficialmente, opina, habló, presidenta, señaló, replicó, vasco, secretario, joseba, ofreció, apuntó, discurso, artículo, defensa, gonzález, vicepresidente

	Cluster 15
	Words: barcelona, calles, paseo, gràcia, baix, llobregat, plaza, cerdà, metro, centro, cuadrado, calle, hospital, adyacente, respectivamente, aeropuerto, sant, garraf, ferial, restaurante, sala, vino, barrio, sants, peatonal, obras, trinitat, patio, junto, renfe, monasterio, carril, subterráneo, ciudad, distrito, túnel, via, augusta, ronda, dalt, sarrià, edificio, bellvitge, hotel, fachada, cuadrados, parque, art, contará, ubicará, construirá, congresos, situada, autovía, avenida, carretera, ferrocarril, 

	Cluster 18
	Words: dar, entrar, actuar, paralizar, haber, ser, permitir, resguardarse, verse, cerrar, poner, decir, considerar, ocasionar, utilizar, ir, llegar, fácilmente, negar, pasar, ponerse, sacar, aceptar, curarse, incluir, hacerlo, cliente, ofrecer, descubrir, abrir, hacer, hallar, acceder, continuar, tener, construirse, convertir, manifestarse, pedir, acudir, dialogar, descartar, darse, carecer, soportar, sobresalir, suceder, decidir, regresar, votar, recurrir, exhibir, cinturones, considerarse, encontrar, acabar,

	Cluster 19
	Words: peajes, congelan, titularidad, revisión, tarifas, empresas, incrementar, sistema, reducción, usuarios, mantenimiento, financiero, prevé, área, metropolitana, mercado, amplia, cuya, interior, mejora, línea, diseño, calidad, uso, producto, usuario, compra, servicio, mejorar, trabajo, mediante, complejo, proyectos, televisión, ifema, información, afecta, cultural, cataluña, realización, potenciar, motor, compañía, espacio, ofrece, gas, movilidad, además, realizado, construcción, creación, histórico,

	Cluster 24
	Words: hacia, gran, mundo, atracción, frente, española, europa, isla, sur, categoría, norte, bloque, pryca, presencia, guerra, rico, posguerra, españa, frontera, francia, daurada, repúblicas, soviéticas, press, embajada, rusa, rusia, internacional, gigante, creó, pueblo, tradicional, atraviesa, siglo, xxi, americano, país, aragón, cantabria, vive, alemania, origen, extranjero, época, ejército, chino, español, fútbol, japón, oriental, occidental, grecia, andorra, mediterráneo, naciones, islámico, arabia, nacionalidad, alemana, oriente, islámica, marroquí, xx, banco, hispano, colonial, denominación, c, exterminio, unido, países, argelia, france,

	Cluster 25
	Words: generalitat, marcha, vigor, gobierno, consejo, ministros, aprobó, decreto, apruebe, nuevo, procedimiento, anunciada, administración, marco, ley, psc, reclamó, consenso, pendiente, caso, general, reforma, parte, mantiene, anuncio, nota, pleno, reunión, fira, institución, dirigentes, debate, voto, informe, respecto, finalmente, mesa, retirar, ciu, pp, erc, plan, proyecto, decisión, oposición, medidas, adoptada, programa, orden, municipal, aprobará, modificación, operación, miembros, ayuntamiento, 

	Cluster 30
	Words: causó, allí, preguntaron, aparece, confesó, recién, cumplidos, dama, arma, mujer, pasión, sentimental, soledad, blanco, raquel, película, bella, quemar, muerte, cabeza, vecino, pequeño, sustituido, cartas, memorias, matrimonio, relato, dejó, abuela, niño, madre, padre, infierno, griterío, emperador, presentaba, hijo, pistola, suceso, llama, pequeña, abandonado, joven, despacho, llorando, resultó, busto, leyenda, llamaba, recuerdo, taxista, aquel, casa, jubilado, falleció, atropellado, atropello, mortal, víctima, encontraba, trabajado, corazón, aleta, niña, hacía, cogió, foto, compañero, acababa, pasó, blanca, lázaro, disco,

	Cluster 32
	Words: interpretan, criticaban, coincidían, usarlo, agresivas, trazados, integrantes, marcados, desórdenes, comporten, votaron, liberales, gobernantes, gobiernan, reproches, enfermería, deterioradas, granjeros, mimar, explicaban, vecinales, críticos, destacaron, planteaban, pensantes, fotoperiodistas, pacifismo, pacifistas, intervienen, oprimidos, contado, surgieron, feroz, mantendrán, avisos, internacionalmente, salvan, disciplinarios, subcomisión, informales, escépticos, supuestas, facetas, traductores, enmarcan, visuales, retiren, parlamentos, huelgas, inhumanos, contundentes, activamente, sabios, magisterio, brigadas, instruyendo, firmantes, atrevidas, étnicos, insaciables, funcionaron, encargados, desinteresadamente, adhesiones, deducido, radicales, consignas, trataron, diabéticos, pediatras, andaluzas, turoperadores, chantajes, prometieron, estudian, opositores,

	Cluster 33
	Words: manuel, carmen, prestigioso, escritora, arturo, calvo, escribió, carlos, cristina, otero, bellas, coronel, nicolás, josé, ignacio, director, richard, alonso, balaguer, nombre, autor, rodríguez, garcía, luis, marqués, dolores, figura, alfonso, rey, juan, casado, vázquez, montalbán, escritor, antonia, mercedes, salisachs, juncadella, lópez, robert, àlex, federico, eduardo, sindicalista, pilar, edil, miguel, antoñanzas, serrano, dirigida, escrito, presidió, ramírez, iglesia, joaquín, patiño, comenta,

	Cluster 37
	Words: ayer, hoy, real, anterior, abril, central, semana, dio, abre, madrid, primera, mañana, llegó, convirtió, puertas, viernes, mediodía, presentación, ciclo, día, mantuvo, abrió, turno, llamada, tarde, vanguardia, estadio, etapa, primero, anteayer, capítulo, abrirá, último, especial, bianual, comenzará, próximo, otoño, próxima, segundo, celebradas, parís, primer, segunda, quinto, puerta, sexto, pasado, noche, miércoles, finalizar, celebra, lunes, exposición, tercero, anoche, jueves, manifestación, acto, noviembre, celebrará, mayo, edición, comienza, enviado, tercer, presenta, lugar, música, ópera, libro, última, teatro, escenario, parada, cuyo, batalla, moscú, asistieron, feria, disputa, comenzó, junio, coincidiendo, italiana, versión, scooter, febrero, certamen, ocasión, danza, puesta, escena, fiesta, domingo, francesa, diciembre, cuarta, sábado, concluirá, celebró, motivo, aniversario, jornada, celebrada, enero, tercera, perdió, llevó, celebraciones, campeonato, deportivo, encuentro, convocado, cita, novedad, celebrado, reunió, martes, cierra, título, clausura, organizado, oro, posteriormente, corte, celebración, puso, pasada, vuelta, culminará, carrera, marzo, gira, inaugura, h, vii, sesión, inauguración, piloto, ceremonia, oficial, honda, londres, consiguió, visita, convocada, ocupa, vi, cumple, conferencia, amsterdam, miss, cerrada, marcó, liceu, término, inició, inauguró, xvi, cena, 