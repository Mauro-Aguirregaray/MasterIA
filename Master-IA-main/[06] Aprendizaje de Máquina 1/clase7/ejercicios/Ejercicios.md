# Ejercicios para practicar

1. Use el dataset [Iris Species](https://archive.ics.uci.edu/dataset/53/iris). El cual consiste en 50 muestras de flores de tres especies distintas (Iris-setosa, Iris-versicolor y Iris-virginica) con diferentes características (Largo del sépalo, ancho del sépalo, largo del pétalo y ancho del pétalo). En este caso, sabemos a qué clase pertenece cada planta, pero podemos con ello evaluar métodos de agrupación.
   1. Implemente 1 modelo usando el algoritmo de K-means.
      1. Con el método del codo, determine el número de clusters. Cuanto clusters se eligió, es igual a la cantidad de especies de planta? Hay una correlación (por ejemplo, supongamos que quedan 4 clusters, en los cuales 2 corresponden perfectamente a un tipo de planta y las otras dos a la otras dos categorías respectivamente)? 
      2. Separe un 20% de los datos del dataset y guardelos.  
      3. Arme el modelo con el número óptimo de clúster usando el 80% de los datos restantes. Con los cluster obtenidos, determine a que clase de planta pertenece mediante algún método de mayoría. 
      4. Clasifique al 20% de los datos, usando el modelo de agrupamiento y el método de mayoría elegida. ¿Qué resultado obtuvo? (Elija la metrica de evaluación que mejor considere). ¿Qué tan correcto fue armando agrupamientos de datos con respecto a la clase que dan origen a los datos?
      5. Compare el procedimiento que realizó con KNN. ¿Que tán similar es al clasificador en cuanto al procedimiento?
   2. Implemente 1 modelo usando el algoritmo de GMM:
      1. Obtenga el número de óptimo de clusters usando Fuerza de predicción
      2. Separe un 20% de los datos del dataset y guardelos.  
      3. Arme el modelo con el número óptimo de clúster usando el 80% de los datos restantes. Con los cluster obtenidos, determine a que clase de planta. 
      4. Clasifique al 20% de los datos, usando el modelo de agrupamiento y el método de mayoría elegida. ¿Qué resultado obtuvo? (Elija la metrica de evaluación que mejor considere). ¿Qué tan correcto fue armando agrupamientos de datos con respecto a la clase que dan origen a los datos? 
      5. Compare con el resultado de K-means y de KNN. ¿Cómo fue el rendimiento de este modelo? 
2. Use el dataset [USArrest.csv](https://www.kaggle.com/datasets/halimedogan/usarrests) el cual es la medición de crímenes en EEUU por estado dividido en diferentes categorías del tipo de arresto. Realice los siguientes ejercicios:
   1. Obtenga la varianza de cada atributo.
   2. Estandarice los atributos usando [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) para que la varianza esté en el mismo rango (y media cero).
   3. Aplique un análisis de componentes principales a los datos (misma cantidad de componentes que de atributos).
   4. Obtenga una [biplot](https://www.jcchouinard.com/python-pca-biplots-machine-learning/) usando los dos primeros componentes de la PCA. La representación Biplot es una representación gráfica de datos multivariantes en dos o tres dimensiones. Las representaciones de las variables son normalmente vectores y los individuos se representan por puntos. Tenga en cuenta que cada observación es un estado, el gráfico debe reflejar esta información.
   5. Desde el gráfico, observe la primera componente a qué atributos les da más pesos, ¿y la segunda? ¿Qué atributos están más cercas y cuáles están más alejados del resto? Justifique con sus palabras
   6. Ahora interprete a nivel estado, el significado del gráfico.
   7. Obtenga la varianza explicada de los 4 componentes. Explique como se distribuyen la misma entre los 4.