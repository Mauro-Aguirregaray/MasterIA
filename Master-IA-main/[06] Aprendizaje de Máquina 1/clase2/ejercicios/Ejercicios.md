# Ejercicios para prácticar

Sos un data scientist que trabaja para una empresa de publicidad que lanzo una campaña de publicidad en una red social. 
Se registró cada usuario al que se le mostró la publicidad, y se pudo obtener datos de la persona tales como 
`genero`, `edad` y `salario estimado`. Además se registró si el usuario luego compró el producto que la publicidad 
hacia referencia. Los datos están en `Social_Network_Ads.csv`. Se busca poder predecir dado un usuario con datos 
socioeconómicos si va a comprar o no el producto.

1. Realice un estudio de variables y de limpieza de datos. Analice las clases, están balanceadas, o no? Qué clase 
nos parece mas importante de las dos?
2. Separe el dataset en entrenamiento y validación.
3. Elija diferentes modelos de clasificación (al menos uno de regresión logística y uno de KNN). Elija las variables 
de entrada en base al análisis del punto 1.
4. Compárelos con dos o más metrica de evaluación. Cual fue el mejor modelo? Todas las métricas coincidieron o 
métricas diferentes evaluan como mejor a diferentes modelos? Discuta los resultados.
5. Para la regresión logistica, cree una curva ROC para evaluar el modelo para ver la calidad del modelo, sin depender 
del valor umbral. Elija un  valor umbral que considere más optimo y vuelva a clasificar usando ese valor. Como 
cambiaron las métricas usadas en el punto 4 con este valor umbral.
6. Utilizando alguna técnica de busqueda de hiper-parámetros, busque para el clasificador kNN los mejores parámetros. 
Se recomienda usar `n_neighbors`, `weights` y 'p', dejando el parámetro de distancia fijo como `'minkowski'`.
