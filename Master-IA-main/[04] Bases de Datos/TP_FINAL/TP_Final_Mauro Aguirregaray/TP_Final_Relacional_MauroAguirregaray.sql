-- Colores más utilizados en los 90
SELECT c.name, 
       COUNT(ip.inventory_id) AS Cantidad_Piezas_Por_Color,
       (SELECT COUNT(ip2.inventory_id)
        FROM sets s2
        JOIN inventories i2 ON i2.set_num = s2.set_num
        JOIN inventory_parts ip2 ON ip2.inventory_id = i2.id
        WHERE s2.year < 2000 AND s2.year > 1989) AS Suma_Total_Cantidad_Piezas
FROM sets s
JOIN inventories i ON i.set_num = s.set_num
JOIN inventory_parts ip ON ip.inventory_id = i.id
JOIN colors c ON ip.color_id = c.id
WHERE s.year < 2000 AND s.year > 1989
GROUP BY c.name
ORDER BY Cantidad_Piezas_Por_Color DESC
LIMIT 10;

-- Colores más utilizados en los 2000
SELECT c.name, 
       COUNT(ip.inventory_id) AS Cantidad_Piezas_Por_Color,
       (SELECT COUNT(ip2.inventory_id)
        FROM sets s2
        JOIN inventories i2 ON i2.set_num = s2.set_num
        JOIN inventory_parts ip2 ON ip2.inventory_id = i2.id
        WHERE s2.year < 2010 AND s2.year > 1999) AS Suma_Total_Cantidad_Piezas
FROM sets s
JOIN inventories i ON i.set_num = s.set_num
JOIN inventory_parts ip ON ip.inventory_id = i.id
JOIN colors c ON ip.color_id = c.id
WHERE s.year < 2010 AND s.year > 1999
GROUP BY c.name
ORDER BY Cantidad_Piezas_Por_Color DESC
LIMIT 10;

-- Colores más utilizados en los 2010
SELECT c.name, 
       COUNT(ip.inventory_id) AS Cantidad_Piezas_Por_Color,
       (SELECT COUNT(ip2.inventory_id)
        FROM sets s2
        JOIN inventories i2 ON i2.set_num = s2.set_num
        JOIN inventory_parts ip2 ON ip2.inventory_id = i2.id
        WHERE s2.year > 2009) AS Suma_Total_Cantidad_Piezas
FROM sets s
JOIN inventories i ON i.set_num = s.set_num
JOIN inventory_parts ip ON ip.inventory_id = i.id
JOIN colors c ON ip.color_id = c.id
WHERE s.year > 2009
GROUP BY c.name
ORDER BY Cantidad_Piezas_Por_Color DESC
LIMIT 10;


-- Cantidad de colores únicos
SELECT DISTINCT c.name, ip.color_id
FROM inventory_parts ip
JOIN colors c ON ip.color_id = c.id
ORDER BY ip.color_id;

-- Cantidad piezas por año
SELECT year, SUM(num_parts) AS Cantidad_Piezas_Promedio_Por_Año
FROM sets
GROUP BY year
ORDER BY year;

-- Promedio de cantidad piezas por set por año
SELECT year, AVG(num_parts) AS Cantidad_Piezas_Promedio_Por_Año
FROM sets
GROUP BY year
ORDER BY year;

SELECT iset.set_num FROM inventory_sets iset
INNER JOIN sets s ON iset.set_num = s.set_num
WHERE iset.set_num = '70904';

-- Temáticas más populares luego de los 2000
SELECT t.name, COUNT(s.set_num) AS Cantidad_de_sets_por_tematica
FROM themes t
JOIN sets s ON s.theme_id = t.id
WHERE s.year >1999
GROUP BY t.name
ORDER BY Cantidad_de_sets_por_tematica DESC;

-- Temáticas más populares antes de los 2000
SELECT t.name, COUNT(s.set_num) AS Cantidad_de_sets_por_tematica
FROM themes t
JOIN sets s ON s.theme_id = t.id
WHERE s.year < 2000
GROUP BY t.name
ORDER BY Cantidad_de_sets_por_tematica DESC;
