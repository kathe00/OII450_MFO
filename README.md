# Moth Flame Optimization
Aplicación de Moth Flame Optimization para resolver instancias de Set Covering Problem. Creado como proyecto para la asignatura Algoritmos Bioinspirados OII450.

### Cambio de parámetros
Todos los parámetros que pueden ser cambiados están definidos en el archivo **main.py**. En la linea 17 se encuentra la instancia que está siendo leída por el algoritmo, para leer otra instancia solo hace falta cambiar el número de instancia SCP definida al final del enlace.

```
instancia = leerInstancia("http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/scp61.txt")
```

Para el correcto funcionamiento de la lectura de la instancia, se recomienda utilizar las instancias definidas en Orlib con el nombre de *scpxx.txt*, tal como *scp41.txt*, *scp51.txt*, etc.

En las lineas 36 a 40 se encuentran los parámetros que utiliza el algoritmo. En la linea 36 se define el número de polillas (agentes buscadores), en las lineas 38 y 39 se definen los límites máximo y mínimo que pueden alcanzar los valores de las soluciones que encuentra el algoritmo. Y finalmente en la linea 40 se encuentra la cantidad máxima de iteraciones.

### Ejecución del programa
Para realizar la prueba del algoritmo solo es necesario ejecutar el archivo **main.py**.
