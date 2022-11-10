"""
 Proyecto OII450 - MFO
 Integrantes:
 - Elian Toro
 - Patricio Cataldo
 - Katherine Sepúlveda
"""

from leerInstancia import leerInstancia
from MFO import MothFlame
import time
import numpy as np
import matplotlib.pyplot as plt


# Leer la instancia
instancia = leerInstancia("http://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/scp61.txt")

# valores de prueba
#instancia.filas = 3
#instancia.columnas = 5
#instancia.costos = np.array([1,5,3,2,4])
#instancia.matriz_A = np.array([[0,1,0,0,1],[0,1,1,1,0],[1,0,0,1,1]])

## Mostrar valores de la instancia
print(f"\nFilas: {instancia.filas}")
print(f"Columnas: {instancia.columnas}")
print("\nVector de costos:")
print(
  f"[{instancia.costos[0]} {instancia.costos[1]} {instancia.costos[2]} ... {instancia.costos[instancia.columnas-3]} {instancia.costos[instancia.columnas-2]} {instancia.costos[instancia.columnas-1]}]"
)
print("\nMatriz A:")
print(instancia.matriz_A)

# Definir parámetros
nro_moths = 10
dim = instancia.columnas
ub = 100
lb = -100
max_iter = 500

# Iniciar Algoritmo
print("\nIniciando MFO...\n")
inicio = time.time()
mfo = MothFlame(nro_moths, dim, ub, lb, max_iter)  # pasar parámetros
bFlameScore, bFlamesPos, convergenceCurve = mfo.mfo(instancia)  # iniciar búsqueda
fin = time.time()

# Mostrar resultado
t_ejecucion = fin - inicio
print()
print(f"Mejor Fitness: {bFlameScore}")
#print(f"Mejor Solucion: {bFlamesPos}")

plt.plot(range(len(convergenceCurve)), convergenceCurve)
#plt.show()
