# Leer instancia SCP a travez de un url

import numpy as np
from urllib.request import urlopen
response = urlopen('http://python.org/')

class leerInstancia():

  filas = 0 # cantidad de filas (restricciones)
  columnas = 0 # cantidad de columnas (variables)
  costos = [] # vector de costos de asignación por columna
  matriz_A = [] # matriz de cobertura

  def __init__(self, ruta:str):

    # Leer
    print(f"Leyendo archivo {ruta} ...")
    
    with urlopen(ruta) as response:
      html_content = response.read()
      encoding = response.headers.get_content_charset('utf-8')
      instancia = html_content.decode(encoding)
      
    print("Instancia leída.")
    
    instancia = instancia.replace('\n',' ') # ajustar separadores
    instancia = instancia.split(' ') # separar valores
    instancia = self.toInteger(instancia) # quitar caracteres vacíos y pasar a nros enteros
    tam = len(instancia)
    print(f"Tam: {tam}")

    # -- Dar formato --
    # leer filas y columnas
    iter = 0 # índice para los datos de la instancia
    
    self.filas = instancia[iter]
    iter+=1
    self.columnas = instancia[iter]
    iter+=1

    # definir tamaño de la matriz y llenar con 0
    self.matriz_A = np.zeros((self.filas, self.columnas),int)

    # leer costos
    for i in range(iter,self.columnas+iter):
      self.costos.append(instancia[i])
      iter+=1


    # generar matriz de covertura
    cant_cov = 0 # cantidad de restricciones que cubre una columna
    cov = [] # restricciones que se cubren por esa columna
    fil_m = 0 # fila de la matriz que se está llenando
    flag = True # true: leer cant, false: leer cov
    cont = 0

    for i in range(iter, len(instancia)):
      if flag:
        # leer la cantidad
        cant_cov = instancia[i]
        cont = 0
        flag = False
      else:
        # leer restriccion
        cov.append(instancia[i])
        cont+=1
        if (cont == cant_cov): # si se leyeron todas
          # modificar entradas de la matriz A
          for col_m in range(self.columnas):
            for j in range(len(cov)):
              if (col_m+1 == cov[j]):
                self.matriz_A[fil_m,col_m] = 1

          fil_m+=1
          cov = []
          flag = True


    # - Lectura de la matriz:
    # Los index de filas van del 0 al 199 y de las columnas del 0 a 999 (en scp41)
    # si me pregunto, por ejemplo, "La restriccion 1 es cubierta por la variable 91?"
    # sería: "la fila 0 tiene un 1 en la columna 90?"
    # entonces, hay que comprobar que matriz_A[0,90] sea = 1
      

  # función para convertir valores en int y eliminar vacíos
  def toInteger(self, inst):
    valores = []
    for i in range(len(inst)):
      if (inst[i] != ''):
        valores.append(int(inst[i]))
    return valores
   
    
    
    