# Implementación del algoritmo Moth Flame Optimization

# Con ayuda del código en  https://github.com/IbraDje/MFO-Python/blob/master/MFO.ipynb

import numpy as np
import random


class MothFlame():
  '''
    Parameters :
    - nsa : Number of Search Agents
    - dim : Dimension of Search Space
    - ub : Upper Bound
    - lb : Lower Bound
    - max_iter : Number of Iterations
    Returns :
    - bFlameScore : Best Flame Score
    - bin_bFlamePos : Best Flame Position
    - ConvergenceCurve : Evolution of the best Flame Score on every iteration
    '''

  def __init__(self, nsa, dim, ub, lb, max_iter):
    # parámetros
    self.nsa = nsa  # nro moths
    self.max_iter = max_iter  # máximo de iteraciones
    self.ub = ub  # valores máximos de cada variable
    self.lb = lb  # valores mínimos de cada variable
    self.dim = dim  # dimensión (nro variables)
    self.fobj = 0  # función objetivo

# --- Solución Inicial ---

  def solucionInicial(self, instancia):
    # Genera una solución inicial con valores random para cada variable, dentro de los límites
    mothPos = np.random.uniform(low=self.lb, high=self.ub, size=(self.nsa, self.dim))
    return mothPos

# --- Funcion Objetivo ---

  def funcionObjetivo(self, M, costos):
    # se calcula el fitness con el producto punto entre cada fila (solución)
    # y el vector de los costos de asignación
    Mfit = np.zeros(self.nsa)
    for i in range(self.nsa):
      Mfit[i] = np.dot(M[i,:],costos)

    return Mfit

# --- Función de Binarización ---

  def binarizar(self, M, fil, cols, best):
    bin_M = np.empty_like(M)
    transf = 0

    for i in range(fil):
      for j in range(cols):
        # Paso 1: Función de transferencia

        #transf = 1 / (1 + np.exp(-2 * M[i, j]))  # S-shape 1
        transf = 1 / (1 + np.exp(M[i, j]))      # S-shape 2
        #transf = 1 / (1 + np.exp(M[i, j])/2)    # S-shape 3
        #transf = 1 / (1 + np.exp(M[i, j])/3)    # S-shape 4

        # Paso 2: Binarización 

        #bin_M[i, j] = self.binStandard(transf)                 # Standard
        #bin_M[i, j] = self.binComplement(transf, bin_M[i, j])  # Complement
        bin_M[i, j] = self.binElitist(transf, best[j])         # Elitist

        
    return bin_M

  def binStandard(self, transf):
    if (random.random() >= transf):
      return 1
    else:
      return 0
    
  def binComplement(self, transf, val):
    if (random.random() >= transf):
      if (val == 1): return 0
      else: return 1
    else:
      return 0

  def binElitist(self, transf, best):
    if (random.random() >= transf):
      return best
    else:
      return 0


# --- Comprobación de Factibilidad ---

  def solFactible(self, sol, A):
    # con el producto punto entre la matriz de cobertura y la solución
    # se obtiene un vector que indica la correcta cobertura de cada restricción
    aux = np.dot(A, sol)
    # si en este vector hay un 0, la solución es infactible
    if (np.prod(aux) == 0): return False, aux

    return True, aux  # factible

# reparación

  def reparar(self, solution, inst):
    sol =  np.reshape(solution, (inst.columnas,))
    set = inst.matriz_A
    
    flag, aux = self.solFactible(sol, inst.matriz_A)

    while not flag:                                             # Mientras la solucion no sea factible
        nz = np.argwhere(aux == 0)                              # Obtengo las restricciones no cubiertas
        id_nz = np.random.choice(nz[0])                         # Selecciono una restricción no cubierta aleatoriamente
        idxRestriccion = np.argwhere((set[id_nz,:]) > 0)        # Obtengo la lista de subsets que cubren la zona seleccionada
        cost = np.take(inst.costos, idxRestriccion)
        a = np.argmin(cost)                                     # Obtengo el/los subset que tiene/n el costo mas bajo 
        idxMenorPeso = idxRestriccion[a]
        sol[np.random.choice(idxMenorPeso)] = 1                 # Asigno 1 a ese subset
        flag, aux = self.solFactible(sol, inst.matriz_A)        # Verifico si la solucion actualizada es factible

    return sol

# --- Función Principal ---

  def mfo(self, instancia):

    # curva de convergencia
    convergenceCurve = np.zeros(shape=(self.max_iter))

    # --- Algoritmo MFO

    # inicializar posiciones
    mothPos = self.solucionInicial(instancia)
    bFlameScore = 100000
    bin_bFlamesPos = np.ones(shape=(instancia.columnas))

    for iteration in range(self.max_iter):  # Main loop
      # Número de llamas, Eq. en fig. 3.12 del paper
      flameNo = int(np.ceil(self.nsa - (iteration + 1) * ((self.nsa - 1) / self.max_iter)))

      # Asegurar que los valores de las polillas esten dentro de los márgenes
      mothPos = np.clip(mothPos, self.lb, self.ub)

      # Binarizar
      binMoth = self.binarizar(mothPos, self.nsa, self.dim, bin_bFlamesPos)

      # Verificar factibilidad 
      #moth_rep = []
      for m in range(self.nsa):  # recorrer moths
        if (self.solFactible(binMoth[m,:], instancia.matriz_A)[0] == False):
          #moth_rep.append(m)
          binMoth[m,:] = self.reparar(binMoth[m,:], instancia)
      
      #print(f"Moths Reparadas: {moth_rep}")

      # Calcular función objetivo
      mothFit = self.funcionObjetivo(binMoth, instancia.costos)

      if iteration == 0:
        # Ordenar la primera generación de polillas
        order = mothFit.argsort(axis=0)
        mothFit = mothFit[order]
        mothPos = mothPos[order, :]
        binMoth = binMoth[order, :]

        # Actualizar las llamas
        binFlames = np.copy(binMoth)
        bFlames = np.copy(mothPos)
        bFlamesFit = np.copy(mothFit)

      else:
        # Ordenar las polillas
        doublePop = np.vstack((bFlames, mothPos))
        binDoublePop = np.vstack((binFlames, binMoth))
        doubleFit = np.hstack((bFlamesFit, mothFit))

        order = doubleFit.argsort(axis=0)
        doubleFit = doubleFit[order]
        doublePop = doublePop[order, :]
        binDoublePop = binDoublePop[order, :]

        # Actualizar las llamas
        bFlames = doublePop[:self.nsa, :]
        binFlames = binDoublePop[:self.nsa, :]
        bFlamesFit = doubleFit[:self.nsa]

      # Actualizar mejor fitnes encontrado
      bFlameScore = bFlamesFit[0]
      bFlamesPos = bFlames[0, :]
      bin_bFlamesPos = binFlames[0, :]

      print(f"Iter: {iteration+1} | Fitness: {bFlameScore}")

      # r decrece linealmente de -1 a -2 para calcular t en Eq. de fig. (3.10)
      r = -1 + (iteration + 1) * ((-1) / self.max_iter)

      # distancia D entre llamas y polillas
      distanceToFlames = np.abs(bFlames - mothPos)

      b = 1
      t = (r - 1) * np.random.rand(self.nsa, self.dim) + 1
      ''' Actualiza la posición de las polillas en relación a sus llamas correspondientes.
      Si el nro de la polilla es mayor al nro total de llamas, entonces se
      actualiza en relación a la última llama. '''
      temp1 = bFlames[:flameNo, :]
      temp2 = bFlames[flameNo - 1, :] * np.ones(shape=(self.nsa - flameNo, self.dim))
      temp2 = np.vstack((temp1, temp2))
      mothPos = distanceToFlames * np.exp(b * t) * np.cos(t * 2 * np.pi) + temp2

      convergenceCurve[iteration] = bFlameScore

    return bFlameScore, bin_bFlamesPos, convergenceCurve
