# %%
from typing import Iterable
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle,starmap
import pandas as pd
# %%
Datos_D = pd.read_csv('perme.csv')
Registros = pd.read_csv('Registros_prueba.csv')

X = Datos_D['poro'].values
y = Datos_D['perme'].values
y = y.reshape(-1,1)
# %%
class Descenso:
    def __init__(self,X:np.ndarray,y:np.ndarray,iteraciones:int,lr:int) -> None:
        self.iteraciones = iteraciones
        self.lr = lr
        self.X = X
        self.y = y
    def sol(self):
        norma = self.lr/len(self.X)
        self.teta = np.ones((self.X.shape[1],1))
        y_pred = np.ones_like(self.y)
        for _ in np.arange(0,self.iteraciones+1):
            self.teta -= norma * self.X.T@(y_pred - self.y)
            if self.teta.ndim < 2:
                self.teta.reshape(-1,1)
            else:
                pass
            y_pred = self.X@self.teta
        return self.teta
# %%
class PHI:
    def __init__(self,X) -> None:
        self.X = X

    @staticmethod
    def checador(matriz):
        """
        Se encarga de ver que tus datos de entrada contengan el termino independiente
        si no lo tiene lo añadira.
        matriz: array que contenga tus datos de entrada
        >>> X = PHI.checador(X) 
        """
        if matriz.ndim < 2:
            matriz = matriz.reshape(-1,1)
            matriz = np.hstack((np.ones_like(matriz),matriz))
        elif (matriz.ndim == 2 and any(matriz[:,0]!=1)):
            matriz = np.hstack((np.ones_like(matriz[:,0:1]),matriz))
        return matriz

    def phi(self,kernel:str,**kargs) -> np.ndarray:
        """
        Puedes pasarle argumentos de acuerdo a tu necesidad de phi
        Por ejemplo, si necesitas una regresion polinómica de grado 4 usa:
        >>> phi(kernel='poli',n=4)
        Para una regresion con funciones base radiales desde -4 hasta 4 podrías usar:
        >>> phi(kernel='rbf',mu=4,sig=4)
        """
        filas,columnas = self.X[:,1:].shape
        if kernel == 'poli':
            phi_matriz = np.repeat(self.X[:,1:],kargs['n'])
            phi_matriz = phi_matriz.reshape(filas,columnas*kargs['n'])
            rango = np.arange(1,kargs['n']+1)
            for f in np.arange(0,filas):
                phi_matriz[f,:] = [i for i in starmap(pow,zip(phi_matriz[f,:],cycle(rango)))]
            return np.hstack((self.X[:,0:1],phi_matriz))
        elif kernel == 'rbf':
            gauss = lambda x,mu,sig=kargs['sig']: np.exp(-(1/2*sig**2)*(x-mu)**2)
            rango = np.linspace(-1,1,kargs['mu'])
            phi_matriz = np.repeat(self.X[:,1:],len(rango))
            phi_matriz = phi_matriz.reshape(filas,columnas*len(rango))
            for f in np.arange(0,filas):
                phi_matriz[f,:] = [i for i in starmap(gauss,zip(phi_matriz[f,:],cycle(rango)))]
            return np.hstack((self.X[:,0:1],phi_matriz))
    @staticmethod
    def Inversion(Registro:np.ndarray,rangos:dict) -> np.ndarray:
        """
        Crea una inversion del registro que le pases
        :param Registro: Array Registro a invertir
        :param rangos: dict Rangos max-min
        >>> rangos = {'g1':50,'g2':100,'p1':0.01,'p2':1000}
        >>> PHI.Inversion(GR,permeabilidad,rangos)
        """
        pendiente = (np.log10(rangos['p2'])-np.log10(rangos['p1'])) / (rangos['g1']- rangos['g2'])
        intercepto = np.log10(rangos['p2']) - rangos['g1']*pendiente
        print(f"Pendiente: {pendiente} | Intercepto: {intercepto}")
        return 10**(intercepto + Registro*pendiente)

# %%
class RedNeuronal:
    """
    Red neuronal para regresion:
    :param X: Matriz con los datos de entrada (no debe tener el termino independiente)
    :param y: Vector con los datos utilizados en el entrenamiento
    :param neuronas: Iterable con el número de neuronas de entrada y ocultas
    >>> RedNeuronal(X,y,{entrada=4,oculta=5})
    """
    def __init__(self,X:np.ndarray,y:np.ndarray,neuronas:dict) -> None:
        self.X = X
        self.y = y
        self.neuronas = neuronas
        self.neuronas['Teta1'] = np.random.normal(size=(self.neuronas['oculta'],self.neuronas['entrada']))
        self.neuronas['bias1'] = np.random.normal(size=(self.neuronas['oculta'],1))
        self.neuronas['Teta2'] = np.random.normal(size=(1,self.neuronas['oculta']))
        self.neuronas['bias2'] = np.random.normal(size=(1))

    def RELU(self,Z):
        return np.maximum(Z, 0)
    def RELU_der(self,Z):
        return Z>0

    def forward(self):
        #Capa entrada
        try:
            self.neuronas['Z1'] = self.neuronas['Teta1']@self.X.T 
        except ValueError as e:
            print(f'Las dimensiones no son compatibles {e.args}')
        self.neuronas['a2'] = self.RELU(self.neuronas['Z1']) + self.neuronas['bias1']

        #Capa de salida
        try:
            self.neuronas['Z2'] = self.neuronas['Teta2']@self.neuronas['a2'] 
        except ValueError as e:
            print(f'Las dimensiones no son compatibles {e.args}')
        self.neuronas['a3'] = self.neuronas['Z2'] + self.neuronas['bias2']
        self.neuronas['a3'].shape = self.y.shape
        return self

    def backprogation(self):
        self.back = {}
        norma = (1/len(self.X))
        #Error de la capa de salida
        self.back['dz3'] = self.neuronas['a3'] - self.y
        try:
            self.back['dteta2'] = self.neuronas['a2']@self.back['dz3'] * norma
        except ValueError:
            print(f'Las dimensiones no son compatibles para dteta2')
        self.back['dbias2'] = np.sum(self.back['dz3']) * norma

        #Error de la capa oculta
        try:
            self.back['dz2'] = (self.back['dz3']@self.neuronas['Teta2']).T * self.RELU_der(self.neuronas['a2'])
        except ValueError:
            print(f'Las dimensiones no son compatibles para dz2')
        try:
            self.back['dteta1'] = self.back['dz2']@self.X * norma
        except ValueError:
                print(f'Las dimensiones no son compatibles para dteta1')
        self.back['dbias1'] = np.sum(self.back['dz2'],axis=1) * norma

        self.back['dteta2'].shape = self.neuronas['Teta2'].shape
        self.back['dteta1'].shape = self.neuronas['Teta1'].shape
        self.back['dbias1'].shape = self.neuronas['bias1'].shape

        return self
    
    def actualizar(self,lr):
        self.neuronas['Teta1'] -= lr*self.back['dteta1']
        self.neuronas['bias1'] -= lr*self.back['dbias1']
        self.neuronas['Teta2'] -= lr*self.back['dteta2']
        self.neuronas['bias2'] -= lr*self.back['dbias2']
# %%
def regul(entero,muestras,permes,default):
    if entero.ndim < 2:
        entero = entero.reshape(-1,1)
    else:
        pass
    entero = np.hstack((entero,np.full_like(entero,default)))
    for muestra,perme in zip(np.unique(muestras),permes):
        idx = np.argmin(abs(entero[:,0] - muestra))
        entero[idx,1] = perme
    return entero