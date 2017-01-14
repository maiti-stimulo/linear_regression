# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import lstsq
# generem una variable x, que carrega totes les dades de l'arxiu housing.data
X = np.loadtxt('./housing.data')
# imprimin la len de les dades
print len(X)
# imprimim x per veure com son aquestes dades
# com que no és important comentem la funcio print x
#print X
# de les dades del housing data, en traiem la última columna
# com que les dades no seran iguales que les de la variable x, l'anomenem y
Y = X[:,-1]
# X[:,-1]= agafem tot x, i li traiem la ultima columna
# imprimim y per veure com son aquestes dades
# com que no és important comentem la funcio print y
#print Y

# definim y-mean, es a dir la mitjana de totes les dades
# Compute the arithmetic mean along the specified axis
#https://docs.scipy.org/doc/numpy/reference/generated/numpy.mean.html
Y_mean = np.mean(Y)
print "la mitjana de les dades es %d" %(Y_mean)
# defimin el MSE
#y_p és només una abreviació de "y_pred", el vector de prediccions de y que ens demana el MSE: https://en.wikipedia.org/wiki/Mean_squared_error
def MSE(y_p, y):

#Sum of array elements over a given axis.
#https://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html
    return np.sum((y-y_p)**2) / len(y)

print MSE(Y_mean, Y)
# partim les dades de x en dos grups, un per train i l'altra per test
n = len(Y)/2

# grup de trainX, sera una matriu amb totes les dades fins a n i traiem la última columna 
trainX = X[:n,:-1]
print"train x"
print len(trainX)
print trainX

# grup de trainY, sera una matriu amb totes les dades de la última columna fins a n 
trainY = Y[:n]
print len(trainY)
print "trainY"
print trainY

# grup de testX, sera una matriu amb totes les dades a partir de n i traiem la última columna 
testX = X[n:,:-1]
# grup de testY, sera una matriu amb totes les dades de la última columna desde n 
testY = Y[n:]
print len(testY)
print "testY"
print testY

# creem tres matrius que contindran les dades de MSEtrain, MSEtest i theta
MSEtrain = []
MSEtest = []
theta = []
# la len del trainX, és el primer element que ocupa la posicio[0], i és una matriu
for i in range(0, len(trainX[0])):

    #Stack arrays in sequence horizontally (column wise)
    #https://docs.scipy.org/doc/numpy/reference/generated/numpy.hstack.html
    #np.ones Return a new array of given shape and type, filled with ones.
    #https://docs.scipy.org/doc/numpy/reference/generated/numpy.ones.html
    #generem 12 matrius, i en casdascuna d'elles la primera columna son 1, i la segona les dades de la primera columna de Y  	  [ 1.       0.00632]

    train = np.hstack((np.ones((n,1)), trainX[:, i:i+1]))
    test  = np.hstack((np.ones((n,1)), testX[:, i:i+1]))
    #((n,1)):: generem una matriu amb n columnes, de 1
    #array([[ 1.],
    #      [ 1.],
    #	    ....

    #linalg = algebra lineal
    #Return the least-squares solution to a linear matrix equation.
    #https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.lstsq.html
    # [0]indica que agafem el primer element, que en aquest cas es una matriu
    # theta ens retorna dotze matrius de una columna i dues files
    theta = np.linalg.lstsq(train, trainY)[0]
    #Dot product of two arrays.
    #Append values to the end of an array.
    #https://docs.scipy.org/doc/numpy/reference/generated/numpy.append.html
    MSEtrain.append(MSE(np.dot(train, theta), trainY))
    MSEtest.append(MSE(np.dot(test, theta), testY))
    #var::Compute the variance along the specified axis.
    #https://docs.scipy.org/doc/numpy/reference/generated/numpy.var.html

    R = 1 - MSEtest[i]/np.var(testY)

    print "Columna %d, R=%3.f" % (i,R)

######
minMSE = []

for i in range(len(MSEtrain)):
    minMSE.append(abs(MSEtrain[i] - MSEtest[i]))
    # argmin::Returns the indices of the minimum values along an axis.
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmin.html
    # min::Element-wise minimum of array elements.
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.minimum.html

print "La columna que generalitza millor es %d amb un valor de %3.f" % (np.argmin(minMSE), np.min(minMSE))

######
#argmax::Return the maximum of an array or maximum along an axis.
#https://docs.scipy.org/doc/numpy/reference/generated/numpy.amax.html
print "El pitjor WSE es la columna %d amb un vlor de %3.f" % (np.argmax(MSEtest), np.max(MSEtest))

#####

print "El millor MSE es la columna %d amb un valor de %3.f" % (np.argmin(MSEtest), np.min(MSEtest))

print "#### Q4 ####"
# generem unes noves variables de train i de test, però traient la pitjor variable de Q3, que era la primera columna
trainXQ4 = trainX[:, 1:]
testXQ4 = testX[:, 1:]
# per comprovar que hem tret la pitjor columna ho plotejem
print "trainXQ4"
print trainXQ4
# creem tres matrius que contindran les dades de MSEtrain, MSEtest i theta, i repetim el mateix procè que a la Q3, però sense la primera columna
MSEtrain = []
MSEtest = []
theta = []

for i in range(0, len(trainXQ4[0])):
    train = np.hstack((np.ones((n,1)), trainXQ4[:, i:i+1]))
    test  = np.hstack((np.ones((n,1)), testXQ4[:, i:i+1]))

    theta = np.linalg.lstsq(train, trainY)[0]

    MSEtrain.append(MSE(np.dot(train, theta), trainY))
    MSEtest.append(MSE(np.dot(test, theta), testY))

    R = 1 - MSEtest[i]/np.var(testY)

    print "Columna %d, R=%3.f" % (i,R)


######
minMSE = []

for i in range(len(MSEtrain)):
    minMSE.append(abs(MSEtrain[i] - MSEtest[i]))

print "La columna que generalitza millor es %d amb un valor de %3.f" % (np.argmin(minMSE), np.min(minMSE))

######

print "El pitjor WSE es la columna %d amb un vlor de %3.f" % (np.argmax(MSEtest), np.max(MSEtest))

#####

print "El millor MSE es la columna %d amb un valor de %3.f" % (np.argmin(MSEtest), np.min(MSEtest))

print "#### Q5 ####"
#Repetim la Q3, però acumulant al train i al test els resultats d'elevar-ho al quadrat, cub i a la quarta.
# el que aconsseguirem es que els valors que ja eren bons fer-los més bons i els dolens fer-los més dolents.
MSEtrain = []
MSEtest = []
theta = []

for i in range(0, len(trainX[0])):
    train = np.hstack((np.ones((n,1)), trainX[:, i:i+1]))
    train = np.hstack((train, trainX[:, i:i+1]**2))
    train = np.hstack((train, trainX[:, i:i+1]**3))
    train = np.hstack((train, trainX[:, i:i+1]**4))

    test  = np.hstack((np.ones((n,1)), testX[:, i:i+1]))
    test = np.hstack((test, testX[:, i:i+1]**2))
    test = np.hstack((test, testX[:, i:i+1]**3))
    test = np.hstack((test, testX[:, i:i+1]**4))

    theta = np.linalg.lstsq(train, trainY)[0]

    MSEtrain.append(MSE(np.dot(train, theta), trainY))
    MSEtest.append(MSE(np.dot(test, theta), testY))

    R = 1 - MSEtest[i]/np.var(testY)

    print "Columna %d, R=%3.f" % (i,R)

######
minMSE = []

for i in range(len(MSEtrain)):
    minMSE.append(abs(MSEtrain[i] - MSEtest[i]))

print "La columna que generalitza millor es %d amb un valor de %3.f" % (np.argmin(minMSE), np.min(minMSE))

######

print "El pitjor WSE es la columna %d amb un vlor de %3.f" % (np.argmax(MSEtest), np.max(MSEtest))

#####

print "El millor MSE es la columna %d amb un valor de %3.f" % (np.argmin(MSEtest), np.min(MSEtest))



