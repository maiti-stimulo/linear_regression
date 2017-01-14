# linear_regression
#exercici weeek 3

Exercici regressio lineal comentat, i amb els links que hem utilitzat per solucionar exercici.
Comento una de les linies de codi que més m'ha costat d'entrendre per  tenir-ho documentat

train = np.hstack((np.ones((n,1)), trainX[:, i:i+1]))

Com funciona np.hstack?

a = np.array((1,2,3))

b = np.array((2,3,4))

 np.hstack((a,b))

array([1, 2, 3, 2, 3, 4])


En el cas que treballem amb matrius.

 a = np.array([[1],[2],[3]])

b = np.array([[2],[3],[4]])

 np.hstack((a,b))

array([[1, 2],

       [2, 3],
       
       [3, 4]])

- La coma separa les dimensions, com que tenim dues dimensions només posem una coma.
- Podem obviar els ":" de la primera dimensió doncs volem tots els valors sense filtre
- "i" és la columna a la que estem accedint amb el for

Nota: trainX[:, i:i+1] -> es equivalent a trainX[, i:i+1]

Quan "i" és igual a 5, per exemple:
trainX[:, i:i+1] és equivalent a trainX[, 5:6], és a dir, "comença per la columna 5
i acaba a la 6"
