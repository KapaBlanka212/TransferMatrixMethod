# код для построения новых спектров
# файл не имеет отношения к другим, но тоже является частью моего диплома
# f = (fi + fi*) / 2 ; f = (fi* + fi+1) / 2;
# t = sqrt(Ti*Ti*); t = sqrt(Ti* * Ti+1)
# t(f) = [(1 - r)^2 * exp(-a * d)]/[1 - r^2*exp(-2*a*d)]
# R(f) = r * (1 + t(f)exp(-ad))
# ТЗ: построить R(f) + t(f), R(f), t(f)
import numpy as np
from scipy import optimize
from math import exp
# CONSTANTS
r = 0.188
d = 3  # см
transmittance_max = np.array([0.98179,0.89973,0.82452,
                 0.7925,0.75701,0.73451,
                 0.69791,0.65811,0.63198,
                 0.63358,0.59938,0.57876])
f_max = np.array([2.09,3.93,5.85,
                 7.83,9.8,11.76,
                 13.71,15.68,17.7,
                 19.63,21.58,23.54]) # ГГц
transmittance_min = np.array([0.64789,0.5902,0.56014,
                 0.5206,0.49568,0.47054,
                 0.43762,0.414,0.40616,
                 0.38098,0.35727])
f_min = np.array([3.07,4.93,6.73,
                 8.83,10.79,12.75,
                 14.73,16.69,18.67,
                 20.64,22.57]) # ГГц
# СОЗДАЁМ ФУКЦИИ, КОТОРЫЕ БУДУТ УЧАВСТВОВАТЬ В РАСЧЁТАХ


def f(f1, f2): # frequency in GGh
    f = (f1 + f2) / 2
    return f


def t(x1, x2):
    t = np.sqrt(x1 * x2)
    return t


def eqn(x):
    eqn1 = t*(1-r**2*exp(-2*x*d)) - (1 - r )**2*exp(-x*d)
    return eqn1


# СОЗДАДИМ ПУСТНЫЕ МАССИВЫ, КОТОРЫЕ МЫ ЗАПОЛНИМ В ДАЛЬНЕЙШЕМ
anser = []
anser1 = []
t2 = []
f2 = []
f1 = []
t1 = []

for i in range(0, 11):
    v = f(f_min[i], f_max[i+1])
    f2.append(v)
    t_ = t(transmittance_min[i], transmittance_max[i+1])
    t2.append(t_)
    v1 = f(f_min[i], f_max[i])
    f1.append(v1)
    t_1 = t(transmittance_min[i], transmittance_max[i])
    t1.append(t_1)

for i in range(0, 11):
    t = t1[i]
    sol = optimize.newton(eqn, 0.5)
    anser.append(sol)


for i in range(0,11):
    t = t2[i]
    sol = optimize.newton(eqn, 0.5)
    anser1.append(sol)



reflectance = r * (1 + t1 * np.exp(-np.array(anser) * d))
reflectance1 = r * (1 + t2 * np.exp(-np.array(anser1) * d))

np.savetxt('K8_ANS//T_max', transmittance_max)
np.savetxt('K8_ANS//T_min', transmittance_min)
np.savetxt('K8_ANS//f_min', f_min)
np.savetxt('K8_ANS//f_max', f_max)
np.savetxt('K8_ANS//alfa', np.hstack([np.array(anser), np.array(anser1)]))
np.savetxt('K8_ANS//R', np.hstack([reflectance, reflectance1]))
np.savetxt('K8_ANS//t+R', np.hstack([t1, t2])+np.hstack([reflectance, reflectance1]))
np.savetxt('K8_ANS//f', np.hstack([f1, f2]))
np.savetxt('K8_ANS//t', np.hstack([t1, t2]))