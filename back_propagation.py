from sympy import *
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from matplotlib.widgets import Button
import ipywidgets as widgets
from lab_utils_backprop import *


#taking an example to compute back propagation
w = 3
a = 3*w + 2
J = a**2
print(f"a = {a}, J = {J}")

#finding dj_da, by finding how J changes as a result of small change in a
a_eplsilon = a + 0.001
J_epsilon = a_eplsilon**2
k = (J_epsilon - J)/0.001
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_da ~= k = {k} ")

sw, sJ, sa = symbols('w,J,a')
sJ = sa**2
sJ.subs([(sa, a)])
dj_da = diff(sJ, sa)

#finding da_dw by computing the change in a observed with small change in w
w_epsilon = w + 0.001
a_epsilon = 2 + 3*w_epsilon
k = (a_epsilon - a)/0.001
print(f"a = {a}, a_epsilon = {a_epsilon}, da_dw ~= k = {k} ")
sa = 2 + 3*sw
da_dw = diff(sa, sw)

#computing dj_dw using the chain rule
dj_dw = da_dw*dj_da

#checking arithmetically as well
w_epsilon = w + 0.001
a_epsilon = 2 + 3*w_epsilon
J_epsilon = a_epsilon**2
k = (J_epsilon - J)/0.001   # difference divided by epsilon
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")
