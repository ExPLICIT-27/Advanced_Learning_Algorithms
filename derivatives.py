from sympy import symbols, diff


#using informal definition of derivatives
J = 3**2
J_epsilon = (3 + 0.001)**2
k = (J_epsilon - J)/0.001
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k:0.6f} ")


#making the k value ergo the derivative more accurate
J = (3)**2
J_epsilon = (3 + 0.000000001)**2
k = (J_epsilon - J)/0.000000001
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")

J, w = symbols('J, w')
J = w**2
print(J)

dj_dw = diff(J, w)
print(dj_dw)
dj_dw.subs([(w, 3)]) #derivative at w = 3



