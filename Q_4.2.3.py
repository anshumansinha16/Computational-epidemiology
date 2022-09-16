import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


alpha = float(input("Enter your value for alpha: "))
delta = float(input("Enter your value for delta: "))
beta = float(input("Enter your value for beta: "))
gamma = float(input("Enter your value for gama: "))
epsilon = float(input("Enter your value for epsilon: "))
rho = float(input("Enter your value for rho: "))
phi = float(input("Enter your value for phi: "))
x0 = float(input("Enter your value for x0 : "))
y0 = float(input("Enter your value for y0 : "))
z0 = float(input("Enter your value for z0 : "))
max_time = int(input("Enter your value for max_time: "))

t = np.linspace(0, max_time, max_time)

# The SIR model differential equations.
def deriv(m, t, alpha, delta, beta, gamma, phi, rho, epsilon):
    x, y, z = m
    dxdt = alpha * x - beta * x * y - phi * x * z
    dydt = gamma * x * y  - delta * y
    dzdt = rho * x * z - epsilon * z
    return dxdt, dydt, dzdt

# Initial conditions vector
m0 = x0, y0, z0
# Integrate the SIR equations over the time grid, t.
ret = odeint(deriv, m0, t, args=( alpha, delta, beta, gamma, phi, rho, epsilon))

x, y, z = ret.T
Output = t, x, y, z

#print(Output)

# Plot the data on three separate curves for S(t), I(t) and R(t)
fig = plt.figure(facecolor='w')
ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
ax.plot(t, x, 'b', alpha=0.5, lw=2, label='x')
ax.plot(t, y, 'r', alpha=0.5, lw=2, label='y')
ax.plot(t, z, 'g', alpha=0.5, lw=2, label='z')

ax.set_xlabel('Time')
ax.set_ylabel('x,y,z')
ax.set_ylim(0,1.2)
ax.yaxis.set_tick_params(length=0)
ax.xaxis.set_tick_params(length=0)
ax.grid(b=True, which='major', c='w', lw=2, ls='-')
legend = ax.legend()
legend.get_frame().set_alpha(0.5)
for spine in ('top', 'right', 'bottom', 'left'):
    ax.spines[spine].set_visible(False)
plt.show()