import numpy as np
import cvxpy as cvx
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation

#vehicle properties
g = -3.7114
mw = 1905
md = 1505

# normal init conditions
r0 = np.array([[5000], [3000], [2000]])
rdot0 = np.array([[-110], [-100], [100]])

#other init conditions
# r0 = np.array([[5000], [100], [5000]])
# rdot0 = np.array([[-50], [-150], [-150]])
# rdot0 = np.array([[0], [0], [0]])

z_init = np.log(mw)
y0 = np.vstack([r0, rdot0, z_init])

Isp = 225
T = 3100
T1 = 0.3*T
T2 = 0.8*T
phi = np.deg2rad(27)
alpha = 1 / (Isp*9.81*np.cos(phi))
nthrust = 6
rho1 = nthrust * T1 * np.cos(phi)
rho2 = nthrust * T2 * np.cos(phi)

dt = 1
Nt = 80
time = np.arange(0, Nt, dt)

#continues time dynamics Ac (7x7), Bc (7x4)
Ac = np.block([[np.zeros((3, 3)), np.eye(3), np.zeros((3, 1))], [np.zeros((4, 7))]])
Bc = np.block([[np.zeros((3, 4))], [np.eye(3), np.zeros((3, 1))], [np.zeros((1, 3)), -alpha]])

#discretize by computing state transition matrix for 0->dt
block_n = Ac.shape[1]+Bc.shape[1]
exp_aug = sp.linalg.expm(dt*np.block([[Ac, Bc], [np.zeros((block_n-Bc.shape[0], block_n))]]))

A = exp_aug[:Ac.shape[0], :Ac.shape[1]]
B = exp_aug[:Ac.shape[0], Ac.shape[1]:]

eigs, eigvecs = np.linalg.eig(A)

#pre calculate all A^k and A^(k-1)B
Ak_list = np.array([np.linalg.matrix_power(A, k) for k in range(0, Nt+1)])
AB_list = np.array([Ak_list[k, ::] @ B for k in range(0,Nt)])

#variable "extraction" matrices
E = np.hstack([np.eye(6), np.zeros((6, 1))])
F = np.hstack([np.zeros((1, 6)), np.array([[1]])])
Eu = np.hstack([np.eye(3), np.zeros((3, 1))])

#SOCP optimization problem formulation for CVXPY
y = cvx.Variable((7, Nt))
p = cvx.Variable((4, Nt-1))

cost = 0
constraints = []
constraints += [y[:, [0]] == y0]

#stay above ground
constraints += [y[0, :] >= 0]

#hard final state constraint
constraints += [y[:6, [-1]] == np.array([[0], [0], [0], [0], [0], [0]])]

#gravity input
g_input = np.vstack([g, np.zeros((3, 1))])

# #wind disturbance
# wind_input = np.vstack(0, wind, np.zeros((3, 1))])

for k in range(Nt-1):
  #dynamics constraints
  constraints += [y[:, [k+1]] == A @ y[:,[k]] + B @ (p[:,[k]] + g_input)]
    
  #convexified thrust constraints
  constraints += [cvx.norm(p[:3, [k]]) <= p[3, k]]
  if k > 0:
    #variables
    z = y[6,k]
    z0 = cvx.log(mw - alpha*rho2*time[k])
    mu1 = rho1*cvx.exp(-z0)
    mu2 = rho2*cvx.exp(-z0)

    #convexified thrust constraint
    constraints += [mu1*(1 - (z-z0) + cvx.power(z-z0, 2)*0.5) <= p[3, k]]
    constraints += [p[3, k] <= mu2*(1-(z-z0))]

    #mass constraints
    constraints += [z0 <= y[6, k]]
    constraints += [y[6, k] <= cvx.log(mw - alpha*rho1*time[k])]

  cost += p[3, k] * dt

# #final state cost (relaxation on hard terminal constraints)
# R = np.diag([10, 10, 10, 100, 10, 10 ])
# cost += cvx.quad_form(y[:6, [-1]], R)

objective = cvx.Minimize(cost)
prob = cvx.Problem(objective, constraints)
opt_val = prob.solve()
solution = y.value
print("solver status: " + prob.status)
print("Final state: ", str(y[:6, -1].value))

############################# plotting #############################

#pos vs time plot
plt.figure(1)
plt.title("pos vs time")
for i in range(3):
  plt.plot(time, y[i, :].value)

traj = y.value
u = p.value

#plot thrust vector
qlen = 100
q = qlen*u[:3, :]
base_x = traj[0, :-1] - q[0, :]
base_y = traj[1, :-1] - q[1, :]
base_z = traj[2, :-1] - q[2, :]

#3d trajectory plot
fig_traj_plot = plt.figure(2, figsize=(8, 8))
fig_traj_plot.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
ax = plt.axes(projection = '3d')
ax.view_init(elev=15, azim=-160)
ax.plot3D(traj[1,:], traj[2,:], traj[0,:])
ax.quiver(base_y, base_z, base_x, q[1,:], q[2,:], q[0,:],
           normalize=False, arrow_length_ratio=0.1, color='red', linewidth=0.5)

#fix aspect ratio of 3d plot
x_lim = ax.get_xlim3d()
y_lim = ax.get_ylim3d()
z_lim = ax.get_zlim3d()

max_lim = max(abs(x_lim[1] - x_lim[0]), 
              abs(y_lim[1] - y_lim[0]), 
              abs(z_lim[1] - z_lim[0]))
x_mid = sum(x_lim)*0.5
y_mid = sum(y_lim)*0.5

ax.set_xlim3d([x_mid-max_lim*0.5, x_mid + max_lim*0.5])
ax.set_ylim3d([y_mid-max_lim*0.5, y_mid + max_lim*0.5])
ax.set_zlim3d([0, max_lim])
ax.plot(ax.get_xlim(), (0, 0), (0, 0), 
        color='black', linestyle='--', linewidth=1)
ax.plot((0, 0), ax.get_ylim(), (0, 0),
        color='black', linestyle='--', linewidth=1)

def shared_traj_plot_properties(ax):
  ax.set_title("fuel optimal trajectory")
  ax.scatter(0, 0, 0, color='green', s=10)
  ax.set_xlabel("y")
  ax.set_ylabel("z")
  ax.set_zlabel("x")

shared_traj_plot_properties(ax)

#control effort plot
plt.figure(3)
plt.title("control effort (thrust/m) vs time")
axis_list = ["x", "y", "z"]
for i in range(p.shape[0]-1):
  plt.plot(time[:-1], p[i, :].value, label="(T/m)_" + axis_list[i])
  plt.legend()

plt.figure(4)
plt.title("mass vs time")
plt.plot(time[:], np.exp(y[6, :].value))

lower_bound = [mw - alpha*rho2*time[k] for k in range(Nt)]
upper_bound = [mw - alpha*rho1*time[k] for k in range(Nt)]
plt.plot(time[:], lower_bound[:])
plt.plot(time[:], upper_bound[:])
plt.legend(["m(t)", "lowerbound", "upperbound"])

############################# animation #############################
fig_anim = plt.figure(5, figsize=(8, 8))


ax_anim = plt.axes(projection = '3d')
ax_anim.view_init(elev=15, azim=-160)
ax_anim.plot3D(traj[1,:], traj[2,:], traj[0,:], linestyle='--', linewidth=0.5, color='black')
shared_traj_plot_properties(ax_anim)
ax_anim.set_xlim(ax.get_xlim())
ax_anim.set_ylim(ax.get_ylim())
ax_anim.set_zlim(ax.get_zlim())

pos_scatter = ax_anim.scatter([traj[1,0]], [traj[2,0]], [traj[0,0]])
quiver = ax_anim.quiver(base_y[0], base_z[0], base_x[0], q[1,0], q[2,0], q[0,0], 
            normalize=False, arrow_length_ratio=0.1, color='red', linewidth=1)

def update(frame):
  pos_scatter._offsets3d = ([traj[1,frame]], [traj[2,frame]], [traj[0,frame]])
  quiver.set_segments([[[base_y[frame], base_z[frame], base_x[frame]], 
                        [traj[1,frame], traj[2,frame], traj[0,frame]]]])

  return pos_scatter, quiver, 

anim_int = 50
animation = FuncAnimation(fig_anim, update, frames= Nt-1, interval= anim_int)
#animation.save('../images/animation.gif', writer='imagemagick', fps=1000/anim_int)

#show plots, wait for input to close
plt.show(block=False)
plt.pause(1)
input()
plt.close()