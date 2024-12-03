import numpy as np
import cvxpy as cvx
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class opt_problem:
  def __init__(self):
    #vehicle and dynamics properties
    self.g = -3.7114
    self.mw = 1905
    self.md = 1505
    self.Isp = 225
    self.T = 3100
    self.T1 = 0.3*self.T
    self.T2 = 0.8*self.T
    self.phi = np.deg2rad(27)
    self.alpha = 1 / (self.Isp*9.81*np.cos(self.phi))
    self.nthrust = 6
    self.rho1 = self.nthrust * self.T1 * np.cos(self.phi)
    self.rho2 = self.nthrust * self.T2 * np.cos(self.phi)
    self.z_init = np.log(self.mw)

    # initial state
    self.r0 = np.array([[5000], [3000], [2000]])
    self.rdot0 = np.array([[-110], [-100], [100]])
    self.y0 = np.vstack([self.r0, self.rdot0, self.z_init])

    self.dt = 1
    self.tmin = (self.md)*np.linalg.norm(self.rdot0)/self.rho2
    self.tmax = (self.mw-self.md)/(self.alpha*self.rho1)

    self.Ntmin = int(self.tmin/self.dt) + 1
    self.Ntmax = int(self.tmax/self.dt)

    self.cost_list = []
    self.Nt_list = []

    #continuouss time dynamics Ac (7x7), Bc (7x4)
    Ac = np.block([[np.zeros((3, 3)), np.eye(3), np.zeros((3, 1))], [np.zeros((4, 7))]])
    Bc = np.block([[np.zeros((3, 4))], [np.eye(3), np.zeros((3, 1))], [np.zeros((1, 3)), -self.alpha]])

    #discretize by computing state transition matrix for 0->dt
    block_n = Ac.shape[1]+Bc.shape[1]
    exp_aug = sp.linalg.expm(self.dt*np.block([[Ac, Bc], [np.zeros((block_n-Bc.shape[0], block_n))]]))

    self.A = exp_aug[:Ac.shape[0], :Ac.shape[1]]
    self.B = exp_aug[:Ac.shape[0], Ac.shape[1]:]

    eigs, eigvecs = np.linalg.eig(self.A)

    #variable "extraction" matrices
    E = np.hstack([np.eye(6), np.zeros((6, 1))])
    F = np.hstack([np.zeros((1, 6)), np.array([[1]])])
    Eu = np.hstack([np.eye(3), np.zeros((3, 1))])

  ############################# cvx problem #############################

  def solve_cvx_problem(self, Nt):
    time_Nt = np.arange(0, Nt*self.dt, self.dt)

    y = cvx.Variable((7, Nt))
    p = cvx.Variable((4, Nt-1))

    cost = 0
    constraints = []
    constraints += [y[:, [0]] == self.y0]

    #stay above ground constraint
    constraints += [y[0, :] >= 0]

    #hard final state constraint
    constraints += [y[:6, [-1]] == np.array([[0], [0], [0], [0], [0], [0]])]

    #gravity input vector
    g_input = np.vstack([self.g, np.zeros((3, 1))])

    for k in range(Nt-1):
      #dynamics constraints
      constraints += [y[:, [k+1]] == self.A @ y[:,[k]] + self.B @ (p[:,[k]] + g_input)]
        
      #relaxed thrust constraints
      constraints += [cvx.norm(p[:3, [k]]) <= p[3, k]]

      z = y[6,k]
      z0 = cvx.log(self.mw - self.alpha*self.rho2*time_Nt[k])
      mu1 = self.rho1*cvx.exp(-z0)
      mu2 = self.rho2*cvx.exp(-z0)

      constraints += [mu1*(1 - (z-z0) + cvx.power(z-z0, 2)*0.5) <= p[3, k]]
      constraints += [p[3, k] <= mu2*(1-(z-z0))]

      constraints += [z0 <= y[6, k]]
      constraints += [y[6, k] <= cvx.log(self.mw - self.alpha*self.rho1*time_Nt[k])]

      cost += p[3, k] * self.dt

    # #final state cost (relaxation on hard terminal constraints)
    # R = np.diag([10, 10, 10, 100, 10, 10 ])
    # cost += cvx.quad_form(y[:6, [-1]], R)

    objective = cvx.Minimize(cost)
    prob = cvx.Problem(objective, constraints)
    prob.solve()
    print("solver status: " + prob.status)

    if prob.status != 'optimal':
      opt_cost = float('inf')
      return opt_cost, None, None
    
    else:
      opt_cost = cost.value
      traj_Nt = y.value
      u_Nt = p.value
      return opt_cost, traj_Nt, u_Nt

  #line search for optimal time of flight using golden section search
  def opt_min_time(self):
    Nt_search = np.arange(self.Ntmin, self.Ntmax, 1)
    g = 2/(1+np.sqrt(5))
    a = 0
    b = len(Nt_search)

    while abs(b - a) > 1:
      c = b - int(np.floor((b - a) * g))
      d = a + int(np.ceil((b - a) * g))
      fc, trajc, uc = self.solve_cvx_problem(Nt_search[c])
      fd, _, _ = self.solve_cvx_problem(Nt_search[d])

      self.cost_list.extend([fc, fd])
      self.Nt_list.extend([Nt_search[c], Nt_search[d]])

      if fc < fd:
          b = d
      else:
          a = c
    
    return Nt_search[c], trajc, uc

opt = opt_problem()
Nt_opt, traj, u = opt.opt_min_time()
time = np.arange(0, Nt_opt*opt.dt, opt.dt)

############################# plotting #############################

#pos vs time plot
plt.figure(1)
plt.title("pos vs time")
labels = []
for i in range(3):
  plt.plot(time, traj[i, :], label='')
plt.legend(['x', 'y', 'z'])
plt.xlabel("time (s)")
plt.ylabel("position (m)")

#plot thrust vector
qlen = 0.05
thrust_vecs = u[:3, :]*np.exp(traj[6, :-1])
q = qlen*thrust_vecs
base_x = traj[0, :-1] - q[0, :]
base_y = traj[1, :-1] - q[1, :]
base_z = traj[2, :-1] - q[2, :]

#3d trajectory plot
fig_traj_plot = plt.figure(2, figsize=(8, 8))
# fig_traj_plot.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.05)
fig_traj_plot.tight_layout()
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

#control effort plots
plt.figure(3)
plt.title("norm(thrust_vector) vs time")
plt.plot(time[:-1], np.linalg.norm(thrust_vecs[:, :], axis=0))
plt.xlabel("time (s)")
plt.ylabel("thrust (N)")

plt.figure(4)
plt.title("thrust vector components vs time")         
for i in range(3):
  plt.plot(time[:-1], thrust_vecs[i, :])
plt.legend(['T_x', 'T_y', 'T_z'])
plt.xlabel("time (s)")
plt.ylabel("thrust (N)")

plt.figure(5)
plt.title("mass vs time")
plt.plot(time[:], np.exp(traj[6, :]))

lower_bound = [opt.mw - opt.alpha*opt.rho2*time[k] for k in range(Nt_opt)]
upper_bound = [opt.mw - opt.alpha*opt.rho1*time[k] for k in range(Nt_opt)]
plt.plot(time[:], lower_bound[:])
plt.plot(time[:], upper_bound[:])
plt.legend(["m(t)", "lowerbound", "upperbound"])
plt.xlabel("time (s)")
plt.ylabel("mass (kg)")

plt.figure(6)
plt.title("cost vs time of flight (Nt timesteps of 1 sec)")
plt.scatter(opt.Nt_list, opt.cost_list)
plt.xlabel("Nt")
plt.ylabel("cost (m/s)")

############################# animation #############################

fig_anim = plt.figure(7, figsize=(8, 8))
fig_anim.tight_layout()
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
animation = FuncAnimation(fig_anim, update, frames= Nt_opt-1, interval= anim_int)

fig_names = ['position', 'trajectory', 'throttle', 'thrusts', 'mass', 'cost_tof']

for i in range(1, 7):
  plt.figure(i).savefig("../images/" + fig_names[i-1] +  ".png", dpi=300)

animation.save('../images/animation.gif', writer='pillow', fps=1000/anim_int)

#show plots, wait for input to close
plt.show(block=False)
plt.pause(1)
input()
plt.close()