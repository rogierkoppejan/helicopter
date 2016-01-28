"""
Helicopter simulator for the generalized hover regime

This implementation of helicopter dynamics is based on:
http://heli.stanford.edu/papers/nips06-aerobatichelicopter.ps.

"""

import math
import random
from quaternion import *


# State indices
U = 0
V = 1
W = 2
X = 3
Y = 4
Z = 5
P = 6
Q = 7
R = 8
ROLL = 9
PTCH = 10
YAW = 11


# Action indices
AILR = 0
ELEV = 1
RUDD = 2
COLL = 3


# Model indices
U_DRAG = 0
V_DRAG = 1
W_DRAG = 3
P_DRAG = 5
Q_DRAG = 7
R_DRAG = 9
W_COLL = 4
P_AILR = 6
Q_ELEV = 8
R_RUDD = 10
SIDE_THRUST = 2


# State limits
LIMITS = tuple([5.0] * 3 + \
               [20.0] * 3 + \
               [4 * math.pi] * 3 + \
               [math.cos(30.0 / 2.0 * math.pi / 180.0)])


def box_mull():
  """ Generate an independent standard normally distributed random number. """
  x1 = 1 - random.random()
  x2 = 1 - random.random()
  return math.sqrt(-2 * math.log(x1)) * math.cos(2 * math.pi * x2)


class Helicopter(object):
  """ Models the behavior of the ghh helicopter simulator. """
  def __init__(self, params, noise_std, dt=0.01, steps=6000):
    assert len(params) == 11
    assert len(noise_std) == 6
    self.params = params
    self.noise_std = noise_std
    self.dt = dt
    self.max_steps = steps

  def _update_noise(self):
    """ Update sensor noise. """
    for i, v in enumerate(self.noise):
      self.noise[i] = 0.8 * v + 0.2 * box_mull() * self.noise_std[i] * 2.0

  def _update_state(self, action):
    """ Update state features. """
    # Saturate all the action features
    action = [min(max(v, -1.0), 1.0) for v in action]
    # Update state
    for i in range(int(0.1 / self.dt)):
      # Update position (x, y, z)
      self.state[X] += self.dt * self.state[U]
      self.state[Y] += self.dt * self.state[V]
      self.state[Z] += self.dt * self.state[W]
      # Compute velocity delta in heli frame
      velocity = inverse_rotate(self.state[U:], self.q)
      delta = [self.params[U_DRAG] * velocity[U] + self.noise[0],
               self.params[V_DRAG] * velocity[V] + \
               self.params[SIDE_THRUST] + self.noise[1],
               self.params[W_DRAG] * velocity[W] + \
               self.params[W_COLL] * action[COLL] + self.noise[2]]
      # Rotate delta vector to world frame
      delta = rotate(delta, self.q)
      # Update velocity (u, v, w)
      self.state[U] += self.dt *  delta[0]
      self.state[V] += self.dt *  delta[1]
      self.state[W] += self.dt * (delta[2] + 9.81)
      # Update orientation (roll, pitch, yaw)
      q = quaternion_from_rotation([self.dt * v for v in self.state[P:]])
      self.q = multiply(self.q, q)
      # Compute angular velocity delta
      delta = [self.params[P_DRAG] * self.state[P] + \
               self.params[P_AILR] * action[AILR] + self.noise[3],
               self.params[Q_DRAG] * self.state[Q] + \
               self.params[Q_ELEV] * action[ELEV] + self.noise[4],
               self.params[R_DRAG] * self.state[R] + \
               self.params[R_RUDD] * action[RUDD] + self.noise[5]]
      # Update angular rates (p, q, r)
      self.state[P] += self.dt * delta[0]
      self.state[Q] += self.dt * delta[1]
      self.state[R] += self.dt * delta[2]

  def _update_status(self):
    """ Update helicopter status, i.e., set terminal = True/False. """
    if any([abs(v) > LIMITS[i] for i, v in enumerate(self.state)]):
      self.terminal = True
    elif abs(self.q[3]) < LIMITS[9]:
      self.terminal = True
    elif self.steps + 1 >= self.max_steps:
      self.terminal = True

  def update(self, action):
    """ Update the helicopter simulator. """
    self._update_noise()
    self._update_state(action)
    self._update_status()
    self.steps += 1
    # Return current state and error
    return self.observation, self.error

  def reset(self):
    """ Reset the simulator to its initial values. """
    self.noise = [0.] * 6
    self.state = [0.0] * 9
    self.q = [0.0, 0.0, 0.0, 1.0]
    self.terminal = False
    self.steps = 0
    # Return current state and error
    return self.observation, self.error

  @property
  def error(self):
    """ Return the error of the current state. """
    if not self.terminal:
      err = sum([v**2 for v in self.state + self.q[:-1]])
    else:
      err = sum([v**2 for v in LIMITS[:9]] + [1.0 - LIMITS[9]**2])
      err *= (self.max_steps - self.steps)
    return err

  @property
  def observation(self):
    """ Return an observation, i.e. current state converted to heli frame. """
    obs = inverse_rotate(self.state[U:], self.q)
    obs.extend(inverse_rotate(self.state[X:], self.q))
    obs.extend(self.state[P:])
    obs.extend(self.q[:-1])
    return obs


class XcellTempest:
  params = (-0.18, -0.43, -0.54, -0.49, -42.15, -12.78, 33.04, -10.12, -33.32, -8.16, 70.54)
  noise_std = (0.1941, 0.2975, 0.6058, 0.1508, 0.2492, 0.0734)
