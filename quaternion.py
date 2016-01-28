"""
Unit quaternion functions for spatial rotation.

Unit quaternions provide a convenient mathematical notation for
representing orientations and rotations of objects in three dimensions.

"""

import math

def conjugate(q):
  """ Return the conjugation of q. """
  return [-q[0], -q[1], -q[2], q[3]]

def multiply(q1, q2):
  """ Multiply quaternions q1 and q2. """
  return [q1[3] * q2[0] + q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1],
          q1[3] * q2[1] - q1[0] * q2[2] + q1[1] * q2[3] + q1[2] * q2[0],
          q1[3] * q2[2] + q1[0] * q2[1] - q1[1] * q2[0] + q1[2] * q2[3],
          q1[3] * q2[3] - q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2]]

def rotate(v, q):
  """ Rotation of a 3D vector using quaternion q. """
  q_tmp = multiply(q, v[:3] + [0.0])
  q_res = multiply(q_tmp, conjugate(q))
  # Return complex part
  return q_res[:-1]

def inverse_rotate(v, q):
  """ Inverse rotation of a 3D vector using quaternion q. """
  return rotate(v, conjugate(q))

def quaternion_from_rotation(v):
  """ Construct quaternion from rotation vector (p, q, r). """
  angle = math.sqrt(sum([x**2 for x in v]))
  if angle < 1e-4: # Avoid division by zero
    q = [x / 2.0 for x in v] + [0.0]
    q[3] = math.sqrt(1.0 - sum([x**2 for x in q[:-1]]))
  else:
    q = [math.sin(angle / 2.0) * (x / angle) for x in v] + \
        [math.cos(angle / 2.0)]
  return q

def quaternion_from_orientation(v):
  """ Obtain quaternion from orientation vector [roll, pitch, yaw]. """
  s = sum([x**2 for x in v])
  qw = math.sqrt(1 - s) # q0^2 + q1^2 + q2^2 + q3^2 = 1
  return v + [qw]
