import numpy as np
import math


def buildERMatrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians. Based on the Euler-Rodrigues formula

    credit: unutbu
    """
    axis = np.asarray(axis)
    axis = axis/math.sqrt(np.dot(axis, axis))
    a = math.cos(theta/2.0)
    b, c, d = -axis*math.sin(theta/2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return np.array([[aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
                     [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
                     [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]])

def polarToCartesianVector(radius, polarAngle, azimuthAngle):
    return [radius*math.sin(polarAngle)*math.cos(azimuthAngle),
            radius*math.sin(polarAngle)*math.sin(azimuthAngle),
            radius*math.cos(polarAngle)]

def randomUnitVector():
    phi = np.random.uniform(0,np.pi*2)
    cos_theta = np.random.uniform(-1,1)
    sin_theta = np.sqrt(1-cos_theta*cos_theta) #theta = np.arccos(cos_theta)
    
    x = sin_theta*np.cos(phi) # np.sin(theta)*np.cos(phi)
    y = sin_theta*np.sin(phi) # np.sin(theta)*np.sin(phi)
    z = cos_theta # np.cos(theta)
    return [x,y,z]