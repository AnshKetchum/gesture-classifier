'''
This script was created as a 
'utility class' used to get
the input for the angle-based
classfier.
'''
import numpy as np
import math


'''
Method that returns the Euclidean Distance between two Cartesian points a and b in form (x,y).
dist = sqrt((x1 - x2)^2 + (y1 - y2)^2)
'''
def dist(a, b):
    dx = a[0] - b[0] #0th index is x, 1rst index is y
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)

'''
Uses Law of Cosines to get angle B
Law of Cosines c^2 = a^2 + b^2 - 2abcosC

Returns angles in degrees.
'''
def get_angle(a,b,c):
    ab = dist(a,b)
    ac = dist(a,c)
    bc = dist(b,c)

    #If a,b,c is collinear 
    if ab == 0 or ac == 0 or bc == 0:
        return -1.0

    dist_diff = (ac * ac) - (ab * ab) - (bc * bc)
    dist_diff = (dist_diff) / (-2.0 * ab * bc)

    if dist_diff < -1:
        dist_diff = -1
    elif dist_diff > 1:
        dist_diff = 1

    theta = math.degrees(math.acos(dist_diff))
    return theta

'''
Returns an array of angles, which will be the input for the angle
based network. Input shape will be (23, )

Notes: Openpose conveniently ouputs array b in a format 
such that the joint i is in between joint i - 1 and joint 
i + 1, making it easy to get the angles..
'''
def get_angles(b):
    angles = []

    for i in range(1,24): 
        angles.append(get_angle(b[i - 1], b[i], b[i + 1]))
    return angles