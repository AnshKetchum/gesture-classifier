import numpy as np
import math


def dist(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return math.sqrt(dx * dx + dy * dy)

#Uses Law of Cosines to get angle B
#Law of Cosines c^2 = a^2 + b^2 -2abcosC
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

    print(theta)
    return theta

def get_angles(b):
    angles = []

    for i in range(1,24):
        angles.append(get_angle(b[i - 1], b[i], b[i + 1]))
    return angles