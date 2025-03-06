def max_angle_preserved(angle1, angle2, max_angle):
    a1 = angle1
    a2 = angle2
    if a1 < 0:
        a1 = a1 + 360
    if a2 < 0:
        a2 = a2 + 360
    dif = abs(a2 - a1)
    # if angle1 < -90 and angle2 > 0:
    #    dif = 180 - dif
    if dif > 180:
        dif = 360 - dif
    return dif <= max_angle, dif