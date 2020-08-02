def T2to3(x,y,image):
    cx = 2044.08
    cy = 1550.39
    fx = 1955.83
    fy = 1955.42
    xaxis = int(y)  ##
    yaxis = int(x)  ##inverse x,y pixel
    zw1 = int(image[xaxis][yaxis])
    while zw1 == 0:
        xaxis += 1
        yaxis += 1
        zw1 = image[xaxis][yaxis]
    xw1 = int((xaxis - cx) * zw1 / fx)
    yw1 = int((yaxis - cy) * zw1 / fy)
    ccord = [xw1, yw1, zw1]
    return ccord