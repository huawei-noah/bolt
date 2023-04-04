#1/usr/bin/python

def read_shape(shape_path):
    shapes = {}
    f = open(shape_path, "r")
    lines = f.readlines();
    for line in lines:
        line = line.strip()
        if (line == ""):
            continue;
        array = line.split(" ")
        if (len(array) == 1):
            print("[ERROR] invalid shape representation: %s" % (line))
            exit(1)
        name = array[0]
        shape = []
        for i in range(1, len(array)):
            shape.append(int(array[i]))
        shapes[name] = shape
    return shapes

def string(array, length):
    sum = array.sum()
    ret = "desc:" + str(array.shape) + " data:"
    array = array.reshape([-1])[:length]
    for i in array:
        ret = ret + str(i) + " "
    ret = ret + " sum:" + str(sum)
    return ret

