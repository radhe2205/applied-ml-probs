import math
import numpy as np

def sigmoid(num):
    return 1/(1+math.exp(-num))

def percept_activation(outputs):
    sigmoid_ = np.vectorize(sigmoid)
    return np.array([sigmoid_(out) for out in outputs])

def der_sigmoid(num):
    return num * (1-num)

def activation_err(outputs):
    der_sigmoid_ = np.vectorize(der_sigmoid)
    return np.array([der_sigmoid_(out) for out in outputs])

def calc_forward_pass(weights, x):
    output = np.concatenate((x, np.ones((1, len(x)))), axis = 1)
    layer_outputs = [output]
    for w in weights:
        output = output.dot(w)
        output = percept_activation(output)
        output = np.concatenate((output, np.ones((1, len(output)))), axis=1)
        layer_outputs.insert(0, output)
    return output, layer_outputs

def back_propagate_error(weights, error, layer_outputs, lr_rate):
    if error == 0:
        return weights

    error_props = - error * activation_err(layer_outputs[0][:, :-1])

    for i, (w,o) in enumerate(zip(weights, layer_outputs[1:])):
        dw = o.T.dot(error_props)
        dw = lr_rate * dw
        weights[i] = w - dw
        error_props = error_props.dot(w.T[:,:-1]) * (activation_err(o[:,:-1]))

    return weights

def get_dataset():
    X = []
    Y = []
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    x = np.array([i,j,k,l])
                    x.shape = (1,4)
                    X.append(x)
                    Y.append((i+j+k+l) % 2)
    return X, Y

def get_errors(X, Y, weights):
    total_errors = 0
    valid_error_threshold = True
    for x, y in zip(X, Y):
        output, layer_outputs = calc_forward_pass(weights, x)
        print("True: " + str(output[0][0]) + " Expected:" + str(y))

        if abs(output[0][0] - y) > 0.05:
            valid_error_threshold = False
        if abs(output[0][0] - y) > 0.5:
            total_errors += 1
    print("Total errors:" + str(total_errors))
    return valid_error_threshold


X, Y = get_dataset()

# initial weights
w1 = np.random.uniform(-1, 1, (5,4))
w2 = np.random.uniform(-1, 1, (5,1))

lr_rate = 0.05

weights = [w1,w2]
loop_count = 0
while True:
    loop_count += 1

    threshold = get_errors(X,Y,weights)
    if threshold:
        break

    # TODO: Add condition for stopping the loop
    for i in np.random.randint(0, len(X), 16):
        x = X[i]
        y = Y[i]
        output, layer_outputs = calc_forward_pass(weights, x)
        error = (y - output[0][0])
        weights.reverse()
        weights = back_propagate_error(weights, error, layer_outputs, 0.2)
        weights.reverse()

get_errors(X, Y, weights)
