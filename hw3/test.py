import copy
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

def back_propagate_error(weights, error, layer_outputs, lr_rate, momentum = 0, w_grad = None):
    error_props = - error * activation_err(layer_outputs[0][:, :-1])

    new_weight_grads = []

    for i, (w,o) in enumerate(zip(weights, layer_outputs[1:])):
        dw = o.T.dot(error_props)
        dw = lr_rate * dw

        if w_grad is not None:
            dw = dw + momentum * w_grad[i]

        new_weight_grads.append(dw)
        weights[i] = w - dw
        error_props = error_props.dot(w.T[:,:-1]) * (activation_err(o[:,:-1]))

    return weights, new_weight_grads

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

def get_errors(X, Y, weights, out = False):
    total_errors = 0
    valid_error_threshold = True
    for x, y in zip(X, Y):
        output, layer_outputs = calc_forward_pass(weights, x)

        if out and abs(output[0][0] -y) > 0.5:
            print("Predicted: " + str(output[0][0]) + " True:" + str(y))

        if abs(output[0][0] - y) > 0.05:
            valid_error_threshold = False
        if abs(output[0][0] - y) > 0.5:
            total_errors += 1
    if out:
        print("Total errors:" + str(total_errors))
    return valid_error_threshold

def get_mse_error(X, Y, weights):
    total_sq = 0
    for x,y in zip(X, Y):
        output, layer_output = calc_forward_pass(weights, x)
        total_sq += (output[0][0] - y) ** 2
    return total_sq / len(X)

def  train_nn(X, Y, weights, lr_rate, momentum):
    loop_count = 0
    epoch_error_map = {}
    w_grads = None
    saved_weights = copy.deepcopy(weights)
    while True:
        loop_count += 1

        if loop_count > 100000:
            loop_count = 0
            weights = saved_weights
            print("Resetting for lr_rate:" + str(lr_rate))

        threshold = get_errors(X, Y, weights, False)
        if threshold:
            epoch_error_map[loop_count]: get_mse_error(X, Y, weights)
            return loop_count, epoch_error_map

        if loop_count%100 == 0:
            epoch_error_map[loop_count]: get_mse_error(X, Y, weights)

        for i in np.random.randint(0, len(X), 16):
            x = X[i]
            y = Y[i]
            output, layer_outputs = calc_forward_pass(weights, x)
            error = (y - output[0][0])
            weights.reverse()
            weights, w_grads = back_propagate_error(weights, error, layer_outputs, lr_rate, momentum, w_grads)
            weights.reverse()

X, Y = get_dataset()

# initial weights
w1 = np.random.uniform(-1, 1, (5,4))
w2 = np.random.uniform(-1, 1, (5,1))


epochs_lr_map = {}
for i in range(5, 55, 5):
    lr_rate = i/100
    epochs_lr_map[lr_rate] = {}
    min_epoch = 1000000000

    for j in range(2):
        weights = [w1, w2]
        epochs_lr_map[lr_rate][j*0.9] = {"total_epochs": 0, "count": 0, "error_map": {}}
        epochs, epoch_error_map = train_nn(X,Y, weights, lr_rate, j*0.9)
        print("lr_rate: " + str(lr_rate) + ", epochs:" + str(epochs))
        epochs_lr_map[lr_rate][j*0.9]["total_epochs"] += epochs
        epochs_lr_map[lr_rate][j*0.9]["count"] += 1
        if epochs < min_epoch:
            epochs_lr_map[lr_rate][j*0.9]["error_map"] = epoch_error_map

print(epochs_lr_map)