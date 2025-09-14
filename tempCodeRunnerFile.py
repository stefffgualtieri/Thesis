    for lin in linear_layers:
        #Weights
        number_weights = lin.weight.numel()
        lin.weight.copy_(best_w[idx:idx + number_weights].view_as(lin.weight))
        idx += number_weights
        #Bias
        if lin.bias is not None:
            number_bias = lin.bias.numel()
            lin.bias.copy_(best_w[idx: idx + number_bias].view_as(lin.bias))
            idx += number_bias