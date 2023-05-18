

class SimpleConvNet:
    def __init__(self, input_dim=(1, 28, 28), conv_params={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1}, 
                hidden_size=100, output_size=10, weight_init_std=0.01):
        filter_num = conv_params['filter_num']
        filter_size = conv_params['filter_size']
        filter_pad = conv_params['pad']
        filter_stride = conv_params['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))

        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['W2'] = (hidden_size, output_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['b3'] = np.zeros(output_size)

        self.layers = OrderDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'], conv_params['stride'], conv_params['pad'])
        