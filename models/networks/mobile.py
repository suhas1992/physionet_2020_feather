import torch 
import torch.nn as nn

expansion_rate = 1

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

# Define a Batchnormalized ReLU-activated Convolutional Layer
class convBNRelu(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        pads = (kernel_size - 1) // 2
        super(convBNRelu, self).__init__(
            nn.Conv1d(in_planes, out_planes, kernel_size, stride=stride, padding=pads,groups=groups, bias=False),
            nn.BatchNorm1d(out_planes),
            nn.LeakyReLU(inplace=True)
        )

# Define a residual botleneck layer
class residual_bottleneck_layer(nn.Module):
    def __init__(self, channel_size, output_channel_size, stride_size=1, expansion_factor=expansion_rate):
        super(residual_bottleneck_layer, self).__init__()

        # Checks if residual conditions can be applied
        self.residual = False

        # Check if conditions meet for residual connection
        if stride_size==1 and channel_size==output_channel_size:
            self.residual = True
            
        # Expanded dimension
        expanded_dim = channel_size*expansion_factor

        # Begin appending the layers
        layers = []
        if expansion_factor != 1:
            layers.append(convBNRelu(channel_size, expanded_dim, kernel_size=1))

        layers.append(convBNRelu(expanded_dim, expanded_dim, stride=stride_size, groups=expanded_dim))
        layers.append(nn.Conv1d(expanded_dim, output_channel_size, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm1d(output_channel_size))

        self.resbConv = nn.Sequential(*layers)

    def forward(self, x):
        if self.residual:
            return x + self.resbConv(x)
        else:
            return self.resbConv(x)

# Define the network
class mobileNet(nn.Module):
    def __init__(self, input_channels, residuals, output_classes):
        super(mobileNet, self).__init__()

        model_layers = []
        # Create initial 32 filter 2d Convolutional Layer
        model_layers.append(convBNRelu(input_channels, out_planes=32, kernel_size=3, stride=2))
        # Number of input channels to the model
        inp = 32

        # Create the residual bottleneck layers
        for t, c, n, s in residuals:
            for i in range(n):
                stride = s if i==0 else 1
                model_layers.append(residual_bottleneck_layer(channel_size=inp, output_channel_size=c, stride_size=stride, expansion_factor=t))
                inp = c

        # Create the final layers
        model_layers.append(convBNRelu(in_planes=inp, out_planes=1280, kernel_size=1,stride=1))
        # Convert it to sequential layers
        self.mobileConv = nn.Sequential(*model_layers)

        # Make it into Sequential layers
        self.finalClassifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, output_classes * 100, bias=False),
            nn.Dropout(0.2),
            nn.Linear(output_classes * 100, output_classes * 50, bias=False),
            nn.Dropout(0.2),
            nn.Linear(output_classes * 50, output_classes * 10, bias=False),
            nn.Dropout(0.2),
            nn.Linear(output_classes * 10, output_classes, bias=False)
        )
        
        # Create a linear layer to pass to Center Loss
        self.linear_closs = nn.Linear(1280, 2300, bias=False)
        self.sig_closs = nn.Sigmoid()

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                #nn.init.zeros_(m.bias)
                
    
    def forward(self, x):
        x = self.mobileConv(x)

        # Perform global averaging
        x = x.mean([2])
        
        closs_output = self.linear_closs(x)
        closs_output = self.sig_closs(closs_output)

        # Put result through a classifier
        return closs_output, self.sig_closs(self.finalClassifier(x))