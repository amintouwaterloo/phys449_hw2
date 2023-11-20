import torch.nn.init

def initialize_ising_boltzmann_net(net):
    # Initialize weights (J) using small random values
    for param in net.parameters():
        if len(param.size()) > 1:  # Check if the parameter is a weight matrix
            torch.nn.init.normal_(param, mean=0, std=0.01)  # You can adjust the mean and std as needed
    
    # Initialize biases (h) using small random values
    for name, param in net.named_parameters():
        if 'bias' in name:
            torch.nn.init.normal_(param, mean=0, std=0.01)  # You can adjust the mean and std as needed

# Example usage:
import torch.nn

class IsingBoltzmannNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(IsingBoltzmannNet, self).__init__()
        self.fc = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

# Instantiate the neural network
input_size = 10  # Adjust according to the number of visible units
hidden_size = 5  # Adjust according to your specific architecture
ising_net = IsingBoltzmannNet(input_size, hidden_size)

# Initialize the neural network parameters
initialize_ising_boltzmann_net(ising_net)