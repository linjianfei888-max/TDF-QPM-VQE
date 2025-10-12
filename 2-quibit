import torch 
import torch.nn as nn 
import torch.optim as optim  
class Qubit2Net(nn.Module):
def __init__(self):
         super().__init__()
         self.fc1 = nn.Linear(4, 32)
         self.fc2 = nn.Linear(32, 4)
         self.sigmoid = nn.Sigmoid()

def forward(self, x):
         x = torch.relu(self.fc1(x))
         psi = self.fc2(x)
         return self.sigmoid(psi)

def hamiltonian_2(psi):
sigma_xx = torch.tensor([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.float)
energy = torch.einsum('bi,ij,bj->b', psi, sigma_xx, psi)     
return energy.mean()  

X = torch.randn(50, 4) 
model = Qubit2Net() 
optimizer = optim.Adam(model.parameters(), lr=0.01) 
initial_params = [p.clone().detach() for p in model.parameters()] 
for epoch in range(500):
     optimizer.zero_grad()
     psi = model(X)
loss = hamiltonian_2(psi)     
loss.backward()     
optimizer.step()  

psi_final = model(X).detach() 
print("2-Qubit Final wavefunction (sample):", psi_final[0].numpy()) print("2-Qubit Energy:", hamiltonian_2(psi_final).item())
