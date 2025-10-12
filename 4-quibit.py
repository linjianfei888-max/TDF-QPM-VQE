class Qubit4Net(nn.Module):
def __init__(self):         
	super().__init__()         
	self.fc1 = nn.Linear(16, 128)         
	self.fc2 = nn.Linear(128, 16)         
	self.sigmoid = nn.Sigmoid()      

def forward(self, x):
    x = torch.relu(self.fc1(x))         
    psi = self.fc2(x)
    return self.sigmoid(psi)

def hamiltonian_4(psi):
sigma_x = torch.tensor([[0, 1], [1, 0]], dtype=torch.float)
H = torch.kron(sigma_x, sigma_x)     
for _ in range(2):      
     H = torch.kron(H, sigma_x)
energy = torch.einsum('bi,ij,bj->b', psi, H, psi)
return energy.mean()  

X = torch.randn(50, 16) 
model = Qubit4Net() 
optimizer = optim.Adam(model.parameters(), lr=0.01) 
initial_params = [p.clone().detach() for p in model.parameters()] 
for epoch in range(500):
optimizer.zero_grad()
psi = model(X)
loss = hamiltonian_4(psi)
loss.backward()     
optimizer.step()  

psi_final = model(X).detach() 
print("4-Qubit Final wavefunction (sample, first 4):", psi_final[0][:4].numpy()) 
print("4-Qubit Energy:", hamiltonian_4(psi_final).item())
