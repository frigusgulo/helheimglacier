import torch 
import gpytorch
import pickle
from matplotlib import pyplot as plt
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


dataset = np.load('/home/dunbar/Research/helheim/data/pcluster.npy',allow_pickle=True)
dataset = dataset.tolist()
train_x = [torch.tensor(data[:,0:2],dtype=torch.float) for data in dataset]
train_y = [torch.tensor(data[:,-1],dtype=torch.float) for data in dataset]


print("Length of training data: {}".format(len(train_x)))

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)

#print(model.covar_module.ScaleKernel.outputscale())
# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.parameters()},  # Includes GaussianLikelihood parameters
], lr=1e-3)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter = 400
settings = gpytorch.settings.debug(state=False)
with settings:
	for i in range(training_iter):
		# Zero gradients from previous iteration
		for batch, (trainx,trainy) in enumerate(zip(train_x,train_y)):


			try:
				optimizer.zero_grad()
				output = model(trainx)
				loss = -mll(output, trainy)

				loss.backward()
				optimizer.step()
				
			
			except:
				#print("Failed at Batch: {}".format(batch))
				#print("Data shape: {} {}\n".format(trainx.shape,trainy.shape))
				print("Data type: {}".format(trainx.dtype))
				pass
			
		print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f   outputscale: %.3f' % (
		    i + 1, training_iter, loss.item(),
		    model.covar_module.base_kernel.lengthscale.item(),
		    #model.covar_module.scale_kernel.outputscale.item(),
		    model.likelihood.noise.item(),
		    model.covar_module.raw_outputscale.item()
		))

for param_name, param in model.named_parameters():
    print(f'Parameter name: {param_name:42} value = {param.item()}')

	# Get into evaluation (predictive posterior) mode


# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
