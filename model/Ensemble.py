import torch
from torch import nn
from AcT import AcT

"""
Code based on: Action Transformer: A Self-Attention Model for Short-Time Pose-Based
Human Action Recognition
From Author: Vittori Mazzia, Simone Angarano1, Francesco Salvetti, Federico Angelin and Marcello Chiaberge
"""
class Ensemble(nn.Module):
	""" Ensemble of AcT models 
		
		Parameters:
		-----------

		Attributes
		----------:

			"""
	def __init__(self, ensemble_size, B=40, T=5, N=17, C=3, nhead=1, num_layer=4, d_last_mlp=256, classes=20):

		super().__init__()
		
		self.ensemble = nn.ModuleList()
		self.ensemble_size = ensemble_size
		self.B = B
		self.T = T
		self.N = N
		self.C = C
		self.nhead = nhead
		self.num_layer = num_layer
		self.d_last_mlp = d_last_mlp
		self.classes = classes 

 
	def load_weights(self, PATH):
		
		for i in range(0, self.ensemble_size):	
			self.ensemble.append(AcT(B=self.B, T=self.T, N=self.N, C=self.C, nhead=self.nhead, num_layer=self.num_layer, d_last_mlp=self.d_last_mlp
										,classes=self.classes))
			self.ensemble[i].load_state_dict(torch.load(PATH+'/models/model_'+str(i+1)+'.pth'))
			self.ensemble[i].eval()

	def forward(self, X_in):
		out = torch.zeros(self.classes)

		for i, l in enumerate(self.ensemble):
			out = l(X_in) + out
		

		return out / self.ensemble_size # [B, T+1, D_model]

"""
---------------------------------------	
	Testing Zone
---------------------------------------	
"""
if __name__=="__main__":
	# Layer-testing zone

	batch_size, T, N, C, nhead, num_layer, classes = 1, 30, 17, 3, 2, 4, 20
	X_in = torch.randn([batch_size,T,N,C])

	model = Ensemble(B=batch_size, T=30, N=17, C=3, nhead=3, num_layer=6, d_last_mlp=502, classes=20)
	out = model(X_in)

	if batch_size == 1:
		print("models voted for: ", out.argmax(0))
	else:
		print("models voted for: ", out.argmax(1))
