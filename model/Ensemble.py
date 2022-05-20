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
		PATH = '../train/trained_models/'
		self.ensemble = []
		
		for i in range(0,ensemble_size):	
			self.ensemble += [AcT(B=B, T=T, N=N, C=C, nhead=nhead, num_layer=num_layer, d_last_mlp=d_last_mlp
										,classes=classes)]
			self.ensemble[i].load_state_dict(torch.load(PATH+'model_'+str(i+1)+'.pth'))
			self.ensemble[i].eval()


		### Model 1 ###
		self.model_1 =  AcT(B=B, T=T, N=N, C=C, nhead=nhead, num_layer=num_layer, d_last_mlp=d_last_mlp, classes=classes)
		self.model_1.load_state_dict(torch.load(PATH+'model_1.pth'))
		self.model_1.eval() 
		### Model 2 ###
		self.model_2 =  AcT(B=B, T=T, N=N, C=C, nhead=nhead, num_layer=num_layer, d_last_mlp=d_last_mlp, classes=classes)
		self.model_2.load_state_dict(torch.load(PATH+'model_2.pth'))
		self.model_2.eval() 
		### Model 3 ###
		self.model_3 =  AcT(B=B, T=T, N=N, C=C, nhead=nhead, num_layer=num_layer, d_last_mlp=d_last_mlp, classes=classes)
		self.model_3.load_state_dict(torch.load(PATH+'model_3.pth'))
		self.model_3.eval() 
		### Model 4 ###
		self.model_4 =  AcT(B=B, T=T, N=N, C=C, nhead=nhead, num_layer=num_layer, d_last_mlp=d_last_mlp, classes=classes)
		self.model_4.load_state_dict(torch.load(PATH+'model_4.pth'))
		self.model_4.eval() 

	def forward(self, X_in):
		out = []

		for i, l in enumerate(self.ensemble):
			out.append(self.ensemble[i](X_in))
		

		
		return sum(out)# [B, T+1, D_model]

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