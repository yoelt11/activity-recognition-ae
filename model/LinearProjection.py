import torch
from torch import nn

class LinearProjection(nn.Module):
	""" This layer creates a linear projection of the input:
		B: batch
		T: Frame number
		C: Channel number
		N: Node number

		Parameters:
			- X = [B,T,N,C] 
		Updatable Variables:
			- W = [N*C,D]
			- x_cls = [1,D]
			- X_pos = [T,D]
			"""
	def __init__(self, B=40, T=5, N=17, C=3, D_model=64):

		super().__init__()

		self.W_lo = nn.Linear(N*C, D_model)
		self.x_cls = torch.randn([1, D_model], requires_grad=True)
		self.X_pos = torch.randn([T+1, D_model], requires_grad=True)
	
	def forward(self, X_in):

		B,T,N,C = X_in.shape

		X_lo = self.W_lo(X_in.flatten(2)) # out_dim = [B, T, D_model]
		
		# we have to adapt x_cls for batch operation; 
		# not really sure if this is the way, since it might cause bprop problems
		# options are .expand() or .repeat() 
		x_bcls = torch.unsqueeze(self.x_cls, 0).expand(B,-1,-1) # out_dim = [B, 1, D_model]
		# same thing for X_pos
		X_bpos = torch.unsqueeze(self.X_pos, 0).expand(B,-1,-1) # out_dim = [B, T+1, D_model]

		X_lp = torch.cat([x_bcls,X_lo],1) + X_bpos # out_dim = [B, T+1, D_model]
		
		return X_lp

#---------------------------------------	
if __name__=="__main__":
	# Layer-testing zone

	B, T, N, C = 40, 5, 17, 3
	X_in = torch.randn([B,T,N,C])

	model = LinearProjection()
	out = model(X_in)

	print(out.shape)