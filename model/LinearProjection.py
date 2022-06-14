import torch
from torch import nn
"""
Code based on: Action Transformer: A Self-Attention Model for Short-Time Pose-Based
Human Action Recognition
From Author: Vittori Mazzia, Simone Angarano1, Francesco Salvetti, Federico Angelin and Marcello Chiaberge
"""
class LinearProjection(nn.Module):
	""" This layer creates a linear projection, it also appends the tokens and adds the position embeddings. 
		This is the previous step before the encoder: [x_bcls, X_2Dpose] + X_pos  
		
		Parameters:
		-----------
			- B: batch
			- T: Frame number
			- C: Channel number
			- N: Node number
			- D_model: the expansion output dimension of the linear projection
		Attributes
		----------:
			- W = linear layer to expand X to D_model dimension [N*C,D]
			- x_cls = cls token [1,D]
			- X_pos = position embedding
			"""
	def __init__(self, B=40, T=5, N=17, C=3, D_model=64):

		super().__init__()

		self.bn = nn.BatchNorm1d(N * C).float()
		self.W_lo = nn.Linear(N*C, D_model, bias=False)
		self.x_cls = nn.Parameter(torch.randn([1, D_model]))
		self.embedding = nn.Embedding(T+1, D_model, norm_type=2.0, max_norm=1.0)

	def forward(self, X_in):

		B, T, N, C = X_in.shape
		# Expansion
		X_lo = self.W_lo(self.bn(X_in.flatten(2).permute(0,2,1).float()).permute(0,2,1)) # out_dim = [B, T, D_model]
		
		# expanding x_bcls to adjust it to the batch size
		x_bcls = self.x_cls.expand([B,-1,-1]) # out_dim = [B, 1, D_model]
		
		positions = torch.arange(start=0, end=T+1).int()
		X_pos = self.embedding(positions) # out_dim = [B, T+1, D_model]
		
		X_lp = torch.cat([x_bcls, X_lo], 1) + X_pos  # out_dim = [B, T+1, D_model]
		
		return X_lp # [B, T+1, D_model]

"""
---------------------------------------	
	Testing Zone
---------------------------------------	
"""
if __name__=="__main__":
	# Layer-testing zone

	B, T, N, C = 40, 30, 17, 3
	X_in = torch.randn([B,T,N,C])

	model = LinearProjection(B,T,N,C,64)
	out = model(X_in)

	print(out.shape)
