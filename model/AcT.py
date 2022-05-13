import torch
from torch import nn
from LinearProjection import LinearProjection

class AcT(nn.Module):

	def __init__(self, B=40, T=5, N=17, C=3, D_model=64, D_mlp=256, classes=20):

		super().__init__()
		
		# Constants
		self.D_model = D_model
		self.B = B

		# Network Layers
		self.linear_projection = LinearProjection(B,T,N,C,D_model)
		encoder_layer = nn.TransformerEncoderLayer(d_model=D_model, nhead=1)
		self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
		self.mlp = nn.Sequential(
					nn.Linear(D_model, D_mlp),
					nn.ReLU(),
					nn.Linear(D_mlp, classes),
					nn.Softmax() # not really sure whether to add this one here or later
					)

	def forward(self, X_in):
		
		lp = self.linear_projection(X_in) # [B, T+1, D_model]

		enc_out = self.transformer_encoder(lp) # [B, T+1, D_model]

		out =  self.mlp(enc_out[:,0,:].squeeze())

		return out

if __name__=="__main__":
	
	B, T, N, C = 40, 5, 17, 3
	X_in = torch.randn([B,T,N,C])

	model = AcT()
	
	out = model(X_in)

	print(out.shape)