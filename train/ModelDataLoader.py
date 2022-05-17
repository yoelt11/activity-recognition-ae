import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TrainDataloader(Dataset):
	def __init__(self, PATH):
		with open(PATH + 'X_train.npy', 'rb') as f:
			self.x = torch.from_numpy(np.load(f))
		with open(PATH + 'y_train.npy', 'rb') as f:
			self.y =  torch.from_numpy(np.load(f))

		self.n_samples = self.y.shape[0]

	def __getitem__(self, index):
		return self.x[index], self.y[index]

	def __len__(self):
		return self.n_samples


class TestDataloader(Dataset):
	def __init__(self, PATH):		
		with open( PATH + 'X_test.npy', 'rb') as f:
			self.x = torch.from_numpy(np.load(f))
		with open(PATH + 'y_test.npy', 'rb') as f:
			self.y =  torch.from_numpy(np.load(f))
		
		self.n_samples = self.y.shape[0]

	def __getitem__(self, index):
		return self.x[index], self.y[index]

	def __len__(self):
		return self.n_samples


"""
---------------------------------------	
	Testing Zone
---------------------------------------	
"""
if __name__=='__main__':
	train_data = TrainDataloader()
	print(f'train size: ', train_data.__len__()) # 12562

	test_data = TestDataloader()
	print(f'test size: ', test_data.__len__()) # 2867
