import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.insert(0, '..')
sys.path.insert(0, './trained_models/')
sys.path.insert(0, '../model/')
from ModelDataLoader import TrainDataloader, TestDataloader
import AcT as ClassificationModel
import yaml


def loadDataset(batch_size, PATH):
    train_dataset = TrainDataloader(PATH=PATH)
    test_dataset = TestDataloader(PATH=PATH)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8, drop_last=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8, drop_last=True)

    return train_loader, test_loader

def check_accuracy(loader, model, batch_size, step, writer):

    num_samples = 0
    model.eval() # set model to evaluation mode
    num_correct = 0
    with torch.no_grad():
        for x, y in loader:
            network_output = model(x)
            # set y values between 0 and 1
            _, predictions = network_output.max(1)
            targets = y
            num_correct += (predictions == targets).sum()
            num_samples += batch_size
    print("correct predictions: ", num_correct.item())
    print("number of samples: ", num_samples)
    print("accuracy percentage: ", num_correct.item()/num_samples)

    writer.add_scalar('Test Accuracy:', num_correct.item()/num_samples, global_step=step)
    
    step += 1
    
    return writer, step

def train_function(loader, model, batch_size, step, writer):
    model.train() # sets model to training mode

    for i, (x, y) in enumerate(loader):

        for param in model.parameters(): # Set grad to zero
            param.grad = None

        network_output = model(x) # predict

        # set y values between 0 and 1

        loss = loss_function(network_output, y)
        loss.backward()
        optimizer.step()

        # output and write metrics
        _, predictions = network_output.max(1)
        targets = y
        num_correct = (predictions == targets).sum()
        running_train_acc = float(num_correct)/float(x.shape[0])
        if i % 25 == 0:
            print("In training Loss: ",loss.item())

        # update tensorboard
        writer.add_scalar('Training Loss', loss, global_step=step)
        writer.add_scalar('Training Accuracy', running_train_acc, global_step=step)

        step += 1

        return writer, step
        s
def load_yaml(PATH='./train_config.yaml'):
    """
        Reads yaml configuration script
    """
    stream = open(PATH, 'r')
    dictionary = yaml.safe_load(stream)
    
    return dictionary

if __name__=='__main__':
    parameters = load_yaml()

    batch_size, T, N, C, nhead, num_layer, d_last_mlp, classes = list(parameters['MODEL_PARAM'].values())
    
    # Load Dataset
    train_loader, test_loader = loadDataset(batch_size, PATH=parameters['DS_PATH'])
    # Load Model
    model = ClassificationModel.AcT(B=batch_size, T=T, N=N, C=C, nhead=nhead, num_layer=num_layer, d_last_mlp=d_last_mlp, classes=classes)
    # For tensorboard
    writer = SummaryWriter(comment=parameters['TB_COMMENT'])
    # Training Parameters
    loss_function = nn.NLLLoss()
    optimizer = optim.AdamW(model.parameters(), lr=parameters['TRAIN_PARAM']['LEARNING_RATE'],  weight_decay=parameters['TRAIN_PARAM']['WEIGHT_DECAY'],)
    num_epochs = parameters['TRAIN_PARAM']['EPOCHS']

    # Training Loop
    train_step = 0
    test_step = 0
    
    for epoch in range(num_epochs):

        writer, train_step = train_function(train_loader, model, batch_size, train_step, writer)
        print("Epoch: ", epoch)
        writer, test_step = check_accuracy(test_loader, model, batch_size, train_step, writer)
        if epoch % 20 == 0:
            torch.save(model.state_dict(), parameters['MODEL_OUT_PATH'])
    torch.save(model.state_dict(), parameters['MODEL_OUT_PATH'])       