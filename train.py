# train.py

from utils import *
from model import *
from config import Config
import sys
import torch.optim as optim
from torch import nn
import torch
import numpy as np 
if __name__=='__main__':
    config = Config()
    train_file = 'data/sst2/train.csv'
    if len(sys.argv) > 2:
        train_file = sys.argv[1]
    test_file = 'data/sst2/test.csv'
    if len(sys.argv) > 3:
        test_file = sys.argv[2]
    
    dataset = Dataset(config)
    dataset.load_data(train_file, test_file)
    
    # Create Model with specified optimizer and loss function
    ##############################################################
    model = Transformer(config, len(dataset.vocab))
    n_all_param = sum([p.nelement() for p in model.parameters()])
    print('#params = {}'.format(n_all_param))

    if torch.cuda.is_available():
        model.cuda()
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    NLLLoss = nn.CrossEntropyLoss()
    model.add_optimizer(optimizer)
    model.add_loss_op(NLLLoss)
    ##############################################################
    
    train_losses = []
    val_accuracies = []
    # state = {'net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
    # torch.save(state, "kkkk")
    max_score=0.1
    
    for i in range(config.max_epochs):
    # for i in range(1):
        print ("Epoch: {}".format(i))
        train_loss,val_accuracy = model.run_epoch(dataset.train_iterator, dataset.val_iterator, i)
        train_losses.append(train_loss)
        val_accuracies.append(val_accuracy)
        train_acc = evaluate_model(model, dataset.train_iterator)
        val_acc = evaluate_model(model, dataset.val_iterator)
        # test_acc = evaluate_model(model, dataset.test_iterator)
        print ('Final Training Accuracy: {:.4f}'.format(train_acc))
        print ('Final Validation Accuracy: {:.4f}'.format(val_acc))
        # if val_acc>max_score:
        #     np.savetxt("embedding.txt",model.position_enc.weight.detach().cpu().numpy(),fmt='%.18e')
        #     max_score=val_acc