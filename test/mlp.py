import torch
import torch.nn.functional as F   
from torchvision import datasets
import numpy as np
from torch.utils.data import Dataset


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP,self).__init__()   
        self.fc1 = torch.nn.Linear(20,16)  
        self.fc2 = torch.nn.Linear(16,10) 
        # self.fc3 = torch.nn.Linear(128,2) 
        
    def forward(self,input):
        # input = input.view(-1,28*28)       
        output = F.relu(self.fc1(input))   
        # output = F.relu(self.fc2(output))
        output = self.fc2(output)
        # print(output)
        return output
class MLP10(torch.nn.Module):
    def __init__(self):
        super(MLP10,self).__init__()   
        self.fc1 = torch.nn.Linear(10,10)  
        self.fc2 = torch.nn.Linear(10,10) 
        # self.fc3 = torch.nn.Linear(128,2) 
        
    def forward(self,input):
        # input = input.view(-1,28*28)       
        output = F.relu(self.fc1(input))   
        # output = F.relu(self.fc2(output))
        output = self.fc2(output)
        # print(output)
        return output
class MLP512(torch.nn.Module):
    def __init__(self):
        super(MLP512,self).__init__()   
        self.fc1 = torch.nn.Linear(512,128)  
        self.fc2 = torch.nn.Linear(128,10) 
        # self.fc3 = torch.nn.Linear(128,2) 
        
    def forward(self,input):
        # input = input.view(-1,28*28)       
        output = F.relu(self.fc1(input))   
        # output = F.relu(self.fc2(output))
        output = self.fc2(output)
        # print(output)
        return output
class MLP1024(torch.nn.Module):
    def __init__(self):
        super(MLP1024,self).__init__()   
        self.fc1 = torch.nn.Linear(1024,512)  
        self.fc2 = torch.nn.Linear(512,10) 
        # self.fc3 = torch.nn.Linear(128,2) 
        
    def forward(self,input):
        # input = input.view(-1,28*28)       
        output = F.relu(self.fc1(input))   
        # output = F.relu(self.fc2(output))
        output = self.fc2(output)
        # print(output)
        return output