import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

# GPU settings
from torch.utils.data import Dataset
from torchvision import transforms

torch.backends.cudnn.benchmark = True
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


class CustomImageDataset(Dataset):
    def __init__(self, inputs, labels):
        assert inputs.shape[0] == labels.shape[0]
        self.inputs = inputs
        self.labels = labels

    def __getitem__(self, index):
        img, label = self.inputs[index], self.labels[index]
        transforms_funcs = get_default_data_transforms(img)[0]
        img = transforms_funcs(img)
        return (img, label)

    def __len__(self):
        return self.inputs.shape[0]


def get_default_data_transforms(img):
    return transforms.Compose([
        transforms.ToPILImage(mode='L'),
        # transforms.RandomCrop(size=28),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3081])]),  # (0.24703223, 0.24348513, 0.26158784)


def compute_grad_aux(global_model, aux_loader):
    '''our own version of ratio'''
    optimizer = optim.SGD(global_model.parameters(), lr=0.1)  # lr doesn't matter here

    grad_square_sum_lst = [0] * 10

    for class_i_data_idx, (input, target) in enumerate(aux_loader):

        # 6. zero gradient, otherwise accumulated
        optimizer.zero_grad()

        # 1. prepare (X, Y) which belongs to same class
        # for ith class's data, with shape of [32, 3, 32, 32] batch size pf 32, with image 3*32*32
        # for ith class'labels, don't need one-hot coding here
        input, target = input.to(device), target.to(device)

        # 2. forward pass get Y_out
        output = global_model(input)  # feed ith class data, network output with shape [32, 10]
        # the output is logits (before softmax), usually output logits as default

        # 3. calculate cross-entropy between Y and Y_out
        loss = F.cross_entropy(output, target)

        # 4. backward pass to get gradients wrt weight using a batch of 32 of data
        loss.backward()

        # 5. record gradients wrt weights from the last layer
        # print(cifarnet.fc2.weight.grad.shape) here is [500, 10]
        # here we only need fetch ith gradient with shape [500]
        # In short, send data from ith class, fetch ith gradient tensor from [500, 10]
        # grad_lst.append(global_model.fc2.weight.grad[class_i_data_idx].cpu().numpy())
        # new
        # the above descripting all wrong
        for name, param in global_model.named_parameters():
            # print(name, param.grad.shape)
            # print((param.grad ** 2).sum().item())
            grad_square_sum_lst[class_i_data_idx] += ((param.grad ** 2)).mean().item()

    return grad_square_sum_lst


# threshold changed the meaning to percentage
# threshold doesn't work here
# grad_square_sum_lst: [10, ]
def compute_ratio(grad_square_sum_lst, temp=1):
    ''' original version in the paper '''

    grad_sum = np.array(grad_square_sum_lst)
    # print(grad_sum)

    grad_sum = grad_sum.min() / grad_sum
    # print(grad_sum)

    # def softmax(grad_sum, temp = 1):
    #     grad_sum = grad_sum - grad_sum.mean()
    #     return np.exp(grad_sum / temp) / np.exp(grad_sum / temp).sum()

    # grad_sum_normalize = softmax(grad_sum, temp)
    grad_sum_normalize = grad_sum / grad_sum.sum()
    # grad_sum_normalize = grad_sum
    return grad_sum_normalize


def compute_ratio_per_client_update(client_models, client_idx, aux_loader):
    ra_dict = {}
    for i, client_model_update in enumerate(client_models):
        grad_square_sum_lst = compute_grad_aux(client_model_update, aux_loader)
        grad_sum_normalize = compute_ratio(grad_square_sum_lst)
        ra_dict[client_idx[i]] = grad_sum_normalize

    return ra_dict
