import torch
import torchvision

    
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./mnist', train=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
torchvision.transforms.Normalize((0.1307,),(0.3081,))])), batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader( torchvision.datasets.MNIST('./mnist', train=False, download=True, transform=torchvision.transforms.Compose(
[torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,),(0.3081,))])), batch_size=batch_size_test, shuffle=True)
example = enumerate(train_loader)
bacth_idx, (example_data, example_targets) = next(example)
print(example_data.shape)