from torchvision import datasets, transforms


def load_data():
    # train
    train_dataset = datasets.MNIST('../images/data', train=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ])),
    train_data = train_dataset[0].data
    train_targets = train_dataset[0].targets

    train_xy = []
    for i in range(train_targets.size(0)):
        train_xy.append((train_data[i].unsqueeze(0).float(), train_targets[i]))

    # test
    test_dataset = datasets.MNIST('./data', train=False,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                  ])),
    test_data = test_dataset[0].data
    test_targets = test_dataset[0].targets

    test_xy = []
    for i in range(test_targets.size(0)):
        test_xy.append((test_data[i].unsqueeze(0).float(), test_targets[i]))

    return train_xy, test_xy
