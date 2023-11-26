import torch
import torchvision
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "ResNet18"
num_epochs = 10
batch_size = 64
learning_rate = 0.001

def load_model(model_name):
    if model_name == "ResNet18":
        return torchvision.models.resnet18()
    else:
        return torchvision.models.resnet50()
    
def load_ds(path_to_train, path_to_valid):
    transform = transforms.Compose([
        # transforms.Resize(256),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(degrees=45),
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        # transforms.CenterCrop(224),
        transforms.ToTensor()
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = torchvision.datasets.CIFAR10(
        root=path_to_train, 
        transform=transform,
        download=True
    )
    valid_dataset = torchvision.datasets.CIFAR10(
        root=path_to_valid,
        transform=transform,
        download=True
    )
    print("Datasets successfully loaded")
    return train_dataset, valid_dataset

def train_model(model, train, valid):
    model = torch.nn.DataParallel(model)
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        print(f'Begin training Epoch {epoch+1}/{num_epochs}')
        acc_loss = 0
        for inputs, labels in train:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc_loss += loss
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {acc_loss:.4f}')

        valid_loss = valid_model(model, valid)
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {valid_loss:.4f}')

    # save the model parameters
    torch.save(model.state_dict(), str(num_epochs) +  '_' + str(batch_size) + '_' + str(learning_rate) + '.pth')

def valid_model(model, valid):
    acc_loss = 0
    criterion = torch.nn.CrossEntropyLoss()

    for inputs, labels in valid:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        acc_loss += loss
    
    return acc_loss

model = load_model(model_name)
train_ds, valid_ds = load_ds("./data", "./data")
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
valid_loader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, shuffle=True, num_workers=0)
train_model(model, train_loader, valid_loader)
