import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
class Generator(nn.Module):
    def __init__(self, input_dim=100, output_dim=1, img_size=28):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, img_size * img_size * output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, self.img_size, self.img_size)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_size=28, output_dim=1):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(img_size * img_size, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


def gradient_penalty(discriminator, real_imgs, fake_imgs, device='cpu'):
    alpha = torch.rand(real_imgs.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_imgs + (1 - alpha) * fake_imgs).requires_grad_(True)
    d_interpolates = discriminator(interpolates)
    gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(d_interpolates.size()).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

class SyntheticDataset(Dataset):
    def __init__(self, num_samples=10000, img_size=28, labeled=True):
        self.num_samples = num_samples
        self.img_size = img_size
        self.labeled = labeled
        self.data = np.random.rand(num_samples, img_size, img_size).astype(np.float32)
        if labeled:
            self.labels = np.random.randint(0, 2, num_samples)  # Binary labels
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = self.data[idx]
        img = self.transform(img)
        if self.labeled:
            label = self.labels[idx]
            return img, torch.tensor(label, dtype=torch.float)
        else:
            return img

num_samples = 10000  
synthetic_labeled_data = SyntheticDataset(num_samples=num_samples, labeled=True)
synthetic_unlabeled_data = SyntheticDataset(num_samples=num_samples, labeled=False)

labeled_data_loader = DataLoader(synthetic_labeled_data, batch_size=64, shuffle=True)
unlabeled_data_loader = DataLoader(synthetic_unlabeled_data, batch_size=64, shuffle=True)

adversarial_loss = nn.BCELoss()
generator = Generator()
discriminator = Discriminator()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, weight_decay=1e-5)
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, weight_decay=1e-5)
scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=10, gamma=0.5)
scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=10, gamma=0.5)
l1_lambda = 1e-5
gp_lambda = 10

num_epochs = 10
g_losses = []
d_losses = []

def add_noise(z, noise_strength=0.1):
    noise = torch.randn_like(z) * noise_strength
    return z + noise

def add_noise_to_images(images, noise_strength=0.1):
    noise = torch.randn_like(images) * noise_strength
    return images + noise


for epoch in range(num_epochs):
    for i, (imgs, _) in enumerate(labeled_data_loader):
        valid = torch.ones(imgs.size(0), 1, requires_grad=False) * 0.9  # Label smoothing for real labels
        fake = torch.zeros(imgs.size(0), 1, requires_grad=False) + 0.1  # Label smoothing for fake labels

        real_imgs = imgs.to(next(discriminator.parameters()).device)

        optimizer_G.zero_grad()
        z = torch.randn(imgs.size(0), 100).to(next(generator.parameters()).device)
        noisy_z = add_noise(z)  # Inject noise into generator input
        gen_imgs = generator(noisy_z)
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        l1_regularization = sum(param.abs().sum() for param in generator.parameters())
        g_loss += l1_lambda * l1_regularization

        g_loss.backward()
        optimizer_G.step()


        optimizer_D.zero_grad()
        noisy_real_imgs = add_noise_to_images(real_imgs)  # Inject noise into real images
        real_loss = adversarial_loss(discriminator(noisy_real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        fake_imgs = generator(noisy_z)  
        gp = gradient_penalty(discriminator, real_imgs, fake_imgs)
        d_loss += gp_lambda * gp

        d_loss.backward()
        optimizer_D.step()

    scheduler_G.step()
    scheduler_D.step()

    g_losses.append(g_loss.item())
    d_losses.append(d_loss.item())

    print(f"Epoch {epoch+1}/{num_epochs}, Generator Loss: {g_loss.item()}, Discriminator Loss: {d_loss.item()}")

num_classes = 4
num_generate_samples = 1000
z_labeled = torch.randn(num_generate_samples, 100)
gen_imgs_labeled = generator(z_labeled)
gen_imgs_labeled = gen_imgs_labeled.detach().numpy()
labels_labeled = np.random.randint(0, num_classes, size=(num_generate_samples,)).astype(np.int64)


z_unlabeled = torch.randn(num_generate_samples, 100)
gen_imgs_unlabeled = generator(z_unlabeled)
gen_imgs_unlabeled = gen_imgs_unlabeled.detach().numpy()

labeled_data = {'images': gen_imgs_labeled, 'labels': labels_labeled}
unlabeled_data = {'images': gen_imgs_unlabeled}

plt.figure(figsize=(10, 5))
plt.plot(g_losses, label="Generator Loss")
plt.plot(d_losses, label="Discriminator Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
print(labeled_data)
print(unlabeled_data)
print("End")

unlabeled_images_tensor = torch.tensor(unlabeled_data['images'], dtype=torch.float32)
labeled_images_tensor = torch.tensor(labeled_data['images'], dtype=torch.float32)
labeled_labels_tensor = torch.tensor(labeled_data['labels'].squeeze(), dtype=torch.long)  # Use .squeeze() if labels are 2D
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Define the ResNet15 model
class ResNet15(nn.Module):
    def __init__(self):
        super(ResNet15, self).__init__()
        self.resnet = models.resnet18(pretrained=False)  # Use ResNet-18 as base model
        self.resnet.fc = nn.Identity()  # Remove the final fully connected layer

    def forward(self, x):
        return self.resnet(x)

# Define the Barlow Twins model
class BarlowTwins(nn.Module):
    def __init__(self, base_encoder, projection_dim):
        super(BarlowTwins, self).__init__()
        self.encoder = base_encoder
        self.projection_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )

    def forward(self, x1, x2):
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        z1 = self.projection_head(h1)
        z2 = self.projection_head(h2)
        return z1, z2

    def loss(self, z1, z2):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        c = torch.matmul(z1.T, z2) / z1.size(0)
        on_diag = torch.diagonal(c).add_(-1).pow(2).sum()
        off_diag = (c**2).sum() - torch.diagonal(c).pow(2).sum()
        return on_diag + off_diag

# Define the Custom Dataset with Data Augmentation
class CustomAugmentedDataset(Dataset):
    def __init__(self, images, transform=None, augmentations=None):
        self.images = images
        self.transform = transform
        self.augmentations = augmentations

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].numpy()  # Convert tensor to numpy array
        image = image.transpose(1, 2, 0)  # Change from (C, H, W) to (H, W, C)

        # Ensure the array is of type uint8
        image = (image * 255).astype(np.uint8)

        # Handle single channel images by converting them to 3 channels
        if image.shape[2] == 1:
            image = np.concatenate([image] * 3, axis=2)

        image = Image.fromarray(image)  # Convert to PIL image

        # Apply augmentations
        aug1, aug2 = self.augmentations
        image_1 = aug1(image)
        image_2 = aug2(image)

        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)

        return image_1, image_2

# Define data augmentations
def get_augmentations():
    return [
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
        ]),
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3)
        ])
    ]

def create_dataloader(images_tensor, batch_size=64):
    augmentations = get_augmentations()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = CustomAugmentedDataset(images=images_tensor, transform=transform, augmentations=augmentations)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

unlabeled_loader = create_dataloader(unlabeled_images_tensor)
labeled_loader = create_dataloader(labeled_images_tensor)
base_encoder = ResNet15()
model = BarlowTwins(base_encoder, projection_dim=128)
criterion = nn.MSELoss()  # Barlow Twins loss function
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 10
g_losses = []
d_losses = []

def nums() :
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for images_1, images_2 in unlabeled_loader:
            z1, z2 = model(images_1, images_2)
            loss = model.loss(z1, z2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(unlabeled_loader)
        g_losses.append(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss}")

    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label="Generator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
from torch.utils.data import DataLoader, random_split
import torch
from torch.utils.data import DataLoader, random_split

def create_train_val_loaders(dataset, batch_size=32, val_split=0.2, num_workers=4, shuffle=True):
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    assert train_size + val_size == dataset_size, "Train and validation sizes do not match the dataset size"
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

train_loader  , val_loader  = create_train_val_loaders(labeled_data, batch_size=32, val_split=0.2, num_workers=4, shuffle=True)

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class ResNet15(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNet15, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(64, 2)  # 2 blocks
        self.layer2 = self._make_layer(128, 2, stride=2)  # 2 blocks
        self.layer3 = self._make_layer(256, 2, stride=2)  # 2 blocks
        self.layer4 = self._make_layer(512, 2, stride=2)  # 2 blocks
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

model = ResNet15(num_classes=4)
print(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)  
num_epochs = 10
for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = correct_train / total_train * 100

    # Validation phase
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_loss = running_val_loss / len(val_loader)
    val_accuracy = correct_val / total_val * 100
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.2f}%, '
          f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')