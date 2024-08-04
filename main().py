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


num_generate_samples = 1000
z_labeled = torch.randn(num_generate_samples, 100)
gen_imgs_labeled = generator(z_labeled)
gen_imgs_labeled = gen_imgs_labeled.detach().numpy()

# Create corresponding labels for the generated labeled data
labels_labeled = np.ones((num_generate_samples, 1))

# Generate 1,000 unlabeled samples
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

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        # Convert tensor to PIL Image if needed
        if self.transform:
            image = transforms.ToPILImage()(image)  # Convert tensor to PIL Image for transformation
            image = self.transform(image)
        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        return image

# Define data augmentations
transform = transforms.Compose([
    transforms.Resize((224, 224)),         # Resize to match ResNet input size
    transforms.RandomHorizontalFlip(),     # Data augmentation
    transforms.RandomRotation(10),         # Data augmentation
])

# Create Data Loaders
unlabeled_dataset = CustomDataset(images=unlabeled_images_tensor, transform=transform)
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=64, shuffle=True)

labeled_dataset = CustomDataset(images=labeled_images_tensor, labels=labeled_labels_tensor, transform=transform)
labeled_loader = DataLoader(labeled_dataset, batch_size=64, shuffle=True)

# Define ResNet model
class ResNet15(nn.Module):
    def __init__(self):
        super(ResNet15, self).__init__()
        self.resnet = models.resnet18(pretrained=False)  # Using ResNet-18 as a proxy
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)  # For RGB images
        self.resnet.fc = nn.Identity()  # Remove final classification layer

    def forward(self, x):
        return self.resnet(x)

# Define NT-Xent Loss
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Compute similarity matrix
        sim = torch.mm(z_i, z_j.t()) / self.temperature

        # Create labels
        labels = torch.arange(batch_size).long().to(z_i.device)

        # Compute loss
        loss = F.cross_entropy(sim, labels)
        return loss

# Initialize model, loss function, and optimizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ResNet15().to(device)
criterion = NTXentLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    for images in unlabeled_loader:
        images = images.to(device)

        # Generate two augmented views
        view1 = images
        view2 = images  # Apply augmentations if necessary

        # Forward pass
        z_i = model(view1)
        z_j = model(view2)

        # Compute loss
        loss = criterion(z_i, z_j)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
