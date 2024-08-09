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
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
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

num_epochs = 100
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

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from PIL import Image

# Function to convert grayscale to RGB
def grayscale_to_rgb(grayscale_image):
    return np.stack([grayscale_image] * 3, axis=-1)

# Sample labeled data
labeled_data = {'images': gen_imgs_labeled, 'labels': labels_labeled}
images = labeled_data['images']
labels = labeled_data['labels']

if images.shape[1] == 1:  # Assuming the channel dimension is at index 1
    images = np.array([grayscale_to_rgb(img.squeeze()) for img in images])

# Split the data into training and validation sets
images_np = images
labels_np = labels

images_train, images_val, labels_train, labels_val = train_test_split(
    images_np, labels_np, test_size=0.20, random_state=42
)

# Convert numpy arrays to tensors
images_train = torch.tensor(images_train, dtype=torch.float32)
labels_train = torch.tensor(labels_train, dtype=torch.long)
images_val = torch.tensor(images_val, dtype=torch.float32)
labels_val = torch.tensor(labels_val, dtype=torch.long)

# Ensure images are in the shape [batch_size, 3, height, width]
images_train = images_train.permute(0, 3, 1, 2)
images_val = images_val.permute(0, 3, 1, 2)

train_dataset = TensorDataset(images_train, labels_train)
val_dataset = TensorDataset(images_val, labels_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Load a pre-trained model and replace the final layer
num_classes = len(torch.unique(labels_train))  
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)  
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

"""
# Training loop
num_epochs = 15
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    corrects = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

    val_loss /= len(val_loader.dataset)
    val_acc = corrects.double() / total
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
"""
train_losses = []
val_losses = []
val_accuracies = []
train_accuracies = []

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    corrects_train = 0
    total_train = 0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        
        # Calculate training accuracy
        _, preds = torch.max(outputs, 1)
        corrects_train += torch.sum(preds == labels.data)
        total_train += labels.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    train_accuracy = corrects_train.double() / total_train
    train_losses.append(epoch_loss)
    train_accuracies.append(train_accuracy.item())
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, Training Accuracy: {train_accuracy:.4f}")
    model.eval()
    val_loss = 0.0
    corrects = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

    val_loss /= len(val_loader.dataset)
    val_acc = corrects.double() / total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc.item())
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', color='blue')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy', color='blue')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
