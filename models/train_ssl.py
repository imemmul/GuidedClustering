import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

class SSLResNet(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        
        self.backbone = resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()
        
        self.projection = nn.Sequential(
            nn.Linear(512, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 256)
        )
        
        self.prediction = nn.Sequential(
            nn.Linear(256, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 256)
        )
        
        self.backbone_target = resnet18(pretrained=True)
        self.backbone_target.fc = nn.Identity()
        
        self.projection_target = nn.Sequential(
            nn.Linear(512, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 256)
        )
        
        for param in self.backbone_target.parameters():
            param.requires_grad = False
        for param in self.projection_target.parameters():
            param.requires_grad = False
            
        self._init_target_network()

    def _init_target_network(self):
        for online_params, target_params in zip(self.backbone.parameters(), self.backbone_target.parameters()):
            target_params.data.copy_(online_params.data)
        for online_params, target_params in zip(self.projection.parameters(), self.projection_target.parameters()):
            target_params.data.copy_(online_params.data)

    @torch.no_grad()
    def update_target_network(self, momentum=0.99):
        """Update target network parameters with momentum"""
        for online_params, target_params in zip(self.backbone.parameters(), self.backbone_target.parameters()):
            target_params.data = momentum * target_params.data + (1 - momentum) * online_params.data
        for online_params, target_params in zip(self.projection.parameters(), self.projection_target.parameters()):
            target_params.data = momentum * target_params.data + (1 - momentum) * online_params.data

    def forward(self, x1, x2):
        f1 = self.backbone(x1)
        z1 = self.projection(f1)
        p1 = self.prediction(z1)
        
        with torch.no_grad():
            f2 = self.backbone_target(x2)
            z2 = self.projection_target(f2)
        
        return p1, z2.detach()

    def get_features(self, x):
        return self.backbone(x)

class BYOLTransform:
    def __init__(self):
        self.transform1 = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.1),
            transforms.RandomSolarize(threshold=128, p=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], 
                               [0.2023, 0.1994, 0.2010])
        ])
        
        self.transform2 = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.1),
            transforms.RandomSolarize(threshold=128, p=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], 
                               [0.2023, 0.1994, 0.2010])
        ])
    
    def __call__(self, x):
        return self.transform1(x), self.transform2(x)

def byol_loss_fn(p, z):
    """BYOL loss function"""
    p = F.normalize(p, dim=-1, p=2)
    z = F.normalize(z, dim=-1, p=2)
    return 2 - 2 * (p * z).sum(dim=-1).mean()

def train_ssl(model, train_loader, epochs=1000, patience=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(
        [
            {'params': model.backbone.parameters(), 'lr': 2e-4},
            {'params': model.projection.parameters(), 'lr': 2e-4},
            {'params': model.prediction.parameters(), 'lr': 2e-4}
        ],
        weight_decay=0.1
    )
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[2e-4, 2e-4, 2e-4],
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.1
    )
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            view1, view2 = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            
            p1, z2 = model(view1, view2)  # Online branch predicts target branch
            p2, z1 = model(view2, view1)  # Symmetric loss term
            
            loss = 0.5 * (byol_loss_fn(p1, z2) + byol_loss_fn(p2, z1))
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            model.update_target_network(momentum=0.99)
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch: {epoch+1}/{epochs} [{batch_idx}/{len(train_loader)}] '
                      f'Loss: {loss.item():.6f} LR: {current_lr:.6f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}')

        if avg_loss < best_loss:
            best_loss = avg_loss

            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
            }, 'best_byol_model.pth')
            print(f'Saved checkpoint at epoch {epoch+1} with loss {best_loss:.4f}')
        else:
            patience_counter += 1
            print(f'Early stopping counter: {patience_counter}/{patience}')

        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break


def evaluate_ssl_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the trained model
    model = SSLResNet().to(device)
    checkpoint = torch.load('./best_byol_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Setup test dataset
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], 
                           [0.2023, 0.1994, 0.2010])
    ])
    
    test_dataset = CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    features = []
    labels = []
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            backbone_features = model.backbone(images)
            z, p = model(images, images)
            
            features.append({
                'backbone': backbone_features.cpu(),
                'projection': z.cpu(),
                'prediction': p.cpu()
            })
            labels.append(targets)
    
    # Concatenate all features
    all_features = {
        k: torch.cat([f[k] for f in features], dim=0)
        for k in features[0].keys()
    }
    all_labels = torch.cat(labels, dim=0).numpy()
    
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    
    def knn_evaluate(features, labels, k=20):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(features, labels)
        predictions = knn.predict(features)
        return accuracy_score(labels, predictions)
    
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    
    def cluster_evaluate(features, labels, n_clusters=10):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features)
        
        nmi = normalized_mutual_info_score(labels, cluster_labels)
        ari = adjusted_rand_score(labels, cluster_labels)
        
        return {
            'NMI': nmi,
            'ARI': ari
        }
    
    results = {}
    for feat_type, features in all_features.items():
        features = F.normalize(features, dim=1).numpy()
        
        knn_acc = knn_evaluate(features, all_labels)
        
        cluster_metrics = cluster_evaluate(features, all_labels)
        
        results[feat_type] = {
            'KNN_Accuracy': knn_acc,
            **cluster_metrics
        }
    
    print("\nEvaluation Results:")
    for feat_type, metrics in results.items():
        print(f"\n{feat_type} features:")
        print(f"KNN Accuracy: {metrics['KNN_Accuracy']:.4f}")
        print(f"NMI Score: {metrics['NMI']:.4f}")
        print(f"ARI Score: {metrics['ARI']:.4f}")
    
    return results

if __name__ == '__main__':
    train_dataset = CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=BYOLTransform()
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=512,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    model = SSLResNet()
    train_ssl(model, train_loader, patience=15, epochs=200)