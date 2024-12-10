import torch
from models.model import SSLResNet, nt_xent_loss
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
import torch.nn.functional as F

torch.manual_seed(42)

class ContrastiveTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),  # Wider scale range
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # brightness, contrast, saturation, hue
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),  # Added blur
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)

def train_ssl(model, train_loader, optimizer, epochs=100):
    model.train()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            view1, view2 = data
            view1, view2 = view1.to(device), view2.to(device)
            
            optimizer.zero_grad()
            
            z1, p1 = model(view1)
            z2, p2 = model(view2)
            
            z1 = F.normalize(z1, dim=1)
            z2 = F.normalize(z2, dim=1)
            p1 = F.normalize(p1, dim=1)
            p2 = F.normalize(p2, dim=1)
            
            loss = 0.5 * (
                nt_xent_loss(p1, z2.detach()) +  # prediction1 should match target2
                nt_xent_loss(p2, z1.detach())    # prediction2 should match target1
            )
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                print(f'Epoch: {epoch+1}/{epochs} [{batch_idx * len(view1)}/{len(train_loader.dataset)} '
                      f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}')

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current Learning Rate: {current_lr:.6f}')
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'best_ssl_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch + 1} epochs')
                break
    
    checkpoint = torch.load('best_ssl_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

if __name__ == '__main__':
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=ContrastiveTransform())
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)

    ssl_model = SSLResNet().to('cuda')
    optimizer = torch.optim.AdamW(ssl_model.parameters(), lr=1e-3, weight_decay=0.05)
    
    ssl_model = train_ssl(ssl_model, train_loader, optimizer, epochs=200)
    
    torch.save(ssl_model.state_dict(), 'ssl_model_final.pth')