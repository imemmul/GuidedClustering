import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, accuracy_score
import numpy as np
from kneed import KneeLocator
from sklearn.neighbors import NearestNeighbors


class SSLResNet(nn.Module):
    def __init__(self, output_dim=256):
        super().__init__()
        
        # Online network
        self.backbone = resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()
        
        # Projection MLP
        self.projection = nn.Sequential(
            nn.Linear(512, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 256)
        )
        
        # Prediction MLP
        self.prediction = nn.Sequential(
            nn.Linear(256, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 256)
        )
        
        # Target network (momentum-updated)
        self.backbone_target = resnet18(pretrained=True)
        self.backbone_target.fc = nn.Identity()
        
        self.projection_target = nn.Sequential(
            nn.Linear(512, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 256)
        )
        
        # Disable gradients for target network
        for param in self.backbone_target.parameters():
            param.requires_grad = False
        for param in self.projection_target.parameters():
            param.requires_grad = False
            
        # Initialize target network with online network params
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
        # Online network forward
        f1 = self.backbone(x1)
        z1 = self.projection(f1)
        p1 = self.prediction(z1)
        
        # Target network forward
        with torch.no_grad():
            f2 = self.backbone_target(x2)
            z2 = self.projection_target(f2)
        
        return p1, z2.detach()

    def get_features(self, x):
        return self.backbone(x)

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy
    """
    from scipy.optimize import linear_sum_assignment
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() * 1.0 / y_pred.size

def extract_features_ssl(model, data_loader):
    model.eval()
    features = []
    labels = []
    device = next(model.parameters()).device
    
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            output = model.get_features(data)
            features.append(output.cpu().numpy())
            labels.append(target.numpy())
            
    return np.concatenate(features), np.concatenate(labels)

def evaluate_clustering(true_labels, pred_labels, name=""):
    """Compute multiple clustering metrics"""
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    acc = cluster_acc(true_labels, pred_labels)
    
    print(f"\n{name} Results:")
    print(f"NMI: {nmi:.4f}")
    print(f"ARI: {ari:.4f}")
    print(f"ACC: {acc:.4f}")
    
    return nmi, ari, acc

class SPECS:
    """
    Spectral embeddings clustering with self-supervision
    """
    def __init__(self, ssl_model, n_clusters=10):
        self.ssl_model = ssl_model
        self.n_clusters = n_clusters
        self.device = next(ssl_model.parameters()).device
        
    def extract_embeddings(self, data):
        """Get embeddings from both backbone and projection head"""
        self.ssl_model.eval()
        with torch.no_grad():
            backbone_features = self.ssl_model.get_features(data)
            proj_features, _ = self.ssl_model(data, data)
            
            backbone_features = F.normalize(backbone_features, dim=1)
            proj_features = F.normalize(proj_features, dim=1)
            
            return backbone_features, proj_features
    
    def find_nearest_neighbors(self, features, k=10):
        """Find k-nearest neighbors for each sample"""
        sim_matrix = torch.mm(features, features.t())
        sim_matrix = torch.exp(sim_matrix / 0.1)  # Temperature scaling
        _, indices = torch.topk(sim_matrix, k=k, dim=1)
        return indices, sim_matrix
    
    def build_similarity_graph(self, backbone_features, proj_features):
        """Build similarity graph using both feature spaces"""
        backbone_nn, backbone_sim = self.find_nearest_neighbors(backbone_features, k=10)
        proj_nn, proj_sim = self.find_nearest_neighbors(proj_features, k=10)
        
        n_samples = backbone_features.size(0)
        similarity_matrix = torch.zeros((n_samples, n_samples), device=self.device)
        
        for i in range(n_samples):
            neighbors = torch.unique(torch.cat([backbone_nn[i], proj_nn[i]]))
            
            b_sim = backbone_sim[i, neighbors]
            p_sim = proj_sim[i, neighbors]
            
            combined_sim = 0.7 * b_sim + 0.3 * p_sim
            similarity_matrix[i, neighbors] = combined_sim
        
        similarity_matrix = 0.5 * (similarity_matrix + similarity_matrix.t())
        
        # Handle numerical stability
        similarity_matrix = similarity_matrix + torch.eye(n_samples, device=self.device) * 1e-5
        
        return F.normalize(similarity_matrix, p=1, dim=1)
    
    def spectral_clustering(self, similarity_matrix):
        """Perform spectral clustering with stability checks"""
        # Compute normalized Laplacian
        degree = similarity_matrix.sum(dim=1)
        laplacian = torch.diag(degree) - similarity_matrix
        normalized_laplacian = torch.diag(1.0 / torch.sqrt(degree + 1e-10)) @ laplacian @ torch.diag(1.0 / torch.sqrt(degree + 1e-10))
        
        eigenvalues, eigenvectors = torch.linalg.eigh(normalized_laplacian)
        
        # Use top-k eigenvectors
        spectral_embeddings = eigenvectors[:, :self.n_clusters]
        
        spectral_embeddings = torch.nan_to_num(spectral_embeddings, nan=0.0)
        
        norms = torch.norm(spectral_embeddings, p=2, dim=1, keepdim=True)
        spectral_embeddings = spectral_embeddings / (norms + 1e-10)
        
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=20)
        labels = kmeans.fit_predict(spectral_embeddings.cpu().numpy())
        
        return torch.tensor(labels, device=self.device)
    
    def compute_confidence(self, backbone_features, assignments):
        """Compute confidence scores"""
        confidence_scores = torch.zeros(len(assignments), device=self.device)
        centers = torch.zeros((self.n_clusters, backbone_features.size(1)), device=self.device)
        
        for k in range(self.n_clusters):
            mask = (assignments == k)
            if mask.sum() > 0:
                centers[k] = backbone_features[mask].mean(0)
                centers[k] = F.normalize(centers[k].unsqueeze(0), dim=0).squeeze()
        
        all_sims = torch.mm(backbone_features, centers.t())
        
        assigned_sims = all_sims[torch.arange(len(assignments)), assignments]
        
        min_conf, max_conf = assigned_sims.min(), assigned_sims.max()
        confidence_scores = 0.5 + 0.35 * (assigned_sims - min_conf) / (max_conf - min_conf + 1e-10)
        
        return confidence_scores
    
    def cluster(self, dataset):
        """Main clustering method"""
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=256, shuffle=False, num_workers=4
        )
        
        all_backbone, all_proj = [], []
        for data, _ in dataloader:
            if isinstance(data, (list, tuple)):
                data = data[0]
            data = data.to(self.device)
            backbone_feat, proj_feat = self.extract_embeddings(data)
            all_backbone.append(backbone_feat)
            all_proj.append(proj_feat)
        
        backbone_features = torch.cat(all_backbone, dim=0)
        proj_features = torch.cat(all_proj, dim=0)
        
        similarity_matrix = self.build_similarity_graph(backbone_features, proj_features)
        assignments = self.spectral_clustering(similarity_matrix)
        confidence_scores = self.compute_confidence(backbone_features, assignments)
        
        return assignments, confidence_scores
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ssl_model = SSLResNet().to(device)
    ssl_model.load_state_dict(torch.load('./best_model.pth')['model_state_dict'])
    ssl_model.eval()

    # Prepare test dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], 
                           [0.2023, 0.1994, 0.2010])
    ])
    
    dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False,
        download=True, 
        transform=transform
    )
    true_labels = np.array(dataset.targets)
    clustering = SPECS(ssl_model, n_clusters=10)
    assignments, confidence_scores = clustering.cluster(dataset)
    ssl_spectral_labels = assignments.cpu().numpy()
    evaluate_clustering(true_labels, ssl_spectral_labels, "SSL-based Spectral Clustering")
    
    print(f"\nConfidence Scores:")
    print(f"Mean: {confidence_scores.mean().item():.4f}")
    print(f"Std: {confidence_scores.std().item():.4f}")
    
    # 2. K-means on SSL backbone features
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
    backbone_features = []
    with torch.no_grad():
        for data, _ in dataloader:
            if isinstance(data, (list, tuple)):
                data = data[0]
            data = data.to(device)
            features = ssl_model.get_features(data)
            backbone_features.append(features.cpu())
    
    backbone_features = torch.cat(backbone_features, dim=0).numpy()
    kmeans_ssl = KMeans(n_clusters=10, random_state=42)
    kmeans_ssl_labels = kmeans_ssl.fit_predict(backbone_features)
    evaluate_clustering(true_labels, kmeans_ssl_labels, "K-means on SSL Backbone Features")
    
    # 3. K-means on SSL projection features
    proj_features = []
    with torch.no_grad():
        for data, _ in dataloader:
            if isinstance(data, (list, tuple)):
                data = data[0]
            data = data.to(device)
            features, _ = ssl_model(data, data)
            proj_features.append(features.cpu())
    
    proj_features = torch.cat(proj_features, dim=0).numpy()
    kmeans_proj = KMeans(n_clusters=10, random_state=42)
    kmeans_proj_labels = kmeans_proj.fit_predict(proj_features)
    evaluate_clustering(true_labels, kmeans_proj_labels, "K-means on SSL Projection Features")
    
    # 4. K-means on raw data
    raw_data = []
    for data, _ in dataloader:
        if isinstance(data, (list, tuple)):
            data = data[0]
        raw_data.append(data.reshape(data.size(0), -1).cpu().numpy())
    
    raw_data = np.concatenate(raw_data, axis=0)
    kmeans_raw = KMeans(n_clusters=10, random_state=42)
    kmeans_raw_labels = kmeans_raw.fit_predict(raw_data)
    evaluate_clustering(true_labels, kmeans_raw_labels, "K-means on Raw Data")
    
    # 6. Agglomerative Clustering on SSL features
    agg = AgglomerativeClustering(n_clusters=10)
    agg_labels = agg.fit_predict(backbone_features)
    evaluate_clustering(true_labels, agg_labels, "Agglomerative Clustering on SSL Features")

    from utils.visualize import visualize_results
    visualize_results(ssl_model, dataset, true_labels, ssl_spectral_labels, kmeans_ssl_labels, confidence_scores)

if __name__ == "__main__":
    main()
    