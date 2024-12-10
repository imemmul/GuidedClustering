import torch

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
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from models.model import SSLResNet
from scipy.optimize import linear_sum_assignment


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy after using the Hungarian algorithm to
    find the best mapping between predicted and true labels
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    acc = sum([w[i, j] for i, j in zip(*ind)]) * 1.0 / y_pred.size
    return acc

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

def evaluate_clustering(features, true_labels, method='kmeans', n_clusters=10):
    if method == 'kmeans':
        clustering = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10).fit(features)
    elif method == 'dbscan':
        clustering = DBSCAN(eps=0.5, min_samples=5).fit(features)
    elif method == 'agglomerative':
        clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(features)
    else:
        raise ValueError("Unsupported method")
    
    pred_labels = clustering.labels_
    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)
    return ari, nmi, pred_labels

def evaluate_ssl_knn(features_train, labels_train, features_test, labels_test, k=5):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(features_train, labels_train)
    pred_labels = knn.predict(features_test)
    accuracy = accuracy_score(labels_test, pred_labels)
    return accuracy

class EnhancedSSLGuidedClustering:
    def __init__(self, ssl_model, n_clusters=10, momentum=0.9, temperature=0.1):
        self.ssl_model = ssl_model
        self.n_clusters = n_clusters
        self.momentum = momentum
        self.temperature = temperature
        self.device = next(ssl_model.parameters()).device

    def compute_features(self, dataset):
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
        features = []
        
        self.ssl_model.eval()
        with torch.no_grad():
            for data, _ in dataloader:
                if isinstance(data, (list, tuple)):
                    data = data[0]
                data = data.to(self.device)
                feat = self.ssl_model.get_features(data)
                # Add L2 normalization
                feat = F.normalize(feat, dim=1)
                features.append(feat.cpu())
                
        return torch.cat(features, dim=0)

    def find_natural_groups(self, dataset, max_iterations=100, epsilon=1e-5):
        features = self.compute_features(dataset)
        
        best_inertia = float('inf')
        best_centroids = None
        
        for _ in range(5):
            kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', n_init=10)
            kmeans.fit(features.numpy())
            if kmeans.inertia_ < best_inertia:
                best_inertia = kmeans.inertia_
                best_centroids = kmeans.cluster_centers_
        
        centroids = torch.tensor(best_centroids, dtype=torch.float32)
        momentum_centroids = centroids.clone()
        
        adaptive_temp = self.temperature
        
        for iteration in range(max_iterations):
            similarities = F.cosine_similarity(
                features.unsqueeze(1).expand(-1, self.n_clusters, -1),
                centroids.unsqueeze(0).expand(features.size(0), -1, -1),
                dim=2
            ) / adaptive_temp
            
            soft_assignments = F.softmax(similarities, dim=1)
            
            # Update centroids with weighted average
            new_centroids = torch.zeros_like(centroids)
            for k in range(self.n_clusters):
                cluster_weights = soft_assignments[:, k].unsqueeze(1)
                new_centroids[k] = (features * cluster_weights).sum(0) / cluster_weights.sum()
                new_centroids[k] = F.normalize(new_centroids[k].unsqueeze(0)).squeeze()
            
            # Adaptive momentum
            centroid_change = torch.norm(centroids - new_centroids)
            adaptive_momentum = self.momentum * (1 - centroid_change / features.size(1))
            
            momentum_centroids = adaptive_momentum * momentum_centroids + (1 - adaptive_momentum) * new_centroids
            momentum_centroids = F.normalize(momentum_centroids, dim=1)
            
            adaptive_temp = self.temperature * (1 + iteration / max_iterations)
            
            if centroid_change < epsilon:
                break
            
            centroids = momentum_centroids.clone()
        
        final_similarities = F.cosine_similarity(
            features.unsqueeze(1).expand(-1, self.n_clusters, -1),
            centroids.unsqueeze(0).expand(features.size(0), -1, -1),
            dim=2
        ) / self.temperature
        
        soft_assignments = F.softmax(final_similarities, dim=1)
        cluster_assignments = torch.argmax(soft_assignments, dim=1)
        confidence_scores = torch.max(soft_assignments, dim=1)[0]
        
        groups = [[] for _ in range(self.n_clusters)]
        for idx, cluster_idx in enumerate(cluster_assignments):
            groups[cluster_idx.item()].append(idx)
            
        return groups, confidence_scores

if __name__ == '__main__':
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    ssl_model = SSLResNet().to('cuda')
    ssl_model.load_state_dict(torch.load('best_ssl_model.pth')['model_state_dict'])
    
    train_features, train_labels = extract_features_ssl(ssl_model, train_loader)
    test_features, test_labels = extract_features_ssl(ssl_model, test_loader)
    
    clustering = EnhancedSSLGuidedClustering(ssl_model)
    groups, confidence_scores = clustering.find_natural_groups(train_dataset)
    pred_labels = torch.zeros(len(train_dataset), dtype=torch.long)
    for i, group in enumerate(groups):
        for idx in group:
            pred_labels[idx] = i
    pred_labels = pred_labels.cpu().numpy()
    ari_ssl, nmi_ssl = adjusted_rand_score(train_labels, pred_labels), normalized_mutual_info_score(train_labels, pred_labels)
    acc_ssl = cluster_acc(train_labels, pred_labels)
    print(f"SSL-Guided Clustering - ARI: {ari_ssl:.4f}, NMI: {nmi_ssl:.4f}, ACC: {acc_ssl:.4f}")
    
    ari_kmeans, nmi_kmeans, _ = evaluate_clustering(train_features, train_labels, method='kmeans')
    print(f"Vanilla k-Means - ARI: {ari_kmeans:.4f}, NMI: {nmi_kmeans:.4f}")
    
    ari_dbscan, nmi_dbscan, _ = evaluate_clustering(train_features, train_labels, method='dbscan')
    print(f"DBSCAN - ARI: {ari_dbscan:.4f}, NMI: {nmi_dbscan:.4f}")
    
    ari_agg, nmi_agg, _ = evaluate_clustering(train_features, train_labels, method='agglomerative')
    print(f"Agglomerative Clustering - ARI: {ari_agg:.4f}, NMI: {nmi_agg:.4f}")
    
    accuracy_knn = evaluate_ssl_knn(train_features, train_labels, test_features, test_labels)
    print(f"Direct SSL k-NN Classification Accuracy: {accuracy_knn:.4f}")
    