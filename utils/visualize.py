import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
import torch

def create_paper_visualizations(backbone_features, proj_features, true_labels, 
                              specs_labels, kmeans_labels, confidence_scores,
                              save_dir='./figures/'):
    """
    Create comprehensive visualizations for the paper
    """
    # Set style for paper-quality plots
    plt.style.use('seaborn-v0_8-paper')
    sns.set_context("paper", font_scale=1.5)
    
    # 1. Feature Space Visualization with t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(backbone_features)
    
    fig = plt.figure(figsize=(15, 5))
    
    # Ground Truth
    plt.subplot(131)
    plt.scatter(features_2d[:, 0], features_2d[:, 1], c=true_labels, 
               cmap='tab10', alpha=0.6, s=20)
    plt.title('Ground Truth')
    plt.xticks([])
    plt.yticks([])
    
    # SPECS Clustering
    plt.subplot(132)
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                         c=specs_labels, cmap='tab10', 
                         alpha=0.6, s=20)
    plt.title('SPECS Clustering')
    plt.xticks([])
    plt.yticks([])
    
    # K-means Clustering
    plt.subplot(133)
    plt.scatter(features_2d[:, 0], features_2d[:, 1], 
               c=kmeans_labels, cmap='tab10', 
               alpha=0.6, s=20)
    plt.title('K-means Clustering')
    plt.xticks([])
    plt.yticks([])
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}clustering_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 2. Confidence Score Analysis
    plt.figure(figsize=(10, 5))
    
    # Confidence Distribution
    plt.subplot(121)
    sns.kdeplot(confidence_scores, fill=True)
    plt.title('Confidence Score Distribution')
    plt.xlabel('Confidence Score')
    plt.ylabel('Density')
    
    # Confidence vs Correctness
    correct_assignments = specs_labels == true_labels
    plt.subplot(122)
    sns.boxplot(x=correct_assignments, y=confidence_scores)
    plt.title('Confidence vs. Correctness')
    plt.xlabel('Correct Assignment')
    plt.ylabel('Confidence Score')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}confidence_analysis.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 3. Feature Space Quality
    plt.figure(figsize=(12, 4))
    
    # Backbone Features
    plt.subplot(121)
    sim_matrix_backbone = backbone_features @ backbone_features.T
    plt.imshow(sim_matrix_backbone, cmap='viridis')
    plt.title('Backbone Feature\nSimilarity Matrix')
    plt.colorbar()
    
    # Projection Features
    plt.subplot(122)
    sim_matrix_proj = proj_features @ proj_features.T
    plt.imshow(sim_matrix_proj, cmap='viridis')
    plt.title('Projection Feature\nSimilarity Matrix')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}feature_quality.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # 4. Cluster Coherence Analysis
    plt.figure(figsize=(15, 5))
    
    # Within-cluster similarity for SPECS
    within_sim_specs = []
    for i in range(10):  # number of clusters
        mask = specs_labels == i
        if np.sum(mask) > 1:
            cluster_features = backbone_features[mask]
            within_sim = np.mean(cluster_features @ cluster_features.T)
            within_sim_specs.append(within_sim)
    
    # Within-cluster similarity for K-means
    within_sim_kmeans = []
    for i in range(10):
        mask = kmeans_labels == i
        if np.sum(mask) > 1:
            cluster_features = backbone_features[mask]
            within_sim = np.mean(cluster_features @ cluster_features.T)
            within_sim_kmeans.append(within_sim)
    
    plt.subplot(131)
    plt.bar(range(len(within_sim_specs)), within_sim_specs)
    plt.title('SPECS Cluster\nCoherence')
    plt.xlabel('Cluster ID')
    plt.ylabel('Average Similarity')
    
    plt.subplot(132)
    plt.bar(range(len(within_sim_kmeans)), within_sim_kmeans)
    plt.title('K-means Cluster\nCoherence')
    plt.xlabel('Cluster ID')
    plt.ylabel('Average Similarity')
    
    # Confidence distribution per cluster
    plt.subplot(133)
    sns.boxplot(x=specs_labels, y=confidence_scores)
    plt.title('Confidence Score\nper Cluster')
    plt.xlabel('Cluster ID')
    plt.ylabel('Confidence Score')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}cluster_analysis.pdf', bbox_inches='tight', dpi=300)
    plt.close()

def visualize_results(model, dataset, true_labels, specs_labels, kmeans_labels, confidence_scores):
    device = next(model.parameters()).device
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
    
    backbone_features, proj_features = [], []
    with torch.no_grad():
        for data, _ in dataloader:
            if isinstance(data, (list, tuple)):
                data = data[0]
            data = data.to(device)
            backbone_feat, proj_feat = model.get_features(data), model(data, data)[0]
            backbone_features.append(backbone_feat.cpu().numpy())
            proj_features.append(proj_feat.cpu().numpy())
    
    backbone_features = np.concatenate(backbone_features, axis=0)
    proj_features = np.concatenate(proj_features, axis=0)
    
    create_paper_visualizations(
        backbone_features=backbone_features,
        proj_features=proj_features,
        true_labels=true_labels,
        specs_labels=specs_labels,
        kmeans_labels=kmeans_labels,
        confidence_scores=confidence_scores.cpu().numpy(),
        save_dir='./figures/'
    )