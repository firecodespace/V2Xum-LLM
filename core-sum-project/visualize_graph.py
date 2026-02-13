#!/usr/bin/env python3
"""
Visualize Frame Coherence Graph
"""

import sys
sys.path.append('/workspace/core-sum-project/src')

import cv2
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Rectangle

def find_video():
    """Find the uploaded video"""
    video_path = "/workspace/core-sum-project/data/Big_Take_2024_-_16mm_Short_Film_1080P.mp4"
    if os.path.exists(video_path):
        return video_path
    return None

def load_frames(video_path, max_frames=600):
    """Load video frames"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_idx = 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_rate = max(1, total_frames // max_frames)
    
    print(f"    Loading {max_frames} frames from {total_frames} total...")
    
    with tqdm(total=max_frames, desc="    ") as pbar:
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_rate == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                pbar.update(1)
            frame_idx += 1
    
    cap.release()
    return frames

def visualize_graph_structure(G, selected_frames, output_path):
    """Visualize the coherence graph structure"""
    
    print(f"\n    Creating graph visualization...")
    
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Use spring layout for positioning
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Get edge weights
    edges = G.edges()
    weights = [G[u][v]['coherence'] for u, v in edges]
    
    # Draw edges with thickness based on coherence
    max_weight = max(weights) if weights else 1
    min_weight = min(weights) if weights else 0
    
    for (u, v), weight in zip(edges, weights):
        # Normalize weight for visualization
        normalized = (weight - min_weight) / (max_weight - min_weight + 1e-6)
        
        # Color: green for high coherence, red for low
        if weight > 0.8:
            color = 'green'
            alpha = 0.6
        elif weight > 0.5:
            color = 'orange'
            alpha = 0.4
        else:
            color = 'red'
            alpha = 0.3
        
        # Draw edge
        nx.draw_networkx_edges(
            G, pos, [(u, v)],
            width=normalized * 3,
            alpha=alpha,
            edge_color=color,
            ax=ax
        )
    
    # Draw nodes
    node_colors = []
    for node in G.nodes():
        frame_idx = selected_frames[node]
        # Color by temporal position
        color_val = frame_idx / max(selected_frames)
        node_colors.append(plt.cm.viridis(color_val))
    
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=500,
        alpha=0.9,
        ax=ax
    )
    
    # Draw labels with frame numbers
    labels = {i: f"{selected_frames[i]}" for i in G.nodes()}
    nx.draw_networkx_labels(
        G, pos, labels,
        font_size=8,
        font_weight='bold',
        ax=ax
    )
    
    ax.set_title("Frame Coherence Graph\n(Node color = temporal position, Edge color = coherence)", 
                 fontsize=14, fontweight='bold')
    ax.axis('off')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', lw=3, label='High coherence (>0.8)'),
        Line2D([0], [0], color='orange', lw=3, label='Medium coherence (0.5-0.8)'),
        Line2D([0], [0], color='red', lw=3, label='Low coherence (<0.5)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Graph saved: {output_path}")

def visualize_coherence_matrix(G, selected_frames, output_path):
    """Visualize coherence as a heatmap"""
    
    print(f"    Creating coherence matrix...")
    
    n = len(selected_frames)
    coherence_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if G.has_edge(i, j):
                coherence_matrix[i, j] = G[i][j]['coherence']
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(coherence_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Set ticks
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"{selected_frames[i]}" for i in range(n)], rotation=45, ha='right')
    ax.set_yticklabels([f"{selected_frames[i]}" for i in range(n)])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Coherence Score', rotation=270, labelpad=20)
    
    ax.set_title("Frame Coherence Matrix\n(Red = Low, Yellow = Medium, Green = High)", 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Frame Index")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Matrix saved: {output_path}")

def visualize_semantic_clusters(G, selected_frames, metrics, output_path):
    """Visualize semantic clusters"""
    
    print(f"    Creating cluster visualization...")
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Draw timeline
    max_frame = max(selected_frames)
    ax.set_xlim(0, max_frame)
    ax.set_ylim(0, len(metrics.semantic_clusters) + 1)
    
    # Color palette for clusters
    colors = plt.cm.tab10(range(len(metrics.semantic_clusters)))
    
    for cluster_idx, cluster in enumerate(metrics.semantic_clusters):
        cluster_frames = sorted([selected_frames[i] for i in cluster])
        
        for frame in cluster_frames:
            # Draw vertical line
            ax.axvline(frame, 
                      ymin=(cluster_idx + 0.2) / (len(metrics.semantic_clusters) + 1),
                      ymax=(cluster_idx + 0.8) / (len(metrics.semantic_clusters) + 1),
                      color=colors[cluster_idx], 
                      linewidth=3, 
                      alpha=0.7)
            
            # Add frame number
            ax.text(frame, cluster_idx + 1, str(frame), 
                   rotation=90, va='bottom', ha='center', fontsize=8)
        
        # Label cluster
        ax.text(-20, cluster_idx + 1, f"Cluster {cluster_idx + 1}\n({len(cluster)} frames)", 
               va='center', ha='right', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor=colors[cluster_idx], alpha=0.3))
    
    ax.set_xlabel("Frame Index", fontsize=12)
    ax.set_ylabel("Semantic Cluster", fontsize=12)
    ax.set_title("Semantic Clusters Timeline\n(Frames grouped by visual/semantic similarity)", 
                 fontsize=14, fontweight='bold')
    ax.set_yticks(range(1, len(metrics.semantic_clusters) + 1))
    ax.set_yticklabels([f"Cluster {i+1}" for i in range(len(metrics.semantic_clusters))])
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Clusters saved: {output_path}")

def create_keyframe_summary(frames, selected_frames, output_path):
    """Create visual summary of selected keyframes"""
    
    print(f"    Creating keyframe summary...")
    
    n_frames = len(selected_frames)
    cols = 5
    rows = (n_frames + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    axes = axes.flatten() if n_frames > 1 else [axes]
    
    for idx, frame_idx in enumerate(selected_frames):
        if frame_idx < len(frames):
            axes[idx].imshow(frames[frame_idx])
            axes[idx].set_title(f"Frame {frame_idx}", fontweight='bold')
            axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(n_frames, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f"Selected Keyframes ({len(selected_frames)} frames)", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Summary saved: {output_path}")

def main():
    print("="*60)
    print("VISUALIZING FRAME COHERENCE GRAPH")
    print("="*60)
    
    # Find video
    video_path = find_video()
    if not video_path:
        print("✗ Video not found")
        return
    
    print(f"\n[1/4] Loading video...")
    frames = load_frames(video_path)
    print(f"    ✓ Loaded {len(frames)} frames")
    
    # Run Stage 1
    print(f"\n[2/4] Running Stage 1...")
    from hierarchical_visual_pyramid import HierarchicalVisualPyramid
    hvp = HierarchicalVisualPyramid(device="cuda")
    hierarchy = hvp.encode(frames)
    print(f"    ✓ Hierarchy built")
    
    # Run Stage 2 + 3 and capture graph
    print(f"\n[3/4] Running Stage 2+3...")
    from cot_engine import CoTEngine
    from frame_coherence_graph import FrameCoherenceGraph
    
    cot_engine = CoTEngine(use_graph_refinement=True)
    result = cot_engine.summarize(hierarchy, target_frames=20)
    
    # Rebuild graph for visualization
    print(f"\n[4/4] Creating visualizations...")
    graph_engine = FrameCoherenceGraph()
    
    llm_frames = result.coherence_check['final_frame_sequence']
    G = graph_engine.build_graph(llm_frames, hierarchy.frames, hierarchy.scenes)
    metrics = result.graph_metrics
    
    # Create output directory
    output_dir = "/workspace/core-sum-project/output/graph_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate visualizations
    visualize_graph_structure(G, llm_frames, f"{output_dir}/coherence_graph.png")
    visualize_coherence_matrix(G, llm_frames, f"{output_dir}/coherence_matrix.png")
    visualize_semantic_clusters(G, llm_frames, metrics, f"{output_dir}/semantic_clusters.png")
    create_keyframe_summary(frames, result.selected_frames, f"{output_dir}/keyframe_summary.png")
    
    print("\n" + "="*60)
    print("✓ VISUALIZATIONS COMPLETE")
    print("="*60)
    print(f"\nSaved to: {output_dir}/")
    print(f"  - coherence_graph.png (network visualization)")
    print(f"  - coherence_matrix.png (heatmap)")
    print(f"  - semantic_clusters.png (timeline)")
    print(f"  - keyframe_summary.png (selected frames)")
    print("="*60)

if __name__ == "__main__":
    main()

