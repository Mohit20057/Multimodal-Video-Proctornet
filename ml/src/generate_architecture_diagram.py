"""
Generates publication-quality Neural Network Architecture Diagrams
styled identically to academic node-and-edge graphs (TikZ style).
Outputs both high-res PNG and vector SVG to ml/models/architecture_diagram.png/.svg
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def draw_neural_net(
    output_png="ml/models/architecture_diagram.png",
    output_svg="ml/models/architecture_diagram.svg"
):
    os.makedirs(os.path.dirname(output_png), exist_ok=True)

    # Canvas setup
    fig = plt.figure(figsize=(15, 9.5), dpi=300)
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')

    # Color palette matching classic academic neural net diagrams:
    # Inputs: soft green, Hidden: soft lavender/blue, Outputs: soft coral/red
    c_input_face = '#D4F0D0'
    c_input_edge = '#2E7D32'

    c_hidden_face = '#D7DCFA'
    c_hidden_edge = '#3949AB'

    c_output_face = '#FFCDD2'
    c_output_edge = '#C62828'

    c_line = '#5C6BC0'  # Classic slate-blue connection lines

    layer_sizes = [5, 6, 6, 6, 4]
    n_layers = len(layer_sizes)
    v_spacing = 1.15
    h_spacing = 2.5

    # Center nodes vertically around y = 0
    layer_positions = []
    for l_idx, n_nodes in enumerate(layer_sizes):
        x = l_idx * h_spacing
        positions = []
        for n_idx in range(n_nodes):
            # Center symmetrically around y = 0
            y = ((n_nodes - 1) / 2.0 - n_idx) * v_spacing
            positions.append((x, y))
        layer_positions.append(positions)

    # Node labels matching reference image
    node_labels = {
        0: [r"$x_1$", r"$x_2$", r"$x_3$", "...", r"$x_n$"],
        1: [r"$a_1^{(1)}$", r"$a_2^{(1)}$", r"$a_3^{(1)}$", r"$a_4^{(1)}$", "...", r"$a_m^{(1)}$"],
        2: [r"$a_1^{(2)}$", r"$a_2^{(2)}$", r"$a_3^{(2)}$", r"$a_4^{(2)}$", "...", r"$a_m^{(2)}$"],
        3: [r"$a_1^{(3)}$", r"$a_2^{(3)}$", r"$a_3^{(3)}$", r"$a_4^{(3)}$", "...", r"$a_m^{(3)}$"],
        4: [r"$y_0$", r"$y_1$", r"$y_2$", r"$y_3$"]
    }

    # Draw connecting lines (synapses)
    for l_idx in range(n_layers - 1):
        curr_positions = layer_positions[l_idx]
        next_positions = layer_positions[l_idx + 1]
        curr_labels = node_labels[l_idx]
        next_labels = node_labels[l_idx + 1]

        for i, (x1, y1) in enumerate(curr_positions):
            for j, (x2, y2) in enumerate(next_positions):
                # Don't draw dense lines directly through ellipsis dots
                if curr_labels[i] == "..." or next_labels[j] == "...":
                    continue
                line = plt.Line2D(
                    [x1, x2], [y1, y2],
                    color=c_line,
                    alpha=0.28,
                    linewidth=1.1,
                    zorder=1
                )
                ax.add_line(line)

    # Draw nodes and node text
    node_radius = 0.32

    for l_idx, positions in enumerate(layer_positions):
        if l_idx == 0:
            face_color = c_input_face
            edge_color = c_input_edge
            text_color = '#1B5E20'
        elif l_idx == n_layers - 1:
            face_color = c_output_face
            edge_color = c_output_edge
            text_color = '#B71C1C'
        else:
            face_color = c_hidden_face
            edge_color = c_hidden_edge
            text_color = '#1A237E'

        labels = node_labels[l_idx]

        for n_idx, (x, y) in enumerate(positions):
            lbl = labels[n_idx]
            if lbl == "...":
                ax.text(
                    x, y, "⋮",
                    fontsize=24,
                    ha='center',
                    va='center',
                    color='#455A64',
                    fontweight='bold',
                    zorder=3
                )
            else:
                circle = patches.Circle(
                    (x, y),
                    radius=node_radius,
                    facecolor=face_color,
                    edgecolor=edge_color,
                    linewidth=1.8,
                    zorder=2
                )
                ax.add_patch(circle)

                ax.text(
                    x, y, lbl,
                    fontsize=13.5,
                    ha='center',
                    va='center',
                    color=text_color,
                    fontweight='medium',
                    zorder=3
                )

    # --- TOP HEADERS (Matching Reference Diagram) ---
    max_top_y = 3.6

    # 1. Input Layer Header
    ax.text(
        0 * h_spacing, max_top_y,
        "input\nlayer",
        fontsize=14,
        ha='center',
        va='bottom',
        color='#2E7D32',
        fontweight='semibold'
    )

    # 2. Hidden Layers Super-Header
    hidden_x_start = 1 * h_spacing
    hidden_x_end = 3 * h_spacing
    hidden_x_mid = (hidden_x_start + hidden_x_end) / 2.0

    ax.text(
        hidden_x_mid, max_top_y + 0.5,
        "hidden layers",
        fontsize=16,
        ha='center',
        va='bottom',
        color='#283593',
        fontweight='semibold'
    )

    # Decorative bracket line spanning hidden layers
    bracket_y = max_top_y + 0.35
    ax.plot(
        [hidden_x_start - 0.4, hidden_x_end + 0.4],
        [bracket_y, bracket_y],
        color='#3949AB',
        linewidth=1.5,
        zorder=2
    )

    # Hidden layer sub-labels (Layer 1, Layer 2, Layer 3)
    ax.text(1 * h_spacing, max_top_y, "layer 1", fontsize=12, ha='center', va='bottom', color='#3949AB')
    ax.text(2 * h_spacing, max_top_y, "layer 2", fontsize=12, ha='center', va='bottom', color='#3949AB')
    ax.text(3 * h_spacing, max_top_y, "layer 3", fontsize=12, ha='center', va='bottom', color='#3949AB')

    # 3. Output Layer Header
    ax.text(
        4 * h_spacing, max_top_y,
        "output\nlayer",
        fontsize=14,
        ha='center',
        va='bottom',
        color='#C62828',
        fontweight='semibold'
    )

    # --- BOTTOM TECHNICAL MAPPINGS FOR OUR MULTIMODAL MODEL ---
    bot_y = -3.5
    sub_col_annotations = [
        ("Input Features", r"$\mathbf{X}_{\text{vid}} \in \mathbb{R}^{B \times 16 \times 3 \times 112^2}$" + "\n" + r"$\mathbf{X}_{\text{aud}} \in \mathbb{R}^{B \times 1 \times 64 \times 300}$", '#2E7D32'),
        ("Video & Audio Convs", "R(2+1)D ResNet-18\n+ Audio 2D-CNN\n" + r"$\mathbf{a}^{(1)} \in \mathbb{R}^{384}$", '#3949AB'),
        ("Cross-Attention", "6x Bidirectional\nTransformer Blocks\n" + r"$\mathbf{a}^{(2)} \in \mathbb{R}^{384}$", '#3949AB'),
        ("Aggregation & Projection", "Dual Temporal Pooling\nAvgPool $\\oplus$ MaxPool\n" + r"$\mathbf{a}^{(3)} \in \mathbb{R}^{256}$", '#3949AB'),
        ("Classifier Head", "4 Class Logits\n" + r"$\mathbf{y} \in \{0, 1, 2, 3\}$" + "\n(Softmax)", '#C62828')
    ]

    for idx, (title, details, col) in enumerate(sub_col_annotations):
        x = idx * h_spacing
        box_text = f"{title}\n{details}"
        ax.text(
            x, bot_y,
            box_text,
            fontsize=10.5,
            ha='center',
            va='top',
            color=col,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FAFAFA', edgecolor='#E0E0E0', linewidth=1.0)
        )

    # Class Legend Card on the right
    legend_x = 4 * h_spacing + 0.7
    legend_text = (
        r"$\bf{Output\ Classes:}$" "\n"
        r"$y_0$: Normal Coding (Focused)" "\n"
        r"$y_1$: 2nd Monitor / Notes" "\n"
        r"$y_2$: Voice Assistance" "\n"
        r"$y_3$: Multi-Face / Absence"
    )
    ax.text(
        legend_x, 0,
        legend_text,
        fontsize=11.5,
        ha='left',
        va='center',
        color='#991B1B',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='#FFF5F5', edgecolor='#FECACA', linewidth=1.5)
    )

    # Title Banner at the very bottom
    footer_text = (
        r"$\bf{Model:}$ MultimodalVideoProctorNet (36 Hidden Layers, 33,946,917 Parameters)  |  "
        r"$\bf{Evaluation:}$ 100% Test Macro-F1 (RTX 4050 GPU)  |  "
        r"$\bf{Runtime:}$ ONNX (150.5ms / 16-frame window)"
    )
    ax.text(
        (n_layers - 1) * h_spacing / 2.0, bot_y - 1.6,
        footer_text,
        fontsize=11,
        ha='center',
        va='center',
        color='#37474F',
        bbox=dict(boxstyle='square,pad=0.5', facecolor='#ECEFF1', edgecolor='#B0BEC5', linewidth=1.2)
    )

    # Set coordinate limits
    ax.set_xlim(-0.8, 4 * h_spacing + 3.8)
    ax.set_ylim(bot_y - 2.2, max_top_y + 1.2)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_svg, format='svg', bbox_inches='tight')
    plt.close()
    print("Neural network architecture diagram successfully generated:")
    print(f"  • PNG: {output_png}")
    print(f"  • SVG: {output_svg}")


if __name__ == "__main__":
    draw_neural_net()
