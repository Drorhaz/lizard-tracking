"""Plotly-based visualization for nose-heading trajectory map."""
from __future__ import annotations
from pathlib import Path
from typing import List
import numpy as np

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️ Plotly not available. Install with: pip install plotly")


def create_nose_heading_map(plot_data: List[dict], 
                            video_name: str,
                            output_path: Path,
                            config,
                            frame_width: int,
                            frame_height: int) -> bool:
    """
    Create interactive Plotly HTML plot showing nose trajectory with time-colored arrows.
    
    Args:
        plot_data: List of dicts with keys: frame_idx, x_norm, y_norm, head_angle_deg, dist_norm
        video_name: Name for plot title
        output_path: Path to save HTML file
        config: AdvancedBehaviorConfig instance
        frame_width: Original frame width
        frame_height: Original frame height
        
    Returns:
        True if successful, False otherwise
    """
    if not PLOTLY_AVAILABLE:
        print("❌ Plotly not available, cannot create nose-heading map")
        return False
    
    if not plot_data:
        print("⚠️ No plot data available")
        return False
    
    # Extract data
    frames = [d['frame_idx'] for d in plot_data]
    x_coords = [d['x_norm'] for d in plot_data]
    y_coords = [d['y_norm'] for d in plot_data]
    angles = [d.get('head_angle_deg') for d in plot_data]
    
    # Normalize time for color encoding [0, 1]
    t_min, t_max = min(frames), max(frames)
    t_norm = [(f - t_min) / (t_max - t_min) if t_max > t_min else 0.5 for f in frames]
    
    # Create figure
    fig = go.Figure()
    
    # Add background grid/arena boundaries
    fig.add_shape(
        type="rect",
        x0=0, y0=0, x1=1, y1=1,
        line=dict(color="lightgray", width=2),
        fillcolor="rgba(240, 240, 240, 0.3)"
    )
    
    # Add target line indicator
    target_line_color = "rgba(255, 0, 0, 0.5)"
    if config.target_line == 'right':
        fig.add_shape(type="line", x0=1, y0=0, x1=1, y1=1, 
                     line=dict(color=target_line_color, width=4))
        fig.add_annotation(x=0.98, y=0.5, text="TARGET", textangle=-90,
                          font=dict(size=14, color="red"))
    elif config.target_line == 'left':
        fig.add_shape(type="line", x0=0, y0=0, x1=0, y1=1,
                     line=dict(color=target_line_color, width=4))
        fig.add_annotation(x=0.02, y=0.5, text="TARGET", textangle=90,
                          font=dict(size=14, color="red"))
    elif config.target_line == 'top':
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=0,
                     line=dict(color=target_line_color, width=4))
        fig.add_annotation(x=0.5, y=0.02, text="TARGET",
                          font=dict(size=14, color="red"))
    else:  # bottom
        fig.add_shape(type="line", x0=0, y0=1, x1=1, y1=1,
                     line=dict(color=target_line_color, width=4))
        fig.add_annotation(x=0.5, y=0.98, text="TARGET",
                          font=dict(size=14, color="red"))
    
    # Add band boundaries (near/middle/far)
    near_line_color = "rgba(0, 255, 0, 0.3)"
    middle_line_color = "rgba(255, 255, 0, 0.3)"
    
    if config.target_line in ['right', 'left']:
        # Vertical target line - bands are horizontal
        near_x = 1 - config.near_max if config.target_line == 'right' else config.near_max
        middle_x = 1 - config.middle_max if config.target_line == 'right' else config.middle_max
        
        fig.add_shape(type="line", x0=near_x, y0=0, x1=near_x, y1=1,
                     line=dict(color=near_line_color, width=1, dash="dash"))
        fig.add_shape(type="line", x0=middle_x, y0=0, x1=middle_x, y1=1,
                     line=dict(color=middle_line_color, width=1, dash="dash"))
        fig.add_annotation(x=near_x, y=0.95, text="near", font=dict(size=10, color="green"))
        fig.add_annotation(x=middle_x, y=0.95, text="middle", font=dict(size=10, color="orange"))
    else:
        # Horizontal target line - bands are vertical
        near_y = 1 - config.near_max if config.target_line == 'bottom' else config.near_max
        middle_y = 1 - config.middle_max if config.target_line == 'bottom' else config.middle_max
        
        fig.add_shape(type="line", x0=0, y0=near_y, x1=1, y1=near_y,
                     line=dict(color=near_line_color, width=1, dash="dash"))
        fig.add_shape(type="line", x0=0, y0=middle_y, x1=1, y1=middle_y,
                     line=dict(color=middle_line_color, width=1, dash="dash"))
        fig.add_annotation(x=0.05, y=near_y, text="near", font=dict(size=10, color="green"))
        fig.add_annotation(x=0.05, y=middle_y, text="middle", font=dict(size=10, color="orange"))
    
    # Add nose positions as scatter points
    fig.add_trace(go.Scatter(
        x=x_coords,
        y=y_coords,
        mode='markers+lines',
        marker=dict(
            size=6,
            color=t_norm,
            colorscale=config.plot_colorscale,
            showscale=True,
            colorbar=dict(
                title="Time<br>(early→late)",
                x=1.15,
            ),
            line=dict(width=1, color='white')
        ),
        line=dict(width=1, color='rgba(100, 100, 100, 0.3)'),
        name='Nose trajectory',
        hovertemplate='<b>Frame %{customdata}</b><br>' +
                     'Position: (%{x:.3f}, %{y:.3f})<br>' +
                     '<extra></extra>',
        customdata=frames,
    ))
    
    # Add heading arrows
    arrow_len = config.arrow_length_norm
    for i, (x, y, angle, t) in enumerate(zip(x_coords, y_coords, angles, t_norm)):
        if angle is None:
            continue
        
        # Convert angle to radians and compute arrow endpoint
        angle_rad = np.radians(angle)
        dx = arrow_len * np.cos(angle_rad)
        dy = arrow_len * np.sin(angle_rad)
        
        # Get color for this time point
        colorscale_colors = {
            'Viridis': [(0, 0.267, 0.004, 0.329), (0.993, 0.906, 0.144, 1.0)],
            'Plasma': [(0.050, 0.030, 0.529, 1.0), (0.940, 0.976, 0.130, 1.0)],
        }
        
        # Simple color interpolation (could use proper colorscale)
        color = f'rgba({int(255*t)}, {int(255*(1-t))}, {int(128*t)}, 0.7)'
        
        fig.add_annotation(
            x=x, y=y,
            ax=x - dx, ay=y - dy,  # Arrow base
            xref='x', yref='y',
            axref='x', ayref='y',
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=color,
        )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f"<b>Nose-Heading Trajectory Map</b><br><sub>{video_name}</sub>",
            x=0.5,
            xanchor='center',
        ),
        xaxis=dict(
            title="X (normalized)",
            range=[0, 1],
            constrain='domain',
            showgrid=True,
            gridcolor='lightgray',
        ),
        yaxis=dict(
            title="Y (normalized)",
            range=[0, 1],
            constrain='domain',
            scaleanchor='x',
            scaleratio=1,
            showgrid=True,
            gridcolor='lightgray',
            autorange='reversed',  # Match image coordinates (y downward)
        ),
        width=900,
        height=900,
        hovermode='closest',
        plot_bgcolor='white',
        showlegend=True,
        legend=dict(x=0.02, y=0.98),
    )
    
    # Save as HTML
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path), include_plotlyjs='cdn')
    print(f"📊 Nose-heading map saved to: {output_path}")
    
    return True


def save_trajectory_csv(plot_data: List[dict], output_path: Path) -> bool:
    """
    Save trajectory data to CSV.
    
    Format: frame_idx, ts_ms, x, y, head_angle_deg, dist_to_target_px, dist_to_target_norm
    """
    import csv
    
    if not plot_data:
        return False
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_idx', 'x_norm', 'y_norm', 'head_angle_deg', 'dist_to_target_norm'])
        
        for d in plot_data:
            writer.writerow([
                d['frame_idx'],
                d['x_norm'],
                d['y_norm'],
                d.get('head_angle_deg', ''),
                d.get('dist_norm', ''),
            ])
    
    print(f"📊 Trajectory CSV saved to: {output_path}")
    return True


def save_events_csv(instructions: List, output_path: Path) -> bool:
    """
    Save behavioral events/instructions to CSV.
    
    Format: ts_ms, frame_idx, instruction, meta_json
    """
    import csv
    
    if not instructions:
        return False
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ts_ms', 'frame_idx', 'instruction', 'meta_json'])
        
        for row in instructions:
            writer.writerow(row)
    
    print(f"📊 Events CSV saved to: {output_path}")
    return True
