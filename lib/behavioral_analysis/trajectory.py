"""Trajectory reconstruction and analysis tools."""
from __future__ import annotations
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
from pathlib import Path
import json
from dataclasses import dataclass
import pandas as pd
import math

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# ============================================================================
# TRAJECTORY RECONSTRUCTION FROM CSV (moved from tools/reconstruct_trajectory.py)
# ============================================================================

def load_csv(path: Path) -> pd.DataFrame:
    """Load and sort trajectory CSV by frame or timestamp."""
    df = pd.read_csv(path)
    by = "frame_idx" if "frame_idx" in df.columns else "ts"
    df = df.sort_values(by=[by]).reset_index(drop=True)
    return df

def infer_px_per_cm(df: pd.DataFrame) -> float | None:
    """Infer pixel-to-cm conversion from calibration data in CSV."""
    if "dist_px" in df.columns and "dist_cm" in df.columns:
        sub = df[~df["dist_cm"].isna() & np.isfinite(df["dist_cm"]) & 
                np.isfinite(df["dist_px"]) & (df["dist_cm"] != 0)]
        if len(sub) >= 5:
            ratio = (sub["dist_px"] / sub["dist_cm"]).median()
            if np.isfinite(ratio) and ratio > 0:
                return float(ratio)
    return None

def smooth_series(x: np.ndarray, win: int) -> np.ndarray:
    """Apply moving average smoothing to 1D array."""
    if win <= 1:
        return x
    win = int(win)
    if win % 2 == 0:
        win += 1
    pad = win // 2
    kern = np.ones(win) / win
    xpad = np.pad(x, (pad, pad), mode="reflect")
    return np.convolve(xpad, kern, mode="valid")

def compute_kinematics(df: pd.DataFrame, smooth: int = 1) -> pd.DataFrame:
    """Compute velocity, acceleration, and heading from trajectory data."""
    if "ts" not in df.columns:
        raise ValueError("CSV must include a 'ts' column (seconds).")
    
    t = df["ts"].to_numpy(dtype=float)
    dt = np.diff(t, prepend=t[0])
    dt[dt <= 0] = np.nan  # avoid div-by-zero
    
    # Extract position coordinates
    x_col = "cx" if "cx" in df.columns else "x"
    y_col = "cy" if "cy" in df.columns else "y"
    
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"CSV must include position columns ({x_col}, {y_col})")
    
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    
    # Apply smoothing if requested
    if smooth > 1:
        x = smooth_series(x, smooth)
        y = smooth_series(y, smooth)
    
    # Compute velocity
    vx = np.diff(x, prepend=x[0]) / dt
    vy = np.diff(y, prepend=y[0]) / dt
    speed = np.sqrt(vx*vx + vy*vy)
    
    # Compute acceleration
    ax = np.diff(vx, prepend=vx[0]) / dt
    ay = np.diff(vy, prepend=vy[0]) / dt
    accel = np.sqrt(ax*ax + ay*ay)
    
    # Compute heading (unit vector from velocity)
    heading_x = np.where(speed > 0, vx / speed, 0.0)
    heading_y = np.where(speed > 0, vy / speed, 0.0)
    
    # Create enriched dataframe
    enriched = df.copy()
    enriched["x_smooth"] = x
    enriched["y_smooth"] = y
    enriched["vx"] = vx
    enriched["vy"] = vy
    enriched["speed"] = speed
    enriched["ax"] = ax
    enriched["ay"] = ay
    enriched["accel"] = accel
    enriched["heading_x"] = heading_x
    enriched["heading_y"] = heading_y
    
    return enriched

def reconstruct_trajectory_from_csv(csv_path: Union[str, Path], 
                                  output_dir: Optional[Union[str, Path]] = None,
                                  smooth_window: int = 1) -> pd.DataFrame:
    """Full trajectory reconstruction pipeline from CSV file."""
    csv_path = Path(csv_path)
    if output_dir is None:
        output_dir = csv_path.parent / "trajectory_analysis"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    # Load and process data
    df = load_csv(csv_path)
    enriched_df = compute_kinematics(df, smooth=smooth_window)
    
    # Save enriched data
    output_csv = output_dir / "trajectory_enriched.csv"
    enriched_df.to_csv(output_csv, index=False)
    
    # Generate summary plots if matplotlib available
    if MATPLOTLIB_AVAILABLE:
        create_trajectory_summary_plots(enriched_df, output_dir)
    
    return enriched_df

def create_trajectory_summary_plots(df: pd.DataFrame, output_dir: Path):
    """Create basic trajectory analysis plots using matplotlib."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Trajectory path
    ax = axes[0, 0]
    ax.plot(df["x_smooth"], df["y_smooth"], 'b-', alpha=0.7, linewidth=1)
    ax.scatter(df["x_smooth"].iloc[0], df["y_smooth"].iloc[0], 
               color='green', s=50, label='Start', zorder=5)
    ax.scatter(df["x_smooth"].iloc[-1], df["y_smooth"].iloc[-1], 
               color='red', s=50, label='End', zorder=5)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Trajectory Path")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Speed over time
    ax = axes[0, 1]
    if "ts" in df.columns:
        ax.plot(df["ts"], df["speed"], 'r-', linewidth=1)
        ax.set_xlabel("Time (s)")
    else:
        ax.plot(df["speed"], 'r-', linewidth=1)
        ax.set_xlabel("Frame")
    ax.set_ylabel("Speed")
    ax.set_title("Speed Profile")
    ax.grid(True, alpha=0.3)
    
    # 3. Acceleration over time
    ax = axes[1, 0]
    if "ts" in df.columns:
        ax.plot(df["ts"], df["accel"], 'orange', linewidth=1)
        ax.set_xlabel("Time (s)")
    else:
        ax.plot(df["accel"], 'orange', linewidth=1)
        ax.set_xlabel("Frame")
    ax.set_ylabel("Acceleration")
    ax.set_title("Acceleration Profile")
    ax.grid(True, alpha=0.3)
    
    # 4. Speed histogram
    ax = axes[1, 1]
    ax.hist(df["speed"].dropna(), bins=30, alpha=0.7, color='blue')
    ax.set_xlabel("Speed")
    ax.set_ylabel("Frequency")
    ax.set_title("Speed Distribution")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "trajectory_summary.png", dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================  
# PLOTLY ARROW VISUALIZATION (moved from tools/plot_arena_arrows_plotly.py)
# ============================================================================

def create_arena_grid_shapes(width: float, height: float, grid_step: float) -> List[Dict]:
    """Create grid shapes for arena visualization."""
    shapes = []
    if grid_step and grid_step > 0:
        # Vertical lines
        for x in range(0, int(width)+1, int(grid_step)):
            shapes.append({
                'type': 'line',
                'x0': x, 'y0': 0, 'x1': x, 'y1': height,
                'line': {'color': 'rgba(0,0,0,0.15)', 'width': 1}
            })
        # Horizontal lines
        for y in range(0, int(height)+1, int(grid_step)):
            shapes.append({
                'type': 'line', 
                'x0': 0, 'y0': y, 'x1': width, 'y1': y,
                'line': {'color': 'rgba(0,0,0,0.15)', 'width': 1}
            })
    return shapes

def plot_trajectory_arrows_plotly(df: pd.DataFrame,
                                 output_path: Union[str, Path] = "trajectory_arrows.html",
                                 arrow_length: float = 30.0,
                                 arena_width: float = 1920,
                                 arena_height: float = 1080,
                                 grid_step: float = 120,
                                 color_by: str = "ts",
                                 stride: int = 1) -> Optional[go.Figure]:
    """Create interactive trajectory plot with direction arrows using Plotly."""
    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Install with: pip install plotly")
        return None
    
    # Subsample data if needed
    if stride > 1:
        df = df.iloc[::stride].copy()
    
    # Extract positions
    x_col = "x_smooth" if "x_smooth" in df.columns else ("cx" if "cx" in df.columns else "x")
    y_col = "y_smooth" if "y_smooth" in df.columns else ("cy" if "cy" in df.columns else "y")
    
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Position columns not found: {x_col}, {y_col}")
    
    x = df[x_col].to_numpy()
    y = df[y_col].to_numpy()
    
    # Determine heading vectors
    if "heading_x" in df.columns and "heading_y" in df.columns:
        hx = df["heading_x"].to_numpy()
        hy = df["heading_y"].to_numpy()
    else:
        # Compute from frame-to-frame movement
        hx = np.diff(x, prepend=x[0])
        hy = np.diff(y, prepend=y[0])
        # Normalize to unit vectors
        mag = np.sqrt(hx*hx + hy*hy)
        hx = np.where(mag > 0, hx / mag, 0)
        hy = np.where(mag > 0, hy / mag, 0)
    
    # Color encoding
    if color_by == "ts" and "ts" in df.columns:
        colors = df["ts"].to_numpy()
        colorbar_title = "Time (s)"
    else:
        colors = np.arange(len(df))
        colorbar_title = "Frame Index"
    
    # Create figure
    fig = go.Figure()
    
    # Add grid shapes
    grid_shapes = create_arena_grid_shapes(arena_width, arena_height, grid_step)
    
    # Add trajectory arrows
    for i in range(len(x)):
        if np.isnan(x[i]) or np.isnan(y[i]):
            continue
        
        # Arrow tip position
        tip_x = x[i] + arrow_length * hx[i]
        tip_y = y[i] + arrow_length * hy[i]
        
        # Add arrow
        fig.add_annotation(
            x=tip_x, y=tip_y,
            ax=x[i], ay=y[i],
            xref='x', yref='y',
            axref='x', ayref='y',
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=px.colors.sequential.Viridis[int(colors[i] / max(colors) * (len(px.colors.sequential.Viridis)-1))],
            showlegend=False
        )
    
    # Add trajectory line
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='lines+markers',
        line=dict(width=1, color='rgba(100,100,100,0.5)'),
        marker=dict(
            size=4,
            color=colors,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title=colorbar_title)
        ),
        name='Trajectory',
        hovertemplate=f'<b>X:</b> %{{x:.1f}}<br><b>Y:</b> %{{y:.1f}}<br><b>{colorbar_title}:</b> %{{marker.color}}<extra></extra>'
    ))
    
    # Layout
    fig.update_layout(
        title="Trajectory with Direction Arrows",
        xaxis=dict(
            title="X Position",
            range=[0, arena_width],
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis=dict(
            title="Y Position", 
            range=[0, arena_height],
            autorange="reversed"  # Flip Y to match image coordinates
        ),
        shapes=grid_shapes,
        showlegend=False,
        width=900,
        height=600
    )
    
    # Save to file
    fig.write_html(output_path)
    return fig


@dataclass
class TrajectoryPoint:
    """Single point in trajectory with metadata."""
    x: float
    y: float
    frame: int
    timestamp: Optional[float] = None
    speed: Optional[float] = None
    direction: Optional[float] = None
    confidence: Optional[float] = None


class TrajectoryAnalyzer:
    """Analyze and visualize animal trajectories."""
    
    def __init__(self, points: Optional[List[TrajectoryPoint]] = None):
        self.points = points or []
        self._cached_metrics = None
    
    @classmethod
    def from_coordinates(cls, coordinates: List[Tuple[float, float]], 
                        frame_numbers: Optional[List[int]] = None) -> 'TrajectoryAnalyzer':
        """Create analyzer from simple coordinate list."""
        if frame_numbers is None:
            frame_numbers = list(range(len(coordinates)))
        
        points = [
            TrajectoryPoint(x=coord[0], y=coord[1], frame=frame)
            for coord, frame in zip(coordinates, frame_numbers)
        ]
        return cls(points)
    
    @classmethod
    def from_csv(cls, filepath: Union[str, Path], 
                 x_col: str = 'x', y_col: str = 'y', frame_col: str = 'frame') -> 'TrajectoryAnalyzer':
        """Load trajectory from CSV file."""
        import pandas as pd
        
        df = pd.read_csv(filepath)
        points = [
            TrajectoryPoint(
                x=row[x_col],
                y=row[y_col], 
                frame=row[frame_col] if frame_col in df.columns else i
            )
            for i, row in df.iterrows()
        ]
        return cls(points)
    
    def add_point(self, point: TrajectoryPoint):
        """Add a single trajectory point."""
        self.points.append(point)
        self._cached_metrics = None  # Invalidate cache
    
    def add_coordinates(self, x: float, y: float, frame: int, **kwargs):
        """Add point from coordinates."""
        point = TrajectoryPoint(x=x, y=y, frame=frame, **kwargs)
        self.add_point(point)
    
    def calculate_metrics(self, fps: float = 30.0) -> Dict[str, Any]:
        """Calculate comprehensive trajectory metrics."""
        if not self.points:
            return {}
        
        if self._cached_metrics is not None:
            return self._cached_metrics
        
        points = self.points
        n_points = len(points)
        
        # Basic statistics
        x_coords = [p.x for p in points]
        y_coords = [p.y for p in points]
        
        metrics = {
            'total_points': n_points,
            'duration_frames': points[-1].frame - points[0].frame if n_points > 1 else 0,
            'duration_seconds': (points[-1].frame - points[0].frame) / fps if n_points > 1 else 0,
            
            # Spatial extent
            'min_x': min(x_coords),
            'max_x': max(x_coords),
            'min_y': min(y_coords),
            'max_y': max(y_coords),
            'center_x': np.mean(x_coords),
            'center_y': np.mean(y_coords),
            'std_x': np.std(x_coords),
            'std_y': np.std(y_coords),
        }
        
        if n_points > 1:
            # Calculate distances and speeds
            distances = []
            speeds = []
            directions = []
            
            for i in range(1, n_points):
                dx = points[i].x - points[i-1].x
                dy = points[i].y - points[i-1].y
                dist = np.sqrt(dx*dx + dy*dy)
                distances.append(dist)
                
                # Speed (pixels per frame)
                frame_diff = points[i].frame - points[i-1].frame
                speed = dist / frame_diff if frame_diff > 0 else 0
                speeds.append(speed)
                
                # Direction (radians)
                if dx != 0 or dy != 0:
                    direction = np.arctan2(dy, dx)
                    directions.append(direction)
            
            # Movement metrics
            metrics.update({
                'total_distance': sum(distances),
                'mean_speed_px_per_frame': np.mean(speeds) if speeds else 0,
                'max_speed_px_per_frame': max(speeds) if speeds else 0,
                'median_speed_px_per_frame': np.median(speeds) if speeds else 0,
                'speed_std': np.std(speeds) if speeds else 0,
                
                # Path efficiency (straight-line distance / total path length)
                'path_efficiency': self._calculate_path_efficiency(),
                
                # Tortuosity
                'tortuosity': self._calculate_tortuosity(),
                
                # Direction change statistics
                'mean_absolute_direction_change': self._calculate_direction_changes(),
                'direction_stability': self._calculate_direction_stability(),
            })
        
        self._cached_metrics = metrics
        return metrics
    
    def _calculate_path_efficiency(self) -> float:
        """Calculate path efficiency (0 = very tortuous, 1 = straight line)."""
        if len(self.points) < 2:
            return 0.0
        
        # Straight-line distance
        start = self.points[0]
        end = self.points[-1]
        straight_distance = np.sqrt((end.x - start.x)**2 + (end.y - start.y)**2)
        
        # Total path length
        total_distance = 0
        for i in range(1, len(self.points)):
            dx = self.points[i].x - self.points[i-1].x
            dy = self.points[i].y - self.points[i-1].y
            total_distance += np.sqrt(dx*dx + dy*dy)
        
        return straight_distance / total_distance if total_distance > 0 else 0
    
    def _calculate_tortuosity(self) -> float:
        """Calculate tortuosity (total path length / straight-line distance)."""
        efficiency = self._calculate_path_efficiency()
        return 1.0 / efficiency if efficiency > 0 else float('inf')
    
    def _calculate_direction_changes(self) -> float:
        """Calculate mean absolute direction change between segments."""
        if len(self.points) < 3:
            return 0.0
        
        direction_changes = []
        prev_direction = None
        
        for i in range(1, len(self.points)):
            dx = self.points[i].x - self.points[i-1].x
            dy = self.points[i].y - self.points[i-1].y
            
            if dx != 0 or dy != 0:
                direction = np.arctan2(dy, dx)
                
                if prev_direction is not None:
                    # Calculate angular difference
                    diff = direction - prev_direction
                    # Normalize to [-pi, pi]
                    diff = ((diff + np.pi) % (2 * np.pi)) - np.pi
                    direction_changes.append(abs(diff))
                
                prev_direction = direction
        
        return np.mean(direction_changes) if direction_changes else 0.0
    
    def _calculate_direction_stability(self) -> float:
        """Calculate direction stability (0 = very unstable, 1 = very stable)."""
        if len(self.points) < 2:
            return 0.0
        
        directions = []
        for i in range(1, len(self.points)):
            dx = self.points[i].x - self.points[i-1].x
            dy = self.points[i].y - self.points[i-1].y
            if dx != 0 or dy != 0:
                direction = np.arctan2(dy, dx)
                directions.append(direction)
        
        if len(directions) < 2:
            return 0.0
        
        # Calculate circular variance
        mean_cos = np.mean(np.cos(directions))
        mean_sin = np.mean(np.sin(directions))
        circular_variance = 1 - np.sqrt(mean_cos**2 + mean_sin**2)
        
        return 1.0 - circular_variance
    
    def smooth_trajectory(self, window_size: int = 5) -> 'TrajectoryAnalyzer':
        """Apply smoothing to trajectory coordinates."""
        if len(self.points) < window_size:
            return TrajectoryAnalyzer(self.points.copy())
        
        smoothed_points = []
        half_window = window_size // 2
        
        for i, point in enumerate(self.points):
            start_idx = max(0, i - half_window)
            end_idx = min(len(self.points), i + half_window + 1)
            
            window_points = self.points[start_idx:end_idx]
            smooth_x = np.mean([p.x for p in window_points])
            smooth_y = np.mean([p.y for p in window_points])
            
            smoothed_point = TrajectoryPoint(
                x=smooth_x,
                y=smooth_y,
                frame=point.frame,
                timestamp=point.timestamp,
                confidence=point.confidence
            )
            smoothed_points.append(smoothed_point)
        
        return TrajectoryAnalyzer(smoothed_points)
    
    def subsample(self, step: int = 2) -> 'TrajectoryAnalyzer':
        """Subsample trajectory by taking every nth point."""
        subsampled_points = self.points[::step]
        return TrajectoryAnalyzer(subsampled_points)
    
    def plot_trajectory_plotly(self, 
                              title: str = "Animal Trajectory",
                              show_arrows: bool = True,
                              arrow_step: int = 10,
                              color_by: str = "time") -> Optional[go.Figure]:
        """Create interactive trajectory plot using Plotly."""
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Install with: pip install plotly")
            return None
        
        if not self.points:
            print("No trajectory points to plot")
            return None
        
        x_coords = [p.x for p in self.points]
        y_coords = [p.y for p in self.points]
        frames = [p.frame for p in self.points]
        
        fig = go.Figure()
        
        # Main trajectory line
        if color_by == "time":
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines+markers',
                marker=dict(
                    size=4,
                    color=frames,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Frame")
                ),
                line=dict(width=2),
                name='Trajectory',
                hovertemplate='<b>Frame:</b> %{text}<br><b>X:</b> %{x:.1f}<br><b>Y:</b> %{y:.1f}<extra></extra>',
                text=frames
            ))
        else:
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines+markers',
                marker=dict(size=4, color='blue'),
                line=dict(width=2),
                name='Trajectory'
            ))
        
        # Add direction arrows
        if show_arrows and len(self.points) > 1:
            arrow_indices = range(0, len(self.points) - 1, arrow_step)
            
            for i in arrow_indices:
                if i + 1 < len(self.points):
                    dx = self.points[i + 1].x - self.points[i].x
                    dy = self.points[i + 1].y - self.points[i].y
                    
                    # Scale arrow
                    length = np.sqrt(dx*dx + dy*dy)
                    if length > 0:
                        scale = min(20, length * 0.5)  # Adjust arrow size
                        dx_scaled = (dx / length) * scale
                        dy_scaled = (dy / length) * scale
                        
                        fig.add_annotation(
                            x=self.points[i].x + dx_scaled,
                            y=self.points[i].y + dy_scaled,
                            ax=self.points[i].x,
                            ay=self.points[i].y,
                            xref='x', yref='y',
                            axref='x', ayref='y',
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=1,
                            arrowcolor='red',
                            showlegend=False
                        )
        
        # Mark start and end points
        fig.add_trace(go.Scatter(
            x=[x_coords[0]], y=[y_coords[0]],
            mode='markers',
            marker=dict(size=12, color='green', symbol='circle'),
            name='Start',
            showlegend=True
        ))
        
        fig.add_trace(go.Scatter(
            x=[x_coords[-1]], y=[y_coords[-1]],
            mode='markers',
            marker=dict(size=12, color='red', symbol='x'),
            name='End',
            showlegend=True
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="X (pixels)",
            yaxis_title="Y (pixels)",
            yaxis=dict(scaleanchor="x", scaleratio=1),  # Equal aspect ratio
            showlegend=True
        )
        
        return fig
    
    def plot_speed_profile(self, fps: float = 30.0) -> Optional[go.Figure]:
        """Plot speed profile over time using Plotly."""
        if not PLOTLY_AVAILABLE:
            print("Plotly not available. Install with: pip install plotly")
            return None
        
        if len(self.points) < 2:
            print("Need at least 2 points for speed profile")
            return None
        
        speeds = []
        times = []
        
        for i in range(1, len(self.points)):
            dx = self.points[i].x - self.points[i-1].x
            dy = self.points[i].y - self.points[i-1].y
            dist = np.sqrt(dx*dx + dy*dy)
            
            frame_diff = self.points[i].frame - self.points[i-1].frame
            speed = dist / frame_diff if frame_diff > 0 else 0
            speeds.append(speed)
            times.append(self.points[i].frame / fps)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times,
            y=speeds,
            mode='lines',
            name='Speed',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title="Speed Profile",
            xaxis_title="Time (seconds)",
            yaxis_title="Speed (pixels/frame)",
            showlegend=False
        )
        
        return fig
    
    def export_trajectory(self, filepath: Union[str, Path], format_type: str = "csv"):
        """Export trajectory to file."""
        filepath = Path(filepath)
        
        if format_type == "csv":
            import pandas as pd
            
            data = {
                'frame': [p.frame for p in self.points],
                'x': [p.x for p in self.points],
                'y': [p.y for p in self.points],
            }
            
            # Add optional columns if available
            if any(p.timestamp is not None for p in self.points):
                data['timestamp'] = [p.timestamp for p in self.points]
            if any(p.speed is not None for p in self.points):
                data['speed'] = [p.speed for p in self.points]
            if any(p.direction is not None for p in self.points):
                data['direction'] = [p.direction for p in self.points]
            if any(p.confidence is not None for p in self.points):
                data['confidence'] = [p.confidence for p in self.points]
            
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)
        
        elif format_type == "json":
            data = [
                {
                    'frame': p.frame,
                    'x': p.x,
                    'y': p.y,
                    'timestamp': p.timestamp,
                    'speed': p.speed,
                    'direction': p.direction,
                    'confidence': p.confidence
                }
                for p in self.points
            ]
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive trajectory summary."""
        metrics = self.calculate_metrics()
        
        return {
            'trajectory_summary': {
                'total_points': len(self.points),
                'spatial_extent': {
                    'width': metrics.get('max_x', 0) - metrics.get('min_x', 0),
                    'height': metrics.get('max_y', 0) - metrics.get('min_y', 0),
                    'center': (metrics.get('center_x', 0), metrics.get('center_y', 0))
                },
                'movement_stats': {
                    'total_distance': metrics.get('total_distance', 0),
                    'path_efficiency': metrics.get('path_efficiency', 0),
                    'mean_speed': metrics.get('mean_speed_px_per_frame', 0),
                    'direction_stability': metrics.get('direction_stability', 0)
                },
                'temporal_info': {
                    'duration_frames': metrics.get('duration_frames', 0),
                    'duration_seconds': metrics.get('duration_seconds', 0)
                }
            },
            'full_metrics': metrics
        }