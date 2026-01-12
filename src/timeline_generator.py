import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from typing import Dict, List
import colorsys


def generate_color_palette(n: int) -> List[str]:
    colors = []
    for i in range(n):
        hue = i / n
        rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )
        colors.append(hex_color)
    return colors


def create_interactive_timeline(results: Dict) -> go.Figure:
    duration = results['video_metadata']['duration']
    fig = make_subplots(
        rows=4,
        cols=1,
        row_heights=[0.3, 0.3, 0.2, 0.2],
        subplot_titles=("Object Tracks", "Scenes", "Transcript Segments", "Key Moments"),
        vertical_spacing=0.08,
        specs=[[{"type": "scatter"}],
               [{"type": "scatter"}],
               [{"type": "scatter"}],
               [{"type": "scatter"}]]
    )
    tracks = results.get('tracks', {})
    unique_classes = list(set(track['class'] for track in tracks.values()))
    colors = generate_color_palette(len(unique_classes))
    class_colors = {cls: color for cls, color in zip(unique_classes, colors)}
    y_position = 0
    class_y_positions = {}
    for track_id, track_info in tracks.items():
        obj_class = track_info['class']
        if obj_class not in class_y_positions:
            class_y_positions[obj_class] = y_position
            y_position += 1
        y_pos = class_y_positions[obj_class]
        fig.add_trace(
            go.Scatter(
                x=[track_info['first_appearance'], track_info['last_appearance']],
                y=[y_pos, y_pos],
                mode='lines',
                line=dict(color=class_colors[obj_class], width=10),
                name=f"{obj_class} #{track_id}",
                text=f"Track {track_id}: {obj_class}<br>"
                     f"Duration: {track_info['duration']:.2f}s<br>"
                     f"Confidence: {track_info['avg_confidence']:.1%}",
                hoverinfo='text',
                showlegend=True,
                legendgroup=obj_class
            ),
            row=1, col=1
        )
    fig.update_yaxes(
        title_text="Object Classes",
        ticktext=list(class_y_positions.keys()),
        tickvals=list(class_y_positions.values()),
        row=1, col=1
    )
    scenes = results.get('scenes', [])
    for i, scene in enumerate(scenes):
        fig.add_vrect(
            x0=scene['start_time'],
            x1=scene['end_time'],
            fillcolor="lightblue" if i % 2 == 0 else "lightgreen",
            opacity=0.3,
            layer="below",
            line_width=0,
            row=2, col=1
        )
        mid_time = (scene['start_time'] + scene['end_time']) / 2
        fig.add_trace(
            go.Scatter(
                x=[mid_time],
                y=[0.5],
                mode='text',
                text=f"Scene {scene['scene_number']}",
                textposition="middle center",
                showlegend=False,
                hovertext=f"{scene['description']}<br>Confidence: {scene['confidence']:.1%}",
                hoverinfo='text'
            ),
            row=2, col=1
        )
    fig.update_yaxes(visible=False, row=2, col=1)
    segments = results.get('audio', {}).get('segments', [])
    for i, segment in enumerate(segments):
        fig.add_trace(
            go.Scatter(
                x=[segment['start'], segment['end']],
                y=[0, 0],
                mode='lines+markers',
                line=dict(color='purple', width=6),
                marker=dict(size=8),
                name=f"Speech {i+1}",
                text=segment['text'],
                hoverinfo='text',
                showlegend=False
            ),
            row=3, col=1
        )
    fig.update_yaxes(visible=False, row=3, col=1)
    key_moments = results.get('summary', {}).get('key_moments', [])
    if key_moments:
        times = [m['timestamp'] for m in key_moments]
        fig.add_trace(
            go.Scatter(
                x=times,
                y=[0.5] * len(times),
                mode='markers',
                marker=dict(
                    size=15,
                    color='red',
                    symbol='star',
                    line=dict(width=2, color='darkred')
                ),
                text=[m['description'] for m in key_moments],
                hoverinfo='text',
                name='Key Moments',
                showlegend=False
            ),
            row=4, col=1
        )
    
    fig.update_yaxes(visible=False, row=4, col=1)
    for i in range(1, 5):
        fig.update_xaxes(
            title_text="Time (seconds)" if i == 4 else "",
            range=[0, duration],
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            row=i, col=1
        )
    fig.update_layout(
        height=800,
        hovermode='closest',
        title_text=f"Video Timeline ({duration:.1f} seconds)",
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )
    
    return fig

def create_simple_timeline_html(results: Dict, output_path: str = None) -> str:
    duration = results['video_metadata']['duration']
    tracks = results.get('tracks', {})
    scenes = results.get('scenes', [])
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Video Timeline</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .timeline {{
                margin-top: 20px;
                border: 1px solid #ddd;
                background: white;
                padding: 10px;
            }}
            .track {{
                height: 30px;
                position: relative;
                margin-bottom: 5px;
                border-bottom: 1px solid #eee;
            }}
            .track-label {{
                display: inline-block;
                width: 150px;
                font-weight: bold;
                font-size: 12px;
            }}
            .track-bar {{
                position: absolute;
                height: 20px;
                border-radius: 3px;
                cursor: pointer;
            }}
            .track-bar:hover {{
                opacity: 0.8;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Video Content Analysis Timeline</h1>
            <p><strong>Duration:</strong> {duration:.2f} seconds</p>
            <p><strong>Objects Tracked:</strong> {len(tracks)}</p>
            <p><strong>Scenes:</strong> {len(scenes)}</p>
            
            <div class="timeline">
                <h3>Object Tracks</h3>
    """
    colors = generate_color_palette(len(set(t['class'] for t in tracks.values())))
    class_colors = {cls: color for cls, color in zip(set(t['class'] for t in tracks.values()), colors)}
    
    for track_id, track_info in tracks.items():
        start_pct = (track_info['first_appearance'] / duration) * 100
        width_pct = (track_info['duration'] / duration) * 100
        color = class_colors[track_info['class']]
        
        html += f"""
                <div class="track">
                    <span class="track-label">{track_info['class']} #{track_id}</span>
                    <div class="track-bar" 
                         style="left: {start_pct + 150}px; width: {width_pct}%; background-color: {color};"
                         title="{track_info['class']} - {track_info['duration']:.2f}s">
                    </div>
                </div>
        """
    
    html += """
            </div>
        </div>
    </body>
    </html>
    """
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(html)
    
    return html