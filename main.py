import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback
import time

# Constants
G = 6.674e-11  # Gravitational constant
M = 5.972e24   # Mass of Earth
R = 6.371e6    # Radius of Earth
rho0 = 1.225   # Air density at sea level
H = 8500       # Scale height

def simulate_rocket(initial_mass, mass_flow_rate, exhaust_velocity, drag_coefficient, 
                    cross_area, initial_speed, launch_angle, azimuth_angle, max_time):
    # Time settings
    dt = 0.1       # Time step
    time = np.arange(0, max_time, dt)
    
    # Initial conditions - Start on Earth's surface
    # Convert spherical to Cartesian coordinates
    # Starting at position (0, R, 0) - on the surface at the equator
    x, y, z = 0, R, 0
    
    # Initial velocity components - tangential to Earth's surface
    vx = initial_speed * np.sin(np.radians(launch_angle)) * np.cos(np.radians(azimuth_angle))
    vy = initial_speed * np.cos(np.radians(launch_angle))
    vz = initial_speed * np.sin(np.radians(launch_angle)) * np.sin(np.radians(azimuth_angle))
    
    m = initial_mass
    
    # Arrays to store trajectory and other variables
    x_traj, y_traj, z_traj = [], [], []
    speed_traj, altitude_traj, mass_traj, time_traj = [], [], [], []
    accel_traj, drag_traj, thrust_traj, gravity_traj = [], [], [], []
    
    # Numerical integration
    for t in time:
        # Calculate altitude above Earth's surface
        altitude = np.sqrt(x**2 + y**2 + z**2) - R
        
        # Stop simulation if rocket hits ground
        if altitude < 0 and len(x_traj) > 1:
            break
            
        # Compute gravity - vector pointing toward Earth's center
        r = np.sqrt(x**2 + y**2 + z**2)
        g_magnitude = G * M / r**2
        g_x = -g_magnitude * x / r
        g_y = -g_magnitude * y / r
        g_z = -g_magnitude * z / r
        
        # Compute air density (exponential decay with altitude)
        rho = rho0 * np.exp(-altitude / H) if altitude >= 0 else rho0
        
        # Compute drag force
        speed = np.sqrt(vx**2 + vy**2 + vz**2)
        Fd_x = -0.5 * rho * speed * drag_coefficient * cross_area * vx if altitude >= 0 else 0
        Fd_y = -0.5 * rho * speed * drag_coefficient * cross_area * vy if altitude >= 0 else 0
        Fd_z = -0.5 * rho * speed * drag_coefficient * cross_area * vz if altitude >= 0 else 0
        
        # Compute thrust force (if there's still fuel)
        if m > initial_mass * 0.1:  # Assume 10% of initial mass is the dry mass
            # Assuming thrust is in the direction of velocity to maximize orbital insertion
            if speed > 0:
                Ft_x = mass_flow_rate * exhaust_velocity * vx / speed
                Ft_y = mass_flow_rate * exhaust_velocity * vy / speed
                Ft_z = mass_flow_rate * exhaust_velocity * vz / speed
            else:
                # Default direction if speed is 0
                Ft_x = mass_flow_rate * exhaust_velocity * np.sin(np.radians(launch_angle)) * np.cos(np.radians(azimuth_angle))
                Ft_y = mass_flow_rate * exhaust_velocity * np.cos(np.radians(launch_angle))
                Ft_z = mass_flow_rate * exhaust_velocity * np.sin(np.radians(launch_angle)) * np.sin(np.radians(azimuth_angle))
            
            dm = mass_flow_rate * dt
        else:
            Ft_x, Ft_y, Ft_z = 0, 0, 0
            dm = 0
        
        # Total forces
        Fx = Ft_x + Fd_x + m * g_x
        Fy = Ft_y + Fd_y + m * g_y
        Fz = Ft_z + Fd_z + m * g_z
        
        # Accelerations
        ax = Fx / m
        ay = Fy / m
        az = Fz / m
        
        # Update velocities
        dvx = ax * dt
        dvy = ay * dt
        dvz = az * dt
        vx += dvx
        vy += dvy
        vz += dvz
        
        # Update positions
        x += vx * dt
        y += vy * dt
        z += vz * dt
        
        # Update mass
        m -= dm
        
        # Store trajectory and other variables
        x_traj.append(x)
        y_traj.append(y)
        z_traj.append(z)
        speed_traj.append(speed)
        altitude_traj.append(altitude)  # Store altitude above Earth's surface
        mass_traj.append(m)
        time_traj.append(t)
        accel_traj.append(np.sqrt(ax**2 + ay**2 + az**2))
        drag_traj.append(np.sqrt(Fd_x**2 + Fd_y**2 + Fd_z**2))
        thrust_traj.append(np.sqrt(Ft_x**2 + Ft_y**2 + Ft_z**2))
        gravity_traj.append(m * g_magnitude)
    
    return (np.array(x_traj), np.array(y_traj), np.array(z_traj), 
            np.array(speed_traj), np.array(altitude_traj), np.array(mass_traj), 
            np.array(time_traj), np.array(accel_traj), np.array(drag_traj), 
            np.array(thrust_traj), np.array(gravity_traj))

def create_rocket_figure(simulation_data, animation_frame=0):
    x_traj, y_traj, z_traj, speed_traj, altitude_traj, mass_traj, time_traj, accel_traj, drag_traj, thrust_traj, gravity_traj = simulation_data
    
    # Create subplot figure with 2 rows and 2 columns
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{'type': 'scene', 'rowspan': 2}, {'type': 'xy'}],
               [None, {'type': 'xy'}]],
        subplot_titles=["3D Trajectory", "Altitude vs Time", "Speed vs Time"],
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )
    
    # Add 3D trajectory
    fig.add_trace(
        go.Scatter3d(
            x=x_traj,
            y=z_traj,
            z=y_traj,
            mode='lines',
            line=dict(color='blue', width=4),
            name="Trajectory"
        ),
        row=1, col=1
    )
    
    # Add rocket position marker for current frame
    fig.add_trace(
        go.Scatter3d(
            x=[x_traj[animation_frame]],
            y=[z_traj[animation_frame]],
            z=[y_traj[animation_frame]],
            mode='markers',
            marker=dict(color='red', size=8),
            name="Rocket"
        ),
        row=1, col=1
    )
    
    # Add Earth (semi-transparent sphere)
    u = np.linspace(0, 2*np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x_earth = R * np.outer(np.cos(u), np.sin(v))
    y_earth = R * np.outer(np.sin(u), np.sin(v))
    z_earth = R * np.outer(np.ones(np.size(u)), np.cos(v))
    
    fig.add_trace(
        go.Surface(
            x=x_earth,
            y=y_earth,
            z=z_earth,
            colorscale=[[0, 'rgb(0, 100, 200)'], [1, 'rgb(0, 200, 255)']],
            opacity=0.3,
            showscale=False,
            name="Earth"
        ),
        row=1, col=1
    )
    
    # Add altitude vs time
    fig.add_trace(
        go.Scatter(
            x=time_traj,
            y=altitude_traj,
            mode='lines',
            line=dict(color='green'),
            name="Altitude"
        ),
        row=1, col=2
    )
    
    # Add altitude marker for current frame
    fig.add_trace(
        go.Scatter(
            x=[time_traj[animation_frame]],
            y=[altitude_traj[animation_frame]],
            mode='markers',
            marker=dict(color='red', size=8),
            name="Current Altitude",
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Add speed vs time
    fig.add_trace(
        go.Scatter(
            x=time_traj,
            y=speed_traj,
            mode='lines',
            line=dict(color='orange'),
            name="Speed"
        ),
        row=2, col=2
    )
    
    # Add speed marker for current frame
    fig.add_trace(
        go.Scatter(
            x=[time_traj[animation_frame]],
            y=[speed_traj[animation_frame]],
            mode='markers',
            marker=dict(color='red', size=8),
            name="Current Speed",
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Update 3D scene
    fig.update_scenes(
        xaxis_title="X Distance (m)",
        yaxis_title="Z Distance (m)",
        zaxis_title="Y Distance (m)",
        aspectmode='data',
        bgcolor='#f5f5f5'
    )
    
    # Update 2D axes
    fig.update_xaxes(title_text="Time (s)", row=1, col=2)
    fig.update_yaxes(title_text="Altitude (m)", row=1, col=2)
    fig.update_xaxes(title_text="Time (s)", row=2, col=2)
    fig.update_yaxes(title_text="Speed (m/s)", row=2, col=2)
    
    # Add current values annotation
    fig.add_annotation(
        text=f"<b>Time:</b> {time_traj[animation_frame]:.1f}s<br><b>Altitude:</b> {altitude_traj[animation_frame]/1000:.1f}km<br><b>Speed:</b> {speed_traj[animation_frame]:.1f}m/s<br><b>Mass:</b> {mass_traj[animation_frame]:.1f}kg<br><b>Acceleration:</b> {accel_traj[animation_frame]:.2f}m/s²",
        align="left",
        showarrow=False,
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        bgcolor="white",
        opacity=0.8
    )
    
    # Update layout
    fig.update_layout(
        title="Rocket Trajectory Simulation - Orbital Insertion",
        height=800,
        margin=dict(l=20, r=20, b=20, t=60),
        showlegend=True,
        legend=dict(x=0.7, y=0.9),
        scene=dict(
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        )
    )
    
    return fig

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server  # Needed for deploying to services like Heroku

# App layout
app.layout = html.Div([
    html.H1("Interactive Rocket Orbital Trajectory Simulator", style={'textAlign': 'center'}),
    
    html.Div([
        html.Div([
            html.H3("Rocket Parameters"),
            
            html.Div([
                html.Label("Initial Mass (kg):"),
                dcc.Slider(
                    id='initial-mass-slider',
                    min=10000, max=500000, step=10000, value=200000,  # Increased default mass
                    marks={10000: '10k', 100000: '100k', 500000: '500k'},
                ),
            ], style={'marginBottom': 20}),
            
            html.Div([
                html.Label("Mass Flow Rate (kg/s):"),
                dcc.Slider(
                    id='mass-flow-slider',
                    min=50, max=2000, step=10, value=500,  # Increased default and max flow rate
                    marks={50: '50', 1000: '1k', 2000: '2k'},
                ),
            ], style={'marginBottom': 20}),
            
            html.Div([
                html.Label("Exhaust Velocity (m/s):"),
                dcc.Slider(
                    id='exhaust-velocity-slider',
                    min=1000, max=15000, step=100, value=3000,  # Increased for orbital velocity
                    marks={1000: '1k', 7500: '7.5k', 15000: '15k'},
                ),
            ], style={'marginBottom': 20}),
            
            html.Div([
                html.Label("Drag Coefficient:"),
                dcc.Slider(
                    id='drag-coeff-slider',
                    min=0.1, max=2.0, step=0.1, value=0.2,  # Lower drag for orbital vehicle
                    marks={0.1: '0.1', 1.0: '1.0', 2.0: '2.0'},
                ),
            ], style={'marginBottom': 20}),
            
            html.Div([
                html.Label("Cross-sectional Area (m²):"),
                dcc.Slider(
                    id='cross-area-slider',
                    min=1, max=50, step=1, value=5,  # Smaller area for orbital vehicle
                    marks={1: '1', 25: '25', 50: '50'},
                ),
            ], style={'marginBottom': 20}),
        ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'}),
        
        html.Div([
            html.H3("Launch Parameters"),
            
            html.Div([
                html.Label("Initial Speed (m/s):"),
                dcc.Slider(
                    id='initial-speed-slider',
                    min=0, max=5000, step=50, value=2000,  # Higher initial speed
                    marks={0: '0', 2500: '2.5k', 5000: '5k'},
                ),
            ], style={'marginBottom': 20}),
            
            html.Div([
                html.Label("Launch Angle (degrees):"),
                dcc.Slider(
                    id='launch-angle-slider',
                    min=0, max=90, step=1, value=45,  # Lower angle for orbital insertion
                    marks={0: '0°', 45: '45°', 90: '90°'},
                ),
            ], style={'marginBottom': 20}),
            
            html.Div([
                html.Label("Azimuth Angle (degrees):"),
                dcc.Slider(
                    id='azimuth-angle-slider',
                    min=0, max=359, step=1, value=90,  # 90 degrees for equatorial orbit
                    marks={0: '0°', 90: '90°', 180: '180°', 270: '270°', 359: '359°'},
                ),
            ], style={'marginBottom': 20}),
            
            html.Div([
                html.Label("Max Simulation Time (s):"),
                dcc.Slider(
                    id='max-time-slider',
                    min=10, max=10000, step=100, value=5000,  # Much longer for orbital period
                    marks={10: '10', 2500: '2.5k', 5000: '5k', 7500: '7.5k', 10000: '10k'},
                ),
            ], style={'marginBottom': 20}),
            
            html.Button('Run Simulation', id='run-button', n_clicks=0, 
                        style={'backgroundColor': '#4CAF50', 'color': 'white', 'padding': '10px 20px', 'border': 'none', 'cursor': 'pointer', 'fontSize': '16px', 'marginTop': '20px'}),
            
            html.Div(id='results-container', style={'marginTop': '20px'}),
        ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'}),
    ]),
    
    dcc.Graph(id='rocket-plot', style={'height': '800px'}),
    
    html.Div([
        html.H3("Animation Control"),
        html.Div([
            html.Button('⏮️ Reset', id='reset-button', n_clicks=0, style={'marginRight': '10px'}),
            html.Button('⏯️ Play/Pause', id='play-button', n_clicks=0, style={'marginRight': '10px'}),
            html.Button('⏭️ Step Forward', id='step-button', n_clicks=0),
        ], style={'marginBottom': '10px'}),
        dcc.Slider(
            id='animation-slider',
            min=0,
            max=100,
            step=1,
            value=0,
            marks={0: '0%', 25: '25%', 50: '50%', 75: '75%', 100: '100%'},
            updatemode='drag'
        ),
    ], style={'marginTop': '20px'}),
    
    # Store simulation data in browser (hidden component)
    dcc.Store(id='simulation-data'),
    dcc.Store(id='animation-state', data={'playing': False, 'frame': 0, 'max_frames': 0, 'interval': 100}),
    dcc.Interval(
        id='animation-interval',
        interval=100,  # in milliseconds
        n_intervals=0,
        disabled=True
    ),
])

@app.callback(
    [Output('simulation-data', 'data'),
     Output('results-container', 'children'),
     Output('animation-slider', 'max'),
     Output('animation-slider', 'marks'),
     Output('animation-slider', 'value'),
     Output('animation-state', 'data')],
    [Input('run-button', 'n_clicks')],
    [State('initial-mass-slider', 'value'),
     State('mass-flow-slider', 'value'),
     State('exhaust-velocity-slider', 'value'),
     State('drag-coeff-slider', 'value'),
     State('cross-area-slider', 'value'),
     State('initial-speed-slider', 'value'),
     State('launch-angle-slider', 'value'),
     State('azimuth-angle-slider', 'value'),
     State('max-time-slider', 'value'),
     State('animation-state', 'data')]
)
def run_simulation(n_clicks, initial_mass, mass_flow_rate, exhaust_velocity, 
                  drag_coefficient, cross_area, initial_speed, launch_angle, 
                  azimuth_angle, max_time, animation_state):
    if n_clicks == 0:
        # Initial state before run button is clicked
        return (None, html.Div(), 100, {0: '0%', 25: '25%', 50: '50%', 75: '75%', 100: '100%'}, 
                0, {'playing': True, 'frame': 0, 'max_frames': 0, 'interval': 100})
    
    # Run simulation
    simulation_data = simulate_rocket(
        initial_mass, mass_flow_rate, exhaust_velocity, drag_coefficient,
        cross_area, initial_speed, launch_angle, azimuth_angle, max_time
    )
    
    # Extract data
    x_traj, y_traj, z_traj, speed_traj, altitude_traj, mass_traj, time_traj, accel_traj, drag_traj, thrust_traj, gravity_traj = simulation_data
    
    # Calculate summary statistics
    max_altitude = max(altitude_traj)
    max_speed = max(speed_traj)
    flight_time = time_traj[-1]
    max_acceleration = max(accel_traj)
    
    # Calculate orbital parameters
    # Determine if orbit was achieved
    orbital_status = "Not achieved"
    if max_altitude > 100000:  # 100 km is usually considered space
        if max(altitude_traj[-100:]) > 100000:  # If still in space at the end
            orbital_status = "Achieved"
    
    # Create results display
    results = html.Div([
        html.H4("Simulation Results:"),
        html.Ul([
            html.Li(f"Maximum Altitude: {max_altitude/1000:.2f} km"),
            html.Li(f"Maximum Speed: {max_speed:.2f} m/s (Orbital velocity at LEO ~7,800 m/s)"),
            html.Li(f"Flight Time: {flight_time:.2f} s"),
            html.Li(f"Maximum Acceleration: {max_acceleration:.2f} m/s²"),
            html.Li(f"Orbit Status: {orbital_status}"),
        ])
    ])
    
    # Set max frames for animation slider (use actual length but step by 5 for efficiency)
    max_frames = len(time_traj) - 1
    step_size = max(1, int(max_frames / 100))  # Divide into 100 steps at most
    
    # Create marks for slider
    marks = {}
    for i in range(0, 101, 25):
        frame_index = int(i * max_frames / 100)
        if frame_index < len(time_traj):
            marks[i] = f"{time_traj[frame_index]:.1f}s"
    
    # Update animation state
    animation_state['max_frames'] = max_frames
    animation_state['frame'] = 0
    animation_state['playing'] = False
    
    # Convert data for storage (transpose to make JSON serialization more efficient)
    return (
        {
            'x': x_traj.tolist(),
            'y': y_traj.tolist(),
            'z': z_traj.tolist(),
            'speed': speed_traj.tolist(),
            'altitude': altitude_traj.tolist(),
            'mass': mass_traj.tolist(),
            'time': time_traj.tolist(),
            'accel': accel_traj.tolist(),
            'drag': drag_traj.tolist(),
            'thrust': thrust_traj.tolist(),
            'gravity': gravity_traj.tolist(),
        },
        results,
        100,  # Always use 0-100 range for slider
        marks,
        0,  # Reset to beginning
        animation_state
    )

# Callback to update the plot based on animation frame
@app.callback(
    Output('rocket-plot', 'figure'),
    [Input('animation-slider', 'value'),
     Input('animation-interval', 'n_intervals')],
    [State('simulation-data', 'data'),
     State('animation-state', 'data')]
)
def update_plot(slider_value, n_intervals, simulation_data, animation_state):
    # Default figure if no data
    if not simulation_data:
        # Create an empty figure with the right structure
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'scene', 'rowspan': 2}, {'type': 'xy'}],
                  [None, {'type': 'xy'}]],
            subplot_titles=["3D Trajectory", "Altitude vs Time", "Speed vs Time"]
        )
        fig.update_layout(
            title="Run simulation to see results",
            height=800
        )
        return fig
    
    # Convert data back to numpy arrays
    x_traj = np.array(simulation_data['x'])
    y_traj = np.array(simulation_data['y'])
    z_traj = np.array(simulation_data['z'])
    speed_traj = np.array(simulation_data['speed'])
    altitude_traj = np.array(simulation_data['altitude'])
    mass_traj = np.array(simulation_data['mass'])
    time_traj = np.array(simulation_data['time'])
    accel_traj = np.array(simulation_data['accel'])
    drag_traj = np.array(simulation_data['drag'])
    thrust_traj = np.array(simulation_data['thrust'])
    gravity_traj = np.array(simulation_data['gravity'])
    
    # Calculate frame index from slider value
    max_frames = animation_state['max_frames']
    
    # Use either slider value or animation state
    if slider_value is not None:
        frame_index = int(slider_value * max_frames / 100)
    else:
        frame_index = animation_state['frame']
    
    # Ensure index is in bounds
    frame_index = min(frame_index, len(time_traj) - 1)
    
    # Create figure with current frame
    sim_data = (
        x_traj, y_traj, z_traj, speed_traj, altitude_traj, 
        mass_traj, time_traj, accel_traj, drag_traj, thrust_traj, gravity_traj
    )
    fig = create_rocket_figure(sim_data, frame_index)
    
    return fig

# Callback to handle animation controls
@app.callback(
    [Output('animation-interval', 'disabled'),
     Output('animation-state', 'data', allow_duplicate=True)],
    [Input('play-button', 'n_clicks'),
     Input('reset-button', 'n_clicks'),
     Input('step-button', 'n_clicks'),
     Input('animation-slider', 'value')],
    [State('animation-state', 'data'),
     State('simulation-data', 'data')],
    prevent_initial_call=True
)
def animation_control(play_clicks, reset_clicks, step_clicks, slider_value, animation_state, simulation_data):
    ctx = dash.callback_context
    if not ctx.triggered or not simulation_data:
        return True, animation_state
    
    # Determine which button was clicked
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'play-button':
        # Toggle play/pause
        animation_state['playing'] = not animation_state['playing']
    
    elif trigger_id == 'reset-button':
        # Reset to beginning
        animation_state['frame'] = 0
        animation_state['playing'] = False
    
    elif trigger_id == 'step-button':
        # Step forward one frame
        animation_state['frame'] = min(animation_state['frame'] + 1, animation_state['max_frames'])
        animation_state['playing'] = False
    
    elif trigger_id == 'animation-slider':
        # Update frame from slider
        animation_state['frame'] = int(slider_value * animation_state['max_frames'] / 100)
        # Don't change play state
    
    return not animation_state['playing'], animation_state

# Callback to update slider based on animation progress
@app.callback(
    Output('animation-slider', 'value', allow_duplicate=True),
    Input('animation-interval', 'n_intervals'),
    [State('animation-state', 'data'),
     State('simulation-data', 'data')],
    prevent_initial_call=True
)
def update_slider_from_animation(n_intervals, animation_state, simulation_data):
    if not simulation_data or not animation_state['playing']:
        raise dash.exceptions.PreventUpdate
    
    # Increment frame
    animation_state['frame'] = min(animation_state['frame'] + 1, animation_state['max_frames'])
    
    # Convert frame to slider value (0-100 range)
    slider_value = int(100 * animation_state['frame'] / animation_state['max_frames'])
    
    # Stop at end
    if animation_state['frame'] >= animation_state['max_frames']:
        animation_state['playing'] = False
    
    return slider_value

if __name__ == '__main__':
    print("Starting Rocket Orbital Trajectory Simulator")
    print("Open your web browser to http://127.0.0.1:8050/ to use the simulator")
    app.run_server(debug=True)