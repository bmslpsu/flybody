"""Utility functions."""

from typing import Sequence

from numpy import cbrt

from IPython.display import HTML
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt

from dm_control.mujoco.wrapper import MjvOption
from dm_control.mujoco.wrapper.mjbindings import mjlib


def rollout_and_render(env, policy, n_steps=100,
                       run_until_termination=False,
                       camera_ids=[-1],
                       **render_kwargs):
    """Rollout policy for n_steps or until termination, and render video.
    Rendering is possible from multiple cameras; in that case, each element in
    returned `frames` is a list of cameras."""
    if isinstance(camera_ids, int):
        camera_ids = [camera_ids]
    timestep = env.reset()
    frames = []
    i = 0
    while ((i < n_steps and not run_until_termination) or 
           (timestep.step_type != 2 and run_until_termination)):
        i += 1
        frame = []
        for camera_id in camera_ids:
            frame.append(
                env.physics.render(camera_id=camera_id, **render_kwargs))
        frame = frame[0] if len(camera_ids) == 1 else frame  # Maybe squeeze.
        frames.append(frame)
        action = policy(timestep.observation)
        timestep = env.step(action)
    return frames

def rollout_ablate_render(env, policy, n_steps=100, camera_ids=[-1], 
                       cut_ratio=[1.0,1.0], cut_isChordwise=[True,True], cut_time=0.5,
                       **render_kwargs):
    """Rollout policy and render video, identical to `rollout_and_render`. Additionally,
    ablate the wings given these additional parameters:
        cut_ratio: The wing area ratio (WAR) of the left and right wings 
        (damaged_area/original_area)
        cut_isChordwise: Reduces the WAR assuming a chordwise cut for each wing. If 
        false, assumes spanwise
        cut_time: Normalized time of cut, = t_cut / t_total = i_cut / n_steps.
    Since cut_time assumes that n_steps is analogous the total time of this simulation,
    run_until_termination is removed as it will always be logically True (simulation
    should never be reset)"""

    if isinstance(camera_ids, int):
        camera_ids = [camera_ids]
    timestep = env.reset()
    frames = []
    i = 0
    i_cut = int(cut_time * n_steps)

    # **render_kwargs should not contain scene_option, as we will be setting
    # that explicitly here
    if "scene_option" in render_kwargs: del render_kwargs["scene_option"]
    scene_opt = MjvOption()

    while (timestep.step_type != 2):
        i += 1
        frame = []

        # Ablate the wing when on its frame
        if((i-1)==i_cut):
            ablate_wings(env,cut_ratio,cut_isChordwise)
            # To visualize this, also turn off aesthetic mesh at the moment
            # of ablation
            scene_opt.geomgroup = [1,0,0,1,1,0]

        for camera_id in camera_ids:
            frame.append(env.physics.render(
                camera_id=camera_id,
                scene_option=scene_opt,
                **render_kwargs))
        frame = frame[0] if len(camera_ids) == 1 else frame  # Maybe squeeze.
        frames.append(frame)
        action = policy(timestep.observation)
        timestep = env.step(action)
    return frames    

def ablate_wings(env, cut_ratio, cut_isChordwise):
    def inertia_ellipsoid(mass, size): #TODO test
        return [
            mass / 5 * (size[1]**2 + size[2]**2),
            mass / 5 * (size[0]**2 + size[2]**2),
            mass / 5 * (size[0]**2 + size[1]**2)
        ]
    
    GEOM_NAMES = (
        ("walker/wing_left_inertial", "walker/wing_left_fluid"), 
        ("walker/wing_right_inertial","walker/wing_right_fluid"))
    BODY_NAMES = ("walker/wing_left", "walker/wing_right")
    
    # Cycle through left and right wing
    for wing in range(2): 
        # attempt ablation on each wing, left and right
        war = cut_ratio[wing] # wing area ratio

        # EDIT SHAPE
        # Cycle through inertial and fluid geoms to change their shapes
        for geom in GEOM_NAMES[wing]:
            size_old = env.physics.named.model.geom_size[geom][:]
            size_new = size_old.copy()

            # Determine which direction this wing is being cut
            if cut_isChordwise[wing]:
                # Apply the chordwise cut to both inertial and fluid geoms
                size_new[2] *= war

                # SIZE: Reduce the wing geom span
                env.physics.named.model.geom_size[geom][:] = size_new
                # POSITION: Realign the wing proximal edge back to its original position TODO make robust
                # env.physics.named.model.geom_pos[geom][1] += size_new-size_old                      
            
            else:
                # Apply the spanwise cut to both inertial and fluid geoms
                    size_new[1] *= war

                    # Reduce the wing geom chord
                    env.physics.named.model.geom_size[geom][:] = size_new
                    # Realign the wing leading edge back to its original position
                    # This is conceptually more challenging 
                    # env.physics.named.model.geom_pos[geom][:]

        # MASS: Change body parameters to adjust for lost wing mass
        # update total mass
        env.physics.named.model.body_mass[BODY_NAMES[wing]] *= war
    
        # update inertia
        env.physics.named.model.body_inertia[BODY_NAMES[wing]][:] = inertia_ellipsoid(
            mass = env.physics.named.model.body_mass[BODY_NAMES[wing]],
            size = env.physics.named.model.geom_size[GEOM_NAMES[wing][0]][:]
        )
        
        # mass center: TODO add
        
        # everything else
        print(env.task._walker.__dict__)
        mjlib.mj_setConst(m=env.physics.model,d=env.physics.data)

def any_substr_in_str(substrings: Sequence[str], string: str) -> bool:
    """Checks if any of substrings is in string."""
    return any(s in string for s in substrings)


def display_video(frames, framerate=30):
    """
    Args:
        frames (array): (n_frames, height, width, 3)
        framerate (int)
    """
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use(
        'Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    plt.close(
        'all')  # Figure auto-closing upon backend switching is deprecated.
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])

    def update(frame):
        im.set_data(frame)
        return [im]

    interval = 1000 / framerate
    anim = animation.FuncAnimation(fig=fig,
                                   func=update,
                                   frames=frames,
                                   interval=interval,
                                   blit=True,
                                   repeat=False)
    return HTML(anim.to_html5_video())


def parse_mujoco_camera(s: str):
    """Parse `Copy camera` XML string from MuJoCo viewer to pos and xyaxes.
    
    Example input string: 
    <camera pos="-4.552 0.024 3.400" xyaxes="0.010 -1.000 0.000 0.382 0.004 0.924"/>
    """
    split = s.split('"')
    pos = split[1]
    xyaxes = split[3]
    pos = [float(s) for s in pos.split()]
    xyaxes = [float(s) for s in xyaxes.split()]
    return pos, xyaxes
