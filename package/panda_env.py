# import numpy as np
# import sys
# from PyQt5.QtWidgets import QApplication
# app = QApplication(sys.argv)

# """
# %run ../../package/init_kinematics.py
# """
# from transforms import ( # type: ignore
#     p2t,
#     pr2t,
#     rpy2r,
#     np_uv,
#     view_in_world,
# )

# """
# %run ../../package/init_qt.py
# """
# from qt_widgets import ( # type: ignore
#     MultiSliderQtWidget,
# )

from ri_motion_v5_package.init_scripts.init_ipython_setup import *
from ri_motion_v5_package.init_scripts.init_qt import *
from ri_motion_v5_package.mujoco_sim import *
from ri_motion_v5_package.kinematics import *
from ri_motion_v5_package.utility import *
from ri_motion_v5_package.qt import *

def get_panda_joint_names():
    """
    Get the joint names of the panda robot.

    Returns:
        list: A list of joint names for the panda robot.
    """
    panda_joints = [
        'panda_joint1','panda_joint2','panda_joint3','panda_joint4',
        'panda_joint5','panda_joint6','panda_joint7',
    ]
    return panda_joints

def get_inspire_joint_names():
    """
    Get the joint names of the inspire hand.

    Returns:
        list: A list of joint names for the inspire hand.
    """
    inspire_joints = [
        'left_thumb_1_joint','left_thumb_2_joint','left_thumb_3_joint','left_thumb_4_joint',
        'left_index_1_joint','left_index_2_joint','left_middle_1_joint','left_middle_2_joint',
        'left_ring_1_joint','left_ring_2_joint','left_little_1_joint','left_little_2_joint',
    ]
    return inspire_joints

def get_T_palm_panda_inspire(env):
    """
    Get the palm pose of the inspire hand on the panda robot in world frame.

    Returns:
        T_palm (np.ndarray): 4x4 homogeneous transformation matrix representing the palm pose of the inspire hand on the panda robot in world frame.
    """
    p_palm        = env.get_p('palm','site')
    p_palm_top    = env.get_p('palm_top','site')
    p_palm_palmar = env.get_p('palm_palmar','site')
    p_palm_front  = env.get_p('palm_front','site')
    ux,uy,uz = np_uv(p_palm_front-p_palm),np_uv(p_palm-p_palm_palmar),np_uv(p_palm_top-p_palm)
    R_palm   = np.vstack((ux,uy,uz)).reshape((3,3)).T
    T_palm   = pr2t(p_palm,R_palm).copy()
    # Return palm pose
    return T_palm

def get_p_offset_palm_to_cylinder():
    """
    Get the offset from the palm to the cylinder center in the palm frame.

    Returns:
        np.ndarray: A 3D vector representing the offset from the palm to the cylinder center in the palm frame.
    """
    return np.array((0.0493,-0.0498,-0.060))

def get_qactive_inspire():
    """
    Get the active joint configuration for the inspire hand.

    Returns:
        np.ndarray: A 1D array representing the active joint configuration for the inspire hand.
    """
    qactive_inspire = [1.16,0.17,0.65,0.65,0.65,0.60]
    return np.array(qactive_inspire)

def get_q_inspire(env):
    """
    Get the full joint configuration for the inspire hand.

    Returns:
        np.ndarray: A 1D array representing the full joint configuration for the inspire hand.
    """
    inspire_joints        = get_inspire_joint_names()
    inspire_active_joints = env.get_active_among_joints(inspire_joints)
    qactive_inspire0      = get_qactive_inspire()
    q_inspire = env.get_qfull_from_qactive(
        full_joints   = inspire_joints,
        active_joints = inspire_active_joints,
        qactive       = qactive_inspire0,
    )
    return q_inspire

"""
Bookshelf environment
"""
def get_q_pandas_bookshelf():
    """
    Get predefined joint configurations for the panda robots in the bookshelf environment.

    Returns:
        dict: A dictionary containing joint configurations for different panda robots.
    """
    q_right_2f = [ 1.41, -1.75, -1.51, -1.85, -2.46,  2.99,  1.51]
    q_right_3f = [ 1.57, -1.44, -1.11, -1.67, -1.77,  2.67,  1.14]
    q_left_2f  = [ 2.49, -1.35, -1.48, -2.45, -2.9 ,  1.35,  2.47]
    q_left_3f  = [ 2.19, -0.67, -0.98, -2.  , -2.11,  1.81,  2.37]
    q_pandas = {
        'right_2f': np.array(q_right_2f),
        'right_3f': np.array(q_right_3f),
        'left_2f' : np.array(q_left_2f),
        'left_3f' : np.array(q_left_3f),
    }
    return q_pandas

def set_panda_bookshelf_env(env,panda_joints,inspire_joints,q_panda=None,q_inspire=None):
    """
    Set up the panda robot environment with a bookshelf and cylinder.

    Parameters:
        env: The environment object to set up.
        panda_joints (list): List of joint names for the panda robot.
        inspire_joints (list): List of joint names for the inspire hand.
        q_panda (np.ndarray, optional): Joint configuration for the panda robot. Defaults to None.
        q_inspire (np.ndarray, optional): Joint configuration for the inspire hand. Defaults to None.
    """
    # Reset environment
    env.reset()

    # Initialize viewer
    env.init_viewer()
    env.viewer.set_transparency(transparent=True)
    env.viewer.set_cam_info(55,2.9,-21,[0.26,-0.18,0.52])

    # Move objects
    env.set_T('body_bookshelf_base','body',pr2t((0.7,-0.6,0),rpy2r([0,0,np.pi])))
    env.set_T('body_bookshelf_deep','body',pr2t((0.7,+0.6,0),rpy2r([0,0,np.pi])))
    env.set_p('body_cylinder','base_body',(0,-0.5,1e-8))

    # Set panda joints
    if q_panda is not None:
        env.forward(q=q_panda,joint_names=panda_joints) # FK panda only

    # Set inspire hand joints
    if q_inspire is not None:
        env.forward(q=q_inspire,joint_names=inspire_joints) # FK inspire only

"""
Cabinet environment
"""
def get_q_pandas_cabinet():
    """
    Get predefined joint configurations for the panda robots in the cabinet environment.

    Returns:
        dict: A dictionary containing joint configurations for different panda robots.
    """
    q_panda_init  = [-1.03,-1.73,1.8,-1.78,-0.27,2.79,2.85]
    q_panda_final = [-1.057,1.122,0.955,-0.944,1.587,2.908,-1.671]
    q_pandas = {
        'init' : np.array(q_panda_init),
        'final': np.array(q_panda_final),
    }
    return q_pandas

def set_panda_cabinet_env(
        env,
        panda_joints,
        inspire_joints,
        q_panda             = None,
        q_inspire           = None,
        width               = 0.6,
        height              = 1.0,
        x_offset            = None,
        y_offset            = None,
        initialize_viewer   = True,
    ):
    """
    Set up the panda robot environment with a cabinet and cylinder.

    Parameters:
        env: The environment object to set up.
        panda_joints (list): List of joint names for the panda robot.
        inspire_joints (list): List of joint names for the inspire hand.
        q_panda (np.ndarray, optional): Joint configuration for the panda robot. Defaults to None.
        q_inspire (np.ndarray, optional): Joint configuration for the inspire hand. Defaults to None.
    """
    # Reset environment
    env.reset()

    # Close viewer if open
    env.close_viewer()

    # Initialize viewer
    if initialize_viewer:
        env.init_viewer(width=width,height=height,x_offset=x_offset,y_offset=y_offset)
        env.viewer.set_transparency(transparent=True)
        env.viewer.set_cam_info(-130.6,2.9,-33,[0.26,-0.17,0.17])

    # Move objects
    env.set_T('body_cabinet_half_closed','body',pr2t((0.5,-0.5,0),rpy2r([0,0,0.5*np.pi])))
    env.set_p('body_cylinder','base_body',(0,-0.5,1e-8))

    # Set panda joints
    if q_panda is not None:
        env.forward(q=q_panda,joint_names=panda_joints) # FK panda only

    # Set inspire hand joints
    if q_inspire is not None:
        env.forward(q=q_inspire,joint_names=inspire_joints) # FK inspire only

"""
Solve IK for panda + inspire to reach targets
"""        
def solve_ik_panda_palm(env,ik_solver,p_palm_trgt,R_palm_trgt):
    """
    Solve inverse kinematics for the panda robot to reach the desired palm target.

    Parameters:
        env: The environment object.
        ik_solver: The IK solver object.
        p_palm_trgt (np.ndarray): 3D position target for the palm.
        R_palm_trgt (np.ndarray): 3x3 rotation matrix target for the palm.

    Returns:
        float: The best IK error achieved.
    """

    # Set IK targets
    p_palm_top_trgt    = p_palm_trgt + 0.1*R_palm_trgt[:,2]
    p_palm_palmar_trgt = p_palm_trgt - 0.1*R_palm_trgt[:,1]
    p_palm_front_trgt  = p_palm_trgt + 0.1*R_palm_trgt[:,0]

    # Append IK targets
    ik_solver.reset_buffers()
    ik_solver.add_ik_target('palm',p_palm_trgt)
    ik_solver.add_ik_target('palm_top',p_palm_top_trgt)
    ik_solver.add_ik_target('palm_palmar',p_palm_palmar_trgt)
    ik_solver.add_ik_target('palm_front',p_palm_front_trgt)
    ik_solver.set_ik_config(max_ik_tick=100,ik_stepsize_rev=0.5) # set ik config

    # Solve IK
    q_rev_best,info = ik_solver.solve_ik(
        env                     = env,
        joints_use              = env.rev_joint_names,
        joint_limit_handle_flag = True,
        nullspace_control_flag  = True,
    )
    ik_err_best,elapsed_time = info['ik_err_best'],info['elapsed_time']

    # Forward to best solution
    env.forward(q=q_rev_best,joint_names=env.rev_joint_names)

    # Return best IK error
    return ik_err_best

def animate_cabinet_env_traj(
        env,
        mode_str,
        traj,
        panda_joints,
        inspire_joints,
        q_pandas,
        q_inspire0,
        p_cylinder_offset0,
        ubuntu_process_events_flag = False,
        app = None,
    ):
    """
    Animate the cabinet environment trajectory.

    Parameters:
        env: The environment object.
        mode_str (str): A string representing the mode of the animation.
        traj (np.ndarray): A 2D array representing the trajectory of joint configurations.
        panda_joints (list): List of joint names for the panda robot.
        inspire_joints (list): List of joint names for the inspire hand.
        q_pandas (dict): A dictionary containing joint configurations for different panda robots.
        q_inspire0 (np.ndarray): Joint configuration for the inspire hand.
        p_cylinder_offset0 (np.ndarray): Offset from the palm to the cylinder center in the palm frame.
    """
    # Trajectory length
    L = traj.shape[0]

    # Initialize env and viewer
    width,height,x_offset,y_offset = 0.8,0.8,0.1,0.15
    set_panda_cabinet_env(
        env,panda_joints,inspire_joints,q_pandas['init'],q_inspire0,
        width,height,x_offset,y_offset,
    )
    sliders = MultiSliderQtWidget(
        title         = "Tick",
        window_width  = 0.8,
        window_height = 0.02,
        x_offset      = 0.1,
        y_offset      = 0.0,
        label_texts   = ['Tick'],
        slider_mins   = [0],
        slider_maxs   = [L-1],
        slider_vals   = [0],
        resolutions   = [1],
    )

    # Loop
    while env.is_viewer_alive():
        # Update
        tick = int(sliders.get_values()[0])
        q = traj[tick,:] # (dim)
        env.forward(q=q,joint_names=panda_joints) # fk panda
        T_palm = get_T_palm_panda_inspire(env) # palm 
        T_cylinder = view_in_world(T=p2t(p_cylinder_offset0),T_wl=T_palm) # cylnder
        env.set_T('body_cylinder','base_body',T_cylinder) # set cylinder pose
        contact_info = env.get_contact_info() # contact info
        min_contact_dist = contact_info['min_contact_dist'] # np.inf or negative
        # Render
        env.render()
        env.viewer_text_overlay("Tick","[%d/%d]"%(tick,L),loc='bottom left')
        env.viewer_text_overlay("Mode",mode_str,loc='top left')
        env.plot_contact_info()
        if ubuntu_process_events_flag:
            app.processEvents()
    
    # Close 
    env.close_viewer()
    sliders.close()
