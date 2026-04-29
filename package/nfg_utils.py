import numpy as np
import sys
sys.path.append('../../kinematics')
from transforms import rpy2r

""" UTILITY FUNCTIONS FOR NFG PROJECT """

def inv_T(T):

    R = T[:3, :3]
    p = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ p
    return T_inv

def env_initialize(env, joints_to_use, forwardbackward = "forward", q=None, robot="panda"):
    env.reset()
    # set position of wall, and bookshelf
    env.set_p('body_bookshelf_deep', 'body', p=(-0.5, 0, 1.01))
    # set position of robot, cylinder

    if robot == "panda":
        env.set_p('panda_link_0', 'body', p=(0.1,0.5,1.0)) # move robot base
        env.set_R('panda_link_0', 'body', R = rpy2r([0, 0, -1.57])) # move robot base
    elif robot == "doosan":
        env.set_p('robot0:base_link', 'body', p=(0.1,0.5,1.0)) # move robot base
        env.set_R('robot0:base_link', 'body', R = rpy2r([0, 0, -1.57])) # move robot base
    env.set_p('body_cylinder_1', "base_body", p=(+0.7,0,1.0)) # move object
    env.set_R('body_cylinder_1', "base_body", R = rpy2r([0, 0, 1.57])) # move object

    if robot=="panda":
        env.forward(q=[1.2],joint_names=['left_thumb_1_joint']) # thumb down, -1.16
    if robot=="doosan":
        env.forward(q=[-1.2],joint_names=['rj_dg_1_2']) # thumb down, -1.16

    if q is not None:
        env.forward(q=q,joint_names=joints_to_use)
    else:
        if robot=="doosan": 
            if forwardbackward == "forward":
                env.forward(q=[-0.8, 2.37, 1.57],joint_names=['robot0:joint_2', 'robot0:joint_3', 'robot0:joint_6'])
            elif forwardbackward == "backward":
                env.forward(q=[0.8, -2.37, -1.57],joint_names=['robot0:joint_2', 'robot0:joint_3', 'robot0:joint_6'])
        elif robot=="panda":
            if forwardbackward == "forward":
                env.forward(q=[0.0, -0.5, 0.0, -1.5, 0.0, 1.0, 2.3],joint_names=['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'])
            elif forwardbackward == "backward":
                env.forward(q=[-0.8, -1.0, 0.0, 1.5, 0.0, -1.0, 0.0],joint_names=['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'])


def map_q_env2warp(env, q_nworld, joints_to_use): # q_nworld: (n_traj x DOF)
    """ FROM WORLD X ENV_Q TO WORLD X WARP_QPOS """
    joint_ids = [env.model.joint(name).qposadr[0] for name in joints_to_use]
    # if q_nworld.ndim == 1:
    #     warp_qpos = np.zeros((env.model.nq), dtype=np.float64)
    # elif q_nworld.ndim == 2:
    warp_qpos = np.zeros((len(q_nworld), env.model.nq), dtype=np.float64)

    for j_idx, joint_id in enumerate(joint_ids):
        # if q_nworld.ndim == 1:
        #     warp_qpos[joint_id] = q_nworld[j_idx]
        # elif q_nworld.ndim == 2:
        warp_qpos[:, joint_id] = q_nworld[:, j_idx]
        # else:
        #     raise ValueError("q_nworld must be 1D or 2D array")
    return warp_qpos

def extract_min_dist(raw_data, shape = [9, 5, 7]):
    """ return: length minimum contact distances of each world """
    n_world = len(raw_data)
    reshaped_data = raw_data.reshape(n_world, shape[0], shape[1], shape[2])
    min_contact_dist_list = reshaped_data[:, :, :, 0]  # min_contact_dist is at index 0
    reshaped_min_cont = min_contact_dist_list.reshape(n_world, -1)
    min_val = np.min(min_contact_dist_list.reshape(n_world, -1), axis=1)

    return min_val




def get_trajs_topk_nd(
        trajs,
        scores,
        k,
        largest = True,   # False: smaller score is better (default)
    ):
    """
    variation of get trajs topk

    input
        - trajs: (L,n_traj,dof) ndarray
        - scores: (n_traj,) ndarray
        - k: number of top trajectories to select
        - largest: bool, whether to select largest scores (True) or smallest scores (False
    output
        - trajs_topk: (L,k,dof) ndarray
        - scores_topk: (k,) ndarray
    """
    trajs = np.asarray(trajs)
    scores = np.asarray(scores)

    if trajs.ndim != 3:
        raise ValueError("trajs must have shape (L,n_traj,dof)")
    if scores.ndim != 1:
        raise ValueError("scores must have shape (n_traj,)")

    n_traj = scores.shape[0]
    if k > n_traj:
        raise ValueError("k must be <= number of trajectories")

    # argsort: ascending
    order = np.argsort(scores)

    if largest:
        idx_topk = order[-k:][::-1]   # largest first
    else:
        idx_topk = order[:k]          # smallest first

    trajs_topk = trajs[:,idx_topk, :]
    scores_topk = scores[idx_topk]

    return trajs_topk,scores_topk


# =========================================================
# env initialize 

# Configuration
panda_joints = [
    'panda_joint1','panda_joint2','panda_joint3','panda_joint4',
    'panda_joint5','panda_joint6','panda_joint7',
]
inspire_joints = [
    'left_thumb_1_joint','left_thumb_2_joint','left_thumb_3_joint','left_thumb_4_joint',
    'left_index_1_joint','left_index_2_joint','left_middle_1_joint','left_middle_2_joint',
    'left_ring_1_joint','left_ring_2_joint','left_little_1_joint','left_little_2_joint',
]

q_start = [-1.03,-1.73,1.8,-1.78,-0.27,2.79,2.85]
q_end = [-1.057,1.122,0.955,-0.944,1.587,2.908,-1.671]
q_panda_init = np.array([0.0,-0.5,0.0,-1.5,0.0,1.0, 2.3])
qactive_inspire0 = [1.16,0.17,0.65,0.65,0.65,0.60]
q_inspire_init = [1.16,0.17,0.14,0.0,0.65,0.7,0.65,0.7,0.65,0.7,0.6,0.65]
p_cylinder_offset0 = np.array((0.0493,-0.0498,-0.060))


def env_initialize(env, cabinet_body_name = "body_cabinet_quarter_closed", q=None):
    # env.set_p('body_cylinder', 'base_body', [0.5, 0.5, 0.0001])
    env.set_p(cabinet_body_name, 'body', [0.5, -0.5, 0.0])
    env.set_R(cabinet_body_name, 'body', R=rpy2r([0,0,np.pi/2]))
    if q is not None:
        env.forward(q=q, joint_names=panda_joints)
        env.forward(q=q_inspire_init, joint_names=inspire_joints)
    else:
        env.forward(q=q_panda_init, joint_names=panda_joints)
        env.forward(q=q_inspire_init, joint_names=inspire_joints)