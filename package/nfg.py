from ri_motion_v5_package.init_scripts.init_ipython_setup import *
from ri_motion_v5_package.init_scripts.init_qt import *
from ri_motion_v5_package.mujoco_sim import *
from ri_motion_v5_package.kinematics import *
from ri_motion_v5_package.utility import *
from ri_motion_v5_package.qt import *
# trajectory tools 
from ri_motion_v5_package.traj_optimization.traj_utils import *
from ri_motion_v5_package.traj_optimization.traj_shaper import *
# gp tools 
from ri_motion_v5_package.gaussian_process.gp_utils import *
from ri_motion_v5_package.gaussian_process.kernels import rbf_kernel


# ============================================================================================

def filter_contact_info_with_body_idxs(
        contact_info,
        body_idxs,
        manual_include_body_idxs = None,
        ):
    """
    Filter out contact info according to bodies & geoms of interest.
    function naming -> understandable (robot contact naming)
    input argument -> robot names
    """

    body_idxs.extend(manual_include_body_idxs) if manual_include_body_idxs is not None else None
    body_ids = body_idxs
    # 1. initialize variables
    n_contact_filtered = 0
    contact_list_filtered = []
    p_contact_list_filtered = []
    geom1_idx_list_filtered = []
    geom2_idx_list_filtered = []
    body1_idx_list_filtered = []
    body2_idx_list_filtered = []
    R_frame_list_filtered = []
    f_world_list_filtered = []
    norm_dir_list_filtered = []
    min_contact_dist_filtered = 0.0
    # 2. check contact info
    for idx, (body1_id, body2_id, geom1_id, geom2_id) in enumerate(zip(
        contact_info['body1_idx_list'],
        contact_info['body2_idx_list'],
        contact_info['geom1_idx_list'],
        contact_info['geom2_idx_list']
    )):
        if (body1_id in body_ids) or (body2_id in body_ids):
            n_contact_filtered += 1
            contact_list_filtered.append(contact_info['contact_list'][idx])
            p_contact_list_filtered.append(contact_info['p_contact_list'][idx])
            geom1_idx_list_filtered.append(geom1_id)
            geom2_idx_list_filtered.append(geom2_id)
            body1_idx_list_filtered.append(body1_id)
            body2_idx_list_filtered.append(body2_id)
            R_frame_list_filtered.append(contact_info['R_frame_list'][idx])
            f_world_list_filtered.append(contact_info['f_world_list'][idx])
            norm_dir_list_filtered.append(contact_info['norm_dir_list'][idx])
            if contact_info['min_contact_dist'] < min_contact_dist_filtered:
                min_contact_dist_filtered = contact_info['min_contact_dist']
    contact_info_filtered =  {
        "n_contact": n_contact_filtered,
        "contact_list": contact_list_filtered,
        "p_contact_list": p_contact_list_filtered,
        "geom1_idx_list": geom1_idx_list_filtered,
        "geom2_idx_list": geom2_idx_list_filtered,
        "body1_idx_list": body1_idx_list_filtered,
        "body2_idx_list": body2_idx_list_filtered,
        "R_frame_list": R_frame_list_filtered,
        "f_world_list": f_world_list_filtered,
        "norm_dir_list": norm_dir_list_filtered,
        "min_contact_dist": min_contact_dist_filtered
    }
    return contact_info_filtered

# ============================================================================================

def filter_contact_info_with_body_idxs_obj_grasp_contact_exclude(
        contact_info,
        robot_body_idxs,
        obj_idxs,
        ):
    """
    Filter out contact info according to bodies & geoms of interest.
    robot body idxs to robot_body_idxs, object body idxs to obj_idxs
    """
    body_idxs_full = robot_body_idxs + obj_idxs
    # 1. initialize variables
    n_contact_filtered = 0
    contact_list_filtered = []
    p_contact_list_filtered = []
    geom1_idx_list_filtered = []
    geom2_idx_list_filtered = []
    body1_idx_list_filtered = []
    body2_idx_list_filtered = []
    R_frame_list_filtered = []
    f_world_list_filtered = []
    norm_dir_list_filtered = []
    min_contact_dist_filtered = 0.0
    # 2. check contact info
    for idx, (body1_id, body2_id, geom1_id, geom2_id) in enumerate(zip(
        contact_info['body1_idx_list'],
        contact_info['body2_idx_list'],
        contact_info['geom1_idx_list'],
        contact_info['geom2_idx_list']
    )):
        if (body1_id in body_idxs_full) or (body2_id in body_idxs_full):
            if body1_id in obj_idxs:
                if body2_id in robot_body_idxs:
                    pass
            if body2_id in obj_idxs:
                if body1_id in robot_body_idxs:
                    pass
            else:
                n_contact_filtered += 1
                contact_list_filtered.append(contact_info['contact_list'][idx])
                p_contact_list_filtered.append(contact_info['p_contact_list'][idx])
                geom1_idx_list_filtered.append(geom1_id)
                geom2_idx_list_filtered.append(geom2_id)
                body1_idx_list_filtered.append(body1_id)
                body2_idx_list_filtered.append(body2_id)
                R_frame_list_filtered.append(contact_info['R_frame_list'][idx])
                f_world_list_filtered.append(contact_info['f_world_list'][idx])
                norm_dir_list_filtered.append(contact_info['norm_dir_list'][idx])
                if contact_info['min_contact_dist'] < min_contact_dist_filtered:
                    min_contact_dist_filtered = contact_info['min_contact_dist']
    contact_info_filtered =  {
        "n_contact": n_contact_filtered,
        "contact_list": contact_list_filtered,
        "p_contact_list": p_contact_list_filtered,
        "geom1_idx_list": geom1_idx_list_filtered,
        "geom2_idx_list": geom2_idx_list_filtered,
        "body1_idx_list": body1_idx_list_filtered,
        "body2_idx_list": body2_idx_list_filtered,
        "R_frame_list": R_frame_list_filtered,
        "f_world_list": f_world_list_filtered,
        "norm_dir_list": norm_dir_list_filtered,
        "min_contact_dist": min_contact_dist_filtered
    }
    return contact_info_filtered

# ============================================================================================\

def sample_trajs_multidim_uniform_coef_deleted(K_chol,n_traj=100,d=3,use_uniform=False,seed=None):
    """
    Sample multi-dimensional trajectories from a Gaussian process with given Cholesky factor of covariance.

    Parameters:
        K_chol (np.ndarray): Cholesky factor of the covariance matrix (L such that K = L L^T).
        n_traj (int): Number of trajectories to sample.
        d (int): Dimensionality of each trajectory.
        use_uniform (bool): If True, use uniform random variables instead of normal.
        seed (int or None): Random seed for reproducibility.

    Returns:
        trajs (np.ndarray): Sampled trajectories of shape (L, n_traj, d).
    """
    # Set seed
    if seed is not None:
        np.random.seed(seed=seed)

    # Get size
    L = K_chol.shape[0]

    if use_uniform:
        U = np.random.uniform(low=0.0,high=1.0,size=(L,n_traj,d))
        Z = np.sqrt(12.0)*(U - 0.5)
        # Z = (U - 0.5)
    else:
        Z = np.random.randn(L,n_traj,d)

    trajs = np.zeros((L,n_traj,d))
    for k in range(d):
        trajs[:,:,k] = safe_matmul(K_chol,Z[:,:,k]) # shape: (L,n_traj)

    return trajs # shape: (L,n_traj,d)


# ============================================================================================

def get_smooth_traj_from_anchors(
        anchors,
        freq=100,
        time=5.0,
        ):
    """
    Get smooth trajectory from anchor points.
    Args:
        anchors: List of anchor points (N x dim)
        freq: Frequency of the trajectory (Hz)
        time: Total time of the trajectory (s)
    Returns:
        L: Length of the trajectory
        times: Time stamps of the trajectory (L,)
        trajs_smt: Smoothed trajectory (L x dim)
    """
    if len(anchors) == 0:
        raise ValueError("Anchors not set. Please set anchors before initializing trajectory.")
    anchor_diffs            = anchors[1:]-anchors[:-1]
    seg_d                   = np.linalg.norm(anchor_diffs,ord=2,axis=1) # (n_anchors-1,), L2 norm
    velocity                = np.sum(seg_d)/time
    n_anchor,dim            = anchors.shape[0],anchors.shape[1]
    # linear interpolation 
    res = get_interp_const_vel_traj_nd(
        anchors = anchors, # (M x dim)
        vel     = velocity, # maximum velocity
        Hz      = freq,
    )
    times        = res['times_interp'] # (L)
    trajs_lin    = res['anchors_interp'] # (L x dim)
    times_anchor = res['times_anchor'] # (M,)
    idxs_anchor  = res['idxs_anchor'] # (M,)
    L,t_max = times.shape[0],times[1]
    # Smooth joint trajectory
    trajs_smt = np.zeros_like(trajs_lin) # (L x dim)
    for d_idx in range(dim):
        res = traj_1d_shaper(
            t        = times, # (L,)
            x_ref    = trajs_lin[:,d_idx], # (L,)
            idxs_eq  = idxs_anchor, # (M,)
            vals_eq  = anchors[:,d_idx], #(M,)
            v_init   = 0.0,
            v_final  = 0.0,
            lambda_j = 1e-6,
        )
        trajs_smt[:,d_idx] = res['z'] # (L,)
    
    print_green(f"[NFG] Generated smooth trajectory from anchors.")
    return L, times, trajs_smt

# ============================================================================================

def get_covariance_matrix_time_hilbert_space(
        times,
        L,
        length_scale_coef = 0.5,
        variance = 1.0,
        kernel = rbf_kernel
):
    """ 
    Get t_out and K_chol for Hilbert space projection.
    Args:
        times: Time stamps of the trajectory (L,)
        L: Length of the trajectory
        length_scale: Length scale for the RBF kernel
        variance: Variance for the RBF kernel
        kernel: Kernel function (default: RBF kernel)
    Returns:
        K_chol: Cholesky decomposition of the kernel matrix (L x L)
        t_out: Time stamps for the output trajectory (L,)   
    """
    t_max  = times[-1]
    t_in   = np.array([0,t_max])
    t_out  = np.linspace(0.0,t_max,L)
    length_scale = length_scale_coef * t_max
    hyp    = {'length_scale':length_scale,'variance':variance}
    K_chol = get_schur_K_chol(t_in,t_out,kernel,hyp, eps = 1e-10)

    print_green(f"[NFG] Generated covariance matrix for Hilbert space projection.")
    return K_chol, t_out

# ============================================================================================

def sample_epsilon_squash_trajs_multidim(
        traj,
        K_chol,
        n_traj,
        dim,
        q_min,
        q_max,
        use_uniform = True,
        seed=1,
        squash_margin = 0.0,
):
    """
    Sample trajectories from the Gaussian Process and squash them with joint limits.
    Args:
        K_chol: Cholesky decomposition of the kernel matrix (L x L)
        n_traj: Number of trajectories to sample
        dim: Dimension of the trajectory (number of joints)
        q_min: Minimum joint limits (dim,)
        q_max: Maximum joint limits (dim,)
    Returns:
        epsilon: Sampled perturbations (n_traj x L x dim)
        trajs_squash: Squashed trajectories (n_traj x L x dim)
    """
    epsilon = sample_trajs_multidim_uniform_coef_deleted(
        K_chol      = K_chol,
        n_traj      = n_traj,
        d           = dim,
        use_uniform = use_uniform,
        seed        = seed, 
    ) 
    traj_perturb = traj[:,None,:] + epsilon
    traj_squash = soft_squash_multidim(
        traj_perturb,
        x_min=q_min,
        x_max=q_max,
        margin=squash_margin,
        dim_axis=2,
    )# L x n_traj x dof

    print_green(f"[NFG] Sampled and squashed trajectories.")
    return epsilon, traj_squash

# ============================================================================================

def calculate_motion_score(traj):
    """
    Calculate trajectory length (= motion score) for all trajectories.
    Args:
        traj: Trajectory to calculate motion score for (L x dim)
    Returns:
        motion_score: length of trajectories (dim)
    """
    traj_diff = np.diff(traj,axis=0)
    traj_length = np.sum(np.linalg.norm(traj_diff,axis=0), axis=1)
    return traj_length

def calculate_contact_score(contact_info):
    """
    Calculate contact score for all trajectories.
    """
    n_contact = contact_info['n_contact'] 
    min_contact_dist = contact_info['min_contact_dist']
    if min_contact_dist > 0: score = 0.0
    else: score = n_contact*min_contact_dist
    return score

# ============================================================================================

def update_traj_with_calculated_gradient(
    traj,
    epsilon,
    scores,
    contact_gain=10.0,
    motion_gain=5.0,
    topk=5,
    power_factor=20,
    trim=True,
    step_size=0.5,
):
    """
    Update the trajectory with the calculated gradient.
    Args:
        traj: Current trajectory (L x dim)
        epsilon: Perturbations for all trajectories (L x n_traj x dim)
        contact_score: Contact scores for all trajectories (n_traj,)
        motion_score: Motion scores for all trajectories (n_traj,)
        contact_gain: Gain for contact score (default: 10.0)
        motion_gain: Gain for motion score (default: 5.0)
        topk: Number of top trajectories to consider for the update (default: 5)
        power_factor: Power factor for weighting the trajectories (default: 20)
        trim: Whether to use trim_scale or fix_scale for gradient scaling (default: True)
        step_size: Step size for gradient scaling (default: 0.5)
    Returns:
        traj_update: Updated trajectory (L x dim)
        gradient: Calculated gradient before scaling (L x dim)
    """
    topk_idx       = np.argsort(scores)[-topk:]
    epsilon_topk   = epsilon[:, topk_idx, :]     # (L, k, n_joints)
    scores_topk    = scores[topk_idx]
    score_sum      = np.sum(scores_topk)/len(scores_topk)
    score_shifted = scores_topk-scores_topk.max()
    score_pt = power_transform(score_shifted, T=0.1, p=power_factor, mode='exp')
    weight = score_pt / score_pt.sum()
    weighted_epsilon = epsilon_topk * weight[np.newaxis, :, np.newaxis]
    # calculate gradient and update 
    gradient = np.sum(weighted_epsilon, axis=1) # L x n x dof
    if trim:
        gradient_fixed = trim_scale(gradient, th=step_size)
        original_gradient = gradient_fixed.copy()
    else:
        gradient_fixed = fix_scale(gradient, th=step_size)
        original_gradient = gradient_fixed.copy()

    traj_update = traj + gradient_fixed
    print_green(f"[NFG] Updated trajectory with calculated gradient.")
    return traj_update, gradient_fixed

# ============================================================================================
# trajectory utils

def plot_traj(
            self,
            traj, # [L x 3] for (x,y,z) sequence or [L x 2] for (x,y) sequence
            rgba          = (1,0,0,1),
            plot_line     = False,
            plot_cylinder = True,
            plot_sphere   = False,
            cylinder_r    = 0.01,
            sphere_r      = 0.025,
            cmap_name     = None, # 'viridis', 'YlOrRd', 'YlGnBu'
            alpha         = 0.5,
        ):
        """
        Plot a trajectory given by a sequence of points with various visualization options.

        Parameters:
            traj (np.ndarray): 
                Trajectory points with shape [L x 3] or [L x 2], where L is the trajectory length. 
                If shape is [L x 2], z-coordinate is automatically set to zero.
            
            rgba (tuple): 
                Default RGBA color tuple for plotting (used when cmap_name is None). 
                Example: (1,0,0,1) for red with full opacity.

            plot_line (bool): 
                Whether to plot lines connecting sequential points in the trajectory.

            plot_cylinder (bool): 
                Whether to plot cylinders between sequential points, representing thicker segments.

            plot_sphere (bool): 
                Whether to plot spheres at each point in the trajectory.

            cylinder_r (float): 
                Radius of the cylinders plotted between trajectory points.

            sphere_r (float): 
                Radius of spheres plotted at trajectory points.

            cmap_name (str or None): 
                Name of a matplotlib colormap (e.g., 'viridis', 'YlOrRd', 'YlGnBu'). 
                If provided, the colors gradually change along the trajectory length, overriding `rgba`.

            alpha (float): 
                Transparency value between 0 (fully transparent) and 1 (fully opaque). 
                Used only when `cmap_name` is specified.

        Returns:
            None

        Example usage:
            >>> traj = np.array([[0,0], [1,1], [2,2]])
            >>> plot_traj(traj, plot_line=True, cmap_name='viridis', alpha=0.8)
        """
        L = traj.shape[0]
        if cmap_name is not None: cmap = plt.get_cmap(cmap_name)
        for idx in range(L-1):
            p_fr = traj[idx,:]
            p_to = traj[idx+1,:]
            if len(p_fr) == 2: p_fr = np.append(p_fr,[0])
            if len(p_to) == 2: p_to = np.append(p_to,[0])

            if cmap_name is not None:
                color_ratio = idx / (L - 1)
                rgb = cmap(color_ratio)[:3]
                rgba = list(rgb) + [alpha]

            if plot_line:
                self.plot_line_fr2to(p_fr=p_fr,p_to=p_to,rgba=rgba)
            if plot_cylinder:
                self.plot_cylinder_fr2to(p_fr=p_fr,p_to=p_to,r=cylinder_r,rgba=rgba)
        
        if plot_sphere:
            for idx in range(L):

                if cmap_name is not None:
                    color_ratio = idx / (L - 1)
                    rgb = cmap(color_ratio)[:3]
                    rgba = list(rgb) + [alpha]

                p = traj[idx,:]
                self.plot_sphere(p=p,r=sphere_r,rgba=rgba)


""" ROTATION UTILS """
def get_R_franka_lpalm(env):
    p_palm        = env.get_p('lpalm','site')
    p_palm_top    = env.get_p('lpalm_top','site')
    p_palm_palmar = env.get_p('lpalm_palmar','site')
    p_palm_front  = env.get_p('lpalm_front','site')
    ux,uy,uz      = np_uv(p_palm_front-p_palm),np_uv(p_palm-p_palm_palmar),np_uv(p_palm_top-p_palm)
    R_franka_lpalm = np.column_stack((ux,uy,uz)).reshape((3,3))
    return R_franka_lpalm
def get_R_doosan_rpalm(env):
    p_palm        = env.get_p('rpalm','site')
    p_palm_top    = env.get_p('rpalm_top','site')
    p_palm_palmar = env.get_p('rpalm_palmar','site')
    p_palm_front  = env.get_p('rpalm_front','site')
    ux,uy,uz      = np_uv(p_palm_front-p_palm),np_uv(p_palm_palmar-p_palm),np_uv(p_palm_top-p_palm)
    R_doosan_rpalm = np.column_stack((ux,uy,uz)).reshape((3,3))
    return R_doosan_rpalm
print ("Ready.")