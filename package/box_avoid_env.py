import numpy as np
import matplotlib.pyplot as plt

class SimpleBoxAvoidEnv:
    """
    Environment to define box-shaped obstacles and compute scores for trajectories.
    """
    def __init__(
            self,
            t_min     = 0.0,
            t_max     = 1.0,
            L         = 100,
            score_type = 'min',
        ):
        """
        Initialize the BoxAvoidEnv.

        Parameters:
            t_min (float): Minimum time value.
            t_max (float): Maximum time value.
            L (int): Number of time steps.
            score_type (str): Type of score to compute ('avg' or 'min').
        """
        self.t_min     = t_min
        self.t_max     = t_max
        self.L         = L
        self.score_type = score_type
        self.ts        = np.linspace(self.t_min,self.t_max,self.L)
        self.boxes     = []

        # Clear any existing boxes
        self.clear_boxes()

    def add_box(self,t0,t1,y0,y1):
        """
        Add a box-shaped obstacle.

        Parameters:
            t0 (float): Start time of the box.
            t1 (float): End time of the box.
            y0 (float): Lower y-bound of the box.
            y1 (float): Upper y-bound of the box.
        """
        self.boxes.append({'t0':float(t0),'t1':float(t1),'y0':float(y0),'y1':float(y1)})

    def clear_boxes(self):
        """
        Clear all defined boxes.
        """
        self.boxes = []

    # ------------------------------------------------------------
    # score computation
    # ------------------------------------------------------------
    def get_score_of_traj(self,traj):
        """
        Compute the score of a single trajectory.

        Parameters:
            traj (np.ndarray): 1D array of trajectory values at each time step (length L).

        Returns:
            float: score of the trajectory (negative if it intersects any box, 0 otherwise).
        """
        scores = np.zeros(self.L)
        for tick in range(self.L):
            t = self.ts[tick]
            y = traj[tick]
            minimum_penetration = 0.0  # best = 0, worse = negative
            for b in self.boxes: # iterate over boxes
                t0 = b['t0']
                t1 = b['t1']
                y0 = b['y0']
                y1 = b['y1']
                if (t>=t0) and (t<=t1) and (y>=y0) and (y<=y1): # if inside box
                    current_penetration = -min(y-y0,y1-y)  # <= 0
                    if current_penetration < minimum_penetration: # track the minimum penetration
                        minimum_penetration = current_penetration
            scores[tick] = minimum_penetration # minimum penetration at this time step (tick)

        # Average / Min over time (these are <= 0)
        avg_score = float(np.mean(scores))
        min_score = float(np.min(scores))

        """
        Return a bonus if no penetration, else return penalty based on score_type.
        """
        if np.all(scores==0.0): # if no penetration at all time steps, return bonus
            dy    = np.diff(traj)
            tv    = float(np.sum(np.abs(dy)))
            bonus = 1.0/(1.0+tv)           # in (0,1], max at tv=0
            return bonus
        else: # if penetration occurred, return penalty
            if self.score_type == 'avg': # average penetration
                return avg_score
            elif self.score_type == 'min': # minimum penetration
                return min_score
            else:
                raise ValueError("Invalid score_type:[%s]"%(self.score_type))
        
    def get_scores_of_trajs(self,trajs):
        """
        Compute the scores of multiple trajectories.

        Parameters:
            trajs (np.ndarray): 2D array of trajectories, each column is a trajectory.

        Returns:
            np.ndarray: Array of scores for each trajectory.
        """
        n_traj = trajs.shape[1]
        scores  = np.zeros(n_traj)
        for t_idx in range(n_traj):
            scores[t_idx] = self.get_score_of_traj(trajs[:,t_idx])
        return scores

    # ------------------------------------------------------------
    # Box + trajectoryplot
    # ------------------------------------------------------------
    def plot(
            self,
            trajs            = None,
            scores           = None,
            traj             = None,
            score_feasible   = None,
            figsize          = (6,3),
            rgba_feasible    = (0,0,1,1),
            rgba_box         = (0.8,0.8,0,1.0),
            rgba_traj        = (0,0,0,1.0),
            alpha_box        = 0.5,
            cmap_name        = 'coolwarm',
            alpha_infeasible = 1.0,
            lw_infeasible    = 0.5,
            lw_feasible      = 1.5,
            lw_traj          = 2.0,
            fs_label         = 8,
            title_str        = None,
            fs_title         = 10,
            zorder_line      = 2,
            zorder_box       = 5,
            zorder_traj      = 10,
            ylims            = None,
            color_single     = 'k',
            show_flag        = True,
            return_rgb_flag  = False,
            rgba_infeasible  = None,
        ):
        """
        Plot the box avoid environment with optional trajectories.

        Parameters:
            trajs (np.ndarray): 2D array of trajectories to plot, each column is a trajectory.
            scores (np.ndarray): 1D array of scores corresponding to trajs.
            traj (np.ndarray): 1D array of a single trajectory to plot.
            score_feasible (float): score threshold for feasibility. If None, no threshold is applied.
            figsize (tuple): Figure size.
            rgba_feasible (tuple): RGBA color for feasible trajectories.
            rgba_box (tuple): RGBA color for boxes.
            rgba_traj (tuple): RGBA color for the single trajectory.
            alpha_box (float): Alpha transparency for boxes.
            cmap_name (str): Colormap name for infeasible trajectories.
            alpha_infeasible (float): Alpha transparency for infeasible trajectories.
            lw_infeasible (float): Line width for infeasible trajectories.
            lw_feasible (float): Line width for feasible trajectories.
            lw_traj (float): Line width for the single trajectory.
            fs_label (int): Font size for axis labels.
            title_str (str): Title string for the plot.
            fs_title (int): Font size for the title.
            zorder_line (int): Z-order for trajectory lines.
            zorder_box (int): Z-order for boxes.
            zorder_traj (int): Z-order for the single trajectory.
            ylims (tuple): Y-axis limits as (ymin, ymax).
            color_single (str): Color for single trajectory plot.  
            show_flag (bool): Whether to display the plot.
            return_rgb_flag (bool): Whether to return the plot as an RGB array.
        """

        # -----------------------------
        # Normalize trajs -> Y (L,n_traj)
        # -----------------------------
        Y = None
        n_traj = 0
        is_single = False

        if trajs is not None:
            Y = np.asarray(trajs,dtype=np.float64)
            if Y.ndim == 1:
                Y = Y.reshape(self.L,1)
                is_single = True
            elif Y.ndim == 2:
                n_traj = Y.shape[1]
            else:
                raise ValueError("trajs must be 1D or 2D array")
            n_traj = Y.shape[1]
            if n_traj == 1:
                is_single = True

        # -----------------------------
        # Normalize scores -> c (n_traj,)
        # -----------------------------
        c = None
        if scores is not None:
            c = np.asarray(scores,dtype=np.float64)
            if c.ndim == 0:
                c = c.reshape(1,)
                is_single = True
            else:
                c = c.reshape(-1)
        elif Y is not None:
            # compute scores if trajs provided
            if is_single:
                c = np.array([self.get_score_of_traj(Y[:,0])],dtype=np.float64)
            else:
                c = self.get_scores_of_trajs(Y)

        # -----------------------------
        # Plotting
        # -----------------------------
        plt.figure(figsize=figsize)

        # boxes
        for b in self.boxes:
            t0=b['t0']; t1=b['t1']; y0=b['y0']; y1=b['y1']
            plt.fill_between(
                [t0,t1],
                [y0,y0],
                [y1,y1],
                color     = rgba_box,
                alpha     = alpha_box,
                edgecolor = (0.0,0.0,0.0,1.0),
                linewidth = 0.5,
                zorder    = zorder_box,
            )

        # Multiple trajectories
        if Y is not None and c is not None:
            if c.shape[0] != Y.shape[1]:
                raise ValueError("scores length must match n_traj")

            # single: always black (or user-specified)
            if is_single:
                plt.plot(
                    self.ts,
                    Y[:,0],
                    lw     = lw_feasible,
                    color  = color_single,
                    zorder = zorder_line,
                )

            else:
                cmap = plt.get_cmap(cmap_name)

                # Case 1: no feasibility semantics (all cmap-colored)
                if score_feasible is None:
                    cmin = float(np.min(c))
                    cmax = float(np.max(c))
                    den  = (cmax-cmin) if (cmax>cmin) else 1.0
                    rgba_list = []

                    for i in range(Y.shape[1]):
                        cnorm = (float(c[i])-cmin)/den
                        cnorm = 1.0 - np.clip(cnorm,0.0,1.0)  # low score -> red, high -> blue
                        rgba  = cmap(cnorm)
                        if rgba_infeasible is not None:
                            rgba = rgba_infeasible
                        plt.plot(self.ts,Y[:,i],lw=lw_infeasible,color=rgba,zorder=zorder_line)
                        rgba_list.append(rgba)

                # Case 2: feasibility semantics (feasible fixed color, infeasible cmap)
                else:
                    idx_feasible   = np.where(c >= score_feasible)[0]
                    idx_infeasible = np.where(c <  score_feasible)[0]

                    if idx_infeasible.size > 0:
                        cmin = float(np.min(c[idx_infeasible]))
                        cmax = float(score_feasible)
                        den  = (cmax-cmin) if (cmax>cmin) else 1.0
                        cnorm = 1.0 - (c[idx_infeasible]-cmin)/den
                    else:
                        cnorm = np.array([],dtype=np.float64)

                    # infeasible
                    for k,i in enumerate(idx_infeasible):
                        rgba = cmap(float(np.clip(cnorm[k],0.0,1.0)))
                        rgba = (rgba[0],rgba[1],rgba[2],alpha_infeasible)
                        if rgba_infeasible is not None:
                            rgba = rgba_infeasible
                        plt.plot(self.ts,Y[:,i],lw=lw_infeasible,color=rgba,zorder=zorder_line)

                    # feasible
                    for i in idx_feasible:
                        plt.plot(self.ts,Y[:,i],lw=lw_feasible,color=rgba_feasible,zorder=zorder_line)

        # Single trajectory
        if traj is not None:
            traj = np.asarray(traj,dtype=np.float64).reshape(-1)
            if traj.shape[0] != self.L:
                raise ValueError("traj length must match L")
            plt.plot(
                self.ts,
                traj,
                lw     = lw_traj,
                color  = rgba_traj,
                zorder = zorder_traj,
            )

        # axes/labels
        plt.xlim([self.t_min,self.t_max])
        if ylims is not None:
            plt.ylim([ylims[0],ylims[1]])

        plt.xlabel("Time (t)",fontsize=fs_label)
        plt.ylabel("Trajectory (y)",fontsize=fs_label)

        if title_str is None:
            if Y is None:
                title_str = "Plot box avoid environment"
            else:
                title_str = "Plot box avoid environment with [%d] trajectories"%(int(Y.shape[1]))
        plt.title(title_str,fontsize=fs_title)

        plt.tight_layout()

        if return_rgb_flag:
            fig    = plt.gcf()
            canvas = fig.canvas
            canvas.draw()
            w,h = canvas.get_width_height()
            buf = np.frombuffer(canvas.buffer_rgba(),dtype=np.uint8)
            rgb = buf.reshape(h,w,4)[...,:3]
            if show_flag:
                plt.show()
            else:
                plt.close(fig)
            return rgb #, rgba_list
        elif show_flag:
            plt.show()
        else:
            plt.close()

    # ------------------------------------------------------------
    # score histogram
    # ------------------------------------------------------------
    def hist(
            self,
            scores,
            score_feasible  = None,      # ← default None
            figsize        = (6,3),
            bins           = 10,
            rgba_feasible  = (0,0,1,1),
            cmap_name      = 'coolwarm',
            fs_label       = 8,
            fs_title       = 10,
            title_str      = "score histogram",
        ):
        """
        Plot histogram of scores.

        Parameters:
            scores (np.ndarray): 1D array of scores.
            score_feasible (float): score threshold for feasibility. If None, no threshold is applied.
            figsize (tuple): Figure size.
            bins (int): Number of histogram bins.
            rgba_feasible (tuple): RGBA color for feasible count text.
            cmap_name (str): Colormap name for infeasible scores.
            fs_label (int): Font size for axis labels.
            fs_title (int): Font size for the title.
            title_str (str): Title string for the plot.
        """

        # Convert scores to numpy array
        scores = np.asarray(scores,dtype=np.float64).reshape(-1)

        plt.figure(figsize=figsize)
        ax = plt.gca()
        cmap = plt.get_cmap(cmap_name)

        # ------------------------------------------------------------
        # Case 1: no feasibility threshold (pure visualization)
        # ------------------------------------------------------------
        if score_feasible is None:
            counts,bin_edges,patches = ax.hist(scores,bins=bins)

            cmin = float(np.min(scores))
            cmax = float(np.max(scores))
            den  = (cmax-cmin) if (cmax>cmin) else 1.0

            bin_centers = 0.5*(bin_edges[:-1]+bin_edges[1:])
            for center,patch in zip(bin_centers,patches):
                cnorm = 1.0 - (center-cmin)/den
                cnorm = np.clip(cnorm,0.0,1.0)
                patch.set_facecolor(cmap(cnorm))

            ax.set_xlabel("score",fontsize=fs_label)
            ax.set_title(title_str,fontsize=fs_title)

            plt.tight_layout()
            plt.show()
            return

        # ------------------------------------------------------------
        # Case 2: feasibility threshold is given
        # ------------------------------------------------------------
        n_feasible = int(np.sum(scores >= score_feasible))
        neg_scores  = scores[scores < score_feasible]

        if neg_scores.size>0:
            counts,bin_edges,patches = ax.hist(neg_scores,bins=bins)

            cmin = float(np.min(neg_scores))
            cmax = score_feasible
            den  = (cmax-cmin) if (cmax>cmin) else 1.0

            bin_centers = 0.5*(bin_edges[:-1]+bin_edges[1:])
            for center,patch in zip(bin_centers,patches):
                cnorm = 1.0 - (center-cmin)/den
                cnorm = np.clip(cnorm,0.0,1.0)
                patch.set_facecolor(cmap(cnorm))
        else:
            ax.text(
                0.5,0.5,
                "No infeasible samples",
                transform=ax.transAxes,
                ha='center',va='center',
                fontsize=fs_label,
            )

        ax.set_xlabel("score (infeasible only)",fontsize=fs_label)
        ax.set_title(title_str,fontsize=fs_title)

        ax.text(
            0.98,0.95,
            "#feasible (score>=[%g]):[%d]"%(score_feasible,n_feasible),
            transform=ax.transAxes,
            ha='right',va='top',
            fontsize=fs_label,
            color=rgba_feasible,
        )

        plt.tight_layout()
        plt.show()
