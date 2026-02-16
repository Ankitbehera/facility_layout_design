"""
================================================================================
Minisum Multiple Facility Location Problem 
================================================================================
"""

import numpy as np
import pulp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ------------------------------------------------------------------------------
# Helper: weighted median
# ------------------------------------------------------------------------------

def weighted_median(values, weights):
    data = sorted(zip(values, weights), key=lambda x: x[0])
    total_w = sum(weights)
    cum_w = 0

    for v, w in data:
        cum_w += w
        if cum_w >= total_w / 2:
            return v


# ------------------------------------------------------------------------------
# Objective function (L1 minisum)
# ------------------------------------------------------------------------------

def minisum_cost(X, existing, w_ji, v_jk):
    """
    X        : list of (xj, yj) for new facilities
    existing : list of (ai, bi)
    w_ji     : weight matrix (n x m)
    v_jk     : interaction matrix (n x n)
    """
    cost = 0.0
    n = len(X)
    m = len(existing)

    # New–existing interactions
    for j in range(n):
        xj, yj = X[j]
        for i in range(m):
            ai, bi = existing[i]
            cost += w_ji[j][i] * (abs(xj - ai) + abs(yj - bi))

    # New–new interactions
    for j in range(n):
        for k in range(j + 1, n):
            xj, yj = X[j]
            xk, yk = X[k]
            cost += v_jk[j][k] * (abs(xj - xk) + abs(yj - yk))

    return cost


# ------------------------------------------------------------------------------
# Coordinate Descent Solver
# ------------------------------------------------------------------------------

def solve_minisum_mfl(existing, w_ji, v_jk, max_iter=50, tol=1e-6):
    """
    Coordinate Descent Algorithm for Minisum MFL (Rectilinear)

    Returns:
        dict with keys:
        - X_opt : optimal new facility locations
        - history : list of iterations
    """

    m = len(existing)
    n = len(w_ji)

    # -----------------------------
    # Step 0: Initial solution
    # (set v_jk = 0 → independent SFL)
    # -----------------------------
    X = []

    for j in range(n):
        a_vals = [existing[i][0] for i in range(m)]
        b_vals = [existing[i][1] for i in range(m)]
        weights = w_ji[j]

        x0 = weighted_median(a_vals, weights)
        y0 = weighted_median(b_vals, weights)
        X.append([x0, y0])

    history = [X.copy()]

    # -----------------------------
    # Iterations
    # -----------------------------
    for _ in range(max_iter):
        X_old = [tuple(p) for p in X]

        for t in range(n):
            # Build artificial data for NF t
            a_vals = []
            b_vals = []
            weights = []

            # Existing facilities
            for i in range(m):
                a_vals.append(existing[i][0])
                b_vals.append(existing[i][1])
                weights.append(w_ji[t][i])

            # Other new facilities
            for k in range(n):
                if k != t:
                    a_vals.append(X[k][0])
                    b_vals.append(X[k][1])
                    weights.append(v_jk[t][k])

            # Update location using weighted median
            X[t][0] = weighted_median(a_vals, weights)
            X[t][1] = weighted_median(b_vals, weights)

        history.append([tuple(p) for p in X])

        # Convergence check
        diff = sum(
            abs(X[t][0] - X_old[t][0]) + abs(X[t][1] - X_old[t][1])
            for t in range(n)
        )

        if diff < tol:
            break

    return {
        "X_opt": [tuple(p) for p in X],
        "history": history,
        "obj": minisum_cost(X, existing, w_ji, v_jk),
    }

def check_multiple_solutions(ef_coords, w_matrix, v_matrix, X_opt):
    """
    Checks for multiple optimal solutions and calculates the valid range.
    Returns a list of warning dictionaries:
    [
      {"type": "property4", "msg": "..."}, 
      {"type": "flat", "msg": "...", "range": "[10, 20]"}
    ]
    """
    messages = []
    n = len(X_opt)
    m = len(ef_coords)
    
    # 1. Check Property 4 (Distinctness)
    # If two NFs share the same x or y, it's a red flag.
    shared_warnings = []
    for j in range(n):
        for k in range(j + 1, n):
            if (abs(X_opt[j][0] - X_opt[k][0]) < 1e-4) and (abs(X_opt[j][1] - X_opt[k][1]) < 1e-4):
                shared_warnings.append(f"NF{j+1} & NF{k+1}")
    
    if shared_warnings:
        messages.append({
            "type": "property4",
            "msg": f"**Coincidence Warning (Property 4 Violated):** {', '.join(shared_warnings)} are at the exact same location. The algorithm may be trapped in a local optimum."
        })

    # 2. Check "Flat Median" Condition (Left Weight == Right Weight)
    for dim, dim_name in [(0, 'x'), (1, 'y')]:
        for j in range(n):
            current_val = X_opt[j][dim]
            
            # Collect all interacting coordinates and their weights
            relevant_points = []
            
            # EF Weights
            for i in range(m):
                w = w_matrix[j][i]
                if w > 0:
                    relevant_points.append({'coord': ef_coords[i][dim], 'weight': w, 'type': f'EF{i+1}'})
            
            # NF Weights (Fixed locations of others)
            for k in range(n):
                if j == k: continue
                v = v_matrix[j][k] if k > j else v_matrix[k][j]
                if v > 0:
                    relevant_points.append({'coord': X_opt[k][dim], 'weight': v, 'type': f'NF{k+1}'})

            # Calculate Forces
            left_weight = sum(p['weight'] for p in relevant_points if p['coord'] < current_val - 1e-5)
            right_weight = sum(p['weight'] for p in relevant_points if p['coord'] > current_val + 1e-5)
            
            # If weights are balanced, we are in a flat region.
            # We need to find the NEAREST neighbors to define the range.
            if abs(left_weight - right_weight) < 1e-4:
                # Find bound_low (nearest point <= current) and bound_high (nearest point >= current)
                # Actually, in a flat region, the range is usually between two critical points.
                
                sorted_coords = sorted([p['coord'] for p in relevant_points])
                
                # Find immediate neighbors in the sorted list
                lower_bound = current_val
                upper_bound = current_val
                
                # Search below
                below = [z for z in sorted_coords if z < current_val - 1e-5]
                if below: lower_bound = max(below)
                
                # Search above
                above = [z for z in sorted_coords if z > current_val + 1e-5]
                if above: upper_bound = min(above)
                
                # Format the range string
                if abs(lower_bound - upper_bound) > 1e-4:
                    range_str = f"[{lower_bound:.2f}, {upper_bound:.2f}]"
                    messages.append({
                        "type": "flat",
                        "facility": f"NF{j+1}",
                        "axis": dim_name,
                        "range": range_str,
                        "msg": f"**Multiple Solutions for NF{j+1} ({dim_name}-axis):** Weights are perfectly balanced. Optimal Range: {range_str}"
                    })

    return messages

# ------------------------------------------------------------------------------
# LP Solver (Rectilinear)
# ------------------------------------------------------------------------------

def solve_minisum_mfl_lp(existing, w_ji, v_jk):
    """
    Solves the Minisum MFL with Rectilinear distance using Linear Programming.
    Decomposes the problem into independent X and Y models.
    Returns detailed variable states for UI display.
    """
    m = len(existing)
    n = len(w_ji)
    
    # Internal helper to solve for one dimension (x or y)
    def solve_1d(coords, weights_ef, weights_nf):
        prob = pulp.LpProblem("Minisum_MFL_1D", pulp.LpMinimize)
        
        # --- Decision Variables ---
        x = [pulp.LpVariable(f"x_{j}", lowBound=None) for j in range(n)]
        
        # r_ji, s_ji for EF-NF interactions
        r = [[pulp.LpVariable(f"r_{j}_{i}", lowBound=0) for i in range(m)] for j in range(n)]
        s = [[pulp.LpVariable(f"s_{j}_{i}", lowBound=0) for i in range(m)] for j in range(n)]
        
        # p_jk, q_jk for NF-NF interactions (only for j < k)
        p_aux = [[None for k in range(n)] for j in range(n)]
        q_aux = [[None for k in range(n)] for j in range(n)]
        
        for j in range(n):
            for k in range(j + 1, n):
                p_aux[j][k] = pulp.LpVariable(f"p_{j}_{k}", lowBound=0)
                q_aux[j][k] = pulp.LpVariable(f"q_{j}_{k}", lowBound=0)

        # --- Objective Function ---
        obj_ef = pulp.lpSum(weights_ef[j][i] * (r[j][i] + s[j][i]) for j in range(n) for i in range(m))
        obj_nf = pulp.lpSum(weights_nf[j][k] * (p_aux[j][k] + q_aux[j][k]) for j in range(n) for k in range(j+1, n))
        
        prob += obj_ef + obj_nf

        # --- Constraints ---
        # 1. EF-NF constraints
        for j in range(n):
            for i in range(m):
                prob += x[j] - r[j][i] + s[j][i] == coords[i]
        
        # 2. NF-NF constraints
        for j in range(n):
            for k in range(j + 1, n):
                prob += x[j] - x[k] - p_aux[j][k] + q_aux[j][k] == 0
        
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # --- Extract Detailed Results ---
        res_x = [pulp.value(var) for var in x]
        
        # Extract auxiliary variables
        res_r = [[pulp.value(r[j][i]) for i in range(m)] for j in range(n)]
        res_s = [[pulp.value(s[j][i]) for i in range(m)] for j in range(n)]
        
        res_p = [[0.0 for k in range(n)] for j in range(n)]
        res_q = [[0.0 for k in range(n)] for j in range(n)]
        
        for j in range(n):
            for k in range(j+1, n):
                res_p[j][k] = pulp.value(p_aux[j][k])
                res_q[j][k] = pulp.value(q_aux[j][k])

        # Active constraints check
        active_ef = []
        for j in range(n):
            for i in range(m):
                if abs(res_r[j][i]) < 1e-5 and abs(res_s[j][i]) < 1e-5:
                    active_ef.append((j, i))
                    
        active_nf = []
        for j in range(n):
            for k in range(j + 1, n):
                if abs(res_p[j][k]) < 1e-5 and abs(res_q[j][k]) < 1e-5:
                    active_nf.append((j, k))

        details = {
            "coords": res_x,
            "r": res_r,
            "s": res_s,
            "p": res_p,
            "q": res_q,
            "active_ef": active_ef,
            "active_nf": active_nf,
            "obj": pulp.value(prob.objective)
        }
        return details

    a_coords = [p[0] for p in existing]
    b_coords = [p[1] for p in existing]
    
    # Solve independent problems
    res_x = solve_1d(a_coords, w_ji, v_jk)
    res_y = solve_1d(b_coords, w_ji, v_jk)
    
    X_opt = list(zip(res_x["coords"], res_y["coords"]))
    
    return {
        "X_opt": X_opt,
        "obj": res_x["obj"] + res_y["obj"],
        "details_x": res_x,
        "details_y": res_y
    }
import numpy as np

# --------------------------------------------------
# 1. Squared Euclidean Solver (Analytical)
# --------------------------------------------------
def solve_squared_euclidean(existing_coords, w_matrix, v_matrix):
    """
    Solves Minisum MFL with Squared Euclidean distance using a linear system.
    Returns dictionary with optimal coords and objective value.
    Reference: Lecture Slide 31 (Weighted Average).
    """
    existing_coords = np.array(existing_coords)
    w_matrix = np.array(w_matrix)
    v_matrix = np.array(v_matrix)
    
    n_new = len(v_matrix)
    m_exist = len(existing_coords)
    
    # Setup System: A * x = B_x and A * y = B_y
    A = np.zeros((n_new, n_new))
    B_x = np.zeros(n_new)
    B_y = np.zeros(n_new)
    
    for j in range(n_new):
        # Diagonal: Sum of all weights connected to j
        sum_w = np.sum(w_matrix[j, :])
        sum_v = np.sum(v_matrix[j, :]) # Row sum (v_jj is 0)
        A[j, j] = sum_w + sum_v
        
        # Off-diagonal: Negative interaction weights
        for k in range(n_new):
            if j != k:
                # v matrix is symmetric, usually v_jk or v_kj
                val = v_matrix[j, k] if k > j else v_matrix[k, j]
                A[j, k] = -val
        
        # RHS Vectors: Weighted sum of fixed coordinates
        B_x[j] = np.sum(w_matrix[j, :] * existing_coords[:, 0])
        B_y[j] = np.sum(w_matrix[j, :] * existing_coords[:, 1])
        
    # Solve
    try:
        x_opt = np.linalg.solve(A, B_x)
        y_opt = np.linalg.solve(A, B_y)
        X_opt = list(zip(x_opt, y_opt))
        
        # Calculate Objective Cost (Squared)
        obj_val = 0.0
        # NF-NF
        for j in range(n_new):
            for k in range(j+1, n_new):
                dist_sq = (x_opt[j] - x_opt[k])**2 + (y_opt[j] - y_opt[k])**2
                val = v_matrix[j, k]
                obj_val += val * dist_sq
        # NF-EF
        for j in range(n_new):
            for i in range(m_exist):
                dist_sq = (x_opt[j] - existing_coords[i, 0])**2 + (y_opt[j] - existing_coords[i, 1])**2
                obj_val += w_matrix[j, i] * dist_sq
                
        return {"X_opt": X_opt, "obj": obj_val, "status": "Optimal"}
        
    except np.linalg.LinAlgError:
        return {"error": "System is singular. Check weights (must be > 0)."}

# --------------------------------------------------
# 2. Standard Euclidean Solver (Iterative Weiszfeld)
# --------------------------------------------------
def solve_euclidean_weiszfeld(existing_coords, w_matrix, v_matrix, max_iter=50, tol=1e-4):
    """
    Solves Minisum MFL with Standard Euclidean distance using Weiszfeld's Algorithm.
    Reference: Lecture Slide 36-38.
    """
    existing_coords = np.array(existing_coords)
    w_matrix = np.array(w_matrix)
    v_matrix = np.array(v_matrix)
    
    n_new = len(v_matrix)
    m_exist = len(existing_coords)
    epsilon = 1e-6 # To avoid division by zero
    
    # Initialize at Center of Gravity (Squared solution is a good guess)
    init_res = solve_squared_euclidean(existing_coords, w_matrix, v_matrix)
    if "error" in init_res:
        current_X = np.mean(existing_coords, axis=0) * np.ones((n_new, 2))
    else:
        current_X = np.array(init_res["X_opt"])
        
    history = [current_X.copy()]
    
    for iteration in range(max_iter):
        next_X = np.zeros_like(current_X)
        
        for j in range(n_new):
            # Numerators and Denominators for Weiszfeld update
            num_x, num_y = 0.0, 0.0
            denom = 0.0
            
            # EF contributions
            for i in range(m_exist):
                dist = np.sqrt(np.sum((current_X[j] - existing_coords[i])**2))
                dist = max(dist, epsilon) # Handle singularity
                
                weight = w_matrix[j, i]
                term = weight / dist
                
                num_x += term * existing_coords[i, 0]
                num_y += term * existing_coords[i, 1]
                denom += term
                
            # NF contributions
            for k in range(n_new):
                if j == k: continue
                val = v_matrix[j, k] if k > j else v_matrix[k, j]
                
                dist = np.sqrt(np.sum((current_X[j] - current_X[k])**2))
                dist = max(dist, epsilon)
                
                term = val / dist
                num_x += term * current_X[k, 0]
                num_y += term * current_X[k, 1]
                denom += term
            
            # Update coordinate j
            if denom > 1e-8:
                next_X[j, 0] = num_x / denom
                next_X[j, 1] = num_y / denom
            else:
                next_X[j] = current_X[j] # No pull, stay put
                
        # Check convergence
        diff = np.sum(np.abs(next_X - current_X))
        history.append(next_X.copy())
        current_X = next_X
        
        if diff < tol:
            break
            
    # Calculate Final Cost (Standard L2)
    obj_val = 0.0
    for j in range(n_new):
        for k in range(j+1, n_new):
            dist = np.sqrt(np.sum((current_X[j] - current_X[k])**2))
            val = v_matrix[j, k] if k > j else v_matrix[k, j]
            obj_val += val * dist
            
    for j in range(n_new):
        for i in range(m_exist):
            dist = np.sqrt(np.sum((current_X[j] - existing_coords[i])**2))
            obj_val += w_matrix[j, i] * dist
            
    return {
        "X_opt": current_X.tolist(),
        "obj": obj_val,
        "history": history,
        "iterations": iteration + 1
    }


# ------------------------------------------------------------------------------
# Plotting Function
# ------------------------------------------------------------------------------

def plot_mfl_solution(existing_coords, new_coords, w_matrix, v_matrix):
    """
    Generates a MFL solution plot with weight labels on connections.
    """
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    
    m = len(existing_coords)
    n = len(new_coords)
    
    # Define color palette
    color_ef = '#1f77b4'  # Professional Blue
    color_nf = '#d62728'  # Professional Red
    color_line_ef = '#aec7e8' # Light Blue for EF-NF
    color_line_nf = '#7f7f7f' # Medium Gray for NF-NF

    # -----------------------------
    # 1. Plot Interactions & Weights
    # -----------------------------
    # EF-NF Interactions (w_ji)
    for j in range(n):
        nf_x, nf_y = new_coords[j]
        for i in range(m):
            weight = w_matrix[j][i]
            if weight > 0:
                ef_x, ef_y = existing_coords[i]
                # Plot line
                ax.plot([nf_x, ef_x], [nf_y, ef_y], color=color_line_ef, 
                        linestyle='--', linewidth=1, alpha=0.7, zorder=1)
                
                # Calculate midpoint for weight label
                mid_x, mid_y = (nf_x + ef_x) / 2, (nf_y + ef_y) / 2
                ax.text(mid_x, mid_y, f'{weight:g}', color='#154360', fontsize=8,
                        ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

    # NF-NF Interactions (v_jk)
    for j in range(n):
        for k in range(j + 1, n):
            weight = v_matrix[j][k]
            if weight > 0:
                nf1_x, nf1_y = new_coords[j]
                nf2_x, nf2_y = new_coords[k]
                # Plot line
                ax.plot([nf1_x, nf2_x], [nf1_y, nf2_y], color=color_line_nf, 
                        linestyle='-', linewidth=1.5, alpha=0.8, zorder=1)
                
                # Calculate midpoint for weight label
                mid_x, mid_y = (nf1_x + nf2_x) / 2, (nf1_y + nf2_y) / 2
                ax.text(mid_x, mid_y, f'{weight:g}', color='black', fontsize=9, fontweight='bold',
                        ha='center', va='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    # -----------------------------
    # 2. Plot Facilities
    # -----------------------------
    # Existing Facilities: Blue circle with a central dot
    ef_xs = [p[0] for p in existing_coords]
    ef_ys = [p[1] for p in existing_coords]
    ax.scatter(ef_xs, ef_ys, facecolors='none', edgecolors=color_ef, 
               marker='o', s=130, linewidths=1.5, zorder=2)
    ax.scatter(ef_xs, ef_ys, c=color_ef, marker='o', s=15, zorder=3)

    # New Facilities: Vibrant Red Star
    nf_xs = [p[0] for p in new_coords]
    nf_ys = [p[1] for p in new_coords]
    ax.scatter(nf_xs, nf_ys, c=color_nf, marker='*', s=200, edgecolors='black', 
               linewidths=0.5, zorder=4)

    # -----------------------------
    # 3. Annotations
    # -----------------------------
    for i, (x, y) in enumerate(existing_coords):
        ax.annotate(f'$P_{{{i+1}}}$', (x, y), textcoords="offset points", 
                    xytext=(0,-18), ha='center', fontsize=9, color=color_ef)

    for j, (x, y) in enumerate(new_coords):
        ax.annotate(f'$X_{{{j+1}}}$', (x, y), textcoords="offset points", 
                    xytext=(0,12), ha='center', fontsize=10, fontweight='bold', color=color_nf)

    # -----------------------------
    # 4. Legend & Formatting
    # -----------------------------
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Existing Facility ($P_i$)',
               markerfacecolor=color_ef, markersize=10),
        Line2D([0], [0], marker='*', color='w', label='New Facility ($X_j$)',
               markerfacecolor=color_nf, markersize=12, markeredgecolor='black'),
        Line2D([0], [0], color=color_line_ef, linestyle='--', label='$w_{ji}$ weights'),
        Line2D([0], [0], color=color_line_nf, linestyle='-', label='$v_{jk}$ weights')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1), fontsize=6)
    
    ax.set_title("Optimal Minisum Facility Layout", fontsize=9, pad=15)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    
    return fig

# ------------------------------------------------------------------------------
# Coordinate Descent Visualization 
# ------------------------------------------------------------------------------

def plot_cost_history(history_obj):
    """
    Plots the convergence of the objective function (Total Cost).
    Style: Matches Tab 2 (Academic/Publication quality).
    """
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    
    iterations = range(len(history_obj))
    
    # Plot Line
    ax.plot(iterations, history_obj, marker='o', markersize=4, 
            linestyle='-', color='#1f77b4', linewidth=1.5, label='Total Cost')
    
    # Styling
    ax.set_title("Objective Function Convergence", fontsize=11, pad=10)
    ax.set_xlabel("Iteration Number", fontsize=9)
    ax.set_ylabel("Total Cost", fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    # Annotation for start and end cost
    start_cost = history_obj[0]
    end_cost = history_obj[-1]
    ax.text(0, start_cost, f'{start_cost:.1f}', fontsize=8, ha='right', va='bottom')
    ax.text(len(history_obj)-1, end_cost, f'{end_cost:.1f}', fontsize=8, ha='left', va='top')
    
    plt.tight_layout()
    return fig

def plot_trajectory(existing_coords, history_X, w_matrix, v_matrix):
    """
    Plots movement paths and final weighted interactions for New Facilities.
    """
    fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
    
    num_nfs = len(history_X[0])
    num_iters = len(history_X)
    optimal_coords = history_X[-1]
    m = len(existing_coords)

    # Colors and styling
    color_ef = '#1f77b4'
    color_nf = '#d62728'
    color_line_ef = '#aec7e8'
    color_line_nf = '#7f7f7f'
    path_colors = ['#d62728', '#ff7f0e', '#9467bd']

    # --------------------------------------------------
    # 1. Plot Final Interactions & Weights (Optimal State)
    # --------------------------------------------------
    # EF-NF Interactions (w_ji)
    for j in range(num_nfs):
        nf_x, nf_y = optimal_coords[j]
        for i in range(m):
            weight = w_matrix[j][i]
            if weight > 0:
                ef_x, ef_y = existing_coords[i]
                ax.plot([nf_x, ef_x], [nf_y, ef_y], color=color_line_ef, 
                        linestyle=':', linewidth=0.8, alpha=0.5, zorder=1)
                
                # Weight label at midpoint
                mid_x, mid_y = (nf_x + ef_x) / 2, (nf_y + ef_y) / 2
                ax.text(mid_x, mid_y, f'{weight:g}', color='#154360', fontsize=7,
                        ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0.5))

    # NF-NF Interactions (v_jk)
    for j in range(num_nfs):
        for k in range(j + 1, num_nfs):
            weight = v_matrix[j][k]
            if weight > 0:
                nf1_x, nf1_y = optimal_coords[j]
                nf2_x, nf2_y = optimal_coords[k]
                ax.plot([nf1_x, nf2_x], [nf1_y, nf2_y], color=color_line_nf, 
                        linestyle='-', linewidth=1.2, alpha=0.6, zorder=1)
                
                # Weight label at midpoint
                mid_x, mid_y = (nf1_x + nf2_x) / 2, (nf1_y + nf2_y) / 2
                ax.text(mid_x, mid_y, f'{weight:g}', color='black', fontsize=8, fontweight='bold',
                        ha='center', va='center', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5))

    # --------------------------------------------------
    # 2. Plot Trajectories (Movement Paths)
    # --------------------------------------------------
    for j in range(num_nfs):
        path_x = [history_X[k][j][0] for k in range(num_iters)]
        path_y = [history_X[k][j][1] for k in range(num_iters)]
        c_line = path_colors[j % len(path_colors)]
        
        # Path Line
        ax.plot(path_x, path_y, linestyle='--', linewidth=1, color=c_line, alpha=0.8, zorder=2)
        # Start Point
        ax.scatter([path_x[0]], [path_y[0]], c='gray', marker='x', s=40, zorder=3)

    # --------------------------------------------------
    # 3. Plot Facilities
    # --------------------------------------------------
    # Existing Facilities
    ef_xs = [p[0] for p in existing_coords]
    ef_ys = [p[1] for p in existing_coords]
    #ax.scatter(ef_xs, ef_ys, facecolors='none', edgecolors=color_ef, marker='o', s=130, linewidths=1.5, zorder=4)
    ax.scatter(ef_xs, ef_ys, c=color_ef, marker='o', s=15, zorder=5)

    # Optimal New Facilities
    nf_xs = [p[0] for p in optimal_coords]
    nf_ys = [p[1] for p in optimal_coords]
    ax.scatter(nf_xs, nf_ys, c=color_nf, marker='*', s=200, edgecolors='black', linewidths=0.5, zorder=6)

    # Labels
    for i, (x, y) in enumerate(existing_coords):
        ax.text(x, y - 1.2, f'$P_{{{i+1}}}$', fontsize=8, ha='center', color=color_ef)
    for j, (x, y) in enumerate(optimal_coords):
        ax.text(x, y + 1.2, f'$X_{{{j+1}}}$', fontsize=9, fontweight='bold', ha='center', color=color_nf)

    # Legend & Formatting
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Existing ($P_i$)', markerfacecolor=color_ef, markersize=8),
        Line2D([0], [0], marker='*', color='w', label='Optimal ($X_j$)', markerfacecolor=color_nf, markersize=12, markeredgecolor='black'),
        Line2D([0], [0], color='gray', linestyle='--', label='Path'),
        Line2D([0], [0], marker='x', color='w', markeredgecolor='gray', label='Start')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=7)
    
    ax.set_title("Optimization Trajectory & Weighted Interactions", fontsize=11, pad=10)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    
    return fig