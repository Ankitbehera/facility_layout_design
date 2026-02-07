"""
================================================================================
Minisum Single Location Problem Functions
================================================================================
This module provides comprehensive solvers and visualization functions for the
Minisum Single Facility Location Problem under different distance metrics:
Functions are organized in the order they are called in app.py for clarity.
================================================================================
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd


# ================================================================================
# SECTION 1: L1 MINISUM PROBLEM - CORE SOLVER
# ================================================================================

def weighted_median(values, weights):
    """
    Compute the weighted median of values with given weights.
    
    The weighted median is the value where cumulative weight reaches or exceeds
    half of the total weight. Returns both lower and upper medians; they are
    equal when the median is unique.
    
    Parameters:
    -----------
    values : list of float
        Coordinate values to find median of
    weights : list of float
        Corresponding weights for each value
    
    Returns:
    --------
    tuple: (lower_median, upper_median)
        If unique, both are equal. If range of median, returns the bounds.
    """
    # Sort values with their corresponding weights
    sorted_data = sorted(zip(values, weights), key=lambda x: x[0])
    total_weight = sum(weights)
    half_weight = total_weight / 2
    
    cumulative = 0
    lower = None
    upper = None
    
    for v, w in sorted_data:
        cumulative += w
        if lower is None and cumulative >= half_weight:
            lower = v
        if cumulative > half_weight:
            upper = v
            break
    
    return lower, upper


def solve_single_facility_L1_median(data):
    """
    Solve the Minisum Single Facility Location Problem with L1 (rectilinear) 
    distance using the Median Method.
    
    This is the classical approach: Find the weighted median of x-coordinates
    and y-coordinates independently. The optimal location is the Cartesian
    product of these two medians (which may be ranges if medians are not unique).
    
    Parameters:
    -----------
    data : list of tuple
        Each element is (a_i, b_i, w_i) where:
        - a_i: x-coordinate of facility i
        - b_i: y-coordinate of facility i
        - w_i: weight (demand) of facility i
    
    Returns:
    --------
    dict with keys:
        - 'x_range': (x_low, x_high) – range of x-coordinates in optimal set
        - 'y_range': (y_low, y_high) – range of y-coordinates in optimal set
        - 'x_opt': representative x (center of range)
        - 'y_opt': representative y (center of range)
        - 'obj': objective value at representative point
    """
    # Extract coordinates and weights
    a_vals = [a for a, _, _ in data]
    b_vals = [b for _, b, _ in data]
    weights = [w for _, _, w in data]
    
    # Find weighted medians independently for each dimension
    x_low, x_high = weighted_median(a_vals, weights)
    y_low, y_high = weighted_median(b_vals, weights)
    
    # Representative optimal point (center of optimal rectangle)
    x_opt = 0.5 * (x_low + x_high)
    y_opt = 0.5 * (y_low + y_high)
    opt_val = obj_L1(x_opt, y_opt, data)
    
    return {
        "x_range": (x_low, x_high),
        "y_range": (y_low, y_high),
        "x_opt": x_opt,
        "y_opt": y_opt,
        "obj": opt_val
    }


def classify_L1_solution(x_range, y_range, tol=1e-6):
    """
    Classify the structure of the L1 optimal solution set.
    
    The optimal set under L1 distance is a rectilinear (axis-aligned) region.
    This function determines whether the optimal set is:
      - A unique point (x* and y* both unique)
      - A horizontal line (y* unique, x* is a range)
      - A vertical line (x* unique, y* is a range)
      - A rectangle (both x* and y* are ranges)
    
    Parameters:
    -----------
    x_range : tuple
        (x_low, x_high) – range of optimal x-coordinates
    y_range : tuple
        (y_low, y_high) – range of optimal y-coordinates
    tol : float
        Tolerance for considering values as equal
    
    Returns:
    --------
    tuple: (solution_type, vertices)
        - solution_type: string describing the region type
        - vertices: list of corner points for visualization
    """
    x_low, x_high = x_range
    y_low, y_high = y_range
    
    # Check for unique point
    if abs(x_low - x_high) < tol and abs(y_low - y_high) < tol:
        return "Unique Point", [(x_low, y_low)]
    
    # Check for vertical line (x unique, y varies)
    if abs(x_low - x_high) < tol:
        return "Vertical Line", [(x_low, y_low), (x_low, y_high)]
    
    # Check for horizontal line (y unique, x varies)
    if abs(y_low - y_high) < tol:
        return "Horizontal Line", [(x_low, y_low), (x_high, y_low)]
    
    # Rectangle (both x and y vary)
    return "Rectangle", [
        (x_low, y_low),
        (x_low, y_high),
        (x_high, y_high),
        (x_high, y_low)
    ]


# ================================================================================
# SECTION 2: GRAPHICAL METHOD HELPERS (TAB 1)
# ================================================================================

def f1_value(x, data):
    """
    Evaluate the univariate function f_1(x) = sum(w_i * |x - a_i|).
    
    This is the x-coordinate portion of the L1 objective function.
    Used for graphical analysis of the piecewise linear function.
    
    Parameters:
    -----------
    x : float
        x-coordinate value
    data : list of tuple
        Each element is (a_i, b_i, w_i)
    
    Returns:
    --------
    float: value of f_1(x)
    """
    return sum(w * abs(x - a) for a, _, w in data)


def f2_value(y, data):
    """
    Evaluate the univariate function f_2(y) = sum(w_i * |y - b_i|).
    
    This is the y-coordinate portion of the L1 objective function.
    Used for graphical analysis of the piecewise linear function.
    
    Parameters:
    -----------
    y : float
        y-coordinate value
    data : list of tuple
        Each element is (a_i, b_i, w_i)
    
    Returns:
    --------
    float: value of f_2(y)
    """
    return sum(w * abs(y - b) for _, b, w in data)


def plot_piecewise_L1(values, func, label):
    """
    Plot the piecewise linear univariate L1 objective function.
    
    Graphs either f_1(x) or f_2(y) as a function of the coordinate,
    marking the points at facility locations and identifying the minimum.
    
    Parameters:
    -----------
    values : list of float
        Coordinate values where function is evaluated (facility locations)
    func : callable
        Function to evaluate (either f1_value or f2_value)
    label : str
        Axis label, either "x" or "y"
    
    Returns:
    --------
    tuple: (fig, x_min, y_min)
        - fig: matplotlib Figure object
        - x_min: x-coordinate of the minimum
        - y_min: minimum function value
    """
    # Evaluate function at unique sorted values
    xs = sorted(set(values))
    ys = [func(x) for x in xs]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    
    # Plot the function
    ax.plot(xs, ys, marker='o')
    
    # Label each point with coordinates
    for x, y in zip(xs, ys):
        ax.text(x, y + 0.03 * max(ys), f"({x}, {int(y)})", fontsize=9)
    
    # Mark the minimum with dashed lines
    min_idx = ys.index(min(ys))
    ax.axvline(xs[min_idx], linestyle="--", alpha=0.6)
    ax.axhline(ys[min_idx], linestyle="--", alpha=0.6)
    
    # Configure axes
    ax.set_xlabel(label)
    ax.set_ylabel(f"f({label})")
    ax.grid(True, linestyle="--", alpha=0.5)
    
    return fig, xs[min_idx], ys[min_idx]


# ================================================================================
# SECTION 3: MEDIAN METHOD HELPER (TAB 2)
# ================================================================================

def build_weighted_median_table(values, weights, labels):
    """
    Build a detailed table showing the weighted median calculation.
    
    Creates a pandas DataFrame that sorts facilities by coordinate value,
    computes cumulative weights, and identifies the median facility.
    Useful for educational display and verification of the median method.
    
    Parameters:
    -----------
    values : list of float
        Coordinate values (either all x-coordinates or all y-coordinates)
    weights : list of float
        Corresponding weights (demands) for each facility
    labels : list
        Facility identifiers (1, 2, 3, ...)
    
    Returns:
    --------
    tuple: (df, median_row, total_weight)
        - df: pandas DataFrame with sorted data and cumulative weights
        - median_row: the row where cumulative weight >= half total weight
        - total_weight: sum of all weights
    """
    # Create DataFrame from input data
    df = pd.DataFrame({
        "Existing Facility": labels,
        "Coordinate": values,
        "Weight": weights
    })
    
    # Sort by coordinate value
    df = df.sort_values("Coordinate").reset_index(drop=True)
    
    # Compute cumulative weights
    df["Cumulative Weight"] = df["Weight"].cumsum()
    total_weight = df["Weight"].sum()
    half_weight = total_weight / 2
    
    # Find the row where cumulative weight first meets/exceeds half
    median_row = df[df["Cumulative Weight"] >= half_weight].iloc[0]
    
    return df, median_row, total_weight


# ================================================================================
# SECTION 4: COST EVALUATION AND UTILITY FUNCTIONS
# ================================================================================

def rectilinear_cost(x, y, data):
    """
    Calculate the rectilinear (L1) distance-weighted cost.
    
    Computes the objective function value:
    f_L1(x, y) = sum(w_i * (|x - a_i| + |y - b_i|))
    
    Parameters:
    -----------
    x, y : float
        Coordinates of the candidate facility location
    data : list of tuple
        Each element is (a_i, b_i, w_i)
    
    Returns:
    --------
    float: total rectilinear cost
    """
    return sum(w * (abs(x - a) + abs(y - b)) for a, b, w in data)


def obj_L1(x, y, data):
    """
    Evaluate L1 objective function (same as rectilinear_cost).
    
    Wrapper for consistency with other objective functions (obj_L2, obj_L2_squared).
    """
    return sum(w * (abs(x - a) + abs(y - b)) for a, b, w in data)


# ================================================================================
# SECTION 5: ISO-CONTOUR FUNCTIONS (TAB 3)
# ================================================================================

def iso_contour(data, x0, y0, n_dirs=360):
    """
    Trace an iso-cost contour passing through a given point using binary search.
    
    For a given point (x0, y0), finds the locus of points (x, y) that have
    the same L1 cost as (x0, y0). Uses binary search in radial directions
    from the center point.
    
    Parameters:
    -----------
    data : list of tuple
        Facility data (a_i, b_i, w_i)
    x0, y0 : float
        Center point (the contour passes through this point)
    n_dirs : int
        Number of radial directions to sample (default 360 for smooth contour)
    
    Returns:
    --------
    tuple: (contour_array, base_cost)
        - contour_array: numpy array of shape (n_dirs, 2) with (x, y) points
        - base_cost: the cost value of the contour
    """
    # Calculate cost at the center point
    base_cost = rectilinear_cost(x0, y0, data)
    
    # Sample angles around the circle
    angles = np.linspace(0, 2*np.pi, n_dirs)
    contour = []
    
    # For each direction, find the distance where cost equals base_cost
    for theta in angles:
        dx, dy = np.cos(theta), np.sin(theta)
        
        # Binary search for the distance where cost = base_cost
        lo, hi = 0, 50
        for _ in range(40):
            mid = (lo + hi) / 2
            x = x0 + mid * dx
            y = y0 + mid * dy
            
            if rectilinear_cost(x, y, data) < base_cost:
                lo = mid
            else:
                hi = mid
        
        contour.append((x0 + lo * dx, y0 + lo * dy))
    
    return np.array(contour), base_cost


def iso_contour_at_cost(data, x_center, y_center, target_cost, n_dirs=360):
    """
    Trace an iso-cost contour for a specific cost value.
    
    Similar to iso_contour(), but instead of starting from a point and finding
    its cost, this function finds the contour for a specified cost value.
    Uses binary search in radial directions from a center point.
    
    Parameters:
    -----------
    data : list of tuple
        Facility data (a_i, b_i, w_i)
    x_center, y_center : float
        Center point around which to search radially
    target_cost : float
        The cost value for the iso-contour
    n_dirs : int
        Number of radial directions to sample
    
    Returns:
    --------
    numpy array of shape (n_dirs, 2) with (x, y) points on the contour
    """
    angles = np.linspace(0, 2*np.pi, n_dirs)
    contour = []
    
    for theta in angles:
        dx, dy = np.cos(theta), np.sin(theta)
        
        # Binary search for distance where cost = target_cost
        lo, hi = 0, 200  # larger search range
        for _ in range(50):
            mid = (lo + hi) / 2
            x = x_center + mid * dx
            y = y_center + mid * dy
            
            if rectilinear_cost(x, y, data) < target_cost:
                lo = mid
            else:
                hi = mid
        
        contour.append((x_center + lo * dx, y_center + lo * dy))
    
    return np.array(contour)


def plot_L1_optimal_set(ax, x_range, y_range):
    """
    Plot the L1 optimal solution set on an existing axes.
    
    Visualizes the optimal region which may be a point, line, or rectangle
    depending on whether the medians are unique or ranges.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes object to draw on
    x_range : tuple
        (x_low, x_high) – range of optimal x-coordinates
    y_range : tuple
        (y_low, y_high) – range of optimal y-coordinates
    
    Returns:
    --------
    None (modifies ax in place)
    """
    x_min, x_max = x_range
    y_min, y_max = y_range
    tol = 1e-6
    
    # Case 1: Unique point (both x and y unique)
    if abs(x_min - x_max) < tol and abs(y_min - y_max) < tol:
        ax.plot(x_min, y_min, 'rs', markersize=9, label="L1 optimum")
    
    # Case 2: Vertical line (x unique, y varies)
    elif abs(x_min - x_max) < tol:
        ax.plot([x_min, x_min], [y_min, y_max],
                'r-', linewidth=3, label="L1 optimal line")
    
    # Case 3: Horizontal line (y unique, x varies)
    elif abs(y_min - y_max) < tol:
        ax.plot([x_min, x_max], [y_min, y_min],
                'r-', linewidth=3, label="L1 optimal line")
    
    # Case 4: Rectangle (both x and y vary)
    else:
        rect = patches.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            linewidth=2,
            edgecolor='red',
            facecolor='none',
            label="L1 optimal region"
        )
        ax.add_patch(rect)


def plot_iso_contours_L1_with_optimal_set(data, contour_points):
    """
    Plot iso-cost contours together with the L1 optimal solution set.
    
    Creates a visualization showing:
      - Existing facility locations (black dots)
      - Iso-cost contours (blue circles)
      - The L1 optimal solution region (red)
    
    Parameters:
    -----------
    data : list of tuple
        Facility data (a_i, b_i, w_i)
    contour_points : list of tuple
        Points through which to draw iso-contours
    
    Returns:
    --------
    tuple: (fig, point_info)
        - fig: matplotlib Figure object
        - point_info: list of dicts with {'label', 'x', 'y', 'cost'} for each contour
    """
    # Solve to get optimal set
    res_L1 = solve_single_facility_L1_median(data)
    x_range = res_L1["x_range"]
    y_range = res_L1["y_range"]
    
    fig, ax = plt.subplots(figsize=(9, 9))
    
    # Plot existing facilities
    for i, (a, b, w) in enumerate(data, start=1):
        ax.plot(a, b, 'ko', markersize=4)
        ax.text(a + 0.1, b + 0.1, f"w{i}={w:g}", fontsize=9)
    
    # Plot iso-contours through specified points
    point_info = []
    for idx, (x0, y0) in enumerate(contour_points):
        label = chr(65 + idx)  # A, B, C, ...
        cost = rectilinear_cost(x0, y0, data)
        contour = iso_contour_at_cost(data, x0, y0, cost)
        
        ax.plot(contour[:, 0], contour[:, 1], color="blue", linewidth=1)
        ax.plot(x0, y0, 'bx')
        ax.text(x0 + 0.1, y0 + 0.1, label, fontsize=11, color="blue")
        ax.text(x0 + 0.1, y0 - 0.5, f"f={cost:.1f}", fontsize=9, color="blue")
        
        point_info.append({
            "label": label,
            "x": x0,
            "y": y0,
            "cost": cost
        })
    
    # Plot the optimal solution region
    plot_L1_optimal_set(ax, x_range, y_range)
    
    # Configure plot
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_title("Rectilinear Iso-Cost Contours with Optimal Set (Median Method)")
    ax.legend(loc="best", frameon=False)
    ax.margins(0.15)
    
    return fig, point_info


# ================================================================================
# SECTION 6: EUCLIDEAN (L2) DISTANCE SOLVER (TAB 4)
# ================================================================================

def solve_single_facility_euclidean(
    data,
    tol=1e-6,
    max_iter=1000,
    verbose=False,
    store_history=False
):
    """
    Solve the Minisum Single Facility Location Problem with Euclidean (L2) 
    distance using the Weiszfeld Method.
    
    The Weiszfeld algorithm is an iterative method that:
    1. Starts at the weighted centroid
    2. Moves in weighted average direction inversely proportional to distances
    3. Repeats until convergence
    
    Parameters:
    -----------
    data : list of tuple
        Each element is (a_i, b_i, w_i)
    tol : float
        Convergence tolerance for step size
    max_iter : int
        Maximum number of iterations
    verbose : bool
        Print iteration details if True
    store_history : bool
        Store iteration history if True
    
    Returns:
    --------
    dict with keys:
        - 'x_opt': optimal x-coordinate
        - 'y_opt': optimal y-coordinate
        - 'iterations': number of iterations performed
        - 'converged': True if converged, False if max_iter reached
        - 'history': list of (iter, x, y) tuples if store_history=True, else None
    """
    # Initial point: weighted centroid
    total_weight = sum(w for _, _, w in data)
    x = sum(w * a for a, _, w in data) / total_weight
    y = sum(w * b for _, b, w in data) / total_weight
    
    history = []
    if store_history:
        history.append((0, x, y))
    
    if verbose:
        print("Iter | x              y              step")
    
    # Weiszfeld iterations
    for k in range(max_iter):
        num_x = num_y = denom = 0.0
        
        for a, b, w in data:
            dist = math.hypot(x - a, y - b)
            
            # If current point coincides with facility, return that facility
            if dist < tol:
                return {
                    "x_opt": a,
                    "y_opt": b,
                    "iterations": k,
                    "converged": True,
                    "history": history if store_history else None
                }
            
            # Weight inversely proportional to distance
            phi = w / dist
            num_x += a * phi
            num_y += b * phi
            denom += phi
        
        # Update location
        x_new = num_x / denom
        y_new = num_y / denom
        step = math.hypot(x_new - x, y_new - y)
        
        if verbose:
            print(f"{k+1:4d} | {x_new:14.8f} {y_new:14.8f} {step:10.8f}")
        
        if store_history:
            history.append((k + 1, x_new, y_new))
        
        # Check for convergence
        if step < tol:
            return {
                "x_opt": x_new,
                "y_opt": y_new,
                "iterations": k + 1,
                "converged": True,
                "history": history if store_history else None
            }
        
        x, y = x_new, y_new
    
    # Failed to converge within max_iter
    return {
        "x_opt": x,
        "y_opt": y,
        "iterations": max_iter,
        "converged": False,
        "history": history if store_history else None
    }


def solve_single_facility_squared_euclidean(data):
    """
    Solve the Minisum Single Facility Location Problem with squared Euclidean 
    (L2^2) distance using the Centroid Method.
    
    This has a closed-form solution: the optimal location is simply the
    weighted centroid (weighted average of all facility locations).
    
    Parameters:
    -----------
    data : list of tuple
        Each element is (a_i, b_i, w_i)
    
    Returns:
    --------
    dict with keys:
        - 'x_opt': optimal x-coordinate (weighted average of x)
        - 'y_opt': optimal y-coordinate (weighted average of y)
        - 'opt_val': objective function value at optimum
    """
    # Compute weighted centroid
    total_weight = sum(w for _, _, w in data)
    x_star = sum(w * a for a, _, w in data) / total_weight
    y_star = sum(w * b for _, b, w in data) / total_weight
    
    # Compute objective value at centroid
    obj_val = sum(
        w * ((x_star - a)**2 + (y_star - b)**2)
        for a, b, w in data
    )
    
    return {
        "x_opt": x_star,
        "y_opt": y_star,
        "opt_val": obj_val
    }


# ================================================================================
# SECTION 7: OBJECTIVE FUNCTIONS FOR COMPARISON
# ================================================================================

def euclidean_objective(x, y, data):
    """Evaluate Euclidean (L2) objective function."""
    return sum(w * math.hypot(x - a, y - b) for a, b, w in data)


def obj_L2(x, y, data):
    """
    Evaluate L2 (Euclidean) objective function.
    f_L2(x, y) = sum(w_i * sqrt((x - a_i)^2 + (y - b_i)^2))
    """
    return sum(w * math.hypot(x - a, y - b) for a, b, w in data)


def obj_L2_squared(x, y, data):
    """
    Evaluate L2^2 (Squared Euclidean) objective function.
    f_L2^2(x, y) = sum(w_i * ((x - a_i)^2 + (y - b_i)^2))
    """
    return sum(w * ((x - a)**2 + (y - b)**2) for a, b, w in data)


# ================================================================================
# SECTION 8: MINKOWSKI (Lp) DISTANCE SOLVER (TAB 5)
# ================================================================================

def solve_single_facility_Lp(data, p, alpha=0.1, tol=1e-6, max_iter=500):
    """
    Solve the Minisum Single Facility Location Problem with Lp (Minkowski) 
    distance using Gradient Descent.
    
    The Lp norm generalizes L1, L2, and L-infinity distances:
      - p = 1: Rectilinear (L1) distance
      - p = 2: Euclidean (L2) distance
      - p → ∞: Chebyshev (L∞) distance
    
    For general p, no closed-form solution exists, so gradient descent is used.
    
    Parameters:
    -----------
    data : list of tuple
        Each element is (a_i, b_i, w_i)
    p : float
        The Minkowski distance parameter (p >= 1)
    alpha : float
        Step size for gradient descent
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum number of iterations
    
    Returns:
    --------
    dict with keys:
        - 'x_opt': optimal x-coordinate
        - 'y_opt': optimal y-coordinate
        - 'obj': objective function value at optimum
    """
    # Initial point: weighted centroid
    total_w = sum(w for _, _, w in data)
    x = sum(w * a for a, _, w in data) / total_w
    y = sum(w * b for _, b, w in data) / total_w
    
    # Gradient descent iterations
    for _ in range(max_iter):
        grad_x = 0.0
        grad_y = 0.0
        
        # Compute gradient
        for a, b, w in data:
            dx = x - a
            dy = y - b
            dist_p = (abs(dx)**p + abs(dy)**p)**(1/p)
            
            if dist_p < tol:
                continue
            
            # Gradient components
            grad_x += w * abs(dx)**(p-1) * np.sign(dx) / dist_p**(p-1)
            grad_y += w * abs(dy)**(p-1) * np.sign(dy) / dist_p**(p-1)
        
        # Update location with gradient step
        x_new = x - alpha * grad_x
        y_new = y - alpha * grad_y
        
        # Check for convergence
        if np.hypot(x_new - x, y_new - y) < tol:
            break
        
        x, y = x_new, y_new
    
    # Evaluate objective at final point
    obj_val = sum(
        w * (abs(x - a)**p + abs(y - b)**p)**(1/p)
        for a, b, w in data
    )
    
    return {
        "x_opt": x,
        "y_opt": y,
        "obj": obj_val
    }


# ================================================================================
# SECTION 9: COMPARISON AND VISUALIZATION FUNCTIONS (TAB 6)
# ================================================================================

def compare_single_facility_models(data):
    """
    Solve and compare all minisum distance models.
    
    Solves the facility location problem using L1, L2, and L2^2 metrics
    and compares their optimal solutions and objective values.
    
    Parameters:
    -----------
    data : list of tuple
        Each element is (a_i, b_i, w_i)
    
    Returns:
    --------
    dict with keys for each model name:
        - "L1 (Rectilinear)": {'x_range', 'y_range', 'x_rep', 'y_rep', 'obj'}
        - "L2 (Euclidean)": {'x', 'y', 'obj'}
        - "L2^2 (Squared Euclidean)": {'x', 'y', 'obj'}
    """
    # Solve each model
    res_L1 = solve_single_facility_L1_median(data)
    res_L2 = solve_single_facility_euclidean(data)
    res_L2sq = solve_single_facility_squared_euclidean(data)
    
    # Representative point for L1 (center of optimal rectangle)
    x_low, x_high = res_L1["x_range"]
    y_low, y_high = res_L1["y_range"]
    x_rep = 0.5 * (x_low + x_high)
    y_rep = 0.5 * (y_low + y_high)
    obj_L1_val = obj_L1(x_rep, y_rep, data)
    
    results = {
        "L1 (Rectilinear)": {
            "x_range": (x_low, x_high),
            "y_range": (y_low, y_high),
            "x_rep": x_rep,
            "y_rep": y_rep,
            "obj": obj_L1_val
        },
        "L2 (Euclidean)": {
            "x": res_L2["x_opt"],
            "y": res_L2["y_opt"],
            "obj": obj_L2(res_L2["x_opt"], res_L2["y_opt"], data)
        },
        "L2^2 (Squared Euclidean)": {
            "x": res_L2sq["x_opt"],
            "y": res_L2sq["y_opt"],
            "obj": obj_L2_squared(res_L2sq["x_opt"], res_L2sq["y_opt"], data)
        }
    }
    
    return results


def plot_optimal_locations(data, results, fig_size=(9, 9)):
    """
    Visualize optimal solutions from all distance models on one plot.
    
    Displays:
      - Existing facility locations (black dots)
      - L1 optimal region (red)
      - L2 optimal point (blue triangle)
      - L2^2 optimal point (green diamond)
    
    Parameters:
    -----------
    data : list of tuple
        Each element is (a_i, b_i, w_i)
    results : dict
        Output from compare_single_facility_models()
    fig_size : tuple
        Figure size (width, height)
    
    Returns:
    --------
    matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Plot existing facilities
    for i, (a, b, w) in enumerate(data, start=1):
        ax.plot(a, b, 'ko', markersize=4)
        ax.text(a + 0.1, b + 0.1, f"w{i}={w:g}", fontsize=9)
    
    # Plot L1 optimal set
    plot_L1_optimal_set(
        ax,
        results["L1 (Rectilinear)"]["x_range"],
        results["L1 (Rectilinear)"]["y_range"]
    )
    
    # Plot L2 optimal point
    ax.plot(
        results["L2 (Euclidean)"]["x"],
        results["L2 (Euclidean)"]["y"],
        'b^', markersize=6, label="L2 (Euclidean)"
    )
    
    # Plot L2^2 optimal point
    ax.plot(
        results["L2^2 (Squared Euclidean)"]["x"],
        results["L2^2 (Squared Euclidean)"]["y"],
        'gd', markersize=6, label="L2² (Squared Euclidean)"
    )
    
    # Configure plot
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_title("Minisum Facility Location Solutions")
    ax.legend(loc="best", frameon=False)
    ax.margins(0.15)
    
    return fig


# ================================================================================
# SECTION 10: LP MINKOWSKI DISTANCE VISUALIZATION (TAB 5)
# ================================================================================

def plot_Lp_solution(data, x_opt, y_opt, p, fig_size=(6, 6)):
    """
    Plot the optimal location for a specific Lp distance model.
    
    Parameters:
    -----------
    data : list of tuple
        Facility data (a_i, b_i, w_i)
    x_opt, y_opt : float
        Optimal facility location coordinates
    p : float
        The Minkowski distance parameter
    fig_size : tuple
        Figure size (width, height)
    
    Returns:
    --------
    matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Plot existing facilities
    for i, (a, b, w) in enumerate(data, start=1):
        ax.plot(a, b, 'ko', markersize=4)
        ax.text(
            a + 0.1,
            b + 0.1,
            f"w{i}={w:g}",
            fontsize=9
        )
    
    # Plot optimal location
    ax.plot(
        x_opt,
        y_opt,
        'rs',
        markersize=8,
        label=f"Lp optimum (p={p})"
    )
    
    # Configure plot
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_title(f"Minkowski (Lp) Optimal Location (p = {p})")
    ax.legend(loc="best", frameon=False)
    ax.margins(0.15)
    
    return fig


def plot_Lp_solution_path(
    data,
    path_x,
    path_y,
    x_current,
    y_current,
    p_current,
    fig_size=(6, 6)
):
    """
    Plot the trajectory of optimal locations as p varies in Lp distance.
    
    Visualizes how the optimal facility location changes as the Minkowski
    parameter p increases from 1 to some maximum value. Useful for understanding
    the convergence behavior as p → ∞ (approaching Chebyshev distance).
    
    Parameters:
    -----------
    data : list of tuple
        Facility data (a_i, b_i, w_i)
    path_x, path_y : list of float
        x and y coordinates of optimal locations for p = 1, 2, ..., p_max
    x_current, y_current : float
        Current optimal location (for the user-selected p)
    p_current : int
        Current value of p
    fig_size : tuple
        Figure size (width, height)
    
    Returns:
    --------
    matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Plot existing facilities
    for i, (a, b, w) in enumerate(data, start=1):
        ax.plot(a, b, 'ko', markersize=4)
        ax.text(a + 0.1, b + 0.1, f"w{i}={w:g}", fontsize=9)
    
    # Plot trajectory (blue path for fixed alpha)
    if len(path_x) > 1:
        ax.plot(
            path_x[:-1],
            path_y[:-1],
            'bo-',
            linewidth=1.5,
            markersize=4,
            label="Optimal path (fixed α)"
        )
    
    # Plot current solution (red point for user-selected alpha)
    ax.plot(
        x_current,
        y_current,
        'rs',
        markersize=8,
        label=f"Current solution (p={p_current}, user α)"
    )
    
    # Configure plot
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_title("Lp Optimal Location Trajectory")
    ax.legend(loc="best", frameon=False)
    ax.margins(0.15)
    
    return fig


def plot_iso_contours_from_lp(data, lp_result, contour_points):
    """
    Plot iso-cost contours for an Lp optimal location.
    
    For visualization of iso-contours at the optimal solution point,
    showing how cost varies around the optimal facility location.
    
    Parameters:
    -----------
    data : list of tuple
        Facility data (a_i, b_i, w_i)
    lp_result : dict
        Result from solve_single_facility_L1_median or other solver
        Should contain 'x_opt' and 'y_opt' keys
    contour_points : list of tuple
        Additional points through which to draw contours
    
    Returns:
    --------
    matplotlib Figure object
    """
    x_star = lp_result["x_opt"]
    y_star = lp_result["y_opt"]
    
    fig, ax = plt.subplots(figsize=(9, 9))
    
    # Plot existing facilities
    for i, (a, b, w) in enumerate(data, start=1):
        ax.plot(a, b, 'ko', markersize=4)
        ax.text(a + 0.1, b + 0.1, f"w{i}={w:g}", fontsize=9)
    
    # Plot optimal point
    ax.plot(x_star, y_star, 'rs', markersize=9, label="LP optimum")
    
    # Plot iso-contours through specified points
    for idx, (x0, y0) in enumerate(contour_points):
        contour, cost_val = iso_contour(data, x0, y0)
        ax.plot(contour[:, 0], contour[:, 1], color="blue")
        ax.plot(x0, y0, 'bx')
        ax.text(x0 + 0.1, y0 + 0.1, f"f={cost_val:.1f}", fontsize=9)
    
    # Configure plot
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_title("Rectilinear Iso-Cost Contours (L1)")
    ax.legend(loc="best", frameon=False)
    
    return fig
