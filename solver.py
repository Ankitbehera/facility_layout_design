# solver.py
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
 
def weighted_median(values, weights):
    """
    Compute weighted median.
    Returns (lower_median, upper_median).
    If unique, both are equal.
    """

    # Sort by values
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

def classify_L1_solution(x_range, y_range, tol=1e-6):
    x_low, x_high = x_range
    y_low, y_high = y_range

    if abs(x_low - x_high) < tol and abs(y_low - y_high) < tol:
        return "Unique Point", [(x_low, y_low)]

    if abs(x_low - x_high) < tol:
        return "Vertical Line", [(x_low, y_low), (x_low, y_high)]

    if abs(y_low - y_high) < tol:
        return "Horizontal Line", [(x_low, y_low), (x_high, y_low)]

    return "Rectangle", [(x_low, y_low),(x_low, y_high),(x_high, y_high),(x_high, y_low),]

def solve_single_facility_L1_median(data):
    """
    Single Facility Location with Rectilinear Distance
    using the Median Method
    """

    a_vals = [a for a, _, _ in data]
    b_vals = [b for _, b, _ in data]
    weights = [w for _, _, w in data]

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

def build_weighted_median_table(values, weights, labels):
    df = pd.DataFrame({
        "Existing Facility": labels,
        "Coordinate": values,
        "Weight": weights
    })

    df = df.sort_values("Coordinate").reset_index(drop=True)
    df["Cumulative Weight"] = df["Weight"].cumsum()

    total_weight = df["Weight"].sum()
    half_weight = total_weight / 2

    median_row = df[df["Cumulative Weight"] >= half_weight].iloc[0]

    return df, median_row, total_weight
    
def rectilinear_cost(x, y, data):
    return sum(w * (abs(x - a) + abs(y - b)) for a, b, w in data)

def plot_L1_optimal_set(ax, x_range, y_range):
    x_min, x_max = x_range
    y_min, y_max = y_range

    tol = 1e-6

    # Unique point
    if abs(x_min - x_max) < tol and abs(y_min - y_max) < tol:
        ax.plot(x_min, y_min, 'rs', markersize=9, label="L1 optimum")

    # Vertical line
    elif abs(x_min - x_max) < tol:
        ax.plot([x_min, x_min], [y_min, y_max],
                'r-', linewidth=3, label="L1 optimal line")

    # Horizontal line
    elif abs(y_min - y_max) < tol:
        ax.plot([x_min, x_max], [y_min, y_min],
                'r-', linewidth=3, label="L1 optimal line")

    # Rectangle
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
        
def iso_contour(data, x0, y0, n_dirs=360):
    base_cost = rectilinear_cost(x0, y0, data)

    angles = np.linspace(0, 2*np.pi, n_dirs)
    contour = []

    for theta in angles:
        dx, dy = np.cos(theta), np.sin(theta)
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
    angles = np.linspace(0, 2*np.pi, n_dirs)
    contour = []

    for theta in angles:
        dx, dy = np.cos(theta), np.sin(theta)
        lo, hi = 0, 200  # large enough

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

# if solve_single_facility_L1(data) is used to get lp_result
def plot_iso_contours_from_lp(data, lp_result, contour_points):
    x_star = lp_result["x_opt"]
    y_star = lp_result["y_opt"]

    fig, ax = plt.subplots(figsize=(9,9))

    # Facilities
    for i, (a, b, w) in enumerate(data, start=1):
        ax.plot(a, b, 'ko', markersize=4)
        ax.text(a + 0.1, b + 0.1, f"w{i}={w:g}", fontsize=9)

    # LP optimum
    ax.plot(x_star, y_star, 'rs', markersize=9, label="LP optimum")

    # Iso-contours
    for idx, (x0, y0) in enumerate(contour_points):
        contour, cost_val = iso_contour(data, x0, y0)
        ax.plot(contour[:, 0], contour[:, 1], color="blue")
        ax.plot(x0, y0, 'bx')
        ax.text(x0 + 0.1, y0 + 0.1, f"f={cost_val:.1f}", fontsize=9)

    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_title("Rectilinear Iso-Cost Contours (L1)")
    ax.legend(loc="best", frameon=False)

    return fig

def plot_iso_contours_L1_with_optimal_set(data, contour_points):
    res_L1 = solve_single_facility_L1_median(data)
    x_range = res_L1["x_range"]
    y_range = res_L1["y_range"]

    fig, ax = plt.subplots(figsize=(9, 9))

    # Facilities
    for i, (a, b, w) in enumerate(data, start=1):
        ax.plot(a, b, 'ko', markersize = 4)
        ax.text(a + 0.1, b + 0.1, f"w{i}={w:g}", fontsize=9)

    point_info = []

    # Iso-contours
    for idx, (x0, y0) in enumerate(contour_points):
        label = chr(65 + idx)

        cost = rectilinear_cost(x0, y0, data)
        contour = iso_contour_at_cost(data, x0, y0, cost)

        ax.plot(contour[:, 0], contour[:, 1], color="blue", linewidth =1)
        ax.plot(x0, y0, 'bx')

        ax.text(x0 + 0.1, y0 + 0.1, label, fontsize=11, color="blue")
        ax.text(x0 + 0.1, y0 - 0.5, f"f={cost:.1f}", fontsize=9, color="blue")

        point_info.append({
            "label": label,
            "x": x0,
            "y": y0,
            "cost": cost
        })

    plot_L1_optimal_set(ax, x_range, y_range)

    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_title("Rectilinear Iso-Cost Contours with Optimal Set (Median Method)")
    ax.legend(loc="best", frameon=False)
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0.15)

    return fig, point_info


def solve_single_facility_squared_euclidean(data):
    """
    Minisum Single Facility Location Problem
    with squared Euclidean distance
    """

    total_weight = sum(w for _, _, w in data)

    x_star = sum(w * a for a, _, w in data) / total_weight
    y_star = sum(w * b for _, b, w in data) / total_weight

    obj_val = sum(
        w * ((x_star - a)**2 + (y_star - b)**2)
        for a, b, w in data
    )

    return {
        "x_opt": x_star,
        "y_opt": y_star,
        "opt_val": obj_val
    }

import math

def solve_single_facility_euclidean(
    data,
    tol=1e-6,
    max_iter=1000,
    verbose=False,
    store_history=False
):
    """
    Minisum Single Facility Location Problem
    with Euclidean (L2) distance using Weiszfeld Method

    Parameters:
    - verbose: print iteration details if True
    - store_history: store iteration history if True
    """

    # Initial point (weighted centroid)
    total_weight = sum(w for _, _, w in data)
    x = sum(w * a for a, _, w in data) / total_weight
    y = sum(w * b for _, b, w in data) / total_weight

    history = []

    if store_history:
        history.append((0, x, y))

    if verbose:
        print("Iter |        x        y        step")

    for k in range(max_iter):
        num_x = num_y = denom = 0.0

        for a, b, w in data:
            dist = math.hypot(x - a, y - b)

            if dist < tol:
                return {
                    "x_opt": a,
                    "y_opt": b,
                    "iterations": k,
                    "converged": True,
                    "history": history if store_history else None
                }

            phi = w / dist
            num_x += a * phi
            num_y += b * phi
            denom += phi

        x_new = num_x / denom
        y_new = num_y / denom

        step = math.hypot(x_new - x, y_new - y)

        if verbose:
            print(f"{k+1:4d} | {x_new:8.5f} {y_new:8.5f} {step:8.6f}")

        if store_history:
            history.append((k + 1, x_new, y_new))

        if step < tol:
            return {
                "x_opt": x_new,
                "y_opt": y_new,
                "iterations": k + 1,
                "converged": True,
                "history": history if store_history else None
            }

        x, y = x_new, y_new

    return {
        "x_opt": x,
        "y_opt": y,
        "iterations": max_iter,
        "converged": False,
        "history": history if store_history else None
    }

def euclidean_objective(x, y, data):
    return sum(w * math.hypot(x - a, y - b) for a, b, w in data)

def obj_L1(x, y, data):
    return sum(w * (abs(x - a) + abs(y - b)) for a, b, w in data)

def obj_L2(x, y, data):
    return sum(w * math.hypot(x - a, y - b) for a, b, w in data)

def obj_L2_squared(x, y, data):
    return sum(w * ((x - a)**2 + (y - b)**2) for a, b, w in data)

def compare_single_facility_models(data):
    """
    Solve and compare L1, L2, and L2^2 minisum problems
    """

    # --- Solve models ---
    res_L1 = solve_single_facility_L1_median(data)
    res_L2 = solve_single_facility_euclidean(data)
    res_L2sq = solve_single_facility_squared_euclidean(data)

    # --- Representative point for L1 (valid for objective) ---
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
            "obj": obj_L2_squared(
                res_L2sq["x_opt"],
                res_L2sq["y_opt"],
                data
            )
        }
    }

    return results

def plot_optimal_locations(data, results, fig_size=(9, 9)):
    fig, ax = plt.subplots(figsize=fig_size)

    # Plot facilities
    for i, (a, b, w) in enumerate(data, start=1):
        ax.plot(a, b, 'ko',markersize =4)
        ax.text(a + 0.1, b + 0.1, f"w{i}={w:g}", fontsize=9)

    # Plot optimal solutions
    plot_L1_optimal_set(
        ax,
        results["L1 (Rectilinear)"]["x_range"],
        results["L1 (Rectilinear)"]["y_range"]
    )


    ax.plot(results["L2 (Euclidean)"]["x"],
            results["L2 (Euclidean)"]["y"],
            'b^', markersize=6, label="L2 (Euclidean)")

    ax.plot(results["L2^2 (Squared Euclidean)"]["x"],
            results["L2^2 (Squared Euclidean)"]["y"],
            'gd', markersize=6, label="L2² (Squared Euclidean)")

    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.set_xlabel("x-axis")
    ax.set_ylabel("y-axis")
    ax.set_title("Minisum Facility Location Solutions")
    ax.legend(loc="best", frameon=False)
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0.15)

    return fig
#------------
#Graphical Approch Functions
#---------------
def f1_value(x, data):
    return sum(w * abs(x - a) for a, _, w in data)

def f2_value(y, data):
    return sum(w * abs(y - b) for _, b, w in data)

def plot_piecewise_L1(values, func, label):
    xs = sorted(set(values))
    ys = [func(x) for x in xs]

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.plot(xs, ys, marker='o')
    
    for x, y in zip(xs, ys):
        ax.text(x, y + 0.03 * max(ys), f"({x}, {int(y)})", fontsize=9)

    min_idx = ys.index(min(ys))
    ax.axvline(xs[min_idx], linestyle="--", alpha=0.6)
    ax.axhline(ys[min_idx], linestyle="--", alpha=0.6)

    ax.set_xlabel(label)
    ax.set_ylabel(f"f({label})")
    ax.grid(True, linestyle="--", alpha=0.5)

    return fig, xs[min_idx], ys[min_idx]
