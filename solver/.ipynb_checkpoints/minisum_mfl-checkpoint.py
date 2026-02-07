"""
================================================================================
Minisum Multiple Facility Location Problem (Rectilinear Distance)
Coordinate Descent + Weighted Median Method
Based on IIT Kharagpur lecture notes (Dr. J. K. Jha)
================================================================================
"""

import numpy as np


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
