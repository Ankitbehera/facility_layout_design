import streamlit as st
import pandas as pd
import pulp
import matplotlib.pyplot as plt
import solver.minisum_mfl as slv
import numpy as np
"""
================================================================================
Minisum Multiple Facility Location Problem (UI)
================================================================================
"""

# ------------------------------------------------------------------------------
# Input Builder
# ------------------------------------------------------------------------------

def build_inputs():
    # ==================================================
    # Sidebar: Input Data (Minisum MFL)
    # ==================================================
    st.sidebar.header("Input Data")

    # --------------------------------------------------
    # 0. Example Loader
    # --------------------------------------------------
    with st.sidebar.expander("Load Example Problem", expanded=False):
        st.caption("Select a preset Example to load data.")
        
        col_ex1, col_ex2 = st.columns(2)
        col_ex3, col_ex4 = st.columns(2)
        
        # Buttons
        load_ex1 = col_ex1.button("Example 1", help="Unique 1")
        load_ex2 = col_ex2.button("Example 2", help="Unique 2")
        load_ex3 = col_ex3.button("Example 3", help="Multiple 1")
        load_ex4 = col_ex4.button("Example 4", help="Unique 3")
        
    # Logic to load examples into Session State
    # We must update "m_input" and "n_input" keys to force the widgets to update visually
    if load_ex1:
        # --- Example 1: Default Textbook ---
        m_new, n_new = 4, 2
        st.session_state.m_val = m_new
        st.session_state.n_val = n_new
        st.session_state["m_input"] = m_new
        st.session_state["n_input"] = n_new
        
        st.session_state.ef_df = pd.DataFrame({
            "a_i (x-coordinate)": [0, 15, 25, 40],
            "b_i (y-coordinate)": [10, 0, 30, 15]
        }, index=[f"EF{i+1}" for i in range(4)])
        
        st.session_state.w_df = pd.DataFrame([
            [10, 6, 0, 0], [0, 2, 16, 8]
        ], index=["NF1", "NF2"], columns=[f"EF{i+1}" for i in range(4)])
        
        # V matrix: v12 = 12
        v_in = pd.DataFrame("", index=["NF1", "NF2"], columns=["NF1", "NF2"])
        v_in.iloc[0, 1] = "12"
        v_in.iloc[1, 0] = "auto-fill"
        st.session_state.v_input = v_in
        st.rerun()

    if load_ex2:
        # --- Example 2: Numerical Example 2 ---
        m_new, n_new = 4, 2
        st.session_state.m_val = m_new
        st.session_state.n_val = n_new
        st.session_state["m_input"] = m_new
        st.session_state["n_input"] = n_new

        st.session_state.ef_df = pd.DataFrame({
            "a_i (x-coordinate)": [0, 4, 6, 10],
            "b_i (y-coordinate)": [2, 0, 8, 4]
        }, index=[f"EF{i+1}" for i in range(4)])
        
        st.session_state.w_df = pd.DataFrame([
            [5, 3, 0, 0], [0, 1, 8, 4]
        ], index=["NF1", "NF2"], columns=[f"EF{i+1}" for i in range(4)])
        
        # V matrix: v12 = 6
        v_in = pd.DataFrame("", index=["NF1", "NF2"], columns=["NF1", "NF2"])
        v_in.iloc[0, 1] = "6"
        v_in.iloc[1, 0] = "auto-fill"
        st.session_state.v_input = v_in
        st.rerun()

    if load_ex3:
        # --- Example 3: 3 NFs, 6 EFs ---
        m_new, n_new = 6, 3
        st.session_state.m_val = m_new
        st.session_state.n_val = n_new
        st.session_state["m_input"] = m_new
        st.session_state["n_input"] = n_new

        st.session_state.ef_df = pd.DataFrame({
            "a_i (x-coordinate)": [0, 15, 10, 5, 20, 25],
            "b_i (y-coordinate)": [10, 0, 25, 15, 20, 5]
        }, index=[f"EF{i+1}" for i in range(6)])
        
        st.session_state.w_df = pd.DataFrame([
            [4, 2, 0, 4, 0, 0],   # NF1
            [2, 2, 4, 0, 0, 7],   # NF2
            [0, 0, 0, 2, 5, 0]    # NF3
        ], index=["NF1", "NF2", "NF3"], columns=[f"EF{i+1}" for i in range(6)])
        
        # V matrix: v12=1, v13=3, v23=2
        v_in = pd.DataFrame("", index=["NF1", "NF2", "NF3"], columns=["NF1", "NF2", "NF3"])
        for r in range(3):
            for c in range(3):
                if r >= c: v_in.iloc[r, c] = "auto-fill"
        v_in.iloc[0, 1] = "1"
        v_in.iloc[0, 2] = "3"
        v_in.iloc[1, 2] = "2"
        st.session_state.v_input = v_in
        st.rerun()

    if load_ex4:
        # --- Example 4: Machine Shop ---
        m_new, n_new = 5, 2
        st.session_state.m_val = m_new
        st.session_state.n_val = n_new
        st.session_state["m_input"] = m_new
        st.session_state["n_input"] = n_new

        st.session_state.ef_df = pd.DataFrame({
            "a_i (x-coordinate)": [10, 10, 15, 20, 25],
            "b_i (y-coordinate)": [25, 15, 30, 10, 25]
        }, index=[f"EF{i+1}" for i in range(5)])
        
        st.session_state.w_df = pd.DataFrame([
            [10, 6, 5, 4, 3],
            [2, 3, 4, 6, 12]
        ], index=["NF1", "NF2"], columns=[f"EF{i+1}" for i in range(5)])
        
        # V matrix: v12 = 4
        v_in = pd.DataFrame("", index=["NF1", "NF2"], columns=["NF1", "NF2"])
        v_in.iloc[0, 1] = "4"
        v_in.iloc[1, 0] = "auto-fill"
        st.session_state.v_input = v_in
        st.rerun()


    # --------------------------------------------------
    # 1. Problem Size Input
    # --------------------------------------------------
    st.sidebar.markdown("#### Problem Dimensions")

    col_m, col_n = st.sidebar.columns(2)

    # Initialize session state for m and n if not present
    if "m_val" not in st.session_state: st.session_state.m_val = 4
    if "n_val" not in st.session_state: st.session_state.n_val = 2

    # Note: We use keys 'm_input' and 'n_input' to sync with the example loader
    with col_m:
        m = st.number_input(
            "Number of Existing Facilities ($m$)",
            min_value=1, step=1,
            value=st.session_state.m_val,
            key="m_input"
        )

    with col_n:
        n = st.number_input(
            "Number of New Facilities ($n$)",
            min_value=1, step=1,
            value=st.session_state.n_val,
            key="n_input"
        )

    # Detect change in dimensions to reset/resize tables
    dims_changed = (m != st.session_state.m_val) or (n != st.session_state.n_val)

    # Update state
    st.session_state.m_val = m
    st.session_state.n_val = n

    st.sidebar.caption("Adjusting $m$ or $n$ will resize tables below.")

    # --------------------------------------------------
    # 2. Existing Facility Locations (a_i, b_i)
    # --------------------------------------------------
    st.sidebar.markdown("### Existing Facility Locations $(a_i,b_i)$")

    # If dims changed or df missing, rebuild EF DataFrame
    if dims_changed or "ef_df" not in st.session_state or len(st.session_state.ef_df) != m:
        st.session_state.ef_df = pd.DataFrame(
            {
                "a_i (x-coordinate)": [10.0 * (i + 1) for i in range(m)],
                "b_i (y-coordinate)": [5.0 * ((i % 3) + 1) * (i + 1) for i in range(m)],
            },
            index=[f"EF{i+1}" for i in range(m)]
        )

    ef_df = st.sidebar.data_editor(
        st.session_state.ef_df,
        key="ef_table",
        num_rows="fixed",
        use_container_width=True
    )

    if ef_df.isnull().any().any():
        st.sidebar.error("Existing facility table contains empty cells.")
        return None

    # --------------------------------------------------
    # 3. EF–NF Interaction Weights w_ji
    # --------------------------------------------------
    st.sidebar.markdown("### EF–NF Interaction Weights $w_{ji}$")

    # If dims changed or df missing, rebuild W DataFrame with UNEVEN weights
    if dims_changed or "w_df" not in st.session_state or st.session_state.w_df.shape != (n, m):
        new_w = []
        for r in range(n):
            row_vals = []
            for c in range(m):
                # Pattern: (row + col + 1) * 3 % 9 + 1 -> creates values like 4, 7, 1...
                val = ((r + c + 2) * 3) % 9 + 1 
                row_vals.append(val)
            new_w.append(row_vals)

        st.session_state.w_df = pd.DataFrame(
            new_w,
            index=[f"NF{j+1}" for j in range(n)],
            columns=[f"EF{i+1}" for i in range(m)]
        )

    w_df = st.sidebar.data_editor(
        st.session_state.w_df,
        key="w_table",
        num_rows="fixed",
        use_container_width=True
    )

    if w_df.isnull().any().any():
        st.sidebar.error("EF–NF weight matrix contains empty cells.")
        return None

    # --------------------------------------------------
    # 4. NF–NF Interaction Weights v_jk (upper triangular)
    # --------------------------------------------------
    st.sidebar.markdown("### NF–NF Interaction Weights $v_{jk}$")

    # If dims changed or df missing, rebuild V input with UNEVEN weights
    if dims_changed or "v_input" not in st.session_state or st.session_state.v_input.shape != (n, n):
        v_input = pd.DataFrame(
            "",
            index=[f"NF{j+1}" for j in range(n)],
            columns=[f"NF{k+1}" for k in range(n)]
        )
        
        for j in range(n):
            for k in range(n):
                if j == k:
                    v_input.iloc[j, k] = "auto-fill"
                elif j > k:
                    v_input.iloc[j, k] = "auto-fill"
                else:
                    # Fill upper triangle (j < k) with uneven pattern
                    # Example pattern: ((j + k) * 5) % 15 + 2 -> values like 2, 7, 12...
                    val = ((j + k + 1) * 5) % 15 + 2
                    v_input.iloc[j, k] = str(val)
        
        st.session_state.v_input = v_input

    v_edit = st.sidebar.data_editor(
        st.session_state.v_input,
        key="v_table",
        num_rows="fixed",
        use_container_width=True
    )

    # Build symmetric numeric matrix v_jk
    v_df = pd.DataFrame(0.0, index=v_edit.index, columns=v_edit.columns)
    for j in range(n):
        for k in range(j + 1, n):
            try:
                val = float(v_edit.iloc[j, k])
            except:
                val = 0.0
            v_df.iloc[j, k] = val
            v_df.iloc[k, j] = val

    st.sidebar.caption(
        r"""
        **Notes:** $v_{jk}$ represents interaction between new facilities. 
        Only fill the upper triangle ($j < k$).
        """
    )

    # --------------------------------------------------
    # Final packaged input
    # --------------------------------------------------
    return {
        "m": m,
        "n": n,
        "ef": ef_df,   # DataFrame: a_i, b_i
        "w": w_df,    # DataFrame: w_ji
        "v": v_df     # DataFrame: symmetric, v_jk with zero diagonal
    }

# =============================================================================
# MAIN PAGE
# =============================================================================
def show_minisum_mfl(data):
    if data is None:
        st.warning("Please complete valid input data to proceed.")
        return
        
    st.title("Minisum Multi-Facility Location Problem")

    # Plot styling 
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    # --------------------------------------------------
    # Tabs 
    # --------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview",
        "Rectilinear (LP)",
        "Coordinate Descent",
        "Euclidean Models"
        
    ])

    # =============================================================================
    # TAB 1: OVERVIEW
    # =============================================================================
    with tab1:
        st.subheader("Multi-Facility Location Problem")

        st.markdown(
            """
            The **Minisum Multi-Facility Location Problem (MFLP)** determines the locations
            of **multiple new facilities** such that the **total interaction cost**
            with existing facilities and between new facilities is minimized.
            """
        )

        st.markdown("### Problem Statement")

        st.latex(
            r"""
            \min \;
            \sum_{1 \le j < k \le n}
            v_{jk} \, d(X_j, X_k)
            \;+\;
            \sum_{j=1}^{n}\sum_{i=1}^{m}
            w_{ji} \, d(X_j, P_i)
            """
        )

        st.markdown("### Interpretation")
        st.markdown(
            """
            - $P_i = (a_i, b_i)$: existing facilities, $i = 1,\ldots,m$  
            - $X_j = (x_j, y_j)$: new facilities to be located, $j = 1,\ldots,n$  
            - $w_{ji}$: interaction weight between EF $i$ and NF $j$  
            - $v_{jk}$: interaction weight between NF $j$ and NF $k$  
            - $d(\cdot,\cdot)$: distance metric (L1, L2, etc.)
            """
        )

        st.markdown("### Input Data Instructions")

        st.markdown(
            """
            The Minisum Multi-Facility Location Problem requires the following inputs:

            - **Existing Facility Locations $(a_i, b_i)$**:  
              Fixed coordinates of existing facilities, $i = 1,\\ldots,m$.

            - **EF–NF Interaction Weights $w_{ji}$**:  
              Weight of interaction between new facility $j$ and existing facility $i$.
              These weights form an $n \\times m$ matrix.

            - **NF–NF Interaction Weights $v_{jk}$**:  
              Interaction between new facilities $j$ and $k$, with $v_{jk} = v_{kj}$ and $v_{jj} = 0$.
              Only values for $j < k$ need to be specified.

            All input values must be non-negative and numeric.
            """
        )

        # --- Input Data Summary Section ---
        show_data = st.checkbox("Show Input Data Summary", value=True)

        if show_data:
            st.markdown("### Input Data Summary")
        
            # Row 1: Existing Facilities
            st.markdown("#### 1. Existing Facility Locations $P_i(a_i, b_i)$")
            st.dataframe(data["ef"], use_container_width=True, height=200)

            # Row 2: Weights (Side-by-Side)
            col_w, col_v = st.columns(2)
        
            with col_w:
                st.markdown("#### 2. EF–NF Weights ($w_{ji}$)")
                st.dataframe(data["w"], use_container_width=True)
                st.caption("Interaction weights between New and Existing facilities.")

            with col_v:
                st.markdown("#### 3. NF–NF Weights ($v_{jk}$)")
                st.dataframe(data["v"], use_container_width=True)
                st.caption("Interaction weights between pairs of New facilities.")

        st.divider()

        # --- Solution Methods Section ---
        st.markdown("### Solution Methods Implemented")
    
        col_m1, col_m2 = st.columns(2)
    
        with col_m1:
            st.markdown("**Rectilinear Distance ($L_1$)**")
            st.markdown(
                """
                * **Linear Programming (LP):** Uses auxiliary variables to find the global optimal solution.
                * **Coordinate Descent:** An iterative approach using the weighted median method.
                """
            )

        with col_m2:
            st.markdown("**Euclidean Distance ($L_2$ / $L_2^2$)**")
            st.markdown(
                """
                * **Squared Euclidean:** Solved using a system of linear equations (Weighted Average).
                * **Standard Euclidean:** Solved via iterative gradient-based methods.
                """
            )

        st.divider()
    
    # =============================================================================
    # TAB 2: RECTILINEAR (LP)
    # =============================================================================
    with tab2:
        st.subheader("Equivalent Linear Programming Formulation")

        m = data["m"]
        n = data["n"]
        ef_df = data["ef"]
        w_df = data["w"]
        v_df = data["v"]

        st.markdown(
            """
            We convert the **Minisum Multi-Facility Location Problem**
            with **rectilinear (L1) distance** into an **equivalent linear program**
            by eliminating absolute values using auxiliary variables. 
            """
        )

        # --------------------------------------------------
        # Original Problem
        # --------------------------------------------------
        st.markdown("### Original Problem")
        st.latex(
            r"""
            \min f(X, Y) = 
            \sum_{1 \le j < k \le n} v_{jk} \left( |x_j - x_k| + |y_j - y_k| \right) 
            + \sum_{j=1}^{n}\sum_{i=1}^{m} w_{ji} \left( |x_j - a_i| + |y_j - b_i| \right)
            """
        )

        # --------------------------------------------------
        # Change of Variables
        # --------------------------------------------------
        st.markdown("### Change of Variables")
        st.write("We define non-negative auxiliary variables to linearize the absolute differences.")

        col_vars_x, col_vars_y = st.columns(2)

        with col_vars_x:
            st.markdown("#### X-Coordinate Variables")
            st.markdown("**1. NF-EF Interactions:**")
            st.latex(
                r"""
                \begin{aligned}
                r_{ji} &: \text{amount NF } j \text{ is to the RIGHT of EF } i \\
                s_{ji} &: \text{amount NF } j \text{ is to the LEFT of EF } i
                \end{aligned}
                """
            )
            st.latex(r"|x_j - a_i| = r_{ji} + s_{ji}")
            
            st.markdown("**2. NF-NF Interactions:**")
            st.latex(
                r"""
                \begin{aligned}
                p_{jk} &: \text{amount NF } j \text{ is to the RIGHT of NF } k \\
                q_{jk} &: \text{amount NF } j \text{ is to the LEFT of NF } k
                \end{aligned}
                """
            )
            st.latex(r"|x_j - x_k| = p_{jk} + q_{jk}")

        with col_vars_y:
            st.markdown("#### Y-Coordinate Variables")
            st.markdown("**1. NF-EF Interactions:**")
            st.latex(
                r"""
                \begin{aligned}
                u_{ji} &: \text{amount NF } j \text{ is ABOVE EF } i \\
                d_{ji} &: \text{amount NF } j \text{ is BELOW EF } i
                \end{aligned}
                """
            )
            st.latex(r"|y_j - b_i| = u_{ji} + d_{ji}")

            st.markdown("**2. NF-NF Interactions:**")
            st.latex(
                r"""
                \begin{aligned}
                a_{jk} &: \text{amount NF } j \text{ is ABOVE NF } k \\
                b_{jk} &: \text{amount NF } j \text{ is BELOW NF } k
                \end{aligned}
                """
            )
            st.latex(r"|y_j - y_k| = a_{jk} + b_{jk}")

        # --------------------------------------------------
        # Equivalent LP
        # --------------------------------------------------
        st.markdown("### Equivalent Linear Program")
        
        st.markdown("**Objective Function:**")
        st.latex(
            r"""
            \begin{aligned}
            \min Z = 
            &\sum_{1 \le j < k \le n} v_{jk}(p_{jk} + q_{jk}) + \sum_{j=1}^{n}\sum_{i=1}^{m} w_{ji}(r_{ji} + s_{ji}) \\
            + &\sum_{1 \le j < k \le n} v_{jk}(a_{jk} + b_{jk}) + \sum_{j=1}^{n}\sum_{i=1}^{m} w_{ji}(u_{ji} + d_{ji})
            \end{aligned}
            """
        )
        
        st.markdown("**Constraints:**")
        st.latex(
            r"""
            \begin{aligned}
            x_j - x_k + p_{jk} - q_{jk} &= 0, & \forall 1 \le j < k \le n \\
            x_j - r_{ji} + s_{ji} &= a_i, & \forall j, i \\
            y_j - y_k + a_{jk} - b_{jk} &= 0, & \forall 1 \le j < k \le n \\
            y_j - u_{ji} + d_{ji} &= b_i, & \forall j, i \\
            \text{All variables} &\ge 0
            \end{aligned}
            """
        )

        # --------------------------------------------------
        # Expanded Linear Objective Function
        # --------------------------------------------------
        st.markdown("### Expanded Linear Objective Function")
        
        obj_terms = []
        
        # X-Axis Terms
        for j in range(n):
            for k in range(j+1, n):
                val = v_df.iloc[j, k]
                if val > 0:
                    obj_terms.append(f"{val:g}(p_{{{j+1}{k+1}}} + q_{{{j+1}{k+1}}})")
        for j in range(n):
            for i in range(m):
                val = w_df.iloc[j, i]
                if val > 0:
                    obj_terms.append(f"{val:g}(r_{{{j+1}{i+1}}} + s_{{{j+1}{i+1}}})")
        
        # Y-Axis Terms
        for j in range(n):
            for k in range(j+1, n):
                val = v_df.iloc[j, k]
                if val > 0:
                    obj_terms.append(f"{val:g}(a_{{{j+1}{k+1}}} + b_{{{j+1}{k+1}}})")
        for j in range(n):
            for i in range(m):
                val = w_df.iloc[j, i]
                if val > 0:
                    obj_terms.append(f"{val:g}(u_{{{j+1}{i+1}}} + d_{{{j+1}{i+1}}})")
        
        if obj_terms:
            # Join with + and line breaks for readability
            equation_latex = r"\min Z = " + " + ".join(obj_terms)
            st.latex(equation_latex)
        else:
            st.write("No non-zero weights defined.")

        # --------------------------------------------------
        # Constraints 
        # --------------------------------------------------
        st.markdown("### Constraints")
        
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.markdown("**X-Coordinates**")
            # EF Constraints
            for j in range(n):
                for i in range(m):
                    # Uses original index i directly
                    val = ef_df.iloc[i, 0]
                    st.latex(rf"x_{{{j+1}}} - r_{{{j+1}{i+1}}} + s_{{{j+1}{i+1}}} = {val:g}")
            
            # NF Constraints
            if n > 1:
                st.markdown("---")
                for j in range(n):
                    for k in range(j+1, n):
                        st.latex(rf"x_{{{j+1}}} - x_{{{k+1}}} + p_{{{j+1}{k+1}}} - q_{{{j+1}{k+1}}} = 0")
                        
        with col_c2:
            st.markdown("**Y-Coordinates**")
            # EF Constraints
            for j in range(n):
                for i in range(m):
                    # Uses original index i directly
                    val = ef_df.iloc[i, 1]
                    st.latex(rf"y_{{{j+1}}} - u_{{{j+1}{i+1}}} + d_{{{j+1}{i+1}}} = {val:g}")
            
            # NF Constraints
            if n > 1:
                st.markdown("---")
                for j in range(n):
                    for k in range(j+1, n):
                        st.latex(rf"y_{{{j+1}}} - y_{{{k+1}}} + a_{{{j+1}{k+1}}} - b_{{{j+1}{k+1}}} = 0")

        # --------------------------------------------------
        # Solve LP
        # --------------------------------------------------
        st.markdown("---")
        solve_lp = st.checkbox("Solve this LP using PuLP", value = True)

        if solve_lp:
            st.subheader("PuLP Solution")
            
            # Prepare data
            existing_coords = data["ef"].values.tolist()
            w_matrix = data["w"].values.tolist()
            v_matrix = data["v"].values.tolist()

            # Call Solver
            with st.spinner("Solving Linear Program..."):
                lp_result = slv.solve_minisum_mfl_lp(existing_coords, w_matrix, v_matrix)

            # -----------------------------
            # Optimal Solution Display
            # -----------------------------
            st.markdown("### Optimal Locations")
            
            res_df = pd.DataFrame(
                lp_result["X_opt"],
                columns=["x*", "y*"],
                index=[f"NF{j+1}" for j in range(n)]
            )
            st.markdown(rf"##### Total Min Cost: {lp_result['obj']}")
            st.table(res_df)
            
            # -----------------------------
            # Decision Variable Tables
            # -----------------------------
            det_x = lp_result["details_x"]
            det_y = lp_result["details_y"]
            
            # --- X Variables ---
            st.markdown("### Decision Variables: X-Axis")
            col_x1, col_x2 = st.columns(2)
            
            with col_x1:
                st.markdown("**NF-EF (r, s)**")
                rows_ef_x = []
                for j in range(n):
                    for i in range(m):
                        rows_ef_x.append({
                            "Var": f"NF{j+1}-EF{i+1}",
                            "r (Right)": f"{det_x['r'][j][i]:.2f}",
                            "s (Left)": f"{det_x['s'][j][i]:.2f}"
                        })
                st.dataframe(pd.DataFrame(rows_ef_x), hide_index=True, use_container_width=True)
                
            with col_x2:
                st.markdown("**NF-NF (p, q)**")
                rows_nf_x = []
                if n > 1:
                    for j in range(n):
                        for k in range(j+1, n):
                            rows_nf_x.append({
                                "Var": f"NF{j+1}-NF{k+1}",
                                "p (Right)": f"{det_x['p'][j][k]:.2f}",
                                "q (Left)": f"{det_x['q'][j][k]:.2f}"
                            })
                    st.dataframe(pd.DataFrame(rows_nf_x), hide_index=True, use_container_width=True)
                else:
                    st.write("N/A (Only 1 New Facility)")

            # --- Y Variables ---
            st.markdown("### Decision Variables: Y-Axis")
            col_y1, col_y2 = st.columns(2)
            
            with col_y1:
                st.markdown("**NF-EF (u, d)**")
                rows_ef_y = []
                for j in range(n):
                    for i in range(m):
                        rows_ef_y.append({
                            "Var": f"NF{j+1}-EF{i+1}",
                            "u (Up)": f"{det_y['r'][j][i]:.2f}",   # Mapping internal 'r' to 'u'
                            "d (Down)": f"{det_y['s'][j][i]:.2f}" # Mapping internal 's' to 'd'
                        })
                st.dataframe(pd.DataFrame(rows_ef_y), hide_index=True, use_container_width=True)

            with col_y2:
                st.markdown("**NF-NF (a, b)**")
                rows_nf_y = []
                if n > 1:
                    for j in range(n):
                        for k in range(j+1, n):
                            rows_nf_y.append({
                                "Var": f"NF{j+1}-NF{k+1}",
                                "a (Above)": f"{det_y['p'][j][k]:.2f}", # Mapping internal 'p' to 'a'
                                "b (Below)": f"{det_y['q'][j][k]:.2f}"  # Mapping internal 'q' to 'b'
                            })
                    st.dataframe(pd.DataFrame(rows_nf_y), hide_index=True, use_container_width=True)
                else:
                    st.write("N/A (Only 1 New Facility)")
            st.markdown("**Interpretation of Active Constraints:**")

            col_interp_x, col_interp_y = st.columns(2)

            with col_interp_x:
                st.markdown("**X-Direction (Horizontal Alignment):**")
                st.markdown(
                    """
                    - If **rⱼᵢ > 0**, New Facility *j* is **to the right** of EF *i*.
                    - If **sⱼᵢ > 0**, New Facility *j* is **to the left** of EF *i*.
                    - If **pⱼₖ > 0**, New Facility *j* is **to the right** of NF *k*.
                    - If **qⱼₖ > 0**, New Facility *j* is **to the left** of NF *k*.
                    """
                )

            with col_interp_y:
                st.markdown("**Y-Direction (Vertical Alignment):**")
                st.markdown(
                    """
                    - If **uⱼᵢ > 0**, New Facility *j* lies **above** EF *i*.
                    - If **dⱼᵢ > 0**, New Facility *j* lies **below** EF *i*.
                    - If **aⱼₖ > 0**, New Facility *j* lies **above** NF *k*.
                    - If **bⱼₖ > 0**, New Facility *j* lies **below** NF *k*.
                    """
                )

            st.markdown(
                """
                **Note:** Active constraints occur when these variables are zero, indicating that a new facility 
                **coincides** with another facility in that dimension. These alignments determine the optimal 
                geometric mesh/grid of the solution.
                """
            )
        
            # --------------------------------------------------
            # Plot LP
            # --------------------------------------------------
            st.markdown("---")
            plot_lp = st.checkbox("Show Optimal Layout Graph", value = True)
            
            if plot_lp:
                # Generate Plot
                fig = slv.plot_mfl_solution(
                    existing_coords, 
                    lp_result['X_opt'], 
                    w_matrix, 
                    v_matrix
                )
                
                # Two columns: Left for text, Right for plot
                col_text, col_blank, col_plot = st.columns([1.2, 0.2,1.2]) 
                
                with col_text:
                    st.markdown("### Visual Analysis")
                    st.markdown(
                        """
                        The optimal layout in a rectilinear Multi-Facility problem is governed by two fundamental properties:

                        1. **Coincidence Property:** An optimum $x$ (or $y$) coordinate for each new facility will coincide with the coordinate of at least one existing facility.
                        
                        2. **Median Property:** When located optimally, each new facility is positioned at the **median location** with respect to all other facilities—both new and existing—with which it interacts.

                        **Interpretation of the "Mesh":**
                        Because of these properties, the optimal solution is mathematically guaranteed to lie on a discrete grid or "mesh" formed by the intersection of the coordinate lines of all existing facilities. 
                        
                        As seen in the plot, the stars (New Facilities) gravitate toward these intersection points to satisfy the median condition, effectively "locking" onto the geometric mesh of the existing facilities ($P_i$).
                        """
                    )
                    
                with col_plot:
                    st.pyplot(fig)
            # --------------------------------------------------
            # 6. Limitations of the LP Formulation
            # --------------------------------------------------
            st.markdown("### Limitations of the LP Formulation")
    
            st.markdown(
                """
                While Linear Programming provides a mathematically robust and guaranteed global 
                optimum, it presents several challenges in practice and pedagogy:
                """
            )

            col_lim_lp1, col_lim_lp2 = st.columns(2)

            with col_lim_lp1:
                st.markdown(
                    """
                    **Structural and Computational Complexity:**
                    * **Variable Inflation:** Linearizing absolute values requires $4n(n-1)/2 + 4nm$ auxiliary variables. For large $n$ and $m$, the model size grows quadratically.
                    * **Metric Specificity:** This formulation is strictly limited to **Rectilinear ($L_1$) distance**. It cannot solve Euclidean ($L_2$) problems without specialized non-linear solvers.
                    """
                )

            with col_lim_lp2:
                st.markdown(
                    """
                    **Educational and Practical Trade-offs:**
                    * **Geometric Intuition:** The use of auxiliary variables ($r, s, p, q$) can obscure the physical "pull" of the facilities compared to the Median Method.
                    * **Multiple Solutions:** If a "flat" optimal region exists, the LP solver typically identifies only one basic feasible solution (a corner point) unless further analysis is performed.
                    """
                )
    # =============================================================================
    # TAB 3: COORDINATE DESCENT
    # =============================================================================
    with tab3:
        st.subheader("Coordinate Descent Algorithm")
        
        # --------------------------------------------------
        # 1. Theoretical Background 
        # --------------------------------------------------       
        st.markdown(
            """
            The Coordinate Descent Method treats the Multi-Facility Location Problem as a sequence of 
            Single-Facility problems. The approach relies on the following fundamental properties 
            of the Rectilinear distance metric:
            """
        )
        
        col_props, col_math = st.columns([1,0.65])
        
        with col_props:
            st.markdown(
                """
                **1. Coincidence Property:**
                An optimum $x$ (or $y$) coordinate for each new facility coincides with the 
                $x$ (or $y$) coordinate of some **existing facility**.
        
                **2. Bound Property:**
                The optimal coordinate for any new facility lies within the range (min to max) 
                of the coordinates obtained by solving the problem with **zero interaction** between new facilities ($v_{jk}=0$).

                **3. Median Property:**
                When located optimally, each new facility resides at the **median location** with respect to all other facilities (both Existing and New) with which it interacts.

                **4. Distinctness Condition (Optimality):**
                If the algorithm converges such that **no two new facilities share the same location** for either $x$ or $y$ coordinates ($x_1$ ≠ $x_2$ ≠ $x_3$…. & $y_1$ ≠ $y_2$ ≠ $y_3$…), the solution is guaranteed to be the global optimum.
        
                ---
                **Algorithm Procedure:**
                1. **Initialize:** Set $v_{jk}=0$ and solve $n$ independent single-facility problems.
                2. **Iterate:** For each New Facility $j$, **fix** the locations of all other New Facilities 
                   and solve for $j$'s optimal location using the Median Method.
                3. **Repeat:** Continue until no facility changes its location ($x^r = x^{r-1}$).
                """
            )
            
        with col_math:
            
            st.markdown("**Minimize independent functions:**")
    
            st.latex(
                r"""
                \min f_1(x_1, \dots, x_n) = 
                \sum_{1 \le j < k \le n} v_{jk} |x_j - x_k| 
                + \sum_{j=1}^{n}\sum_{i=1}^{m} w_{ji} |x_j - a_i|
                """
            )
    
            st.latex(
                r"""
                \min f_2(y_1, \dots, y_n) = 
                \sum_{1 \le j < k \le n} v_{jk} |y_j - y_k| 
                + \sum_{j=1}^{n}\sum_{i=1}^{m} w_{ji} |y_j - b_i|
                """
            )
            st.caption("Each step reduces to finding the weighted median.")

        st.divider()

        # --------------------------------------------------
        # 2. Execution 
        # --------------------------------------------------
        # Parameters
        MAX_ITER = 50
        TOLERANCE = 1e-5
        
        # Prepare data
        ef_coords = data["ef"].values.tolist()
        w_matrix = data["w"].values.tolist()
        v_matrix = data["v"].values.tolist()
        
        # Run Solver
        result_cd = slv.solve_minisum_mfl(
            ef_coords, 
            w_matrix, 
            v_matrix, 
            max_iter=MAX_ITER, 
            tol=TOLERANCE
        )
        
        # Calculate cost history
        cost_history = []
        for step_coords in result_cd["history"]:
            c = slv.minisum_cost(step_coords, ef_coords, w_matrix, v_matrix)
            cost_history.append(c)
        
        # --------------------------------------------------
        # 3. Detailed Step-by-Step Instructions
        # --------------------------------------------------
        st.markdown("### Detailed Computational Steps")
        st.markdown("The following steps detail the mathematical procedure for each iteration.")

        # --- Helper to format the objective string (Sorted by Coordinate Value) ---
        def format_objective_latex(nf_idx, axis_val, axis_name, current_nfs, efs, w_mat, v_mat):
            # We collect all terms as tuples: (coordinate_value, latex_string)
            # This allows us to sort by the coordinate value before displaying.
            terms_list = []
        
            # 1. EF terms: w_ji * |x - a_i|
            for i, ef in enumerate(efs):
                weight = w_mat[nf_idx][i]
                coord = ef[axis_val]
                if weight > 0:
                    # Store (coord, string)
                    terms_list.append((coord, f"{weight:g}|{axis_name}_{{{nf_idx+1}}} - {coord:g}|"))
        
            # 2. NF terms: v_jk * |x - x_k|
            # Note: current_nfs contains the FIXED locations of other NFs
            for k, nf in enumerate(current_nfs):
                if k == nf_idx: continue
            
                # Access symmetric v matrix carefully
                weight = v_mat[nf_idx][k] if k > nf_idx else v_mat[k][nf_idx]
                coord = nf[axis_val]
            
                if weight > 0:
                    # Store (coord, string)
                    terms_list.append((coord, f"{weight:g}|{axis_name}_{{{nf_idx+1}}} - {coord:g}|"))
        
            # 3. SORT the list based on the coordinate value (the first item in tuple)
            # This ensures the equation reads from left-to-right on the number line.
            terms_list.sort(key=lambda t: t[0])
        
            # 4. Extract just the strings
            final_terms = [t[1] for t in terms_list]
        
            # 5. Join and format
            if not final_terms: return "0"
            full_eq = " + ".join(final_terms)
        
            # Handle line breaking for long equations
            if len(full_eq) > 60:
                mid = len(final_terms) // 2
                part1 = " + ".join(final_terms[:mid])
                part2 = " + ".join(final_terms[mid:])
                return rf"\begin{{aligned}} f({axis_name}_{{{nf_idx+1}}}) &= {part1} \\ &+ {part2} \end{{aligned}}"
            else:
                return rf"f({axis_name}_{{{nf_idx+1}}}) = {full_eq}"

        # --- Simulation Loop for Display ---
        # We replay the logic to generate the text
        
        # ITERATION 0: Initialization
        with st.expander("Iteration 0: Initialization ($v_{jk}=0$)", expanded=False):
            st.markdown(
                """
                **Step 1:** Set all NF-NF interactions ($v_{jk}$) to 0. 
                Solve $n$ independent single-facility problems using the Median Method.
                """
            )
            # Display Iteration 0 result (from history[0])
            init_coords = result_cd["history"][0]
            
            cols = st.columns(len(init_coords))
            for j, (x, y) in enumerate(init_coords):
                with cols[j]:
                    st.markdown(f"**NF{j+1} Initial:**")
                    st.latex(rf"x_{{{j+1}}} = {x:g}, \quad y_{{{j+1}}} = {y:g}")
        
        # SUBSEQUENT ITERATIONS
        # Note: result_cd['history'] stores the state at the END of each iteration.
        # We need to reconstruct the intermediate "moves" for the explanation.
        
        current_state = list(result_cd["history"][0]) # Start with iter 0
        
        # Loop through history (excluding the 0th init state)
        for iter_idx, next_state_snapshot in enumerate(result_cd["history"][1:], 1):
            
            with st.expander(f"Iteration {iter_idx}", expanded=False):
                st.caption("We update each facility sequentially, using the most recent locations of the others.")
                
                # We need to simulate the sequential update of NF1, then NF2...
                # inside this iteration to show the "Fixing" logic.
                
                temp_state_for_display = [list(c) for c in current_state]
                
                for j in range(len(temp_state_for_display)):
                    # Columns for X and Y analysis
                    col_x, col_y = st.columns(2)
                    
                    # --- X Coordinate Logic ---
                    with col_x:
                        st.markdown(f"**Update $x_{{{j+1}}}$:**")
                        
                        # Description of fixed facilities
                        fixed_desc = []
                        for k, (fx, fy) in enumerate(temp_state_for_display):
                            if k != j:
                                fixed_desc.append(f"NF{k+1} at $x={fx:g}$")
                        
                        st.markdown(f"Fix {', '.join(fixed_desc)}.")
                        
                        # Generate Equation
                        latex_eq = format_objective_latex(
                            j, 0, "x", temp_state_for_display, ef_coords, w_matrix, v_matrix
                        )
                        st.latex(latex_eq)
                        
                        # Show result (from the actual history snapshot for accuracy)
                        new_x = next_state_snapshot[j][0]
                        st.markdown(f"Apply Median Method $\\rightarrow$ **$x_{{{j+1}}} = {new_x:g}$**")

                    # --- Y Coordinate Logic ---
                    with col_y:
                        st.markdown(f"**Update $y_{{{j+1}}}$:**")
                        
                        # Description of fixed facilities
                        fixed_desc_y = []
                        for k, (fx, fy) in enumerate(temp_state_for_display):
                            if k != j:
                                fixed_desc_y.append(f"NF{k+1} at $y={fy:g}$")
                        
                        st.markdown(f"Fix {', '.join(fixed_desc_y)}.")
                        
                        # Generate Equation
                        latex_eq_y = format_objective_latex(
                            j, 1, "y", temp_state_for_display, ef_coords, w_matrix, v_matrix
                        )
                        st.latex(latex_eq_y)
                        
                        # Show result
                        new_y = next_state_snapshot[j][1]
                        st.markdown(f"Apply Median Method $\\rightarrow$ **$y_{{{j+1}}} = {new_y:g}$**")
                    
                    st.divider()
                    
                    # Update our temp state so the next NF sees this move
                    temp_state_for_display[j] = (new_x, new_y)
                
                # Update main tracker
                current_state = next_state_snapshot

        st.divider()
        # --------------------------------------------------
        # 4. Results Summary
        # --------------------------------------------------
        st.markdown("### Optimization Results")
        
        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Final Total Cost", f"{result_cd['obj']:.4f}")
        m2.metric("Iterations Used", f"{len(result_cd['history']) - 1}")
        m3.metric("Convergence Status", 
                  "Converged" if len(result_cd["history"]) - 1 < MAX_ITER else "Max Iter Reached")
        
        # Optimal Locations Table
        st.markdown("**Optimal Coordinates $(x^*, y^*)$**")
        df_res = pd.DataFrame(
            result_cd["X_opt"], 
            columns=["x*", "y*"], 
            index=[f"NF{j+1}" for j in range(len(result_cd["X_opt"]))]
        )
        st.table(df_res.T)
        

        # Check for multiple solutions / warnings
        warnings = slv.check_multiple_solutions(
            ef_coords, w_matrix, v_matrix, result_cd["X_opt"]
        )
        
        if warnings:
            st.markdown("##### ⚠️ Optimality Analysis")
            for w in warnings:
                if w["type"] == "property4":
                    st.error(w["msg"])
                    st.caption(
                        "When two new facilities merge, the algorithm cannot distinguish "
                        "between them to separate them further. This often indicates a local optimum."
                    )
                elif w["type"] == "flat":
                    st.warning(w["msg"])
                    # Visualization of the range
                    st.caption(
                        f"The objective function is 'flat' (slope = 0) between "
                        f"${w['range']}$. Any value in this interval is equally optimal "
                        "given the fixed positions of the other facilities."
                    )
        else:
            st.success("✅ **Unique Solution:** The algorithm converged to a strict median (Unique Global Optimum).")
            
        st.divider()
       
        # --------------------------------------------------
        # 5. Visualization
        # --------------------------------------------------
        st.markdown("### Convergence Analysis")

        col_plot1,col_blank ,col_plot2 = st.columns([1,0.2,1])

        with col_plot1:
            st.markdown("**Cost Reduction**")
            # Visualization of the objective function value over iterations 
            fig_cost = slv.plot_cost_history(cost_history)
            st.pyplot(fig_cost)
            st.caption("Total cost decreases monotonically at each iteration until convergence is reached.")

        with col_plot2:
            st.markdown("**Movement Trajectory and Interactions**")
            # This plot now shows the movement path alongside final weighted interactions 
            # We pass ef_coords, w_matrix, and v_matrix to display the weights on the graph
            fig_traj = slv.plot_trajectory(
                ef_coords, 
                result_cd["history"], 
                w_matrix, 
                v_matrix
            )
            st.pyplot(fig_traj)
            st.caption(
                "Dashed lines indicate the iteration path. "
                "Solid and dotted lines show final weighted interactions ($v_{jk}$ and $w_{ji}$)."
            )
        # --------------------------------------------------
        # 6. Limitations of the Approach
        # --------------------------------------------------
        st.markdown("### Limitations of Coordinate Descent")
        
        st.markdown(
            """
            While the Coordinate Descent method is simple to implement and provides a clear 
            step-by-step visualization of the median property, it has specific theoretical 
            limitations that need to be considered:
            """
        )

        col_lim1, col_lim2 = st.columns(2)

        with col_lim1:
            st.markdown(
                """
                **Global Optimality Constraints:**
                * **Property 4 Requirement:** The method is guaranteed to give a global optimal solution 
                  only when all new facilities have different values for both x and y coordinates.
                * **Local Optima Traps:** If some x or y coordinates are equal, the algorithm may satisfy 
                  median conditions at a non-optimal location.
                """
            )

        with col_lim2:
            st.markdown(
                """
                **Computational and Robustness Issues:**
                * **Flat Regions:** In cases with perfectly balanced weights, the approach may stop at 
                  any point within the optimal range rather than identifying the full set of solutions.
                * **LP Alternative:** For problems where facilities coincide or where absolute precision 
                  is required despite "flat" objective functions, a Linear Programming (LP) approach is 
                  recommended.
                """
            )
        
    # =============================================================================
    # TAB 4: EUCLIDEAN MODELS
    # =============================================================================
    with tab4:
        st.subheader("Euclidean Distance Models")
        
        st.markdown(
            """
            In this section, we transition from the "grid-like" movement of the Rectilinear metric 
            to **straight-line**  distances. 
            
            Unlike the Rectilinear model where $x$ and $y$ can be solved independently, 
            Euclidean models usually require solving for coordinates simultaneously in continuous space.
            """
        )
        
        st.divider()

        # --------------------------------------------------
        # 1. Squared Euclidean (Analytical Solution)
        # --------------------------------------------------
        st.markdown("### 1. Squared Euclidean Distance Model ($L_2^2$)")
                    
        st.markdown(
            """
            This model minimizes the sum of weighted **squared** Euclidean distances. 
            Mathematically, it behaves like a **Spring-Mass System**:
            * Every weight ($w_{ji}$ or $v_{jk}$) acts as a spring constant.
            * The optimal location is the equilibrium point where all spring forces cancel out.
                
            **Why use it?**
            * **Differentiability:** The squared term eliminates square roots, making the function differentiable everywhere.
            * **Analytical Solution:** We can find the exact global optimum by solving a system of linear equations. No iteration is needed.
            """
        )

        st.markdown("**Mathematical Formulation:**")
        st.latex(
            r"""
            \begin{aligned}
            \min \; f\big((x_1,y_1),\ldots,(x_n,y_n)\big)
            &=
            \sum_{1 \le j < k \le n}
            v_{jk}\Big[(x_j-x_k)^2 + (y_j-y_k)^2\Big] 
            & +
            \sum_{j=1}^{n}\sum_{i=1}^{m}
            w_{ji}\Big[(x_j-a_i)^2 + (y_j-b_i)^2\Big]
            \end{aligned}
            """
            )
        
        st.markdown("**Solution: Step-by-Step Derivation**")
        st.caption("Since $x$ and $y$ terms are separable in the squared function, we solve two independent linear systems.")

        col_sol_x, col_sol_y = st.columns(2)

        with col_sol_x:
            st.markdown("**1. X-Coordinates**")
            st.markdown("Set partial derivative w.r.t $x_j$ to 0:")
            st.latex(r"\frac{\partial f}{\partial x_j} = 2\sum_i w_{ji}(x_j - a_i) + 2\sum_k v_{jk}(x_j - x_k) = 0")
            
            st.markdown("Rearrange to isolate $x_j$:")
            st.latex(r"x_j \underbrace{\left( \sum_i w_{ji} + \sum_k v_{jk} \right)}_{\text{Total Weight } A_{jj}} - \sum_{k \neq j} v_{jk} x_k = \sum_i w_{ji} a_i")
            
            st.markdown("Matrix Form ($\mathbf{A}\mathbf{x} = \mathbf{B}_x$):")
            st.latex(r"\begin{bmatrix} A_{11} & \dots & -v_{1n} \\ \vdots & \ddots & \vdots \\ -v_{n1} & \dots & A_{nn} \end{bmatrix} \begin{bmatrix} x_1 \\ \vdots \\ x_n \end{bmatrix} = \begin{bmatrix} \sum w_{1i}a_i \\ \vdots \\ \sum w_{ni}a_i \end{bmatrix}")
            
            

        with col_sol_y:
            st.markdown("**2. Y-Coordinates**")
            st.markdown("Set partial derivative w.r.t $y_j$ to 0:")
            st.latex(r"\frac{\partial f}{\partial y_j} = 2\sum_i w_{ji}(y_j - b_i) + 2\sum_k v_{jk}(y_j - y_k) = 0")
            
            st.markdown("Rearrange to isolate $y_j$:")
            st.latex(r"y_j \underbrace{\left( \sum_i w_{ji} + \sum_k v_{jk} \right)}_{\text{Total Weight } A_{jj}} - \sum_{k \neq j} v_{jk} y_k = \sum_i w_{ji} b_i")
            
            st.markdown("Matrix Form ($\mathbf{A}\mathbf{y} = \mathbf{B}_y$):")
            st.latex(r"\begin{bmatrix} A_{11} & \dots & -v_{1n} \\ \vdots & \ddots & \vdots \\ -v_{n1} & \dots & A_{nn} \end{bmatrix} \begin{bmatrix} y_1 \\ \vdots \\ y_n \end{bmatrix} = \begin{bmatrix} \sum w_{1i}b_i \\ \vdots \\ \sum w_{ni}b_i \end{bmatrix}")
                        
        st.latex(
            r"""
            \begin{aligned}
            \text{Diagonal Term:}\quad
            & A_{jj} = \sum_{i=1}^m w_{ji} + \sum_{k=1}^n v_{jk} \\[6pt]
            \text{Off\text{-}Diagonal Term:}\quad
            & A_{jk} = -v_{jk}
            \end{aligned}
            """
            )

        st.info(
        "The coefficient matrix $\mathbf{A}$ is the same for both $x$ and $y$. "
        "It is strictly diagonally dominant (diagonal element > sum of off-diagonals), ensuring a unique solution exists."
    )

        # --------------------------------------------------
        # Dynamic Matrix Construction for Input Data
        # --------------------------------------------------
        st.markdown("##### Solution for Current Input Data")
        
        # 1. Prepare Data
        ef_coords = data["ef"].values
        w_matrix = data["w"].values
        v_matrix = data["v"].values
        n_new = len(v_matrix)

        # 2. Build and Solve System
        # Build Matrix A (Coefficient Matrix)
        A_num = np.zeros((n_new, n_new))
        B_x_num = np.zeros(n_new)
        B_y_num = np.zeros(n_new)

        for j in range(n_new):
            # Diagonal A_jj
            sum_w = np.sum(w_matrix[j, :])
            sum_v = np.sum(v_matrix[j, :]) 
            A_num[j, j] = sum_w + sum_v
            
            # RHS Vectors
            B_x_num[j] = np.sum(w_matrix[j, :] * ef_coords[:, 0])
            B_y_num[j] = np.sum(w_matrix[j, :] * ef_coords[:, 1])

            for k in range(n_new):
                if j != k:
                    val = v_matrix[j, k] if k > j else v_matrix[k, j]
                    A_num[j, k] = -val

        # Solve for display
        try:
            x_sol_vec = np.linalg.solve(A_num, B_x_num)
            y_sol_vec = np.linalg.solve(A_num, B_y_num)
        except np.linalg.LinAlgError:
            x_sol_vec = np.zeros(n_new)
            y_sol_vec = np.zeros(n_new)

        # 3. Create LaTeX Strings
        # Matrix A
        A_rows = []
        for row in A_num:
            A_rows.append(" & ".join([f"{val:g}" for val in row]))
        A_str = r"\begin{bmatrix} " + r" \\ ".join(A_rows) + r" \end{bmatrix}"

        # Vector Bx
        Bx_str = r"\begin{bmatrix} " + r" \\ ".join([f"{val:g}" for val in B_x_num]) + r" \end{bmatrix}"
        # Vector By
        By_str = r"\begin{bmatrix} " + r" \\ ".join([f"{val:g}" for val in B_y_num]) + r" \end{bmatrix}"
        
        # Result Vectors
        x_res_str = r"\begin{bmatrix} " + r" \\ ".join([f"{val:.2f}" for val in x_sol_vec]) + r" \end{bmatrix}"
        y_res_str = r"\begin{bmatrix} " + r" \\ ".join([f"{val:.2f}" for val in y_sol_vec]) + r" \end{bmatrix}"
        
        # Variable Vectors
        x_vars = r"\begin{bmatrix} " + r" \\ ".join([f"x_{j+1}" for j in range(n_new)]) + r" \end{bmatrix}"
        y_vars = r"\begin{bmatrix} " + r" \\ ".join([f"y_{j+1}" for j in range(n_new)]) + r" \end{bmatrix}"

        # 4. Display
        col_mat_x, col_mat_y = st.columns(2)

        with col_mat_x:
            st.markdown("**X-System**")
            st.latex(f"{A_str} {x_vars} = {Bx_str}")
            st.markdown("$\downarrow$ *Solving for X*")
            st.latex(rf"{x_vars} = {x_res_str}")

        with col_mat_y:
            st.markdown("**Y-System**")
            st.latex(f"{A_str} {y_vars} = {By_str}")
            st.markdown("$\downarrow$ *Solving for Y*")
            st.latex(rf"{y_vars} = {y_res_str}")

    # --------------------------------------------------
    # --- Execution (Squared) ---
    # --------------------------------------------------
        
        ef_coords = data["ef"].values.tolist()
        w_matrix = data["w"].values.tolist()
        v_matrix = data["v"].values.tolist()
        
        res_sq = slv.solve_squared_euclidean(ef_coords, w_matrix, v_matrix)
        
        if "error" in res_sq:
            st.error(res_sq["error"])
        else:
            st.markdown("##### Optimization Results ($L_2^2$)")
            
            st.metric("Total Minimum Cost", f"{res_sq['obj']:.4f}")
          
            # Results Table
            st.markdown("**Optimal Coordinates $(x^*, y^*)$**")
            df_sq = pd.DataFrame(res_sq["X_opt"], columns=["x", "y"], index=[f"NF{j+1}" for j in range(n)])
            st.table(df_sq.T)

        st.divider()

        # --------------------------------------------------
        # 2. Standard Euclidean (Iterative Solution)
        # --------------------------------------------------
        st.markdown("### 2. Standard Euclidean Distance Model ($L_2$)")
    
        st.markdown(
            """
            This model minimizes the **true straight-line distance**. It represents the actual physical 
            distance for transport (e.g., air travel, conveyors).
        
            **The Mathematical Challenge:**
            * The objective function contains square roots: $D_{ji} = \sqrt{(x_j-a_i)^2 + (y_j-b_i)^2}$.
            * **The Singularity:** If a new facility $X_j$ sits exactly on top of an existing facility ($X_j = P_i$), 
                the distance is zero. The derivative involves dividing by distance ($1/0$), which is undefined.
        
            **Solution Method:**
            We use the **Weiszfeld Iterative Procedure**. It approximates the solution by repeatedly 
            adjusting weights inversely proportional to the current distance estimates.
            """
        )

        st.markdown("**Mathematical Formulation:**")
        st.latex(
            r"""
            \begin{aligned}
            \min \; f\big((x_1,y_1),\ldots,(x_n,y_n)\big)
            &=
            \sum_{1 \le j < k \le n}
            v_{jk}\,
            \sqrt{(x_j-x_k)^2 + (y_j-y_k)^2} 
            +
            \sum_{j=1}^{n}\sum_{i=1}^{m}
            w_{ji}\,
            \sqrt{(x_j-a_i)^2 + (y_j-b_i)^2}
            \end{aligned}
            """
            )

        st.markdown("**Weiszfeld Iteration Method**")
        st.caption("Since we cannot solve for $x$ and $y$ directly (no closed-form solution), we update them iteratively based on the previous position $(x^{(k)}, y^{(k)})$.")

        col_std_x, col_std_y = st.columns(2)

        with col_std_x:
            st.markdown("**X-Update Formula**")
            st.latex(
                r"""
                x_j^{(k+1)} = \frac{\sum_i \frac{w_{ji} a_i}{D_{ji}^{(k)}} + \sum_k \frac{v_{jk} x_k^{(k)}}{D_{jk}^{(k)}}}{\sum_i \frac{w_{ji}}{D_{ji}^{(k)}} + \sum_k \frac{v_{jk}}{D_{jk}^{(k)}}}
                """
            )
            st.caption("New $x$ is the weighted average of target $x$-coordinates, weighted by inverse distance.")

        with col_std_y:
            st.markdown("**Update Formula**")
            st.latex(
                r"""
                y_j^{(k+1)} = \frac{\sum_i \frac{w_{ji} b_i}{D_{ji}^{(k)}} + \sum_k \frac{v_{jk} y_k^{(k)}}{D_{jk}^{(k)}}}{\sum_i \frac{w_{ji}}{D_{ji}^{(k)}} + \sum_k \frac{v_{jk}}{D_{jk}^{(k)}}}
                """
            )
            st.caption("New $y$ is the weighted average of target $y$-coordinates, weighted by inverse distance.")

        # --- Initialization Display ---
        st.markdown("**Initialization ($k=0$):**")
        st.markdown(
            """
            The iterative algorithm requires a starting point $(x^{(0)}, y^{(0)})$. 
            We use the **Squared Euclidean solution** (calculated above) as the initial guess, as it provides a robust starting position typically close to the true optimum.
            """
        )
    
        # Retrieve starting values from the Squared Euclidean result
        start_vals = []
        if "X_opt" in res_sq:
            for j, (sx, sy) in enumerate(res_sq["X_opt"]):
                start_vals.append(rf"NF{j+1}^{(0)} = ({sx:.2f}, {sy:.2f})")
        
            # Display nicely formatted math string
            st.latex(r"\quad ".join(start_vals))
        else:
            st.info("Starting Values: Center of Gravity of Existing Facilities")

        st.info(
            "**Note on Singularities:** A small constant $\epsilon$ is usually added to the distance $D$ in the denominator "
            "to prevent division by zero if a facility lands exactly on another location."
        )
    
    

        # --------------------------------------------------
        # Execution (Weiszfeld) & Visualization
        # --------------------------------------------------
        res_std = slv.solve_euclidean_weiszfeld(ef_coords, w_matrix, v_matrix)
    
        st.markdown("##### Optimization Results ($L_2$)")
    
        # 1. Key Metrics Row
        c1, c2, c3 = st.columns(3)
        c1.metric("Minimum Total Cost", f"{res_std['obj']:.4f}")
        c2.metric("Iterations Required", f"{res_std['iterations']}")
        c3.metric("Convergence Status", 
                  "Converged" if res_std['iterations'] < 50 else "Max Iteration Reached")
        
        # 2. Detailed Results & Plot
        col_res, col_plot = st.columns([0.8, 0.8])
    
        with col_res:
            st.markdown("**Optimal Coordinates $(x^*, y^*)$**")
        
            # Format Table: NF as Rows, x/y as Columns
            df_std = pd.DataFrame(res_std["X_opt"], columns=["x", "y"], index=[f"NF{j+1}" for j in range(n)])
            st.dataframe(df_std.style.format("{:.4f}"), use_container_width=True)
        
            # Collapsible Iteration History
            with st.expander("View Iteration Steps", expanded=False):
                history_data = []
                for k, snapshot in enumerate(res_std["history"]):
                    row = {"Iter": k}
                    for j, (x, y) in enumerate(snapshot):
                        # Format tuple as string for compact display
                        row[f"NF{j+1}"] = f"({x:.2f}, {y:.2f})"
                    history_data.append(row)
            
                # Display scrollable dataframe
                st.dataframe(
                    pd.DataFrame(history_data).set_index("Iter"), 
                    height=250, 
                    use_container_width=True
                )

        with col_plot:
            st.markdown("**Convergence Trajectory**")
        
            # Create a compact figure for the layout
            fig_traj = slv.plot_trajectory(ef_coords, res_std["history"], w_matrix, v_matrix)
        
            # Adjust figure size for sidebar layout constraint
            fig_traj.set_size_inches(6, 6) 
            fig_traj.axes[0].set_title("Euclidean Optimization Path (Curved Descent)", fontsize=10)
        
            st.pyplot(fig_traj, use_container_width=True)
            st.caption("Notice how the path follows a smooth curve (Gradient Descent) compared to the 'staircase' steps of Coordinate Descent.")