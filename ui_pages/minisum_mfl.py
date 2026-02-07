import streamlit as st
import pandas as pd
import pulp
import matplotlib.pyplot as plt
import solver.minisum_mfl as slv
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
    # Problem size
    # --------------------------------------------------
    st.sidebar.markdown("#### Manual Input")

    col_m, col_n = st.sidebar.columns(2)

    with col_m:
        m = st.number_input(
            "Number of existing facilities $m$",
            min_value=1,
            step=1,
            value=4
        )

    with col_n:
        n = st.number_input(
            "Number of new facilities $n$",
            min_value=1,
            step=1,
            value=2
        )

    st.sidebar.caption(
        "Changing $m$ or $n$ will reset the corresponding input tables."
    )

    # --------------------------------------------------
    # Existing Facility Locations (a_i, b_i)
    # --------------------------------------------------
    st.sidebar.markdown("### Existing Facility Locations $(a_i,b_i)$")

    if "ef_df" not in st.session_state or len(st.session_state.ef_df) != m:

        # ---- Textbook example ----
        if m == 4:
            st.session_state.ef_df = pd.DataFrame(
                {
                    "a_i (x-coordinate)": [0, 15, 25, 40],
                    "b_i (y-coordinate)": [10, 0, 30, 15],
                },
                index=[f"EF{i+1}" for i in range(4)]
            )

        # ---- Generic default ----
        else:
            st.session_state.ef_df = pd.DataFrame(
                {
                    "a_i (x-coordinate)": [10.0 * (i + 1) for i in range(m)],
                    "b_i (y-coordinate)": [5.0 * (i + 1) for i in range(m)],
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
    # EF–NF Interaction Weights w_ji
    # --------------------------------------------------
    st.sidebar.markdown("### EF–NF Interaction Weights $w_{ji}$")

    if "w_df" not in st.session_state or st.session_state.w_df.shape != (n, m):

        # ---- Textbook example ----
        if m == 4 and n == 2:
            st.session_state.w_df = pd.DataFrame(
                [
                    [10, 6, 0, 0],   # NF1
                    [0, 2, 16, 8],   # NF2
                ],
                index=["NF1", "NF2"],
                columns=["EF1", "EF2", "EF3", "EF4"]
            )

        # ---- Generic default ----
        else:
            st.session_state.w_df = pd.DataFrame(
                1.0,
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
    # NF–NF Interaction Weights v_jk (upper triangular)
    # --------------------------------------------------
    st.sidebar.markdown("### NF–NF Interaction Weights $v_{jk}$")

    if "v_input" not in st.session_state or st.session_state.v_input.shape != (n, n):

        v_input = pd.DataFrame(
            "",
            index=[f"NF{j+1}" for j in range(n)],
            columns=[f"NF{k+1}" for k in range(n)]
        )

        # Lock diagonal and lower triangle
        for j in range(n):
            for k in range(n):
                if j >= k:
                    v_input.iloc[j, k] = "auto-fill"

        # ---- Textbook example ----
        if n == 2:
            v_input.iloc[0, 1] = "12"   # v12 = 12

        # ---- Generic default (n > 2) ----
        else:
            for j in range(n):
                for k in range(j + 1, n):
                    v_input.iloc[j, k] = str(2 * (j + k + 2))

        st.session_state.v_input = v_input


    v_edit = st.sidebar.data_editor(
        st.session_state.v_input,
        key="v_table",
        num_rows="fixed",
        use_container_width=True
    )

    # Build symmetric numeric matrix v_jk
    v_df = pd.DataFrame(
        0.0,
        index=v_edit.index,
        columns=v_edit.columns
    )

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
        **Notes on $v_{jk}$ interaction matrix:**

        • $v_{jk} = v_{kj}$  
        • $v_{jj} = 0$  
        • Cells marked "**auto-fill**" are handled automatically  
        • Specify values only for $j < k$
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

# --------------------------------------------------
# For Upload and Manual Input, Use when upload and manual needed
# --------------------------------------------------
def build_inputs_upload():
    # ==================================================
    # Sidebar: Input Data (Minisum MFL)
    # ==================================================
    st.sidebar.header("Input Data")

    # --------------------------------------------------
    # Input Mode
    # --------------------------------------------------
    input_mode = st.sidebar.radio(
        "Input Mode",
        ["Manual Input", "Upload CSV Files"],
        key="input_mode_mfl"
    )

    # ==================================================
    # CSV INPUT MODE
    # ==================================================
    if input_mode == "Upload CSV Files":
        st.sidebar.markdown("### Upload CSV Files")

        ef_file = st.sidebar.file_uploader(
            "Existing Facilities CSV (columns: a, b)",
            type=["csv"],
            key="ef_csv"
        )

        w_file = st.sidebar.file_uploader(
            "EF–NF Weight Matrix CSV ($w_{ji}$)",
            type=["csv"],
            key="w_csv"
        )

        if ef_file is None or w_file is None:
            st.sidebar.info("Please upload both CSV files.")
            return None

        # -----------------------------
        # Existing Facilities
        # -----------------------------
        try:
            ef_df = pd.read_csv(ef_file)

            if list(ef_df.columns) != ["a", "b"]:
                st.sidebar.error("EF CSV must have columns: a, b")
                return None

            if ef_df.isnull().any().any():
                st.sidebar.error("EF CSV contains empty cells.")
                return None

            ef_df = ef_df.astype(float)
            m = len(ef_df)

            ef_df.index = [f"EF{i+1}" for i in range(m)]
            ef_df.columns = ["a_i (x-coordinate)", "b_i (y-coordinate)"]

            st.session_state.ef_df = ef_df

        except Exception as e:
            st.sidebar.error(f"Invalid EF CSV: {e}")
            return None

        # -----------------------------
        # EF–NF Weights
        # -----------------------------
        try:
            w_df = pd.read_csv(w_file)

            if w_df.isnull().any().any():
                st.sidebar.error("Weight CSV contains empty cells.")
                return None

            w_df = w_df.astype(float)
            n = len(w_df)

            if w_df.shape[1] != m:
                st.sidebar.error(
                    "Number of columns in weight CSV must match EF CSV."
                )
                return None

            w_df.index = [f"NF{i+1}" for i in range(n)]
            w_df.columns = [f"EF{i+1}" for i in range(m)]

            st.session_state.w_df = w_df

        except Exception as e:
            st.sidebar.error(f"Invalid weight CSV: {e}")
            return None

        # -----------------------------
        # NF–NF Interaction (manual)
        # -----------------------------
        st.sidebar.markdown("### NF–NF Interaction Weights $v_{jk}$")

        if "v_input" not in st.session_state or st.session_state.v_input.shape != (n, n):
            v_input = pd.DataFrame(
                "",
                index=[f"NF{j+1}" for j in range(n)],
                columns=[f"NF{k+1}" for k in range(n)]
            )
            for j in range(n):
                for k in range(n):
                    if j >= k:
                        v_input.iloc[j, k] = "auto-fill"
            st.session_state.v_input = v_input

        v_edit = st.sidebar.data_editor(
            st.session_state.v_input,
            key="v_table_upload",
            num_rows="fixed",
            use_container_width=True
        )

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
            **Notes on $v_{jk}$ interaction matrix:**

            • $v_{jk} = v_{kj}$  
            • $v_{jj} = 0$  
            • Cells marked *auto-fill* are handled automatically  
            • Specify values only for $j < k$
            """
        )

        return {
            "m": m,
            "n": n,
            "ef": ef_df,
            "w": w_df,
            "v": v_df
        }

    # ==================================================
    # MANUAL INPUT MODE
    # ==================================================
    st.sidebar.markdown("#### Manual Input")

    col_m, col_n = st.sidebar.columns(2)

    with col_m:
        m = st.number_input(
            "Number of existing facilities $m$",
            min_value=1,
            step=1,
            value=3
        )

    with col_n:
        n = st.number_input(
            "Number of new facilities $n$",
            min_value=1,
            step=1,
            value=2
        )

    st.sidebar.caption(
        "Changing $m$ or $n$ will reset the corresponding input tables."
    )

    # -----------------------------
    # Existing Facilities
    # -----------------------------
    st.sidebar.markdown("### Existing Facility Locations $(a_i,b_i)$")

    if "ef_df" not in st.session_state or len(st.session_state.ef_df) != m:
        st.session_state.ef_df = pd.DataFrame(
            {
                "a_i (x-coordinate)": [10.0 * (i + 1) for i in range(m)],
                "b_i (y-coordinate)": [5.0 * (i + 1) for i in range(m)],
            },
            index=[f"EF{i+1}" for i in range(m)]
        )

    ef_df = st.sidebar.data_editor(
        st.session_state.ef_df,
        key="ef_table_manual",
        num_rows="fixed",
        use_container_width=True
    )

    if ef_df.isnull().any().any():
        st.sidebar.error("Existing facility table contains empty cells.")
        return None

    # -----------------------------
    # EF–NF Weights
    # -----------------------------
    st.sidebar.markdown("### EF–NF Interaction Weights $w_{ji}$")

    if "w_df" not in st.session_state or st.session_state.w_df.shape != (n, m):
        st.session_state.w_df = pd.DataFrame(
            1.0,
            index=[f"NF{j+1}" for j in range(n)],
            columns=[f"EF{i+1}" for i in range(m)]
        )

    w_df = st.sidebar.data_editor(
        st.session_state.w_df,
        key="w_table_manual",
        num_rows="fixed",
        use_container_width=True
    )

    if w_df.isnull().any().any():
        st.sidebar.error("EF–NF weight matrix contains empty cells.")
        return None

    # -----------------------------
    # NF–NF Interaction
    # -----------------------------
    st.sidebar.markdown("### NF–NF Interaction Weights $v_{jk}$")

    if "v_input" not in st.session_state or st.session_state.v_input.shape != (n, n):
        v_input = pd.DataFrame(
            "",
            index=[f"NF{j+1}" for j in range(n)],
            columns=[f"NF{k+1}" for k in range(n)]
        )
        for j in range(n):
            for k in range(n):
                if j >= k:
                    v_input.iloc[j, k] = "auto-fill"
        st.session_state.v_input = v_input

    v_edit = st.sidebar.data_editor(
        st.session_state.v_input,
        key="v_table_manual",
        num_rows="fixed",
        use_container_width=True
    )

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
        **Notes on $v_{jk}$ interaction matrix:**

        • $v_{jk} = v_{kj}$  
        • $v_{jj} = 0$  
        • Cells marked *auto-fill* are handled automatically  
        • Specify values only for $j < k$
        """
    )

    return {
        "m": m,
        "n": n,
        "ef": ef_df,
        "w": w_df,
        "v": v_df
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview",
        "Rectilinear (LP)",
        "Coordinate Descent",
        "Euclidean Models",
        "LP vs Coordinate Descent"
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
        # Constraints (Dynamic)
        # --------------------------------------------------
        st.markdown("### Constraints (Dynamic)")
        
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            st.markdown("**X-Coordinates**")
            # EF Constraints
            for j in range(n):
                for i in range(m):
                    st.latex(rf"x_{{{j+1}}} - r_{{{j+1}{i+1}}} + s_{{{j+1}{i+1}}} = {ef_df.iloc[i, 0]}")
            # NF Constraints
            for j in range(n):
                for k in range(j+1, n):
                    st.latex(rf"x_{{{j+1}}} - x_{{{k+1}}} + p_{{{j+1}{k+1}}} - q_{{{j+1}{k+1}}} = 0")
            
        with col_c2:
            st.markdown("**Y-Coordinates**")
            # EF Constraints
            for j in range(n):
                for i in range(m):
                    st.latex(rf"y_{{{j+1}}} - u_{{{j+1}{i+1}}} + d_{{{j+1}{i+1}}} = {ef_df.iloc[i, 1]}")
            # NF Constraints
            for j in range(n):
                for k in range(j+1, n):
                    st.latex(rf"y_{{{j+1}}} - y_{{{k+1}}} + a_{{{j+1}{k+1}}} - b_{{{j+1}{k+1}}} = 0")

        # --------------------------------------------------
        # Solve LP
        # --------------------------------------------------
        st.markdown("---")
        solve_lp = st.checkbox("Solve this LP using PuLP")

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
            plot_lp = st.checkbox("Show Optimal Layout Graph")
            
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

    # =============================================================================
    # TAB 3: COORDINATE DESCENT
    # =============================================================================
    with tab3:
        st.subheader("Coordinate Descent Method")

        st.markdown("### Method Explanation")
        st.markdown(
            """
            The Coordinate Descent method alternately optimizes the
            x- and y-coordinates of new facilities while keeping the
            other coordinates fixed.
            """
        )

        st.markdown("### Implementation")
        st.info("Step-by-step coordinate descent implementation coming next.")

        st.markdown("### Limitations")
        st.markdown(
            """
            - May fail when facilities coincide  
            - Requires good initialization  
            - Not guaranteed to converge for all instances  
            """
        )

    # =============================================================================
    # TAB 4: EUCLIDEAN MODELS
    # =============================================================================
    with tab4:
        st.subheader("Euclidean Distance Models")

        st.markdown("### Squared Euclidean (L2²)")
        st.latex(
            r"""
            x_j^* =
            \frac{\sum_i w_{ji} a_i + \sum_k v_{jk} x_k}
            {\sum_i w_{ji} + \sum_k v_{jk}}
            """
        )

        st.markdown("### Euclidean (L2)")
        st.markdown(
            """
            Solved using iterative fixed-point or gradient-based methods.
            """
        )

        st.markdown("### Gradient Equation")
        st.info("Gradient expressions and iterations will be added.")

    # =============================================================================
    # TAB 5: LP vs COORDINATE DESCENT
    # =============================================================================
    with tab5:
        st.subheader("LP vs Coordinate Descent")

        st.markdown(
            """
            | Aspect | Linear Programming | Coordinate Descent |
            |------|--------------------|-------------------|
            | Optimality | Global | Local / Conditional |
            | Robustness | High | Medium |
            | Computation | High | Low |
            | Degeneracy Handling | Yes | No |
            """
        )

        st.markdown(
            """
            **Key takeaway:**  
            LP provides a mathematically guaranteed solution, while
            Coordinate Descent offers computational efficiency and
            geometric insight.
            """
        )