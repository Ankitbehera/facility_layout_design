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
        ["Manual Input", "Upload CSV Files"]
    )

    # ==================================================
    # CSV INPUT MODE
    # ==================================================
    if input_mode == "Upload CSV Files":
        st.sidebar.markdown("### Upload CSV Files")

        ef_file = st.sidebar.file_uploader(
            "Existing Facilities CSV (a,b)",
            type=["csv"],
            key="ef_csv"
        )

        w_file = st.sidebar.file_uploader(
            "EF–NF Weight Matrix CSV (w_ji)",
            type=["csv"],
            key="w_csv"
        )

        if ef_file is None or w_file is None:
            st.sidebar.info("Please upload both CSV files.")
            return None

        # -----------------------------
        # Read Existing Facilities
        # -----------------------------
        try:
            ef_df = pd.read_csv(ef_file)

            if list(ef_df.columns) != ["a", "b"]:
                st.sidebar.error("EF CSV must have columns: a, b")
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
        # Read EF–NF Weights
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
                    "Number of EF columns in weight CSV must match EF CSV."
                )
                return None

            w_df.index = [f"NF{i+1}" for i in range(n)]
            w_df.columns = [f"EF{i+1}" for i in range(m)]

            st.session_state.w_df = w_df

        except Exception as e:
            st.sidebar.error(f"Invalid weight CSV: {e}")
            return None

        # -----------------------------
        # NF–NF Interaction (Manual only)
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
                        v_input.iloc[j, k] = "—"
            st.session_state.v_input = v_input

        v_edit = st.sidebar.data_editor(
            st.session_state.v_input,
            num_rows="fixed",
            use_container_width=True
        )
        st.session_state.v_input = v_edit

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
        m = st.number_input("Number of existing facilities $m$", 1, step=1, value=3)
    with col_n:
        n = st.number_input("Number of new facilities $n$", 1, step=1, value=2)

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
        num_rows="fixed",
        use_container_width=True
    )
    st.session_state.ef_df = ef_df

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
        num_rows="fixed",
        use_container_width=True
    )
    st.session_state.w_df = w_df

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
                    v_input.iloc[j, k] = "—"
        st.session_state.v_input = v_input

    v_edit = st.sidebar.data_editor(
        st.session_state.v_input,
        num_rows="fixed",
        use_container_width=True
    )
    st.session_state.v_input = v_edit

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

# ------------------------------------------------------------------------------
# Page
# ------------------------------------------------------------------------------

# =============================================================================
# MAIN PAGE
# =============================================================================
def show_minisum_mfl(data):
    st.title("Minisum Multi-Facility Location Problem")

    # Plot styling (copied from SFL)
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    # --------------------------------------------------
    # Tabs (As YOU defined)
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

        show_data = st.checkbox("Show Input Data Summary")

        if show_data:
            st.markdown("### Input Data Summary")

            # --------------------------------------------------
            # Existing Facilities
            # --------------------------------------------------
            with st.expander("Existing Facility Locations $(a_i, b_i)$", expanded=True):
                st.dataframe(
                    data["ef"],
                    hide_index=False,
                    use_container_width=True
                )

            # --------------------------------------------------
            # EF–NF Interaction Weights
            # --------------------------------------------------
            with st.expander("EF–NF Interaction Weights $w_{ji}$",expanded=True):
                st.dataframe(
                    data["w"],
                    hide_index=False,
                    use_container_width=True
                )

            # --------------------------------------------------
            # NF–NF Interaction Weights
            # --------------------------------------------------
            with st.expander("NF–NF Interaction Weights $v_{jk}$",expanded=True):
                st.dataframe(
                    data["v"],
                    hide_index=False,
                    use_container_width=True
                )

                st.caption(
                    r"""
                    $v_{jk} = v_{kj}$,  $v_{jj} = 0$  
                    Only interactions between distinct new facilities are considered.
                    """
                )

        st.markdown("### Solution Methods Implemented")
        st.markdown(
            """
            - **Rectilinear (L1)** — Equivalent Linear Programming  
            - **Rectilinear (L1)** — Coordinate Descent  
            - **Euclidean (L2² and L2)** — Iterative methods  
            """
        )


    # =============================================================================
    # TAB 2: RECTILINEAR (LP)
    # =============================================================================
    with tab2:
        st.subheader("Rectilinear Minisum — Linear Programming Formulation")

        st.markdown("### Problem Decomposition")
        st.latex(
            r"""
            f(\mathbf{x},\mathbf{y}) = f_x(x_1,\dots,x_n) + f_y(y_1,\dots,y_n)
            """
        )

        st.markdown("### Equivalent LP Formulation")
        st.markdown(
            """
            Absolute values are eliminated using auxiliary variables
            for EF–NF and NF–NF interactions.
            """
        )

        st.info("Expanded LP, constraints, and PuLP solver will be added here.")

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