"""
================================================================================
Minisum Single Location Problem Page
================================================================================
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import solver.minisum_sfl as slv
import io
import pulp

def build_inputs():
    # --------------------------------------------------
    # Sidebar: Data Input
    # --------------------------------------------------
    st.sidebar.header("Input Data")

    # ==================================================
    # CASE 1: MANUAL INPUT (DEFAULT)
    # ==================================================
    st.sidebar.subheader("Manual Input")

    m = st.sidebar.number_input(
        "Number of existing facilities",
        min_value=1,
        step=1,
        value=4
    )

    st.sidebar.markdown("### Facility Locations")

    default_data = [
        (0.0, 2.0, 3.0),
        (0.0, 4.0, 2.0),
        (2.0, 0.0, 3.0),
        (4.0, 0.0, 2.0),
    ]

    manual_data = []

    for i in range(m):
        col1, col2, col3 = st.sidebar.columns(3)

        if i < len(default_data):
            a0, b0, w0 = default_data[i]
        else:
            a0 = float(i + 1)
            b0 = float(i + 1)
            w0 = 1.0 / m

        a = col1.number_input(
            f"$a_{i+1}$",
            key=f"a{i}",
            value=a0,
            step=0.1,
            format="%.2f"
        )

        b = col2.number_input(
            f"$b_{i+1}$",
            key=f"b{i}",
            value=b0,
            step=0.1,
            format="%.2f"
        )

        w = col3.number_input(
            f"$w_{i+1}$",
            key=f"w{i}",
            min_value=0.0,
            value=w0
        )

        manual_data.append((a, b, w))

    # ==================================================
    # CASE 2: CSV UPLOAD (OVERRIDES MANUAL)
    # ==================================================
    st.sidebar.markdown("---")
    st.sidebar.subheader("Upload CSV (Optional)")

    uploaded_file = st.sidebar.file_uploader(
        "Upload Minisum SFL data (CSV)",
        type=["csv"]
    )

    has_header = st.sidebar.checkbox("My data has headers", value=True)

    st.sidebar.markdown(
        """
        **CSV format:**  
        - Columns: `a, b, w`  
        - All values numeric  
        - No empty cells  
        """
    )

    if uploaded_file is not None:
        try:
            # ---- Read CSV ----
            if has_header:
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file, header=None)
                df.columns = ["a", "b", "w"]

            # ---- Validation (EXACTLY as before) ----
            if list(df.columns) != ["a", "b", "w"]:
                st.sidebar.error("CSV must contain exactly columns: a, b, w")
                return []

            if df.isnull().any().any():
                st.sidebar.error("CSV contains empty cells")
                return []

            # ---- Type conversion ----
            df = df.astype(float)

            st.sidebar.success("CSV loaded successfully (manual input ignored)")
            return list(df.itertuples(index=False, name=None))

        except Exception as e:
            st.sidebar.error(f"Invalid CSV file: {e}")
            return []

    # ==================================================
    # DEFAULT: USE MANUAL INPUT
    # ==================================================
    return manual_data



def show_minisum_sfl(data):
    st.title("Minisum Single Facility Location Problem")
    
    # st.markdown(
    # """
    # This app solves the **Minisum Single Facility Location Problem** using:
    # - Rectilinear (L1) distance — *Graphical Approch*
    # - Rectilinear (L1) distance — *Median Method*
    # - Euclidean (L2) distance — *Weiszfeld Method*
    # - Squared Euclidean (L2²) distance — *Centroid*
    # - Minkowski Distance Model (Lp) — *Gradient Descent*
    # """
    # )
    
    if "plot_iso" not in st.session_state:
        st.session_state.plot_iso = True
    
    #Plotting Style
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
    tab0,tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        [
            "Overview",
            "Rectilinear (Graphical Approach)",
            "Rectilinear (Median Method)",
            "Iso-Contours",
            "Euclidean Models (L2² and L2)",
            "Minkowski distance(Lp)",
            "Comparison",
            "Equivalent LP (L1 Minisum)"
        ]
    )
    # --------------------------------------------------
    # TAB 0: Overview
    # --------------------------------------------------    
    
    with tab0:
        st.subheader("Single-Facility Minisum Location Problem")
    
        left_col, right_col = st.columns([2, 0.8])
    
        # ============================
        # LEFT COLUMN — EXPLANATION
        # ============================
        with left_col:
            st.markdown(
                """
            
                The **Minisum Single Facility Location Problem** determines the location
                of **one new facility** such that the **total weighted distance**
                to all existing facilities is minimized.
    
                This model is widely used for:
                - Warehouses and distribution centers  
                - Machines inside factories  
                - Service facilities with frequent interactions  
    
                ---
                """
            )    
            st.markdown(
                """
                ### Rectilinear Minisum Single Facility Location Problem (L1)
            
                In the **Minisum Single Facility Location Problem**, the objective is to locate
                a new facility at $(x,y)$ such that the **total weighted rectilinear distance**
                to all existing facilities is minimized.
            
                The mathematical formulation is:
                """
            )
            
            st.latex(
                r"""
                \min_{x,y} \; f(x,y)
                =
                \sum_{i=1}^{m}
                w_i \left(
                |x-a_i| + |y-b_i|
                \right)
                """
            )
            
            st.markdown(
                """
                #### Interpretation
            
                - Each existing facility $i$ is located at $(a_i, b_i)$
                - $w_i$ represents the **intensity of interaction**
                  (e.g., number of trips, flow, or demand)
                - Rectilinear distance assumes **movement only along horizontal and vertical directions**
                - The objective minimizes the **total cost of movement**.
            
                Because rectilinear distance is **additively separable**, the problem can be written as:
                """
            )
            
            st.latex(
                r"""
                f(x,y) = f_1(x) + f_2(y),
                \quad
                \text{where}
                \quad
                f_1(x)=\sum_{i=1}^{m} w_i |x-a_i|,
                \quad
                f_2(y)=\sum_{i=1}^{m} w_i |y-b_i|
                """
            )
            
            st.markdown(
                """
                This key property allows the **x-coordinate and y-coordinate**
                of the optimal facility to be determined **independently**,
                leading to graphical and median-based solution methods.
                """
            )

            st.markdown(
                """
                ---
                ### Input Data Instructions
            
                Use the **sidebar** by clicking the **>>** icon on the left-hand side of the screen.
                Data can be provided using one of the following methods:
            
                **1. CSV Upload**
                - File format: `a, b, w`
                - Each row corresponds to one existing facility
                - `w` represents the interaction weight (e.g., number of trips, flow, or demand)
            
                **2. Manual Input**
                - Specify the number of existing facilities
                - Enter the coordinates `ai`, `bi` and corresponding weights `wi`
                - This option is suitable for classroom examples and exploratory analysis
                """
            )

            show_data = st.checkbox("Show existing facility data")
            if show_data:
                df_data = pd.DataFrame(
                    data,
                    columns=["a (x-coordinate)", "b (y-coordinate)", "w (weight)"]
                )
            
                st.markdown("#### Existing Facility Data")
                st.dataframe(df_data, hide_index=True, use_container_width=True)
                
            st.markdown(    
                """
                ---
                ### Solution Methods Implemented
    
                This page solves the problem using multiple distance models
                and solution techniques:
    
                - **Rectilinear (L1)** — Graphical Approach  
                - **Rectilinear (L1)** — Median Method  
                - **Rectilinear (L1)** — Iso-Cost Contours  
                - **Squared Euclidean (L2²)** — Centroid Method  
                - **Euclidean (L2)** — Weiszfeld Algorithm  
                - **Minkowski (Lp)** — Gradient Descent  
                - **Equivalent Linear Programming Formulation**
                """
            )
            
        # ============================
        # RIGHT COLUMN — NAVIGATION
        # ============================
        with right_col:
            st.markdown("### Page Structure")
        
            st.markdown(
                """
                | Tab Name | Description |
                |---------|-------------|
                | **Overview** | Problem description, interpretation, and data input instructions |
                | **Rectilinear (Graphical Approach)** | Visual solution using piecewise linear functions |
                | **Rectilinear (Median Method)** | Closed-form solution using weighted medians |
                | **Iso-Contours** | Iso-cost contours and near-optimal alternative locations |
                | **Euclidean Models (L2² and L2)** | Centroid and Weiszfeld solution methods |
                | **Minkowski distance (Lp)** | Generalized distance model solved via gradient descent |
                | **Comparison** | Side-by-side comparison of all distance models |
                | **Equivalent LP (L1 Minisum)** | Linear programming formulation of the L1 minisum problem |
                """
            )
        
            st.info(
                "Use the tabs at the top of the page to navigate between sections."
            )

    # --------------------------------------------------
    # TAB 1: Rectilinear (Graphical Approch)
    # --------------------------------------------------
    with tab1:
        st.subheader("Rectilinear Distance (L1) – Graphical Approach")
    
        # --------------------------------------------------
        # Solve once (authoritative solution set)
        # --------------------------------------------------
        res_L1 = slv.solve_single_facility_L1_median(data)
        x_low, x_high = res_L1["x_range"]
        y_low, y_high = res_L1["y_range"]
    
        st.latex(r"\min_{x,y} f_{L1}(x,y) = \sum_{i=1}^{m} w_i \big(|x-a_i| + |y-b_i|\big)")
        st.markdown(
            """
            Writing the functions $f_1(x)$ and $f_2(y)$ such that the
            coordinates of the existing facilities appear in **non-decreasing order**.
            """
        )
        # -----------------------------------------
        # Checkbox to show/hide data
        # -----------------------------------------
        show_data = st.checkbox("Show Data")
        
        if show_data:
            df_data = pd.DataFrame(
                data,
                columns=["a (x-coordinate)", "b (y-coordinate)", "w (weight)"]
            )
        
            st.markdown("### Existing Facility Data")
            st.dataframe(df_data, hide_index=True)
            
        col_x, col_y = st.columns(2)
    
        # ============================
        # LEFT COLUMN: f1(x)
        # ============================
        with col_x:
            st.markdown("### Graphical solution for x-coordinate")
    
            st.latex(r"f_1(x) = \sum_{i=1}^{m} w_i |x - a_i|")
    
            # Expanded f1(x) in sorted order
            terms_x = [
                rf"{w:g}|x-{a:g}|"
                for a, w in sorted([(a, w) for a, _, w in data])
            ]
            st.latex(r"f_1(x) = " + " + ".join(terms_x))
    
            a_vals = [a for a, _, _ in data]
            fig_x, x_star_plot, f1_star = slv.plot_piecewise_L1(
                a_vals,
                lambda x: slv.f1_value(x, data),
                "x"
            )
    
            st.pyplot(fig_x)
    
            # Correct conclusion for x*
            if x_low == x_high:
                st.latex(
                    rf"""
                    f_1(x) \text{{ is minimum at }} x^* = {x_low:g},
                    \quad f_1(x^*) = {int(f1_star)}
                    """
                )
            else:
                st.latex(
                    rf"""
                    f_1(x) \text{{ is minimum for all }} x \in [{x_low:g}, {x_high:g}]
                    """
                )
    
        # ============================
        # RIGHT COLUMN: f2(y)
        # ============================
        with col_y:
            st.markdown("### Graphical solution for y-coordinate")
    
            st.latex(r"f_2(y) = \sum_{i=1}^{m} w_i |y - b_i|")
    
            # Expanded f2(y) in sorted order
            terms_y = [
                rf"{w:g}|y-{b:g}|"
                for b, w in sorted([(b, w) for _, b, w in data])
            ]
            st.latex(r"f_2(y) = " + " + ".join(terms_y))
    
            b_vals = [b for _, b, _ in data]
            fig_y, y_star_plot, f2_star = slv.plot_piecewise_L1(
                b_vals,
                lambda y: slv.f2_value(y, data),
                "y"
            )
    
            st.pyplot(fig_y)
    
            # Correct conclusion for y*
            if y_low == y_high:
                st.latex(
                    rf"""
                    f_2(y) \text{{ is minimum at }} y^* = {y_low:g},
                    \quad f_2(y^*) = {int(f2_star)}
                    """
                )
            else:
                st.latex(
                    rf"""
                    f_2(y) \text{{ is minimum for all }} y \in [{y_low:g}, {y_high:g}]
                    """
                )
    
        # ============================
        # FINAL RESULT (POINT or SET)
        # ============================
        st.markdown("---")
    
        if x_low == x_high and y_low == y_high:
            st.latex(
                rf"""
                \text{{Optimal location}} = (x^*, y^*) = ({x_low:g}, {y_low:g})
                """
            )
    
            st.latex(
                rf"""
                f(x^*, y^*) = f_1(x^*) + f_2(y^*)
                = {int(f1_star)} + {int(f2_star)}
                = {int(f1_star + f2_star)}
                """
            )
        else:
            st.latex(
                rf"""
                \text{{Optimal location set}}
                =
                \{{(x,y)\mid x \in [{x_low:g},{x_high:g}],\;
                y \in [{y_low:g},{y_high:g}]\}}
                """
            )
    
            st.markdown(
                "<p style='text-align:center; font-size:0.85em; color:gray;'>"
                "All points within this region are optimal and yield the same minimum rectilinear cost."
                "</p>",
                unsafe_allow_html=True
            )
    
    #
    
    
    # --------------------------------------------------
    # TAB 2: Rectilinear (Median Method)
    # --------------------------------------------------
    with tab2:
        st.subheader("Rectilinear Distance (L1) – Median Method")
        st.latex(r"\min_{x,y} f_{L1}(x,y) = \sum_{i=1}^{m} w_i \big(|x-a_i| + |y-b_i|\big)")
        # --------------------------------------------------
        # Solve once (used everywhere below)
        # --------------------------------------------------
        res_L1 = slv.solve_single_facility_L1_median(data)
        x_low, x_high = res_L1["x_range"]
        y_low, y_high = res_L1["y_range"]
    
        # --------------------------------------------------
        # Two equal columns: x- and y- derivation
        # --------------------------------------------------
        col_x, col_y = st.columns(2)
    
        # ============================
        # LEFT COLUMN: x-coordinate
        # ============================
        with col_x:
            st.markdown("### To find x-coordinate")
    
            # f1(x) – general form
            st.latex(
                r"""
                f_1(x) = \sum_{i=1}^{m} w_i \lvert x - a_i \rvert
                """
            )
    
            # Expanded f1(x) in non-decreasing order of a_i
            terms_x = [
                rf"{w:g}\lvert x - {a:g}\rvert"
                for a, w in sorted(
                    [(a, w) for a, _, w in data],
                    key=lambda t: t[0]
                )
            ]
            
            st.latex(r"f_1(x) = " + " + ".join(terms_x))
    
            # Weighted median table (x)
            a_vals = [a for a, _, _ in data]
            weights = [w for _, _, w in data]
            labels = list(range(1, len(data) + 1))
    
            df_x, median_x, total_w = slv.build_weighted_median_table(
                a_vals, weights, labels
            )
    
            st.dataframe(df_x, hide_index=True)
    
            st.latex(
                rf"""
                \text{{Median}} = \frac{{\sum w_i}}{{2}}
                = \frac{{{total_w:g}}}{{2}} = {total_w/2:g}
                """
            )
    
            st.write(
                f"**Median occurs at facility {median_x['Existing Facility']}**"
            )
    
            # Correct conclusion for x*
            if x_low == x_high:
                st.latex(rf"x^* = {x_low:g}")
            else:
                st.latex(rf"x^* \in [{x_low:g}, {x_high:g}]")
    
        # ============================
        # RIGHT COLUMN: y-coordinate
        # ============================
        with col_y:
            st.markdown("### To find y-coordinate")
    
            # f2(y) – general form
            st.latex(
                r"""
                f_2(y) = \sum_{i=1}^{m} w_i \lvert y - b_i \rvert
                """
            )
    
            # Expanded f2(y) in non-decreasing order of b_i
            terms_y = [
                rf"{w:g}\lvert y - {b:g}\rvert"
                for b, w in sorted(
                    [(b, w) for _, b, w in data],
                    key=lambda t: t[0]
                )
            ]
            
            st.latex(r"f_2(y) = " + " + ".join(terms_y))
    
            # Weighted median table (y)
            b_vals = [b for _, b, _ in data]
    
            df_y, median_y, total_w = slv.build_weighted_median_table(
                b_vals, weights, labels
            )
    
            st.dataframe(df_y, hide_index=True)
    
            st.latex(
                rf"""
                \text{{Median}} = \frac{{\sum w_i}}{{2}}
                = \frac{{{total_w:g}}}{{2}} = {total_w/2:g}
                """
            )
    
            st.write(
                f"**Median occurs at facility {median_y['Existing Facility']}**"
            )
    
            # Correct conclusion for y*
            if y_low == y_high:
                st.latex(rf"y^* = {y_low:g}")
            else:
                st.latex(rf"y^* \in [{y_low:g}, {y_high:g}]")
    
        # --------------------------------------------------
        # FINAL RESULT (SET or POINT — CORRECTLY)
        # --------------------------------------------------
        st.markdown("---")
    
        if x_low == x_high and y_low == y_high:
            st.latex(
                rf"""
                \Rightarrow \text{{Optimal location}} = ({x_low:g}, {y_low:g})
                """
            )
        else:
            st.latex(
                rf"""
                \Rightarrow \text{{Optimal location set}}
                =
                \{{(x,y)\mid x \in [{x_low:g},{x_high:g}],\;
                y \in [{y_low:g},{y_high:g}]\}}
                """
            )
    
        # --------------------------------------------------
        # RESULTS SUMMARY
        # --------------------------------------------------
        st.markdown("---")
        st.subheader("Results Summary")
    
        st.write("**Optimal x-range:**", f"[{x_low:g}, {x_high:g}]")
        st.write("**Optimal y-range:**", f"[{y_low:g}, {y_high:g}]")
    
        solution_type, vertices = slv.classify_L1_solution(
            res_L1["x_range"],
            res_L1["y_range"]
        )
    
        st.write(
            "**Optimal solution:**",
            "Unique optimal solution exists"
            if solution_type == "Unique Point"
            else "Multiple optimal solutions exist"
        )
    
        st.write("**Solution region type:**", solution_type)
    
        st.markdown("**Solution region vertices:**")
        cols = st.columns(len(vertices))
        for col, (x, y), i in zip(cols, vertices, range(1, len(vertices) + 1)):
            col.markdown(f"**V{i}**  \n({x}, {y})")
    
        st.markdown(
            f"**Objective value:** {res_L1['obj']:.4f}"
        )

    
    # --------------------------------------------------
    # TAB 3: Iso-Contours
    # --------------------------------------------------
    with tab3:
        st.subheader("Rectilinear Iso-Cost Contours")
    
        # ---- two-column layout ----
        left_col, right_col = st.columns([1.5, 1.3])
    
        # ============================
        # LEFT COLUMN: Inputs + Table
        # ============================
        with left_col:
            n = st.number_input(
                "Number of contour points",
                min_value=1,
                step=1,
                value=2
            )
    
            st.markdown("### Points through which contours pass")
    
            default_contour_points = [
                (6.0, 0.0),
                (3.0, 3.0),
            ]
    
            contour_points = []
    
            for i in range(n):
                col1, col2 = st.columns(2)
    
                if i < len(default_contour_points):
                    x0, y0 = default_contour_points[i]
                else:
                    x0, y0 = (i + 1), (i + 0.5)
    
                x = col1.number_input(
                    f"x{i+1}",
                    key=f"x{i}",
                    value=float(x0),
                    step=0.5,
                    format="%.2f"
                )

                y = col2.number_input(
                    f"y{i+1}",
                    key=f"y{i}",
                    value=float(y0),
                    step=0.5,
                    format="%.2f"
                )
    
                contour_points.append((x, y))
    
            # ---- table ----
            _, point_info = slv.plot_iso_contours_L1_with_optimal_set(
                data,
                contour_points
            )
    
            df = pd.DataFrame(point_info)
            df.rename(columns={"label": "Contour Point"}, inplace=True)
    
            st.markdown("### Cost at Contour Points")
            st.dataframe(df, hide_index=True)
    
        # ============================
        # RIGHT COLUMN: Visualization
        # ============================
        with right_col:
            fig, _ = slv.plot_iso_contours_L1_with_optimal_set(
                data,
                contour_points
            )
    
            st.pyplot(fig)
    
    # --------------------------------------------------
    # TAB 4: Euclidean Models
    # --------------------------------------------------
    with tab4:
        st.subheader("Euclidean Distance Models - (L1 & L2)")
    
        # ---- two-column layout ----
        col1, col2 = st.columns(2)
    
        # ============================
        # LEFT: Squared Euclidean
        # ============================
        with col1:
            st.markdown("#### Squared Euclidean (L2²) — Centroid Method")
                    # ---- Objective functions (LaTeX) ----
            st.latex(
                r"""
                \min_{x,y}\; f_{L2^2}(x,y)
                = \sum_{i=1}^{m} w_i \big[(x-a_i)^2 + (y-b_i)^2\big]
                """
            )
            st.markdown("**Optimality condition (Centroid):**")
            
            st.latex(
                r"""
                x^* = \frac{\sum_{i=1}^{m} w_i a_i}{\sum_{i=1}^{m} w_i},
                \qquad
                y^* = \frac{\sum_{i=1}^{m} w_i b_i}{\sum_{i=1}^{m} w_i}
                """
            )
            
            res_L2sq = slv.solve_single_facility_squared_euclidean(data)
    
            st.markdown(
                f"**Optimal location:** ({res_L2sq["x_opt"]:.4f},{res_L2sq["y_opt"]:.4f})"
            )

            st.markdown(f"**Objective value:** {res_L2sq["opt_val"]:.4f}")
    
        # ============================
        # RIGHT: Euclidean (Weiszfeld)
        # ============================
        with col2:
            st.markdown("#### Euclidean (L2) — Weiszfeld Method")
             # ---- Objective functions (LaTeX) ----
            st.latex(
                r"""
                \min_{x,y}\; f_{L2}(x,y)
                = \sum_{i=1}^{m} w_i \sqrt{(x-a_i)^2 + (y-b_i)^2}
                """
            )
            st.markdown("**Initial point:**")
            
            st.latex(
                r"""
                x^{(0)} = \frac{\sum_{i=1}^{m} w_i a_i}{\sum_{i=1}^{m} w_i},
                \qquad
                y^{(0)} = \frac{\sum_{i=1}^{m} w_i b_i}{\sum_{i=1}^{m} w_i}
                """
            )
            
            st.markdown("**Weiszfeld iteration (k-th step):**")
            
            st.latex(
                r"""
                x^{(k)} =
                \frac{\sum_{i=1}^{m} a_i \phi_i(x^{(k-1)},y^{(k-1)})}
                {\sum_{i=1}^{m} \phi_i(x^{(k-1)},y^{(k-1)})}
                """
            )
            
            st.latex(
                r"""
                y^{(k)} =
                \frac{\sum_{i=1}^{m} b_i \phi_i(x^{(k-1)},y^{(k-1)})}
                {\sum_{i=1}^{m} \phi_i(x^{(k-1)},y^{(k-1)})}
                """
            )
            
            st.latex(
                r"""
                \phi_i(x,y) =
                \frac{w_i}{\sqrt{(x-a_i)^2 + (y-b_i)^2}}
                """
            )
    
            show_iter = st.checkbox("Show iteration history")
    
            res_L2 = slv.solve_single_facility_euclidean(
                data,
                store_history=show_iter
            )
    
            st.markdown(
                f"""
                **Optimal location:** ({res_L2['x_opt']:.4f}, {res_L2['y_opt']:.4f})  
                **Iterations:** {res_L2['iterations']}  
                **Converged:** {res_L2['converged']}  
                """
            )

    
            obj_val_L2 = slv.obj_L2(
                res_L2["x_opt"],
                res_L2["y_opt"],
                data
            )
    
            st.markdown(f"**Objective value:** {obj_val_L2:.4f}")
    
            if show_iter and res_L2["history"] is not None:
                hist_df = pd.DataFrame(
                    res_L2["history"],
                    columns=["Iteration", "x", "y"]
                )
                st.dataframe(hist_df, hide_index=True)
    
    # --------------------------------------------------
    # Minkowski distance Model (Lp) 
    # -----------------------------------------
    with tab5:
        st.subheader("Minkowski Distance Model (Lp)")
    
        # ---- Two-column layout ----
        left_col, right_col = st.columns([1.8, 1.2])
    
        # ============================
        # LEFT COLUMN: Theory + Inputs
        # ============================
        with left_col:
            st.markdown(
                """
                This model generalizes **L1**, **L2**, and **L∞** distances
                using the **Lp (Minkowski) norm**.
                """
            )
    
            st.markdown("### Solution Method: Gradient Descent")
    
            st.markdown(
                """
                For general values of **p**, the Minkowski objective function does not admit
                a closed-form solution. Therefore, the problem is solved using
                **Gradient Descent**, an iterative numerical optimization method.
                """
            )
    
            # ---- p selector ----
            p = st.slider(
                "Select value of p (integer)",
                min_value=1,
                max_value=100,
                step=1,
                value=2
            )
    
            # ---- alpha selector ----
            alpha = st.slider(
                "Step size (α)",
                min_value=0.001,
                max_value=0.5,
                value=0.1,
                step=0.001,
                format="%.3f"
            )
    
            st.markdown("**Meaning of the step size (α):**")
    
            st.markdown(
                """
                - **α (alpha)** controls the magnitude of movement in the direction of the
                  negative gradient.
                - Smaller values of α lead to **stable but slower convergence**.
                - Larger values of α lead to **faster convergence**, but may cause
                  **oscillations or divergence**.
                """
            )
    
            # ---- Objective function ----
            st.latex(
                r"""
                \min_{x,y} \; f_p(x,y)
                =
                \sum_{i=1}^{m} w_i
                \left(
                |x-a_i|^p + |y-b_i|^p
                \right)^{1/p}
                """
            )
    
            # ---- Initial point ----
            st.markdown("**Initial point:**")
    
            st.latex(
                r"""
                x^{(0)} =
                \frac{\sum_{i=1}^{m} w_i a_i}{\sum_{i=1}^{m} w_i},
                \qquad
                y^{(0)} =
                \frac{\sum_{i=1}^{m} w_i b_i}{\sum_{i=1}^{m} w_i}
                """
            )
    
            # ---- Iterative update ----
            st.markdown("**Iterative optimization idea:**")
    
            st.latex(
                r"""
                (x^{(k)}, y^{(k)})
                =
                (x^{(k-1)}, y^{(k-1)})
                -
                \alpha \nabla f_p(x^{(k-1)}, y^{(k-1)})
                """
            )
            st.latex(
                r"""
                \nabla f_p\!\left(x^{(k)},y^{(k)}\right)
                =
                \sum_{i=1}^{m}
                w_i
                \frac{
                \begin{pmatrix}
                |x^{(k)}-a_i|^{p-1}\mathrm{sgn}(x^{(k)}-a_i), \\
                |y^{(k)}-b_i|^{p-1}\mathrm{sgn}(y^{(k)}-b_i)
                \end{pmatrix}
                }{
                \left(
                |x^{(k)}-a_i|^p + |y^{(k)}-b_i|^p
                \right)^{\frac{p-1}{p}}
                }
                """
            )
    
            st.markdown("**Special cases:**")
    
            st.latex(
                r"""
                \begin{aligned}
                p = 1 &\Rightarrow \text{Rectilinear (L1)} \\
                p = 2 &\Rightarrow \text{Euclidean (L2)} \\
                p \to \infty &\Rightarrow \text{Chebyshev ($L{\infty}$)}
                \end{aligned}
                """
            )
    
        # ============================
        # RIGHT COLUMN: Visualization
        # ============================
        alpha_ref = 0.1  # fixed alpha for path computation
        
        with right_col:
            # ---- Blue path (fixed alpha) ----
            path_x = []
            path_y = []
        
            for p_val in range(1, p + 1):
                res_tmp = slv.solve_single_facility_Lp(
                    data,
                    p_val,
                    alpha=alpha_ref
                )
                path_x.append(float(res_tmp["x_opt"]))
                path_y.append(float(res_tmp["y_opt"]))
        
            # ---- Red point (user-selected alpha) ----
            res_Lp = slv.solve_single_facility_Lp(
                data,
                p,
                alpha=alpha
            )
        
            x_opt = float(res_Lp["x_opt"])
            y_opt = float(res_Lp["y_opt"])
            obj_val = float(res_Lp["obj"])
        
            st.markdown("### Optimal Solution")
        
            st.markdown(
                f"""
                **Optimal location:** ({x_opt:.4f}, {y_opt:.4f})  
                **Objective value:** {obj_val:.4f}
                """
            )

            fig = slv.plot_Lp_solution_path(
                data,
                path_x,
                path_y,
                x_opt,
                y_opt,
                p
            )
        
            st.pyplot(fig)
        
            st.caption(
                "The blue trajectory shows the optimal location as p increases "
                "using a fixed step size. The red point shows the current solution "
                "obtained with the selected α."
            )
    
    
    
    # --------------------------------------------------
    # TAB 6: Comparison
    # --------------------------------------------------
    
    with tab6:
        st.subheader("Comparison of Distance Models")
    
        # ---- two-column layout ----
        left_col, right_col = st.columns([1.5, 1.3])
    
        results = slv.compare_single_facility_models(data)
    
        # ============================
        # LEFT COLUMN: Table
        # ============================
        with left_col:
            st.markdown("### Numerical Comparison")
    
            comp_data = {
                "Model": [],
                "x": [],
                "y": [],
                "Objective Value": []
            }
    
            for model, res in results.items():
                if model == "L1 (Rectilinear)":
                    x_low, x_high = res["x_range"]
                    y_low, y_high = res["y_range"]
    
                    if x_low == x_high:
                        x_display = f"{x_low:g}"
                        y_display = f"{y_low:g}"
                    else:
                        x_display = f"[{x_low:g}, {x_high:g}]"
                        y_display = f"[{y_low:g}, {y_high:g}]"
    
                    comp_data["Model"].append(model)
                    comp_data["x"].append(x_display)
                    comp_data["y"].append(y_display)
                    comp_data["Objective Value"].append(res["obj"])
    
                else:
                    comp_data["Model"].append(model)
                    comp_data["x"].append(res["x"])
                    comp_data["y"].append(res["y"])
                    comp_data["Objective Value"].append(res["obj"])
    
            df = pd.DataFrame(comp_data)
            st.dataframe(df, hide_index=True)
    
        # ============================
        # RIGHT COLUMN: Plot
        # ============================
        with right_col:
            fig = slv.plot_optimal_locations(data, results)
            st.pyplot(fig)
    # --------------------------------------------------
    # TAB 7: Equivalent LP for Minisum SFL (L1)
    # --------------------------------------------------
    with tab7:
        st.subheader("Equivalent Linear Programming Formulation")
        
        m = len(data)
    
        st.markdown(
            """
            We convert the **Minisum Single Facility Location Problem**
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
            \min_{x,y} \; f_{L1}(x,y)
            =
            \sum_{i=1}^{m} w_i
            \big(
            |x-a_i| + |y-b_i|
            \big)
            """
        )
    
        # --------------------------------------------------
        # Change of Variables Explanation
        # --------------------------------------------------
        st.markdown("### Change of Variables")
    
        st.markdown(
            """
            For each existing facility \\(i\\), define the following nonnegative variables:
            """
        )
    
        st.latex(
            r"""
            \begin{aligned}
            r_i &: \text{amount by which the new facility is to the RIGHT of facility } i \\
            s_i &: \text{amount by which the new facility is to the LEFT of facility } i \\
            u_i &: \text{amount by which the new facility is ABOVE facility } i \\
            v_i &: \text{amount by which the new facility is BELOW facility } i
            \end{aligned}
            """
        )
    
        st.markdown(
            """
            These variables allow us to write absolute values as linear expressions:
            """
        )
    
        st.latex(
            r"""
            |x-a_i| = r_i + s_i,
            \qquad
            |y-b_i| = u_i + v_i
            """
        )
        # --------------------------------------------------
        # Final LP Statement
        # --------------------------------------------------
        st.markdown("### Equivalent Linear Program")
    
        st.latex(
            r"""
            \begin{aligned}
            \min \quad &
            \sum_{i=1}^{m} w_i
            ( r_i + s_i + u_i + v_i ) \\
            \text{s.t.} \quad &
            x - r_i + s_i = a_i, \quad i=1,\ldots,m \\
            &
            y - u_i + v_i = b_i, \quad i=1,\ldots,m \\
            &
            r_i, s_i, u_i, v_i \ge 0, \quad i=1,\ldots,m \\
            &
            x, y \text{ unrestricted}
            \end{aligned}
            """
        )
        # --------------------------------------------------
        # Equivalent LP Objective
        # --------------------------------------------------
        st.markdown("### Expanded Linear Objective Function")
    
        terms = [
            rf"{w:g}(r_{{{i}}} + s_{{{i}}} + u_{{{i}}} + v_{{{i}}})"
            for i, (_, _, w) in enumerate(data, start=1)
        ]
        
        st.latex(
            r"\min f(x,y) =\; " + " + ".join(terms)
        )
    
        # --------------------------------------------------
        # Constraints (Expanded for all i)
        # --------------------------------------------------
        st.markdown("### Constraints")

        col_x, col_y = st.columns([1.5, 1.5])
        
        # -----------------------------
        # LEFT COLUMN: X-constraints
        # -----------------------------
        with col_x:
            st.markdown("#### X-coordinate constraints")
            for i, (a, _, _) in enumerate(data, start=1):
                st.latex(
                    rf"x - r_{{{i}}} + s_{{{i}}} = {a}"
                )
        
        # -----------------------------
        # RIGHT COLUMN: Y-constraints
        # -----------------------------
        with col_y:
            st.markdown("#### Y-coordinate constraints")
            for i, (_, b, _) in enumerate(data, start=1):
                st.latex(
                    rf"y - u_{{{i}}} + v_{{{i}}} = {b}"
                )

        # --------------------------------------------------
        # Nonnegativity & Free Variables
        # --------------------------------------------------
        #st.markdown("### Variable Restrictions")
    
        st.latex(
            r"""
            r_i,\; s_i,\; u_i,\; v_i \ge 0
            \quad
            \forall i = 1,2,\ldots,m
            """
        )
    
        st.latex(
            r"""
            x,\; y \;\; \text{unrestricted in sign}
            """
        )
        # --------------------------------------------------
        # Solve LP using PuLP (Optional)
        # --------------------------------------------------
        st.markdown("---")
        solve_lp = st.checkbox("Solve this LP using PuLP")
        
        if solve_lp:
            st.subheader("PuLP Solution")
        
            # -----------------------------
            # Create LP problem
            # -----------------------------
            prob = pulp.LpProblem(
                "Minisum_SFL_L1",
                pulp.LpMinimize
            )
        
            # -----------------------------
            # Decision Variables
            # -----------------------------
            x = pulp.LpVariable("x", lowBound=None)
            y = pulp.LpVariable("y", lowBound=None)
        
            r = {
                i: pulp.LpVariable(f"r_{i}", lowBound=0)
                for i in range(1, m + 1)
            }
            s = {
                i: pulp.LpVariable(f"s_{i}", lowBound=0)
                for i in range(1, m + 1)
            }
            u = {
                i: pulp.LpVariable(f"u_{i}", lowBound=0)
                for i in range(1, m + 1)
            }
            v = {
                i: pulp.LpVariable(f"v_{i}", lowBound=0)
                for i in range(1, m + 1)
            }
        
            # -----------------------------
            # Objective Function
            # -----------------------------
            prob += pulp.lpSum(
                w * (r[i] + s[i] + u[i] + v[i])
                for i, (_, _, w) in enumerate(data, start=1)
            )
        
            # -----------------------------
            # Constraints
            # -----------------------------
            for i, (a, b, _) in enumerate(data, start=1):
                prob += x - r[i] + s[i] == a
                prob += y - u[i] + v[i] == b
        
            # -----------------------------
            # Solve
            # -----------------------------
            prob.solve(pulp.PULP_CBC_CMD(msg=False))
        
            # -----------------------------
            # Results
            # -----------------------------
            st.markdown("### Optimal Solution")
        
            st.latex(
                rf"""
                x^* = {pulp.value(x):.4f},
                \qquad
                y^* = {pulp.value(y):.4f}
                """
            )
        
            st.latex(
                rf"""
                f^* = {pulp.value(prob.objective):.4f}
                """
            )
        
            # -----------------------------
            # Decision Variable Table
            # -----------------------------
            rows = []
        
            for i in range(1, m + 1):
                rows.append({
                    "i": i,
                    "r_i": pulp.value(r[i]),
                    "s_i": pulp.value(s[i]),
                    "u_i": pulp.value(u[i]),
                    "v_i": pulp.value(v[i]),
                })
        
            df_vars = pd.DataFrame(rows)
        
            st.markdown("### Decision Variable Values")
            st.dataframe(
                df_vars,
                hide_index=True,
                use_container_width=True
            )
        
            st.caption(
                "The PuLP solver returns one optimal solution of the linear program. "
                "When the Median Method yields multiple optimal locations, "
                "the LP selects a representative extreme point from the optimal solution set."
            )
            # --------------------------------------------------
            # Active Constraints – Geometric Interpretation (Correct)
            # --------------------------------------------------
            st.markdown("### Active Constraints (Geometric Interpretation)")
            
            tol = 1e-6
            rows = []
            
            for i, (a, b, _) in enumerate(data, start=1):
                ri = pulp.value(r[i])
                si = pulp.value(s[i])
                ui = pulp.value(u[i])
                vi = pulp.value(v[i])
            
                # ---- X direction ----
                if ri > tol:
                    x_relation = "Optimal x is to the RIGHT of facility"
                elif si > tol:
                    x_relation = "Optimal x is to the LEFT of facility"
                else:
                    x_relation = "Optimal x coincides with facility"
                
                # ---- Y direction ----
                if ui > tol:
                    y_relation = "Optimal y is ABOVE the facility"
                elif vi > tol:
                    y_relation = "Optimal y is BELOW the facility"
                else:
                    y_relation = "Optimal y coincides with facility"

                rows.append({
                    "Facility i": i,
                    "Relative position in x-direction": x_relation,
                    "Relative position in y-direction": y_relation
                })
            
            df_active = pd.DataFrame(rows)
            
            st.dataframe(
                df_active,
                hide_index=True,
                use_container_width=True
            )

            st.markdown(
                """
                **Interpretation of Active Constraints:**
            
                - If **rᵢ > 0**, the new facility lies **to the right** of facility *i*
                - If **sᵢ > 0**, the new facility lies **to the left** of facility *i*
                - If **uᵢ > 0**, the new facility lies **above** facility *i*
                - If **vᵢ > 0**, the new facility lies **below** facility *i*
            
                Active constraints correspond to **binding absolute-value terms**
                and determine the geometry of the LP optimal solution.
                """
            )
            