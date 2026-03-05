"""
================================================================================
Minimax Single Location Problem Page
================================================================================
"""
import streamlit as st
import solver.minimax_sfl as slv
import pandas as pd
from streamlit_plotly_events import plotly_events

def build_inputs():
    # --------------------------------------------------
    # Sidebar: Minimax SFL Input
    # --------------------------------------------------
    st.sidebar.header("Input Data")

    # Initialize Session State Variables (Unique keys for Minimax SFL)
    if "m_val_mm_sf" not in st.session_state: 
        st.session_state.m_val_mm_sf = 5
    if "uploaded_csv_id_mm_sf" not in st.session_state:
        st.session_state.uploaded_csv_id_mm_sf = None

    # ==================================================
    # 1. DATA LOADERS (Examples & CSV)
    # ==================================================
    st.sidebar.subheader("Load Data")

    # --- A. Example Loader ---
    with st.sidebar.expander("Load Example Problem", expanded=False):
        st.caption("Select a preset Example to load data.")
        
        col_ex1, col_ex2 = st.columns(2)
        col_ex3, col_ex4 = st.columns(2)
        
        # Buttons
        load_ex1 = col_ex1.button("Example 1", help="5 Demand Points")
        load_ex2 = col_ex2.button("Example 2", help="19 Demand Points (Tompkins 10.9(d))")
        load_ex3 = col_ex3.button("Example 3", help="6 Demand Points (Tompkins 10.8)")
        load_ex4 = col_ex4.button("Example 4", help="12 Demand Points (Francis 9.1)")
        
        if load_ex1:
            m_new = 5
            st.session_state.m_val_mm_sf = m_new
            st.session_state["m_input_mm_sf"] = m_new
            st.session_state.mm_sf_df = pd.DataFrame({
                "a (x-coord)": [4.0, 5.0, 13.0, 10.0, 4.0],
                "b (y-coord)": [3.0, 11.0, 13.0, 6.0, 6.0]
            }, index=[f"DP{i+1}" for i in range(m_new)])
            st.rerun()

        if load_ex2:
            m_new = 19
            st.session_state.m_val_mm_sf = m_new
            st.session_state["m_input_mm_sf"] = m_new
            st.session_state.mm_sf_df = pd.DataFrame({
                "a (x-coord)": [2.0, 2.0, 4.0, 4.0, 4.0, 6.0, 6.0, 6.0, 6.0, 8.0, 8.0, 10.0, 12.0, 12.0, 12.0, 12.0, 14.0, 14.0, 14.0],
                "b (y-coord)": [2.0, 4.0, 6.0, 10.0, 12.0, 2.0, 4.0, 10.0, 12.0, 4.0, 8.0, 10.0, 4.0, 8.0, 12.0, 14.0, 2.0, 12.0, 14.0]

            }, index=[f"DP{i+1}" for i in range(m_new)])
            st.rerun()

        if load_ex3:
            m_new = 4
            st.session_state.m_val_mm_sf = m_new
            st.session_state["m_input_mm_sf"] = m_new
            st.session_state.mm_sf_df = pd.DataFrame({
                "a (x-coord)": [20.0, 25.0, 13.0, 25.0, 4.0, 18.0],
                "b (y-coord)": [15.0, 25.0, 32.0, 14.0, 21.0, 8.0]

            }, index=[f"DP{i+1}" for i in range(m_new)])
            st.rerun()

        if load_ex4:
            m_new = 12
            st.session_state.m_val_mm_sf = m_new
            st.session_state["m_input_mm_sf"] = m_new
            st.session_state.mm_sf_df = pd.DataFrame({
                "a (x-coord)": [0.0, 2.0, 3.0, 3.0, 4.0, 8.0, 9.0, 8.0, 7.0, 5.0, 4.0, 2.0],
                "b (y-coord)": [3.0, 7.0, 7.0, 10.0, 7.0, 7.0, 6.0, 5.0, 6.0, 6.0, 3.0, 1.0]

            }, index=[f"DP{i+1}" for i in range(m_new)])
            st.rerun()

    # --- B. CSV Loader ---
    with st.sidebar.expander("Upload CSV Data", expanded=False):
        uploaded_file = st.file_uploader(
            "Upload Minimax SFL data (CSV)",
            type=["csv"]
        )
        has_header = st.checkbox("My data has headers", value=True)
        st.caption("CSV format: 2 numeric columns (a, b). No empty cells.")

        # Process newly uploaded file
        if uploaded_file is not None:
            # Check if this is a NEW upload by comparing file_ids
            if st.session_state.uploaded_csv_id_mm_sf != uploaded_file.file_id:
                try:
                    if has_header:
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_csv(uploaded_file, header=None)
                    
                    # Validation
                    if len(df.columns) != 2:
                        st.error("CSV must contain exactly 2 columns: a, b")
                    elif df.isnull().any().any():
                        st.error("CSV contains empty cells")
                    else:
                        df = df.astype(float)
                        df.columns = ["a (x-coord)", "b (y-coord)"]
                        
                        m_new = len(df)
                        df.index = [f"DP{i+1}" for i in range(m_new)]
                        
                        # Update session state with CSV data
                        st.session_state.m_val_mm_sf = m_new
                        st.session_state["m_input_mm_sf"] = m_new
                        st.session_state.mm_sf_df = df
                        st.session_state.uploaded_csv_id_mm_sf = uploaded_file.file_id
                        
                        st.rerun() # Refresh to show CSV data in the table
                except Exception as e:
                    st.error(f"Invalid CSV file: {e}")
                    # Prevent infinite looping on a bad file
                    st.session_state.uploaded_csv_id_mm_sf = uploaded_file.file_id 
        else:
            # Reset tracker if user clicks "X" to remove the uploaded file
            st.session_state.uploaded_csv_id_mm_sf = None

    # ==================================================
    # 2. TABULAR DATA EDITOR (Manual Input / Edits)
    # ==================================================
    st.sidebar.markdown("---")
    st.sidebar.subheader("Manual Data")

    m = st.sidebar.number_input(
        "Number of demand points ($m$)",
        min_value=1, 
        step=1,
        value=st.session_state.m_val_mm_sf,
        key="m_input_mm_sf"
    )

    dims_changed = (m != st.session_state.m_val_mm_sf)
    st.session_state.m_val_mm_sf = m

    # If dimensions changed manually, or dataframe doesn't exist, rebuild a blank one
    if dims_changed or "mm_sf_df" not in st.session_state or len(st.session_state.mm_sf_df) != m:
        st.session_state.mm_sf_df = pd.DataFrame({
            "a (x-coord)": [float(i + 1) for i in range(m)],
            "b (y-coord)": [float(i + 1) for i in range(m)]
        }, index=[f"DP{i+1}" for i in range(m)])

    # The data_editor now holds user manual data OR the uploaded CSV / Example data
    mm_sf_df = st.sidebar.data_editor(
        st.session_state.mm_sf_df,
        key="mm_sf_table",
        num_rows="fixed",
        use_container_width=True
    )

    if mm_sf_df.isnull().any().any():
        st.sidebar.error("Demand Point table contains empty cells.")
        return []

    # ==================================================
    # 3. RETURN DATA
    # ==================================================
    # Extract final data from DataFrame to list of tuples: [(a, b), ...]
    return list(mm_sf_df.itertuples(index=False, name=None))


def show_minimax_sfl(data):
    # --------------------------------------------------
    # Page Title & Intro
    # --------------------------------------------------
    st.title("Minimax/Maximin Single Facility Location Problem")

    # --------------------------------------------------
    # Tabs
    # --------------------------------------------------
    tab1, tab3 = st.tabs(
        [
            "Minimax Rectilinear (Solved Equivalent LPP)",
            #"Maximin Rectilinear (Solved Equivalent LPP)",
            "Minimax Equilidean (Elzinga-Hearn Algo.)"
        ]
    )
    # --------------------------------------------------
    # TAB 1: Rectilinear (Solved Equivalent LPP)
    # --------------------------------------------------
    with tab1:
        # --------------------------------------------------
        # Solve the problem (once)
        # --------------------------------------------------
        result = slv.solve_minimax_sfl_L1(data)
    
        Z = result["Z"]
        (x1, y1), (x2, y2) = result["segment"]
        c1, c2, c3, c4, c5 = result["c_vals"]
        st.subheader("Single-Facility Rectilinear Minimax Location Problem")
        st.markdown(
            """
            The **Minimax Single Facility Location Problem** determines the location
            of a new facility such that the **maximum distance** to any demand point
            is as small as possible.
    
            This model is especially suitable when **worst-case performance**
            is more important than average performance, such as:
            - Emergency services (hospitals, fire stations)
            - Disaster response centers
            - Critical service facilities
            """
        )
        st.latex(
            r"""
            \min f(x,y)
            =
            \max_{1 \le i \le m}
            \left\{
            |x-a_i| + |y-b_i|
            \right\}
            """
        )
        
        st.latex(
            r"""
            \begin{aligned}
            \min \quad \; & Z \\
            \text{s.t. }\quad &
            |x-a_i| + |y-b_i| \le Z, \quad \forall i
            \end{aligned}
            """
        )
        st.markdown(
            """
            This objective ensures that the **farthest demand point**
            from the new facility is as close as possible.
            """
        )
        # -----------------------------------------
        # Checkbox to show/hide data
        # -----------------------------------------
        show_data = st.checkbox("Show Data")
        
        if show_data:
            df_data = pd.DataFrame(
                data,
                columns=["a (x-coordinate)", "b (y-coordinate)"]
            )
        
            st.markdown("### Existing Facility Data")
            st.dataframe(df_data, hide_index=True)
            
        # --------------------------------------------------
        # Two-column layout
        # --------------------------------------------------
        left_col, right_col = st.columns([1.8, 1.2])
    
        # ==================================================
        # LEFT COLUMN — THEORY + RESULTS
        # ==================================================
        with left_col:
    
            # --------------------------------------------------
            # Mathematical Model
            # --------------------------------------------------
            st.subheader("Mathematical Formulation")
    
            st.markdown(
                """
                To obtain the minimax solution we let.
                """
            )
    
            st.latex(
                rf"""
                \begin{{aligned}}
                c_1 &= \min (a_i + b_i) \quad = {c1:.3f} \\
                c_2 &= \max (a_i + b_i) \quad = {c2:.3f} \\
                c_3 &= \min (-a_i + b_i) \quad = {c3:.3f} \\
                c_4 &= \max (-a_i + b_i) \quad = {c4:.3f} \\
                c_5 &= \max (c_2 - c_1,\; c_4 - c_3) \quad = {c5:.3f}
                \end{{aligned}}
                """
            )

            st.markdown(
                """
                Optimum solutions to the minimax location problem are all points on a line segment connecting the Points:
                """
            )
            st.latex(
                r"""
                \begin{aligned}
                (x_1^*,y_1^*)
                &= \frac{1}{2}
                \left(
                c_1 - c_3,\;
                c_1 + c_3 + c_5
                \right) \\
                (x_2^*,y_2^*)
                &= \frac{1}{2}
                \left(
                c_2 - c_4,\;
                c_2 + c_4 - c_5
                \right)
                \end{aligned}
                """
            )
            st.markdown("**Point 1**")
            
            st.latex(
                rf"""
                (x_1^*,y_1^*)
                =
                \tfrac{{1}}{{2}}
                ({c1:.3f}-{c3:.3f},\;
                {c1:.3f}+{c3:.3f}+{c5:.3f})
                =
                ({x1:.3f}, {y1:.3f})
                """
            )
            
            st.markdown("**Point 2**")
            
            st.latex(
                rf"""
                (x_2^*,y_2^*)
                =
                \tfrac{{1}}{{2}}
                ({c2:.3f}-{c4:.3f},\;
                {c2:.3f}+{c4:.3f}-{c5:.3f})
                =
                ({x2:.3f}, {y2:.3f})
                """
            )

            st.subheader("Equation of the Optimal Location Line")
            
            st.markdown(
                """
                At the optimum, the maximum distance constraint becomes **active**.
                Therefore, all optimal solutions satisfy a **single linear equation**.
                """
            )
            if c2 - c1 >= c4 - c3:
                st.markdown("Since $c_2 - c_1 \\ge c_4 - c_3$, the active constraint is:")
                st.latex(
                    rf"""
                    x + y = \frac{{c_1 + c_2}}{{2}}
                    = \frac{{{c1:.3f} + {c2:.3f}}}{{2}}
                    = {(c1 + c2)/2:.3f}
                    """
                )
            else:
                st.markdown("Since $c_4 - c_3 > c_2 - c_1$, the active constraint is:")
                st.latex(
                    rf"""
                    -x + y = \frac{{c_3 + c_4}}{{2}}
                    = \frac{{{c3:.3f} + {c4:.3f}}}{{2}}
                    = {(c3 + c4)/2:.3f}
                    """
                )





        # ==================================================
        # RIGHT COLUMN — PLOT
        # ==================================================
        with right_col:
            # --------------------------------------------------
            # Optimal Objective Value
            # --------------------------------------------------
            st.subheader("Optimal Objective Value")
    
            st.latex(
                rf"""
                Z^* = \frac{{1}}{{2}}
                \max \left\{{ c_2 - c_1,\; c_4 - c_3 \right\}}
                = \frac{{1}}{{2}} ({c5:.3f})
                = {Z:.3f}
                """
            )
            st.latex(
                rf"""
                Point 1 =
                ({x1:.3f}, {y1:.3f})\\
                Point 2 = ({x2:.3f}, {y2:.3f})
                """
            )
            
            st.subheader("Graphical Interpretation")
    
            fig = slv.plot_minimax_solution_L1(data, result)
            st.pyplot(fig)
    
            st.caption(
                "Blue points represent demand locations. "
                "The red line segment represents the complete set of optimal solutions."
            )
    
    # # --------------------------------------------------
    # # TAB 2: Maximin Rectilinear (Geometric Reformulation)
    # # --------------------------------------------------
    # with tab2:
    
    #     st.subheader("Single-Facility Rectilinear Maximin Location Problem")
    
    #     # --------------------------------------------------
    #     # Mathematical model
    #     # --------------------------------------------------
    #     st.latex(
    #         r"""
    #         \max f(x,y)
    #         =
    #         \min_{1 \le i \le m}
    #         \left\{
    #         |x-a_i| + |y-b_i|
    #         \right\}
    #         """
    #     )
    
    #     st.latex(
    #         r"""
    #         \begin{aligned}
    #         \max \quad & Z \\
    #         \text{s.t.} \quad
    #         & |x-a_i| + |y-b_i| \ge Z, \quad \forall i
    #         \end{aligned}
    #         """
    #     )
    
    #     st.markdown(
    #         """
    #         This problem models the location of an **obnoxious facility**
    #         (e.g., waste dump, polluting plant), where the goal is to
    #         **maximize the distance to the nearest demand point**.
    #         """
    #     )
    
    #     # --------------------------------------------------
    #     # Solve using correct geometric maximin solver
    #     # --------------------------------------------------
    #     result = slv.solve_maximin_sfl_L1(data)
    
    #     Z_star = result["Z"]                 # true distance (>= 0)
    #     x_star, y_star = result["point"]
    #     c1, c2, c3, c4 = result["c_vals"]
    
    #     # --------------------------------------------------
    #     # Two-column layout
    #     # --------------------------------------------------
    #     left_col, right_col = st.columns([1.8, 1.2])
    
    #     # ==================================================
    #     # LEFT COLUMN — THEORY
    #     # ==================================================
    #     with left_col:
    
    #         st.subheader("Geometric Reformulation")
    
    #         st.markdown(
    #             """
    #             Define the following constants based on the demand points:
    #             """
    #         )
    
    #         st.latex(
    #             rf"""
    #             \begin{{aligned}}
    #             c_1 &= \max (a_i + b_i) = {c1:.3f} \\
    #             c_2 &= \max (a_i - b_i) = {c2:.3f} \\
    #             c_3 &= \max (-a_i + b_i) = {c3:.3f} \\
    #             c_4 &= \max (-a_i - b_i) = {c4:.3f}
    #             \end{{aligned}}
    #             """
    #         )
    
    #         st.markdown(
    #             """
    #             Using these constants, define the **geometric slack function**:
    #             """
    #         )
    
    #         st.latex(
    #             r"""
    #             Z_{\text{geom}}(x,y)
    #             =
    #             \min
    #             \left\{
    #             x+y-c_1,\;
    #             x-y-c_2,\;
    #             -x+y-c_3,\;
    #             -x-y-c_4
    #             \right\}
    #             """
    #         )
    
    #         st.markdown(
    #             """
    #             The **true maximin objective value** is:
    #             """
    #         )
    
    #         st.latex(
    #             r"""
    #             f(x,y) = \max\{0,\; Z_{\text{geom}}(x,y)\}
    #             """
    #         )
    
    #         st.markdown(
    #             """
    #             The optimal solution is obtained by choosing \\((x,y)\\)
    #             so that the **minimum of the four linear expressions is maximized**.
    #             Unlike the minimax problem, **no universal closed-form solution exists**.
    #             """
    #         )
    
    #     # ==================================================
    #     # RIGHT COLUMN — RESULTS & INTERPRETATION
    #     # ==================================================
    #     with right_col:
    
    #         st.subheader("Numerical Solution")
    
    #         st.latex(
    #             rf"""
    #             Z^* = \min_i
    #             \left\{{ |x^*-a_i| + |y^*-b_i| \right\}}
    #             = {Z_star:.3f}
    #             """
    #         )
    
    #         st.latex(
    #             rf"""
    #             (x^*,y^*) = ({x_star:.3f},\; {y_star:.3f})
    #             """
    #         )
    
    #         st.markdown(
    #             """
    #             **Key observations:**
    #             - The maximin objective value is **never negative**
    #             - The problem is **always feasible**
    #             - Without explicit bounds, the problem may be **unbounded**
    #             - The solution shown is a **representative optimal point**
    #             """
    #         )
    
    #         st.subheader("Graphical Interpretation")
    
    #         fig = slv.plot_minimax_solution_L1(
    #             data,
    #             {"segment": [(x_star, y_star), (x_star, y_star)]}
    #         )
    #         st.pyplot(fig)
    
    #         st.caption(
    #             "The plotted point represents a location that maximizes the distance "
    #             "to the nearest demand point under rectilinear distance."
    #         )
    # --------------------------------------------------
    # TAB 3: Minimax Euclidean (Elzinga-Hearn Algo.)
    # --------------------------------------------------
    with tab3:
        st.subheader("Minimax Euclidean Distance (L2) – Elzinga–Hearn Algorithm")
    
        st.latex(
            r"""
            \min_{x,y} \; \max_{i}
            \sqrt{(x-a_i)^2 + (y-b_i)^2}
            """
        )
    
        st.markdown(
            """
            The **Elzinga–Hearn Algorithm** solves the minimax Euclidean
            single-facility location problem by finding the
            **minimum enclosing circle** of the demand points.
    
            The optimal facility location is the **center of this circle**,
            and the minimax objective value is its **radius**.
            """
        )
    
        st.markdown("**Key theoretical properties:**")
        st.markdown(
            """
            - The optimal circle is defined by **either two or three points**
            - Two points → diameter case  
            - Three points → circumcircle of an acute triangle  
            - The solution is **unique** for Euclidean distance
            """
        )
    
        if not data:
            st.warning("Please provide demand point data.")
            return
    
        # --------------------------------------------------
        # Solve (STATIC DATA)
        # --------------------------------------------------
        result = slv.solve_minimax_sfl_L2_elzinga_hearn(data)
    
        # --------------------------------------------------
        # Two-column layout
        # --------------------------------------------------
        left_col, right_col = st.columns([1.8, 1.2])
    
        # ==================================================
        # LEFT COLUMN — DATA TABLE
        # ==================================================
        with left_col:
            df_data = pd.DataFrame(
                data,
                columns=["x-coordinate", "y-coordinate"]
            )
            df_data.insert(0, "Point", [f"P{i+1}" for i in range(len(df_data))])
    
            st.markdown("### Demand Points")
            st.dataframe(df_data, hide_index=True)
            st.markdown("### Defining Points in Elzinga–Hearn Algorithm")

            st.latex(
                r"""
                
                \quad
                \text{A demand point } (a_i,b_i) \text{ is called a defining point if}
                """
            )
            
            st.latex(
                r"""
                \sqrt{(x^*-a_i)^2 + (y^*-b_i)^2} = Z^*
                """
            )
                       
            st.markdown(
                """
                **Interpretation:**
                - Defining points lie **exactly on the boundary** of the minimum enclosing circle  
                - All other demand points lie **strictly inside** the circle  
                - Only defining points **determine the optimal solution**
                """
            )
            
            st.markdown("### Number of Defining Points")
            
            st.markdown(
                """
                - **Two defining points**  
                  → Optimal facility is the **midpoint** of these two points  
                  → This is the *diameter case*
            
                - **Three defining points**  
                  → These points form an **acute triangle**  
                  → Optimal facility is the **circumcenter** of the triangle  
            
                In two-dimensional space, the minimum enclosing circle is determined by
                **at most three defining points**.
                """
            )
            
            st.caption(
                "Defining points are the active constraints of the minimax Euclidean "
                "facility location problem and uniquely characterize the optimal solution."
            )

            # --------------------------------------------------
            # Defining points table
            # --------------------------------------------------
            tol = 1e-6
            defining_points = []
            
            for i, (x, y) in enumerate(data):
                dist = ((x - result["x"])**2 + (y - result["y"])**2) ** 0.5
                if abs(dist - result["Z"]) <= tol:
                    defining_points.append((f"P{i+1}", x, y, dist))
            
            if defining_points:
                df_def = pd.DataFrame(
                    defining_points,
                    columns=[
                        "Point",
                        "x-coordinate",
                        "y-coordinate",
                        "Distance to optimal facility"
                    ]
                )
            
                st.markdown("### Defining Points")
                st.dataframe(df_def, hide_index=True)
            
                st.caption(
                    "Defining points lie exactly on the boundary of the minimum enclosing circle "
                    "and determine the optimal solution."
                )
            else:
                st.info("No defining points detected (numerical tolerance issue).")
    
        # ==================================================
        # RIGHT COLUMN — PLOT + RESULTS
        # ==================================================
        with right_col:
            # st.markdown(
            #     "<div style='height:200px'></div>",
            #     unsafe_allow_html=True
            # )
    
            st.subheader("Graphical Interpretation")
    
            fig = slv.plot_minimax_solution_L2(data, result)
            st.pyplot(fig)
    
            st.subheader("Optimal Solution")
    
            st.markdown(
                f"""
                **Optimal location:** ({result['x']:.4f}, {result['y']:.4f})  
                **Optimal objective value (Z):** {result['Z']:.4f}
                """
            )
    
            st.caption(
                "The circle represents the minimum enclosing circle of the demand points. "
                "Its center gives the minimax Euclidean facility location."
            )
