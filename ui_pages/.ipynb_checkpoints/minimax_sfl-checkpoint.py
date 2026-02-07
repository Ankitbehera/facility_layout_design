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

    uploaded_file = st.sidebar.file_uploader(
        "Upload Minimax SFL data (CSV)",
        type=["csv"]
    )

    has_header = st.sidebar.checkbox("My data has headers", value=True)

    st.sidebar.markdown(
        """
        **CSV format:**  
        - Columns: `a, b`  
        - All values numeric  
        - No empty cells  
        """
    )

    # --------------------------------------------------
    # CASE 1: CSV UPLOADED → USE CSV DIRECTLY
    # --------------------------------------------------
    if uploaded_file is not None:
        try:
            if has_header:
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file, header=None)
                df.columns = ["a", "b"]

            # Validation
            if list(df.columns) != ["a", "b"]:
                st.sidebar.error("CSV must contain exactly columns: a, b")
                return []

            if df.isnull().any().any():
                st.sidebar.error("CSV contains empty cells")
                return []

            df = df.astype(float)
            st.sidebar.success("CSV loaded successfully")

            #  RETURN DATA DIRECTLY
            return list(df.itertuples(index=False, name=None))

        except Exception as e:
            st.sidebar.error(f"Invalid CSV file: {e}")
            return []

    # --------------------------------------------------
    # CASE 2: NO CSV → MANUAL INPUT
    # --------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.subheader("Manual Input")

    m = st.sidebar.number_input(
        "Number of demand points",
        min_value=1,
        step=1,
        value=5
    )

    st.sidebar.markdown("### Demand Point Locations")

    default_points = [
        (4.0, 3.0),
        (5.0, 11.0),
        (13.0, 13.0),
        (10.0, 6.0),
        (4.0, 6.0),
    ]

    data = []

    for i in range(m):
        col1, col2 = st.sidebar.columns(2)

        if i < len(default_points):
            a0, b0 = default_points[i]
        else:
            a0 = float(i + 1)
            b0 = float(i + 1)

        a = col1.number_input(
            f"a{i+1}",
            key=f"mm_a{i}",
            value=a0,
            step=0.1,
            format="%.2f"
        )

        b = col2.number_input(
            f"b{i+1}",
            key=f"mm_b{i}",
            value=b0,
            step=0.1,
            format="%.2f"
        )

        data.append((a, b))

    return data


def show_minimax_sfl(data):
    # --------------------------------------------------
    # Page Title & Intro
    # --------------------------------------------------
    st.title("Minimax/Maximin Single Facility Location Problem")

    # --------------------------------------------------
    # Tabs
    # --------------------------------------------------
    tab1, tab2, tab3 = st.tabs(
        [
            "Minimax Rectilinear (Solved Equivalent LPP)",
            "Maximin Rectilinear (Solved Equivalent LPP)",
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
                "Black points represent demand locations. "
                "The red line segment represents the complete set of optimal solutions."
            )
    
    # --------------------------------------------------
    # TAB 2: Maximin Rectilinear (Geometric Reformulation)
    # --------------------------------------------------
    with tab2:
    
        st.subheader("Single-Facility Rectilinear Maximin Location Problem")
    
        # --------------------------------------------------
        # Mathematical model
        # --------------------------------------------------
        st.latex(
            r"""
            \max f(x,y)
            =
            \min_{1 \le i \le m}
            \left\{
            |x-a_i| + |y-b_i|
            \right\}
            """
        )
    
        st.latex(
            r"""
            \begin{aligned}
            \max \quad & Z \\
            \text{s.t.} \quad
            & |x-a_i| + |y-b_i| \ge Z, \quad \forall i
            \end{aligned}
            """
        )
    
        st.markdown(
            """
            This problem models the location of an **obnoxious facility**
            (e.g., waste dump, polluting plant), where the goal is to
            **maximize the distance to the nearest demand point**.
            """
        )
    
        # --------------------------------------------------
        # Solve using correct geometric maximin solver
        # --------------------------------------------------
        result = slv.solve_maximin_sfl_L1(data)
    
        Z_star = result["Z"]                 # true distance (>= 0)
        x_star, y_star = result["point"]
        c1, c2, c3, c4 = result["c_vals"]
    
        # --------------------------------------------------
        # Two-column layout
        # --------------------------------------------------
        left_col, right_col = st.columns([1.8, 1.2])
    
        # ==================================================
        # LEFT COLUMN — THEORY
        # ==================================================
        with left_col:
    
            st.subheader("Geometric Reformulation")
    
            st.markdown(
                """
                Define the following constants based on the demand points:
                """
            )
    
            st.latex(
                rf"""
                \begin{{aligned}}
                c_1 &= \max (a_i + b_i) = {c1:.3f} \\
                c_2 &= \max (a_i - b_i) = {c2:.3f} \\
                c_3 &= \max (-a_i + b_i) = {c3:.3f} \\
                c_4 &= \max (-a_i - b_i) = {c4:.3f}
                \end{{aligned}}
                """
            )
    
            st.markdown(
                """
                Using these constants, define the **geometric slack function**:
                """
            )
    
            st.latex(
                r"""
                Z_{\text{geom}}(x,y)
                =
                \min
                \left\{
                x+y-c_1,\;
                x-y-c_2,\;
                -x+y-c_3,\;
                -x-y-c_4
                \right\}
                """
            )
    
            st.markdown(
                """
                The **true maximin objective value** is:
                """
            )
    
            st.latex(
                r"""
                f(x,y) = \max\{0,\; Z_{\text{geom}}(x,y)\}
                """
            )
    
            st.markdown(
                """
                The optimal solution is obtained by choosing \\((x,y)\\)
                so that the **minimum of the four linear expressions is maximized**.
                Unlike the minimax problem, **no universal closed-form solution exists**.
                """
            )
    
        # ==================================================
        # RIGHT COLUMN — RESULTS & INTERPRETATION
        # ==================================================
        with right_col:
    
            st.subheader("Numerical Solution")
    
            st.latex(
                rf"""
                Z^* = \min_i
                \left\{{ |x^*-a_i| + |y^*-b_i| \right\}}
                = {Z_star:.3f}
                """
            )
    
            st.latex(
                rf"""
                (x^*,y^*) = ({x_star:.3f},\; {y_star:.3f})
                """
            )
    
            st.markdown(
                """
                **Key observations:**
                - The maximin objective value is **never negative**
                - The problem is **always feasible**
                - Without explicit bounds, the problem may be **unbounded**
                - The solution shown is a **representative optimal point**
                """
            )
    
            st.subheader("Graphical Interpretation")
    
            fig = slv.plot_minimax_solution_L1(
                data,
                {"segment": [(x_star, y_star), (x_star, y_star)]}
            )
            st.pyplot(fig)
    
            st.caption(
                "The plotted point represents a location that maximizes the distance "
                "to the nearest demand point under rectilinear distance."
            )
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
