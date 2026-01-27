# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import solver as slv

# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(
    page_title="Minisum Single Facility Location",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Minisum Single Facility Location Problem")

st.markdown(
"""
This app solves the **Minisum Single Facility Location Problem** using:
- Rectilinear (L1) distance — *Graphical Approch*
- Rectilinear (L1) distance — *Median Method*
- Euclidean (L2) distance — *Weiszfeld Method*
- Squared Euclidean (L2²) distance — *Centroid*
"""
)

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
# Sidebar: Data Input
# --------------------------------------------------
st.sidebar.header("Input Data")

m = st.sidebar.number_input(
    "Number of existing facilities",
    min_value=1,
    step=1,
    value=4
)

st.sidebar.markdown("### Facility Locations")

default_data = [
    (0, 20, 0.3),
    (0, 40, 0.2),
    (20, 0, 0.3),
    (40, 0, 0.2),
]

data = []

for i in range(m):
    col1, col2, col3 = st.sidebar.columns(3)

    if i < len(default_data):
        a0, b0, w0 = default_data[i]
    else:
        a0 = 10 * (i + 1)
        b0 = 10 * (i + 1)
        w0 = 1.0 / m

    a = col1.number_input(f"a{i+1}", key=f"a{i}", value=a0)
    b = col2.number_input(f"b{i+1}", key=f"b{i}", value=b0)
    w = col3.number_input(
        f"w{i+1}",
        key=f"w{i}",
        min_value=0.0,
        value=w0
    )

    data.append((a, b, w))

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Rectilinear (Graphical Approach)",
        "Rectilinear (Median Method)",
        "Iso-Contours",
        "Euclidean Models",
        "Comparison"
    ]
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

        st.caption(
            "Each point in this region attains the minimum value of the rectilinear objective function."
        )


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

    st.write("**Objective value:**", res_L1["obj"])

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
            (60, 0),
            (30, 30),
        ]

        contour_points = []

        for i in range(n):
            col1, col2 = st.columns(2)

            if i < len(default_contour_points):
                x0, y0 = default_contour_points[i]
            else:
                x0, y0 = 10 * (i + 1), 10 * (i + 1)

            x = col1.number_input(
                f"x{chr(65+i)}",
                key=f"cx{i}",
                value=x0
            )
            y = col2.number_input(
                f"y{chr(65+i)}",
                key=f"cy{i}",
                value=y0
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
    st.subheader("Euclidean Distance Models")

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

        st.write(
            "**Optimal location:**",
            (res_L2sq["x_opt"], res_L2sq["y_opt"])
        )
        st.write("**Objective value:**", res_L2sq["opt_val"])

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

        st.write(
            "**Optimal location:**",
            (res_L2["x_opt"], res_L2["y_opt"])
        )
        st.write("**Iterations:**", res_L2["iterations"])
        st.write("**Converged:**", res_L2["converged"])

        obj_val_L2 = slv.obj_L2(
            res_L2["x_opt"],
            res_L2["y_opt"],
            data
        )

        st.write("**Objective value:**", obj_val_L2)

        if show_iter and res_L2["history"] is not None:
            hist_df = pd.DataFrame(
                res_L2["history"],
                columns=["Iteration", "x", "y"]
            )
            st.dataframe(hist_df, hide_index=True)

# --------------------------------------------------
# TAB 5: Comparison
# --------------------------------------------------
with tab5:
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
# Footer
# --------------------------------------------------
st.markdown("---")
st.caption("Minisum Single Facility Location App — Streamlit")
