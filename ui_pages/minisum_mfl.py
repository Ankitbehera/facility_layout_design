import streamlit as st
import pandas as pd
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
    st.sidebar.header("Input Data")

    m = st.sidebar.number_input(
        "Number of existing facilities",
        min_value=1,
        step=1,
        value=3
    )

    n = st.sidebar.number_input(
        "Number of new facilities",
        min_value=1,
        step=1,
        value=2
    )

    st.sidebar.markdown("### Existing Facility Locations")

    existing = []
    for i in range(m):
        col1, col2 = st.sidebar.columns(2)
        a = col1.number_input(f"a{i+1}", value=float((i+1)*10), step=1.0)
        b = col2.number_input(f"b{i+1}", value=float((i+1)*5), step=1.0)
        existing.append((a, b))

    st.sidebar.markdown("### Weights: New → Existing (wᵢⱼ)")

    w_ji = []
    for j in range(n):
        row = []
        st.sidebar.markdown(f"**New Facility {j+1}**")
        for i in range(m):
            w = st.sidebar.number_input(
                f"w{j+1}{i+1}",
                min_value=0.0,
                value=1.0,
                step=1.0
            )
            row.append(w)
        w_ji.append(row)

    st.sidebar.markdown("### Interaction Weights Between New Facilities (vⱼₖ)")

    v_jk = [[0.0]*n for _ in range(n)]
    for j in range(n):
        for k in range(j+1, n):
            v = st.sidebar.number_input(
                f"v{j+1}{k+1}",
                min_value=0.0,
                value=1.0,
                step=1.0
            )
            v_jk[j][k] = v
            v_jk[k][j] = v

    return existing, w_ji, v_jk


# ------------------------------------------------------------------------------
# Page
# ------------------------------------------------------------------------------

def show_minisum_mfl():
    st.title("Minisum Multiple Facility Location Problem")

    st.markdown(
        """
        This module solves the **Minisum Multi-Facility Location Problem**
        using **Rectilinear (L1) distance** and the **Coordinate Descent Method**.

        The algorithm is based on IIT Kharagpur lecture notes and guarantees
        optimality when median conditions are satisfied.
        """
    )

    existing, w_ji, v_jk = build_inputs()

    if st.button("Solve Minisum MFL"):
        result = slv.solve_minisum_mfl(existing, w_ji, v_jk)

        st.subheader("Optimal Locations of New Facilities")

        df = pd.DataFrame(
            result["X_opt"],
            columns=["x*", "y*"],
            index=[f"NF{i+1}" for i in range(len(result["X_opt"]))]
        )

        st.dataframe(df)

        st.markdown(f"**Objective value:** `{result['obj']:.4f}`")

        with st.expander("Iteration History"):
            for it, X in enumerate(result["history"]):
                st.write(f"Iteration {it}: {X}")
