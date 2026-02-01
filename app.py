# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches


import ui_pages as pg

st.set_page_config(
        page_title="Facility Location App",
        layout="wide"
    )

# -------------------------------
# Navigation state
# -------------------------------
pages = [
    "Home",
    "Minisum Single Facility Location",
    "Minimax/Maximin SFL",
    "Minisum Multiple Facility Location",
    "Minimax Multiple Facility Location",
    "Location Allocation Problems",
    "References",
]

if "active_page" not in st.session_state:
    st.session_state.active_page = "Home"

# ---- APPLY ONE-TIME NAV JUMP (THIS IS THE KEY) ----
if "_nav_jump" in st.session_state:
    st.session_state.active_page = st.session_state._nav_jump
    del st.session_state._nav_jump  # one-time use

# ---- NOW create the navigation widget ----
page = st.segmented_control(
    "",
    pages,
    selection_mode="single",
    key="active_page",
)

if page == "Home":
    pg.home.show_home()

elif page == "Minisum Single Facility Location":
    data = pg.minisum_sfl.build_inputs()
    pg.minisum_sfl.show_minisum_sfl(data)

elif page == "Minimax/Maximin SFL":
    data = pg.minimax_sfl.build_inputs()
    pg.minimax_sfl.show_minimax_sfl(data)

elif page == "Minisum Multiple Facility Location":
    pg.minisum_mfl.show_minisum_mfl()

elif page == "Minimax Multiple Facility Location":
    pg.minimax_mfl.show_minimax_mfl()

elif page == "Location Allocation Problems":
    pg.lap.show_lap()
    
elif page == "References":
    pg.references.show_references()

# --------------------------------------------------
# Footer
# ---------------------------------------
st.markdown("---")
st.caption("Facility Layout and Design App by — Ankit Behera")
st.caption(
    "Course: Facility Layout and Design | "
    "Department of Industrial & Systems Engineering | "
    "Indian Institute of Technology Kharagpur"
)
