import streamlit as st


def _go_to(page_name: str):
    # Request a one-time navigation jump
    st.session_state._nav_jump = page_name
    st.rerun()


def show_home():
    st.title("Facility Layout and Design App")
    st.caption(
        "An interactive learning platform for Facility Location and Design — "
        "inspired by academic rigor and classroom-based understanding at IIT Kharagpur"
    )

    # --------------------------------------------------
    # Two-column layout
    # --------------------------------------------------
    left_col, right_col = st.columns([1.8, 0.6])

    # ============================
    # LEFT COLUMN: Description
    # ============================
    with left_col:
        st.markdown(
            """
            Facility Layout and Design is a core subject in **Operations Research**
            and **Industrial & Systems Engineering**, concerned with determining
            optimal locations for facilities such as warehouses, manufacturing plants,
            hospitals, fire stations, and service centers. These decisions directly
            impact system efficiency, service quality, and long-term operational cost,
            making them strategically important in both public and private sectors.

            ---
            ### Objective of the Application

            This application serves as a **one-stop educational platform** for
            understanding fundamental and advanced concepts in **Facility Layout
            and Design**.

            The app is primarily intended for **conceptual learning and academic
            exploration**. It is designed to work with **small datasets**, enabling 
            users to clearly interpret results and relate them
            to theoretical principles discussed in class.

            The application is organized into multiple modules, each corresponding
            to a specific category of facility location problems. Through these
            modules, users can:
            - Interactively input and modify problem data  
            - Apply classical analytical and algorithmic solution approaches  
            - Visualize optimal locations, solution regions, and cost structures  
            - Compare outcomes across different distance measures and models  

            This interactive experimentation helps bridge the gap between
            **theoretical formulations** and **practical interpretation**.

            ---
            ### Acknowledgement

            This application was developed based on concepts and material covered
            in the course **Facility Layout and Design** taught by **Prof. J. K. Jha**
            at **IIT Kharagpur**. The lecture notes and classroom discussions were
            used as the primary academic reference for this work.

            """
        )

    # ============================
    # RIGHT COLUMN: Navigation
    # ============================
    with right_col:
        st.subheader("Single-Facility Problems")

        if st.button("Minisum Single Facility Location", use_container_width=True):
            _go_to("Minisum Single Facility Location")

        if st.button("Minimax Single Facility Location", use_container_width=True):
            _go_to("Minimax Single Facility Location")

        st.markdown("---")

        st.subheader("Multi-Facility & Allocation")

        if st.button("Minisum Multiple Facility Location", use_container_width=True):
            _go_to("Minisum Multiple Facility Location")

        if st.button("Location Allocation Problems", use_container_width=True):
            _go_to("Location Allocation Problems")

        st.markdown("---")

        if st.button("References", use_container_width=True):
            _go_to("References")
