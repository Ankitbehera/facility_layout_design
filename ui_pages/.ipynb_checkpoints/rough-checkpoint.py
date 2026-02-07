# --------------------------------------------------
# TAB 3: Minimax Euclidean (Elzinga-Hearn Algo.) without Interaction
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


# --------------------------------------------------
    # TAB 3: Minimax Euclidean (Elzinga-Hearn Algo.)
    # --------------------------------------------------
    with tab3:
        st.subheader("Minimax Euclidean Distance (L2)")
        st.markdown("### Elzinga–Hearn Algorithm")
    
        # --------------------------------------------------
        # Initialize / sync interactive points
        # --------------------------------------------------
        if "euclid_points" not in st.session_state:
            st.session_state.euclid_points = list(data)
    
        if "base_data_snapshot" not in st.session_state:
            st.session_state.base_data_snapshot = list(data)
    
        # Reset interactive points ONLY if base data changed
        if list(data) != st.session_state.base_data_snapshot:
            st.session_state.euclid_points = list(data)
            st.session_state.base_data_snapshot = list(data)
    
        left_col, right_col = st.columns([1.5, 1.5])
    
        # ==================================================
        # LEFT — THEORY + RESULTS + TABLE
        # ==================================================
        with left_col:
            st.latex(
                r"""
                \min_{x,y} \; \max_{i}
                \sqrt{(x-a_i)^2 + (y-b_i)^2}
                """
            )
    
            st.markdown(
                """
                The **Elzinga–Hearn Algorithm** finds the
                **minimum enclosing circle** of the demand points.
                The center of this circle is the optimal facility location.
                """
            )
    
            # -----------------------------------------
            # Solve ONLY if points exist
            # -----------------------------------------
            if st.session_state.euclid_points:
                result = slv.solve_minimax_sfl_L2_elzinga_hearn(
                    st.session_state.euclid_points
                )
    
                st.subheader("Optimal Solution")
                st.markdown(
                    f"""
                    **Optimal location:** ({result['x']:.4f}, {result['y']:.4f})  
                    **Optimal objective value (Z):** {result['Z']:.4f}
                    """
                )
            else:
                result = None
                st.warning("No demand points available. Add points using the plot.")
    
            # -----------------------------------------
            # Checkbox to show/hide data
            # -----------------------------------------
            show_data = st.checkbox("Show Demand Point Table")
    
            if show_data:
                df_data = pd.DataFrame(
                    st.session_state.euclid_points,
                    columns=["x-coordinate", "y-coordinate"]
                )
                df_data.insert(0, "Point", [f"P{i+1}" for i in range(len(df_data))])
    
                st.markdown("### Demand Points")
                st.dataframe(df_data, hide_index=True)
    
        # ==================================================
        # RIGHT — INTERACTIVE PLOT
        # ==================================================
        with right_col:
            st.subheader("Graphical Interpretation")
    
            enable_click = st.checkbox("Enable interactive point input")
    
            if result is not None:
                fig = slv.plot_minimax_solution_L2_interactive(
                    st.session_state.euclid_points,
                    result,
                    show_labels=True
                )
            else:
                # Empty placeholder plot
                fig = slv.plot_minimax_solution_L2_interactive(
                    [],
                    {"x": 0, "y": 0, "Z": 0},
                    show_labels=False
                )
    
            if enable_click:
                clear = st.button("Clear demand points")
    
                if clear:
                    st.session_state.euclid_points = []
                    st.rerun()
    
                clicked = plotly_events(
                    fig,
                    click_event=True,
                    hover_event=False,
                    select_event=False,
                    override_height=450
                )
    
                if clicked:
                    st.session_state.euclid_points.append(
                        (clicked[0]["x"], clicked[0]["y"])
                    )
                    st.rerun()
            else:
                st.plotly_chart(fig, use_container_width=True)
