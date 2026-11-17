import streamlit as st
import pandas as pd
import plotly.express as px

from analysis import analyze_journey

# PAGE CONFIGURATION
st.set_page_config(
    page_title="DriveSense AI",
    page_icon="🏎️",
    layout="wide"
)

# CUSTOM STYLING
st.markdown("""
<style>
.main {
    background-color: #ffffff;
}
h1 {
    color: #087FCA;
    font-weight: 800;
}
.metric-card {
    background-color: #f4f7fa;
    padding: 18px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)


# HEADER
st.title("DriveSense AI")
st.markdown(
    "### Where Driving Dynamics Meet Intelligent Performance"
)
st.markdown(
    "Explore driving behaviour, road dynamics and performance insights "
    "through machine learning."
)

st.divider()


# SIDEBAR
st.sidebar.title("DRIVESENSE AI")
st.sidebar.markdown("Performance Analysis System")

drive_mode = st.sidebar.selectbox(
    "Drive Mode",
    ["4H_ALLGRIP", "2H"]
)

st.sidebar.markdown("---")
st.sidebar.caption("Machine Learning Powered Driving Analysis")


# DATA UPLOAD
st.subheader("Journey Analysis")

uploaded_file = st.file_uploader(
    "Upload Journey Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        journey_df = pd.read_csv(uploaded_file)

        st.success("Journey data loaded successfully.")

        st.subheader("Journey Data Preview")
        st.dataframe(journey_df.head(10), use_container_width=True)

        st.write(f"Total Records: {len(journey_df)}")

        if st.button("Analyse Driving Performance", type="primary"):

            with st.spinner("Analysing driving dynamics..."):

                results = analyze_journey(journey_df)

            st.success("Analysis completed successfully!")

            # PERFORMANCE METRICS
            st.subheader("Performance Overview")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Data Points",
                len(results)
            )

            event_counts = results["predicted_event"].value_counts()

            col2.metric(
                "Driving Events",
                len(event_counts)
            )

            col3.metric(
                "Detected Events",
                int(
                    (results["predicted_event"] != "Normal driving").sum()
                )
            )

            st.divider()

            # EVENT DISTRIBUTION
            st.subheader("Driving Event Distribution")

            event_data = (
                results["predicted_event"]
                .value_counts()
                .reset_index()
            )

            event_data.columns = ["Event", "Count"]

            fig = px.bar(
                event_data,
                x="Event",
                y="Count",
                color="Event",
                title="Detected Driving Events"
            )

            st.plotly_chart(fig, use_container_width=True)

            # DRIVING DYNAMICS
            st.subheader("Driving Dynamics")

            available_metrics = [
                col for col in [
                    "speed",
                    "longitudinal_acc",
                    "lateral_acc",
                    "road_curvature"
                ]
                if col in results.columns
            ]

            if available_metrics:

                selected_metric = st.selectbox(
                    "Select Performance Parameter",
                    available_metrics
                )

                fig = px.line(
                    results,
                    y=selected_metric,
                    title=f"{selected_metric.replace('_', ' ').title()} Analysis"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            # CONTEXTUAL INSIGHTS
            st.subheader("Driving Insights")

            for event, count in event_counts.items():

                st.markdown(f"**{event}:** {count} occurrences")

            st.divider()

            st.subheader("Detailed Analysis")

            st.dataframe(
                results,
                use_container_width=True
            )

            # DOWNLOAD RESULTS
            csv = results.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Analysis Report",
                data=csv,
                file_name="drivesense_analysis.csv",
                mime="text/csv"
            )

    except Exception as e:

        st.error(f"Analysis failed: {e}")

else:

    st.info(
        "Upload a simulated journey CSV file to begin performance analysis."
    )

    st.markdown("""
    **DriveSense AI Features**

    - Driving behaviour classification
    - Acceleration and braking analysis
    - Sharp-turn detection
    - Road curvature analysis
    - Machine learning based performance insights
    """)

st.divider()

st.caption(
    "DriveSense AI | Intelligent Driving Performance Analysis"
)
