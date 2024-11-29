import streamlit as st
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# Function for separating phenotype data by race
def seperateByRace(file, race, race_col):
    """
    Searches the phenotype dataframe for the race.demographic column
    and returns a new dataframe with only the rows that contain the race.
    """
    return file[file[race_col].str.contains(race, case=False, na=False)]

# Function to match sample IDs from phenotype to counts data
def matchingDNA(phenotypeDATA, countsDATA, phenotype_sample_col, counts_sample_col):
    """
    Matches the race phenotype data to the counts data and returns a racial-dataframe with the matching data.
    """
    colA1 = phenotypeDATA[phenotype_sample_col]
    newFile = countsDATA[['Ensembl_ID'] + [col for col in colA1 if col in countsDATA.columns]]
    return pd.DataFrame(newFile)

# Streamlit App Configuration
st.set_page_config(
    page_title="Differential Gene Expression Analysis with PyDESeq2",
    page_icon=":dna:",
    layout="wide"
)

# Custom CSS for improved styling
st.markdown("""
    <style>
    .reportview-container {
        background-color: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background-color: #e6e9ef;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

# Create a navigation bar
page = st.selectbox("Select a Page", ["Home", "Dataset Segregation", "Differential Gene Expression Analysis", "About"])

# Home Page
if page == "Home":
    st.title("Welcome to the Genomic Data Analysis App")
    st.write("This app provides tools for segregating datasets by race and performing differential gene expression analysis. Use the navigation bar to explore the functionalities.")

# Dataset Segregation Page
elif page == "Dataset Segregation":
    st.title("🧬 Dataset Segregation by Race")
    st.subheader("Separate and Match Genetic Data Across Racial Demographics")
    # Include the Dataset Segregation code here (the one you provided earlier)
    # ... (insert your previous dataset segregation code here)

# Differential Gene Expression Analysis Page
elif page == "Differential Gene Expression Analysis":
    st.title("Differential Gene Expression Analysis with PyDESeq2")

    # File Uploads
    st.header("Input Files")
    phenotypeFile = st.file_uploader("Upload Phenotype Data (CSV)", type=["csv"])
    countsFile = st.file_uploader("Upload Counts Data (CSV)", type=["csv"])

    if phenotypeFile and countsFile:
        phenotype = pd.read_csv(phenotypeFile)
        counts = pd.read_csv(countsFile)
        preprocess_deg = counts.set_index("Ensembl_ID")

        # Configurable Columns
        st.header("Column Selections")
        phenotype_race_col = st.selectbox(
            "Select Race Column in Phenotype Data", phenotype.columns
        )
        phenotype_sample_col = st.selectbox(
            "Select Sample ID Column in Phenotype Data", phenotype.columns
        )
        counts_sample_col = st.selectbox(
            "Select Sample ID Column in Counts Data", counts.columns
        )

        # Process by Race
        st.header("Race Selection and Preprocessing")
        phenotype_unique_races = phenotype[phenotype_race_col].unique()
        selected_race = st.selectbox("Select Race to Analyze", phenotype_unique_races)

        phenotype_race_df = seperateByRace(phenotype, selected_race, phenotype_race_col)
        st.write("phenotype_race_df", phenotype_race_df)

        counts_race_df = matchingDNA(
            phenotype_race_df, counts, phenotype_sample_col, counts_sample_col
        )
        st.write("counts_race_df", counts_race_df)

        preprocessed_counts_data = counts_race_df.set_index("Ensembl_ID")
        preprocessed_counts_data = preprocessed_counts_data.fillna(0)
        preprocessed_counts_data = preprocessed_counts_data.round().astype(int)
        preprocessed_counts_data = preprocessed_counts_data[preprocessed_counts_data.sum(axis=1) > 0]
        preprocessed_counts_data = preprocessed_counts_data.T
        st.write("Preprocessed Counts Data", preprocessed_counts_data)

        # Create Metadata
        def create_metadata(counts_data):
            metadata = pd.DataFrame(index=counts_data.index)
            metadata['label'] = [
                'cancer' if '-01' in sample else 'normal' for sample in metadata.index
            ]
            return metadata

        metadata = create_metadata(preprocessed_counts_data)
        st.write("Metadata", metadata)

        # DEG Analysis
        st.header("DEG Analysis")
        contrast_choice = st.radio("Select Contrast for Analysis", ["cancer", "normal"])

        def initiate_deg(counts_data, metadata):
            dds = DeseqDataSet(
                counts=counts_data,
                metadata=metadata,
                design_factors="label"
            )
            dds.deseq2()
            stat_res = DeseqStats(dds, contrast=("label", "cancer", "normal"))
            stat_res.summary()
            return stat_res.results_df

        deg_stats_results = initiate_deg(preprocessed_counts_data, metadata)
        st.write("DEG Statistics Results", deg_stats_results)

        # Filter DEG Results
        st.header("DEG Filtering Options")
        cutoff_padj = st.number_input("Cutoff for padj", value=0.05)
        cutoff_log2FoldChange = st.number_input("Cutoff for log2FoldChange", value=0.5)
        cutoff_baseMean = st.number_input("Cutoff for baseMean", value=10)
        cutoff_pvalue = st.number_input("Cutoff for pvalue", value=0.05)
        cutoff_lfcSE = st.number_input("Cutoff for lfcSE", value=0.1)
        cutoff_stat = st.number_input("Cutoff for stat", value=0.1)

        def filter_deg_results(deg_results):
            deg_results = deg_results[deg_results['padj'] < cutoff_padj]
            deg_results = deg_results[deg_results['log2FoldChange'].abs() > cutoff_log2FoldChange]
            deg_results = deg_results[deg_results['baseMean'] > cutoff_baseMean]
            deg_results = deg_results[deg_results['pvalue'] < cutoff_pvalue]
            deg_results = deg_results[deg_results['lfcSE'] > cutoff_lfcSE]
            deg_results = deg_results[deg_results['stat'].abs() > cutoff_stat]
            return deg_results

        filtered_deg_results = filter_deg_results(deg_stats_results)
        st.write("Filtered DEG Results", filtered_deg_results)

        # Display DEG Genes
        st.write("DEG Genes", filtered_deg_results.index.to_list())

# About Page
elif page == "About":
    st.title("About This App")
    st.write("This app allows users to segregate and match genetic data based on racial demographics, as well as perform differential gene expression (DEG) analysis using the PyDESeq2 package.")
