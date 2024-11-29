import streamlit as st
import pandas as pd
import os

def seperateByRace(file, target, name, race):
    """Separates Data by Race"""
    racer = file[file["race.demographic"].str.contains(race, case=False, na=False)]
    output_file_path = os.path.join(target, f"{name}.csv")
    racer.to_csv(output_file_path, index=False)
    return output_file_path

def matchingDNA(race, counts, target, name):
    """Matches Sample ID from phenotypes to counts"""
    phenotypeData = pd.read_csv(race)
    colA1 = phenotypeData.iloc[:, 0]
    newFile = counts[['Ensembl_ID'] + [col for col in colA1 if col in counts.columns[1:]]]
    output_file_path = os.path.join(target, f"{name}.csv")
    newFile.to_csv(output_file_path, index=False)
    return output_file_path

# Streamlit App Configuration
st.set_page_config(
    page_title="Race-Based Dataset Segregation",
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

# App Title with Subheader
st.title("🧬 Dataset Segregation by Race")
st.subheader("Separate and Match Genetic Data Across Racial Demographics")

# Create two columns for file uploaders
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 📄 Phenotype File")
    phenotype_file = st.file_uploader("Upload Phenotype CSV", type=["csv"], key="phenotype")

with col2:
    st.markdown("### 📊 Counts File")
    counts_file = st.file_uploader("Upload Counts CSV", type=["csv"], key="counts")

# Race Selection with Info
st.markdown("### 🌍 Race Selection")
st.info("Choose one or more racial demographics to process")

# Improved Race Selection
races = ['white', 'black or african american', 'not reported', 'asian', 'american indian or alaska native']
selected_races = st.multiselect(
    "Select Races to Separate", 
    races, 
    help="Select the racial demographics you want to segregate and analyze"
)

# Process Section with Expander
with st.expander("⚙️ Process Data"):
    if st.button("Process Datasets", key="process_button"):
        if phenotype_file and counts_file and selected_races:
            try:
                # Existing processing logic remains the same
                os.makedirs("temp", exist_ok=True)
                phenotype_path = os.path.join("temp", phenotype_file.name)
                counts_path = os.path.join("temp", counts_file.name)

                with open(phenotype_path, "wb") as f:
                    f.write(phenotype_file.read())  
                with open(counts_path, "wb") as f:
                    f.write(counts_file.read())

                phenotype_data = pd.read_csv(phenotype_path)
                counts_data = pd.read_csv(counts_path)

                os.makedirs("temp", exist_ok=True)

                # Create a container to display processing results
                results_container = st.container()

                with results_container:
                    st.markdown("### 🔍 Processing Results")
                    for race in selected_races:
                        # Existing race processing logic
                        race_file_path = seperateByRace(phenotype_data, "temp", race, race)
                        output_file_path = matchingDNA(race_file_path, counts_data, "temp", f"matched_{race}")
                        
                        # Improved result display
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.success(f"Processed Race: {race}")
                        with col2:
                            with open(output_file_path, "rb") as file:
                                st.download_button(
                                    label="Download", 
                                    data=file, 
                                    file_name=f"matched_{race}.csv", 
                                    mime="text/csv",
                                    key=f"download_{race}"
                                )

            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please upload both files and select at least one race.")