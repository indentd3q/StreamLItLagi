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

# Streamlit App
st.title("Dataset Segregation by Race")

# File Upload Section
st.header("Upload Files")
phenotype_file = st.file_uploader("Upload Phenotype TSV File", type=["tsv"])
counts_file = st.file_uploader("Upload Counts CSV File", type=["csv"])

# Race Selection and Output Settings
st.header("Output Settings")
races = ['white', 'black or african american', 'not reported', 'asian', 'american indian or alaska native']
selected_races = st.multiselect("Select Races to Separate", races)
target_dir = st.text_input("Enter Target Directory Path", value="./output")

# Process Button
if st.button("Process"):
    if phenotype_file and counts_file and selected_races:
        try:
            # Save uploaded files temporarily
            os.makedirs("temp", exist_ok=True)
            phenotype_path = os.path.join("temp", phenotype_file.name)
            counts_path = os.path.join("temp", counts_file.name)

            with open(phenotype_path, "wb") as f:
                f.write(phenotype_file.read())
            with open(counts_path, "wb") as f:
                f.write(counts_file.read())

            phenotype_data = pd.read_csv(phenotype_path, sep='\t')
            counts_data = pd.read_csv(counts_path, sep=',')

            os.makedirs(target_dir, exist_ok=True)

            for race in selected_races:
                # Separate by race
                race_file_path = seperateByRace(phenotype_data, target_dir, race, race)

                # Match DNA Samples
                output_file_path = matchingDNA(race_file_path, counts_data, target_dir, f"matched_{race}")
                st.success(f"Processed Race: {race}. Output saved at: {output_file_path}")

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload both files and select at least one race.")import streamlit as st
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

# Streamlit App
st.title("Dataset Segregation by Race")

# File Upload Section
st.header("Upload Files")
phenotype_file = st.file_uploader("Upload Phenotype TSV File", type=["tsv"])
counts_file = st.file_uploader("Upload Counts CSV File", type=["csv"])

# Race Selection and Output Settings
st.header("Output Settings")
races = ['white', 'black or african american', 'not reported', 'asian', 'american indian or alaska native']
selected_races = st.multiselect("Select Races to Separate", races)
target_dir = st.text_input("Enter Target Directory Path", value="./output")

# Process Button
if st.button("Process"):
    if phenotype_file and counts_file and selected_races:
        try:
            # Save uploaded files temporarily
            os.makedirs("temp", exist_ok=True)
            phenotype_path = os.path.join("temp", phenotype_file.name)
            counts_path = os.path.join("temp", counts_file.name)

            with open(phenotype_path, "wb") as f:
                f.write(phenotype_file.read())
            with open(counts_path, "wb") as f:
                f.write(counts_file.read())

            phenotype_data = pd.read_csv(phenotype_path, sep='\t')
            counts_data = pd.read_csv(counts_path, sep=',')

            os.makedirs(target_dir, exist_ok=True)

            for race in selected_races:
                # Separate by race
                race_file_path = seperateByRace(phenotype_data, target_dir, race, race)

                # Match DNA Samples
                output_file_path = matchingDNA(race_file_path, counts_data, target_dir, f"matched_{race}")
                st.success(f"Processed Race: {race}. Output saved at: {output_file_path}")

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload both files and select at least one race.")
