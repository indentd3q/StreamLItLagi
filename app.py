import streamlit as st
import pandas as pd
import os

def datasortrace(dataset_path, file1_path, file2_path, target, name):
    """Sort Dataset by Race"""
    # Read in the datasets
    dataset = pd.read_csv(dataset_path, encoding='utf-8')  # Use the parameter value
    file1 = pd.read_excel(file1_path)  # Use the parameter value
    file2 = pd.read_excel(file2_path)  # Use the parameter value

    colA1 = file1.iloc[:, 0].tolist()
    colA2 = file2.iloc[:, 0].tolist()

    # Select columns from the dataset
    newFile2 = dataset[[column for column in colA2 if column in dataset.columns]]
    newFile1 = pd.concat([dataset.iloc[:, 0], dataset[[column for column in colA1 if column in dataset.columns]]], axis=1)

    # Combine dataframes and save to target directory
    newFile = pd.concat([newFile1, newFile2], ignore_index=False, axis=1)

    # Ensure target directory exists
    os.makedirs(target, exist_ok=True)
    
    # Save the final dataframe to a CSV file
    output_file_path = os.path.join(target, name)
    newFile.to_csv(output_file_path, index=False)
    return output_file_path

# Streamlit App
st.title("Dataset Sorting by Race")

# Upload dataset
st.header("Upload Files")
dataset_file = st.file_uploader("Upload Dataset CSV File", type=["csv"])
file1_file = st.file_uploader("Upload File1 Excel File", type=["xlsx"])
file2_file = st.file_uploader("Upload File2 Excel File", type=["xlsx"])

# Input target directory and output file name
st.header("Output Settings")
target_dir = st.text_input("Enter Target Directory Path", value="./output")
output_filename = st.text_input("Enter Output File Name", value="sorted_dataset.csv")

# Process button
if st.button("Process"):
    if dataset_file and file1_file and file2_file and output_filename:
        try:
            # Save uploaded files temporarily
            dataset_path = os.path.join("temp", dataset_file.name)
            file1_path = os.path.join("temp", file1_file.name)
            file2_path = os.path.join("temp", file2_file.name)

            os.makedirs("temp", exist_ok=True)
            with open(dataset_path, "wb") as f:
                f.write(dataset_file.read())
            with open(file1_path, "wb") as f:
                f.write(file1_file.read())
            with open(file2_path, "wb") as f:
                f.write(file2_file.read())

            # Call the function
            output_path = datasortrace(dataset_path, file1_path, file2_path, target_dir, output_filename)
            st.success(f"File created successfully: {output_path}")

            # Optionally, allow the user to download the result
            with open(output_path, "rb") as f:
                st.download_button(
                    label="Download Sorted Dataset",
                    data=f,
                    file_name=output_filename,
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload all required files and provide a valid output filename.")
