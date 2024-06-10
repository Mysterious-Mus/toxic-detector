import sys
sys.path.append('./experiments')

import pandas as pd
import shutil
import os

# dataset construction entrance method
def sample_data(generate_new: bool, prepared_data_folder: str, **kwargs):
    output_folder = os.path.join(kwargs['experiment_folder'], 'data') # experiment_folder should stay in kwargs
    os.makedirs(output_folder, exist_ok=True)

    if generate_new:
        sample_data_mixing(**kwargs)
    else:
        def safe_copy(src, dst):
            """
            Copies a file from src to dst. If the resolved absolute paths of src and dst are the same, the copy operation is skipped.
            
            :param src: Source file path.
            :param dst: Destination file path.
            """
            # Resolve the absolute paths to handle cases with relative paths or symbolic links
            abs_src = os.path.abspath(src)
            abs_dst = os.path.abspath(dst)
            
            # Check if the source and destination are the same
            if abs_src != abs_dst:
                shutil.copy(abs_src, abs_dst)
                print(f"Copied from {src} to {dst}.")
            else:
                print(f"Skipping copy as source and destination are the same: {src}")
        # Copy from the folder
        if prepared_data_folder is not None:
            safe_copy(f'{prepared_data_folder}/train_data.csv', f'{output_folder}/train_data.csv')
            safe_copy(f'{prepared_data_folder}/test_data.csv', f'{output_folder}/test_data.csv')

def sample_data_mixing(experiment_folder: str,
                length_limit: int, benign_dataset_path: str,
                benign_sample_train: int, benign_sample_test: int, malicious_dataset_path: str,
                malicious_dataset_selection: dict, malicious_sample_train: int, malicious_sample_test: int,
                benign_prompt_column_name: str, malicious_prompt_column_name: str, **kwargs) -> None:
    output_folder = os.path.join(experiment_folder, 'data')
    # Load the benign dataset
    benign_df = pd.read_csv(benign_dataset_path)

    # discard the data if too long
    benign_df = benign_df[benign_df[benign_prompt_column_name].apply(lambda x: len(x) <= length_limit)]

    # Sample from the benign dataset
    benign_train = benign_df.sample(min(len(benign_df), benign_sample_train))
    benign_test = benign_df.drop(benign_train.index).sample(min(len(benign_df) - len(benign_train), benign_sample_test))

    # Load the malicious dataset
    malicious_df = pd.read_csv(malicious_dataset_path)

    if malicious_dataset_selection is not None:
        # Select the rows that match the specified category
        for key, val in malicious_dataset_selection.items():
            malicious_df = malicious_df[malicious_df[key] == val]

    # discard the data if too long
    malicious_df = malicious_df[malicious_df[malicious_prompt_column_name].apply(lambda x: len(x) <= length_limit)]

    # Sample from the malicious dataset
    malicious_train = malicious_df.sample(min(len(malicious_df), malicious_sample_train))
    malicious_test = malicious_df.drop(malicious_train.index).sample(min(len(malicious_df) - len(malicious_train), malicious_sample_test))

    # Drop other columns and rename the prompt column
    benign_train = benign_train[[benign_prompt_column_name]].rename(columns={benign_prompt_column_name: 'prompt'}).assign(label='benign')
    benign_test = benign_test[[benign_prompt_column_name]].rename(columns={benign_prompt_column_name: 'prompt'}).assign(label='benign')

    malicious_train = malicious_train[[malicious_prompt_column_name]].rename(columns={malicious_prompt_column_name: 'prompt'}).assign(label='malicious')
    malicious_test = malicious_test[[malicious_prompt_column_name]].rename(columns={malicious_prompt_column_name: 'prompt'}).assign(label='malicious')

    # Concatenate the train and test datasets
    train_df = pd.concat([benign_train, malicious_train])
    test_df = pd.concat([benign_test, malicious_test])

    # Save the train and test datasets
    train_df.to_csv(f'{output_folder}/train_data.csv', index=False)
    test_df.to_csv(f'{output_folder}/test_data.csv', index=False)
