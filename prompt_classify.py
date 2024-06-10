"""
This is a script to test our malicious prompt categorization methodology.

For preparation, an experiment folder should be prepared, contain a `config.yaml` for all the experiment configs.

Step1: Load the dataset
the dataset is a mixed sample of normal and malicious prompts.
"""

import argparse
import yaml
import os
from src.classification.dataset_construction import sample_data
import pandas as pd
import time
import os
from sklearn.metrics import recall_score, confusion_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample data based on configuration in experiment folder.')
    parser.add_argument('experiment_folder', type=str, help='Path to the experiment folder.')

    args = parser.parse_args()

    experiment_folder = args.experiment_folder

    # Construct the path to the config file
    config_path = os.path.join(experiment_folder, 'config.yaml')

    # Load the config file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract the data config
    data_config = config['data']

    # Call the sample_data function
    sample_data(
        experiment_folder=experiment_folder,
        **data_config
    )

    # load the model
    from src.hookedLLM import HookedLLM # this is slow
    hooked_llm = HookedLLM(**config['model'])
    
    # load the train data
    train_data = pd.read_csv(f'{experiment_folder}/data/train_data.csv')

    positive_samples = train_data[train_data['label'] == 'malicious']['prompt'].tolist()
    negative_samples = train_data[train_data['label'] == 'benign']['prompt'].tolist()
    config['classifier']['positive_samples'] = positive_samples
    config['classifier']['negative_samples'] = negative_samples

    # configure training feature export
    if config['export_features']:
        config['classifier']['export_path'] = f'{experiment_folder}/features/train'

    # construct the classifying direction
    from src.classification.classifier import ActivationUsage
    start_time = time.time()
    classifier = ActivationUsage.from_config(config['classifier'], hooked_llm)
    train_time = time.time() - start_time

    # Assuming classifier and experiment_folder are defined elsewhere in your code

    # score every prompt in the training data
    start_time = time.time()
    train_data['score'], train_data['pred'] = classifier.batch_score_and_classify(train_data['prompt'].tolist())
    classify_train_time = time.time() - start_time
    print(f"Scoring and classifying training data took {time.time() - start_time:.2f} seconds")
    print(train_data)

    output_folder = f'{experiment_folder}/stats'
    os.makedirs(output_folder, exist_ok=True)

    # save the scored data
    start_time = time.time()
    train_data.to_csv(f'{output_folder}/train_data.csv', index=False)
    print(f"Saving training data took {time.time() - start_time:.2f} seconds")

    # score, classify and save the test data
    start_time = time.time()
    test_data = pd.read_csv(f'{experiment_folder}/data/test_data.csv')
    print(f"Loading test data took {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    test_data['score'], test_data['pred'] = classifier.batch_score_and_classify(test_data['prompt'].tolist())
    classify_test_time = time.time() - start_time
    print(f"Scoring and classifying test data took {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    test_data.to_csv(f'{output_folder}/test_data.csv', index=False)
    print(f"Saving test data took {time.time() - start_time:.2f} seconds")


    classification_threshold = classifier.classify_threshold

    # if we want features exported, get out another classifier using the test set as training set, so we can get the test data features exported
    if config['export_features']:
        config['classifier']['export_path'] = f'{experiment_folder}/features/test'
        positive_samples = test_data[test_data['label'] == 'malicious']['prompt'].tolist()
        negative_samples = test_data[test_data['label'] == 'benign']['prompt'].tolist()
        config['classifier']['positive_samples'] = positive_samples
        config['classifier']['negative_samples'] = negative_samples
        ActivationUsage.from_config(config['classifier'], hooked_llm)

    def plot_and_save(data: pd.DataFrame, savepath: str, statspath: str):
        # Calculate recall
        y_true = data['label']
        y_pred = data['pred']

        # Calculate false positive rate and precision
        tp, fn, fp, tn = confusion_matrix(y_true, y_pred, labels=['malicious', 'benign']).ravel()
        fpr = fp / (fp + tn)
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)

        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall)

        # Save statistics to CSV
        stats = pd.DataFrame({
            'Metric': ['Recall', 'False Positive Rate', 'Precision', 'F1 Score'],
            'Value': [recall, fpr, precision, f1]
        })
        stats.to_csv(statspath, index=False)

    start_time = time.time()
    plot_and_save(train_data, f'{output_folder}/score_train_histogram.png', f'{output_folder}/train_stats.csv')
    plot_and_save(test_data, f'{output_folder}/score_test_histogram.png', f'{output_folder}/test_stats.csv')
    print(f"Saving stats and plot took {time.time() - start_time:.2f} seconds")

    # save train and classify time as a csv
    pd.DataFrame({
        'Train Set Size': [len(train_data)],
        'Epochs': [config['classifier']['NNCfg']['training']['epochs']],
        'Train Time': [train_time],
        'Average Classify Time': [(classify_train_time + classify_test_time) / (len(train_data) + len(test_data))],
    }).to_csv(f'{output_folder}/time.csv', index=False)