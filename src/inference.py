import pandas as pd
import joblib
from sklearn.metrics import accuracy_score


def evaluate_accuracy(model_path, label_encoder_path, valid_file_path):
    # Load the model and label encoder
    voting_classifier = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)

    # Load the validation dataset without labels
    valid_df = pd.read_csv(valid_file_path)
    x_valid = valid_df['url']

    # Predict the validation dataset
    y_pred_valid = voting_classifier.predict(x_valid)

    # Transform predictions back to original labels
    y_pred_valid_labels = label_encoder.inverse_transform(y_pred_valid)

    # Compare predictions with actual labels for the validation dataset
    y_true_valid = valid_df['label']

    # Calculate accuracy for the validation dataset
    output_accuracy_valid = accuracy_score(y_true_valid, y_pred_valid_labels)

    return output_accuracy_valid


def create_result_csv(model_path, label_encoder_path, test_file_path):
    # Assuming your model and label encoder are already loaded
    # Load the model and label encoder
    voting_classifier = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)

    # Load the test dataset without labels
    test_df = pd.read_csv(test_file_path, encoding='utf-16le')

    # Extract features from the test dataset
    X_test = test_df['vba_code']

    # Predict the test dataset
    y_pred_test = voting_classifier.predict(X_test)

    # Transform predictions back to original labels
    y_pred_test_labels = label_encoder.inverse_transform(y_pred_test)

    # Create a DataFrame with the predictions
    predictions_df = pd.DataFrame(y_pred_test_labels, columns=['prediction'])

    # Save the predictions to a new CSV file
    predictions_file_path = 'test_prediction.csv'
    predictions_df.to_csv(predictions_file_path, index=False, encoding='utf-16-le')

    print(f"Predictions saved to: {predictions_file_path}")


def compare_csv_columns(file1, file2):
    # Load CSV files
    df1 = pd.read_csv(file1, encoding='utf-16-le')
    df2 = pd.read_csv(file2, encoding='utf-16-le')

    # Print column names for debugging
    print("Columns in df1:", df1.columns)
    print("Columns in df2:", df2.columns)

    # Get the specified column from each DataFrame
    col1 = df1["prediction"]  # Use "prediction" column from df1
    col2 = df2["label"]

    i = 0

    # Compare the columns and print indices where they are not equal
    unequal_indices = col1.index[col1 != col2].tolist()

    if not unequal_indices:
        print(f"The columns in 'prediction' are equal in both CSV files.")
    else:
        print(f"The columns in 'prediction' are not equal at the following indices:")
        for index in unequal_indices:
            print(f"Index: {index}, df1 Prediction: {col1[index]}, df2 label: {col2[index]}")
            i += 1

        print(f"Total: {i}/{len(col1)}")
        print(f"Percentage of inequality: {(1 - (i / int(len(col1))))}%")


def merge_csv(csvs, output_file='merged_file.csv'):
    csvs_read = []

    for csv in csvs:
        csvs_read.append(pd.read_csv(csv))

    # Concatenate 'url' and 'label' columns from all DataFrames
    merged_df = pd.concat([df[['url', 'label']] for df in csvs_read], ignore_index=True)

    # If you want to save the merged DataFrame to a new CSV file
    merged_df.to_csv(output_file, index=False)


if __name__ == '__main__':

    model_path = 'model.joblib'
    label_encoder_path = 'label_encoder.joblib'
    test_file_path = '../datasets/dataset_phishing.csv'
    valid_file_path = '../datasets/dataset_phishing.csv'

    accuracy_valid = evaluate_accuracy(model_path, label_encoder_path, valid_file_path)
    print(f'Accuracy on the validation dataset: {accuracy_valid * 100:}%')


    #merge_csv(["file1.csv", "file2.csv", "file3.csv"])

    # create_result_csv(model_path, label_encoder_path, valid_file_path)

    # compare_csv_columns('test_prediction.csv', valid_file_path)

    # create_result_csv(model_path, label_encoder_path, test_file_path)
