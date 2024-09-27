import os
import pandas as pd
from transformers import BertForSequenceClassification, Trainer, TrainingArguments, BertTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import torch
from torch.utils.data import Dataset
import numpy as np




class TextClassificationModel:
    """
    A text classification model using a pre-trained BERT model. This class provides
    methods for loading data, training, evaluating, and saving the model.
    """
    def __init__(self, model_name='bert-base-uncased', cache_dir=None):
        """
        Initializes the model, tokenizer, and device.
        :param model_name: Name of the pre-trained BERT model (default: bert-base-uncased)
        :param cache_dir: Directory where the model cache is stored (default: None)
        """
        self.model_name = model_name
        self.cache_dir = cache_dir if cache_dir else os.getcwd()  # Use current directory if no cache dir is provided
        self.tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=self.cache_dir)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.df = None  # DataFrame to store the data
        self.class_mapping = {}  # Dictionary to map class labels to indices
        self.scaler = StandardScaler()  # For scaling numeric data
        self.encoder = OneHotEncoder(sparse_output=False)  # For encoding categorical data


    def load_data(self, file_path):
        """
        Loads the data from a CSV or Excel file.
        :param file_path: Path to the file (CSV or Excel)
        :return: List of column names in the dataset
        """
        # Determine if file is CSV or Excel
        if file_path.endswith('.csv'):
            self.df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            self.df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")

        # Return the column names in the dataset
        print("Columns found in the dataset:")
        print(self.df.columns.tolist())
        return self.df.columns.tolist()

    def set_columns(self, label_column, text_columns, numeric_columns):
        """
        Sets the label, text, and numeric columns for training.
        :param label_column: The column containing the target labels
        :param text_columns: The columns containing text data
        :param numeric_columns: The columns containing numeric/categorical data
        :return: Processed text, numeric, and label data
        """
        # Check if the selected columns exist in the dataset
        if label_column not in self.df.columns or any(col not in self.df.columns for col in text_columns + numeric_columns):
            raise ValueError("Invalid column selection. Ensure the columns exist in the dataset.")

        # Process the text data
        text_data = self.df[text_columns].astype(str).values.tolist()

        # Normalize numeric/categorical features if they exist
        if numeric_columns:
            numeric_data = self.df[numeric_columns]
            numeric_data = self.scaler.fit_transform(numeric_data)
        else:
            numeric_data = np.array([]).reshape(len(text_data), 0)

        # Convert text labels to numeric indices
        if not np.issubdtype(self.df[label_column].dtype, np.number):
            unique_labels = sorted(self.df[label_column].unique())
            print(f"Text labels found, converting them to numeric: {unique_labels}")
            self.class_mapping = {label: idx for idx, label in enumerate(unique_labels)}
            y_data = self.df[label_column].map(self.class_mapping).tolist()
        else:
            y_data = self.df[label_column].tolist()
            unique_labels = sorted(set(y_data))
            self.class_mapping = {label: idx for idx, label in enumerate(unique_labels)}

        return text_data, numeric_data, y_data

    def tokenize_data(self, texts):
        """
        Tokenizes the text data using the BERT tokenizer.
        :param texts: List of texts to be tokenized
        :return: Tokenized data
        """
        return self.tokenizer(texts, truncation=True, padding=True, max_length=512)

    def compute_metrics(self, p):
        """
        Computes metrics (accuracy, precision, recall, f1-score) during evaluation.
        :param p: EvalPrediction object containing predictions and labels
        :return: Dictionary of computed metrics
        """
        labels = p.label_ids
        preds = p.predictions.argmax(-1)
        precision, recall, f1, support = precision_recall_fscore_support(labels, preds, average=None)
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': float(acc),
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist(),
            'f1_per_class': f1.tolist(),
            'support_per_class': support.tolist()
        }

    def print_class_wise_metrics(self, eval_results):
        """
        Prints class-wise precision, recall, F1-score, and support.
        :param eval_results: The evaluation results containing metrics per class
        """
        precision, recall, f1, support = eval_results['eval_precision_per_class'], eval_results['eval_recall_per_class'], eval_results['eval_f1_per_class'], eval_results['eval_support_per_class']
        print("\nClass-wise Precision, Recall, F1-score, and Support:")
        print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-score':<10} {'Support':<10}")
        for idx, _ in enumerate(precision):
            class_name = list(self.class_mapping.keys())[list(self.class_mapping.values()).index(idx)]
            print(f"{class_name:<15} {precision[idx]:<10.4f} {recall[idx]:<10.4f} {f1[idx]:<10.4f} {support[idx]:<10}")

    def create_dataset(self, encodings, numeric_data, labels):
        """
        Creates a PyTorch dataset from encodings, numeric data, and labels.
        :param encodings: Encoded text data
        :param numeric_data: Numeric/categorical data
        :param labels: Target labels
        :return: A PyTorch Dataset object
        """
        class CustomDataset(Dataset):
            def __init__(self, encodings, numeric_data, labels):
                self.encodings = encodings
                self.numeric_data = numeric_data
                self.labels = labels

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                item['numeric_data'] = torch.tensor(self.numeric_data[idx], dtype=torch.float32)
                item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
                return item

        return CustomDataset(encodings, numeric_data, labels)

    def train(self, text_train, num_train, y_train, text_test, num_test, y_test, save_model_path='./saved_model'):
        """
        Trains the model using the given training and testing data.
        :param text_train: Training text data
        :param num_train: Training numeric/categorical data
        :param y_train: Training labels
        :param text_test: Testing text data
        :param num_test: Testing numeric/categorical data
        :param y_test: Testing labels
        :param save_model_path: Path where the model will be saved
        :return: Evaluation results after training
        """
        # Tokenize the text data
        train_encodings = self.tokenize_data(text_train)
        test_encodings = self.tokenize_data(text_test)

        # Create datasets for training and testing
        train_dataset = self.create_dataset(train_encodings, num_train, y_train)
        test_dataset = self.create_dataset(test_encodings, num_test, y_test)

        # Load pre-trained BERT model
        model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=len(set(y_train)), cache_dir=self.cache_dir).to(self.device)

        # Set training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            evaluation_strategy="epoch",
            logging_dir='./logs',
            logging_steps=10,
            save_strategy="epoch",
            load_best_model_at_end=True
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=self.compute_metrics
        )

        # Train the model
        trainer.train()

        # Save the model
        model.save_pretrained(save_model_path)
        print(f"Model saved to {save_model_path}")

        # Evaluate the model on the test dataset
        eval_results = trainer.evaluate()

        # Print the class-wise metrics
        self.print_class_wise_metrics(eval_results)

        return eval_results

    def load_and_evaluate(self, save_model_path, text_test, num_test, y_test):
        """
        Loads a pre-trained model and evaluates it on the given test data.
        :param save_model_path: Path where the pre-trained model is saved
        :param text_test: Test text data
        :param num_test: Test numeric/categorical data
        :param y_test: Test labels
        :return: Evaluation results
        """
        # Load the saved model
        model = BertForSequenceClassification.from_pretrained(save_model_path, num_labels=len(set(y_test))).to(self.device)

        # Tokenize the test data
        test_encodings = self.tokenize_data(text_test)

        # Create the test dataset
        test_dataset = self.create_dataset(test_encodings, num_test, y_test)

        # Set evaluation arguments
        eval_args = TrainingArguments(
            output_dir='./results',
            per_device_eval_batch_size=4
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=eval_args,
            eval_dataset=test_dataset,
            compute_metrics=self.compute_metrics
        )

        # Evaluate the model on the test dataset
        eval_results = trainer.evaluate()

        # Print the class-wise metrics
        self.print_class_wise_metrics(eval_results)

        return eval_results


def main():
    """
    Main function to run the model training and evaluation process.
    """
    model = TextClassificationModel()

    # Step 1: User provides the training file path
    train_file_path = input("Please enter the path to the training file (CSV or Excel): ").strip()
    train_columns = model.load_data(train_file_path)

    # Step 2: User selects the label column, text columns, and numeric columns
    label_column = input(f"Please choose the label column from: {train_columns}: ").strip()
    text_columns = input(f"Please choose the text columns (comma-separated): ").split(',')
    numeric_columns_input = input(f"Please choose the numeric columns (comma-separated, or leave blank if none): ").strip()
    numeric_columns = numeric_columns_input.split(',') if numeric_columns_input else []

    # Process the selected columns for training
    text_train, num_train, y_train = model.set_columns(label_column, [col.strip() for col in text_columns], [col.strip() for col in numeric_columns])

    # Step 3: User provides the test file path
    test_file_path = input("Please enter the path to the test file (CSV or Excel): ").strip()
    test_columns = model.load_data(test_file_path)

    # Use the same label, text, and numeric columns for the test dataset
    print(f"Using the same columns for testing: label = {label_column}, text = {text_columns}, numeric = {numeric_columns}")
    text_test, num_test, y_test = model.set_columns(label_column, [col.strip() for col in text_columns], [col.strip() for col in numeric_columns])

    # Step 4: User provides the model save path
    save_model_path = input("Please enter the path where the model should be saved (default: ./saved_model): ").strip() or './saved_model'

    # Step 5: Train the model and save it
    eval_results = model.train(text_train, num_train, y_train, text_test, num_test, y_test, save_model_path)
    print("Evaluation results after training:", eval_results)

    # Step 6: Load the saved model and evaluate it again
    loaded_eval_results = model.load_and_evaluate(save_model_path, text_test, num_test, y_test)
    print("Evaluation results after loading the saved model:", loaded_eval_results)


if __name__ == "__main__":
    main()
