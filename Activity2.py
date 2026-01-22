# Imports
import os # Standard Python module that helps us interact with the operating system
import pandas as pd # Library for data analysis.
import tkinter as tk # Standard Python library for graphical interfaces
from tkinter import filedialog, messagebox # To open the system file explorer and display pop-up windows
import numpy as np # Base library for numerical calculations in Python


# CSV Dataset Loader with File Explorer

# select_csv_file: Function to open the user's file explorer, select the respective file, 
# verify that it is a CSV file, and return the absolute path.
# Parameters: none
# Return: Absolute path of the CSV file
def select_csv_file() -> str:
    # In case of errors, we try to:
    try:
        root = tk.Tk() # Create the tkinter root window.
        root.withdraw() # Hide the main window to show only the file explorer.

        # Open the file explorer where only csv files can be selected and Save the file path.
        file_path = filedialog.askopenfilename(
            title="Select a CSV dataset", # We choose a title for this action
            filetypes=[("CSV Files", "*.csv")] # Only csv files can be selected.
        )

        # If the user cancels the action, the path will be empty.
        if not file_path:
            raise FileNotFoundError("No file was selected.") # A controlled error is thrown to prevent continuing without data.
        
        # Convert the name to lowercase and verify that it ends in .csv.
        if not file_path.lower().endswith(".csv"):
            raise ValueError("The selected file is not a CSV file.") # A controlled error is thrown to prevent continuing without data.
        
        return os.path.abspath(file_path) # We return the absolute path of the file.
    
    # Capture any errors 
    except Exception as error:
        raise error # Throw them again.

# load_dataset: Load a CSV file from disk and convert it to a pandas dataframe
# Parameters:
# csv_path: str -> string with the path to the CSV file
# Return: Return a pandas dataframe
def load_dataset(csv_path: str) -> pd.DataFrame:
    # We try in case of errors.
    try:
        dataset = pd.read_csv(csv_path) # We read the CSV file and create a dataframe from it, interpreting headers, columns, and data.
        return dataset # We return the dataframe ready for the program.
    #  If so original error.mething fails, we capture the error
    except Exception as error:
        raise IOError(f"Error reading CSV file: {error}") # We raise an IOError with a message.


# Dataset Cleaning Utilities

# normalize_missing_values: Function that helps standardize missing values in the dataset.
# Parameters:
# dataset: pd.DataFrame -> An uncleaned DataFrame.
# Return: The same dataset but with missing values normalized to “Nan.”
def normalize_missing_values(dataset: pd.DataFrame) -> pd.DataFrame:

    # List of patterns that are common missing data types that we usually encounter.
    missing_patterns = [
        "NA", "N/A", "NAN", "NaN", "NULL", "null", "", " "
    ]

    # Searches all values for any missing_patterns value and replaces them with np.nan.
    dataset = dataset.replace(missing_patterns, np.nan) # Searches all values for any missing_patterns value and replaces them with np.nan.
    return dataset # Return the normalized dataset.

# trim_string_columns: remove leading and trailing whitespace
# Parameters:
# dataset: pd.DataFrame -> Dataset with unnecessary whitespace
# Return: Return the same dataset but with clean strings
def trim_string_columns(dataset: pd.DataFrame) -> pd.DataFrame:

    # Go through all the columns in the DataFrame.
    for column in dataset.columns:
        dataset[column] = dataset[column].apply( # Apply the function to each value in the column.
            # Lambda function: Check if the value is text and, if so, remove leading and trailing spaces. 
            # If it is not a string, leave it intact.
            lambda value: value.strip() if isinstance(value, str) else value
        )
    return dataset # Return the DataFrame with white spaces removed.

# convert_numeric_columns: Convert all possible columns to numeric types such as int or float
# Parameters: 
# dataset: pd.DataFrame -> Dataset to continue cleaning
# Return: Return the same DataFrame but even more organized and clean
def convert_numeric_columns(dataset: pd.DataFrame) -> pd.DataFrame:

    # We go through all the columns in the dataset.
    for column in dataset.columns:
        # We try in case of errors.
        try:
            dataset[column] = pd.to_numeric(dataset[column]) # We try to convert the entire column to either integers or floats.
        # If the conversion fails, the error is ignored.
        except (ValueError, TypeError): 
            pass # The column remains intact.
    
    return dataset # Return the dataframe with the correct data types.

# clean_dataset: We thoroughly clean all data using normalization functions, 
# removing all types of bad rows and converting types.
# Parameters: 
# dataset: pd.DataFrame -> Original dataset from the CSV, not yet cleaned.
# Return: The completely cleaned and sorted dataset.
def clean_dataset(dataset: pd.DataFrame) -> pd.DataFrame:

    original_row_count = len(dataset) # Save how many rows the dataset originally had

    dataset = normalize_missing_values(dataset) # Normalize missing values

    dataset = trim_string_columns(dataset) # Trim string columns

    rows_with_nan = dataset.isna().any(axis=1).sum() # Count rows with missing values

    dataset = dataset.dropna()  # Drop rows witn any NaN

    duplicate_rows = dataset.duplicated().sum() # Count duplicate rows

    dataset = dataset.drop_duplicates() # Drop duplicate rows

    dataset = convert_numeric_columns(dataset) # Convert numeric columns

    dataset = dataset.reset_index(drop=True) # Reset index

    final_row_count = len(dataset) # Save how many rows remained after cleaning

    # We created a console report on what was lost, cleaned, and remained.
    print("\n========== DATA CLEANING REPORT ==========")
    print(f"Original rows: {original_row_count}")
    print(f"Rows removed due to missing values: {rows_with_nan}")
    print(f"Duplicate rows removed: {duplicate_rows}")
    print(f"Final rows after cleaning: {final_row_count}")
    print("==========================================\n")

    return dataset # We returned the final dataset for the program.

# Feature & Target Selection 

# select_target_and_features: Function that allows the user to graphically select the target columns and predictor columns.
# Parameters:
# dataset: pd.DataFrame -> Cleaned dataset containing all columns relevant to the program.
# Return: A tuple with the target column in the form of a string and the predictor columns in the form of a list of strings.
def select_target_and_features(dataset: pd.DataFrame) -> tuple[str, list[str]]:

    selected_target = {"value": None} # State variable to save the target column
    selected_features = {"values": []} # State variable to save the predictor columns

    # Internal function
    # confirm_selection(): Executed when the user presses the confirm selection button
    # Parameters: None
    # Return: None
    def confirm_selection():
        # We try in case of errors.
        try:
            target_indices = target_listbox.curselection() # Return the tuples of the selected indexes for each column
            feature_indices = feature_listbox.curselection() # Return the tuples of the selected indexes for each column

            # Validate the target column; there can only be one dependent variable.
            if len(target_indices) != 1:
                raise ValueError("You must select exactly ONE target column.") # Raise an error if another column is selected.
            
            # Validate that predictor columns are selected.
            if len(feature_indices) < 1:
                raise ValueError("You must select at least ONE feature column.") # Raise an error if no column has been selected
            
            target_column = target_listbox.get(target_indices[0]) # Create a list of the columns selected as features
            feature_columns = [feature_listbox.get(i) for i in feature_indices] # Obtain the actual names of the columns

            # Validate that the feature columns are not repeated with the target
            if target_column in feature_columns:
                raise ValueError("Target column cannot be part of feature columns.") # Raise an error if the user has done so.
            
            selected_target["value"] = target_column # Save the column selection made by the user.
            selected_features["values"] = feature_columns # Save the column selection made by the user.

            selector_window.destroy() # Close the window

        # Raise an error
        except ValueError as error:
            messagebox.showerror("Selection Error", str(error)) # Display an error message to the user

    # GUI Window
    selector_window = tk.Toplevel() # Creating the window
    selector_window.title("Select Target and Feature Columns") # Giving it a title
    selector_window.geometry("700x400")  # Setting the window size

    selector_window.resizable(False,False) # Preventing resizing

    # Labels
    # Labels to display column data and how many can be selected
    tk.Label(selector_window, text="Target Column (Select ONE)", font=("Arial", 10, "bold")).grid(row=0, column=0, padx=10, pady=5)
    tk.Label(selector_window, text="Feature Columns (Select ONE or MORE)", font=("Arial", 10, "bold")).grid(row=0, column=1, padx=10, pady=5)

    # Listboxes 
    target_listbox = tk.Listbox(selector_window, exportselection=False) # Prevent the list from getting lost
    feature_listbox = tk.Listbox(selector_window, selectmode=tk.MULTIPLE) # Allow selection of more than one column

    # Scroll through the columns in the dataset
    for column in dataset.columns:
        target_listbox.insert(tk.END, column) # Insert the columns into their respective fields
        feature_listbox.insert(tk.END, column) # Insert the columns into their respective fields
    
    target_listbox.grid(row=1, column=0, padx=10, pady=5, sticky="nsew") # Configure the size and position of the listbox
    feature_listbox.grid(row=1, column=1, padx=10, pady=5, sticky="nsew") # Configure the size and position of the listbox

    # Scrollbars
    target_scroll = tk.Scrollbar(selector_window, command=target_listbox.yview) # Allow handling datasets with many columns.
    feature_scroll = tk.Scrollbar(selector_window, command=feature_listbox.yview) # Connect the scroll bar to the list box.

    target_listbox.config(yscrollcommand=target_scroll.set) # We configure the list box with the scroll bar.
    feature_listbox.config(yscrollcommand=feature_scroll.set) # We configure the list box with the scroll bar.

    target_scroll.grid(row=1, column=0, sticky="nse") # We configure the scroll for this section.
    feature_listbox.grid(row=1, column=1, sticky="nse") # We configure the scroll for this section.

    # Create a confirmation button.
    confirm_button = tk.Button(
        selector_window, # We connect with the window
        text="Confirm Selection", # Button text
        command= confirm_selection, # Command to achieve
        bg="#4CAF50", # Color
        fg="white", # Template color
        padx=10, # Shape
        pady=5 # Shape
    )

    confirm_button.grid(row=2, column=0, columnspan=2, pady=15) # Button size configuration

    selector_window.grab_set() # Wait for the program.
    selector_window.wait_window() # Do not continue until the window is closed.

    # If none of the columns are selected.
    if selected_target["value"] is None:
        raise ValueError("No columns were selected.") # Raise an error with a message.
    
    return selected_target["value"], selected_features["values"] # Return the columns so that the model can learn.


# Prediction Input Window 

# collect_prediction_input: Open the window to request the values to be predicted
# Parameters:
# dataset: pd.DataFrame: cleaned and used dataset for training
# feature_columns: list[str] : list of selected columns as features
# Return: A single-row DataFrame
def collect_prediction_input(dataset: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:

    user_inputs = {} # Valid final values
    entries = {} # Widgets associated with each feature

    feature_metadata = {} # Dictionary describing how each feature should behave

    # Iterate over each selected feature to detect types
    for feature in feature_columns:
        # If it detects a column that is numeric
        if pd.api.types.is_numeric_dtype(dataset[feature]):
            # Store Data
            feature_metadata[feature] = {
                "type": "numeric", # Store logical data
                "dtype": dataset[feature].dtype # Store real data
            }
        # Otherwise
        else:
            # Store Data
            feature_metadata[feature] = {
                "type": "categorical", # If categorical data is stored
                "valid_values": sorted(dataset[feature].unique().tolist()) # Store the list of valid real values
            }


    # Validation logic

    # def validate_inputs(): Executed each time the user enters data.
    # Parameters: None
    # Return: None.
    def validate_inputs():
        # We try in case of errors.
        try:
            temp_data = {} # Temporary dictionary to validate everything before accepting the prediction.

            # Iterate over each input field.
            for feature, entry in entries.items():
                raw_value = entry.get() # Obtain the text entered by the user.

                # If the field is completely empty
                if raw_value == "":
                    raise ValueError(f"Field '{feature}' cannot be empty.") # Throw an error to the user that a field is empty
                
                meta = feature_metadata[feature] # Get the definition of the feature

                # If the data is numeric
                if meta["type"] == "numeric":
                    value = float(raw_value) # Try to convert the number to decimal

                    # Convert the numbers to their original version
                    if np.issubdtype(meta["dtype"], np.integer):
                        value = int(value) # Perform the conversion
                
                # Otherwise
                else:
                    # If it fails, then the button is disabled
                    if raw_value not in meta["valid_values"]: # Avoid values that are not trained.
                        
                        # Raise the error.
                        raise ValueError(
                            f"Invalid value for '{feature}'.\n" # Message to the user.
                            f"Valid values: {meta['valid_values']}" # Message to the user.
                        )
                    value = raw_value # Save the value if it is valid.
                
                temp_data[feature] = value # It is stored temporarily.
            
            predict_button.config(state=tk.NORMAL) # Activate the predict button.
            user_inputs.clear() # Save the final values of the column. 
            user_inputs.update(temp_data) # Save the final values of the column. 
        
        # In case of failure
        except Exception:
            predict_button.config(state=tk.DISABLED) # The predict button is disabled
    
    # Submit handler
    # def submit:  close the window directly
    # Parameters: none
    # Return: None
    def submit():
        input_window.destroy() # Close the window directly

    # GUI Window    
    input_window = tk.Toplevel() # Modal secondary window
    input_window.title("Enter Values for Prediction") # Window title
    input_window.geometry("600x400") # Window dimensions
    input_window.resizable(True, True) # Whether the window can be resized

    canvas = tk.Canvas(input_window) # Create a canvas inside the window
    # Create a scrollbar inside the window
    scrollbar = tk.Scrollbar(input_window, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas) # Insert the frame into the canvas

    # Connect the scrollbar and configure it for each column
    scrollable_frame.bind(
        "<Configure>",
        # Lambda function to configure each canvas for each box 
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    # We create the window with the active canvas
    canvas.create_window((0, 0), window = scrollable_frame, anchor = "nw")

    # We configure the canvas appropriately.  
    canvas.configure(yscrollcommand=scrollbar.set)

    # We move the canvas to the left of the window.
    canvas.pack(side="left",fill="both",expand=True)

    # We move the scrollbar to the right of the window.
    scrollbar.pack(side="right", fill="y")

    # Dynamic input fields
    # We iterate through each input feature to create a label text.
    for row, feature in enumerate(feature_columns):
        meta = feature_metadata[feature] # We add the data types that the user must enter.

        label_text = feature # We label the field

        # If the type is categorical
        if meta["type"] == "categorical": 
            label_text += f" (values: {meta['valid_values']})" # We add the text to the label
        
        tk.Label(
            scrollable_frame, # We add the elements to the tag
            text=label_text, # We add the text
            anchor="w" # Your width type
        ).grid(row=row, column=0, padx=10, pady=5, sticky="w") # Size configuration

        entry = tk.Entry(scrollable_frame) # We create the text entry
        entry.grid(row=row, column=1, padx=10, pady=5, sticky="ew") #We set the size of the input
        #We configure the data entry action to validate the text in real time.
        entry.bind("<KeyRelease>", lambda event: validate_inputs())

        entries[feature] = entry # We save the entry in its respective column.
    
    scrollable_frame.columnconfigure(1, weight=1) # We configure the window scrollbar

    # Predict Button
    # We created the predict button
    predict_button = tk.Button(
        input_window, # We add it to the window.
        text="Predict", # Button text
        state=tk.DISABLED, # Status disabled
        command=submit, # We add functionality to the button
        bg="#2196F3", # Button color
        fg="white", # Button text color
        padx=10, # Button size
        pady=5 # Button size
    )

    predict_button.pack(pady=15) # We correctly added the button.

    input_window.grab_set() # We leave the window open
    input_window.wait_window() # We wait for the window to execute

    # If you do not have the inputs prepared
    if not user_inputs:
        raise ValueError("Prediction input was cancelled or incomplete.") # We raise an error that displays a message to the user.
    
    return pd.DataFrame([user_inputs]) # We return the dictionary converted into a 1-row dataframe.


# Naive Bayes Classifier
# class NaiveBayesClassifier: Trainable classifier for fitting and prediction, and statistics can be reused
class NaiveBayesClassifier:

    # __init__: Class constructor function
    # Parameter: 
    # laplace_alpha: float = 1.0 -> This is the alpha of the Laplace smoothing formula
    def __init__(self, laplace_alpha: float = 1.0):
        self.alpha = laplace_alpha # Save alpha for later use
        self.classes = None # To store possible classes and unique values
        self.class_priors = {} # The results of the logarithms are saved
        self.feature_types = {} # The predictive columns are saved
        self.feature_stats = {} # All learned knowledge is saved, such as means, variances, and categorical counts
    
    # Detect feature types
    # _detect_feature_types: Internal function to detect feature types.
    # Parameters: 
    # X: pd.DataFrame -> The predictive columns of the dataset that were selected
    # Return: None.
    def _detect_feature_types(self, X: pd.DataFrame):
        # Iterate through each feature.
        for column in X.columns:
            # If it is detected that it is numeric.
            if pd.api.types.is_numeric_dtype(X[column]):
                self.feature_types[column] = "numeric" # Add numeric type to the columns.
            # Otherwise
            else:
                self.feature_types[column] = "categorical" # Add categorical type to the columns.

    # Train model
    # fit: Function to perform actual model training
    # Parameters: 
    # X: pd.DataFrame -> Predictive columns of the dataset
    # y: pd.Series -> Target column of the dataset
    # Return: None
    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.classes = y.unique() # We obtain all classes of the target variable.
        self._detect_feature_types(X) # We detect whether each predictive value is numerical or categorical. 

        total_samples = len(y) # Total number of samples that exist.

        # Compute priors
        # We iterate each class that we have.
        for cls in self.classes:
            # We calculate the number of samples that exist AMONG the
            #  number of classes of a certain type that exist.
            self.class_priors[cls] = np.log(
                len(y[y == cls]) / total_samples
            )
        
        # Compute likehood statistics
        # We begin the likelihood calculations.
        # Where we iterate again by class, that is, we enter by class.
        for cls in self.classes:
            self.feature_stats[cls] = {} # We start the structure by classes.
            X_cls = X[y == cls] # This will be to create the subsets.

            # We iterate through each predictive value.
            for feature in X.columns:
                # If it is a numerical value, we apply the Gaussian distribution.
                if self.feature_types[feature] == "numeric":
                    mean = X_cls[feature].mean() # We calculate the mean of this predictive value.
                    variance = X_cls[feature].var(ddof=0) # We calculate the variance of this predictive value.

                    # Numeric stability
                    # If the variance is 0, we have to fix it.
                    if variance == 0:
                        variance = 1e-9 # We fix it using epsilon.

                    # We save the parameters of the distribution.
                    self.feature_stats[cls][feature] = {
                        "mean": mean, # We save the mean.
                        "var": variance # We save the variance.
                    }

                # Otherwise 
                else:
                    # We store the number of occurrences that have a certain 
                    # value in the predictive column in a dictionary.
                    value_counts = X_cls[feature].value_counts().to_dict()
                    # We store the values of the Laplace formula.
                    self.feature_stats[cls][feature] = {
                        "counts": value_counts, # The numerator.
                        "total": sum(value_counts.values()), # The denominator.
                        "unique": len(value_counts) # The “k” in the formula. 
                    }

    # Gaussian log probability
    # _gaussian_log_probability_: exact recreation of the Gaussian formula
    # Parameters:
    # x -> Value to predict
    # mean -> mean to be calculated
    # var -> variance to be calculated
    # Return: None
    def _gaussian_log_probability_(self, x , mean, var):
        # Return the exact implementation of the Gaussian formula
        return(
            -0.5 * np.log(2 * np.pi * var)
            - ((x - mean) ** 2) / (2 * var)
        )

    # Predict a single instance
    # predict_single: Make a prediction for a specific instance, predict for a row
    # Parameters:
    # x: pd.Series -> Predictive values of the dataframe
    # Return: Prediction for each class in dictionary form
    def predict_single(self, x: pd.Series) -> dict:
        log_posteriors = {} # We will store the prediction results in this dictionary.

        # We iterate through each class.
        for cls in self.classes: 
            log_prob = self.class_priors[cls] # We calculate each prior for each class.
            
            # We iterate through each feature and its value.
            for feature, value in x.items():
                stats = self.feature_stats[cls][feature] # We retrieve the learned statistics.
                
                # If the predictive value is numerical,
                if self.feature_types[feature] == "numeric":
                    # we add it to the logarithm probability together with the value to be predicted, 
                    # mean, variance, and everything else already calculated.
                    log_prob += self._gaussian_log_probability_(
                        value, # Value to predict
                        stats["mean"], # The Mean
                        stats["var"] # The variance
                    )

                # Otherwise
                else: 
                    # if it is categorical, we first normalize the values that do not appear.
                    count = stats["counts"].get(value, 0)
                    numerator = count + self.alpha # We create the numerator.
                    denominator = stats["total"] + self.alpha * stats["unique"] # We create the denominator for the formulas.

                    log_prob += np.log(numerator / denominator) # We add the categorical log-likelihood.
            
            log_posteriors[cls] = log_prob # We save the final result of the class we were calculating.

        return log_posteriors # We return the dictionary with the results of each predictive class.
    
    # Predict class
    # predict: create the final prediction
    # Parameters:
    # X: pd.DataFrame -> we pass all the values that will form the prediction
    # Return: we return the predictions made, now if they are final
    def predict(self, X: pd.DataFrame):
        predictions = [] # List to store the final predictions

        # Iterate over each row of the columns
        for _, row in X.iterrows():
            posteriors = self.predict_single(row) # Calculate the log-posterior of each completed column
            predicted_class = max(posteriors, key=posteriors.get) # Select the maximum log that came out of the calculations
            predictions.append((predicted_class, posteriors)) # We add the class that won and also the logs for each class.

        return predictions # We return the final prediction containing the result and the logs.

# Prediction Result Window (GUI)
# show_prediction_result: Visually display the model result
# Parameters:
# predicted_class -> final class chosen by the model
# log_posteriors: dict -> dictionary of probabilities for each category
# on_new_prediction -> callback that runs after the function
# Return: None
def show_prediction_result( predicted_class, log_posteriors: dict, on_new_prediction):
    result_window = tk.Toplevel() # Create the window
    result_window.title("Prediction Result") # Text inside the element
    result_window.geometry("500x400") # Window dimensions
    result_window.resizable(False, False) # If the window can be resizable

    # Create the label
    tk.Label(
        result_window, # Add the label to the window
        text="Prediction Result", # Text inside the element
        font=("Arial", 16, "bold") # Text source
    ).pack(pady=10) # Size

    # Create the label
    tk.Label(
        result_window, # Add the label to the window
        text=f"Predicted Class:\n{predicted_class}", # Text inside the element
        font=("Arial", 14), # Text source
        fg="#2E7D32" # Color
    ).pack(pady=10) # Position of the element

    frame = tk.Frame(result_window) # We add the frame to the window
    frame.pack(pady=10, fill="both", expand=True) # Position of the element

    # Create the label
    tk.Label(
        frame, # Contains the frame
        text="Log-Posterior Probabilities", # Text inside the element
        font=("Arial", 11, "bold") # Text source
    ).pack(pady=5) # Position of the element

    # We iterate to display each class
    for cls, value in log_posteriors.items():
        # The winning class will be green, the others will be black
        color = "#2E7D32" if cls == predicted_class else "black"

        # We created a label
        tk.Label(   
            frame, # We add the frame
            text=f"{cls}: {value:.6f}", # Text source
            fg=color, # Text color
            anchor="w" # Broaden
        ).pack(fill="x", padx=20) # Position configuration within the window

    button_frame = tk.Frame(result_window) # We add the button to the window in a certain frame.
    button_frame.pack(pady=15) # Position configuration within the window

    # Let's create the button
    tk.Button(
        button_frame, # We add the button 
        text="New Prediction", # Text inside the element
        # Lambda function to either close the program or create a new prediction 
        command=lambda: [result_window.destroy(), on_new_prediction()],
        bg="#1976D2", # Selected color
        fg="white", # Text color
        padx=10, # Size
        pady=5 # Size
    ).grid(row=0, column=0, padx=10) # Position configuration within the window

    # Let's create the button
    tk.Button(
        button_frame, # We add the button 
        text="Close", # Text inside the element
        command=result_window.destroy, # Button command
        bg="#9E9E9E", # Selected color
        fg="white", # Text color
        padx=10, # Size
        pady=5 # Size
    ).grid(row=0, column=1, padx=10) # Position configuration within the window

    result_window.grab_set() # Leave the window open
    result_window.wait_window() # Waiting for the program to respond


# main: main execution function
# Parameters: None
# Return: None
def main():
    # We try to execute the code
    try:
        print("Please select your csv dataset...") # Informative messages

        csv_file_path = select_csv_file() # Open the file explorer
        print(f"CSV file selected:\n{csv_file_path}") # Informative messages

        dataset = load_dataset(csv_file_path) # Convert the CSV to a dataframe

        print("\nDataset loaded successfully!") # Informative messages
        print("Dataset preview:") # Informative messages
        print(dataset.head()) # Show dataset

        dataset = clean_dataset(dataset) # Data cleaning

        print("Cleaned dataset preview:") # Informative messages
        print(dataset.head()) # Show dataset

        # Target variables and predictor variables
        target_column, feature_columns = select_target_and_features(dataset)

        print("Target column selected: ") # Informative messages
        print(target_column) # The target column

        print("\nFeature columns selected:") # Informative messages

        # We iterate over the columns of predictive values.
        for feature in feature_columns:
            print(f"- {feature}")  # Informative messages

        # We train the model
        X_train = dataset[feature_columns] # We train the features
        y_train = dataset[target_column] # We train the classes

        model = NaiveBayesClassifier(laplace_alpha=1.0) # We initialize the hybrid Naive Bayes with categorical and numerical variables.
        model.fit(X_train, y_train) # Model training

        # run_prediction: internal closure function to reuse predictions
        # Parameters: None
        # Return: None
        def run_prediction():
            # User input and also open the dynamic GUI
            input_data = collect_prediction_input(dataset, feature_columns)

            print("\nPrediction input received:") # Informative messages
            print(input_data) # Show the inputs of the user

            predictions = model.predict(input_data) # The prediction is made and the results are returned
            predicted_class, log_posteriors = predictions[0] # Unpack the results obtained

            print("\n========== PREDICTION RESULT ==========") # Informative messages
            print("Log-Posterior Probabilities per Class:") # Informative messages
            # We iterate to obtain the classes and their respective final values.
            for cls, value in log_posteriors.items(): 
                print(f"Class '{cls}': {value:.6f}") # Informative messages

            print("\nFinal Predicted Class:") # Informative messages
            print(predicted_class) # Winning class
            print("=======================================\n") # Informative messages
            
            # Display results in the interface
            show_prediction_result(
                predicted_class, # winning class
                log_posteriors, # probability logs
                on_new_prediction=run_prediction # New prediction
            )

        run_prediction() # The first decision we make within the program.
        
        # Final message to the user regarding the status of the dataset.
        messagebox.showinfo(
            "Success",
            "CSV file loaded and cleaned successfully.\nCheck the console for dataset preview."
        )

    # Error if the file is canceled
    except FileNotFoundError as error:
        # Error message if this error occurs
        messagebox.showwarning("File Selection Error", str(error))

    # Error if the selection or validation is canceled
    except ValueError as error:
        # Error message if this error occurs
        messagebox.showerror("Invalid File", str(error))

    # Error reading the CSV file
    except IOError as error:
        # Error message if this error occurs
        messagebox.showerror("File Read Error", str(error))

    # Unexpected error
    except Exception as error:
        # Error message if this error occurs
        messagebox.showerror("Unexpected Error", f"An unexpected error occurred:\n{error}")

# Program entry point  
if __name__ == "__main__":
    main() # Main function