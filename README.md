# Predicting Telework Hours

## 📈 Project Overview

This project focuses on predicting the average weekly hours teleworked by individuals based on various demographic and employment-related factors. Leveraging machine learning models such as Random Forest, XGBoost, and Neural Networks, the project aims to provide accurate predictions and insights into the key drivers influencing telework engagement.

## 📂 Project Structure
```
Predicting_Telework_Hours/
├── README.md                # Project documentation
├── environment.yml          # Conda environment configuration file
├── requirements.txt         # Python dependencies
├── scaler_emp.pkl           # Saved scaler for preprocessing
├── src/                     # Source code for the project
├── data/                    # Directory for datasets
│   ├── cleaned/             # Cleaned datasets
│   │   ├── table_1_refined.csv
│   │   ├── table_2_refined.csv
│   │   ├── table_3_refined.csv
│   │   ├── table_4_refined.csv
│   │   ├── table_5_refined.csv
│   │   ├── table_6_refined.csv
│   │   ├── table_7_refined.csv
│   │   └── table_8_refined.csv
│   ├── processed/           # Processed and encoded data files
│   │   ├── merged_emp_df_encoded.csv
│   │   ├── X_train.pkl
│   │   ├── X_test.pkl
│   │   ├── y_train.pkl
│   │   └── y_test.pkl
│   └── raw/                 # Raw unprocessed data (if any)
├── notebooks/               # Jupyter notebooks for analysis
│   ├── 1_EDA.ipynb          # Exploratory Data Analysis
│   └── 2_Modeling.ipynb     # Model training and evaluation
├── models/                  # Saved machine learning models
│   ├── tuned_random_forest_telework_model.pkl
│   ├── tuned_xgboost_telework_model.pkl
│   └── telework_neural_network_model.pth
├── plots/                   # Visualizations and plots
│   ├── actual_vs_predicted_comparison.png
│   ├── correlation_heatmap.png
│   ├── boxplot_telework_by_employment_status.png
│   └── telework_hours_distribution.png
└── report/                  # Directory for additional reports or results

```



## 🔧 Installation

### 🐍 Using `requirements.txt`

1. **Create a virtual environment (optional but recommended):**

    ```bash
    python3 -m venv telework_env
    source telework_env/bin/activate  # On Windows: telework_env\Scripts\activate
    ```

2. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

### 🐳 Using `environment.yml`

1. **Install Conda** if you haven't already. You can download it from [here](https://docs.conda.io/en/latest/miniconda.html).

2. **Create the Conda environment:**

    ```bash
    conda env create -f environment.yml
    ```

3. **Activate the environment:**

    ```bash
    conda activate telework_env
    ```

## 📚 Usage

The project is organized into two main Jupyter notebooks:

1. **Exploratory Data Analysis (EDA):** `1_EDA.ipynb`

    - **Purpose:** Load, clean, and explore the dataset.
    - **Steps:**
        - Data Loading and Cleaning
        - Data Integration
        - Feature Engineering
        - Visualization and Insights

2. **Model Training and Evaluation:** `2_Modeling.ipynb`

    - **Purpose:** Train machine learning models to predict telework hours.
    - **Steps:**
        - Data Preparation
        - Model Training (Random Forest, XGBoost, Neural Network)
        - Model Evaluation
        - Model Comparison
        - Saving Trained Models

### 📊 Visualizations

All generated plots are saved in the `plots/` directory for easy reference and sharing.

### 🗃️ Data Management

- **Raw Data:** Stored in `data/cleaned/` directory.
- **Processed Data:** Cleaned and merged datasets are saved in `data/processed/`.
- **Models:** Trained models are stored in the `models/` directory.
- **Scaler:** The `StandardScaler` used during feature scaling is saved as `scaler_emp.pkl`.

## 🛠️ Dependencies

All project dependencies are listed in the `requirements.txt` and `environment.yml` files. Below is a brief overview of the main packages used:

- **Data Manipulation:**
    - `pandas`
    - `numpy`

- **Visualization:**
    - `matplotlib`
    - `seaborn`

- **Machine Learning:**
    - `scikit-learn`
    - `xgboost`

- **Deep Learning:**
    - `torch` (PyTorch)

- **Utilities:**
    - `joblib`
    - `pathlib`

## 📄 Files Description

- **`requirements.txt`**: Lists all Python dependencies required to run the project.
- **`environment.yml`**: Conda environment configuration file.
- **`README.md`**: Project overview and instructions.
- **`notebooks/1_EDA.ipynb`**: Jupyter notebook for Exploratory Data Analysis.
- **`notebooks/2_Modeling.ipynb`**: Jupyter notebook for Model Training and Evaluation.
- **`data/cleaned/`**: Directory containing cleaned CSV files.
- **`data/processed/merged_emp_df_encoded.csv`**: Processed and encoded dataset ready for modeling.
- **`models/`**: Directory containing trained model files.
- **`scaler_emp.pkl`**: Saved scaler used for feature scaling.
- **`plots/`**: Directory containing all generated plots and visualizations.

## 🤝 Contributing

Contributions are welcome! Please follow these steps to contribute:

1. **Fork the repository.**
2. **Create a new branch:**

    ```bash
    git checkout -b feature/YourFeatureName
    ```

3. **Commit your changes:**

    ```bash
    git commit -m "Add new feature"
    ```

4. **Push to the branch:**

    ```bash
    git push origin feature/YourFeatureName
    ```

5. **Open a Pull Request.**

## 📫 Contact

For any inquiries or feedback, please contact [Abdur Rahman](ar7165@nyu.edu).

---

**Notes:**

- **Replace `[Abdur Rahman]` and `[ar7165@nyu.edu]`** with your actual name and contact email.
- **Plot Directory:** Ensure that the `plots/` directory exists and contains all your generated visualization files.
- **Data Files:** Make sure all data files are correctly placed in the `data/cleaned/` and `data/processed/` directories as per the project structure.
- **Model Files:** After training your models, save them in the `models/` directory for easy access and deployment.

Feel free to customize the `README.md` further to include more specific details about your project, such as results, key findings, or future work.

---

If you need further assistance or modifications to these files, feel free to ask!
