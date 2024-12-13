# Predicting Telework Engagement: Estimating Average Weekly Telework Hours Using Machine Learning

## Table of Contents
- [Introduction](#introduction)
- [Data Description](#data-description)
- [Models and Methods](#models-and-methods)
- [Results and Interpretation](#results-and-interpretation)
- [Conclusion and Next Steps](#conclusion-and-next-steps)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Contact](#contact)

## Introduction
In the evolving landscape of remote work, understanding telework engagement is crucial for organizations aiming to optimize workforce management strategies. This project develops predictive models to estimate the average weekly telework hours of employees across various demographics and occupations.

## Data Description
The dataset is sourced from the U.S. Bureau of Labor Statistics' Current Population Survey (CPS) data, focusing on telework metrics from October. Key attributes include demographic details, occupational information, and telework hours.

## Models and Methods
We employed three machine learning models:
- **Random Forest Regressor**
- **XGBoost Regressor**
- **Neural Network (PyTorch)**

Models were evaluated using Mean Squared Error (MSE) and R-squared (R²) metrics.

## Results and Interpretation
XGBoost achieved the best performance with an MSE of 0.1916 and R² of 0.9761. Feature importance analysis revealed that [Key Features] significantly influence telework hours. The neural network underperformed, likely due to [Reasons].

## Conclusion and Next Steps
Our analysis provides actionable insights for organizations to tailor telework policies effectively. Future work includes deploying the model for real-time predictions and integrating additional data sources for enhanced accuracy.

## Repository Structure



## Installation
1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/Predicting_Telework_Hours.git
    cd Predicting_Telework_Hours
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage
- **Exploratory Data Analysis:**
    ```bash
    jupyter notebook notebooks/01_EDA.ipynb
    ```

- **Data Preprocessing:**
    ```bash
    jupyter notebook notebooks/02_Data_Preprocessing.ipynb
    ```

- **Model Training:**
    ```bash
    jupyter notebook notebooks/03_Model_Training.ipynb
    ```

- **Model Evaluation:**
    ```bash
    jupyter notebook notebooks/04_Model_Evaluation.ipynb
    ```

- **Final Analysis:**
    ```bash
    jupyter notebook notebooks/05_Final_Analysis.ipynb
    ```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
For any inquiries, please contact [your.email@example.com](mailto:your.email@example.com).
