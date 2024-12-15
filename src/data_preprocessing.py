import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

def preprocess_table_1(table_1):
    # Renaming columns for consistency
    table_1 = table_1.rename(columns={
        'Total persons  at work1': 'Total_Persons_At_Work',
        'Teleworked some hours2': 'Teleworked_Some_Hours',
        'Teleworked all hours': 'Teleworked_All_Hours',
        'Persons who did not telework or work at home for pay': 'Did_Not_Telework',
        'Total persons  at work1.1': 'Total_Persons_At_Work_Percent',
        'Total.1': 'Total_Teleworked_Worked_Home_Percent',
        'Teleworked some hours2.1': 'Teleworked_Some_Hours_Percent',
        'Teleworked all hours.1': 'Teleworked_All_Hours_Percent',
        'Persons who did not telework or work at home for pay.1': 'Did_Not_Telework_Percent'
    })
    return table_1

def preprocess_table_2(table_2):
    # Renaming columns for consistency
    table_2 = table_2.rename(columns={
        'Total persons  at work1': 'Total_Persons_At_Work',
        'Teleworked some hours2': 'Teleworked_Some_Hours',
        'Teleworked all hours': 'Teleworked_All_Hours',
        'Persons who did not telework or work at home for pay': 'Did_Not_Telework',
        'Total persons  at work1.1': 'Total_Persons_At_Work_Percent',
        'Total.1': 'Total_Teleworked_Worked_Home_Percent',
        'Teleworked some hours2.1': 'Teleworked_Some_Hours_Percent',
        'Teleworked all hours.1': 'Teleworked_All_Hours_Percent',
        'Persons who did not telework or work at home for pay.1': 'Did_Not_Telework_Percent'
    })
    return table_2

def preprocess_table_3(table_3):
    # Renaming columns for consistency
    table_3 = table_3.rename(columns={
        'Number of persons (in thousands)': 'Number_Persons_Thousands',
        'Average weekly hours teleworked or worked at home for pay': 'Avg_Weekly_Hours_Teleworked',
        'Average weekly hours worked': 'Avg_Weekly_Hours_Worked',
        'Hours teleworked or worked at home for pay as a percent of hours worked2': 'Telework_Hours_Percent',
        'Number of persons (in thousands).1': 'Teleworked_Hours_Thousands',
        'Average weekly hours teleworked or worked at home for pay.1': 'Avg_Weekly_Hours_Teleworked_1',
        'Average weekly hours worked.1': 'Avg_Weekly_Hours_Worked_1',
        'Hours teleworked or worked at home for pay as a percent of hours worked2.1': 'Telework_Hours_Percent_1'
    })
    return table_3

def preprocess_table_4(table_4):
    # Renaming columns for consistency
    rename_mapping_table4 = {
        'Number of persons (in thousands)': 'Teleworked_Hours_Thousands',
        'Average weekly hours teleworked or worked at home for pay': 'Avg_Weekly_Hours_Teleworked',
        'Average weekly hours worked': 'Avg_Weekly_Hours_Worked',
        'Hours teleworked or worked at home for pay as a percent of hours worked2': 'Telework_Hours_Percent',
        'Number of persons (in thousands).1': 'Teleworked_Hours_Thousands_1',
        'Average weekly hours teleworked or worked at home for pay.1': 'Avg_Weekly_Hours_Teleworked_1',
        'Average weekly hours worked.1': 'Avg_Weekly_Hours_Worked_1',
        'Hours teleworked or worked at home for pay as a percent of hours worked2.1': 'Telework_Hours_Percent_1'
    }
    table_4.rename(columns=rename_mapping_table4, inplace=True)
    return table_4

def impute_and_convert_numeric(df, columns):
    """
    Imputes missing values using median and converts columns to numeric.
    """
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    imputer = SimpleImputer(strategy='median')
    df[columns] = imputer.fit_transform(df[columns])
    return df
