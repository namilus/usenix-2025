import numpy as np
import pandas as pd

# preprocessing code taken from
# https://www.kaggle.com/code/elikplim/analysing-predicting-adults-income
# and
# https://github.com/agayev169/Adult/blob/master/train.py
CATEGORICAL_COLUMNS = {
    'workclass' : ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'],
    'education' : ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'],
    'marital_status' : ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'],
    'occupation' : ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'],
    'relationship' : ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
    'race' : ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'],
    'sex' : ['Female', 'Male'],
    'native_country' : ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'],
    'income' : ['<=50K', '>50K']
}

ALL_COLUMNS = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

def process_data(f):
    df = pd.read_csv(f, header=None, names=ALL_COLUMNS, sep=", ", engine='python')
    # make age numeric
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df = df.replace({'?': np.nan}).dropna()
    # normalise numeric columns
    df['age']           /= df['age'].max()
    df['fnlwgt']        /= df['fnlwgt'].max()
    df['education_num'] /= df['education_num'].max()
    df['capital_gain']  /= df['capital_gain'].max()
    df['capital_loss']  /= df['capital_loss'].max()
    print(df.head())

    columns2transform = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country', 'income']
    for c in columns2transform:
        ccategories = CATEGORICAL_COLUMNS[c]
        if c == 'income':
            df[c] = pd.factorize(df[c])[0]
        else:
            df[c] = pd.factorize(df[c])[0] + 1
            # normalise
            df[c] = df[c] / len(ccategories)


    arr = df.to_numpy()

    X = arr[:,:-1]
    y = arr[:,-1]

    return X, y
    
    
    



def main():
    train_file = "./adult.data"
    test_file = "./adult.test"

    x_train, y_train = process_data(train_file)
    x_test, y_test = process_data(test_file)

    np.savez("adult.npz",
             x_train = x_train,
             y_train = y_train,
             x_test  = x_test,
             y_test  = y_test)

if __name__ == "__main__":
    main()
    
