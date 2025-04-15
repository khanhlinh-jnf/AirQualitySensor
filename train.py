import pandas as pd
import advance_model as advance_models
import linear_model	as linear_models

if __name__ == '__main__':
    df = pd.read_csv('data/train.csv')
    print("Training Linear Models")
    linear_models.train_linear_models(df)
    print("Training Advance Models")
    advance_models.train_advance_models(df)
    