import pandas as pd
from src.data_preparation.process_data import prepare_features_and_data
from src.model.model import model_final


def main():
   
    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = prepare_features_and_data()
    rmse=model_final(X_train, Y_train, X_valid, Y_valid, X_test)
    print('Stacking Model RMSE on Validation data:', rmse)


if __name__ == "__main__":
    main()
   
