# AirQualitySensor

# 💖 How to run code

## 1.  Install the required libraries using the command below:
```bash
pip install -r requirements.txt
```
## 2. Run the code using the command below:

### To train linear and non-linear models
```bash
python train.py
```
### To run the models on the test set.
```bash
python utils.py
```
### Input parameters from the keyboard for prediction with final model.
```bash
python predict.py
```
## 3. Choose the model you want to run from the list of available models in the code.

## 4. Input the required parameters when prompted. The code will then run the selected model and display the results.

# Project Structure
```
AirQualitySensor/
├── data/
│   ├── dummy_test.csv                   # Dummy test data for model evaluation
│   └── train.csv                        # Training data for model development
├── models/								 # Directory for storing trained models
│   ├── linear_models/                   # Directory for linear models
│   ├── advanced_models/                 # Directory for advanced models
│   └── decision_tree_models/            # Directory for decision tree models with each max_depth
├── advance_model.py                     # Script for advanced model implementation
├── analyzeDT.py                         # Analyze the DecisionTree model.
├── linear_model.py                      # Script for linear model implementation
├── predict.py                           # Script for making predictions
├── README.md                            
├── requirements.txt                     # List of required Python libraries
├── train.py                             # Script for training models
└── utils.py                             # Script for running models on the test set
```
### Note
- After running the code, the trained models will be saved in the `models/` directory. You can load these models on Google Drive at link [here](https://drive.google.com/drive/u/1/folders/1nMeRZxAm8O-InNP7UvCb0tsuvo7ii0ym?fbclid=IwY2xjawJsqNhleHRuA2FlbQIxMAABHq1jUhRKirHKiB-v_hXWifXtYLMeJeTzzNSWILs3enULqM_eAclUijvJINn0_aem_3yZdkM9mTUZxqzNfE1E-sg) for use.
- In each folder after run `train.py` and `utils.py`, will have results_summary.csv and results_on_test.csv files. These files contain the results of MAE of each model.