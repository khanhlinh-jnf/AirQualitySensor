# AirQualitySensor

# 💖 How to run code

1.  Install the required libraries using the command below:
```bash
pip install -r requirements.txt
```
2. Run the code using the command below:
```bash
python train.py
python utils.py
python predict.py
```
3. Choose the model you want to run from the list of available models in the code.

4. Input the required parameters when prompted. The code will then run the selected model and display the results.

## Project Structure
```
AirQualitySensor/
├── data/
│   ├── dummy_test.csv                   # Dummy test data for model evaluation
│   └── train.csv                        # Training data for model development
├── models_pycache/
│   ├── advance_model.cpython-313.pyc    # Compiled bytecode for advance_model.py
│   └── linear_model.cpython-313.pyc     # Compiled bytecode for linear_model.py
├── advance_model.py                     # Script for advanced model implementation
├── linear_model.py                      # Script for linear model implementation
├── predict.py                           # Script for making predictions
├── README.md                            
├── requirements.txt                     # List of required Python libraries
├── train.py                             # Script for training models
└── utils.py                             # Utility functions for the project
```