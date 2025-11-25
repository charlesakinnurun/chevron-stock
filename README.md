# Chevron Stock
![Chevron](/image.jpg)

## Procedures
- Import libraries
    - pandas
    - scikit-learn
    - numpy
    - seaborn
    - matplotlib
    - yfinance
- Data Collection
    - Data download from the Yahoo Finance API
- Data Preprocessing
    - Check for missing values
    - Check for duplicated rows
- Feature Engineering
    - Features: Lag_1, Lag_7, SMA_5, SMA_20
    - Target: Close
- Pre-Training Visualization
![pre-training-visualization](/output1.png)
- Data Splitting
    - Spltting the data into training and testing sets
    - Using a time-series splitting (no shuffling) is crucical for financial  data
    - Use 80% for training and 20% for testing
- Data Scaling
    - Initialize the Standard Scaler which are essential for Linear, Ridge, Lasso Regression
- Model Definition
    - Linear Regression
    - Ridge Regression
    - Lasso Regression
    - Random Forest Regressor
- Hyperparameter Tuning
    - alpha
    - max_depth
    - min_samples
    - n_estimators
- Model Training, Prediction and Evaluation
    - The Best Model is the Ridge Regression (Tuned) with an RMSE of 1.4299

| Model                          | RMSE      | MAE       | R²        |
|--------------------------------|-----------|-----------|-----------|
| Ridge Regression (Tuned)       | 1.429873  | 2.044537  | 0.826168  |
| Linear Regression              | 1.430764  | 2.047087  | 0.825843  |
| Random Forest (Tuned)          | 1.468070  | 2.155228  | 0.805292  |
| Lasso Regression               | 1.491463  | 2.224462  | 0.808400  |

- Post-Training Visualization
![pots-training-visualizaion](/output2.png)
- New Input Predicton


## Tech Stack and Tools
- Programming language
    - Python 
- libraries
    - scikit-learn
    - pandas
    - numpy
    - seaborn
    - matplotlib
    - yfinance
- Environment
    - Jupyter Notebook
    - Anaconda
- IDE
    - VSCode

You can install all dependencies via:
```
pip install -r requirements.txt
```

## Usage Instructions
To run this project locally:
1. Clone the repository:
```
git clone https://github.com/charlesakinnurun/chevron-stock.git
cd chevron-stock
```
2. Install required packages
```
pip install -r requirements.txt
```
3. Open the notebook:
```
jupyter notebook model.ipynb

```

## Project Structure
```
chevron-stock/
│
├── model.ipynb  
|── model.py    
|── marketing_campaign.csv  
├── requirements.txt 
├── customer.jpg       
├── output1.png        
├── output2.png        
├── SECURITY.md        
├── CONTRIBUTING.md    
├── CODE_OF_CONDUCT.md 
├── LICENSE
└── README.md          

```
