# Stock Ranking Prediction using FinGAT and Transformer Variants

This repository contains code and resources for stock ranking prediction using the **FinGAT** model and its Transformer-based variant.

📄 **Report Link**: [Project Report](https://www.overleaf.com/read/bckpdknmzhsv#8a4fff)

---

## 🔧 Repository Contents

- **`preprocess.ipynb`**  
  - **First step**: Run this notebook to preprocess raw data located in the `stock_data` directory.  
  - Handles data cleaning, feature engineering, and preparation for model training.  
  - Outputs processed data in `stock_data/processed`.

- **`FinGAT.ipynb`**  
  - Implementation of the original **FinGAT** model.  
  - After preprocessing, run this notebook to train the model.  
  - Saves the trained model as `fingat_model.pth` in the root directory.

- **`FinGAT_Transformer.ipynb`**  
  - Modified version of FinGAT that uses a **Transformer** instead of a GRU for sequential learning.  
  - Requires preprocessed data from `preprocess.ipynb`.

- **`run_model.ipynb`**  
  - Final step: Run this notebook to use the pretrained model for predictions on stock data (Jan 1 to Mar 22, 2025).  
  - Generates evaluation metrics (**R²**, **MRR**, **movement accuracy**, etc.) and saves predictions as a CSV file.

---

## 🚀 Workflow Sequence

Follow these steps to use the repository:

1. **Data Preparation**:  
   - Run `preprocess.ipynb`.  
   - Outputs cleaned and processed data in `stock_data/processed`.

2. **Model Training**:  
   - Run `FinGAT.ipynb`.  
   - Trains the FinGAT model and saves the trained weights as `fingat_model.pth`.

3. **Prediction**:  
   - Run `run_model.ipynb`.  
   - Loads the pretrained model and generates predictions for stock data from Jan 1 to Mar 22, 2025.

---

## 📂 Directory Structure

