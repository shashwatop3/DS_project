# Stock Ranking Prediction using FinGAT and Transformer Variants

This repository contains code and resources for stock ranking prediction using the FinGAT model and its Transformer-based variant.

📄 **Report Link**: [Project Report](https://www.overleaf.com/read/bckpdknmzhsv#8a4fff)

---

## 🔧 Repository Contents

- **`run_model.ipynb`**  
  Runs the **pretrained model** trained on stock data from _January 1 to March 22, 2025_.  
  Includes:  
  - Evaluation using key metrics (**R²**, **MRR**, **movement accuracy**, etc.)  
  - CSV file generation for predictions.

- **`FinGAT.ipynb`**  
  Implementation of the _original **FinGAT**_ model for stock ranking.

- **`FinGAT_Transformer.ipynb`**  
  Modified version of FinGAT using a **Transformer** instead of a **GRU** for sequential learning.

- **`preprocess.ipynb`**  
  Demonstrates the **data preprocessing** pipeline used before training the models.
