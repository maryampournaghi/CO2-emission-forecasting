# CO2-emission-forecasting
Python scripts comparing ML and LSTM models for short-term and daily CO₂ emission forecasting with SHAP interpretability.
🧠 How to Run This Project

This repository contains Python scripts comparing machine learning (ML) and LSTM models for short-term and daily CO₂ emission forecasting with SHAP interpretability.

🔧 1. Clone or download the repository
git clone https://github.com/maryampournaghi/CO2-emission-forecasting.git
cd CO2-emission-forecasting

📦 2. Install dependencies

It’s recommended to use a virtual environment:

python -m venv venv
source venv/bin/activate       # (on macOS/Linux)
venv\Scripts\activate          # (on Windows)
pip install -r requirements.txt

📊 3. Run the main code

Execute the forecasting script:

python notebooks/CO2_forecasting.py

📁 4. Input data

The main dataset is located in:

data/data.xlsx

📈 5. Output

The script automatically saves:

SHAP summary and feature-importance plots

Model performance comparisons (RMSE, MAE, R²)

Figures in the project root or /results folder if created

🧾 Citation

If you use this code, please cite the related paper:
Pournaghi Keykele, M., & Ravanshadnia, M. (2025). Predicting residential CO₂ emissions: Machine learning and deep learning approaches for short-term forecasting. International Journal of Sustainable Building Technology and Urban Development, 16(3), 373–387. https://doi.org/10.22712/susb.20250024
