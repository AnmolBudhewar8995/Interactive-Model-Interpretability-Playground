# Interactive Model Interpretability Playground

A Streamlit-based web application for interactive machine learning model interpretability. Upload your dataset, train a Random Forest model (classification or regression), and explore various interpretability techniques including SHAP explanations, feature importance, partial dependence plots (PDP), and individual conditional expectation (ICE) plots.

## Features

- **Data Upload**: Support for CSV files (e.g., Iris, Titanic datasets)
- **Automatic Model Training**: Trains Random Forest models based on target variable type
- **Global Feature Importance**: Visualize feature importance rankings
- **SHAP Explanations**: Summary plots and local force plots for model explanations
- **PDP & ICE Plots**: Partial dependence and individual conditional expectation plots
- **Interactive UI**: User-friendly interface with sliders and selectors

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- SHAP
- Matplotlib

## Installation

1. Clone or download this repository.

2. Create a virtual environment (if not already done):
   ```
   python3 -m venv .venv
   ```

3. Activate the virtual environment:
   - On macOS/Linux:
     ```
     source .venv/bin/activate
     ```
   - On Windows:
     ```
     .venv\Scripts\activate
     ```

4. Install the required packages:
   ```
   pip install streamlit pandas numpy scikit-learn shap matplotlib
   ```

## Usage

1. Ensure the virtual environment is activated.

2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

3. Open your browser to the provided URL (usually `http://localhost:8501`).

4. Upload a CSV dataset via the sidebar.

5. Select the target column.

6. Adjust test size and train the model.

7. Explore the interpretability visualizations.

## Example Datasets

- Iris dataset: Classification example
- Titanic dataset: Classification example with mixed data types
- Boston Housing: Regression example

## Contributing

Feel free to submit issues, feature requests, or pull requests.

## License

This project is open-source. Please check the license file for details.
