# Diabetes Prediction - Kaggle Playground S5E12

Solution for predicting diabetes  using data for Kaggle Playground Series S5E12.

## Approach

* **Manual Ensemble:** Implements a soft-voting classifier combining XGBoost, LightGBM, and CatBoost with custom preprocessing pipelines.
* **AutoML Studio:** Utilizes AutoGluon optimized for ROC-AUC with high-quality presets and stacking/bagging strategies.

## Project Structure

* **data/:** folder for competition data
* **notebook/:** Jupyter notebooks for the manual ensemble and AutoGluon training.
* **src/:** Python scripts for data loading, preprocessing, and adversarial validation.
* **outputs/:** Stores submission CSVs.
* **docker-compose.yml:** Configuration for the containerized environment.

## Environment & Usage

* **Prerequisites:** Docker and NVIDIA GPU Drivers, competition data in **data/raw** folder .
* **Setup:** Clone the repository and run `docker-compose up` to start the container.
* **Access:** Jupyter Lab is accessible at `http://localhost:8888`.

## Key Techniques

* **Adversarial Validation:** Identifies distribution differences between train and test sets to refine cross-validation.
* **Pseudo-Labeling:** Retrains models using high-confidence predictions from the test set to improve generalization.
* **Feature Engineering:** use field-specific features

## License

This project is open-source and available under the MIT License.