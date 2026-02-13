# MLOps Pipeline Automation: Titanic Dataset

**Author:** Muhammad Saad Shafqat  
**Role:** AI/ML Engineer  

## Project Overview
This repository contains a fully automated Machine Learning Operations (MLOps) pipeline. The project demonstrates how to orchestrate a complete ML workflow—from data ingestion to model evaluation—using **GNU Make** as the central controller.

## Submission Components
As per the assignment guidelines, this repository includes all 3 required components:

1. **Source Code (`scripts/`):** Modular Python scripts for each phase of the pipeline.
2. **Makefile:** The automation controller ensuring reproducibility.
3. **Generated Outputs:** The finalized artifacts, including:
   - `data/` (Raw and Preprocessed datasets)
   - `features/` (Engineered features)
   - `models/` (Serialized Random Forest model)
   - `results/` (Predictions and Evaluation metrics)

## ⚙️ Prerequisites
To execute this project on a clean system, ensure you have the following installed:
* **Python 3.x**
* **GNU Make**

## Execution Instructions
This project is designed to be 100% reproducible. Manual execution of Python scripts is not required. Use the following `make` commands in your terminal:

**1. Install Dependencies**
```bash
make setup