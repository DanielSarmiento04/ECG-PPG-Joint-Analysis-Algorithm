# 📁 Project Structure

```
blood_pressure_pipeline/
│
├── README.md                    # 📖 Main project documentation
├── requirements.txt             # 📦 Python dependencies
├── demo_pipeline.py            # 🚀 Complete demonstration script
│
├── src/                        # 🧬 Core pipeline modules
│   ├── __init__.py            
│   ├── data/                   # 📊 Data processing components
│   │   ├── __init__.py
│   │   ├── data_validator.py   # ✅ Signal quality & data validation
│   │   └── feature_extractor.py # 🔧 Enhanced feature engineering  
│   ├── models/                 # 🤖 Machine learning components
│   │   ├── __init__.py
│   │   ├── model_trainer.py    # 🎯 Fixed XGBoost + ensemble training
│   │   └── model_evaluator.py  # 🏥 Clinical validation metrics
│   ├── pipelines/              # 🔄 Pipeline orchestration
│   │   ├── __init__.py
│   │   └── main_pipeline.py    # 🎼 Complete pipeline conductor
│   └── utils/                  # 🛠️ Utility functions
│       ├── __init__.py
│       └── logging_config.py   # 📝 Structured logging setup
│
├── config/                     # ⚙️ Configuration files
│   └── model_config.yaml      # 📄 Model & pipeline configuration
│
├── data/                       # 📁 Data directory (user data goes here)
│   └── (place Final_data_base.xlsx here)
│
├── models/                     # 💾 Saved models & artifacts
│   └── (trained models saved here)
│
├── reports/                    # 📊 Evaluation reports & analysis
│   ├── plots/                  # 📈 Generated visualizations
│   └── (evaluation reports saved here)
│
├── logs/                       # 📝 Pipeline execution logs
│   └── (log files generated here)
│
└── tests/                      # 🧪 Unit & integration tests
    ├── __init__.py
    ├── test_data_validator.py
    ├── test_feature_extractor.py
    ├── test_model_trainer.py
    ├── test_model_evaluator.py
    └── test_pipeline.py
```

## 📋 Component Descriptions

### 🧬 **Core Modules (`src/`)**

#### 📊 **Data Processing (`src/data/`)**
- **`data_validator.py`**: Comprehensive validation for physiological signals and BP values
  - Signal quality assessment
  - Physiological range validation  
  - Outlier detection and cleaning
  - Missing data handling

- **`feature_extractor.py`**: Enhanced feature engineering from ECG/PPG signals
  - Time-domain features (50+ metrics)
  - Frequency-domain analysis
  - Wavelet decomposition features
  - Cross-signal correlation features
  - Heart Rate Variability (HRV) metrics

#### 🤖 **Machine Learning (`src/models/`)**
- **`model_trainer.py`**: Fixed XGBoost training + ensemble methods
  - Enhanced XGBoost with proper regularization
  - Random Forest optimization
  - Ensemble methods (Voting + Stacking)
  - Cross-validation with clinical groups
  - Model persistence and versioning

- **`model_evaluator.py`**: Clinical validation and performance metrics
  - British Hypertension Society (BHS) grading
  - AAMI standard compliance
  - Bland-Altman analysis
  - Clinical accuracy thresholds
  - Comprehensive reporting

#### 🔄 **Pipeline Orchestration (`src/pipelines/`)**
- **`main_pipeline.py`**: Complete pipeline coordination
  - Data loading and validation
  - Feature extraction orchestration
  - Model training coordination
  - Evaluation and reporting
  - Error handling and logging

#### 🛠️ **Utilities (`src/utils/`)**
- **`logging_config.py`**: Structured logging configuration
  - Multiple log levels
  - File and console output
  - Structured formatting
  - Performance tracking

### ⚙️ **Configuration (`config/`)**
- **`model_config.yaml`**: Centralized configuration
  - Model hyperparameters
  - Feature engineering options
  - Evaluation settings
  - Pipeline behavior

### 📁 **Data & Output Directories**
- **`data/`**: Input data storage
- **`models/`**: Trained model artifacts
- **`reports/`**: Evaluation reports and plots
- **`logs/`**: Execution logs and debugging info

### 🧪 **Testing (`tests/`)**
- Unit tests for each component
- Integration tests for full pipeline
- Performance benchmarks
- Data validation tests

## 🚀 **Usage Flow**

1. **Place your data**: Copy `Final_data_base.xlsx` to `data/` directory
2. **Configure**: Modify `config/model_config.yaml` if needed
3. **Run demo**: Execute `python demo_pipeline.py`
4. **Review results**: Check `reports/` for evaluation outputs
5. **Use models**: Load trained models from `models/` directory

## 🔧 **Customization Points**

### **Add New Features**
- Extend `feature_extractor.py` with new signal processing methods
- Update configuration to enable/disable feature groups

### **Add New Models**
- Add new algorithms to `model_trainer.py`
- Update ensemble configurations
- Extend evaluation metrics in `model_evaluator.py`

### **Modify Validation**
- Customize clinical thresholds in `model_evaluator.py`
- Add new validation rules in `data_validator.py`
- Implement custom quality metrics

### **Extend Pipeline**
- Add preprocessing steps to `main_pipeline.py`
- Implement custom data loaders
- Add automated hyperparameter optimization

## 📊 **Data Flow**

```
📄 Final_data_base.xlsx
        ↓
🔍 data_validator.py (validation & cleaning)
        ↓
🔧 feature_extractor.py (enhanced feature engineering)
        ↓
🤖 model_trainer.py (training with fixed XGBoost)
        ↓
🏥 model_evaluator.py (clinical validation)
        ↓
📊 reports/ (comprehensive evaluation)
```

## 🎯 **Key Improvements Over Original**

1. **Modular Architecture**: Separated concerns into focused modules
2. **Enhanced XGBoost**: Fixed performance issues with proper regularization
3. **Clinical Validation**: Added medical-standard evaluation metrics
4. **Data Quality**: Comprehensive validation and cleaning
5. **Configuration Management**: YAML-based configuration system
6. **Ensemble Methods**: Multiple model combination strategies
7. **Comprehensive Logging**: Detailed execution tracking
8. **Production Ready**: Error handling, versioning, and deployment support

## 🔄 **Development Workflow**

1. **Make changes** to individual modules
2. **Test components** using unit tests
3. **Run integration tests** with `demo_pipeline.py`
4. **Review outputs** in `reports/` directory
5. **Commit changes** with proper documentation
6. **Deploy** using production configuration

---

This modular structure transforms the original monolithic script into a maintainable, scalable, and production-ready machine learning pipeline that addresses all identified performance and architectural issues.
