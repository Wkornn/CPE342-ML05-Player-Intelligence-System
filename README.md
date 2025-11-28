# Player Intelligence System - CPE342 ML

A comprehensive machine learning system for gaming industry applications, implementing five core tasks: anti-cheat detection, player segmentation, spending prediction, game title detection, and account security monitoring.

## 🎯 Project Overview

This project develops ML solutions for critical gaming industry challenges using ensemble methods, advanced feature engineering, and domain-specific optimizations. Each task addresses real-world gaming platform needs with production-ready models.

## 📋 Tasks & Performance

| Task | Problem Type | Best Model | Performance | Key Technique |
|------|-------------|------------|-------------|---------------|
| **Task 1** | Anti-Cheat Detection | XGBoost Ensemble | 83.84% F2-score | Behavioral feature engineering |
| **Task 2** | Player Segmentation | Voting Classifier | 76% Accuracy | SMOTE balancing |
| **Task 3** | Spending Prediction | Two-Stage Ensemble | MAE $2,967 (30%) | Zero-inflation handling |
| **Task 4** | Game Title Detection | Vision Transformer | Stable convergence | TTA with 7 augmentations |
| **Task 5** | Security Monitoring | Isolation Forest + PCA | 17% anomaly rate | Unsupervised ensemble |

## 🛠️ Technical Stack

- **ML Frameworks**: scikit-learn, XGBoost, CatBoost, LightGBM, PyTorch
- **Computer Vision**: timm (Vision Transformers), torchvision
- **Optimization**: Optuna (Bayesian hyperparameter tuning)
- **Data Processing**: pandas, numpy, SMOTE
- **Visualization**: matplotlib, seaborn

## 📁 Project Structure

```
Player-Intelligence-System---CPE342-ML/
├── Raw/                          # Original development notebooks
│   ├── task1/                    # Anti-cheat detection
│   ├── task2/                    # Player segmentation  
│   ├── task3/                    # Spending prediction
│   ├── task4/                    # Game title detection
│   └── task5/                    # Security monitoring
├── task1/                        # Refined implementations
├── task2/
├── task3/
├── task4/
├── task5/
├── task1_report_sections.md      # Detailed technical reports
├── task2_report_sections.md
├── task3_report_sections.md
├── task4_report_sections.md
├── task5_report_sections.md
└── README.md
```

## 🚀 Key Features

### Advanced ML Techniques
- **Ensemble Methods**: Multiple model combinations for robust predictions
- **Feature Engineering**: Gaming-specific behavioral and temporal features
- **Hyperparameter Optimization**: Optuna-based Bayesian optimization
- **Class Imbalance Handling**: SMOTE, balanced weights, custom thresholds

### Domain-Specific Solutions
- **Two-Stage Modeling**: Handles zero-inflated spending data
- **Behavioral Analysis**: Detects cheating patterns and anomalous behavior
- **Computer Vision**: ViT with advanced augmentation strategies
- **Real-Time Capable**: Optimized for production deployment

## 📊 Results Summary

### Task 1: Anti-Cheat Detection
- **Objective**: Identify cheating players using behavioral patterns
- **Approach**: Ensemble of XGBoost, CatBoost, LightGBM with F2-score optimization
- **Key Insight**: Performance-to-account-age ratios highly predictive of cheating

### Task 2: Player Segmentation  
- **Objective**: Classify players into 4 behavioral segments
- **Approach**: SMOTE-balanced voting classifier with engineered features
- **Key Insight**: Social and spending patterns primary differentiators

### Task 3: Spending Prediction
- **Objective**: Predict monthly player spending amounts
- **Approach**: Two-stage model (CatBoost classifier + LightGBM regressor)
- **Key Insight**: 48.2% zero-spenders require specialized handling

### Task 4: Game Title Detection
- **Objective**: Classify game screenshots into 5 title categories
- **Approach**: Vision Transformer with Mixup/CutMix and 7-fold TTA
- **Key Insight**: Transfer learning + augmentation crucial for small datasets

### Task 5: Account Security Monitoring
- **Objective**: Detect compromised accounts using behavioral anomalies
- **Approach**: Ensemble of Isolation Forest + PCA reconstruction error
- **Key Insight**: Temporal behavioral consistency key security indicator

## 🔧 Installation & Usage

### Prerequisites
```bash
pip install pandas numpy scikit-learn xgboost catboost lightgbm
pip install torch torchvision timm
pip install optuna matplotlib seaborn
pip install imbalanced-learn kaggle
```

### Running the Models
1. **Download Data**: Each notebook includes Kaggle data download functionality
2. **Execute Notebooks**: Run task-specific Jupyter notebooks in order
3. **View Results**: Check generated submission files and performance metrics

### Key Configuration Parameters
- **Ensemble Models**: 200+ Optuna trials for hyperparameter optimization
- **Cross-Validation**: 3-5 fold stratified validation
- **Class Balancing**: SMOTE oversampling and balanced class weights
- **Regularization**: L1/L2 penalties and early stopping

## 👥 Team Contributions

- **Research Lead**: Literature review and pipeline architecture design
- **EDA Specialist**: Comprehensive data analysis across all tasks
- **Tabular ML Developer**: Tasks 1 & 2 implementation and optimization
- **Mixed ML Developer**: Tasks 3 & 4 covering both tabular and computer vision
- **Security ML Developer**: Task 5 unsupervised anomaly detection

## 🎓 Key Learnings

### Technical Insights
- **Ensemble Superiority**: Consistently outperformed single models across all tasks
- **Feature Engineering**: Domain-specific features crucial for gaming applications
- **Data Handling**: Gaming data requires specialized preprocessing (skewness, imbalance)
- **Evaluation Metrics**: Business-aligned metrics (F2-score, MAE) more meaningful

### Gaming Industry Applications
- **Revenue Optimization**: ML-driven spending predictions enable targeted monetization
- **Security & Trust**: Automated cheat and anomaly detection protect game integrity  
- **Player Experience**: Segmentation enables personalized gaming experiences
- **Operational Efficiency**: Automated content classification streamlines operations

## 🔮 Future Improvements

- **Graph Neural Networks**: Model player social networks and guild relationships
- **Real-Time Streaming**: Incorporate live gameplay micro-interactions
- **Multi-Task Learning**: Joint optimization across related gaming problems
- **Federated Learning**: Privacy-preserving distributed model training
- **Explainable AI**: Interpretable models for regulatory compliance

## 📄 License

This project is developed for educational purposes as part of CPE342 Machine Learning coursework.

## 🤝 Contributing

This is an academic project. For questions or discussions about the implementation, please refer to the detailed technical reports in the `task*_report_sections.md` files.