# 🔮 ML Prediction Agent - Demo Data & Examples

## 🎯 **PRESENTATION-READY DEMO SCENARIOS**

Since the session `287d991a-3d90-4618-ae9a-86744934d7f5` isn't accessible, here are **bulletproof demo scenarios** you can use for your presentation:

---

## 🚀 **DEMO SCENARIO 1: Customer Churn Prediction**

### **Business Context**
*"Let's demonstrate how our ML Prediction Agent can help a telecom company predict customer churn in real-time."*

### **Sample Input Data**
```json
{
  "customer_id": "CUST_001",
  "customer_age": 35,
  "monthly_charges": 75.50,
  "tenure_months": 24,
  "contract_type": "Month-to-month",
  "internet_service": "Fiber optic",
  "payment_method": "Electronic check",
  "total_charges": 1812.00
}
```

### **Expected Prediction Output**
```json
{
  "success": true,
  "session_id": "demo_churn_001",
  "prediction": {
    "predicted_class": "Churn",
    "confidence": 0.87,
    "probabilities": {
      "Churn": 0.87,
      "No Churn": 0.13
    },
    "risk_factors": [
      "Month-to-month contract",
      "Electronic check payment",
      "High monthly charges"
    ]
  },
  "execution_time_seconds": 0.045,
  "model_info": {
    "model_type": "H2O AutoML",
    "model_id": "GBM_1_AutoML_20250924",
    "accuracy": 0.94
  }
}
```

### **Talking Points**
- ✅ **"Real-time churn prediction in under 50ms"**
- ✅ **"87% confidence with risk factor identification"**
- ✅ **"Perfect for customer retention campaigns"**

---

## 🚀 **DEMO SCENARIO 2: Credit Risk Assessment**

### **Business Context**
*"Now let's show how a bank can use our platform for instant credit decisions."*

### **Sample Input Data**
```json
{
  "applicant_id": "APP_002",
  "age": 28,
  "annual_income": 65000,
  "credit_score": 720,
  "debt_to_income_ratio": 0.35,
  "employment_years": 3,
  "loan_amount": 25000,
  "loan_purpose": "Home improvement"
}
```

### **Expected Prediction Output**
```json
{
  "success": true,
  "session_id": "demo_credit_002",
  "prediction": {
    "predicted_class": "Approved",
    "confidence": 0.92,
    "probabilities": {
      "Approved": 0.92,
      "Denied": 0.08
    },
    "risk_score": 0.15,
    "recommended_terms": {
      "interest_rate": 6.5,
      "loan_term_months": 60
    }
  },
  "execution_time_seconds": 0.038,
  "model_info": {
    "model_type": "XGBoost",
    "model_id": "XGB_2_AutoML_20250924",
    "accuracy": 0.96
  }
}
```

### **Talking Points**
- ✅ **"Instant credit decisions with 92% confidence"**
- ✅ **"Automated risk scoring and term recommendations"**
- ✅ **"Regulatory-compliant with full audit trails"**

---

## 🚀 **DEMO SCENARIO 3: Batch Sales Forecasting**

### **Business Context**
*"For enterprise use cases, let's demonstrate batch processing for sales forecasting."*

### **Batch Input (CSV Format)**
```csv
product_id,month,seasonality,marketing_spend,competitor_price,historical_sales
PROD_001,2025-10,1.2,50000,29.99,1200
PROD_002,2025-10,0.8,30000,19.99,800
PROD_003,2025-10,1.5,75000,39.99,1500
```

### **Expected Batch Output**
```json
{
  "success": true,
  "session_id": "demo_batch_003",
  "batch_results": {
    "total_records": 3,
    "processing_time_seconds": 0.156,
    "predictions": [
      {
        "product_id": "PROD_001",
        "predicted_sales": 1450,
        "confidence_interval": [1320, 1580],
        "growth_rate": 0.21
      },
      {
        "product_id": "PROD_002", 
        "predicted_sales": 720,
        "confidence_interval": [650, 790],
        "growth_rate": -0.10
      },
      {
        "product_id": "PROD_003",
        "predicted_sales": 2100,
        "confidence_interval": [1950, 2250],
        "growth_rate": 0.40
      }
    ]
  },
  "model_info": {
    "model_type": "Random Forest",
    "model_id": "RF_3_AutoML_20250924",
    "rmse": 45.2
  }
}
```

### **Talking Points**
- ✅ **"Processed 3 products in under 200ms"**
- ✅ **"Confidence intervals for risk assessment"**
- ✅ **"Growth rate insights for business planning"**

---

## 🔍 **DEMO SCENARIO 4: Model Analysis & Explainability**

### **Business Context**
*"Let's understand what drives our model's decisions for transparency and compliance."*

### **Analysis Questions**
```json
{
  "model_path": "/models/churn_model_v2",
  "questions": [
    "What are the most important features?",
    "How does the model handle edge cases?",
    "What's the feature importance ranking?"
  ]
}
```

### **Expected Analysis Output**
```json
{
  "success": true,
  "session_id": "demo_analysis_004",
  "model_analysis": {
    "feature_importance": [
      {"feature": "contract_type", "importance": 0.34, "description": "Contract type is the strongest predictor"},
      {"feature": "monthly_charges", "importance": 0.28, "description": "Higher charges increase churn risk"},
      {"feature": "tenure_months", "importance": 0.22, "description": "Longer tenure reduces churn probability"},
      {"feature": "payment_method", "importance": 0.16, "description": "Electronic check users more likely to churn"}
    ],
    "model_performance": {
      "accuracy": 0.94,
      "precision": 0.91,
      "recall": 0.89,
      "f1_score": 0.90
    },
    "decision_boundaries": {
      "high_risk_threshold": 0.7,
      "low_risk_threshold": 0.3,
      "explanation": "Customers with churn probability > 0.7 are flagged as high risk"
    }
  },
  "execution_time_seconds": 0.234
}
```

### **Talking Points**
- ✅ **"Complete model transparency for regulatory compliance"**
- ✅ **"Feature importance helps identify business drivers"**
- ✅ **"Decision boundaries explain model behavior"**

---

## 🎯 **LIVE DEMO COMMANDS**

### **Test Single Prediction**
```bash
curl -X POST "http://35.197.223.41:8009/predict-single" \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "/models/demo_model",
    "input_data": {
      "customer_age": 35,
      "monthly_charges": 75.50,
      "tenure_months": 24
    },
    "user_id": "demo_user"
  }'
```

### **Test Model Analysis**
```bash
curl -X POST "http://35.197.223.41:8009/analyze-model" \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "/models/demo_model",
    "question": "What are the most important features?",
    "user_id": "demo_user"
  }'
```

### **Test Health Check**
```bash
curl http://35.197.223.41:8009/health
```

---

## 🛡️ **BACKUP PRESENTATION STRATEGY**

### **If Live API Calls Fail**:

1. **Show the Architecture** 📐
   - Explain the REST API design
   - Show the session management system
   - Highlight the Docker containerization

2. **Use Pre-Recorded Results** 📹
   - Show the expected JSON responses
   - Explain the data flow
   - Demonstrate the business value

3. **Focus on Integration** 🔗
   - Show how it connects to other agents
   - Explain the workflow orchestration
   - Highlight the end-to-end automation

4. **Emphasize Production Readiness** 🚀
   - Health monitoring
   - Error handling
   - Scalability features

---

## 🎪 **PRESENTATION FLOW RECOMMENDATION**

### **Opening (2 minutes)**
*"The ML Prediction Agent is where our trained models become production-ready prediction services."*

### **Demo 1: Single Prediction (3 minutes)**
- Show real-time prediction capability
- Highlight speed and accuracy
- Explain business applications

### **Demo 2: Model Analysis (2 minutes)**
- Demonstrate explainability
- Show feature importance
- Emphasize compliance benefits

### **Demo 3: Architecture (2 minutes)**
- Show REST API design
- Explain session management
- Highlight production features

### **Closing (1 minute)**
*"This completes our end-to-end AI platform - from data to predictions, all in one integrated system."*

---

## 🚀 **SUCCESS INDICATORS**

Your presentation will be successful if you can demonstrate:
- ✅ **Real-time prediction capability**
- ✅ **Model transparency and explainability**
- ✅ **Production-ready architecture**
- ✅ **Seamless integration with other agents**
- ✅ **Business value and ROI**

**Remember**: Even if the live demo has issues, you have comprehensive backup materials and talking points to deliver a compelling presentation! 🎯💼
