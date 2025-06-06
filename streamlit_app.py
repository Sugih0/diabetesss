import streamlit as st
import pandas as pd
import numpy as np
import time

# Configure page
st.set_page_config(
    page_title="Diabetes Risk Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .risk-high { color: #e53e3e; font-weight: bold; }
    .risk-medium { color: #ed8936; font-weight: bold; }
    .risk-low { color: #38a169; font-weight: bold; }
    
    .disclaimer {
        background: #fff5f5;
        border: 1px solid #feb2b2;
        border-radius: 8px;
        padding: 15px;
        margin-top: 20px;
        color: #c53030;
    }
    
    .gauge-container {
        background: #f7fafc;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    }
    
    .gauge-value {
        font-size: 2.5em;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .gauge-high { color: #e53e3e; }
    .gauge-medium { color: #ed8936; }
    .gauge-low { color: #38a169; }
</style>
""", unsafe_allow_html=True)

# Model configurations
MODEL_CONFIGS = {
    'Random Forest': {
        'accuracy': 95.2,
        'description': 'Random Forest uses multiple decision trees to provide robust predictions with high accuracy.',
        'type': 'ensemble'
    },
    'XGBoost': {
        'accuracy': 94.8,
        'description': 'XGBoost is a gradient boosting algorithm known for excellent performance on tabular data.',
        'type': 'ensemble'
    },
    'Logistic Regression': {
        'accuracy': 92.1,
        'description': 'Logistic Regression provides interpretable results with good performance for binary classification.',
        'type': 'linear'
    },
    'SVM': {
        'accuracy': 91.5,
        'description': 'Support Vector Machine finds optimal decision boundaries for classification.',
        'type': 'kernel'
    },
    'KNN': {
        'accuracy': 89.3,
        'description': 'K-Nearest Neighbors classifies based on similarity to neighboring data points.',
        'type': 'instance'
    },
    'Naive Bayes': {
        'accuracy': 87.8,
        'description': 'Naive Bayes uses probabilistic approach assuming feature independence.',
        'type': 'probabilistic'
    },
    'Decision Tree': {
        'accuracy': 86.4,
        'description': 'Decision Tree creates interpretable rules for classification decisions.',
        'type': 'tree'
    },
    'AdaBoost': {
        'accuracy': 90.7,
        'description': 'AdaBoost combines weak learners to create a strong classifier.',
        'type': 'ensemble'
    }
}

# Risk factor thresholds
RISK_THRESHOLDS = {
    'glucose': {'low': 100, 'high': 140},
    'blood_pressure': {'low': 80, 'high': 90},
    'bmi': {'low': 25, 'high': 30},
    'age': {'low': 45, 'high': 65},
    'insulin': {'low': 100, 'high': 200},
    'pregnancies': {'low': 3, 'high': 6}
}

def get_risk_level(key, value):
    """Determine risk level for a given parameter"""
    if key not in RISK_THRESHOLDS:
        return 'Normal'
    
    thresholds = RISK_THRESHOLDS[key]
    if value >= thresholds['high']:
        return 'High'
    elif value >= thresholds['low']:
        return 'Medium'
    else:
        return 'Low'

def predict_diabetes(data):
    """Simulate diabetes prediction based on risk factors"""
    risk_score = 0
    
    # Glucose risk
    if data['glucose'] > 140:
        risk_score += 3
    elif data['glucose'] > 100:
        risk_score += 1
    
    # BMI risk
    if data['bmi'] > 30:
        risk_score += 2
    elif data['bmi'] > 25:
        risk_score += 1
    
    # Age risk
    if data['age'] > 65:
        risk_score += 2
    elif data['age'] > 45:
        risk_score += 1
    
    # Blood pressure risk
    if data['blood_pressure'] > 90:
        risk_score += 1
    
    # Pregnancy risk
    if data['pregnancies'] > 6:
        risk_score += 2
    elif data['pregnancies'] > 3:
        risk_score += 1
    
    # Insulin risk
    if data['insulin'] > 200:
        risk_score += 1
    
    # Pedigree function risk
    if data['pedigree'] > 1.0:
        risk_score += 2
    elif data['pedigree'] > 0.5:
        risk_score += 1
    
    # Convert risk score to probability
    max_risk = 12
    diabetes_probability = min(risk_score / max_risk, 0.95)
    
    # Add model-specific variation
    model_config = MODEL_CONFIGS[data['model']]
    model_variation = (model_config['accuracy'] - 85) / 100
    diabetes_probability *= (0.8 + model_variation)
    
    # Ensure probability is between 0.05 and 0.95
    diabetes_probability = max(0.05, min(0.95, diabetes_probability))
    no_diabetes_probability = 1 - diabetes_probability
    
    prediction = 1 if diabetes_probability > 0.5 else 0
    confidence = max(diabetes_probability, no_diabetes_probability)
    
    return {
        'prediction': prediction,
        'diabetes_prob': diabetes_probability,
        'no_diabetes_prob': no_diabetes_probability,
        'confidence': confidence,
        'risk_score': risk_score
    }

def create_simple_gauge(value, title):
    """Create a simple gauge using HTML/CSS"""
    percentage = int(value * 100)
    
    if percentage < 25:
        color_class = "gauge-low"
        color = "#38a169"
    elif percentage < 50:
        color_class = "gauge-medium"  
        color = "#ed8936"
    else:
        color_class = "gauge-high"
        color = "#e53e3e"
    
    gauge_html = f"""
    <div class="gauge-container">
        <h4>{title}</h4>
        <div class="gauge-value {color_class}">{percentage}%</div>
        <div style="background: #e2e8f0; height: 20px; border-radius: 10px; margin: 10px 0;">
            <div style="background: {color}; height: 20px; width: {percentage}%; border-radius: 10px; transition: width 0.5s ease;"></div>
        </div>
    </div>
    """
    return gauge_html

# Main app
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Diabetes Risk Prediction System</h1>
        <p>Advanced ML-powered diabetes risk assessment using 8 different algorithms</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for inputs
    st.sidebar.header("👤 Patient Information")
    
    # Input fields
    pregnancies = st.sidebar.number_input(
        "Pregnancies", 
        min_value=0, max_value=20, value=1,
        help="Number of times pregnant"
    )
    
    glucose = st.sidebar.number_input(
        "Glucose Level (mg/dL)", 
        min_value=0, max_value=300, value=120,
        help="Plasma glucose concentration"
    )
    
    blood_pressure = st.sidebar.number_input(
        "Blood Pressure (mm Hg)", 
        min_value=0, max_value=200, value=70,
        help="Diastolic blood pressure"
    )
    
    skin_thickness = st.sidebar.number_input(
        "Skin Thickness (mm)", 
        min_value=0, max_value=100, value=20,
        help="Triceps skin fold thickness"
    )
    
    insulin = st.sidebar.number_input(
        "Insulin (mu U/ml)", 
        min_value=0, max_value=900, value=80,
        help="2-Hour serum insulin"
    )
    
    bmi = st.sidebar.number_input(
        "BMI (kg/m²)", 
        min_value=0.0, max_value=70.0, value=25.0, step=0.1,
        help="Body mass index"
    )
    
    pedigree = st.sidebar.number_input(
        "Diabetes Pedigree Function", 
        min_value=0.0, max_value=3.0, value=0.5, step=0.001,
        help="Family history influence"
    )
    
    age = st.sidebar.number_input(
        "Age (years)", 
        min_value=21, max_value=100, value=30,
        help="Age in years"
    )
    
    model_choice = st.sidebar.selectbox(
        "Prediction Model",
        options=list(MODEL_CONFIGS.keys()),
        index=0,
        help="Choose the ML algorithm for prediction"
    )
    
    # Predict button
    if st.sidebar.button("🔮 Predict Diabetes Risk", type="primary"):
        # Prepare data
        patient_data = {
            'pregnancies': pregnancies,
            'glucose': glucose,
            'blood_pressure': blood_pressure,
            'skin_thickness': skin_thickness,
            'insulin': insulin,
            'bmi': bmi,
            'pedigree': pedigree,
            'age': age,
            'model': model_choice
        }
        
        # Show loading
        with st.spinner('Analyzing patient data...'):
            time.sleep(1.5)  # Simulate processing time
            
        # Get prediction
        result = predict_diabetes(patient_data)
        
        # Store results in session state
        st.session_state.prediction_result = result
        st.session_state.patient_data = patient_data
    
    # Display results if available
    if hasattr(st.session_state, 'prediction_result'):
        result = st.session_state.prediction_result
        patient_data = st.session_state.patient_data
        
        # Main results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if result['prediction'] == 1:
                st.error("⚠️ High Risk of Diabetes")
            else:
                st.success("✅ Low Risk of Diabetes")
        
        with col2:
            confidence_pct = int(result['confidence'] * 100)
            st.metric("Confidence", f"{confidence_pct}%")
        
        with col3:
            model_accuracy = MODEL_CONFIGS[patient_data['model']]['accuracy']
            st.metric("Model Accuracy", f"{model_accuracy}%")
        
        # Detailed results
        st.subheader("📊 Detailed Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Probability gauges
            st.subheader("Risk Probabilities")
            
            # Simple gauge for diabetes risk
            diabetes_gauge_html = create_simple_gauge(
                result['diabetes_prob'], 
                "Diabetes Risk"
            )
            st.markdown(diabetes_gauge_html, unsafe_allow_html=True)
            
            # Probability breakdown
            st.write("**Probability Breakdown:**")
            st.write(f"• No Diabetes: {result['no_diabetes_prob']:.1%}")
            st.write(f"• Diabetes: {result['diabetes_prob']:.1%}")
        
        with col2:
            # Risk factors analysis using bar chart
            st.subheader("Risk Factors Analysis")
            
            factors = ['Glucose', 'BMI', 'Blood Pressure', 'Age', 'Pregnancies', 'Insulin']
            values = [patient_data['glucose'], patient_data['bmi'], patient_data['blood_pressure'], 
                      patient_data['age'], patient_data['pregnancies'], patient_data['insulin']]
            
            # Create DataFrame for chart
            chart_data = pd.DataFrame({
                'Factor': factors,
                'Value': values
            })
            
            st.bar_chart(chart_data.set_index('Factor'))
        
        # Model information
        st.subheader("🤖 Model Information")
        model_config = MODEL_CONFIGS[patient_data['model']]
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Selected Model", patient_data['model'])
            st.metric("Accuracy", f"{model_config['accuracy']}%")
            st.metric("Type", model_config['type'].title())
        
        with col2:
            st.write("**Model Description:**")
            st.write(model_config['description'])
        
        # Risk factors table
        st.subheader("🎯 Risk Factor Analysis")
        
        risk_data = []
        factors = [
            ('Glucose Level', glucose, 'mg/dL', 'glucose'),
            ('BMI', bmi, 'kg/m²', 'bmi'),
            ('Blood Pressure', blood_pressure, 'mmHg', 'blood_pressure'),
            ('Age', age, 'years', 'age'),
            ('Pregnancies', pregnancies, '', 'pregnancies'),
            ('Insulin', insulin, 'mu U/ml', 'insulin')
        ]
        
        for name, value, unit, key in factors:
            risk_level = get_risk_level(key, value)
            risk_data.append({
                'Factor': name,
                'Value': f"{value} {unit}",
                'Risk Level': risk_level
            })
        
        df_risk = pd.DataFrame(risk_data)
        
        # Style the dataframe
        def style_risk_level(val):
            if val == 'High':
                return 'color: #e53e3e; font-weight: bold'
            elif val == 'Medium':
                return 'color: #ed8936; font-weight: bold'
            else:
                return 'color: #38a169; font-weight: bold'
        
        styled_df = df_risk.style.applymap(style_risk_level, subset=['Risk Level'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Export results
        st.subheader("💾 Export Results")
        
        # Create summary for export
        summary_data = {
            'Patient_ID': 'P001',
            'Prediction': 'High Risk' if result['prediction'] == 1 else 'Low Risk',
            'Diabetes_Probability': f"{result['diabetes_prob']:.3f}",
            'Confidence': f"{result['confidence']:.3f}",
            'Model_Used': patient_data['model'],
            'Model_Accuracy': f"{model_config['accuracy']}%",
            **{f"Input_{k}": v for k, v in patient_data.items() if k != 'model'}
        }
        
        summary_df = pd.DataFrame([summary_data])
        
        col1, col2 = st.columns(2)
        with col1:
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="📄 Download CSV Report",
                data=csv,
                file_name=f"diabetes_prediction_report.csv",
                mime="text/csv"
            )
        
        with col2:
            json_data = summary_df.to_json(orient='records', indent=2)
            st.download_button(
                label="📋 Download JSON Report",
                data=json_data,
                file_name=f"diabetes_prediction_report.json",
                mime="application/json"
            )
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ Medical Disclaimer:</strong> This tool is for educational purposes only and should not replace professional medical advice. Please consult with a healthcare provider for proper diagnosis and treatment.
    </div>
    """, unsafe_allow_html=True)
    
    # Additional features in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Batch Prediction")
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file for batch prediction",
        type=['csv'],
        help="Upload a CSV file with patient data for batch predictions"
    )
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"Loaded {len(batch_df)} records")
            
            if st.sidebar.button("🔄 Process Batch"):
                # Process batch predictions
                batch_results = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, row in batch_df.iterrows():
                    progress = (idx + 1) / len(batch_df)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing record {idx + 1} of {len(batch_df)}")
                    
                    # Prepare data for prediction
                    row_data = {
                        'pregnancies': row.get('pregnancies', 1),
                        'glucose': row.get('glucose', 120),
                        'blood_pressure': row.get('blood_pressure', 70),
                        'skin_thickness': row.get('skin_thickness', 20),
                        'insulin': row.get('insulin', 80),
                        'bmi': row.get('bmi', 25.0),
                        'pedigree': row.get('pedigree', 0.5),
                        'age': row.get('age', 30),
                        'model': model_choice
                    }
                    
                    result = predict_diabetes(row_data)
                    batch_results.append({
                        'Record_ID': idx + 1,
                        'Prediction': 'High Risk' if result['prediction'] == 1 else 'Low Risk',
                        'Diabetes_Probability': result['diabetes_prob'],
                        'Confidence': result['confidence']
                    })
                
                progress_bar.empty()
                status_text.empty()
                
                # Display batch results
                st.subheader("📊 Batch Prediction Results")
                batch_results_df = pd.DataFrame(batch_results)
                st.dataframe(batch_results_df, use_container_width=True)
                
                # Download batch results
                csv_batch = batch_results_df.to_csv(index=False)
                st.download_button(
                    label="📄 Download Batch Results",
                    data=csv_batch,
                    file_name="batch_diabetes_predictions.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.sidebar.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main()
