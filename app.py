import streamlit as st
import pandas as pd
import plotly.express as px
import time
from datetime import datetime
import warnings
from model import HeartDiseaseDetector
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
import io
            

warnings.filterwarnings('ignore')

# Configuration
st.set_page_config(
    page_title="HeartGuard Pro",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Styles */
.stApp {
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
    font-family: 'Inter', sans-serif;
    color: #ffffff;
}

/* Remove default Streamlit styling */
.stApp > header {
    background: transparent;
}

.stApp > .main > div {
    padding-top: 1rem;
}

/* Header Styles */
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2.5rem 2rem;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}

.main-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(45deg, transparent 30%, rgba(255,255,255,0.1) 50%, transparent 70%);
    animation: shimmer 3s infinite;
}

@keyframes shimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

.main-header h1 {
    color: white;
    font-size: 3rem;
    font-weight: 700;
    margin: 0;
    text-shadow: 2px 2px 10px rgba(0,0,0,0.3);
    position: relative;
    z-index: 1;
}

.main-header p {
    color: rgba(255,255,255,0.9);
    font-size: 1.2rem;
    font-weight: 400;
    margin-top: 0.5rem;
    position: relative;
    z-index: 1;
}

.heart-icon {
    font-size: 3rem;
    margin-right: 1rem;
    animation: heartbeat 1.5s infinite;
    display: inline-block;
    filter: drop-shadow(0 0 10px rgba(255, 75, 92, 0.5));
}

@keyframes heartbeat {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.1); }
}

/* Sidebar Styles */
.css-1d391kg {
    background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
}

.sidebar-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem 1rem;
    border-radius: 15px;
    margin-bottom: 1.5rem;
    text-align: center;
    box-shadow: 0 10px 25px rgba(102, 126, 234, 0.2);
}

.sidebar-header h3 {
    color: white;
    margin: 0;
    font-weight: 600;
    font-size: 1.1rem;
}

/* Card Styles */
.metric-card {
    background: linear-gradient(145deg, #1e1e3f 0%, #2a2a4a 100%);
    padding: 1.5rem;
    border-radius: 15px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    text-align: center;
    margin: 1rem 0;
    border: 1px solid rgba(255,255,255,0.1);
    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 40px rgba(0,0,0,0.4);
    border-color: rgba(102, 126, 234, 0.3);
}

.metric-card h3 {
    color: #667eea;
    font-size: 2rem;
    margin: 0.5rem 0;
    font-weight: 600;
}

.metric-card h4 {
    color: #ffffff;
    font-size: 1rem;
    margin: 0;
    font-weight: 500;
    opacity: 0.9;
}

.metric-card p {
    color: #a0a0a0;
    font-size: 0.9rem;
    margin: 0.5rem 0 0 0;
}

/* Prediction Result Styles */
.prediction-result {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    padding: 2.5rem;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 20px 50px rgba(102, 126, 234, 0.4);
    margin: 2rem 0;
    position: relative;
    overflow: hidden;
}

.prediction-result::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
    animation: pulse 4s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 0.3; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(1.05); }
}

.prediction-result h2 {
    font-size: 2.5rem;
    margin: 0;
    font-weight: 700;
    color: white;
    text-shadow: 2px 2px 10px rgba(0,0,0,0.3);
    position: relative;
    z-index: 1;
}

.prediction-result p {
    font-size: 1.1rem;
    margin: 0.5rem 0;
    line-height: 1.6;
    color: rgba(255,255,255,0.95);
    position: relative;
    z-index: 1;
}

/* Risk Level Styles */
.risk-low {
    background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    color: white;
}

.risk-medium {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
}

.risk-high {
    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
    color: white;
}

.risk-extreme {
    background: linear-gradient(135deg, #8b0000 0%, #ff0000 100%);
    color: white;
}

/* Info Cards */
.info-card {
    background: linear-gradient(145deg, #1e1e3f 0%, #2a2a4a 100%);
    padding: 1.2rem;
    border-radius: 12px;
    margin: 0.8rem 0;
    border: 1px solid rgba(255,255,255,0.1);
    color: #ffffff;
    transition: all 0.3s ease;
}

.info-card:hover {
    transform: translateX(5px);
    border-color: rgba(102, 126, 234, 0.3);
    box-shadow: 0 5px 20px rgba(0,0,0,0.2);
}

.tips-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.2rem;
    border-radius: 12px;
    margin: 0.8rem 0;
    color: white;
    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.2);
    transition: all 0.3s ease;
}

.tips-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
}

/* Button Styles */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    padding: 0.8rem 2rem;
    border-radius: 25px;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.3s ease;
    box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    border: 1px solid rgba(255,255,255,0.1);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
}

/* Input Styles */
.stSelectbox > div > div {
    background: linear-gradient(145deg, #1e1e3f 0%, #2a2a4a 100%);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    color: white;
}

.stNumberInput > div > div > input {
    background: linear-gradient(145deg, #1e1e3f 0%, #2a2a4a 100%);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    color: white;
}

.stTextInput > div > div > input {
    background: linear-gradient(145deg, #1e1e3f 0%, #2a2a4a 100%);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    color: white;
}

/* Progress Bar */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
}

/* Metrics */
.metric-container {
    background: linear-gradient(145deg, #1e1e3f 0%, #2a2a4a 100%);
    padding: 1rem;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.1);
    margin: 0.5rem 0;
}

/* Footer */
.footer {
    background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%);
    padding: 2rem;
    text-align: center;
    margin-top: 3rem;
    border-top: 1px solid rgba(255,255,255,0.1);
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: #1a1a2e;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
}

/* Responsive Design */
@media (max-width: 768px) {
    .main-header h1 {
        font-size: 2rem;
    }
    
    .heart-icon {
        font-size: 2rem;
    }
    
    .prediction-result h2 {
        font-size: 2rem;
    }
}
</style>
""", unsafe_allow_html=True)

class HeartGuardApp:
    def __init__(self):
        self.initialize_session_state()
        self.detector = self.load_models()
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        defaults = {
            'prediction_made': False,
            'patient_data': {},
            'risk_percentage': 0,
            'risk_level': '',
            'risk_color': '',
            'risk_message': '',
            'prediction_history': []
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    @st.cache_resource
    def load_models(_self):
        """Load the trained models and preprocessors"""
        try:
            detector = HeartDiseaseDetector()
            detector.load_model(name='svm')
            return detector
        except Exception as e:
            st.error(f"❌ Error loading models: {str(e)}")
            return None
    
    def render_header(self):
        """Render the main header"""
        st.markdown("""
        <div class="main-header">
            <h1><span class="heart-icon">🫀</span>HeartGuard Pro</h1>
            <p>Advanced AI-Powered Heart Disease Risk Assessment</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with input forms"""
        with st.sidebar:
            st.markdown("""
            <div class="sidebar-header">
                <h3>🏥 Patient Information</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Personal Information
            st.subheader("👤 Personal Details")
            patient_name = st.text_input("Patient Name", placeholder="Enter patient name")
            
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input("Age", min_value=1, max_value=120, value=50)
            with col2:
                sex = st.selectbox("Sex", ["Male", "Female"])
            
            # Medical Parameters
            st.subheader("🩺 Medical Parameters")
            chest_pain = st.selectbox("Chest Pain Type", [
                "Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"
            ])
            
            col1, col2 = st.columns(2)
            with col1:
                resting_bp = st.number_input("Resting BP (mmHg)", min_value=0, max_value=300, value=120)
                cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=600, value=200)
                max_hr = st.number_input("Max Heart Rate", min_value=50, max_value=250, value=150)
            
            with col2:
                fasting_bs = st.selectbox("Fasting BS > 120 mg/dl", ["No", "Yes"])
                resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
                exercise_angina = st.selectbox("Exercise Angina", ["No", "Yes"])
            
            # Additional Measurements
            st.subheader("📊 Additional Measurements")
            oldpeak = st.number_input("ST Depression", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
            st_slope = st.selectbox("ST Slope", ["Upsloping", "Flat", "Downsloping"])
            
            # Lifestyle Factors
            st.subheader("🏃‍♂️ Lifestyle Factors")
            col1, col2 = st.columns(2)
            with col1:
                weight = st.number_input("Weight (kg)", min_value=1, max_value=300, value=70)
                height = st.number_input("Height (cm)", min_value=1, max_value=250, value=170)
            with col2:
                smoking = st.selectbox("Smoking", ["Never", "Former", "Current"])
                activity = st.selectbox("Activity Level", ["Low", "Moderate", "High"])
            
            # Calculate and display BMI
            bmi = weight / ((height/100) ** 2)
            bmi_status = self.get_bmi_status(bmi)
            
            st.markdown(f"""
            <div class="metric-container">
                <h4>BMI: {bmi:.1f}</h4>
                <p>Status: {bmi_status}</p>
            </div>
            """, unsafe_allow_html=True)
            
            return {
                'patient_name': patient_name,
                'age': age,
                'sex': sex,
                'chest_pain': chest_pain,
                'resting_bp': resting_bp,
                'cholesterol': cholesterol,
                'max_hr': max_hr,
                'fasting_bs': fasting_bs,
                'resting_ecg': resting_ecg,
                'exercise_angina': exercise_angina,
                'oldpeak': oldpeak,
                'st_slope': st_slope,
                'weight': weight,
                'height': height,
                'smoking': smoking,
                'activity': activity,
                'bmi': bmi
            }
    
    def get_bmi_status(self, bmi):
        """Get BMI status based on value"""
        if bmi < 18.5:
            return "Underweight"
        elif bmi < 25:
            return "Normal"
        elif bmi < 30:
            return "Overweight"
        else:
            return "Obese"
    
    def map_input_values(self, inputs):
        """Map input values to model format"""
        mappings = {
            'chest_pain': {
                "Typical Angina": "TA", "Atypical Angina": "ATA", 
                "Non-Anginal Pain": "NAP", "Asymptomatic": "ASY"
            },
            'resting_ecg': {
                "Normal": "Normal", "ST-T Wave Abnormality": "ST", 
                "Left Ventricular Hypertrophy": "LVH"
            },
            'st_slope': {
                "Upsloping": "Up", "Flat": "Flat", "Downsloping": "Down"
            }
        }
        
        return {
            'Age': inputs['age'],
            'Sex': 'M' if inputs['sex'] == 'Male' else 'F',
            'ChestPainType': mappings['chest_pain'].get(inputs['chest_pain'], "ATA"),
            'RestingBP': inputs['resting_bp'],
            'Cholesterol': inputs['cholesterol'],
            'FastingBS': 1 if inputs['fasting_bs'] == "Yes" else 0,
            'RestingECG': mappings['resting_ecg'].get(inputs['resting_ecg'], "Normal"),
            'MaxHR': inputs['max_hr'],
            'ExerciseAngina': "Y" if inputs['exercise_angina'] == "Yes" else "N",
            'Oldpeak': inputs['oldpeak'],
            'ST_Slope': mappings['st_slope'].get(inputs['st_slope'], "Flat")
        }
    
    def calculate_risk_level(self, risk_percentage):
        """Calculate risk level and styling"""
        if risk_percentage > 95:
            return "Extreme", "risk-extreme", "🚨 Emergency - Immediate medical attention required!"
        elif risk_percentage > 70:
            return "High", "risk-high", "⚠️ High risk - Immediate consultation recommended"
        elif risk_percentage > 30:
            return "Moderate", "risk-medium", "📋 Moderate risk - Regular monitoring advised"
        else:
            return "Low", "risk-low", "✅ Low risk - Continue healthy lifestyle"
    
    def generate_recommendations(self, inputs):
        """Generate personalized recommendations"""
        recommendations = []
        
        # Health-based recommendations
        if inputs['resting_bp'] > 140:
            recommendations.append("🩺 Monitor blood pressure regularly - consider medication consultation")
        if inputs['cholesterol'] > 200:
            recommendations.append("🥗 Adopt heart-healthy diet - reduce saturated fats and cholesterol")
        if inputs['exercise_angina'] == "Yes":
            recommendations.append("🏃‍♂️ Consult cardiologist before starting exercise program")
        if inputs['bmi'] > 25:
            recommendations.append("⚖️ Consider weight management - aim for BMI 18.5-24.9")
        if inputs['smoking'] != "Never":
            recommendations.append("🚭 Quit smoking immediately - single most important change")
        if inputs['max_hr'] < (220 - inputs['age']) :
            recommendations.append("💪 Improve cardiovascular fitness with regular exercise")
        if inputs['activity'] == "Low":
            recommendations.append("🚶‍♂️ Increase physical activity - aim for 150 minutes/week")
        
        # Default recommendations if none specific
        if not recommendations:
            recommendations = [
                "✅ Maintain current healthy lifestyle",
                "📅 Schedule regular health check-ups",
                "🏃‍♂️ Continue regular physical activity",
                "🥗 Follow balanced, heart-healthy diet",
                "😴 Ensure 7-9 hours of quality sleep",
                "🧘‍♂️ Practice stress management techniques"
            ]
        
        return recommendations
    
    def render_health_metrics(self, inputs):
        """Render health metrics sidebar"""
        st.subheader("📊 Health Metrics")
        
        # Blood Pressure Status
        bp_status = "High" if inputs['resting_bp'] > 140 else "Normal" if inputs['resting_bp'] > 100 else "Low"
        bp_color = "#ff6b6b" if bp_status == "High" else "#11998e" if bp_status == "Normal" else "#f093fb"
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>Blood Pressure</h4>
            <h3 style="color: {bp_color};">{inputs['resting_bp']} mmHg</h3>
            <p>{bp_status}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Cholesterol Status
        chol_status = "High" if inputs['cholesterol'] > 240 else "Borderline" if inputs['cholesterol'] > 200 else "Normal"
        chol_color = "#ff6b6b" if chol_status == "High" else "#f093fb" if chol_status == "Borderline" else "#11998e"
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>Cholesterol</h4>
            <h3 style="color: {chol_color};">{inputs['cholesterol']} mg/dl</h3>
            <p>{chol_status}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Heart Rate Status
        hr_status = "Excellent" if inputs['max_hr'] > (220 - inputs['age']) else "Good" if inputs['max_hr'] > (180 - inputs['age']) else "Concerning"
        hr_color = "#11998e" if hr_status == "Excellent" else "#f093fb" if hr_status == "Good" else "#ff6b6b"
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>Max Heart Rate</h4>
            <h3 style="color: {hr_color};">{inputs['max_hr']} bpm</h3>
            <p>{hr_status}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # BMI Status
        bmi_status = self.get_bmi_status(inputs['bmi'])
        bmi_color = "#11998e" if bmi_status == "Normal" else "#f093fb" if bmi_status in ["Overweight", "Underweight"] else "#ff6b6b"
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>BMI</h4>
            <h3 style="color: {bmi_color};">{inputs['bmi']:.1f}</h3>
            <p>{bmi_status}</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_daily_tips(self):
        """Render daily health tips"""
        st.subheader("💡 Daily Health Tips")
        
        tips = [
            "🚶‍♀️ Take 10,000 steps daily for cardiovascular health",
            "💧 Stay hydrated with 8-10 glasses of water",
            "🧘‍♂️ Practice meditation for stress reduction",
            "😴 Maintain 7-9 hours of quality sleep",
            "🥗 Include omega-3 rich foods in diet",
            "🚭 Avoid tobacco and limit alcohol",
            "📱 Use health apps to track progress"
        ]
        
        for tip in tips:
            st.markdown(f"""
            <div class="info-card">
                <p>{tip}</p>
            </div>
            """, unsafe_allow_html=True)
    
    def export_report(self, inputs):
        """Generate and export health report as PDF"""
        if not st.session_state.prediction_made:
            return
        
        st.subheader("📄 Health Report")
    
        if st.button("📋 Generate Comprehensive Report"):
            try:

                # Constants
                COLORS = {
                    'primary': colors.HexColor('#2E86AB'),
                    'secondary': colors.HexColor('#A23B72'),
                    'background': colors.HexColor('#F0F0F0')
                }
            
                # Build report data structure
                report_sections = {
                    'Patient Information': {
                        'Name': inputs['patient_name'],
                        'Age': inputs['age'],
                        'Sex': inputs['sex'],
                        'Assessment Date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    },
                    'Risk Assessment': {
                        'Risk Percentage': f"{st.session_state.risk_percentage:.1f}%",
                        'Risk Level': st.session_state.risk_level,
                        'Risk Message': st.session_state.risk_message
                    },
                    'Vital Signs': {
                        'Resting Blood Pressure': f"{inputs['resting_bp']} mmHg",
                        'Cholesterol Level': f"{inputs['cholesterol']} mg/dl",
                        'Maximum Heart Rate': f"{inputs['max_hr']} bpm",
                        'BMI': f"{inputs['bmi']:.1f}"
                    },
                    'Medical History': {
                        'Chest Pain Type': inputs['chest_pain'],
                        'Fasting Blood Sugar': inputs['fasting_bs'],
                        'Resting ECG': inputs['resting_ecg'],
                        'Exercise Angina': inputs['exercise_angina'],
                        'ST Depression': inputs['oldpeak'],
                        'ST Slope': inputs['st_slope']
                    },
                    'Lifestyle Factors': {
                        'Smoking Status': inputs['smoking'],
                        'Physical Activity': inputs['activity'],
                        'Weight': f"{inputs['weight']} kg",
                        'Height': f"{inputs['height']} cm"
                    }
                }
            
                # Create PDF buffer
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=A4,rightMargin=48, leftMargin=48, topMargin=48, bottomMargin=16)
            
                # Define styles
                base_styles = getSampleStyleSheet()
                title_style = ParagraphStyle('Title', parent=base_styles['Title'],
                                       fontSize=24, spaceAfter=12,
                                       textColor=COLORS['primary'], alignment=TA_CENTER)
            
                heading_style = ParagraphStyle('Heading', parent=base_styles['Heading2'],
                                         fontSize=14, spaceAfter=12,
                                         textColor=COLORS['secondary'])
            
                footer_style = ParagraphStyle('Footer', parent=base_styles['Normal'],
                                        fontSize=8, textColor=colors.grey,
                                        alignment=TA_CENTER)
            
                # Table style
                table_style = TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                    ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('PADDING', (0, 0), (-1, -1), 6),
                    ('LEFTPADDING', (0, 0), (-1, -1), 20),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey),
                    ('BACKGROUND', (0, 0), (0, -1), COLORS['background']),
                ])
            
                # Build content
                story = [
                    Paragraph("HeartGuard Health Report", title_style),
                    Spacer(1, 15)
                ]
            
                # Add sections
                for section_name, section_data in report_sections.items():
                    story.append(Paragraph(section_name, heading_style))
                
                    # Create table from section data
                    table_data = [[f"{key}:", str(value)] for key, value in section_data.items()]
                    table = Table(table_data, colWidths=[2.5*inch, 3.5*inch])
                    table.setStyle(table_style)
                    story.append(table)
                    story.append(Spacer(1, 10))
            
                # Add footer
                story.extend([
                    Spacer(1, 15),
                    Paragraph("This report is generated by HeartGuard AI system for informational purposes only.", footer_style),
                    Paragraph("Please consult with healthcare professionals for medical advice.", footer_style)
                ])
            
                # Generate PDF
                doc.build(story)
                pdf_data = buffer.getvalue()
                buffer.close()
            
                # Create filename
                patient_name = (inputs['patient_name'].replace(' ', '_') 
                            if inputs['patient_name'] else 'Patient')
                filename = f"HeartGuard_Report_{patient_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            
                # Download button
                st.download_button(
                    label="⬇️ Download Report (PDF)",
                    data=pdf_data,
                    file_name=filename,
                    mime="application/pdf"
                )
            
                st.success("✅ PDF Report generated successfully!")
            except Exception as e:
                st.error(f"❌ Error generating PDF report: {str(e)}")
    
    def run(self):
        """Main app execution"""
        # Check if models loaded successfully
        if self.detector is None:
            st.error("❌ Unable to load prediction models. Please check model files.")
            st.stop()
        
        # Render header
        self.render_header()
        
        # Get input data from sidebar
        inputs = self.render_sidebar()
        
        # Main content area
        main_col1, main_col2 = st.columns([2, 1])
        
        with main_col1:
            st.subheader("🔬 Heart Disease Risk Assessment")
            
            # Analysis button
            if st.button("🔍 Analyze Heart Disease Risk", use_container_width=True):
                if not inputs['patient_name']:
                    st.warning("⚠️ Please enter patient name before analysis.")
                    return
                
                with st.spinner("🔄 Analyzing patient data..."):
                    # Progress bar for better UX
                    progress_bar = st.progress(0)
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    try:
                        # Map input values
                        patient_data = self.map_input_values(inputs)
                        
                        # Make prediction
                        result = self.detector.predict_single(patient_data, model_name='svm')
                        risk_percentage = result['probability_disease'] * 100
                        
                        # Calculate risk level
                        risk_level, risk_color, risk_message = self.calculate_risk_level(risk_percentage)
                        
                        # Store results in session state
                        st.session_state.prediction_made = True
                        st.session_state.patient_data = patient_data
                        st.session_state.risk_percentage = risk_percentage
                        st.session_state.risk_level = risk_level
                        st.session_state.risk_color = risk_color
                        st.session_state.risk_message = risk_message
                        
                        # Add to prediction history
                        prediction_record = {
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'patient_name': inputs['patient_name'],
                            'risk_percentage': risk_percentage,
                            'risk_level': risk_level
                        }
                        st.session_state.prediction_history.append(prediction_record)
                        
                        st.success("✅ Analysis completed successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")
                        st.error("Please check your input values and try again.")
            
            # Display results if prediction was made
            if st.session_state.prediction_made:
                # Main prediction result
                st.markdown(f"""
                <div class="prediction-result {st.session_state.risk_color}">
                    <h2>Heart Disease Risk: {st.session_state.risk_percentage:.1f}%</h2>
                    <p><strong>Risk Level:</strong> {st.session_state.risk_level}</p>
                    <p>{st.session_state.risk_message}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Risk factor breakdown
                st.subheader("🎯 Risk Factor Breakdown")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    age_risk = min(inputs['age'] / 80 * 100, 100)
                    st.metric("Age Factor", f"{age_risk:.0f}%", 
                             delta="High" if age_risk > 70 else "Moderate" if age_risk > 40 else "Low")
                
                with col2:
                    bp_risk = min(inputs['resting_bp'] / 200 * 100, 100)
                    st.metric("BP Factor", f"{bp_risk:.0f}%",
                             delta="High" if bp_risk > 70 else "Moderate" if bp_risk > 60 else "Normal")
                
                with col3:
                    chol_risk = min(inputs['cholesterol'] / 400 * 100, 100)
                    st.metric("Cholesterol Factor", f"{chol_risk:.0f}%",
                             delta="High" if chol_risk > 60 else "Moderate" if chol_risk > 50 else "Normal")
                
                # Personalized recommendations
                st.subheader("💡 Personalized Recommendations")
                recommendations = self.generate_recommendations(inputs)
                
                for i, rec in enumerate(recommendations):
                    st.markdown(f"""
                    <div class="tips-card">
                        <p><strong>{i+1}.</strong> {rec}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                
                # Timeline visualization if multiple predictions
                p_name= inputs['patient_name']
                p_history = [
                                record for record in st.session_state.prediction_history
                                if record.get('patient_name') == p_name
                            ]
                if len(p_history) > 1:
                    st.subheader("📈 Risk History")
                    history_df = pd.DataFrame(p_history)
                    
                    fig = px.line(
                        history_df, 
                        x='timestamp', 
                        y='risk_percentage',
                        title=f'Risk Assessment History - {p_name}',
                        markers=True
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='white',
                        title_font_color='white'
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        with main_col2:
            # Health metrics display
            self.render_health_metrics(inputs)
            
            # Daily health tips
            self.render_daily_tips()
        
        # Export report section
        if st.session_state.prediction_made:
            st.markdown("---")
            self.export_report(inputs)
        
        # Emergency contact info
        st.markdown("---")
        st.subheader("🚨 Emergency Information")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="info-card">
                <h4>🚑 Emergency Services</h4>
                <p><strong>Call:</strong> 911 (US/Canada)</p>
                <p><strong>Call:</strong> 102 (Nepal)</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-card">
                <h4>❤️ Heart Attack Signs</h4>
                <p>• Chest pain/discomfort</p>
                <p>• Shortness of breath</p>
                <p>• Nausea, sweating</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="info-card">
                <h4>⚡ Immediate Actions</h4>
                <p>• Call emergency services</p>
                <p>• Take aspirin if available</p>
                <p>• Stay calm and rest</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Footer
        st.markdown("""
        <div class="footer">
            <h4>🫀 HeartGuard Pro</h4>
            <p>Advanced AI-Powered Heart Disease Detection System</p>
            <p><small>© 2025 HeartGuard Pro. For educational purposes only. Always consult healthcare professionals for medical advice.</small></p>
            <p><small>This tool is not a substitute for professional medical diagnosis or treatment.</small></p>
        </div>
        """, unsafe_allow_html=True)

# Initialize and run the app
if __name__ == "__main__":
    app = HeartGuardApp()
    app.run()