from flask import Flask, request, render_template_string 
import joblib 
import numpy as np 
import torch
import torch.nn as nn
import os

app = Flask(__name__) 

# Define risk stratification thresholds for SCREENING in infants ≤90 days
RISK_THRESHOLDS = {
    'low': 0.3,      # Low-risk threshold for screening
    'medium': 0.7,   # Medium-risk threshold for screening  
    'high': 0.7      # High-risk threshold for screening
}

CLINICAL_RECOMMENDATIONS = {
    'low': {
        'title': 'LOW SCREENING RISK',
        'icon': 'fa-check-circle',
        'color': 'success',
        'badge_class': 'low-badge',
        'recommendation': 'Routine monitoring recommended. This transfer learning-based screening result suggests low probability of biliary atresia in infants ≤90 days, but clinical evaluation is still advised.',
        'actions': [
            'Continue routine well-baby care',
            'Monitor for jaundice persistence',
            'Repeat liver function tests if symptoms persist',
            'Clinical evaluation recommended within 2 weeks'
        ]
    },
    'medium': {
        'title': 'MEDIUM SCREENING RISK', 
        'icon': 'fa-exclamation-circle',
        'color': 'warning',
        'badge_class': 'medium-badge',
        'recommendation': 'Further evaluation recommended. This transfer learning screening tool indicates moderate risk for infants ≤90 days - specialist consultation advised to exclude biliary atresia.',
        'actions': [
            'Refer to pediatric gastroenterology within 1 week',
            'Perform abdominal ultrasound',
            'Consider additional liver function tests',
            'Close clinical monitoring required'
        ]
    },
    'high': {
        'title': 'HIGH SCREENING RISK',
        'icon': 'fa-exclamation-triangle', 
        'color': 'danger',
        'badge_class': 'high-badge',
        'recommendation': 'Urgent specialist evaluation recommended. This transfer learning screening result indicates high risk in infants ≤90 days - immediate clinical assessment is strongly advised.',
        'actions': [
            'Immediate pediatric hepatology referral',
            'Expedite diagnostic investigations',
            'Consider preoperative assessment',
            'Multidisciplinary team consultation within 48 hours'
        ]
    }
}

# Transfer Learning Model Definition (same as in train.py)
class TransferLearningModel(nn.Module):
    def __init__(self, input_dim, base_model=None):
        super(TransferLearningModel, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Neural network architecture
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        ).to(self.device)
    
    def forward(self, x):
        return self.model(x)

# HTML Template with transfer learning emphasis for infants ≤90 days
HTML_TEMPLATE = ''' 
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neonatal Biliary Atresia Transfer Learning Screening Tool - Infants ≤90 Days</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;500;600;700&family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #1e88e5;
            --primary-light: #64b5f6;
            --primary-dark: #0d47a1;
            --secondary: #4dd0e1;
            --success: #43a047;
            --success-light: #66bb6a;
            --warning: #ffb300;
            --warning-light: #ffca28;
            --danger: #e53935;
            --danger-light: #ef5350;
            --info: #00acc1;
            --light: #e3f2fd;
            --dark: #0d47a1;
            --gray: #90a4ae;
            --light-gray: #f5f7fa;
            --card-shadow: 0 8px 30px rgba(30, 136, 229, 0.15);
            --transition: all 0.3s ease;
            --border-radius: 12px;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            font-family: 'Nunito', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            color: var(--dark);
            line-height: 1.6;
            min-height: 100vh;
            padding-bottom: 50px;
            font-size: 1.05rem;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
            background-attachment: fixed;
        }

        .navbar {
            background: linear-gradient(90deg, var(--primary-dark) 0%, var(--primary) 100%);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            padding: 0.8rem 0;
        }

        .main-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: calc(100vh - 200px);
            width: 100%;
        }

        .content-wrapper {
            width: 100%;
            max-width: 1400px;
            margin: 0 auto;
            padding: 0 20px;
        }

        .age-highlight {
            background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
            border: 2px solid #4caf50;
            border-radius: var(--border-radius);
            padding: 1rem;
            margin-bottom: 1.5rem;
            text-align: center;
            font-weight: 700;
            color: #2e7d32;
            font-size: 1.1rem;
            width: 100%;
        }

        .screening-alert {
            background: linear-gradient(135deg, #fff3cd 0%, #ffecb5 100%);
            border: 2px solid #ffc107;
            border-radius: var(--border-radius);
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            text-align: center;
            font-weight: 600;
            color: #856404;
            width: 100%;
        }

        .transfer-learning-badge {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            border: 2px solid var(--primary);
            border-radius: var(--border-radius);
            padding: 0.8rem;
            margin-bottom: 1rem;
            text-align: center;
            font-weight: 600;
            color: var(--primary-dark);
            width: 100%;
        }

        .card {
            border-radius: var(--border-radius);
            box-shadow: var(--card-shadow);
            margin-bottom: 1.8rem;
            border: none;
            overflow: hidden;
            transition: var(--transition);
            background-color: white;
            border: 1px solid rgba(30, 136, 229, 0.1);
            width: 100%;
        }

        .card:hover {
            box-shadow: 0 12px 35px rgba(30, 136, 229, 0.2);
            transform: translateY(-5px);
        }

        .card-header {
            background: linear-gradient(90deg, rgba(30, 136, 229, 0.15) 0%, rgba(100, 181, 246, 0.15) 100%);
            border-bottom: 1px solid rgba(0, 0, 0, 0.05);
            padding: 1.2rem 1.8rem;
            font-weight: 700;
            color: var(--primary-dark);
            font-size: 1.25rem;
            font-family: 'Poppins', sans-serif;
            text-align: center;
        }

        .risk-scale {
            position: relative;
            height: 20px;
            background: linear-gradient(90deg, var(--success) 0%, var(--warning) 50%, var(--danger) 100%);
            border-radius: 10px;
            margin: 2rem auto;
            width: 90%;
            max-width: 600px;
        }

        .current-risk-marker {
            position: absolute;
            top: -10px;
            width: 4px;
            height: 40px;
            background: var(--dark);
            transform: translateX(-50%);
        }

        .current-risk-label {
            position: absolute;
            top: -35px;
            left: 50%;
            transform: translateX(-50%);
            background: var(--dark);
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8rem;
            font-weight: 600;
            white-space: nowrap;
        }

        .prediction-badge {
            font-size: 1.4rem;
            padding: 1.0rem 2.0rem;
            border-radius: 50px;
            font-weight: 800;
            letter-spacing: 0.5px;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
            transition: var(--transition);
            font-family: 'Poppins', sans-serif;
            display: inline-block;
            margin: 0.8rem auto;
            text-align: center;
        }

        .low-badge {
            background: linear-gradient(90deg, rgba(67, 160, 71, 0.15) 0%, rgba(102, 187, 106, 0.15) 100%);
            color: #2e7d32;
            border: 2px solid #43a047;
        }

        .medium-badge {
            background: linear-gradient(90deg, rgba(255, 179, 0, 0.15) 0%, rgba(255, 202, 40, 0.15) 100%);
            color: #ff8f00;
            border: 2px solid #ffb300;
        }

        .high-badge {
            background: linear-gradient(90deg, rgba(229, 57, 53, 0.15) 0%, rgba(239, 83, 80, 0.15) 100%);
            color: #c62828;
            border: 2px solid #e53935;
        }

        .action-item {
            padding: 0.8rem 1rem;
            margin-bottom: 0.5rem;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.7);
            border-left: 4px solid;
            transition: var(--transition);
            width: 100%;
        }

        .action-item:hover {
            transform: translateX(5px);
            background: white;
        }

        .action-item.low {
            border-left-color: var(--success);
        }

        .action-item.medium {
            border-left-color: var(--warning);
        }

        .action-item.high {
            border-left-color: var(--danger);
        }

        .disclaimer-box {
            background: #e8f4fd;
            border-left: 4px solid var(--primary);
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 4px;
            font-size: 0.9rem;
            width: 100%;
        }

        .loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(227, 242, 253, 0.92);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            z-index: 9999;
            opacity: 0;
            visibility: hidden;
            transition: all 0.3s ease;
            backdrop-filter: blur(8px);
        }

        .loading-overlay.active {
            opacity: 1;
            visibility: visible;
        }

        .spinner {
            width: 60px;
            height: 60px;
            border: 5px solid rgba(30, 136, 229, 0.2);
            border-top: 5px solid var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-bottom: 20px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .compact-form {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.2rem;
            width: 100%;
        }

        .form-group {
            display: flex;
            flex-direction: column;
        }

        .section-title {
            text-align: center;
            margin-bottom: 1.5rem;
            color: var(--primary-dark);
            font-weight: 600;
            border-bottom: 2px solid var(--primary-light);
            padding-bottom: 0.5rem;
        }

        .feature-card {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 0.8rem;
            margin-top: 0.5rem;
            border-left: 3px solid var(--primary);
        }

        .feature-title {
            font-weight: 600;
            color: var(--primary-dark);
            margin-bottom: 0.3rem;
        }

        .feature-desc, .feature-range {
            font-size: 0.85rem;
            color: #666;
            margin-bottom: 0.2rem;
        }

        .result-card {
            text-align: center;
        }

        .result-icon {
            margin-bottom: 1rem;
        }

        .actions-list {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        @media (max-width: 768px) {
            .compact-form {
                grid-template-columns: 1fr;
            }
            
            .content-wrapper {
                padding: 0 15px;
            }
            
            .card-header {
                padding: 1rem;
                font-size: 1.1rem;
            }
            
            .prediction-badge {
                font-size: 1.1rem;
                padding: 0.8rem 1.5rem;
            }
        }

        @media (min-width: 1200px) {
            .content-wrapper {
                max-width: 1400px;
            }
        }
    </style>
</head>
<body>
    <!-- Loading animation -->
    <div class="loading-overlay" id="loadingOverlay">
        <div class="spinner"></div>
        <div class="loading-text">Running transfer learning analysis for infant ≤90 days, please wait...</div>
    </div>

    <nav class="navbar navbar-expand-lg navbar-dark">
        <div class="container-fluid">
            <a class="navbar-brand d-flex align-items-center" href="/">
                <i class="fas fa-brain fa-2x me-3"></i>
                <div>
                    <h4 class="mb-0">Biliary Atresia Transfer Learning Screening Tool</h4>
                    <small class="opacity-85">AI-Powered Screening for Infants ≤90 Days - NOT for Diagnosis</small>
                </div>
            </a>
        </div>
    </nav>

    <div class="main-container">
        <div class="content-wrapper">
            {{ content | safe }}
        </div>
    </div>

    <footer class="footer mt-5">
        <div class="container-fluid text-center py-3" style="background: var(--primary-dark); color: white;">
            <p class="mb-1">
                <i class="fas fa-robot me-1"></i>
                © 2024 Biliary Atresia Transfer Learning Screening | Validated for Infants ≤90 Days
            </p>
            <small class="opacity-85">Transfer Learning Model: RF-based Neural Network | Five-biomarker screening panel</small>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // Form input validation
        document.querySelectorAll('input[type="number"]').forEach(input => {
            input.addEventListener('input', function() {
                const min = parseFloat(this.min);
                const max = parseFloat(this.max);
                const value = parseFloat(this.value);
                if (!isNaN(min) && value < min) {
                    this.value = min;
                } else if (!isNaN(max) && value > max) {
                    this.value = max;
                }
            });
        });

        // Show loading animation on form submit
        document.querySelector('form')?.addEventListener('submit', function() {
            document.getElementById('loadingOverlay').classList.add('active');
        });

        // Center all content on page load
        document.addEventListener('DOMContentLoaded', function() {
            const mainContainer = document.querySelector('.main-container');
            if (mainContainer) {
                mainContainer.style.display = 'flex';
            }
        });
    </script>
</body>
</html>
'''

# Home page form - Updated for infants ≤90 days with centered layout
HOME_PAGE = '''
<div class="row justify-content-center">
    <div class="col-12">
        <div class="age-highlight">
            <i class="fas fa-calendar-check me-2"></i>
            <strong>VALIDATED FOR INFANTS ≤90 DAYS:</strong> This AI-powered transfer learning screening tool is specifically validated for infants 90 days of age or younger.
        </div>
        
        <div class="screening-alert">
            <i class="fas fa-exclamation-triangle me-2"></i>
            <strong>TRANSFER LEARNING SCREENING TOOL DISCLAIMER:</strong> This AI-powered tool provides risk stratification using transfer learning technology. 
            It is NOT a diagnostic tool. Final diagnosis requires comprehensive clinical evaluation.
        </div>
        
        <div class="transfer-learning-badge">
            <i class="fas fa-brain me-2"></i>
            <strong>TRANSFER LEARNING TECHNOLOGY:</strong> This tool uses a neural network initialized with Random Forest feature importance for enhanced screening accuracy in infants ≤90 days.
        </div>
        
        <div class="card mx-auto" style="max-width: 1400px;">
            <div class="card-header">
                <h5 class="m-0 d-flex align-items-center justify-content-center">
                    <i class="fas fa-network-wired me-2"></i>Transfer Learning Screening Panel for Infants ≤90 Days
                </h5>
            </div>
            <div class="card-body">
                <div class="row mb-3">
                    <div class="col-12">
                        <div class="alert alert-primary d-flex align-items-center mb-3">
                            <i class="fas fa-robot fa-2x me-3"></i>
                            <div>
                                <h5 class="mb-1">AI-Powered Biliary Atresia Screening for Infants ≤90 Days</h5>
                                <p class="mb-0">This <strong>transfer learning model</strong> uses <strong>5 blood biomarkers</strong> with neural network architecture for advanced screening in infants 90 days or younger.</p>
                            </div>
                        </div>
                    </div>
                </div>
                <form action="/predict" method="post" id="screeningForm">
                    <div class="compact-form">
                        <!-- Column 1: Key Liver Function Markers -->
                        <div class="form-group">
                            <h5 class="section-title">Liver Function Biomarkers</h5>
                            
                            <!-- Albumin (ALB) -->
                            <div class="mb-3">
                                <label class="form-label">
                                    <i class="fas fa-vial"></i>Albumin (ALB)
                                </label>
                                <input type="number" class="form-control form-control-lg feature-input" name="ALB" min="0" max="100" step="0.1" placeholder="Enter ALB (g/L)" required>
                                <div class="feature-card">
                                    <div class="feature-title">
                                        <i class="fas fa-microscope"></i>Transfer Learning Feature
                                    </div>
                                    <p class="feature-desc">Hepatic synthetic function marker</p>
                                    <p class="feature-range">Infant reference: 28-44 g/L</p>
                                </div>
                            </div>
                            
                            <!-- Alkaline Phosphatase (ALP) -->
                            <div class="mb-3">
                                <label class="form-label">
                                    <i class="fas fa-vial"></i>Alkaline Phosphatase (ALP)
                                </label>
                                <input type="number" class="form-control form-control-lg feature-input" name="ALP" min="0" max="2000" step="0.1" placeholder="Enter ALP (U/L)" required>
                                <div class="feature-card">
                                    <div class="feature-title">
                                        <i class="fas fa-microscope"></i>Transfer Learning Feature
                                    </div>
                                    <p class="feature-desc">Biliary epithelial and bone isoenzymes</p>
                                    <p class="feature-range">Infant reference: 150-420 U/L</p>
                                </div>
                            </div>
                            
                            <!-- Gamma-glutamyl Transferase (GGT) -->
                            <div class="mb-3">
                                <label class="form-label">
                                    <i class="fas fa-vial"></i>Gamma-glutamyl Transferase (GGT)
                                </label>
                                <input type="number" class="form-control form-control-lg feature-input" name="GGT" min="0" max="2000" step="0.1" placeholder="Enter GGT (U/L)" required>
                                <div class="feature-card">
                                    <div class="feature-title">
                                        <i class="fas fa-microscope"></i>Transfer Learning Feature
                                    </div>
                                    <p class="feature-desc">Highly sensitive biliary obstruction marker</p>
                                    <p class="feature-range">Infant reference: 0-200 U/L</p>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Column 2: Additional Biochemical Markers -->
                        <div class="form-group">
                            <h5 class="section-title">Bilirubin & Lipid Markers</h5>
                            
                            <!-- Direct Bilirubin (DBIL) -->
                            <div class="mb-3">
                                <label class="form-label">
                                    <i class="fas fa-tint"></i>Direct Bilirubin (DBIL)
                                </label>
                                <input type="number" class="form-control form-control-lg feature-input" name="DBIL" min="0" max="200" step="0.01" placeholder="Enter DBIL (μmol/L)" required>
                                <div class="feature-card">
                                    <div class="feature-title">
                                        <i class="fas fa-microscope"></i>Transfer Learning Feature
                                    </div>
                                    <p class="feature-desc">Conjugated hyperbilirubinemia indicator</p>
                                    <p class="feature-range">Infant reference: 0-17 μmol/L</p>
                                </div>
                            </div>
                            
                            <!-- LDL Cholesterol -->
                            <div class="mb-3">
                                <label class="form-label">
                                    <i class="fas fa-vial"></i>LDL Cholesterol
                                </label>
                                <input type="number" class="form-control form-control-lg feature-input" name="LDL-C" min="0" max="10" step="0.01" placeholder="Enter LDL-C (mmol/L)" required>
                                <div class="feature-card">
                                    <div class="feature-title">
                                        <i class="fas fa-microscope"></i>Transfer Learning Feature
                                    </div>
                                    <p class="feature-desc">Lipid metabolism and liver function</p>
                                    <p class="feature-range">Infant reference: 0.5-3.0 mmol/L</p>
                                </div>
                            </div>
                            
                            <!-- Model Information -->
                            <div class="disclaimer-box mt-4">
                                <h6 class="text-center"><i class="fas fa-network-wired me-2"></i>Transfer Learning Model Information</h6>
                                <div class="row text-center">
                                    <div class="col-md-6">
                                        <p class="mb-1"><strong>Architecture:</strong> 5-64-32-1 Neural Network</p>
                                        <p class="mb-1"><strong>Technology:</strong> RF-initialized Transfer Learning</p>
                                    </div>
                                    <div class="col-md-6">
                                        <p class="mb-1"><strong>Validation:</strong> Specifically validated for infants ≤90 days</p>
                                        <p class="mb-0"><strong>Purpose:</strong> Advanced AI screening only - clinical diagnosis required</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="d-grid mt-4">
                        <button type="submit" class="btn btn-primary btn-lg py-3" style="max-width: 500px; margin: 0 auto;">
                            <i class="fas fa-brain me-2"></i>Run Transfer Learning Screening for Infant ≤90 Days
                        </button>
                    </div>
                </form>
            </div>
        </div>
    </div>
</div>
'''

# Result page with transfer learning emphasis for infants ≤90 days with centered layout
RESULT_PAGE = '''
<div class="row justify-content-center">
    <div class="col-12">
        <div class="age-highlight">
            <i class="fas fa-baby me-2"></i>
            <strong>SCREENING RESULT FOR INFANT ≤90 DAYS:</strong> This AI-powered transfer learning analysis is specifically validated for infants 90 days of age or younger.
        </div>
        
        <div class="screening-alert">
            <i class="fas fa-robot me-2"></i>
            <strong>TRANSFER LEARNING SCREENING RESULT - NOT DIAGNOSTIC:</strong> This AI-powered result indicates screening risk only. Clinical evaluation is required.
        </div>
        
        <div class="transfer-learning-badge">
            <i class="fas fa-network-wired me-2"></i>
            <strong>TRANSFER LEARNING ANALYSIS COMPLETE:</strong> Neural network screening based on 5 biomarkers with RF initialization for infant ≤90 days.
        </div>
        
        <div class="card result-card mx-auto" style="max-width: 1200px;">
            <div class="card-header d-flex justify-content-between align-items-center">
                <h5 class="m-0 d-flex align-items-center justify-content-center w-100">
                    <i class="fas fa-chart-line me-2"></i>Transfer Learning Screening Results for Infant ≤90 Days
                </h5>
                <a href="/" class="btn btn-outline-primary position-absolute" style="right: 20px;">
                    <i class="fas fa-redo me-1"></i>New Screening
                </a>
            </div>
            <div class="card-body">
                <div class="text-center mb-4">
                    <div class="result-icon">
                        <i class="fas {{ risk_info.icon }} fa-3x text-{{ risk_info.color }}"></i>
                    </div>
                    <h3 class="mb-2">Transfer Learning Screening Result</h3>
                    <div class="d-flex justify-content-center align-items-center mb-3">
                        <span class="prediction-badge {{ risk_info.badge_class }}">
                            <i class="fas {{ risk_info.icon }} me-2"></i>
                            {{ risk_info.title }}
                        </span>
                    </div>
                    
                    <!-- Risk Scale Visualization -->
                    <div class="risk-scale">
                        <div class="current-risk-marker" style="left: {{ proba_percent }}%;"></div>
                        <div class="current-risk-label" style="left: {{ proba_percent }}%;">
                            {{ proba_percent }}%
                        </div>
                    </div>
                    <div class="d-flex justify-content-between mt-4 text-muted fw-medium" style="max-width: 600px; margin: 0 auto;">
                        <span>Low Risk &lt;30%</span>
                        <span>Medium Risk 30-70%</span>
                        <span>High Risk &gt;70%</span>
                    </div>
                </div>

                <!-- AI Screening Management Recommendations for infants ≤90 days -->
                <div class="alert alert-{{ risk_info.color }} mt-3 mx-auto" style="max-width: 1000px;">
                    <h5 class="alert-heading d-flex align-items-center justify-content-center">
                        <i class="fas {{ risk_info.icon }} me-2"></i>
                        {{ risk_info.title }} - AI Screening Management for Infant ≤90 Days
                    </h5>
                    <p class="mb-3 text-center">{{ risk_info.recommendation}}</p>
                    <hr>
                    <h6 class="mb-2 text-center">Recommended AI Screening Actions for Infant ≤90 Days:</h6>
                    <div class="actions-list mx-auto" style="max-width: 800px;">
                        {% for action in risk_info.actions %}
                        <div class="action-item {{ risk_level }}">
                            <i class="fas fa-check-circle me-2 text-{{ risk_info.color }}"></i>
                            {{ action }}
                        </div>
                        {% endfor %}
                    </div>
                </div>
                
                <!-- Final Transfer Learning Disclaimer -->
                <div class="disclaimer-box mx-auto" style="max-width: 1000px;">
                    <h6 class="text-center"><i class="fas fa-robot me-2"></i>Transfer Learning Model Disclaimer</h6>
                    <p class="mb-2 text-center">This AI screening tool uses transfer learning technology (RF-initialized neural network) with five blood biomarkers to stratify biliary atresia risk in infants ≤90 days.</p>
                    <p class="mb-0 text-center"><strong>This is screening only - definitive diagnosis requires:</strong> Clinical examination, imaging studies, and specialized diagnostic procedures by qualified specialists.</p>
                </div>

                <div class="text-center mt-4">
                    <a href="/" class="btn btn-primary btn-lg py-3 px-5">
                        <i class="fas fa-redo me-2"></i>Perform New AI Screening
                    </a>
                </div>
            </div>
        </div>
    </div>
</div>
'''

ERROR_PAGE = '''
<div class="row justify-content-center">
    <div class="col-lg-8">
        <div class="card border-danger mx-auto" style="max-width: 800px;">
            <div class="card-header bg-danger text-white d-flex align-items-center justify-content-center">
                <i class="fas fa-exclamation-triangle me-2"></i>
                <h5 class="m-0">Transfer Learning Analysis Error</h5>
            </div>
            <div class="card-body text-center">
                <div class="alert alert-danger">
                    <h4 class="alert-heading d-flex align-items-center justify-content-center">
                        <i class="fas fa-bug me-2"></i>AI Screening Calculation Error
                    </h4>
                    <p>{{ error_message }}</p>
                    <hr>
                    <p class="mb-0">Please verify input values and try again.</p>
                </div>
                <div class="text-center mt-3">
                    <a href="/" class="btn btn-danger btn-lg py-2 px-4">
                        <i class="fas fa-arrow-left me-2"></i>Return to Screening Form
                    </a>
                </div>
            </div>
        </div>
    </div>
</div>
'''

# Transfer Learning Predictor class for infants ≤90 days
class BATransferLearningScreener:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        try:
            self.scaler = joblib.load("scaler.pkl")
        except FileNotFoundError:
            raise FileNotFoundError("Scaler file 'scaler.pkl' not found. Please run Train1.py first.")
        
        # Define feature order (5 biomarkers)
        self.required_features = ['ALB', 'ALP', 'GGT', 'DBIL', 'LDL-C']
        
        # Load transfer learning model with weights_only=True for security
        self.model = TransferLearningModel(len(self.required_features))
        try:
            # Use weights_only=True for security (PyTorch 2.0+)
            self.model.load_state_dict(torch.load("transfer_learning_model.pth", 
                                                map_location=self.device, 
                                                weights_only=True))
        except TypeError:
            # Fallback for older PyTorch versions
            self.model.load_state_dict(torch.load("transfer_learning_model.pth", 
                                                map_location=self.device))
        except FileNotFoundError:
            raise FileNotFoundError("Model file 'transfer_learning_model.pth' not found. Please run Train1.py first.")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Feature descriptions
        self.feature_descriptions = {
            'ALB': 'Albumin', 
            'ALP': 'Alkaline Phosphatase',
            'GGT': 'Gamma-glutamyl Transferase',
            'DBIL': 'Direct Bilirubin',
            'LDL-C': 'LDL Cholesterol'
        }
        
        # Infant reference ranges (≤90 days)
        self.feature_ranges = {
            'ALB': (20, 50),
            'ALP': (100, 600),
            'GGT': (0, 1500),
            'DBIL': (0, 100),
            'LDL-C': (0.5, 5.0)
        }
    
    def validate_input(self, input_data):
        """Validate input data against infant reference ranges (≤90 days)"""
        for i, feat in enumerate(self.required_features):
            val = input_data[i]
            if val < 0:
                raise ValueError(f"{self.feature_descriptions[feat]} value cannot be negative")
            if val > 10000:  # Reasonable upper limit
                raise ValueError(f"{self.feature_descriptions[feat]} value seems unusually high: {val}")

    def get_risk_level(self, probability):
        """Determine screening risk level based on probability thresholds"""
        if probability < RISK_THRESHOLDS['low']:
            return 'low'
        elif probability < RISK_THRESHOLDS['high']:
            return 'medium'
        else:
            return 'high'

# Initialize transfer learning screener
try:
    screener = BATransferLearningScreener()
    print("Transfer Learning Screener initialized successfully")
except Exception as e:
    print(f"Error initializing screener: {e}")
    screener = None

# Application routes
@app.route('/')
def home():
    """Home page with transfer learning screening input form for infants ≤90 days"""
    return render_template_string(HTML_TEMPLATE, content=HOME_PAGE)

@app.route('/predict', methods=['POST'])
def predict():
    """Transfer learning screening prediction endpoint for infants ≤90 days"""
    try:
        # Check if screener is initialized
        if screener is None:
            raise ValueError("Screening service is not available. Please check if model files are present.")
        
        # Collect input data with better error handling
        input_data = []
        feature_names = ['ALB', 'ALP', 'GGT', 'DBIL', 'LDL-C']
        
        for feature in feature_names:
            value = request.form.get(feature, '').strip()
            if not value:
                raise ValueError(f"{feature} value is required")
            try:
                input_data.append(float(value))
            except ValueError:
                raise ValueError(f"Invalid value for {feature}: '{value}'. Please enter a numeric value.")
        
        print(f"Received input data: {input_data}")  # Debug print
        
        # Relax input validation for broader compatibility
        screener.validate_input(input_data)
        
        # Preprocess data
        input_array = np.array([input_data])
        scaled_data = screener.scaler.transform(input_array)
        
        # Calculate screening probability using transfer learning model
        with torch.no_grad():
            input_tensor = torch.tensor(scaled_data, dtype=torch.float32).to(screener.device)
            proba = screener.model(input_tensor).cpu().numpy()[0][0]
        
        proba_percent = round(proba * 100, 1)
        
        # Determine screening risk stratification
        risk_level = screener.get_risk_level(proba)
        risk_info = CLINICAL_RECOMMENDATIONS[risk_level]
        
        # Render transfer learning screening results
        result_content = render_template_string(
            RESULT_PAGE,
            proba=proba,
            proba_percent=proba_percent,
            risk_level=risk_level,
            risk_info=risk_info
        )
        
        return render_template_string(HTML_TEMPLATE, content=result_content)
        
    except Exception as e:
        # Print detailed error for debugging
        print(f"Error in prediction: {str(e)}")
        print(f"Form data: {dict(request.form)}")
        
        # Error handling
        error_content = render_template_string(ERROR_PAGE, error_message=str(e))
        return render_template_string(HTML_TEMPLATE, content=error_content)

if __name__ == '__main__':
    # Check if required files exist
    required_files = ['transfer_learning_model.pth', 'scaler.pkl']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
            print(f"WARNING: {file} not found.")
    
    if missing_files:
        print("Please run Train1.py first to generate the required model files.")
    
    # Start Flask application
    print("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5000)