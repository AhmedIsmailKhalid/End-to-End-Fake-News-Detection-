import os
import io
import sys
import json
import time
import hashlib
import logging
import requests
import subprocess
import pandas as pd
import altair as alt
import streamlit as st
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any


# Import the new path manager
try:
    from path_config import path_manager
except ImportError:
    # Add current directory to path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append('/app')
    from path_config import path_manager

# Configure logging with error handling for restricted environments
def setup_streamlit_logging():
    """Setup logging with fallback for restricted file access"""
    try:
        # Try to create a log file in logs directory
        log_file_path = path_manager.get_logs_path('streamlit_app.log')
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Test write access
        with open(log_file_path, 'a') as test_file:
            test_file.write('')
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file_path),
                logging.StreamHandler()
            ]
        )
        return True
    except (PermissionError, OSError):
        # Fallback to console-only logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        return False

# Setup logging
file_logging_enabled = setup_streamlit_logging()
logger = logging.getLogger(__name__)

if not file_logging_enabled:
    logger.warning("File logging disabled due to permission restrictions")

# Log environment info at startup
logger.info(f"Streamlit starting in {path_manager.environment} environment")


class StreamlitAppManager:
    """Manages Streamlit application state and functionality with dynamic paths"""

    def __init__(self):
        self.setup_config()
        self.setup_api_client()
        self.initialize_session_state()

    def setup_config(self):
        """Setup application configuration"""
        self.config = {
            'api_url': "http://localhost:8000",
            'max_upload_size': 1000 * 1024 * 1024,  # 1000 MB
            'supported_file_types': ['csv', 'txt', 'json'],
            'max_text_length': 10000,
            'prediction_timeout': 30,
            'refresh_interval': 60,
            'max_batch_size': 100
        }

    def setup_api_client(self):
        """Setup API client with error handling"""
        self.session = requests.Session()
        self.session.timeout = self.config['prediction_timeout']

        # Test API connection
        self.api_available = self.test_api_connection()

    def test_api_connection(self) -> bool:
        """Test API connection"""
        try:
            response = self.session.get(
                f"{self.config['api_url']}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []

        if 'upload_history' not in st.session_state:
            st.session_state.upload_history = []

        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = datetime.now()

        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = False

    def get_cv_results_from_api(self):
        """Get cross-validation results from API"""
        try:
            if not self.api_available:
                return None
            
            response = self.session.get(
                f"{self.config['api_url']}/cv/results",
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return {'error': 'No CV results available'}
            else:
                return None
        except Exception as e:
            logger.warning(f"Could not fetch CV results: {e}")
            return None
    
    def get_model_comparison_from_api(self):
        """Get model comparison results from API"""
        try:
            if not self.api_available:
                return None
            
            response = self.session.get(
                f"{self.config['api_url']}/cv/comparison",
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                return {'error': 'No comparison results available'}
            else:
                return None
        except Exception as e:
            logger.warning(f"Could not fetch model comparison: {e}")
            return None


    def get_validation_statistics_from_api(self):
        """Get validation statistics from API"""
        try:
            if not self.api_available:
                return None
            
            response = self.session.get(
                f"{self.config['api_url']}/validation/statistics",
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except Exception as e:
            logger.warning(f"Could not fetch validation statistics: {e}")
            return None
    
    def get_validation_health_from_api(self):
        """Get validation system health from API"""
        try:
            if not self.api_available:
                return None
            
            response = self.session.get(
                f"{self.config['api_url']}/validation/health",
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except Exception as e:
            logger.warning(f"Could not fetch validation health: {e}")
            return None

    def get_validation_quality_report_from_api(self):
        """Get validation quality report from API"""
        try:
            if not self.api_available:
                return None
            response = self.session.get(f"{self.config['api_url']}/validation/quality-report", timeout=10)
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            logger.warning(f"Could not fetch quality report: {e}")
            return None


    def get_monitoring_metrics_from_api(self):
        """Get current monitoring metrics from API"""
        try:
            if not self.api_available:
                return None
            response = self.session.get(f"{self.config['api_url']}/monitor/metrics/current", timeout=10)
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            logger.warning(f"Could not fetch monitoring metrics: {e}")
            return None
    
    def get_monitoring_alerts_from_api(self):
        """Get monitoring alerts from API"""
        try:
            if not self.api_available:
                return None
            response = self.session.get(f"{self.config['api_url']}/monitor/alerts", timeout=10)
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            logger.warning(f"Could not fetch monitoring alerts: {e}")
            return None
    
    def get_prediction_patterns_from_api(self, hours: int = 24):
        """Get prediction patterns from API"""
        try:
            if not self.api_available:
                return None
            response = self.session.get(f"{self.config['api_url']}/monitor/patterns?hours={hours}", timeout=10)
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            logger.warning(f"Could not fetch prediction patterns: {e}")
            return None


    def get_automation_status_from_api(self):
        """Get automation status from API"""
        try:
            if not self.api_available:
                return None
            response = self.session.get(f"{self.config['api_url']}/automation/status", timeout=10)
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            logger.warning(f"Could not fetch automation status: {e}")
            return None

    # Blue-Green Deployment
    def get_deployment_status_from_api(self):
        """Get deployment status from API"""
        try:
            if not self.api_available:
                return None
            response = self.session.get(f"{self.config['api_url']}/deployment/status", timeout=10)
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            logger.warning(f"Could not fetch deployment status: {e}")
            return None
    
    def get_traffic_status_from_api(self):
        """Get traffic routing status from API"""
        try:
            if not self.api_available:
                return None
            response = self.session.get(f"{self.config['api_url']}/deployment/traffic", timeout=10)
            return response.json() if response.status_code == 200 else None
        except Exception as e:
            logger.warning(f"Could not fetch traffic status: {e}")
            return None


# Initialize app manager
app_manager = StreamlitAppManager()

# Page configuration
st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
    }
    
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
    
    .environment-info {
        background-color: #e7f3ff;
        color: #004085;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #b3d7ff;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def load_json_file(file_path: Path, default: Any = None) -> Any:
    """Safely load JSON file with error handling"""
    try:
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return default or {}
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return default or {}

def show_logs_section():
    """Display system logs in Streamlit"""
    st.subheader("System Logs")
    
    log_files = {
        "Activity Log": path_manager.get_activity_log_path(),
        "Prediction Log": path_manager.get_logs_path("prediction_log.json"),
        "Scheduler Log": path_manager.get_logs_path("scheduler_execution.json"),
        "Drift History": path_manager.get_logs_path("drift_history.json"),
        "Drift Alerts": path_manager.get_logs_path("drift_alerts.json"),
        "Prediction Monitor": path_manager.get_logs_path("monitor/predictions.json"),
        "Metrics Log": path_manager.get_logs_path("monitor/metrics.json"),
        "Alerts Log": path_manager.get_logs_path("monitor/alerts.json")
    }
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_log = st.selectbox("Select log file:", list(log_files.keys()))
    
    with col2:
        max_entries = st.number_input("Max entries:", min_value=10, max_value=1000, value=50)
    
    if st.button("Load Log", type="primary"):
        log_path = log_files[selected_log]
        
        if log_path.exists():
            try:
                with open(log_path, 'r') as f:
                    log_data = json.load(f)
                
                if log_data:
                    st.info(f"Total entries: {len(log_data)}")
                    
                    if len(log_data) > max_entries:
                        log_data = log_data[-max_entries:]
                        st.warning(f"Showing last {max_entries} entries")
                    
                    with st.expander("Raw JSON Data"):
                        st.json(log_data)
                    
                    if isinstance(log_data, list) and log_data:
                        df = pd.DataFrame(log_data)
                        st.dataframe(df, use_container_width=True)
                else:
                    st.warning("Log file is empty")
                    
            except Exception as e:
                st.error(f"Error reading log: {e}")
        else:
            st.warning(f"Log file not found: {log_path}")

def render_cv_results_section():
    """Render cross-validation results section"""
    st.subheader("🎯 Cross-Validation Results")
    
    cv_results = app_manager.get_cv_results_from_api()
    
    if cv_results is None:
        st.warning("API not available - showing local CV results if available")
        
        # Try to load local metadata
        try:
            from path_config import path_manager
            metadata_path = path_manager.get_metadata_path()
            
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    cv_results = {'cross_validation': metadata.get('cross_validation', {})}
            else:
                st.info("No local CV results found")
                return
        except Exception as e:
            st.error(f"Could not load local CV results: {e}")
            return
    
    if cv_results and 'error' not in cv_results:
        # Display model information
        if 'model_version' in cv_results:
            st.info(f"**Model Version:** {cv_results.get('model_version', 'Unknown')} | "
                   f"**Type:** {cv_results.get('model_type', 'Unknown')} | "
                   f"**Trained:** {cv_results.get('training_timestamp', 'Unknown')}")
        
        cv_data = cv_results.get('cross_validation', {})
        
        if cv_data:
            # CV Methodology
            methodology = cv_data.get('methodology', {})
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("CV Folds", methodology.get('n_splits', 'Unknown'))
            with col2:
                st.metric("CV Type", methodology.get('cv_type', 'StratifiedKFold'))
            with col3:
                st.metric("Random State", methodology.get('random_state', 42))
            
            # Performance Metrics Summary
            st.subheader("📊 Performance Summary")
            
            test_scores = cv_data.get('test_scores', {})
            if test_scores:
                
                metrics_cols = st.columns(len(test_scores))
                for idx, (metric, scores) in enumerate(test_scores.items()):
                    with metrics_cols[idx]:
                        if isinstance(scores, dict):
                            mean_val = scores.get('mean', 0)
                            std_val = scores.get('std', 0)
                            st.metric(
                                f"{metric.upper()}",
                                f"{mean_val:.4f}",
                                delta=f"±{std_val:.4f}"
                            )
                
                # Detailed CV Scores Visualization
                st.subheader("📈 Cross-Validation Scores by Metric")
                
                # Create a comprehensive chart
                chart_data = []
                fold_results = cv_data.get('individual_fold_results', [])
                
                if fold_results:
                    for fold_result in fold_results:
                        fold_num = fold_result.get('fold', 0)
                        test_scores_fold = fold_result.get('test_scores', {})
                        
                        for metric, score in test_scores_fold.items():
                            chart_data.append({
                                'Fold': f"Fold {fold_num}",
                                'Metric': metric.upper(),
                                'Score': score,
                                'Type': 'Test'
                            })
                        
                        # Add train scores if available
                        train_scores_fold = fold_result.get('train_scores', {})
                        for metric, score in train_scores_fold.items():
                            chart_data.append({
                                'Fold': f"Fold {fold_num}",
                                'Metric': metric.upper(),
                                'Score': score,
                                'Type': 'Train'
                            })
                
                if chart_data:
                    df_cv = pd.DataFrame(chart_data)
                    
                    # Create separate charts for each metric
                    for metric in df_cv['Metric'].unique():
                        metric_data = df_cv[df_cv['Metric'] == metric]
                        
                        fig = px.bar(
                            metric_data,
                            x='Fold',
                            y='Score',
                            color='Type',
                            title=f'{metric} Scores Across CV Folds',
                            barmode='group'
                        )
                        
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Performance Indicators
                st.subheader("🔍 Model Quality Indicators")
                
                performance_indicators = cv_data.get('performance_indicators', {})
                col1, col2 = st.columns(2)
                
                with col1:
                    overfitting_score = performance_indicators.get('overfitting_score', 'Unknown')
                    if isinstance(overfitting_score, (int, float)):
                        if overfitting_score < 0.05:
                            st.success(f"**Overfitting Score:** {overfitting_score:.4f} (Low)")
                        elif overfitting_score < 0.15:
                            st.warning(f"**Overfitting Score:** {overfitting_score:.4f} (Moderate)")
                        else:
                            st.error(f"**Overfitting Score:** {overfitting_score:.4f} (High)")
                    else:
                        st.info(f"**Overfitting Score:** {overfitting_score}")
                
                with col2:
                    stability_score = performance_indicators.get('stability_score', 'Unknown')
                    if isinstance(stability_score, (int, float)):
                        if stability_score > 0.9:
                            st.success(f"**Stability Score:** {stability_score:.4f} (High)")
                        elif stability_score > 0.7:
                            st.warning(f"**Stability Score:** {stability_score:.4f} (Moderate)")
                        else:
                            st.error(f"**Stability Score:** {stability_score:.4f} (Low)")
                    else:
                        st.info(f"**Stability Score:** {stability_score}")
                
                # Statistical Validation Results
                if 'statistical_validation' in cv_results:
                    st.subheader("📈 Statistical Validation")
                    
                    stat_validation = cv_results['statistical_validation']
                    
                    for metric, validation_data in stat_validation.items():
                        if isinstance(validation_data, dict):
                            with st.expander(f"Statistical Tests - {metric.upper()}"):
                                
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write(f"**Improvement:** {validation_data.get('improvement', 0):.4f}")
                                    st.write(f"**Effect Size:** {validation_data.get('effect_size', 0):.4f}")
                                
                                with col2:
                                    sig_improvement = validation_data.get('significant_improvement', False)
                                    if sig_improvement:
                                        st.success("**Significant Improvement:** Yes")
                                    else:
                                        st.info("**Significant Improvement:** No")
                                
                                # Display test results
                                tests = validation_data.get('tests', {})
                                if tests:
                                    st.write("**Statistical Test Results:**")
                                    for test_name, test_result in tests.items():
                                        if isinstance(test_result, dict):
                                            p_value = test_result.get('p_value', 1.0)
                                            significant = test_result.get('significant', False)
                                            
                                            status = "✅ Significant" if significant else "❌ Not Significant"
                                            st.write(f"- {test_name}: p-value = {p_value:.4f} ({status})")
                
                # Promotion Validation
                if 'promotion_validation' in cv_results:
                    st.subheader("🚀 Model Promotion Validation")
                    
                    promotion_val = cv_results['promotion_validation']
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        confidence = promotion_val.get('decision_confidence', 'Unknown')
                        if isinstance(confidence, (int, float)):
                            st.metric("Decision Confidence", f"{confidence:.2%}")
                        else:
                            st.metric("Decision Confidence", str(confidence))
                    
                    with col2:
                        st.write(f"**Promotion Reason:**")
                        st.write(promotion_val.get('promotion_reason', 'Unknown'))
                    
                    with col3:
                        st.write(f"**Comparison Method:**")
                        st.write(promotion_val.get('comparison_method', 'Unknown'))
                
                # Raw CV Data (expandable)
                with st.expander("🔍 Detailed CV Data"):
                    st.json(cv_data)
                    
            else:
                st.info("No detailed CV test scores available")
        else:
            st.info("No cross-validation data available")
    else:
        error_msg = cv_results.get('error', 'Unknown error') if cv_results else 'No CV results available'
        st.warning(f"Cross-validation results not available: {error_msg}")

def render_validation_statistics_section():
    """Render validation statistics section"""
    st.subheader("📊 Data Validation Statistics")
    
    validation_stats = app_manager.get_validation_statistics_from_api()
    
    if validation_stats and validation_stats.get('statistics_available'):
        overall_metrics = validation_stats.get('overall_metrics', {})
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Validations", overall_metrics.get('total_validations', 0))
        with col2:
            st.metric("Articles Processed", overall_metrics.get('total_articles_processed', 0))
        with col3:
            success_rate = overall_metrics.get('overall_success_rate', 0)
            st.metric("Success Rate", f"{success_rate:.1%}")
        with col4:
            quality_score = overall_metrics.get('average_quality_score', 0)
            st.metric("Avg Quality", f"{quality_score:.3f}")
    else:
        st.info("No validation statistics available yet.")

def render_validation_quality_report():
    """Render validation quality report section"""
    st.subheader("📋 Data Quality Report")
    
    quality_report = app_manager.get_validation_quality_report_from_api()
    
    if quality_report and 'error' not in quality_report:
        overall_stats = quality_report.get('overall_statistics', {})
        quality_assessment = quality_report.get('quality_assessment', {})
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Articles", overall_stats.get('total_articles', 0))
            st.metric("Success Rate", f"{overall_stats.get('overall_success_rate', 0):.1%}")
        with col2:
            quality_level = quality_assessment.get('quality_level', 'unknown')
            if quality_level == 'excellent':
                st.success(f"Quality Level: {quality_level.title()}")
            elif quality_level == 'good':
                st.info(f"Quality Level: {quality_level.title()}")
            elif quality_level == 'fair':
                st.warning(f"Quality Level: {quality_level.title()}")
            else:
                st.error(f"Quality Level: {quality_level.title()}")
        
        recommendations = quality_report.get('recommendations', [])
        if recommendations:
            st.subheader("💡 Recommendations")
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
    else:
        st.info("Quality report not available yet.")
    

def render_model_comparison_section():
    """Render model comparison results section"""
    st.subheader("⚖️ Model Comparison Results")
    
    comparison_results = app_manager.get_model_comparison_from_api()
    
    if comparison_results is None:
        st.warning("API not available - comparison results not accessible")
        return
    
    if comparison_results and 'error' not in comparison_results:
        
        # Comparison Summary
        summary = comparison_results.get('summary', {})
        models_compared = comparison_results.get('models_compared', {})
        
        st.info(f"**Comparison:** {models_compared.get('model1_name', 'Model 1')} vs "
                f"{models_compared.get('model2_name', 'Model 2')} | "
                f"**Timestamp:** {comparison_results.get('comparison_timestamp', 'Unknown')}")
        
        # Decision Summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            decision = summary.get('decision', False)
            if decision:
                st.success("**Decision:** Promote New Model")
            else:
                st.info("**Decision:** Keep Current Model")
        
        with col2:
            confidence = summary.get('confidence', 0)
            st.metric("Decision Confidence", f"{confidence:.2%}")
        
        with col3:
            st.write("**Reason:**")
            st.write(summary.get('reason', 'Unknown'))
        
        # Performance Comparison
        st.subheader("📊 Performance Comparison")
        
        prod_performance = comparison_results.get('model_performance', {}).get('production_model', {})
        cand_performance = comparison_results.get('model_performance', {}).get('candidate_model', {})
        
        # Create comparison chart
        if prod_performance.get('test_scores') and cand_performance.get('test_scores'):
            
            comparison_data = []
            
            prod_scores = prod_performance['test_scores']
            cand_scores = cand_performance['test_scores']
            
            for metric in set(prod_scores.keys()) & set(cand_scores.keys()):
                prod_mean = prod_scores[metric].get('mean', 0)
                cand_mean = cand_scores[metric].get('mean', 0)
                
                comparison_data.extend([
                    {'Model': 'Production', 'Metric': metric.upper(), 'Score': prod_mean},
                    {'Model': 'Candidate', 'Metric': metric.upper(), 'Score': cand_mean}
                ])
            
            if comparison_data:
                df_comparison = pd.DataFrame(comparison_data)
                
                fig = px.bar(
                    df_comparison,
                    x='Metric',
                    y='Score',
                    color='Model',
                    title='Model Performance Comparison',
                    barmode='group'
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Detailed Metric Comparisons
        st.subheader("🔍 Detailed Metric Analysis")
        
        metric_comparisons = comparison_results.get('metric_comparisons', {})
        
        if metric_comparisons:
            for metric, comparison_data in metric_comparisons.items():
                if isinstance(comparison_data, dict):
                    
                    with st.expander(f"{metric.upper()} Analysis"):
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            improvement = comparison_data.get('improvement', 0)
                            rel_improvement = comparison_data.get('relative_improvement', 0)
                            
                            if improvement > 0:
                                st.success(f"**Improvement:** +{improvement:.4f}")
                                st.success(f"**Relative:** +{rel_improvement:.2f}%")
                            else:
                                st.info(f"**Improvement:** {improvement:.4f}")
                                st.info(f"**Relative:** {rel_improvement:.2f}%")
                        
                        with col2:
                            effect_size = comparison_data.get('effect_size', 0)
                            
                            if abs(effect_size) > 0.8:
                                st.success(f"**Effect Size:** {effect_size:.4f} (Large)")
                            elif abs(effect_size) > 0.5:
                                st.warning(f"**Effect Size:** {effect_size:.4f} (Medium)")
                            else:
                                st.info(f"**Effect Size:** {effect_size:.4f} (Small)")
                        
                        with col3:
                            sig_improvement = comparison_data.get('significant_improvement', False)
                            practical_sig = comparison_data.get('practical_significance', False)
                            
                            if sig_improvement:
                                st.success("**Statistical Significance:** Yes")
                            else:
                                st.info("**Statistical Significance:** No")
                            
                            if practical_sig:
                                st.success("**Practical Significance:** Yes")
                            else:
                                st.info("**Practical Significance:** No")
                        
                        # Statistical test results
                        tests = comparison_data.get('tests', {})
                        if tests:
                            st.write("**Statistical Tests:**")
                            for test_name, test_result in tests.items():
                                if isinstance(test_result, dict):
                                    p_value = test_result.get('p_value', 1.0)
                                    significant = test_result.get('significant', False)
                                    
                                    status = "✅" if significant else "❌"
                                    st.write(f"- {test_name}: p = {p_value:.4f} {status}")
        
        # CV Methodology
        cv_methodology = comparison_results.get('cv_methodology', {})
        if cv_methodology:
            st.subheader("🎯 Cross-Validation Methodology")
            st.info(f"**CV Folds:** {cv_methodology.get('cv_folds', 'Unknown')} | "
                   f"**Session ID:** {comparison_results.get('session_id', 'Unknown')}")
        
        # Raw comparison data (expandable)
        with st.expander("🔍 Raw Comparison Data"):
            st.json(comparison_results)
            
    else:
        error_msg = comparison_results.get('error', 'Unknown error') if comparison_results else 'No comparison results available'
        st.warning(f"Model comparison results not available: {error_msg}")
        

def save_prediction_to_history(text: str, prediction: str, confidence: float):
    """Save prediction to session history"""
    prediction_entry = {
        'timestamp': datetime.now().isoformat(),
        'text': text[:100] + "..." if len(text) > 100 else text,
        'prediction': prediction,
        'confidence': confidence,
        'text_length': len(text)
    }

    st.session_state.prediction_history.append(prediction_entry)

    # Keep only last 50 predictions
    if len(st.session_state.prediction_history) > 50:
        st.session_state.prediction_history = st.session_state.prediction_history[-50:]


def make_prediction_request(text: str) -> Dict[str, Any]:
    """Make prediction request to API"""
    try:
        if not app_manager.api_available:
            return {'error': 'API is not available'}

        response = app_manager.session.post(
            f"{app_manager.config['api_url']}/predict",
            json={"text": text},
            timeout=app_manager.config['prediction_timeout']
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {'error': f'API Error: {response.status_code} - {response.text}'}

    except requests.exceptions.Timeout:
        return {'error': 'Request timed out. Please try again.'}
    except requests.exceptions.ConnectionError:
        return {'error': 'Cannot connect to prediction service.'}
    except Exception as e:
        return {'error': f'Unexpected error: {str(e)}'}


def validate_text_input(text: str) -> tuple[bool, str]:
    """Validate text input"""
    if not text or not text.strip():
        return False, "Please enter some text to analyze."

    if len(text) < 10:
        return False, "Text must be at least 10 characters long."

    if len(text) > app_manager.config['max_text_length']:
        return False, f"Text must be less than {app_manager.config['max_text_length']} characters."

    # Check for suspicious content
    suspicious_patterns = ['<script', 'javascript:', 'data:']
    if any(pattern in text.lower() for pattern in suspicious_patterns):
        return False, "Text contains suspicious content."

    return True, "Valid"


def create_confidence_gauge(confidence: float, prediction: str):
    """Create confidence gauge visualization"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Confidence: {prediction}"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "red" if prediction == "Fake" else "green"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(height=300)
    return fig


def create_prediction_history_chart():
    """Create prediction history visualization"""
    if not st.session_state.prediction_history:
        return None

    df = pd.DataFrame(st.session_state.prediction_history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['confidence_percent'] = df['confidence'] * 100

    fig = px.scatter(
        df,
        x='timestamp',
        y='confidence_percent',
        color='prediction',
        size='text_length',
        hover_data=['text'],
        title="Prediction History",
        labels={'confidence_percent': 'Confidence (%)', 'timestamp': 'Time'}
    )

    fig.update_layout(height=400)
    return fig

def create_cv_performance_chart(cv_results: dict) -> Optional[Any]:
    """Create a comprehensive CV performance visualization"""
    try:
        if not cv_results or 'cross_validation' not in cv_results:
            return None
        
        cv_data = cv_results['cross_validation']
        fold_results = cv_data.get('individual_fold_results', [])
        
        if not fold_results:
            return None
        
        # Prepare data for visualization
        chart_data = []
        
        for fold_result in fold_results:
            fold_num = fold_result.get('fold', 0)
            test_scores = fold_result.get('test_scores', {})
            train_scores = fold_result.get('train_scores', {})
            
            for metric, score in test_scores.items():
                chart_data.append({
                    'Fold': fold_num,
                    'Metric': metric.upper(),
                    'Score': score,
                    'Type': 'Test',
                    'Fold_Label': f"Fold {fold_num}"
                })
            
            for metric, score in train_scores.items():
                chart_data.append({
                    'Fold': fold_num,
                    'Metric': metric.upper(), 
                    'Score': score,
                    'Type': 'Train',
                    'Fold_Label': f"Fold {fold_num}"
                })
        
        if not chart_data:
            return None
        
        df_cv = pd.DataFrame(chart_data)
        
        # Create faceted chart showing all metrics
        fig = px.box(
            df_cv[df_cv['Type'] == 'Test'],  # Focus on test scores
            x='Metric',
            y='Score',
            title='Cross-Validation Performance Distribution',
            points='all'
        )
        
        # Add mean lines
        for metric in df_cv['Metric'].unique():
            metric_data = df_cv[(df_cv['Metric'] == metric) & (df_cv['Type'] == 'Test')]
            mean_score = metric_data['Score'].mean()
            
            fig.add_hline(
                y=mean_score,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {mean_score:.3f}"
            )
        
        fig.update_layout(
            height=500,
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Failed to create CV chart: {e}")
        return None


def render_environment_info():
    """Render environment information"""
    env_info = path_manager.get_environment_info()
    
    st.markdown(f"""
    <div class="environment-info">
        <h4>🌍 Environment Information</h4>
        <p><strong>Environment:</strong> {env_info['environment']}</p>
        <p><strong>Base Directory:</strong> {env_info['base_dir']}</p>
        <p><strong>Data Directory:</strong> {env_info['data_dir']}</p>
        <p><strong>Model Directory:</strong> {env_info['model_dir']}</p>
    </div>
    """, unsafe_allow_html=True)


# Main application
def main():
    """Main Streamlit application"""

    # Header
    st.markdown('<h1 class="main-header">📰 Fake News Detection System</h1>',
                unsafe_allow_html=True)

    # Environment info
    render_environment_info()

    # API Status indicator
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if app_manager.api_available:
            st.markdown(
                '<div class="success-message">🟢 API Service: Online</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="error-message">🔴 API Service: Offline</div>', unsafe_allow_html=True)

    # Main content area
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🔍 Prediction",
        "📊 Batch Analysis", 
        "📈 Analytics",
        "🎯 Model Training",
        "📋 Logs",
        "⚙️ System Status",
        "📊 Monitoring"  # New monitoring tab
    ])


    # Tab 1: Individual Prediction
    with tab1:
        st.header("Single Text Analysis")

        # Input methods
        input_method = st.radio(
            "Choose input method:",
            ["Type Text", "Upload File"],
            horizontal=True
        )

        user_text = ""

        if input_method == "Type Text":
            user_text = st.text_area(
                "Enter news article text:",
                height=200,
                placeholder="Paste or type the news article you want to analyze..."
            )

        else:  # Upload File
            uploaded_file = st.file_uploader(
                "Upload text file:",
                type=['txt', 'csv'],
                help="Upload a text file containing the article to analyze"
            )

            if uploaded_file:
                try:
                    if uploaded_file.type == "text/plain":
                        user_text = str(uploaded_file.read(), "utf-8")
                    elif uploaded_file.type == "text/csv":
                        df = pd.read_csv(uploaded_file)
                        if 'text' in df.columns:
                            user_text = df['text'].iloc[0] if len(
                                df) > 0 else ""
                        else:
                            st.error("CSV file must contain a 'text' column")

                    st.success(
                        f"File uploaded successfully! ({len(user_text)} characters)")

                except Exception as e:
                    st.error(f"Error reading file: {e}")

        # Prediction section
        col1, col2 = st.columns([3, 1])

        with col1:
            if st.button("🧠 Analyze Text", type="primary", use_container_width=True):
                if user_text:
                    # Validate input
                    is_valid, validation_message = validate_text_input(
                        user_text)

                    if not is_valid:
                        st.error(validation_message)
                    else:
                        # Show progress
                        with st.spinner("Analyzing text..."):
                            result = make_prediction_request(user_text)

                        if 'error' in result:
                            st.error(f"❌ {result['error']}")
                        else:
                            # Display results
                            prediction = result['prediction']
                            confidence = result['confidence']

                            # Save to history
                            save_prediction_to_history(
                                user_text, prediction, confidence)

                            # Results display
                            col_result1, col_result2 = st.columns(2)

                            with col_result1:
                                if prediction == "Fake":
                                    st.markdown(f"""
                                    <div class="error-message">
                                        <h3>🚨 Prediction: FAKE NEWS</h3>
                                        <p>Confidence: {confidence:.2%}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div class="success-message">
                                        <h3>✅ Prediction: REAL NEWS</h3>
                                        <p>Confidence: {confidence:.2%}</p>
                                    </div>
                                    """, unsafe_allow_html=True)

                            with col_result2:
                                # Confidence gauge
                                fig_gauge = create_confidence_gauge(
                                    confidence, prediction)
                                st.plotly_chart(
                                    fig_gauge, use_container_width=True)

                            # Additional information
                            with st.expander("📋 Analysis Details"):
                                st.json({
                                    "model_version": result.get('model_version', 'Unknown'),
                                    "processing_time": f"{result.get('processing_time', 0):.3f} seconds",
                                    "timestamp": result.get('timestamp', ''),
                                    "text_length": len(user_text),
                                    "word_count": len(user_text.split()),
                                    "environment": path_manager.environment
                                })
                else:
                    st.warning("Please enter text to analyze.")

        with col2:
            if st.button("🔄 Clear Text", use_container_width=True):
                st.rerun()

    # Tab 2: Batch Analysis (simplified for space)
    with tab2:
        st.header("Batch Text Analysis")
        
        # File upload for batch processing
        batch_file = st.file_uploader(
            "Upload CSV file for batch analysis:",
            type=['csv'],
            help="CSV file should contain a 'text' column with articles to analyze"
        )

        if batch_file:
            try:
                df = pd.read_csv(batch_file)

                if 'text' not in df.columns:
                    st.error("CSV file must contain a 'text' column")
                else:
                    st.success(f"File loaded: {len(df)} articles found")

                    # Preview data
                    st.subheader("Data Preview")
                    st.dataframe(df.head(10))

                    # Batch processing
                    if st.button("🚀 Process Batch", type="primary"):
                        if len(df) > app_manager.config['max_batch_size']:
                            st.warning(
                                f"Only processing first {app_manager.config['max_batch_size']} articles")
                            df = df.head(app_manager.config['max_batch_size'])

                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        results = []

                        for i, row in df.iterrows():
                            status_text.text(
                                f"Processing article {i+1}/{len(df)}...")
                            progress_bar.progress((i + 1) / len(df))

                            result = make_prediction_request(row['text'])

                            if 'error' not in result:
                                results.append({
                                    'text': row['text'][:100] + "...",
                                    'prediction': result['prediction'],
                                    'confidence': result['confidence'],
                                    'processing_time': result.get('processing_time', 0)
                                })
                            else:
                                results.append({
                                    'text': row['text'][:100] + "...",
                                    'prediction': 'Error',
                                    'confidence': 0,
                                    'processing_time': 0
                                })

                        # Display results
                        results_df = pd.DataFrame(results)

                        # Summary statistics
                        col1, col2, col3, col4 = st.columns(4)

                        with col1:
                            st.metric("Total Processed", len(results_df))

                        with col2:
                            fake_count = len(
                                results_df[results_df['prediction'] == 'Fake'])
                            st.metric("Fake News", fake_count)

                        with col3:
                            real_count = len(
                                results_df[results_df['prediction'] == 'Real'])
                            st.metric("Real News", real_count)

                        with col4:
                            avg_confidence = results_df['confidence'].mean()
                            st.metric("Avg Confidence",
                                      f"{avg_confidence:.2%}")

                        # Results visualization
                        if len(results_df) > 0:
                            fig = px.histogram(
                                results_df,
                                x='prediction',
                                color='prediction',
                                title="Batch Analysis Results"
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        # Download results
                        csv_buffer = io.StringIO()
                        results_df.to_csv(csv_buffer, index=False)

                        st.download_button(
                            label="📥 Download Results",
                            data=csv_buffer.getvalue(),
                            file_name=f"batch_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )

            except Exception as e:
                st.error(f"Error processing file: {e}")

    # Tab 3: Analytics
    with tab3:
        st.header("System Analytics")
        
        # Add CV and Model Comparison sections
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("🔄 Refresh CV Results", use_container_width=True):
                st.rerun()
        
        with col2:
            show_detailed_cv = st.checkbox("Show Detailed CV Analysis", value=True)
        
        if show_detailed_cv:
            # Render cross-validation results
            render_cv_results_section()
            
            # Add separator
            st.divider()
            
            # Render model comparison results
            render_model_comparison_section()
            
            # Add separator
            st.divider()
    
        # Prediction history (existing content)
        if st.session_state.prediction_history:
            st.subheader("Recent Predictions")
    
            # History chart
            fig_history = create_prediction_history_chart()
            if fig_history:
                st.plotly_chart(fig_history, use_container_width=True)
    
            # History table
            history_df = pd.DataFrame(st.session_state.prediction_history)
            st.dataframe(history_df.tail(20), use_container_width=True)
    
        else:
            st.info(
                "No prediction history available. Make some predictions to see analytics.")
    
        # System metrics (existing content with CV enhancement)
        st.subheader("System Metrics")
    
        # Load various log files for analytics
        try:
            # API health check with CV information
            if app_manager.api_available:
                response = app_manager.session.get(
                    f"{app_manager.config['api_url']}/metrics")
                if response.status_code == 200:
                    metrics = response.json()
    
                    # Basic metrics
                    api_metrics = metrics.get('api_metrics', {})
                    model_info = metrics.get('model_info', {})
                    cv_summary = metrics.get('cross_validation_summary', {})
    
                    col1, col2, col3, col4 = st.columns(4)
    
                    with col1:
                        st.metric("Total API Requests",
                                  api_metrics.get('total_requests', 0))
    
                    with col2:
                        st.metric("Unique Clients", 
                                  api_metrics.get('unique_clients', 0))
    
                    with col3:
                        st.metric("Model Version", 
                                  model_info.get('model_version', 'Unknown'))
    
                    with col4:
                        status = model_info.get('model_health', 'unknown')
                        st.metric("Model Status", status)
    
                    # Cross-validation summary metrics
                    if cv_summary.get('cv_available', False):
                        st.subheader("Cross-Validation Summary")
                        
                        cv_col1, cv_col2, cv_col3, cv_col4 = st.columns(4)
                        
                        with cv_col1:
                            cv_folds = cv_summary.get('cv_folds', 'Unknown')
                            st.metric("CV Folds", cv_folds)
                        
                        with cv_col2:
                            cv_f1 = cv_summary.get('cv_f1_mean')
                            cv_f1_std = cv_summary.get('cv_f1_std')
                            if cv_f1 is not None and cv_f1_std is not None:
                                st.metric("CV F1 Score", f"{cv_f1:.4f}", f"±{cv_f1_std:.4f}")
                            else:
                                st.metric("CV F1 Score", "N/A")
                        
                        with cv_col3:
                            cv_acc = cv_summary.get('cv_accuracy_mean')
                            cv_acc_std = cv_summary.get('cv_accuracy_std')
                            if cv_acc is not None and cv_acc_std is not None:
                                st.metric("CV Accuracy", f"{cv_acc:.4f}", f"±{cv_acc_std:.4f}")
                            else:
                                st.metric("CV Accuracy", "N/A")
                        
                        with cv_col4:
                            overfitting = cv_summary.get('overfitting_score')
                            if overfitting is not None:
                                if overfitting < 0.05:
                                    st.metric("Overfitting", f"{overfitting:.4f}", "Low", delta_color="normal")
                                elif overfitting < 0.15:
                                    st.metric("Overfitting", f"{overfitting:.4f}", "Moderate", delta_color="off")
                                else:
                                    st.metric("Overfitting", f"{overfitting:.4f}", "High", delta_color="inverse")
                            else:
                                st.metric("Overfitting", "N/A")
    
                    # Environment details
                    st.subheader("Environment Details")
                    env_info = metrics.get('environment_info', {})
                    env_data = env_info.get('environment', 'Unknown')
                    st.info(f"Running in: {env_data}")
                    
                    # Available files
                    datasets = env_info.get('available_datasets', {})
                    models = env_info.get('available_models', {})
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Available Datasets:**")
                        for name, exists in datasets.items():
                            status = "✅" if exists else "❌"
                            st.write(f"{status} {name}")
                    
                    with col2:
                        st.write("**Available Models:**")
                        for name, exists in models.items():
                            status = "✅" if exists else "❌"
                            st.write(f"{status} {name}")
    
        except Exception as e:
            st.warning(f"Could not load API metrics: {e}")

    # Tab 4: Model Training  
    with tab4:
        st.header("Custom Model Training")

        st.info("Upload your own dataset to retrain the model with custom data.")

        # File upload for training
        training_file = st.file_uploader(
            "Upload training dataset (CSV):",
            type=['csv'],
            help="CSV file should contain 'text' and 'label' columns (label: 0=Real, 1=Fake)"
        )

        if training_file:
            try:
                df_train = pd.read_csv(training_file)

                required_columns = ['text', 'label']
                missing_columns = [
                    col for col in required_columns if col not in df_train.columns]

                if missing_columns:
                    st.error(f"Missing required columns: {missing_columns}")
                else:
                    st.success(
                        f"Training file loaded: {len(df_train)} samples")

                    # Data validation
                    label_counts = df_train['label'].value_counts()

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Dataset Overview")
                        st.write(f"Total samples: {len(df_train)}")
                        st.write(f"Real news (0): {label_counts.get(0, 0)}")
                        st.write(f"Fake news (1): {label_counts.get(1, 0)}")

                    with col2:
                        # Label distribution chart
                        fig_labels = px.pie(
                            values=label_counts.values,
                            names=['Real', 'Fake'],
                            title="Label Distribution"
                        )
                        st.plotly_chart(fig_labels)

                    # Training options
                    st.subheader("Training Configuration")

                    col1, col2 = st.columns(2)

                    with col1:
                        test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
                        max_features = st.number_input(
                            "Max Features", 1000, 20000, 10000, 1000)

                    with col2:
                        cross_validation = st.checkbox(
                            "Cross Validation", value=True)
                        hyperparameter_tuning = st.checkbox(
                            "Hyperparameter Tuning", value=False)

                    # Start training
                    if st.button("🏃‍♂️ Start Training", type="primary"):
                        # Save training data to the appropriate location
                        custom_data_path = path_manager.get_data_path('custom_upload.csv')
                        custom_data_path.parent.mkdir(parents=True, exist_ok=True)
                        df_train.to_csv(custom_data_path, index=False)

                        # Progress simulation
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        training_steps = [
                            "Preprocessing data...",
                            "Splitting dataset...",
                            "Training model...",
                            "Evaluating performance...",
                            "Saving model..."
                        ]

                        for i, step in enumerate(training_steps):
                            status_text.text(step)
                            progress_bar.progress(
                                (i + 1) / len(training_steps))
                            time.sleep(2)  # Simulate processing time

                        # Run actual training
                        try:
                            result = subprocess.run(
                                [sys.executable, str(path_manager.get_model_path() / "train.py"),
                                 "--data_path", str(custom_data_path)],
                                capture_output=True,
                                text=True,
                                timeout=300,
                                cwd=str(path_manager.base_paths['base'])
                            )

                            if result.returncode == 0:
                                st.success(
                                    "🎉 Training completed successfully!")

                                # Try to extract accuracy from output
                                try:
                                    output_lines = result.stdout.strip().split('\n')
                                    for line in output_lines:
                                        if 'accuracy' in line.lower():
                                            st.info(
                                                f"Model performance: {line}")
                                except:
                                    pass

                                # Reload API model
                                if app_manager.api_available:
                                    try:
                                        reload_response = app_manager.session.post(
                                            f"{app_manager.config['api_url']}/model/reload"
                                        )
                                        if reload_response.status_code == 200:
                                            st.success(
                                                "✅ Model reloaded in API successfully!")
                                    except:
                                        st.warning(
                                            "⚠️ Model trained but API reload failed")

                            else:
                                st.error(f"Training failed: {result.stderr}")

                        except subprocess.TimeoutExpired:
                            st.error(
                                "Training timed out. Please try with a smaller dataset.")
                        except Exception as e:
                            st.error(f"Training error: {e}")

            except Exception as e:
                st.error(f"Error loading training file: {e}")

    # Tab 5: Logs
    with tab5:
        show_logs_section()

    # Tab 6: System Status
    with tab6:
        render_system_status()

    # Tab 7: Monitoring
    with tab7:
        st.header("Real-time System Monitoring")
        
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("🔄 Refresh Monitoring", use_container_width=True):
                st.rerun()
        
        render_monitoring_dashboard()
        st.divider()
        render_monitoring_alerts()
        st.divider()
        render_automation_status()
        st.divider()
        render_deployment_status()

def render_system_status():
    """Render system status tab"""
    st.header("System Status & Monitoring")

    # Auto-refresh toggle
    col1, col2 = st.columns([1, 4])
    with col1:
        st.session_state.auto_refresh = st.checkbox(
            "Auto Refresh", value=st.session_state.auto_refresh)

    with col2:
        if st.button("🔄 Refresh Now"):
            st.session_state.last_refresh = datetime.now()
            st.rerun()

    # Environment Information
    st.subheader("🌍 Environment Information")
    env_info = path_manager.get_environment_info()
    
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Environment:** {env_info['environment']}")
        st.write(f"**Base Directory:** {env_info['base_dir']}")
        st.write(f"**Working Directory:** {env_info.get('current_working_directory', 'N/A')}")
        
    with col2:
        st.write(f"**Data Directory:** {env_info['data_dir']}")
        st.write(f"**Model Directory:** {env_info['model_dir']}")
        st.write(f"**Logs Directory:** {env_info.get('logs_dir', 'N/A')}")

    # System health overview
    st.subheader("🏥 System Health")

    if app_manager.api_available:
        try:
            health_response = app_manager.session.get(
                f"{app_manager.config['api_url']}/health")
            if health_response.status_code == 200:
                health_data = health_response.json()

                # Overall status
                overall_status = health_data.get('status', 'unknown')
                if overall_status == 'healthy':
                    st.success("🟢 System Status: Healthy")
                else:
                    st.error("🔴 System Status: Unhealthy")

                # Basic health display
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.subheader("🤖 Model Health")
                    model_health = health_data.get('model_health', {})
                    for key, value in model_health.items():
                        if key not in ['test_prediction', 'model_path', 'data_path', 'environment']:
                            display_key = key.replace('_', ' ').title()
                            if isinstance(value, bool):
                                status = "✅" if value else "❌"
                                st.write(f"**{display_key}:** {status}")
                            else:
                                st.write(f"**{display_key}:** {value}")

        except Exception as e:
            st.error(f"Failed to get health status: {e}")
    else:
        st.error("🔴 API Service is not available")

    # Add the validation sections as specified in the document
    st.divider()
    render_validation_statistics_section()
    st.divider() 
    render_validation_quality_report()

    # Model information
    st.subheader("🎯 Model Information")
    metadata = load_json_file(path_manager.get_metadata_path(), {})
    if metadata:
        col1, col2 = st.columns(2)
        with col1:
            for key in ['model_version', 'test_accuracy', 'test_f1', 'model_type']:
                if key in metadata:
                    display_key = key.replace('_', ' ').title()
                    value = metadata[key]
                    if isinstance(value, float):
                        st.metric(display_key, f"{value:.4f}")
                    else:
                        st.metric(display_key, str(value))
        with col2:
            for key in ['train_size', 'timestamp', 'environment']:
                if key in metadata:
                    display_key = key.replace('_', ' ').title()
                    value = metadata[key]
                    if key == 'timestamp':
                        try:
                            dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
                            value = dt.strftime('%Y-%m-%d %H:%M:%S')
                        except:
                            pass
                    st.write(f"**{display_key}:** {value}")
    else:
        st.warning("No model metadata available")

def render_monitoring_dashboard():
    """Render real-time monitoring dashboard"""
    st.subheader("📊 Real-time Monitoring Dashboard")
    
    monitoring_data = app_manager.get_monitoring_metrics_from_api()
    
    if monitoring_data:
        # Current metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        pred_metrics = monitoring_data.get('prediction_metrics', {})
        system_metrics = monitoring_data.get('system_metrics', {})
        api_metrics = monitoring_data.get('api_metrics', {})
        
        with col1:
            st.metric("Predictions/Min", f"{pred_metrics.get('predictions_per_minute', 0):.1f}")
            st.metric("Avg Confidence", f"{pred_metrics.get('avg_confidence', 0):.2f}")
        
        with col2:
            st.metric("Response Time", f"{api_metrics.get('avg_response_time', 0):.2f}s")
            st.metric("Error Rate", f"{api_metrics.get('error_rate', 0):.1%}")
        
        with col3:
            st.metric("CPU Usage", f"{system_metrics.get('cpu_percent', 0):.1f}%")
            st.metric("Memory Usage", f"{system_metrics.get('memory_percent', 0):.1f}%")
        
        with col4:
            anomaly_score = pred_metrics.get('anomaly_score', 0)
            st.metric("Anomaly Score", f"{anomaly_score:.3f}")
            if anomaly_score > 0.3:
                st.warning("High anomaly score detected!")
    else:
        st.warning("Monitoring data not available")

def render_monitoring_alerts():
    """Render monitoring alerts section"""
    st.subheader("🚨 Active Alerts")
    
    alerts_data = app_manager.get_monitoring_alerts_from_api()
    
    if alerts_data:
        active_alerts = alerts_data.get('active_alerts', [])
        alert_stats = alerts_data.get('alert_statistics', {})
        
        # Alert statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Active Alerts", alert_stats.get('active_alerts', 0))
        with col2:
            st.metric("Critical Alerts", alert_stats.get('critical_alerts_active', 0))
        with col3:
            st.metric("24h Alert Rate", f"{alert_stats.get('alert_rate_per_hour', 0):.1f}/hr")
        
        # Active alerts display
        if active_alerts:
            for alert in active_alerts:
                alert_type = alert.get('type', 'info')
                if alert_type == 'critical':
                    st.error(f"🔴 **{alert.get('title', 'Unknown')}**: {alert.get('message', '')}")
                elif alert_type == 'warning':
                    st.warning(f"🟡 **{alert.get('title', 'Unknown')}**: {alert.get('message', '')}")
                else:
                    st.info(f"🔵 **{alert.get('title', 'Unknown')}**: {alert.get('message', '')}")
        else:
            st.success("No active alerts")
    else:
        st.warning("Alert data not available")


def render_automation_status():
    """Render automation system status"""
    st.subheader("🤖 Automated Retraining Status")
    
    automation_data = app_manager.get_automation_status_from_api()
    
    if automation_data:
        automation_system = automation_data.get('automation_system', {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Monitoring Active", "Yes" if automation_system.get('monitoring_active') else "No")
        with col2:
            st.metric("Total Auto Trainings", automation_system.get('total_automated_trainings', 0))
        with col3:
            st.metric("Queued Jobs", automation_system.get('queued_jobs', 0))
        
        if automation_system.get('last_automated_training'):
            st.info(f"Last automated training: {automation_system['last_automated_training']}")
        
        if automation_system.get('in_cooldown'):
            st.warning("System in cooldown period")
    else:
        st.warning("Automation status not available")


def render_deployment_status():
    """Render deployment system status"""
    st.subheader("🚀 Blue-Green Deployment Status")
    
    deployment_data = app_manager.get_deployment_status_from_api()
    traffic_data = app_manager.get_traffic_status_from_api()
    
    if deployment_data:
        current_deployment = deployment_data.get('current_deployment')
        active_version = deployment_data.get('active_version')
        traffic_split = deployment_data.get('traffic_split', {})
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Active Version", active_version['version_id'] if active_version else "None")
        with col2:
            st.metric("Blue Traffic", f"{traffic_split.get('blue', 0)}%")
        with col3:
            st.metric("Green Traffic", f"{traffic_split.get('green', 0)}%")
        
        if current_deployment:
            st.info(f"Current deployment: {current_deployment['deployment_id']} ({current_deployment['status']})")
    else:
        st.warning("Deployment status not available")

# Auto-refresh logic
if st.session_state.auto_refresh:
    time_since_refresh = datetime.now() - st.session_state.last_refresh
    if time_since_refresh > timedelta(seconds=app_manager.config['refresh_interval']):
        st.session_state.last_refresh = datetime.now()
        st.rerun()
        
# Run main application
if __name__ == "__main__":
    main()