"""
Evaluation Display Component
Renders RAG evaluation metrics in a comprehensive, user-friendly format

Design Pattern: Component Pattern - Self-contained, reusable UI component
"""

import streamlit as st
from typing import Dict, Any, Optional
import plotly.graph_objects as go
import logging

logger = logging.getLogger(__name__)


class EvaluationDisplay:
    """
    Component for displaying RAG evaluation metrics.
    
    Provides both compact and full display modes for evaluation results.
    """
    
    def __init__(self):
        """Initialize evaluation display component."""
        self.metric_colors = {
            "excellent": "#00C853",  # Green
            "good": "#64DD17",       # Light green
            "fair": "#FFB300",       # Orange
            "needs_improvement": "#DD2C00"  # Red
        }
        
        self.metric_labels = {
            "answer_relevance": "📝 Answer Relevance",
            "context_relevance": "🔍 Context Relevance",
            "groundedness": "✅ Groundedness",
            "context_precision": "🎯 Context Precision",
            "context_recall": "📊 Context Recall",
            "answer_correctness": "✔️ Answer Correctness"
        }
        
        self.metric_descriptions = {
            "answer_relevance": "Does the answer address the user's question?",
            "context_relevance": "Are the retrieved documents relevant to the query?",
            "groundedness": "Is the answer based only on retrieved documents? (Hallucination check)",
            "context_precision": "Are the most relevant documents ranked highest?",
            "context_recall": "Did we retrieve all necessary information?",
            "answer_correctness": "Is the answer factually correct?"
        }
    
    def render(self, evaluation_data: Dict[str, Any]):
        """
        Render complete evaluation display.
        
        Args:
            evaluation_data: Evaluation data from API response
        """
        if not evaluation_data:
            st.info("ℹ️ No evaluation data available")
            return
        
        # Header
        self._render_header(evaluation_data)
        
        # Overall score (prominent)
        self._render_overall_score(evaluation_data)
        
        # Individual metrics
        self._render_metrics(evaluation_data)
        
        # Visualization
        self._render_chart(evaluation_data)
        
        # Summary and recommendations
        self._render_summary(evaluation_data)
    
    def render_compact(self, evaluation_data: Dict[str, Any]):
        """
        Render compact evaluation display (for message metadata).
        
        Args:
            evaluation_data: Evaluation data from API response
        """
        if not evaluation_data:
            return
        
        with st.expander("📊 Quality Metrics", expanded=False):
            overall_score = evaluation_data.get("overall_score", 0.0)
            
            # Overall score badge
            col1, col2 = st.columns([1, 3])
            with col1:
                status = self._get_status_from_score(overall_score)
                color = self.metric_colors[status.lower().replace(" ", "_")]
                st.markdown(
                    f"<div style='text-align: center; padding: 10px; "
                    f"background-color: {color}; color: white; border-radius: 5px;'>"
                    f"<strong>{overall_score:.0%}</strong><br/>"
                    f"<small>{status}</small></div>",
                    unsafe_allow_html=True
                )
            
            with col2:
                # Quick metrics
                metrics_data = evaluation_data.get("metrics", {})
                if metrics_data:
                    # Show top 3 core metrics
                    key_metrics = ["answer_relevance", "context_relevance", "groundedness"]
                    available_metrics = [m for m in key_metrics if m in metrics_data and metrics_data[m]]
                    
                    if available_metrics:
                        cols = st.columns(len(available_metrics))
                        for idx, metric_key in enumerate(available_metrics):
                            metric = metrics_data[metric_key]
                            with cols[idx]:
                                st.metric(
                                    label=self.metric_labels.get(metric_key, metric_key),
                                    value=f"{metric['score']:.0%}",
                                    help=metric.get('explanation', '')
                                )
            
            # Summary
            summary = evaluation_data.get("summary", "")
            if summary:
                st.caption(f"💡 {summary}")
    
    def _render_header(self, evaluation_data: Dict[str, Any]):
        """Render evaluation header."""
        st.markdown("### 📊 RAG Evaluation Report")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            query = evaluation_data.get("query", "N/A")
            st.caption(f"**Query:** {query[:100]}..." if len(query) > 100 else f"**Query:** {query}")
        
        with col2:
            strategy = evaluation_data.get("strategy", "N/A")
            st.caption(f"**Strategy:** {strategy}")
    
    def _render_overall_score(self, evaluation_data: Dict[str, Any]):
        """Render prominent overall score display."""
        overall_score = evaluation_data.get("overall_score", 0.0)
        status = self._get_status_from_score(overall_score)
        color = self.metric_colors[status.lower().replace(" ", "_")]
        
        st.markdown(
            f"""
            <div style='text-align: center; padding: 30px; margin: 20px 0;
                        background: linear-gradient(135deg, {color}22 0%, {color}11 100%);
                        border: 2px solid {color}; border-radius: 15px;'>
                <h1 style='color: {color}; margin: 0; font-size: 3em;'>{overall_score:.0%}</h1>
                <h3 style='color: {color}; margin: 10px 0 0 0;'>{status}</h3>
                <p style='color: #666; margin: 5px 0 0 0;'>Overall RAG Quality Score</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    def _render_metrics(self, evaluation_data: Dict[str, Any]):
        """Render individual metric cards."""
        st.markdown("#### 📈 Detailed Metrics")
        
        metrics_data = evaluation_data.get("metrics", {})
        
        if not metrics_data:
            st.info("No detailed metrics available")
            return
        
        # Render metrics in rows of 2
        metric_keys = [k for k, v in metrics_data.items() if v is not None]
        
        for i in range(0, len(metric_keys), 2):
            cols = st.columns(2)
            
            for idx, col in enumerate(cols):
                if i + idx < len(metric_keys):
                    metric_key = metric_keys[i + idx]
                    metric = metrics_data[metric_key]
                    
                    if metric:
                        with col:
                            self._render_metric_card(metric_key, metric)
    
    def _render_metric_card(self, metric_key: str, metric: Dict[str, Any]):
        """Render individual metric card."""
        score = metric.get("score", 0.0)
        explanation = metric.get("explanation", "No explanation available")
        status = metric.get("status", self._get_status_from_score(score))
        method = metric.get("method", "N/A")
        
        status_lower = status.lower().replace(" ", "_")
        color = self.metric_colors.get(status_lower, "#666")
        
        # Metric card
        st.markdown(
            f"""
            <div style='padding: 15px; margin: 10px 0; 
                        border-left: 4px solid {color}; 
                        background-color: #f8f9fa; border-radius: 5px;'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <h4 style='margin: 0; color: #333;'>{self.metric_labels.get(metric_key, metric_key)}</h4>
                    <span style='font-size: 1.5em; font-weight: bold; color: {color};'>{score:.0%}</span>
                </div>
                <p style='margin: 10px 0 5px 0; color: #666; font-size: 0.9em;'>{explanation}</p>
                <div style='display: flex; justify-content: space-between; margin-top: 10px;'>
                    <small style='color: {color}; font-weight: 600;'>{status}</small>
                    <small style='color: #999;'>Method: {method}</small>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Show description on hover
        with st.expander(f"ℹ️ About {self.metric_labels.get(metric_key, metric_key)}", expanded=False):
            st.write(self.metric_descriptions.get(metric_key, "No description available"))
    
    def _render_chart(self, evaluation_data: Dict[str, Any]):
        """Render metrics visualization chart."""
        st.markdown("#### 📊 Metrics Visualization")
        
        metrics_data = evaluation_data.get("metrics", {})
        
        if not metrics_data:
            return
        
        # Prepare data for chart
        labels = []
        values = []
        colors = []
        
        for metric_key, metric in metrics_data.items():
            if metric and metric.get("score") is not None:
                labels.append(self.metric_labels.get(metric_key, metric_key))
                score = metric.get("score", 0.0)
                values.append(score * 100)  # Convert to percentage
                
                # Color based on score
                if score >= 0.8:
                    colors.append(self.metric_colors["excellent"])
                elif score >= 0.6:
                    colors.append(self.metric_colors["good"])
                elif score >= 0.4:
                    colors.append(self.metric_colors["fair"])
                else:
                    colors.append(self.metric_colors["needs_improvement"])
        
        if not labels:
            return
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=values,
                y=labels,
                orientation='h',
                marker=dict(color=colors),
                text=[f"{v:.1f}%" for v in values],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="RAG Quality Metrics",
            xaxis_title="Score (%)",
            yaxis_title="Metric",
            height=max(300, len(labels) * 60),
            showlegend=False,
            xaxis=dict(range=[0, 100]),
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_summary(self, evaluation_data: Dict[str, Any]):
        """Render summary and recommendations."""
        st.markdown("#### 💡 Summary & Recommendations")
        
        summary = evaluation_data.get("summary", "")
        
        if summary:
            st.info(f"**Summary:** {summary}")
        
        # Generate recommendations based on metrics
        recommendations = self._generate_recommendations(evaluation_data)
        
        if recommendations:
            st.markdown("**Recommendations:**")
            for rec in recommendations:
                st.markdown(f"- {rec}")
    
    def _generate_recommendations(self, evaluation_data: Dict[str, Any]) -> list:
        """Generate actionable recommendations based on metrics."""
        recommendations = []
        metrics_data = evaluation_data.get("metrics", {})
        
        # Check each metric and provide recommendations
        if "answer_relevance" in metrics_data:
            metric = metrics_data["answer_relevance"]
            if metric and metric.get("score", 0) < 0.6:
                recommendations.append(
                    "🔍 **Improve Answer Relevance:** Consider refining your query or trying a different RAG strategy"
                )
        
        if "context_relevance" in metrics_data:
            metric = metrics_data["context_relevance"]
            if metric and metric.get("score", 0) < 0.6:
                recommendations.append(
                    "📚 **Improve Context Relevance:** Increase document retrieval limit or adjust chunking strategy"
                )
        
        if "groundedness" in metrics_data:
            metric = metrics_data["groundedness"]
            if metric and metric.get("score", 0) < 0.7:
                recommendations.append(
                    "⚠️ **Reduce Hallucinations:** The model may be generating unsupported information. "
                    "Try a more conservative generation strategy or increase retrieved documents."
                )
        
        if "context_precision" in metrics_data:
            metric = metrics_data["context_precision"]
            if metric and metric.get("score", 0) < 0.6:
                recommendations.append(
                    "🎯 **Improve Ranking:** Consider using ReRank or Sentence Window strategy for better document ranking"
                )
        
        if "context_recall" in metrics_data:
            metric = metrics_data["context_recall"]
            if metric and metric.get("score", 0) < 0.6:
                recommendations.append(
                    "📊 **Improve Recall:** Retrieve more documents or use a strategy with better coverage like Auto-Merging"
                )
        
        # Strategy-specific recommendations
        overall_score = evaluation_data.get("overall_score", 0)
        if overall_score < 0.6:
            recommendations.append(
                "🚀 **Try Advanced Strategy:** Consider using Fusion RAG or Auto-Merging for better results"
            )
        
        # If no issues, provide positive feedback
        if not recommendations:
            recommendations.append(
                "✨ **Excellent Performance:** All metrics are within acceptable ranges. Keep up the good work!"
            )
        
        return recommendations
    
    def _get_status_from_score(self, score: float) -> str:
        """Get status label from score."""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        else:
            return "Needs Improvement"