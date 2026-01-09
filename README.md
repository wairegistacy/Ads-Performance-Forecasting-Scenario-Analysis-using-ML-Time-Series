# Ads Performance Forecasting & Scenario Analysis

Forecasting and decision-support system for digital advertising performance using machine learning and time-series analysis.

This project demonstrates how forecasting models can be translated into actionable planning insights for Ads, Strategy, and Operations teams. Rather than focusing only on prediction accuracy, the system emphasizes evaluation rigor, uncertainty-aware forecasting, and scenario-based decision support.

## 🔍 Problem Statement

Advertising teams need reliable forecasts of key performance indicators (KPIs) such as:
- impressions
- clicks
- spend
- conversions

These forecasts support decisions around:
- budget planning
- campaign optimization
- performance risk management

However, real-world Ads data is:
- noisy
- seasonal
- multi-campaign
- sensitive to changes in spend and engagement metrics

Simple averages or single-point forecasts are insufficient for operational decision-making.

## 🎯 Project Objectives

This project aims to:
1. Build a robust daily forecasting pipeline for Ads performance
2. Evaluate forecasts using time-based backtesting
3. Enable scenario analysis to support planning decisions under changing conditions
4. Frame results in a way that is interpretable and useful for stakeholders

## Steps
Built a machine learning–based forecasting system to predict daily Ads performance metrics (clicks, impressions, cost) across multiple campaigns.
Engineered lag features, rolling statistics, and calendar features to model temporal patterns in Ads data.
Evaluated models using time-based validation and rolling backtests (MAE, RMSE, MAPE).
Implemented scenario analysis to quantify the impact of changes in impressions and CTR on expected clicks, supporting Ads planning and budget decision-making.
Developed an interactive Streamlit planning application for forecasting and what-if analysis.

## 📊 Dataset
Real-world Ads performance dataset (campaign-level, daily granularity)
Metrics include:
- impressions
- clicks
- cost
- conversions
Data spans 91 consecutive days across multiple campaigns

A standardized data contract is enforced to ensure reproducibility and extensibility.
