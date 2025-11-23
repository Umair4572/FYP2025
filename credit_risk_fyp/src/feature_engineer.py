"""
Feature Engineering Module for Credit Risk Assessment FYP
Creates derived features, ratios, interactions, and aggregations
"""

import logging
from typing import Dict, List, Optional, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import datetime

from .config import FEATURE_ENGINEERING_CONFIG
from .utils import save_pickle, load_pickle

# Setup logger
logger = logging.getLogger('credit_risk_fyp.feature_engineer')


class FeatureEngineer:
    """
    Comprehensive feature engineering for credit risk data.

    Creates:
    - Financial ratios (loan-to-income, installment-to-income, etc.)
    - Credit behavior indicators (delinquency score, account diversity)
    - Interaction features (int_rate × dti, fico × dti)
    - Time-based features (credit age, loan season)
    - Aggregation features
    - Binned/discretized features
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize FeatureEngineer.

        Args:
            config: Optional feature engineering configuration
        """
        self.config = config or FEATURE_ENGINEERING_CONFIG
        self.feature_names = []
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> 'FeatureEngineer':
        """
        Learn parameters for feature engineering (e.g., quantile bins).

        Args:
            df: Training DataFrame

        Returns:
            Self for method chaining
        """
        logger.info("Fitting feature engineer...")

        # Calculate quantiles for binning if needed
        self.bin_edges = {}

        if self.config['binning_strategy'] == 'quantile':
            n_bins = self.config['n_bins']

            # Income bins
            if 'annual_inc' in df.columns:
                self.bin_edges['income'] = df['annual_inc'].quantile(
                    np.linspace(0, 1, n_bins + 1)
                ).values

            # Loan amount bins
            if 'loan_amnt' in df.columns:
                self.bin_edges['loan_amnt'] = df['loan_amnt'].quantile(
                    np.linspace(0, 1, n_bins + 1)
                ).values

        self.is_fitted = True
        logger.info("Feature engineer fitting complete")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering transformations.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with engineered features
        """
        if not self.is_fitted:
            logger.warning("Feature engineer not fitted. Fitting on current data...")
            self.fit(df)

        logger.info("Engineering features...")

        df = df.copy()
        initial_features = len(df.columns)

        # Create different types of features based on config
        if self.config['create_ratios']:
            df = self.create_ratio_features(df)

        if self.config['create_time_features']:
            df = self.create_time_features(df)

        df = self.create_credit_features(df)

        if self.config['create_interactions']:
            df = self.create_interaction_features(df)

        if self.config['create_aggregations']:
            df = self.create_aggregation_features(df)

        # Binning
        df = self.create_binned_features(df)

        final_features = len(df.columns)
        new_features = final_features - initial_features

        logger.info(f"Created {new_features} new features (total: {final_features})")

        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with engineered features
        """
        self.fit(df)
        return self.transform(df)

    def create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create financial ratio features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with ratio features added
        """
        logger.info("Creating ratio features...")

        # Loan to income ratio
        if 'loan_amnt' in df.columns and 'annual_inc' in df.columns:
            df['loan_to_income'] = df['loan_amnt'] / (df['annual_inc'] + 1)

        # Installment to income ratio
        if 'installment' in df.columns and 'annual_inc' in df.columns:
            df['installment_to_income'] = (df['installment'] * 12) / (df['annual_inc'] + 1)

        # DTI ratio normalized
        if 'dti' in df.columns:
            df['dti_ratio'] = df['dti'] / 100.0

        # Credit utilization
        if 'revol_bal' in df.columns and 'revol_util' in df.columns:
            df['credit_utilization'] = df['revol_util'] / 100.0

        # Payment coverage (can borrower afford monthly payment?)
        if 'installment' in df.columns and 'annual_inc' in df.columns:
            df['payment_coverage'] = df['installment'] / ((df['annual_inc'] / 12) + 1)

        # Revolving balance to income
        if 'revol_bal' in df.columns and 'annual_inc' in df.columns:
            df['revol_bal_to_income'] = df['revol_bal'] / (df['annual_inc'] + 1)

        # Funded amount ratio
        if 'funded_amnt' in df.columns and 'loan_amnt' in df.columns:
            df['funded_ratio'] = df['funded_amnt'] / (df['loan_amnt'] + 1)

        return df

    def create_credit_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create credit behavior indicators.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with credit features added
        """
        logger.info("Creating credit behavior features...")

        # Account diversity (open vs total accounts)
        if 'open_acc' in df.columns and 'total_acc' in df.columns:
            df['account_diversity'] = df['open_acc'] / (df['total_acc'] + 1)

        # Recent inquiry rate
        if 'inq_last_6mths' in df.columns:
            df['recent_inquiry_rate'] = df['inq_last_6mths'] / 6.0

        # Delinquency score (weighted sum of negative events)
        delinq_score = 0
        if 'delinq_2yrs' in df.columns:
            delinq_score += df['delinq_2yrs'] * 2
        if 'pub_rec' in df.columns:
            delinq_score += df['pub_rec'] * 3
        if 'pub_rec_bankruptcies' in df.columns:
            delinq_score += df['pub_rec_bankruptcies'] * 5

        df['delinquency_score'] = delinq_score

        # Total number of accounts
        if 'total_acc' in df.columns and 'open_acc' in df.columns:
            df['closed_acc'] = df['total_acc'] - df['open_acc']

        # Average account balance
        if 'total_bal_ex_mort' in df.columns and 'total_acc' in df.columns:
            df['avg_account_balance'] = df['total_bal_ex_mort'] / (df['total_acc'] + 1)

        return df

    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with time features added
        """
        logger.info("Creating time-based features...")

        # Credit age
        if 'earliest_cr_line' in df.columns:
            # Try to parse dates
            try:
                earliest_dates = pd.to_datetime(df['earliest_cr_line'], errors='coerce')
                reference_date = pd.Timestamp('2020-12-31')  # Use dataset end date

                # Calculate days since earliest credit line
                days_diff = (reference_date - earliest_dates).dt.days

                # Convert to years and months
                df['credit_age_years'] = days_diff / 365.25
                df['credit_age_months'] = days_diff / 30.44

            except Exception as e:
                logger.warning(f"Could not parse earliest_cr_line: {e}")

        # Loan issue date features
        if 'issue_d' in df.columns:
            try:
                issue_dates = pd.to_datetime(df['issue_d'], errors='coerce')

                # Extract components
                df['issue_year'] = issue_dates.dt.year
                df['issue_month'] = issue_dates.dt.month
                df['issue_quarter'] = issue_dates.dt.quarter

                # Season (financial quarters)
                df['loan_season'] = df['issue_month'].map({
                    1: 'Q1', 2: 'Q1', 3: 'Q1',
                    4: 'Q2', 5: 'Q2', 6: 'Q2',
                    7: 'Q3', 8: 'Q3', 9: 'Q3',
                    10: 'Q4', 11: 'Q4', 12: 'Q4'
                })

            except Exception as e:
                logger.warning(f"Could not parse issue_d: {e}")

        # Employment length features
        if 'emp_length' in df.columns:
            # Parse employment length (e.g., "10+ years", "< 1 year")
            df['emp_length_numeric'] = df['emp_length'].replace({
                '< 1 year': 0,
                '1 year': 1,
                '2 years': 2,
                '3 years': 3,
                '4 years': 4,
                '5 years': 5,
                '6 years': 6,
                '7 years': 7,
                '8 years': 8,
                '9 years': 9,
                '10+ years': 10
            })

        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between important variables.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with interaction features added
        """
        logger.info("Creating interaction features...")

        # Interest rate × DTI
        if 'int_rate' in df.columns and 'dti' in df.columns:
            df['int_rate_x_dti'] = df['int_rate'] * df['dti']

        # Loan amount × Interest rate
        if 'loan_amnt' in df.columns and 'int_rate' in df.columns:
            df['loan_amnt_x_int_rate'] = df['loan_amnt'] * df['int_rate']

        # Interest rate × Term
        if 'int_rate' in df.columns and 'term' in df.columns:
            # Parse term (e.g., " 36 months", " 60 months")
            term_numeric = df['term'].str.extract('(\d+)', expand=False).astype(float)
            df['int_rate_x_term'] = df['int_rate'] * term_numeric

        # FICO × DTI
        # Check for FICO range columns
        fico_col = None
        if 'fico_range_high' in df.columns:
            fico_col = 'fico_range_high'
        elif 'fico_range_low' in df.columns:
            fico_col = 'fico_range_low'

        if fico_col and 'dti' in df.columns:
            df['fico_x_dti'] = df[fico_col] * df['dti']

        # Income × Employment length
        if 'annual_inc' in df.columns and 'emp_length_numeric' in df.columns:
            df['income_x_emp_length'] = df['annual_inc'] * df['emp_length_numeric']

        return df

    def create_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create aggregation and statistical features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with aggregation features added
        """
        logger.info("Creating aggregation features...")

        # Total credit accounts
        total_credit_cols = []
        for col in ['open_acc', 'total_acc', 'num_actv_bc_tl', 'num_bc_tl', 'num_sats']:
            if col in df.columns:
                total_credit_cols.append(col)

        if total_credit_cols:
            df['total_credit_accounts'] = df[total_credit_cols].sum(axis=1)

        # Total balance across all accounts
        balance_cols = []
        for col in ['tot_cur_bal', 'total_bal_ex_mort', 'total_bc_limit', 'total_rev_hi_lim']:
            if col in df.columns:
                balance_cols.append(col)

        if balance_cols:
            df['total_balance_all'] = df[balance_cols].sum(axis=1)

        return df

    def create_binned_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create binned/discretized features.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with binned features added
        """
        logger.info("Creating binned features...")

        # Income buckets
        if 'annual_inc' in df.columns:
            if 'income' in self.bin_edges:
                df['income_bucket'] = pd.cut(
                    df['annual_inc'],
                    bins=self.bin_edges['income'],
                    labels=range(len(self.bin_edges['income']) - 1),
                    include_lowest=True
                ).astype(float)
            else:
                # Use fixed bins
                df['income_bucket'] = pd.cut(
                    df['annual_inc'],
                    bins=[0, 30000, 50000, 75000, 100000, np.inf],
                    labels=[0, 1, 2, 3, 4],
                    include_lowest=True
                ).astype(float)

        # FICO buckets (credit score ranges)
        fico_col = None
        if 'fico_range_high' in df.columns:
            fico_col = 'fico_range_high'
        elif 'fico_range_low' in df.columns:
            fico_col = 'fico_range_low'

        if fico_col:
            df['fico_bucket'] = pd.cut(
                df[fico_col],
                bins=[0, 600, 660, 720, 780, 850],
                labels=[0, 1, 2, 3, 4],
                include_lowest=True
            ).astype(float)

        # Loan amount buckets
        if 'loan_amnt' in df.columns:
            if 'loan_amnt' in self.bin_edges:
                df['loan_amnt_bucket'] = pd.cut(
                    df['loan_amnt'],
                    bins=self.bin_edges['loan_amnt'],
                    labels=range(len(self.bin_edges['loan_amnt']) - 1),
                    include_lowest=True
                ).astype(float)
            else:
                df['loan_amnt_bucket'] = pd.cut(
                    df['loan_amnt'],
                    bins=[0, 5000, 10000, 20000, 30000, np.inf],
                    labels=[0, 1, 2, 3, 4],
                    include_lowest=True
                ).astype(float)

        # DTI buckets
        if 'dti' in df.columns:
            df['dti_bucket'] = pd.cut(
                df['dti'],
                bins=[0, 10, 20, 30, 40, np.inf],
                labels=[0, 1, 2, 3, 4],
                include_lowest=True
            ).astype(float)

        return df

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Save fitted feature engineer to file.

        Args:
            filepath: Path to save file
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted feature engineer")

        save_pickle(self, filepath)
        logger.info(f"Saved feature engineer to {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'FeatureEngineer':
        """
        Load fitted feature engineer from file.

        Args:
            filepath: Path to saved feature engineer

        Returns:
            Loaded FeatureEngineer instance
        """
        feature_engineer = load_pickle(filepath)
        logger.info(f"Loaded feature engineer from {filepath}")
        return feature_engineer
