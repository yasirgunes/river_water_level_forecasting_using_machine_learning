#!/usr/bin/env python3
"""
Dataset Analysis for Hydrological Water Level Prediction
=========================================================

This module provides comprehensive analysis of the combined meteorological and 
hydrological dataset used for water level/discharge prediction models.

Based on USGS site 03443000 water level data combined with meteorological data
from Meteostat API for the corresponding location (35.4333, -82.0333, 660m).

Dataset characteristics:
- Target variables: stage_m (water level in meters), discharge_cms (discharge in cubic meters/second)
- Meteorological features: tavg, prcp, wspd, pres, rhum
- Time period: 1995-10-01 to 2025-06-01
- Frequency: Daily observations

Author: Generated analysis following hydrological modeling conventions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import jarque_bera, shapiro, anderson
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class HydrologicalDatasetAnalyzer:
    """
    Comprehensive dataset analyzer for hydrological time series data
    following conventions from water resources engineering literature.
    """
    
    def __init__(self, data_path="combined_dataset.csv"):
        """
        Initialize the analyzer with the dataset.
        
        Parameters:
        -----------
        data_path : str
            Path to the combined dataset CSV file
        """
        self.data_path = data_path
        self.data = None
        self.load_data()
        
    def load_data(self):
        """Load and prepare the dataset for analysis."""
        try:
            self.data = pd.read_csv(self.data_path, parse_dates=['datetime'], index_col='datetime')
            print(f"Dataset loaded successfully: {self.data.shape}")
            print(f"Date range: {self.data.index.min()} to {self.data.index.max()}")
        except FileNotFoundError:
            print(f"Error: Could not find {self.data_path}")
            return None
    
    def basic_statistics(self):
        """
        Generate comprehensive descriptive statistics following 
        hydrological analysis conventions.
        """
        print("\n" + "="*80)
        print("BASIC DATASET STATISTICS")
        print("="*80)
        
        # Basic info
        print(f"Dataset shape: {self.data.shape}")
        print(f"Time span: {(self.data.index.max() - self.data.index.min()).days} days")
        print(f"Missing values per column:")
        for col in self.data.columns:
            missing = self.data[col].isnull().sum()
            percentage = (missing / len(self.data)) * 100
            print(f"  {col}: {missing} ({percentage:.2f}%)")
        
        print(f"\nDescriptive Statistics:")
        print("-" * 50)
        
        # Enhanced descriptive statistics
        desc_stats = self.data.describe()
        
        # Add additional statistics relevant for hydrology
        additional_stats = pd.DataFrame(index=['skewness', 'kurtosis', 'cv'], 
                                       columns=self.data.columns)
        
        for col in self.data.columns:
            additional_stats.loc['skewness', col] = stats.skew(self.data[col].dropna())
            additional_stats.loc['kurtosis', col] = stats.kurtosis(self.data[col].dropna())
            additional_stats.loc['cv', col] = self.data[col].std() / self.data[col].mean()  # Coefficient of variation
        
        # Combine statistics
        combined_stats = pd.concat([desc_stats, additional_stats])
        print(combined_stats.round(4))
        
        return combined_stats
    
    def temporal_analysis(self):
        """
        Analyze temporal patterns in the hydrological data.
        """
        print("\n" + "="*80)
        print("TEMPORAL PATTERN ANALYSIS")
        print("="*80)
        
        # Create time-based features for analysis
        data_temp = self.data.copy()
        data_temp['year'] = data_temp.index.year
        data_temp['month'] = data_temp.index.month
        data_temp['day_of_year'] = data_temp.index.dayofyear
        data_temp['season'] = data_temp['month'].map({12: 'Winter', 1: 'Winter', 2: 'Winter',
                                                     3: 'Spring', 4: 'Spring', 5: 'Spring',
                                                     6: 'Summer', 7: 'Summer', 8: 'Summer',
                                                     9: 'Fall', 10: 'Fall', 11: 'Fall'})
        
        # Seasonal statistics for key variables
        print("Seasonal Statistics for Water Level (stage_m):")
        print("-" * 50)
        seasonal_stage = data_temp.groupby('season')['stage_m'].agg(['mean', 'std', 'min', 'max'])
        print(seasonal_stage.round(3))
        
        print("\nSeasonal Statistics for Precipitation (prcp):")
        print("-" * 50)
        seasonal_prcp = data_temp.groupby('season')['prcp'].agg(['mean', 'std', 'min', 'max'])
        print(seasonal_prcp.round(3))
        
        # Monthly averages
        print("\nMonthly Averages:")
        print("-" * 50)
        monthly_avg = data_temp.groupby('month')[['stage_m', 'prcp', 'tavg']].mean()
        print(monthly_avg.round(3))
        
        # Inter-annual variability
        print("\nInter-annual Statistics:")
        print("-" * 50)
        annual_stats = data_temp.groupby('year')[['stage_m', 'prcp']].agg(['mean', 'std'])
        print(f"Annual stage variation (CV): {(annual_stats[('stage_m', 'std')] / annual_stats[('stage_m', 'mean')]).mean():.3f}")
        print(f"Annual precipitation variation (CV): {(annual_stats[('prcp', 'std')] / annual_stats[('prcp', 'mean')]).mean():.3f}")
        
        return data_temp
    
    def correlation_analysis(self):
        """
        Comprehensive correlation analysis between variables.
        """
        print("\n" + "="*80)
        print("CORRELATION ANALYSIS")
        print("="*80)
        
        # Pearson correlation matrix
        corr_matrix = self.data.corr()
        print("Pearson Correlation Matrix:")
        print("-" * 30)
        print(corr_matrix.round(3))
        
        # Focus on correlations with target variables
        print("\nCorrelations with Water Level (stage_m):")
        print("-" * 40)
        stage_corr = corr_matrix['stage_m'].drop('stage_m').sort_values(key=abs, ascending=False)
        for var, corr in stage_corr.items():
            print(f"{var:15s}: {corr:6.3f}")
        
        if 'discharge_cms' in self.data.columns:
            print("\nCorrelations with Discharge (discharge_cms):")
            print("-" * 45)
            discharge_corr = corr_matrix['discharge_cms'].drop(['discharge_cms', 'stage_m']).sort_values(key=abs, ascending=False)
            for var, corr in discharge_corr.items():
                print(f"{var:15s}: {corr:6.3f}")
        
        # Lag correlations (important for time series)
        print("\nLag Correlations (Precipitation vs Water Level):")
        print("-" * 50)
        for lag in range(0, 8):
            if lag == 0:
                lag_corr = self.data['prcp'].corr(self.data['stage_m'])
            else:
                lag_corr = self.data['prcp'].corr(self.data['stage_m'].shift(-lag))
            print(f"Lag {lag} days: {lag_corr:6.3f}")
        
        return corr_matrix
    
    def extreme_events_analysis(self):
        """
        Analyze extreme events in hydrological data (floods/droughts).
        """
        print("\n" + "="*80)
        print("EXTREME EVENTS ANALYSIS")
        print("="*80)
        
        # Define thresholds for extreme events
        stage_data = self.data['stage_m'].dropna()
        
        # Percentile-based thresholds (common in hydrology)
        p95_stage = stage_data.quantile(0.95)  # High flow threshold
        p5_stage = stage_data.quantile(0.05)   # Low flow threshold
        p99_stage = stage_data.quantile(0.99)  # Extreme high flow
        p1_stage = stage_data.quantile(0.01)   # Extreme low flow
        
        print(f"Water Level Thresholds (m):")
        print(f"  Extreme low (1st percentile):  {p1_stage:.3f}")
        print(f"  Low flow (5th percentile):     {p5_stage:.3f}")
        print(f"  High flow (95th percentile):   {p95_stage:.3f}")
        print(f"  Extreme high (99th percentile): {p99_stage:.3f}")
        
        # Count extreme events
        extreme_high = (stage_data >= p99_stage).sum()
        extreme_low = (stage_data <= p1_stage).sum()
        high_flow = (stage_data >= p95_stage).sum()
        low_flow = (stage_data <= p5_stage).sum()
        
        print(f"\nExtreme Event Counts:")
        print(f"  Extreme high flow events: {extreme_high}")
        print(f"  High flow events:         {high_flow}")
        print(f"  Low flow events:          {low_flow}")
        print(f"  Extreme low flow events:  {extreme_low}")
        
        # Precipitation extremes
        prcp_data = self.data['prcp'].dropna()
        p95_prcp = prcp_data.quantile(0.95)
        p99_prcp = prcp_data.quantile(0.99)
        
        print(f"\nPrecipitation Thresholds:")
        print(f"  Heavy precipitation (95th percentile): {p95_prcp:.1f} mm")
        print(f"  Extreme precipitation (99th percentile): {p99_prcp:.1f} mm")
        
        # Drought analysis (consecutive days with low precipitation)
        dry_days = (prcp_data == 0).astype(int)
        dry_spells = []
        current_spell = 0
        
        for day in dry_days:
            if day == 1:
                current_spell += 1
            else:
                if current_spell > 0:
                    dry_spells.append(current_spell)
                current_spell = 0
        
        if current_spell > 0:
            dry_spells.append(current_spell)
        
        if dry_spells:
            print(f"\nDry Spell Analysis:")
            print(f"  Maximum consecutive dry days: {max(dry_spells)}")
            print(f"  Average dry spell length: {np.mean(dry_spells):.1f} days")
            print(f"  Number of dry spells ≥7 days: {sum(1 for x in dry_spells if x >= 7)}")
        
        return {
            'stage_thresholds': {'p1': p1_stage, 'p5': p5_stage, 'p95': p95_stage, 'p99': p99_stage},
            'prcp_thresholds': {'p95': p95_prcp, 'p99': p99_prcp},
            'extreme_counts': {'extreme_high': extreme_high, 'high_flow': high_flow, 
                             'low_flow': low_flow, 'extreme_low': extreme_low}
        }
    
    def data_quality_assessment(self):
        """
        Assess data quality issues relevant for time series modeling.
        """
        print("\n" + "="*80)
        print("DATA QUALITY ASSESSMENT")
        print("="*80)
        
        # Missing data patterns
        print("Missing Data Patterns:")
        print("-" * 25)
        missing_data = self.data.isnull()
        
        # Consecutive missing values
        for col in self.data.columns:
            if missing_data[col].any():
                # Find consecutive missing value groups
                missing_groups = missing_data[col].groupby((~missing_data[col]).cumsum())
                max_consecutive = missing_groups.sum().max()
                print(f"{col}: Max consecutive missing: {max_consecutive} days")
        
        # Outlier detection using IQR method
        print("\nOutlier Detection (IQR method):")
        print("-" * 35)
        
        for col in self.data.select_dtypes(include=[np.number]).columns:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((self.data[col] < lower_bound) | (self.data[col] > upper_bound)).sum()
            outlier_pct = (outliers / len(self.data)) * 100
            print(f"{col:15s}: {outliers:4d} outliers ({outlier_pct:5.2f}%)")
        
        # Data continuity check
        print("\nData Continuity:")
        print("-" * 20)
        date_range = pd.date_range(start=self.data.index.min(), 
                                 end=self.data.index.max(), 
                                 freq='D')
        missing_dates = len(date_range) - len(self.data)
        print(f"Expected daily records: {len(date_range)}")
        print(f"Actual records: {len(self.data)}")
        print(f"Missing dates: {missing_dates}")
        
        # Temporal gaps larger than 1 day
        time_diffs = self.data.index.to_series().diff()
        large_gaps = time_diffs[time_diffs > pd.Timedelta(days=1)]
        if not large_gaps.empty:
            print(f"Large temporal gaps (>1 day): {len(large_gaps)}")
            for gap_date, gap_size in large_gaps.items():
                print(f"  {gap_date.date()}: {gap_size}")
    
    def normality_tests(self):
        """
        Test normality of variables for model assumption validation.
        """
        print("\n" + "="*80)
        print("NORMALITY TESTS")
        print("="*80)
        
        print(f"{'Variable':15s} {'Shapiro-Wilk':>15s} {'Jarque-Bera':>15s} {'Anderson-Darling':>18s}")
        print("-" * 65)
        
        for col in self.data.select_dtypes(include=[np.number]).columns:
            data_clean = self.data[col].dropna()
            
            # Shapiro-Wilk test (good for smaller samples)
            if len(data_clean) <= 5000:  # Shapiro-Wilk limit
                shapiro_stat, shapiro_p = shapiro(data_clean[:5000])
                shapiro_result = f"{shapiro_p:.4f}"
            else:
                shapiro_result = "N/A (n>5000)"
            
            # Jarque-Bera test
            jb_stat, jb_p = jarque_bera(data_clean)
            
            # Anderson-Darling test
            ad_stat, ad_critical, ad_significance = anderson(data_clean, dist='norm')
            # Check if statistic exceeds critical value at 5% significance
            ad_result = "Normal" if ad_stat < ad_critical[2] else "Non-normal"
            
            print(f"{col:15s} {shapiro_result:>15s} {jb_p:>15.4f} {ad_result:>18s}")
        
        print("\nInterpretation:")
        print("- p-value > 0.05 suggests data is normally distributed")
        print("- p-value ≤ 0.05 suggests data is not normally distributed")
    
    def generate_summary_report(self):
        """
        Generate a comprehensive summary report.
        """
        print("\n" + "="*80)
        print("DATASET SUMMARY REPORT")
        print("="*80)
        
        # Dataset overview
        print("Dataset Overview:")
        print(f"  • Time series length: {len(self.data)} daily observations")
        print(f"  • Time period: {self.data.index.min().date()} to {self.data.index.max().date()}")
        print(f"  • Variables: {len(self.data.columns)} ({', '.join(self.data.columns)})")
        
        # Key statistics
        stage_stats = self.data['stage_m'].describe()
        print(f"\nWater Level Statistics:")
        print(f"  • Range: {stage_stats['min']:.2f} - {stage_stats['max']:.2f} m")
        print(f"  • Mean: {stage_stats['mean']:.2f} m")
        print(f"  • Standard deviation: {stage_stats['std']:.2f} m")
        print(f"  • Coefficient of variation: {stage_stats['std']/stage_stats['mean']:.3f}")
        
        prcp_stats = self.data['prcp'].describe()
        print(f"\nPrecipitation Statistics:")
        print(f"  • Annual total (mean): {prcp_stats['mean'] * 365:.0f} mm/year")
        print(f"  • Maximum daily: {prcp_stats['max']:.1f} mm")
        print(f"  • Days with precipitation: {(self.data['prcp'] > 0).sum()} ({(self.data['prcp'] > 0).mean()*100:.1f}%)")
        
        # Data quality summary
        total_missing = self.data.isnull().sum().sum()
        missing_pct = (total_missing / (len(self.data) * len(self.data.columns))) * 100
        print(f"\nData Quality:")
        print(f"  • Total missing values: {total_missing} ({missing_pct:.2f}%)")
        print(f"  • Data completeness: {100-missing_pct:.2f}%")
        
        # Modeling recommendations
        print(f"\nModeling Recommendations:")
        print(f"  • Time series is suitable for machine learning with {len(self.data)} observations")
        print(f"  • Consider seasonal patterns in model features")
        print(f"  • Include lag features for precipitation (correlation analysis shows delayed response)")
        print(f"  • Monitor for extreme events in validation")
        
        # Station information
        print(f"\nUSGS Station Information:")
        print(f"  • Station ID: 03443000")
        print(f"  • Location: 35.4333°N, 82.0333°W, 660m elevation")
        print(f"  • Parameter: 00065 (Gage height, feet) converted to meters")
        print(f"  • Data source: USGS National Water Information System")
    
    def create_visualizations(self, save_plots=True):
        """
        Create comprehensive visualizations for the dataset.
        """
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Figure 1: Time series overview
        fig, axes = plt.subplots(3, 1, figsize=(15, 12))
        
        # Water level time series
        axes[0].plot(self.data.index, self.data['stage_m'], linewidth=0.8, alpha=0.8)
        axes[0].set_title('Water Level Time Series', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Stage (m)')
        axes[0].grid(True, alpha=0.3)
        
        # Precipitation time series
        axes[1].bar(self.data.index, self.data['prcp'], width=1, alpha=0.7)
        axes[1].set_title('Daily Precipitation', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Precipitation (mm)')
        axes[1].grid(True, alpha=0.3)
        
        # Temperature time series
        axes[2].plot(self.data.index, self.data['tavg'], linewidth=0.8, alpha=0.8, color='red')
        axes[2].set_title('Average Temperature', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Temperature (°C)')
        axes[2].set_xlabel('Date')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('time_series_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Figure 2: Correlation heatmap
        plt.figure(figsize=(10, 8))
        corr_matrix = self.data.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Variable Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        if save_plots:
            plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Figure 3: Seasonal patterns
        data_temp = self.data.copy()
        data_temp['month'] = data_temp.index.month
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Monthly boxplots for water level
        monthly_stage = [self.data[self.data.index.month == month]['stage_m'].values 
                        for month in range(1, 13)]
        axes[0,0].boxplot(monthly_stage, labels=['Jan','Feb','Mar','Apr','May','Jun',
                                               'Jul','Aug','Sep','Oct','Nov','Dec'])
        axes[0,0].set_title('Monthly Water Level Distribution', fontweight='bold')
        axes[0,0].set_ylabel('Stage (m)')
        axes[0,0].grid(True, alpha=0.3)
        
        # Monthly precipitation
        monthly_prcp = data_temp.groupby('month')['prcp'].mean()
        axes[0,1].bar(range(1,13), monthly_prcp.values, alpha=0.7)
        axes[0,1].set_title('Average Monthly Precipitation', fontweight='bold')
        axes[0,1].set_ylabel('Precipitation (mm)')
        axes[0,1].set_xlabel('Month')
        axes[0,1].set_xticks(range(1,13))
        axes[0,1].set_xticklabels(['Jan','Feb','Mar','Apr','May','Jun',
                                  'Jul','Aug','Sep','Oct','Nov','Dec'])
        axes[0,1].grid(True, alpha=0.3)
        
        # Scatter plot: Precipitation vs Water Level
        axes[1,0].scatter(self.data['prcp'], self.data['stage_m'], alpha=0.5, s=10)
        axes[1,0].set_xlabel('Precipitation (mm)')
        axes[1,0].set_ylabel('Water Level (m)')
        axes[1,0].set_title('Precipitation vs Water Level', fontweight='bold')
        axes[1,0].grid(True, alpha=0.3)
        
        # Distribution of water level
        axes[1,1].hist(self.data['stage_m'].dropna(), bins=50, alpha=0.7, density=True)
        axes[1,1].set_xlabel('Water Level (m)')
        axes[1,1].set_ylabel('Density')
        axes[1,1].set_title('Water Level Distribution', fontweight='bold')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_plots:
            plt.savefig('seasonal_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations generated successfully!")
        if save_plots:
            print("Plots saved as PNG files in current directory")
    
    def run_complete_analysis(self, save_plots=True):
        """
        Run the complete dataset analysis workflow.
        """
        print("HYDROLOGICAL DATASET ANALYSIS")
        print("="*80)
        print("Analysis following conventions from water resources literature")
        print(f"Dataset: {self.data_path}")
        print("="*80)
        
        # Run all analysis components
        self.basic_statistics()
        self.temporal_analysis()
        self.correlation_analysis()
        self.extreme_events_analysis()
        self.data_quality_assessment()
        self.normality_tests()
        self.generate_summary_report()
        self.create_visualizations(save_plots=save_plots)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print("Refer to the generated plots and statistics for model development guidance.")


if __name__ == "__main__":
    # Run the complete analysis
    analyzer = HydrologicalDatasetAnalyzer("combined_dataset.csv")
    
    if analyzer.data is not None:
        analyzer.run_complete_analysis(save_plots=True)
    else:
        print("Could not load dataset. Please ensure 'combined_dataset.csv' exists in the current directory.")
        print("Run the data preparation notebooks first to generate the dataset.")
