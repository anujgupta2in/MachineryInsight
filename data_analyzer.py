import pandas as pd
import numpy as np
from typing import Optional, Dict, List

class MachineryDataAnalyzer:
    """Class for analyzing ship machinery data"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._clean_data()
    
    def _clean_data(self):
        """Clean and preprocess the data"""
        # Handle the critical machinery column - check first column if it's unnamed
        if 'Critical_Machinery' in self.df.columns:
            self.df['Is_Critical'] = self.df['Critical_Machinery'].astype(str).str.upper() == 'C'
        elif len(self.df.columns) > 0:
            # Check if first column contains 'C' values for critical machinery
            first_col = self.df.iloc[:, 0].astype(str).str.upper()
            self.df['Is_Critical'] = first_col == 'C'
        else:
            self.df['Is_Critical'] = False
        
        # Clean string columns
        string_columns = ['Maker', 'Model', 'Component Type', 'Particulars', 
                         'Machinery', 'Shell Component', 'Component Name']
        
        for col in string_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.strip()
                self.df[col] = self.df[col].replace(['nan', 'NaN', ''], 'Unknown')
    
    def get_total_machinery_count(self) -> int:
        """Get total count of machinery entries"""
        return len(self.df)
    
    def get_unique_machinery_count(self) -> int:
        """Get count of unique machinery"""
        return self.df['Machinery'].nunique()
    
    def get_unique_machinery_with_details(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Get unique machinery with their makers, models, and components"""
        # Use provided dataframe or default to self.df
        data = df if df is not None else self.df
        
        # Get unique combinations of machinery, maker, model, and component
        machinery_details = data.groupby(['Machinery', 'Maker', 'Model', 'Component Name']).size().reset_index(name='Count')
        
        # Create a summary with component names grouped per machinery-maker-model combination
        grouped = machinery_details.groupby(['Machinery', 'Maker', 'Model']).agg({
            'Component Name': lambda x: ', '.join(sorted(x.unique())),
            'Count': 'sum'
        }).reset_index()
        
        # Rearrange columns as requested: Machinery, Component, Maker, Model, Count
        grouped = grouped[['Machinery', 'Component Name', 'Maker', 'Model', 'Count']]
        
        # Rename Component Name column for clarity
        grouped = grouped.rename(columns={'Component Name': 'Component'})
        
        grouped = grouped.sort_values(['Machinery', 'Count'], ascending=[True, False])
        
        return grouped
    
    def get_critical_machinery_count(self) -> int:
        """Get count of critical machinery"""
        return self.df['Is_Critical'].sum()
    
    def get_total_components_count(self) -> int:
        """Get total count of unique components"""
        return self.df['Component Name'].nunique()
    
    def get_unique_makers_count(self) -> int:
        """Get count of unique makers"""
        return self.df['Maker'].nunique()
    
    def get_unique_models_count(self) -> int:
        """Get count of unique models"""
        return self.df['Model'].nunique()
    
    def apply_filters(self, maker: Optional[str] = None, model: Optional[str] = None, 
                     machinery: Optional[str] = None, component: Optional[str] = None, 
                     critical_filter: str = 'All') -> pd.DataFrame:
        """Apply filters to the dataframe"""
        filtered_df = self.df.copy()
        
        if maker:
            filtered_df = filtered_df[filtered_df['Maker'] == maker]
        
        if model:
            filtered_df = filtered_df[filtered_df['Model'] == model]
        
        if machinery:
            filtered_df = filtered_df[filtered_df['Machinery'] == machinery]
        
        if component:
            filtered_df = filtered_df[filtered_df['Component Name'] == component]
        
        if critical_filter == 'Critical Only':
            filtered_df = filtered_df[filtered_df['Is_Critical'] == True]
        elif critical_filter == 'Non-Critical Only':
            filtered_df = filtered_df[filtered_df['Is_Critical'] == False]
        
        return filtered_df
    
    def get_maker_analysis(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Get maker analysis with counts and percentages"""
        if df is None:
            df = self.df
        
        maker_counts = df['Maker'].value_counts().reset_index()
        maker_counts.columns = ['Maker', 'Count']
        maker_counts['Percentage'] = (maker_counts['Count'] / len(df) * 100).round(2)
        
        return maker_counts
    
    def get_maker_model_matrix(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Get maker-model combination analysis"""
        if df is None:
            df = self.df
        
        maker_model = df.groupby(['Maker', 'Model']).size().reset_index(name='Count')
        maker_model = maker_model.sort_values('Count', ascending=False)
        
        return maker_model
    
    def get_component_hierarchy(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Get component hierarchy analysis"""
        if df is None:
            df = self.df
        
        hierarchy = df.groupby(['Machinery', 'Shell Component', 'Component Type', 'Component Name']).size().reset_index(name='Count')
        hierarchy = hierarchy.sort_values('Count', ascending=False)
        
        return hierarchy
    
    def get_component_maker_analysis(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Get component analysis by maker"""
        if df is None:
            df = self.df
        
        component_maker = df.groupby(['Maker', 'Component Type']).size().reset_index(name='Count')
        component_maker = component_maker.sort_values('Count', ascending=False)
        
        return component_maker
    
    def get_critical_machinery_analysis(self, df: Optional[pd.DataFrame] = None) -> Dict:
        """Get critical machinery analysis"""
        if df is None:
            df = self.df
        
        # Ensure Is_Critical column exists
        if 'Is_Critical' not in df.columns:
            # Create the column if it doesn't exist
            if 'Critical_Machinery' in df.columns:
                df = df.copy()
                df['Is_Critical'] = df['Critical_Machinery'].astype(str).str.upper() == 'C'
            elif len(df.columns) > 0:
                df = df.copy()
                first_col = df.iloc[:, 0].astype(str).str.upper()
                df['Is_Critical'] = first_col == 'C'
            else:
                df = df.copy()
                df['Is_Critical'] = False
        
        total = len(df)
        critical = df['Is_Critical'].sum()
        non_critical = total - critical
        
        return {
            'total': total,
            'critical': critical,
            'non_critical': non_critical,
            'critical_percentage': (critical / total * 100) if total > 0 else 0
        }
    
    def get_critical_by_maker(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Get critical machinery breakdown by maker"""
        if df is None:
            df = self.df
        
        # Ensure Is_Critical column exists
        if 'Is_Critical' not in df.columns:
            if 'Critical_Machinery' in df.columns:
                df = df.copy()
                df['Is_Critical'] = df['Critical_Machinery'].astype(str).str.upper() == 'C'
            elif len(df.columns) > 0:
                df = df.copy()
                first_col = df.iloc[:, 0].astype(str).str.upper()
                df['Is_Critical'] = first_col == 'C'
            else:
                df = df.copy()
                df['Is_Critical'] = False
        
        critical_maker = df[df['Is_Critical'] == True]['Maker'].value_counts().reset_index()
        critical_maker.columns = ['Maker', 'Critical_Count']
        
        return critical_maker
    
    def get_critical_machinery_details(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Get detailed information about critical machinery"""
        if df is None:
            df = self.df
        
        # Ensure Is_Critical column exists
        if 'Is_Critical' not in df.columns:
            if 'Critical_Machinery' in df.columns:
                df = df.copy()
                df['Is_Critical'] = df['Critical_Machinery'].astype(str).str.upper() == 'C'
            elif len(df.columns) > 0:
                df = df.copy()
                first_col = df.iloc[:, 0].astype(str).str.upper()
                df['Is_Critical'] = first_col == 'C'
            else:
                df = df.copy()
                df['Is_Critical'] = False
        
        critical_df = df[df['Is_Critical'] == True].copy()
        
        # Select and arrange columns as requested: Machinery, Component, Maker, Model, Particulars
        columns = ['Machinery', 'Component Name', 'Maker', 'Model', 'Particulars']
        
        # Rename Component Name to Component for consistency
        result_df = critical_df[columns].copy()
        result_df = result_df.rename(columns={'Component Name': 'Component'})
        
        return result_df
    
    def get_detailed_hierarchy(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Get detailed hierarchy with machinery and component information"""
        if df is None:
            df = self.df
        
        # Create a detailed hierarchy showing Machinery -> Component Name relationships
        hierarchy_data = df.groupby(['Machinery', 'Component Name']).agg({
            'Maker': 'first',
            'Model': 'first',
            'Component Type': 'first',
            'Shell Component': 'first'
        }).reset_index()
        
        # Add count of occurrences for each machinery-component combination
        counts = df.groupby(['Machinery', 'Component Name']).size().reset_index(name='Count')
        hierarchy_data = hierarchy_data.merge(counts, on=['Machinery', 'Component Name'])
        
        # Rename Component Name to Component for consistency
        hierarchy_data = hierarchy_data.rename(columns={'Component Name': 'Component'})
        
        # Reorder columns: Machinery, Component, then other details
        column_order = ['Machinery', 'Component', 'Component Type', 'Shell Component', 'Maker', 'Model', 'Count']
        hierarchy_data = hierarchy_data[column_order]
        
        # Sort by machinery and component count
        hierarchy_data = hierarchy_data.sort_values(['Machinery', 'Count'], ascending=[True, False])
        
        return hierarchy_data
    
    def get_machinery_breakdown(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Get detailed breakdown of machinery and their components"""
        if df is None:
            df = self.df
        
        # Select relevant columns and remove duplicates
        breakdown_cols = ['Machinery', 'Shell Component', 'Component Type', 
                         'Component Name', 'Maker', 'Model', 'Is_Critical']
        
        breakdown_df = df[breakdown_cols].drop_duplicates()
        breakdown_df = breakdown_df.sort_values(['Machinery', 'Shell Component', 'Component Type'])
        
        return breakdown_df
    
    def get_summary_statistics(self, df: Optional[pd.DataFrame] = None) -> Dict:
        """Get comprehensive summary statistics"""
        if df is None:
            df = self.df
        
        stats = {
            'total_records': len(df),
            'unique_machinery': df['Machinery'].nunique(),
            'unique_makers': df['Maker'].nunique(),
            'unique_models': df['Model'].nunique(),
            'unique_component_types': df['Component Type'].nunique(),
            'unique_components': df['Component Name'].nunique(),
            'critical_machinery': df['Is_Critical'].sum(),
            'critical_percentage': (df['Is_Critical'].sum() / len(df) * 100) if len(df) > 0 else 0
        }
        
        return stats
