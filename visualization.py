import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class MachineryVisualizer:
    """Class for creating visualizations of machinery data"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
        # Color schemes
        self.colors = px.colors.qualitative.Set3
        self.critical_colors = ['#ff6b6b', '#4ecdc4']  # Red for critical, teal for non-critical
    
    def create_maker_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Create maker distribution bar chart"""
        maker_counts = df['Maker'].value_counts().head(15)
        
        fig = px.bar(
            x=maker_counts.values,
            y=maker_counts.index,
            orientation='h',
            title="Top 15 Makers by Equipment Count",
            labels={'x': 'Count', 'y': 'Maker'},
            color=maker_counts.values,
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            height=500,
            showlegend=False,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        return fig
    
    def create_model_distribution(self, df: pd.DataFrame, maker: str) -> go.Figure:
        """Create model distribution for a specific maker"""
        maker_df = df[df['Maker'] == maker]
        model_counts = maker_df['Model'].value_counts().head(10)
        
        fig = px.pie(
            values=model_counts.values,
            names=model_counts.index,
            title=f"Model Distribution for {maker}",
            color_discrete_sequence=self.colors
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=500)
        
        return fig
    
    def create_top_models_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create top models chart across all makers"""
        model_counts = df['Model'].value_counts().head(15)
        
        fig = px.bar(
            x=model_counts.index,
            y=model_counts.values,
            title="Top 15 Models by Equipment Count",
            labels={'x': 'Model', 'y': 'Count'},
            color=model_counts.values,
            color_continuous_scale='plasma'
        )
        
        fig.update_layout(
            height=500,
            xaxis_tickangle=-45,
            showlegend=False
        )
        
        return fig
    
    def create_component_type_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Create component type distribution"""
        component_counts = df['Component Type'].value_counts().head(15)
        
        fig = px.treemap(
            names=component_counts.index,
            values=component_counts.values,
            title="Component Type Distribution (Top 15)",
            color=component_counts.values,
            color_continuous_scale='RdYlBu'
        )
        
        fig.update_layout(height=500)
        
        return fig
    
    def create_machinery_component_relationship(self, df: pd.DataFrame) -> go.Figure:
        """Create machinery vs component relationship chart"""
        # Get top machinery types and their component counts
        machinery_components = df.groupby('Machinery')['Component Name'].nunique().sort_values(ascending=False).head(10)
        
        fig = px.bar(
            x=machinery_components.index,
            y=machinery_components.values,
            title="Number of Unique Components by Machinery Type (Top 10)",
            labels={'x': 'Machinery', 'y': 'Unique Components'},
            color=machinery_components.values,
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            height=500,
            xaxis_tickangle=-45,
            showlegend=False
        )
        
        return fig
    
    def create_critical_machinery_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create critical vs non-critical machinery pie chart"""
        # Handle the critical machinery column
        if 'Is_Critical' in df.columns:
            critical_counts = df['Is_Critical'].value_counts()
        else:
            # Fallback to checking the first column for 'C' values
            if 'Critical_Machinery' in df.columns:
                critical_mask = df['Critical_Machinery'].astype(str).str.upper() == 'C'
            elif len(df.columns) > 0:
                critical_mask = df.iloc[:, 0].astype(str).str.upper() == 'C'
            else:
                critical_mask = pd.Series([False] * len(df))
            critical_counts = critical_mask.value_counts()
        
        labels = ['Non-Critical', 'Critical']
        values = [critical_counts.get(False, 0), critical_counts.get(True, 0)]
        
        fig = px.pie(
            values=values,
            names=labels,
            title="Critical vs Non-Critical Machinery Distribution",
            color_discrete_sequence=self.critical_colors
        )
        
        fig.update_traces(
            textposition='inside', 
            textinfo='percent+label+value',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )
        
        fig.update_layout(height=400)
        
        return fig
    
    def create_hierarchy_treemap(self, df: pd.DataFrame) -> go.Figure:
        """Create hierarchical treemap visualization"""
        # Prepare data for treemap
        hierarchy_data = []
        
        # Add machinery level
        machinery_counts = df['Machinery'].value_counts()
        for machinery, count in machinery_counts.items():
            if machinery and machinery.strip() and machinery != 'Unknown':
                hierarchy_data.append({
                    'ids': f"Machinery-{machinery}",
                    'labels': machinery,
                    'parents': "",
                    'values': count
                })
        
        # Add shell component level
        shell_component_counts = df.groupby(['Machinery', 'Shell Component']).size()
        for (machinery, shell_component), count in shell_component_counts.items():
            if (machinery and machinery.strip() and machinery != 'Unknown' and 
                shell_component and shell_component.strip() and shell_component != 'Unknown'):
                hierarchy_data.append({
                    'ids': f"Shell-{machinery}-{shell_component}",
                    'labels': shell_component,
                    'parents': f"Machinery-{machinery}",
                    'values': count
                })
        
        if not hierarchy_data:
            # Fallback simple treemap
            component_counts = df['Component Type'].value_counts().head(10)
            fig = px.treemap(
                names=component_counts.index,
                values=component_counts.values,
                title="Component Type Distribution"
            )
        else:
            # Create the treemap
            hierarchy_df = pd.DataFrame(hierarchy_data)
            
            fig = go.Figure(go.Treemap(
                ids=hierarchy_df['ids'],
                labels=hierarchy_df['labels'],
                parents=hierarchy_df['parents'],
                values=hierarchy_df['values'],
                branchvalues="total",
                maxdepth=3,
                textinfo="label+value+percent parent"
            ))
            
            fig.update_layout(title="Machinery Hierarchy Treemap")
        
        fig.update_layout(height=600)
        
        return fig
    
    def create_maker_model_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create maker-model combination heatmap"""
        # Get top makers and models
        top_makers = df['Maker'].value_counts().head(10).index
        
        # Create pivot table
        heatmap_data = df[df['Maker'].isin(top_makers)].groupby(['Maker', 'Model']).size().unstack(fill_value=0)
        
        # Limit to top models for readability
        if heatmap_data.shape[1] > 15:
            top_models = heatmap_data.sum().nlargest(15).index
            heatmap_data = heatmap_data[top_models]
        
        fig = px.imshow(
            heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            color_continuous_scale='viridis',
            title="Maker-Model Combination Heatmap (Top Makers & Models)",
            labels={'x': 'Model', 'y': 'Maker', 'color': 'Count'}
        )
        
        fig.update_layout(
            height=500,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_critical_by_maker_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create critical machinery distribution by maker"""
        # Handle the critical machinery column
        if 'Is_Critical' in df.columns:
            critical_df = df[df['Is_Critical'] == True]
        else:
            # Fallback to checking the first column for 'C' values
            if 'Critical_Machinery' in df.columns:
                critical_df = df[df['Critical_Machinery'].astype(str).str.upper() == 'C']
            elif len(df.columns) > 0:
                critical_df = df[df.iloc[:, 0].astype(str).str.upper() == 'C']
            else:
                critical_df = pd.DataFrame()
        
        if len(critical_df) == 0:
            # No critical machinery found
            fig = go.Figure()
            fig.add_annotation(
                text="No critical machinery found in the dataset",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            fig.update_layout(title="Critical Machinery by Maker", height=400)
            return fig
        
        critical_by_maker = critical_df['Maker'].value_counts().head(10)
        
        fig = px.bar(
            x=critical_by_maker.index,
            y=critical_by_maker.values,
            title="Critical Machinery Count by Maker (Top 10)",
            labels={'x': 'Maker', 'y': 'Critical Machinery Count'},
            color=critical_by_maker.values,
            color_continuous_scale='reds'
        )
        
        fig.update_layout(
            height=400,
            xaxis_tickangle=-45,
            showlegend=False
        )
        
        return fig
    
    def create_component_timeline_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create a chart showing component distribution"""
        # Since we don't have date information, create a component type vs machinery chart
        component_machinery = df.groupby(['Component Type', 'Machinery']).size().reset_index(name='Count')
        
        # Get top 10 component types
        top_components = df['Component Type'].value_counts().head(10).index
        filtered_data = component_machinery[component_machinery['Component Type'].isin(top_components)]
        
        fig = px.scatter(
            filtered_data,
            x='Component Type',
            y='Machinery',
            size='Count',
            color='Count',
            title="Component Type vs Machinery Distribution",
            labels={'Count': 'Equipment Count'},
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            height=500,
            xaxis_tickangle=-45
        )
        
        return fig
