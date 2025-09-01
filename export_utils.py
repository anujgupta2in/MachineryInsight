import pandas as pd
import io
from typing import Dict, Any
import xlsxwriter

class StyledExporter:
    """Utility class for exporting dataframes with conditional formatting preserved"""
    
    def __init__(self):
        self.colors = {
            'machinery_bg': '#e8f5e8',
            'machinery_text': '#2e7d32',
            'component_bg': '#fff8e1', 
            'component_text': '#f57f17',
            'critical_bg': '#ffecb3',
            'alt_row_1': '#f8f9fa',
            'alt_row_2': '#ffffff',
            'hierarchy_alt_1': '#f1f8e9',
            'hierarchy_alt_2': '#f8fdf6'
        }
    
    def export_machinery_list_excel(self, df: pd.DataFrame) -> bytes:
        """Export machinery list with green/yellow formatting"""
        output = io.BytesIO()
        workbook = xlsxwriter.Workbook(output, {'in_memory': True})
        worksheet = workbook.add_worksheet('Machinery List')
        
        # Define formats
        machinery_format = workbook.add_format({
            'bg_color': self.colors['machinery_bg'],
            'font_color': self.colors['machinery_text'],
            'bold': True,
            'align': 'left'
        })
        
        component_format = workbook.add_format({
            'bg_color': self.colors['component_bg'],
            'font_color': self.colors['component_text'],
            'align': 'left'
        })
        
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#d0d0d0',
            'align': 'center'
        })
        
        alt_row_1 = workbook.add_format({
            'bg_color': self.colors['alt_row_1'],
            'align': 'left'
        })
        
        alt_row_2 = workbook.add_format({
            'bg_color': self.colors['alt_row_2'],
            'align': 'left'
        })
        
        number_format = workbook.add_format({
            'num_format': '#,##0',
            'align': 'right'
        })
        
        # Write headers
        for col, header in enumerate(df.columns):
            worksheet.write(0, col, header, header_format)
        
        # Write data with formatting
        for row_idx, (_, row) in enumerate(df.iterrows(), 1):
            for col_idx, (col_name, value) in enumerate(row.items()):
                if col_name == 'Machinery':
                    worksheet.write(row_idx, col_idx, value, machinery_format)
                elif col_name == 'Component':
                    worksheet.write(row_idx, col_idx, value, component_format)
                elif col_name == 'Count':
                    worksheet.write(row_idx, col_idx, value, number_format)
                else:
                    alt_format = alt_row_1 if row_idx % 2 == 0 else alt_row_2
                    worksheet.write(row_idx, col_idx, value, alt_format)
        
        # Auto-adjust column widths
        for col_idx, col_name in enumerate(df.columns):
            max_length = max(
                df[col_name].astype(str).str.len().max(),
                len(col_name)
            )
            worksheet.set_column(col_idx, col_idx, min(max_length + 2, 50))
        
        workbook.close()
        output.seek(0)
        return output.getvalue()
    
    def export_critical_machinery_excel(self, df: pd.DataFrame) -> bytes:
        """Export critical machinery with warning colors"""
        output = io.BytesIO()
        workbook = xlsxwriter.Workbook(output, {'in_memory': True})
        worksheet = workbook.add_worksheet('Critical Machinery')
        
        # Define formats
        machinery_format = workbook.add_format({
            'bg_color': self.colors['machinery_bg'],
            'font_color': self.colors['machinery_text'],
            'bold': True,
            'align': 'left'
        })
        
        component_format = workbook.add_format({
            'bg_color': self.colors['component_bg'],
            'font_color': self.colors['component_text'],
            'align': 'left'
        })
        
        critical_format_1 = workbook.add_format({
            'bg_color': self.colors['critical_bg'],
            'align': 'left'
        })
        
        critical_format_2 = workbook.add_format({
            'bg_color': self.colors['component_bg'],
            'align': 'left'
        })
        
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#ffcc00',
            'align': 'center'
        })
        
        # Write headers
        for col, header in enumerate(df.columns):
            worksheet.write(0, col, header, header_format)
        
        # Write data with formatting
        for row_idx, (_, row) in enumerate(df.iterrows(), 1):
            for col_idx, (col_name, value) in enumerate(row.items()):
                if col_name == 'Machinery':
                    worksheet.write(row_idx, col_idx, value, machinery_format)
                elif col_name == 'Component':
                    worksheet.write(row_idx, col_idx, value, component_format)
                else:
                    critical_format = critical_format_1 if row_idx % 2 == 0 else critical_format_2
                    worksheet.write(row_idx, col_idx, value, critical_format)
        
        # Auto-adjust column widths
        for col_idx, col_name in enumerate(df.columns):
            max_length = max(
                df[col_name].astype(str).str.len().max(),
                len(col_name)
            )
            worksheet.set_column(col_idx, col_idx, min(max_length + 2, 50))
        
        workbook.close()
        output.seek(0)
        return output.getvalue()
    
    def export_hierarchy_excel(self, df: pd.DataFrame) -> bytes:
        """Export hierarchy with green theme"""
        output = io.BytesIO()
        workbook = xlsxwriter.Workbook(output, {'in_memory': True})
        worksheet = workbook.add_worksheet('Machinery Hierarchy')
        
        # Define formats
        machinery_format = workbook.add_format({
            'bg_color': self.colors['machinery_bg'],
            'font_color': self.colors['machinery_text'],
            'bold': True,
            'align': 'left'
        })
        
        component_format = workbook.add_format({
            'bg_color': self.colors['component_bg'],
            'font_color': self.colors['component_text'],
            'align': 'left'
        })
        
        # Format for components with count > 1 (highlighted)
        component_highlighted_format = workbook.add_format({
            'bg_color': '#ffcc80',
            'font_color': '#e65100',
            'bold': True,
            'align': 'left'
        })
        
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#81c784',
            'align': 'center'
        })
        
        alt_row_1 = workbook.add_format({
            'bg_color': self.colors['hierarchy_alt_1'],
            'align': 'left'
        })
        
        alt_row_2 = workbook.add_format({
            'bg_color': self.colors['hierarchy_alt_2'],
            'align': 'left'
        })
        
        number_format = workbook.add_format({
            'num_format': '#,##0',
            'align': 'right',
            'bg_color': self.colors['hierarchy_alt_1']
        })
        
        # Write headers
        for col, header in enumerate(df.columns):
            worksheet.write(0, col, header, header_format)
        
        # Write data with formatting
        for row_idx, (_, row) in enumerate(df.iterrows(), 1):
            for col_idx, (col_name, value) in enumerate(row.items()):
                if col_name == 'Machinery':
                    worksheet.write(row_idx, col_idx, value, machinery_format)
                elif col_name == 'Component':
                    # Check if count > 1 for conditional formatting
                    if 'Count' in df.columns and row['Count'] > 1:
                        worksheet.write(row_idx, col_idx, value, component_highlighted_format)
                    else:
                        worksheet.write(row_idx, col_idx, value, component_format)
                elif col_name == 'Count':
                    worksheet.write(row_idx, col_idx, value, number_format)
                else:
                    alt_format = alt_row_1 if row_idx % 2 == 0 else alt_row_2
                    worksheet.write(row_idx, col_idx, value, alt_format)
        
        # Auto-adjust column widths
        for col_idx, col_name in enumerate(df.columns):
            max_length = max(
                df[col_name].astype(str).str.len().max(),
                len(col_name)
            )
            worksheet.set_column(col_idx, col_idx, min(max_length + 2, 50))
        
        workbook.close()
        output.seek(0)
        return output.getvalue()
    
    def create_summary_excel(self, machinery_df: pd.DataFrame, critical_df: pd.DataFrame, 
                           hierarchy_df: pd.DataFrame, raw_df: pd.DataFrame, vessel_name: str = "Ship") -> bytes:
        """Create a comprehensive Excel file with all tables in separate sheets"""
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Write each dataframe to a separate sheet
            machinery_df.to_excel(writer, sheet_name='Machinery List', index=False)
            critical_df.to_excel(writer, sheet_name='Critical Machinery', index=False)
            hierarchy_df.to_excel(writer, sheet_name='Hierarchy Details', index=False)
            raw_df.to_excel(writer, sheet_name='Raw Data', index=False)
            
            # Get the workbook and add formatting
            workbook = writer.book
            
            # Format each sheet
            self._format_worksheet(writer.sheets['Machinery List'], machinery_df, workbook, 'machinery')
            self._format_worksheet(writer.sheets['Critical Machinery'], critical_df, workbook, 'critical')
            self._format_worksheet(writer.sheets['Hierarchy Details'], hierarchy_df, workbook, 'hierarchy')
            self._format_worksheet(writer.sheets['Raw Data'], raw_df, workbook, 'raw')
        
        output.seek(0)
        return output.getvalue()
    
    def _format_worksheet(self, worksheet, df: pd.DataFrame, workbook, sheet_type: str):
        """Apply formatting to a worksheet based on type"""
        # Define common formats
        machinery_format = workbook.add_format({
            'bg_color': self.colors['machinery_bg'],
            'font_color': self.colors['machinery_text'],
            'bold': True
        })
        
        component_format = workbook.add_format({
            'bg_color': self.colors['component_bg'],
            'font_color': self.colors['component_text']
        })
        
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#d0d0d0' if sheet_type != 'critical' else '#ffcc00'
        })
        
        # Apply header formatting
        for col_idx, col_name in enumerate(df.columns):
            worksheet.write(0, col_idx, col_name, header_format)
        
        # Apply column-specific formatting
        for col_idx, col_name in enumerate(df.columns):
            if col_name == 'Machinery':
                for row_idx in range(1, len(df) + 1):
                    worksheet.write(row_idx, col_idx, df.iloc[row_idx-1, col_idx], machinery_format)
            elif 'Component' in col_name:
                for row_idx in range(1, len(df) + 1):
                    worksheet.write(row_idx, col_idx, df.iloc[row_idx-1, col_idx], component_format)
        
        # Auto-adjust column widths
        for col_idx, col_name in enumerate(df.columns):
            max_length = max(
                df[col_name].astype(str).str.len().max() if len(df) > 0 else 0,
                len(col_name)
            )
            worksheet.set_column(col_idx, col_idx, min(max_length + 2, 50))
