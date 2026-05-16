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
    
    def export_machinery_list_excel(self, df: pd.DataFrame, duplicates: set = None) -> bytes:
        """Export machinery list with green/yellow formatting and pink highlighting for duplicates"""
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
        
        # Pink format for duplicate components
        component_duplicate_format = workbook.add_format({
            'bg_color': '#ffc0cb',
            'font_color': '#d81b60',
            'bold': True,
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
                    # Check if this is a duplicate component
                    if duplicates and 'Machinery' in df.columns and (row['Machinery'], row['Component']) in duplicates:
                        worksheet.write(row_idx, col_idx, value, component_duplicate_format)
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
    
    def export_critical_machinery_excel(self, df: pd.DataFrame, duplicates: set = None) -> bytes:
        """Export critical machinery with warning colors and pink highlighting for duplicates"""
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
        
        # Pink format for duplicate components
        component_duplicate_format = workbook.add_format({
            'bg_color': '#ffc0cb',
            'font_color': '#d81b60',
            'bold': True,
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
                    # Check if this is a duplicate component
                    if duplicates and 'Machinery' in df.columns and (row['Machinery'], row['Component']) in duplicates:
                        worksheet.write(row_idx, col_idx, value, component_duplicate_format)
                    else:
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
    
    def export_hierarchy_excel(self, df: pd.DataFrame, duplicates: set = None) -> bytes:
        """Export hierarchy with green theme and pink highlighting for duplicates"""
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
        
        # Pink format for duplicate components (priority over count > 1)
        component_duplicate_format = workbook.add_format({
            'bg_color': '#ffc0cb',
            'font_color': '#d81b60',
            'bold': True,
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
                    # Check if this is a duplicate component (priority)
                    if duplicates and 'Machinery' in df.columns and (row['Machinery'], row['Component']) in duplicates:
                        worksheet.write(row_idx, col_idx, value, component_duplicate_format)
                    # Check if count > 1 for conditional formatting
                    elif 'Count' in df.columns and row['Count'] > 1:
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
    
    def export_records_difference_excel(self, v1_name: str, v2_name: str,
                                         only_in_v1: pd.DataFrame, only_in_v2: pd.DataFrame,
                                         in_both_diff: pd.DataFrame,
                                         df1_comp: pd.DataFrame, df2_comp: pd.DataFrame,
                                         comp_col: str = 'Component Name') -> bytes:
        """Export Total Records Difference analysis to Excel with multiple sheets"""
        output = io.BytesIO()
        workbook = xlsxwriter.Workbook(output, {'in_memory': True})

        # --- Common formats ---
        header_fmt = workbook.add_format({'bold': True, 'bg_color': '#455a64', 'font_color': '#ffffff', 'align': 'center', 'border': 1})
        machinery_fmt = workbook.add_format({'bg_color': '#e8f5e8', 'font_color': '#2e7d32', 'bold': True, 'align': 'left'})
        v1_only_fmt = workbook.add_format({'bg_color': '#e3f2fd', 'font_color': '#1565c0', 'align': 'left'})
        v2_only_fmt = workbook.add_format({'bg_color': '#fce4ec', 'font_color': '#c62828', 'align': 'left'})
        count_diff_fmt = workbook.add_format({'bg_color': '#fff8e1', 'font_color': '#e65100', 'align': 'left'})
        pos_diff_fmt = workbook.add_format({'bg_color': '#e3f2fd', 'font_color': '#1565c0', 'bold': True, 'align': 'center'})
        neg_diff_fmt = workbook.add_format({'bg_color': '#fce4ec', 'font_color': '#c62828', 'bold': True, 'align': 'center'})
        normal_fmt = workbook.add_format({'align': 'left'})
        number_fmt = workbook.add_format({'num_format': '#,##0', 'align': 'center'})
        section_fmt = workbook.add_format({'bold': True, 'bg_color': '#90caf9', 'font_color': '#0d47a1', 'align': 'left', 'border': 1})

        # =====================================================================
        # Sheet 1: Summary — machinery-level diff table
        # =====================================================================
        ws_summary = workbook.add_worksheet('Summary')
        summary_headers = ['Machinery', v1_name, v2_name, 'Difference']
        for ci, h in enumerate(summary_headers):
            ws_summary.write(0, ci, h, header_fmt)

        for ri, (_, row) in enumerate(in_both_diff[['Machinery', v1_name, v2_name, 'Difference']].iterrows(), 1):
            ws_summary.write(ri, 0, row['Machinery'], machinery_fmt)
            ws_summary.write(ri, 1, int(row[v1_name]), number_fmt)
            ws_summary.write(ri, 2, int(row[v2_name]), number_fmt)
            diff_val = int(row['Difference'])
            ws_summary.write(ri, 3, diff_val, pos_diff_fmt if diff_val > 0 else neg_diff_fmt)

        ws_summary.set_column(0, 0, 45)
        ws_summary.set_column(1, 3, 18)

        # =====================================================================
        # Sheet 2: Unique Machinery — machinery only in one vessel
        # =====================================================================
        ws_unique = workbook.add_worksheet('Unique Machinery')
        ws_unique.write(0, 0, f'Only in {v1_name}', section_fmt)
        ws_unique.write(0, 1, 'Records', header_fmt)
        ri = 1
        for _, row in only_in_v1.iterrows():
            ws_unique.write(ri, 0, row['Machinery'], v1_only_fmt)
            ws_unique.write(ri, 1, int(row['Records']), number_fmt)
            ri += 1

        ri += 1
        ws_unique.write(ri, 0, f'Only in {v2_name}', section_fmt)
        ws_unique.write(ri, 1, 'Records', header_fmt)
        ri += 1
        for _, row in only_in_v2.iterrows():
            ws_unique.write(ri, 0, row['Machinery'], v2_only_fmt)
            ws_unique.write(ri, 1, int(row['Records']), number_fmt)
            ri += 1

        ws_unique.set_column(0, 0, 45)
        ws_unique.set_column(1, 1, 15)

        # =====================================================================
        # Sheet 3: Component Differences — full component breakdown
        # =====================================================================
        ws_comp = workbook.add_worksheet('Component Differences')
        comp_headers = ['Machinery', 'Component Name', 'Status', f'{v1_name} Count', f'{v2_name} Count', 'Difference']
        for ci, h in enumerate(comp_headers):
            ws_comp.write(0, ci, h, header_fmt)

        ri = 1
        if comp_col and comp_col in df1_comp.columns and comp_col in df2_comp.columns:
            for _, mach_row in in_both_diff.iterrows():
                machinery = mach_row['Machinery']
                comps1 = df1_comp[df1_comp['Machinery'] == machinery][comp_col].tolist()
                comps2 = df2_comp[df2_comp['Machinery'] == machinery][comp_col].tolist()
                set1, set2 = set(comps1), set(comps2)

                # Only in V1
                for comp in sorted(set1 - set2):
                    ws_comp.write(ri, 0, machinery, machinery_fmt)
                    ws_comp.write(ri, 1, comp, v1_only_fmt)
                    ws_comp.write(ri, 2, f'Only in {v1_name}', v1_only_fmt)
                    ws_comp.write(ri, 3, comps1.count(comp), number_fmt)
                    ws_comp.write(ri, 4, 0, number_fmt)
                    ws_comp.write(ri, 5, comps1.count(comp), pos_diff_fmt)
                    ri += 1

                # Only in V2
                for comp in sorted(set2 - set1):
                    ws_comp.write(ri, 0, machinery, machinery_fmt)
                    ws_comp.write(ri, 1, comp, v2_only_fmt)
                    ws_comp.write(ri, 2, f'Only in {v2_name}', v2_only_fmt)
                    ws_comp.write(ri, 3, 0, number_fmt)
                    ws_comp.write(ri, 4, comps2.count(comp), number_fmt)
                    ws_comp.write(ri, 5, -comps2.count(comp), neg_diff_fmt)
                    ri += 1

                # Count differences in shared components
                for comp in sorted(set1 & set2):
                    c1, c2 = comps1.count(comp), comps2.count(comp)
                    if c1 != c2:
                        ws_comp.write(ri, 0, machinery, machinery_fmt)
                        ws_comp.write(ri, 1, comp, count_diff_fmt)
                        ws_comp.write(ri, 2, 'Count Difference', count_diff_fmt)
                        ws_comp.write(ri, 3, c1, number_fmt)
                        ws_comp.write(ri, 4, c2, number_fmt)
                        diff_val = c1 - c2
                        ws_comp.write(ri, 5, diff_val, pos_diff_fmt if diff_val > 0 else neg_diff_fmt)
                        ri += 1

        ws_comp.set_column(0, 0, 45)
        ws_comp.set_column(1, 1, 40)
        ws_comp.set_column(2, 2, 22)
        ws_comp.set_column(3, 5, 16)

        workbook.close()
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
