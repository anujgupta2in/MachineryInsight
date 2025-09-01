import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from data_analyzer import MachineryDataAnalyzer
from visualization import MachineryVisualizer
from export_utils import StyledExporter

# Page configuration
st.set_page_config(
    page_title="Ship Machinery Analysis",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for multiple vessels
if 'vessels_data' not in st.session_state:
    st.session_state.vessels_data = {}
if 'selected_vessel' not in st.session_state:
    st.session_state.selected_vessel = None
if 'comparison_mode' not in st.session_state:
    st.session_state.comparison_mode = False

def load_vessel_data(uploaded_file):
    """Load and process machinery data for a single vessel"""
    try:
        # Read CSV file
        df = pd.read_csv(uploaded_file)
        
        # Extract vessel name from filename
        vessel_name = extract_vessel_name(uploaded_file.name)
        
        # Clean column names and handle the first unnamed column
        df.columns = df.columns.str.strip()
        
        # Rename the first column to 'Critical_Machinery' if it's unnamed
        if df.columns[0] in ['', 'Unnamed: 0']:
            df.rename(columns={df.columns[0]: 'Critical_Machinery'}, inplace=True)
        
        # Clean data
        df = df.fillna('')
        df = df.replace('""', '')
        
        # Create analyzer and visualizer
        analyzer = MachineryDataAnalyzer(df)
        visualizer = MachineryVisualizer(df)
        
        # Store vessel data
        vessel_data = {
            'df': df,
            'analyzer': analyzer,
            'visualizer': visualizer,
            'vessel_name': vessel_name,
            'filename': uploaded_file.name,
            'total_records': len(df)
        }
        
        return True, vessel_name, vessel_data
    except Exception as e:
        st.error(f"Error loading data for {uploaded_file.name}: {str(e)}")
        return False, None, None

def load_multiple_vessels(uploaded_files):
    """Load and process multiple vessel files"""
    loaded_vessels = {}
    failed_files = []
    
    for uploaded_file in uploaded_files:
        success, vessel_name, vessel_data = load_vessel_data(uploaded_file)
        if success:
            loaded_vessels[vessel_name] = vessel_data
            st.success(f"✅ Loaded {vessel_name} - {vessel_data['total_records']} records")
        else:
            failed_files.append(uploaded_file.name)
    
    return loaded_vessels, failed_files

def extract_vessel_name(filename):
    """Extract vessel name from filename"""
    try:
        # Remove file extension
        name_without_ext = filename.rsplit('.', 1)[0]
        
        # Split by spaces and extract vessel name parts
        parts = name_without_ext.split()
        
        # Look for pattern like "NYK Isabel Machinery" and extract "NYK Isabel"
        if len(parts) >= 2:
            # Check if the pattern contains "Machinery" keyword
            if 'Machinery' in parts:
                machinery_index = parts.index('Machinery')
                if machinery_index >= 2:
                    return ' '.join(parts[:machinery_index])
            
            # Fallback: take first two parts if they look like vessel name
            if len(parts) >= 2:
                return ' '.join(parts[:2])
        
        # Final fallback
        return parts[0] if parts else "Vessel"
        
    except Exception:
        return "Vessel"

def display_vessel_comparison():
    """Display comparison view for multiple vessels"""
    st.markdown("### 📊 Multi-Vessel Comparison Dashboard")
    
    # Comparison metrics
    st.header("🔢 Vessel Comparison Metrics")
    
    comparison_data = []
    for vessel_name, data in st.session_state.vessels_data.items():
        analyzer = data['analyzer']
        comparison_data.append({
            'Vessel': vessel_name,
            'Total Records': analyzer.get_total_machinery_count(),
            'Unique Machinery': analyzer.get_unique_machinery_count(),
            'Critical Machinery': analyzer.get_critical_machinery_count(),
            'Total Components': analyzer.get_total_components_count(),
            'Unique Makers': analyzer.get_unique_makers_count(),
            'Unique Models': analyzer.get_unique_models_count()
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display comparison table
    st.dataframe(comparison_df, use_container_width=True)
    
    # Detailed machinery comparison analysis
    st.header("🔍 Detailed Machinery Comparison Analysis")
    
    # Get all vessel data for detailed comparison
    vessel_names = list(st.session_state.vessels_data.keys())
    
    if len(vessel_names) >= 2:
        # Create tabs for different comparison views
        comp_tab1, comp_tab2, comp_tab3, comp_tab4 = st.tabs([
            "⚙️ Machinery Differences",
            "🏭 Maker & Model Differences", 
            "❌ Missing Machinery",
            "📊 Export Analysis"
        ])
        
        with comp_tab1:
            st.subheader("Machinery Present in Each Vessel")
            
            # Get machinery sets for each vessel with trimmed data
            vessel_machinery = {}
            for vessel_name, data in st.session_state.vessels_data.items():
                df = data['df'].copy()
                # Trim machinery column before comparison
                df['Machinery'] = df['Machinery'].astype(str).str.strip()
                machinery_set = set(df['Machinery'].dropna().unique())
                # Remove empty strings
                machinery_set = {m for m in machinery_set if m and m.strip()}
                vessel_machinery[vessel_name] = machinery_set
            
            # Display machinery for each vessel
            for vessel_name, machinery_set in vessel_machinery.items():
                with st.expander(f"🚢 {vessel_name} - {len(machinery_set)} unique machinery types"):
                    machinery_list = sorted(list(machinery_set))
                    for i in range(0, len(machinery_list), 3):
                        cols = st.columns(3)
                        for j, col in enumerate(cols):
                            if i + j < len(machinery_list):
                                col.write(f"• {machinery_list[i + j]}")
            
            # Common machinery across all vessels
            if len(vessel_machinery) > 1:
                all_machinery = list(vessel_machinery.values())
                common_machinery = set.intersection(*all_machinery)
                
                st.subheader(f"🤝 Common Machinery Across All Vessels ({len(common_machinery)})")
                if common_machinery:
                    common_list = sorted(list(common_machinery))
                    for i in range(0, len(common_list), 4):
                        cols = st.columns(4)
                        for j, col in enumerate(cols):
                            if i + j < len(common_list):
                                col.write(f"✓ {common_list[i + j]}")
                else:
                    st.info("No machinery is common across all vessels")
        
        with comp_tab2:
            st.subheader("Component Maker & Model Differences")
            
            # Helper function to normalize strings for comparison
            def normalize_for_comparison(text):
                if pd.isna(text) or text == '':
                    return ''
                return str(text).strip().replace(' ', '').upper()
            
            # Analyze component-level differences only
            component_comparison = {}
            
            for vessel_name, data in st.session_state.vessels_data.items():
                df = data['df'].copy()
                
                # Group by machinery and component for component-level comparison
                if 'Component Name' in df.columns:
                    # Get detailed maker-model combinations for each component
                    component_details = df[df['Component Name'].notna() & df['Machinery'].notna()].copy()
                    
                    # Group by machinery, component, maker, and model to get unique combinations
                    maker_model_combinations = component_details.groupby(['Machinery', 'Component Name', 'Maker', 'Model']).size().reset_index(name='count')
                    
                    # Group by machinery and component to collect all maker-model pairs
                    for (machinery, component), group in maker_model_combinations.groupby(['Machinery', 'Component Name']):
                        machinery = str(machinery).strip() if pd.notna(machinery) else ''
                        component = str(component).strip() if pd.notna(component) else ''
                        key = f"{machinery} - {component}"
                        
                        if machinery and component and key not in component_comparison:
                            component_comparison[key] = {}
                        if machinery and component:
                            # Create maker-model pairs
                            maker_model_pairs = []
                            normalized_pairs = []
                            
                            for _, row in group.iterrows():
                                maker = str(row['Maker']).strip() if pd.notna(row['Maker']) else ''
                                model = str(row['Model']).strip() if pd.notna(row['Model']) else ''
                                if maker or model:
                                    maker_model_pairs.append({'maker': maker, 'model': model})
                                    normalized_pairs.append({
                                        'maker': normalize_for_comparison(maker),
                                        'model': normalize_for_comparison(model)
                                    })
                            
                            component_comparison[key][vessel_name] = {
                                'machinery': machinery,
                                'component': component,
                                'maker_model_pairs': maker_model_pairs,
                                'normalized_pairs': normalized_pairs
                            }
            
            # Categorize component differences
            component_maker_differences = []
            component_model_differences = []
            component_both_differences = []
            
            for key, vessel_data in component_comparison.items():
                if len(vessel_data) > 1:  # Component exists in multiple vessels
                    # Compare normalized maker-model pairs
                    all_normalized_pairs = []
                    for info in vessel_data.values():
                        vessel_pairs = set()
                        for pair in info['normalized_pairs']:
                            vessel_pairs.add(f"{pair['maker']}|{pair['model']}")
                        all_normalized_pairs.append(vessel_pairs)
                    
                    # Check if there are different maker-model combinations
                    all_pairs_union = set().union(*all_normalized_pairs)
                    has_differences = len(all_pairs_union) > len(all_normalized_pairs[0]) or not all(pairs == all_normalized_pairs[0] for pairs in all_normalized_pairs)
                    
                    if has_differences:
                        # Determine type of difference
                        all_makers = set()
                        all_models = set()
                        for info in vessel_data.values():
                            for pair in info['normalized_pairs']:
                                all_makers.add(pair['maker'])
                                all_models.add(pair['model'])
                        
                        makers_by_vessel = []
                        models_by_vessel = []
                        for info in vessel_data.values():
                            vessel_makers = set(pair['maker'] for pair in info['normalized_pairs'])
                            vessel_models = set(pair['model'] for pair in info['normalized_pairs'])
                            makers_by_vessel.append(vessel_makers)
                            models_by_vessel.append(vessel_models)
                        
                        has_maker_diff = len(set().union(*makers_by_vessel)) > 1
                        has_model_diff = len(set().union(*models_by_vessel)) > 1
                        
                        if has_maker_diff and has_model_diff:
                            component_both_differences.append((key, vessel_data))
                        elif has_maker_diff:
                            component_maker_differences.append((key, vessel_data))
                        elif has_model_diff:
                            component_model_differences.append((key, vessel_data))
            
            # Display component differences
            if component_both_differences:
                st.subheader("🔧 Components with Different Makers AND Models")
                for key, vessel_data in component_both_differences:
                    with st.expander(f"🔩 {key}"):
                        cols = st.columns(len(vessel_data))
                        for i, (vessel_name, info) in enumerate(vessel_data.items()):
                            with cols[i]:
                                st.write(f"**{vessel_name}**")
                                st.write(f"**Machinery:** {info['machinery']}")
                                st.write(f"**Component:** {info['component']}")
                                st.write("**Maker-Model Combinations:**")
                                for pair in info['maker_model_pairs']:
                                    maker = pair['maker'] if pair['maker'] else 'N/A'
                                    model = pair['model'] if pair['model'] else 'N/A'
                                    st.write(f"• **{maker}** - {model}")
            
            if component_maker_differences:
                st.subheader("🏭 Components with Different Makers Only")
                for key, vessel_data in component_maker_differences:
                    with st.expander(f"🔩 {key}"):
                        cols = st.columns(len(vessel_data))
                        for i, (vessel_name, info) in enumerate(vessel_data.items()):
                            with cols[i]:
                                st.write(f"**{vessel_name}**")
                                st.write(f"**Machinery:** {info['machinery']}")
                                st.write(f"**Component:** {info['component']}")
                                st.write("**Maker-Model Combinations:**")
                                for pair in info['maker_model_pairs']:
                                    maker = pair['maker'] if pair['maker'] else 'N/A'
                                    model = pair['model'] if pair['model'] else 'N/A'
                                    st.write(f"• **{maker}** - {model}")
            
            if component_model_differences:
                st.subheader("📋 Components with Different Models Only")
                for key, vessel_data in component_model_differences:
                    with st.expander(f"🔩 {key}"):
                        cols = st.columns(len(vessel_data))
                        for i, (vessel_name, info) in enumerate(vessel_data.items()):
                            with cols[i]:
                                st.write(f"**{vessel_name}**")
                                st.write(f"**Machinery:** {info['machinery']}")
                                st.write(f"**Component:** {info['component']}")
                                st.write("**Maker-Model Combinations:**")
                                for pair in info['maker_model_pairs']:
                                    maker = pair['maker'] if pair['maker'] else 'N/A'
                                    model = pair['model'] if pair['model'] else 'N/A'
                                    st.write(f"• **{maker}** - {model}")
            
            if not (component_both_differences or component_maker_differences or component_model_differences):
                st.info("No maker or model differences found for components across vessels")
        
        with comp_tab3:
            st.subheader("Missing Machinery Analysis")
            
            # Pairwise comparison of vessels
            vessel_list = list(st.session_state.vessels_data.keys())
            
            for i in range(len(vessel_list)):
                for j in range(i + 1, len(vessel_list)):
                    vessel1 = vessel_list[i]
                    vessel2 = vessel_list[j]
                    
                    df1 = st.session_state.vessels_data[vessel1]['df'].copy()
                    df2 = st.session_state.vessels_data[vessel2]['df'].copy()
                    
                    # Trim machinery columns before comparison
                    df1['Machinery'] = df1['Machinery'].astype(str).str.strip()
                    df2['Machinery'] = df2['Machinery'].astype(str).str.strip()
                    
                    machinery1 = set(df1['Machinery'].dropna().unique())
                    machinery2 = set(df2['Machinery'].dropna().unique())
                    
                    # Remove empty strings
                    machinery1 = {m for m in machinery1 if m and m.strip()}
                    machinery2 = {m for m in machinery2 if m and m.strip()}
                    
                    missing_in_vessel2 = machinery1 - machinery2
                    missing_in_vessel1 = machinery2 - machinery1
                    
                    st.subheader(f"🔄 {vessel1} vs {vessel2}")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Machinery in {vessel1} but missing in {vessel2}** ({len(missing_in_vessel2)})")
                        if missing_in_vessel2:
                            for machinery in sorted(missing_in_vessel2):
                                st.write(f"❌ {machinery}")
                        else:
                            st.write("✅ No missing machinery")
                    
                    with col2:
                        st.write(f"**Machinery in {vessel2} but missing in {vessel1}** ({len(missing_in_vessel1)})")
                        if missing_in_vessel1:
                            for machinery in sorted(missing_in_vessel1):
                                st.write(f"❌ {machinery}")
                        else:
                            st.write("✅ No missing machinery")
        
        with comp_tab4:
            st.subheader("Export Comparison Analysis")
            
            # Filters for comparison analysis
            st.subheader("🔍 Filters")
            filter_col1, filter_col2, filter_col3 = st.columns(3)
            
            # Create comprehensive comparison data with trimmed columns
            all_machinery = set()
            all_makers = set()
            for data in st.session_state.vessels_data.values():
                df = data['df'].copy()
                df['Machinery'] = df['Machinery'].astype(str).str.strip()
                df['Maker'] = df['Maker'].astype(str).str.strip()
                machinery_set = set(df['Machinery'].dropna().unique())
                machinery_set = {m for m in machinery_set if m and m.strip()}
                all_machinery.update(machinery_set)
                maker_set = set(df['Maker'].dropna().unique())
                maker_set = {m for m in maker_set if m and m.strip()}
                all_makers.update(maker_set)
            
            with filter_col1:
                # Machinery filter
                selected_machinery_filter = st.selectbox(
                    "Filter by Machinery", 
                    ['All'] + sorted(list(all_machinery)),
                    key="export_machinery_filter"
                )
            
            with filter_col2:
                # Maker filter
                selected_maker_filter = st.selectbox(
                    "Filter by Maker", 
                    ['All'] + sorted(list(all_makers)),
                    key="export_maker_filter"
                )
            
            with filter_col3:
                # Presence filter
                presence_filter = st.selectbox(
                    "Filter by Presence", 
                    ['All', 'Present in All Vessels', 'Missing in Some Vessels', 'Present in Only One Vessel'],
                    key="export_presence_filter"
                )
            
            # Create comparison matrix
            comparison_matrix = []
            machinery_to_process = all_machinery if selected_machinery_filter == 'All' else {selected_machinery_filter}
            
            for machinery in sorted(machinery_to_process):
                row = {'Machinery': machinery}
                vessel_present_count = 0
                should_include_row = True
                
                # Check maker filter first
                if selected_maker_filter != 'All':
                    machinery_has_maker = False
                    for vessel_name, data in st.session_state.vessels_data.items():
                        df = data['df'].copy()
                        df['Machinery'] = df['Machinery'].astype(str).str.strip()
                        df['Maker'] = df['Maker'].astype(str).str.strip()
                        machinery_data = df[df['Machinery'] == machinery]
                        if len(machinery_data) > 0:
                            makers = [m.strip() for m in machinery_data['Maker'].dropna().unique() if m and m.strip()]
                            if selected_maker_filter in makers:
                                machinery_has_maker = True
                                break
                    if not machinery_has_maker:
                        should_include_row = False
                
                if should_include_row:
                    for vessel_name, data in st.session_state.vessels_data.items():
                        df = data['df'].copy()
                        
                        # Trim all relevant columns
                        string_columns = ['Machinery', 'Maker', 'Model']
                        for col in string_columns:
                            if col in df.columns:
                                df[col] = df[col].astype(str).str.strip()
                        
                        machinery_data = df[df['Machinery'] == machinery]
                        
                        if len(machinery_data) > 0:
                            makers = [m for m in machinery_data['Maker'].dropna().unique() if m and m.strip()]
                            models = [m for m in machinery_data['Model'].dropna().unique() if m and m.strip()]
                            
                            # Apply maker filter to the display data
                            if selected_maker_filter != 'All':
                                filtered_data = machinery_data[machinery_data['Maker'] == selected_maker_filter]
                                if len(filtered_data) > 0:
                                    makers = [m for m in filtered_data['Maker'].dropna().unique() if m and m.strip()]
                                    models = [m for m in filtered_data['Model'].dropna().unique() if m and m.strip()]
                                else:
                                    makers = []
                                    models = []
                            
                            if makers or models:
                                row[f"{vessel_name}_Present"] = "Yes"
                                row[f"{vessel_name}_Makers"] = ', '.join(makers)
                                row[f"{vessel_name}_Models"] = ', '.join(models)
                                row[f"{vessel_name}_Count"] = len(machinery_data) if selected_maker_filter == 'All' else len(filtered_data)
                                vessel_present_count += 1
                            else:
                                row[f"{vessel_name}_Present"] = "No"
                                row[f"{vessel_name}_Makers"] = ""
                                row[f"{vessel_name}_Models"] = ""
                                row[f"{vessel_name}_Count"] = 0
                        else:
                            row[f"{vessel_name}_Present"] = "No"
                            row[f"{vessel_name}_Makers"] = ""
                            row[f"{vessel_name}_Models"] = ""
                            row[f"{vessel_name}_Count"] = 0
                    
                    # Apply presence filter
                    total_vessels = len(st.session_state.vessels_data)
                    include_in_results = True
                    
                    if presence_filter == 'Present in All Vessels' and vessel_present_count != total_vessels:
                        include_in_results = False
                    elif presence_filter == 'Missing in Some Vessels' and vessel_present_count == total_vessels:
                        include_in_results = False
                    elif presence_filter == 'Present in Only One Vessel' and vessel_present_count != 1:
                        include_in_results = False
                    
                    if include_in_results:
                        comparison_matrix.append(row)
            
            comparison_df = pd.DataFrame(comparison_matrix)
            
            if len(comparison_df) > 0:
                st.write("**Filtered Machinery Comparison Matrix**")
                st.write(f"Showing {len(comparison_df)} machinery entries")
                st.dataframe(comparison_df, use_container_width=True, height=400)
            else:
                st.info("No machinery matches the selected filters")
            
            # Export options
            col1, col2 = st.columns(2)
            
            with col1:
                csv_data = comparison_df.to_csv(index=False)
                st.download_button(
                    label="📄 Download Detailed Comparison CSV",
                    data=csv_data,
                    file_name="detailed_vessel_machinery_comparison.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Summary metrics CSV
                summary_csv = comparison_df.to_csv(index=False)
                st.download_button(
                    label="📊 Download Summary Metrics CSV",
                    data=summary_csv,
                    file_name="vessel_comparison_metrics.csv",
                    mime="text/csv"
                )
    
    else:
        st.info("Upload at least 2 vessel files to see detailed comparisons")

def display_single_vessel_analysis(df, analyzer, visualizer, vessel_name):
    """Display analysis for a single vessel"""
    
    # Display key metrics
    st.header("📊 Key Metrics")
    
    # Get summary statistics
    total_machinery_entries = analyzer.get_total_machinery_count()
    unique_machinery = analyzer.get_unique_machinery_count()
    critical_machinery = analyzer.get_critical_machinery_count()
    total_components = analyzer.get_total_components_count()
    unique_makers = analyzer.get_unique_makers_count()
    unique_models = analyzer.get_unique_models_count()
    
    # Display metrics in columns
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("Total Machinery & Components", total_machinery_entries)
    with col2:
        st.metric("Unique Machinery", unique_machinery)
    with col3:
        st.metric("Critical Machinery", critical_machinery)
    with col4:
        st.metric("Total Components", total_components)
    with col5:
        st.metric("Unique Makers", unique_makers)
    with col6:
        st.metric("Unique Models", unique_models)
    
    # Sidebar filters
    with st.sidebar:
        st.header("🔍 Filters")
        
        # Maker filter
        makers = ['All'] + sorted(df['Maker'].unique().tolist())
        selected_maker = st.selectbox("Select Maker", makers)
        
        # Model filter
        if selected_maker != 'All':
            models = ['All'] + sorted(df[df['Maker'] == selected_maker]['Model'].unique().tolist())
        else:
            models = ['All'] + sorted(df['Model'].unique().tolist())
        selected_model = st.selectbox("Select Model", models)
        
        # Machinery filter
        machineries = ['All'] + sorted(df['Machinery'].unique().tolist())
        selected_machinery = st.selectbox("Select Machinery", machineries)
        
        # Component filter
        if selected_machinery != 'All':
            components = ['All'] + sorted(df[df['Machinery'] == selected_machinery]['Component Name'].unique().tolist())
        else:
            components = ['All'] + sorted(df['Component Name'].unique().tolist())
        selected_component = st.selectbox("Select Component", components)
        
        # Critical machinery filter
        critical_filter = st.selectbox("Critical Machinery", ['All', 'Critical Only', 'Non-Critical Only'])
    
    # Apply filters
    filtered_df = analyzer.apply_filters(
        maker=selected_maker if selected_maker != 'All' else None,
        model=selected_model if selected_model != 'All' else None,
        machinery=selected_machinery if selected_machinery != 'All' else None,
        component=selected_component if selected_component != 'All' else None,
        critical_filter=critical_filter
    )
    
    # Display filtered metrics
    if len(filtered_df) != len(df):
        st.info(f"Showing {len(filtered_df)} out of {len(df)} records after applying filters")
    
    # Tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs([
        "⚙️ Unique Machinery",
        "⚠️ Critical Machinery", 
        "🏗️ Machinery Hierarchy",
        "📋 Raw Data"
    ])
    
    with tab1:
        st.header("Unique Machinery with Maker & Model Details")
        
        # Get unique machinery details using filtered data
        machinery_details = analyzer.get_unique_machinery_with_details(filtered_df)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Machinery List with Maker & Model")
            
            if len(machinery_details) > 0:
                # Apply conditional formatting with consistent colors
                def highlight_machinery_component(row):
                    styles = []
                    for idx, col in enumerate(machinery_details.columns):
                        if col == 'Machinery':
                            styles.append('background-color: #e8f5e8; color: #2e7d32; font-weight: bold')
                        elif col == 'Component':
                            styles.append('background-color: #fff8e1; color: #f57f17')
                        else:
                            styles.append('background-color: #f8f9fa' if row.name % 2 == 0 else 'background-color: #ffffff')
                    return styles
                
                machinery_styled = machinery_details.style.format({
                    'Count': '{:,.0f}'
                }).apply(highlight_machinery_component, axis=1).set_properties(**{
                    'text-align': 'left',
                    'white-space': 'nowrap'
                })
                
                st.dataframe(machinery_styled, use_container_width=True, height=600)
            else:
                st.info("No machinery found with the current filter selection")
            
            # Enhanced export options
            if len(machinery_details) > 0:
                st.subheader("Export Options")
                exporter = StyledExporter()
                
                col_export1, col_export2 = st.columns(2)
                
                with col_export1:
                    # CSV export
                    csv = machinery_details.to_csv(index=False)
                    st.download_button(
                        label="📄 Download CSV",
                        data=csv,
                        file_name=f"{vessel_name}_machinery_list.csv",
                        mime="text/csv"
                    )
                
                with col_export2:
                    # Excel export with formatting
                    excel_data = exporter.export_machinery_list_excel(machinery_details)
                    st.download_button(
                        label="📊 Download Styled Excel",
                        data=excel_data,
                        file_name=f"{vessel_name}_machinery_list.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        
        with col2:
            st.subheader("Summary")
            if len(machinery_details) > 0:
                st.metric("Total Unique Machinery", len(machinery_details['Machinery'].unique()))
                st.metric("Total Maker-Model Combinations", len(machinery_details))
                
                # Top machinery by component count
                top_machinery = machinery_details.groupby('Machinery')['Count'].sum().sort_values(ascending=False).head(10)
                st.subheader("Top 10 Machinery by Component Count")
                for machinery, count in top_machinery.items():
                    st.write(f"**{machinery}**: {count} components")
            else:
                st.info("No machinery found with current filters")

    with tab2:
        st.header("Critical Machinery Analysis")
        
        # Critical machinery details
        st.subheader("Critical Machinery Details")
        critical_details = analyzer.get_critical_machinery_details(filtered_df)
        
        # Apply conditional formatting for critical machinery with consistent colors
        def highlight_critical_machinery_component(row):
            styles = []
            for idx, col in enumerate(critical_details.columns):
                if col == 'Machinery':
                    styles.append('background-color: #e8f5e8; color: #2e7d32; font-weight: bold')
                elif col == 'Component':
                    styles.append('background-color: #fff8e1; color: #f57f17')
                else:
                    # Critical items get orange/red background for other columns
                    styles.append('background-color: #ffecb3' if row.name % 2 == 0 else 'background-color: #fff8e1')
            return styles
        
        critical_styled = critical_details.style.apply(highlight_critical_machinery_component, axis=1).set_properties(**{
            'text-align': 'left',
            'white-space': 'nowrap'
        })
        
        st.dataframe(critical_styled, use_container_width=True, height=600)
        
        # Enhanced export options for critical machinery
        st.subheader("Export Critical Machinery")
        exporter = StyledExporter()
        
        col_crit1, col_crit2 = st.columns(2)
        
        with col_crit1:
            # CSV export
            csv = critical_details.to_csv(index=False)
            st.download_button(
                label="📄 Download CSV",
                data=csv,
                file_name=f"{vessel_name}_critical_machinery.csv",
                mime="text/csv"
            )
        
        with col_crit2:
            # Excel export with formatting
            excel_data = exporter.export_critical_machinery_excel(critical_details)
            st.download_button(
                label="⚠️ Download Styled Excel",
                data=excel_data,
                file_name=f"{vessel_name}_critical_machinery.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with tab3:
        st.header("Machinery Hierarchy Visualization")
        
        # Machinery hierarchy tree
        hierarchy_fig = visualizer.create_hierarchy_treemap(filtered_df)
        st.plotly_chart(hierarchy_fig, use_container_width=True)
        
        # Detailed hierarchy table
        st.subheader("Detailed Hierarchy")
        detailed_hierarchy = analyzer.get_detailed_hierarchy(filtered_df)
        
        # Apply conditional formatting for hierarchy with consistent colors
        def highlight_hierarchy_machinery_component(row):
            styles = []
            for idx, col in enumerate(detailed_hierarchy.columns):
                if col == 'Machinery':
                    styles.append('background-color: #e8f5e8; color: #2e7d32; font-weight: bold')
                elif col == 'Component':
                    # Highlight components with count > 1
                    if 'Count' in detailed_hierarchy.columns and row['Count'] > 1:
                        styles.append('background-color: #ffcc80; color: #e65100; font-weight: bold')
                    else:
                        styles.append('background-color: #fff8e1; color: #f57f17')
                else:
                    styles.append('background-color: #f1f8e9' if row.name % 2 == 0 else 'background-color: #f8fdf6')
            return styles
        
        hierarchy_styled = detailed_hierarchy.style.format({
            'Count': '{:,.0f}'
        }).apply(highlight_hierarchy_machinery_component, axis=1).set_properties(**{
            'text-align': 'left',
            'white-space': 'nowrap'
        })
        
        st.dataframe(hierarchy_styled, use_container_width=True)
        
        # Export options for hierarchy
        st.subheader("Export Hierarchy Data")
        exporter = StyledExporter()
        
        col_hier1, col_hier2 = st.columns(2)
        
        with col_hier1:
            # CSV export
            csv = detailed_hierarchy.to_csv(index=False)
            st.download_button(
                label="📄 Download CSV",
                data=csv,
                file_name=f"{vessel_name}_hierarchy.csv",
                mime="text/csv"
            )
        
        with col_hier2:
            # Excel export with formatting
            excel_data = exporter.export_hierarchy_excel(detailed_hierarchy)
            st.download_button(
                label="🏗️ Download Styled Excel",
                data=excel_data,
                file_name=f"{vessel_name}_hierarchy.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        # Machinery component breakdown
        st.subheader("Machinery Component Breakdown")
        machinery_breakdown = analyzer.get_machinery_breakdown(filtered_df)
        
        # Create expandable sections for each machinery
        for machinery in machinery_breakdown['Machinery'].unique():
            if machinery and machinery.strip():
                with st.expander(f"🔧 {machinery}"):
                    machinery_data = machinery_breakdown[machinery_breakdown['Machinery'] == machinery]
                    
                    # Apply conditional formatting for machinery breakdown with consistent colors
                    breakdown_data = machinery_data[['Component Type', 'Component Name', 'Maker', 'Model']]
                    
                    def highlight_breakdown_component(row):
                        styles = []
                        for idx, col in enumerate(breakdown_data.columns):
                            if col == 'Component Name':
                                styles.append('background-color: #fff8e1; color: #f57f17')
                            else:
                                styles.append('background-color: #f0f8ff' if row.name % 2 == 0 else 'background-color: #f8f8ff')
                        return styles
                    
                    breakdown_styled = breakdown_data.style.apply(highlight_breakdown_component, axis=1).set_properties(**{
                        'text-align': 'left',
                        'white-space': 'nowrap'
                    })
                    
                    st.dataframe(breakdown_styled, use_container_width=True)
    
    with tab4:
        st.header("Raw Data View")
        
        # Search functionality
        search_term = st.text_input("Search in data", placeholder="Enter search term...")
        
        if search_term:
            # Search across all text columns
            text_columns = ['Maker', 'Model', 'Component Type', 'Particulars', 
                          'Machinery', 'Shell Component', 'Component Name']
            search_mask = False
            for col in text_columns:
                if col in filtered_df.columns:
                    search_mask |= filtered_df[col].str.contains(search_term, case=False, na=False)
            display_df = filtered_df[search_mask]
        else:
            display_df = filtered_df
        
        # Display data with pagination
        st.write(f"Showing {len(display_df)} records")
        
        # Enhanced export options for raw data
        st.subheader("Export Raw Data")
        exporter = StyledExporter()
        
        col_raw1, col_raw2, col_raw3 = st.columns(3)
        
        with col_raw1:
            # CSV export
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📄 Download Filtered CSV",
                data=csv,
                file_name=f"{vessel_name}_filtered_data.csv",
                mime="text/csv"
            )
        
        with col_raw2:
            # Complete dataset CSV
            complete_csv = df.to_csv(index=False)
            st.download_button(
                label="📋 Download Complete CSV",
                data=complete_csv,
                file_name=f"{vessel_name}_complete_data.csv",
                mime="text/csv"
            )
        
        with col_raw3:
            # Comprehensive Excel report
            machinery_details = analyzer.get_unique_machinery_with_details()
            critical_details = analyzer.get_critical_machinery_details(df)
            hierarchy_details = analyzer.get_detailed_hierarchy(df)
            
            excel_summary = exporter.create_summary_excel(
                machinery_details, critical_details, hierarchy_details, 
                df, vessel_name
            )
            st.download_button(
                label="📊 Complete Excel Report",
                data=excel_summary,
                file_name=f"{vessel_name}_complete_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        # Apply conditional formatting to raw data
        raw_data_styled = display_df.style.set_properties(**{
            'text-align': 'left',
            'white-space': 'nowrap'
        }).apply(lambda x: ['background-color: #f8f9fa' if x.name % 2 == 0 else 'background-color: #ffffff' 
                           for _ in range(len(x))], axis=1)
        
        # Display the dataframe
        st.dataframe(raw_data_styled, use_container_width=True, height=600)

def main():
    st.title("⚙️ Ship Machinery Data Analysis")
    
    # Sidebar for file upload and vessel selection
    with st.sidebar:
        st.header("📁 Data Upload")
        
        # File upload - now accepts multiple files
        uploaded_files = st.file_uploader(
            "Upload Machinery CSV files",
            type=['csv'],
            accept_multiple_files=True,
            help="Upload one or more vessel machinery CSV files for analysis"
        )
        
        if uploaded_files:
            if st.button("Load/Reload All Files"):
                with st.spinner("Loading and processing vessel data..."):
                    loaded_vessels, failed_files = load_multiple_vessels(uploaded_files)
                    
                    if loaded_vessels:
                        st.session_state.vessels_data.update(loaded_vessels)
                        
                        # Set default selected vessel if none selected
                        if not st.session_state.selected_vessel and loaded_vessels:
                            st.session_state.selected_vessel = list(loaded_vessels.keys())[0]
                        
                        if failed_files:
                            st.warning(f"Failed to load: {', '.join(failed_files)}")
                        
                        st.rerun()
        
        # Vessel selection
        if st.session_state.vessels_data:
            st.header("🚢 Vessel Selection")
            
            vessel_names = list(st.session_state.vessels_data.keys())
            
            # Vessel selector
            selected_vessel = st.selectbox(
                "Select Vessel to Analyze",
                vessel_names,
                index=vessel_names.index(st.session_state.selected_vessel) if st.session_state.selected_vessel in vessel_names else 0
            )
            
            if selected_vessel != st.session_state.selected_vessel:
                st.session_state.selected_vessel = selected_vessel
                st.rerun()
            
            # Show loaded vessels summary
            st.subheader("📋 Loaded Vessels")
            for vessel_name, data in st.session_state.vessels_data.items():
                status = "🔍 Analyzing" if vessel_name == st.session_state.selected_vessel else "📂 Loaded"
                st.write(f"{status} **{vessel_name}** - {data['total_records']} records")
            
            # Comparison mode toggle
            if len(st.session_state.vessels_data) > 1:
                st.session_state.comparison_mode = st.checkbox(
                    "📊 Comparison Mode", 
                    value=st.session_state.comparison_mode,
                    help="Compare metrics across all loaded vessels"
                )
    
    # Main content
    if st.session_state.vessels_data:
        # Check if comparison mode is enabled
        if st.session_state.comparison_mode and len(st.session_state.vessels_data) > 1:
            display_vessel_comparison()
        else:
            # Single vessel analysis
            if st.session_state.selected_vessel:
                vessel_data = st.session_state.vessels_data[st.session_state.selected_vessel]
                df = vessel_data['df']
                analyzer = vessel_data['analyzer']
                visualizer = vessel_data['visualizer']
                vessel_name = vessel_data['vessel_name']
                
                # Dynamic title based on selected vessel
                st.markdown(f"### {vessel_name} Machinery Analysis Dashboard")
                
                display_single_vessel_analysis(df, analyzer, visualizer, vessel_name)
    else:
        # Welcome message when no data is loaded
        st.info("👈 Please upload a CSV file to begin the analysis")
        
        # Show sample of expected data format
        st.subheader("Expected Data Format")
        st.markdown("""
        The CSV file should contain the following columns:
        - **Critical Machinery**: Indicator for critical machinery (C for critical)
        - **Suffix**: Machinery suffix identifier
        - **Maker**: Manufacturer name
        - **Model**: Model number/name
        - **Component Type**: Type of component
        - **Particulars**: Additional specifications
        - **Machinery**: Main machinery category
        - **Shell Component**: Shell component information
        - **Component Name**: Specific component name
        """)

if __name__ == "__main__":
    main()
