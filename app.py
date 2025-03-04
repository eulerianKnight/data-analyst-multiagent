import streamlit as st
import pandas as pd
import sqlite3
import anthropic
import json
from io import StringIO
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'table_created' not in st.session_state:
    st.session_state.table_created = False
if 'table_name' not in st.session_state:
    st.session_state.table_name = "uploaded_data"
if 'transformations_applied' not in st.session_state:
    st.session_state.transformations_applied = []
if 'edited_df' not in st.session_state:
    st.session_state.edited_df = None
if 'original_filtered_df' not in st.session_state:
    st.session_state.original_filtered_df = None

# Database functions
def initialize_db(db_name="data/app_data.db"):
    os.makedirs(os.path.dirname(db_name), exist_ok=True)
    conn = sqlite3.connect(db_name)
    return conn

def create_table_from_df(conn, df, table_name="uploaded_data"):
    """Create a table in SQLite based on DataFrame structure."""
    # Drop table if it exists
    cursor = conn.cursor()
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    
    # Create table
    df.to_sql(table_name, conn, index=False, if_exists="replace")
    conn.commit()
    return table_name

def get_table_schema(conn, table_name):
    """Get the schema of a table."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    return cursor.fetchall()

def query_table_with_filter(conn, table_name, filter_col, filter_value, col_type):
    """Query the table with the specified filter."""
    if filter_value is None or (isinstance(filter_value, str) and filter_value.strip() == ""):
        # No filter value provided, return all data
        return pd.read_sql(f"SELECT * FROM {table_name}", conn)
    
    if 'int' in col_type or 'real' in col_type or 'float' in col_type or 'double' in col_type:
        # For numeric columns
        min_val, max_val = filter_value
        query = f"SELECT * FROM {table_name} WHERE {filter_col} BETWEEN ? AND ?"
        return pd.read_sql(query, conn, params=[min_val, max_val])
    elif 'text' in col_type or 'varchar' in col_type or 'char' in col_type:
        # For text columns
        query = f"SELECT * FROM {table_name} WHERE {filter_col} LIKE ?"
        return pd.read_sql(query, conn, params=[f"%{filter_value}%"])
    else:
        # Default case
        return pd.read_sql(f"SELECT * FROM {table_name}", conn)

def save_changes_to_db(conn, table_name, original_df, edited_df):
    """Save changes from the edited dataframe back to the database."""
    # Remove the original_index column we added
    if 'original_index' in edited_df.columns:
        edited_df = edited_df.drop('original_index', axis=1)
    
    # Compare original and edited dataframes to identify changes
    if not original_df.equals(edited_df):
        # Drop and recreate the table with the new data
        cursor = conn.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.commit()
        
        edited_df.to_sql(table_name, conn, index=False, if_exists="replace")
        conn.commit()
        
        return True
    
    return False

# Claude Agent functions
def data_analysis_agent(client, df):
    """Agent for data analysis and exploration."""
    df_info = StringIO()
    df.info(buf=df_info)
    df_info_str = df_info.getvalue()
    
    prompt = f"""
    Analyze this dataset and provide insights:
    
    Data Preview:
    {df.head(5).to_string()}
    
    DataFrame Info:
    {df_info_str}
    
    Basic Statistics:
    {df.describe().to_string()}
    
    Please provide:
    1. A summary of what this dataset contains
    2. Key observations about the data quality (missing values, outliers, etc.)
    3. Potential use cases for this data
    
    Format your response as plain text.
    """
    
    response = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=1000,
        system="You are a data analysis expert who specializes in exploratory data analysis.",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return response.content[0].text

def get_transformation_suggestions(client, df_head, df_info, column_types):
    """Get transformation suggestions from Claude based on DataFrame analysis."""
    prompt = f"""
    I have a CSV file with the following structure:
    
    Data Preview:
    {df_head.to_string()}
    
    DataFrame Info:
    {df_info}
    
    Column Types:
    {column_types}
    
    Please suggest 3-5 useful data transformations for this dataset. For each transformation, provide:
    1. A short description of the transformation
    2. The Python code to execute it
    3. Why this transformation would be useful
    
    IMPORTANT: In your code examples, always use the variable name 'df' for the DataFrame.
    
    Format your response as a JSON array of objects with keys 'description', 'code', and 'reason'.
    """
    
    response = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=1000,
        temperature=0,
        system="You are a data analysis assistant that specializes in pandas transformations.",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    # Extract the JSON from Claude's response
    try:
        response_text = response.content[0].text
        json_start = response_text.find('[')
        json_end = response_text.rfind(']') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            transformations = json.loads(json_str)
        else:
            # Fallback to parsing the full response if we can't find clear JSON delimiters
            transformations = json.loads(response_text)
        
        return transformations
    except Exception as e:
        st.error(f"Error parsing Claude's response: {e}")
        st.text(response.content[0].text)
        return []

def schema_design_agent(client, df):
    """Agent for database schema design and validation."""
    df_info = StringIO()
    df.info(buf=df_info)
    df_info_str = df_info.getvalue()
    
    prompt = f"""
    I need to store this DataFrame in a SQLite database:
    
    Data Preview:
    {df.head(5).to_string()}
    
    DataFrame Info:
    {df_info_str}
    
    Please suggest:
    1. Appropriate column types for SQLite
    2. Any constraints that should be applied (PRIMARY KEY, NOT NULL, etc.)
    3. Any indexes that should be created for better query performance
    
    Format your response as a JSON object with keys 'column_types', 'constraints', and 'indexes'.
    """
    
    response = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=1000,
        system="You are a database expert who specializes in schema design.",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    # Extract the JSON from the response
    try:
        response_text = response.content[0].text
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            json_str = response_text[json_start:json_end]
            schema_suggestions = json.loads(json_str)
        else:
            schema_suggestions = json.loads(response_text)
            
        return schema_suggestions
    except Exception as e:
        st.error(f"Error parsing Claude's schema suggestions: {e}")
        st.text(response.content[0].text)
        return {
            "column_types": {},
            "constraints": [],
            "indexes": []
        }

def query_generation_agent(client, table_schema, table_name, query_description):
    """Agent for generating SQL queries based on user request."""
    schema_str = "\n".join([f"- {col[1]} ({col[2]})" for col in table_schema])
    
    prompt = f"""
    I need an SQL query for a SQLite database with the following schema:
    
    Table name: {table_name}
    
    Columns:
    {schema_str}
    
    User request: {query_description}
    
    IMPORTANT: Use EXACTLY this table name in your query: {table_name}
    
    Please generate the SQL query that fulfills this request.
    Format your response as a single SQL query without any additional text.
    """
    
    response = client.messages.create(
        model="claude-3-7-sonnet-20250219",
        max_tokens=500,
        temperature=0,
        system="You are an SQL expert who specializes in writing efficient queries.",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    # Try to extract just the SQL query
    response_text = response.content[0].text
    
    # Look for SQL in code blocks
    import re
    sql_match = re.search(r"```sql\n(.*?)\n```", response_text, re.DOTALL)
    
    if sql_match:
        return sql_match.group(1).strip()
    
    # If no code block, just return the response
    return response_text.strip()

def apply_transformation(df, transformation_code):
    """Apply a transformation to the DataFrame using exec."""
    # Create a local copy of the dataframe to be used in the exec context
    df_transformed = df.copy()
    
    # Define a secure execution context with only the necessary variables
    exec_context = {
        'df_transformed': df_transformed,
        'df': df_transformed,  # Add df as alias to support Claude's suggestions
        'pd': pd,
        'np': __import__('numpy'),  # Add numpy for common transformations
    }
    
    try:
        # Execute the transformation code
        exec(transformation_code, exec_context)
        # Return the transformed dataframe
        return exec_context['df_transformed'], None
    except Exception as e:
        return df, str(e)

def generate_filter_ui(conn, table_name):
    """Generate filter UI elements based on table schema."""
    schema = get_table_schema(conn, table_name)
    
    # Get all column names
    columns = [col[1] for col in schema]
    
    # Let user select a column to filter on
    filter_col = st.selectbox("Select column to filter on:", columns)
    
    # Get column type
    col_type = [col[2] for col in schema if col[1] == filter_col][0].lower()
    
    # Create appropriate filter input based on column type
    filter_value = None
    if 'int' in col_type or 'integer' in col_type:
        # For numeric columns, provide a range slider
        # First, get min/max values
        cursor = conn.cursor()
        cursor.execute(f"SELECT MIN({filter_col}), MAX({filter_col}) FROM {table_name}")
        min_val, max_val = cursor.fetchone()
        if min_val is not None and max_val is not None:
            filter_value = st.slider(f"Filter by {filter_col}", float(min_val), float(max_val), (float(min_val), float(max_val)))
    elif 'text' in col_type or 'varchar' in col_type or 'char' in col_type:
        # For text columns, provide a text input
        filter_value = st.text_input(f"Filter by {filter_col} (contains):")
    elif 'real' in col_type or 'float' in col_type or 'double' in col_type:
        # For float columns, provide a range slider
        cursor = conn.cursor()
        cursor.execute(f"SELECT MIN({filter_col}), MAX({filter_col}) FROM {table_name}")
        min_val, max_val = cursor.fetchone()
        if min_val is not None and max_val is not None:
            filter_value = st.slider(f"Filter by {filter_col}", float(min_val), float(max_val), (float(min_val), float(max_val)))
    
    return filter_col, filter_value, col_type

def create_editable_dataframe(df):
    """Create an editable version of the dataframe in Streamlit."""
    # Create a copy of the dataframe with an index column for tracking
    df_edit = df.copy().reset_index().rename(columns={"index": "original_index"})
    
    # Use Streamlit's data editor
    edited_df = st.data_editor(df_edit)
    
    return edited_df

def generate_sample_data():
    """Generate a sample sales dataset."""
    # Create a sample sales dataset
    import numpy as np
    from datetime import datetime, timedelta
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate dates
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(100)]
    
    # Generate product IDs and names
    product_ids = ['P' + str(i).zfill(3) for i in range(1, 11)]
    product_names = [
        'Laptop', 'Smartphone', 'Tablet', 'Monitor', 'Keyboard',
        'Mouse', 'Headphones', 'Speaker', 'Camera', 'Printer'
    ]
    
    # Generate regions
    regions = ['North', 'South', 'East', 'West', 'Central']
    
    # Generate data
    data = []
    for _ in range(500):
        date = np.random.choice(dates)
        product_idx = np.random.randint(0, len(product_ids))
        product_id = product_ids[product_idx]
        product_name = product_names[product_idx]
        quantity = np.random.randint(1, 11)
        price = np.random.uniform(10, 1000)
        region = np.random.choice(regions)
        
        data.append({
            'date': date.strftime('%Y-%m-%d'),
            'product_id': product_id,
            'product_name': product_name,
            'quantity': quantity,
            'price': round(price, 2),
            'total': round(quantity * price, 2),
            'region': region
        })
    
    return pd.DataFrame(data)

# Main application
st.title("Multi-Agent CSV Processing System")

# Get API key from environment variable
api_key = os.getenv("ANTHROPIC_API_KEY")

# Provide UI option to override the API key if needed
api_key_override = st.sidebar.text_input("Override Anthropic API key (optional):", type="password")
if api_key_override:
    api_key = api_key_override

# Check if API key is available
if not api_key:
    st.sidebar.warning("No Anthropic API key found. Set the ANTHROPIC_API_KEY in your .env file or enter it above.")
    use_claude = False
else:
    use_claude = True
    try:
        client = anthropic.Anthropic(api_key=api_key)
        st.sidebar.success("Claude agents are enabled!")
    except Exception as e:
        st.sidebar.error(f"Error initializing Claude client: {e}")
        use_claude = False

# Initialize database connection
conn = initialize_db()

# App sections in tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Upload & Analyze", 
    "2. Transform", 
    "3. Database Storage", 
    "4. Query & Edit"
])

with tab1:
    st.header("Upload CSV Data")
    upload_option = st.radio(
        "Choose data source:",
        ["Upload CSV file", "Use sample data"]
    )
    
    if upload_option == "Upload CSV file":
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            try:
                st.session_state.df = pd.read_csv(uploaded_file)
                st.success(f"File uploaded successfully: {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
    else:
        if st.button("Generate sample sales data"):
            st.session_state.df = generate_sample_data()
            st.success("Sample sales data generated")
    
    if st.session_state.df is not None:
        st.subheader("Data Preview")
        st.dataframe(st.session_state.df.head())
        
        # Data analysis with Claude
        if use_claude:
            if st.button("Analyze Data with Claude"):
                with st.spinner("Claude is analyzing your data..."):
                    analysis = data_analysis_agent(client, st.session_state.df)
                    st.markdown(analysis)
        else:
            st.info("Enter your Anthropic API key to enable data analysis with Claude.")

with tab2:
    st.header("Data Transformation")
    
    if st.session_state.df is not None:
        # Display current data
        st.subheader("Current Data")
        st.dataframe(st.session_state.df.head())
        
        if use_claude:
            if st.button("Get Transformation Suggestions"):
                with st.spinner("Claude is generating transformation suggestions..."):
                    df_info = StringIO()
                    st.session_state.df.info(buf=df_info)
                    df_info_str = df_info.getvalue()
                    
                    column_types = st.session_state.df.dtypes.to_dict()
                    column_types = {str(k): str(v) for k, v in column_types.items()}
                    
                    transformations = get_transformation_suggestions(
                        client, st.session_state.df.head(), df_info_str, column_types
                    )
                    
                    # Store transformations in session state
                    st.session_state.transformations = transformations
            
            # Display transformation suggestions if available
            if 'transformations' in st.session_state:
                for i, transform in enumerate(st.session_state.transformations):
                    with st.expander(f"Transformation {i+1}: {transform['description']}"):
                        st.markdown(f"**Reason:** {transform['reason']}")
                        st.code(transform['code'], language="python")
                        
                        if st.button(f"Apply Transformation {i+1}"):
                            with st.spinner("Applying transformation..."):
                                transformed_df, error = apply_transformation(
                                    st.session_state.df, transform['code']
                                )
                                
                                if error:
                                    st.error(f"Error applying transformation: {error}")
                                else:
                                    st.success("Transformation applied successfully!")
                                    st.session_state.df = transformed_df
                                    st.session_state.transformations_applied.append(transform['description'])
                                    
                                    # Show the transformed data
                                    st.subheader("Transformed Data")
                                    st.dataframe(transformed_df.head())
            
            # Show applied transformations
            if st.session_state.transformations_applied:
                st.subheader("Applied Transformations:")
                for i, transform in enumerate(st.session_state.transformations_applied):
                    st.markdown(f"{i+1}. {transform}")
        else:
            st.info("Enter your Anthropic API key to enable transformation suggestions with Claude.")
    else:
        st.info("Please upload or generate data in the 'Upload & Analyze' tab first.")

with tab3:
    st.header("Store in SQLite Database")
    
    if st.session_state.df is not None:
        table_name = st.text_input("Table name:", "uploaded_data")
        
        if st.button("Store in Database"):
            with st.spinner("Creating database table..."):
                if use_claude:
                    # Get schema suggestions from Claude
                    schema_suggestions = schema_design_agent(client, st.session_state.df)
                    
                    # Display schema suggestions
                    st.subheader("Schema Suggestions from Claude")
                    st.json(schema_suggestions)
                
                # Create table in database
                st.session_state.table_name = create_table_from_df(conn, st.session_state.df, table_name)
                st.session_state.table_created = True
                
                st.success(f"Data stored in SQLite table: {st.session_state.table_name}")
    else:
        st.info("Please upload or generate data in the 'Upload & Analyze' tab first.")

with tab4:
    st.header("Query and Edit Data")
    
    if not st.session_state.table_created:
        st.info("Please store your data in the database in the 'Database Storage' tab first.")
    else:
        st.subheader("Filter Data")
        
        # Generate filter UI
        filter_col, filter_value, col_type = generate_filter_ui(conn, st.session_state.table_name)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Apply Filter"):
                with st.spinner("Querying database..."):
                    filtered_df = query_table_with_filter(
                        conn, st.session_state.table_name, filter_col, filter_value, col_type
                    )
                    
                    # Store original dataframe for comparison when saving changes
                    st.session_state.original_filtered_df = filtered_df.copy()
                    
                    # Display as editable dataframe
                    st.subheader("Editable Results")
                    st.session_state.edited_df = create_editable_dataframe(filtered_df)
        
        if use_claude:
            # Custom query section
            st.subheader("Custom Natural Language Query")
            query_description = st.text_input("Describe what you want to query (e.g., 'Show me total sales by region'):")
            
            if query_description and st.button("Generate and Run Query"):
                with st.spinner("Generating SQL query..."):
                    # Get table schema
                    table_schema = get_table_schema(conn, st.session_state.table_name)
                    
                    # Generate query with Claude
                    sql_query = query_generation_agent(client, table_schema, st.session_state.table_name, query_description)
                    
                    st.code(sql_query, language="sql")
                    
                    try:
                        # Execute query
                        query_results = pd.read_sql(sql_query, conn)
                        st.write("Query Results:")
                        st.dataframe(query_results)
                    except Exception as e:
                        st.error(f"Error executing query: {e}")
        
        # Save changes section
        if st.session_state.edited_df is not None and st.session_state.original_filtered_df is not None:
            with col2:
                if st.button("Save Changes to Database"):
                    with st.spinner("Saving changes..."):
                        # Remove the original_index column from original df for comparison
                        original_df_no_index = st.session_state.original_filtered_df.copy()
                        if 'original_index' in original_df_no_index.columns:
                            original_df_no_index = original_df_no_index.drop('original_index', axis=1)
                        
                        changes_saved = save_changes_to_db(
                            conn, st.session_state.table_name, 
                            original_df_no_index, st.session_state.edited_df
                        )
                        
                        if changes_saved:
                            st.success("Changes saved successfully!")
                        else:
                            st.info("No changes detected.")

# Cleanup on app close
conn.close()