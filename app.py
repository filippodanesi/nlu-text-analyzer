import streamlit as st
import pandas as pd
import json
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from ibm_watson.natural_language_understanding_v1 import Features, KeywordsOptions, EntitiesOptions, ConceptsOptions, CategoriesOptions, RelationsOptions, SentimentOptions

# Configurazione della pagina Streamlit
st.set_page_config(page_title="IBM Watson NLU Analyzer", layout="wide")

# Minimalist Vercel-inspired CSS styling
st.markdown("""
<style>
    /* Global styles */
    body {font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;}
    .main {background-color: white; color: black;}
    
    /* Header styling */
    .header {padding: 1.5rem 0; border-bottom: 1px solid #eaeaea; margin-bottom: 0.5rem;}
    .header h1 {font-size: 1.5rem; font-weight: 600; margin: 0;}
    
    /* Description styling */
    .description {padding: 0.75rem 0 1.5rem 0; color: #666; font-size: 0.875rem; line-height: 1.5; border-bottom: 1px solid #eaeaea; margin-bottom: 1.5rem;}
    
    /* Panel styling */
    .panel {border: 1px solid #eaeaea; border-radius: 5px; padding: 1.5rem; margin-bottom: 1.5rem;}
    .panel-header {font-size: 0.875rem; font-weight: 600; margin-bottom: 1rem; color: #666;}
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {gap: 0; border-bottom: 1px solid #eaeaea;}
    .stTabs [data-baseweb="tab"] {font-size: 0.875rem; color: #666; padding: 0.75rem 1rem;}
    .stTabs [aria-selected="true"] {color: black !important; border-bottom: 2px solid black !important; font-weight: 600;}
    
    /* Button styling */
    .stButton button {background-color: black; color: white; border-radius: 5px;}
    
    /* Table styling */
    div[data-testid="stTable"] table {border-collapse: collapse; width: 100%;}
    div[data-testid="stTable"] thead tr th {background-color: #fafafa; border-bottom: 1px solid #eaeaea; text-align: left; padding: 0.75rem; font-size: 0.75rem; font-weight: 600; color: #666;}
    div[data-testid="stTable"] tbody tr td {border-bottom: 1px solid #eaeaea; padding: 0.75rem; font-size: 0.875rem;}
    div[data-testid="stTable"] tbody tr:hover {background-color: #fafafa;}
    
    /* Target keyword highlighting */
    .target-keyword {color: #10b981; font-weight: 600;}
    
    /* Dataframe styling for highlighted rows */
    .highlight-green {
        background-color: rgba(16, 185, 129, 0.1) !important;
    }
    
    /* Code/JSON styling */
    .json-box {background-color: #fafafa; border: 1px solid #eaeaea; border-radius: 5px; padding: 1rem; font-family: 'Menlo', 'Monaco', 'Courier New', monospace; font-size: 0.75rem; overflow-x: auto;}
    
    /* Utility classes */
    .monospace {font-family: 'Menlo', 'Monaco', 'Courier New', monospace; font-size: 0.75rem;}
    .subtle-text {color: #666; font-size: 0.75rem;}
    .divider {border-top: 1px solid #eaeaea; margin: 1.5rem 0;}
    
    /* Stats box */
    .stats-box {background-color: #fafafa; border: 1px solid #eaeaea; border-radius: 5px; padding: 1rem; margin-top: 1rem;}
    .stats-header {font-weight: 600; font-size: 0.75rem; margin-bottom: 0.5rem; color: #666;}
    .stats-item {font-size: 0.75rem; margin-bottom: 0.25rem; color: #333;}
    
    /* Target keywords box */
    .target-box {background-color: #fafafa; border: 1px solid #eaeaea; border-radius: 5px; padding: 1rem; margin: 1rem 0;}
    .target-header {font-weight: 600; font-size: 0.75rem; margin-bottom: 0.5rem; color: #666;}
    .target-pills {display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.5rem;}
    .target-pill {background-color: #f3f4f6; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.75rem; color: #333;}
</style>
<div class="header">
    <h1>IBM Watson Natural Language Understanding API</h1>
</div>
<div class="description">
    Enterprise-grade natural language processing for extracting metadata from text such as keywords, 
    entities, categories, and relationships. This tool leverages advanced linguistic analysis and 
    machine learning to identify key textual elements and provide insights about content structure.
</div>
""", unsafe_allow_html=True)

# Check for API credentials in Streamlit Cloud secrets or local secrets
# This approach works both locally and on Streamlit Cloud
try:
    # First try Streamlit Cloud secrets
    has_secrets = "ibm_watson" in st.secrets and "api_key" in st.secrets["ibm_watson"] and "url" in st.secrets["ibm_watson"]
    api_key = st.secrets["ibm_watson"]["api_key"]
    url = st.secrets["ibm_watson"]["url"]
    use_secrets = True
except Exception:
    has_secrets = False
    use_secrets = False
    api_key = None
    url = None

# Sidebar for API configuration
with st.sidebar:
    st.markdown("### API Configuration")
    
    # API Key and URL input
    if has_secrets:
        st.success("Using API credentials from Streamlit secrets")
        use_api_form = st.checkbox("Enter API credentials manually instead", value=False)
        
        if use_api_form:
            api_key = st.text_input("API Key", type="password", 
                                  help="Your IBM Watson NLU API key. Required for API authentication.")
            url = st.text_input("Service URL", 
                             help="Full URL of your IBM Watson NLU instance including the instance ID.")
    else:
        st.warning("No secrets detected. Please enter your API credentials below.")
        api_key = st.text_input("API Key", type="password", 
                              help="Your IBM Watson NLU API key. Required for API authentication.")
        
        url_options = {
            "Dallas (us-south)": "https://api.us-south.natural-language-understanding.watson.cloud.ibm.com",
            "Washington DC (us-east)": "https://api.us-east.natural-language-understanding.watson.cloud.ibm.com",
            "Frankfurt (eu-de)": "https://api.eu-de.natural-language-understanding.watson.cloud.ibm.com",
            "London (eu-gb)": "https://api.eu-gb.natural-language-understanding.watson.cloud.ibm.com",
            "Sydney (au-syd)": "https://api.au-syd.natural-language-understanding.watson.cloud.ibm.com",
            "Tokyo (jp-tok)": "https://api.jp-tok.natural-language-understanding.watson.cloud.ibm.com",
            "Seoul (kr-seo)": "https://api.kr-seo.natural-language-understanding.watson.cloud.ibm.com",
            "Custom": "custom"
        }
        
        selected_region = st.selectbox("Region", options=list(url_options.keys()), 
                                     help="Select the region where your IBM Watson NLU instance is deployed.")
        
        if selected_region == "Custom":
            url = st.text_input("Custom API URL", 
                             help="Enter the full URL of your IBM Watson NLU instance.")
        else:
            url = f"{url_options[selected_region]}/instances/"
            instance_id = st.text_input("Instance ID", 
                                      help="The instance ID from your IBM Watson NLU service credentials.")
            if instance_id:
                url += instance_id
    
    st.markdown("### Analysis Features")
    
    # Feature selection (organized as in IBM's UI)
    st.markdown("#### Extraction")
    analyze_keywords = st.checkbox("Keywords", value=True, 
                                 help="Extract relevant keywords from text.")
    analyze_entities = st.checkbox("Entities", value=True, 
                                 help="Identify people, organizations, locations, and other entities.")
    analyze_concepts = st.checkbox("Concepts", value=True, 
                                 help="Identify general concepts that aren't necessarily directly referenced in the text.")
    analyze_relations = st.checkbox("Relations", value=False, 
                                  help="Recognize when two entities are related and identify the type of relation.")
    
    st.markdown("#### Classification")
    analyze_categories = st.checkbox("Categories", value=True, 
                                   help="Categorize content into a hierarchical taxonomy.")
    
    st.markdown("#### Analysis Parameters")
    advanced_params = st.expander("Advanced Parameters")
    with advanced_params:
        keywords_limit = st.number_input("Keywords limit", min_value=1, max_value=50, value=10, 
                                       help="Maximum number of keywords to return.")
        entities_limit = st.number_input("Entities limit", min_value=1, max_value=50, value=10, 
                                       help="Maximum number of entities to return.")
        concepts_limit = st.number_input("Concepts limit", min_value=1, max_value=50, value=5, 
                                       help="Maximum number of concepts to return.")
        categories_limit = st.number_input("Categories limit", min_value=1, max_value=10, value=3, 
                                         help="Maximum number of categories to return.")
        language = st.selectbox("Language", options=["en", "ar", "de", "es", "fr", "it", "ja", "ko", "nl", "pt", "zh"], 
                              help="Language of the text. If not specified, the service will attempt to detect the language.")

# Main content - Input panel
st.markdown('<div class="panel">', unsafe_allow_html=True)
st.markdown('<div class="panel-header">INPUT</div>', unsafe_allow_html=True)

# Removed URL option as requested, keeping only Text and Text file
text_input_method = st.radio("Select input method", ["Text", "Text file"])

if text_input_method == "Text":
    text_to_analyze = st.text_area("Enter text to analyze", height=150, 
                                  help="Raw text to be analyzed.")
    input_type = "text"
else:
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"], 
                                    help="Text file to be analyzed.")
    if uploaded_file is not None:
        text_to_analyze = uploaded_file.getvalue().decode("utf-8")
        st.text_area("File content", text_to_analyze, height=150)
        input_type = "text"
    else:
        text_to_analyze = ""
        input_type = "text"

# Target keywords input
target_keywords = st.text_input("Target keywords (comma-separated)", 
                               help="Enter keywords to highlight in the results. These will be shown in green when they appear.")

# Display target keywords as pills if provided
if target_keywords:
    # Convert target keywords to list, clean up and normalize
    target_keywords_list = [kw.strip().lower() for kw in target_keywords.split(',') if kw.strip()]
    if target_keywords_list:
        st.markdown('<div class="target-box">', unsafe_allow_html=True)
        st.markdown('<div class="target-header">TARGET KEYWORDS</div>', unsafe_allow_html=True)
        st.markdown('<div class="target-pills">', unsafe_allow_html=True)
        for kw in target_keywords_list:
            st.markdown(f'<div class="target-pill">{kw}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

analyze_button = st.button("Analyze", help="Submit the text for analysis")
st.markdown('</div>', unsafe_allow_html=True)

# Results container
results_container = st.container()

# Execute analysis when button is clicked
if analyze_button and text_to_analyze and api_key:
    try:
        # Process target keywords
        target_keywords_list = [kw.strip().lower() for kw in target_keywords.split(',') if kw.strip()]
        
        # Configure IBM Watson authentication
        authenticator = IAMAuthenticator(api_key)
        natural_language_understanding = NaturalLanguageUnderstandingV1(
            version='2022-04-07',
            authenticator=authenticator
        )
        natural_language_understanding.set_service_url(url)
        
        # Prepare features to analyze
        features = {}
        
        if analyze_keywords:
            features["keywords"] = KeywordsOptions(sentiment=True, emotion=False, limit=keywords_limit)
        
        if analyze_entities:
            features["entities"] = EntitiesOptions(sentiment=True, emotion=False, limit=entities_limit)
        
        if analyze_concepts:
            features["concepts"] = ConceptsOptions(limit=concepts_limit)
        
        if analyze_categories:
            features["categories"] = CategoriesOptions(limit=categories_limit)
            
        if analyze_relations:
            features["relations"] = RelationsOptions()
        
        # Configure input based on type
        params = {
            "features": Features(**features),
            "language": language
        }
        
        if input_type == "text":
            params["text"] = text_to_analyze
        
        # API call
        with st.spinner("Analyzing..."):
            response = natural_language_understanding.analyze(**params).get_result()
            
            with results_container:
                st.success("Analysis completed successfully")
                
                # Add text stats
                st.markdown('<div class="stats-box">', unsafe_allow_html=True)
                st.markdown('<div class="stats-header">TEXT STATISTICS</div>', unsafe_allow_html=True)
                
                # Count words, sentences, and characters
                word_count = len(text_to_analyze.split())
                sentence_count = text_to_analyze.count('.') + text_to_analyze.count('!') + text_to_analyze.count('?')
                char_count = len(text_to_analyze)
                
                # Display stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f'<div class="stats-item">Words: {word_count}</div>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<div class="stats-item">Sentences: {sentence_count}</div>', unsafe_allow_html=True)
                with col3:
                    st.markdown(f'<div class="stats-item">Characters: {char_count}</div>', unsafe_allow_html=True)
                
                # Display detected language if available
                if "language" in response:
                    st.markdown(f'<div class="stats-item">Detected Language: {response["language"].upper()}</div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Main tabs as in IBM's interface
                main_tabs = st.tabs(["Extraction", "Classification", "Linguistics", "Custom"])
                
                with main_tabs[0]:  # Extraction tab
                    if analyze_keywords or analyze_entities or analyze_concepts or analyze_relations:
                        # Subtabs for extraction types
                        extraction_tabs = st.tabs(["Entities", "Keywords", "Concepts", "Relations"])
                        
                        # Entities tab
                        with extraction_tabs[0]:
                            if analyze_entities and "entities" in response:
                                entities_df = pd.DataFrame(response["entities"])
                                if not entities_df.empty:
                                    # Simplify dataframe to show only relevant columns
                                    display_cols = ["text", "type", "relevance"]
                                    if "sentiment" in entities_df.columns:
                                        entities_df["sentiment_score"] = entities_df["sentiment"].apply(
                                            lambda x: x.get("score", 0) if isinstance(x, dict) else 0
                                        )
                                        display_cols.append("sentiment_score")
                                    
                                    # Sort and display
                                    entities_df = entities_df.sort_values(by="relevance", ascending=False)
                                    
                                    # For entities that match target keywords, apply style
                                    if target_keywords_list:
                                        # Check if entities match any target keywords
                                        def highlight_target_entities(row):
                                            text = row['text'].lower()
                                            for kw in target_keywords_list:
                                                if kw == text or kw in text.split():
                                                    return ['highlight-green'] * len(row)
                                            return [''] * len(row)
                                        
                                        # Apply styling
                                        st.dataframe(
                                            entities_df[display_cols].style.apply(highlight_target_entities, axis=1),
                                            use_container_width=True
                                        )
                                    else:
                                        st.dataframe(entities_df[display_cols], use_container_width=True)
                                else:
                                    st.info("No entities found in the analyzed text.")
                            else:
                                st.info("Enable the Entities option in the sidebar to view entity analysis results.")
                        
                        # Keywords tab
                        with extraction_tabs[1]:
                            if analyze_keywords and "keywords" in response:
                                keywords_df = pd.DataFrame(response["keywords"])
                                if not keywords_df.empty:
                                    # Simplify dataframe
                                    if "sentiment" in keywords_df.columns:
                                        keywords_df["sentiment_score"] = keywords_df["sentiment"].apply(
                                            lambda x: x.get("score", 0) if isinstance(x, dict) else 0
                                        )
                                    
                                    # Sort and display
                                    keywords_df = keywords_df.sort_values(by="relevance", ascending=False)
                                    display_cols = ["text", "relevance"]
                                    if "sentiment_score" in keywords_df.columns:
                                        display_cols.append("sentiment_score")
                                    
                                    # For keywords that match target keywords, apply style
                                    if target_keywords_list:
                                        # Check if keywords match any target keywords
                                        def highlight_target_keywords(row):
                                            text = row['text'].lower()
                                            for kw in target_keywords_list:
                                                if kw == text or kw in text.split():
                                                    return ['highlight-green'] * len(row)
                                            return [''] * len(row)
                                        
                                        # Apply styling
                                        st.dataframe(
                                            keywords_df[display_cols].style.apply(highlight_target_keywords, axis=1),
                                            use_container_width=True
                                        )
                                    else:
                                        st.dataframe(keywords_df[display_cols], use_container_width=True)
                                else:
                                    st.info("No keywords found in the analyzed text.")
                            else:
                                st.info("Enable the Keywords option in the sidebar to view keyword analysis results.")
                        
                        # Concepts tab
                        with extraction_tabs[2]:
                            if analyze_concepts and "concepts" in response:
                                concepts_df = pd.DataFrame(response["concepts"])
                                if not concepts_df.empty:
                                    # Sort and display
                                    concepts_df = concepts_df.sort_values(by="relevance", ascending=False)
                                    
                                    # For concepts that match target keywords, apply style
                                    if target_keywords_list:
                                        # Check if concepts match any target keywords
                                        def highlight_target_concepts(row):
                                            text = row['text'].lower()
                                            for kw in target_keywords_list:
                                                if kw == text or kw in text.split():
                                                    return ['highlight-green'] * len(row)
                                            return [''] * len(row)
                                        
                                        # Apply styling
                                        st.dataframe(
                                            concepts_df[["text", "relevance"]].style.apply(highlight_target_concepts, axis=1),
                                            use_container_width=True
                                        )
                                    else:
                                        st.dataframe(concepts_df[["text", "relevance"]], use_container_width=True)
                                else:
                                    st.info("No concepts found in the analyzed text.")
                            else:
                                st.info("Enable the Concepts option in the sidebar to view concept analysis results.")
                        
                        # Relations tab
                        with extraction_tabs[3]:
                            if analyze_relations and "relations" in response:
                                relations = response["relations"]
                                if relations:
                                    relations_data = []
                                    for rel in relations:
                                        relation_type = rel.get("type", "")
                                        sentence = rel.get("sentence", "")
                                        score = rel.get("score", 0)
                                        
                                        # Extract entities involved in the relation
                                        arguments = []
                                        for arg in rel.get("arguments", []):
                                            entity_text = arg.get("text", "")
                                            entity_type = ""
                                            entities = arg.get("entities", [])
                                            if entities and len(entities) > 0:
                                                entity_type = entities[0].get("type", "")
                                            arguments.append(f"{entity_text} ({entity_type})")
                                        
                                        relation_args = " → ".join(arguments)
                                        relations_data.append({
                                            "Relation Type": relation_type,
                                            "Elements": relation_args,
                                            "Sentence": sentence,
                                            "Confidence": score
                                        })
                                    
                                    relations_df = pd.DataFrame(relations_data)
                                    
                                    # Highlight relations that contain target keywords
                                    if target_keywords_list:
                                        def highlight_target_relations(row):
                                            elements = row['Elements'].lower()
                                            sentence = row['Sentence'].lower() if isinstance(row['Sentence'], str) else ""
                                            
                                            for kw in target_keywords_list:
                                                if kw in elements or kw in sentence:
                                                    return ['highlight-green'] * len(row)
                                            return [''] * len(row)
                                        
                                        # Apply styling
                                        st.dataframe(
                                            relations_df.style.apply(highlight_target_relations, axis=1),
                                            use_container_width=True
                                        )
                                    else:
                                        st.dataframe(relations_df, use_container_width=True)
                                else:
                                    st.info("No relations found in the analyzed text.")
                            else:
                                st.info("Enable the Relations option in the sidebar to view relation analysis results.")
                    else:
                        st.info("Select at least one extraction option in the sidebar.")
                
                with main_tabs[1]:  # Classification tab
                    if analyze_categories and "categories" in response:
                        categories_df = pd.DataFrame(response["categories"])
                        if not categories_df.empty:
                            # Rename columns for better clarity
                            if "label" in categories_df.columns and "score" in categories_df.columns:
                                categories_df = categories_df.rename(columns={"label": "Category", "score": "Confidence"})
                            
                            # Highlight categories that contain target keywords
                            if target_keywords_list:
                                def highlight_target_categories(row):
                                    category = row['Category'].lower() if isinstance(row['Category'], str) else ""
                                    
                                    for kw in target_keywords_list:
                                        if kw in category:
                                            return ['highlight-green'] * len(row)
                                    return [''] * len(row)
                                
                                # Apply styling
                                st.dataframe(
                                    categories_df.style.apply(highlight_target_categories, axis=1),
                                    use_container_width=True
                                )
                            else:
                                st.dataframe(categories_df, use_container_width=True)
                        else:
                            st.info("No categories found in the analyzed text.")
                    else:
                        st.info("Enable the Categories option in the sidebar to view classification results.")
                
                with main_tabs[2]:  # Linguistics tab
                    st.info("To enable linguistic analysis, select the corresponding options in the sidebar.")
                
                with main_tabs[3]:  # Custom tab
                    st.info("To use custom models, configure the appropriate settings in the sidebar.")
                
                # Raw JSON results
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                st.markdown("### API Response")
                
                with st.expander("View raw JSON response"):
                    st.markdown('<div class="json-box">', unsafe_allow_html=True)
                    st.json(response)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Add copy to clipboard button
                    if st.button("Copy JSON to clipboard"):
                        st.code(json.dumps(response, indent=2))
                        st.success("JSON copied to clipboard!")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.markdown(f"<div class='json-box'>Error details: {str(e)}</div>", unsafe_allow_html=True)

elif analyze_button and not text_to_analyze:
    st.warning("Please enter text to analyze")

elif analyze_button and not api_key:
    st.warning("Please enter your IBM Watson NLU API Key")

# Instructions for Streamlit Cloud secrets
if not has_secrets:
    with st.expander("How to set up secrets in Streamlit Cloud"):
        st.markdown("""
        ### Setting up secrets for Streamlit Cloud
        
        To securely store your IBM Watson NLU credentials in Streamlit Cloud:
        
        1. Deploy this app to Streamlit Cloud
        2. Go to your app settings
        3. Navigate to the "Secrets" section
        4. Add your secrets in TOML format:
        
        ```toml
        [ibm_watson]
        api_key = "your_api_key_here"
        url = "https://api.region.natural-language-understanding.watson.cloud.ibm.com/instances/your_instance_id"
        ```
        
        5. Save the secrets and restart your app
        
        This allows you to keep your API credentials secure and separate from your code.
        """)

# Footer
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="subtle-text">IBM Watson Natural Language Understanding API Explorer - v1.0.0</div>', unsafe_allow_html=True)
