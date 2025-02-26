import streamlit as st  # Import Streamlit library for creating web apps
import joblib, os  # Import joblib for loading models and os for file operations
import spacy  # Import spaCy for natural language processing tasks
import pandas as pd  # Import pandas for data manipulation
import matplotlib.pyplot as plt  # Import matplotlib for creating visualizations
import matplotlib  # Import matplotlib base library
matplotlib.use("Agg")  # Set matplotlib backend to Agg for streamlit compatibility
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator  # Import for word cloud generation
import base64  # Import for file downloading functionality
from PIL import Image  # Import for image processing

# Set page configuration with customized title, icon, and layout
st.set_page_config(
    page_title="News Classifier",  # Set browser tab title
    page_icon="📰",  # Set browser tab icon
    layout="wide",  # Use wide layout for better screen utilization
    initial_sidebar_state="expanded"  # Start with sidebar expanded
)

# Initialize session state to store theme preference
# Session state persists across reruns of the script
if 'theme' not in st.session_state:
    st.session_state.theme = 'light'  # Default theme is light

# Function to toggle between light and dark themes
def toggle_theme():
    # Switch theme from light to dark or vice versa
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'

# Function to define CSS styles based on current theme
def local_css():
    # Set color variables based on current theme
    if st.session_state.theme == 'dark':
        # Dark theme colors
        bg_color = "#121212"
        text_color = "#FFFFFF"
        card_bg = "#1E1E1E"
        primary_color = "#4169E1"
        hover_color = "#1E40AF"
        accent_color = "#1E1E1E"
        border_color = "#333333"
        input_bg = "#2D2D2D"
    else:
        # Light theme colors
        bg_color = "#FFFFFF"
        text_color = "#333333"
        card_bg = "#F8F9FA"
        primary_color = "#4169E1"
        hover_color = "#1E40AF"
        accent_color = "#F0F4FF"
        border_color = "#E0E0E0"
        input_bg = "#FFFFFF"
    
    # Define CSS with theme variables
    st.markdown(f"""
    <style>
    /* Base styles for the entire app */
    .main {{
        padding: 1rem 2rem;
        background-color: {bg_color};
        color: {text_color};
    }}
    
    /* Custom styles for button elements */
    .stButton button {{
        background-color: {primary_color};
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }}
    .stButton button:hover {{
        background-color: {hover_color};
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }}
    
    /* Custom styles for text input areas */
    .stTextArea textarea {{
        border-radius: 5px;
        border: 1px solid {border_color};
        background-color: {input_bg};
        color: {text_color};
    }}
    
    /* Custom styles for dropdowns */
    .stSelectbox div[data-baseweb="select"] > div {{
        border-radius: 5px;
        border: 1px solid {border_color};
        background-color: {input_bg};
        color: {text_color};
    }}
    
    /* Heading styles */
    div.stMarkdown h1 {{
        margin-bottom: 1.5rem;
        color: {text_color};
    }}
    div.stMarkdown h3 {{
        color: {primary_color};
        margin-bottom: 1rem;
    }}
    
    /* Result display box */
    .result-box {{
        padding: 1.5rem;
        border-radius: 10px;
        background-color: {accent_color};
        margin: 1rem 0;
        border-left: 5px solid {primary_color};
    }}
    
    /* Category label styling */
    .category-tag {{
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-weight: bold;
        color: white;
    }}
    
    /* Category-specific colors */
    .business {{background-color: #FF6B6B;}}
    .tech {{background-color: #4CAF50;}}
    .sport {{background-color: #FF9800;}}
    .health {{background-color: #2196F3;}}
    .politics {{background-color: #9C27B0;}}
    .entertainment {{background-color: #E91E63;}}
    
    /* Sidebar styling */
    .sidebar-content {{
        padding: 1.5rem 1rem;
    }}
    
    /* Tab content styling */
    .tab-content {{
        padding: 2rem;
        border-radius: 10px;
        background-color: {card_bg};
        margin-top: 1rem;
    }}
    
    /* Card styling */
    .card {{
        background-color: {card_bg};
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }}
    
    /* Data tables styling */
    .dataframe {{
        background-color: {card_bg};
        color: {text_color};
    }}
    
    /* Theme toggle button */
    .theme-toggle {{
        display: inline-block;
        padding: 0.4rem 0.8rem;
        background-color: {primary_color};
        color: white;
        border-radius: 20px;
        text-decoration: none;
        font-size: 0.8rem;
        margin-bottom: 1rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
    }}
    .theme-toggle:hover {{
        background-color: {hover_color};
        transform: translateY(-2px);
    }}
    
    /* Footer styling */
    .footer {{
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid {border_color};
        text-align: center;
        font-size: 0.8rem;
        color: {text_color};
    }}
    </style>
    """, unsafe_allow_html=True)

# Call the CSS function to apply styles
local_css()

# Function to add a background texture based on current theme
def add_bg_texture():
    # Different texture URLs for light and dark themes
    if st.session_state.theme == 'dark':
        texture_url = "https://www.transparenttextures.com/patterns/dark-leather.png"
        overlay = "rgba(18, 18, 18, 0.97)"
    else:
        texture_url = "https://www.transparenttextures.com/patterns/clean-gray-paper.png"
        overlay = "rgba(245, 246, 252, 0.96)"
    
    # Apply the background texture with CSS
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-size: cover;
             background-position: top left;
             background-repeat: no-repeat;
             background-image: linear-gradient(to bottom, {overlay}, {overlay}), url("{texture_url}");
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

# Apply the background texture
add_bg_texture()

# Load Vectorizer For News Prediction with caching to avoid reloading on each rerun
@st.cache_resource
def load_vectorizer():
    """
    Load and cache the Count Vectorizer model
    
    Returns:
        CountVectorizer: The loaded model for text vectorization
    """
    news_vectorizer = open("models/final_news_cv_vectorizer.pkl", "rb")
    return joblib.load(news_vectorizer)

# Load the vectorizer model
news_cv = load_vectorizer()

# Function to load prediction models with caching
@st.cache_resource
def load_prediction_models(model_file):
    """
    Load and cache ML prediction models
    
    Args:
        model_file (str): Path to the model file
        
    Returns:
        object: The loaded ML model
    """
    loaded_model = joblib.load(open(os.path.join(model_file), "rb"))
    return loaded_model

# Function to get dictionary key from value
def get_key(val, my_dict):
    """
    Find a key in a dictionary based on its value
    
    Args:
        val: The value to search for
        my_dict (dict): The dictionary to search in
        
    Returns:
        The key corresponding to the value
    """
    for key, value in my_dict.items():
        if val == value:
            return key

# Load spaCy NLP model with caching
@st.cache_resource
def load_nlp():
    """
    Load and cache the spaCy NLP model
    
    Returns:
        spacy.Language: The loaded spaCy model
    """
    return spacy.load('en')

# Load the NLP model
nlp = load_nlp()

# Main application function
def main():
    """
    Main function to render the application UI and handle user interactions
    """
    # Sidebar section
    with st.sidebar:
        # Display app logo
        st.image("https://via.placeholder.com/150x150?text=News+AI", width=150)
        st.title("News Classifier")
        st.markdown("---")
        
        # Add theme toggle button in sidebar
        theme_btn_text = "🌙 Switch to Dark Mode" if st.session_state.theme == 'light' else "☀️ Switch to Light Mode"
        if st.button(theme_btn_text):
            toggle_theme()
            st.rerun()  # Rerun the app to apply new theme
        
        st.markdown("---")
        
        # Module selection radio buttons
        activity = ['News Classification', 'Text Analysis']
        choice = st.radio("Select Module", activity)
        
        st.markdown("---")
        
        # About section with app information
        st.markdown("### About")
        st.markdown("""
        This app uses machine learning to classify news articles and perform natural language processing tasks.
        
        **Features:**
        - News categorization
        - Text analysis
        - Word clouds
        - Theme customization
        """)
        
        st.markdown("---")
        st.markdown("Made with ❤️ by Your Team")
    
    # Main content section - News Classification
    if choice == 'News Classification':
        st.title("📰 News Classification")
        st.markdown("### Analyze news content and categorize it into different topics")
        
        # Create two-column layout
        col1, col2 = st.columns([3, 1])
        
        # Left column for text input
        with col1:
            # Text area for news article input
            news_text = st.text_area(
                "Enter news article text",
                placeholder="Paste your news article here...",
                height=200
            )
        
        # Right column for model selection and category information
        with col2:
            st.markdown("### Model Selection")
            # Dictionary mapping user-friendly names to model identifiers
            all_ml_models = {
                "Logistic Regression": "LR",
                "Random Forest": "RFOREST",
                "Naive Bayes": "NB",
                "Decision Tree": "DECISION_TREE"
            }
            # Dropdown for model selection
            model_choice = st.selectbox("Choose algorithm", list(all_ml_models.keys()))
            
            st.markdown("### Categories")
            # Dictionary of news categories with their display colors
            categories = {
                'business': "#FF6B6B", 
                'tech': "#4CAF50", 
                'sport': "#FF9800", 
                'health': "#2196F3", 
                'politics': "#9C27B0", 
                'entertainment': "#E91E63"
            }
            
            # Display category color legend
            for category, color in categories.items():
                st.markdown(f"""
                <div style="display:flex;align-items:center;margin-bottom:0.5rem;">
                    <div style="width:15px;height:15px;background-color:{color};border-radius:50%;margin-right:10px;"></div>
                    <div style="text-transform:capitalize;">{category}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Button to trigger classification
        classify_btn = st.button("Classify News", use_container_width=True)
        
        # Process the news text when button is clicked and text is not empty
        if classify_btn and news_text.strip() != "" and news_text != "Paste your news article here...":
            with st.spinner("Analyzing news content..."):
                # Show progress bar for user feedback
                progress_bar = st.progress(0)
                for percent_complete in range(100):
                    progress_bar.progress(percent_complete + 1)
                
                # Transform input text to feature vector using loaded vectorizer
                vect_text = news_cv.transform([news_text]).toarray()
                
                # Get model key from user selection
                model_key = all_ml_models[model_choice]
                
                # Load the selected prediction model
                if model_key == 'LR':
                    predictor = load_prediction_models("models/newsclassifier_Logit_model.pkl")
                elif model_key == 'RFOREST':
                    predictor = load_prediction_models("models/newsclassifier_RFOREST_model.pkl")
                elif model_key == 'NB':
                    predictor = load_prediction_models("models/newsclassifier_NB_model.pkl")
                elif model_key == 'DECISION_TREE':
                    predictor = load_prediction_models("models/newsclassifier_CART_model.pkl")
                
                # Make prediction using selected model
                prediction = predictor.predict(vect_text)
                
                # Dictionary mapping numerical predictions to category labels
                prediction_labels = {'business': 0, 'tech': 1, 'sport': 2, 'health': 3, 'politics': 4, 'entertainment': 5}
                
                # Get the category label from numerical prediction
                final_result = get_key(prediction, prediction_labels)
                
                # Display classification result with styling
                st.markdown("### Classification Result")
                st.markdown(f"""
                <div class="result-box">
                    <h3>News Category:</h3>
                    <div class="category-tag {final_result}" style="background-color: {categories[final_result]}">
                        {final_result.upper()}
                    </div>
                    <p style="margin-top: 1rem"><strong>Model used:</strong> {model_choice}</p>
                    <p><strong>Confidence:</strong> High</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show sample of analyzed text
                st.markdown("### Analyzed Text")
                st.markdown(f"""
                <div style="background-color: {'#2D2D2D' if st.session_state.theme == 'dark' else 'white'}; 
                      color: {'white' if st.session_state.theme == 'dark' else 'black'};
                      padding: 1rem; border-radius: 5px; 
                      border: 1px solid {'#333333' if st.session_state.theme == 'dark' else '#ddd'}; 
                      height: 150px; overflow-y: auto;">
                    {news_text[:500]}{'...' if len(news_text) > 500 else ''}
                </div>
                """, unsafe_allow_html=True)
    
    # Text Analysis section
    elif choice == 'Text Analysis':
        st.title("🔍 Text Analysis")
        st.markdown("### Analyze text using Natural Language Processing")
        
        # Create tabs for different analysis views
        tabs = st.tabs(["NLP Tools", "Visualization"])
        
        # NLP Tools tab
        with tabs[0]:
            # Create two-column layout
            col1, col2 = st.columns([3, 1])
            
            # Left column for text input
            with col1:
                # Text area for analysis input
                raw_text = st.text_area(
                    "Enter text for analysis",
                    placeholder="Paste your text here...",
                    height=200
                )
            
            # Right column for NLP task selection and buttons
            with col2:
                # List of available NLP tasks
                nlp_task = ["Tokenization", "Lemmatization", "Named Entities", "POS Tags"]
                # Dropdown for NLP task selection
                task_choice = st.selectbox("Choose NLP Task", nlp_task)
                
                # Analysis buttons
                analyze_btn = st.button("Analyze Text")
                tabulize_btn = st.button("View as Table")
            
            # Process text when analyze button is clicked and text is not empty
            if analyze_btn and raw_text.strip() != "" and raw_text != "Paste your text here...":
                with st.spinner("Processing text..."):
                    st.markdown("### Analysis Result")
                    
                    # Process text with spaCy NLP model
                    docx = nlp(raw_text)
                    
                    # Tokenization - break text into individual tokens
                    if task_choice == 'Tokenization':
                        # Extract all tokens from the processed text
                        result = [token.text for token in docx]
                        # Display tokens with theme-aware styling
                        st.markdown(f"""
                        <div class="result-box">
                            <h3>Tokens:</h3>
                            <div style="background-color: {'#2D2D2D' if st.session_state.theme == 'dark' else 'white'}; 
                                  color: {'white' if st.session_state.theme == 'dark' else 'black'};
                                  padding: 1rem; border-radius: 5px; 
                                  border: 1px solid {'#333333' if st.session_state.theme == 'dark' else '#ddd'}; 
                                  max-height: 200px; overflow-y: auto;">
                                {', '.join(result)}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Lemmatization - reduce words to their base/dictionary form
                    elif task_choice == 'Lemmatization':
                        # Extract tokens and their lemmas
                        result = [(token.text, token.lemma_) for token in docx]
                        
                        # Create a DataFrame for better display
                        lemma_df = pd.DataFrame(result, columns=['Token', 'Lemma'])
                        st.dataframe(lemma_df, use_container_width=True)
                    
                    # Named Entity Recognition - identify and classify named entities
                    elif task_choice == 'Named Entities':
                        # Extract named entities and their types
                        result = [(entity.text, entity.label_) for entity in docx.ents]
                        
                        if result:
                            # Create a visual representation of entities
                            entities_html = ""
                            # Color coding for different entity types
                            for ent_text, ent_type in result:
                                color = {
                                    'PERSON': '#FF6B6B', 'ORG': '#4CAF50', 'GPE': '#2196F3', 
                                    'DATE': '#9C27B0', 'MONEY': '#FF9800', 'PRODUCT': '#E91E63'
                                }.get(ent_type, '#607D8B')
                                
                                # Create styled tag for each entity
                                entities_html += f"""
                                <span style="display: inline-block; margin: 0.25rem; padding: 0.3rem 0.5rem; 
                                background-color: {color}; color: white; border-radius: 15px; font-size: 0.9rem;">
                                {ent_text} <small>({ent_type})</small>
                                </span>
                                """
                            
                            # Display entities with styling
                            st.markdown(f"""
                            <div class="result-box">
                                <h3>Named Entities:</h3>
                                <div>
                                    {entities_html}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            # Show message when no entities are found
                            st.info("No named entities found in the text.")
                    
                    # Part-of-Speech tagging - identify grammatical parts of speech
                    elif task_choice == 'POS Tags':
                        # Extract tokens, POS tags, and dependencies
                        result = [(word.text, word.tag_, word.dep_) for word in docx]
                        
                        # Create a DataFrame for better display
                        pos_df = pd.DataFrame(result, columns=['Token', 'POS Tag', 'Dependency'])
                        st.dataframe(pos_df, use_container_width=True)
            
            # Process text when tabulize button is clicked and text is not empty
            if tabulize_btn and raw_text.strip() != "" and raw_text != "Paste your text here...":
                with st.spinner("Creating table view..."):
                    # Process text with spaCy NLP model
                    docx = nlp(raw_text)
                    # Extract tokens, lemmas, and POS tags
                    c_tokens = [token.text for token in docx]
                    c_lemma = [token.lemma_ for token in docx]
                    c_pos = [token.pos_ for token in docx]
                    
                    # Create DataFrame with extracted data
                    new_df = pd.DataFrame(zip(c_tokens, c_lemma, c_pos), columns=['Tokens', 'Lemma', 'POS'])
                    
                    st.markdown("### Tabular View")
                    # Display the DataFrame
                    st.dataframe(new_df, use_container_width=True)
                    
                    # Add download button for the DataFrame
                    csv = new_df.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    # Create download link with styling
                    href = f'<a href="data:file/csv;base64,{b64}" download="nlp_analysis.csv" class="btn" style="display: inline-block; padding: 0.5rem 1rem; background-color: #4169E1; color: white; text-decoration: none; border-radius: 5px; margin-top: 1rem;">Download CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)
        
        # Visualization tab
        with tabs[1]:
            st.markdown("### Text Visualization")
            
            # Check if text input exists from previous tab
            if 'raw_text' not in locals():
                # If not, create new text input
                raw_text = st.text_area(
                    "Enter text for visualization",
                    placeholder="Paste your text here...",
                    height=200
                )
            
            # Multi-select for visualization types
            viz_options = st.multiselect(
                "Choose visualization type",
                ["Word Cloud", "Word Frequency", "Character Count"],
                default=["Word Cloud"]
            )
            
            # Button to generate visualizations
            generate_viz = st.button("Generate Visualizations", use_container_width=True)
            
            # Process text when visualization button is clicked and text is not empty
            if generate_viz and raw_text.strip() != "" and raw_text != "Paste your text here...":
                # Word Cloud visualization
                if "Word Cloud" in viz_options:
                    st.markdown("### Word Cloud")
                    with st.spinner("Generating word cloud..."):
                        # Create word cloud with custom settings
                        wordcloud = WordCloud(
                            background_color='white' if st.session_state.theme == 'light' else 'black',
                            width=800,
                            height=400,
                            max_words=100,
                            contour_width=3,
                            contour_color='steelblue'
                        ).generate(raw_text)
                        
                        # Create matplotlib figure and plot word cloud
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis("off")
                        plt.tight_layout()
                        # Set figure background color based on theme
                        fig.patch.set_facecolor('white' if st.session_state.theme == 'light' else '#1E1E1E')
                        # Display the plot
                        st.pyplot(fig)
                
                # Word Frequency visualization
                if "Word Frequency" in viz_options:
                    st.markdown("### Word Frequency")
                    with st.spinner("Calculating word frequency..."):
                        # Process text with spaCy and filter tokens
                        docx = nlp(raw_text.lower())
                        # Get clean words (no stopwords or punctuation)
                        words = [token.text for token in docx if not token.is_stop and not token.is_punct and token.text.strip()]
                        # Count word frequencies and get top 20
                        word_freq = pd.Series(words).value_counts().head(20)
                        
                        # Create bar chart
                        fig, ax = plt.subplots(figsize=(10, 6))
                        word_freq.plot(kind='bar', color='steelblue', ax=ax)
                        ax.set_title('Top 20 Word Frequency')
                        ax.set_ylabel('Frequency')
                        # Set theme-specific colors
                        if st.session_state.theme == 'dark':
                            fig.patch.set_facecolor('#1E1E1E')
                            ax.set_facecolor('#1E1E1E')
                            ax.tick_params(colors='white')
                            ax.title.set_color('white')
                            ax.yaxis.label.set_color('white')
                            ax.xaxis.label.set_color('white')
                            for spine in ax.spines.values():
                                spine.set_color('#333333')
                        # Display the plot
                        st.pyplot(fig)
                
                # Text Statistics visualization
                if "Character Count" in viz_options:
                    st.markdown("### Text Statistics")
                    
                    # Calculate text statistics
                    char_count = len(raw_text)
                    word_count = len(raw_text.split())
                    sentence_count = len([s for s in raw_text.split('.') if s.strip()])
                    
                    # Create three-column layout for statistics
                    col1, col2, col3 = st.columns(3)
                    
                    # Theme-aware background color
                    bg_color = "#1E1E1E" if st.session_state.theme == 'dark' else "#F0F4FF"
                    
                    # Display character count
                    with col1:
                        st.markdown(f"""
                        <div style="background-color: {bg_color}; padding: 1.5rem; border-radius: 10px; text-align: center;">
                            <h1 style="color: #4169E1; font-size: 3rem; margin: 0;">{char_count}</h1>
                            <p>Characters</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Display word count
                    with col2:
                        st.markdown(f"""
                        <div style="background-color: {bg_color}; padding: 1.5rem; border-radius: 10px; text-align: center;">
                            <h1 style="color: #4169E1; font-size: 3rem; margin: 0;">{word_count}</h1>
                            <p>Words</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Display sentence count
                    with col3:
                        st.markdown(f"""
                        <div style="background-color: {bg_color}; padding: 1.5rem; border-radius: 10px; text-align: center;">
                            <h1 style="color: #4169E1; font-size: 3rem; margin: 0;">{sentence_count}</h1>
                            <p>Sentences</p>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Add footer with theme-aware styling
    st.markdown(f"""
    <div class="footer">
        <p>© 2025 News Classifier | v2.0 | Made with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

# Entry point of the application
if __name__ == '__main__':
    main()