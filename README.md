# IBM Watson Natural Language Understanding Analyzer

A minimalist text analysis tool powered by IBM Watson NLU, built with Streamlit. This application provides advanced natural language processing capabilities to extract metadata from text such as keywords, entities, categories, and relationships.

![IBM Watson NLU Analyzer Screenshot](https://raw.githubusercontent.com/filippodanesi/nlu-text-analyzer/main/screenshot.png)

## Features

- **Text Analysis**: Analyze raw text or text files using IBM Watson NLU
- **Multiple Analysis Types**:
  - Keywords extraction with relevance scores
  - Entity recognition (people, organizations, locations)
  - Concept identification
  - Category classification
  - Relationship extraction
- **Target Keywords**: Highlight specific keywords or topics of interest in the results
- **Clean Interface**: Minimalist design with a focus on readability and usability

## Live Demo

Try the application live on [Streamlit Cloud](https://ibm-nlu-analyzer.streamlit.app).

## Deployment on Streamlit Cloud

### 1. Fork this repository

Click the "Fork" button at the top right of this repository to create your own copy.

### 2. Deploy to Streamlit Cloud

1. Sign in to [Streamlit Cloud](https://streamlit.io/cloud)
2. Click "New app"
3. Select your forked repository
4. Select the main branch and app.py file
5. Click "Deploy"

### 3. Add your IBM Watson credentials as secrets

1. Go to your deployed app's settings page
2. Navigate to the "Secrets" section
3. Add your IBM Watson NLU credentials in TOML format:

```toml
[ibm_watson]
api_key = "your_api_key_here"
url = "https://api.your-region.natural-language-understanding.watson.cloud.ibm.com/instances/your-instance-id"
```

4. Save the secrets and restart your app

## Local Development

### Prerequisites

- Python 3.8+
- IBM Watson NLU API credentials

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ibm-nlu-analyzer.git
cd ibm-nlu-analyzer
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.streamlit/secrets.toml` file with your IBM Watson NLU credentials:
```toml
[ibm_watson]
api_key = "your_api_key_here"
url = "https://api.your-region.natural-language-understanding.watson.cloud.ibm.com/instances/your-instance-id"
```

4. Run the application:
```bash
streamlit run app.py
```

## Getting IBM Watson NLU Credentials

1. Create an IBM Cloud account at [cloud.ibm.com](https://cloud.ibm.com/registration)
2. Create a Natural Language Understanding service instance
3. Go to "Service Credentials" and create new credentials
4. Copy the API key and URL for use in the application

## Technologies Used

- [Streamlit](https://streamlit.io) - Web application framework
- [IBM Watson NLU](https://www.ibm.com/cloud/watson-natural-language-understanding) - Natural language processing API
- [Pandas](https://pandas.pydata.org/) - Data manipulation library

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- IBM Watson for providing the NLU API
- Streamlit for the excellent web app framework
