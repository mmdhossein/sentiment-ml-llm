# Sentiment-ml-llm pipeline

A hybrid Persian text sentiment analysis pipeline combining a PyTorch LSTM model with OpenAI's GPT-4o-mini via LangChain for two-stage verification.

---

## Architecture

Input Text → PyTorch LSTM Model → Sentiment Score
                                        ↓
                              Score > Threshold?
                                ↙           ↘
                              No            Yes
                               ↓             ↓
                          Return ML     LangChain + GPT-4o-mini
                           Result         Verification
                                              ↓
                                       Final Decision

## How It Works

1. **Stage 1 - ML Model**: A PyTorch LSTM-based `SentimentClassifier` processes the Persian input text and outputs a bad-sentiment probability score.
2. **Stage 2 - LLM Verification**: If the score exceeds a defined threshold, the text is forwarded to `gpt-4o-mini` via a LangChain chain for confirmation.
3. **Final Output**: The LLM returns a structured JSON response (`is_bad`, `confidence`, `reasoning`) which drives the final classification decision.

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| ML Model | PyTorch (LSTM) |
| LLM | OpenAI GPT-4o-mini |
| LLM Framework | LangChain |
| Visualization | Plotly |
| Language | Python 3.10+ |

---

## Project Structure


persian-sentiment-ml-llm/
├── app.py                  # Main Streamlit application
├── model/
│   ├── sentiment_model.py  # PyTorch LSTM model definition
│   ├── vocabulary.py       # PersianVocabulary class
│   └── predictor.py        # SentimentPredictor class
├── llm/
│   └── verifier.py         # LangChain + OpenAI verification chain
├── saved_model/
│   └── model.pt            # Trained model weights
├── requirements.txt
└── .streamlit/
    └── secrets.toml        # API keys (not committed)

---

## Setup & Installation

### 1. Clone the repository

bash
git clone https://github.com/your-username/persian-sentiment-ml-llm.git
cd persian-sentiment-ml-llm

### 2. Create and activate a virtual environment

bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate

### 3. Install dependencies

bash
pip install -r requirements.txt

### 4. Configure API Keys

Create the Streamlit secrets file:

bash
mkdir -p .streamlit
touch .streamlit/secrets.toml

Add your OpenAI API key:

toml
# .streamlit/secrets.toml
OPENAI_API_KEY = "sk-..."

> **Never commit this file.** Add `.streamlit/secrets.toml` to your `.gitignore`.

---

## Running the App

bash
streamlit run app.py

The app will be available at `http://localhost:8501`.

---

## Dependencies

txt
streamlit
torch
langchain
langchain-openai
plotly

Install all at once:

bash
pip install streamlit torch langchain langchain-openai plotly

---

## Pipeline Logic

python
THRESHOLD = 0.7  # Configurable bad-sentiment threshold

# Stage 1
score = ml_model.predict(text)

# Stage 2
if score > THRESHOLD:
    result = llm_chain.invoke({"text": text})
    # result = {"is_bad": true, "confidence": 0.92, "reasoning": "..."}

The LLM is only called when the ML model flags a text as potentially bad, reducing API costs.

---

## LLM Output Schema

The LangChain chain enforces a structured JSON output:

json
{
  "is_bad": true,
  "confidence": 0.91,
  "reasoning": "The text contains explicit negative language targeting..."
}

---

## Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Your OpenAI API key (set in `.streamlit/secrets.toml`) |

---

## .gitignore Recommendation

gitignore
venv/
__pycache__/
*.pyc
.streamlit/secrets.toml
saved_model/*.pt

---

## License

MIT License. See `LICENSE` for details.


Copy this into a `README.md` file at the root of your repo. Update the GitHub URL and adjust file paths if your structure differs.
