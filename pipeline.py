import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import re
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# ==================== 1. Streamlit Configuration ====================
st.set_page_config(
    page_title="Sentiment Analysis Pipeline",
    page_icon="",
    layout="wide"
)

# Configuration
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")  # or set directly
MODEL_NAME = "gpt-4o-mini"  # Change to "gpt-4o" for better accuracy
THRESHOLD = 0.1
CONFIDENCE_THRESHOLD = 0.6

# ==================== 2. Model Architecture ====================
class PersianVocabulary:
    def __init__(self):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<SOS>', 3: '<EOS>'}

    def build_from_dict(self, word2idx):
        self.word2idx = word2idx
        self.idx2word = {v: k for k, v in word2idx.items()}

    def tokenize_persian(self, text):
        if not isinstance(text, str):
            return []
        text = re.sub(r'[^\u0600-\u06FF\s\.\,\!\?]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().split()

    def numericalize(self, text, max_length=100):
        words = self.tokenize_persian(text)
        if len(words) > max_length - 2:
            words = words[:max_length - 2]
        words = ['<SOS>'] + words + ['<EOS>']
        indices = [self.word2idx.get(w, self.word2idx['<UNK>']) for w in words]
        if len(indices) < max_length:
            indices += [self.word2idx['<PAD>']] * (max_length - len(indices))
        else:
            indices = indices[:max_length]
            indices[-1] = self.word2idx['<EOS>']
        return indices

    def __len__(self):
        return len(self.word2idx)


class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256,
                 output_dim=3, n_layers=2, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers,
                            bidirectional=True, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, text):
        embedded = self.embedding(text)
        lstm_output, _ = self.lstm(embedded)
        attn = torch.softmax(self.attention(lstm_output), dim=1)
        context = torch.sum(attn * lstm_output, dim=1)
        out = self.dropout(context)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        return self.fc2(out)

# ==================== 3. Load Model ====================
@st.cache_resource
def load_model_and_vocab(pth_path='best_sentiment_model.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        checkpoint = torch.load(pth_path, map_location=device)
        vocab = PersianVocabulary()
        if 'vocab_word2idx' in checkpoint:
            vocab.build_from_dict(checkpoint['vocab_word2idx'])
        elif 'vocab' in checkpoint:
            vocab = checkpoint['vocab']
        max_length = checkpoint.get('max_length', 100)
        model = SentimentClassifier(
            vocab_size=checkpoint.get('vocab_size', len(vocab)),
            embedding_dim=64, hidden_dim=128, output_dim=3, n_layers=2, dropout=0.1
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device).eval()
        st.success(f"Model loaded | Vocab: {len(vocab)} words")
        return model, vocab, max_length, device, checkpoint
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, 100, device, None

# ==================== 4. LangChain OpenAI Integration ====================
@st.cache_resource
def get_llm_chain():
    """Build LangChain chain: prompt | LLM | JSON parser"""
    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=OPENAI_API_KEY,
        temperature=0.1,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a sentiment analysis assistant for Persian text.
Analyze if the text expresses BAD sentiment (dissatisfaction, complaint, negative experience).

Respond ONLY with valid JSON:
{{
    "is_bad": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation in English"
}}

Consider:
- Negative words (بد, ضعیف, خراب, etc.)
- Complaints about quality/service
- Expressions of disappointment
- Recommendations against purchase"""),
        ("human", "Analyze this Persian text: {text}")
    ])

    parser = JsonOutputParser()
    return prompt | llm | parser


def check_with_openai(text):
    """Check if text has bad sentiment using LangChain + OpenAI"""
    try:
        chain = get_llm_chain()
        result = chain.invoke({"text": text})
        return result
    except Exception as e:
        return {"is_bad": False, "confidence": 0.0, "reasoning": f"Error: {str(e)}"}

# ==================== 5. Predictor ====================
class SentimentPredictor:
    def __init__(self, model, vocab, max_length, device):
        self.model = model
        self.vocab = vocab
        self.max_length = max_length
        self.device = device
        self.label_map = {0: 1, 1: 2, 2: 3}

    def predict(self, text, return_probabilities=False):
        try:
            indices = self.vocab.numericalize(text, self.max_length)
            tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(tensor)probs = torch.softmax(outputs, dim=1)
                _, pred = torch.max(outputs, dim=1)
            suggestion = self.label_map[pred.item()]
            p = probs[0].cpu().numpy()
            result = {
                'suggestion': suggestion,
                'probabilities': {'good': float(p[0]), 'neutral': float(p[1]), 'bad': float(p[2])},
                'is_bad_ml': p[2] > THRESHOLD,
                'confidence': float(p[2])
            }
            return (suggestion, result) if return_probabilities else result
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None

# ==================== 6. Visualization ====================
def create_pipeline_diagram(step_status):
    nodes = {
        'start':     {'label': 'Start','x': 0, 'y': 2, 'color': '#4CAF50'},
        'ml':        {'label': 'ML Model',        'x': 2, 'y': 2, 'color': '#2196F3'},
        'threshold': {'label': f'Check >{THRESHOLD:.0%}', 'x': 4, 'y': 2, 'color': '#FF9800'},
        'openai':    {'label': 'OpenAI Check',    'x': 6, 'y': 2, 'color': '#9C27B0'},
        'decision':  {'label': 'Final Decision',  'x': 8, 'y': 2, 'color': '#607D8B'},
        'ok':        {'label': '✅ OK',            'x': 8, 'y': 1, 'color': '#4CAF50'},
        'bad':       {'label': '⚠️ BAD',           'x': 8, 'y': 3, 'color': '#F44336'},
    }
    fig = go.Figure()
    for nid, n in nodes.items():
        active = nid in step_status.get('active_nodes', [])
        fig.add_trace(go.Scatter(
            x=[n['x']], y=[n['y']], mode='markers+text',
            marker=dict(size=40, color=n['color'] if active else '#E0E0E0',
                        line=dict(width=3, color='white')),
            text=[n['label']], textposition="middle center",
            textfont=dict(size=11, color='white'), hoverinfo='text', showlegend=False
        ))
    edges = [('start','ml'),('ml','threshold'),('threshold','openai'),
             ('threshold','decision'),('openai','decision'),('decision','ok'),('decision','bad')]
    for s, e in edges:
        sn, en = nodes[s], nodes[e]
        active = (s, e) in step_status.get('active_edges', [])
        fig.add_trace(go.Scatter(
            x=[sn['x'], en['x']], y=[sn['y'], en['y']], mode='lines',
            line=dict(color='green' if active else 'gray', width=3 if active else 1),
            hoverinfo='none', showlegend=False
        ))
    fig.update_layout(
        title=dict(text="Sentiment Analysis Pipeline Flow", x=0.5, font=dict(size=20)),
        showlegend=False, plot_bgcolor='white', height=400,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 9]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 4]),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig


def create_sentiment_gauge(probabilities, current_suggestion):
    fig = make_subplots(rows=1, cols=3, subplot_titles=('Good', 'Neutral', 'Bad'),
                        specs=[[{'type': 'indicator'}] * 3])
    for i, (sentiment, color, label_id) in enumerate([
        ('good', '#4CAF50', 1), ('neutral', '#FF9800', 2), ('bad', '#F44336', 3)
    ]):
        fig.add_trace(go.Indicator(
            mode="gauge+number", value=probabilities[sentiment] * 100,
            number=dict(suffix="%", font=dict(size=20)),
            gauge={'axis': {'range': [0, 100]}, 'bar': {'color': color, 'thickness': 0.8},
                   'bgcolor': 'white', 'borderwidth': 2,
                   'bordercolor': color if current_suggestion == label_id else 'gray',
                   'steps': [{'range': [0, 100], 'color': '#F5F5F5'}],
                   'threshold': {'line': {'color': 'black', 'width': 4},'thickness': 0.8, 'value': THRESHOLD * 100}},
        ), row=1, col=i + 1)
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20),
                      title_text="Sentiment Probability Distribution", title_x=0.5)
    return fig


def create_timeline(steps):
    if not steps:
        return go.Figure()
    df = pd.DataFrame(steps)
    fig = go.Figure(data=[go.Bar(
        x=df['step'], y=df['duration'], text=df['status'],
        marker_color=df['color'], textposition='auto',
        textfont=dict(color='white', size=12)
    )])
    fig.update_layout(title="Processing Timeline", xaxis_title="Step",
                      yaxis_title="Duration (seconds)", showlegend=False,
                      height=250, plot_bgcolor='white')
    return fig

# ==================== 7. Analysis Pipeline ====================
def run_analysis_pipeline(text, predictor):
    steps_data = []
    status = {'active_nodes': ['start'], 'active_edges': [], 'current_step': 'Starting...'}

    # Step 1: ML Model
    status['active_nodes'].append('ml')
    status['active_edges'].append(('start', 'ml'))
    with st.spinner("Analyzing with ML model..."):
        t0 = time.time()
        ml_result = predictor.predict(text)
        steps_data.append({'step': 'ML Model', 'duration': round(time.time() - t0, 3),
                            'status': 'Completed', 'color': '#2196F3'})if not ml_result:
        return None

    # Step 2: Threshold check
    status['active_nodes'].append('threshold')
    status['active_edges'].append(('ml', 'threshold'))

    openai_result = None
    if ml_result['is_bad_ml']:
        # Step 3: OpenAI verification
        status['active_nodes'].append('openai')
        status['active_edges'].append(('threshold', 'openai'))
        with st.spinner("Verifying with OpenAI..."):
            t0 = time.time()
            openai_result = check_with_openai(text)
            steps_data.append({'step': 'OpenAI Check', 'duration': round(time.time() - t0, 3),
                                'status': 'Completed', 'color': '#9C27B0'})

        status['active_nodes'].append('decision')
        status['active_edges'].append(('openai', 'decision'))if openai_result.get('is_bad', False) and openai_result.get('confidence', 0) > CONFIDENCE_THRESHOLD:
            final_decision = "BAD"
            status['active_nodes'].append('bad')
            status['active_edges'].append(('decision', 'bad'))
        else:
            final_decision = "OK"
            status['active_nodes'].append('ok')
            status['active_edges'].append(('decision', 'ok'))
    else:
        status['active_nodes'] += ['decision', 'ok']
        status['active_edges'] += [('threshold', 'decision'), ('decision', 'ok')]
        final_decision = "OK"

    status['current_step'] = f"Analysis complete: {final_decision}"
    return {
        'text': text, 'ml_result': ml_result, 'ollama_result': openai_result,
        'final_decision': final_decision, 'steps_data': steps_data,
        'pipeline_status': status
    }

# ==================== 8. Streamlit UI ====================
def main():
    if 'analysis_history' not in st.session_state:
        st.session_state['analysis_history'] = []

    st.title("Sentiment Analysis Pipeline")
    st.markdown("""
    Two-step pipeline:
    1. **ML Model**: Predicts sentiment (1=Good, 2=Neutral, 3=Bad)
    2. **OpenAI Verification**: Double-checks bad sentiments using GPT
    """)

    with st.sidebar:
        st.header("Configuration")

        # API Key input if not set
        if not OPENAI_API_KEY:
            api_key_input = st.text_input("OpenAI API Key", type="password")
            if api_key_input:
                import os; os.environ["OPENAI_API_KEY"] = api_key_input

        global THRESHOLD, CONFIDENCE_THRESHOLD
        THRESHOLD = st.slider("Bad Sentiment Threshold", 0.0, 1.0, 0.1, 0.01)
        CONFIDENCE_THRESHOLD = st.slider("OpenAI Confidence Threshold", 0.0, 1.0, 0.6, 0.05)

        st.divider()
        st.header("Statistics")
        history = st.session_state['analysis_history']
        if history:
            total = len(history)
            bad_count = sum(1 for r in history if r['final_decision'] == "BAD")
            c1, c2 = st.columns(2)
            c1.metric("Total", total)
            c2.metric("Bad", bad_count)
            st.metric("Bad Rate", f"{bad_count/total*100:.1f}%")else:
            st.info("No analyses yet")

        if st.button("Clear History"):
            st.session_state['analysis_history'] = []
            st.rerun()

        # OpenAI status
        st.divider()
        key = OPENAI_API_KEY or st.session_state.get('api_key', '')
        if key:
            st.success(f"✅ OpenAI configured | Model: {MODEL_NAME}")
        else:
            st.error("❌ OpenAI API key not set")
        st.caption(f"Bad threshold: >{THRESHOLD:.0%}")
        st.caption(f"Confidence threshold: >{CONFIDENCE_THRESHOLD:.0%}")

    model, vocab, max_length, device, checkpoint = load_model_and_vocab()
    if model is None:
        st.error("Could not load model. Place `best_sentiment_model.pth` in the working directory.")
        return

    predictor = SentimentPredictor(model, vocab, max_length, device)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("Enter Persian Text")
        example_texts = {
            "Good": "این محصول واقعا عالی بود. کیفیت فوق العاده",
            "Neutral": "محصول متوسطی بود، نه خوب نه بد",
            "Bad": "بدترین خرابیم بود، پولم را هدر دادم"
        }
        selected = None
        cols = st.columns(3)
        for i, (label, txt) in enumerate(example_texts.items()):
            if cols[i].button(f"{label} Example", use_container_width=True):
                selected = txt

        text_input = st.text_area("Enter text:", value=selected or "", height=100,
                                   placeholder="مثال: محصول بدی بود...")
        analyze_clicked = st.button("Analyze Sentiment", type="primary",disabled=not text_input.strip(), use_container_width=True)

    with col2:
        st.subheader("Status")
        current = st.session_state.get('current_status', 'Ready')
        st.markdown(f"**{current}**")

    if analyze_clicked and text_input.strip():
        pipeline_ph = st.empty()
        pipeline_ph.plotly_chart(create_pipeline_diagram({'active_nodes': ['start'], 'active_edges': []}),
                                  use_container_width=True)

        result = run_analysis_pipeline(text_input, predictor)
        if result:
            st.session_state['current_status'] = result['pipeline_status']['current_step']
            st.session_state['analysis_history'].append(result)

            pipeline_ph.plotly_chart(create_pipeline_diagram(result['pipeline_status']),
                                      use_container_width=True)

            st.subheader("Results")
            c1, c2, c3, c4 = st.columns(4)
            s = result['ml_result']['suggestion']
            c1.metric("ML Prediction", f"{'Good' if s==1 else 'Neutral' if s==2 else 'Bad'} ({s})")
            c2.metric("Bad Probability", f"{result['ml_result']['probabilities']['bad']:.1%}")
            if result['ollama_result']:
                c3.metric("OpenAI Confidence", f"{result['ollama_result'].get('confidence', 0):.1%}")
            else:
                c3.metric("OpenAI Check", "Skipped")
            decision = result['final_decision']
            color = "red" if decision == "BAD" else "green"
            c4.markdown(f"""<div style="background:{color}20;padding:10px;border-radius:5px;
                border-left:5px solid {color}">
                <h3 style="margin:0;color:{color}">Final Decision</h3>
                <h1 style="margin:0">{'⚠️' if decision=='BAD' else '✅'} {decision}</h1>
                </div>""", unsafe_allow_html=True)

            tab1, tab2, tab3 = st.tabs(["Probabilities", "OpenAI Details", "Timeline"])
            with tab1:
                st.plotly_chart(create_sentiment_gauge(result['ml_result']['probabilities'],
                                                        result['ml_result']['suggestion']),
                                use_container_width=True)
            with tab2:
                if result['ollama_result']:
                    st.json(result['ollama_result'])reasoning = result['ollama_result'].get('reasoning', '')
                    st.write(f"**Reasoning**: {reasoning}")
                else:
                    st.info("OpenAI check was skipped (ML confidence below threshold).")
            with tab3:
                if result['steps_data']:
                    st.plotly_chart(create_timeline(result['steps_data']), use_container_width=True)

            with st.expander("History (Last 10)"):
                rows = [{'#': i+1,
                         'Text': r['text'][:50]+'...' if len(r['text'])>50 else r['text'],
                         'ML': f"Suggestion {r['ml_result']['suggestion']}",
                         'Bad Prob': f"{r['ml_result']['probabilities']['bad']:.1%}",
                         'Decision': r['final_decision']}
                        for i, r in enumerate(st.session_state['analysis_history'][-10:])]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if checkpoint:
        with st.sidebar:
            st.divider()
            st.header("Model Info")
            st.caption(f"Vocab: {len(vocab)} | Max length: {max_length}")
            if 'val_accuracy' in checkpoint:
                st.caption(f"Val accuracy: {checkpoint['val_accuracy']:.2%}")

if __name__ == "__main__":
    main()
