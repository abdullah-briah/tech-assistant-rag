# src/app.py
import sys
from pathlib import Path
import streamlit as st
from rag_engine import TechnicalAssistantRAG

# === Add project directory to sys.path ===
src_dir = Path(__file__).parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

# === Load external CSS relative to app.py ===
def load_css(file_name: str):
    """Load CSS file located in the same folder as app.py."""
    css_path = Path(__file__).parent / file_name
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Call the CSS loader
load_css("styles.css")

# === Title and description ===
st.markdown(
    """
    <div style="display:flex; align-items:center; justify-content:center; gap:8px; margin-bottom:8px;">
        <h1 style="margin:0;">🤖 Teknik Asistan</h1>
        <span class="pulse-dot"></span>
    </div>
    <div class="caption">Teknik ve yapay zeka asistanınız</div>
    """,
    unsafe_allow_html=True
)

# === Initialize RAG engine ===
@st.cache_resource
def get_rag_engine():
    return TechnicalAssistantRAG()

rag = get_rag_engine()

# === Initialize chat session ===
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Merhaba! 🌟 Size yardımcı olmak için buradayım."}
    ]

# === Helper functions ===
def append_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})
    if role == "assistant":
        st.rerun()

def render_messages():
    for msg in st.session_state.messages:
        content_html = msg["content"].replace("\n", "<br>")
        st.chat_message(msg["role"]).markdown(content_html, unsafe_allow_html=True)

def display_welcome_buttons():
    st.markdown("### 🧠 Bugün hangi konuda konuşalım?")
    col1, col2 = st.columns(2)

    # Topics
    col1_topics = [
        ("🤖 Yapay Zeka (YZ)", "YZ ile ilgili sorularınızı cevaplamak için hazırım. 💡"),
        ("📊 Veri Bilimi", "Veri Bilimi ile ilgili sorularınızı yanıtlayabilirim. 💡"),
        ("📈 Makine Öğrenimi (ML)", "ML hakkında sorularınızı bekliyorum. 💡")
    ]
    col2_topics = [
        ("🧠 Derin Öğrenme", "Derin öğrenme ile ilgili sorularınızı yanıtlıyorum. 💡"),
        ("🧩 RAG & LLM'ler", "RAG ve LLM konularında destek olabilirim. 💡"),
        ("🛠️ Geliştirici Araçları", "Geliştirici araçları hakkında sorularınızı cevaplarım. 💡")
    ]

    for (label1, content1), (label2, content2) in zip(col1_topics, col2_topics):
        with col1:
            if st.button(label1):
                append_message("assistant", content1)
        with col2:
            if st.button(label2):
                append_message("assistant", content2)

    st.divider()

# === Render messages ===
render_messages()

# === Show welcome buttons if first assistant message ===
if len(st.session_state.messages) == 1 and st.session_state.messages[0]["role"] == "assistant":
    display_welcome_buttons()

# === Get user input ===
if prompt := st.chat_input("Sorunuzu buraya yazın..."):
    append_message("user", prompt)
    prompt_lower = prompt.strip().lower()
    response = None

    greetings = ["merhaba", "selam", "günaydın", "iyi günler", "hey", "merhabalar"]
    thanks_words = ["teşekkür", "sağ ol", "eyvallah", "çok sağ ol", "thanks"]
    about_bot_phrases = ["sen kimsin", "ne iş yapıyorsun", "ne yaparsın"]
    creator_phrases = ["seni kim yaptı", "seni kim yapti", "seni kim oluşturdu", "seni kim programladı"]

    if any(word in prompt_lower for word in greetings):
        response = "Merhaba! Size teknik konularda nasıl yardımcı olabilirim? 😊"
    elif any(word in prompt_lower for word in thanks_words):
        response = "Rica ederim! Başka bir sorunuz var mı? 🙌"
    elif any(phrase in prompt_lower for phrase in creator_phrases):
        response = ("Ben, Abdullah Ahmed Briah tarafından geliştirildim. 👨‍💻\n"
                    "Yönetim Bilişim Sistemleri öğrencisidir ve veri bilimi ile yapay zeka konularına ilgi duymaktadır.\n"
                    "Sorularına teknik konularda yardımcı olmak için buradayım. 🤖")
    elif any(phrase in prompt_lower for phrase in about_bot_phrases):
        response = ("Ben, teknik sorularınızı yanıtlamak için tasarlanmış bir yapay zeka asistanıyım. "
                    "RAG teknolojisiyle doğru ve güncel bilgiler sunuyorum.")
    else:
        if prompt_lower in ["evet", "evet var", "tamam", "ok", "olur", "tabii", "elbette"]:
            response = "Tabii! Sorunuzu dinliyorum. 😊"
        elif prompt_lower in ["hayır", "yok", "hayir"]:
            response = "Tamam, başka bir sorunuz olursa buradayım. 🙌"
        else:
            with st.spinner("Düşünüyor..."):
                response = rag.ask(prompt)

    response_html = response.replace("\n", "<br>")
    append_message("assistant", response)
    st.chat_message("assistant").markdown(response_html, unsafe_allow_html=True)
