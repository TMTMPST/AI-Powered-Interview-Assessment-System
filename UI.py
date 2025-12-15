import streamlit as st
import pandas as pd
import tempfile
import subprocess
import json
import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from faster_whisper import WhisperModel

# =========================
# CONFIGURATIONS
# =========================
WHISPER_MODEL_SIZE = "large-v3"   # atau "medium" / "small"

# GPU Optimization for RTX 3060 12GB
def detect_device():
    try:
        import torch
        if torch.cuda.is_available():
            # Get GPU info
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"🎮 GPU Detected: {gpu_name} ({gpu_memory:.1f} GB)")

            # For RTX 3060 12GB, use float16 for best performance
            # float16 is faster than int8 on modern NVIDIA GPUs
            return "cuda", "float16"
    except Exception as e:
        print(f"⚠️ CUDA detection failed: {e}")
    return "cpu", "int8"

WHISPER_DEVICE, WHISPER_COMPUTE_TYPE = detect_device()

# Performance settings
NUM_WORKERS = 4  # For parallel processing
WHISPER_NUM_WORKERS = 4  # Whisper transcription workers
WHISPER_BEAM_SIZE = 5  # Balance between speed and accuracy
FFMPEG_THREADS = 6  # Use multiple CPU threads for ffmpeg

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.1"         # sesuaikan dengan yang ada di mesinmu

MAX_SCORE_PER_QUESTION = 4
PASS_THRESHOLD = 75               # final score threshold


# =========================
# HELPER FUNCTIONS
# =========================

@st.cache_resource
def load_whisper_model():
    """Load Whisper model optimized for RTX 3060."""
    return WhisperModel(
        WHISPER_MODEL_SIZE,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
        num_workers=WHISPER_NUM_WORKERS,  # Parallel CPU workers for preprocessing
        cpu_threads=NUM_WORKERS,  # CPU threads for operations
    )


def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to a temporary location and return its path."""
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def extract_audio(video_path: str) -> str:
    """Use ffmpeg to extract mono 16kHz WAV audio from the video (optimized)."""
    audio_fd, audio_path = tempfile.mkstemp(suffix=".wav")
    os.close(audio_fd)

    cmd = [
        "ffmpeg",
        "-y",               # overwrite
        "-threads", str(FFMPEG_THREADS),  # Multi-threaded decoding
        "-i", video_path,
        "-vn",              # No video (faster)
        "-ac", "1",         # mono
        "-ar", "16000",     # 16 kHz
        "-acodec", "pcm_s16le",  # Fast codec
        "-threads", str(FFMPEG_THREADS),  # Multi-threaded encoding
        audio_path
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path


def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio to text using Faster-Whisper (GPU optimized)."""
    model = load_whisper_model()

    # Optimized transcription settings for speed
    segments, info = model.transcribe(
        audio_path,
        beam_size=WHISPER_BEAM_SIZE,
        vad_filter=True,  # Voice Activity Detection - skip silence (FASTER!)
        vad_parameters=dict(
            min_silence_duration_ms=500,  # Skip silences > 500ms
            speech_pad_ms=200,
        ),
        condition_on_previous_text=False,  # Faster, less context-dependent
        no_speech_threshold=0.6,  # Skip non-speech segments
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
    )

    text = " ".join(seg.text.strip() for seg in segments)
    return text.strip()


def build_llm_prompt(question: str, transcript: str, rubric: str) -> dict:
    """Build messages payload for Ollama (chat format) - Professional HRD style."""
    system_prompt = f"""You are a STRICT and PROFESSIONAL Senior HR Director with 15+ years of experience in technical hiring.

YOUR EVALUATION PHILOSOPHY:
- Score 4 is EXCEPTIONAL - reserved for truly outstanding answers that exceed expectations
- Score 3 is GOOD - solid answer but missing some depth or specifics
- Score 2 is AVERAGE - basic understanding shown but lacks substance
- Score 1 is BELOW AVERAGE - vague, generic, or mostly irrelevant
- Score 0 is UNACCEPTABLE - no answer, completely off-topic, or nonsensical

STRICT EVALUATION CRITERIA:
1. SPECIFICITY: Does the candidate provide CONCRETE examples, numbers, metrics, or real experiences? Generic answers get lower scores.
2. RELEVANCE: Does the answer DIRECTLY address the question asked? Tangential answers are penalized.
3. DEPTH: Does the candidate demonstrate deep understanding or just surface-level knowledge?
4. STRUCTURE: Is the answer well-organized and coherent?
5. PROFESSIONAL INSIGHT: Does the answer show real-world experience vs. textbook knowledge?

RED FLAGS (reduce score):
- Vague phrases like "I would do..." without specific examples
- No concrete metrics, numbers, or measurable outcomes
- Answer seems memorized or overly generic
- Doesn't directly address the question asked
- Claims without supporting evidence or details

SCORING DISTRIBUTION:
- Score 0: No answer or completely off-topic
- Score 1: Minimal effort, lacks substance, very generic
- Score 2: Basic answer with some relevance but missing key details
- Score 3: Good answer with specific examples and reasonable depth
- Score 4: Excellent answer with concrete examples, metrics, clear expertise, and comprehensive coverage

IMPORTANT:
- Evaluate OBJECTIVELY based on the content provided
- Use the FULL score range (0-4) based on answer quality
- Award higher scores when candidates provide specific details and evidence
- Award lower scores for vague or incomplete answers
- Ignore speech errors, accent issues, or transcription artifacts

You MUST respond with ONLY a valid JSON object:
{{"score": <0-{MAX_SCORE_PER_QUESTION}>, "reason": "<2-3 sentence professional explanation>"}}"""

    user_prompt = f"""INTERVIEW QUESTION:
"{question}"

CANDIDATE'S TRANSCRIPT (from speech-to-text):
"{transcript}"

SCORING RUBRIC:
{rubric}

Evaluate this response objectively. Score based on:
1. How well they answered the question
2. Specificity of examples and details provided
3. Depth of knowledge demonstrated
4. Relevance and coherence

Use the FULL score range (0-4). Give credit where deserved.
Return ONLY the JSON object with score and reason."""

    return {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {
            "temperature": 0.5,  # Balanced temperature for more varied, objective evaluation
            "top_p": 0.9,
            "num_predict": 200,  # Ensure complete response
        }
    }


def call_ollama(question: str, transcript: str, rubric: str) -> dict:
    """Call Ollama to get score + reason. Returns {'score': int, 'reason': str}."""
    try:
        payload = build_llm_prompt(question, transcript, rubric)
        resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        content = data["message"]["content"].strip()

        # Try to parse JSON directly
        # Sometimes model may wrap JSON in text, so be defensive
        json_str = content
        # If there is extra text, try to locate first '{' and last '}'
        if not content.strip().startswith("{"):
            start = content.find("{")
            end = content.rfind("}")
            if start != -1 and end != -1 and end > start:
                json_str = content[start:end+1]

        result = json.loads(json_str)
        score = int(result.get("score", 0))
        reason = str(result.get("reason", ""))
        return {
            "score": max(0, min(MAX_SCORE_PER_QUESTION, score)),
            "reason": reason,
        }

    except Exception as e:
        # Fallback: mark for human review
        return {
            "score": 0,
            "reason": f"Error evaluating with AI: {e}. Needs human review.",
        }


def compute_final_scores(question_rows, transcript: str, project_score: float | None):
    """Evaluate all questions and compute interview + final score (PARALLEL)."""

    # Prepare questions list
    questions_list = []
    for idx, row in question_rows.iterrows():
        q_text = str(row.get("question", "")).strip()
        rubric = str(row.get("rubric", "")).strip()
        if q_text:
            questions_list.append((idx, q_text, rubric))

    total_questions = len(questions_list)
    if total_questions == 0:
        return [], 0.0, 0.0, "Need Human Review"

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    per_question_results = [None] * total_questions
    completed = 0

    # Parallel evaluation using ThreadPoolExecutor
    # This speeds up LLM calls significantly
    def evaluate_question(args):
        idx, q_text, rubric = args
        result = call_ollama(q_text, transcript, rubric)
        return idx, q_text, rubric, result

    # Use 2-3 parallel workers for Ollama (adjust based on your GPU VRAM)
    # Too many parallel requests can overwhelm the LLM
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(evaluate_question, q): q for q in questions_list}

        for future in as_completed(futures):
            completed += 1
            progress_bar.progress(completed / total_questions)
            status_text.text(f"Mengevaluasi pertanyaan {completed} dari {total_questions}...")

            idx, q_text, rubric, result = future.result()

            # Find position in results
            pos = next(i for i, q in enumerate(questions_list) if q[0] == idx)
            per_question_results[pos] = {
                "question": q_text,
                "rubric": rubric,
                "score": result["score"],
                "reason": result["reason"],
            }

    progress_bar.empty()
    status_text.empty()

    # Calculate scores
    total_score = sum(r["score"] for r in per_question_results if r)
    answered_questions = len([r for r in per_question_results if r])

    max_possible = max(1, answered_questions * MAX_SCORE_PER_QUESTION)
    interview_score_scaled = (total_score / max_possible) * 100.0

    if project_score is None:
        final_total_score = interview_score_scaled
    else:
        final_total_score = (
            PROJECT_WEIGHT * project_score +
            INTERVIEW_WEIGHT * interview_score_scaled
        )

    decision = "PASSED" if final_total_score >= PASS_THRESHOLD else "Need Human Review"

    return per_question_results, interview_score_scaled, final_total_score, decision


# =========================
# STREAMLIT UI
# =========================

st.set_page_config(
    page_title="AI Interview Assessment",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional look
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.5rem;
        padding: 1rem 0;
        border-bottom: 3px solid #3b82f6;
    }

    .subtitle {
        font-size: 1rem;
        color: #64748b;
        margin-bottom: 2rem;
    }

    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        padding: 0.75rem 1.25rem;
        border-radius: 8px;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        font-size: 1.1rem;
    }

    /* Info boxes */
    .info-box {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem 1.25rem;
        border-radius: 4px;
        margin: 1rem 0;
    }

    .success-box {
        background-color: #f0fdf4;
        border-left: 4px solid #22c55e;
        padding: 1rem 1.25rem;
        border-radius: 4px;
        margin: 1rem 0;
    }

    .warning-box {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem 1.25rem;
        border-radius: 4px;
        margin: 1rem 0;
    }

    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.875rem;
        font-weight: 600;
    }

    .status-online {
        background-color: #dcfce7;
        color: #166534;
    }

    .status-offline {
        background-color: #fee2e2;
        color: #991b1b;
    }

    /* Cards */
    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 6px;
        transition: all 0.3s;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Check if Ollama is running
@st.cache_data(ttl=60)
def check_ollama_connection():
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        return resp.status_code == 200
    except:
        return False

# Title and header
st.markdown('<h1 class="main-title">AI-Powered Interview Assessment System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Automated candidate evaluation using advanced AI models</p>', unsafe_allow_html=True)

# System status
ollama_status = check_ollama_connection()
if ollama_status:
    st.markdown("""
    <div class="success-box">
        <strong>System Status:</strong> <span class="status-badge status-online">ONLINE</span><br>
        <small>All services are operational</small>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="warning-box">
        <strong>System Status:</strong> <span class="status-badge status-offline">OFFLINE</span><br>
        <small>Ollama service is not running. Please start with: <code>ollama serve</code></small>
    </div>
    """, unsafe_allow_html=True)

# -------------------------
# Sidebar: global settings
# -------------------------
with st.sidebar:
    st.markdown("### Configuration")
    st.markdown("---")
    PASS_THRESHOLD = st.slider(
        "Pass Threshold (%)",
        0, 100, PASS_THRESHOLD, 5,
        help="Minimum score required to pass the assessment"
    )

    st.markdown("---")
    st.markdown("#### System Information")
    st.markdown(f"""
    - **STT Model:** Faster-Whisper (large-v3)
    - **LLM:** Ollama (Llama 3.1)
    - **Device:** {WHISPER_DEVICE.upper()}
    - **Max Score:** {MAX_SCORE_PER_QUESTION} per question
    """)

# -------------------------
# Section 1: Questions & Rubrics
# -------------------------

st.markdown('<div class="section-header">STEP 1: Questions & Rubric Management</div>', unsafe_allow_html=True)

# Add save/load functionality
col_save, col_load = st.columns([1, 1])

with col_save:
    st.markdown("**Save Template**")
    # Prepare download data
    if "questions_df" in st.session_state:
        template_data = st.session_state.questions_df.to_dict('records')
    else:
        template_data = []

    json_str = json.dumps(template_data, indent=2, ensure_ascii=False)
    st.download_button(
        label="Download Template JSON",
        data=json_str,
        file_name="questions_template.json",
        mime="application/json",
        help="Save current questions as JSON template"
    )

with col_load:
    st.markdown("**Load Template**")
    uploaded_template = st.file_uploader("Upload JSON Template", type="json", label_visibility="collapsed")
    if uploaded_template is not None and not st.session_state.get("template_loaded", False):
        try:
            data_template = json.load(uploaded_template)
            if isinstance(data_template, list):
                st.session_state.questions_df = pd.DataFrame(data_template)
                st.success("Template loaded successfully")
                st.session_state["template_loaded"] = True
            else:
                st.error("Invalid template format")
        except Exception as e:
            st.error(f"Failed to load: {e}")

# Reset flag button
if st.session_state.get("template_loaded", False):
    if st.button("Load Another Template"):
        st.session_state["template_loaded"] = False
        st.rerun()

st.markdown('<div style="margin: 20px 0;"></div>', unsafe_allow_html=True)

# Default questions if not in session state
if "questions_df" not in st.session_state:
    st.session_state.questions_df = pd.DataFrame(
        [
            {
                "question": "Tell me about a recent machine learning project you worked on.",
                "rubric": (
                    "Score 4: Explains specific project, personal role, dataset, model, and results.\n"
                    "Score 3: Explains project clearly but lacks technical details.\n"
                    "Score 2: Very general explanation, minimal details.\n"
                    "Score 1: Answer not relevant or very brief.\n"
                    "Score 0: No answer."
                ),
            }
        ]
    )

num_questions = st.number_input(
    "Number of Questions",
    min_value=1,
    max_value=20,
    value=len(st.session_state.questions_df) if not st.session_state.questions_df.empty else 5,
    help="Define how many interview questions"
)

# Sync data editor with session state
questions_df = st.data_editor(
    st.session_state.questions_df,
    num_rows="dynamic",
    width="stretch",
    key="questions_editor",
    column_config={
        "question": st.column_config.TextColumn("Question", width="large"),
        "rubric": st.column_config.TextColumn("Rubric (0–4)", width="large"),
    }
)

# Update session state when user edits the table
st.session_state.questions_df = questions_df

st.caption("HR dapat menambah/menghapus baris dan mengisi rubrik detail per pertanyaan.")


# -------------------------
# Section 2: Upload video & project score
# -------------------------

st.subheader("2. Upload Video/Audio Jawaban Interview")
# -------------------------
# Section 2: Upload Interview Files
# -------------------------

st.markdown('<div class="section-header">STEP 2: Upload Interview Videos/Audios</div>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <strong>Instructions:</strong> Upload video/audio files in order matching your questions.<br>
    Video 1 → Question 1, Video 2 → Question 2, etc.
</div>
""", unsafe_allow_html=True)

# Supported file formats
VIDEO_EXTENSIONS = ["mp4", "mkv", "mov", "avi", "webm"]
AUDIO_EXTENSIONS = ["mp3", "wav", "m4a", "flac", "ogg", "aac"]
ALL_EXTENSIONS = VIDEO_EXTENSIONS + AUDIO_EXTENSIONS

media_files = st.file_uploader(
    "Upload media files (in question order)",
    type=ALL_EXTENSIONS,
    accept_multiple_files=True,
    help=f"Supported formats: {', '.join(ALL_EXTENSIONS)}"
)

if media_files:
    # Sort files by name to ensure correct order
    media_files = sorted(media_files, key=lambda x: x.name)

    num_questions = len(questions_df[questions_df["question"].fillna("").str.strip() != ""])

    st.markdown(f"""
    <div class="success-box">
        <strong>{len(media_files)} files uploaded</strong> (for {num_questions} questions)
    </div>
    """, unsafe_allow_html=True)

    if len(media_files) != num_questions:
        st.markdown(f"""
        <div class="warning-box">
            File count ({len(media_files)}) does not match question count ({num_questions})
        </div>
        """, unsafe_allow_html=True)

    for idx, media_file in enumerate(media_files):
        # Check if it's video or audio based on extension
        file_ext = os.path.splitext(media_file.name)[1].lower().lstrip(".")
        is_audio_file = file_ext in AUDIO_EXTENSIONS

        # Get corresponding question
        q_label = f"Question {idx + 1}" if idx < num_questions else "Extra"

        with st.expander(f"{q_label}: {media_file.name}", expanded=False):
            if is_audio_file:
                st.audio(media_file)
            else:
                st.video(media_file)


# -------------------------
# Section 3: Run Assessment
# -------------------------

st.markdown('<div class="section-header">STEP 3: Run AI Assessment</div>', unsafe_allow_html=True)

run_button = st.button("Run Assessment", type="primary", use_container_width=True)

if run_button:
    if not media_files:
        st.error("Please upload interview media files first.")
    elif questions_df["question"].fillna("").str.strip().eq("").all():
        st.error("At least one question must be filled.")
    else:
        # Sort files by name
        media_files_sorted = sorted(media_files, key=lambda x: x.name)

        # Get valid questions
        valid_questions = questions_df[questions_df["question"].fillna("").str.strip() != ""].reset_index(drop=True)

        if len(media_files_sorted) != len(valid_questions):
            st.warning(f"File count ({len(media_files_sorted)}) does not match question count ({len(valid_questions)}). Will process the matching pairs.")

        # Process each video with its corresponding question
        per_question_results = []
        total_score = 0

        num_to_process = min(len(media_files_sorted), len(valid_questions))

        progress_bar = st.progress(0)

        for idx in range(num_to_process):
            media_file = media_files_sorted[idx]
            question_row = valid_questions.iloc[idx]

            q_text = str(question_row.get("question", "")).strip()
            rubric = str(question_row.get("rubric", "")).strip()

            st.markdown(f"---")
            st.markdown(f"### Pertanyaan {idx + 1}: {q_text[:50]}{'...' if len(q_text) > 50 else ''}")
            st.caption(f"File: {media_file.name}")

            # Check file type
            file_ext = os.path.splitext(media_file.name)[1].lower().lstrip(".")
            is_audio_file = file_ext in AUDIO_EXTENSIONS

            if is_audio_file:
                spinner_text = f"Processing audio {idx + 1}/{num_to_process}: transcription..."
            else:
                spinner_text = f"Processing video {idx + 1}/{num_to_process}: extract audio & transcription..."

            with st.spinner(spinner_text):
                try:
                    media_path = save_uploaded_file(media_file)

                    if is_audio_file:
                        if file_ext == "wav":
                            audio_path = media_path
                        else:
                            audio_path = extract_audio(media_path)
                    else:
                        audio_path = extract_audio(media_path)

                    transcript = transcribe_audio(audio_path)
                except Exception as e:
                    st.error(f"Failed to process {media_file.name}: {e}")
                    per_question_results.append({
                        "question": q_text,
                        "rubric": rubric,
                        "transcript": f"Error: {e}",
                        "score": 0,
                        "reason": "Failed to process file"
                    })
                    continue

            with st.expander("View Transcript"):
                st.text_area("Transcript", transcript, height=100, key=f"transcript_{idx}")

            # Evaluate this specific question with its transcript
            with st.spinner(f"Evaluating answer for question {idx + 1}..."):
                result = call_ollama(q_text, transcript, rubric)

            score = result["score"]
            reason = result["reason"]
            total_score += score

            # Show result
            col_score, col_reason = st.columns([1, 3])
            with col_score:
                st.metric(f"Score", f"{score}/{MAX_SCORE_PER_QUESTION}")
            with col_reason:
                st.markdown(f"**Evaluation:** {reason}")

            per_question_results.append({
                "question": q_text,
                "rubric": rubric,
                "transcript": transcript,
                "score": score,
                "reason": reason,
            })

            # Update progress
            progress_bar.progress((idx + 1) / num_to_process)

        progress_bar.empty()

        # Calculate final scores
        answered_questions = len(per_question_results)
        max_possible = max(1, answered_questions * MAX_SCORE_PER_QUESTION)
        interview_score_scaled = (total_score / max_possible) * 100.0
        final_total_score = interview_score_scaled

        decision = "PASSED" if final_total_score >= PASS_THRESHOLD else "NEEDS REVIEW"

        # Summary
        st.markdown("---")
        st.markdown('<div class="section-header">Assessment Results</div>', unsafe_allow_html=True)

        # Results table
        results_df = pd.DataFrame(per_question_results)
        st.dataframe(
            results_df[["question", "score", "reason"]],
            use_container_width=True,
            column_config={
                "question": st.column_config.TextColumn("Question", width="medium"),
                "score": st.column_config.NumberColumn("Score", width="small"),
                "reason": st.column_config.TextColumn("Evaluation", width="large"),
            }
        )

        # Score metrics
        st.markdown('<div style="margin: 20px 0;"></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("Interview Score", f"{interview_score_scaled:.1f}/100")
        c2.metric("Final Score", f"{final_total_score:.1f}/100")

        st.markdown('<div style="margin: 20px 0;"></div>', unsafe_allow_html=True)

        if decision == "PASSED":
            st.markdown(f"""
            <div class="success-box">
                <h3 style="margin: 0;">Decision: {decision}</h3>
                <p style="margin: 5px 0 0 0;">Candidate has met the passing criteria</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="warning-box">
                <h3 style="margin: 0;">Decision: {decision}</h3>
                <p style="margin: 5px 0 0 0;">Score is below threshold, human review required</p>
            </div>
            """, unsafe_allow_html=True)

        # Export results
        st.markdown('<div style="margin: 30px 0 10px 0;"><strong>Export Results</strong></div>', unsafe_allow_html=True)

        export_data = {
            "assessorProfile": {
                "id": 1,
                "name": "AI Auto-Assessor",
                "photoUrl": "https://ui-avatars.com/api/?name=AI"
            },
            "decision": decision,
            "scoresOverview": {
                "interview": round(interview_score_scaled, 2),
                "total": round(final_total_score, 2)
            },
            "reviewChecklistResult": {
                "interviews": {
                    "minScore": 0,
                    "maxScore": MAX_SCORE_PER_QUESTION,
                    "scores": per_question_results
                }
            }
        }

        json_str = json.dumps(export_data, indent=2, ensure_ascii=False)

        col_export1, col_export2 = st.columns(2)

        with col_export1:
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name="assessment_result.json",
                mime="application/json",
                use_container_width=True
            )

        with col_export2:
            csv_data = results_df[["question", "score", "reason"]].to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv_data,
                file_name="assessment_result.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.markdown("""
        <div class="success-box" style="margin-top: 20px;">
            Assessment completed successfully
        </div>
        """, unsafe_allow_html=True)

