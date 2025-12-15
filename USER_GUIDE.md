# User Guide - AI-Powered Interview Assessment System

## Table of Contents

1. [System Overview](#system-overview)
2. [Prerequisites](#prerequisites)
3. [Getting Started](#getting-started)
4. [Step-by-Step Usage](#step-by-step-usage)
5. [Tips & Best Practices](#tips--best-practices)
6. [Troubleshooting](#troubleshooting)

---

## System Overview

This AI-powered system automatically evaluates candidate interview responses using:

- **Speech-to-Text**: Faster-Whisper (large-v3) for accurate transcription
- **LLM Evaluation**: Ollama with Llama 3.1 for intelligent scoring
- **GPU Acceleration**: Optimized for NVIDIA GPUs (CUDA)

**Key Features:**

- Upload multiple video/audio files (one per question)
- Automatic transcription and evaluation
- Customizable question templates and scoring rubrics
- Export results in JSON and CSV formats
- Professional, emoji-free interface

---

## Prerequisites

### Required Software

1. **Python 3.13+** with virtual environment
2. **Ollama** - Must be running locally
3. **CUDA** - For GPU acceleration (optional but recommended)
4. **ffmpeg** - For audio/video processing

### Hardware Recommendations

- **Minimum**: 8GB RAM, CPU-only mode
- **Recommended**: 16GB+ RAM, NVIDIA GPU with 6GB+ VRAM
- **Optimal**: 32GB RAM, RTX 3060 or better

### Installation

```bash
# 1. Activate virtual environment
source ./venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start Ollama (in separate terminal)
ollama serve

# 4. Pull Llama 3.1 model (if not already installed)
ollama pull llama3.1
```

---

## Getting Started

### Launch the Application

```bash
# Make sure you're in the project directory and venv is activated
streamlit run UI.py
```

The app will open in your default browser at `http://localhost:8501`

### System Status Check

When the app loads, check the **System Status** indicator at the top:

- **🟢 ONLINE**: All services operational (Ollama is running)
- **🔴 OFFLINE**: Ollama service not detected - start with `ollama serve`

---

## Step-by-Step Usage

### STEP 1: Questions & Rubric Management

#### Option A: Create New Questions

1. Set **Number of Questions** using the input field
2. Click on the table to edit questions and rubrics
3. Fill in:
   - **Question**: The interview question to ask
   - **Rubric**: Scoring criteria (describe what each score 0-4 means)

**Example Question:**

```
Question: Tell me about a recent machine learning project you worked on.

Rubric:
Score 4: Explains specific project, personal role, dataset, model, and results.
Score 3: Explains project clearly but lacks technical details.
Score 2: Very general explanation, minimal details.
Score 1: Answer not relevant or very brief.
Score 0: No answer.
```

#### Option B: Use Template

1. Click **Download Template JSON** to save current questions
2. Click **Upload JSON Template** and select a saved template file
3. Questions will load automatically
4. Click **Load Another Template** if you want to change templates

**Template Format:**

```json
[
  {
    "question": "Your question here",
    "rubric": "Score 4: ...\nScore 3: ...\nScore 2: ...\nScore 1: ...\nScore 0: ..."
  }
]
```

---

### STEP 2: Upload Interview Videos/Audios

#### Supported Formats

**Video**: MP4, MKV, MOV, AVI, WebM
**Audio**: MP3, WAV, M4A, FLAC, OGG, AAC

#### Upload Instructions

1. Click **Browse files** or drag-and-drop files
2. Select multiple files (one per question)
3. **IMPORTANT**: Files are matched to questions by order
   - Video 1 → Question 1
   - Video 2 → Question 2
   - And so on...

#### File Naming Tip

Name your files so they sort correctly:

```
01_question1.mp4
02_question2.mp4
03_question3.mp4
```

#### Preview

- Expand each file entry to preview video/audio
- Verify the file matches the intended question
- Check file count matches question count

---

### STEP 3: Run AI Assessment

1. Click **Run Assessment** button
2. The system will process each file:
   - Extract audio (for video files)
   - Transcribe speech to text
   - Evaluate answer against rubric
   - Display score and reasoning

#### During Processing

- Progress bar shows current status
- Each question displays:
  - **Transcript**: Expand to view full transcription
  - **Score**: X/4 points
  - **Evaluation**: AI's reasoning for the score

#### Processing Time

- **Per video**: ~30-60 seconds (GPU) or 2-5 minutes (CPU)
- **Total time**: Depends on number and length of videos

---

### STEP 4: Review Results

#### Assessment Summary

After all questions are processed, you'll see:

**Results Table**

- Question text
- Score received
- Evaluation reasoning

**Score Metrics**

- **Interview Score**: Total score as percentage (0-100%)
- **Final Score**: Same as interview score

**Decision**

- **PASSED**: Score ≥ threshold (configurable in sidebar)
- **NEEDS REVIEW**: Score < threshold

#### Export Options

**Download JSON**

- Complete assessment data with all details
- Suitable for integration with other systems
- Includes assessor profile, scores, and full responses

**Download CSV**

- Simple spreadsheet format
- Contains: Question, Score, Evaluation
- Easy to open in Excel/Google Sheets

---

## Configuration (Sidebar)

### Pass Threshold

- Adjust the minimum passing percentage (0-100%)
- Default: 70%
- Candidates below this threshold are marked "NEEDS REVIEW"

### System Information

View current configuration:

- **STT Model**: Faster-Whisper (large-v3)
- **LLM**: Ollama (Llama 3.1)
- **Device**: CUDA (GPU) or CPU
- **Max Score**: 4 per question

---

## Tips & Best Practices

### For Best Results

#### Question Design

- ✅ Ask specific, focused questions
- ✅ Include clear rubrics with measurable criteria
- ✅ Define what differentiates each score level
- ❌ Avoid overly broad or multi-part questions

#### Video Quality

- ✅ Clear audio with minimal background noise
- ✅ Good microphone quality
- ✅ Candidate speaks clearly and at moderate pace
- ❌ Avoid videos with multiple speakers or overlapping audio

#### File Organization

- ✅ Name files sequentially (01, 02, 03...)
- ✅ Keep videos focused (1 question = 1 video)
- ✅ Upload files in correct order
- ❌ Don't mix multiple questions in one video

### Scoring Expectations

The AI evaluates based on:

1. **Specificity**: Concrete examples vs vague statements
2. **Relevance**: Directly answers the question
3. **Depth**: Demonstrates understanding beyond surface level
4. **Structure**: Organized, coherent response
5. **Evidence**: Metrics, outcomes, real experience

**Typical Distribution:**

- Score 0-1: Poor/irrelevant answers
- Score 2: Basic, somewhat vague answers
- Score 3: Good answers with specific details
- Score 4: Excellent, comprehensive answers with evidence

---

## Troubleshooting

### System Status: OFFLINE

**Problem**: Ollama service not detected
**Solution**:

```bash
# Start Ollama in a separate terminal
ollama serve
```

### "No answer" Transcripts

**Problem**: Whisper returns empty or "no answer"
**Possible Causes:**

- Audio quality too poor
- File corrupted
- Very short video (< 2 seconds)

**Solutions:**

- Check audio quality
- Try re-encoding video with ffmpeg
- Ensure candidate is speaking audibly

### Slow Processing

**Problem**: Each video takes 3-5 minutes
**Solutions:**

- Check if GPU is being used (see System Information)
- Install PyTorch with CUDA support
- Close other GPU-intensive applications
- Consider shorter video clips

### Score Always 2

**Problem**: AI gives same score repeatedly
**Solution**:

- Temperature is now balanced (0.5)
- Ensure transcripts are accurate
- Check rubrics are clear and specific
- Verify Ollama is responding correctly

### File Upload Errors

**Problem**: Can't upload video files
**Solutions:**

- Check file format is supported
- Ensure file size is reasonable (< 500MB recommended)
- Try converting to MP4 format
- Check disk space

### Template Won't Load

**Problem**: Upload template but questions don't appear
**Solution**:

- Check JSON format is correct (array of objects)
- Click "Load Another Template" to reset flag
- Refresh the page
- Create new template from scratch

---

## Keyboard Shortcuts

| Shortcut | Action                        |
| -------- | ----------------------------- |
| `R`      | Rerun the Streamlit app       |
| `Ctrl+C` | Stop the server (in terminal) |

---

## Advanced Usage

### Batch Processing

To process multiple candidates:

1. Create folders for each candidate
2. Use separate browser tabs for each assessment
3. Export results with candidate names in filenames

### Custom Scoring Scale

To use different max scores:

1. Edit `MAX_SCORE_PER_QUESTION` in `UI.py` (line ~18)
2. Update all rubrics to match new scale
3. Restart application

### GPU Memory Issues

If you get CUDA out-of-memory errors:

```python
# Edit UI.py, change compute_type
compute_type="int8"  # Instead of float16
```

---

## Support & Contact

For issues, improvements, or questions:

- Check this guide first
- Review DEPLOYMENT.md for setup help
- Check QUICKSTART.md for installation
- Report bugs via GitHub issues

---

**Version**: 1.0
**Last Updated**: December 2025
**Compatible with**: Python 3.13+, Streamlit 1.x, Ollama with Llama 3.1
