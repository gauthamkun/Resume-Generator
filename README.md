# ATS Resume Builder (Streamlit)

A Streamlit app that generates a one-page ATS-friendly resume tailored to a target job description.

## Inputs
- Knowledge/memory file (TXT, PDF, DOCX): complete history of your work and achievements.
- Current resume file (TXT, PDF, DOCX).
- Job description (paste as text).
- Optional notes (what to emphasize).

## Output
- Structured on-screen resume preview (sectioned and readable).
- Downloadable `.txt` resume text.
- Downloadable `.pdf` rendered from structured JSON for consistent one-page layout.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then set your API key directly in code:
- Open `/Users/user/Documents/New project/app.py`
- Set `OPENAI_API_KEY = "your_key_here"`

Alternative: use environment variable `OPENAI_API_KEY`.

## Run
```bash
streamlit run app.py
```

## How it works
1. Reads uploaded memory/resume files.
2. Saves them locally for reuse.
3. Sends inputs + JD to the model with strict JSON-only instructions.
4. Parses and sanitizes JSON:
   - removes `match_notes`
   - keeps only personal/independent projects
   - filters study-material projects
   - deduplicates skills across categories
5. Renders polished preview and dynamic one-page PDF with ReportLab.

## Notes
- Model output is expected as strict JSON (not plain resume prose).
- `match_notes` are not shown or included in downloads.
- Saved memory is stored locally under `.streamlit_resume_builder/`.
