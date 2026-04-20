# Makes `app/` a proper package so Python always resolves `from app.visualise import …`
# to ROOT/app/visualise.py — not to app/streamlit/app.py which Streamlit adds to sys.path.
