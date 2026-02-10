"""Streamlit entrypoint kept under src/app/ for the final project structure.

Run with:
  python -m streamlit run src/app/main.py

This delegates to the existing Streamlit app in web/app.py, preserving backwards compatibility.
"""

from web.app import main


if __name__ == "__main__":
    main()
