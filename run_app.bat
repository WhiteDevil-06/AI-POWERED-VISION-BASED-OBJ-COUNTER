@echo off
echo 📦 STARTING BOX SENSE AI DASHBOARD...
call .\env\Scripts\activate.bat
streamlit run dashboard.py
pause
