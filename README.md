first you need to copy these in a notepad and name it "requirements":

streamlit
transformers
seaborn
matplotlib
steamreviews
steamfront
torch
tqdm

then run this in your terminal: pip install -r requirements.txt

finally, copy paste this code in a notepad and change "path_to_app" with the actual path to the code where you saved it then save it as "all type" file: 

@echo off
cd "path_to_app"
streamlit run game_analyzer.py
pause
