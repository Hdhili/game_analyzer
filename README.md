-copy these in a notepad named "requirements' :


streamlit

transformers

seaborn

matplotlib

steamreviews

steamfront

torch

tqdm


-then run this in your terminal : 


pip install -r requirements.txt


-finaly copy this in a notepad and change "path_to_the_code" with the actual path to the code where you saved it, and save as "All type" file:


@echo off

cd "path_to_the_code"

streamlit run game_analyzer.py

pause
