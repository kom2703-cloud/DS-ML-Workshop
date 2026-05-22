pip install streamlit-nightly -q
!streamlit run clean_app.py &>/dev/null&  # Run Streamlit in the background

import time
import urllib.request

print("Waiting for Streamlit to start...")
time.sleep(5)  # Give Streamlit a few seconds to start

try:
    # Get the URL of the running Streamlit app
    url = urllib.request.urlopen("http://localhost:8501").geturl()
    print(f"Streamlit app is running at: {url}")
except Exception as e:
    print(f"Error fetching Streamlit URL: {e}")
    print("You might need to check the Colab logs or port forwarding.")
