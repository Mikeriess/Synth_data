Conversation Inspector Tool
-------------------------

This tool provides a web interface for comparing original and synthetic conversations side by side.

Setup:
1. Install required Python packages:
   pip install datasets pandas openpyxl

2. Prepare the data by running:
   python prepare_data.py
   This will create conversations.json from the HuggingFace dataset

3. Start a local web server:
   python server.py
   
4. Open in browser:
   http://localhost:8000

Usage:
- View original conversations on the left, synthetic on the right
- Use Previous/Next buttons to navigate between conversations
- Conversation ID and count are shown at the top
- Each message shows the poster ID and message text

Files:
index.html - The web interface
inspector.js - JavaScript code for loading and displaying conversations
prepare_data.py - Script to prepare data from HuggingFace dataset
conversations.json - Generated JSON file containing the conversation data

Notes:
- Make sure you have access to the HuggingFace dataset specified in prepare_data.py
- The interface requires a modern web browser with JavaScript enabled
- All data is loaded locally, no external server required
- The tool works offline once conversations.json is generated
- Annotations are saved to annotations.json in the inspect folder
- Each annotation includes a timestamp and conversation ID
- The file is created automatically on first save
- Annotations are also saved in browser localStorage for UI state 