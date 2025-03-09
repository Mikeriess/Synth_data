Conversation Inspector Tool
-------------------------

This tool provides a web interface for comparing original and synthetic conversations side by side.

Setup:
1. Install required Python packages:
   pip install datasets pandas openpyxl

2. Start a local web server:
   python server.py
   
3. Open in browser:
   http://localhost:8000

Usage:
- On the landing page, enter a HuggingFace dataset ID to download and process
- After processing, you'll be redirected to the inspection interface
- View original conversations on the left, synthetic on the right
- Use Previous/Next buttons to navigate between conversations
- Conversation ID and count are shown at the top
- Each message shows the poster ID and message text
- Rate and annotate conversations using the provided controls

Files:
dataset_selection.html - The landing page for dataset selection
index.html - The inspection interface
inspector.js - JavaScript code for loading and displaying conversations
prepare_data.py - Script to prepare data from HuggingFace dataset
prepare_from_file.py - Script to prepare data from processed JSON file
server.py - HTTP server with dataset processing capabilities
conversations.json - Generated JSON file containing the conversation data

Notes:
- Make sure you have access to the HuggingFace dataset you specify
- The interface requires a modern web browser with JavaScript enabled
- All data is loaded locally, no external server required
- Annotations are saved to annotations.json in the inspect folder
- Each annotation includes a timestamp and conversation ID
- The file is created automatically on first save
- Annotations are also saved in browser localStorage for UI state 