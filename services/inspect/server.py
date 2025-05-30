from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import os
from datetime import datetime
import pandas as pd
import io
import sys
import subprocess
from pathlib import Path
import urllib.parse
import shutil
import glob
from urllib.parse import parse_qs
import numpy as np
from collections import Counter
from datasets import load_dataset

# Add parent directory to path to import dataset_processor
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from dataset_processor import process_dataset

class AnnotationHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        # Handle request to list existing datasets
        if self.path == '/list_datasets':
            try:
                # Get all dataset folders
                dataset_folders = glob.glob('datasets/*')
                datasets = []
                
                for folder in dataset_folders:
                    # Skip the default folder
                    if folder.endswith('/default'):
                        continue
                    
                    # Try to load dataset info
                    info_file = os.path.join(folder, 'dataset_info.json')
                    if os.path.exists(info_file):
                        with open(info_file, 'r') as f:
                            dataset_info = json.load(f)
                        
                        # Add annotation count if available
                        annotations_file = dataset_info.get('annotations_file')
                        if annotations_file and os.path.exists(annotations_file):
                            try:
                                with open(annotations_file, 'r') as f:
                                    annotations = json.load(f)
                                dataset_info['annotation_count'] = len(annotations)
                            except:
                                dataset_info['annotation_count'] = 0
                        
                        datasets.append(dataset_info)
                    else:
                        # If no info file, create a basic entry
                        folder_name = os.path.basename(folder)
                        datasets.append({
                            'folder': folder,
                            'timestamp': folder_name
                        })
                
                # Sort datasets by timestamp (newest first)
                datasets.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(datasets).encode())
                return
            except Exception as e:
                print(f"Error listing datasets: {e}")
                self.send_error(500, str(e))
                return
        
        # Handle request to load an existing dataset
        if self.path.startswith('/load_dataset'):
            try:
                # Parse query parameters
                query = parse_qs(self.path.split('?', 1)[1])
                folder = query.get('folder', [''])[0]
                
                if not folder:
                    raise ValueError("No folder specified")
                
                # Check if the folder exists
                if not os.path.exists(folder):
                    raise ValueError(f"Folder not found: {folder}")
                
                # Load dataset info
                info_file = os.path.join(folder, 'dataset_info.json')
                if os.path.exists(info_file):
                    with open(info_file, 'r') as f:
                        dataset_info = json.load(f)
                else:
                    # Create basic info if not available
                    folder_name = os.path.basename(folder)
                    dataset_info = {
                        'folder': folder,
                        'timestamp': folder_name,
                        'dataset_id': f"dataset_{folder_name}",
                        'conversations_file': os.path.join(folder, 'conversations.json'),
                        'annotations_file': os.path.join(folder, 'annotations.json')
                    }
                
                # Update current dataset
                with open('current_dataset.json', 'w') as f:
                    json.dump(dataset_info, f, indent=2)
                
                # Redirect to the inspection interface
                self.send_response(302)
                self.send_header('Location', '/index.html')
                self.end_headers()
                return
            except Exception as e:
                print(f"Error loading dataset: {e}")
                self.send_error(500, str(e))
                return
        
        # Check if the path is for conversations.json or annotations.json
        if self.path == '/conversations.json' or self.path == '/annotations.json':
            # Determine which file to serve
            file_key = "conversations_file" if self.path == '/conversations.json' else "annotations_file"
            
            # Get the path from current dataset info
            current_dataset = {}
            if os.path.exists("current_dataset.json"):
                with open("current_dataset.json", "r") as f:
                    current_dataset = json.load(f)
            
            file_path = current_dataset.get(file_key)
            
            if file_path and os.path.exists(file_path):
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                
                with open(file_path, 'rb') as f:
                    self.wfile.write(f.read())
                return
            elif self.path == '/annotations.json':
                # If annotations file doesn't exist, return an empty array
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(b'[]')
                return
        
        # Serve the dataset selection page as the landing page
        if self.path == '/' or self.path == '':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            with open(Path(__file__).parent / 'dataset_selection.html', 'rb') as file:
                self.wfile.write(file.read())
            return
        
        # Handle request to switch datasets
        if self.path == '/switch_dataset':
            # Redirect to the dataset selection page
            self.send_response(302)
            self.send_header('Location', '/')
            self.end_headers()
            return
        
        # Handle request for dataset statistics
        if self.path == '/dataset_stats':
            try:
                # Get the current dataset info
                current_dataset = {}
                if os.path.exists("current_dataset.json"):
                    with open("current_dataset.json", "r") as f:
                        current_dataset = json.load(f)
                
                # Get the conversations file path
                conversations_file = current_dataset.get("conversations_file")
                if not conversations_file or not os.path.exists(conversations_file):
                    raise Exception("No dataset is currently loaded")
                
                # Load the conversations
                with open(conversations_file, 'r') as f:
                    conversations = json.load(f)
                
                # Process the dataset
                stats = self.calculate_dataset_stats(conversations)
                
                # Send the statistics as JSON
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(stats).encode())
                return
            except Exception as e:
                print(f"Error calculating dataset statistics: {e}")
                self.send_error(500, str(e))
                return
        
        # Add this to the AnnotationHandler class's do_GET method
        elif self.path == '/prompt_file':
            try:
                # Get the current dataset info
                current_dataset = {}
                if os.path.exists("current_dataset.json"):
                    with open("current_dataset.json", "r") as f:
                        current_dataset = json.load(f)
                
                # Get the dataset folder
                dataset_folder = current_dataset.get("folder")
                if not dataset_folder:
                    raise Exception("No dataset is currently loaded")
                
                # Get the prompt file path
                prompt_file = os.path.join(dataset_folder, "dialogue_prompt.txt")
                
                # If the prompt file doesn't exist, return a default message
                if not os.path.exists(prompt_file):
                    self.send_response(200)
                    self.send_header('Content-type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(b"No prompt file found for this dataset.")
                    return
                
                # Read and return the prompt file
                with open(prompt_file, 'rb') as f:
                    prompt_content = f.read()
                
                self.send_response(200)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(prompt_content)
                return
            except Exception as e:
                print(f"Error serving prompt file: {e}")
                self.send_error(500, str(e))
                return
        
        # Add this to the AnnotationHandler class's do_GET method
        elif self.path == '/config_file':
            try:
                # Get the current dataset info
                current_dataset = {}
                if os.path.exists("current_dataset.json"):
                    with open("current_dataset.json", "r") as f:
                        current_dataset = json.load(f)
                
                # Get the dataset folder
                dataset_folder = current_dataset.get("folder")
                if not dataset_folder:
                    raise Exception("No dataset is currently loaded")
                
                # Get the config file path
                config_file = os.path.join(dataset_folder, "generation_config.json")
                
                # If the config file doesn't exist, return a default message
                if not os.path.exists(config_file):
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"message": "No config file found for this dataset."}).encode())
                    return
                
                # Read and return the config file
                with open(config_file, 'rb') as f:
                    config_content = f.read()
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(config_content)
                return
            except Exception as e:
                print(f"Error serving config file: {e}")
                self.send_error(500, str(e))
                return
        
        # For all other GET requests, use the default handler
        return SimpleHTTPRequestHandler.do_GET(self)
    
    def do_POST(self):
        if self.path == '/process_dataset':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length).decode('utf-8')
            
            # Parse form data
            form_data = {}
            for field in post_data.split('&'):
                key, value = field.split('=', 1)
                # Properly decode URL-encoded values
                form_data[key] = urllib.parse.unquote(value)
            
            dataset_id = form_data.get('dataset_id', '')
            config = form_data.get('config', '') or None
            split = form_data.get('split', 'train')
            max_samples = form_data.get('max_samples', '')
            
            if max_samples:
                try:
                    max_samples = int(max_samples)
                except ValueError:
                    max_samples = None
            
            try:
                # Create a unique folder with just datetime as name
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                dataset_folder = os.path.join("datasets", timestamp)
                os.makedirs(dataset_folder, exist_ok=True)
                
                # Process the dataset
                output_file = process_dataset(dataset_id, config, split, max_samples, output_dir=dataset_folder)
                
                # Use the processed dataset to prepare inspection data
                script_path = Path(__file__).parent / 'prepare_from_file.py'
                conversations_file = os.path.join(dataset_folder, "conversations.json")
                subprocess.run([
                    sys.executable, 
                    str(script_path), 
                    output_file, 
                    conversations_file
                ])
                
                # Create an empty annotations file as a placeholder
                annotations_file = os.path.join(dataset_folder, "annotations.json")
                with open(annotations_file, 'w') as f:
                    json.dump([], f)
                
                # Create a file to track the current dataset
                current_dataset_info = {
                    "dataset_id": dataset_id,
                    "config": config,
                    "split": split,
                    "timestamp": timestamp,
                    "folder": dataset_folder,
                    "conversations_file": conversations_file,
                    "annotations_file": annotations_file,
                    "raw_data_file": output_file
                }
                
                with open(os.path.join(dataset_folder, "dataset_info.json"), "w") as f:
                    json.dump(current_dataset_info, f, indent=2)
                
                # Also save to the root directory for the UI to find
                with open("current_dataset.json", "w") as f:
                    json.dump(current_dataset_info, f, indent=2)
                
                # Redirect to the inspection interface
                self.send_response(302)
                self.send_header('Location', '/index.html')
                self.end_headers()
                
            except Exception as e:
                # Send error response
                self.send_response(500)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                
                error_html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Error</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; padding: 20px; }}
                        .error {{ color: red; background-color: #ffeeee; padding: 10px; border-radius: 5px; }}
                        .back {{ margin-top: 20px; }}
                    </style>
                </head>
                <body>
                    <h1>Error Processing Dataset</h1>
                    <div class="error">{str(e)}</div>
                    <div class="back"><a href="/">Back to Dataset Selection</a></div>
                </body>
                </html>
                """
                self.wfile.write(error_html.encode())
            
            return
            
        elif self.path == '/save_annotation':
            # Read the request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            annotation = json.loads(post_data.decode('utf-8'))
            
            # Get the current dataset folder
            current_dataset = {}
            if os.path.exists("current_dataset.json"):
                with open("current_dataset.json", "r") as f:
                    current_dataset = json.load(f)
            
            # Get annotations file path from the current dataset info
            annotations_file = current_dataset.get("annotations_file")
            
            if not annotations_file:
                self.send_error(500, "No dataset is currently loaded")
                return
            
            # Load existing annotations
            annotations = []
            if os.path.exists(annotations_file):
                try:
                    with open(annotations_file, 'r') as f:
                        annotations = json.load(f)
                    print(f"Loaded {len(annotations)} existing annotations from {annotations_file}")
                except Exception as e:
                    print(f"Error loading annotations: {e}")
            
            # Find and update existing annotation or append new one
            conversation_id = annotation['conversation_id']
            existing_index = next(
                (i for i, a in enumerate(annotations) 
                 if a['conversation_id'] == conversation_id), 
                None
            )
            
            if existing_index is not None:
                annotations[existing_index] = annotation
                print(f"Updated existing annotation for {conversation_id}")
            else:
                annotations.append(annotation)
                print(f"Added new annotation for {conversation_id}")
            
            # Save back to file
            try:
                with open(annotations_file, 'w') as f:
                    json.dump(annotations, f, indent=2)
                print(f"Successfully saved annotations file with {len(annotations)} entries to {annotations_file}")
                # Send success response
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'success'}).encode())
            except Exception as e:
                print(f"Error saving annotations: {e}")
                self.send_error(500, f"Failed to save annotation: {str(e)}")
            return
        
        elif self.path == '/export_excel':
            try:
                # Read the request body
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                request_data = json.loads(post_data.decode('utf-8'))
                annotator = request_data.get('annotator', '')

                # Get the current dataset folder
                current_dataset = {}
                if os.path.exists("current_dataset.json"):
                    with open("current_dataset.json", "r") as f:
                        current_dataset = json.load(f)
                
                # Get file paths from the current dataset info
                annotations_file = current_dataset.get("annotations_file")
                conversations_file = current_dataset.get("conversations_file")
                
                if not annotations_file or not conversations_file:
                    raise Exception("No dataset is currently loaded")
                
                # Load annotations and conversations from the dataset folder
                with open(annotations_file, 'r') as f:
                    annotations = json.load(f)
                with open(conversations_file, 'r') as f:
                    conversations = json.load(f)
                
                # Create conversations lookup dict
                conv_lookup = {c['conversation_id']: c for c in conversations}
                
                # Convert annotations to DataFrame
                df = pd.DataFrame(annotations)
                
                # Reorder columns for better readability
                columns = [
                    'conversation_id', 'timestamp',
                    'annotator',
                    'language', 'factuality', 'knowledge', 'similarity',
                    'not_qa', 'invalid_gen', 'incomplete_orig',
                    'notes'
                ]
                
                # Add annotator column
                df['annotator'] = annotator
                
                df = df[columns]
                
                # Create conversations sheet with full content
                conv_rows = []
                for annotation in annotations:
                    conv_id = annotation['conversation_id']
                    if conv_id in conv_lookup:
                        conv = conv_lookup[conv_id]
                        # Format original conversation
                        orig_messages = conv.get('orig_messages', [])
                        orig_text = "\n".join([
                            f"Person {msg.get('user', '?')}: {msg.get('text', '')}" 
                            for msg in orig_messages
                        ])
                        
                        # Format synthetic conversation
                        synth_messages = conv.get('synthetic_messages', [])
                        synth_text = "\n".join([
                            f"Person {msg.get('user', '?')}: {msg.get('text', '')}" 
                            for msg in synth_messages
                        ])
                        
                        # Add to rows
                        conv_rows.append({
                            'conversation_id': conv_id,
                            'original_conversation': orig_text,
                            'synthetic_conversation': synth_text,
                            'annotator': annotator,
                            'language': annotation.get('language', ''),
                            'factuality': annotation.get('factuality', ''),
                            'knowledge': annotation.get('knowledge', ''),
                            'similarity': annotation.get('similarity', ''),
                            'not_qa': annotation.get('not_qa', False),
                            'invalid_gen': annotation.get('invalid_gen', False),
                            'incomplete_orig': annotation.get('incomplete_orig', False),
                            'notes': annotation.get('notes', '')
                        })
                
                # Create Excel file in memory
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    # Create conversations DataFrame as the only sheet
                    if conv_rows:
                        conv_df = pd.DataFrame(conv_rows)
                        conv_df.to_excel(writer, index=False, sheet_name='Annotations')
                        # Adjust column widths for better readability
                        worksheet = writer.sheets['Annotations']
                        worksheet.column_dimensions['B'].width = 100  # Original conversation
                        worksheet.column_dimensions['C'].width = 100  # Synthetic conversation
                
                # Save a copy of the Excel file to the dataset folder
                excel_file = os.path.join(os.path.dirname(annotations_file), f"annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
                with open(excel_file, 'wb') as f:
                    f.write(output.getvalue())
                print(f"Saved Excel file to {excel_file}")
                
                # Prepare response
                self.send_response(200)
                self.send_header('Content-Type', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
                self.send_header('Content-Disposition', 'attachment; filename=annotations.xlsx')
                self.end_headers()
                
                # Send file
                self.wfile.write(output.getvalue())
                
            except Exception as e:
                print(f"Export error: {str(e)}")  # Log the error
                self.send_error(500, str(e))
            return
        
        # Handle unknown POST requests
        self.send_error(404, "Endpoint not found")
        return

    def calculate_dataset_stats(self, conversations):
        # Initialize statistics
        message_counts = {
            'orig_counts': [],
            'synth_counts': [],
            'scatter_data': []
        }
        
        message_lengths = {
            'orig_lengths': [],
            'synth_lengths': [],
            'orig_words': [],
            'synth_words': []
        }
        
        user_counts = {
            'orig_users': [],
            'synth_users': []
        }
        
        # Extract context statistics directly from top-level fields
        context_stats = {
            'used_counts': [],
            'avail_counts': [],
            'tokens_used': [],
            'tokens_avail': []
        }
        
        # At the beginning of calculate_dataset_stats
        if conversations:
            print("First conversation keys:", list(conversations[0].keys()))
            if 'metadata' in conversations[0]:
                print("Metadata keys:", list(conversations[0]['metadata'].keys()))
            
            # Print the actual context values if they exist
            for key in ['context_msg_used', 'context_msg_available', 
                        'context_tokens_used', 'context_tokens_available']:
                if key in conversations[0]:
                    print(f"{key}: {conversations[0][key]}")
        
        # Process each conversation
        for conv in conversations:
            # Check if this is using the new format (synthetic_messages) or old format (synth_messages)
            orig_messages_field = 'orig_messages'
            synth_messages_field = 'synthetic_messages' if 'synthetic_messages' in conv else 'synth_messages'
            
            # Get message counts
            orig_count = len(conv.get(orig_messages_field, []))
            synth_count = len(conv.get(synth_messages_field, []))
            
            message_counts['orig_counts'].append(orig_count)
            message_counts['synth_counts'].append(synth_count)
            message_counts['scatter_data'].append({
                'x': orig_count,
                'y': synth_count,
                'id': conv.get('conversation_id', 'unknown')
            })
            
            # Get message lengths and word counts
            for msg in conv.get(orig_messages_field, []):
                text = msg.get('text', '')
                message_lengths['orig_lengths'].append(len(text))
                message_lengths['orig_words'].append(len(text.split()))
            
            for msg in conv.get(synth_messages_field, []):
                text = msg.get('text', '')
                message_lengths['synth_lengths'].append(len(text))
                message_lengths['synth_words'].append(len(text.split()))
            
            # Get unique users
            orig_users = set()
            synth_users = set()
            
            for msg in conv.get(orig_messages_field, []):
                # Try different fields that might contain user ID
                user_id = None
                if 'user' in msg:
                    user_id = msg['user']
                elif 'poster_id' in msg:
                    user_id = msg['poster_id']
                
                if user_id is not None:
                    # Convert to integer if it's a float
                    if isinstance(user_id, (int, float)):
                        user_id = int(float(user_id))
                    orig_users.add(user_id)
            
            for msg in conv.get(synth_messages_field, []):
                # Try different fields that might contain user ID
                user_id = None
                if 'user' in msg:
                    user_id = msg['user']
                elif 'poster_id' in msg:
                    user_id = msg['poster_id']
                
                if user_id is not None:
                    # Convert to integer if it's a float
                    if isinstance(user_id, (int, float)):
                        user_id = int(float(user_id))
                    synth_users.add(user_id)
            
            user_counts['orig_users'].append(len(orig_users))
            user_counts['synth_users'].append(len(synth_users))
            
            # Extract context statistics directly from top-level fields
            context_msg_used = conv.get('context_msg_used', 0)
            context_msg_available = conv.get('context_msg_available', 0)
            context_tokens_used = conv.get('context_tokens_used', 0)
            context_tokens_available = conv.get('context_tokens_available', 0)
            
            # Convert to integers if needed
            if isinstance(context_msg_used, str):
                try:
                    context_msg_used = int(context_msg_used)
                except ValueError:
                    context_msg_used = 0
            
            if isinstance(context_msg_available, str):
                try:
                    context_msg_available = int(context_msg_available)
                except ValueError:
                    context_msg_available = 0
            
            if isinstance(context_tokens_used, str):
                try:
                    context_tokens_used = int(context_tokens_used)
                except ValueError:
                    context_tokens_used = 0
            
            if isinstance(context_tokens_available, str):
                try:
                    context_tokens_available = int(context_tokens_available)
                except ValueError:
                    context_tokens_available = 0
            
            # Add to arrays
            context_stats['used_counts'].append(context_msg_used)
            context_stats['avail_counts'].append(context_msg_available)
            context_stats['tokens_used'].append(context_tokens_used)
            context_stats['tokens_avail'].append(context_tokens_available)
        
        # Calculate message count statistics
        orig_counts = np.array(message_counts['orig_counts'])
        synth_counts = np.array(message_counts['synth_counts'])
        
        message_counts.update({
            'orig_avg': float(np.mean(orig_counts)),
            'synth_avg': float(np.mean(synth_counts)),
            'orig_median': float(np.median(orig_counts)),
            'synth_median': float(np.median(synth_counts)),
            'orig_min': int(np.min(orig_counts)),
            'synth_min': int(np.min(synth_counts)),
            'orig_max': int(np.max(orig_counts)),
            'synth_max': int(np.max(synth_counts)),
            'correlation': float(np.corrcoef(orig_counts, synth_counts)[0, 1])
        })
        
        # Calculate message length statistics
        orig_lengths = np.array(message_lengths['orig_lengths'])
        synth_lengths = np.array(message_lengths['synth_lengths'])
        orig_words = np.array(message_lengths['orig_words'])
        synth_words = np.array(message_lengths['synth_words'])
        
        # Create histograms for message lengths with more bins
        bins = list(range(0, 2001, 25))  # 0-2000 in steps of 25 (80 bins instead of 20)
        orig_hist, _ = np.histogram(orig_lengths, bins=bins)
        synth_hist, _ = np.histogram(synth_lengths, bins=bins)
        
        message_lengths.update({
            'orig_chars_avg': float(np.mean(orig_lengths)),
            'synth_chars_avg': float(np.mean(synth_lengths)),
            'orig_words_avg': float(np.mean(orig_words)),
            'synth_words_avg': float(np.mean(synth_words)),
            'orig_chars_total': int(np.sum(orig_lengths)),
            'synth_chars_total': int(np.sum(synth_lengths)),
            'orig_words_total': int(np.sum(orig_words)),
            'synth_words_total': int(np.sum(synth_words)),
            'bins': bins[:-1],  # Remove the last bin edge
            'orig_hist': orig_hist.tolist(),
            'synth_hist': synth_hist.tolist()
        })
        
        # Calculate user count statistics
        orig_user_counts = Counter(user_counts['orig_users'])
        synth_user_counts = Counter(user_counts['synth_users'])
        
        # Group counts into bins (1, 2, 3, 4, 5, 6+)
        orig_counts_binned = [
            orig_user_counts.get(1, 0),
            orig_user_counts.get(2, 0),
            orig_user_counts.get(3, 0),
            orig_user_counts.get(4, 0),
            orig_user_counts.get(5, 0),
            sum(orig_user_counts.get(i, 0) for i in range(6, 100))
        ]
        
        synth_counts_binned = [
            synth_user_counts.get(1, 0),
            synth_user_counts.get(2, 0),
            synth_user_counts.get(3, 0),
            synth_user_counts.get(4, 0),
            synth_user_counts.get(5, 0),
            sum(synth_user_counts.get(i, 0) for i in range(6, 100))
        ]
        
        user_counts.update({
            'orig_counts': orig_counts_binned,
            'synth_counts': synth_counts_binned
        })
        
        # Calculate context statistics similar to message count statistics
        used_counts = np.array(context_stats['used_counts'])
        avail_counts = np.array(context_stats['avail_counts'])
        tokens_used = np.array(context_stats['tokens_used'])
        tokens_avail = np.array(context_stats['tokens_avail'])
        
        # Only calculate if we have data
        if len(used_counts) > 0:
            context_stats.update({
                'used_avg': float(np.mean(used_counts)),
                'avail_avg': float(np.mean(avail_counts)),
                'used_median': float(np.median(used_counts)),
                'avail_median': float(np.median(avail_counts)),
                'used_min': int(np.min(used_counts)),
                'avail_min': int(np.min(avail_counts)),
                'used_max': int(np.max(used_counts)),
                'avail_max': int(np.max(avail_counts)),
                'used_sum': int(np.sum(used_counts)),
                'avail_sum': int(np.sum(avail_counts)),
                'tokens_used_avg': float(np.mean(tokens_used)),
                'tokens_avail_avg': float(np.mean(tokens_avail)),
                'tokens_used_min': int(np.min(tokens_used)),
                'tokens_avail_min': int(np.min(tokens_avail)),
                'tokens_used_max': int(np.max(tokens_used)),
                'tokens_avail_max': int(np.max(tokens_avail)),
                'tokens_used_sum': int(np.sum(tokens_used)),
                'tokens_avail_sum': int(np.sum(tokens_avail)),
                'usage_ratio': float(np.mean(used_counts) / np.mean(avail_counts)) if np.mean(avail_counts) > 0 else 0,
                'token_usage': float(np.mean(tokens_used) / np.mean(tokens_avail)) if np.mean(tokens_avail) > 0 else 0
            })
        else:
            # Provide default values if no data
            context_stats.update({
                'used_avg': 0.0,
                'avail_avg': 0.0,
                'used_median': 0.0,
                'avail_median': 0.0,
                'used_min': 0,
                'avail_min': 0,
                'used_max': 0,
                'avail_max': 0,
                'used_sum': 0,
                'avail_sum': 0,
                'tokens_used_avg': 0.0,
                'tokens_avail_avg': 0.0,
                'tokens_used_min': 0,
                'tokens_avail_min': 0,
                'tokens_used_max': 0,
                'tokens_avail_max': 0,
                'tokens_used_sum': 0,
                'tokens_avail_sum': 0,
                'usage_ratio': 0.0,
                'token_usage': 0.0
            })
        
        # Add dataset info to the response
        dataset_info = {
            'dataset_id': 'Unknown',
            'conversation_count': len(conversations)
        }

        # Try to get the current dataset info
        try:
            if os.path.exists("current_dataset.json"):
                with open("current_dataset.json", "r") as f:
                    current_dataset = json.load(f)
                    dataset_info['dataset_id'] = current_dataset.get('dataset_id', 'Unknown')
        except Exception as e:
            print(f"Error loading current dataset info: {e}")

        # Return all statistics
        return {
            'message_counts': message_counts,
            'message_lengths': message_lengths,
            'user_counts': user_counts,
            'context_stats': context_stats,
            'dataset_info': dataset_info,
            'conversations': conversations  # Include full conversations for additional processing
        }

    def process_huggingface_dataset(self, dataset_name, dataset_id):
        """Process a HuggingFace dataset for inspection."""
        try:
            # Load the dataset
            dataset = load_dataset(dataset_name)
            
            # Convert to pandas DataFrame for easier processing
            df = dataset['train'].to_pandas()
            
            # Check if this is the new format dataset
            is_new_format = all(col in df.columns for col in [
                'orig_messages', 'synthetic_messages', 'context_msg_used', 
                'context_msg_available', 'context_tokens_used', 'context_tokens_available'
            ])
            
            # Create output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"datasets/{timestamp}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Process the dataset based on its format
            if is_new_format:
                # New format - direct mapping
                conversations = []
                
                for _, row in df.iterrows():
                    # Convert row to dict for easier access
                    row_dict = row.to_dict()
                    
                    # Create conversation entry
                    conv = {
                        'conversation_id': str(row_dict['conversation_id']),
                        'orig_messages': row_dict['orig_messages'],
                        'synthetic_messages': row_dict['synthetic_messages'],
                        # Add context statistics as top-level fields
                        'context_msg_used': row_dict['context_msg_used'],
                        'context_msg_available': row_dict['context_msg_available'],
                        'context_tokens_used': row_dict['context_tokens_used'],
                        'context_tokens_available': row_dict['context_tokens_available']
                    }
                    
                    # Add metadata if available
                    if 'metadata' in row_dict and row_dict['metadata'] is not None:
                        # If metadata is a string, try to parse it as JSON
                        if isinstance(row_dict['metadata'], str):
                            try:
                                conv['metadata'] = json.loads(row_dict['metadata'])
                            except json.JSONDecodeError:
                                conv['metadata'] = {'model': row_dict.get('model', 'unknown')}
                        else:
                            conv['metadata'] = row_dict['metadata']
                    else:
                        conv['metadata'] = {'model': row_dict.get('model', 'unknown')}
                    
                    conversations.append(conv)
            else:
                # Old format - needs conversion
                # ... (existing code for old format)
                pass
            
            # Save conversations to file
            conversations_file = os.path.join(output_dir, "conversations.json")
            with open(conversations_file, 'w', encoding='utf-8') as f:
                json.dump(conversations, f, ensure_ascii=False, indent=2)
            
            # Create empty annotations file
            annotations_file = os.path.join(output_dir, "annotations.json")
            with open(annotations_file, 'w', encoding='utf-8') as f:
                json.dump({}, f)
            
            # Create dataset info file
            dataset_info = {
                "dataset_id": dataset_id,
                "source": dataset_name,
                "timestamp": timestamp,
                "folder": output_dir,
                "conversations_file": conversations_file,
                "annotations_file": annotations_file
            }
            
            info_file = os.path.join(output_dir, "dataset_info.json")
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(dataset_info, f, indent=2)
            
            # Set as current dataset
            with open("current_dataset.json", 'w', encoding='utf-8') as f:
                json.dump(dataset_info, f, indent=2)
            
            return dataset_info
        except Exception as e:
            print(f"Error processing HuggingFace dataset: {e}")
            raise

def run(port=8000):
    # Create datasets directory if it doesn't exist
    os.makedirs("datasets", exist_ok=True)
    
    # Create a default dataset folder if needed
    default_folder = os.path.join("datasets", "default")
    os.makedirs(default_folder, exist_ok=True)
    
    # If no current dataset exists, create a default one
    if not os.path.exists("current_dataset.json"):
        default_info = {
            "dataset_id": "default",
            "folder": default_folder,
            "annotations_file": os.path.join(default_folder, "annotations.json"),
            "conversations_file": os.path.join(default_folder, "conversations.json")
        }
        with open("current_dataset.json", "w") as f:
            json.dump(default_info, f, indent=2)
    
    server_address = ('', port)
    httpd = HTTPServer(server_address, AnnotationHandler)
    print(f'Starting server on port {port}...')
    httpd.serve_forever()

if __name__ == '__main__':
    run() 