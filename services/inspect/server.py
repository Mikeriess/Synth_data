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
        
        # Process each conversation
        for conv in conversations:
            # Get message counts
            orig_count = len(conv.get('orig_messages', []))
            synth_count = len(conv.get('synthetic_messages', []))
            
            message_counts['orig_counts'].append(orig_count)
            message_counts['synth_counts'].append(synth_count)
            message_counts['scatter_data'].append({
                'x': orig_count,
                'y': synth_count,
                'id': conv.get('conversation_id', 'unknown')
            })
            
            # Get message lengths and word counts
            for msg in conv.get('orig_messages', []):
                text = msg.get('text', '')
                message_lengths['orig_lengths'].append(len(text))
                message_lengths['orig_words'].append(len(text.split()))
            
            for msg in conv.get('synthetic_messages', []):
                text = msg.get('text', '')
                message_lengths['synth_lengths'].append(len(text))
                message_lengths['synth_words'].append(len(text.split()))
            
            # Get unique users
            orig_users = set()
            synth_users = set()
            
            for msg in conv.get('orig_messages', []):
                if 'user' in msg:
                    user_id = msg['user']
                    if isinstance(user_id, (int, float)):
                        user_id = int(float(user_id))
                    orig_users.add(user_id)
            
            for msg in conv.get('synthetic_messages', []):
                if 'user' in msg:
                    user_id = msg['user']
                    if isinstance(user_id, (int, float)):
                        user_id = int(float(user_id))
                    synth_users.add(user_id)
            
            user_counts['orig_users'].append(len(orig_users))
            user_counts['synth_users'].append(len(synth_users))
        
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
        
        # Create histograms for message lengths
        bins = list(range(0, 1001, 50))  # 0-1000 in steps of 50
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
        
        # Add context statistics if available
        context_stats = {
            'messages_used': [],
            'messages_total': [],
            'tokens_used': [],
            'max_tokens': []
        }
        
        # Extract context statistics from dataset fields or metadata
        for conv in conversations:
            # First try to get from the direct fields (new format)
            if 'context_msg_used' in conv and 'context_msg_available' in conv:
                context_stats['messages_used'].append(conv.get('context_msg_used', 0))
                context_stats['messages_total'].append(conv.get('context_msg_available', 0))
                context_stats['tokens_used'].append(conv.get('context_tokens_used', 0))
                context_stats['max_tokens'].append(conv.get('context_tokens_available', 0))
            # Fall back to metadata (old format)
            elif 'metadata' in conv and 'context_stats' in conv['metadata']:
                stats = conv['metadata']['context_stats']
                context_stats['messages_used'].append(stats.get('messages_used', 0))
                context_stats['messages_total'].append(stats.get('total_messages', 0))
                context_stats['tokens_used'].append(stats.get('tokens_used', 0))
                context_stats['max_tokens'].append(stats.get('max_tokens', 0))
        
        # Calculate averages if data exists
        if context_stats['messages_used']:
            context_stats['avg_messages_used'] = sum(context_stats['messages_used']) / len(context_stats['messages_used'])
            context_stats['avg_messages_total'] = sum(context_stats['messages_total']) / len(context_stats['messages_total'])
            context_stats['avg_tokens_used'] = sum(context_stats['tokens_used']) / len(context_stats['tokens_used'])
            context_stats['avg_max_tokens'] = sum(context_stats['max_tokens']) / len(context_stats['max_tokens'])
            context_stats['usage_ratio'] = context_stats['avg_messages_used'] / context_stats['avg_messages_total'] if context_stats['avg_messages_total'] > 0 else 0
            context_stats['token_usage'] = context_stats['avg_tokens_used'] / context_stats['avg_max_tokens'] if context_stats['avg_max_tokens'] > 0 else 0
        
        # Return all statistics
        return {
            'message_counts': message_counts,
            'message_lengths': message_lengths,
            'user_counts': user_counts,
            'context_stats': context_stats,
            'conversations': conversations  # Include full conversations for additional processing
        }

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