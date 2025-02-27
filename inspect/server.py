from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import os
from datetime import datetime
import pandas as pd
import io

class AnnotationHandler(SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/save_annotation':
            # Read the request body
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            annotation = json.loads(post_data.decode('utf-8'))
            
            # Load existing annotations if file exists
            annotations = []
            if os.path.exists('annotations.json'):
                with open('annotations.json', 'r') as f:
                    annotations = json.load(f)
            
            # Find and update existing annotation or append new one
            conversation_id = annotation['conversation_id']
            existing_index = next(
                (i for i, a in enumerate(annotations) 
                 if a['conversation_id'] == conversation_id), 
                None
            )
            
            if existing_index is not None:
                annotations[existing_index] = annotation
            else:
                annotations.append(annotation)
            
            # Save back to file
            with open('annotations.json', 'w') as f:
                json.dump(annotations, f, indent=2)
            
            # Send response
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'success'}).encode())
            return
        
        elif self.path == '/export_excel':
            try:
                # Read the request body
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                request_data = json.loads(post_data.decode('utf-8'))
                annotator = request_data.get('annotator', '')

                # Load annotations and conversations
                with open('annotations.json', 'r') as f:
                    annotations = json.load(f)
                with open('conversations.json', 'r') as f:
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
        
        return SimpleHTTPRequestHandler.do_POST(self)

def run(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, AnnotationHandler)
    print(f'Starting server on port {port}...')
    httpd.serve_forever()

if __name__ == '__main__':
    run() 