from flask import Flask, render_template, request, redirect, url_for, jsonify
import subprocess
import os
import json
from dataset_processor import process_dataset

app = Flask(__name__)

# Existing routes and functionality
# ...

@app.route('/')
def index():
    """Landing page with dataset selection form"""
    return render_template('dataset_selection.html')

@app.route('/process_dataset', methods=['POST'])
def process_dataset_route():
    """Handle dataset processing request"""
    dataset_id = request.form.get('dataset_id')
    config = request.form.get('config') or None
    split = request.form.get('split') or 'train'
    max_samples = request.form.get('max_samples')
    
    if max_samples:
        try:
            max_samples = int(max_samples)
        except ValueError:
            max_samples = None
    
    try:
        # Process the dataset
        output_file = process_dataset(dataset_id, config, split, max_samples)
        
        # Update the configuration to use the processed dataset
        with open('prompts/generation_config.json', 'r') as f:
            config = json.load(f)
        
        # Update the config to point to the new dataset
        config['data_path'] = output_file
        
        with open('prompts/generation_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        # Redirect to the annotation interface
        return redirect(url_for('annotation_interface'))
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/annotation')
def annotation_interface():
    """Existing annotation interface"""
    # This should point to your existing annotation route
    # Replace with your actual route
    return redirect(url_for('existing_annotation_route'))

if __name__ == '__main__':
    app.run(debug=True) 