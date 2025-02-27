import pytest
import os
import json
from generate_dialogues import load_experiment_config

def test_load_experiment_config():
    """Test loading configuration from experiment folder"""
    
    # Test valid experiment
    config = load_experiment_config('LM_dialogues1')
    assert isinstance(config, dict)
    assert 'generation_config' in config
    assert 'dataset_config' in config
    assert 'files' in config

def test_nonexistent_experiment():
    """Test handling of non-existent experiment folder"""
    with pytest.raises(ValueError) as exc_info:
        load_experiment_config('nonexistent_experiment')
    assert "Experiment folder 'experiments/nonexistent_experiment' does not exist" in str(exc_info.value)

def test_missing_config_file():
    """Test handling of missing config file in experiment folder"""
    # Create temporary empty experiment folder
    os.makedirs('experiments/empty_experiment', exist_ok=True)
    
    with pytest.raises(ValueError) as exc_info:
        load_experiment_config('empty_experiment')
    assert "Config file not found in experiments/empty_experiment/generation_config.json" in str(exc_info.value)
    
    # Cleanup
    os.rmdir('experiments/empty_experiment')

def test_experiment_specific_prompt():
    """Test loading experiment-specific prompt file"""
    # Create temporary experiment with custom prompt
    exp_path = 'experiments/test_experiment'
    os.makedirs(exp_path, exist_ok=True)
    
    # Create config file
    config = {
        "generation_config": {"model": "test-model"},
        "dataset_config": {"source_dataset": "test/dataset"},
        "files": {"prompt_file": "prompts/dialogue_prompt.txt"}
    }
    with open(f"{exp_path}/generation_config.json", 'w') as f:
        json.dump(config, f)
    
    # Create custom prompt file
    with open(f"{exp_path}/dialogue_prompt.txt", 'w') as f:
        f.write("Custom prompt")
    
    # Test loading
    loaded_config = load_experiment_config('test_experiment')
    assert loaded_config['files']['prompt_file'] == f"{exp_path}/dialogue_prompt.txt"
    
    # Cleanup
    os.remove(f"{exp_path}/generation_config.json")
    os.remove(f"{exp_path}/dialogue_prompt.txt")
    os.rmdir(exp_path)

def test_config_structure():
    """Test that loaded config has required fields"""
    config = load_experiment_config('LM_dialogues1')
    
    # Check generation config
    assert 'model' in config['generation_config']
    assert 'temperature' in config['generation_config']
    assert 'max_tokens' in config['generation_config']
    
    # Check dataset config
    assert 'source_dataset' in config['dataset_config']
    assert 'output_dataset_name' in config['dataset_config']
    assert 'private' in config['dataset_config']
    
    # Check files config
    assert 'prompt_file' in config['files']
    assert 'checkpoint_dir' in config['files']

if __name__ == '__main__':
    pytest.main([__file__]) 