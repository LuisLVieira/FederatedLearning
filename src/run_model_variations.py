#!/usr/bin/env python3
"""
Script to test federated learning configuration with varying model.
Keeps all other parameters constant and only modifies model.
"""

import json
import subprocess
import sys
import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def save_config(config_path, config):
    """Save configuration to JSON file."""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

def run_experiment(config_path, model):
    """Run a single experiment with specified number of model."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Starting experiment with {model}")
    logger.info(f"{'='*80}\n")
    
    # Load current config
    config = load_config(config_path)
    original_model = config['model_config']['model']
    
    # Update model
    config['model_config']['model'] = model

    # Update experiment name to include model
    base_experiment_name = ('_').join(config['experiment_name'].split('_')[:1])
    config['experiment_name'] = f"{base_experiment_name}_{model}"
    
    logger.info(f"Config updated: model={model}, experiment_name={config['experiment_name']}")
    
    # Save modified config
    save_config(config_path, config)
    
    try:
        # Run main.py with the modified config
        result = subprocess.run(
            [sys.executable, 'main.py', '--config', config_path],
            cwd=Path(config_path).parent.parent,
            capture_output=False
        )
        
        if result.returncode == 0:
            logger.info(f"✓ Experiment with {model} completed successfully")
            return True
        else:
            logger.error(f"✗ Experiment with {model} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Error running experiment with {model}: {str(e)}")
        return False
    finally:
        # Restore original config
        config['model_config']['model'] = original_model
        config['experiment_name'] = base_experiment_name
        save_config(config_path, config)

def main():
    """Main function to run all experiments."""
    config_path = 'config/config.json'
    
    # Validate config file exists
    if not Path(config_path).exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    models = [
        "simple_tl_resnet18",
        "custom_fc_resnet18",
        "efficientnet_b0",
        "mobilenet_v3_small",
        "custom_layer4_fc_resnet18",
    ]
    results = {}
    
    logger.info(f"Starting experiments with varying model: {models}")
    
    for model in models:
        success = run_experiment(config_path, model)
        results[model] = 'SUCCESS' if success else 'FAILED'
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("EXPERIMENT SUMMARY")
    logger.info(f"{'='*80}")
    
    for model in models:
        status = results[model]
        symbol = "✓" if status == 'SUCCESS' else "✗"
        logger.info(f"{symbol} model: {model:2d} - {status}")
    
    # Count successes and failures
    successes = sum(1 for v in results.values() if v == 'SUCCESS')
    failures = sum(1 for v in results.values() if v == 'FAILED')
    
    logger.info(f"\nTotal: {successes} successful, {failures} failed out of {len(models)} experiments")
    logger.info(f"{'='*80}\n")

if __name__ == '__main__':
    main()
