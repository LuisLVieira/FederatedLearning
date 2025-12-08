#!/usr/bin/env python3
"""
Script to test federated learning configuration with varying number of epochs (1-15).
Keeps all other parameters constant and only modifies num_epochs.
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

def run_experiment(config_path, num_epochs):
    """Run a single experiment with specified number of epochs."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Starting experiment with {num_epochs} epoch(s)")
    logger.info(f"{'='*80}\n")
    
    # Load current config
    config = load_config(config_path)
    original_epochs = config['model_config']['epochs']
    
    # Update num_epochs
    config['model_config']['epochs'] = num_epochs

    
    # Update experiment name to include epoch count
    base_experiment_name = config['experiment_name'].split('_epochs_')[0]
    config['experiment_name'] = f"{base_experiment_name}_epochs_{num_epochs}"
    
    logger.info(f"Config updated: num_epochs={num_epochs}, experiment_name={config['experiment_name']}")
    
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
            logger.info(f"✓ Experiment with {num_epochs} epoch(s) completed successfully")
            return True
        else:
            logger.error(f"✗ Experiment with {num_epochs} epoch(s) failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Error running experiment with {num_epochs} epoch(s): {str(e)}")
        return False
    finally:
        # Restore original config
        config['model_config']['epochs'] = original_epochs
        config['experiment_name'] = base_experiment_name
        save_config(config_path, config)

def main():
    """Main function to run all experiments."""
    config_path = 'config/config.json'
    
    # Validate config file exists
    if not Path(config_path).exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    # Test with epoch counts from 1 to 10
    epoch_counts = list(range(1, 11))
    results = {}
    
    logger.info(f"Starting experiments with varying epoch counts: {epoch_counts}")
    
    for num_epochs in epoch_counts:
        success = run_experiment(config_path, num_epochs)
        results[num_epochs] = 'SUCCESS' if success else 'FAILED'
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("EXPERIMENT SUMMARY")
    logger.info(f"{'='*80}")
    
    for num_epochs in epoch_counts:
        status = results[num_epochs]
        symbol = "✓" if status == 'SUCCESS' else "✗"
        logger.info(f"{symbol} epochs: {num_epochs:2d} - {status}")
    
    # Count successes and failures
    successes = sum(1 for v in results.values() if v == 'SUCCESS')
    failures = sum(1 for v in results.values() if v == 'FAILED')
    
    logger.info(f"\nTotal: {successes} successful, {failures} failed out of {len(epoch_counts)} experiments")
    logger.info(f"{'='*80}\n")

if __name__ == '__main__':
    main()
