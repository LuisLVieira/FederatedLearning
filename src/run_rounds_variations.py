#!/usr/bin/env python3
"""
Script to test federated learning configuration with varying number of rounds (1-15).
Keeps all other parameters constant and only modifies rounds.
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

def run_experiment(config_path, rounds):
    """Run a single experiment with specified number of rounds."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Starting experiment with {rounds} round (s)")
    logger.info(f"{'='*80}\n")
    
    # Load current config
    config = load_config(config_path)
    original_rounds = config['rounds']
    
    # Update rounds
    config['rounds'] = rounds

    
    # Update experiment name to include round  count
    base_experiment_name = config['experiment_name'].split('_rounds_')[0]
    config['experiment_name'] = f"{base_experiment_name}_rounds_{rounds}"
    
    logger.info(f"Config updated: rounds={rounds}, experiment_name={config['experiment_name']}")
    
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
            logger.info(f"✓ Experiment with {rounds} round (s) completed successfully")
            return True
        else:
            logger.error(f"✗ Experiment with {rounds} round (s) failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Error running experiment with {rounds} round (s): {str(e)}")
        return False
    finally:
        # Restore original config
        config['rounds'] = original_rounds
        config['experiment_name'] = base_experiment_name
        save_config(config_path, config)

def main():
    """Main function to run all experiments."""
    config_path = 'config/config.json'
    
    # Validate config file exists
    if not Path(config_path).exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    # Test with rounds from 10 to 23
    rounds_list = list(range(10, 24))
    results = {}
    
    logger.info(f"Starting experiments with varying rounds: {rounds_list}")
    
    for rounds in rounds_list:
        success = run_experiment(config_path, rounds)
        results[rounds] = 'SUCCESS' if success else 'FAILED'
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("EXPERIMENT SUMMARY")
    logger.info(f"{'='*80}")
    
    for rounds in rounds_list:
        status = results[rounds]
        symbol = "✓" if status == 'SUCCESS' else "✗"
        logger.info(f"{symbol} rounds: {rounds:2d} - {status}")
    
    # Count successes and failures
    successes = sum(1 for v in results.values() if v == 'SUCCESS')
    failures = sum(1 for v in results.values() if v == 'FAILED')
    
    logger.info(f"\nTotal: {successes} successful, {failures} failed out of {len(rounds_list)} experiments")
    logger.info(f"{'='*80}\n")

if __name__ == '__main__':
    main()
