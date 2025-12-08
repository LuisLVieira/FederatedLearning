#!/usr/bin/env python3
"""
Script to test federated learning configuration with varying aggregator.
Keeps all other parameters constant and only modifies aggregator.
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

def run_experiment(config_path, aggregation):
    """Run a single experiment with specified number of aggregation."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Starting experiment with {aggregation}")
    logger.info(f"{'='*80}\n")
    
    # Load current config
    config = load_config(config_path)
    original_aggregation = config['aggregation']
    
    # Update aggregation
    config['aggregation'] = aggregation

    
    # Update experiment name to include aggregator
    config['experiment_name'] = f"{aggregation}_KidneyData"
    
    logger.info(f"Config updated: aggregation={aggregation}, experiment_name={config['experiment_name']}")
    
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
            logger.info(f"✓ Experiment with {aggregation} epoch(s) completed successfully")
            return True
        else:
            logger.error(f"✗ Experiment with {aggregation} epoch(s) failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Error running experiment with {aggregation}: {str(e)}")
        return False
    finally:
        # Restore original config
        config['aggregation'] = original_aggregation
        config['experiment_name'] = config['experiment_name']
        save_config(config_path, config)

def main():
    """Main function to run all experiments."""
    config_path = 'config/config.json'
    
    # Validate config file exists
    if not Path(config_path).exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    # Test with aggregators from 1 to 10
    aggregators = ['fedadagrad', 'fedadam', 'fedyogi', 'krum', 'dp_fedavg_adaptive']
    results = {}
    
    logger.info(f"Starting experiments with varying aggregators: {aggregators}")
    
    for aggregation in aggregators:
        success = run_experiment(config_path, aggregation)
        results[aggregation] = 'SUCCESS' if success else 'FAILED'
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("EXPERIMENT SUMMARY")
    logger.info(f"{'='*80}")
    
    for aggregation in aggregators:
        status = results[aggregation]
        symbol = "✓" if status == 'SUCCESS' else "✗"
        logger.info(f"{symbol} aggregation: {aggregation:20s} - {status}")
    
    # Count successes and failures
    successes = sum(1 for v in results.values() if v == 'SUCCESS')
    failures = sum(1 for v in results.values() if v == 'FAILED')
    
    logger.info(f"\nTotal: {successes} successful, {failures} failed out of {len(aggregators)} experiments")
    logger.info(f"{'='*80}\n")

if __name__ == '__main__':
    main()
