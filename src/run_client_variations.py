#!/usr/bin/env python3
"""
Script to test federated learning configuration with varying number of clients (1-15).
Keeps all other parameters constant and only modifies num_clients.
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

def run_experiment(config_path, num_clients):
    """Run a single experiment with specified number of clients."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Starting experiment with {num_clients} client(s)")
    logger.info(f"{'='*80}\n")
    
    # Load current config
    config = load_config(config_path)
    original_clients = config['num_clients']
    
    # Update num_clients
    config['num_clients'] = num_clients
    
    # Update experiment name to include client count
    base_experiment_name = config['experiment_name'].split('_clients_')[0]
    config['experiment_name'] = f"{base_experiment_name}_clients_{num_clients}"
    
    logger.info(f"Config updated: num_clients={num_clients}, experiment_name={config['experiment_name']}")
    
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
            logger.info(f"✓ Experiment with {num_clients} client(s) completed successfully")
            return True
        else:
            logger.error(f"✗ Experiment with {num_clients} client(s) failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Error running experiment with {num_clients} client(s): {str(e)}")
        return False
    finally:
        # Restore original config
        config['num_clients'] = original_clients
        config['experiment_name'] = base_experiment_name
        save_config(config_path, config)

def main():
    """Main function to run all experiments."""
    config_path = 'config/config.json'
    
    # Validate config file exists
    if not Path(config_path).exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    # Test with client counts from 5 to 12
    client_counts = list(range(5, 13))
    results = {}
    
    logger.info(f"Starting experiments with varying client counts: {client_counts}")
    
    for num_clients in client_counts:
        success = run_experiment(config_path, num_clients)
        results[num_clients] = 'SUCCESS' if success else 'FAILED'
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("EXPERIMENT SUMMARY")
    logger.info(f"{'='*80}")
    
    for num_clients in client_counts:
        status = results[num_clients]
        symbol = "✓" if status == 'SUCCESS' else "✗"
        logger.info(f"{symbol} Clients: {num_clients:2d} - {status}")
    
    # Count successes and failures
    successes = sum(1 for v in results.values() if v == 'SUCCESS')
    failures = sum(1 for v in results.values() if v == 'FAILED')
    
    logger.info(f"\nTotal: {successes} successful, {failures} failed out of {len(client_counts)} experiments")
    logger.info(f"{'='*80}\n")

if __name__ == '__main__':
    main()
