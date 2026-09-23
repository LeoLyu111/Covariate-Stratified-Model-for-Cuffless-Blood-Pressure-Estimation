import os
import argparse
import numpy as np
from utils.common import set_random_seeds, get_device
from experiments.runner import run_experiment

def main():
    parser = argparse.ArgumentParser(description='BP Estimation using Transformer Models with Comprehensive Analysis')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing processed data')
    parser.add_argument(
        '--data_format', type=str, default='subject_cv',
        choices=['subject_cv', 'measurements'],
        help='Input layout: subject CV folds or measurement session files'
    )
    parser.add_argument('--fold', type=int, default=None, help='Fold number for subject_cv data (1-5)')
    parser.add_argument(
        '--train_measurements', type=int, nargs='+', default=[1, 2, 3],
        help='Measurement sessions used for training'
    )
    parser.add_argument(
        '--test_measurements', type=int, nargs='+', default=[4],
        help='Measurement sessions used for testing'
    )
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory for results')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, required=True, 
                       choices=['base', 'full_hypertension', 'full_all', 
                               'stratified_hypertension', 'stratified_gender', 'stratified_hypertension_gender'],
                       help='Model type to train')
    parser.add_argument('--input_signals', type=str, default='ppg_ecg', choices=['ppg_ecg', 'ppg'],
                       help='Input signals to use: ppg_ecg (both) or ppg (only PPG)')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    
    # Model architecture arguments
    parser.add_argument('--d_model', type=int, default=128, help='Transformer d_model')
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=2, help='Number of encoder layers')
    parser.add_argument('--dim_feedforward', type=int, default=128, help='Feedforward dimension')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()

    if args.data_format == 'subject_cv' and args.fold is None:
        parser.error('--fold is required when --data_format subject_cv')

    if args.data_format == 'measurements':
        train_tag = '-'.join(str(value) for value in args.train_measurements)
        test_tag = '-'.join(str(value) for value in args.test_measurements)
        run_name = f'measurements_train_{train_tag}_test_{test_tag}'
    else:
        run_name = f'fold_{args.fold}'
    
    # Set random seeds
    set_random_seeds(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print configuration
    print("=" * 50)
    print("BP Estimation Experiment Configuration")
    print("=" * 50)
    print(f"Data directory: {args.data_dir}")
    print(f"Data format: {args.data_format}")
    if args.data_format == 'measurements':
        print(f"Train measurements: {args.train_measurements}")
        print(f"Test measurements: {args.test_measurements}")
    else:
        print(f"Fold: {args.fold}")
    print(f"Model type: {args.model_type}")
    print(f"Input signals: {args.input_signals}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"d_model: {args.d_model}")
    print(f"Number of heads: {args.nhead}")
    print(f"Encoder layers: {args.num_encoder_layers}")
    print(f"Feedforward dim: {args.dim_feedforward}")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    print(f"Device: {get_device()}")
    print("=" * 50)
    
    # Run experiment
    results = run_experiment(
        data_dir=args.data_dir,
        fold_number=args.fold,
        model_type=args.model_type,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        dim_feedforward=args.dim_feedforward,
        input_signals=args.input_signals,
        num_epochs=args.epochs,
        data_format=args.data_format,
        train_measurements=args.train_measurements,
        test_measurements=args.test_measurements,
        run_name=run_name
    )
    
    # Save results
    results_file = os.path.join(
        args.output_dir,
        f'{run_name}_{args.model_type}_{args.input_signals}_results.npy'
    )
    np.save(results_file, results)
    
    print(f"\nExperiment completed!")
    print(f"Results saved to: {results_file}")

if __name__ == "__main__":
    main()
