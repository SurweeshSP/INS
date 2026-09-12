import os
import subprocess
import sys

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    
    scripts_to_run = [
        os.path.join(models_dir, 'model1_bilstm_gru', 'train.py'),
        os.path.join(models_dir, 'model2_transformer_cnn', 'train.py'),
        os.path.join(models_dir, 'model3_wavelet_lstm', 'train.py')
    ]
    
    for script in scripts_to_run:
        model_name = os.path.basename(os.path.dirname(script))
        print(f"\n==================================================")
        print(f"Starting training for {model_name}")
        print(f"==================================================\n")
        
        if not os.path.exists(script):
            print(f"Error: {script} does not exist.")
            continue
            
        try:
            # Run the training script in its own directory so relative paths work
            script_dir = os.path.dirname(script)
            subprocess.run([sys.executable, script], cwd=script_dir, check=True)
            print(f"\nSuccessfully finished training {model_name}")
        except subprocess.CalledProcessError as e:
            print(f"\nError occurred while training {model_name}: {e}")
            print(f"Exiting train_all.py early due to error.")
            sys.exit(1)
        except Exception as e:
            print(f"\nUnexpected error occurred: {e}")
            sys.exit(1)

    print("\nAll model training completed successfully!")

if __name__ == "__main__":
    main()
