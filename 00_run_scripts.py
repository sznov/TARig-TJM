import argparse
import subprocess

def main():
    # -- Run all scripts --
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--root-dir', type=str, required=True)
    arg_parser.add_argument('--skip-pretrain-attn', action='store_true')
    arg_parser.add_argument('--skip-gen-binvox', action='store_true')
    arg_parser.add_argument('--skip-gen-dataset', action='store_true')
    args = arg_parser.parse_args()
    root_dir = args.root_dir
    
    if not args.skip_pretrain_attn:
        print('Running compute_pretrain_attn.py')
        # Compute pretrained attention
        subprocess.run(['python', '-m', 'geometric_proc.compute_pretrain_attn', 
                        '--dataset-folder', root_dir])

    if not args.skip_gen_binvox:
        print('Running gen_binvox.py')
        # Generate binvox
        subprocess.run(['python', 'gen_binvox.py', '--dataset-folder', root_dir])

    if not args.skip_gen_dataset:
        print('Running gen_dataset.py')   
        # Generate dataset (vertices, edges, joints, etc.) 
        subprocess.run(['python', 'gen_dataset.py', '--dataset-folder', root_dir])

if __name__ == '__main__':

    main()

