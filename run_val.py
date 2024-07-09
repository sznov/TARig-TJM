import argparse
import math
import os
import subprocess


def main():
    # -- Run all scripts --

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--root-dir', type=str) 
    args = arg_parser.parse_args()

    root_dir = args.root_dir

    run_primary_joint_val(root_dir)


def run_primary_joint_val(root_dir):
    train_dir = os.path.join(root_dir, 'train')
    val_dir = os.path.join(root_dir, 'val')
    test_dir = os.path.join(root_dir, 'test')
    checkpoint_dir = os.path.join(root_dir, 'checkpoints', 'tarig_prim_joint')
    log_dir = os.path.join(root_dir, 'logs', 'tarig_prim_joint')

    for dir_path in [train_dir, val_dir, test_dir, checkpoint_dir, log_dir]:
        os.makedirs(dir_path, exist_ok=True)

    print('Running run_primary_joint_train.py')
    # Run primary training
    subprocess.run(['python', 
                    '-u', 
                    'run_primary_joint_train.py',
                    '-e',
                    '--resume', os.path.join(checkpoint_dir, 'model_best.pth.tar'),
                    '--train_folder', train_dir, 
                    '--val_folder', val_dir,
                    '--test_folder', test_dir,
                    '--checkpoint', checkpoint_dir,
                    '--logdir', log_dir,
                    '--test_batch', '2'])
    


if __name__ == '__main__':
    main()

