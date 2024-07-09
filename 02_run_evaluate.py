import argparse
import os
import subprocess

NUM_EPOCHS = 100
LR = 1e-3
TRAIN_BATCH = 1
TEST_BATCH = 1

def main():
    # -- Run all scripts --
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--root-dir', type=str, required=True)
    args = arg_parser.parse_args()
    root_dir = args.root_dir
    
    run_primary_joint_train(root_dir)


def run_primary_joint_train(root_dir):
    train_dir = os.path.join(root_dir, 'train')
    val_dir = os.path.join(root_dir, 'val')
    test_dir = os.path.join(root_dir, 'test')
    checkpoint_dir = os.path.join(root_dir, 'checkpoints', 'tarig_prim_joint')
    log_dir = os.path.join(root_dir, 'logs', 'tarig_prim_joint')

    for dir_path in [train_dir, val_dir, test_dir, checkpoint_dir, log_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Get number of joints from keypoint_names.txt (number of lines)
    with open(os.path.join(root_dir, 'keypoint_names.txt'), 'r') as f:
        num_joints = len(f.readlines())
    
    num_joints = int(num_joints)

    print('Running run_primary_joint_train.py')
    # Run primary training
    subprocess.run(['python', 
                    '-u', 
                    'run_primary_joint_train.py',
                    '--resume', os.path.join(checkpoint_dir, 'model_best.pth.tar'),
                    '-e',
                    '--train_folder', train_dir, 
                    '--val_folder', val_dir,
                    '--test_folder', test_dir,
                    '--checkpoint', checkpoint_dir,
                    '--logdir', log_dir,
                    '--train_batch', str(int(TRAIN_BATCH)), 
                    '--test_batch', str(int(TEST_BATCH)), 
                    '--lr', str(float(LR)), 
                    '--epoch', str(int(NUM_EPOCHS)),
                    '--num-joints', str(num_joints)])
    


if __name__ == '__main__':
    main()

