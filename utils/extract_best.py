import re
import sys 
import argparse


def split_trials(text):
    # Split at each trial "finished" line, keeping the delimiter
    blocks = re.split(r"(\[I .*?Trial \d+)", text, flags=re.DOTALL)

    # Reconstruct: every two parts = [trial_content, trial_summary]
    trial_blocks = []
    for i in range(0, len(blocks)-1, 2):
        trial_blocks.append(blocks[i] + blocks[i+1])

    return trial_blocks


def get_epoch_block(text, epoch_num):
    pattern = rf"(Epoch {epoch_num},.*?)(?=Epoch {epoch_num+1},|$)"
    match = re.search(pattern, text, flags=re.DOTALL)
    return match.group(1) if match else None


def parse_log_file(filepath):
    with open(filepath, "r") as f:
        text = f.read()

    # Find the best trial
    trial_pattern = re.compile(
        r"Best is trial (\d+) with value: ([\d\.]+).",
        re.DOTALL
    )
    trials = trial_pattern.findall(text)
    # trials = set(trials)  # Remove duplicates
    best_trial, best_val_ade = max(trials, key=lambda x: int(x[0]))
    best_trial, best_val_ade = int(best_trial), float(best_val_ade.rstrip('.'))

    # get best block
    trial_blocks = split_trials(text)
    best_block = trial_blocks[best_trial-1]
    val_match = re.findall(r"Epoch (\d+), val_ADE=([\d\.]+), val_FDE=([\d\.]+)", best_block)
    best_epoch, best_val_ade_, best_val_fde = min(val_match, key=lambda x: float(x[1]))
    best_epoch, best_val_ade_, best_val_fde = int(best_epoch), float(best_val_ade_), float(best_val_fde)
    log_directory = re.search(r"Log directory: (.+)", best_block).group(1)
    assert (best_val_ade - best_val_ade_) < 0.001, "Mismatch in best val ADE values!"

    # get best train
    train_match = re.findall(r"Epoch (\d+), train_ADE=([\d\.]+), train_FDE=([\d\.]+)", best_block)
    # test_match = re.findall(r"test_ADE=([\d\.]+), test_FDE=([\d\.]+)", best_block)
    best_train_ade, best_train_fde = train_match[best_epoch-1][1:]
    best_train_ade, best_train_fde = float(best_train_ade), float(best_train_fde)
    print('Log directory:', log_directory)
    print('trial', best_trial)
    print('\nBest epoch:', best_epoch)
    print(f'val_ADE/FDE & train_ADE/FDE:\n{best_val_ade_}\n{best_val_fde}\n\n{best_train_ade}\n{best_train_fde}\n')

    # get best test
    best_test = get_epoch_block(best_block, best_epoch).split('\n')
    for line in best_test:
        print(line)

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filepath", default=None, type=str)
    args = parser.parse_args()

    # Open file for writing (or append with "a")
    sys.stdout = open("best.txt", "w")
    parse_log_file(args.filepath)