import json
import pandas as pd
import matplotlib.pyplot as plt
import sys

def parse_test_accuracy(filepath):
    epochs = []
    accs = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('{') and '"mode": "test"' in line:
                try:
                    data = json.loads(line)
                    epoch = data.get("epoch")
                    acc = data.get("acc")
                    if acc is not None and not pd.isna(acc):
                        epochs.append(epoch)
                        accs.append(acc)
                except json.JSONDecodeError:
                    continue
    return epochs, accs

def smooth_plot(epochs, accs, label, color, window=5):
    df = pd.DataFrame({'epoch': epochs, 'acc': accs})
    df = df.groupby('epoch').mean().reset_index()

    smoothed = df['acc'].rolling(window, center=True, min_periods=1).mean()
    min_band = df['acc'].rolling(window, center=True, min_periods=1).min()
    max_band = df['acc'].rolling(window, center=True, min_periods=1).max()

    plt.plot(df['epoch'], smoothed, label=label, color=color)
    plt.fill_between(df['epoch'], min_band, max_band, color=color, alpha=0.2)

def main():
    log_no_comm = sys.argv[1]
    log_comm = sys.argv[2]

    epochs1, accs1 = parse_test_accuracy(log_no_comm)
    epochs2, accs2 = parse_test_accuracy(log_comm)

    plt.figure(figsize=(5, 4))
    smooth_plot(epochs1, accs1, label='Accuracy (no comm)', color='darkorange')
    smooth_plot(epochs2, accs2, label='Accuracy (comm)', color='steelblue')

    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy Comparison')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('test_accuracy_comparison.pdf')
    plt.close()

if __name__ == "__main__":
    main()
