import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from datetime import datetime

def initialize_plot(num_epochs):
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Palatino Linotype'],
        'font.size': 14
    })

    plt.ion()
    fig, ax = plt.subplots()
    train_loss_line, = ax.plot([], [], label='Train Loss (combined)')
    val_loss_segmentation_line, = ax.plot([], [], label='Val Loss (Segmentation)')
    val_loss_localization_line, = ax.plot([], [], label='Val Loss (Localization)')
    ax.set_xlim(0, num_epochs)
    ax.legend()

    # Customize the tick formatter for the y-axis to show scientific notation
    formatter_y = ticker.ScalarFormatter(useMathText=True)
    formatter_y.set_scientific(True)
    formatter_y.set_powerlimits((-1, 1))

    # Customize the tick formatter for the x-axis to show integer numbers
    formatter_x = ticker.FuncFormatter(lambda x, _: f'{int(x)}')

    # Apply the formatter to both axes
    ax.xaxis.set_major_formatter(formatter_x)
    ax.yaxis.set_major_formatter(formatter_y)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')

    return fig, ax, train_loss_line, val_loss_segmentation_line, val_loss_localization_line

def update_plot(fig, ax, train_loss_line, val_loss_segmentation_line, val_loss_localization_line, epoch, train_losses, val_losses_segmentation, val_losses_localization):
    train_loss_line.set_xdata(range(1, epoch+2))
    train_loss_line.set_ydata(train_losses)
    val_loss_segmentation_line.set_xdata(range(1, epoch+2))
    val_loss_segmentation_line.set_ydata(val_losses_segmentation)
    val_loss_localization_line.set_xdata(range(1, epoch+2))
    val_loss_localization_line.set_ydata(val_losses_localization)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)

def finalize_plot(fig):
    # Get today's date
    today = datetime.today().strftime('%Y-%m-%d')
    filename = f'results/training_loss_{today}.png'
    plt.ioff()
    plt.savefig(filename)
    plt.show()