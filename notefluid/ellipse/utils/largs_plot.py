import matplotlib.pyplot as plt


def largs_plot_speed(df):
    plt.figure(figsize=(6, 6))
    ax = plt.axes()

    def plot_larg(row):
        ax.arrow(row['lx'], row['ly'], 100 * row['ux'], 100 * row['uy'], color='b',
                 linewidth=0.5, head_width=0.5,
                 head_length=0.5)

    df.apply(plot_larg, axis=1)
    plt.xlim(df['lx'].min() - df['ux'].abs().max() - 5, df['lx'].max() + df['ux'].abs().max() + 5)
    plt.ylim(df['ly'].min() - df['uy'].abs().max() - 5, df['ly'].max() + df['uy'].abs().max() + 5)

    plt.grid(b=True, which='major')
