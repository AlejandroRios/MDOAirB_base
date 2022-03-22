import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from aux_tools import corrdot
# df.to_pickle('doe.pkl')
df = pd.read_pickle('doe.pkl')
sns.set(style='white', font_scale=1.4)

print(df.head())
print(df.info())
df = df.drop(df[df.profit == 0].index)
print(df.head())
print(df.info())
plot_type = 1

if plot_type == 0:

    # Simple plot
    ax = sns.pairplot(df,corner=True)

elif plot_type == 1:


    # Complete plot
    # based on: https://stackoverflow.com/questions/48139899/correlation-matrix-plot-with-coefficients-on-one-side-scatterplots-on-another
    ax = sns.PairGrid(df, aspect=1.4, diag_sharey=False)
    ax.map_lower(sns.regplot, lowess=True, line_kws={'color': 'black'})
    ax.map_diag(sns.histplot)
    ax.map_upper(corrdot)

for ax in ax.axes[:,0]:
    ax.get_yaxis().set_label_coords(-0.30,0.5)

# Plot window
plt.tight_layout()
plt.show()