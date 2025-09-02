import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Set style parameters
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

# Read data
rituals_df = pd.read_csv("data/rituals_codes.csv")

# Calculate text lengths and filter out short texts
text_lengths = rituals_df['text'].str.len()
text_lengths = text_lengths[text_lengths > 10]

# Create figure
fig, ax = plt.subplots(figsize=(3.5, 2.5), dpi=300)

# Create histogram
sns.histplot(data=text_lengths, 
            bins=30,
            stat='count',
            color='#2f5d7e',
            alpha=0.6,
            edgecolor='white',
            linewidth=0.5)

# Format axes
def format_k(x, p):
    return f'{int(x/1000)}k' if x >= 1000 else str(int(x))

ax.xaxis.set_major_formatter(FuncFormatter(format_k))
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}'))  # Changed to integer format
ax.yaxis.set_major_locator(plt.MaxNLocator(5))  # Set number of y-axis ticks

# Labels and titles
ax.set_xlabel('Character count')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Ethnographic Text Lengths', 
             pad=10, 
             fontsize=10, 
             fontweight='bold')

# Adjust layout
plt.tight_layout()

# Save figure
plt.savefig('figures/text_length_distribution.pdf', 
            bbox_inches='tight',
            dpi=300)
plt.savefig('figures/text_length_distribution.png',
            bbox_inches='tight', 
            dpi=300)
plt.close()
