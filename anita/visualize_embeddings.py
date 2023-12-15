from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import numpy as np
import matplotlib

# Extract embeddings and characters
model = Word2Vec.load('anita/small-wv')
keyed_vecs = model.wv
chars = list(keyed_vecs.index_to_key)
embeddings = np.array([keyed_vecs[char] for char in chars])

# Reduce dimensionality for visualization
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot the embeddings
matplotlib.rcParams.update(
    {
        'font.family': 'Heiti TC',
        'axes.unicode_minus': False
    }
)
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)

# Annotate points with characters
for i, char in enumerate(chars):
    plt.annotate(char, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=8)

plt.title('t-SNE Visualization of Character Embeddings')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.show()
