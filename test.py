import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors


# Passo 1: Converti la matrice di correlazione in una matrice di distanza
distance_matrix = 1 - np.abs(corr_matrix)

# Passo 2: Applica k-NN per trovare i k vicini più prossimi
k = int(len(descriptor_df.colums) /100)
knn = NearestNeighbors(n_neighbors=k + 1, metric='precomputed')
knn.fit(distance_matrix)

distances, indices = knn.kneighbors(distance_matrix)

# Passo 3: Calcola la somma delle distanze per ogni descrittore
# Maggiore è la somma, meno correlato è il descrittore ai suoi vicini
sum_distances = distances[:, 1:].sum(axis=1)  # Escludi la distanza con sé stesso (0)

# Passo 4: Seleziona descrittori con un approccio greedy
selected_features = []
remaining_features = corr_matrix.columns.tolist()
n_features_to_select = 5  # Numero di descrittori da selezionare (modifica in base alle tue esigenze)

while len(selected_features) < n_features_to_select and remaining_features:
    # Trova il descrittore con la somma delle distanze più alta tra quelli rimanenti
    remaining_indices = [corr_matrix.columns.get_loc(f) for f in remaining_features]
    valid_indices = [i for i in range(len(sum_distances)) if i in remaining_indices]
    if not valid_indices:
        break
    best_idx = valid_indices[np.argmax([sum_distances[i] for i in valid_indices])]
    best_feature = corr_matrix.columns[best_idx]
    
    # Aggiungi il descrittore selezionato
    selected_features.append(best_feature)
    
    # Rimuovi il descrittore selezionato e i suoi k vicini
    neighbors = [corr_matrix.columns[i] for i in indices[best_idx, 1:]]  # Escludi il descrittore stesso
    remaining_features = [f for f in remaining_features if f not in [best_feature] + neighbors]

print("Descrittori selezionati:", selected_features)

# Passo 5: Verifica la matrice di correlazione ristretta ai descrittori selezionati
selected_corr_matrix = corr_matrix.loc[selected_features, selected_features]
print("Matrice di correlazione dei descrittori selezionati:\n", selected_corr_matrix)


                 
