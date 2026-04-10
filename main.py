import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ---------------------------
# Funzioni di base
# ---------------------------

def crea_matrice_adiacenza_random(n_pagine, min_link=1, max_link=5):
    A = np.zeros((n_pagine, n_pagine))
    for pagina in range(n_pagine):
        link_possibili = [i for i in range(n_pagine) if i != pagina]
        n_link = np.random.randint(min_link, max_link + 1)
        destinazioni = np.random.choice(link_possibili, size=n_link, replace=False)
        for dest in destinazioni:
            A[dest, pagina] = 1
    return A

def normalizza_matrice(A):
    M = np.zeros_like(A)
    for j in range(A.shape[1]):
        somma_colonna = np.sum(A[:, j])
        if somma_colonna != 0:
            M[:, j] = A[:, j] / somma_colonna
        else:
            M[:, j] = 1 / A.shape[0]
    return M

def pagerank(M, d=0.85, tol=1e-6, max_iter=100, teleport_vector=None):
    N = M.shape[0]
    if teleport_vector is None:
        v = np.ones(N) / N
    else:
        v = teleport_vector / np.sum(teleport_vector)
    p = v.copy()
    for _ in range(max_iter):
        p_nuovo = d * M @ p + (1 - d) * v
        delta = np.linalg.norm(p_nuovo - p, 1)
        if delta < tol:
            break
        p = p_nuovo
    return p

def crea_teleport_vector(df_preferenze, nome_utente):
    riga = df_preferenze[df_preferenze['utente'] == nome_utente]
    preferenze = riga.drop(columns=['utente']).values.flatten()
    if np.sum(preferenze) == 0:
        return np.ones_like(preferenze) / len(preferenze)
    return preferenze / np.sum(preferenze)

# ---------------------------
# Dati
# ---------------------------

np.random.seed(42)
n_pagine = 20
n_utenti = 5

# Se vuoi, puoi rimuovere la randomizzazione per caricare dati reali

A = crea_matrice_adiacenza_random(n_pagine)
M = normalizza_matrice(A)

preferenze = np.random.rand(n_utenti, n_pagine)
df_preferenze = pd.DataFrame(preferenze, columns=[f"pagina_{i}" for i in range(n_pagine)])
df_preferenze.insert(0, 'utente', [f"Utente_{i+1}" for i in range(n_utenti)])

df_preferenze.to_csv("preferenze_utenti.csv", index=False)

# ---------------------------
# STREAMLIT APP
# ---------------------------

st.title("📊 PageRank Personalizzato per Utente")

utente_selezionato = st.selectbox(
    "Seleziona un utente:",
    df_preferenze['utente'].tolist()
)

# Crea teleport vector per l'utente selezionato
teleport = crea_teleport_vector(df_preferenze, utente_selezionato)

# Calcola il PageRank personalizzato
pr = pagerank(M, teleport_vector=teleport)

df_result = pd.DataFrame({
    'Pagina': [f"pagina_{i}" for i in range(n_pagine)],
    'PageRank': pr
}).sort_values(by='PageRank', ascending=False)

st.subheader(f"PageRank per {utente_selezionato}")
st.dataframe(df_result)

# Grafico a barre
st.bar_chart(data=df_result.set_index('Pagina'))

