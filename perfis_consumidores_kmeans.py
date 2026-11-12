# %% [markdown]
# # Trabalho de IA - Identificação de perfis de consumidores (K-means)
#
# UniEVANGÉLICA - Engenharia de Software - Inteligência Artificial - Turma A
# Prof. Lucas Marcos - Tema 17
#
# Dupla: Arthur Victor Rodrigues (2310579) e Sergio Cavalcante (2321154)
#
# Passos: gerar a base, verificar e limpar os dados, padronizar, escolher o k,
# treinar o K-means e analisar os grupos.

# %%
# bibliotecas
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score
from sklearn.decomposition import PCA

SEED = 42  # semente pra dar sempre o mesmo resultado

os.makedirs("dados", exist_ok=True)
os.makedirs("figuras", exist_ok=True)
pd.set_option("display.width", 120)

# %% [markdown]
# ## 1. Gerando a base de dados
#
# Como não temos dados reais de uma loja, vamos criar 300 clientes fictícios.
# Usamos 3 comportamentos diferentes (100 clientes cada):
# - clientes que compram bastante e com frequência
# - clientes que compram pouco mas gastam muito
# - clientes que estão sumidos
#
# A coluna `perfil_gerador` guarda de qual comportamento cada cliente veio, mas ela
# **não é usada no treino**, só no final para conferir o resultado.
#
# Também colocamos de propósito 6 linhas duplicadas e 15 valores vazios, para mostrar a limpeza.

# %%
rng = np.random.default_rng(SEED)
n = 100

def gerar_clientes(rec_media, rec_desvio, freq_media, ticket_media, ticket_desvio, nome):
    clientes = pd.DataFrame()
    clientes["recencia_dias"] = np.clip(rng.normal(rec_media, rec_desvio, n), 0, 180).round()
    clientes["frequencia_compras"] = np.clip(rng.poisson(freq_media, n), 1, None)
    clientes["ticket_medio_reais"] = np.clip(rng.normal(ticket_media, ticket_desvio, n), 15, None).round(2)
    clientes["perfil_gerador"] = nome
    return clientes

frequentes = gerar_clientes(12, 7, 14, 110, 30, "frequente")
alto_valor = gerar_clientes(50, 18, 4, 380, 70, "alto_valor_ocasional")
inativos = gerar_clientes(125, 30, 2, 55, 18, "inativo")

base = pd.concat([frequentes, alto_valor, inativos], ignore_index=True)
base = base.sample(frac=1, random_state=SEED).reset_index(drop=True)  # embaralha
base.insert(0, "cliente_id", ["C" + str(i).zfill(4) for i in range(1, len(base) + 1)])

# colocando duplicados e valores vazios de propósito
duplicados = base.sample(6, random_state=SEED)
base = pd.concat([base, duplicados], ignore_index=True)

for coluna, qtd in [("recencia_dias", 5), ("frequencia_compras", 4), ("ticket_medio_reais", 6)]:
    linhas = base.sample(qtd, random_state=SEED + qtd).index
    base.loc[linhas, coluna] = np.nan

base.to_csv("dados/clientes_simulados.csv", index=False)
print("Tamanho da base:", base.shape)
print(base.head())

# %% [markdown]
# ## 2. Verificando a base

# %%
df = pd.read_csv("dados/clientes_simulados.csv")

print("Total de linhas:", len(df))
print("Linhas duplicadas:", df.duplicated().sum())
print("Valores vazios por coluna:")
print(df.isna().sum())
print(df.describe().round(2))

# %% [markdown]
# ## 3. Limpeza
#
# - tiramos as linhas duplicadas
# - preenchemos os vazios com a mediana da coluna (a mediana sofre menos com valores muito altos ou baixos)

# %%
colunas = ["recencia_dias", "frequencia_compras", "ticket_medio_reais"]

qtd_antes = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print("Duplicados removidos:", qtd_antes - len(df), "| linhas agora:", len(df))

medianas = df[colunas].median()
print("Medianas usadas:")
print(medianas.round(2))

df[colunas] = df[colunas].fillna(medianas)
print("Vazios que sobraram:", int(df[colunas].isna().sum().sum()))

# %% [markdown]
# ## 4. Olhando os dados

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
for i, col in enumerate(colunas):
    axes[i].hist(df[col], bins=30, color="#4C72B0", edgecolor="white")
    axes[i].set_title(col)
fig.suptitle("Distribuição dos atributos (depois da limpeza)")
fig.tight_layout()
fig.savefig("figuras/01_distribuicoes.png", dpi=150)
plt.show()

# correlação entre os atributos
print(df[colunas].corr().round(2))
# recência e frequência têm correlação negativa: quem compra mais vezes comprou há menos tempo

# %% [markdown]
# ## 5. Padronização
#
# O K-means usa distância, então precisamos deixar os atributos na mesma escala.
# Senão o ticket médio (centenas de reais) pesa muito mais que a frequência.
#
# Fórmula: z = (x - média) / desvio padrão

# %%
scaler = StandardScaler()
X = scaler.fit_transform(df[colunas])

print("Média depois de padronizar:", X.mean(axis=0).round(3))
print("Desvio padrão depois de padronizar:", X.std(axis=0).round(3))

# %% [markdown]
# ## 6. Escolhendo o número de grupos (k)
#
# Testamos k de 2 até 6 e olhamos:
# - **inércia** (método do cotovelo)
# - **silhueta** (quanto mais perto de 1, melhor)

# %%
resultados = []
for k in range(2, 7):
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=SEED)
    km.fit(X)
    sil = silhouette_score(X, km.labels_)
    resultados.append({"k": k, "inercia": km.inertia_, "silhueta": sil})

tabela_k = pd.DataFrame(resultados)
tabela_k.to_csv("dados/escolha_k.csv", index=False)
print(tabela_k.round(3))

# pega o k com a maior silhueta
melhor_k = int(tabela_k.loc[tabela_k["silhueta"].idxmax(), "k"])
print("Melhor k pela silhueta:", melhor_k)

# %%
fig, ax1 = plt.subplots(figsize=(7, 4))
ax1.plot(tabela_k["k"], tabela_k["inercia"], "o-", color="#4C72B0")
ax1.set_xlabel("k (número de grupos)")
ax1.set_ylabel("Inércia", color="#4C72B0")
ax1.set_xticks(list(tabela_k["k"]))

ax2 = ax1.twinx()
ax2.plot(tabela_k["k"], tabela_k["silhueta"], "s--", color="#DD8452")
ax2.set_ylabel("Silhueta média", color="#DD8452")

ax1.axvline(melhor_k, color="gray", ls=":", lw=1)
ax1.set_title("Inércia e silhueta por k")
fig.tight_layout()
fig.savefig("figuras/02_escolha_k.png", dpi=150)
plt.show()

# %% [markdown]
# Deu k = 3: é onde a silhueta é maior e também onde a inércia para de cair muito (cotovelo).

# %% [markdown]
# ## 7. Treinando o modelo final

# %%
modelo = KMeans(n_clusters=melhor_k, init="k-means++", n_init=10, random_state=SEED)
modelo.fit(X)

df["grupo"] = modelo.labels_
silhueta_final = silhouette_score(X, modelo.labels_)

print("k:", melhor_k)
print("Inércia:", round(modelo.inertia_, 1))
print("Silhueta média:", round(silhueta_final, 3))
print("Iterações até convergir:", modelo.n_iter_)
print("Clientes por grupo:")
print(df["grupo"].value_counts().sort_index())

# %% [markdown]
# ## 8. Testando a estabilidade
#
# Rodamos o K-means de novo com 10 sementes diferentes e comparamos com o modelo final
# usando o ARI (Índice de Rand Ajustado). Se der perto de 1, os grupos não mudam.

# %%
lista_ari = []
for s in range(10):
    teste = KMeans(n_clusters=melhor_k, init="k-means++", n_init=1, random_state=100 + s)
    rotulos = teste.fit_predict(X)
    lista_ari.append(adjusted_rand_score(modelo.labels_, rotulos))

print("ARI médio:", round(np.mean(lista_ari), 3))
print("ARI mínimo:", round(np.min(lista_ari), 3))

# %% [markdown]
# ## 9. Entendendo cada grupo
#
# Voltamos os valores para as unidades normais (dias, compras e reais) e usamos a mediana de cada grupo.
# Os nomes dos perfis fomos nós que demos, olhando os números.

# %%
centroides = pd.DataFrame(scaler.inverse_transform(modelo.cluster_centers_), columns=colunas)

perfil = df.groupby("grupo")[colunas].median().round(1)
perfil["clientes"] = df["grupo"].value_counts().sort_index()
sil_cliente = silhouette_samples(X, modelo.labels_)
perfil["silhueta_media"] = pd.Series(sil_cliente).groupby(df["grupo"]).mean().round(3)

def dar_nome(linha):
    if linha["recencia_dias"] <= 30 and linha["frequencia_compras"] >= 8:
        return "Frequentes"
    elif linha["ticket_medio_reais"] >= 250:
        return "Alto valor ocasional"
    elif linha["recencia_dias"] >= 90:
        return "Inativos / em risco"
    else:
        return "Intermediário"

perfil["perfil"] = perfil.apply(dar_nome, axis=1)
df["perfil"] = df["grupo"].map(perfil["perfil"])

perfil.to_csv("dados/perfis_resultado.csv")
df.to_csv("dados/clientes_com_grupo.csv", index=False)
print(perfil)

# %% [markdown]
# ## 10. Gráficos dos grupos

# %%
cores = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for g in sorted(df["grupo"].unique()):
    dados_g = df[df["grupo"] == g]
    axes[0].scatter(dados_g["frequencia_compras"], dados_g["ticket_medio_reais"],
                    s=18, alpha=0.75, color=cores[g], label=str(g) + " - " + perfil.loc[g, "perfil"])
axes[0].scatter(centroides["frequencia_compras"], centroides["ticket_medio_reais"],
                marker="X", s=200, color="black", label="Centroides")
axes[0].set_xlabel("Frequência de compras (180 dias)")
axes[0].set_ylabel("Ticket médio (R$)")
axes[0].set_title("Frequência x ticket médio")
axes[0].legend(fontsize=8)

for g in sorted(df["grupo"].unique()):
    dados_g = df[df["grupo"] == g]
    axes[1].scatter(dados_g["recencia_dias"], dados_g["frequencia_compras"], s=18, alpha=0.75, color=cores[g])
axes[1].scatter(centroides["recencia_dias"], centroides["frequencia_compras"], marker="X", s=200, color="black")
axes[1].set_xlabel("Recência (dias desde a última compra)")
axes[1].set_ylabel("Frequência de compras")
axes[1].set_title("Recência x frequência")

fig.tight_layout()
fig.savefig("figuras/03_grupos.png", dpi=150)
plt.show()

# %%
# PCA só pra conseguir ver os 3 atributos em um gráfico 2D
pca = PCA(n_components=2, random_state=SEED)
Z = pca.fit_transform(X)

fig, ax = plt.subplots(figsize=(6.5, 5))
for g in sorted(df["grupo"].unique()):
    m = (df["grupo"] == g).values
    ax.scatter(Z[m, 0], Z[m, 1], s=18, alpha=0.75, color=cores[g], label=perfil.loc[g, "perfil"])
variancia = pca.explained_variance_ratio_.sum() * 100
ax.set_title("Grupos em 2D com PCA (" + str(round(variancia)) + "% da variância)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("figuras/04_pca.png", dpi=150)
plt.show()

# %%
# centroides padronizados: acima de 0 = acima da média, abaixo de 0 = abaixo da média
fig, ax = plt.subplots(figsize=(9, 4))
centroides_z = pd.DataFrame(modelo.cluster_centers_, columns=colunas, index=perfil["perfil"])
centroides_z.plot(kind="bar", ax=ax, color=["#4C72B0", "#DD8452", "#55A868"])
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("Valor padronizado (z)")
ax.set_xlabel("")
ax.set_title("Centroides padronizados de cada grupo")
plt.xticks(rotation=0)
fig.tight_layout()
fig.savefig("figuras/05_centroides.png", dpi=150)
plt.show()

# %% [markdown]
# ## 11. Conferindo com a base original
#
# Isso só dá pra fazer porque a base é simulada e sabemos de onde cada cliente veio.
# Com dados reais não teria essa "resposta".

# %%
print(pd.crosstab(df["perfil"], df["perfil_gerador"]))
print()
print("ARI grupos x perfil de origem:", round(adjusted_rand_score(df["perfil_gerador"], df["grupo"]), 3))

# %%
# quais clientes ficaram no grupo "errado"?
errados = df[(df["perfil"] == "Frequentes") & (df["perfil_gerador"] == "alto_valor_ocasional")]
print(errados)
# os dois estavam com o ticket vazio e receberam a mediana (113,70), por isso foram pro grupo Frequentes

# %% [markdown]
# ## 12. Testando com clientes novos

# %%
novos = pd.DataFrame({
    "recencia_dias": [5, 60, 150],
    "frequencia_compras": [18, 3, 1],
    "ticket_medio_reais": [95.0, 420.0, 40.0],
})
novos["grupo"] = modelo.predict(scaler.transform(novos[colunas]))
novos["perfil"] = novos["grupo"].map(perfil["perfil"])
print(novos)
