# Trabalho de IA - Perfis de consumidores com K-means

Tema 17 - Inteligência Artificial - UniEVANGÉLICA (Eng. de Software, Turma A) - Prof. Lucas Marcos

Dupla: Arthur Victor Rodrigues (2310579) e Sergio Cavalcante (2321154)

## Como rodar
No Google Colab: abrir o `perfis_consumidores_kmeans.ipynb` e clicar em "Executar tudo".
Não precisa subir nenhum arquivo, porque o próprio notebook gera a base.

No computador:
```
pip install -r requirements.txt
python perfis_consumidores_kmeans.py
```

## Arquivos
- `perfis_consumidores_kmeans.ipynb`: notebook principal (já executado)
- `perfis_consumidores_kmeans.py`: o mesmo código em .py
- `dados/clientes_simulados.csv`: base gerada (306 linhas, com duplicados e vazios)
- `dados/clientes_com_grupo.csv`: base limpa (300) com o grupo de cada cliente
- `dados/escolha_k.csv`: inércia e silhueta de k = 2 até 6
- `dados/perfis_resultado.csv`: resumo dos grupos
- `figuras/`: gráficos
- `docs/Trabalho_IA_Perfis_Consumidores.pdf`: documento do trabalho

## Resultado
Deu k = 3, com silhueta de 0,650. Os grupos foram Frequentes (102), Alto valor ocasional (98) e Inativos (100).
