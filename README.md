# ead_enem2024

**Análise Exploratória de Dados e Visualização do ENEM 2024**

Dashboard interativo desenvolvido em Python + Streamlit para exploração dos microdados do ENEM 2024, com dados agregados ao nível municipal e conectado a um banco PostgreSQL.

---

## 📋 Sobre o Projeto

O projeto transforma os microdados brutos do ENEM 2024 em um painel analítico interativo. A unidade de análise é o **município**: a partir de três tabelas relacionais, os dados são agrupados por município (`co_municipio_prova`) e consolidados em um único DataFrame, permitindo visualizar disparidades regionais em desempenho, perfil socioeconômico e participação.

---

## 🗂️ Estrutura do Repositório

```
ead_enem2024/
├── app.py            # Dashboard Streamlit (4 páginas)
├── database.py       # Conexão PostgreSQL e pipeline de dados
├── analysis.py       # Funções estatísticas e constantes analíticas
├── requirements.txt  # Dependências Python
└── .streamlit/
    └── secrets.toml  # Template de credenciais (não versionado)
```

---

## 🗄️ Modelo de Dados

O banco de dados contém três tabelas reais:

| Tabela | Descrição |
|---|---|
| `ed_enem_2024_participantes` | Microdados de inscrição e questionário socioeconômico (Q001–Q023) |
| `ed_enem_2024_resultados` | Notas por participante (CN, CH, LC, MT, Redação) |
| `municipio` | Tabela de referência de municípios (código, nome, UF) |

Como não há chave primária/estrangeira direta entre participantes e resultados, o pipeline agrega ambas as tabelas ao nível municipal via `GROUP BY co_municipio_prova` e então realiza o `JOIN` com a tabela de municípios pela coluna `cod_7` (código IBGE do município com 7 dígitos).

### Colunas de notas (após renomeação no pipeline)

| Nome interno | Descrição |
|---|---|
| `nota_ciencias_natureza` | Média municipal em Ciências da Natureza |
| `nota_ciencias_humanas` | Média municipal em Ciências Humanas |
| `nota_linguagens` | Média municipal em Linguagens e Códigos |
| `nota_matematica` | Média municipal em Matemática |
| `nota_redacao` | Média municipal em Redação |
| `nota_geral_media` | Média geral das 5 áreas |

---

## 📊 Páginas do Dashboard

### 🏠 Visão Geral
- KPIs nacionais: total de inscritos, média de idade, médias de notas
- Gráfico de barras com inscritos por UF
- Gráfico de barras com notas por área de conhecimento
- **Prévia dos dados municipais** (primeiras 200 linhas) exibindo o **nome do município** (`municipio`) como primeira coluna

### 📊 Variáveis Qualitativas
- Distribuição de municípios por UF
- Composição demográfica (gênero, raça/cor, tipo de escola)

### 📈 Variáveis Quantitativas
- Histogramas e box plots das médias municipais de notas
- Estatísticas descritivas e teste de normalidade (Shapiro-Wilk)

### 🔗 Análise de Correlação
- Heatmap de correlação (Pearson/Spearman/Kendall) entre notas e proporções demográficas
- Scatter matrix interativa

---

## ⚙️ Tecnologias Utilizadas

- **Python 3.11+**
- **Streamlit** ≥ 1.32 — interface do dashboard
- **Pandas** ≥ 2.2 — manipulação de dados
- **Plotly** ≥ 5.20 — gráficos interativos
- **SciPy** ≥ 1.12 — testes estatísticos
- **psycopg2-binary** ≥ 2.9 — conexão PostgreSQL

---

## 🚀 Como Executar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Configurar credenciais do banco

Crie o arquivo `.streamlit/secrets.toml` com as credenciais do PostgreSQL:

```toml
[database]
host     = "SEU_HOST"
port     = 5432
dbname   = "SEU_BANCO"
user     = "SEU_USUARIO"
password = "SUA_SENHA"
```

> No Streamlit Cloud, configure via **Settings → Secrets**.

### 3. Executar o dashboard

```bash
streamlit run app.py
```

---

## 🔧 Tarefas Realizadas no Desenvolvimento

1. **Estruturação do banco de dados**: definição das três tabelas (`participantes`, `resultados`, `municipio`) e estratégia de agregação municipal via `GROUP BY co_municipio_prova`.

2. **Pipeline de dados (`database.py`)**: implementação das três queries SQL de agregação exaustiva (participantes: contagens por gênero, raça, estado civil, tipo de escola, questionário Q001–Q023; resultados: médias de notas por município; municípios: referência IBGE). As queries são executadas no PostgreSQL e o resultado é consolidado via merge.

3. **Renomeação das colunas de notas**: as colunas originais `media_cn`, `media_ch`, `media_lc`, `media_mt`, `media_redacao`, `media_geral` foram renomeadas no pipeline para `nota_ciencias_natureza`, `nota_ciencias_humanas`, `nota_linguagens`, `nota_matematica`, `nota_redacao`, `nota_geral_media`, seguindo o mapeamento definido no script Colab.

4. **Dashboard Streamlit (`app.py`)**: criação das quatro páginas analíticas com KPIs, gráficos interativos (Plotly), tabela de prévia dos dados e navegação lateral.

5. **Módulo de análise (`analysis.py`)**: implementação das funções estatísticas (`descriptive_stats`, `frequency_table`, `correlation_matrix`, `normality_test`, `inscribed_by_uf`, `mean_scores_by_group`, `apply_labels`) e definição das constantes de colunas utilizadas no dashboard.

6. **Troca da coluna de código de município pelo nome** na tabela de prévia dos dados municipais (Visão Geral): a coluna `cod_7` foi removida da exibição e `municipio` (nome) passa a ser a primeira coluna visível, tornando a tabela mais legível sem alterar o DataFrame interno.

7. **Formatação numérica no padrão brasileiro**: todos os números exibidos no dashboard usam `.` como separador de milhares e `,` como separador de casas decimais (ex.: `1.234.567`, `525,3`). Isso foi aplicado nos KPIs, rótulos de barras, anotações de histogramas, estatísticas descritivas, percentis e nos eixos/tooltips de todos os gráficos via template global do Plotly.

8. **Revisão de textos**: verificação de consistência nos rótulos e títulos para evitar ambiguidade entre "nota média por município" (valor agregado já calculado no pipeline) e "média nacional" (agregado ponderado dos valores municipais exibido nos KPIs e gráficos nacionais).

---

## 📁 Dados

Os microdados do ENEM 2024 são disponibilizados pelo **INEP** (Instituto Nacional de Estudos e Pesquisas Educacionais Anísio Teixeira) e podem ser obtidos em:  
🔗 https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem
