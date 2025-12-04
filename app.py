import streamlit as st
import pandas as pd
import joblib

# Mesmas colunas usadas no treino
COLS_MODELO = [
    "diff_rank",
    "diff_elo_global",
    "diff_elo_surface",
    "diff_avg_aces",
    "diff_avg_dfs",
    "diff_p1_in",
    "diff_p1_won",
    "diff_momentum",
    "diff_h2h",
    "surface_code",
]

@st.cache_resource
def load_model():
    model = joblib.load("atp_model_small.pkl")
    return model

@st.cache_data
def load_data():
    df = pd.read_csv("matches_for_app.csv", parse_dates=["tourney_date"])
    return df

# Configuração da página
st.set_page_config(
    page_title="Previsão de Partidas ATP",
    page_icon="🎾",
    layout="centered",
)

st.title("🎾 Previsão de Partidas ATP")
st.write(
    """
Aplicação simples para demonstrar o **deploy** do modelo de previsão de partidas ATP.

- O modelo foi treinado com partidas de 2018–2023  
- Aqui usamos partidas reais de **2024** e mostramos a previsão do modelo para elas
"""
)

model = load_model()
df = load_data()

st.sidebar.header("Escolha uma partida de 2024")

# Criar uma lista de opções legíveis para o usuário
opcoes = []
for idx, row in df.iterrows():
    label = f"{row['tourney_date'].date()} — {row['winner_name']} vs {row['loser_name']}"
    opcoes.append((label, idx))

labels = [label for label, _ in opcoes]
label_escolhida = st.sidebar.selectbox("Selecione a partida:", labels)

# Recuperar o índice da linha escolhida
idx_escolhido = [i for (label, i) in opcoes if label == label_escolhida][0]
linha = df.loc[idx_escolhido]

st.subheader("Detalhes da partida selecionada")
col1, col2 = st.columns(2)
with col1:
    st.metric("Jogador 1 (colunas do modelo)", linha["winner_name"])
with col2:
    st.metric("Jogador 2", linha["loser_name"])

st.caption("Obs.: as features do modelo são calculadas como **Jogador 1 − Jogador 2**.")

if st.button("📊 Fazer previsão para esta partida"):
    # Montar vetor de features
    X = linha[COLS_MODELO].values.reshape(1, -1)

    prob = model.predict_proba(X)[0, 1]  # probabilidade da classe 1
    pred = model.predict(X)[0]

    if pred == 1:
        vencedor_previsto = linha["winner_name"]
    else:
        vencedor_previsto = linha["loser_name"]

    st.success(f"✅ Vencedor previsto pelo modelo: **{vencedor_previsto}**")
    st.write(f"Probabilidade (classe 1 = Jogador 1 vencer): **{prob:.3f}**")

    # Como é uma partida histórica, podemos mostrar o resultado real:
    vencedor_real = linha["winner_name"]
    if vencedor_previsto == vencedor_real:
        st.info(f"🟢 O modelo **acertou**. Vencedor real: **{vencedor_real}**.")
    else:
        st.error(f"🔴 O modelo **errou**. Vencedor real: **{vencedor_real}**.")
