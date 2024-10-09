import streamlit as st
import pandas as pd
from io import StringIO
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

def carregar_dados(arquivo):
    """Carrega o dataset de transações da padaria a partir de um arquivo CSV enviado pelo usuário."""
    try:
        return pd.read_csv(arquivo)
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {e}")
        return None

def preparar_transacoes(df):
    """Prepara as transações a partir do dataframe."""
    transacoes = df.groupby('TransactionNo')['Items'].apply(list).tolist()
    return transacoes

def aplicar_apriori(transacoes, min_support, min_confidence):
    """Aplica o algoritmo Apriori para gerar regras de associação."""
    te = TransactionEncoder()
    transacao_te = te.fit(transacoes).transform(transacoes)
    df_transacao = pd.DataFrame(transacao_te, columns=te.columns_)
    
    items_frequentes_apriori = apriori(df_transacao, min_support=min_support, use_colnames=True)
    
    if items_frequentes_apriori.empty:
        st.warning("Nenhum item frequente encontrado com o suporte mínimo fornecido.")
        return None
    
    regras_apriori = association_rules(items_frequentes_apriori, metric='confidence', min_threshold=min_confidence)
    
    if regras_apriori.empty:
        st.warning("Nenhuma regra de associação encontrada com a confiança mínima fornecida.")
        return None
    
    # Traduzir colunas e adicionar coluna de combinação antecedente -> consequente
    regras_apriori.rename(columns={
        'antecedents': 'Antecedentes',
        'consequents': 'Consequentes',
        'support': 'Suporte',
        'confidence': 'Confiança'
    }, inplace=True)
    regras_apriori['Combinação'] = regras_apriori['Antecedentes'].apply(lambda x: ', '.join(list(x))) + ' -> ' + regras_apriori['Consequentes'].apply(lambda x: ', '.join(list(x)))
    
    return regras_apriori

def main():
    st.title('Regras de Associação com Apriori')
    st.write('Este aplicativo permite gerar regras de associação a partir de um dataset de transações utilizando o algoritmo Apriori.')
    
    arquivo = st.file_uploader('Faça o upload do arquivo de dados (CSV):', type=['csv'], accept_multiple_files=False)
    min_support = st.slider('Suporte Mínimo:', min_value=0.01, max_value=0.5, value=0.02, step=0.01)
    min_confidence = st.slider('Confiança Mínima:', min_value=0.1, max_value=1.0, value=0.5, step=0.1)
    
    # Aumentar o tamanho máximo do arquivo para 1GB
    # st.set_option('server.maxUploadSize', 1024)
    
    if st.button('Carregar e Aplicar Apriori') and arquivo is not None:
        df = carregar_dados(arquivo)
        if df is not None:
            transacoes = preparar_transacoes(df)
            regras = aplicar_apriori(transacoes, min_support=min_support, min_confidence=min_confidence)
            
            if regras is not None:
                st.write('Regras de Associação Encontradas:')
                st.markdown("""
                **Legenda das Colunas:**
                - **Combinação**: Representa a relação entre os itens encontrados, onde o conjunto de 'Antecedentes' está associado ao conjunto de 'Consequentes'.
                - **Suporte**: Proporção de transações no dataset que contém os itens da regra.
                - **Confiança**: Probabilidade de que o 'Consequente' ocorra dado que o 'Antecedente' ocorreu.
                """)
                st.dataframe(regras[['Combinação', 'Suporte', 'Confiança']])
                st.download_button('Baixar Regras em CSV', regras.to_csv(index=False), file_name='regras_associacao.csv', mime='text/csv')

if __name__ == "__main__":
    main()