import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import faiss
import json
from fastapi import FastAPI
from flask import Flask

app = Flask(__name__)

@app.route("/")
def ping():
    return "It's works"

@app.route("/find/<user_id>", methods=['GET'])
def find_restaurant(user_id):
    # --- Mock ---
    with open('data.json', 'r') as file:
        mock_data = json.load(file)

    # --- Entrada do usuário ---
    # id_digitado = int(input("Digite o ID do usuário: "))
    id_digitado = int(user_id)

    avaliacoes_df = pd.DataFrame(mock_data['Avaliacao'])

    # Definir o usuário para a recomendação
    usuario_id_para_recomendar = id_digitado

    # Restaurantes já avaliados
    restaurantes_ja_visitados = set(
        avaliacoes_df[avaliacoes_df['id_usuario'] == usuario_id_para_recomendar]['id_restaurante']
    )

    # Restaurantes base (avaliados com nota >= 4)
    avaliacoes_positivas = avaliacoes_df[
        (avaliacoes_df['id_usuario'] == usuario_id_para_recomendar) & (avaliacoes_df['nota'] >= 3)
    ]
    restaurantes_base_recomendacao = avaliacoes_positivas['id_restaurante'].tolist()

    # Restaurantes DataFrame
    restaurantes_df = pd.DataFrame(mock_data['Restaurante'])

    # --- Pré-processamento ---
    # 1. OneHotEncoding da categoria
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    categorias_encoded = ohe.fit_transform(restaurantes_df[['tipo_restaurante']])
    categoria_df = pd.DataFrame(categorias_encoded, columns=ohe.get_feature_names_out(['tipo_restaurante']))

    # 2. Normalização de preço
    scaler = MinMaxScaler()
    precos_escalados = scaler.fit_transform(restaurantes_df[['preco_medio']])
    precos_df = pd.DataFrame(precos_escalados, columns=['preco_medio_escalado'])

    # 3. Vetores finais
    restaurante_features_df = pd.concat([categoria_df, precos_df], axis=1)
    restaurant_vectors = restaurante_features_df.to_numpy().astype('float32')

    # --- FAISS ---
    dimension = restaurant_vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(restaurant_vectors)

    # --- Recomendações ---
    recomendacoes_finais = set()
    k = 5
    for ref_id in restaurantes_base_recomendacao:
        ref_index = restaurantes_df[restaurantes_df['idRestaurante'] == ref_id].index[0]
        user_vector = restaurant_vectors[ref_index].reshape(1, -1)
        distances, indices = index.search(user_vector, k)
        for i in indices[0]:
            restaurante_id_recomendado = restaurantes_df.iloc[i]['idRestaurante']
            recomendacoes_finais.add(restaurante_id_recomendado)

    # Remover já visitados
    recomendacoes_filtradas = [r for r in recomendacoes_finais if r not in restaurantes_ja_visitados]
    restaurantes_recomendados_df = restaurantes_df[restaurantes_df['idRestaurante'].isin(recomendacoes_filtradas)]

    # --- Função auxiliar ---
    def find_nome_por_id(id_busca):
        return next((u["nome"] for u in mock_data["Usuarios"] if u["idUsuarios"] == id_busca), None).upper()

    # --- Output ---
    recommended_restaurants = []

    print(f"\n--- RECOMENDAÇÃO DE RESTAURANTES PARA O USUÁRIO {find_nome_por_id(id_digitado)} ---")
    if restaurantes_recomendados_df.empty:
        print("Nenhuma nova recomendação encontrada.")
        return {"message": "Nenhuma nova recomendação encontrada.", "restaurants": []}
    else:
        for _, row in restaurantes_recomendados_df.iterrows():
            print(f"Nome: {row['nome']} | Categoria: {row['tipo_restaurante']} | Preço Médio: R${row['preco_medio']:.2f}")
            restaurant_data = {
                "nome": row['nome'],
                "categoria": row['tipo_restaurante'],
                "preco_medio": float(f"{row['preco_medio']:.2f}")
            }
            
            recommended_restaurants.append(restaurant_data)

        return {
            "user": find_nome_por_id(id_digitado), 
            "total_restaurants": len(recommended_restaurants),
            "restaurants": recommended_restaurants
        }

if __name__ == '__main__':
    app.run(debug=True)