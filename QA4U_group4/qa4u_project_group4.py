import numpy as np
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import jijmodeling as jm
import jijmodeling_transpiler as jmt
from openjij import SASampler

# 使用する調味料と味覚ベクトル
seasoning_names = [
    "醤油", "みりん", "塩", "酒", "酢", "だし", "ごま油", "オリーブオイル",
    "砂糖", "マヨネーズ", "ソース", "スパイス", "豆板醤", "バター", "カレー粉"
]
flavor_labels = ["旨味", "苦味", "塩味", "酸味", "甘味", "渋味"]

seasoning_data = {
    "醤油":        [8, 1, 6, 1, 2, 1],
    "みりん":      [3, 1, 1, 1, 8, 1],
    "塩":          [1, 1, 10, 1, 1, 1],
    "酒":          [2, 1, 1, 2, 3, 1],
    "酢":          [1, 1, 1, 9, 1, 1],
    "だし":        [9, 1, 1, 1, 1, 1],
    "ごま油":      [3, 3, 1, 1, 2, 2],
    "オリーブオイル": [2, 3, 1, 1, 1, 3],
    "砂糖":        [1, 1, 1, 1, 10, 1],
    "マヨネーズ":  [4, 2, 6, 3, 3, 1],
    "ソース":      [6, 2, 6, 4, 4, 1],
    "スパイス":    [2, 6, 1, 1, 1, 3],
    "豆板醤":      [3, 2, 2, 1, 1, 6],
    "バター":      [5, 4, 3, 2, 5, 1],
    "カレー粉":    [2, 5, 2, 1, 2, 6]
}
seasoning_array = np.array(list(seasoning_data.values()))


# Streamlit UI
st.title("量子アニーリングで新しい味を作るアプリ")
st.write("ベースとなる味覚を1〜10で設定してください。")

base_flavor = []
for flavor in flavor_labels:
  val = st.slider(f"{flavor}", 1, 10, 5)
  base_flavor.append(val)
base_flavor = np.array(base_flavor)

num_selected = st.slider("使いたい調味料の数", 1, len(seasoning_names), 1)

if st.button("最適な調味料を選ぶ"):
  #QUBO最適化
  N = len(seasoning_names)
  x = jm.BinaryVar("x", shape=(N,))
  i = jm.Element("i", belong_to=(0,N))
  j = jm.Element("j", belong_to=(0,N))
  Z = jm.Placeholder("Z", ndim=2)
  problem = jm.Problem("Maximize flavor distance", sense=jm.ProblemSense.MAXIMIZE)
  problem += jm.sum([i,j], Z[i,j] * x[i] * x[j])
  problem += jm.Constraint("num_selected_seasonings", jm.sum(i, x[i]) == num_selected)

  #ベースとの差分からQUBOを作る
  base_difference = seasoning_array - base_flavor
  Zmat = np.dot(base_difference, base_difference.T)
  instance_data = {"Z": Zmat}
  compiled_model = jmt.core.compile_model(problem, instance_data, {})
  pubo_builder = jmt.core.pubo.transpile_to_pubo(compiled_model)
  qubo, const = pubo_builder.get_qubo_dict(multipliers={"num_selected_seasonings": 10.0})

  #サンプル取得
  sampler = SASampler()
  sampleset = sampler.sample_qubo(qubo, num_reads=10)
  result_index = np.array(sampleset.record[0][0], dtype=bool)
  selected = np.array(seasoning_names)[result_index]

  st.subheader("選ばれた調味料")
  st.write(", ".join(selected))