import numpy as np
import streamlit as st
import jijmodeling as jm
import jijmodeling_transpiler as jmt
from openjij import SQASampler
sampler = SQASampler()


#使用する調味料
seasoning_names = [
    "醤油",
    "みりん",
    "塩",
    "酒",
    "酢",
    "だし",
    "ごま油",
    "オリーブオイル",
    "砂糖",
    "マヨネーズ",
    "ソース",
    "スパイス",
    "豆板醤",
    "バター",
    "カレー粉"
]

#調味料の味覚ベクトル　["旨味", "苦味", "塩味", "酸味", "甘味", "渋味"]
seasoning =  [[8, 1, 6, 1, 2, 1],
              [3, 1, 1, 1, 8, 1],
              [1, 1, 10, 1, 1, 1],
              [2, 1, 1, 2, 3, 1],
              [1, 1, 1, 9, 1, 1],
              [9, 1, 1, 1, 1, 1],
              [3, 3, 1, 1, 2, 2],
              [2, 3, 1, 1, 1, 3],
              [1, 1, 1, 1, 10, 1],
              [4, 2, 6, 3, 3, 1],
              [6, 2, 6, 4, 4, 1],
              [2, 6, 1, 1, 1, 3],
              [3, 2, 2, 1, 1, 6],
              [5, 4, 3, 2, 5, 1],
              [2, 5, 2, 1, 2, 6]]
seasoning = np.array(seasoning)

#料理の味覚ベクトル
dish_flavor_profiles = {
    "ラーメン":       [8.0, 1.0, 7.0, 1.0, 2.0, 1.0],
    "トマトスープ":   [6.0, 1.0, 4.0, 4.5, 3.5, 1.0],
    "カレー":         [7.0, 2.0, 6.0, 2.0, 3.0, 1.0],
    "味噌汁":         [9.0, 0.5, 6.0, 1.0, 1.5, 1.0],
    "すし酢ご飯":     [5.0, 0.5, 4.0, 6.0, 3.0, 0.5],
    "お好み焼き":     [8.0, 1.0, 6.0, 2.0, 4.0, 1.0],
}


selected_dish = st.selectbox("よく食べる料理を選んでください", list(dish_flavor_profiles.keys()))
base_dish_flavor = np.array(dish_flavor_profiles[selected_dish])

num_seasonings = st.slider("使う調味料の数(※この通りになるとは限りません。値はあくまで目安です。)", 1, len(seasoning_names), 1)

if st.button("違う味を楽しむ"):
  N = len(seasoning)
  M = 6
  unit = []
  for m in range(M):
    unit.append(m)
  unit = np.array(unit)

  x = jm.BinaryVar("x", shape=(N, M))
  i = jm.Element("i", belong_to=(0,N))
  j = jm.Element("j", belong_to=(0,M))
  k = jm.Element("k", belong_to=(0,M))
  l = jm.Element("l", belong_to=(0,M))
  Z = jm.Placeholder("Z", ndim=4)

  problem = jm.Problem("Maximizing the distance", sense = jm.ProblemSense.MAXIMIZE)
  problem += jm.sum([i,j,k,l], Z[i,j,k,l] * x[i,k] * x[j,l])
  problem += jm.Constraint("one unit",jm.sum(k, x[i,k]) == 1, forall=i)
  problem += jm.Constraint("num seasonings", jm.sum(i, x[i,0])==(len(seasoning) - num_seasonings))

  Zmat = np.zeros((N,N,M,M))
  for i in range(N):
    for j in range(N):
      for k in range(M):
        for l in range(M):
          if k == 0 and l == 0:
            Zmat[(i,j,k,l)] = 0
          else:
            difference = (((seasoning[i] * (k)) + (seasoning[j] * (l))) / (k + l)) - base_dish_flavor
            score = np.dot(difference, difference.T)
            Zmat[(i,j,k,l)] = score
  
  instance_data = {"Z": Zmat}
  compiled_model = jmt.core.compile_model(problem, instance_data, {})
  pubo_builder = jmt.core.pubo.transpile_to_pubo(compiled_model=compiled_model)
  qubo, const = pubo_builder.get_qubo_dict(multipliers={"one unit": 10.0, "num seasonings": 10})

  num_reads=10
  sampleset = sampler.sample_qubo(qubo, num_reads=num_reads)

  for i in range(num_reads):
    if sampleset.record[i][1] == sampleset.data_vectors["energy"].min():
      best_sample = sampleset.record[i][0].reshape(N,M)
      break

  selected_seasoning = []
  unit_num = []
  for n in range(N):
    for m in range(M):
      if best_sample[n][m] == 1 and m > 0:
        selected_seasoning.append(seasoning_names[n])
        unit_num.append(m)
  
  st.subheader("選ばれた調味料(比率)")
  for i in range(len(selected_seasoning)):
    st.write(selected_seasoning[i],":", unit_num[i])