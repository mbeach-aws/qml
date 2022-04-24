Last update: 2022-04-24  00:00:47 (All times shown in Eastern time)
# List of differences in demonstration outputs

# Table of contents

1. [tutorial_qnn_module_tf.html](#demo0)
2. [tutorial_variational_classifier.html](#demo1)
3. [tutorial_vqe.html](#demo2)
4. [tutorial_expressivity_fourier_series.html](#demo3)
5. [tutorial_qgrnn.html](#demo4)
6. [tutorial_adaptive_circuits.html](#demo5)
7. [tutorial_backprop.html](#demo6)
8. [tutorial_rosalin.html](#demo7)
9. [tutorial_error_mitigation.html](#demo8)
10. [tutorial_measurement_optimize.html](#demo9)
11. [tutorial_quantum_chemistry.html](#demo10)
12. [tutorial_chemical_reactions.html](#demo11)
13. [tutorial_jax_transformations.html](#demo12)
14. [tutorial_vqe_spin_sectors.html](#demo13)
15. [tutorial_sc_qubits.html](#demo14)
16. [tutorial_quantum_transfer_learning.html](#demo15)
17. [tutorial_vqe_qng.html](#demo16)
18. [tutorial_quanvolution.html](#demo17)


Number of demos different/all demos: 18/60

## 1. tutorial_qnn_module_tf.html <a name="demo0"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_qnn_module_tf.html):

```
30/30 - 11s - loss: 0.4997 - accuracy: 0.5000 - val_loss: 0.5081 - val_accuracy: 0.4400 - 11s/epoch - 371ms/step
30/30 - 11s - loss: 0.4673 - accuracy: 0.6200 - val_loss: 0.4488 - val_accuracy: 0.6200 - 11s/epoch - 365ms/step
30/30 - 11s - loss: 0.3230 - accuracy: 0.8267 - val_loss: 0.2562 - val_accuracy: 0.8400 - 11s/epoch - 375ms/step
30/30 - 11s - loss: 0.2124 - accuracy: 0.8867 - val_loss: 0.1997 - val_accuracy: 0.8400 - 11s/epoch - 367ms/step
30/30 - 11s - loss: 0.1800 - accuracy: 0.8933 - val_loss: 0.1841 - val_accuracy: 0.8400 - 11s/epoch - 373ms/step
30/30 - 11s - loss: 0.1593 - accuracy: 0.8667 - val_loss: 0.2177 - val_accuracy: 0.8400 - 11s/epoch - 373ms/step
30/30 - 22s - loss: 0.5189 - accuracy: 0.4000 - val_loss: 0.4945 - val_accuracy: 0.5400 - 22s/epoch - 742ms/step
30/30 - 22s - loss: 0.4822 - accuracy: 0.6200 - val_loss: 0.4412 - val_accuracy: 0.7200 - 22s/epoch - 737ms/step
30/30 - 22s - loss: 0.3850 - accuracy: 0.7133 - val_loss: 0.2898 - val_accuracy: 0.7800 - 22s/epoch - 737ms/step
30/30 - 22s - loss: 0.2720 - accuracy: 0.7867 - val_loss: 0.2185 - val_accuracy: 0.8200 - 22s/epoch - 739ms/step
30/30 - 22s - loss: 0.2056 - accuracy: 0.8400 - val_loss: 0.1893 - val_accuracy: 0.8400 - 22s/epoch - 727ms/step
30/30 - 22s - loss: 0.1753 - accuracy: 0.8400 - val_loss: 0.1856 - val_accuracy: 0.8400 - 22s/epoch - 739ms/step
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_qnn_module_tf.html):

```
30/30 - 13s - loss: 0.4997 - accuracy: 0.5000 - val_loss: 0.5081 - val_accuracy: 0.4400 - 13s/epoch - 431ms/step
30/30 - 13s - loss: 0.4673 - accuracy: 0.6200 - val_loss: 0.4488 - val_accuracy: 0.6200 - 13s/epoch - 430ms/step
30/30 - 13s - loss: 0.3230 - accuracy: 0.8267 - val_loss: 0.2562 - val_accuracy: 0.8400 - 13s/epoch - 430ms/step
30/30 - 13s - loss: 0.2124 - accuracy: 0.8867 - val_loss: 0.1997 - val_accuracy: 0.8400 - 13s/epoch - 440ms/step
30/30 - 13s - loss: 0.1800 - accuracy: 0.8933 - val_loss: 0.1841 - val_accuracy: 0.8400 - 13s/epoch - 432ms/step
30/30 - 13s - loss: 0.1593 - accuracy: 0.8667 - val_loss: 0.2177 - val_accuracy: 0.8400 - 13s/epoch - 439ms/step
30/30 - 26s - loss: 0.5189 - accuracy: 0.4000 - val_loss: 0.4945 - val_accuracy: 0.5400 - 26s/epoch - 853ms/step
30/30 - 26s - loss: 0.4822 - accuracy: 0.6200 - val_loss: 0.4412 - val_accuracy: 0.7200 - 26s/epoch - 853ms/step
30/30 - 26s - loss: 0.3850 - accuracy: 0.7133 - val_loss: 0.2898 - val_accuracy: 0.7800 - 26s/epoch - 862ms/step
30/30 - 25s - loss: 0.2720 - accuracy: 0.7867 - val_loss: 0.2185 - val_accuracy: 0.8200 - 25s/epoch - 850ms/step
30/30 - 27s - loss: 0.2056 - accuracy: 0.8400 - val_loss: 0.1893 - val_accuracy: 0.8400 - 27s/epoch - 901ms/step
30/30 - 26s - loss: 0.1753 - accuracy: 0.8400 - val_loss: 0.1856 - val_accuracy: 0.8400 - 26s/epoch - 863ms/step
```

---

## 2. tutorial_variational_classifier.html <a name="demo1"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_variational_classifier.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Iter:     2 | Cost: 1.9287800 | Accuracy: 0.5000000
Iter:     3 | Cost: 2.0341238 | Accuracy: 0.5000000
Iter:     4 | Cost: 1.6372574 | Accuracy: 0.5000000
Iter:     5 | Cost: 1.3025395 | Accuracy: 0.6250000
Iter:     6 | Cost: 1.4555019 | Accuracy: 0.3750000
Iter:     7 | Cost: 1.4492786 | Accuracy: 0.5000000
Iter:     8 | Cost: 0.6510286 | Accuracy: 0.8750000
Iter:     9 | Cost: 0.0566074 | Accuracy: 1.0000000
Iter:    10 | Cost: 0.0053045 | Accuracy: 1.0000000
Iter:    11 | Cost: 0.0809483 | Accuracy: 1.0000000
Iter:    12 | Cost: 0.1115426 | Accuracy: 1.0000000
Iter:    13 | Cost: 0.1460257 | Accuracy: 1.0000000
Iter:    14 | Cost: 0.0877037 | Accuracy: 1.0000000
Iter:    15 | Cost: 0.0361311 | Accuracy: 1.0000000
Iter:    16 | Cost: 0.0040937 | Accuracy: 1.0000000
Iter:    17 | Cost: 0.0004899 | Accuracy: 1.0000000
Iter:    18 | Cost: 0.0005290 | Accuracy: 1.0000000
Iter:    19 | Cost: 0.0024304 | Accuracy: 1.0000000
Iter:    20 | Cost: 0.0062137 | Accuracy: 1.0000000
Iter:    21 | Cost: 0.0088864 | Accuracy: 1.0000000
Iter:    22 | Cost: 0.0201912 | Accuracy: 1.0000000
Iter:    23 | Cost: 0.0060335 | Accuracy: 1.0000000
Iter:    24 | Cost: 0.0036153 | Accuracy: 1.0000000
Iter:    25 | Cost: 0.0012741 | Accuracy: 1.0000000
Iter:     2 | Cost: 1.3309953 | Acc train: 0.4933333 | Acc validation: 0.5600000
Iter:     3 | Cost: 1.1582178 | Acc train: 0.4533333 | Acc validation: 0.5600000
Iter:     4 | Cost: 0.9795035 | Acc train: 0.4800000 | Acc validation: 0.5600000
Iter:     5 | Cost: 0.8857893 | Acc train: 0.6400000 | Acc validation: 0.7600000
Iter:     6 | Cost: 0.8587935 | Acc train: 0.7066667 | Acc validation: 0.7600000
Iter:     7 | Cost: 0.8496204 | Acc train: 0.7200000 | Acc validation: 0.6800000
Iter:     8 | Cost: 0.8200972 | Acc train: 0.7333333 | Acc validation: 0.6800000
Iter:     9 | Cost: 0.8027511 | Acc train: 0.7466667 | Acc validation: 0.6800000
Iter:    10 | Cost: 0.7695152 | Acc train: 0.8000000 | Acc validation: 0.7600000
Iter:    11 | Cost: 0.7437432 | Acc train: 0.8133333 | Acc validation: 0.9600000
Iter:    12 | Cost: 0.7569196 | Acc train: 0.6800000 | Acc validation: 0.7600000
Iter:    13 | Cost: 0.7887487 | Acc train: 0.6533333 | Acc validation: 0.7200000
Iter:    14 | Cost: 0.8401458 | Acc train: 0.6133333 | Acc validation: 0.6400000
Iter:    15 | Cost: 0.8651830 | Acc train: 0.5600000 | Acc validation: 0.6000000
Iter:    16 | Cost: 0.8726113 | Acc train: 0.5600000 | Acc validation: 0.6000000
Iter:    17 | Cost: 0.8389732 | Acc train: 0.6133333 | Acc validation: 0.6400000
Iter:    18 | Cost: 0.8004839 | Acc train: 0.6266667 | Acc validation: 0.6400000
Iter:    19 | Cost: 0.7592044 | Acc train: 0.6800000 | Acc validation: 0.7600000
Iter:    20 | Cost: 0.7332872 | Acc train: 0.7733333 | Acc validation: 0.8000000
Iter:    21 | Cost: 0.7184319 | Acc train: 0.8800000 | Acc validation: 0.9600000
Iter:    22 | Cost: 0.7336631 | Acc train: 0.8133333 | Acc validation: 0.7200000
Iter:    23 | Cost: 0.7503193 | Acc train: 0.6533333 | Acc validation: 0.6400000
Iter:    24 | Cost: 0.7608474 | Acc train: 0.5866667 | Acc validation: 0.5200000
Iter:    25 | Cost: 0.7443533 | Acc train: 0.6533333 | Acc validation: 0.6400000
Iter:    26 | Cost: 0.7383224 | Acc train: 0.7066667 | Acc validation: 0.6400000
Iter:    27 | Cost: 0.7322155 | Acc train: 0.7466667 | Acc validation: 0.6800000
Iter:    28 | Cost: 0.7384175 | Acc train: 0.6533333 | Acc validation: 0.6400000
Iter:    29 | Cost: 0.7393227 | Acc train: 0.6400000 | Acc validation: 0.6400000
Iter:    30 | Cost: 0.7251903 | Acc train: 0.7200000 | Acc validation: 0.6800000
Iter:    31 | Cost: 0.7125040 | Acc train: 0.7866667 | Acc validation: 0.6800000
Iter:    32 | Cost: 0.6932690 | Acc train: 0.9066667 | Acc validation: 0.9200000
Iter:    33 | Cost: 0.6800562 | Acc train: 0.9200000 | Acc validation: 1.0000000
Iter:    34 | Cost: 0.6763140 | Acc train: 0.9200000 | Acc validation: 0.9600000
Iter:    35 | Cost: 0.6790040 | Acc train: 0.8933333 | Acc validation: 0.8800000
Iter:    36 | Cost: 0.6936199 | Acc train: 0.7600000 | Acc validation: 0.7200000
Iter:    37 | Cost: 0.6767184 | Acc train: 0.8266667 | Acc validation: 0.8000000
Iter:    38 | Cost: 0.6712470 | Acc train: 0.8266667 | Acc validation: 0.8000000
Iter:    39 | Cost: 0.6747390 | Acc train: 0.7600000 | Acc validation: 0.7600000
Iter:    40 | Cost: 0.6845696 | Acc train: 0.6666667 | Acc validation: 0.6400000
Iter:    41 | Cost: 0.6703303 | Acc train: 0.7333333 | Acc validation: 0.7200000
Iter:    42 | Cost: 0.6238401 | Acc train: 0.8933333 | Acc validation: 0.8400000
Iter:    43 | Cost: 0.6028185 | Acc train: 0.9066667 | Acc validation: 0.9200000
Iter:    44 | Cost: 0.5936355 | Acc train: 0.9066667 | Acc validation: 0.9200000
Iter:    45 | Cost: 0.5722417 | Acc train: 0.9200000 | Acc validation: 0.9600000
Iter:    46 | Cost: 0.5617923 | Acc train: 0.9200000 | Acc validation: 0.9600000
Iter:    47 | Cost: 0.5413240 | Acc train: 0.9466667 | Acc validation: 1.0000000
Iter:    48 | Cost: 0.5239643 | Acc train: 0.9466667 | Acc validation: 1.0000000
Iter:    49 | Cost: 0.5100842 | Acc train: 0.9466667 | Acc validation: 1.0000000
Iter:    50 | Cost: 0.5006861 | Acc train: 0.9466667 | Acc validation: 1.0000000
Iter:    51 | Cost: 0.4821672 | Acc train: 0.9466667 | Acc validation: 1.0000000
Iter:    52 | Cost: 0.4579575 | Acc train: 0.9600000 | Acc validation: 1.0000000
Iter:    53 | Cost: 0.4397479 | Acc train: 1.0000000 | Acc validation: 1.0000000
Iter:    54 | Cost: 0.4326879 | Acc train: 0.9600000 | Acc validation: 0.9200000
Iter:    55 | Cost: 0.4351511 | Acc train: 0.9466667 | Acc validation: 0.9200000
Iter:    56 | Cost: 0.4328988 | Acc train: 0.9333333 | Acc validation: 0.9200000
Iter:    57 | Cost: 0.4149892 | Acc train: 0.9333333 | Acc validation: 0.9200000
Iter:    58 | Cost: 0.3755246 | Acc train: 0.9600000 | Acc validation: 0.9200000
Iter:    59 | Cost: 0.3468994 | Acc train: 1.0000000 | Acc validation: 1.0000000
Iter:    60 | Cost: 0.3297071 | Acc train: 1.0000000 | Acc validation: 1.0000000
 </code>
 </pre>
 </details>

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_variational_classifier.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Iter:     2 | Cost: 1.9717733 | Accuracy: 0.5000000
Iter:     3 | Cost: 1.8182812 | Accuracy: 0.5000000
Iter:     4 | Cost: 1.5042404 | Accuracy: 0.5000000
Iter:     5 | Cost: 1.1477739 | Accuracy: 0.5000000
Iter:     6 | Cost: 1.2734990 | Accuracy: 0.6250000
Iter:     7 | Cost: 0.8290628 | Accuracy: 0.5000000
Iter:     8 | Cost: 0.3226183 | Accuracy: 1.0000000
Iter:     9 | Cost: 0.1436206 | Accuracy: 1.0000000
Iter:    10 | Cost: 0.2982810 | Accuracy: 1.0000000
Iter:    11 | Cost: 0.3064355 | Accuracy: 1.0000000
Iter:    12 | Cost: 0.1682335 | Accuracy: 1.0000000
Iter:    13 | Cost: 0.0892512 | Accuracy: 1.0000000
Iter:    14 | Cost: 0.0381562 | Accuracy: 1.0000000
Iter:    15 | Cost: 0.0170359 | Accuracy: 1.0000000
Iter:    16 | Cost: 0.0109353 | Accuracy: 1.0000000
Iter:    17 | Cost: 0.0108388 | Accuracy: 1.0000000
Iter:    18 | Cost: 0.0139196 | Accuracy: 1.0000000
Iter:    19 | Cost: 0.0123980 | Accuracy: 1.0000000
Iter:    20 | Cost: 0.0085416 | Accuracy: 1.0000000
Iter:    21 | Cost: 0.0053549 | Accuracy: 1.0000000
Iter:    22 | Cost: 0.0065759 | Accuracy: 1.0000000
Iter:    23 | Cost: 0.0024883 | Accuracy: 1.0000000
Iter:    24 | Cost: 0.0029102 | Accuracy: 1.0000000
Iter:    25 | Cost: 0.0023471 | Accuracy: 1.0000000
Iter:     2 | Cost: 1.3312057 | Acc train: 0.4933333 | Acc validation: 0.5600000
Iter:     3 | Cost: 1.1589332 | Acc train: 0.4533333 | Acc validation: 0.5600000
Iter:     4 | Cost: 0.9806934 | Acc train: 0.4800000 | Acc validation: 0.5600000
Iter:     5 | Cost: 0.8865623 | Acc train: 0.6133333 | Acc validation: 0.7600000
Iter:     6 | Cost: 0.8580769 | Acc train: 0.6933333 | Acc validation: 0.7600000
Iter:     7 | Cost: 0.8473132 | Acc train: 0.7200000 | Acc validation: 0.6800000
Iter:     8 | Cost: 0.8177533 | Acc train: 0.7333333 | Acc validation: 0.6800000
Iter:     9 | Cost: 0.8001100 | Acc train: 0.7466667 | Acc validation: 0.6800000
Iter:    10 | Cost: 0.7681053 | Acc train: 0.8000000 | Acc validation: 0.7600000
Iter:    11 | Cost: 0.7440015 | Acc train: 0.8133333 | Acc validation: 0.9600000
Iter:    12 | Cost: 0.7583777 | Acc train: 0.6800000 | Acc validation: 0.7600000
Iter:    13 | Cost: 0.7896372 | Acc train: 0.6533333 | Acc validation: 0.7200000
Iter:    14 | Cost: 0.8397790 | Acc train: 0.6133333 | Acc validation: 0.6400000
Iter:    15 | Cost: 0.8632423 | Acc train: 0.5733333 | Acc validation: 0.6000000
Iter:    16 | Cost: 0.8693517 | Acc train: 0.5733333 | Acc validation: 0.6000000
Iter:    17 | Cost: 0.8350625 | Acc train: 0.6266667 | Acc validation: 0.6400000
Iter:    18 | Cost: 0.7966558 | Acc train: 0.6266667 | Acc validation: 0.6400000
Iter:    19 | Cost: 0.7563381 | Acc train: 0.6800000 | Acc validation: 0.7600000
Iter:    20 | Cost: 0.7315459 | Acc train: 0.7733333 | Acc validation: 0.8000000
Iter:    21 | Cost: 0.7182359 | Acc train: 0.8800000 | Acc validation: 0.9600000
Iter:    22 | Cost: 0.7339132 | Acc train: 0.8133333 | Acc validation: 0.7200000
Iter:    23 | Cost: 0.7498571 | Acc train: 0.6533333 | Acc validation: 0.6400000
Iter:    24 | Cost: 0.7593763 | Acc train: 0.6000000 | Acc validation: 0.6400000
Iter:    25 | Cost: 0.7423832 | Acc train: 0.6800000 | Acc validation: 0.6400000
Iter:    26 | Cost: 0.7361186 | Acc train: 0.7333333 | Acc validation: 0.6800000
Iter:    27 | Cost: 0.7300201 | Acc train: 0.7600000 | Acc validation: 0.6800000
Iter:    28 | Cost: 0.7360923 | Acc train: 0.6666667 | Acc validation: 0.6400000
Iter:    29 | Cost: 0.7371325 | Acc train: 0.6533333 | Acc validation: 0.6400000
Iter:    30 | Cost: 0.7234520 | Acc train: 0.7466667 | Acc validation: 0.6800000
Iter:    31 | Cost: 0.7112089 | Acc train: 0.8133333 | Acc validation: 0.7600000
Iter:    32 | Cost: 0.6923495 | Acc train: 0.9200000 | Acc validation: 0.9200000
Iter:    33 | Cost: 0.6792510 | Acc train: 0.9200000 | Acc validation: 1.0000000
Iter:    34 | Cost: 0.6756795 | Acc train: 0.9200000 | Acc validation: 0.9600000
Iter:    35 | Cost: 0.6787274 | Acc train: 0.8933333 | Acc validation: 0.8000000
Iter:    36 | Cost: 0.6938181 | Acc train: 0.7466667 | Acc validation: 0.6800000
Iter:    37 | Cost: 0.6765225 | Acc train: 0.8266667 | Acc validation: 0.8000000
Iter:    38 | Cost: 0.6708057 | Acc train: 0.8266667 | Acc validation: 0.8000000
Iter:    39 | Cost: 0.6740248 | Acc train: 0.7600000 | Acc validation: 0.7200000
Iter:    40 | Cost: 0.6835014 | Acc train: 0.6666667 | Acc validation: 0.6400000
Iter:    41 | Cost: 0.6686811 | Acc train: 0.7333333 | Acc validation: 0.7200000
Iter:    42 | Cost: 0.6217025 | Acc train: 0.8933333 | Acc validation: 0.8400000
Iter:    43 | Cost: 0.6004307 | Acc train: 0.9066667 | Acc validation: 0.9200000
Iter:    44 | Cost: 0.5910768 | Acc train: 0.9066667 | Acc validation: 0.9200000
Iter:    45 | Cost: 0.5694770 | Acc train: 0.9200000 | Acc validation: 0.9600000
Iter:    46 | Cost: 0.5588385 | Acc train: 0.9200000 | Acc validation: 0.9600000
Iter:    47 | Cost: 0.5381473 | Acc train: 0.9466667 | Acc validation: 1.0000000
Iter:    48 | Cost: 0.5205658 | Acc train: 0.9466667 | Acc validation: 1.0000000
Iter:    49 | Cost: 0.5064983 | Acc train: 0.9466667 | Acc validation: 1.0000000
Iter:    50 | Cost: 0.4969733 | Acc train: 0.9466667 | Acc validation: 1.0000000
Iter:    51 | Cost: 0.4782257 | Acc train: 0.9466667 | Acc validation: 1.0000000
Iter:    52 | Cost: 0.4536953 | Acc train: 0.9600000 | Acc validation: 1.0000000
Iter:    53 | Cost: 0.4350300 | Acc train: 1.0000000 | Acc validation: 1.0000000
Iter:    54 | Cost: 0.4272909 | Acc train: 0.9733333 | Acc validation: 0.9600000
Iter:    55 | Cost: 0.4288929 | Acc train: 0.9466667 | Acc validation: 0.9200000
Iter:    56 | Cost: 0.4261037 | Acc train: 0.9333333 | Acc validation: 0.9200000
Iter:    57 | Cost: 0.4082663 | Acc train: 0.9466667 | Acc validation: 0.9200000
Iter:    58 | Cost: 0.3698736 | Acc train: 0.9600000 | Acc validation: 0.9200000
Iter:    59 | Cost: 0.3420686 | Acc train: 1.0000000 | Acc validation: 1.0000000
Iter:    60 | Cost: 0.3253480 | Acc train: 1.0000000 | Acc validation: 1.0000000
 </code>
 </pre>
 </details>

---

## 3. tutorial_vqe.html <a name="demo2"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_vqe.html):

```
The Hamiltonian is    (-0.2427450172749822) [Z2]
+ (-0.2427450172749822) [Z3]
+ (-0.04207254303152995) [I0]
+ (0.17771358191549907) [Z0]
+ (0.17771358191549919) [Z1]
+ (0.12293330460167415) [Z0 Z2]
+ (0.12293330460167415) [Z1 Z3]
+ (0.16768338881432715) [Z0 Z3]
+ (0.16768338881432715) [Z1 Z2]
+ (0.17059759240560826) [Z0 Z1]
+ (0.17627661476093917) [Z2 Z3]
+ (-0.04475008421265302) [Y0 Y1 X2 X3]
+ (-0.04475008421265302) [X0 X1 Y2 Y3]
+ (0.04475008421265302) [Y0 X1 X2 Y3]
+ (0.04475008421265302) [X0 Y1 Y2 X3]
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_vqe.html):

```
The Hamiltonian is    (-0.24274501250450498) [Z2]
+ (-0.24274501250450498) [Z3]
+ (-0.04207255204041782) [I0]
+ (0.17771358235506415) [Z0]
+ (0.17771358235506418) [Z1]
+ (0.12293330446066005) [Z0 Z2]
+ (0.12293330446066005) [Z1 Z3]
+ (0.16768338851190992) [Z0 Z3]
+ (0.16768338851190992) [Z1 Z2]
+ (0.17059759275428316) [Z0 Z1]
+ (0.17627661386364413) [Z2 Z3]
+ (-0.04475008405124988) [Y0 Y1 X2 X3]
+ (-0.04475008405124988) [X0 X1 Y2 Y3]
+ (0.04475008405124988) [Y0 X1 X2 Y3]
+ (0.04475008405124988) [X0 Y1 Y2 X3]
```

---

## 4. tutorial_expressivity_fourier_series.html <a name="demo3"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_expressivity_fourier_series.html):

```
Cost at step  10: 0.017166449445319226
Cost at step  20: 0.005497199314426231
Cost at step  30: 0.004784402394898537
Cost at step  40: 0.00401548143455807
Cost at step  50: 0.0013998102989809839
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_expressivity_fourier_series.html):

```
Cost at step  10: 0.01716644944531955
Cost at step  20: 0.005497199314425853
Cost at step  30: 0.004784402394898465
Cost at step  40: 0.004015481434558411
Cost at step  50: 0.0013998102989816457
```

---

## 5. tutorial_qgrnn.html <a name="demo4"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_qgrnn.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Cost at Step 290: -0.9999918144420527
0.56                |  0.5988034096092949
1.24                |   1.348386551200518
1.67                |  1.7862070648455826
-0.79               | -0.8425475506159126
Cost at Step 15: -0.9981871533122741
-1.44               | -1.4067983643944049
-1.43               |  -1.352963862717384
Cost at Step 20: -0.9995130692146866
Cost at Step 170: -0.999988674732169
1.18                |  1.0349129419830865
-0.93               | -1.0635874966599725
Cost at Step 25: -0.9998181560069559
Non-Existing Edge Parameters: [-0.0012651471928293249, -0.0036534472423338637]
Cost at Step 180: -0.9999914836764836
Cost at Step 30: -0.9997713453918132
Cost at Step 45: -0.999879644915471
Cost at Step 195: -0.9999899532834507
Cost at Step 200: -0.9999880990041627
Cost at Step 215: -0.999988616757543
Cost at Step 220: -0.9999875727826187
Cost at Step 75: -0.9999839418035988
Weights at Step 220: [ 5.94793970e-01 -8.57403941e-04  1.33609793e+00  1.78453102e+00
Cost at Step 80: -0.9999900754964173
Cost at Step 225: -0.9999895237582312
Cost at Step 85: -0.9999892005586409
Cost at Step 230: -0.9999922217151828
Cost at Step 235: -0.9999912692867913
Cost at Step 90: -0.9999860418671424
Cost at Step 240: -0.999989935769989
Cost at Step 105: -0.9999860417314501
Cost at Step 115: -0.9999907527375035
Cost at Step 270: -0.9999877865048056
Cost at Step 130: -0.9999895629847482
Cost at Step 135: -0.9999893671898541
Cost at Step 280: -0.9999912773773975
Cost at Step 140: -0.9999909995915325
Cost at Step 285: -0.9999871922879655
Cost at Step 145: -0.9999882781659928
 </code>
 </pre>
 </details>

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_qgrnn.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Cost at Step 290: -0.9999918144420524
0.56                |  0.5988034096092887
1.24                |  1.3483865512005284
1.67                |  1.7862070648455801
-0.79               |  -0.842547550615921
Cost at Step 15: -0.9981871533122743
-1.44               | -1.4067983643944137
-1.43               | -1.3529638627173912
Cost at Step 20: -0.9995130692146865
Cost at Step 170: -0.9999886747321688
1.18                |  1.0349129419830911
-0.93               | -1.0635874966599628
Cost at Step 25: -0.9998181560069561
Non-Existing Edge Parameters: [-0.0012651471928236588, -0.0036534472423248887]
Cost at Step 180: -0.9999914836764835
Cost at Step 30: -0.9997713453918133
Cost at Step 45: -0.9998796449154709
Cost at Step 195: -0.9999899532834509
Cost at Step 200: -0.9999880990041629
Cost at Step 215: -0.9999886167575431
Cost at Step 220: -0.9999875727826188
Cost at Step 75: -0.9999839418035987
Weights at Step 220: [ 5.94793970e-01 -8.57403942e-04  1.33609793e+00  1.78453102e+00
Cost at Step 80: -0.9999900754964174
Cost at Step 225: -0.9999895237582316
Cost at Step 85: -0.9999892005586408
Cost at Step 230: -0.9999922217151831
Cost at Step 235: -0.9999912692867914
Cost at Step 90: -0.9999860418671422
Cost at Step 240: -0.9999899357699892
Cost at Step 105: -0.9999860417314502
Cost at Step 115: -0.9999907527375037
Cost at Step 270: -0.9999877865048059
Cost at Step 130: -0.9999895629847484
Cost at Step 135: -0.9999893671898538
Cost at Step 280: -0.9999912773773976
Cost at Step 140: -0.9999909995915327
Cost at Step 285: -0.9999871922879656
Cost at Step 145: -0.9999882781659927
 </code>
 </pre>
 </details>

---

## 6. tutorial_adaptive_circuits.html <a name="demo5"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_adaptive_circuits.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Excitation : [0, 1, 2, 3], Gradient: -0.012782175157661512
Excitation : [0, 1, 2, 5], Gradient: 1.490777987167569e-19
Excitation : [0, 1, 2, 7], Gradient: 1.6940658945086002e-19
Excitation : [0, 1, 2, 9], Gradient: 0.03426451170167927
Excitation : [0, 1, 3, 4], Gradient: 9.486769009248164e-20
Excitation : [0, 1, 3, 6], Gradient: -8.809142651444727e-20
Excitation : [0, 1, 3, 8], Gradient: -0.034264511701679164
Excitation : [0, 1, 4, 5], Gradient: -0.02358152902067629
Excitation : [0, 1, 5, 8], Gradient: 4.0657581468206826e-20
Excitation : [0, 1, 6, 7], Gradient: -0.023581529020676284
Excitation : [0, 1, 7, 8], Gradient: 7.453889935837894e-20
Excitation : [0, 1, 8, 9], Gradient: -0.12362273485599037
Excitation : [0, 2], Gradient: -0.0050625362393359585
Excitation : [0, 4], Gradient: 4.6519925288489786e-18
Excitation : [0, 6], Gradient: -2.7172099608030946e-18
Excitation : [0, 8], Gradient: -0.0009448044625780737
Excitation : [1, 3], Gradient: 0.004926616877004509
Excitation : [1, 5], Gradient: -1.7672231140684465e-18
Excitation : [1, 7], Gradient: 7.767609666719258e-20
Excitation : [1, 9], Gradient: 0.001453553485404749
n = 0,  E = -7.86266587 H, t = 2.76 s
n = 1,  E = -7.87094621 H, t = 2.75 s
n = 2,  E = -7.87563100 H, t = 2.75 s
n = 3,  E = -7.87829146 H, t = 2.76 s
n = 4,  E = -7.87981705 H, t = 2.76 s
n = 5,  E = -7.88070477 H, t = 2.76 s
n = 6,  E = -7.88123143 H, t = 2.75 s
n = 7,  E = -7.88155161 H, t = 2.74 s
n = 8,  E = -7.88175217 H, t = 2.75 s
n = 9,  E = -7.88188237 H, t = 2.75 s
n = 10,  E = -7.88197041 H, t = 3.08 s
n = 11,  E = -7.88203267 H, t = 2.80 s
n = 12,  E = -7.88207879 H, t = 2.78 s
n = 13,  E = -7.88211452 H, t = 2.80 s
n = 14,  E = -7.88214335 H, t = 2.78 s
n = 15,  E = -7.88216743 H, t = 2.80 s
n = 16,  E = -7.88218814 H, t = 2.78 s
n = 17,  E = -7.88220634 H, t = 2.78 s
n = 18,  E = -7.88222261 H, t = 2.78 s
n = 19,  E = -7.88223734 H, t = 2.78 s
    with 11264 stored elements in COOrdinate format>
n = 0,  E = -7.86266587 H, t = 0.13 s
n = 1,  E = -7.87094621 H, t = 0.14 s
n = 2,  E = -7.87563100 H, t = 0.13 s
n = 3,  E = -7.87829146 H, t = 0.13 s
n = 4,  E = -7.87981705 H, t = 0.13 s
n = 5,  E = -7.88070477 H, t = 0.13 s
n = 6,  E = -7.88123143 H, t = 0.13 s
n = 7,  E = -7.88155161 H, t = 0.13 s
n = 8,  E = -7.88175217 H, t = 0.13 s
n = 9,  E = -7.88188237 H, t = 0.13 s
n = 10,  E = -7.88197041 H, t = 0.13 s
n = 11,  E = -7.88203267 H, t = 0.13 s
n = 12,  E = -7.88207879 H, t = 0.13 s
n = 13,  E = -7.88211452 H, t = 0.13 s
n = 14,  E = -7.88214335 H, t = 0.13 s
n = 15,  E = -7.88216743 H, t = 0.13 s
n = 16,  E = -7.88218814 H, t = 0.13 s
n = 17,  E = -7.88220634 H, t = 0.13 s
n = 18,  E = -7.88222261 H, t = 0.13 s
n = 19,  E = -7.88223734 H, t = 0.13 s
 </code>
 </pre>
 </details>

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_adaptive_circuits.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Excitation : [0, 1, 2, 3], Gradient: -0.012782168177307606
Excitation : [0, 1, 2, 5], Gradient: -1.4907779871675686e-19
Excitation : [0, 1, 2, 7], Gradient: 3.3203691532368573e-19
Excitation : [0, 1, 2, 9], Gradient: -0.034264503590498444
Excitation : [0, 1, 3, 4], Gradient: 5.42101086242753e-20
Excitation : [0, 1, 3, 6], Gradient: 3.388131789017204e-20
Excitation : [0, 1, 3, 8], Gradient: 0.03426450359049812
Excitation : [0, 1, 4, 5], Gradient: -0.023581524376608452
Excitation : [0, 1, 5, 8], Gradient: 3.659182332138582e-19
Excitation : [0, 1, 6, 7], Gradient: -0.023581524376612435
Excitation : [0, 1, 7, 8], Gradient: -2.1006417091906605e-19
Excitation : [0, 1, 8, 9], Gradient: -0.12362273286723915
Excitation : [0, 2], Gradient: 0.00506254462734164
Excitation : [0, 4], Gradient: -1.4017270458719007e-17
Excitation : [0, 6], Gradient: 7.757114430397982e-19
Excitation : [0, 8], Gradient: -0.0009448055850321408
Excitation : [1, 3], Gradient: -0.004926625111533418
Excitation : [1, 5], Gradient: 5.458181086482301e-18
Excitation : [1, 7], Gradient: -1.251831553924807e-19
Excitation : [1, 9], Gradient: 0.0014535553836053277
n = 0,  E = -7.86266588 H, t = 3.15 s
n = 1,  E = -7.87094622 H, t = 2.79 s
n = 2,  E = -7.87563101 H, t = 3.02 s
n = 3,  E = -7.87829147 H, t = 3.02 s
n = 4,  E = -7.87981706 H, t = 3.34 s
n = 5,  E = -7.88070478 H, t = 2.77 s
n = 6,  E = -7.88123144 H, t = 3.02 s
n = 7,  E = -7.88155162 H, t = 3.01 s
n = 8,  E = -7.88175219 H, t = 3.04 s
n = 9,  E = -7.88188238 H, t = 3.03 s
n = 10,  E = -7.88197042 H, t = 3.36 s
n = 11,  E = -7.88203269 H, t = 2.77 s
n = 12,  E = -7.88207881 H, t = 3.03 s
n = 13,  E = -7.88211453 H, t = 3.03 s
n = 14,  E = -7.88214336 H, t = 3.03 s
n = 15,  E = -7.88216745 H, t = 3.01 s
n = 16,  E = -7.88218815 H, t = 3.03 s
n = 17,  E = -7.88220635 H, t = 3.37 s
n = 18,  E = -7.88222262 H, t = 2.81 s
n = 19,  E = -7.88223735 H, t = 3.11 s
    with 11776 stored elements in COOrdinate format>
n = 0,  E = -7.86266588 H, t = 0.15 s
n = 1,  E = -7.87094622 H, t = 0.16 s
n = 2,  E = -7.87563101 H, t = 0.16 s
n = 3,  E = -7.87829147 H, t = 0.16 s
n = 4,  E = -7.87981706 H, t = 0.15 s
n = 5,  E = -7.88070478 H, t = 0.15 s
n = 6,  E = -7.88123144 H, t = 0.15 s
n = 7,  E = -7.88155162 H, t = 0.15 s
n = 8,  E = -7.88175219 H, t = 0.16 s
n = 9,  E = -7.88188238 H, t = 0.15 s
n = 10,  E = -7.88197042 H, t = 0.15 s
n = 11,  E = -7.88203269 H, t = 0.16 s
n = 12,  E = -7.88207881 H, t = 0.16 s
n = 13,  E = -7.88211453 H, t = 0.15 s
n = 14,  E = -7.88214336 H, t = 0.15 s
n = 15,  E = -7.88216745 H, t = 0.15 s
n = 16,  E = -7.88218815 H, t = 0.15 s
n = 17,  E = -7.88220635 H, t = 0.15 s
n = 18,  E = -7.88222262 H, t = 0.15 s
n = 19,  E = -7.88223735 H, t = 0.15 s
 </code>
 </pre>
 </details>

---

## 7. tutorial_backprop.html <a name="demo6"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_backprop.html):

```
[[-6.51887722e-02 -2.72891905e-02 -2.83424776e-17 -9.33934621e-02
  -7.61067572e-01  4.10464615e-17]]
Forward pass (best of 3): 0.01910195250002289 sec per loop
Gradient computation (best of 3): 5.370657376899999 sec per loop
6.87670290000824
Forward pass (best of 3): 0.05093871469998703 sec per loop
Backward pass (best of 3): 0.1557415779000621 sec per loop
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_backprop.html):

```
[-6.51887722e-02 -2.72891905e-02 -2.83424776e-17 -9.33934621e-02
 -7.61067572e-01  4.10464615e-17]
Forward pass (best of 3): 0.022038751499985666 sec per loop
Gradient computation (best of 3): 6.103847609399963 sec per loop
7.93395053999484
Forward pass (best of 3): 0.058189682800002626 sec per loop
Backward pass (best of 3): 0.1780649433999315 sec per loop
```

---

## 8. tutorial_rosalin.html <a name="demo7"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_rosalin.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Step 8: cost = -7.398432638481054 shots_used = 21600
Step 10: cost = -7.374281342889539 shots_used = 26400
Step 11: cost = -7.287845575489478 shots_used = 28800
Step 16: cost = -7.278024044019372 shots_used = 40800
Step 17: cost = -7.36144993117844 shots_used = 43200
Step 18: cost = -7.442410269187304 shots_used = 45600
Step 19: cost = -7.511315452968971 shots_used = 48000
Step 22: cost = -7.6083225347795524 shots_used = 55200
Step 24: cost = -7.594076978496342 shots_used = 60000
Step 26: cost = -7.572266109391966 shots_used = 64800
Step 28: cost = -7.5819359681781515 shots_used = 69600
Step 29: cost = -7.610907153836915 shots_used = 72000
Step 31: cost = -7.697526604943237 shots_used = 76800
Step 32: cost = -7.746903397102402 shots_used = 79200
Step 34: cost = -7.820874827421244 shots_used = 84000
Step 35: cost = -7.840729913365394 shots_used = 86400
Step 36: cost = -7.858055514435198 shots_used = 88800
Step 37: cost = -7.868752507617257 shots_used = 91200
Step 38: cost = -7.87777500140351 shots_used = 93600
Step 39: cost = -7.884473822847109 shots_used = 96000
Step 41: cost = -7.885930519767962 shots_used = 100800
Step 42: cost = -7.8810028907659415 shots_used = 103200
Step 45: cost = -7.8505811532515715 shots_used = 110400
Step 48: cost = -7.8534445769956935 shots_used = 117600
Step 49: cost = -7.8580183681147915 shots_used = 120000
Step 52: cost = -7.850626102015814 shots_used = 127200
Step 53: cost = -7.8489696312739445 shots_used = 129600
Step 54: cost = -7.852176020039672 shots_used = 132000
Step 55: cost = -7.86142787416302 shots_used = 134400
Step 57: cost = -7.87624175909495 shots_used = 139200
Step 58: cost = -7.880465487489517 shots_used = 141600
Step 60: cost = -7.877251725772391 shots_used = 146400
Step 62: cost = -7.864543163588924 shots_used = 151200
Step 63: cost = -7.862715331323388 shots_used = 153600
Step 64: cost = -7.861607002909468 shots_used = 156000
Step 67: cost = -7.867196452121142 shots_used = 163200
Step 68: cost = -7.869567827264065 shots_used = 165600
Step 70: cost = -7.866578292195072 shots_used = 170400
Step 74: cost = -7.861466111241276 shots_used = 180000
Step 75: cost = -7.864825877239139 shots_used = 182400
Step 77: cost = -7.86349761416901 shots_used = 187200
Step 78: cost = -7.8603268453556385 shots_used = 189600
Step 80: cost = -7.855890069918927 shots_used = 194400
Step 81: cost = -7.857406216142362 shots_used = 196800
Step 83: cost = -7.870058142679085 shots_used = 201600
Step 85: cost = -7.883842326350928 shots_used = 206400
Step 86: cost = -7.882633952688098 shots_used = 208800
Step 88: cost = -7.872184015334465 shots_used = 213600
Step 90: cost = -7.860606976666903 shots_used = 218400
Step 93: cost = -7.868586359942557 shots_used = 225600
Step 94: cost = -7.8747571561059235 shots_used = 228000
Step 95: cost = -7.879893808186248 shots_used = 230400
Step 97: cost = -7.883690480348111 shots_used = 235200
Step 98: cost = -7.882061100381098 shots_used = 237600
Step 99: cost = -7.878907675503154 shots_used = 240000
 </code>
 </pre>
 </details>

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_rosalin.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Step 8: cost = -7.398432638481055 shots_used = 21600
Step 10: cost = -7.374281342889536 shots_used = 26400
Step 11: cost = -7.28784557548948 shots_used = 28800
Step 16: cost = -7.278024044019375 shots_used = 40800
Step 17: cost = -7.361449931178441 shots_used = 43200
Step 18: cost = -7.442410269187305 shots_used = 45600
Step 19: cost = -7.5113154529689705 shots_used = 48000
Step 22: cost = -7.608322534779556 shots_used = 55200
Step 24: cost = -7.594076978496339 shots_used = 60000
Step 26: cost = -7.5722661093919665 shots_used = 64800
Step 28: cost = -7.581935968178152 shots_used = 69600
Step 29: cost = -7.610907153836916 shots_used = 72000
Step 31: cost = -7.697526604943236 shots_used = 76800
Step 32: cost = -7.746903397102401 shots_used = 79200
Step 34: cost = -7.8208748274212425 shots_used = 84000
Step 35: cost = -7.840729913365392 shots_used = 86400
Step 36: cost = -7.858055514435195 shots_used = 88800
Step 37: cost = -7.868752507617255 shots_used = 91200
Step 38: cost = -7.877775001403509 shots_used = 93600
Step 39: cost = -7.884473822847107 shots_used = 96000
Step 41: cost = -7.885930519767964 shots_used = 100800
Step 42: cost = -7.881002890765939 shots_used = 103200
Step 45: cost = -7.850581153251572 shots_used = 110400
Step 48: cost = -7.853444576995692 shots_used = 117600
Step 49: cost = -7.858018368114793 shots_used = 120000
Step 52: cost = -7.850626102015812 shots_used = 127200
Step 53: cost = -7.848969631273944 shots_used = 129600
Step 54: cost = -7.8521760200396695 shots_used = 132000
Step 55: cost = -7.861427874163019 shots_used = 134400
Step 57: cost = -7.876241759094949 shots_used = 139200
Step 58: cost = -7.88046548748952 shots_used = 141600
Step 60: cost = -7.877251725772392 shots_used = 146400
Step 62: cost = -7.864543163588923 shots_used = 151200
Step 63: cost = -7.862715331323386 shots_used = 153600
Step 64: cost = -7.8616070029094685 shots_used = 156000
Step 67: cost = -7.867196452121144 shots_used = 163200
Step 68: cost = -7.869567827264067 shots_used = 165600
Step 70: cost = -7.866578292195071 shots_used = 170400
Step 74: cost = -7.8614661112412785 shots_used = 180000
Step 75: cost = -7.864825877239138 shots_used = 182400
Step 77: cost = -7.863497614169012 shots_used = 187200
Step 78: cost = -7.86032684535564 shots_used = 189600
Step 80: cost = -7.855890069918925 shots_used = 194400
Step 81: cost = -7.85740621614236 shots_used = 196800
Step 83: cost = -7.870058142679084 shots_used = 201600
Step 85: cost = -7.883842326350927 shots_used = 206400
Step 86: cost = -7.882633952688099 shots_used = 208800
Step 88: cost = -7.872184015334466 shots_used = 213600
Step 90: cost = -7.860606976666908 shots_used = 218400
Step 93: cost = -7.868586359942558 shots_used = 225600
Step 94: cost = -7.874757156105917 shots_used = 228000
Step 95: cost = -7.8798938081862495 shots_used = 230400
Step 97: cost = -7.883690480348113 shots_used = 235200
Step 98: cost = -7.8820611003811 shots_used = 237600
Step 99: cost = -7.878907675503153 shots_used = 240000
 </code>
 </pre>
 </details>

---

## 9. tutorial_error_mitigation.html <a name="demo8"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_error_mitigation.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
 0: ──RY(4.56)──╭C──RY(5.93)───RY(-5.93)──────────────────────────────────────╭C──RY(-4.56)──┤
 1: ──RY(3.6)───╰Z──RY(5.9)───╭C──────────RY(5.18)──RY(-5.18)──╭C──RY(-5.9)───╰Z──RY(-3.6)───┤
 2: ──RY(4.05)──╭C──RY(3.32)──╰Z──────────RY(1.07)──RY(-1.07)──╰Z──RY(-3.32)──╭C──RY(-4.05)──┤
 3: ──RY(3.51)──╰Z──RY(3.66)───RY(-3.66)──────────────────────────────────────╰Z──RY(-3.51)──┤
Globally-folded circuit with a scale factor of 2:
 0: ──RY(4.56)──╭C──RY(5.93)───RY(-5.93)──────────────────────────────────────╭C──RY(-4.56)──RY(4.56)──╭C─────────────────────────────────────────────────────────────────────────╭C──RY(-4.56)──┤
 1: ──RY(3.6)───╰Z──RY(5.9)───╭C──────────RY(5.18)──RY(-5.18)──╭C──RY(-5.9)───╰Z──RY(-3.6)───RY(3.6)───╰Z──RY(5.9)───╭C──RY(5.18)──RY(-5.18)──RY(5.18)──RY(-5.18)──╭C──RY(-5.9)───╰Z──RY(-3.6)───┤
 2: ──RY(4.05)──╭C──RY(3.32)──╰Z──────────RY(1.07)──RY(-1.07)──╰Z──RY(-3.32)──╭C──RY(-4.05)──RY(4.05)──╭C──RY(3.32)──╰Z──RY(1.07)──RY(-1.07)──RY(1.07)──RY(-1.07)──╰Z──RY(-3.32)──╭C──RY(-4.05)──┤
 3: ──RY(3.51)──╰Z──RY(3.66)───RY(-3.66)──────────────────────────────────────╰Z──RY(-3.51)──RY(3.51)──╰Z─────────────────────────────────────────────────────────────────────────╰Z──RY(-3.51)──┤
Globally-folded circuit with a scale factor of 3:
 0: ──RY(4.56)──╭C──RY(5.93)───RY(-5.93)──────────────────────────────────────╭C──RY(-4.56)──RY(4.56)──╭C──RY(5.93)───RY(-5.93)──────────────────────────────────────╭C──RY(-4.56)──RY(4.56)──╭C──RY(5.93)───RY(-5.93)──────────────────────────────────────╭C──RY(-4.56)──┤
 1: ──RY(3.6)───╰Z──RY(5.9)───╭C──────────RY(5.18)──RY(-5.18)──╭C──RY(-5.9)───╰Z──RY(-3.6)───RY(3.6)───╰Z──RY(5.9)───╭C──────────RY(5.18)──RY(-5.18)──╭C──RY(-5.9)───╰Z──RY(-3.6)───RY(3.6)───╰Z──RY(5.9)───╭C──────────RY(5.18)──RY(-5.18)──╭C──RY(-5.9)───╰Z──RY(-3.6)───┤
 2: ──RY(4.05)──╭C──RY(3.32)──╰Z──────────RY(1.07)──RY(-1.07)──╰Z──RY(-3.32)──╭C──RY(-4.05)──RY(4.05)──╭C──RY(3.32)──╰Z──────────RY(1.07)──RY(-1.07)──╰Z──RY(-3.32)──╭C──RY(-4.05)──RY(4.05)──╭C──RY(3.32)──╰Z──────────RY(1.07)──RY(-1.07)──╰Z──RY(-3.32)──╭C──RY(-4.05)──┤
 3: ──RY(3.51)──╰Z──RY(3.66)───RY(-3.66)──────────────────────────────────────╰Z──RY(-3.51)──RY(3.51)──╰Z──RY(3.66)───RY(-3.66)──────────────────────────────────────╰Z──RY(-3.51)──RY(3.51)──╰Z──RY(3.66)───RY(-3.66)──────────────────────────────────────╰Z──RY(-3.51)──┤
[array([1.]), array([1.]), array([1.])]
[array([0.71729164]), array([0.54368629]), array([0.3777036])]
ZNE result: 0.8985196547410967
0.9599709819121317
0: ──RY(4.56)──╭C──RY(5.93)───RY(-5.93)──────────────────────────────────────────────╭C──RY(-4.56)──┤
1: ──RY(3.6)───╰Z──RY(5.9)───╭C──────────╭C──╭C──RY(5.18)──RY(-5.18)──╭C──RY(-5.9)───╰Z──RY(-3.6)───┤
2: ──RY(4.05)──╭C──RY(3.32)──╰Z──────────╰Z──╰Z──RY(1.07)──RY(-1.07)──╰Z──RY(-3.32)──╭C──RY(-4.05)──┤
3: ──RY(3.51)──╰Z──RY(3.66)───RY(-3.66)──────────────────────────────────────────────╰Z──RY(-3.51)──┤
0: ──RY(4.56)──╭C──RY(5.93)───RY(-5.93)───────────────────────────────────────────────────────────╭C──RY(-4.56)──┤
1: ──RY(3.6)───╰Z──RY(5.9)───╭C──────────RY(5.18)──RY(-5.18)──RY(5.18)──RY(-5.18)──╭C──RY(-5.9)───╰Z──RY(-3.6)───┤
2: ──RY(4.05)──╭C──RY(3.32)──╰Z──────────RY(1.07)──RY(-1.07)───────────────────────╰Z──RY(-3.32)──╭C──RY(-4.05)──┤
3: ──RY(3.51)──╰Z──RY(3.66)───RY(-3.66)───────────────────────────────────────────────────────────╰Z──RY(-3.51)──┤
0: ──RY(4.56)──╭C──RY(5.93)───RY(-5.93)──RY(5.93)──RY(-5.93)─────────────────╭C──RY(-4.56)──┤
1: ──RY(3.6)───╰Z──RY(5.9)───╭C──────────RY(5.18)──RY(-5.18)──╭C──RY(-5.9)───╰Z──RY(-3.6)───┤
2: ──RY(4.05)──╭C──RY(3.32)──╰Z──────────RY(1.07)──RY(-1.07)──╰Z──RY(-3.32)──╭C──RY(-4.05)──┤
3: ──RY(3.51)──╰Z──RY(3.66)───RY(-3.66)──────────────────────────────────────╰Z──RY(-3.51)──┤
0.9589759497509437
 </code>
 </pre>
 </details>

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_error_mitigation.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
0: ──RY(4.56)─╭C──RY(5.93)──RY(-5.93)────────────────────────────────────╭C
1: ──RY(3.60)─╰Z──RY(5.90)─╭C──────────RY(5.18)──RY(-5.18)─╭C──RY(-5.90)─╰Z
2: ──RY(4.05)─╭C──RY(3.32)─╰Z──────────RY(1.07)──RY(-1.07)─╰Z──RY(-3.32)─╭C
3: ──RY(3.51)─╰Z──RY(3.66)──RY(-3.66)────────────────────────────────────╰Z
───RY(-4.56)─┤
───RY(-3.60)─┤
───RY(-4.05)─┤
───RY(-3.51)─┤
Globally-folded circuit with a scale factor of 2:
0: ──RY(4.56)─╭C──RY(5.93)──RY(-5.93)────────────────────────────────────╭C
1: ──RY(3.60)─╰Z──RY(5.90)─╭C──────────RY(5.18)──RY(-5.18)─╭C──RY(-5.90)─╰Z
2: ──RY(4.05)─╭C──RY(3.32)─╰Z──────────RY(1.07)──RY(-1.07)─╰Z──RY(-3.32)─╭C
3: ──RY(3.51)─╰Z──RY(3.66)──RY(-3.66)────────────────────────────────────╰Z
───RY(-4.56)──RY(4.56)─╭C───────────────────────────────────────────────────────
───RY(-3.60)──RY(3.60)─╰Z──RY(5.90)─╭C──RY(5.18)──RY(-5.18)──RY(5.18)──RY(-5.18)
───RY(-4.05)──RY(4.05)─╭C──RY(3.32)─╰Z──RY(1.07)──RY(-1.07)──RY(1.07)──RY(-1.07)
───RY(-3.51)──RY(3.51)─╰Z───────────────────────────────────────────────────────
────────────────╭C──RY(-4.56)─┤
──╭C──RY(-5.90)─╰Z──RY(-3.60)─┤
──╰Z──RY(-3.32)─╭C──RY(-4.05)─┤
────────────────╰Z──RY(-3.51)─┤
Globally-folded circuit with a scale factor of 3:
0: ──RY(4.56)─╭C──RY(5.93)──RY(-5.93)────────────────────────────────────╭C
1: ──RY(3.60)─╰Z──RY(5.90)─╭C──────────RY(5.18)──RY(-5.18)─╭C──RY(-5.90)─╰Z
2: ──RY(4.05)─╭C──RY(3.32)─╰Z──────────RY(1.07)──RY(-1.07)─╰Z──RY(-3.32)─╭C
3: ──RY(3.51)─╰Z──RY(3.66)──RY(-3.66)────────────────────────────────────╰Z
───RY(-4.56)──RY(4.56)─╭C──RY(5.93)──RY(-5.93)────────────────────────
───RY(-3.60)──RY(3.60)─╰Z──RY(5.90)─╭C──────────RY(5.18)──RY(-5.18)─╭C
───RY(-4.05)──RY(4.05)─╭C──RY(3.32)─╰Z──────────RY(1.07)──RY(-1.07)─╰Z
───RY(-3.51)──RY(3.51)─╰Z──RY(3.66)──RY(-3.66)────────────────────────
─────────────╭C──RY(-4.56)──RY(4.56)─╭C──RY(5.93)──RY(-5.93)──────────
 </code>
 </pre>
 </details>

---

## 10. tutorial_measurement_optimize.html <a name="demo9"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_measurement_optimize.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
  (-0.24274280513140423) [Z2]
+ (-0.24274280513140423) [Z3]
+ (-0.04207897647782287) [I0]
+ (0.1777128746513993) [Z1]
+ (0.17771287465139948) [Z0]
+ (0.12293305056183795) [Z0 Z2]
+ (0.12293305056183795) [Z1 Z3]
+ (0.16768319457718955) [Z0 Z3]
+ (0.16768319457718955) [Z1 Z2]
+ (0.17059738328801047) [Z0 Z1]
+ (0.1762764080431957) [Z2 Z3]
+ (-0.04475014401535161) [Y0 Y1 X2 X3]
+ (-0.04475014401535161) [X0 X1 Y2 Y3]
+ (0.04475014401535161) [Y0 X1 X2 Y3]
+ (0.04475014401535161) [X0 Y1 Y2 X3]
Cost function value: -0.544996018090293
Number of Hamiltonian terms/required measurements: 2050
   (-46.46390678868896) [I0]
+ (0.7829661725950188) [Z10]
+ (0.7829661725950189) [Z11]
+ (0.8084581961720487) [Z12]
+ (0.808458196172049) [Z13]
+ (1.203440228914564) [Z4]
+ (1.2034402289145643) [Z5]
+ (1.3096862988615419) [Z7]
+ (1.3096862988615423) [Z6]
+ (1.369352563471818) [Z8]
+ (1.3693525634718182) [Z9]
+ (1.6538942226831712) [Z3]
+ (1.6538942226831714) [Z2]
+ (12.412630742111762) [Z0]
+ (12.412630742111762) [Z1]
+ (-8.194261371769373e-06) [Y10 Y12]
+ (-8.194261371769373e-06) [X10 X12]
+ (-1.854060857958372e-06) [Y5 Y7]
+ (-1.854060857958372e-06) [X5 X7]
+ (-7.764994118105853e-07) [Y3 Y5]
+ (-7.764994118105853e-07) [X3 X5]
+ (-5.929765815942634e-07) [Y4 Y6]
+ (-5.929765815942634e-07) [X4 X6]
+ (1.6021167406205364e-06) [Y2 Y4]
+ (1.6021167406205364e-06) [X2 X4]
+ (7.954413175773528e-06) [Y11 Y13]
+ (7.954413175773528e-06) [X11 X13]
+ (0.003276971931231669) [Y1 Y3]
+ (0.003276971931231669) [X1 X3]
+ (0.10433064780651398) [Y0 Y2]
+ (0.10433064780651398) [X0 X2]
+ (0.11270386920332212) [Z10 Z12]
+ (0.11270386920332212) [Z11 Z13]
+ (0.11383573679388657) [Z4 Z12]
+ (0.11383573679388657) [Z5 Z13]
+ (0.11952438964682671) [Z6 Z10]
+ (0.11952438964682671) [Z7 Z11]
+ (0.124899909172376) [Z4 Z10]
+ (0.124899909172376) [Z5 Z11]
+ (0.12495807739503229) [Z2 Z4]
+ (0.12495807739503229) [Z3 Z5]
+ (0.1279950249246842) [Z2 Z10]
+ (0.1279950249246842) [Z3 Z11]
+ (0.13401715261963695) [Z6 Z12]
+ (0.13401715261963695) [Z7 Z13]
+ (0.1370119167404075) [Z4 Z6]
+ (0.1370119167404075) [Z5 Z7]
+ (0.1373495306426133) [Z6 Z11]
+ (0.1373495306426133) [Z7 Z10]
+ (0.13739104762683235) [Z2 Z6]
+ (0.13739104762683235) [Z3 Z7]
+ (0.13766872645852576) [Z8 Z10]
+ (0.13766872645852576) [Z9 Z11]
+ (0.1401128986535481) [Z2 Z12]
+ (0.1401128986535481) [Z3 Z13]
+ (0.14138905291942802) [Z10 Z13]
+ (0.14138905291942802) [Z11 Z12]
+ (0.14257997712485757) [Z4 Z11]
+ (0.14257997712485757) [Z5 Z10]
+ (0.14722943218766166) [Z8 Z11]
+ (0.14722943218766166) [Z9 Z10]
+ (0.14899430575065548) [Z4 Z7]
+ (0.14899430575065548) [Z5 Z6]
+ (0.14926355147388898) [Z10 Z11]
+ (0.14960702684445298) [Z4 Z8]
+ (0.14960702684445298) [Z5 Z9]
+ (0.14973486803496922) [Z8 Z12]
+ (0.14973486803496922) [Z9 Z13]
+ (0.150714081210083) [Z2 Z8]
+ (0.150714081210083) [Z3 Z9]
+ (0.15138327161428838) [Z6 Z13]
+ (0.15138327161428838) [Z7 Z12]
+ (0.15215040708869043) [Z4 Z13]
+ (0.15215040708869043) [Z5 Z12]
+ (0.15337968243314154) [Z2 Z11]
+ (0.15337968243314154) [Z3 Z10]
+ (0.15435748657223625) [Z12 Z13]
+ (0.15569010671752456) [Z2 Z13]
+ (0.15569010671752456) [Z3 Z12]
+ (0.15582269051553105) [Z8 Z13]
+ (0.15582269051553105) [Z9 Z12]
+ (0.1567639617643099) [Z4 Z9]
+ (0.1567639617643099) [Z5 Z8]
+ (0.1575531479798567) [Z4 Z5]
+ (0.1607976453483857) [Z2 Z5]
+ (0.1607976453483857) [Z3 Z4]
+ (0.16756653265461266) [Z6 Z8]
+ (0.16756653265461266) [Z7 Z9]
+ (0.16853486561579956) [Z2 Z7]
+ (0.16853486561579956) [Z3 Z6]
+ (0.1814399144030387) [Z6 Z9]
+ (0.1814399144030387) [Z7 Z8]
+ (0.1818908579075139) [Z2 Z3]
+ (0.18690820476912573) [Z2 Z9]
+ (0.18690820476912573) [Z3 Z8]
+ (0.19299723935364202) [Z0 Z10]
+ (0.19299723935364202) [Z1 Z11]
+ (0.19392534613270182) [Z6 Z7]
+ (0.19661770890342112) [Z0 Z4]
+ (0.19661770890342112) [Z1 Z5]
+ (0.19936354537360793) [Z0 Z5]
+ (0.19936354537360793) [Z1 Z4]
+ (0.20072866460441724) [Z0 Z11]
+ (0.20072866460441724) [Z1 Z10]
+ (0.2110265984979148) [Z0 Z12]
+ (0.2110265984979148) [Z1 Z13]
+ (0.21631037498631775) [Z0 Z13]
+ (0.21631037498631775) [Z1 Z12]
+ (0.2200397733437609) [Z8 Z9]
+ (0.23671080783830423) [Z0 Z2]
+ (0.23671080783830423) [Z1 Z3]
+ (0.24164663936017156) [Z0 Z6]
+ (0.24164663936017156) [Z1 Z7]
+ (0.2485348337131421) [Z0 Z7]
+ (0.2485348337131421) [Z1 Z6]
+ (0.25129445674591694) [Z0 Z3]
+ (0.25129445674591694) [Z1 Z2]
+ (0.2723251830660565) [Z0 Z8]
+ (0.2723251830660565) [Z1 Z9]
+ (0.27883454426723375) [Z0 Z9]
+ (0.27883454426723375) [Z1 Z8]
+ (1.186176373486046) [Z0 Z1]
+ (-1.226048498919638e-05) [Y4 Z5 Y6]
+ (-1.226048498919638e-05) [X4 Z5 X6]
+ (-1.2260484989196377e-05) [Y5 Z6 Y7]
+ (-1.2260484989196377e-05) [X5 Z6 X7]
+ (-1.0722312158394612e-05) [Y11 Z12 Y13]
+ (-1.0722312158394612e-05) [X11 Z12 X13]
+ (-1.0722312158394606e-05) [Y10 Z11 Y12]
+ (-1.0722312158394606e-05) [X10 Z11 X12]
+ (-3.887051672079001e-06) [Y2 Z3 Y4]
+ (-3.887051672079001e-06) [X2 Z3 X4]
+ (-3.887051672079001e-06) [Y3 Z4 Y5]
+ (-3.887051672079001e-06) [X3 Z4 X5]
+ (0.12507032579772007) [Y1 Z2 Y3]
+ (0.12507032579772007) [X1 Z2 X3]
+ (0.12507032579772015) [Y0 Z1 Y2]
+ (0.12507032579772015) [X0 Z1 X2]
+ (-0.038314670294803864) [Y4 Y5 X12 X13]
+ (-0.038314670294803864) [X4 X5 Y12 Y13]
+ (-0.03619412355904272) [Y2 Y3 X8 X9]
+ (-0.03619412355904272) [X2 X3 Y8 Y9]
+ (-0.035839567953353406) [Y2 Y3 X4 X5]
+ (-0.035839567953353406) [X2 X3 Y4 Y5]
+ (-0.0311438179889672) [Y2 Y3 X6 X7]
+ (-0.0311438179889672) [X2 X3 Y6 Y7]
+ (-0.028685183716105893) [Y10 Y11 X12 X13]
+ (-0.028685183716105893) [X10 X11 Y12 Y13]
+ (-0.025996177598021086) [Y3 Z4 Z5 Y7]
+ (-0.025996177598021086) [X3 Z4 Z5 X7]
+ (-0.025384657508457358) [Y2 Y3 X10 X11]
+ (-0.025384657508457358) [X2 X3 Y10 Y11]
+ (-0.019028242443847196) [Y3 Y4 X11 X12]
+ (-0.019028242443847196) [X3 X4 Y11 Y12]
+ (-0.017825140995786578) [Y6 Y7 X10 X11]
+ (-0.017825140995786578) [X6 X7 Y10 Y11]
+ (-0.017680067952481556) [Y4 Y5 X10 X11]
+ (-0.017680067952481556) [X4 X5 Y10 Y11]
+ (-0.0173661189946514) [Y6 Y7 X12 X13]
+ (-0.0173661189946514) [X6 X7 Y12 Y13]
+ (-0.015577208063976469) [Y2 Y3 X12 X13]
+ (-0.015577208063976469) [X2 X3 Y12 Y13]
+ (-0.0145836489076127) [Y0 Y1 X2 X3]
+ (-0.0145836489076127) [X0 X1 Y2 Y3]
+ (-0.013873381748426063) [Y6 Y7 X8 X9]
+ (-0.013873381748426063) [X6 X7 Y8 Y9]
+ (-0.01198238901024799) [Y4 Y5 X6 X7]
+ (-0.01198238901024799) [X4 X5 Y6 Y7]
+ (-0.01128519020084097) [Y5 X6 X11 Y12]
+ (-0.01128519020084097) [X5 Y6 Y11 X12]
+ (-0.009560705729135905) [Y8 Y9 X10 X11]
+ (-0.009560705729135905) [X8 X9 Y10 Y11]
+ (-0.008125251921381034) [Y1 X2 X8 Y9]
+ (-0.008125251921381034) [Y1 Y2 Y8 Y9]
+ (-0.008125251921381034) [X1 X2 X8 X9]
+ (-0.008125251921381034) [X1 Y2 Y8 X9]
+ (-0.007731425250775242) [Y0 Y1 X10 X11]
+ (-0.007731425250775242) [X0 X1 Y10 Y11]
+ (-0.007156934919856943) [Y4 Y5 X8 X9]
+ (-0.007156934919856943) [X4 X5 Y8 Y9]
+ (-0.00688819435297054) [Y0 Y1 X6 X7]
+ (-0.00688819435297054) [X0 X1 Y6 Y7]
+ (-0.006509361201177226) [Y0 Y1 X8 X9]
+ (-0.006509361201177226) [X0 X1 Y8 Y9]
+ (-0.006087822480561857) [Y8 Y9 X12 X13]
+ (-0.006087822480561857) [X8 X9 Y12 Y13]
+ (-0.0052837764884029505) [Y0 Y1 X12 X13]
+ (-0.0052837764884029505) [X0 X1 Y12 Y13]
+ (-0.00514339176882516) [Y3 X4 X5 Y6]
+ (-0.00514339176882516) [X3 Y4 Y5 X6]
+ (-0.004684903388155212) [Y1 X2 X6 Y7]
+ (-0.004684903388155212) [Y1 Y2 Y6 Y7]
+ (-0.004684903388155212) [X1 X2 X6 X7]
+ (-0.004684903388155212) [X1 Y2 Y6 X7]
+ (-0.004575007626639212) [Y1 X2 X12 Y13]
+ (-0.004575007626639212) [Y1 Y2 Y12 Y13]
+ (-0.004575007626639212) [X1 X2 X12 X13]
+ (-0.004575007626639212) [X1 Y2 Y12 X13]
+ (-0.004424855449441851) [Y1 X2 X4 Y5]
+ (-0.004424855449441851) [Y1 Y2 Y4 Y5]
+ (-0.004424855449441851) [X1 X2 X4 X5]
+ (-0.004424855449441851) [X1 Y2 Y4 X5]
+ (-0.0034795118903343525) [Y2 Z3 Z5 Y6]
+ (-0.0034795118903343525) [X2 Z3 Z5 X6]
+ (-0.0034795118903343525) [Y3 Z4 Z6 Y7]
+ (-0.0034795118903343525) [X3 Z4 Z6 X7]
+ (-0.0027458364701868038) [Y0 Y1 X4 X5]
+ (-0.0027458364701868038) [X0 X1 Y4 Y5]
+ (-0.0017992194936630329) [Y1 X2 X10 Y11]
+ (-0.0017992194936630329) [Y1 Y2 Y10 Y11]
+ (-0.0017992194936630329) [X1 X2 X10 X11]
+ (-0.0017992194936630329) [X1 Y2 Y10 X11]
+ (-0.0002921986261110113) [Y7 Y8 X9 X10]
+ (-0.0002921986261110113) [X7 X8 Y9 Y10]
+ (-8.194261371769373e-06) [Z10 Y11 Z12 Y13]
+ (-8.194261371769373e-06) [Z10 X11 Z12 X13]
+ (-7.801707500027575e-06) [Y2 Z3 Y4 Z11]
+ (-7.801707500027575e-06) [X2 Z3 X4 Z11]
+ (-7.801707500027575e-06) [Y3 Z4 Y5 Z10]
+ (-7.801707500027575e-06) [X3 Z4 X5 Z10]
+ (-4.643051068237288e-06) [Y3 X4 X10 Y11]
+ (-4.643051068237288e-06) [Y3 Y4 Y10 Y11]
+ (-4.643051068237288e-06) [X3 X4 X10 X11]
+ (-4.643051068237288e-06) [X3 Y4 Y10 X11]
+ (-4.588855155483756e-06) [Y4 Z5 Y6 Z13]
+ (-4.588855155483756e-06) [X4 Z5 X6 Z13]
+ (-4.588855155483756e-06) [Y5 Z6 Y7 Z12]
+ (-4.588855155483756e-06) [X5 Z6 X7 Z12]
+ (-4.556569217890057e-06) [Y5 X6 X12 Y13]
+ (-4.556569217890057e-06) [Y5 Y6 Y12 Y13]
+ (-4.556569217890057e-06) [X5 X6 X12 X13]
+ (-4.556569217890057e-06) [X5 Y6 Y12 X13]
+ (-3.6945132943250566e-06) [Y4 X5 X11 Y12]
+ (-3.6945132943250566e-06) [Y4 Y5 Y11 Y12]
+ (-3.6945132943250566e-06) [X4 X5 X11 X12]
+ (-3.6945132943250566e-06) [X4 Y5 Y11 X12]
+ (-3.3440815564680474e-06) [Z0 Y5 Z6 Y7]
+ (-3.3440815564680474e-06) [Z0 X5 Z6 X7]
+ (-3.3440815564680474e-06) [Z1 Y4 Z5 Y6]
+ (-3.3440815564680474e-06) [Z1 X4 Z5 X6]
+ (-3.158656431790286e-06) [Y2 Z3 Y4 Z10]
+ (-3.158656431790286e-06) [X2 Z3 X4 Z10]
+ (-3.158656431790286e-06) [Y3 Z4 Y5 Z11]
+ (-3.158656431790286e-06) [X3 Z4 X5 Z11]
+ (-3.09934924359987e-06) [Z0 Y4 Z5 Y6]
+ (-3.09934924359987e-06) [Z0 X4 Z5 X6]
+ (-3.09934924359987e-06) [Z1 Y5 Z6 Y7]
+ (-3.09934924359987e-06) [Z1 X5 Z6 X7]
+ (-2.8909678816085235e-06) [Z6 Y11 Z12 Y13]
+ (-2.8909678816085235e-06) [Z6 X11 Z12 X13]
+ (-2.8909678816085235e-06) [Z7 Y10 Z11 Y12]
+ (-2.8909678816085235e-06) [Z7 X10 Z11 X12]
+ (-2.1776646050276355e-06) [Z0 Y10 Z11 Y12]
+ (-2.1776646050276355e-06) [Z0 X10 Z11 X12]
+ (-2.1776646050276355e-06) [Z1 Y11 Z12 Y13]
+ (-2.1776646050276355e-06) [Z1 X11 Z12 X13]
+ (-1.88185018321198e-06) [Y4 Z5 Y6 Z9]
+ (-1.88185018321198e-06) [X4 Z5 X6 Z9]
+ (-1.88185018321198e-06) [Y5 Z6 Y7 Z8]
+ (-1.88185018321198e-06) [X5 Z6 X7 Z8]
+ (-1.8551201215023133e-06) [Z6 Y10 Z11 Y12]
+ (-1.8551201215023133e-06) [Z6 X10 Z11 X12]
+ (-1.8551201215023133e-06) [Z7 Y11 Z12 Y13]
+ (-1.8551201215023133e-06) [Z7 X11 Z12 X13]
+ (-1.854060857958372e-06) [Y4 Z5 Y6 Z7]
+ (-1.854060857958372e-06) [X4 Z5 X6 Z7]
+ (-1.8163031697152533e-06) [Z4 Y11 Z12 Y13]
+ (-1.8163031697152533e-06) [Z4 X11 Z12 X13]
+ (-1.8163031697152533e-06) [Z5 Y10 Z11 Y12]
+ (-1.8163031697152533e-06) [Z5 X10 Z11 X12]
+ (-1.6923978285634726e-06) [Y4 Z5 Y6 Z10]
+ (-1.6923978285634726e-06) [X4 Z5 X6 Z10]
+ (-1.6923978285634726e-06) [Y5 Z6 Y7 Z11]
+ (-1.6923978285634726e-06) [X5 Z6 X7 Z11]
+ (-1.6148794139208836e-06) [Z0 Y11 Z12 Y13]
+ (-1.6148794139208836e-06) [Z0 X11 Z12 X13]
+ (-1.6148794139208836e-06) [Z1 Y10 Z11 Y12]
+ (-1.6148794139208836e-06) [Z1 X10 Z11 X12]
+ (-1.5973171977956993e-06) [Z8 Y10 Z11 Y12]
+ (-1.5973171977956993e-06) [Z8 X10 Z11 X12]
+ (-1.5973171977956993e-06) [Z9 Y11 Z12 Y13]
+ (-1.5973171977956993e-06) [Z9 X11 Z12 X13]
+ (-1.4548424489918288e-06) [Y3 X4 X6 Y7]
+ (-1.4548424489918288e-06) [Y3 Y4 Y6 Y7]
+ (-1.4548424489918288e-06) [X3 X4 X6 X7]
+ (-1.4548424489918288e-06) [X3 Y4 Y6 X7]
+ (-1.3980449081438063e-06) [Y4 Z5 Y6 Z8]
+ (-1.3980449081438063e-06) [X4 Z5 X6 Z8]
+ (-1.3980449081438063e-06) [Y5 Z6 Y7 Z9]
+ (-1.3980449081438063e-06) [X5 Z6 X7 Z9]
+ (-1.1954890098547266e-06) [Y2 Z3 Y4 Z7]
+ (-1.1954890098547266e-06) [X2 Z3 X4 Z7]
+ (-1.1954890098547266e-06) [Y3 Z4 Y5 Z6]
+ (-1.1954890098547266e-06) [X3 Z4 X5 Z6]
+ (-1.190850808134371e-06) [Z0 Y3 Z4 Y5]
+ (-1.190850808134371e-06) [Z0 X3 Z4 X5]
+ (-1.190850808134371e-06) [Z1 Y2 Z3 Y4]
+ (-1.190850808134371e-06) [Z1 X2 Z3 X4]
+ (-1.1708301370385038e-06) [Z2 Y5 Z6 Y7]
+ (-1.1708301370385038e-06) [Z2 X5 Z6 X7]
+ (-1.1708301370385038e-06) [Z3 Y4 Z5 Y6]
+ (-1.1708301370385038e-06) [Z3 X4 Z5 X6]
+ (-1.0632283423316172e-06) [Z2 Y10 Z11 Y12]
+ (-1.0632283423316172e-06) [Z2 X10 Z11 X12]
+ (-1.0632283423316172e-06) [Z3 Y11 Z12 Y13]
+ (-1.0632283423316172e-06) [Z3 X11 Z12 X13]
+ (-1.03584776010621e-06) [Y6 X7 X11 Y12]
+ (-1.03584776010621e-06) [Y6 Y7 Y11 Y12]
+ (-1.03584776010621e-06) [X6 X7 X11 X12]
+ (-1.03584776010621e-06) [X6 Y7 Y11 X12]
+ (-9.509249751996858e-07) [Z2 Y4 Z5 Y6]
+ (-9.509249751996858e-07) [Z2 X4 Z5 X6]
+ (-9.509249751996858e-07) [Z3 Y5 Z6 Y7]
+ (-9.509249751996858e-07) [Z3 X5 Z6 X7]
+ (-9.344557776820712e-07) [Z8 Y11 Z12 Y13]
+ (-9.344557776820712e-07) [Z8 X11 Z12 X13]
+ (-9.344557776820712e-07) [Z9 Y10 Z11 Y12]
+ (-9.344557776820712e-07) [Z9 X10 Z11 X12]
+ (-8.337746752583534e-07) [Z0 Y2 Z3 Y4]
+ (-8.337746752583534e-07) [Z0 X2 Z3 X4]
+ (-8.337746752583534e-07) [Z1 Y3 Z4 Y5]
+ (-8.337746752583534e-07) [Z1 X3 Z4 X5]
+ (-7.956895371861962e-07) [Y3 X4 X8 Y9]
+ (-7.956895371861962e-07) [Y3 Y4 Y8 Y9]
+ (-7.956895371861962e-07) [X3 X4 X8 X9]
+ (-7.956895371861962e-07) [X3 Y4 Y8 X9]
+ (-7.764994118105853e-07) [Y2 Z3 Y4 Z5]
+ (-7.764994118105853e-07) [X2 Z3 X4 Z5]
+ (-5.929765815942634e-07) [Z4 Y5 Z6 Y7]
+ (-5.929765815942634e-07) [Z4 X5 Z6 X7]
+ (-5.770052993874209e-07) [Y2 Z3 Y4 Z9]
+ (-5.770052993874209e-07) [X2 Z3 X4 Z9]
+ (-5.770052993874209e-07) [Y3 Z4 Y5 Z8]
+ (-5.770052993874209e-07) [X3 Z4 X5 Z8]
+ (-5.471647744402701e-07) [Y1 Y2 X11 X12]
+ (-5.471647744402701e-07) [X1 X2 Y11 Y12]
+ (-4.838052750681737e-07) [Y5 X6 X8 Y9]
+ (-4.838052750681737e-07) [Y5 Y6 Y8 Y9]
+ (-4.838052750681737e-07) [X5 X6 X8 X9]
+ (-4.838052750681737e-07) [X5 Y6 Y8 X9]
+ (-3.5707613287601753e-07) [Y0 X1 X3 Y4]
+ (-3.5707613287601753e-07) [Y0 Y1 Y3 Y4]
+ (-3.5707613287601753e-07) [X0 X1 X3 X4]
+ (-3.5707613287601753e-07) [X0 Y1 Y3 X4]
+ (-2.4473231286817734e-07) [Y0 X1 X5 Y6]
+ (-2.4473231286817734e-07) [Y0 Y1 Y5 Y6]
+ (-2.4473231286817734e-07) [X0 X1 X5 X6]
+ (-2.4473231286817734e-07) [X0 Y1 Y5 X6]
+ (-2.1990516183881777e-07) [Y2 X3 X5 Y6]
+ (-2.1990516183881777e-07) [Y2 Y3 Y5 Y6]
+ (-2.1990516183881777e-07) [X2 X3 X5 X6]
+ (-2.1990516183881777e-07) [X2 Y3 Y5 X6]
+ (-1.933241277017308e-07) [Y1 X2 X3 Y4]
+ (-1.933241277017308e-07) [X1 Y2 Y3 X4]
+ (-1.291969486071391e-07) [Y1 Z2 Z3 Y5]
+ (-1.291969486071391e-07) [X1 Z2 Z3 X5]
+ (1.7379332623664975e-07) [Y0 Z1 Z3 Y4]
+ (1.7379332623664975e-07) [X0 Z1 Z3 X4]
+ (1.7379332623664975e-07) [Y1 Z2 Z4 Y5]
+ (1.7379332623664975e-07) [X1 Z2 Z4 X5]
+ (1.933241277017308e-07) [Y1 Y2 X3 X4]
+ (1.933241277017308e-07) [X1 X2 Y3 Y4]
+ (2.1868423779877532e-07) [Y2 Z3 Y4 Z8]
+ (2.1868423779877532e-07) [X2 Z3 X4 Z8]
+ (2.1868423779877532e-07) [Y3 Z4 Y5 Z9]
+ (2.1868423779877532e-07) [X3 Z4 X5 Z9]
+ (2.593534391371023e-07) [Y2 Z3 Y4 Z6]
+ (2.593534391371023e-07) [X2 Z3 X4 Z6]
+ (2.593534391371023e-07) [Y3 Z4 Y5 Z7]
+ (2.593534391371023e-07) [X3 Z4 X5 Z7]
+ (3.6060718678311913e-07) [Y0 Z1 Z2 Y4]
+ (3.6060718678311913e-07) [X0 Z1 Z2 X4]
+ (3.6060718678311913e-07) [Y1 Z3 Z4 Y5]
+ (3.6060718678311913e-07) [X1 Z3 Z4 X5]
+ (5.471647744402701e-07) [Y1 X2 X11 Y12]
+ (5.471647744402701e-07) [X1 Y2 Y11 X12]
+ (5.627851911067517e-07) [Y0 X1 X11 Y12]
+ (5.627851911067517e-07) [Y0 Y1 Y11 Y12]
+ (5.627851911067517e-07) [X0 X1 X11 X12]
+ (5.627851911067517e-07) [X0 Y1 Y11 X12]
+ (6.628614201136285e-07) [Y8 X9 X11 Y12]
+ (6.628614201136285e-07) [Y8 Y9 Y11 Y12]
+ (6.628614201136285e-07) [X8 X9 X11 X12]
+ (6.628614201136285e-07) [X8 Y9 Y11 X12]
+ (1.109440759006623e-06) [Z2 Y11 Z12 Y13]
+ (1.109440759006623e-06) [Z2 X11 Z12 X13]
+ (1.109440759006623e-06) [Z3 Y10 Z11 Y12]
+ (1.109440759006623e-06) [Z3 X10 Z11 X12]
+ (1.6021167406205364e-06) [Z2 Y3 Z4 Y5]
+ (1.6021167406205364e-06) [Z2 X3 Z4 X5]
+ (1.878210124609803e-06) [Z4 Y10 Z11 Y12]
+ (1.878210124609803e-06) [Z4 X10 Z11 X12]
+ (1.878210124609803e-06) [Z5 Y11 Z12 Y13]
+ (1.878210124609803e-06) [Z5 X11 Z12 X13]
+ (2.1726691013382406e-06) [Y2 X3 X11 Y12]
+ (2.1726691013382406e-06) [Y2 Y3 Y11 Y12]
+ (2.1726691013382406e-06) [X2 X3 X11 X12]
+ (2.1726691013382406e-06) [X2 Y3 Y11 X12]
+ (3.117447945833612e-06) [Y0 Z2 Z3 Y4]
+ (3.117447945833612e-06) [X0 Z2 Z3 X4]
+ (3.539054184357821e-06) [Y2 Z3 Y4 Z12]
+ (3.539054184357821e-06) [X2 Z3 X4 Z12]
+ (3.539054184357821e-06) [Y3 Z4 Y5 Z13]
+ (3.539054184357821e-06) [X3 Z4 X5 Z13]
+ (4.281913884542774e-06) [Y4 Z5 Y6 Z11]
+ (4.281913884542774e-06) [X4 Z5 X6 Z11]
+ (4.281913884542774e-06) [Y5 Z6 Y7 Z10]
+ (4.281913884542774e-06) [X5 Z6 X7 Z10]
+ (5.275883121830806e-06) [Y3 X4 X12 Y13]
+ (5.275883121830806e-06) [Y3 Y4 Y12 Y13]
+ (5.275883121830806e-06) [X3 X4 X12 X13]
+ (5.275883121830806e-06) [X3 Y4 Y12 X13]
+ (5.974311713106246e-06) [Y5 X6 X10 Y11]
+ (5.974311713106246e-06) [Y5 Y6 Y10 Y11]
+ (5.974311713106246e-06) [X5 X6 X10 X11]
+ (5.974311713106246e-06) [X5 Y6 Y10 X11]
+ (7.95441317577353e-06) [Y10 Z11 Y12 Z13]
+ (7.95441317577353e-06) [X10 Z11 X12 Z13]
+ (8.814937306188628e-06) [Y2 Z3 Y4 Z13]
+ (8.814937306188628e-06) [X2 Z3 X4 Z13]
+ (8.814937306188628e-06) [Y3 Z4 Y5 Z12]
+ (8.814937306188628e-06) [X3 Z4 X5 Z12]
+ (0.0002921986261110113) [Y7 X8 X9 Y10]
+ (0.0002921986261110113) [X7 Y8 Y9 X10]
+ (0.0004956762314915686) [Y2 Z4 Z5 Y6]
+ (0.0004956762314915686) [X2 Z4 Z5 X6]
+ (0.0011059037691896754) [Y0 Z1 Y2 Z5]
+ (0.0011059037691896754) [X0 Z1 X2 Z5]
+ (0.0011059037691896754) [Y1 Z2 Y3 Z4]
+ (0.0011059037691896754) [X1 Z2 X3 Z4]
+ (0.001663879878490808) [Y2 Z3 Z4 Y6]
+ (0.001663879878490808) [X2 Z3 Z4 X6]
+ (0.001663879878490808) [Y3 Z5 Z6 Y7]
+ (0.001663879878490808) [X3 Z5 Z6 X7]
+ (0.001756070701841229) [Y0 Z1 Y2 Z11]
+ (0.001756070701841229) [X0 Z1 X2 Z11]
+ (0.001756070701841229) [Y1 Z2 Y3 Z10]
+ (0.001756070701841229) [X1 Z2 X3 Z10]
+ (0.0023262306231580715) [Y0 Z1 Y2 Z13]
+ (0.0023262306231580715) [X0 Z1 X2 Z13]
+ (0.0023262306231580715) [Y1 Z2 Y3 Z12]
+ (0.0023262306231580715) [X1 Z2 X3 Z12]
+ (0.0027458364701868038) [Y0 X1 X4 Y5]
+ (0.0027458364701868038) [X0 Y1 Y4 X5]
+ (0.0029297686747510434) [Y0 Z1 Y2 Z9]
+ (0.0029297686747510434) [X0 Z1 X2 Z9]
+ (0.0029297686747510434) [Y1 Z2 Y3 Z8]
+ (0.0029297686747510434) [X1 Z2 X3 Z8]
+ (0.003276971931231669) [Y0 Z1 Y2 Z3]
+ (0.003276971931231669) [X0 Z1 X2 Z3]
+ (0.0033476175306661627) [Y0 Z1 Y2 Z7]
+ (0.0033476175306661627) [X0 Z1 X2 Z7]
+ (0.0033476175306661627) [Y1 Z2 Y3 Z6]
+ (0.0033476175306661627) [X1 Z2 X3 Z6]
+ (0.0035552901955042617) [Y0 Z1 Y2 Z10]
+ (0.0035552901955042617) [X0 Z1 X2 Z10]
+ (0.0035552901955042617) [Y1 Z2 Y3 Z11]
+ (0.0035552901955042617) [X1 Z2 X3 Z11]
+ (0.00514339176882516) [Y3 Y4 X5 X6]
+ (0.00514339176882516) [X3 X4 Y5 Y6]
+ (0.0052837764884029505) [Y0 X1 X12 Y13]
+ (0.0052837764884029505) [X0 Y1 Y12 X13]
+ (0.0055307592186315275) [Y0 Z1 Y2 Z4]
+ (0.0055307592186315275) [X0 Z1 X2 Z4]
+ (0.0055307592186315275) [Y1 Z2 Y3 Z5]
+ (0.0055307592186315275) [X1 Z2 X3 Z5]
+ (0.006087822480561857) [Y8 X9 X12 Y13]
+ (0.006087822480561857) [X8 Y9 Y12 X13]
+ (0.006509361201177226) [Y0 X1 X8 Y9]
+ (0.006509361201177226) [X0 Y1 Y8 X9]
+ (0.00688819435297054) [Y0 X1 X6 Y7]
+ (0.00688819435297054) [X0 Y1 Y6 X7]
+ (0.006901238249797282) [Y0 Z1 Y2 Z12]
+ (0.006901238249797282) [X0 Z1 X2 Z12]
+ (0.006901238249797282) [Y1 Z2 Y3 Z13]
+ (0.006901238249797282) [X1 Z2 X3 Z13]
+ (0.007156934919856943) [Y4 X5 X8 Y9]
+ (0.007156934919856943) [X4 Y5 Y8 X9]
+ (0.007731425250775242) [Y0 X1 X10 Y11]
+ (0.007731425250775242) [X0 Y1 Y10 X11]
+ (0.008032520918821374) [Y0 Z1 Y2 Z6]
+ (0.008032520918821374) [X0 Z1 X2 Z6]
+ (0.008032520918821374) [Y1 Z2 Y3 Z7]
+ (0.008032520918821374) [X1 Z2 X3 Z7]
+ (0.009560705729135905) [Y8 X9 X10 Y11]
+ (0.009560705729135905) [X8 Y9 Y10 X11]
+ (0.011055020596132078) [Y0 Z1 Y2 Z8]
+ (0.011055020596132078) [X0 Z1 X2 Z8]
+ (0.011055020596132078) [Y1 Z2 Y3 Z9]
+ (0.011055020596132078) [X1 Z2 X3 Z9]
+ (0.01128519020084097) [Y5 Y6 X11 X12]
+ (0.01128519020084097) [X5 X6 Y11 Y12]
+ (0.011307274008848144) [Y7 Z8 Z9 Y11]
+ (0.011307274008848144) [X7 Z8 Z9 X11]
+ (0.01198238901024799) [Y4 X5 X6 Y7]
+ (0.01198238901024799) [X4 Y5 Y6 X7]
+ (0.013873381748426063) [Y6 X7 X8 Y9]
+ (0.013873381748426063) [X6 Y7 Y8 X9]
+ (0.0145836489076127) [Y0 X1 X2 Y3]
+ (0.0145836489076127) [X0 Y1 Y2 X3]
+ (0.015577208063976469) [Y2 X3 X12 Y13]
+ (0.015577208063976469) [X2 Y3 Y12 X13]
+ (0.0173661189946514) [Y6 X7 X12 Y13]
+ (0.0173661189946514) [X6 Y7 Y12 X13]
+ (0.017680067952481556) [Y4 X5 X10 Y11]
+ (0.017680067952481556) [X4 Y5 Y10 X11]
+ (0.017825140995786578) [Y6 X7 X10 Y11]
+ (0.017825140995786578) [X6 Y7 Y10 X11]
+ (0.019028242443847196) [Y3 X4 X11 Y12]
+ (0.019028242443847196) [X3 Y4 Y11 X12]
+ (0.025384657508457358) [Y2 X3 X10 Y11]
+ (0.025384657508457358) [X2 Y3 Y10 X11]
+ (0.028685183716105893) [Y10 X11 X12 Y13]
+ (0.028685183716105893) [X10 Y11 Y12 X13]
+ (0.02981242451734587) [Y6 Z7 Z8 Y10]
+ (0.02981242451734587) [X6 Z7 Z8 X10]
+ (0.02981242451734587) [Y7 Z9 Z10 Y11]
+ (0.02981242451734587) [X7 Z9 Z10 X11]
+ (0.030104623143456882) [Y6 Z7 Z9 Y10]
+ (0.030104623143456882) [X6 Z7 Z9 X10]
+ (0.030104623143456882) [Y7 Z8 Z10 Y11]
+ (0.030104623143456882) [X7 Z8 Z10 X11]
+ (0.030787505389143988) [Y6 Z8 Z9 Y10]
+ (0.030787505389143988) [X6 Z8 Z9 X10]
+ (0.0311438179889672) [Y2 X3 X6 Y7]
+ (0.0311438179889672) [X2 Y3 Y6 X7]
+ (0.035839567953353406) [Y2 X3 X4 Y5]
+ (0.035839567953353406) [X2 Y3 Y4 X5]
+ (0.03619412355904272) [Y2 X3 X8 Y9]
+ (0.03619412355904272) [X2 Y3 Y8 X9]
+ (0.038314670294803864) [Y4 X5 X12 Y13]
+ (0.038314670294803864) [X4 Y5 Y12 X13]
+ (0.10433064780651398) [Z0 Y1 Z2 Y3]
+ (0.10433064780651398) [Z0 X1 Z2 X3]
+ (-0.12133276911042279) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133276911042279) [X2 Z3 Z4 Z5 X6]
+ (-0.12133276911042278) [Y3 Z4 Z5 Z6 Y7]
+ (-0.12133276911042278) [X3 Z4 Z5 Z6 X7]
+ (3.20207687968376e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.20207687968376e-06) [X0 Z1 Z2 Z3 X4]
+ (3.2020768796837605e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.2020768796837605e-06) [X1 Z2 Z3 Z4 X5]
+ (0.22848106564918938) [Y6 Z7 Z8 Z9 Y10]
+ (0.22848106564918938) [X6 Z7 Z8 Z9 X10]
+ (0.22848106564918944) [Y7 Z8 Z9 Z10 Y11]
+ (0.22848106564918944) [X7 Z8 Z9 Z10 X11]
+ (-0.032767657823290414) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.032767657823290414) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.032767657823290414) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.032767657823290414) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845273076) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.027115036845273076) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845273076) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.027115036845273076) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.025996177598021083) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.025996177598021083) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.017561202409646138) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.017561202409646138) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.017561202409646138) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.017561202409646138) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.014564531231173008) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.014564531231173008) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.014564531231173008) [X7 Z8 Z9 X10 X12 X13]
+ (-0.014564531231173008) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.01221504099761402) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.01221504099761402) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.01221504099761402) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.01221504099761402) [X4 Z5 X6 X11 Z12 X13]
+ (-0.01221504099761402) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.01221504099761402) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.01221504099761402) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.01221504099761402) [X5 Z6 X7 X10 Z11 X12]
+ (-0.011756013419819224) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.011756013419819224) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.011756013419819224) [X3 Z4 Z5 X6 X8 X9]
+ (-0.011756013419819224) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.0087648275756887) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.0087648275756887) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.0087648275756887) [X2 Z3 Z4 X5 X11 X12]
+ (-0.0087648275756887) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.0087648275756887) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.0087648275756887) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.0087648275756887) [X3 X4 X10 Z11 Z12 X13]
+ (-0.0087648275756887) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.008125251921381034) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.008125251921381034) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.007306759928833009) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.007306759928833009) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.007306759928833009) [X4 X5 X7 Z8 Z9 X10]
+ (-0.007306759928833009) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.005805188989826914) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.005805188989826914) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.005805188989826914) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.005805188989826914) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.005652620978017336) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.005652620978017336) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.005652620978017336) [X0 X1 X3 Z4 Z5 X6]
+ (-0.005652620978017336) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.00514339176882516) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.00514339176882516) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.00514339176882516) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.00514339176882516) [X2 Z3 X4 X5 Z6 X7]
+ (-0.004684903388155211) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.004684903388155211) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668620318776301) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.004668620318776301) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.004575007626639212) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.004575007626639212) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.004424855449441851) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.004424855449441851) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.004158797381840088) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.004158797381840088) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.004158797381840088) [X3 Z4 Z5 X6 X12 X13]
+ (-0.004158797381840088) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.003493790359890148) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.003493790359890148) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.003493790359890148) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.003493790359890148) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.0027790267990255163) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.0027790267990255163) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.002293956611352467) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.002293956611352467) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.0017992194936630327) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.0017992194936630327) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.0017278753941369536) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.0017278753941369536) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.0009298507967730506) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298507967730506) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298507967730506) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298507967730506) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533856254125389) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.0008533856254125389) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.0008145313270956607) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.0008145313270956607) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.0008145313270956607) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.0008145313270956607) [X3 Z4 Z5 Z6 X7 Z11]
+ (-7.73503688058815e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.73503688058815e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.73503688058815e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.73503688058815e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-8.774817864174164e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.774817864174164e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.774817864174164e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.774817864174164e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.51836221536367e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.51836221536367e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.51836221536367e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.51836221536367e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.44434467557869e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.44434467557869e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.44434467557869e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.44434467557869e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.524373848146317e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.524373848146317e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.524373848146317e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.524373848146317e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-6.290028432912597e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.290028432912597e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.290028432912597e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.290028432912597e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.974311713106246e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (-5.974311713106246e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (-5.275883121830807e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.275883121830807e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.643051068237289e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.643051068237289e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.556569217890057e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (-4.556569217890057e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (-4.253224225371402e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.253224225371402e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-3.76965945173768e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.76965945173768e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.6945132943250558e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.6945132943250558e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.6102971303345233e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.6102971303345233e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.6102971303345233e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.6102971303345233e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.313145499938101e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.313145499938101e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.277483195293267e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.277483195293267e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.277483195293267e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.277483195293267e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.2112283482082166e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.2112283482082166e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.2112283482082166e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.2112283482082166e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.151346311003452e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.151346311003452e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.0882507111265363e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.0882507111265363e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.1726691013382406e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.1726691013382406e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.454842448991829e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.454842448991829e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.3304731885954745e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.3304731885954745e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.228333782451074e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.228333782451074e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.03584776010621e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.03584776010621e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-7.956895371861962e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.956895371861962e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.73319774190041e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.73319774190041e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.73319774190041e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.73319774190041e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.628614201136285e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.628614201136285e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.556281914330292e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.556281914330292e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.556281914330292e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.556281914330292e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.418291574360731e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.418291574360731e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.418291574360731e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.418291574360731e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.927453082391173e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.927453082391173e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.927453082391173e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.927453082391173e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.627851911067517e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.627851911067517e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.287660624451284e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.287660624451284e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.287660624451284e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.287660624451284e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.287660624451284e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.287660624451284e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.287660624451284e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.287660624451284e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.838052750681737e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (-4.838052750681737e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (-3.5707613287601753e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.5707613287601753e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.328139350412559e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.328139350412559e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.086826564954983e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.086826564954983e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.086826564954983e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.086826564954983e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.4473231286817734e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (-2.4473231286817734e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (-2.3713289477208715e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.3713289477208715e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.3713289477208715e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.3713289477208715e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.199051618388178e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-2.199051618388178e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-1.933241277017308e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.933241277017308e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.933241277017308e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.933241277017308e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.8394209153421932e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.8394209153421932e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.8394209153421932e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.8394209153421932e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.5510539176369912e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (-1.5510539176369912e-07) [X0 Z1 X2 X4 Z5 X6]
+ (-1.5510539176369912e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (-1.5510539176369912e-07) [X1 Z2 X3 X5 Z6 X7]
+ (-1.3807781479993594e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (-1.3807781479993594e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (-1.3807781479993594e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (-1.3807781479993594e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (-1.3807781479993592e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (-1.3807781479993592e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (-1.3807781479993592e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (-1.3807781479993592e-07) [X0 Z1 X2 X5 Z6 X7]
+ (-1.3807781479993592e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (-1.3807781479993592e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (-1.3807781479993592e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (-1.3807781479993592e-07) [X1 Z2 X3 X4 Z5 X6]
+ (-1.291969486071391e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.291969486071391e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.1076325599185191e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.1076325599185191e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.1076325599185191e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.1076325599185191e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.1076325599185191e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.1076325599185191e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.1076325599185191e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.1076325599185191e-07) [X1 Z2 X3 X10 Z11 X12]
+ (8.057446595092381e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.057446595092381e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.057446595092381e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.057446595092381e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.649310134655942e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.649310134655942e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.649310134655942e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.649310134655942e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.8394209153421934e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.8394209153421934e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.8394209153421934e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.8394209153421934e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.199051618388178e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (2.199051618388178e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (2.4473231286817734e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (2.4473231286817734e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.236259961186466e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.236259961186466e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.236259961186466e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.236259961186466e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.328139350412559e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.328139350412559e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.5707613287601753e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.5707613287601753e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.838052750681737e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (4.838052750681737e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (5.627851911067517e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.627851911067517e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (6.628614201136285e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.628614201136285e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (7.956895371861962e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.956895371861962e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.30653665144212e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.30653665144212e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.30653665144212e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.30653665144212e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.03584776010621e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.03584776010621e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.228333782451074e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.228333782451074e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.23933632163971e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.23933632163971e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.23933632163971e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.23933632163971e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.3304731885954745e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.3304731885954745e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.454842448991829e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.454842448991829e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.1726691013382406e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.1726691013382406e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.0882507111265363e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.0882507111265363e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.117447945833612e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.117447945833612e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.151346311003452e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.151346311003452e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.313145499938101e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.313145499938101e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.334331289155188e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.334331289155188e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.6945132943250558e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.6945132943250558e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (4.183932559197975e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.183932559197975e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.556569217890057e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (4.556569217890057e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (4.643051068237289e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.643051068237289e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.275883121830807e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.275883121830807e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.974311713106246e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (5.974311713106246e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (0.0002921986261110113) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.0002921986261110113) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.0002921986261110113) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.0002921986261110113) [X6 Z7 X8 X9 Z10 X11]
+ (0.0004956762314915686) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.0004956762314915686) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.0006650070219499394) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.0006650070219499394) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.0006650070219499394) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.0006650070219499394) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.0008533856254125389) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.0008533856254125389) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.001609531381721385) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.001609531381721385) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.001609531381721385) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.001609531381721385) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.0016676041811440616) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.0016676041811440616) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.0016676041811440616) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.0016676041811440616) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.0017278753941369536) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.0017278753941369536) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.0017992194936630327) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.0017992194936630327) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.002293956611352467) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.002293956611352467) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.002462917007133924) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.002462917007133924) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.002462917007133924) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.002462917007133924) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.003961560792496529) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.003961560792496529) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.003961560792496529) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.003961560792496529) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.004424855449441851) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.004424855449441851) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.004575007626639212) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.004575007626639212) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.004668620318776301) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.004668620318776301) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.004684903388155211) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.004684903388155211) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.005324835234221691) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.005324835234221691) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.005324835234221691) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.005324835234221691) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.00536865935810963) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.00536865935810963) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.00536865935810963) [X2 X3 X7 Z8 Z9 X10]
+ (0.00536865935810963) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.007960880725921566) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.007960880725921566) [X4 Z5 X6 X10 Z11 X12]
+ (0.007960880725921566) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.007960880725921566) [X5 Z6 X7 X11 Z12 X13]
+ (0.008125251921381034) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.008125251921381034) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.00889073152269462) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.00889073152269462) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.00889073152269462) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.00889073152269462) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.010263414868158498) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.010263414868158498) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.010263414868158498) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.010263414868158498) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.010540425907671543) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.010540425907671543) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.010540425907671543) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.010540425907671543) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.010960074940542585) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.010960074940542585) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.010960074940542585) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.010960074940542585) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.011307274008848145) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.011307274008848145) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.014411099430130908) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.014411099430130908) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.014411099430130908) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.014411099430130908) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.01522563075722657) [Y3 Z4 Z5 X6 X10 Y11]
+ (0.01522563075722657) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (0.01522563075722657) [X3 Z4 Z5 X6 X10 X11]
+ (0.01522563075722657) [X3 Z4 Z5 Y6 Y10 X11]
+ (0.015588250102380189) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (0.015588250102380189) [X2 Z3 X4 X10 Z11 X12]
+ (0.015588250102380189) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (0.015588250102380189) [X3 Z4 X5 X11 Z12 X13]
+ (0.01826683486937559) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.01826683486937559) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.01826683486937559) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.01826683486937559) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.01902042317304) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.01902042317304) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.01902042317304) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.01902042317304) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.02017592172353559) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.02017592172353559) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.02017592172353559) [X4 Z5 Z6 X7 X11 X12]
+ (0.02017592172353559) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.02017592172353559) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.02017592172353559) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.02017592172353559) [X5 X6 X10 Z11 Z12 X13]
+ (0.02017592172353559) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.024353077678068887) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.024353077678068887) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.024353077678068887) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.024353077678068887) [X2 Z3 X4 X11 Z12 X13]
+ (0.024353077678068887) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.024353077678068887) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.024353077678068887) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.024353077678068887) [X3 Z4 X5 X10 Z11 X12]
+ (0.02438908253114963) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.02438908253114963) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.02438908253114963) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.02438908253114963) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.02510495713884455) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.02510495713884455) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.02510495713884455) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.02510495713884455) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.030787505389143988) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.030787505389143988) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.045879470781297796) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.045879470781297796) [X0 Z2 Z3 Z4 Z5 X6]
+ (0.05600733087780778) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.05600733087780778) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.05600733087780778) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.05600733087780778) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.056084681246613664) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.056084681246613664) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.056084681246613664) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.056084681246613664) [Z1 X7 Z8 Z9 Z10 X11]
+ (-6.631277928133168e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.631277928133168e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.631277928133167e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.631277928133167e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.59508600678326e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.59508600678326e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.595086006783259e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.595086006783259e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.042743277013783144) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.042743277013783144) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (0.042743277013783144) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.042743277013783144) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.047642612176383096) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.047642612176383096) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.047642612176383096) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.047642612176383096) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.041718813839821754) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.041718813839821754) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.041718813839821754) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.041718813839821754) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.039564416322893245) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.039564416322893245) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.039564416322893245) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.039564416322893245) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.039359168022053) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.039359168022053) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.039359168022053) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.039359168022053) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.03931805194719752) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03931805194719752) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.03931805194719752) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03931805194719752) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.035608378988312595) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.035608378988312595) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.02990378951262481) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.02990378951262481) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.02990378951262481) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.02990378951262481) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.028730779551905523) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.028730779551905523) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.028730779551905523) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.028730779551905523) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.025637238296026838) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.025637238296026838) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.025637238296026838) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.025637238296026838) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.024755463292890925) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.024755463292890925) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.024755463292890925) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.024755463292890925) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.024282117354692958) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.024282117354692958) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.023145130929529058) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.023145130929529058) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.022528440196012984) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.022528440196012984) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.021433810721600825) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.021433810721600825) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.021433810721600825) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.021433810721600825) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251586) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.019257505095251586) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019028242443847196) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.019028242443847196) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.018889030304942895) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-0.018889030304942895) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-0.018889030304942895) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.018889030304942895) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.01602460368917959) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.01602460368917959) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.01522563075722657) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.01522563075722657) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.014603704729162071) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.014603704729162071) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.014564531231173008) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.014564531231173008) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.011756013419819224) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.011756013419819224) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.01128519020084097) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-0.01128519020084097) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-0.009841749246962628) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.009841749246962628) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.009612634606847251) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.009612634606847251) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.009612634606847251) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.009612634606847251) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.008469978791023987) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.008469978791023987) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.007306759928833009) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.007306759928833009) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005923798336561343) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.005923798336561343) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.005652620978017336) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.005652620978017336) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.00536865935810963) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.00536865935810963) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.004158797381840088) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.004158797381840088) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0033566705638328723) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.0033566705638328723) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.0033566705638328723) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.0033566705638328723) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.0032675138544235407) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.0032675138544235407) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.0032675138544235407) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.0032675138544235407) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.0027790267990255163) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.0027790267990255163) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.0026860409778065946) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.0026860409778065946) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.0026860409778065946) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.0026860409778065946) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.002293956611352467) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.002293956611352467) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.002293956611352467) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.002293956611352467) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.0009581655836696419) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.0009581655836696419) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.0009581655836696419) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.0009581655836696419) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.0009581655836696419) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.0009581655836696419) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.0009581655836696419) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.0009581655836696419) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.0002463643756957243) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0002463643756957243) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303550585) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.00013840177303550585) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.00013840177303550585) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.00013840177303550585) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.73503688058815e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.73503688058815e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.610358530490267e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.610358530490267e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.610358530490267e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.610358530490267e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.531680879453455e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.531680879453455e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.531680879453455e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.531680879453455e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.806102774704692e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.806102774704692e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.806102774704692e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.806102774704692e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.089799467201424e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.089799467201424e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.089799467201424e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.089799467201424e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.652209669046866e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.652209669046866e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.652209669046866e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.652209669046866e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.4818518335139955e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.4818518335139955e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.4818518335139955e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.4818518335139955e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.071480736194055e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.071480736194055e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.071480736194055e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.071480736194055e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.7346220385106356e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.7346220385106356e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.7346220385106356e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.7346220385106356e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.728843147029078e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.728843147029078e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.728843147029078e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.728843147029078e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.253224225371402e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.253224225371402e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.76965945173768e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.76965945173768e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.5443954290896734e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.5443954290896734e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.5443954290896734e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.5443954290896734e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.5443954290896734e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.5443954290896734e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.5443954290896734e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.5443954290896734e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.360956320172346e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.360956320172346e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.360956320172346e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.360956320172346e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-2.1032156045380653e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.1032156045380653e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.1032156045380653e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.1032156045380653e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.011122098070779e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.011122098070779e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.011122098070779e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.011122098070779e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9429468365398185e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.9429468365398185e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9429468365398185e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9429468365398185e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174769646403e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.6541174769646403e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174769646403e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.6541174769646403e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.5224930675811794e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.5224930675811794e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.5224930675811794e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.5224930675811794e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.5224930675811794e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930675811794e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930675811794e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.5224930675811794e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.228333782451074e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.228333782451074e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.228333782451074e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.228333782451074e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.98877028808124e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.98877028808124e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.98877028808124e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.98877028808124e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.867765103681205e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765103681205e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765103681205e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.867765103681205e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.189990974865645e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.189990974865645e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.175246206713223e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175246206713223e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.471647744402701e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.471647744402701e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.561447179631377e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.561447179631377e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.561447179631377e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.561447179631377e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.523389677635892e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.523389677635892e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.4273231084498627e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.4273231084498627e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.4273231084498627e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.4273231084498627e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.328139350412559e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350412559e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.328139350412559e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350412559e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.086826564954982e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.086826564954982e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.888293595751781e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.888293595751781e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.888293595751781e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.888293595751781e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.3713289477208715e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.3713289477208715e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.8394209153421932e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.8394209153421932e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-8.05744659509238e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.05744659509238e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (4.537178097072084e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.537178097072084e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.537178097072084e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (4.537178097072084e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (8.05744659509238e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.05744659509238e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (9.209350646728657e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350646728657e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350646728657e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.209350646728657e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783553287164e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783553287164e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783553287164e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783553287164e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.8394209153421932e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.8394209153421932e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.3713289477208715e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.3713289477208715e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (3.086826564954982e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.086826564954982e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (4.523389677635892e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.523389677635892e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (5.471647744402701e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.471647744402701e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.175246206713223e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175246206713223e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.189990974865645e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.189990974865645e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.3304731885954745e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.3304731885954745e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.3304731885954745e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.3304731885954745e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.6288532434222723e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.6288532434222723e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.6288532434222723e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.6288532434222723e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.6893489513831933e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.6893489513831933e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.6893489513831933e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.6893489513831933e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (2.745518400281549e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.745518400281549e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.745518400281549e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.745518400281549e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.745518400281549e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.745518400281549e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.745518400281549e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.745518400281549e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.211842018964373e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.211842018964373e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.211842018964373e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.211842018964373e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.211842018964373e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.211842018964373e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.211842018964373e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.211842018964373e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.313145499938101e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.313145499938101e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.313145499938101e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.313145499938101e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (3.334331289155188e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.334331289155188e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (4.183932559197975e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.183932559197975e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (7.73503688058815e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.73503688058815e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.0002463643756957243) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.0002463643756957243) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0004458535128840867) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.0004458535128840867) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.0004458535128840867) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.0004458535128840867) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.0005940221543005483) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005483) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005483) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005483) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005483) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005483) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.0005940221543005483) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005483) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.0008533856254125387) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.0008533856254125387) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.0008533856254125387) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.0008533856254125387) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.0010435246534907664) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.0010435246534907664) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.0010435246534907664) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.0010435246534907664) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.00128030609734968) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.00128030609734968) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.00128030609734968) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.00128030609734968) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.0013038004788126956) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (0.0013038004788126956) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (0.0013038004788126956) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (0.0013038004788126956) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (0.0022619660624823377) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (0.0022619660624823377) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (0.0022619660624823377) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (0.0022619660624823377) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (0.0022619660624823377) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (0.0022619660624823377) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (0.0022619660624823377) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (0.0022619660624823377) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (0.003989841456619293) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (0.003989841456619293) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (0.003989841456619293) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (0.003989841456619293) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (0.004158797381840088) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.004158797381840088) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.004311038507914307) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.004311038507914307) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.004311038507914307) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.004311038507914307) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.004636976661182552) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.004636976661182552) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.004636976661182552) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.004636976661182552) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.005114473831660389) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.005114473831660389) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.005114473831660389) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.005114473831660389) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.005114473831660389) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660389) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660389) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005114473831660389) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.005241535382803855) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (0.005241535382803855) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (0.005241535382803855) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (0.005241535382803855) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (0.00526264247307685) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.00526264247307685) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.00526264247307685) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.00526264247307685) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.00536865935810963) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.00536865935810963) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.00537993715583936) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.00537993715583936) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.00537993715583936) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.00537993715583936) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.005652620978017336) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.005652620978017336) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.005708495985960936) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.005708495985960936) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.005708495985960936) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.005708495985960936) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.005923798336561343) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.005923798336561343) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.007306759928833009) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.007306759928833009) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.008469978791023987) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.008469978791023987) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.009841749246962628) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.009841749246962628) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.01128519020084097) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (0.01128519020084097) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (0.011756013419819224) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.011756013419819224) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.014564531231173008) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.014564531231173008) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.014603704729162071) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.014603704729162071) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.01522563075722657) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.01522563075722657) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.01602460368917959) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.01602460368917959) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.019028242443847196) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.019028242443847196) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.019257505095251586) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.019257505095251586) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.045879470781297796) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.045879470781297796) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.36937089366156156) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.36937089366156156) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.36937089366156156) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.36937089366156156) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.2816425776702276) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.2816425776702276) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.2816425776702275) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.2816425776702275) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.09065144207036463) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.09065144207036463) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.09065144207036463) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.09065144207036463) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0868473758986361) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0868473758986361) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0868473758986361) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0868473758986361) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.07635021950634988) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07635021950634988) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07635021950634988) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07635021950634988) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214006) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.06752385099214006) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214006) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.06752385099214006) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.035608378988312595) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.035608378988312595) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.034903343373661924) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.034903343373661924) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.034903343373661924) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.034903343373661924) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.02459186088383001) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.02459186088383001) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.02459186088383001) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.02459186088383001) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.024282117354692954) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.024282117354692954) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.023145130929529058) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.023145130929529058) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.022528440196012984) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.022528440196012984) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019538050311314645) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.019538050311314645) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.019538050311314645) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.019538050311314645) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.017091553155898838) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.017091553155898838) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.017091553155898838) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.017091553155898838) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.01602460368917959) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.01602460368917959) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.01602460368917959) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.01602460368917959) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.010311482489831918) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831918) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831918) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010311482489831918) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.009841749246962626) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962626) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.009841749246962626) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962626) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209822) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209822) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209822) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008826368514209822) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008541996625454799) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454799) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454799) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454799) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454799) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454799) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454799) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008541996625454799) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008469978791023987) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023987) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.008469978791023987) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023987) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.004668620318776301) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.004668620318776301) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.003876470899336929) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.003876470899336929) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.003804066171728533) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728533) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728533) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003804066171728533) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178873) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178873) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0033566705638328723) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.0033566705638328723) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.0032675138544235403) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.0032675138544235403) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0021413612231015997) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.0021413612231015997) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.0017278753941369536) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.0017278753941369536) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.0016407548553124146) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0016407548553124146) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214169178) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0014528843214169178) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214169178) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0014528843214169178) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870896771024368) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007870896771024368) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0005192743499487701) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.0005192743499487701) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.00019400857029756142) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00019400857029756142) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303550585) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.00013840177303550585) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.141625221156769e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.141625221156769e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.141625221156769e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.141625221156769e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-5.071480736194055e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.071480736194055e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.151346311003452e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.151346311003452e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0882507111265363e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.0882507111265363e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.9885117061120012e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.9885117061120012e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.874299071210645e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.874299071210645e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.3609563201723465e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.3609563201723465e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.3002946561342375e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.3002946561342375e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.1468376506853182e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.1468376506853182e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.1468376506853182e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.1468376506853182e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.352332102618319e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.352332102618319e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.352332102618319e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.352332102618319e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.091637198614134e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637198614134e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637198614134e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637198614134e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637198614134e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637198614134e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637198614134e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.091637198614134e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.074305985469693e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.074305985469693e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.074305985469693e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.074305985469693e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.90012898587494e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.90012898587494e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.90012898587494e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.90012898587494e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867765103681205e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.867765103681205e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.56069246440067e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.56069246440067e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.56069246440067e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.56069246440067e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.56069246440067e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.56069246440067e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.56069246440067e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.56069246440067e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.997018421882503e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.997018421882503e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.997018421882503e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.997018421882503e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.997018421882503e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.997018421882503e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.997018421882503e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.997018421882503e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.56824752097824e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.56824752097824e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.56824752097824e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.56824752097824e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.376739308239047e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.376739308239047e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.376739308239047e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.376739308239047e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.376739308239047e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.376739308239047e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.376739308239047e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.376739308239047e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.888293595751781e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.888293595751781e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6863815453337207e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.6863815453337207e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035783553287164e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7035783553287164e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.209350646728658e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.209350646728658e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244108506e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773244108506e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244108506e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773244108506e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244108506e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773244108506e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.379773244108506e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773244108506e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253793912213e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253793912213e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9742253793912213e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9742253793912213e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0474716555374167e-08) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.0474716555374167e-08) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.0474716555374167e-08) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.0474716555374167e-08) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.209350646728658e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.209350646728658e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.071728218311243e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.071728218311243e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.071728218311243e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.071728218311243e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.200428749321666e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.200428749321666e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.200428749321666e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.200428749321666e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.7035783553287164e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7035783553287164e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.3120943050812317e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.3120943050812317e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.3120943050812317e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.3120943050812317e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.6863815453337207e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815453337207e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.888293595751781e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.888293595751781e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.0922506158713994e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506158713994e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.0922506158713994e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506158713994e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.0922506158713994e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506158713994e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.0922506158713994e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506158713994e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.444597853872238e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.444597853872238e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.444597853872238e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.444597853872238e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.6849150949069334e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.6849150949069334e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.6849150949069334e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.6849150949069334e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.246974425024465e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.246974425024465e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.246974425024465e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.246974425024465e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.246974425024465e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.246974425024465e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.246974425024465e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.246974425024465e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.867765103681205e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.867765103681205e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.3002946561342375e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.3002946561342375e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.3609563201723465e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.3609563201723465e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.874299071210645e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.874299071210645e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.883676575867854e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.883676575867854e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.947356011519291e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.947356011519291e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.947356011519291e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.947356011519291e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117061120012e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9885117061120012e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.0882507111265363e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.0882507111265363e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.151346311003452e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.151346311003452e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.846201671027228e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.846201671027228e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.846201671027228e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.846201671027228e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.071480736194055e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.071480736194055e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.10552672167468e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.10552672167468e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.10552672167468e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.10552672167468e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.146496327161466e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.146496327161466e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.146496327161466e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.146496327161466e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.159350501598212e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.159350501598212e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.159350501598212e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.159350501598212e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.427988656131584e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.427988656131584e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.427988656131584e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.427988656131584e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.935867717631292e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.935867717631292e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.935867717631292e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.935867717631292e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273347635485e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.253273347635485e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.979825792885325e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.979825792885325e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.979825792885325e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.979825792885325e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (4.205548411216446e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.205548411216446e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.205548411216446e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.205548411216446e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00013840177303550585) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.00013840177303550585) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.00018787053389549677) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.00018787053389549677) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.00018787053389549677) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.00018787053389549677) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00019400857029756142) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00019400857029756142) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0002463643756957243) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0002463643756957243) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0002463643756957243) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0002463643756957243) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192743499487701) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.0005192743499487701) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.0007156734248908691) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007156734248908691) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0007156734248908691) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007156734248908691) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024368) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007870896771024368) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.0015324835230730199) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.0015324835230730199) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.0015324835230730199) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.0015324835230730199) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.0016407548553124146) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0016407548553124146) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0017278753941369536) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.0017278753941369536) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.002446497155415807) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.002446497155415807) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.002446497155415807) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.002446497155415807) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.0032675138544235403) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.0032675138544235403) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.0033566705638328723) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.0033566705638328723) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.0034841573002178873) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0034841573002178873) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.003876470899336929) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.003876470899336929) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.004668620318776301) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.004668620318776301) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278072) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.004767272188278072) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.004767272188278072) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278072) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.005286546538226843) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.005286546538226843) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.005286546538226843) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.005286546538226843) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.00540895442240995) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.00540895442240995) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.00540895442240995) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.00540895442240995) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.005923798336561343) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561343) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.005923798336561343) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561343) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.010715508469796785) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010715508469796785) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010715508469796785) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010715508469796785) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.010757563953908948) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010757563953908948) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010757563953908948) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010757563953908948) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.014603704729162071) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.014603704729162071) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.014603704729162071) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.014603704729162071) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.01929956057936375) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01929956057936375) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01929956057936375) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01929956057936375) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01929956057936375) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01929956057936375) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.01929956057936375) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01929956057936375) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.05859198873386151) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.05859198873386151) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.7759505268340893e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.7759505268340893e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.7759505268340893e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.7759505268340893e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.07165035181002567) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.07165035181002567) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.07165035181002567) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.07165035181002567) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251586) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019257505095251586) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010311482489831918) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311482489831918) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008826368514209822) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209822) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.007597464029770606) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.007597464029770606) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.007597464029770606) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.007597464029770606) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311871) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311871) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311871) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311871) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311871) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311871) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311871) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311871) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676623) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005348051582676623) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005348051582676623) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676623) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.003804066171728533) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003804066171728533) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002984166168121919) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.002984166168121919) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.002984166168121919) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.002984166168121919) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.002446497155415808) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.002446497155415808) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.0022494124470939835) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0022494124470939835) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0022494124470939835) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0022494124470939835) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0021413612231015997) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.0021413612231015997) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.0018638942824587346) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587346) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587346) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587346) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587346) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587346) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0018638942824587346) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587346) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0016407548553124146) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124146) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0016407548553124146) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124146) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0012223378081538223) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538223) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538223) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538223) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538223) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538223) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538223) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0012223378081538223) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0010283292378562613) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0010283292378562613) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0010283292378562613) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0010283292378562613) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1463061452372669e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1463061452372669e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.874299071210645e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.874299071210645e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.874299071210645e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.874299071210645e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.3002946561342375e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.3002946561342375e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.3002946561342375e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.3002946561342375e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.0444941297352302e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.0444941297352302e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.0444941297352302e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.0444941297352302e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.95607922934491e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.95607922934491e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.95607922934491e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.95607922934491e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105515036628909e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105515036628909e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105515036628909e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105515036628909e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661347212645995e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661347212645995e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661347212645995e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661347212645995e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540341413255192e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540341413255192e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.189990974865645e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.189990974865645e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.876621657857019e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876621657857019e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876621657857019e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876621657857019e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175246206713223e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175246206713223e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.523389677635892e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.523389677635892e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.076732531647536e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.076732531647536e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.076732531647536e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.076732531647536e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.013471458720934e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.013471458720934e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.904599884097109e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.904599884097109e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.904599884097109e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.904599884097109e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.666731754448035e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.666731754448035e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.666731754448035e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.666731754448035e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641927160019e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.8505641927160019e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.656930931550381e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.656930931550381e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.656930931550381e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.656930931550381e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.8505641927160019e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.8505641927160019e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815453337207e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815453337207e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.6863815453337207e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815453337207e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458720934e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.013471458720934e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.523389677635892e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.523389677635892e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.6704023902713163e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.6704023902713163e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.6704023902713163e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.6704023902713163e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175246206713223e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175246206713223e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.189990974865645e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.189990974865645e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540341413255192e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540341413255192e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949476486851107e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949476486851107e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.7924939576008286e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939576008286e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939576008286e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.7924939576008286e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.8836765758678546e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.8836765758678546e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.9885117061120012e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117061120012e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117061120012e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117061120012e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273347635485e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.253273347635485e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4017109734405522e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4017109734405522e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4017109734405522e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4017109734405522e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603692006348e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.5809603692006348e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603692006348e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.5809603692006348e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0005192743499487701) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487701) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.0005192743499487701) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487701) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.0007870896771024368) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024368) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024368) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024368) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0011726348316441898) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0011726348316441898) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0011726348316441898) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0011726348316441898) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012366478019245216) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.0012366478019245216) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.0012366478019245216) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.0012366478019245216) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.0022009640695004507) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0022009640695004507) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0022009640695004507) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0022009640695004507) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980123) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980123) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0023949726397980123) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980123) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0023949726397980123) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0023949726397980123) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0023949726397980123) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0023949726397980123) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002446497155415808) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.002446497155415808) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.003804066171728533) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.003804066171728533) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0038764708993369286) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.0038764708993369286) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.0038764708993369286) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.0038764708993369286) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.00422081397004644) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.00422081397004644) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.00422081397004644) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.00422081397004644) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.008826368514209822) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.008826368514209822) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.010311482489831918) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010311482489831918) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019257505095251586) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019257505095251586) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.05859198873386151) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.05859198873386151) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.3987009014424412e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.3987009014424412e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.3987009014424408e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.3987009014424408e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841573002178873) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841573002178873) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0029841661681219195) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.0029841661681219195) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.00019400857029756142) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00019400857029756142) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061452372669e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061452372669e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.7924939576008286e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7924939576008286e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540341413255192e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413255192e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540341413255192e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413255192e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.8505641927160019e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641927160019e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641927160019e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641927160019e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458720935e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471458720935e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458720935e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471458720935e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.949476486851107e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.949476486851107e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.7924939576008286e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7924939576008286e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029756142) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029756142) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0029841661681219195) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.0029841661681219195) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.0034841573002178873) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0034841573002178873) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
Expectation value of XYI =  0.022659767960222316
Expectation value of XIZ =  0.07715357869738931
[0.27361669 0.00898685 0.26297431 0.00732554 0.21720814 0.00116213
 0.22790267 0.00082366]
Expectation value of XYI =  0.02265976796022237
Expectation value of XIZ =  0.07715357869738954
[0.02265977 0.07715358]
[RY(-1.5707963267948966, wires=[0]), RX(1.5707963267948966, wires=[1])]
[PauliZ(wires=[0]) @ PauliZ(wires=[1]), PauliZ(wires=[0]) @ PauliZ(wires=[2])]
pennylane.qnodes.base.QuantumFunctionError: Only observables that are qubit-wise commuting
Pauli words can be returned on the same wire
Minimum number of QWC groupings found: 2
Group 0:
Y0 X2 X3
Y0 Y1 X2 X3
X2 X3
Group 1:
Z0 Z1 Z2
Z0 Z1 Z2 Z3
Z0
Z0 Z1
Term expectation values:
Group 0 expectation values: [-0.14012997  0.01555488  0.18967764]
Group 1 expectation values: [0.93755207 0.94996042 0.96302938 0.96118149]
<H> =  3.8768259168631207
3.8768259168631207
Number of Hamiltonian terms/required measurements: 2050
Number of required measurements after optimization: 523
 </code>
 </pre>
 </details>

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_measurement_optimize.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
  (-0.24274280036030488) [Z2]
+ (-0.24274280036030488) [Z3]
+ (-0.04207898548721717) [I0]
+ (0.17771287509137346) [Z0]
+ (0.17771287509137346) [Z1]
+ (0.12293305042061924) [Z0 Z2]
+ (0.12293305042061924) [Z1 Z3]
+ (0.1676831942744919) [Z0 Z3]
+ (0.1676831942744919) [Z1 Z2]
+ (0.1705973836365957) [Z0 Z1]
+ (0.1762764071454046) [Z2 Z3]
+ (-0.04475014385387267) [Y0 Y1 X2 X3]
+ (-0.04475014385387267) [X0 X1 Y2 Y3]
+ (0.04475014385387267) [Y0 X1 X2 Y3]
+ (0.04475014385387267) [X0 Y1 Y2 X3]
Cost function value: -0.5657291586576485
Number of Hamiltonian terms/required measurements: 2110
   (-46.46390691195639) [I0]
+ (0.7829652070217062) [Z11]
+ (0.7829652070217064) [Z10]
+ (0.8084591005240847) [Z13]
+ (0.8084591005240849) [Z12]
+ (1.2034393392162677) [Z5]
+ (1.203439339216268) [Z4]
+ (1.3096876619515256) [Z7]
+ (1.309687661951526) [Z6]
+ (1.369352571274335) [Z9]
+ (1.3693525712743355) [Z8]
+ (1.6538938305636546) [Z3]
+ (1.653893830563655) [Z2]
+ (12.412630713955505) [Z0]
+ (12.412630713955505) [Z1]
+ (-7.95422479824004e-06) [Y11 Y13]
+ (-7.95422479824004e-06) [X11 X13]
+ (-1.6021751094425738e-06) [Y2 Y4]
+ (-1.6021751094425738e-06) [X2 X4]
+ (5.929280653392023e-07) [Y4 Y6]
+ (5.929280653392023e-07) [X4 X6]
+ (7.765082203324298e-07) [Y3 Y5]
+ (7.765082203324298e-07) [X3 X5]
+ (1.8540565467409491e-06) [Y5 Y7]
+ (1.8540565467409491e-06) [X5 X7]
+ (8.194105025175619e-06) [Y10 Y12]
+ (8.194105025175619e-06) [X10 X12]
+ (0.003276965063200823) [Y1 Y3]
+ (0.003276965063200823) [X1 X3]
+ (0.10433061483969047) [Y0 Y2]
+ (0.10433061483969047) [X0 X2]
+ (0.11270381857119745) [Z10 Z12]
+ (0.11270381857119745) [Z11 Z13]
+ (0.11383573684190564) [Z4 Z12]
+ (0.11383573684190564) [Z5 Z13]
+ (0.11952441015517323) [Z6 Z10]
+ (0.11952441015517323) [Z7 Z11]
+ (0.12489977361701074) [Z4 Z10]
+ (0.12489977361701074) [Z5 Z11]
+ (0.12495799327865759) [Z2 Z4]
+ (0.12495799327865759) [Z3 Z5]
+ (0.127994927997404) [Z2 Z10]
+ (0.127994927997404) [Z3 Z11]
+ (0.13401737371849506) [Z6 Z12]
+ (0.13401737371849506) [Z7 Z13]
+ (0.13701191913320687) [Z4 Z6]
+ (0.13701191913320687) [Z5 Z7]
+ (0.13734942208666434) [Z6 Z11]
+ (0.13734942208666434) [Z7 Z10]
+ (0.137391123747511) [Z2 Z6]
+ (0.137391123747511) [Z3 Z7]
+ (0.1376685913246082) [Z8 Z10]
+ (0.1376685913246082) [Z9 Z11]
+ (0.14011294748302064) [Z2 Z12]
+ (0.14011294748302064) [Z3 Z13]
+ (0.14138903587613463) [Z10 Z13]
+ (0.14138903587613463) [Z11 Z12]
+ (0.14257991127040476) [Z4 Z11]
+ (0.14257991127040476) [Z5 Z10]
+ (0.14722930781995497) [Z8 Z11]
+ (0.14722930781995497) [Z9 Z10]
+ (0.1489942617150456) [Z4 Z7]
+ (0.1489942617150456) [Z5 Z6]
+ (0.1492634705692818) [Z10 Z11]
+ (0.14960692557711816) [Z4 Z8]
+ (0.14960692557711816) [Z5 Z9]
+ (0.14973497004652467) [Z8 Z12]
+ (0.14973497004652467) [Z9 Z13]
+ (0.15071405482156747) [Z2 Z8]
+ (0.15071405482156747) [Z3 Z9]
+ (0.1513834269801586) [Z6 Z13]
+ (0.1513834269801586) [Z7 Z12]
+ (0.15215040621457274) [Z4 Z13]
+ (0.15215040621457274) [Z5 Z12]
+ (0.1533795916913557) [Z2 Z11]
+ (0.1533795916913557) [Z3 Z10]
+ (0.15435760063000475) [Z12 Z13]
+ (0.15569017455667172) [Z2 Z13]
+ (0.15569017455667172) [Z3 Z12]
+ (0.1558228068505096) [Z8 Z13]
+ (0.1558228068505096) [Z9 Z12]
+ (0.15676384610696634) [Z4 Z9]
+ (0.15676384610696634) [Z5 Z8]
+ (0.15755303804209367) [Z4 Z5]
+ (0.16079755046333477) [Z2 Z5]
+ (0.16079755046333477) [Z3 Z4]
+ (0.1675666935673512) [Z6 Z8]
+ (0.1675666935673512) [Z7 Z9]
+ (0.16853492794132535) [Z2 Z7]
+ (0.16853492794132535) [Z3 Z6]
+ (0.18144009363567953) [Z6 Z9]
+ (0.18144009363567953) [Z7 Z8]
+ (0.18189081242793587) [Z2 Z3]
+ (0.18690814830809194) [Z2 Z9]
+ (0.18690814830809194) [Z3 Z8]
+ (0.19299700266841233) [Z0 Z10]
+ (0.19299700266841233) [Z1 Z11]
+ (0.1939257433559061) [Z6 Z7]
+ (0.19661749959337) [Z0 Z4]
+ (0.19661749959337) [Z1 Z5]
+ (0.19936332690914366) [Z0 Z5]
+ (0.19936332690914366) [Z1 Z4]
+ (0.20072843551444322) [Z0 Z11]
+ (0.20072843551444322) [Z1 Z10]
+ (0.2110268123176738) [Z0 Z12]
+ (0.2110268123176738) [Z1 Z13]
+ (0.21631059807106104) [Z0 Z13]
+ (0.21631059807106104) [Z1 Z12]
+ (0.22003977241170355) [Z8 Z9]
+ (0.23671071738496285) [Z0 Z2]
+ (0.23671071738496285) [Z1 Z3]
+ (0.24164696831124496) [Z0 Z6]
+ (0.24164696831124496) [Z1 Z7]
+ (0.24853517285356913) [Z0 Z7]
+ (0.24853517285356913) [Z1 Z6]
+ (0.25129435571056036) [Z0 Z3]
+ (0.25129435571056036) [Z1 Z2]
+ (0.27232518449590626) [Z0 Z8]
+ (0.27232518449590626) [Z1 Z9]
+ (0.2788345457300503) [Z0 Z9]
+ (0.2788345457300503) [Z1 Z8]
+ (1.1861764482930046) [Z0 Z1]
+ (3.886639631053698e-06) [Y2 Z3 Y4]
+ (3.886639631053698e-06) [X2 Z3 X4]
+ (3.886639631053698e-06) [Y3 Z4 Y5]
+ (3.886639631053698e-06) [X3 Z4 X5]
+ (1.0722748257714844e-05) [Y10 Z11 Y12]
+ (1.0722748257714844e-05) [X10 Z11 X12]
+ (1.0722748257714846e-05) [Y11 Z12 Y13]
+ (1.0722748257714846e-05) [X11 Z12 X13]
+ (1.2260276944187822e-05) [Y4 Z5 Y6]
+ (1.2260276944187822e-05) [X4 Z5 X6]
+ (1.2260276944187822e-05) [Y5 Z6 Y7]
+ (1.2260276944187822e-05) [X5 Z6 X7]
+ (0.12507036882207412) [Y1 Z2 Y3]
+ (0.12507036882207412) [X1 Z2 X3]
+ (0.12507036882207415) [Y0 Z1 Y2]
+ (0.12507036882207415) [X0 Z1 X2]
+ (-0.03831466937266708) [Y4 Y5 X12 X13]
+ (-0.03831466937266708) [X4 X5 Y12 Y13]
+ (-0.03619409348652447) [Y2 Y3 X8 X9]
+ (-0.03619409348652447) [X2 X3 Y8 Y9]
+ (-0.03583955718467718) [Y2 Y3 X4 X5]
+ (-0.03583955718467718) [X2 X3 Y4 Y5]
+ (-0.031143804193814368) [Y2 Y3 X6 X7]
+ (-0.031143804193814368) [X2 X3 Y6 Y7]
+ (-0.0307874407269806) [Y6 Z8 Z9 Y10]
+ (-0.0307874407269806) [X6 Z8 Z9 X10]
+ (-0.030104525280106133) [Y6 Z7 Z9 Y10]
+ (-0.030104525280106133) [X6 Z7 Z9 X10]
+ (-0.030104525280106133) [Y7 Z8 Z10 Y11]
+ (-0.030104525280106133) [X7 Z8 Z10 X11]
+ (-0.02981229960800575) [Y6 Z7 Z8 Y10]
+ (-0.02981229960800575) [X6 Z7 Z8 X10]
+ (-0.02981229960800575) [Y7 Z9 Z10 Y11]
+ (-0.02981229960800575) [X7 Z9 Z10 X11]
+ (-0.02868521730493718) [Y10 Y11 X12 X13]
+ (-0.02868521730493718) [X10 X11 Y12 Y13]
+ (-0.02599620626470892) [Y3 Z4 Z5 Y7]
+ (-0.02599620626470892) [X3 Z4 Z5 X7]
+ (-0.025384663693951686) [Y2 Y3 X10 X11]
+ (-0.025384663693951686) [X2 X3 Y10 Y11]
+ (-0.019028318717598386) [Y3 Y4 X11 X12]
+ (-0.019028318717598386) [X3 X4 Y11 Y12]
+ (-0.017825011931491108) [Y6 Y7 X10 X11]
+ (-0.017825011931491108) [X6 X7 Y10 Y11]
+ (-0.01768013765339401) [Y4 Y5 X10 X11]
+ (-0.01768013765339401) [X4 X5 Y10 Y11]
+ (-0.017366053261663555) [Y6 Y7 X12 X13]
+ (-0.017366053261663555) [X6 X7 Y12 Y13]
+ (-0.01557722707365107) [Y2 Y3 X12 X13]
+ (-0.01557722707365107) [X2 X3 Y12 Y13]
+ (-0.014583638325597528) [Y0 Y1 X2 X3]
+ (-0.014583638325597528) [X0 X1 Y2 Y3]
+ (-0.013873400068328323) [Y6 Y7 X8 X9]
+ (-0.013873400068328323) [X6 X7 Y8 Y9]
+ (-0.011982342581838742) [Y4 Y5 X6 X7]
+ (-0.011982342581838742) [X4 X5 Y6 Y7]
+ (-0.01130720803595055) [Y7 Z8 Z9 Y11]
+ (-0.01130720803595055) [X7 Z8 Z9 X11]
+ (-0.011285144615357718) [Y5 X6 X11 Y12]
+ (-0.011285144615357718) [X5 Y6 Y11 X12]
+ (-0.009560716495346785) [Y8 Y9 X10 X11]
+ (-0.009560716495346785) [X8 X9 Y10 Y11]
+ (-0.008125248410398368) [Y1 X2 X8 Y9]
+ (-0.008125248410398368) [Y1 Y2 Y8 Y9]
+ (-0.008125248410398368) [X1 X2 X8 X9]
+ (-0.008125248410398368) [X1 Y2 Y8 X9]
+ (-0.007731432846030907) [Y0 Y1 X10 X11]
+ (-0.007731432846030907) [X0 X1 Y10 Y11]
+ (-0.007156920529848183) [Y4 Y5 X8 X9]
+ (-0.007156920529848183) [X4 X5 Y8 Y9]
+ (-0.006888204542324161) [Y0 Y1 X6 X7]
+ (-0.006888204542324161) [X0 X1 Y6 Y7]
+ (-0.006509361234144078) [Y0 Y1 X8 X9]
+ (-0.006509361234144078) [X0 X1 Y8 Y9]
+ (-0.006087836803984877) [Y8 Y9 X12 X13]
+ (-0.006087836803984877) [X8 X9 Y12 Y13]
+ (-0.0052837857533872) [Y0 Y1 X12 X13]
+ (-0.0052837857533872) [X0 X1 Y12 Y13]
+ (-0.005143382384883107) [Y3 X4 X5 Y6]
+ (-0.005143382384883107) [X3 Y4 Y5 X6]
+ (-0.004684920227376266) [Y1 X2 X6 Y7]
+ (-0.004684920227376266) [Y1 Y2 Y6 Y7]
+ (-0.004684920227376266) [X1 X2 X6 X7]
+ (-0.004684920227376266) [X1 Y2 Y6 X7]
+ (-0.0045750151885197) [Y1 X2 X12 Y13]
+ (-0.0045750151885197) [Y1 Y2 Y12 Y13]
+ (-0.0045750151885197) [X1 X2 X12 X13]
+ (-0.0045750151885197) [X1 Y2 Y12 X13]
+ (-0.004424843669057876) [Y1 X2 X4 Y5]
+ (-0.004424843669057876) [Y1 Y2 Y4 Y5]
+ (-0.004424843669057876) [X1 X2 X4 X5]
+ (-0.004424843669057876) [X1 Y2 Y4 X5]
+ (-0.003479421726134383) [Y2 Z3 Z5 Y6]
+ (-0.003479421726134383) [X2 Z3 Z5 X6]
+ (-0.003479421726134383) [Y3 Z4 Z6 Y7]
+ (-0.003479421726134383) [X3 Z4 Z6 X7]
+ (-0.002745827315773686) [Y0 Y1 X4 X5]
+ (-0.002745827315773686) [X0 X1 Y4 Y5]
+ (-0.0017991930085310929) [Y1 X2 X10 Y11]
+ (-0.0017991930085310929) [Y1 Y2 Y10 Y11]
+ (-0.0017991930085310929) [X1 X2 X10 X11]
+ (-0.0017991930085310929) [X1 Y2 Y10 X11]
+ (-0.00029222567210038025) [Y7 X8 X9 Y10]
+ (-0.00029222567210038025) [X7 Y8 Y9 X10]
+ (-8.814793550181946e-06) [Y2 Z3 Y4 Z13]
+ (-8.814793550181946e-06) [X2 Z3 X4 Z13]
+ (-8.814793550181946e-06) [Y3 Z4 Y5 Z12]
+ (-8.814793550181946e-06) [X3 Z4 X5 Z12]
+ (-7.95422479824004e-06) [Y10 Z11 Y12 Z13]
+ (-7.95422479824004e-06) [X10 Z11 X12 Z13]
+ (-5.974176940837152e-06) [Y5 X6 X10 Y11]
+ (-5.974176940837152e-06) [Y5 Y6 Y10 Y11]
+ (-5.974176940837152e-06) [X5 X6 X10 X11]
+ (-5.974176940837152e-06) [X5 Y6 Y10 X11]
+ (-5.275783513437838e-06) [Y3 X4 X12 Y13]
+ (-5.275783513437838e-06) [Y3 Y4 Y12 Y13]
+ (-5.275783513437838e-06) [X3 X4 X12 X13]
+ (-5.275783513437838e-06) [X3 Y4 Y12 X13]
+ (-4.28181208604779e-06) [Y4 Z5 Y6 Z11]
+ (-4.28181208604779e-06) [X4 Z5 X6 Z11]
+ (-4.28181208604779e-06) [Y5 Z6 Y7 Z10]
+ (-4.28181208604779e-06) [X5 Z6 X7 Z10]
+ (-3.5390100367438913e-06) [Y2 Z3 Y4 Z12]
+ (-3.5390100367438913e-06) [X2 Z3 X4 Z12]
+ (-3.5390100367438913e-06) [Y3 Z4 Y5 Z13]
+ (-3.5390100367438913e-06) [X3 Z4 X5 Z13]
+ (-3.117366415691129e-06) [Y0 Z2 Z3 Y4]
+ (-3.117366415691129e-06) [X0 Z2 Z3 X4]
+ (-2.1726380430164708e-06) [Y2 X3 X11 Y12]
+ (-2.1726380430164708e-06) [Y2 Y3 Y11 Y12]
+ (-2.1726380430164708e-06) [X2 X3 X11 X12]
+ (-2.1726380430164708e-06) [X2 Y3 Y11 X12]
+ (-1.878149593697101e-06) [Z4 Y10 Z11 Y12]
+ (-1.878149593697101e-06) [Z4 X10 Z11 X12]
+ (-1.878149593697101e-06) [Z5 Y11 Z12 Y13]
+ (-1.878149593697101e-06) [Z5 X11 Z12 X13]
+ (-1.6021751094425738e-06) [Z2 Y3 Z4 Y5]
+ (-1.6021751094425738e-06) [Z2 X3 Z4 X5]
+ (-1.1094125416898342e-06) [Z2 Y11 Z12 Y13]
+ (-1.1094125416898342e-06) [Z2 X11 Z12 X13]
+ (-1.1094125416898342e-06) [Z3 Y10 Z11 Y12]
+ (-1.1094125416898342e-06) [Z3 X10 Z11 X12]
+ (-6.628427357492552e-07) [Y8 X9 X11 Y12]
+ (-6.628427357492552e-07) [Y8 Y9 Y11 Y12]
+ (-6.628427357492552e-07) [X8 X9 X11 X12]
+ (-6.628427357492552e-07) [X8 Y9 Y11 X12]
+ (-5.627722093485102e-07) [Y0 X1 X11 Y12]
+ (-5.627722093485102e-07) [Y0 Y1 Y11 Y12]
+ (-5.627722093485102e-07) [X0 X1 X11 X12]
+ (-5.627722093485102e-07) [X0 Y1 Y11 X12]
+ (-5.471606100974703e-07) [Y1 X2 X11 Y12]
+ (-5.471606100974703e-07) [X1 Y2 Y11 X12]
+ (-3.606069291270702e-07) [Y0 Z1 Z2 Y4]
+ (-3.606069291270702e-07) [X0 Z1 Z2 X4]
+ (-3.606069291270702e-07) [Y1 Z3 Z4 Y5]
+ (-3.606069291270702e-07) [X1 Z3 Z4 X5]
+ (-2.5941461374281907e-07) [Y2 Z3 Y4 Z6]
+ (-2.5941461374281907e-07) [X2 Z3 X4 Z6]
+ (-2.5941461374281907e-07) [Y3 Z4 Y5 Z7]
+ (-2.5941461374281907e-07) [X3 Z4 X5 Z7]
+ (-2.187230390585744e-07) [Y2 Z3 Y4 Z8]
+ (-2.187230390585744e-07) [X2 Z3 X4 Z8]
+ (-2.187230390585744e-07) [Y3 Z4 Y5 Z9]
+ (-2.187230390585744e-07) [X3 Z4 X5 Z9]
+ (-1.9332121410179726e-07) [Y1 Y2 X3 X4]
+ (-1.9332121410179726e-07) [X1 X2 Y3 Y4]
+ (-1.672857150258913e-07) [Y0 Z1 Z3 Y4]
+ (-1.672857150258913e-07) [X0 Z1 Z3 X4]
+ (-1.672857150258913e-07) [Y1 Z2 Z4 Y5]
+ (-1.672857150258913e-07) [X1 Z2 Z4 X5]
+ (-3.226105728792616e-09) [Y1 Y2 X5 X6]
+ (-3.226105728792616e-09) [X1 X2 Y5 Y6]
+ (3.226105728792616e-09) [Y1 X2 X5 Y6]
+ (3.226105728792616e-09) [X1 Y2 Y5 X6]
+ (3.228403458888779e-08) [Y4 Z5 Y6 Z12]
+ (3.228403458888779e-08) [X4 Z5 X6 Z12]
+ (3.228403458888779e-08) [Y5 Z6 Y7 Z13]
+ (3.228403458888779e-08) [X5 Z6 X7 Z13]
+ (1.2919458314441224e-07) [Y1 Z2 Z3 Y5]
+ (1.2919458314441224e-07) [X1 Z2 Z3 X5]
+ (1.9332121410179726e-07) [Y1 X2 X3 Y4]
+ (1.9332121410179726e-07) [X1 Y2 Y3 X4]
+ (2.1989637872781432e-07) [Y2 X3 X5 Y6]
+ (2.1989637872781432e-07) [Y2 Y3 Y5 Y6]
+ (2.1989637872781432e-07) [X2 X3 X5 X6]
+ (2.1989637872781432e-07) [X2 Y3 Y5 X6]
+ (2.447264822426652e-07) [Y0 X1 X5 Y6]
+ (2.447264822426652e-07) [Y0 Y1 Y5 Y6]
+ (2.447264822426652e-07) [X0 X1 X5 X6]
+ (2.447264822426652e-07) [X0 Y1 Y5 X6]
+ (3.5706355246819515e-07) [Y0 X1 X3 Y4]
+ (3.5706355246819515e-07) [Y0 Y1 Y3 Y4]
+ (3.5706355246819515e-07) [X0 X1 X3 X4]
+ (3.5706355246819515e-07) [X0 Y1 Y3 X4]
+ (4.837953357572026e-07) [Y5 X6 X8 Y9]
+ (4.837953357572026e-07) [Y5 Y6 Y8 Y9]
+ (4.837953357572026e-07) [X5 X6 X8 X9]
+ (4.837953357572026e-07) [X5 Y6 Y8 X9]
+ (5.471606100974703e-07) [Y1 Y2 X11 X12]
+ (5.471606100974703e-07) [X1 X2 Y11 Y12]
+ (5.769436626293367e-07) [Y2 Z3 Y4 Z9]
+ (5.769436626293367e-07) [X2 Z3 X4 Z9]
+ (5.769436626293367e-07) [Y3 Z4 Y5 Z8]
+ (5.769436626293367e-07) [X3 Z4 X5 Z8]
+ (5.929280653392023e-07) [Z4 Y5 Z6 Y7]
+ (5.929280653392023e-07) [Z4 X5 Z6 X7]
+ (7.765082203324298e-07) [Y2 Z3 Y4 Z5]
+ (7.765082203324298e-07) [X2 Z3 X4 Z5]
+ (7.956667016879111e-07) [Y3 X4 X8 Y9]
+ (7.956667016879111e-07) [Y3 Y4 Y8 Y9]
+ (7.956667016879111e-07) [X3 X4 X8 X9]
+ (7.956667016879111e-07) [X3 Y4 Y8 X9]
+ (8.336695678945763e-07) [Z0 Y2 Z3 Y4]
+ (8.336695678945763e-07) [Z0 X2 Z3 X4]
+ (8.336695678945763e-07) [Z1 Y3 Z4 Y5]
+ (8.336695678945763e-07) [Z1 X3 Z4 X5]
+ (9.344969748373289e-07) [Z8 Y11 Z12 Y13]
+ (9.344969748373289e-07) [Z8 X11 Z12 X13]
+ (9.344969748373289e-07) [Z9 Y10 Z11 Y12]
+ (9.344969748373289e-07) [Z9 X10 Z11 X12]
+ (9.509134546629256e-07) [Z2 Y4 Z5 Y6]
+ (9.509134546629256e-07) [Z2 X4 Z5 X6]
+ (9.509134546629256e-07) [Z3 Y5 Z6 Y7]
+ (9.509134546629256e-07) [Z3 X5 Z6 X7]
+ (1.0357924798551008e-06) [Y6 X7 X11 Y12]
+ (1.0357924798551008e-06) [Y6 Y7 Y11 Y12]
+ (1.0357924798551008e-06) [X6 X7 X11 X12]
+ (1.0357924798551008e-06) [X6 Y7 Y11 X12]
+ (1.0632255013264197e-06) [Z2 Y10 Z11 Y12]
+ (1.0632255013264197e-06) [Z2 X10 Z11 X12]
+ (1.0632255013264197e-06) [Z3 Y11 Z12 Y13]
+ (1.0632255013264197e-06) [Z3 X11 Z12 X13]
+ (1.1708098333900894e-06) [Z2 Y5 Z6 Y7]
+ (1.1708098333900894e-06) [Z2 X5 Z6 X7]
+ (1.1708098333900894e-06) [Z3 Y4 Z5 Y6]
+ (1.1708098333900894e-06) [Z3 X4 Z5 X6]
+ (1.1907331203628178e-06) [Z0 Y3 Z4 Y5]
+ (1.1907331203628178e-06) [Z0 X3 Z4 X5]
+ (1.1907331203628178e-06) [Z1 Y2 Z3 Y4]
+ (1.1907331203628178e-06) [Z1 X2 Z3 X4]
+ (1.195392078180528e-06) [Y2 Z3 Y4 Z7]
+ (1.195392078180528e-06) [X2 Z3 X4 Z7]
+ (1.195392078180528e-06) [Y3 Z4 Y5 Z6]
+ (1.195392078180528e-06) [X3 Z4 X5 Z6]
+ (1.3980242990681041e-06) [Y4 Z5 Y6 Z8]
+ (1.3980242990681041e-06) [X4 Z5 X6 Z8]
+ (1.3980242990681041e-06) [Y5 Z6 Y7 Z9]
+ (1.3980242990681041e-06) [X5 Z6 X7 Z9]
+ (1.454806691923347e-06) [Y3 X4 X6 Y7]
+ (1.454806691923347e-06) [Y3 Y4 Y6 Y7]
+ (1.454806691923347e-06) [X3 X4 X6 X7]
+ (1.454806691923347e-06) [X3 Y4 Y6 X7]
+ (1.597339710586584e-06) [Z8 Y10 Z11 Y12]
+ (1.597339710586584e-06) [Z8 X10 Z11 X12]
+ (1.597339710586584e-06) [Z9 Y11 Z12 Y13]
+ (1.597339710586584e-06) [Z9 X11 Z12 X13]
+ (1.614960770170498e-06) [Z0 Y11 Z12 Y13]
+ (1.614960770170498e-06) [Z0 X11 Z12 X13]
+ (1.614960770170498e-06) [Z1 Y10 Z11 Y12]
+ (1.614960770170498e-06) [Z1 X10 Z11 X12]
+ (1.6923648547919647e-06) [Y4 Z5 Y6 Z10]
+ (1.6923648547919647e-06) [X4 Z5 X6 Z10]
+ (1.6923648547919647e-06) [Y5 Z6 Y7 Z11]
+ (1.6923648547919647e-06) [X5 Z6 X7 Z11]
+ (1.8163673402103037e-06) [Z4 Y11 Z12 Y13]
+ (1.8163673402103037e-06) [Z4 X11 Z12 X13]
+ (1.8163673402103037e-06) [Z5 Y10 Z11 Y12]
+ (1.8163673402103037e-06) [Z5 X10 Z11 X12]
+ (1.8540565467409491e-06) [Y4 Z5 Y6 Z7]
+ (1.8540565467409491e-06) [X4 Z5 X6 Z7]
+ (1.8551374635243298e-06) [Z6 Y10 Z11 Y12]
+ (1.8551374635243298e-06) [Z6 X10 Z11 X12]
+ (1.8551374635243298e-06) [Z7 Y11 Z12 Y13]
+ (1.8551374635243298e-06) [Z7 X11 Z12 X13]
+ (1.8818196348253068e-06) [Y4 Z5 Y6 Z9]
+ (1.8818196348253068e-06) [X4 Z5 X6 Z9]
+ (1.8818196348253068e-06) [Y5 Z6 Y7 Z8]
+ (1.8818196348253068e-06) [X5 Z6 X7 Z8]
+ (2.1777329795188532e-06) [Z0 Y10 Z11 Y12]
+ (2.1777329795188532e-06) [Z0 X10 Z11 X12]
+ (2.1777329795188532e-06) [Z1 Y11 Z12 Y13]
+ (2.1777329795188532e-06) [Z1 X11 Z12 X13]
+ (2.8909299433798642e-06) [Z6 Y11 Z12 Y13]
+ (2.8909299433798642e-06) [Z6 X11 Z12 X13]
+ (2.8909299433798642e-06) [Z7 Y10 Z11 Y12]
+ (2.8909299433798642e-06) [Z7 X10 Z11 X12]
+ (3.099296651701755e-06) [Z0 Y4 Z5 Y6]
+ (3.099296651701755e-06) [Z0 X4 Z5 X6]
+ (3.099296651701755e-06) [Z1 Y5 Z6 Y7]
+ (3.099296651701755e-06) [Z1 X5 Z6 X7]
+ (3.1585594097854885e-06) [Y2 Z3 Y4 Z10]
+ (3.1585594097854885e-06) [X2 Z3 X4 Z10]
+ (3.1585594097854885e-06) [Y3 Z4 Y5 Z11]
+ (3.1585594097854885e-06) [X3 Z4 X5 Z11]
+ (3.344023133944276e-06) [Z0 Y5 Z6 Y7]
+ (3.344023133944276e-06) [Z0 X5 Z6 X7]
+ (3.344023133944276e-06) [Z1 Y4 Z5 Y6]
+ (3.344023133944276e-06) [Z1 X4 Z5 X6]
+ (3.6945169339065373e-06) [Y4 X5 X11 Y12]
+ (3.6945169339065373e-06) [Y4 Y5 Y11 Y12]
+ (3.6945169339065373e-06) [X4 X5 X11 X12]
+ (3.6945169339065373e-06) [X4 Y5 Y11 X12]
+ (4.556473827556362e-06) [Y5 X6 X12 Y13]
+ (4.556473827556362e-06) [Y5 Y6 Y12 Y13]
+ (4.556473827556362e-06) [X5 X6 X12 X13]
+ (4.556473827556362e-06) [X5 Y6 Y12 X13]
+ (4.5887578621443825e-06) [Y4 Z5 Y6 Z13]
+ (4.5887578621443825e-06) [X4 Z5 X6 Z13]
+ (4.5887578621443825e-06) [Y5 Z6 Y7 Z12]
+ (4.5887578621443825e-06) [X5 Z6 X7 Z12]
+ (4.642979033031959e-06) [Y3 X4 X10 Y11]
+ (4.642979033031959e-06) [Y3 Y4 Y10 Y11]
+ (4.642979033031959e-06) [X3 X4 X10 X11]
+ (4.642979033031959e-06) [X3 Y4 Y10 X11]
+ (7.801538442816364e-06) [Y2 Z3 Y4 Z11]
+ (7.801538442816364e-06) [X2 Z3 X4 Z11]
+ (7.801538442816364e-06) [Y3 Z4 Y5 Z10]
+ (7.801538442816364e-06) [X3 Z4 X5 Z10]
+ (8.194105025175619e-06) [Z10 Y11 Z12 Y13]
+ (8.194105025175619e-06) [Z10 X11 Z12 X13]
+ (0.00029222567210038025) [Y7 Y8 X9 X10]
+ (0.00029222567210038025) [X7 X8 Y9 Y10]
+ (0.0004957972946627267) [Y2 Z4 Z5 Y6]
+ (0.0004957972946627267) [X2 Z4 Z5 X6]
+ (0.0011058984792153333) [Y0 Z1 Y2 Z5]
+ (0.0011058984792153333) [X0 Z1 X2 Z5]
+ (0.0011058984792153333) [Y1 Z2 Y3 Z4]
+ (0.0011058984792153333) [X1 Z2 X3 Z4]
+ (0.0016639606587487228) [Y2 Z3 Z4 Y6]
+ (0.0016639606587487228) [X2 Z3 Z4 X6]
+ (0.0016639606587487228) [Y3 Z5 Z6 Y7]
+ (0.0016639606587487228) [X3 Z5 Z6 X7]
+ (0.0017560659607343775) [Y0 Z1 Y2 Z11]
+ (0.0017560659607343775) [X0 Z1 X2 Z11]
+ (0.0017560659607343775) [Y1 Z2 Y3 Z10]
+ (0.0017560659607343775) [X1 Z2 X3 Z10]
+ (0.0023262348456269868) [Y0 Z1 Y2 Z13]
+ (0.0023262348456269868) [X0 Z1 X2 Z13]
+ (0.0023262348456269868) [Y1 Z2 Y3 Z12]
+ (0.0023262348456269868) [X1 Z2 X3 Z12]
+ (0.002745827315773686) [Y0 X1 X4 Y5]
+ (0.002745827315773686) [X0 Y1 Y4 X5]
+ (0.002929768276302397) [Y0 Z1 Y2 Z9]
+ (0.002929768276302397) [X0 Z1 X2 Z9]
+ (0.002929768276302397) [Y1 Z2 Y3 Z8]
+ (0.002929768276302397) [X1 Z2 X3 Z8]
+ (0.003276965063200823) [Y0 Z1 Y2 Z3]
+ (0.003276965063200823) [X0 Z1 X2 Z3]
+ (0.0033476264688852026) [Y0 Z1 Y2 Z7]
+ (0.0033476264688852026) [X0 Z1 X2 Z7]
+ (0.0033476264688852026) [Y1 Z2 Y3 Z6]
+ (0.0033476264688852026) [X1 Z2 X3 Z6]
+ (0.0035552589692654695) [Y0 Z1 Y2 Z10]
+ (0.0035552589692654695) [X0 Z1 X2 Z10]
+ (0.0035552589692654695) [Y1 Z2 Y3 Z11]
+ (0.0035552589692654695) [X1 Z2 X3 Z11]
+ (0.005143382384883107) [Y3 Y4 X5 X6]
+ (0.005143382384883107) [X3 X4 Y5 Y6]
+ (0.0052837857533872) [Y0 X1 X12 Y13]
+ (0.0052837857533872) [X0 Y1 Y12 X13]
+ (0.005530742148273209) [Y0 Z1 Y2 Z4]
+ (0.005530742148273209) [X0 Z1 X2 Z4]
+ (0.005530742148273209) [Y1 Z2 Y3 Z5]
+ (0.005530742148273209) [X1 Z2 X3 Z5]
+ (0.006087836803984877) [Y8 X9 X12 Y13]
+ (0.006087836803984877) [X8 Y9 Y12 X13]
+ (0.006509361234144078) [Y0 X1 X8 Y9]
+ (0.006509361234144078) [X0 Y1 Y8 X9]
+ (0.006888204542324161) [Y0 X1 X6 Y7]
+ (0.006888204542324161) [X0 Y1 Y6 X7]
+ (0.006901250034146687) [Y0 Z1 Y2 Z12]
+ (0.006901250034146687) [X0 Z1 X2 Z12]
+ (0.006901250034146687) [Y1 Z2 Y3 Z13]
+ (0.006901250034146687) [X1 Z2 X3 Z13]
+ (0.007156920529848183) [Y4 X5 X8 Y9]
+ (0.007156920529848183) [X4 Y5 Y8 X9]
+ (0.007731432846030907) [Y0 X1 X10 Y11]
+ (0.007731432846030907) [X0 Y1 Y10 X11]
+ (0.008032546696261468) [Y0 Z1 Y2 Z6]
+ (0.008032546696261468) [X0 Z1 X2 Z6]
+ (0.008032546696261468) [Y1 Z2 Y3 Z7]
+ (0.008032546696261468) [X1 Z2 X3 Z7]
+ (0.009560716495346785) [Y8 X9 X10 Y11]
+ (0.009560716495346785) [X8 Y9 Y10 X11]
+ (0.011055016686700763) [Y0 Z1 Y2 Z8]
+ (0.011055016686700763) [X0 Z1 X2 Z8]
+ (0.011055016686700763) [Y1 Z2 Y3 Z9]
+ (0.011055016686700763) [X1 Z2 X3 Z9]
+ (0.011285144615357718) [Y5 Y6 X11 X12]
+ (0.011285144615357718) [X5 X6 Y11 Y12]
+ (0.011982342581838742) [Y4 X5 X6 Y7]
+ (0.011982342581838742) [X4 Y5 Y6 X7]
+ (0.013873400068328323) [Y6 X7 X8 Y9]
+ (0.013873400068328323) [X6 Y7 Y8 X9]
+ (0.014583638325597528) [Y0 X1 X2 Y3]
+ (0.014583638325597528) [X0 Y1 Y2 X3]
+ (0.01557722707365107) [Y2 X3 X12 Y13]
+ (0.01557722707365107) [X2 Y3 Y12 X13]
+ (0.017366053261663555) [Y6 X7 X12 Y13]
+ (0.017366053261663555) [X6 Y7 Y12 X13]
+ (0.01768013765339401) [Y4 X5 X10 Y11]
+ (0.01768013765339401) [X4 Y5 Y10 X11]
+ (0.017825011931491108) [Y6 X7 X10 Y11]
+ (0.017825011931491108) [X6 Y7 Y10 X11]
+ (0.019028318717598386) [Y3 X4 X11 Y12]
+ (0.019028318717598386) [X3 Y4 Y11 X12]
+ (0.025384663693951686) [Y2 X3 X10 Y11]
+ (0.025384663693951686) [X2 Y3 Y10 X11]
+ (0.02868521730493718) [Y10 X11 X12 Y13]
+ (0.02868521730493718) [X10 Y11 Y12 X13]
+ (0.031143804193814368) [Y2 X3 X6 Y7]
+ (0.031143804193814368) [X2 Y3 Y6 X7]
+ (0.03583955718467718) [Y2 X3 X4 Y5]
+ (0.03583955718467718) [X2 Y3 Y4 X5]
+ (0.03619409348652447) [Y2 X3 X8 Y9]
+ (0.03619409348652447) [X2 Y3 Y8 X9]
+ (0.03831466937266708) [Y4 X5 X12 Y13]
+ (0.03831466937266708) [X4 Y5 Y12 X13]
+ (0.10433061483969047) [Z0 Y1 Z2 Y3]
+ (0.10433061483969047) [Z0 X1 Z2 X3]
+ (-0.2284794631512675) [Y6 Z7 Z8 Z9 Y10]
+ (-0.2284794631512675) [X6 Z7 Z8 Z9 X10]
+ (-0.2284794631512674) [Y7 Z8 Z9 Z10 Y11]
+ (-0.2284794631512674) [X7 Z8 Z9 Z10 X11]
+ (-0.12133242245214228) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133242245214228) [X2 Z3 Z4 Z5 X6]
+ (-0.12133242245214226) [Y3 Z4 Z5 Z6 Y7]
+ (-0.12133242245214226) [X3 Z4 Z5 Z6 X7]
+ (-3.204142251791999e-06) [Y0 Z1 Z2 Z3 Y4]
+ (-3.204142251791999e-06) [X0 Z1 Z2 Z3 X4]
+ (-3.204142251791999e-06) [Y1 Z2 Z3 Z4 Y5]
+ (-3.204142251791999e-06) [X1 Z2 Z3 Z4 X5]
+ (-0.05608449432707725) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (-0.05608449432707725) [Z0 X6 Z7 Z8 Z9 X10]
+ (-0.05608449432707725) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (-0.05608449432707725) [Z1 X7 Z8 Z9 Z10 X11]
+ (-0.05600713562149556) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (-0.05600713562149556) [Z0 X7 Z8 Z9 Z10 X11]
+ (-0.05600713562149556) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (-0.05600713562149556) [Z1 X6 Z7 Z8 Z9 X10]
+ (-0.032767485885956765) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.032767485885956765) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.032767485885956765) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.032767485885956765) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.0307874407269806) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (-0.0307874407269806) [Z6 X7 Z8 Z9 Z10 X11]
+ (-0.027114878571570026) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.027114878571570026) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.027114878571570026) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.027114878571570026) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.02599620626470892) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.02599620626470892) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.025104907973022787) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (-0.025104907973022787) [X6 Z7 Z8 Z9 X10 Z12]
+ (-0.025104907973022787) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (-0.025104907973022787) [X7 Z8 Z9 Z10 X11 Z13]
+ (-0.024388989992246812) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (-0.024388989992246812) [Z2 X7 Z8 Z9 Z10 X11]
+ (-0.024388989992246812) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (-0.024388989992246812) [Z3 X6 Z7 Z8 Z9 X10]
+ (-0.019020373880017562) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (-0.019020373880017562) [Z2 X6 Z7 Z8 Z9 X10]
+ (-0.019020373880017562) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (-0.019020373880017562) [Z3 X7 Z8 Z9 Z10 X11]
+ (-0.018266758584781426) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (-0.018266758584781426) [Z4 X6 Z7 Z8 Z9 X10]
+ (-0.018266758584781426) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (-0.018266758584781426) [Z5 X7 Z8 Z9 Z10 X11]
+ (-0.01756111644839479) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.01756111644839479) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.01756111644839479) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.01756111644839479) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.012214985319199173) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.012214985319199173) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.012214985319199173) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.012214985319199173) [X4 Z5 X6 X11 Z12 X13]
+ (-0.012214985319199173) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.012214985319199173) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.012214985319199173) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.012214985319199173) [X5 Z6 X7 X10 Z11 X12]
+ (-0.011755995239956712) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.011755995239956712) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.011755995239956712) [X3 Z4 Z5 X6 X8 X9]
+ (-0.011755995239956712) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.01130720803595055) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (-0.01130720803595055) [X6 Z7 Z8 Z9 X10 Z11]
+ (-0.010959994618254724) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (-0.010959994618254724) [Z4 X7 Z8 Z9 Z10 X11]
+ (-0.010959994618254724) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (-0.010959994618254724) [Z5 X6 Z7 Z8 Z9 X10]
+ (-0.010540434336103226) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (-0.010540434336103226) [X6 Z7 Z8 Z9 X10 Z13]
+ (-0.010540434336103226) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (-0.010540434336103226) [X7 Z8 Z9 Z10 X11 Z12]
+ (-0.00876485821836479) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.00876485821836479) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.00876485821836479) [X2 Z3 Z4 X5 X11 X12]
+ (-0.00876485821836479) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.00876485821836479) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.00876485821836479) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.00876485821836479) [X3 X4 X10 Z11 Z12 X13]
+ (-0.00876485821836479) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.008125248410398366) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.008125248410398366) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.005805121208438083) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.005805121208438083) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.005805121208438083) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.005805121208438083) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.005652607314386735) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.005652607314386735) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.005652607314386735) [X0 X1 X3 Z4 Z5 X6]
+ (-0.005652607314386735) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.005368616112229252) [Y2 X3 X7 Z8 Z9 Y10]
+ (-0.005368616112229252) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (-0.005368616112229252) [X2 X3 X7 Z8 Z9 X10]
+ (-0.005368616112229252) [X2 Y3 Y7 Z8 Z9 X10]
+ (-0.0051433823848831065) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.0051433823848831065) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.0051433823848831065) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.0051433823848831065) [X2 Z3 X4 X5 Z6 X7]
+ (-0.004684920227376266) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.004684920227376266) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668615265776027) [Y1 Y2 X7 Z8 Z9 X10]
+ (-0.004668615265776027) [X1 X2 Y7 Z8 Z9 Y10]
+ (-0.0045750151885197) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.0045750151885197) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.004424843669057875) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.004424843669057875) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.004158830716220912) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.004158830716220912) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.004158830716220912) [X3 Z4 Z5 X6 X12 X13]
+ (-0.004158830716220912) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.00349380036881937) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.00349380036881937) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.00349380036881937) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.00349380036881937) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.0027790407641030204) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.0027790407641030204) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.0022939556229147383) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.0022939556229147383) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.0017991930085310929) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.0017991930085310929) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.0017278745819650529) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.0017278745819650529) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.0009298407038414555) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298407038414555) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298407038414555) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298407038414555) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533831053501369) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.0008533831053501369) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.0008144692856159059) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.0008144692856159059) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.0008144692856159059) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.0008144692856159059) [X3 Z4 Z5 Z6 X7 Z11]
+ (-0.00029222567210038025) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (-0.00029222567210038025) [Y6 Z7 Y8 X9 Z10 X11]
+ (-0.00029222567210038025) [X6 Z7 X8 Y9 Z10 Y11]
+ (-0.00029222567210038025) [X6 Z7 X8 X9 Z10 X11]
+ (-8.774724389044863e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.774724389044863e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.774724389044863e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.774724389044863e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.518288916259601e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.518288916259601e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.518288916259601e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.518288916259601e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.444267520883706e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.444267520883706e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.444267520883706e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.444267520883706e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.524204573938458e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.524204573938458e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.524204573938458e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.524204573938458e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-6.290019794303291e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.290019794303291e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.290019794303291e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.290019794303291e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.974176940837152e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (-5.974176940837152e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (-5.275783513437838e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (-5.275783513437838e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (-4.642979033031959e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (-4.642979033031959e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (-4.556473827556362e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (-4.556473827556362e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (-4.253118822328782e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.253118822328782e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-3.7695836427468e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.7695836427468e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.6945169339065373e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (-3.6945169339065373e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (-3.6102422857248228e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.6102422857248228e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.6102422857248228e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.6102422857248228e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.313017108304804e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.313017108304804e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.2774383158231876e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.2774383158231876e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.2774383158231876e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.2774383158231876e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.211187465631052e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.211187465631052e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.211187465631052e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.211187465631052e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.151296001765156e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.151296001765156e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.117366415691129e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (-3.117366415691129e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (-3.0882457540241676e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (-3.0882457540241676e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (-2.1726380430164708e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (-2.1726380430164708e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (-1.454806691923347e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (-1.454806691923347e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (-1.3304568681611574e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.3304568681611574e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.239311400016185e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (-1.239311400016185e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (-1.239311400016185e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (-1.239311400016185e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (-1.2282691219558765e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.2282691219558765e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.0357924798551008e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (-1.0357924798551008e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (-9.306343071069541e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (-9.306343071069541e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (-9.306343071069541e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (-9.306343071069541e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (-7.956667016879111e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (-7.956667016879111e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (-6.628427357492552e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (-6.628427357492552e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (-5.627722093485102e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (-5.627722093485102e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (-4.837953357572026e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (-4.837953357572026e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (-3.5706355246819515e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (-3.5706355246819515e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (-3.3280396990163514e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.3280396990163514e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.236183438470862e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (-3.236183438470862e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (-3.236183438470862e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (-3.236183438470862e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (-2.447264822426652e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (-2.447264822426652e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (-2.1989637872781432e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (-2.1989637872781432e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (-1.8290428667914177e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (-1.8290428667914177e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (-1.8290428667914177e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (-1.8290428667914177e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (-8.64912962926013e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (-8.64912962926013e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (-8.64912962926013e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (-8.64912962926013e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (-8.05746540853431e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (-8.05746540853431e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (-8.05746540853431e-08) [X1 Z2 Z3 X4 X10 X11]
+ (-8.05746540853431e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (-1.839565788423159e-08) [Y0 Z1 X2 X10 Z11 Y12]
+ (-1.839565788423159e-08) [X0 Z1 Y2 Y10 Z11 X12]
+ (-1.839565788423159e-08) [Y1 Z2 X3 X11 Z12 Y13]
+ (-1.839565788423159e-08) [X1 Z2 Y3 Y11 Z12 X13]
+ (1.0351498237751102e-09) [Y0 Z1 Z2 Z3 Y4 Z7]
+ (1.0351498237751102e-09) [X0 Z1 Z2 Z3 X4 Z7]
+ (1.0351498237751102e-09) [Y1 Z2 Z3 Z4 Y5 Z6]
+ (1.0351498237751102e-09) [X1 Z2 Z3 Z4 X5 Z6]
+ (2.2702082707835305e-08) [Y0 Z1 Z2 X3 X5 Y6]
+ (2.2702082707835305e-08) [Y0 Z1 Z2 Y3 Y5 Y6]
+ (2.2702082707835305e-08) [X0 Z1 Z2 X3 X5 X6]
+ (2.2702082707835305e-08) [X0 Z1 Z2 Y3 Y5 X6]
+ (2.2702082707835305e-08) [Y1 X2 X4 Z5 Z6 Y7]
+ (2.2702082707835305e-08) [Y1 Y2 Y4 Z5 Z6 Y7]
+ (2.2702082707835305e-08) [X1 X2 X4 Z5 Z6 X7]
+ (2.2702082707835305e-08) [X1 Y2 Y4 Z5 Z6 X7]
+ (2.5928188436707542e-08) [Y0 Z1 X2 X4 Z5 Y6]
+ (2.5928188436707542e-08) [X0 Z1 Y2 Y4 Z5 X6]
+ (2.5928188436707542e-08) [Y1 Z2 X3 X5 Z6 Y7]
+ (2.5928188436707542e-08) [X1 Z2 Y3 Y5 Z6 X7]
+ (1.1076529214184303e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (1.1076529214184303e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (1.1076529214184303e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (1.1076529214184303e-07) [X0 Z1 X2 X11 Z12 X13]
+ (1.1076529214184303e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (1.1076529214184303e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (1.1076529214184303e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (1.1076529214184303e-07) [X1 Z2 X3 X10 Z11 X12]
+ (1.2919458314441224e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (1.2919458314441224e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (1.3484968877883303e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (1.3484968877883303e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (1.3484968877883303e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (1.3484968877883303e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (1.3807579450767308e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (1.3807579450767308e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (1.3807579450767308e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (1.3807579450767308e-07) [X0 Z1 X2 X5 Z6 X7]
+ (1.3807579450767308e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (1.3807579450767308e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (1.3807579450767308e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (1.3807579450767308e-07) [X1 Z2 X3 X4 Z5 X6]
+ (1.607778772156439e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (1.607778772156439e-07) [X0 Z1 X2 X4 Z5 X6]
+ (1.607778772156439e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (1.607778772156439e-07) [X1 Z2 X3 X5 Z6 X7]
+ (1.83939436502822e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (1.83939436502822e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (1.83939436502822e-07) [X1 Z2 Z3 X4 X6 X7]
+ (1.83939436502822e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (1.9332121410179726e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (1.9332121410179726e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (1.9332121410179726e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (1.9332121410179726e-07) [X0 Z1 X2 X3 Z4 X5]
+ (2.1989637872781432e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (2.1989637872781432e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (2.371270475544849e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (2.371270475544849e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (2.371270475544849e-07) [X1 Z2 Z3 X4 X8 X9]
+ (2.371270475544849e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (2.447264822426652e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (2.447264822426652e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (3.0867709290856695e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (3.0867709290856695e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (3.0867709290856695e-07) [X1 Z2 Z3 X4 X12 X13]
+ (3.0867709290856695e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (3.3280396990163514e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.3280396990163514e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.5706355246819515e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (3.5706355246819515e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (4.837953357572026e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (4.837953357572026e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (5.287649522136046e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (5.287649522136046e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (5.287649522136046e-07) [X0 Z1 Z2 X3 X11 X12]
+ (5.287649522136046e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (5.287649522136046e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (5.287649522136046e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (5.287649522136046e-07) [X1 X2 X10 Z11 Z12 X13]
+ (5.287649522136046e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (5.627722093485102e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (5.627722093485102e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (5.927350306817163e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (5.927350306817163e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (5.927350306817163e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (5.927350306817163e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (6.395302443551427e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (6.395302443551427e-07) [X0 Z1 X2 X10 Z11 X12]
+ (6.395302443551427e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (6.395302443551427e-07) [X1 Z2 X3 X11 Z12 X13]
+ (6.57925902240018e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (6.57925902240018e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (6.57925902240018e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (6.57925902240018e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (6.628427357492552e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (6.628427357492552e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (6.733096847665986e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (6.733096847665986e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (6.733096847665986e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (6.733096847665986e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (7.956667016879111e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (7.956667016879111e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (1.0357924798551008e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (1.0357924798551008e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (1.2282691219558765e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.2282691219558765e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.3304568681611574e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.3304568681611574e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.454806691923347e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (1.454806691923347e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (2.1726380430164708e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (2.1726380430164708e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (3.0882457540241676e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (3.0882457540241676e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (3.151296001765156e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.151296001765156e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.313017108304804e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.313017108304804e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.3342618847263805e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.3342618847263805e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.6945169339065373e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (3.6945169339065373e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (4.183808884344539e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.183808884344539e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.556473827556362e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (4.556473827556362e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (4.642979033031959e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (4.642979033031959e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (5.275783513437838e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (5.275783513437838e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (5.974176940837152e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (5.974176940837152e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (7.735870558168802e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (7.735870558168802e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (7.735870558168802e-05) [X0 X1 X7 Z8 Z9 X10]
+ (7.735870558168802e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (0.0004957972946627267) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.0004957972946627267) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.000665030347401546) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.000665030347401546) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.000665030347401546) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.000665030347401546) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.0008533831053501369) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.0008533831053501369) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.0016095335156705328) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.0016095335156705328) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.0016095335156705328) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.0016095335156705328) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.0016676137497706857) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.0016676137497706857) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.0016676137497706857) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.0016676137497706857) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.0017278745819650529) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.0017278745819650529) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.0017991930085310929) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.0017991930085310929) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.0022939556229147383) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.0022939556229147383) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.0024629166210206697) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.0024629166210206697) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.0024629166210206697) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.0024629166210206697) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.003961569372685424) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.003961569372685424) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.003961569372685424) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.003961569372685424) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.004424843669057875) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.004424843669057875) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.0045750151885197) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.0045750151885197) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.004668615265776027) [Y1 X2 X7 Z8 Z9 Y10]
+ (0.004668615265776027) [X1 Y2 Y7 Z8 Z9 X10]
+ (0.004684920227376266) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.004684920227376266) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.005324817364129981) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.005324817364129981) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.005324817364129981) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.005324817364129981) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.007306763966526703) [Y4 X5 X7 Z8 Z9 Y10]
+ (0.007306763966526703) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (0.007306763966526703) [X4 X5 X7 Z8 Z9 X10]
+ (0.007306763966526703) [X4 Y5 Y7 Z8 Z9 X10]
+ (0.00796083963673095) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.00796083963673095) [X4 Z5 X6 X10 Z11 X12]
+ (0.00796083963673095) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.00796083963673095) [X5 Z6 X7 X11 Z12 X13]
+ (0.008125248410398366) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.008125248410398366) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.008890680340572405) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.008890680340572405) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.008890680340572405) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.008890680340572405) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.0102634604992336) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.0102634604992336) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.0102634604992336) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.0102634604992336) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.014411189770843366) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.014411189770843366) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.014411189770843366) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.014411189770843366) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.01456447363691956) [Y7 Z8 Z9 X10 X12 Y13]
+ (0.01456447363691956) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (0.01456447363691956) [X7 Z8 Z9 X10 X12 X13]
+ (0.01456447363691956) [X7 Z8 Z9 Y10 Y12 X13]
+ (0.015225659056459274) [Y3 Z4 Z5 X6 X10 Y11]
+ (0.015225659056459274) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (0.015225659056459274) [X3 Z4 Z5 X6 X10 X11]
+ (0.015225659056459274) [X3 Z4 Z5 Y6 Y10 X11]
+ (0.015588277863363566) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (0.015588277863363566) [X2 Z3 X4 X10 Z11 X12]
+ (0.015588277863363566) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (0.015588277863363566) [X3 Z4 X5 X11 Z12 X13]
+ (0.020175824955930125) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.020175824955930125) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.020175824955930125) [X4 Z5 Z6 X7 X11 X12]
+ (0.020175824955930125) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.020175824955930125) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.020175824955930125) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.020175824955930125) [X5 X6 X10 Z11 Z12 X13]
+ (0.020175824955930125) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.024353136081728365) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.024353136081728365) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.024353136081728365) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.024353136081728365) [X2 Z3 X4 X11 Z12 X13]
+ (0.024353136081728365) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.024353136081728365) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.024353136081728365) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.024353136081728365) [X3 Z4 X5 X10 Z11 X12]
+ (0.04587942402500817) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.04587942402500817) [X0 Z2 Z3 Z4 Z5 X6]
+ (-6.631262027482713e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.631262027482713e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.631262027482713e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.631262027482713e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.5950813879371718e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.5950813879371718e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.5950813879371714e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.5950813879371714e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.042743260056773105) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.042743260056773105) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (0.04274326005677312) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.04274326005677312) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (-0.019257452998969894) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019257452998969894) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.019028318717598386) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.019028318717598386) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.016024666091731657) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (-0.016024666091731657) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (-0.015225659056459274) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.015225659056459274) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.014603742409704273) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (-0.014603742409704273) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (-0.01456447363691956) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.01456447363691956) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.011755995239956712) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.011755995239956712) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.011285144615357718) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-0.011285144615357718) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-0.009841802921692996) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.009841802921692996) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008469833338938592) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.008469833338938592) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.007306763966526703) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.007306763966526703) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.005923799555839527) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-0.005923799555839527) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-0.005708479853126795) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (-0.005708479853126795) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (-0.005708479853126795) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (-0.005708479853126795) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (-0.005652607314386735) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.005652607314386735) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.005368616112229251) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.005368616112229251) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005262631032840544) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (-0.005262631032840544) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (-0.005262631032840544) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (-0.005262631032840544) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (-0.005114464086062279) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (-0.005114464086062279) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (-0.005114464086062279) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (-0.005114464086062279) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (-0.005114464086062279) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005114464086062279) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005114464086062279) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.005114464086062279) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.004158830716220912) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.004158830716220912) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.003356667921508994) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.003356667921508994) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.003356667921508994) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.003356667921508994) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.0032675148969666017) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.0032675148969666017) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.0032675148969666017) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.0032675148969666017) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.00277904076410302) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.00277904076410302) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.0026860422746892293) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.0026860422746892293) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.0026860422746892293) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.0026860422746892293) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.0022939556229147383) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939556229147383) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.0022939556229147383) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939556229147383) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.0009581676927241776) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.0009581676927241776) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.0009581676927241776) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.0009581676927241776) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.0009581676927241776) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.0009581676927241776) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.0009581676927241776) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.0009581676927241776) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.0005940157670645164) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (-0.0005940157670645164) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (-0.0005940157670645164) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (-0.0005940157670645164) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (-0.0005940157670645164) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (-0.0005940157670645164) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (-0.0005940157670645164) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (-0.0005940157670645164) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (-0.0004458488202862509) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (-0.0004458488202862509) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (-0.0004458488202862509) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (-0.0004458488202862509) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (-0.00024644081366334753) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00024644081366334753) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.0001383860368052346) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.0001383860368052346) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.0001383860368052346) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.0001383860368052346) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.735870558168802e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (-7.735870558168802e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.610337519421159e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.610337519421159e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.610337519421159e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.610337519421159e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316614323959114e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.5316614323959114e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316614323959114e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5316614323959114e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.805982188332285e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.805982188332285e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.805982188332285e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.805982188332285e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.089728737682231e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.089728737682231e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.089728737682231e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.089728737682231e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.65210643654299e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.65210643654299e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.65210643654299e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.65210643654299e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481752090221865e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.481752090221865e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481752090221865e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.481752090221865e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.07140373382127e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.07140373382127e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.07140373382127e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.07140373382127e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.734578454511015e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.734578454511015e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.734578454511015e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.734578454511015e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.7287815005021285e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.7287815005021285e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.7287815005021285e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.7287815005021285e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.253118822328782e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.253118822328782e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.7695836427468e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.7695836427468e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.745510670254525e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (-2.745510670254525e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (-2.745510670254525e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (-2.745510670254525e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (-2.745510670254525e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (-2.745510670254525e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (-2.745510670254525e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (-2.745510670254525e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (-2.360947237181946e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.360947237181946e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.360947237181946e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.360947237181946e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-2.1031634988537685e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.1031634988537685e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.1031634988537685e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.1031634988537685e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0110740494082117e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.0110740494082117e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0110740494082117e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.0110740494082117e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.942946542612522e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.942946542612522e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.942946542612522e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.942946542612522e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.654090079553864e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.654090079553864e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.654090079553864e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.654090079553864e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.5224582109382712e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.5224582109382712e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.5224582109382712e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.5224582109382712e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.5224582109382712e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224582109382712e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224582109382712e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.5224582109382712e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.2282691219558765e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2282691219558765e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.2282691219558765e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2282691219558765e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.8676087025226e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.8676087025226e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.8676087025226e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.8676087025226e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.189870551176575e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.189870551176575e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.175164961676847e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (-6.175164961676847e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (-5.471606100974703e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (-5.471606100974703e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (-4.523339429499919e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.523339429499919e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3280396990163514e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.3280396990163514e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.3280396990163514e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.3280396990163514e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.0867709290856695e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (-3.0867709290856695e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (-2.8885646306039264e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8885646306039264e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8885646306039264e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.8885646306039264e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.371270475544849e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (-2.371270475544849e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (-1.83939436502822e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (-1.83939436502822e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (-8.05746540853431e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (-8.05746540853431e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (-6.772951848841522e-08) [Y1 Z2 Z3 X4 X7 Z8 Z9 Y10]
+ (-6.772951848841522e-08) [X1 Z2 Z3 Y4 Y7 Z8 Z9 X10]
+ (-3.226105728792616e-09) [Y0 Z1 Z2 Y3 X4 Z5 Z6 X7]
+ (-3.226105728792616e-09) [X0 Z1 Z2 X3 Y4 Z5 Z6 Y7]
+ (3.226105728792616e-09) [Y0 Z1 Z2 X3 X4 Z5 Z6 Y7]
+ (3.226105728792616e-09) [X0 Z1 Z2 Y3 Y4 Z5 Z6 X7]
+ (6.04679037237649e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (6.04679037237649e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (6.04679037237649e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (6.04679037237649e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (6.772951848841522e-08) [Y1 Z2 Z3 Y4 X7 Z8 Z9 X10]
+ (6.772951848841522e-08) [X1 Z2 Z3 X4 Y7 Z8 Z9 Y10]
+ (8.05746540853431e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (8.05746540853431e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (9.20894494434968e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.20894494434968e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.20894494434968e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.20894494434968e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.703543463209163e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.703543463209163e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.703543463209163e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.703543463209163e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.83939436502822e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (1.83939436502822e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (2.371270475544849e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (2.371270475544849e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (3.0867709290856695e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (3.0867709290856695e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (3.427350837733288e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (3.427350837733288e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (3.427350837733288e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (3.427350837733288e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (4.523339429499919e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.523339429499919e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (4.561117055317844e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (4.561117055317844e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (4.561117055317844e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (4.561117055317844e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (5.471606100974703e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (5.471606100974703e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (6.175164961676847e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (6.175164961676847e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (7.189870551176575e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.189870551176575e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (7.988467892992585e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (7.988467892992585e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (7.988467892992585e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (7.988467892992585e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (1.3304568681611574e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.3304568681611574e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.3304568681611574e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.3304568681611574e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.6288377908251503e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.6288377908251503e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.6288377908251503e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.6288377908251503e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.6893056945503247e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.6893056945503247e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.6893056945503247e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.6893056945503247e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (3.2117639054871864e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.2117639054871864e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.2117639054871864e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.2117639054871864e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.2117639054871864e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.2117639054871864e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.2117639054871864e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.2117639054871864e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.313017108304804e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.313017108304804e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.313017108304804e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.313017108304804e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (3.3342618847263805e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.3342618847263805e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (3.544357459554434e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (3.544357459554434e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (3.544357459554434e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (3.544357459554434e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (3.544357459554434e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (3.544357459554434e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (3.544357459554434e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (3.544357459554434e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (4.183808884344539e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.183808884344539e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (7.735870558168802e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (7.735870558168802e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.00024644081366334753) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (0.00024644081366334753) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (0.0008533831053501367) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.0008533831053501367) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.0008533831053501367) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.0008533831053501367) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.0010435237096566967) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.0010435237096566967) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.0010435237096566967) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.0010435237096566967) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.0012803055942792632) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.0012803055942792632) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.0012803055942792632) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.0012803055942792632) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.0013038029824944185) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (0.0013038029824944185) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (0.0013038029824944185) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (0.0013038029824944185) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (0.0022619706752185966) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (0.0022619706752185966) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (0.0022619706752185966) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (0.0022619706752185966) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (0.0022619706752185966) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (0.0022619706752185966) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (0.0022619706752185966) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (0.0022619706752185966) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (0.003989845257183649) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (0.003989845257183649) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (0.003989845257183649) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (0.003989845257183649) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (0.004158830716220912) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.004158830716220912) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.004311038606623297) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.004311038606623297) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.004311038606623297) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.004311038606623297) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.004636973515788257) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.004636973515788257) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.004636973515788257) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.004636973515788257) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.0052415435960653615) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (0.0052415435960653615) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (0.0052415435960653615) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (0.0052415435960653615) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (0.005368616112229251) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.005368616112229251) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.005379929632870597) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.005379929632870597) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.005379929632870597) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.005379929632870597) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.005652607314386735) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.005652607314386735) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.005923799555839527) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (0.005923799555839527) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (0.007306763966526703) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (0.007306763966526703) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.008469833338938592) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.008469833338938592) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.009612546721656843) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (0.009612546721656843) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (0.009612546721656843) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (0.009612546721656843) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (0.009841802921692996) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.009841802921692996) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.011285144615357718) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (0.011285144615357718) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (0.011755995239956712) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.011755995239956712) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.01456447363691956) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.01456447363691956) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.014603742409704273) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (0.014603742409704273) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (0.015225659056459274) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.015225659056459274) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.016024666091731657) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (0.016024666091731657) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (0.01888899507732967) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (0.01888899507732967) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (0.01888899507732967) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.01888899507732967) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.019028318717598386) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.019028318717598386) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.019257452998969894) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (0.019257452998969894) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (0.021433980116771913) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.021433980116771913) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.021433980116771913) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.021433980116771913) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.022528354253334996) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.022528354253334996) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.023145221660820288) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (0.023145221660820288) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (0.024282031618361972) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (0.024282031618361972) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (0.02475550797984469) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.02475550797984469) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.02475550797984469) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.02475550797984469) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.025637212813388502) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (0.025637212813388502) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (0.025637212813388502) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (0.025637212813388502) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (0.02873079799902266) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.02873079799902266) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.02873079799902266) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.02873079799902266) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.029903813455710505) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.029903813455710505) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.029903813455710505) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.029903813455710505) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.03560840034934819) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.03560840034934819) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.03931810723858705) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.03931810723858705) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.03931810723858705) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.03931810723858705) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.03935925038954896) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.03935925038954896) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.03935925038954896) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.03935925038954896) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.03956454805225039) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (0.03956454805225039) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (0.03956454805225039) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.03956454805225039) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.041718814052922236) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (0.041718814052922236) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (0.041718814052922236) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (0.041718814052922236) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (0.04587942402500817) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04587942402500817) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (0.047642613608761764) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (0.047642613608761764) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (0.047642613608761764) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (0.047642613608761764) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (0.281643357518165) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.281643357518165) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.281643357518165) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.281643357518165) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.3693713755963207) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.3693713755963207) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.3693713755963208) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.3693713755963208) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0585921517848763) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0585921517848763) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.019299499855675226) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.019299499855675226) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019299499855675226) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.019299499855675226) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019299499855675226) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019299499855675226) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.019299499855675226) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019299499855675226) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.014603742409704273) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (-0.014603742409704273) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (-0.014603742409704273) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (-0.014603742409704273) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (-0.010757524199594552) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010757524199594552) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010757524199594552) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010757524199594552) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.01071547734288922) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.01071547734288922) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.01071547734288922) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.01071547734288922) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005923799555839527) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-0.005923799555839527) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-0.005923799555839527) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-0.005923799555839527) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-0.005408970757089847) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.005408970757089847) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.005408970757089847) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.005408970757089847) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.005286569055744121) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.005286569055744121) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.005286569055744121) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.005286569055744121) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.004767276643738928) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.004767276643738928) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.004767276643738928) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.004767276643738928) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.004668615265776027) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.004668615265776027) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.003876482195629717) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (-0.003876482195629717) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (-0.0034841545794502857) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0034841545794502857) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003356667921508994) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.003356667921508994) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.0032675148969666013) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.0032675148969666013) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0024464634226051968) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.0024464634226051968) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.0024464634226051968) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.0024464634226051968) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.0017278745819650529) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.0017278745819650529) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.001640759115897476) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.001640759115897476) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0015324885614601308) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.0015324885614601308) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.0015324885614601308) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.0015324885614601308) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.0007870893706149089) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0007870893706149089) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.000715673706319644) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.000715673706319644) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.000715673706319644) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.000715673706319644) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0005192924120051922) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.0005192924120051922) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.00024644081366334753) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00024644081366334753) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00024644081366334753) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00024644081366334753) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0001940103063223774) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0001940103063223774) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.00018787485985513827) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.00018787485985513827) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.00018787485985513827) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.00018787485985513827) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0001383860368052346) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.0001383860368052346) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-4.2046856705330385e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-4.2046856705330385e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-4.2046856705330385e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.2046856705330385e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-5.07140373382127e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.07140373382127e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.151296001765156e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.151296001765156e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0882457540241676e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-3.0882457540241676e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-2.988412606533134e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.988412606533134e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8742485959314243e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.8742485959314243e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.360947237181946e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.360947237181946e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.3001959065423764e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.3001959065423764e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.1468226359031131e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.1468226359031131e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.1468226359031131e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.1468226359031131e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.398527785441439e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.398527785441439e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.398527785441439e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.398527785441439e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.091539949493761e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091539949493761e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091539949493761e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.091539949493761e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.091539949493761e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091539949493761e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091539949493761e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.091539949493761e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.027844297371528e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.027844297371528e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.027844297371528e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.027844297371528e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.900025839037526e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.900025839037526e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.900025839037526e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900025839037526e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.8676087025226e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.8676087025226e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.560554039247249e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560554039247249e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560554039247249e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.560554039247249e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.560554039247249e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560554039247249e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560554039247249e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.560554039247249e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.246849507253142e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (-7.246849507253142e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (-7.246849507253142e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (-7.246849507253142e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (-7.246849507253142e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (-7.246849507253142e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (-7.246849507253142e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (-7.246849507253142e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (-3.5682005199894546e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.5682005199894546e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.5682005199894546e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.5682005199894546e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.376686409534422e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.376686409534422e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.376686409534422e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.376686409534422e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.376686409534422e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.376686409534422e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.376686409534422e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.376686409534422e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.8885646306039264e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.8885646306039264e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6863217076261497e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.6863217076261497e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.2498976541285106e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-2.2498976541285106e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-2.2498976541285106e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-2.2498976541285106e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-1.703543463209163e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.703543463209163e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1782131085537407e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-1.1782131085537407e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-1.1782131085537407e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-1.1782131085537407e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-1.0716845455783274e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-1.0716845455783274e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-1.0716845455783274e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-1.0716845455783274e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-9.20894494434968e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.20894494434968e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.37973746193191e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.37973746193191e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.37973746193191e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.37973746193191e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.37973746193191e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.37973746193191e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.37973746193191e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.37973746193191e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.706834880704536e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.706834880704536e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.706834880704536e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.706834880704536e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.204004476683488e-08) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (-3.204004476683488e-08) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (-3.204004476683488e-08) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (-3.204004476683488e-08) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (3.568947372162439e-08) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (3.568947372162439e-08) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (3.568947372162439e-08) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (3.568947372162439e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (3.568947372162439e-08) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.568947372162439e-08) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (3.568947372162439e-08) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (3.568947372162439e-08) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (9.20894494434968e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.20894494434968e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.703543463209163e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.703543463209163e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863217076261497e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863217076261497e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.8885646306039264e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.8885646306039264e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.09216197886683e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.09216197886683e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.09216197886683e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.09216197886683e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.09216197886683e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.09216197886683e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.09216197886683e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.09216197886683e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.4490567160826336e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.4490567160826336e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.4490567160826336e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.4490567160826336e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.769457163754438e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.769457163754438e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.769457163754438e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.769457163754438e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (4.996951853132356e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (4.996951853132356e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (4.996951853132356e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (4.996951853132356e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (4.996951853132356e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (4.996951853132356e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (4.996951853132356e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (4.996951853132356e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (7.8676087025226e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.8676087025226e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.3001959065423764e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.3001959065423764e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.360947237181946e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.360947237181946e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.8742485959314243e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.8742485959314243e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.883653184327338e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.883653184327338e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.9473314763774853e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9473314763774853e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.9473314763774853e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9473314763774853e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.988412606533134e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.988412606533134e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.0882457540241676e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (3.0882457540241676e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (3.151296001765156e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.151296001765156e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.846190908177345e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.846190908177345e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.846190908177345e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.846190908177345e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.07140373382127e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.07140373382127e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.105462464171707e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.105462464171707e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.105462464171707e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.105462464171707e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.146386814724492e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.146386814724492e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.146386814724492e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.146386814724492e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.159294518526235e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.159294518526235e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.159294518526235e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.159294518526235e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.427926689283483e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.427926689283483e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.427926689283483e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.427926689283483e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.9357440829086675e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.9357440829086675e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.9357440829086675e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.9357440829086675e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.2531852221290635e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.2531852221290635e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.979711060103131e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.979711060103131e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.979711060103131e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.979711060103131e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (7.141566429526485e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (7.141566429526485e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (7.141566429526485e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.141566429526485e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0001383860368052346) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.0001383860368052346) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.0001940103063223774) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (0.0001940103063223774) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (0.0005192924120051922) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.0005192924120051922) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.0007870893706149089) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.0007870893706149089) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.0014528842560423392) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0014528842560423392) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0014528842560423392) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0014528842560423392) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.001640759115897476) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.001640759115897476) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0017278745819650529) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.0017278745819650529) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.00214134896533342) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (0.00214134896533342) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (0.0032675148969666013) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.0032675148969666013) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.003356667921508994) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.003356667921508994) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.0034841545794502857) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0034841545794502857) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0038040631544515317) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0038040631544515317) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0038040631544515317) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0038040631544515317) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.003876482195629717) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (0.003876482195629717) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (0.004668615265776027) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.004668615265776027) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.008469833338938592) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.008469833338938592) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.008469833338938592) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.008469833338938592) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.00854197565608067) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00854197565608067) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (0.00854197565608067) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (0.00854197565608067) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00854197565608067) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00854197565608067) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00854197565608067) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00854197565608067) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.008826387566628595) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.008826387566628595) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.008826387566628595) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.008826387566628595) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.009841802921692996) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.009841802921692996) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.009841802921692996) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.009841802921692996) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.010311472181968588) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010311472181968588) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010311472181968588) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010311472181968588) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.016024666091731657) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (0.016024666091731657) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (0.016024666091731657) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (0.016024666091731657) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (0.017091621919560734) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.017091621919560734) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.017091621919560734) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.017091621919560734) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.019538085342165935) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.019538085342165935) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.019538085342165935) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.019538085342165935) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.022528354253334996) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.022528354253334996) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.023145221660820284) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (0.023145221660820284) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (0.02428203161836197) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (0.02428203161836197) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (0.024591832094859396) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.024591832094859396) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.024591832094859396) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.024591832094859396) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.03490330427682799) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.03490330427682799) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.03490330427682799) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.03490330427682799) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.03560840034934819) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.03560840034934819) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.06752398179244637) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.06752398179244637) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.06752398179244637) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.06752398179244637) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.07635036935907495) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.07635036935907495) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.07635036935907495) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.07635036935907495) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.08684736029995833) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.08684736029995833) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.08684736029995833) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.08684736029995833) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.09065142345440985) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.09065142345440985) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.09065142345440985) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.09065142345440985) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0716505624847426) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0716505624847426) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07165056248474258) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07165056248474258) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.775872156742351e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.775872156742351e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.775872156742351e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.775872156742351e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.05859215178487631) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.05859215178487631) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.019257452998969894) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.019257452998969894) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311472181968588) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010311472181968588) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008826387566628595) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.008826387566628595) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.004220835996971584) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.004220835996971584) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.004220835996971584) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.004220835996971584) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.003876482195629717) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (-0.003876482195629717) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (-0.003876482195629717) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (-0.003876482195629717) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (-0.0038040631544515317) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0038040631544515317) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.002446463422605197) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.002446463422605197) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.002394967154185249) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002394967154185249) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.002394967154185249) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002394967154185249) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.002394967154185249) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.002394967154185249) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.002394967154185249) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.002394967154185249) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.002200956847862872) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.002200956847862872) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.002200956847862872) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.002200956847862872) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.001236655922496217) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.001236655922496217) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.001236655922496217) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.001236655922496217) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.0011726297841433227) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0011726297841433227) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0011726297841433227) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0011726297841433227) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0007870893706149089) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0007870893706149089) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870893706149089) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0007870893706149089) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0005192924120051922) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.0005192924120051922) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.0005192924120051922) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.0005192924120051922) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-1.1462851123969703e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1462851123969703e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.8742485959314243e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742485959314243e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.8742485959314243e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742485959314243e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.3001959065423764e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.3001959065423764e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.3001959065423764e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.3001959065423764e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.0444741751646106e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.0444741751646106e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.0444741751646106e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.0444741751646106e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.955903688719832e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.955903688719832e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.955903688719832e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.955903688719832e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105341200799898e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105341200799898e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105341200799898e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105341200799898e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661200414287549e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661200414287549e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661200414287549e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661200414287549e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540204345474976e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540204345474976e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.189870551176575e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.189870551176575e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.876530456311099e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876530456311099e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876530456311099e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876530456311099e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175164961676847e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-6.175164961676847e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-4.523339429499919e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.523339429499919e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.076662726104703e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.076662726104703e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.076662726104703e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.076662726104703e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.013398883686404e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.013398883686404e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.9045374061711297e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.9045374061711297e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.9045374061711297e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.9045374061711297e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.666679774459261e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.666679774459261e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.666679774459261e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.666679774459261e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505624879198664e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.8505624879198664e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.846699579759078e-08) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-7.846699579759078e-08) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-6.772951848841522e-08) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (-6.772951848841522e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (-4.099829516466275e-08) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-4.099829516466275e-08) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.099829516466275e-08) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.099829516466275e-08) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (6.772951848841522e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (6.772951848841522e-08) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (7.846699579759078e-08) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (7.846699579759078e-08) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (1.6570092799063005e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.6570092799063005e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.6570092799063005e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.6570092799063005e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.8505624879198664e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.8505624879198664e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863217076261497e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863217076261497e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.6863217076261497e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863217076261497e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013398883686404e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.013398883686404e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.523339429499919e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.523339429499919e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.670408163595712e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.670408163595712e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.670408163595712e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.670408163595712e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175164961676847e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (6.175164961676847e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (7.189870551176575e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.189870551176575e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540204345474976e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540204345474976e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949307828638243e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949307828638243e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.7924638502676126e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924638502676126e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924638502676126e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.7924638502676126e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.883653184327338e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.883653184327338e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.988412606533134e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.988412606533134e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.988412606533134e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.988412606533134e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.2531852221290635e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.2531852221290635e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4016916571833366e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4016916571833366e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4016916571833366e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4016916571833366e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809380422100865e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.5809380422100865e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809380422100865e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.5809380422100865e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0010283270637195488) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0010283270637195488) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0010283270637195488) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0010283270637195488) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012223373700419261) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.0012223373700419261) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (0.0012223373700419261) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (0.0012223373700419261) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0012223373700419261) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0012223373700419261) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0012223373700419261) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012223373700419261) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.001640759115897476) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.001640759115897476) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.001640759115897476) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.001640759115897476) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.001863893138550432) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.001863893138550432) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.001863893138550432) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.001863893138550432) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.001863893138550432) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.001863893138550432) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.001863893138550432) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.001863893138550432) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.00214134896533342) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (0.00214134896533342) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (0.002249414060628551) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002249414060628551) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002249414060628551) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002249414060628551) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002446463422605197) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.002446463422605197) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.002984180074475367) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.002984180074475367) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.002984180074475367) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.002984180074475367) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.0038040631544515317) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0038040631544515317) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.005348047718000716) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.005348047718000716) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.005348047718000716) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.005348047718000716) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.005733568640078837) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.005733568640078837) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.005733568640078837) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.005733568640078837) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.005733568640078837) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.005733568640078837) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.005733568640078837) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.005733568640078837) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0075974617786292685) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0075974617786292685) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0075974617786292685) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0075974617786292685) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.008826387566628595) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.008826387566628595) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.010311472181968588) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.010311472181968588) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019257452998969894) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019257452998969894) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.398665373810666e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.398665373810666e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.3986653738106656e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.3986653738106656e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841545794502857) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0034841545794502857) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.002984180074475367) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.002984180074475367) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.0001940103063223774) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0001940103063223774) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1462851123969702e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1462851123969702e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
 </code>
 </pre>
 </details>

---

## 11. tutorial_quantum_chemistry.html <a name="demo10"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_quantum_chemistry.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pyscf/lib/misc.py:46: H5pyDeprecationWarning: Using default_file_mode other than 'r' is deprecated. Pass the mode to h5py.File() instead.
  h5py.get_config().default_file_mode = 'a'
Qubit Hamiltonian of the water molecule
(-46.46390678868895+0j) [] +
(-0.014583648907612575+0j) [X0 X1 Y2 Y3] +
(-3.570761329788667e-07+0j) [X0 X1 Y2 Z3 Z4 Y5] +
(-0.005652620978017361+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7] +
(-0.008826368514209846+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.7924939577442292e-06+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.570761329788667e-07+0j) [X0 X1 X3 X4] +
(-0.005652620978017361+0j) [X0 X1 X3 Z4 Z5 X6] +
(-0.008826368514209848+0j) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.7924939577442292e-06+0j) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.002745836470186808+0j) [X0 X1 Y4 Y5] +
(-2.4473231291178265e-07+0j) [X0 X1 Y4 Z5 Z6 Y7] +
(-7.86776510475254e-07+0j) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0038040661717285438+0j) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.4473231291178265e-07+0j) [X0 X1 X5 X6] +
(-7.86776510475254e-07+0j) [X0 X1 X5 Z6 Z7 Z8 Z9 X10] +
(-0.0038040661717285438+0j) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.00688819435297061+0j) [X0 X1 Y6 Y7] +
(-7.735036880593441e-05+0j) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11] +
(1.7035783555083448e-07+0j) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.735036880593441e-05+0j) [X0 X1 X7 Z8 Z9 X10] +
(1.7035783555083448e-07+0j) [X0 X1 X7 Z8 Z9 Z10 Z11 X12] +
(-0.006509361201177245+0j) [X0 X1 Y8 Y9] +
(-0.007731425250775312+0j) [X0 X1 Y10 Y11] +
(5.62785191154046e-07+0j) [X0 X1 Y10 Z11 Z12 Y13] +
(5.62785191154046e-07+0j) [X0 X1 X11 X12] +
(-0.005283776488402975+0j) [X0 X1 Y12 Y13] +
(0.014583648907612575+0j) [X0 Y1 Y2 X3] +
(3.570761329788667e-07+0j) [X0 Y1 Y2 Z3 Z4 X5] +
(0.005652620978017361+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7] +
(0.008826368514209846+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-1.7924939577442292e-06+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.570761329788667e-07+0j) [X0 Y1 Y3 X4] +
(-0.005652620978017361+0j) [X0 Y1 Y3 Z4 Z5 X6] +
(-0.008826368514209848+0j) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.7924939577442292e-06+0j) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.002745836470186808+0j) [X0 Y1 Y4 X5] +
(2.4473231291178265e-07+0j) [X0 Y1 Y4 Z5 Z6 X7] +
(7.86776510475254e-07+0j) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0038040661717285438+0j) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.4473231291178265e-07+0j) [X0 Y1 Y5 X6] +
(-7.86776510475254e-07+0j) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.0038040661717285438+0j) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.00688819435297061+0j) [X0 Y1 Y6 X7] +
(7.735036880593441e-05+0j) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11] +
(-1.7035783555083448e-07+0j) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.735036880593441e-05+0j) [X0 Y1 Y7 Z8 Z9 X10] +
(1.7035783555083448e-07+0j) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12] +
(0.006509361201177245+0j) [X0 Y1 Y8 X9] +
(0.007731425250775312+0j) [X0 Y1 Y10 X11] +
(-5.62785191154046e-07+0j) [X0 Y1 Y10 Z11 Z12 X13] +
(5.62785191154046e-07+0j) [X0 Y1 Y11 X12] +
(0.005283776488402975+0j) [X0 Y1 Y12 X13] +
(0.12507032579771696+0j) [X0 Z1 X2] +
(-1.9332412773115847e-07+0j) [X0 Z1 X2 X3 Z4 X5] +
(-0.0022939566113524637+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 X7] +
(-0.0016407548553123907+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(3.0134714589866395e-07+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.9332412773115847e-07+0j) [X0 Z1 X2 Y3 Z4 Y5] +
(-0.0022939566113524637+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7] +
(-0.0016407548553123907+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(3.0134714589866395e-07+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0032769719312315615+0j) [X0 Z1 X2 Z3] +
(-1.551053917693341e-07+0j) [X0 Z1 X2 X4 Z5 X6] +
(-1.1468376507776878e-06+0j) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.007597464029770587+0j) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.3807781482097296e-07+0j) [X0 Z1 X2 Y4 Z5 Y6] +
(-7.900128986469083e-07+0j) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.005348051582676588+0j) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.005530759218631479+0j) [X0 Z1 X2 Z4] +
(-1.3807781482097296e-07+0j) [X0 Z1 X2 X5 Z6 X7] +
(-3.376739308614999e-07+0j) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.001863894282458714+0j) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.3807781482097296e-07+0j) [X0 Z1 X2 Y5 Z6 Y7] +
(-3.376739308614999e-07+0j) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.001863894282458714+0j) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0011059037691896342+0j) [X0 Z1 X2 Z5] +
(0.005708495985960896+0j) [X0 Z1 X2 X6 Z7 Z8 Z9 X10] +
(-8.352332103123893e-07+0j) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
1.97422537908315e-08j [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0052626424730768204+0j) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10] +
(-8.074305986063435e-07+0j) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.008032520918821338+0j) [X0 Z1 X2 Z6] +
(0.0005940221543005213+0j) [X0 Z1 X2 X7 Z8 Z9 Z10 X11] +
(-8.37977324397798e-08+0j) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005940221543005213+0j) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11] +
(-8.37977324397798e-08+0j) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.003347617530666148+0j) [X0 Z1 X2 Z7] +
(0.011055020596132002+0j) [X0 Z1 X2 Z8] +
(0.0029297686747509883+0j) [X0 Z1 X2 Z9] +
(-6.418291574603341e-07+0j) [X0 Z1 X2 X10 Z11 X12] +
(-6.556281914756785e-07+0j) [X0 Z1 X2 Y10 Z11 Y12] +
(0.003555290195504214+0j) [X0 Z1 X2 Z10] +
(-1.1076325599874888e-07+0j) [X0 Z1 X2 X11 Z12 X13] +
(-1.1076325599874888e-07+0j) [X0 Z1 X2 Y11 Z12 Y13] +
(0.0017560707018411947+0j) [X0 Z1 X2 Z11] +
(0.00690123824979723+0j) [X0 Z1 X2 Z12] +
(0.0023262306231580307+0j) [X0 Z1 X2 Z13] +
(-3.568247521307796e-07+0j) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.002249412447093998+0j) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.0474716555193246e-08+0j) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0004458535128840775+0j) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10] +
(-1.974225379050873e-08+0j) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004424855449441846+0j) [X0 Z1 Z2 X3 Y4 Y5] +
(-4.523389677854083e-07+0j) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0034841573002178743+0j) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-8.09163719916188e-07+0j) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10] +
(-0.005733569747311872+0j) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.00468490338815519+0j) [X0 Z1 Z2 X3 Y6 Y7] +
(0.004668620318776298+0j) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11] +
(-7.189990975487444e-07+0j) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005114473831660375+0j) [X0 Z1 Z2 X3 X7 Z8 Z9 X10] +
(-7.560692464904286e-07+0j) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12] +
(-0.008125251921381013+0j) [X0 Z1 Z2 X3 Y8 Y9] +
(-0.0017992194936630188+0j) [X0 Z1 Z2 X3 Y10 Y11] +
(-5.471647744794065e-07+0j) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13] +
(-5.287660624591082e-07+0j) [X0 Z1 Z2 X3 X11 X12] +
(-0.0045750076266392+0j) [X0 Z1 Z2 X3 Y12 Y13] +
(0.004424855449441846+0j) [X0 Z1 Z2 Y3 Y4 X5] +
(4.523389677854083e-07+0j) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0034841573002178743+0j) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.09163719916188e-07+0j) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.005733569747311872+0j) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.00468490338815519+0j) [X0 Z1 Z2 Y3 Y6 X7] +
(-0.004668620318776298+0j) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11] +
(7.189990975487444e-07+0j) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005114473831660375+0j) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10] +
(-7.560692464904286e-07+0j) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12] +
(0.008125251921381013+0j) [X0 Z1 Z2 Y3 Y8 X9] +
(0.0017992194936630188+0j) [X0 Z1 Z2 Y3 Y10 X11] +
(5.471647744794065e-07+0j) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13] +
(-5.287660624591082e-07+0j) [X0 Z1 Z2 Y3 Y11 X12] +
(0.0045750076266392+0j) [X0 Z1 Z2 Y3 Y12 X13] +
(3.2020768803993203e-06+0j) [X0 Z1 Z2 Z3 X4] +
(0.0008533856254125586+0j) [X0 Z1 Z2 Z3 X4 X5 Z6 X7] +
(0.0007870896771024534+0j) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0008533856254125586+0j) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7] +
(0.0007870896771024534+0j) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.2919694865836442e-07+0j) [X0 Z1 Z2 Z3 X4 Z5] +
(4.4445978544791876e-07+0j) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10] +
(0.001172634831644182+0j) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.6849150954784265e-07+0j) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10] +
(0.002200964069500467+0j) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.8394209157650005e-07+0j) [X0 Z1 Z2 Z3 X4 Z6] +
(4.0922506162495e-07+0j) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11] +
(0.002394972639798031+0j) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.0922506162495e-07+0j) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11] +
(0.002394972639798031+0j) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.236259961825759e-07+0j) [X0 Z1 Z2 Z3 X4 Z8] +
(8.649310135129234e-08+0j) [X0 Z1 Z2 Z3 X4 Z9] +
(0.0013038004788126936+0j) [X0 Z1 Z2 Z3 X4 X10 Z11 X12] +
(0.003989841456619312+0j) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12] +
(-6.733197742448576e-07+0j) [X0 Z1 Z2 Z3 X4 Z10] +
(0.0022619660624823563+0j) [X0 Z1 Z2 Z3 X4 X11 Z12 X13] +
(0.0022619660624823563+0j) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13] +
(-5.92745308294272e-07+0j) [X0 Z1 Z2 Z3 X4 Z11] +
(1.239336321749658e-06+0j) [X0 Z1 Z2 Z3 X4 Z12] +
(9.306536652357458e-07+0j) [X0 Z1 Z2 Z3 X4 Z13] +
(-0.0010283292378562856+0j) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.002686040977806617+0j) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12] +
(-1.8394209157650005e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7] +
(-0.0001940085702975637+0j) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0012223378081538494+0j) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-2.371328948312835e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9] +
(8.057446595058543e-08+0j) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11] +
(0.0017278753941369542+0j) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13] +
(-0.0009581655836696625+0j) [X0 Z1 Z2 Z3 Z4 X5 X11 X12] +
(-3.086826565139122e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13] +
(1.8394209157650005e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7] +
(0.0001940085702975637+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0012223378081538494+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(2.371328948312835e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9] +
(-8.057446595058543e-08+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11] +
(-0.0017278753941369542+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13] +
(-0.0009581655836696625+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12] +
(3.086826565139122e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13] +
(0.04274327701378257+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6] +
(0.0005192743499487757+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(-1.850564193087162e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005192743499487757+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(-1.850564193087162e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.002779026799025576+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7] +
(0.004636976661182579+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8] +
(0.0012803060973496682+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9] +
(2.3120943051740326e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12] +
(1.071728218299437e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12] +
(0.005379937155839377+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10] +
(7.246974425643725e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13] +
(7.246974425643725e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13] +
(0.005241535382803889+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11] +
(0.004311038507914325+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12] +
(0.0010435246534907538+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13] +
(1.200428749406797e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12] +
(-0.0033566705638329117+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9] +
(-0.00013840177303548894+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11] +
(-6.175246207344288e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(-4.997018422479458e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12] +
(-0.003267513854423571+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13] +
(0.0033566705638329117+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9] +
(0.00013840177303548894+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11] +
(6.175246207344288e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(-4.997018422479458e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12] +
(0.003267513854423571+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13] +
(0.003876470899336967+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(-7.540341414077669e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.003876470899336967+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(-7.540341414077669e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(0.07165035181002477+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0021413612231016253+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(0.004220813970046463+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(0.0012366478019245105+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(-0.002984166168121952+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(0.002984166168121952+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-1.3987009011638351e-05+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(8.949476488578566e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-6.876621657743388e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(-7.661347212725077e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(0.0015324835230730023+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10] +
(-2.9045998834367544e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(0.0054089544224099695+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10] +
(-1.0444941297514424e-06+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(0.004767272188278113+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10] +
(-8.105515036477304e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(0.005286546538226888+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10] +
(-9.956079229564465e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0016095313817213745+0j) [X0 Z1 Z2 Z3 Z4 X6] +
(-7.141625221158626e-05+0j) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10] +
(-2.666731754383278e-07+0j) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.002462917007133933+0j) [X0 Z1 Z2 Z3 Z5 X6] +
(0.0007156734248908669+0j) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10] +
(-3.076732531574469e-07+0j) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.6060718682034623e-07+0j) [X0 Z1 Z2 X4] +
(0.003961560792496528+0j) [X0 Z1 Z2 Z4 Z5 X6] +
(0.00018787053389544264+0j) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.656930932654137e-07+0j) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.7379332624955106e-07+0j) [X0 Z1 Z3 X4] +
(0.0016676041811440646+0j) [X0 Z1 Z3 Z4 Z5 X6] +
(-0.0014528843214169482+0j) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(4.670402391640776e-07+0j) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.10433064780651373+0j) [X0 X2] +
(3.1174479465887913e-06+0j) [X0 Z2 Z3 X4] +
(0.04587947078129834+0j) [X0 Z2 Z3 Z4 Z5 X6] +
(0.058591988733862045+0j) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-1.1463061453144651e-05+0j) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.014583648907612575+0j) [Y0 X1 X2 Y3] +
(3.570761329788667e-07+0j) [Y0 X1 X2 Z3 Z4 Y5] +
(0.005652620978017361+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7] +
(0.008826368514209846+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.7924939577442292e-06+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.570761329788667e-07+0j) [Y0 X1 X3 Y4] +
(-0.005652620978017361+0j) [Y0 X1 X3 Z4 Z5 Y6] +
(-0.008826368514209848+0j) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.7924939577442292e-06+0j) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.002745836470186808+0j) [Y0 X1 X4 Y5] +
(2.4473231291178265e-07+0j) [Y0 X1 X4 Z5 Z6 Y7] +
(7.86776510475254e-07+0j) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.0038040661717285438+0j) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.4473231291178265e-07+0j) [Y0 X1 X5 Y6] +
(-7.86776510475254e-07+0j) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.0038040661717285438+0j) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.00688819435297061+0j) [Y0 X1 X6 Y7] +
(7.735036880593441e-05+0j) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11] +
(-1.7035783555083448e-07+0j) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.735036880593441e-05+0j) [Y0 X1 X7 Z8 Z9 Y10] +
(1.7035783555083448e-07+0j) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12] +
(0.006509361201177245+0j) [Y0 X1 X8 Y9] +
(0.007731425250775312+0j) [Y0 X1 X10 Y11] +
(-5.62785191154046e-07+0j) [Y0 X1 X10 Z11 Z12 Y13] +
(5.62785191154046e-07+0j) [Y0 X1 X11 Y12] +
(0.005283776488402975+0j) [Y0 X1 X12 Y13] +
(-0.014583648907612575+0j) [Y0 Y1 X2 X3] +
(-3.570761329788667e-07+0j) [Y0 Y1 X2 Z3 Z4 X5] +
(-0.005652620978017361+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7] +
(-0.008826368514209846+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.7924939577442292e-06+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.570761329788667e-07+0j) [Y0 Y1 Y3 Y4] +
(-0.005652620978017361+0j) [Y0 Y1 Y3 Z4 Z5 Y6] +
(-0.008826368514209848+0j) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.7924939577442292e-06+0j) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.002745836470186808+0j) [Y0 Y1 X4 X5] +
(-2.4473231291178265e-07+0j) [Y0 Y1 X4 Z5 Z6 X7] +
(-7.86776510475254e-07+0j) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0038040661717285438+0j) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.4473231291178265e-07+0j) [Y0 Y1 Y5 Y6] +
(-7.86776510475254e-07+0j) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.0038040661717285438+0j) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.00688819435297061+0j) [Y0 Y1 X6 X7] +
(-7.735036880593441e-05+0j) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11] +
(1.7035783555083448e-07+0j) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.735036880593441e-05+0j) [Y0 Y1 Y7 Z8 Z9 Y10] +
(1.7035783555083448e-07+0j) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.006509361201177245+0j) [Y0 Y1 X8 X9] +
(-0.007731425250775312+0j) [Y0 Y1 X10 X11] +
(5.62785191154046e-07+0j) [Y0 Y1 X10 Z11 Z12 X13] +
(5.62785191154046e-07+0j) [Y0 Y1 Y11 Y12] +
(-0.005283776488402975+0j) [Y0 Y1 X12 X13] +
(-3.568247521307796e-07+0j) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.002249412447093998+0j) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0004458535128840775+0j) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10] +
(-1.974225379050873e-08+0j) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.0474716555193246e-08+0j) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.12507032579771696+0j) [Y0 Z1 Y2] +
(-1.9332412773115847e-07+0j) [Y0 Z1 Y2 X3 Z4 X5] +
(-0.0022939566113524637+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7] +
(-0.0016407548553123907+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(3.0134714589866395e-07+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.9332412773115847e-07+0j) [Y0 Z1 Y2 Y3 Z4 Y5] +
(-0.0022939566113524637+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7] +
(-0.0016407548553123907+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(3.0134714589866395e-07+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0032769719312315615+0j) [Y0 Z1 Y2 Z3] +
(-1.3807781482097296e-07+0j) [Y0 Z1 Y2 X4 Z5 X6] +
(-7.900128986469083e-07+0j) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.005348051582676588+0j) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.551053917693341e-07+0j) [Y0 Z1 Y2 Y4 Z5 Y6] +
(-1.1468376507776878e-06+0j) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.007597464029770587+0j) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.005530759218631479+0j) [Y0 Z1 Y2 Z4] +
(-1.3807781482097296e-07+0j) [Y0 Z1 Y2 X5 Z6 X7] +
(-3.376739308614999e-07+0j) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.001863894282458714+0j) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.3807781482097296e-07+0j) [Y0 Z1 Y2 Y5 Z6 Y7] +
(-3.376739308614999e-07+0j) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.001863894282458714+0j) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0011059037691896342+0j) [Y0 Z1 Y2 Z5] +
(0.0052626424730768204+0j) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10] +
(-8.074305986063435e-07+0j) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.005708495985960896+0j) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10] +
-1.97422537908315e-08j [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-8.352332103123893e-07+0j) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.008032520918821338+0j) [Y0 Z1 Y2 Z6] +
(0.0005940221543005213+0j) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11] +
(-8.37977324397798e-08+0j) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005940221543005213+0j) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11] +
(-8.37977324397798e-08+0j) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.003347617530666148+0j) [Y0 Z1 Y2 Z7] +
(0.011055020596132002+0j) [Y0 Z1 Y2 Z8] +
(0.0029297686747509883+0j) [Y0 Z1 Y2 Z9] +
(-6.556281914756785e-07+0j) [Y0 Z1 Y2 X10 Z11 X12] +
(-6.418291574603341e-07+0j) [Y0 Z1 Y2 Y10 Z11 Y12] +
(0.003555290195504214+0j) [Y0 Z1 Y2 Z10] +
(-1.1076325599874888e-07+0j) [Y0 Z1 Y2 X11 Z12 X13] +
(-1.1076325599874888e-07+0j) [Y0 Z1 Y2 Y11 Z12 Y13] +
(0.0017560707018411947+0j) [Y0 Z1 Y2 Z11] +
(0.00690123824979723+0j) [Y0 Z1 Y2 Z12] +
(0.0023262306231580307+0j) [Y0 Z1 Y2 Z13] +
(0.004424855449441846+0j) [Y0 Z1 Z2 X3 X4 Y5] +
(4.523389677854083e-07+0j) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.0034841573002178743+0j) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-8.09163719916188e-07+0j) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.005733569747311872+0j) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.00468490338815519+0j) [Y0 Z1 Z2 X3 X6 Y7] +
(-0.004668620318776298+0j) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11] +
(7.189990975487444e-07+0j) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005114473831660375+0j) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10] +
(-7.560692464904286e-07+0j) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12] +
(0.008125251921381013+0j) [Y0 Z1 Z2 X3 X8 Y9] +
(0.0017992194936630188+0j) [Y0 Z1 Z2 X3 X10 Y11] +
(5.471647744794065e-07+0j) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13] +
(-5.287660624591082e-07+0j) [Y0 Z1 Z2 X3 X11 Y12] +
(0.0045750076266392+0j) [Y0 Z1 Z2 X3 X12 Y13] +
(-0.004424855449441846+0j) [Y0 Z1 Z2 Y3 X4 X5] +
(-4.523389677854083e-07+0j) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0034841573002178743+0j) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.09163719916188e-07+0j) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.005733569747311872+0j) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.00468490338815519+0j) [Y0 Z1 Z2 Y3 X6 X7] +
(0.004668620318776298+0j) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11] +
(-7.189990975487444e-07+0j) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005114473831660375+0j) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10] +
(-7.560692464904286e-07+0j) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.008125251921381013+0j) [Y0 Z1 Z2 Y3 X8 X9] +
(-0.0017992194936630188+0j) [Y0 Z1 Z2 Y3 X10 X11] +
(-5.471647744794065e-07+0j) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13] +
(-5.287660624591082e-07+0j) [Y0 Z1 Z2 Y3 Y11 Y12] +
(-0.0045750076266392+0j) [Y0 Z1 Z2 Y3 X12 X13] +
(-0.0010283292378562856+0j) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.002686040977806617+0j) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12] +
(3.2020768803993203e-06+0j) [Y0 Z1 Z2 Z3 Y4] +
(0.0008533856254125586+0j) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7] +
(0.0007870896771024534+0j) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0008533856254125586+0j) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7] +
(0.0007870896771024534+0j) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.2919694865836442e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z5] +
(4.6849150954784265e-07+0j) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10] +
(0.002200964069500467+0j) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.4445978544791876e-07+0j) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10] +
(0.001172634831644182+0j) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.8394209157650005e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z6] +
(4.0922506162495e-07+0j) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11] +
(0.002394972639798031+0j) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.0922506162495e-07+0j) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11] +
(0.002394972639798031+0j) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.236259961825759e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z8] +
(8.649310135129234e-08+0j) [Y0 Z1 Z2 Z3 Y4 Z9] +
(0.003989841456619312+0j) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12] +
(0.0013038004788126936+0j) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12] +
(-6.733197742448576e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z10] +
(0.0022619660624823563+0j) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13] +
(0.0022619660624823563+0j) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13] +
(-5.92745308294272e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z11] +
(1.239336321749658e-06+0j) [Y0 Z1 Z2 Z3 Y4 Z12] +
(9.306536652357458e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z13] +
(1.8394209157650005e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7] +
(0.0001940085702975637+0j) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0012223378081538494+0j) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(2.371328948312835e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9] +
(-8.057446595058543e-08+0j) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11] +
(-0.0017278753941369542+0j) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13] +
(-0.0009581655836696625+0j) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12] +
(3.086826565139122e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13] +
(-1.8394209157650005e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7] +
(-0.0001940085702975637+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0012223378081538494+0j) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-2.371328948312835e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9] +
(8.057446595058543e-08+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11] +
(0.0017278753941369542+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13] +
(-0.0009581655836696625+0j) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12] +
(-3.086826565139122e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13] +
(1.200428749406797e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12] +
(0.04274327701378257+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6] +
(0.0005192743499487757+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(-1.850564193087162e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005192743499487757+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(-1.850564193087162e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.002779026799025576+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7] +
(0.004636976661182579+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8] +
(0.0012803060973496682+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9] +
(1.071728218299437e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12] +
(2.3120943051740326e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12] +
(0.005379937155839377+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10] +
(7.246974425643725e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13] +
(7.246974425643725e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13] +
(0.005241535382803889+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11] +
(0.004311038507914325+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12] +
(0.0010435246534907538+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13] +
(0.0033566705638329117+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9] +
(0.00013840177303548894+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11] +
(6.175246207344288e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(-4.997018422479458e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12] +
(0.003267513854423571+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13] +
(-0.0033566705638329117+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9] +
(-0.00013840177303548894+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11] +
(-6.175246207344288e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(-4.997018422479458e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12] +
(-0.003267513854423571+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13] +
(0.003876470899336967+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(-7.540341414077669e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.003876470899336967+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(-7.540341414077669e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(0.07165035181002477+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0021413612231016253+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(0.004220813970046463+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(0.0012366478019245105+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(0.002984166168121952+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-0.002984166168121952+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-1.3987009011638351e-05+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(8.949476488578566e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-6.876621657743388e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(-7.661347212725077e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(0.0015324835230730023+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10] +
(-2.9045998834367544e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(0.0054089544224099695+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10] +
(-1.0444941297514424e-06+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(0.004767272188278113+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10] +
(-8.105515036477304e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(0.005286546538226888+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10] +
(-9.956079229564465e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0016095313817213745+0j) [Y0 Z1 Z2 Z3 Z4 Y6] +
(-7.141625221158626e-05+0j) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10] +
(-2.666731754383278e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.002462917007133933+0j) [Y0 Z1 Z2 Z3 Z5 Y6] +
(0.0007156734248908669+0j) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10] +
(-3.076732531574469e-07+0j) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(3.6060718682034623e-07+0j) [Y0 Z1 Z2 Y4] +
(0.003961560792496528+0j) [Y0 Z1 Z2 Z4 Z5 Y6] +
(0.00018787053389544264+0j) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.656930932654137e-07+0j) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.7379332624955106e-07+0j) [Y0 Z1 Z3 Y4] +
(0.0016676041811440646+0j) [Y0 Z1 Z3 Z4 Z5 Y6] +
(-0.0014528843214169482+0j) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(4.670402391640776e-07+0j) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.10433064780651373+0j) [Y0 Y2] +
(3.1174479465887913e-06+0j) [Y0 Z2 Z3 Y4] +
(0.04587947078129834+0j) [Y0 Z2 Z3 Z4 Z5 Y6] +
(0.058591988733862045+0j) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-1.1463061453144651e-05+0j) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(12.412630742111787+0j) [Z0] +
(0.10433064780651373+0j) [Z0 X1 Z2 X3] +
(3.1174479465887913e-06+0j) [Z0 X1 Z2 Z3 Z4 X5] +
(0.04587947078129834+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7] +
(0.058591988733862045+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-1.1463061453144651e-05+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.10433064780651373+0j) [Z0 Y1 Z2 Y3] +
(3.1174479465887913e-06+0j) [Z0 Y1 Z2 Z3 Z4 Y5] +
(0.04587947078129834+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7] +
(0.058591988733862045+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.1463061453144651e-05+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.1861763734860524+0j) [Z0 Z1] +
(-8.337746758960303e-07+0j) [Z0 X2 Z3 X4] +
(-0.027115036845273246+0j) [Z0 X2 Z3 Z4 Z5 X6] +
(-0.06752385099214055+0j) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.4017109735929768e-05+0j) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-8.337746758960303e-07+0j) [Z0 Y2 Z3 Y4] +
(-0.027115036845273246+0j) [Z0 Y2 Z3 Z4 Z5 Y6] +
(-0.06752385099214055+0j) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.4017109735929768e-05+0j) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.23671080783830392+0j) [Z0 Z2] +
(-1.1908508088748971e-06+0j) [Z0 X3 Z4 X5] +
(-0.03276765782329061+0j) [Z0 X3 Z4 Z5 Z6 X7] +
(-0.0763502195063504+0j) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.5809603693673997e-05+0j) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.1908508088748971e-06+0j) [Z0 Y3 Z4 Y5] +
(-0.03276765782329061+0j) [Z0 Y3 Z4 Z5 Z6 Y7] +
(-0.0763502195063504+0j) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.5809603693673997e-05+0j) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.2512944567459165+0j) [Z0 Z3] +
(-3.0993492437218658e-06+0j) [Z0 X4 Z5 X6] +
(-1.5316808795857735e-05+0j) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.08684737589863636+0j) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-3.0993492437218658e-06+0j) [Z0 Y4 Z5 Y6] +
(-1.5316808795857735e-05+0j) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.08684737589863636+0j) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.19661770890342156+0j) [Z0 Z4] +
(-3.344081556633648e-06+0j) [Z0 X5 Z6 X7] +
(-1.6103585306332987e-05+0j) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.09065144207036491+0j) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.344081556633648e-06+0j) [Z0 Y5 Z6 Y7] +
(-1.6103585306332987e-05+0j) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.09065144207036491+0j) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.1993635453736084+0j) [Z0 Z5] +
(0.0560846812466133+0j) [Z0 X6 Z7 Z8 Z9 X10] +
(-6.652209669488607e-06+0j) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0560846812466133+0j) [Z0 Y6 Z7 Z8 Z9 Y10] +
(-6.652209669488607e-06+0j) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.24164663936017272+0j) [Z0 Z6] +
(0.05600733087780736+0j) [Z0 X7 Z8 Z9 Z10 X11] +
(-6.481851833937773e-06+0j) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.05600733087780736+0j) [Z0 Y7 Z8 Z9 Z10 Y11] +
(-6.481851833937773e-06+0j) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.24853483371314333+0j) [Z0 Z7] +
(0.27232518306605713+0j) [Z0 Z8] +
(0.2788345442672344+0j) [Z0 Z9] +
(-2.1776646049623762e-06+0j) [Z0 X10 Z11 X12] +
(-2.1776646049623762e-06+0j) [Z0 Y10 Z11 Y12] +
(0.19299723935364282+0j) [Z0 Z10] +
(-1.6148794138083305e-06+0j) [Z0 X11 Z12 X13] +
(-1.6148794138083305e-06+0j) [Z0 Y11 Z12 Y13] +
(0.20072866460441813+0j) [Z0 Z11] +
(0.21102659849791566+0j) [Z0 Z12] +
(0.21631037498631867+0j) [Z0 Z13] +
(1.9332412773115847e-07+0j) [X1 X2 Y3 Y4] +
(0.0022939566113524637+0j) [X1 X2 Y3 Z4 Z5 Y6] +
(0.0016407548553123907+0j) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-3.0134714589866395e-07+0j) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004424855449441846+0j) [X1 X2 X4 X5] +
(-8.09163719916188e-07+0j) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.005733569747311872+0j) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-4.523389677854083e-07+0j) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.0034841573002178743+0j) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004684903388155189+0j) [X1 X2 X6 X7] +
(0.005114473831660375+0j) [X1 X2 X6 Z7 Z8 Z9 Z10 X11] +
(-7.560692464904286e-07+0j) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.004668620318776298+0j) [X1 X2 Y7 Z8 Z9 Y10] +
(-7.189990975487444e-07+0j) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.008125251921381013+0j) [X1 X2 X8 X9] +
(-0.0017992194936630188+0j) [X1 X2 X10 X11] +
(-5.287660624591082e-07+0j) [X1 X2 X10 Z11 Z12 X13] +
(-5.471647744794065e-07+0j) [X1 X2 Y11 Y12] +
(-0.004575007626639199+0j) [X1 X2 X12 X13] +
(-1.9332412773115847e-07+0j) [X1 Y2 Y3 X4] +
(-0.0022939566113524637+0j) [X1 Y2 Y3 Z4 Z5 X6] +
(-0.0016407548553123907+0j) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(3.0134714589866395e-07+0j) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004424855449441846+0j) [X1 Y2 Y4 X5] +
(-8.09163719916188e-07+0j) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.005733569747311872+0j) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.523389677854083e-07+0j) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10] +
(0.0034841573002178743+0j) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004684903388155189+0j) [X1 Y2 Y6 X7] +
(0.005114473831660375+0j) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11] +
(-7.560692464904286e-07+0j) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.004668620318776298+0j) [X1 Y2 Y7 Z8 Z9 X10] +
(7.189990975487444e-07+0j) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.008125251921381013+0j) [X1 Y2 Y8 X9] +
(-0.0017992194936630188+0j) [X1 Y2 Y10 X11] +
(-5.287660624591082e-07+0j) [X1 Y2 Y10 Z11 Z12 X13] +
(5.471647744794065e-07+0j) [X1 Y2 Y11 X12] +
(-0.004575007626639199+0j) [X1 Y2 Y12 X13] +
(0.125070325797717+0j) [X1 Z2 X3] +
(-1.3807781482097296e-07+0j) [X1 Z2 X3 X4 Z5 X6] +
(-3.376739308614999e-07+0j) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.001863894282458714+0j) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.3807781482097296e-07+0j) [X1 Z2 X3 Y4 Z5 Y6] +
(-3.376739308614999e-07+0j) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.001863894282458714+0j) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0011059037691896342+0j) [X1 Z2 X3 Z4] +
(-1.551053917693341e-07+0j) [X1 Z2 X3 X5 Z6 X7] +
(-1.1468376507776878e-06+0j) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.007597464029770587+0j) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.3807781482097296e-07+0j) [X1 Z2 X3 Y5 Z6 Y7] +
(-7.900128986469083e-07+0j) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.005348051582676588+0j) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005530759218631479+0j) [X1 Z2 X3 Z5] +
(0.0005940221543005213+0j) [X1 Z2 X3 X6 Z7 Z8 Z9 X10] +
(-8.37977324397798e-08+0j) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0005940221543005213+0j) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10] +
(-8.37977324397798e-08+0j) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.003347617530666148+0j) [X1 Z2 X3 Z6] +
(0.005708495985960896+0j) [X1 Z2 X3 X7 Z8 Z9 Z10 X11] +
(-8.352332103123893e-07+0j) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
1.97422537908315e-08j [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0052626424730768204+0j) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11] +
(-8.074305986063435e-07+0j) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.008032520918821338+0j) [X1 Z2 X3 Z7] +
(0.0029297686747509883+0j) [X1 Z2 X3 Z8] +
(0.011055020596132002+0j) [X1 Z2 X3 Z9] +
(-1.1076325599874888e-07+0j) [X1 Z2 X3 X10 Z11 X12] +
(-1.1076325599874888e-07+0j) [X1 Z2 X3 Y10 Z11 Y12] +
(0.0017560707018411947+0j) [X1 Z2 X3 Z10] +
(-6.418291574603341e-07+0j) [X1 Z2 X3 X11 Z12 X13] +
(-6.556281914756785e-07+0j) [X1 Z2 X3 Y11 Z12 Y13] +
(0.003555290195504214+0j) [X1 Z2 X3 Z11] +
(0.0023262306231580307+0j) [X1 Z2 X3 Z12] +
(0.00690123824979723+0j) [X1 Z2 X3 Z13] +
(-3.568247521307796e-07+0j) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.002249412447093998+0j) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.0474716555193246e-08+0j) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0004458535128840775+0j) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11] +
(-1.974225379050873e-08+0j) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0008533856254125586+0j) [X1 Z2 Z3 X4 Y5 Y6] +
(-0.0007870896771024534+0j) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10] +
(-1.8394209157650005e-07+0j) [X1 Z2 Z3 X4 X6 X7] +
(-0.0012223378081538494+0j) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0001940085702975637+0j) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12] +
(-2.371328948312835e-07+0j) [X1 Z2 Z3 X4 X8 X9] +
(8.057446595058543e-08+0j) [X1 Z2 Z3 X4 X10 X11] +
(-0.0009581655836696625+0j) [X1 Z2 Z3 X4 X10 Z11 Z12 X13] +
(0.0017278753941369542+0j) [X1 Z2 Z3 X4 Y11 Y12] +
(-3.086826565139122e-07+0j) [X1 Z2 Z3 X4 X12 X13] +
(0.0008533856254125586+0j) [X1 Z2 Z3 Y4 Y5 X6] +
(0.0007870896771024534+0j) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10] +
(-1.8394209157650005e-07+0j) [X1 Z2 Z3 Y4 Y6 X7] +
(-0.0012223378081538494+0j) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0001940085702975637+0j) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12] +
(-2.371328948312835e-07+0j) [X1 Z2 Z3 Y4 Y8 X9] +
(8.057446595058543e-08+0j) [X1 Z2 Z3 Y4 Y10 X11] +
(-0.0009581655836696625+0j) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13] +
(-0.0017278753941369542+0j) [X1 Z2 Z3 Y4 Y11 X12] +
(-3.086826565139122e-07+0j) [X1 Z2 Z3 Y4 Y12 X13] +
(3.202076880399321e-06+0j) [X1 Z2 Z3 Z4 X5] +
(4.0922506162495e-07+0j) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10] +
(0.002394972639798031+0j) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.0922506162495e-07+0j) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10] +
(0.002394972639798031+0j) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.4445978544791876e-07+0j) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11] +
(0.001172634831644182+0j) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.6849150954784265e-07+0j) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11] +
(0.002200964069500467+0j) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.8394209157650005e-07+0j) [X1 Z2 Z3 Z4 X5 Z7] +
(8.649310135129234e-08+0j) [X1 Z2 Z3 Z4 X5 Z8] +
(3.236259961825759e-07+0j) [X1 Z2 Z3 Z4 X5 Z9] +
(0.0022619660624823563+0j) [X1 Z2 Z3 Z4 X5 X10 Z11 X12] +
(0.0022619660624823563+0j) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12] +
(-5.92745308294272e-07+0j) [X1 Z2 Z3 Z4 X5 Z10] +
(0.0013038004788126936+0j) [X1 Z2 Z3 Z4 X5 X11 Z12 X13] +
(0.003989841456619312+0j) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13] +
(-6.733197742448576e-07+0j) [X1 Z2 Z3 Z4 X5 Z11] +
(9.306536652357458e-07+0j) [X1 Z2 Z3 Z4 X5 Z12] +
(1.239336321749658e-06+0j) [X1 Z2 Z3 Z4 X5 Z13] +
(-0.0010283292378562856+0j) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.002686040977806617+0j) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13] +
(-0.0005192743499487757+0j) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10] +
(1.850564193087162e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.0033566705638329117+0j) [X1 Z2 Z3 Z4 Z5 X6 X8 X9] +
(-0.00013840177303548894+0j) [X1 Z2 Z3 Z4 Z5 X6 X10 X11] +
(-4.997018422479458e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13] +
(-6.175246207344288e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12] +
(-0.003267513854423571+0j) [X1 Z2 Z3 Z4 Z5 X6 X12 X13] +
(0.0005192743499487757+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10] +
(-1.850564193087162e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.0033566705638329117+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9] +
(-0.00013840177303548894+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11] +
(-4.997018422479458e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13] +
(6.175246207344288e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12] +
(-0.003267513854423571+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13] +
(0.04274327701378257+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7] +
(0.0012803060973496682+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8] +
(0.004636976661182579+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9] +
(7.246974425643725e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12] +
(7.246974425643725e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12] +
(0.005241535382803889+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10] +
(2.3120943051740326e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13] +
(1.071728218299437e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13] +
(0.005379937155839377+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11] +
(0.0010435246534907538+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12] +
(0.004311038507914325+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13] +
(1.200428749406797e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13] +
(-0.0038764708993369672+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10] +
(7.540341414077669e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(0.0038764708993369672+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10] +
(-7.540341414077669e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(-0.002984166168121952+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-0.002984166168121952+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(0.07165035181002481+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0012366478019245105+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(0.004220813970046463+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(-1.3987009011638344e-05+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(8.949476488578565e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(-7.661347212725077e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(-0.0021413612231016253+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11] +
(-6.876621657743388e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(0.0054089544224099695+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11] +
(-1.0444941297514424e-06+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(0.0015324835230730023+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11] +
(-2.9045998834367544e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(0.005286546538226888+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11] +
(-9.956079229564465e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.002779026799025576+0j) [X1 Z2 Z3 Z4 Z5 X7] +
(0.004767272188278113+0j) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11] +
(-8.105515036477304e-07+0j) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.002462917007133933+0j) [X1 Z2 Z3 Z4 Z6 X7] +
(0.0007156734248908669+0j) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11] +
(-3.076732531574469e-07+0j) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2919694865836442e-07+0j) [X1 Z2 Z3 X5] +
(0.0016095313817213745+0j) [X1 Z2 Z3 Z5 Z6 X7] +
(-7.141625221158626e-05+0j) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-2.666731754383278e-07+0j) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.7379332624955106e-07+0j) [X1 Z2 Z4 X5] +
(0.0016676041811440646+0j) [X1 Z2 Z4 Z5 Z6 X7] +
(-0.0014528843214169482+0j) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(4.670402391640776e-07+0j) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0032769719312315615+0j) [X1 X3] +
(3.6060718682034623e-07+0j) [X1 Z3 Z4 X5] +
(0.003961560792496528+0j) [X1 Z3 Z4 Z5 Z6 X7] +
(0.00018787053389544264+0j) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.656930932654137e-07+0j) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.9332412773115847e-07+0j) [Y1 X2 X3 Y4] +
(-0.0022939566113524637+0j) [Y1 X2 X3 Z4 Z5 Y6] +
(-0.0016407548553123907+0j) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(3.0134714589866395e-07+0j) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004424855449441846+0j) [Y1 X2 X4 Y5] +
(-8.09163719916188e-07+0j) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.005733569747311872+0j) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(4.523389677854083e-07+0j) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10] +
(0.0034841573002178743+0j) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004684903388155189+0j) [Y1 X2 X6 Y7] +
(0.005114473831660375+0j) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11] +
(-7.560692464904286e-07+0j) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.004668620318776298+0j) [Y1 X2 X7 Z8 Z9 Y10] +
(7.189990975487444e-07+0j) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.008125251921381013+0j) [Y1 X2 X8 Y9] +
(-0.0017992194936630188+0j) [Y1 X2 X10 Y11] +
(-5.287660624591082e-07+0j) [Y1 X2 X10 Z11 Z12 Y13] +
(5.471647744794065e-07+0j) [Y1 X2 X11 Y12] +
(-0.004575007626639199+0j) [Y1 X2 X12 Y13] +
(1.9332412773115847e-07+0j) [Y1 Y2 X3 X4] +
(0.0022939566113524637+0j) [Y1 Y2 X3 Z4 Z5 X6] +
(0.0016407548553123907+0j) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-3.0134714589866395e-07+0j) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004424855449441846+0j) [Y1 Y2 Y4 Y5] +
(-8.09163719916188e-07+0j) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.005733569747311872+0j) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-4.523389677854083e-07+0j) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10] +
(-0.0034841573002178743+0j) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004684903388155189+0j) [Y1 Y2 Y6 Y7] +
(0.005114473831660375+0j) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11] +
(-7.560692464904286e-07+0j) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.004668620318776298+0j) [Y1 Y2 X7 Z8 Z9 X10] +
(-7.189990975487444e-07+0j) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12] +
(-0.008125251921381013+0j) [Y1 Y2 Y8 Y9] +
(-0.0017992194936630188+0j) [Y1 Y2 Y10 Y11] +
(-5.287660624591082e-07+0j) [Y1 Y2 Y10 Z11 Z12 Y13] +
(-5.471647744794065e-07+0j) [Y1 Y2 X11 X12] +
(-0.004575007626639199+0j) [Y1 Y2 Y12 Y13] +
(-3.568247521307796e-07+0j) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.002249412447093998+0j) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0004458535128840775+0j) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11] +
(-1.974225379050873e-08+0j) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.0474716555193246e-08+0j) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.125070325797717+0j) [Y1 Z2 Y3] +
(-1.3807781482097296e-07+0j) [Y1 Z2 Y3 X4 Z5 X6] +
(-3.376739308614999e-07+0j) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.001863894282458714+0j) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.3807781482097296e-07+0j) [Y1 Z2 Y3 Y4 Z5 Y6] +
(-3.376739308614999e-07+0j) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.001863894282458714+0j) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0011059037691896342+0j) [Y1 Z2 Y3 Z4] +
(-1.3807781482097296e-07+0j) [Y1 Z2 Y3 X5 Z6 X7] +
(-7.900128986469083e-07+0j) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.005348051582676588+0j) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.551053917693341e-07+0j) [Y1 Z2 Y3 Y5 Z6 Y7] +
(-1.1468376507776878e-06+0j) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.007597464029770587+0j) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005530759218631479+0j) [Y1 Z2 Y3 Z5] +
(0.0005940221543005213+0j) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10] +
(-8.37977324397798e-08+0j) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0005940221543005213+0j) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10] +
(-8.37977324397798e-08+0j) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.003347617530666148+0j) [Y1 Z2 Y3 Z6] +
(0.0052626424730768204+0j) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11] +
(-8.074305986063435e-07+0j) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005708495985960896+0j) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11] +
-1.97422537908315e-08j [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.352332103123893e-07+0j) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.008032520918821338+0j) [Y1 Z2 Y3 Z7] +
(0.0029297686747509883+0j) [Y1 Z2 Y3 Z8] +
(0.011055020596132002+0j) [Y1 Z2 Y3 Z9] +
(-1.1076325599874888e-07+0j) [Y1 Z2 Y3 X10 Z11 X12] +
(-1.1076325599874888e-07+0j) [Y1 Z2 Y3 Y10 Z11 Y12] +
(0.0017560707018411947+0j) [Y1 Z2 Y3 Z10] +
(-6.556281914756785e-07+0j) [Y1 Z2 Y3 X11 Z12 X13] +
(-6.418291574603341e-07+0j) [Y1 Z2 Y3 Y11 Z12 Y13] +
(0.003555290195504214+0j) [Y1 Z2 Y3 Z11] +
(0.0023262306231580307+0j) [Y1 Z2 Y3 Z12] +
(0.00690123824979723+0j) [Y1 Z2 Y3 Z13] +
(0.0008533856254125586+0j) [Y1 Z2 Z3 X4 X5 Y6] +
(0.0007870896771024534+0j) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10] +
(-1.8394209157650005e-07+0j) [Y1 Z2 Z3 X4 X6 Y7] +
(-0.0012223378081538494+0j) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0001940085702975637+0j) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12] +
(-2.371328948312835e-07+0j) [Y1 Z2 Z3 X4 X8 Y9] +
(8.057446595058543e-08+0j) [Y1 Z2 Z3 X4 X10 Y11] +
(-0.0009581655836696625+0j) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13] +
(-0.0017278753941369542+0j) [Y1 Z2 Z3 X4 X11 Y12] +
(-3.086826565139122e-07+0j) [Y1 Z2 Z3 X4 X12 Y13] +
(-0.0008533856254125586+0j) [Y1 Z2 Z3 Y4 X5 X6] +
(-0.0007870896771024534+0j) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10] +
(-1.8394209157650005e-07+0j) [Y1 Z2 Z3 Y4 Y6 Y7] +
(-0.0012223378081538494+0j) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0001940085702975637+0j) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12] +
(-2.371328948312835e-07+0j) [Y1 Z2 Z3 Y4 Y8 Y9] +
(8.057446595058543e-08+0j) [Y1 Z2 Z3 Y4 Y10 Y11] +
(-0.0009581655836696625+0j) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13] +
(0.0017278753941369542+0j) [Y1 Z2 Z3 Y4 X11 X12] +
(-3.086826565139122e-07+0j) [Y1 Z2 Z3 Y4 Y12 Y13] +
(-0.0010283292378562856+0j) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.002686040977806617+0j) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13] +
(3.202076880399321e-06+0j) [Y1 Z2 Z3 Z4 Y5] +
(4.0922506162495e-07+0j) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10] +
(0.002394972639798031+0j) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.0922506162495e-07+0j) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10] +
(0.002394972639798031+0j) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.6849150954784265e-07+0j) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11] +
(0.002200964069500467+0j) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.4445978544791876e-07+0j) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11] +
(0.001172634831644182+0j) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.8394209157650005e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z7] +
(8.649310135129234e-08+0j) [Y1 Z2 Z3 Z4 Y5 Z8] +
(3.236259961825759e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z9] +
(0.0022619660624823563+0j) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12] +
(0.0022619660624823563+0j) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12] +
(-5.92745308294272e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z10] +
(0.003989841456619312+0j) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13] +
(0.0013038004788126936+0j) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13] +
(-6.733197742448576e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z11] +
(9.306536652357458e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z12] +
(1.239336321749658e-06+0j) [Y1 Z2 Z3 Z4 Y5 Z13] +
(0.0005192743499487757+0j) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10] +
(-1.850564193087162e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.0033566705638329117+0j) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9] +
(-0.00013840177303548894+0j) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11] +
(-4.997018422479458e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13] +
(6.175246207344288e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12] +
(-0.003267513854423571+0j) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13] +
(-0.0005192743499487757+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10] +
(1.850564193087162e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-0.0033566705638329117+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9] +
(-0.00013840177303548894+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11] +
(-4.997018422479458e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13] +
(-6.175246207344288e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12] +
(-0.003267513854423571+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13] +
(1.200428749406797e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13] +
(0.04274327701378257+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7] +
(0.0012803060973496682+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8] +
(0.004636976661182579+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9] +
(7.246974425643725e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12] +
(7.246974425643725e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12] +
(0.005241535382803889+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10] +
(1.071728218299437e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13] +
(2.3120943051740326e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13] +
(0.005379937155839377+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11] +
(0.0010435246534907538+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12] +
(0.004311038507914325+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13] +
(0.0038764708993369672+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10] +
(-7.540341414077669e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-0.0038764708993369672+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10] +
(7.540341414077669e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-0.002984166168121952+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(-0.002984166168121952+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(0.07165035181002481+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.0012366478019245105+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(0.004220813970046463+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(-1.3987009011638344e-05+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(8.949476488578565e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(-7.661347212725077e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(-0.0021413612231016253+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11] +
(-6.876621657743388e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(0.0054089544224099695+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11] +
(-1.0444941297514424e-06+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(0.0015324835230730023+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11] +
(-2.9045998834367544e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(0.005286546538226888+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11] +
(-9.956079229564465e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.002779026799025576+0j) [Y1 Z2 Z3 Z4 Z5 Y7] +
(0.004767272188278113+0j) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11] +
(-8.105515036477304e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.002462917007133933+0j) [Y1 Z2 Z3 Z4 Z6 Y7] +
(0.0007156734248908669+0j) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11] +
(-3.076732531574469e-07+0j) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.2919694865836442e-07+0j) [Y1 Z2 Z3 Y5] +
(0.0016095313817213745+0j) [Y1 Z2 Z3 Z5 Z6 Y7] +
(-7.141625221158626e-05+0j) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-2.666731754383278e-07+0j) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.7379332624955106e-07+0j) [Y1 Z2 Z4 Y5] +
(0.0016676041811440646+0j) [Y1 Z2 Z4 Z5 Z6 Y7] +
(-0.0014528843214169482+0j) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(4.670402391640776e-07+0j) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0032769719312315615+0j) [Y1 Y3] +
(3.6060718682034623e-07+0j) [Y1 Z3 Z4 Y5] +
(0.003961560792496528+0j) [Y1 Z3 Z4 Z5 Z6 Y7] +
(0.00018787053389544264+0j) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.656930932654137e-07+0j) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(12.412630742111787+0j) [Z1] +
(-1.1908508088748971e-06+0j) [Z1 X2 Z3 X4] +
(-0.03276765782329061+0j) [Z1 X2 Z3 Z4 Z5 X6] +
(-0.0763502195063504+0j) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.5809603693673997e-05+0j) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.1908508088748971e-06+0j) [Z1 Y2 Z3 Y4] +
(-0.03276765782329061+0j) [Z1 Y2 Z3 Z4 Z5 Y6] +
(-0.0763502195063504+0j) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.5809603693673997e-05+0j) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.2512944567459165+0j) [Z1 Z2] +
(-8.337746758960303e-07+0j) [Z1 X3 Z4 X5] +
(-0.027115036845273246+0j) [Z1 X3 Z4 Z5 Z6 X7] +
(-0.06752385099214055+0j) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.4017109735929768e-05+0j) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.337746758960303e-07+0j) [Z1 Y3 Z4 Y5] +
(-0.027115036845273246+0j) [Z1 Y3 Z4 Z5 Z6 Y7] +
(-0.06752385099214055+0j) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.4017109735929768e-05+0j) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.23671080783830392+0j) [Z1 Z3] +
(-3.344081556633648e-06+0j) [Z1 X4 Z5 X6] +
(-1.6103585306332987e-05+0j) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.09065144207036491+0j) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-3.344081556633648e-06+0j) [Z1 Y4 Z5 Y6] +
(-1.6103585306332987e-05+0j) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.09065144207036491+0j) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.1993635453736084+0j) [Z1 Z4] +
(-3.0993492437218658e-06+0j) [Z1 X5 Z6 X7] +
(-1.5316808795857735e-05+0j) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.08684737589863636+0j) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.0993492437218658e-06+0j) [Z1 Y5 Z6 Y7] +
(-1.5316808795857735e-05+0j) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.08684737589863636+0j) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.19661770890342156+0j) [Z1 Z5] +
(0.05600733087780736+0j) [Z1 X6 Z7 Z8 Z9 X10] +
(-6.481851833937773e-06+0j) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.05600733087780736+0j) [Z1 Y6 Z7 Z8 Z9 Y10] +
(-6.481851833937773e-06+0j) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.24853483371314333+0j) [Z1 Z6] +
(0.0560846812466133+0j) [Z1 X7 Z8 Z9 Z10 X11] +
(-6.652209669488607e-06+0j) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0560846812466133+0j) [Z1 Y7 Z8 Z9 Z10 Y11] +
(-6.652209669488607e-06+0j) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.24164663936017272+0j) [Z1 Z7] +
(0.2788345442672344+0j) [Z1 Z8] +
(0.27232518306605713+0j) [Z1 Z9] +
(-1.6148794138083305e-06+0j) [Z1 X10 Z11 X12] +
(-1.6148794138083305e-06+0j) [Z1 Y10 Z11 Y12] +
(0.20072866460441813+0j) [Z1 Z10] +
(-2.1776646049623762e-06+0j) [Z1 X11 Z12 X13] +
(-2.1776646049623762e-06+0j) [Z1 Y11 Z12 Y13] +
(0.19299723935364282+0j) [Z1 Z11] +
(0.21631037498631867+0j) [Z1 Z12] +
(0.21102659849791566+0j) [Z1 Z13] +
(-0.03583956795335352+0j) [X2 X3 Y4 Y5] +
(-2.199051618818372e-07+0j) [X2 X3 Y4 Z5 Z6 Y7] +
(-2.360956320482492e-06+0j) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.010311482489831667+0j) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.199051618818372e-07+0j) [X2 X3 X5 X6] +
(-2.3609563204824914e-06+0j) [X2 X3 X5 Z6 Z7 Z8 Z9 X10] +
(-0.010311482489831667+0j) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.031143817988967034+0j) [X2 X3 Y6 Y7] +
(0.0053686593581093985+0j) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11] +
(9.209350652029754e-08+0j) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0053686593581093985+0j) [X2 X3 X7 Z8 Z9 X10] +
(9.209350652029754e-08+0j) [X2 X3 X7 Z8 Z9 Z10 Z11 X12] +
(-0.036194123559042474+0j) [X2 X3 Y8 Y9] +
(-0.025384657508457503+0j) [X2 X3 Y10 Y11] +
(2.1726691016779745e-06+0j) [X2 X3 Y10 Z11 Z12 Y13] +
(2.1726691016779745e-06+0j) [X2 X3 X11 X12] +
(-0.01557720806397646+0j) [X2 X3 Y12 Y13] +
(0.03583956795335352+0j) [X2 Y3 Y4 X5] +
(2.199051618818372e-07+0j) [X2 Y3 Y4 Z5 Z6 X7] +
(2.360956320482492e-06+0j) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.010311482489831667+0j) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.199051618818372e-07+0j) [X2 Y3 Y5 X6] +
(-2.3609563204824914e-06+0j) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.010311482489831667+0j) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.031143817988967034+0j) [X2 Y3 Y6 X7] +
(-0.0053686593581093985+0j) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11] +
(-9.209350652029754e-08+0j) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0053686593581093985+0j) [X2 Y3 Y7 Z8 Z9 X10] +
(9.209350652029754e-08+0j) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12] +
(0.036194123559042474+0j) [X2 Y3 Y8 X9] +
(0.025384657508457503+0j) [X2 Y3 Y10 X11] +
(-2.1726691016779745e-06+0j) [X2 Y3 Y10 Z11 Z12 X13] +
(2.1726691016779745e-06+0j) [X2 Y3 Y11 X12] +
(0.01557720806397646+0j) [X2 Y3 Y12 X13] +
(-3.8870516758612256e-06+0j) [X2 Z3 X4] +
(-0.005143391768825085+0j) [X2 Z3 X4 X5 Z6 X7] +
(-0.009841749246962631+0j) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.988511706664223e-06+0j) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.005143391768825085+0j) [X2 Z3 X4 Y5 Z6 Y7] +
(-0.009841749246962631+0j) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.988511706664223e-06+0j) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.76499412104935e-07+0j) [X2 Z3 X4 Z5] +
(1.6893489515456822e-06+0j) [X2 Z3 X4 X6 Z7 Z8 Z9 X10] +
(0.010757563953908943+0j) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.5371780953306294e-08+0j) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10] +
(4.2055484112201325e-05+0j) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.593534389680191e-07+0j) [X2 Z3 X4 Z6] +
(3.2118420191802892e-06+0j) [X2 Z3 X4 X7 Z8 Z9 Z10 X11] +
(0.019299560579363825+0j) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.2118420191802892e-06+0j) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11] +
(0.019299560579363825+0j) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.1954890102747775e-06+0j) [X2 Z3 X4 Z7] +
(2.18684237619931e-07+0j) [X2 Z3 X4 Z8] +
(-5.770052997875832e-07+0j) [X2 Z3 X4 Z9] +
(0.015588250102380219+0j) [X2 Z3 X4 X10 Z11 X12] +
(0.005324835234221658+0j) [X2 Z3 X4 Y10 Z11 Y12] +
(-3.1586564322847177e-06+0j) [X2 Z3 X4 Z10] +
(0.024353077678069074+0j) [X2 Z3 X4 X11 Z12 X13] +
(0.024353077678069074+0j) [X2 Z3 X4 Y11 Z12 Y13] +
(-7.80170750091976e-06+0j) [X2 Z3 X4 Z11] +
(3.5390541845155326e-06+0j) [X2 Z3 X4 Z12] +
(8.814937306911245e-06+0j) [X2 Z3 X4 Z13] +
(1.6288532436080791e-06+0j) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10] +
(0.010715508469796738+0j) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.010263414868158556+0j) [X2 Z3 Y4 Y10 Z11 X12] +
(-1.4548424492427964e-06+0j) [X2 Z3 Z4 X5 Y6 Y7] +
(-3.1513463112426866e-06+0j) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11] +
(-0.01925750509525162+0j) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.5224930676346074e-06+0j) [X2 Z3 Z4 X5 X7 Z8 Z9 X10] +
(-0.008541996625454884+0j) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-7.956895374075141e-07+0j) [X2 Z3 Z4 X5 Y8 Y9] +
(-4.643051068635042e-06+0j) [X2 Z3 Z4 X5 Y10 Y11] +
(-0.019028242443847414+0j) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13] +
(-0.008764827575688855+0j) [X2 Z3 Z4 X5 X11 X12] +
(5.275883122395712e-06+0j) [X2 Z3 Z4 X5 Y12 Y13] +
(1.4548424492427964e-06+0j) [X2 Z3 Z4 Y5 Y6 X7] +
(3.1513463112426866e-06+0j) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11] +
(0.01925750509525162+0j) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.5224930676346074e-06+0j) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10] +
(-0.008541996625454884+0j) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(7.956895374075141e-07+0j) [X2 Z3 Z4 Y5 Y8 X9] +
(4.643051068635042e-06+0j) [X2 Z3 Z4 Y5 Y10 X11] +
(0.019028242443847414+0j) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13] +
(-0.008764827575688855+0j) [X2 Z3 Z4 Y5 Y11 X12] +
(-5.275883122395712e-06+0j) [X2 Z3 Z4 Y5 Y12 X13] +
(-0.12133276911042447+0j) [X2 Z3 Z4 Z5 X6] +
(-0.008469978791023871+0j) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(2.6863815441743586e-07+0j) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.008469978791023871+0j) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(2.6863815441743586e-07+0j) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.025996177598021364+0j) [X2 Z3 Z4 Z5 X6 Z7] +
(-0.005805188989826893+0j) [X2 Z3 Z4 Z5 X6 Z8] +
(-0.017561202409646204+0j) [X2 Z3 Z4 Z5 X6 Z9] +
(-7.988770289638443e-07+0j) [X2 Z3 Z4 Z5 X6 X10 Z11 X12] +
(-3.427323109166914e-07+0j) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12] +
(-0.000814531327095695+0j) [X2 Z3 Z4 Z5 X6 Z10] +
(2.745518400484067e-06+0j) [X2 Z3 Z4 Z5 X6 X11 Z12 X13] +
(2.745518400484067e-06+0j) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13] +
(0.014411099430130903+0j) [X2 Z3 Z4 Z5 X6 Z11] +
(0.0006650070219498542+0j) [X2 Z3 Z4 Z5 X6 Z12] +
(-0.003493790359890189+0j) [X2 Z3 Z4 Z5 X6 Z13] +
(-4.561447180471532e-07+0j) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12] +
(-0.011756013419819307+0j) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9] +
(0.015225630757226598+0j) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11] +
(-3.0882507114007586e-06+0j) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(-3.544395429447912e-06+0j) [X2 Z3 Z4 Z5 Z6 X7 X11 X12] +
(-0.004158797381840043+0j) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13] +
(0.011756013419819307+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9] +
(-0.015225630757226598+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11] +
(3.0882507114007586e-06+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(-3.544395429447912e-06+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12] +
(0.004158797381840043+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13] +
(0.014603704729162167+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(-2.8742990715356804e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.014603704729162167+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(-2.8742990715356804e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(-0.2816425776702309+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-1.300294656301964e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(-1.300294656301964e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(-0.024282117354693045+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(-0.019538050311314815+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(-0.017091553155898907+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(0.00244649715541591+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(-0.00244649715541591+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(5.7759505275970614e-05+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(2.8836765762282103e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(5.146496327748959e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(3.846201671446995e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(-0.0393591680220532+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10] +
(7.979825793772298e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(-0.024755463292891033+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10] +
(5.105526722236617e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(-0.021433810721601033+0j) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10] +
(5.1593505022450165e-06+0j) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(-0.0299037895126249+0j) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10] +
(5.4279886566624525e-06+0j) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0016638798784907832+0j) [X2 Z3 Z4 X6] +
(-0.018889030304942895+0j) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10] +
(2.9473560117242756e-06+0j) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.0034795118903343013+0j) [X2 Z3 Z5 X6] +
(-0.028730779551905526+0j) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10] +
(5.935867718388499e-06+0j) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.602116740625765e-06+0j) [X2 X4] +
(0.0004956762314917326+0j) [X2 Z4 Z5 X6] +
(-0.03560837898831254+0j) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(7.253273348457477e-06+0j) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.03583956795335352+0j) [Y2 X3 X4 Y5] +
(2.199051618818372e-07+0j) [Y2 X3 X4 Z5 Z6 Y7] +
(2.360956320482492e-06+0j) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.010311482489831667+0j) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.199051618818372e-07+0j) [Y2 X3 X5 Y6] +
(-2.3609563204824914e-06+0j) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.010311482489831667+0j) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.031143817988967034+0j) [Y2 X3 X6 Y7] +
(-0.0053686593581093985+0j) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11] +
(-9.209350652029754e-08+0j) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0053686593581093985+0j) [Y2 X3 X7 Z8 Z9 Y10] +
(9.209350652029754e-08+0j) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12] +
(0.036194123559042474+0j) [Y2 X3 X8 Y9] +
(0.025384657508457503+0j) [Y2 X3 X10 Y11] +
(-2.1726691016779745e-06+0j) [Y2 X3 X10 Z11 Z12 Y13] +
(2.1726691016779745e-06+0j) [Y2 X3 X11 Y12] +
(0.01557720806397646+0j) [Y2 X3 X12 Y13] +
(-0.03583956795335352+0j) [Y2 Y3 X4 X5] +
(-2.199051618818372e-07+0j) [Y2 Y3 X4 Z5 Z6 X7] +
(-2.360956320482492e-06+0j) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.010311482489831667+0j) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.199051618818372e-07+0j) [Y2 Y3 Y5 Y6] +
(-2.3609563204824914e-06+0j) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.010311482489831667+0j) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.031143817988967034+0j) [Y2 Y3 X6 X7] +
(0.0053686593581093985+0j) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11] +
(9.209350652029754e-08+0j) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0053686593581093985+0j) [Y2 Y3 Y7 Z8 Z9 Y10] +
(9.209350652029754e-08+0j) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.036194123559042474+0j) [Y2 Y3 X8 X9] +
(-0.025384657508457503+0j) [Y2 Y3 X10 X11] +
(2.1726691016779745e-06+0j) [Y2 Y3 X10 Z11 Z12 X13] +
(2.1726691016779745e-06+0j) [Y2 Y3 Y11 Y12] +
(-0.01557720806397646+0j) [Y2 Y3 X12 X13] +
(1.6288532436080791e-06+0j) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10] +
(0.010715508469796738+0j) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.010263414868158556+0j) [Y2 Z3 X4 X10 Z11 Y12] +
(-3.8870516758612256e-06+0j) [Y2 Z3 Y4] +
(-0.005143391768825085+0j) [Y2 Z3 Y4 X5 Z6 X7] +
(-0.009841749246962631+0j) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.988511706664223e-06+0j) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.005143391768825085+0j) [Y2 Z3 Y4 Y5 Z6 Y7] +
(-0.009841749246962631+0j) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.988511706664223e-06+0j) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.76499412104935e-07+0j) [Y2 Z3 Y4 Z5] +
(4.5371780953306294e-08+0j) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10] +
(4.2055484112201325e-05+0j) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.6893489515456822e-06+0j) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10] +
(0.010757563953908943+0j) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.593534389680191e-07+0j) [Y2 Z3 Y4 Z6] +
(3.2118420191802892e-06+0j) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11] +
(0.019299560579363825+0j) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.2118420191802892e-06+0j) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11] +
(0.019299560579363825+0j) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.1954890102747775e-06+0j) [Y2 Z3 Y4 Z7] +
(2.18684237619931e-07+0j) [Y2 Z3 Y4 Z8] +
(-5.770052997875832e-07+0j) [Y2 Z3 Y4 Z9] +
(0.005324835234221658+0j) [Y2 Z3 Y4 X10 Z11 X12] +
(0.015588250102380219+0j) [Y2 Z3 Y4 Y10 Z11 Y12] +
(-3.1586564322847177e-06+0j) [Y2 Z3 Y4 Z10] +
(0.024353077678069074+0j) [Y2 Z3 Y4 X11 Z12 X13] +
(0.024353077678069074+0j) [Y2 Z3 Y4 Y11 Z12 Y13] +
(-7.80170750091976e-06+0j) [Y2 Z3 Y4 Z11] +
(3.5390541845155326e-06+0j) [Y2 Z3 Y4 Z12] +
(8.814937306911245e-06+0j) [Y2 Z3 Y4 Z13] +
(1.4548424492427964e-06+0j) [Y2 Z3 Z4 X5 X6 Y7] +
(3.1513463112426866e-06+0j) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11] +
(0.01925750509525162+0j) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.5224930676346074e-06+0j) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10] +
(-0.008541996625454884+0j) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(7.956895374075141e-07+0j) [Y2 Z3 Z4 X5 X8 Y9] +
(4.643051068635042e-06+0j) [Y2 Z3 Z4 X5 X10 Y11] +
(0.019028242443847414+0j) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13] +
(-0.008764827575688855+0j) [Y2 Z3 Z4 X5 X11 Y12] +
(-5.275883122395712e-06+0j) [Y2 Z3 Z4 X5 X12 Y13] +
(-1.4548424492427964e-06+0j) [Y2 Z3 Z4 Y5 X6 X7] +
(-3.1513463112426866e-06+0j) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11] +
(-0.01925750509525162+0j) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.5224930676346074e-06+0j) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10] +
(-0.008541996625454884+0j) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-7.956895374075141e-07+0j) [Y2 Z3 Z4 Y5 X8 X9] +
(-4.643051068635042e-06+0j) [Y2 Z3 Z4 Y5 X10 X11] +
(-0.019028242443847414+0j) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13] +
(-0.008764827575688855+0j) [Y2 Z3 Z4 Y5 Y11 Y12] +
(5.275883122395712e-06+0j) [Y2 Z3 Z4 Y5 X12 X13] +
(-4.561447180471532e-07+0j) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12] +
(-0.12133276911042447+0j) [Y2 Z3 Z4 Z5 Y6] +
(-0.008469978791023871+0j) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(2.6863815441743586e-07+0j) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.008469978791023871+0j) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(2.6863815441743586e-07+0j) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.025996177598021364+0j) [Y2 Z3 Z4 Z5 Y6 Z7] +
(-0.005805188989826893+0j) [Y2 Z3 Z4 Z5 Y6 Z8] +
(-0.017561202409646204+0j) [Y2 Z3 Z4 Z5 Y6 Z9] +
(-3.427323109166914e-07+0j) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12] +
(-7.988770289638443e-07+0j) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12] +
(-0.000814531327095695+0j) [Y2 Z3 Z4 Z5 Y6 Z10] +
(2.745518400484067e-06+0j) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13] +
(2.745518400484067e-06+0j) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13] +
(0.014411099430130903+0j) [Y2 Z3 Z4 Z5 Y6 Z11] +
(0.0006650070219498542+0j) [Y2 Z3 Z4 Z5 Y6 Z12] +
(-0.003493790359890189+0j) [Y2 Z3 Z4 Z5 Y6 Z13] +
(0.011756013419819307+0j) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9] +
(-0.015225630757226598+0j) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11] +
(3.0882507114007586e-06+0j) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(-3.544395429447912e-06+0j) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12] +
(0.004158797381840043+0j) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13] +
(-0.011756013419819307+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9] +
(0.015225630757226598+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11] +
(-3.0882507114007586e-06+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(-3.544395429447912e-06+0j) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12] +
(-0.004158797381840043+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13] +
(0.014603704729162167+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(-2.8742990715356804e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.014603704729162167+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(-2.8742990715356804e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(-0.2816425776702309+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-1.300294656301964e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(-1.300294656301964e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(-0.024282117354693045+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(-0.019538050311314815+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(-0.017091553155898907+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(-0.00244649715541591+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(0.00244649715541591+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(5.7759505275970614e-05+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.8836765762282103e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(5.146496327748959e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(3.846201671446995e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(-0.0393591680220532+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10] +
(7.979825793772298e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(-0.024755463292891033+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10] +
(5.105526722236617e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(-0.021433810721601033+0j) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10] +
(5.1593505022450165e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(-0.0299037895126249+0j) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10] +
(5.4279886566624525e-06+0j) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0016638798784907832+0j) [Y2 Z3 Z4 Y6] +
(-0.018889030304942895+0j) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10] +
(2.9473560117242756e-06+0j) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.0034795118903343013+0j) [Y2 Z3 Z5 Y6] +
(-0.028730779551905526+0j) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10] +
(5.935867718388499e-06+0j) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.602116740625765e-06+0j) [Y2 Y4] +
(0.0004956762314917326+0j) [Y2 Z4 Z5 Y6] +
(-0.03560837898831254+0j) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(7.253273348457477e-06+0j) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.6538942226831692+0j) [Z2] +
(1.602116740625765e-06+0j) [Z2 X3 Z4 X5] +
(0.0004956762314917326+0j) [Z2 X3 Z4 Z5 Z6 X7] +
(-0.03560837898831254+0j) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(7.253273348457477e-06+0j) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.602116740625765e-06+0j) [Z2 Y3 Z4 Y5] +
(0.0004956762314917326+0j) [Z2 Y3 Z4 Z5 Z6 Y7] +
(-0.03560837898831254+0j) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(7.253273348457477e-06+0j) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.18189085790751308+0j) [Z2 Z3] +
(-9.509249751227719e-07+0j) [Z2 X4 Z5 X6] +
(-4.7288431473541736e-06+0j) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.024591860883829923+0j) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-9.509249751227719e-07+0j) [Z2 Y4 Z5 Y6] +
(-4.7288431473541736e-06+0j) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.024591860883829923+0j) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.12495807739503192+0j) [Z2 Z4] +
(-1.170830137004609e-06+0j) [Z2 X5 Z6 X7] +
(-7.089799467836665e-06+0j) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.03490334337366159+0j) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.170830137004609e-06+0j) [Z2 Y5 Z6 Y7] +
(-7.089799467836665e-06+0j) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.03490334337366159+0j) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.16079764534838542+0j) [Z2 Z5] +
(0.019020423173039813+0j) [Z2 X6 Z7 Z8 Z9 X10] +
(-2.1032156047047804e-06+0j) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.019020423173039813+0j) [Z2 Y6 Z7 Z8 Z9 Y10] +
(-2.1032156047047804e-06+0j) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.13739104762683213+0j) [Z2 Z6] +
(0.02438908253114921+0j) [Z2 X7 Z8 Z9 Z10 X11] +
(-2.0111220981844834e-06+0j) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.02438908253114921+0j) [Z2 Y7 Z8 Z9 Z10 Y11] +
(-2.0111220981844834e-06+0j) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.16853486561579917+0j) [Z2 Z7] +
(0.15071408121008262+0j) [Z2 Z8] +
(0.18690820476912512+0j) [Z2 Z9] +
(-1.0632283423207093e-06+0j) [Z2 X10 Z11 X12] +
(-1.0632283423207093e-06+0j) [Z2 Y10 Z11 Y12] +
(0.127995024924684+0j) [Z2 Z10] +
(1.1094407593572654e-06+0j) [Z2 X11 Z12 X13] +
(1.1094407593572654e-06+0j) [Z2 Y11 Z12 Y13] +
(0.1533796824331415+0j) [Z2 Z11] +
(0.14011289865354803+0j) [Z2 Z12] +
(0.1556901067175245+0j) [Z2 Z13] +
(0.005143391768825085+0j) [X3 X4 Y5 Y6] +
(0.009841749246962631+0j) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10] +
(-2.988511706664223e-06+0j) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.4548424492427964e-06+0j) [X3 X4 X6 X7] +
(-1.5224930676346074e-06+0j) [X3 X4 X6 Z7 Z8 Z9 Z10 X11] +
(-0.008541996625454884+0j) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.1513463112426866e-06+0j) [X3 X4 Y7 Z8 Z9 Y10] +
(-0.01925750509525162+0j) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12] +
(-7.956895374075142e-07+0j) [X3 X4 X8 X9] +
(-4.6430510686350405e-06+0j) [X3 X4 X10 X11] +
(-0.008764827575688855+0j) [X3 X4 X10 Z11 Z12 X13] +
(-0.019028242443847414+0j) [X3 X4 Y11 Y12] +
(5.275883122395712e-06+0j) [X3 X4 X12 X13] +
(-0.005143391768825085+0j) [X3 Y4 Y5 X6] +
(-0.009841749246962631+0j) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10] +
(2.988511706664223e-06+0j) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.4548424492427964e-06+0j) [X3 Y4 Y6 X7] +
(-1.5224930676346074e-06+0j) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11] +
(-0.008541996625454884+0j) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.1513463112426866e-06+0j) [X3 Y4 Y7 Z8 Z9 X10] +
(0.01925750509525162+0j) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12] +
(-7.956895374075142e-07+0j) [X3 Y4 Y8 X9] +
(-4.6430510686350405e-06+0j) [X3 Y4 Y10 X11] +
(-0.008764827575688855+0j) [X3 Y4 Y10 Z11 Z12 X13] +
(0.019028242443847414+0j) [X3 Y4 Y11 X12] +
(5.275883122395712e-06+0j) [X3 Y4 Y12 X13] +
(-3.887051675861227e-06+0j) [X3 Z4 X5] +
(3.2118420191802892e-06+0j) [X3 Z4 X5 X6 Z7 Z8 Z9 X10] +
(0.019299560579363825+0j) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.2118420191802892e-06+0j) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10] +
(0.019299560579363825+0j) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.1954890102747775e-06+0j) [X3 Z4 X5 Z6] +
(1.6893489515456822e-06+0j) [X3 Z4 X5 X7 Z8 Z9 Z10 X11] +
(0.010757563953908943+0j) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.5371780953306294e-08+0j) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11] +
(4.2055484112201325e-05+0j) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.593534389680191e-07+0j) [X3 Z4 X5 Z7] +
(-5.770052997875832e-07+0j) [X3 Z4 X5 Z8] +
(2.18684237619931e-07+0j) [X3 Z4 X5 Z9] +
(0.024353077678069074+0j) [X3 Z4 X5 X10 Z11 X12] +
(0.024353077678069074+0j) [X3 Z4 X5 Y10 Z11 Y12] +
(-7.80170750091976e-06+0j) [X3 Z4 X5 Z10] +
(0.015588250102380219+0j) [X3 Z4 X5 X11 Z12 X13] +
(0.005324835234221658+0j) [X3 Z4 X5 Y11 Z12 Y13] +
(-3.1586564322847177e-06+0j) [X3 Z4 X5 Z11] +
(8.814937306911245e-06+0j) [X3 Z4 X5 Z12] +
(3.5390541845155326e-06+0j) [X3 Z4 X5 Z13] +
(1.6288532436080791e-06+0j) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11] +
(0.010715508469796738+0j) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.010263414868158556+0j) [X3 Z4 Y5 Y11 Z12 X13] +
(0.00846997879102387+0j) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10] +
(-2.6863815441743586e-07+0j) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.01175601341981931+0j) [X3 Z4 Z5 X6 X8 X9] +
(0.015225630757226598+0j) [X3 Z4 Z5 X6 X10 X11] +
(-3.544395429447912e-06+0j) [X3 Z4 Z5 X6 X10 Z11 Z12 X13] +
(-3.0882507114007586e-06+0j) [X3 Z4 Z5 X6 Y11 Y12] +
(-0.004158797381840043+0j) [X3 Z4 Z5 X6 X12 X13] +
(-0.00846997879102387+0j) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10] +
(2.6863815441743586e-07+0j) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.01175601341981931+0j) [X3 Z4 Z5 Y6 Y8 X9] +
(0.015225630757226598+0j) [X3 Z4 Z5 Y6 Y10 X11] +
(-3.544395429447912e-06+0j) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13] +
(3.0882507114007586e-06+0j) [X3 Z4 Z5 Y6 Y11 X12] +
(-0.004158797381840043+0j) [X3 Z4 Z5 Y6 Y12 X13] +
(-0.12133276911042447+0j) [X3 Z4 Z5 Z6 X7] +
(-0.017561202409646204+0j) [X3 Z4 Z5 Z6 X7 Z8] +
(-0.005805188989826893+0j) [X3 Z4 Z5 Z6 X7 Z9] +
(2.745518400484067e-06+0j) [X3 Z4 Z5 Z6 X7 X10 Z11 X12] +
(2.745518400484067e-06+0j) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12] +
(0.014411099430130903+0j) [X3 Z4 Z5 Z6 X7 Z10] +
(-7.988770289638443e-07+0j) [X3 Z4 Z5 Z6 X7 X11 Z12 X13] +
(-3.427323109166914e-07+0j) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13] +
(-0.000814531327095695+0j) [X3 Z4 Z5 Z6 X7 Z11] +
(-0.003493790359890189+0j) [X3 Z4 Z5 Z6 X7 Z12] +
(0.0006650070219498542+0j) [X3 Z4 Z5 Z6 X7 Z13] +
(-4.561447180471532e-07+0j) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13] +
(-0.014603704729162167+0j) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10] +
(2.8742990715356804e-06+0j) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(0.014603704729162167+0j) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10] +
(-2.8742990715356804e-06+0j) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(1.300294656301964e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12] +
(0.00244649715541591+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-1.300294656301964e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12] +
(0.00244649715541591+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(-0.281642577670231+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.017091553155898907+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(-0.019538050311314815+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(5.775950527597063e-05+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(2.8836765762282108e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(3.846201671446995e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(-0.024282117354693048+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11] +
(5.146496327748959e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(-0.024755463292891033+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11] +
(5.105526722236617e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(-0.0393591680220532+0j) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11] +
(7.979825793772298e-06+0j) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(-0.0299037895126249+0j) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11] +
(5.4279886566624525e-06+0j) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.025996177598021364+0j) [X3 Z4 Z5 X7] +
(-0.021433810721601033+0j) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11] +
(5.1593505022450165e-06+0j) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0034795118903343013+0j) [X3 Z4 Z6 X7] +
(-0.028730779551905526+0j) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11] +
(5.935867718388499e-06+0j) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.764994121049351e-07+0j) [X3 X5] +
(0.0016638798784907832+0j) [X3 Z5 Z6 X7] +
(-0.018889030304942895+0j) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.9473560117242756e-06+0j) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.005143391768825085+0j) [Y3 X4 X5 Y6] +
(-0.009841749246962631+0j) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10] +
(2.988511706664223e-06+0j) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.4548424492427964e-06+0j) [Y3 X4 X6 Y7] +
(-1.5224930676346074e-06+0j) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11] +
(-0.008541996625454884+0j) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.1513463112426866e-06+0j) [Y3 X4 X7 Z8 Z9 Y10] +
(0.01925750509525162+0j) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12] +
(-7.956895374075142e-07+0j) [Y3 X4 X8 Y9] +
(-4.6430510686350405e-06+0j) [Y3 X4 X10 Y11] +
(-0.008764827575688855+0j) [Y3 X4 X10 Z11 Z12 Y13] +
(0.019028242443847414+0j) [Y3 X4 X11 Y12] +
(5.275883122395712e-06+0j) [Y3 X4 X12 Y13] +
(0.005143391768825085+0j) [Y3 Y4 X5 X6] +
(0.009841749246962631+0j) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10] +
(-2.988511706664223e-06+0j) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.4548424492427964e-06+0j) [Y3 Y4 Y6 Y7] +
(-1.5224930676346074e-06+0j) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11] +
(-0.008541996625454884+0j) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.1513463112426866e-06+0j) [Y3 Y4 X7 Z8 Z9 X10] +
(-0.01925750509525162+0j) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12] +
(-7.956895374075142e-07+0j) [Y3 Y4 Y8 Y9] +
(-4.6430510686350405e-06+0j) [Y3 Y4 Y10 Y11] +
(-0.008764827575688855+0j) [Y3 Y4 Y10 Z11 Z12 Y13] +
(-0.019028242443847414+0j) [Y3 Y4 X11 X12] +
(5.275883122395712e-06+0j) [Y3 Y4 Y12 Y13] +
(1.6288532436080791e-06+0j) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11] +
(0.010715508469796738+0j) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.010263414868158556+0j) [Y3 Z4 X5 X11 Z12 Y13] +
(-3.887051675861227e-06+0j) [Y3 Z4 Y5] +
(3.2118420191802892e-06+0j) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10] +
(0.019299560579363825+0j) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.2118420191802892e-06+0j) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10] +
(0.019299560579363825+0j) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.1954890102747775e-06+0j) [Y3 Z4 Y5 Z6] +
(4.5371780953306294e-08+0j) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11] +
(4.2055484112201325e-05+0j) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.6893489515456822e-06+0j) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11] +
(0.010757563953908943+0j) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.593534389680191e-07+0j) [Y3 Z4 Y5 Z7] +
(-5.770052997875832e-07+0j) [Y3 Z4 Y5 Z8] +
(2.18684237619931e-07+0j) [Y3 Z4 Y5 Z9] +
(0.024353077678069074+0j) [Y3 Z4 Y5 X10 Z11 X12] +
(0.024353077678069074+0j) [Y3 Z4 Y5 Y10 Z11 Y12] +
(-7.80170750091976e-06+0j) [Y3 Z4 Y5 Z10] +
(0.005324835234221658+0j) [Y3 Z4 Y5 X11 Z12 X13] +
(0.015588250102380219+0j) [Y3 Z4 Y5 Y11 Z12 Y13] +
(-3.1586564322847177e-06+0j) [Y3 Z4 Y5 Z11] +
(8.814937306911245e-06+0j) [Y3 Z4 Y5 Z12] +
(3.5390541845155326e-06+0j) [Y3 Z4 Y5 Z13] +
(-0.00846997879102387+0j) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10] +
(2.6863815441743586e-07+0j) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.01175601341981931+0j) [Y3 Z4 Z5 X6 X8 Y9] +
(0.015225630757226598+0j) [Y3 Z4 Z5 X6 X10 Y11] +
(-3.544395429447912e-06+0j) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13] +
(3.0882507114007586e-06+0j) [Y3 Z4 Z5 X6 X11 Y12] +
(-0.004158797381840043+0j) [Y3 Z4 Z5 X6 X12 Y13] +
(0.00846997879102387+0j) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10] +
(-2.6863815441743586e-07+0j) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-0.01175601341981931+0j) [Y3 Z4 Z5 Y6 Y8 Y9] +
(0.015225630757226598+0j) [Y3 Z4 Z5 Y6 Y10 Y11] +
(-3.544395429447912e-06+0j) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13] +
(-3.0882507114007586e-06+0j) [Y3 Z4 Z5 Y6 X11 X12] +
(-0.004158797381840043+0j) [Y3 Z4 Z5 Y6 Y12 Y13] +
(-4.561447180471532e-07+0j) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13] +
(-0.12133276911042447+0j) [Y3 Z4 Z5 Z6 Y7] +
(-0.017561202409646204+0j) [Y3 Z4 Z5 Z6 Y7 Z8] +
(-0.005805188989826893+0j) [Y3 Z4 Z5 Z6 Y7 Z9] +
(2.745518400484067e-06+0j) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12] +
(2.745518400484067e-06+0j) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12] +
(0.014411099430130903+0j) [Y3 Z4 Z5 Z6 Y7 Z10] +
(-3.427323109166914e-07+0j) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13] +
(-7.988770289638443e-07+0j) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13] +
(-0.000814531327095695+0j) [Y3 Z4 Z5 Z6 Y7 Z11] +
(-0.003493790359890189+0j) [Y3 Z4 Z5 Z6 Y7 Z12] +
(0.0006650070219498542+0j) [Y3 Z4 Z5 Z6 Y7 Z13] +
(0.014603704729162167+0j) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10] +
(-2.8742990715356804e-06+0j) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-0.014603704729162167+0j) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10] +
(2.8742990715356804e-06+0j) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-1.300294656301964e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12] +
(0.00244649715541591+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(1.300294656301964e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12] +
(0.00244649715541591+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(-0.281642577670231+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.017091553155898907+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(-0.019538050311314815+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(5.775950527597063e-05+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.8836765762282108e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(3.846201671446995e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(-0.024282117354693048+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11] +
(5.146496327748959e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(-0.024755463292891033+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11] +
(5.105526722236617e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(-0.0393591680220532+0j) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11] +
(7.979825793772298e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(-0.0299037895126249+0j) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11] +
(5.4279886566624525e-06+0j) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.025996177598021364+0j) [Y3 Z4 Z5 Y7] +
(-0.021433810721601033+0j) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11] +
(5.1593505022450165e-06+0j) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0034795118903343013+0j) [Y3 Z4 Z6 Y7] +
(-0.028730779551905526+0j) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11] +
(5.935867718388499e-06+0j) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.764994121049351e-07+0j) [Y3 Y5] +
(0.0016638798784907832+0j) [Y3 Z5 Z6 Y7] +
(-0.018889030304942895+0j) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.9473560117242756e-06+0j) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.6538942226831694+0j) [Z3] +
(-1.170830137004609e-06+0j) [Z3 X4 Z5 X6] +
(-7.089799467836665e-06+0j) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.03490334337366159+0j) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.170830137004609e-06+0j) [Z3 Y4 Z5 Y6] +
(-7.089799467836665e-06+0j) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.03490334337366159+0j) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.16079764534838542+0j) [Z3 Z4] +
(-9.509249751227719e-07+0j) [Z3 X5 Z6 X7] +
(-4.7288431473541736e-06+0j) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.024591860883829923+0j) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-9.509249751227719e-07+0j) [Z3 Y5 Z6 Y7] +
(-4.7288431473541736e-06+0j) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.024591860883829923+0j) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.12495807739503192+0j) [Z3 Z5] +
(0.02438908253114921+0j) [Z3 X6 Z7 Z8 Z9 X10] +
(-2.0111220981844834e-06+0j) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.02438908253114921+0j) [Z3 Y6 Z7 Z8 Z9 Y10] +
(-2.0111220981844834e-06+0j) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.16853486561579917+0j) [Z3 Z6] +
(0.019020423173039813+0j) [Z3 X7 Z8 Z9 Z10 X11] +
(-2.1032156047047804e-06+0j) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.019020423173039813+0j) [Z3 Y7 Z8 Z9 Z10 Y11] +
(-2.1032156047047804e-06+0j) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.13739104762683213+0j) [Z3 Z7] +
(0.18690820476912512+0j) [Z3 Z8] +
(0.15071408121008262+0j) [Z3 Z9] +
(1.1094407593572654e-06+0j) [Z3 X10 Z11 X12] +
(1.1094407593572654e-06+0j) [Z3 Y10 Z11 Y12] +
(0.1533796824331415+0j) [Z3 Z10] +
(-1.0632283423207093e-06+0j) [Z3 X11 Z12 X13] +
(-1.0632283423207093e-06+0j) [Z3 Y11 Z12 Y13] +
(0.127995024924684+0j) [Z3 Z11] +
(0.1556901067175245+0j) [Z3 Z12] +
(0.14011289865354803+0j) [Z3 Z13] +
(-0.011982389010247934+0j) [X4 X5 Y6 Y7] +
(-0.007306759928832958+0j) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11] +
(-2.8882935959874856e-07+0j) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.007306759928832958+0j) [X4 X5 X7 Z8 Z9 X10] +
(-2.8882935959874856e-07+0j) [X4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-0.007156934919856937+0j) [X4 X5 Y8 Y9] +
(-0.01768006795248153+0j) [X4 X5 Y10 Y11] +
(-3.694513294571336e-06+0j) [X4 X5 Y10 Z11 Z12 Y13] +
(-3.694513294571336e-06+0j) [X4 X5 X11 X12] +
(-0.038314670294803906+0j) [X4 X5 Y12 Y13] +
(0.011982389010247934+0j) [X4 Y5 Y6 X7] +
(0.007306759928832958+0j) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11] +
(2.8882935959874856e-07+0j) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.007306759928832958+0j) [X4 Y5 Y7 Z8 Z9 X10] +
(-2.8882935959874856e-07+0j) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(0.007156934919856937+0j) [X4 Y5 Y8 X9] +
(0.01768006795248153+0j) [X4 Y5 Y10 X11] +
(3.694513294571336e-06+0j) [X4 Y5 Y10 Z11 Z12 X13] +
(-3.694513294571336e-06+0j) [X4 Y5 Y11 X12] +
(0.038314670294803906+0j) [X4 Y5 Y12 X13] +
(-1.2260484988919033e-05+0j) [X4 Z5 X6] +
(-1.2283337826144905e-06+0j) [X4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(0.00024636437569587377+0j) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2283337826144905e-06+0j) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(0.00024636437569587377+0j) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.8540608580084018e-06+0j) [X4 Z5 X6 Z7] +
(-1.3980449081154841e-06+0j) [X4 Z5 X6 Z8] +
(-1.8818501832636375e-06+0j) [X4 Z5 X6 Z9] +
(0.007960880725921535+0j) [X4 Z5 X6 X10 Z11 X12] +
(-0.0009298507967730229+0j) [X4 Z5 X6 Y10 Z11 Y12] +
(-1.6923978285318794e-06+0j) [X4 Z5 X6 Z10] +
(-0.01221504099761388+0j) [X4 Z5 X6 X11 Z12 X13] +
(-0.01221504099761388+0j) [X4 Z5 X6 Y11 Z12 Y13] +
(4.281913885183156e-06+0j) [X4 Z5 X6 Z11] +
(-4.588855155759114e-06+0j) [X4 Z5 X6 Z13] +
(0.008890731522694558+0j) [X4 Z5 Y6 Y10 Z11 X12] +
(-4.838052751481535e-07+0j) [X4 Z5 Z6 X7 Y8 Y9] +
(5.974311713715035e-06+0j) [X4 Z5 Z6 X7 Y10 Y11] +
(0.011285190200840855+0j) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(0.020175921723535415+0j) [X4 Z5 Z6 X7 X11 X12] +
(-4.5565692182563635e-06+0j) [X4 Z5 Z6 X7 Y12 Y13] +
(4.838052751481535e-07+0j) [X4 Z5 Z6 Y7 Y8 X9] +
(-5.974311713715035e-06+0j) [X4 Z5 Z6 Y7 Y10 X11] +
(-0.011285190200840855+0j) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(0.020175921723535415+0j) [X4 Z5 Z6 Y7 Y11 X12] +
(4.5565692182563635e-06+0j) [X4 Z5 Z6 Y7 Y12 X13] +
(1.3304731887571244e-06+0j) [X4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(0.0059237983365613475+0j) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(1.3304731887571244e-06+0j) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(0.0059237983365613475+0j) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(-6.631277928614517e-05+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.016024603689179528+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(-0.016024603689179528+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(3.3343312896068007e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(-4.734622038897867e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(-9.80610277552518e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(-5.0714807366273135e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(5.0714807366273135e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-0.3693708936615625+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.023145130929529113+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-0.009612634606847324+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(-0.025637238296026855+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(-8.774817864890834e-06+0j) [X4 Z5 Z6 Z7 Z8 X10] +
(-0.04764261217638312+0j) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(-7.444344676133709e-06+0j) [X4 Z5 Z6 Z7 Z9 X10] +
(-0.041718813839821775+0j) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(-6.29002843332208e-06+0j) [X4 Z5 Z6 Z8 Z9 X10] +
(-0.03956441632289351+0j) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(-7.518362215936571e-06+0j) [X4 Z5 Z7 Z8 Z9 X10] +
(-0.03931805194719763+0j) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(-5.92976581527343e-07+0j) [X4 X6] +
(-4.253224225740007e-06+0j) [X4 Z6 Z7 Z8 Z9 X10] +
(-0.022528440196012967+0j) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.011982389010247934+0j) [Y4 X5 X6 Y7] +
(0.007306759928832958+0j) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11] +
(2.8882935959874856e-07+0j) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.007306759928832958+0j) [Y4 X5 X7 Z8 Z9 Y10] +
(-2.8882935959874856e-07+0j) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(0.007156934919856937+0j) [Y4 X5 X8 Y9] +
(0.01768006795248153+0j) [Y4 X5 X10 Y11] +
(3.694513294571336e-06+0j) [Y4 X5 X10 Z11 Z12 Y13] +
(-3.694513294571336e-06+0j) [Y4 X5 X11 Y12] +
(0.038314670294803906+0j) [Y4 X5 X12 Y13] +
(-0.011982389010247934+0j) [Y4 Y5 X6 X7] +
(-0.007306759928832958+0j) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11] +
(-2.8882935959874856e-07+0j) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.007306759928832958+0j) [Y4 Y5 Y7 Z8 Z9 Y10] +
(-2.8882935959874856e-07+0j) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.007156934919856937+0j) [Y4 Y5 X8 X9] +
(-0.01768006795248153+0j) [Y4 Y5 X10 X11] +
(-3.694513294571336e-06+0j) [Y4 Y5 X10 Z11 Z12 X13] +
(-3.694513294571336e-06+0j) [Y4 Y5 Y11 Y12] +
(-0.038314670294803906+0j) [Y4 Y5 X12 X13] +
(0.008890731522694558+0j) [Y4 Z5 X6 X10 Z11 Y12] +
(-1.2260484988919033e-05+0j) [Y4 Z5 Y6] +
(-1.2283337826144905e-06+0j) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(0.00024636437569587377+0j) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2283337826144905e-06+0j) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(0.00024636437569587377+0j) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.8540608580084018e-06+0j) [Y4 Z5 Y6 Z7] +
(-1.3980449081154841e-06+0j) [Y4 Z5 Y6 Z8] +
(-1.8818501832636375e-06+0j) [Y4 Z5 Y6 Z9] +
(-0.0009298507967730229+0j) [Y4 Z5 Y6 X10 Z11 X12] +
(0.007960880725921535+0j) [Y4 Z5 Y6 Y10 Z11 Y12] +
(-1.6923978285318794e-06+0j) [Y4 Z5 Y6 Z10] +
(-0.01221504099761388+0j) [Y4 Z5 Y6 X11 Z12 X13] +
(-0.01221504099761388+0j) [Y4 Z5 Y6 Y11 Z12 Y13] +
(4.281913885183156e-06+0j) [Y4 Z5 Y6 Z11] +
(-4.588855155759114e-06+0j) [Y4 Z5 Y6 Z13] +
(4.838052751481535e-07+0j) [Y4 Z5 Z6 X7 X8 Y9] +
(-5.974311713715035e-06+0j) [Y4 Z5 Z6 X7 X10 Y11] +
(-0.011285190200840855+0j) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(0.020175921723535415+0j) [Y4 Z5 Z6 X7 X11 Y12] +
(4.5565692182563635e-06+0j) [Y4 Z5 Z6 X7 X12 Y13] +
(-4.838052751481535e-07+0j) [Y4 Z5 Z6 Y7 X8 X9] +
(5.974311713715035e-06+0j) [Y4 Z5 Z6 Y7 X10 X11] +
(0.011285190200840855+0j) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(0.020175921723535415+0j) [Y4 Z5 Z6 Y7 Y11 Y12] +
(-4.5565692182563635e-06+0j) [Y4 Z5 Z6 Y7 X12 X13] +
(1.3304731887571244e-06+0j) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(0.0059237983365613475+0j) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(1.3304731887571244e-06+0j) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(0.0059237983365613475+0j) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(-6.631277928614517e-05+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.016024603689179528+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(-0.016024603689179528+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(3.3343312896068007e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(-4.734622038897867e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(-9.80610277552518e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(5.0714807366273135e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-5.0714807366273135e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-0.3693708936615625+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.023145130929529113+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-0.009612634606847324+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(-0.025637238296026855+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(-8.774817864890834e-06+0j) [Y4 Z5 Z6 Z7 Z8 Y10] +
(-0.04764261217638312+0j) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(-7.444344676133709e-06+0j) [Y4 Z5 Z6 Z7 Z9 Y10] +
(-0.041718813839821775+0j) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(-6.29002843332208e-06+0j) [Y4 Z5 Z6 Z8 Z9 Y10] +
(-0.03956441632289351+0j) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(-7.518362215936571e-06+0j) [Y4 Z5 Z7 Z8 Z9 Y10] +
(-0.03931805194719763+0j) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(-5.92976581527343e-07+0j) [Y4 Y6] +
(-4.253224225740007e-06+0j) [Y4 Z6 Z7 Z8 Z9 Y10] +
(-0.022528440196012967+0j) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.2034402289145634+0j) [Z4] +
(-5.92976581527343e-07+0j) [Z4 X5 Z6 X7] +
(-4.253224225740007e-06+0j) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.022528440196012967+0j) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-5.92976581527343e-07+0j) [Z4 Y5 Z6 Y7] +
(-4.253224225740007e-06+0j) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.022528440196012967+0j) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.15755314797985667+0j) [Z4 Z5] +
(0.0182668348693754+0j) [Z4 X6 Z7 Z8 Z9 X10] +
(-1.6541174771646875e-06+0j) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0182668348693754+0j) [Z4 Y6 Z7 Z8 Z9 Y10] +
(-1.6541174771646875e-06+0j) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.13701191674040764+0j) [Z4 Z6] +
(0.010960074940542444+0j) [Z4 X7 Z8 Z9 Z10 X11] +
(-1.9429468367634364e-06+0j) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.010960074940542444+0j) [Z4 Y7 Z8 Z9 Z10 Y11] +
(-1.9429468367634364e-06+0j) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.14899430575065556+0j) [Z4 Z7] +
(0.14960702684445293+0j) [Z4 Z8] +
(0.15676396176430987+0j) [Z4 Z9] +
(1.8782101247831819e-06+0j) [Z4 X10 Z11 X12] +
(1.8782101247831819e-06+0j) [Z4 Y10 Z11 Y12] +
(0.1248999091723761+0j) [Z4 Z10] +
(-1.8163031697881547e-06+0j) [Z4 X11 Z12 X13] +
(-1.8163031697881547e-06+0j) [Z4 Y11 Z12 Y13] +
(0.14257997712485765+0j) [Z4 Z11] +
(0.1138357367938867+0j) [Z4 Z12] +
(0.1521504070886906+0j) [Z4 Z13] +
(1.2283337826144905e-06+0j) [X5 X6 Y7 Z8 Z9 Y10] +
(-0.00024636437569587377+0j) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-4.838052751481535e-07+0j) [X5 X6 X8 X9] +
(5.9743117137150366e-06+0j) [X5 X6 X10 X11] +
(0.020175921723535415+0j) [X5 X6 X10 Z11 Z12 X13] +
(0.011285190200840855+0j) [X5 X6 Y11 Y12] +
(-4.556569218256364e-06+0j) [X5 X6 X12 X13] +
(-1.2283337826144905e-06+0j) [X5 Y6 Y7 Z8 Z9 X10] +
(0.00024636437569587377+0j) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-4.838052751481535e-07+0j) [X5 Y6 Y8 X9] +
(5.9743117137150366e-06+0j) [X5 Y6 Y10 X11] +
(0.020175921723535415+0j) [X5 Y6 Y10 Z11 Z12 X13] +
(-0.011285190200840855+0j) [X5 Y6 Y11 X12] +
(-4.556569218256364e-06+0j) [X5 Y6 Y12 X13] +
(-1.2260484988919043e-05+0j) [X5 Z6 X7] +
(-1.8818501832636375e-06+0j) [X5 Z6 X7 Z8] +
(-1.3980449081154841e-06+0j) [X5 Z6 X7 Z9] +
(-0.01221504099761388+0j) [X5 Z6 X7 X10 Z11 X12] +
(-0.01221504099761388+0j) [X5 Z6 X7 Y10 Z11 Y12] +
(4.281913885183156e-06+0j) [X5 Z6 X7 Z10] +
(0.007960880725921535+0j) [X5 Z6 X7 X11 Z12 X13] +
(-0.0009298507967730229+0j) [X5 Z6 X7 Y11 Z12 Y13] +
(-1.6923978285318794e-06+0j) [X5 Z6 X7 Z11] +
(-4.588855155759114e-06+0j) [X5 Z6 X7 Z12] +
(0.008890731522694558+0j) [X5 Z6 Y7 Y11 Z12 X13] +
(-1.3304731887571244e-06+0j) [X5 Z6 Z7 X8 Y9 Y10] +
(-0.0059237983365613475+0j) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(1.3304731887571244e-06+0j) [X5 Z6 Z7 Y8 Y9 X10] +
(0.0059237983365613475+0j) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(0.016024603689179528+0j) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12] +
(-5.071480736627314e-06+0j) [X5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-0.016024603689179528+0j) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12] +
(-5.071480736627314e-06+0j) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(-6.631277928614517e-05+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-9.80610277552518e-06+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(-4.734622038897867e-06+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(-0.3693708936615624+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.023145130929529113+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(-0.025637238296026855+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(3.3343312896068007e-06+0j) [X5 Z6 Z7 Z8 Z9 X11] +
(-0.009612634606847324+0j) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(-7.444344676133709e-06+0j) [X5 Z6 Z7 Z8 Z10 X11] +
(-0.041718813839821775+0j) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(-8.774817864890834e-06+0j) [X5 Z6 Z7 Z9 Z10 X11] +
(-0.04764261217638312+0j) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(-7.518362215936571e-06+0j) [X5 Z6 Z8 Z9 Z10 X11] +
(-0.03931805194719763+0j) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.854060858008402e-06+0j) [X5 X7] +
(-6.29002843332208e-06+0j) [X5 Z7 Z8 Z9 Z10 X11] +
(-0.03956441632289351+0j) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2283337826144905e-06+0j) [Y5 X6 X7 Z8 Z9 Y10] +
(0.00024636437569587377+0j) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-4.838052751481535e-07+0j) [Y5 X6 X8 Y9] +
(5.9743117137150366e-06+0j) [Y5 X6 X10 Y11] +
(0.020175921723535415+0j) [Y5 X6 X10 Z11 Z12 Y13] +
(-0.011285190200840855+0j) [Y5 X6 X11 Y12] +
(-4.556569218256364e-06+0j) [Y5 X6 X12 Y13] +
(1.2283337826144905e-06+0j) [Y5 Y6 X7 Z8 Z9 X10] +
(-0.00024636437569587377+0j) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-4.838052751481535e-07+0j) [Y5 Y6 Y8 Y9] +
(5.9743117137150366e-06+0j) [Y5 Y6 Y10 Y11] +
(0.020175921723535415+0j) [Y5 Y6 Y10 Z11 Z12 Y13] +
(0.011285190200840855+0j) [Y5 Y6 X11 X12] +
(-4.556569218256364e-06+0j) [Y5 Y6 Y12 Y13] +
(0.008890731522694558+0j) [Y5 Z6 X7 X11 Z12 Y13] +
(-1.2260484988919043e-05+0j) [Y5 Z6 Y7] +
(-1.8818501832636375e-06+0j) [Y5 Z6 Y7 Z8] +
(-1.3980449081154841e-06+0j) [Y5 Z6 Y7 Z9] +
(-0.01221504099761388+0j) [Y5 Z6 Y7 X10 Z11 X12] +
(-0.01221504099761388+0j) [Y5 Z6 Y7 Y10 Z11 Y12] +
(4.281913885183156e-06+0j) [Y5 Z6 Y7 Z10] +
(-0.0009298507967730229+0j) [Y5 Z6 Y7 X11 Z12 X13] +
(0.007960880725921535+0j) [Y5 Z6 Y7 Y11 Z12 Y13] +
(-1.6923978285318794e-06+0j) [Y5 Z6 Y7 Z11] +
(-4.588855155759114e-06+0j) [Y5 Z6 Y7 Z12] +
(1.3304731887571244e-06+0j) [Y5 Z6 Z7 X8 X9 Y10] +
(0.0059237983365613475+0j) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-1.3304731887571244e-06+0j) [Y5 Z6 Z7 Y8 X9 X10] +
(-0.0059237983365613475+0j) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-0.016024603689179528+0j) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12] +
(-5.071480736627314e-06+0j) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(0.016024603689179528+0j) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12] +
(-5.071480736627314e-06+0j) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(-6.631277928614517e-05+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-9.80610277552518e-06+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(-4.734622038897867e-06+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(-0.3693708936615624+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.023145130929529113+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(-0.025637238296026855+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(3.3343312896068007e-06+0j) [Y5 Z6 Z7 Z8 Z9 Y11] +
(-0.009612634606847324+0j) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(-7.444344676133709e-06+0j) [Y5 Z6 Z7 Z8 Z10 Y11] +
(-0.041718813839821775+0j) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(-8.774817864890834e-06+0j) [Y5 Z6 Z7 Z9 Z10 Y11] +
(-0.04764261217638312+0j) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(-7.518362215936571e-06+0j) [Y5 Z6 Z8 Z9 Z10 Y11] +
(-0.03931805194719763+0j) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.854060858008402e-06+0j) [Y5 Y7] +
(-6.29002843332208e-06+0j) [Y5 Z7 Z8 Z9 Z10 Y11] +
(-0.03956441632289351+0j) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.2034402289145636+0j) [Z5] +
(0.010960074940542444+0j) [Z5 X6 Z7 Z8 Z9 X10] +
(-1.9429468367634364e-06+0j) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.010960074940542444+0j) [Z5 Y6 Z7 Z8 Z9 Y10] +
(-1.9429468367634364e-06+0j) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.14899430575065556+0j) [Z5 Z6] +
(0.0182668348693754+0j) [Z5 X7 Z8 Z9 Z10 X11] +
(-1.6541174771646875e-06+0j) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0182668348693754+0j) [Z5 Y7 Z8 Z9 Z10 Y11] +
(-1.6541174771646875e-06+0j) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.13701191674040764+0j) [Z5 Z7] +
(0.15676396176430987+0j) [Z5 Z8] +
(0.14960702684445293+0j) [Z5 Z9] +
(-1.8163031697881547e-06+0j) [Z5 X10 Z11 X12] +
(-1.8163031697881547e-06+0j) [Z5 Y10 Z11 Y12] +
(0.14257997712485765+0j) [Z5 Z10] +
(1.8782101247831819e-06+0j) [Z5 X11 Z12 X13] +
(1.8782101247831819e-06+0j) [Z5 Y11 Z12 Y13] +
(0.1248999091723761+0j) [Z5 Z11] +
(0.1521504070886906+0j) [Z5 Z12] +
(0.1138357367938867+0j) [Z5 Z13] +
(-0.013873381748426176+0j) [X6 X7 Y8 Y9] +
(-0.017825140995786335+0j) [X6 X7 Y10 Y11] +
(-1.0358477602912778e-06+0j) [X6 X7 Y10 Z11 Z12 Y13] +
(-1.0358477602912778e-06+0j) [X6 X7 X11 X12] +
(-0.01736611899465143+0j) [X6 X7 Y12 Y13] +
(0.013873381748426176+0j) [X6 Y7 Y8 X9] +
(0.017825140995786335+0j) [X6 Y7 Y10 X11] +
(1.0358477602912778e-06+0j) [X6 Y7 Y10 Z11 Z12 X13] +
(-1.0358477602912778e-06+0j) [X6 Y7 Y11 X12] +
(0.01736611899465143+0j) [X6 Y7 Y12 X13] +
(0.00029219862611112846+0j) [X6 Z7 X8 X9 Z10 X11] +
(-3.328139351018249e-07+0j) [X6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.00029219862611112846+0j) [X6 Z7 X8 Y9 Z10 Y11] +
(-3.328139351018249e-07+0j) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(0.22848106564918627+0j) [X6 Z7 Z8 Z9 X10] +
(3.3131455004053552e-06+0j) [X6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(3.3131455004053552e-06+0j) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(0.011307274008848116+0j) [X6 Z7 Z8 Z9 X10 Z11] +
(0.025104957138844426+0j) [X6 Z7 Z8 Z9 X10 Z12] +
(0.010540425907671421+0j) [X6 Z7 Z8 Z9 X10 Z13] +
(-0.014564531231173008+0j) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(0.014564531231173008+0j) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-2.5950860069718607e-05+0j) [X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.183932559624061e-06+0j) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-6.5243738488920305e-06+0j) [X6 Z7 Z8 Z9 Z10 X12] +
(-3.2112283484866753e-06+0j) [X6 Z7 Z8 Z9 Z11 X12] +
(0.029812424517345545+0j) [X6 Z7 Z8 X10] +
(-3.2774831955297125e-06+0j) [X6 Z7 Z8 Z10 Z11 X12] +
(0.030104623143456674+0j) [X6 Z7 Z9 X10] +
(-3.6102971306315375e-06+0j) [X6 Z7 Z9 Z10 Z11 X12] +
(0.030787505389143835+0j) [X6 Z8 Z9 X10] +
(-3.7696594520255588e-06+0j) [X6 Z8 Z9 Z10 Z11 X12] +
(0.013873381748426176+0j) [Y6 X7 X8 Y9] +
(0.017825140995786335+0j) [Y6 X7 X10 Y11] +
(1.0358477602912778e-06+0j) [Y6 X7 X10 Z11 Z12 Y13] +
(-1.0358477602912778e-06+0j) [Y6 X7 X11 Y12] +
(0.01736611899465143+0j) [Y6 X7 X12 Y13] +
(-0.013873381748426176+0j) [Y6 Y7 X8 X9] +
(-0.017825140995786335+0j) [Y6 Y7 X10 X11] +
(-1.0358477602912778e-06+0j) [Y6 Y7 X10 Z11 Z12 X13] +
(-1.0358477602912778e-06+0j) [Y6 Y7 Y11 Y12] +
(-0.01736611899465143+0j) [Y6 Y7 X12 X13] +
(0.00029219862611112846+0j) [Y6 Z7 Y8 X9 Z10 X11] +
(-3.328139351018249e-07+0j) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.00029219862611112846+0j) [Y6 Z7 Y8 Y9 Z10 Y11] +
(-3.328139351018249e-07+0j) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(0.22848106564918627+0j) [Y6 Z7 Z8 Z9 Y10] +
(3.3131455004053552e-06+0j) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(3.3131455004053552e-06+0j) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(0.011307274008848116+0j) [Y6 Z7 Z8 Z9 Y10 Z11] +
(0.025104957138844426+0j) [Y6 Z7 Z8 Z9 Y10 Z12] +
(0.010540425907671421+0j) [Y6 Z7 Z8 Z9 Y10 Z13] +
(0.014564531231173008+0j) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-0.014564531231173008+0j) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-2.5950860069718607e-05+0j) [Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.183932559624061e-06+0j) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-6.5243738488920305e-06+0j) [Y6 Z7 Z8 Z9 Z10 Y12] +
(-3.2112283484866753e-06+0j) [Y6 Z7 Z8 Z9 Z11 Y12] +
(0.029812424517345545+0j) [Y6 Z7 Z8 Y10] +
(-3.2774831955297125e-06+0j) [Y6 Z7 Z8 Z10 Z11 Y12] +
(0.030104623143456674+0j) [Y6 Z7 Z9 Y10] +
(-3.6102971306315375e-06+0j) [Y6 Z7 Z9 Z10 Z11 Y12] +
(0.030787505389143835+0j) [Y6 Z8 Z9 Y10] +
(-3.7696594520255588e-06+0j) [Y6 Z8 Z9 Z10 Z11 Y12] +
(1.3096862988615432+0j) [Z6] +
(0.030787505389143835+0j) [Z6 X7 Z8 Z9 Z10 X11] +
(-3.7696594520255588e-06+0j) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.030787505389143835+0j) [Z6 Y7 Z8 Z9 Z10 Y11] +
(-3.7696594520255588e-06+0j) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.19392534613270268+0j) [Z6 Z7] +
(0.16756653265461288+0j) [Z6 Z8] +
(0.18143991440303905+0j) [Z6 Z9] +
(-1.8551201215396418e-06+0j) [Z6 X10 Z11 X12] +
(-1.8551201215396418e-06+0j) [Z6 Y10 Z11 Y12] +
(0.119524389646827+0j) [Z6 Z10] +
(-2.89096788183092e-06+0j) [Z6 X11 Z12 X13] +
(-2.89096788183092e-06+0j) [Z6 Y11 Z12 Y13] +
(0.13734953064261335+0j) [Z6 Z11] +
(0.13401715261963731+0j) [Z6 Z12] +
(0.15138327161428872+0j) [Z6 Z13] +
(-0.00029219862611112846+0j) [X7 X8 Y9 Y10] +
(3.3281393510182486e-07+0j) [X7 X8 Y9 Z10 Z11 Y12] +
(0.00029219862611112846+0j) [X7 Y8 Y9 X10] +
(-3.3281393510182486e-07+0j) [X7 Y8 Y9 Z10 Z11 X12] +
(-3.3131455004053552e-06+0j) [X7 Z8 Z9 X10 Y11 Y12] +
(-0.014564531231173008+0j) [X7 Z8 Z9 X10 X12 X13] +
(3.3131455004053552e-06+0j) [X7 Z8 Z9 Y10 Y11 X12] +
(-0.014564531231173008+0j) [X7 Z8 Z9 Y10 Y12 X13] +
(0.22848106564918635+0j) [X7 Z8 Z9 Z10 X11] +
(0.010540425907671421+0j) [X7 Z8 Z9 Z10 X11 Z12] +
(0.025104957138844426+0j) [X7 Z8 Z9 Z10 X11 Z13] +
(-2.5950860069718597e-05+0j) [X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.18393255962406e-06+0j) [X7 Z8 Z9 Z10 Z11 X13] +
(-3.2112283484866753e-06+0j) [X7 Z8 Z9 Z10 Z12 X13] +
(0.011307274008848116+0j) [X7 Z8 Z9 X11] +
(-6.5243738488920305e-06+0j) [X7 Z8 Z9 Z11 Z12 X13] +
(0.030104623143456674+0j) [X7 Z8 Z10 X11] +
(-3.6102971306315375e-06+0j) [X7 Z8 Z10 Z11 Z12 X13] +
(0.029812424517345545+0j) [X7 Z9 Z10 X11] +
(-3.2774831955297125e-06+0j) [X7 Z9 Z10 Z11 Z12 X13] +
(0.00029219862611112846+0j) [Y7 X8 X9 Y10] +
(-3.3281393510182486e-07+0j) [Y7 X8 X9 Z10 Z11 Y12] +
(-0.00029219862611112846+0j) [Y7 Y8 X9 X10] +
(3.3281393510182486e-07+0j) [Y7 Y8 X9 Z10 Z11 X12] +
(3.3131455004053552e-06+0j) [Y7 Z8 Z9 X10 X11 Y12] +
(-0.014564531231173008+0j) [Y7 Z8 Z9 X10 X12 Y13] +
(-3.3131455004053552e-06+0j) [Y7 Z8 Z9 Y10 X11 X12] +
(-0.014564531231173008+0j) [Y7 Z8 Z9 Y10 Y12 Y13] +
(0.22848106564918635+0j) [Y7 Z8 Z9 Z10 Y11] +
(0.010540425907671421+0j) [Y7 Z8 Z9 Z10 Y11 Z12] +
(0.025104957138844426+0j) [Y7 Z8 Z9 Z10 Y11 Z13] +
(-2.5950860069718597e-05+0j) [Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(4.18393255962406e-06+0j) [Y7 Z8 Z9 Z10 Z11 Y13] +
(-3.2112283484866753e-06+0j) [Y7 Z8 Z9 Z10 Z12 Y13] +
(0.011307274008848116+0j) [Y7 Z8 Z9 Y11] +
(-6.5243738488920305e-06+0j) [Y7 Z8 Z9 Z11 Z12 Y13] +
(0.030104623143456674+0j) [Y7 Z8 Z10 Y11] +
(-3.6102971306315375e-06+0j) [Y7 Z8 Z10 Z11 Z12 Y13] +
(0.029812424517345545+0j) [Y7 Z9 Z10 Y11] +
(-3.2774831955297125e-06+0j) [Y7 Z9 Z10 Z11 Z12 Y13] +
(1.3096862988615436+0j) [Z7] +
(0.18143991440303905+0j) [Z7 Z8] +
(0.16756653265461288+0j) [Z7 Z9] +
(-2.89096788183092e-06+0j) [Z7 X10 Z11 X12] +
(-2.89096788183092e-06+0j) [Z7 Y10 Z11 Y12] +
(0.13734953064261335+0j) [Z7 Z10] +
(-1.8551201215396418e-06+0j) [Z7 X11 Z12 X13] +
(-1.8551201215396418e-06+0j) [Z7 Y11 Z12 Y13] +
(0.119524389646827+0j) [Z7 Z11] +
(0.15138327161428872+0j) [Z7 Z12] +
(0.13401715261963731+0j) [Z7 Z13] +
(-0.009560705729135999+0j) [X8 X9 Y10 Y11] +
(6.628614202165169e-07+0j) [X8 X9 Y10 Z11 Z12 Y13] +
(6.628614202165167e-07+0j) [X8 X9 X11 X12] +
(-0.006087822480561868+0j) [X8 X9 Y12 Y13] +
(0.009560705729135999+0j) [X8 Y9 Y10 X11] +
(-6.628614202165169e-07+0j) [X8 Y9 Y10 Z11 Z12 X13] +
(6.628614202165167e-07+0j) [X8 Y9 Y11 X12] +
(0.006087822480561868+0j) [X8 Y9 Y12 X13] +
(0.009560705729135999+0j) [Y8 X9 X10 Y11] +
(-6.628614202165169e-07+0j) [Y8 X9 X10 Z11 Z12 Y13] +
(6.628614202165167e-07+0j) [Y8 X9 X11 Y12] +
(0.006087822480561868+0j) [Y8 X9 X12 Y13] +
(-0.009560705729135999+0j) [Y8 Y9 X10 X11] +
(6.628614202165169e-07+0j) [Y8 Y9 X10 Z11 Z12 X13] +
(6.628614202165167e-07+0j) [Y8 Y9 Y11 Y12] +
(-0.006087822480561868+0j) [Y8 Y9 X12 X13] +
(1.3693525634718167+0j) [Z8] +
(0.2200397733437609+0j) [Z8 Z9] +
(-1.5973171978040146e-06+0j) [Z8 X10 Z11 X12] +
(-1.5973171978040146e-06+0j) [Z8 Y10 Z11 Y12] +
(0.13766872645852585+0j) [Z8 Z10] +
(-9.344557775874979e-07+0j) [Z8 X11 Z12 X13] +
(-9.344557775874979e-07+0j) [Z8 Y11 Z12 Y13] +
(0.14722943218766182+0j) [Z8 Z11] +
(0.1497348680349694+0j) [Z8 Z12] +
(0.15582269051553127+0j) [Z8 Z13] +
(1.369352563471817+0j) [Z9] +
(-9.344557775874979e-07+0j) [Z9 X10 Z11 X12] +
(-9.344557775874979e-07+0j) [Z9 Y10 Z11 Y12] +
(0.14722943218766182+0j) [Z9 Z10] +
(-1.5973171978040146e-06+0j) [Z9 X11 Z12 X13] +
(-1.5973171978040146e-06+0j) [Z9 Y11 Z12 Y13] +
(0.13766872645852585+0j) [Z9 Z11] +
(0.15582269051553127+0j) [Z9 Z12] +
(0.1497348680349694+0j) [Z9 Z13] +
(-0.02868518371610595+0j) [X10 X11 Y12 Y13] +
(0.02868518371610595+0j) [X10 Y11 Y12 X13] +
(-1.0722312157054516e-05+0j) [X10 Z11 X12] +
(7.954413176624898e-06+0j) [X10 Z11 X12 Z13] +
(-8.194261372584273e-06+0j) [X10 X12] +
(0.02868518371610595+0j) [Y10 X11 X12 Y13] +
(-0.02868518371610595+0j) [Y10 Y11 X12 X13] +
(-1.0722312157054516e-05+0j) [Y10 Z11 Y12] +
(7.954413176624898e-06+0j) [Y10 Z11 Y12 Z13] +
(-8.194261372584273e-06+0j) [Y10 Y12] +
(0.7829661725950187+0j) [Z10] +
(-8.194261372584273e-06+0j) [Z10 X11 Z12 X13] +
(-8.194261372584273e-06+0j) [Z10 Y11 Z12 Y13] +
(0.14926355147388926+0j) [Z10 Z11] +
(0.11270386920332232+0j) [Z10 Z12] +
(0.14138905291942827+0j) [Z10 Z13] +
(-1.0722312157054511e-05+0j) [X11 Z12 X13] +
(7.954413176624898e-06+0j) [X11 X13] +
(-1.0722312157054511e-05+0j) [Y11 Z12 Y13] +
(7.954413176624898e-06+0j) [Y11 Y13] +
(0.7829661725950188+0j) [Z11] +
(0.14138905291942827+0j) [Z11 Z12] +
(0.11270386920332232+0j) [Z11 Z13] +
(0.8084581961720485+0j) [Z12] +
(0.15435748657223655+0j) [Z12 Z13] +
(0.8084581961720487+0j) [Z13]
Number of qubits: 14
Qubit Hamiltonian
  (-46.46390678868892) [I0]
+ (0.7829661725950201) [Z10]
+ (0.7829661725950201) [Z11]
+ (0.80845819617205) [Z12]
+ (0.8084581961720501) [Z13]
+ (1.2034402289145647) [Z4]
+ (1.2034402289145647) [Z5]
+ (1.3096862988615432) [Z7]
+ (1.3096862988615436) [Z6]
+ (1.369352563471819) [Z8]
+ (1.3693525634718198) [Z9]
+ (1.653894222683171) [Z2]
+ (1.6538942226831712) [Z3]
+ (12.412630742111766) [Z0]
+ (12.412630742111766) [Z1]
+ (-8.194261371265773e-06) [Y10 Y12]
+ (-8.194261371265773e-06) [X10 X12]
+ (-1.8540608581314767e-06) [Y5 Y7]
+ (-1.8540608581314767e-06) [X5 X7]
+ (-7.764994118958317e-07) [Y3 Y5]
+ (-7.764994118958317e-07) [X3 X5]
+ (-5.929765817069853e-07) [Y4 Y6]
+ (-5.929765817069853e-07) [X4 X6]
+ (1.6021167406396052e-06) [Y2 Y4]
+ (1.6021167406396052e-06) [X2 X4]
+ (7.954413175425421e-06) [Y11 Y13]
+ (7.954413175425421e-06) [X11 X13]
+ (0.0032769719312316396) [Y1 Y3]
+ (0.0032769719312316396) [X1 X3]
+ (0.10433064780651416) [Y0 Y2]
+ (0.10433064780651416) [X0 X2]
+ (0.11270386920332202) [Z10 Z12]
+ (0.11270386920332202) [Z11 Z13]
+ (0.11383573679388652) [Z4 Z12]
+ (0.11383573679388652) [Z5 Z13]
+ (0.1195243896468266) [Z6 Z10]
+ (0.1195243896468266) [Z7 Z11]
+ (0.12489990917237588) [Z4 Z10]
+ (0.12489990917237588) [Z5 Z11]
+ (0.12495807739503204) [Z2 Z4]
+ (0.12495807739503204) [Z3 Z5]
+ (0.12799502492468387) [Z2 Z10]
+ (0.12799502492468387) [Z3 Z11]
+ (0.13401715261963695) [Z6 Z12]
+ (0.13401715261963695) [Z7 Z13]
+ (0.13701191674040744) [Z4 Z6]
+ (0.13701191674040744) [Z5 Z7]
+ (0.13734953064261307) [Z6 Z11]
+ (0.13734953064261307) [Z7 Z10]
+ (0.1373910476268321) [Z2 Z6]
+ (0.1373910476268321) [Z3 Z7]
+ (0.13766872645852563) [Z8 Z10]
+ (0.13766872645852563) [Z9 Z11]
+ (0.14011289865354792) [Z2 Z12]
+ (0.14011289865354792) [Z3 Z13]
+ (0.14138905291942788) [Z10 Z13]
+ (0.14138905291942788) [Z11 Z12]
+ (0.14257997712485743) [Z4 Z11]
+ (0.14257997712485743) [Z5 Z10]
+ (0.14722943218766155) [Z8 Z11]
+ (0.14722943218766155) [Z9 Z10]
+ (0.1489943057506554) [Z4 Z7]
+ (0.1489943057506554) [Z5 Z6]
+ (0.14926355147388878) [Z10 Z11]
+ (0.14960702684445298) [Z4 Z8]
+ (0.14960702684445298) [Z5 Z9]
+ (0.14973486803496922) [Z8 Z12]
+ (0.14973486803496922) [Z9 Z13]
+ (0.15071408121008273) [Z2 Z8]
+ (0.15071408121008273) [Z3 Z9]
+ (0.15138327161428833) [Z6 Z13]
+ (0.15138327161428833) [Z7 Z12]
+ (0.15215040708869038) [Z4 Z13]
+ (0.15215040708869038) [Z5 Z12]
+ (0.1533796824331412) [Z2 Z11]
+ (0.1533796824331412) [Z3 Z10]
+ (0.15435748657223614) [Z12 Z13]
+ (0.15569010671752437) [Z2 Z13]
+ (0.15569010671752437) [Z3 Z12]
+ (0.15582269051553105) [Z8 Z13]
+ (0.15582269051553105) [Z9 Z12]
+ (0.1567639617643099) [Z4 Z9]
+ (0.1567639617643099) [Z5 Z8]
+ (0.15755314797985662) [Z4 Z5]
+ (0.16079764534838545) [Z2 Z5]
+ (0.16079764534838545) [Z3 Z4]
+ (0.1675665326546126) [Z6 Z8]
+ (0.1675665326546126) [Z7 Z9]
+ (0.16853486561579922) [Z2 Z7]
+ (0.16853486561579922) [Z3 Z6]
+ (0.1814399144030387) [Z6 Z9]
+ (0.1814399144030387) [Z7 Z8]
+ (0.18189085790751328) [Z2 Z3]
+ (0.18690820476912534) [Z2 Z9]
+ (0.18690820476912534) [Z3 Z8]
+ (0.19299723935364205) [Z0 Z10]
+ (0.19299723935364205) [Z1 Z11]
+ (0.19392534613270182) [Z6 Z7]
+ (0.1966177089034212) [Z0 Z4]
+ (0.1966177089034212) [Z1 Z5]
+ (0.199363545373608) [Z0 Z5]
+ (0.199363545373608) [Z1 Z4]
+ (0.20072866460441735) [Z0 Z11]
+ (0.20072866460441735) [Z1 Z10]
+ (0.21102659849791489) [Z0 Z12]
+ (0.21102659849791489) [Z1 Z13]
+ (0.21631037498631786) [Z0 Z13]
+ (0.21631037498631786) [Z1 Z12]
+ (0.2200397733437609) [Z8 Z9]
+ (0.23671080783830387) [Z0 Z2]
+ (0.23671080783830387) [Z1 Z3]
+ (0.2416466393601717) [Z0 Z6]
+ (0.2416466393601717) [Z1 Z7]
+ (0.24853483371314228) [Z0 Z7]
+ (0.24853483371314228) [Z1 Z6]
+ (0.25129445674591655) [Z0 Z3]
+ (0.25129445674591655) [Z1 Z2]
+ (0.2723251830660566) [Z0 Z8]
+ (0.2723251830660566) [Z1 Z9]
+ (0.27883454426723386) [Z0 Z9]
+ (0.27883454426723386) [Z1 Z8]
+ (1.1861763734860475) [Z0 Z1]
+ (-1.2260484989991275e-05) [Y4 Z5 Y6]
+ (-1.2260484989991275e-05) [X4 Z5 X6]
+ (-1.2260484989991269e-05) [Y5 Z6 Y7]
+ (-1.2260484989991269e-05) [X5 Z6 X7]
+ (-1.0722312156883441e-05) [Y11 Z12 Y13]
+ (-1.0722312156883441e-05) [X11 Z12 X13]
+ (-1.072231215688344e-05) [Y10 Z11 Y12]
+ (-1.072231215688344e-05) [X10 Z11 X12]
+ (-3.887051674017731e-06) [Y2 Z3 Y4]
+ (-3.887051674017731e-06) [X2 Z3 X4]
+ (-3.8870516740177295e-06) [Y3 Z4 Y5]
+ (-3.8870516740177295e-06) [X3 Z4 X5]
+ (0.1250703257977233) [Y0 Z1 Y2]
+ (0.1250703257977233) [X0 Z1 X2]
+ (0.1250703257977233) [Y1 Z2 Y3]
+ (0.1250703257977233) [X1 Z2 X3]
+ (-0.038314670294803864) [Y4 Y5 X12 X13]
+ (-0.038314670294803864) [X4 X5 Y12 Y13]
+ (-0.036194123559042606) [Y2 Y3 X8 X9]
+ (-0.036194123559042606) [X2 X3 Y8 Y9]
+ (-0.035839567953353406) [Y2 Y3 X4 X5]
+ (-0.035839567953353406) [X2 X3 Y4 Y5]
+ (-0.031143817988967124) [Y2 Y3 X6 X7]
+ (-0.031143817988967124) [X2 X3 Y6 Y7]
+ (-0.02868518371610587) [Y10 Y11 X12 X13]
+ (-0.02868518371610587) [X10 X11 Y12 Y13]
+ (-0.025996177598021114) [Y3 Z4 Z5 Y7]
+ (-0.025996177598021114) [X3 Z4 Z5 X7]
+ (-0.02538465750845734) [Y2 Y3 X10 X11]
+ (-0.02538465750845734) [X2 X3 Y10 Y11]
+ (-0.019028242443847238) [Y3 Y4 X11 X12]
+ (-0.019028242443847238) [X3 X4 Y11 Y12]
+ (-0.017825140995786474) [Y6 Y7 X10 X11]
+ (-0.017825140995786474) [X6 X7 Y10 Y11]
+ (-0.01768006795248152) [Y4 Y5 X10 X11]
+ (-0.01768006795248152) [X4 X5 Y10 Y11]
+ (-0.01736611899465137) [Y6 Y7 X12 X13]
+ (-0.01736611899465137) [X6 X7 Y12 Y13]
+ (-0.015577208063976443) [Y2 Y3 X12 X13]
+ (-0.015577208063976443) [X2 X3 Y12 Y13]
+ (-0.014583648907612724) [Y0 Y1 X2 X3]
+ (-0.014583648907612724) [X0 X1 Y2 Y3]
+ (-0.013873381748426077) [Y6 Y7 X8 X9]
+ (-0.013873381748426077) [X6 X7 Y8 Y9]
+ (-0.011982389010247953) [Y4 Y5 X6 X7]
+ (-0.011982389010247953) [X4 X5 Y6 Y7]
+ (-0.011285190200840907) [Y5 X6 X11 Y12]
+ (-0.011285190200840907) [X5 Y6 Y11 X12]
+ (-0.009560705729135923) [Y8 Y9 X10 X11]
+ (-0.009560705729135923) [X8 X9 Y10 Y11]
+ (-0.008125251921381046) [Y1 X2 X8 Y9]
+ (-0.008125251921381046) [Y1 Y2 Y8 Y9]
+ (-0.008125251921381046) [X1 X2 X8 X9]
+ (-0.008125251921381046) [X1 Y2 Y8 X9]
+ (-0.007731425250775312) [Y0 Y1 X10 X11]
+ (-0.007731425250775312) [X0 X1 Y10 Y11]
+ (-0.0071569349198569426) [Y4 Y5 X8 X9]
+ (-0.0071569349198569426) [X4 X5 Y8 Y9]
+ (-0.006888194352970557) [Y0 Y1 X6 X7]
+ (-0.006888194352970557) [X0 X1 Y6 Y7]
+ (-0.006509361201177241) [Y0 Y1 X8 X9]
+ (-0.006509361201177241) [X0 X1 Y8 Y9]
+ (-0.00608782248056186) [Y8 Y9 X12 X13]
+ (-0.00608782248056186) [X8 X9 Y12 Y13]
+ (-0.005283776488402962) [Y0 Y1 X12 X13]
+ (-0.005283776488402962) [X0 X1 Y12 Y13]
+ (-0.005143391768825138) [Y3 X4 X5 Y6]
+ (-0.005143391768825138) [X3 Y4 Y5 X6]
+ (-0.004684903388155217) [Y1 X2 X6 Y7]
+ (-0.004684903388155217) [Y1 Y2 Y6 Y7]
+ (-0.004684903388155217) [X1 X2 X6 X7]
+ (-0.004684903388155217) [X1 Y2 Y6 X7]
+ (-0.004575007626639207) [Y1 X2 X12 Y13]
+ (-0.004575007626639207) [Y1 Y2 Y12 Y13]
+ (-0.004575007626639207) [X1 X2 X12 X13]
+ (-0.004575007626639207) [X1 Y2 Y12 X13]
+ (-0.00442485544944186) [Y1 X2 X4 Y5]
+ (-0.00442485544944186) [Y1 Y2 Y4 Y5]
+ (-0.00442485544944186) [X1 X2 X4 X5]
+ (-0.00442485544944186) [X1 Y2 Y4 X5]
+ (-0.0034795118903343165) [Y2 Z3 Z5 Y6]
+ (-0.0034795118903343165) [X2 Z3 Z5 X6]
+ (-0.0034795118903343165) [Y3 Z4 Z6 Y7]
+ (-0.0034795118903343165) [X3 Z4 Z6 X7]
+ (-0.0027458364701868107) [Y0 Y1 X4 X5]
+ (-0.0027458364701868107) [X0 X1 Y4 Y5]
+ (-0.001799219493663005) [Y1 X2 X10 Y11]
+ (-0.001799219493663005) [Y1 Y2 Y10 Y11]
+ (-0.001799219493663005) [X1 X2 X10 X11]
+ (-0.001799219493663005) [X1 Y2 Y10 X11]
+ (-0.00029219862611103874) [Y7 Y8 X9 X10]
+ (-0.00029219862611103874) [X7 X8 Y9 Y10]
+ (-8.194261371265773e-06) [Z10 Y11 Z12 Y13]
+ (-8.194261371265773e-06) [Z10 X11 Z12 X13]
+ (-7.801707499572791e-06) [Y2 Z3 Y4 Z11]
+ (-7.801707499572791e-06) [X2 Z3 X4 Z11]
+ (-7.801707499572791e-06) [Y3 Z4 Y5 Z10]
+ (-7.801707499572791e-06) [X3 Z4 X5 Z10]
+ (-4.64305106791888e-06) [Y3 X4 X10 Y11]
+ (-4.64305106791888e-06) [Y3 Y4 Y10 Y11]
+ (-4.64305106791888e-06) [X3 X4 X10 X11]
+ (-4.64305106791888e-06) [X3 Y4 Y10 X11]
+ (-4.588855155293474e-06) [Y4 Z5 Y6 Z13]
+ (-4.588855155293474e-06) [X4 Z5 X6 Z13]
+ (-4.588855155293474e-06) [Y5 Z6 Y7 Z12]
+ (-4.588855155293474e-06) [X5 Z6 X7 Z12]
+ (-4.556569217519392e-06) [Y5 X6 X12 Y13]
+ (-4.556569217519392e-06) [Y5 Y6 Y12 Y13]
+ (-4.556569217519392e-06) [X5 X6 X12 X13]
+ (-4.556569217519392e-06) [X5 Y6 Y12 X13]
+ (-3.694513294000249e-06) [Y4 X5 X11 Y12]
+ (-3.694513294000249e-06) [Y4 Y5 Y11 Y12]
+ (-3.694513294000249e-06) [X4 X5 X11 X12]
+ (-3.694513294000249e-06) [X4 Y5 Y11 X12]
+ (-3.3440815567404786e-06) [Z0 Y5 Z6 Y7]
+ (-3.3440815567404786e-06) [Z0 X5 Z6 X7]
+ (-3.3440815567404786e-06) [Z1 Y4 Z5 Y6]
+ (-3.3440815567404786e-06) [Z1 X4 Z5 X6]
+ (-3.1586564316539102e-06) [Y2 Z3 Y4 Z10]
+ (-3.1586564316539102e-06) [X2 Z3 X4 Z10]
+ (-3.1586564316539102e-06) [Y3 Z4 Y5 Z11]
+ (-3.1586564316539102e-06) [X3 Z4 X5 Z11]
+ (-3.0993492438440066e-06) [Z0 Y4 Z5 Y6]
+ (-3.0993492438440066e-06) [Z0 X4 Z5 X6]
+ (-3.0993492438440066e-06) [Z1 Y5 Z6 Y7]
+ (-3.0993492438440066e-06) [Z1 X5 Z6 X7]
+ (-2.8909678814525525e-06) [Z6 Y11 Z12 Y13]
+ (-2.8909678814525525e-06) [Z6 X11 Z12 X13]
+ (-2.8909678814525525e-06) [Z7 Y10 Z11 Y12]
+ (-2.8909678814525525e-06) [Z7 X10 Z11 X12]
+ (-2.1776646046572356e-06) [Z0 Y10 Z11 Y12]
+ (-2.1776646046572356e-06) [Z0 X10 Z11 X12]
+ (-2.1776646046572356e-06) [Z1 Y11 Z12 Y13]
+ (-2.1776646046572356e-06) [Z1 X11 Z12 X13]
+ (-1.88185018337197e-06) [Y4 Z5 Y6 Z9]
+ (-1.88185018337197e-06) [X4 Z5 X6 Z9]
+ (-1.88185018337197e-06) [Y5 Z6 Y7 Z8]
+ (-1.88185018337197e-06) [X5 Z6 X7 Z8]
+ (-1.8551201212263152e-06) [Z6 Y10 Z11 Y12]
+ (-1.8551201212263152e-06) [Z6 X10 Z11 X12]
+ (-1.8551201212263152e-06) [Z7 Y11 Z12 Y13]
+ (-1.8551201212263152e-06) [Z7 X11 Z12 X13]
+ (-1.854060858131477e-06) [Y4 Z5 Y6 Z7]
+ (-1.854060858131477e-06) [X4 Z5 X6 Z7]
+ (-1.816303169490883e-06) [Z4 Y11 Z12 Y13]
+ (-1.816303169490883e-06) [Z4 X11 Z12 X13]
+ (-1.816303169490883e-06) [Z5 Y10 Z11 Y12]
+ (-1.816303169490883e-06) [Z5 X10 Z11 X12]
+ (-1.6923978284852553e-06) [Y4 Z5 Y6 Z10]
+ (-1.6923978284852553e-06) [X4 Z5 X6 Z10]
+ (-1.6923978284852553e-06) [Y5 Z6 Y7 Z11]
+ (-1.6923978284852553e-06) [X5 Z6 X7 Z11]
+ (-1.6148794135450254e-06) [Z0 Y11 Z12 Y13]
+ (-1.6148794135450254e-06) [Z0 X11 Z12 X13]
+ (-1.6148794135450254e-06) [Z1 Y10 Z11 Y12]
+ (-1.6148794135450254e-06) [Z1 X10 Z11 X12]
+ (-1.5973171975634865e-06) [Z8 Y10 Z11 Y12]
+ (-1.5973171975634865e-06) [Z8 X10 Z11 X12]
+ (-1.5973171975634865e-06) [Z9 Y11 Z12 Y13]
+ (-1.5973171975634865e-06) [Z9 X11 Z12 X13]
+ (-1.4548424491606586e-06) [Y3 X4 X6 Y7]
+ (-1.4548424491606586e-06) [Y3 Y4 Y6 Y7]
+ (-1.4548424491606586e-06) [X3 X4 X6 X7]
+ (-1.4548424491606586e-06) [X3 Y4 Y6 X7]
+ (-1.3980449082510977e-06) [Y4 Z5 Y6 Z8]
+ (-1.3980449082510977e-06) [X4 Z5 X6 Z8]
+ (-1.3980449082510977e-06) [Y5 Z6 Y7 Z9]
+ (-1.3980449082510977e-06) [X5 Z6 X7 Z9]
+ (-1.195489010080914e-06) [Y2 Z3 Y4 Z7]
+ (-1.195489010080914e-06) [X2 Z3 X4 Z7]
+ (-1.195489010080914e-06) [Y3 Z4 Y5 Z6]
+ (-1.195489010080914e-06) [X3 Z4 X5 Z6]
+ (-1.1908508085264566e-06) [Z0 Y3 Z4 Y5]
+ (-1.1908508085264566e-06) [Z0 X3 Z4 X5]
+ (-1.1908508085264566e-06) [Z1 Y2 Z3 Y4]
+ (-1.1908508085264566e-06) [Z1 X2 Z3 X4]
+ (-1.1708301371334494e-06) [Z2 Y5 Z6 Y7]
+ (-1.1708301371334494e-06) [Z2 X5 Z6 X7]
+ (-1.1708301371334494e-06) [Z3 Y4 Z5 Y6]
+ (-1.1708301371334494e-06) [Z3 X4 Z5 X6]
+ (-1.0632283421047847e-06) [Z2 Y10 Z11 Y12]
+ (-1.0632283421047847e-06) [Z2 X10 Z11 X12]
+ (-1.0632283421047847e-06) [Z3 Y11 Z12 Y13]
+ (-1.0632283421047847e-06) [Z3 X11 Z12 X13]
+ (-1.0358477602262376e-06) [Y6 X7 X11 Y12]
+ (-1.0358477602262376e-06) [Y6 Y7 Y11 Y12]
+ (-1.0358477602262376e-06) [X6 X7 X11 X12]
+ (-1.0358477602262376e-06) [X6 Y7 Y11 X12]
+ (-9.509249752767806e-07) [Z2 Y4 Z5 Y6]
+ (-9.509249752767806e-07) [Z2 X4 Z5 X6]
+ (-9.509249752767806e-07) [Z3 Y5 Z6 Y7]
+ (-9.509249752767806e-07) [Z3 X5 Z6 X7]
+ (-9.344557774630524e-07) [Z8 Y11 Z12 Y13]
+ (-9.344557774630524e-07) [Z8 X11 Z12 X13]
+ (-9.344557774630524e-07) [Z9 Y10 Z11 Y12]
+ (-9.344557774630524e-07) [Z9 X10 Z11 X12]
+ (-8.337746755907143e-07) [Z0 Y2 Z3 Y4]
+ (-8.337746755907143e-07) [Z0 X2 Z3 X4]
+ (-8.337746755907143e-07) [Z1 Y3 Z4 Y5]
+ (-8.337746755907143e-07) [Z1 X3 Z4 X5]
+ (-7.95689537312934e-07) [Y3 X4 X8 Y9]
+ (-7.95689537312934e-07) [Y3 Y4 Y8 Y9]
+ (-7.95689537312934e-07) [X3 X4 X8 X9]
+ (-7.95689537312934e-07) [X3 Y4 Y8 X9]
+ (-7.764994118958317e-07) [Y2 Z3 Y4 Z5]
+ (-7.764994118958317e-07) [X2 Z3 X4 Z5]
+ (-5.929765817069853e-07) [Z4 Y5 Z6 Y7]
+ (-5.929765817069853e-07) [Z4 X5 Z6 X7]
+ (-5.770052995989079e-07) [Y2 Z3 Y4 Z9]
+ (-5.770052995989079e-07) [X2 Z3 X4 Z9]
+ (-5.770052995989079e-07) [Y3 Z4 Y5 Z8]
+ (-5.770052995989079e-07) [X3 Z4 X5 Z8]
+ (-5.47164774413403e-07) [Y1 Y2 X11 X12]
+ (-5.47164774413403e-07) [X1 X2 Y11 Y12]
+ (-4.838052751208722e-07) [Y5 X6 X8 Y9]
+ (-4.838052751208722e-07) [Y5 Y6 Y8 Y9]
+ (-4.838052751208722e-07) [X5 X6 X8 X9]
+ (-4.838052751208722e-07) [X5 Y6 Y8 X9]
+ (-3.570761329357422e-07) [Y0 X1 X3 Y4]
+ (-3.570761329357422e-07) [Y0 Y1 Y3 Y4]
+ (-3.570761329357422e-07) [X0 X1 X3 X4]
+ (-3.570761329357422e-07) [X0 Y1 Y3 X4]
+ (-2.447323128964723e-07) [Y0 X1 X5 Y6]
+ (-2.447323128964723e-07) [Y0 Y1 Y5 Y6]
+ (-2.447323128964723e-07) [X0 X1 X5 X6]
+ (-2.447323128964723e-07) [X0 Y1 Y5 X6]
+ (-2.19905161856669e-07) [Y2 X3 X5 Y6]
+ (-2.19905161856669e-07) [Y2 Y3 Y5 Y6]
+ (-2.19905161856669e-07) [X2 X3 X5 X6]
+ (-2.19905161856669e-07) [X2 Y3 Y5 X6]
+ (-1.9332412772416081e-07) [Y1 X2 X3 Y4]
+ (-1.9332412772416081e-07) [X1 Y2 Y3 X4]
+ (-1.291969486290224e-07) [Y1 Z2 Z3 Y5]
+ (-1.291969486290224e-07) [X1 Z2 Z3 X5]
+ (1.7379332624950845e-07) [Y0 Z1 Z3 Y4]
+ (1.7379332624950845e-07) [X0 Z1 Z3 X4]
+ (1.7379332624950845e-07) [Y1 Z2 Z4 Y5]
+ (1.7379332624950845e-07) [X1 Z2 Z4 X5]
+ (1.9332412772416081e-07) [Y1 Y2 X3 X4]
+ (1.9332412772416081e-07) [X1 X2 Y3 Y4]
+ (2.1868423771402616e-07) [Y2 Z3 Y4 Z8]
+ (2.1868423771402616e-07) [X2 Z3 X4 Z8]
+ (2.1868423771402616e-07) [Y3 Z4 Y5 Z9]
+ (2.1868423771402616e-07) [X3 Z4 X5 Z9]
+ (2.5935343907974454e-07) [Y2 Z3 Y4 Z6]
+ (2.5935343907974454e-07) [X2 Z3 X4 Z6]
+ (2.5935343907974454e-07) [Y3 Z4 Y5 Z7]
+ (2.5935343907974454e-07) [X3 Z4 X5 Z7]
+ (3.606071868158054e-07) [Y0 Z1 Z2 Y4]
+ (3.606071868158054e-07) [X0 Z1 Z2 X4]
+ (3.606071868158054e-07) [Y1 Z3 Z4 Y5]
+ (3.606071868158054e-07) [X1 Z3 Z4 X5]
+ (5.47164774413403e-07) [Y1 X2 X11 Y12]
+ (5.47164774413403e-07) [X1 Y2 Y11 X12]
+ (5.627851911122099e-07) [Y0 X1 X11 Y12]
+ (5.627851911122099e-07) [Y0 Y1 Y11 Y12]
+ (5.627851911122099e-07) [X0 X1 X11 X12]
+ (5.627851911122099e-07) [X0 Y1 Y11 X12]
+ (6.628614201004345e-07) [Y8 X9 X11 Y12]
+ (6.628614201004345e-07) [Y8 Y9 Y11 Y12]
+ (6.628614201004345e-07) [X8 X9 X11 X12]
+ (6.628614201004345e-07) [X8 Y9 Y11 X12]
+ (1.1094407592260139e-06) [Z2 Y11 Z12 Y13]
+ (1.1094407592260139e-06) [Z2 X11 Z12 X13]
+ (1.1094407592260139e-06) [Z3 Y10 Z11 Y12]
+ (1.1094407592260139e-06) [Z3 X10 Z11 X12]
+ (1.6021167406396052e-06) [Z2 Y3 Z4 Y5]
+ (1.6021167406396052e-06) [Z2 X3 Z4 X5]
+ (1.8782101245093662e-06) [Z4 Y10 Z11 Y12]
+ (1.8782101245093662e-06) [Z4 X10 Z11 X12]
+ (1.8782101245093662e-06) [Z5 Y11 Z12 Y13]
+ (1.8782101245093662e-06) [Z5 X11 Z12 X13]
+ (2.172669101330798e-06) [Y2 X3 X11 Y12]
+ (2.172669101330798e-06) [Y2 Y3 Y11 Y12]
+ (2.172669101330798e-06) [X2 X3 X11 X12]
+ (2.172669101330798e-06) [X2 Y3 Y11 X12]
+ (3.1174479463132206e-06) [Y0 Z2 Z3 Y4]
+ (3.1174479463132206e-06) [X0 Z2 Z3 X4]
+ (3.539054184083227e-06) [Y2 Z3 Y4 Z12]
+ (3.539054184083227e-06) [X2 Z3 X4 Z12]
+ (3.539054184083227e-06) [Y3 Z4 Y5 Z13]
+ (3.539054184083227e-06) [X3 Z4 X5 Z13]
+ (4.281913884334636e-06) [Y4 Z5 Y6 Z11]
+ (4.281913884334636e-06) [X4 Z5 X6 Z11]
+ (4.281913884334636e-06) [Y5 Z6 Y7 Z10]
+ (4.281913884334636e-06) [X5 Z6 X7 Z10]
+ (5.2758831215934065e-06) [Y3 X4 X12 Y13]
+ (5.2758831215934065e-06) [Y3 Y4 Y12 Y13]
+ (5.2758831215934065e-06) [X3 X4 X12 X13]
+ (5.2758831215934065e-06) [X3 Y4 Y12 X13]
+ (5.9743117128198905e-06) [Y5 X6 X10 Y11]
+ (5.9743117128198905e-06) [Y5 Y6 Y10 Y11]
+ (5.9743117128198905e-06) [X5 X6 X10 X11]
+ (5.9743117128198905e-06) [X5 Y6 Y10 X11]
+ (7.954413175425423e-06) [Y10 Z11 Y12 Z13]
+ (7.954413175425423e-06) [X10 Z11 X12 Z13]
+ (8.814937305676634e-06) [Y2 Z3 Y4 Z13]
+ (8.814937305676634e-06) [X2 Z3 X4 Z13]
+ (8.814937305676634e-06) [Y3 Z4 Y5 Z12]
+ (8.814937305676634e-06) [X3 Z4 X5 Z12]
+ (0.00029219862611103874) [Y7 X8 X9 Y10]
+ (0.00029219862611103874) [X7 Y8 Y9 X10]
+ (0.000495676231491633) [Y2 Z4 Z5 Y6]
+ (0.000495676231491633) [X2 Z4 Z5 X6]
+ (0.0011059037691896626) [Y0 Z1 Y2 Z5]
+ (0.0011059037691896626) [X0 Z1 X2 Z5]
+ (0.0011059037691896626) [Y1 Z2 Y3 Z4]
+ (0.0011059037691896626) [X1 Z2 X3 Z4]
+ (0.0016638798784908214) [Y2 Z3 Z4 Y6]
+ (0.0016638798784908214) [X2 Z3 Z4 X6]
+ (0.0016638798784908214) [Y3 Z5 Z6 Y7]
+ (0.0016638798784908214) [X3 Z5 Z6 X7]
+ (0.0017560707018412236) [Y0 Z1 Y2 Z11]
+ (0.0017560707018412236) [X0 Z1 X2 Z11]
+ (0.0017560707018412236) [Y1 Z2 Y3 Z10]
+ (0.0017560707018412236) [X1 Z2 X3 Z10]
+ (0.002326230623158068) [Y0 Z1 Y2 Z13]
+ (0.002326230623158068) [X0 Z1 X2 Z13]
+ (0.002326230623158068) [Y1 Z2 Y3 Z12]
+ (0.002326230623158068) [X1 Z2 X3 Z12]
+ (0.0027458364701868107) [Y0 X1 X4 Y5]
+ (0.0027458364701868107) [X0 Y1 Y4 X5]
+ (0.002929768674751024) [Y0 Z1 Y2 Z9]
+ (0.002929768674751024) [X0 Z1 X2 Z9]
+ (0.002929768674751024) [Y1 Z2 Y3 Z8]
+ (0.002929768674751024) [X1 Z2 X3 Z8]
+ (0.00327697193123164) [Y0 Z1 Y2 Z3]
+ (0.00327697193123164) [X0 Z1 X2 Z3]
+ (0.003347617530666161) [Y0 Z1 Y2 Z7]
+ (0.003347617530666161) [X0 Z1 X2 Z7]
+ (0.003347617530666161) [Y1 Z2 Y3 Z6]
+ (0.003347617530666161) [X1 Z2 X3 Z6]
+ (0.0035552901955042283) [Y0 Z1 Y2 Z10]
+ (0.0035552901955042283) [X0 Z1 X2 Z10]
+ (0.0035552901955042283) [Y1 Z2 Y3 Z11]
+ (0.0035552901955042283) [X1 Z2 X3 Z11]
+ (0.005143391768825138) [Y3 Y4 X5 X6]
+ (0.005143391768825138) [X3 X4 Y5 Y6]
+ (0.005283776488402962) [Y0 X1 X12 Y13]
+ (0.005283776488402962) [X0 Y1 Y12 X13]
+ (0.005530759218631522) [Y0 Z1 Y2 Z4]
+ (0.005530759218631522) [X0 Z1 X2 Z4]
+ (0.005530759218631522) [Y1 Z2 Y3 Z5]
+ (0.005530759218631522) [X1 Z2 X3 Z5]
+ (0.00608782248056186) [Y8 X9 X12 Y13]
+ (0.00608782248056186) [X8 Y9 Y12 X13]
+ (0.006509361201177241) [Y0 X1 X8 Y9]
+ (0.006509361201177241) [X0 Y1 Y8 X9]
+ (0.006888194352970557) [Y0 X1 X6 Y7]
+ (0.006888194352970557) [X0 Y1 Y6 X7]
+ (0.006901238249797275) [Y0 Z1 Y2 Z12]
+ (0.006901238249797275) [X0 Z1 X2 Z12]
+ (0.006901238249797275) [Y1 Z2 Y3 Z13]
+ (0.006901238249797275) [X1 Z2 X3 Z13]
+ (0.0071569349198569426) [Y4 X5 X8 Y9]
+ (0.0071569349198569426) [X4 Y5 Y8 X9]
+ (0.007731425250775312) [Y0 X1 X10 Y11]
+ (0.007731425250775312) [X0 Y1 Y10 X11]
+ (0.008032520918821376) [Y0 Z1 Y2 Z6]
+ (0.008032520918821376) [X0 Z1 X2 Z6]
+ (0.008032520918821376) [Y1 Z2 Y3 Z7]
+ (0.008032520918821376) [X1 Z2 X3 Z7]
+ (0.009560705729135923) [Y8 X9 X10 Y11]
+ (0.009560705729135923) [X8 Y9 Y10 X11]
+ (0.011055020596132071) [Y0 Z1 Y2 Z8]
+ (0.011055020596132071) [X0 Z1 X2 Z8]
+ (0.011055020596132071) [Y1 Z2 Y3 Z9]
+ (0.011055020596132071) [X1 Z2 X3 Z9]
+ (0.011285190200840907) [Y5 Y6 X11 X12]
+ (0.011285190200840907) [X5 X6 Y11 Y12]
+ (0.01130727400884818) [Y7 Z8 Z9 Y11]
+ (0.01130727400884818) [X7 Z8 Z9 X11]
+ (0.011982389010247953) [Y4 X5 X6 Y7]
+ (0.011982389010247953) [X4 Y5 Y6 X7]
+ (0.013873381748426077) [Y6 X7 X8 Y9]
+ (0.013873381748426077) [X6 Y7 Y8 X9]
+ (0.014583648907612724) [Y0 X1 X2 Y3]
+ (0.014583648907612724) [X0 Y1 Y2 X3]
+ (0.015577208063976443) [Y2 X3 X12 Y13]
+ (0.015577208063976443) [X2 Y3 Y12 X13]
+ (0.01736611899465137) [Y6 X7 X12 Y13]
+ (0.01736611899465137) [X6 Y7 Y12 X13]
+ (0.01768006795248152) [Y4 X5 X10 Y11]
+ (0.01768006795248152) [X4 Y5 Y10 X11]
+ (0.017825140995786474) [Y6 X7 X10 Y11]
+ (0.017825140995786474) [X6 Y7 Y10 X11]
+ (0.019028242443847238) [Y3 X4 X11 Y12]
+ (0.019028242443847238) [X3 Y4 Y11 X12]
+ (0.02538465750845734) [Y2 X3 X10 Y11]
+ (0.02538465750845734) [X2 Y3 Y10 X11]
+ (0.02868518371610587) [Y10 X11 X12 Y13]
+ (0.02868518371610587) [X10 Y11 Y12 X13]
+ (0.02981242451734581) [Y6 Z7 Z8 Y10]
+ (0.02981242451734581) [X6 Z7 Z8 X10]
+ (0.02981242451734581) [Y7 Z9 Z10 Y11]
+ (0.02981242451734581) [X7 Z9 Z10 X11]
+ (0.030104623143456848) [Y6 Z7 Z9 Y10]
+ (0.030104623143456848) [X6 Z7 Z9 X10]
+ (0.030104623143456848) [Y7 Z8 Z10 Y11]
+ (0.030104623143456848) [X7 Z8 Z10 X11]
+ (0.030787505389143946) [Y6 Z8 Z9 Y10]
+ (0.030787505389143946) [X6 Z8 Z9 X10]
+ (0.031143817988967124) [Y2 X3 X6 Y7]
+ (0.031143817988967124) [X2 Y3 Y6 X7]
+ (0.035839567953353406) [Y2 X3 X4 Y5]
+ (0.035839567953353406) [X2 Y3 Y4 X5]
+ (0.036194123559042606) [Y2 X3 X8 Y9]
+ (0.036194123559042606) [X2 Y3 Y8 X9]
+ (0.038314670294803864) [Y4 X5 X12 Y13]
+ (0.038314670294803864) [X4 Y5 Y12 X13]
+ (0.10433064780651416) [Z0 Y1 Z2 Y3]
+ (0.10433064780651416) [Z0 X1 Z2 X3]
+ (-0.12133276911042268) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133276911042268) [X2 Z3 Z4 Z5 X6]
+ (-0.12133276911042258) [Y3 Z4 Z5 Z6 Y7]
+ (-0.12133276911042258) [X3 Z4 Z5 Z6 X7]
+ (3.202076880196519e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.202076880196519e-06) [X1 Z2 Z3 Z4 X5]
+ (3.2020768801965195e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.2020768801965195e-06) [X0 Z1 Z2 Z3 X4]
+ (0.2284810656491887) [Y7 Z8 Z9 Z10 Y11]
+ (0.2284810656491887) [X7 Z8 Z9 Z10 X11]
+ (0.2284810656491888) [Y6 Z7 Z8 Z9 Y10]
+ (0.2284810656491888) [X6 Z7 Z8 Z9 X10]
+ (-0.03276765782329041) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.03276765782329041) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.03276765782329041) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.03276765782329041) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845273062) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.027115036845273062) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845273062) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.027115036845273062) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.025996177598021114) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.025996177598021114) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.017561202409646117) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.017561202409646117) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.017561202409646117) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.017561202409646117) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.014564531231172975) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.014564531231172975) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.014564531231172975) [X7 Z8 Z9 X10 X12 X13]
+ (-0.014564531231172975) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.012215040997613953) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.012215040997613953) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.012215040997613953) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.012215040997613953) [X4 Z5 X6 X11 Z12 X13]
+ (-0.012215040997613953) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.012215040997613953) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.012215040997613953) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.012215040997613953) [X5 Z6 X7 X10 Z11 X12]
+ (-0.011756013419819219) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.011756013419819219) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.011756013419819219) [X3 Z4 Z5 X6 X8 X9]
+ (-0.011756013419819219) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.008764827575688732) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.008764827575688732) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.008764827575688732) [X2 Z3 Z4 X5 X11 X12]
+ (-0.008764827575688732) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.008764827575688732) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.008764827575688732) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.008764827575688732) [X3 X4 X10 Z11 Z12 X13]
+ (-0.008764827575688732) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.008125251921381046) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.008125251921381046) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.007306759928832976) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.007306759928832976) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.007306759928832976) [X4 X5 X7 Z8 Z9 X10]
+ (-0.007306759928832976) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.005805188989826898) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.005805188989826898) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.005805188989826898) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.005805188989826898) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.005652620978017347) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.005652620978017347) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.005652620978017347) [X0 X1 X3 Z4 Z5 X6]
+ (-0.005652620978017347) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.005143391768825138) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.005143391768825138) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.005143391768825138) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.005143391768825138) [X2 Z3 X4 X5 Z6 X7]
+ (-0.004684903388155217) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.004684903388155217) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668620318776307) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.004668620318776307) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.004575007626639208) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.004575007626639208) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.00442485544944186) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.00442485544944186) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.004158797381840065) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.004158797381840065) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.004158797381840065) [X3 Z4 Z5 X6 X12 X13]
+ (-0.004158797381840065) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.003493790359890133) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.003493790359890133) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.003493790359890133) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.003493790359890133) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.0027790267990255315) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.0027790267990255315) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.0022939566113524715) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.0022939566113524715) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.001799219493663005) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.001799219493663005) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.0017278753941369785) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.0017278753941369785) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.000929850796773048) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.000929850796773048) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.000929850796773048) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.000929850796773048) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533856254125439) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.0008533856254125439) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.0008145313270956443) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.0008145313270956443) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.0008145313270956443) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.0008145313270956443) [X3 Z4 Z5 Z6 X7 Z11]
+ (-7.735036880590801e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.735036880590801e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.735036880590801e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.735036880590801e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-8.774817863620579e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.774817863620579e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.774817863620579e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.774817863620579e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.518362214909445e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.518362214909445e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.518362214909445e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.518362214909445e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.4443446750516686e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.4443446750516686e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.4443446750516686e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.4443446750516686e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.524373847733292e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.524373847733292e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.524373847733292e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.524373847733292e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-6.2900284323371874e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.2900284323371874e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.2900284323371874e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.2900284323371874e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.9743117128198905e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (-5.9743117128198905e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (-5.275883121593407e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.275883121593407e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.64305106791888e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.64305106791888e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.556569217519392e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (-4.556569217519392e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (-4.253224225180554e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.253224225180554e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-3.769659451237846e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.769659451237846e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.694513294000249e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.694513294000249e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.610297129855416e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.610297129855416e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.610297129855416e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.610297129855416e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.3131454999000433e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.3131454999000433e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.277483194786914e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.277483194786914e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.277483194786914e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.277483194786914e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.2112283478332478e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.2112283478332478e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.2112283478332478e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.2112283478332478e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.151346310779012e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.151346310779012e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.0882507108814977e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.0882507108814977e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.172669101330798e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.172669101330798e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.4548424491606586e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.4548424491606586e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.330473188568911e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.330473188568911e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.2283337825722566e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.2283337825722566e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.0358477602262376e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.0358477602262376e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-7.95689537312934e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.95689537312934e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.733197741118205e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.733197741118205e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.733197741118205e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.733197741118205e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.628614201004342e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.628614201004342e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.556281913983721e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.556281913983721e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.556281913983721e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.556281913983721e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.418291574088104e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.418291574088104e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.418291574088104e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.418291574088104e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.927453081777569e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.927453081777569e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.927453081777569e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.927453081777569e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.627851911122099e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.627851911122099e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.287660624280344e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.287660624280344e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.287660624280344e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.287660624280344e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.287660624280344e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.287660624280344e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.287660624280344e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.287660624280344e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.838052751208722e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (-4.838052751208722e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (-3.570761329357422e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.570761329357422e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.328139350685017e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.328139350685017e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.086826565355608e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.086826565355608e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.086826565355608e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.086826565355608e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.4473231289647237e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (-2.4473231289647237e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (-2.3713289480550654e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.3713289480550654e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.3713289480550654e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.3713289480550654e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.19905161856669e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-2.19905161856669e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-1.933241277241608e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.933241277241608e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.933241277241608e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.933241277241608e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.8394209156046416e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.8394209156046416e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.8394209156046416e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.8394209156046416e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.5510539177003106e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (-1.5510539177003106e-07) [X0 Z1 X2 X4 Z5 X6]
+ (-1.5510539177003106e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (-1.5510539177003106e-07) [X1 Z2 X3 X5 Z6 X7]
+ (-1.380778148189225e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (-1.380778148189225e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (-1.380778148189225e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (-1.380778148189225e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (-1.380778148189225e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (-1.380778148189225e-07) [X0 Z1 X2 X5 Z6 X7]
+ (-1.380778148189225e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (-1.380778148189225e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (-1.380778148189225e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (-1.380778148189225e-07) [X1 Z2 X3 X4 Z5 X6]
+ (-1.380778148189225e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (-1.380778148189225e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (-1.291969486290224e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.291969486290224e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.107632559828725e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.107632559828725e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.107632559828725e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.107632559828725e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.107632559828725e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.107632559828725e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.107632559828725e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.107632559828725e-07) [X1 Z2 X3 X10 Z11 X12]
+ (8.057446593406361e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.057446593406361e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.057446593406361e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.057446593406361e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.649310136017495e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.649310136017495e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.649310136017495e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.649310136017495e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.839420915604641e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.839420915604641e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.839420915604641e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.839420915604641e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.19905161856669e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (2.19905161856669e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (2.4473231289647237e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (2.4473231289647237e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.236259961656814e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.236259961656814e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.236259961656814e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.236259961656814e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.328139350685017e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.328139350685017e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.570761329357422e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.570761329357422e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.838052751208722e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (4.838052751208722e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (5.627851911122099e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.627851911122099e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (6.628614201004342e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.628614201004342e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (7.95689537312934e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.95689537312934e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.306536651074336e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.306536651074336e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.306536651074336e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.306536651074336e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.0358477602262376e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.0358477602262376e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.2283337825722566e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.2283337825722566e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.2393363216429944e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.2393363216429944e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.2393363216429944e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.2393363216429944e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.330473188568911e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.330473188568911e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.4548424491606586e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.4548424491606586e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.172669101330798e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.172669101330798e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.0882507108814977e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.0882507108814977e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.1174479463132206e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.1174479463132206e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.151346310779012e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.151346310779012e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.3131454999000433e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.3131454999000433e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.3343312890210605e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.3343312890210605e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.694513294000249e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.694513294000249e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (4.183932559161012e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.183932559161012e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.556569217519392e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (4.556569217519392e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (4.64305106791888e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.64305106791888e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.275883121593407e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.275883121593407e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.9743117128198905e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (5.9743117128198905e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (0.00029219862611103874) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.00029219862611103874) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.00029219862611103874) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.00029219862611103874) [X6 Z7 X8 X9 Z10 X11]
+ (0.000495676231491633) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.000495676231491633) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.0006650070219499323) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.0006650070219499323) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.0006650070219499323) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.0006650070219499323) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.0008533856254125439) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.0008533856254125439) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.001609531381721385) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.001609531381721385) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.001609531381721385) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.001609531381721385) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.0016676041811440577) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.0016676041811440577) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.0016676041811440577) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.0016676041811440577) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.0017278753941369785) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.0017278753941369785) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.001799219493663005) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.001799219493663005) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.0022939566113524715) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.0022939566113524715) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.002462917007133929) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.002462917007133929) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.002462917007133929) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.002462917007133929) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.00396156079249653) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.00396156079249653) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.00396156079249653) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.00396156079249653) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.00442485544944186) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.00442485544944186) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.004575007626639208) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.004575007626639208) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.004668620318776307) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.004668620318776307) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.004684903388155217) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.004684903388155217) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.0053248352342216655) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.0053248352342216655) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.0053248352342216655) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.0053248352342216655) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.005368659358109557) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.005368659358109557) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.005368659358109557) [X2 X3 X7 Z8 Z9 X10]
+ (0.005368659358109557) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.00796088072592156) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.00796088072592156) [X4 Z5 X6 X10 Z11 X12]
+ (0.00796088072592156) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.00796088072592156) [X5 Z6 X7 X11 Z12 X13]
+ (0.008125251921381046) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.008125251921381046) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.008890731522694609) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.008890731522694609) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.008890731522694609) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.008890731522694609) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.010263414868158509) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.010263414868158509) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.010263414868158509) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.010263414868158509) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.010540425907671572) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.010540425907671572) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.010540425907671572) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.010540425907671572) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.010960074940542597) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.010960074940542597) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.010960074940542597) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.010960074940542597) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.011307274008848182) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.011307274008848182) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.014411099430130914) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.014411099430130914) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.014411099430130914) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.014411099430130914) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.015225630757226558) [Y3 Z4 Z5 X6 X10 Y11]
+ (0.015225630757226558) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (0.015225630757226558) [X3 Z4 Z5 X6 X10 X11]
+ (0.015225630757226558) [X3 Z4 Z5 Y6 Y10 X11]
+ (0.015588250102380175) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (0.015588250102380175) [X2 Z3 X4 X10 Z11 X12]
+ (0.015588250102380175) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (0.015588250102380175) [X3 Z4 X5 X11 Z12 X13]
+ (0.01826683486937557) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.01826683486937557) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.01826683486937557) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.01826683486937557) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.019020423173039987) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.019020423173039987) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.019020423173039987) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.019020423173039987) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.020175921723535512) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.020175921723535512) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.020175921723535512) [X4 Z5 Z6 X7 X11 X12]
+ (0.020175921723535512) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.020175921723535512) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.020175921723535512) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.020175921723535512) [X5 X6 X10 Z11 Z12 X13]
+ (0.020175921723535512) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.024353077678068907) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.024353077678068907) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.024353077678068907) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.024353077678068907) [X2 Z3 X4 X11 Z12 X13]
+ (0.024353077678068907) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.024353077678068907) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.024353077678068907) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.024353077678068907) [X3 Z4 X5 X10 Z11 X12]
+ (0.02438908253114954) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.02438908253114954) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.02438908253114954) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.02438908253114954) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.025104957138844548) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.025104957138844548) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.025104957138844548) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.025104957138844548) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.030787505389143946) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.030787505389143946) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.04587947078129791) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.04587947078129791) [X0 Z2 Z3 Z4 Z5 X6]
+ (0.05600733087780766) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.05600733087780766) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.05600733087780766) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.05600733087780766) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661357) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.05608468124661357) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661357) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.05608468124661357) [Z1 X7 Z8 Z9 Z10 X11]
+ (-6.631277927662665e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.631277927662665e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-6.631277927662664e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.631277927662664e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.5950860064433847e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.5950860064433847e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.595086006443384e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.595086006443384e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.042743277013783554) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.042743277013783554) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (0.042743277013783554) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.042743277013783554) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.04764261217638313) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.04764261217638313) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.04764261217638313) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.04764261217638313) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.04171881383982179) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.04171881383982179) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.04171881383982179) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.04171881383982179) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.03956441632289333) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.03956441632289333) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.03956441632289333) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03956441632289333) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03935916802205301) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.03935916802205301) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.03935916802205301) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.03935916802205301) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.03931805194719756) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03931805194719756) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.03931805194719756) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03931805194719756) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03560837898831247) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.03560837898831247) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.029903789512624783) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.029903789512624783) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.029903789512624783) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.029903789512624783) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.02873077955190547) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.02873077955190547) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.02873077955190547) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.02873077955190547) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.02563723829602682) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.02563723829602682) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.02563723829602682) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.02563723829602682) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.024755463292890925) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.024755463292890925) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.024755463292890925) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.024755463292890925) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.024282117354692927) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.024282117354692927) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.023145130929529065) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.023145130929529065) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.022528440196013) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.022528440196013) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.021433810721600874) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.021433810721600874) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.021433810721600874) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.021433810721600874) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251558) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.019257505095251558) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019028242443847238) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.019028242443847238) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.018889030304942857) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-0.018889030304942857) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-0.018889030304942857) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.018889030304942857) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.01602460368917954) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.01602460368917954) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.015225630757226558) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.015225630757226558) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.014603704729162087) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.014603704729162087) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.014564531231172975) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.014564531231172975) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.011756013419819219) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.011756013419819219) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.011285190200840907) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-0.011285190200840907) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-0.009841749246962614) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.009841749246962614) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.009612634606847283) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.009612634606847283) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.009612634606847283) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.009612634606847283) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.008469978791023913) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.008469978791023913) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.007306759928832976) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.007306759928832976) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005923798336561343) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.005923798336561343) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.005652620978017347) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.005652620978017347) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.005368659358109558) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005368659358109558) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.004158797381840065) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.004158797381840065) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.003356670563832885) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.003356670563832885) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.003356670563832885) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.003356670563832885) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.003267513854423552) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.003267513854423552) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.003267513854423552) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.003267513854423552) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.0027790267990255315) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.0027790267990255315) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.002686040977806623) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.002686040977806623) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.002686040977806623) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.002686040977806623) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.002293956611352472) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.002293956611352472) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.002293956611352472) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.002293956611352472) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.000958165583669644) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.000958165583669644) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.000958165583669644) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.000958165583669644) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.000958165583669644) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.000958165583669644) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.000958165583669644) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.000958165583669644) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.0002463643756957749) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0002463643756957749) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303551073) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.00013840177303551073) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.00013840177303551073) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.00013840177303551073) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.7350368805908e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.7350368805908e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585303873055e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585303873055e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.6103585303873055e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.6103585303873055e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808793520564e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.5316808793520564e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808793520564e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5316808793520564e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.80610277411484e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.80610277411484e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.80610277411484e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.80610277411484e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.0897994668277924e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.0897994668277924e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.0897994668277924e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.0897994668277924e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.6522096681083545e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.6522096681083545e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.6522096681083545e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.6522096681083545e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851832541479e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.481851832541479e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851832541479e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.481851832541479e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.071480735927862e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.071480735927862e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.071480735927862e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.071480735927862e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.734622038186978e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.734622038186978e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.734622038186978e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.734622038186978e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.728843146714317e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.728843146714317e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.728843146714317e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.728843146714317e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.253224225180554e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.253224225180554e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.769659451237846e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.769659451237846e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.544395428899358e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.544395428899358e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.544395428899358e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.544395428899358e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.544395428899358e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.544395428899358e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.544395428899358e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.544395428899358e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.360956320113475e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.360956320113475e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.360956320113475e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.360956320113475e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-2.1032156042049336e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.1032156042049336e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.1032156042049336e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.1032156042049336e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220976048363e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.0111220976048363e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220976048363e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.0111220976048363e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9429468363577437e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.9429468363577437e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9429468363577437e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9429468363577437e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174766204587e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.6541174766204587e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174766204587e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.6541174766204587e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.5224930674705486e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.5224930674705486e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.5224930674705486e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.5224930674705486e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.5224930674705486e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930674705486e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930674705486e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.5224930674705486e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.2283337825722566e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337825722566e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.2283337825722566e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337825722566e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.988770288797835e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.988770288797835e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.988770288797835e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.988770288797835e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.867765103524932e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765103524932e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765103524932e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.867765103524932e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.189990974300326e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.189990974300326e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.175246206428805e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175246206428805e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.47164774413403e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.47164774413403e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.561447180178596e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.561447180178596e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.561447180178596e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.561447180178596e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.5233896769722744e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.5233896769722744e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.4273231086192385e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.4273231086192385e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.4273231086192385e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.4273231086192385e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.328139350685017e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350685017e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.328139350685017e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350685017e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.086826565355608e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.086826565355608e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.8882935973728525e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8882935973728525e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8882935973728525e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.8882935973728525e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.3713289480550654e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.3713289480550654e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.8394209156046413e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.8394209156046413e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-8.057446593406361e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.057446593406361e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (4.537178095476928e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.537178095476928e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.537178095476928e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (4.537178095476928e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (8.057446593406361e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.057446593406361e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (9.209350660009705e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350660009705e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350660009705e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.209350660009705e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783556687509e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783556687509e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783556687509e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783556687509e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.8394209156046413e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.8394209156046413e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.3713289480550654e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.3713289480550654e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (3.086826565355608e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.086826565355608e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (4.5233896769722744e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.5233896769722744e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (5.47164774413403e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.47164774413403e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.175246206428805e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175246206428805e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.189990974300326e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.189990974300326e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.330473188568911e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.330473188568911e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.330473188568911e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.330473188568911e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.628853243308463e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.628853243308463e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.628853243308463e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.628853243308463e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.6893489512489827e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.6893489512489827e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.6893489512489827e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.6893489512489827e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (2.745518400019574e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.745518400019574e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.745518400019574e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.745518400019574e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.745518400019574e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.745518400019574e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.745518400019574e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.745518400019574e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.2118420187195313e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420187195313e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.2118420187195313e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420187195313e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.2118420187195313e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420187195313e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.2118420187195313e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420187195313e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.3131454999000433e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.3131454999000433e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.3131454999000433e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.3131454999000433e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (3.3343312890210605e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.3343312890210605e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (4.183932559161012e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.183932559161012e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (7.7350368805908e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.7350368805908e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.0002463643756957749) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.0002463643756957749) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00044585351288409496) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.00044585351288409496) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.00044585351288409496) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.00044585351288409496) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.0005940221543005423) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005423) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005423) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005423) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005423) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005423) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.0005940221543005423) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005423) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.0008533856254125439) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.0008533856254125439) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.0008533856254125439) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.0008533856254125439) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.0010435246534907612) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.0010435246534907612) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.0010435246534907612) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.0010435246534907612) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.0012803060973496784) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.0012803060973496784) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.0012803060973496784) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.0012803060973496784) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.0013038004788127019) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (0.0013038004788127019) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (0.0013038004788127019) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (0.0013038004788127019) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (0.002261966062482346) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (0.002261966062482346) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (0.002261966062482346) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (0.002261966062482346) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (0.002261966062482346) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (0.002261966062482346) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (0.002261966062482346) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (0.002261966062482346) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (0.003989841456619325) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (0.003989841456619325) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (0.003989841456619325) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (0.003989841456619325) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (0.004158797381840065) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.004158797381840065) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.004311038507914313) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.004311038507914313) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.004311038507914313) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.004311038507914313) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.004636976661182563) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.004636976661182563) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.004636976661182563) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.004636976661182563) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.0051144738316604025) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.0051144738316604025) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.0051144738316604025) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.0051144738316604025) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.0051144738316604025) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.0051144738316604025) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.0051144738316604025) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.0051144738316604025) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.005241535382803872) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (0.005241535382803872) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (0.005241535382803872) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (0.005241535382803872) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (0.005262642473076849) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.005262642473076849) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.005262642473076849) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.005262642473076849) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.005368659358109558) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005368659358109558) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005379937155839383) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.005379937155839383) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.005379937155839383) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.005379937155839383) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.005652620978017347) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.005652620978017347) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.005708495985960945) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.005708495985960945) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.005708495985960945) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.005708495985960945) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.005923798336561343) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.005923798336561343) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.007306759928832976) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.007306759928832976) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.008469978791023913) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.008469978791023913) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.009841749246962614) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.009841749246962614) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.011285190200840907) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (0.011285190200840907) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (0.011756013419819219) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.011756013419819219) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.014564531231172975) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.014564531231172975) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.014603704729162087) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.014603704729162087) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.015225630757226558) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.015225630757226558) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.01602460368917954) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.01602460368917954) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.019028242443847238) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.019028242443847238) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.019257505095251558) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.019257505095251558) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.045879470781297914) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.045879470781297914) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.36937089366156245) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.36937089366156245) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.36937089366156245) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.36937089366156245) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.28164257767022854) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.28164257767022854) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.2816425776702283) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.2816425776702283) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.09065144207036473) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.09065144207036473) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.09065144207036473) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.09065144207036473) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0868473758986362) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0868473758986362) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0868473758986362) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0868473758986362) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.07635021950634997) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07635021950634997) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07635021950634997) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07635021950634997) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214009) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.06752385099214009) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214009) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.06752385099214009) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03560837898831247) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.03560837898831247) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.034903343373661806) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.034903343373661806) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.034903343373661806) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.034903343373661806) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.02459186088382998) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.02459186088382998) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.02459186088382998) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.02459186088382998) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.024282117354692927) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.024282117354692927) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.023145130929529065) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.023145130929529065) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.022528440196013) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.022528440196013) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.01953805031131466) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.01953805031131466) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.01953805031131466) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.01953805031131466) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.017091553155898827) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.017091553155898827) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.017091553155898827) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.017091553155898827) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.016024603689179538) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.016024603689179538) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.016024603689179538) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.016024603689179538) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.010311482489831818) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831818) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831818) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010311482489831818) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.009841749246962614) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962614) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.009841749246962614) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962614) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209886) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209886) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209886) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008826368514209886) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008541996625454807) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454807) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454807) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454807) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454807) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454807) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454807) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008541996625454807) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008469978791023913) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023913) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.008469978791023913) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023913) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.004668620318776307) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.004668620318776307) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.0038764708993369486) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.0038764708993369486) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.003804066171728542) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728542) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728542) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003804066171728542) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178895) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178895) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003356670563832884) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.003356670563832884) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.003267513854423552) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.003267513854423552) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0021413612231015776) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.0021413612231015776) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.0017278753941369785) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.0017278753941369785) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.0016407548553124156) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0016407548553124156) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214168927) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0014528843214168927) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214168927) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0014528843214168927) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870896771024448) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007870896771024448) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0005192743499487756) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.0005192743499487756) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.00019400857029756255) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00019400857029756255) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303551073) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.00013840177303551073) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.14162522115509e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.14162522115509e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.14162522115509e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.14162522115509e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-5.071480735927862e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.071480735927862e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.151346310779012e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.151346310779012e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0882507108814977e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.0882507108814977e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.988511706296515e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.988511706296515e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8742990710983366e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.8742990710983366e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.360956320113475e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.360956320113475e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.3002946560662123e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.3002946560662123e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.1468376505801358e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.1468376505801358e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.1468376505801358e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.1468376505801358e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.352332101868221e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.352332101868221e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.352332101868221e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.352332101868221e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.091637197686036e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637197686036e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637197686036e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637197686036e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637197686036e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637197686036e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637197686036e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.091637197686036e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.074305984758116e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.074305984758116e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.074305984758116e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.074305984758116e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.900128985087594e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.900128985087594e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.900128985087594e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900128985087594e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867765103524932e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.867765103524932e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.560692463784358e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692463784358e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692463784358e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692463784358e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692463784358e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692463784358e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692463784358e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.560692463784358e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.997018421364028e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.997018421364028e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.997018421364028e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.997018421364028e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.997018421364028e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.997018421364028e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.997018421364028e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.997018421364028e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.568247520713763e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.568247520713763e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.568247520713763e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.568247520713763e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.37673930811532e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.37673930811532e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.37673930811532e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.37673930811532e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.37673930811532e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.37673930811532e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.37673930811532e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.37673930811532e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.8882935973728525e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.8882935973728525e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6863815427870304e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.6863815427870304e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035783556687509e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7035783556687509e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.209350660009705e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.209350660009705e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773242708254e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773242708254e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773242708254e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773242708254e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773242708254e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773242708254e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.379773242708254e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773242708254e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253790283034e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253790283034e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9742253790283034e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9742253790283034e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0474716553397946e-08) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.0474716553397946e-08) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.0474716553397946e-08) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.0474716553397946e-08) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.209350660009705e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.209350660009705e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0717282180370066e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.0717282180370066e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.0717282180370066e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.0717282180370066e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.200428749310027e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.200428749310027e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.200428749310027e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.200428749310027e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.7035783556687509e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7035783556687509e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.3120943049834555e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.3120943049834555e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.3120943049834555e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.3120943049834555e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.6863815427870304e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815427870304e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.8882935973728525e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.8882935973728525e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.0922506156735886e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506156735886e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.0922506156735886e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506156735886e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.0922506156735886e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506156735886e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.0922506156735886e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506156735886e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.444597853988046e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.444597853988046e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.444597853988046e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.444597853988046e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.684915094805791e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.684915094805791e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.684915094805791e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.684915094805791e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.24697442446581e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.24697442446581e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.24697442446581e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.24697442446581e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.24697442446581e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.24697442446581e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.24697442446581e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.24697442446581e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.867765103524932e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.867765103524932e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.3002946560662123e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.3002946560662123e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.360956320113475e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.360956320113475e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.8742990710983366e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.8742990710983366e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.883676575814257e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.883676575814257e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.947356011261857e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.947356011261857e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.947356011261857e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.947356011261857e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.988511706296515e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.988511706296515e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.0882507108814977e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.0882507108814977e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.151346310779012e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.151346310779012e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.846201670908735e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.846201670908735e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.846201670908735e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.846201670908735e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.071480735927862e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.071480735927862e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.105526721534736e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.105526721534736e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.105526721534736e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.105526721534736e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.146496326974948e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.146496326974948e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.146496326974948e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.146496326974948e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.1593505015623604e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.1593505015623604e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.1593505015623604e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.1593505015623604e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.4279886558410635e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.4279886558410635e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.4279886558410635e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.4279886558410635e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.935867717558372e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.935867717558372e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.935867717558372e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.935867717558372e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273347421868e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.253273347421868e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.979825792633072e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.979825792633072e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.979825792633072e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.979825792633072e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (4.205548411217487e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.205548411217487e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.205548411217487e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.205548411217487e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00013840177303551073) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.00013840177303551073) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.00018787053389552266) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.00018787053389552266) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.00018787053389552266) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.00018787053389552266) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00019400857029756255) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00019400857029756255) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00024636437569577495) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024636437569577495) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00024636437569577495) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024636437569577495) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192743499487756) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.0005192743499487756) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.0007156734248908938) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007156734248908938) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0007156734248908938) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007156734248908938) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024448) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007870896771024448) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.0015324835230730608) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.0015324835230730608) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.0015324835230730608) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.0015324835230730608) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.0016407548553124156) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0016407548553124156) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0017278753941369785) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.0017278753941369785) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.0024464971554158336) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.0024464971554158336) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.0024464971554158336) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.0024464971554158336) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.003267513854423552) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.003267513854423552) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.003356670563832884) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.003356670563832884) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.0034841573002178895) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0034841573002178895) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0038764708993369486) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.0038764708993369486) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.004668620318776307) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.004668620318776307) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278124) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.004767272188278124) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.004767272188278124) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278124) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.0052865465382269) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.0052865465382269) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.0052865465382269) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.0052865465382269) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.0054089544224100086) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.0054089544224100086) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.0054089544224100086) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.0054089544224100086) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.005923798336561343) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561343) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.005923798336561343) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561343) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.010715508469796749) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010715508469796749) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010715508469796749) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010715508469796749) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.010757563953908925) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010757563953908925) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010757563953908925) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010757563953908925) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.014603704729162087) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.014603704729162087) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.014603704729162087) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.014603704729162087) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.019299560579363734) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363734) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363734) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363734) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363734) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363734) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.019299560579363734) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363734) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.058591988733862066) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.058591988733862066) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.775950526698355e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.775950526698355e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.7759505266983574e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.7759505266983574e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.07165035181003045) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.07165035181003045) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.07165035181003052) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.07165035181003052) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251558) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019257505095251558) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010311482489831818) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311482489831818) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008826368514209886) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209886) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.007597464029770618) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.007597464029770618) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.007597464029770618) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.007597464029770618) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311881) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311881) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311881) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311881) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311881) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311881) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311881) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311881) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676629) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005348051582676629) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005348051582676629) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676629) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.003804066171728542) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003804066171728542) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0029841661681219355) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.0029841661681219355) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.0029841661681219355) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.0029841661681219355) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.0024464971554158336) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.0024464971554158336) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.0022494124470939917) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0022494124470939917) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0022494124470939917) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0022494124470939917) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.002141361223101578) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.002141361223101578) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.0018638942824587383) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587383) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587383) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587383) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587383) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587383) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0018638942824587383) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587383) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0016407548553124159) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124159) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0016407548553124159) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124159) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0012223378081538262) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538262) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538262) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538262) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538262) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538262) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538262) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0012223378081538262) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0010283292378562637) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0010283292378562637) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0010283292378562637) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0010283292378562637) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.146306145214541e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.146306145214541e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.8742990710983366e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990710983366e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.8742990710983366e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990710983366e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.3002946560662123e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.3002946560662123e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.3002946560662123e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.3002946560662123e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.0444941297212506e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.0444941297212506e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.0444941297212506e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.0444941297212506e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.956079229168257e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.956079229168257e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.956079229168257e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.956079229168257e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105515036306174e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105515036306174e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105515036306174e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105515036306174e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661347212437077e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661347212437077e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661347212437077e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661347212437077e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540341412932578e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540341412932578e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.189990974300326e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.189990974300326e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.876621657540056e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876621657540056e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876621657540056e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876621657540056e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175246206428805e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175246206428805e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.5233896769722744e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.5233896769722744e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0767325319569685e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.0767325319569685e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0767325319569685e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.0767325319569685e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.013471458563128e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.013471458563128e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.9045998842799276e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.9045998842799276e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.9045998842799276e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.9045998842799276e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.6667317548803457e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.6667317548803457e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.6667317548803457e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6667317548803457e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641928620843e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.8505641928620843e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309315169258e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.6569309315169258e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309315169258e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.6569309315169258e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.8505641928620843e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.8505641928620843e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815427870304e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815427870304e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.6863815427870304e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815427870304e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458563128e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.013471458563128e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.5233896769722744e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.5233896769722744e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.670402390080054e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.670402390080054e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.670402390080054e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.670402390080054e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175246206428805e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175246206428805e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.189990974300326e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.189990974300326e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540341412932578e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540341412932578e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949476486298767e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949476486298767e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.7924939575594127e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939575594127e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939575594127e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.7924939575594127e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.8836765758142573e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.8836765758142573e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.988511706296515e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.988511706296515e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.988511706296515e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.988511706296515e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273347421868e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.253273347421868e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4017109733964774e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4017109733964774e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4017109733964774e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4017109733964774e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603691524186e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.5809603691524186e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603691524186e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.5809603691524186e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0005192743499487756) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487756) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.0005192743499487756) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487756) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.0007870896771024448) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024448) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024448) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024448) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0011726348316441924) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0011726348316441924) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0011726348316441924) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0011726348316441924) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012366478019245485) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.0012366478019245485) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.0012366478019245485) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.0012366478019245485) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.0022009640695004563) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0022009640695004563) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0022009640695004563) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0022009640695004563) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798019) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798019) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798019) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798019) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798019) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798019) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002394972639798019) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798019) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0024464971554158336) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.0024464971554158336) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.003804066171728542) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.003804066171728542) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.003876470899336949) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.003876470899336949) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.003876470899336949) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.003876470899336949) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.0042208139700464845) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.0042208139700464845) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.0042208139700464845) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.0042208139700464845) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.008826368514209886) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.008826368514209886) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.010311482489831818) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010311482489831818) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019257505095251558) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019257505095251558) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.058591988733862066) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.058591988733862066) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.3987009017113989e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.3987009017113989e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.3987009017113985e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.3987009017113985e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841573002178895) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841573002178895) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0029841661681219355) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.0029841661681219355) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.00019400857029756255) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00019400857029756255) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061452145408e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061452145408e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.792493957559413e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.792493957559413e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540341412932578e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341412932578e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540341412932578e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341412932578e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.8505641928620843e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641928620843e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641928620843e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641928620843e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714585631283e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.0134714585631283e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714585631283e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.0134714585631283e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.949476486298768e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.949476486298768e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.792493957559413e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.792493957559413e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029756255) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029756255) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0029841661681219355) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.0029841661681219355) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.0034841573002178895) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0034841573002178895) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
List of core orbitals: [0, 1, 2]
List of active orbitals: [3, 4, 5, 6]
Number of qubits: 8
Number of qubits required to perform quantum simulations: 8
Hamiltonian of the water molecule
  (-73.1387323135253) [I0]
+ (-0.18066792656583347) [Z6]
+ (-0.18066792656583347) [Z7]
+ (-0.15961432501809925) [Z4]
+ (-0.15961432501809925) [Z5]
+ (0.17419956155055746) [Z2]
+ (0.1741995615505575) [Z3]
+ (0.22757269005453454) [Z0]
+ (0.22757269005453462) [Z1]
+ (-8.194261372363013e-06) [Y4 Y6]
+ (-8.194261372363013e-06) [X4 X6]
+ (7.954413176287087e-06) [Y5 Y7]
+ (7.954413176287087e-06) [X5 X7]
+ (0.112703869203322) [Z4 Z6]
+ (0.112703869203322) [Z5 Z7]
+ (0.11952438964682657) [Z0 Z4]
+ (0.11952438964682657) [Z1 Z5]
+ (0.1340171526196368) [Z0 Z6]
+ (0.1340171526196368) [Z1 Z7]
+ (0.13734953064261313) [Z0 Z5]
+ (0.13734953064261313) [Z1 Z4]
+ (0.13766872645852568) [Z2 Z4]
+ (0.13766872645852568) [Z3 Z5]
+ (0.14138905291942788) [Z4 Z7]
+ (0.14138905291942788) [Z5 Z6]
+ (0.1472294321876616) [Z2 Z5]
+ (0.1472294321876616) [Z3 Z4]
+ (0.14926355147388876) [Z4 Z5]
+ (0.1497348680349691) [Z2 Z6]
+ (0.1497348680349691) [Z3 Z7]
+ (0.15138327161428816) [Z0 Z7]
+ (0.15138327161428816) [Z1 Z6]
+ (0.15435748657223602) [Z6 Z7]
+ (0.15582269051553094) [Z2 Z7]
+ (0.15582269051553094) [Z3 Z6]
+ (0.16756653265461252) [Z0 Z2]
+ (0.16756653265461252) [Z1 Z3]
+ (0.18143991440303858) [Z0 Z3]
+ (0.18143991440303858) [Z1 Z2]
+ (0.19392534613270157) [Z0 Z1]
+ (0.2200397733437609) [Z2 Z3]
+ (-7.03788751046816e-06) [Y5 Z6 Y7]
+ (-7.03788751046816e-06) [X5 Z6 X7]
+ (-7.0378875104681574e-06) [Y4 Z5 Y6]
+ (-7.0378875104681574e-06) [X4 Z5 X6]
+ (-0.02868518371610587) [Y4 Y5 X6 X7]
+ (-0.02868518371610587) [X4 X5 Y6 Y7]
+ (-0.01782514099578657) [Y0 Y1 X4 X5]
+ (-0.01782514099578657) [X0 X1 Y4 Y5]
+ (-0.017366118994651403) [Y0 Y1 X6 X7]
+ (-0.017366118994651403) [X0 X1 Y6 Y7]
+ (-0.013873381748426063) [Y0 Y1 X2 X3]
+ (-0.013873381748426063) [X0 X1 Y2 Y3]
+ (-0.009560705729135895) [Y2 Y3 X4 X5]
+ (-0.009560705729135895) [X2 X3 Y4 Y5]
+ (-0.006087822480561847) [Y2 Y3 X6 X7]
+ (-0.006087822480561847) [X2 X3 Y6 Y7]
+ (-0.00029219862611101267) [Y1 Y2 X3 X4]
+ (-0.00029219862611101267) [X1 X2 Y3 Y4]
+ (-8.194261372363013e-06) [Z4 Y5 Z6 Y7]
+ (-8.194261372363013e-06) [Z4 X5 Z6 X7]
+ (-2.8909678817793226e-06) [Z0 Y5 Z6 Y7]
+ (-2.8909678817793226e-06) [Z0 X5 Z6 X7]
+ (-2.8909678817793226e-06) [Z1 Y4 Z5 Y6]
+ (-2.8909678817793226e-06) [Z1 X4 Z5 X6]
+ (-1.8551201216339564e-06) [Z0 Y4 Z5 Y6]
+ (-1.8551201216339564e-06) [Z0 X4 Z5 X6]
+ (-1.8551201216339564e-06) [Z1 Y5 Z6 Y7]
+ (-1.8551201216339564e-06) [Z1 X5 Z6 X7]
+ (-1.597317197902417e-06) [Z2 Y4 Z5 Y6]
+ (-1.597317197902417e-06) [Z2 X4 Z5 X6]
+ (-1.597317197902417e-06) [Z3 Y5 Z6 Y7]
+ (-1.597317197902417e-06) [Z3 X5 Z6 X7]
+ (-1.0358477601453654e-06) [Y0 X1 X5 Y6]
+ (-1.0358477601453654e-06) [Y0 Y1 Y5 Y6]
+ (-1.0358477601453654e-06) [X0 X1 X5 X6]
+ (-1.0358477601453654e-06) [X0 Y1 Y5 X6]
+ (-9.344557777360644e-07) [Z2 Y5 Z6 Y7]
+ (-9.344557777360644e-07) [Z2 X5 Z6 X7]
+ (-9.344557777360644e-07) [Z3 Y4 Z5 Y6]
+ (-9.344557777360644e-07) [Z3 X4 Z5 X6]
+ (6.628614201663525e-07) [Y2 X3 X5 Y6]
+ (6.628614201663525e-07) [Y2 Y3 Y5 Y6]
+ (6.628614201663525e-07) [X2 X3 X5 X6]
+ (6.628614201663525e-07) [X2 Y3 Y5 X6]
+ (7.954413176287087e-06) [Y4 Z5 Y6 Z7]
+ (7.954413176287087e-06) [X4 Z5 X6 Z7]
+ (0.00029219862611101267) [Y1 X2 X3 Y4]
+ (0.00029219862611101267) [X1 Y2 Y3 X4]
+ (0.006087822480561847) [Y2 X3 X6 Y7]
+ (0.006087822480561847) [X2 Y3 Y6 X7]
+ (0.009560705729135895) [Y2 X3 X4 Y5]
+ (0.009560705729135895) [X2 Y3 Y4 X5]
+ (0.011307274008848178) [Y1 Z2 Z3 Y5]
+ (0.011307274008848178) [X1 Z2 Z3 X5]
+ (0.013873381748426063) [Y0 X1 X2 Y3]
+ (0.013873381748426063) [X0 Y1 Y2 X3]
+ (0.017366118994651403) [Y0 X1 X6 Y7]
+ (0.017366118994651403) [X0 Y1 Y6 X7]
+ (0.01782514099578657) [Y0 X1 X4 Y5]
+ (0.01782514099578657) [X0 Y1 Y4 X5]
+ (0.02868518371610587) [Y4 X5 X6 Y7]
+ (0.02868518371610587) [X4 Y5 Y6 X7]
+ (0.029812424517345906) [Y0 Z1 Z2 Y4]
+ (0.029812424517345906) [X0 Z1 Z2 X4]
+ (0.029812424517345906) [Y1 Z3 Z4 Y5]
+ (0.029812424517345906) [X1 Z3 Z4 X5]
+ (0.030104623143456917) [Y0 Z1 Z3 Y4]
+ (0.030104623143456917) [X0 Z1 Z3 X4]
+ (0.030104623143456917) [Y1 Z2 Z4 Y5]
+ (0.030104623143456917) [X1 Z2 Z4 X5]
+ (0.03078750538914401) [Y0 Z2 Z3 Y4]
+ (0.03078750538914401) [X0 Z2 Z3 X4]
+ (0.04375263801066066) [Y1 Z2 Z3 Z4 Y5]
+ (0.04375263801066066) [X1 Z2 Z3 Z4 X5]
+ (0.04375263801066068) [Y0 Z1 Z2 Z3 Y4]
+ (0.04375263801066068) [X0 Z1 Z2 Z3 X4]
+ (-0.014564531231173006) [Y1 Z2 Z3 X4 X6 Y7]
+ (-0.014564531231173006) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-0.014564531231173006) [X1 Z2 Z3 X4 X6 X7]
+ (-0.014564531231173006) [X1 Z2 Z3 Y4 Y6 X7]
+ (-6.524373848623513e-06) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (-6.524373848623513e-06) [X0 Z1 Z2 Z3 Z4 X6]
+ (-6.524373848623513e-06) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (-6.524373848623513e-06) [X1 Z2 Z3 Z5 Z6 X7]
+ (-3.7696594520372537e-06) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (-3.7696594520372537e-06) [X0 Z2 Z3 Z4 Z5 X6]
+ (-3.6102971306172883e-06) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (-3.6102971306172883e-06) [X0 Z1 Z3 Z4 Z5 X6]
+ (-3.6102971306172883e-06) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (-3.6102971306172883e-06) [X1 Z2 Z4 Z5 Z6 X7]
+ (-3.3131455001610044e-06) [Y1 Z2 Z3 Y4 X5 X6]
+ (-3.3131455001610044e-06) [X1 Z2 Z3 X4 Y5 Y6]
+ (-3.2774831955579204e-06) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (-3.2774831955579204e-06) [X0 Z1 Z2 Z4 Z5 X6]
+ (-3.2774831955579204e-06) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (-3.2774831955579204e-06) [X1 Z3 Z4 Z5 Z6 X7]
+ (-3.2112283484625094e-06) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (-3.2112283484625094e-06) [X0 Z1 Z2 Z3 Z5 X6]
+ (-3.2112283484625094e-06) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (-3.2112283484625094e-06) [X1 Z2 Z3 Z4 Z6 X7]
+ (-1.0358477601453654e-06) [Y0 Y1 X4 Z5 Z6 X7]
+ (-1.0358477601453654e-06) [X0 X1 Y4 Z5 Z6 Y7]
+ (-6.628614201663525e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (-6.628614201663525e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (-3.328139350593681e-07) [Y1 X2 X3 Z4 Z5 Y6]
+ (-3.328139350593681e-07) [X1 Y2 Y3 Z4 Z5 X6]
+ (3.328139350593681e-07) [Y1 Y2 X3 Z4 Z5 X6]
+ (3.328139350593681e-07) [X1 X2 Y3 Z4 Z5 Y6]
+ (6.628614201663525e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (6.628614201663525e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (1.0358477601453654e-06) [Y0 X1 X4 Z5 Z6 Y7]
+ (1.0358477601453654e-06) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.3131455001610044e-06) [Y1 Z2 Z3 X4 X5 Y6]
+ (3.3131455001610044e-06) [X1 Z2 Z3 Y4 Y5 X6]
+ (4.183932559460193e-06) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (4.183932559460193e-06) [X1 Z2 Z3 Z4 Z5 X7]
+ (0.00029219862611101267) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (0.00029219862611101267) [Y0 Z1 Y2 X3 Z4 X5]
+ (0.00029219862611101267) [X0 Z1 X2 Y3 Z4 Y5]
+ (0.00029219862611101267) [X0 Z1 X2 X3 Z4 X5]
+ (0.010540425907671553) [Y0 Z1 Z2 Z3 Y4 Z7]
+ (0.010540425907671553) [X0 Z1 Z2 Z3 X4 Z7]
+ (0.010540425907671553) [Y1 Z2 Z3 Z4 Y5 Z6]
+ (0.010540425907671553) [X1 Z2 Z3 Z4 X5 Z6]
+ (0.011307274008848178) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (0.011307274008848178) [X0 Z1 Z2 Z3 X4 Z5]
+ (0.025104957138844558) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (0.025104957138844558) [X0 Z1 Z2 Z3 X4 Z6]
+ (0.025104957138844558) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (0.025104957138844558) [X1 Z2 Z3 Z4 X5 Z7]
+ (0.03078750538914401) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (0.03078750538914401) [Z0 X1 Z2 Z3 Z4 X5]
+ (-5.105396549351089e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-5.105396549351089e-06) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-5.105396549351086e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (-5.105396549351086e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (-0.014564531231173006) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-0.014564531231173006) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-3.7696594520372537e-06) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-3.7696594520372537e-06) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-3.328139350593681e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-3.328139350593681e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-3.328139350593681e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-3.328139350593681e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (3.3131455001610044e-06) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (3.3131455001610044e-06) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (3.3131455001610044e-06) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (3.3131455001610044e-06) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (4.183932559460193e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (4.183932559460193e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (0.014564531231173006) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (0.014564531231173006) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
 </code>
 </pre>
 </details>

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_quantum_chemistry.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Number of qubits: 14
Qubit Hamiltonian
  (-46.46390691194562) [I0]
+ (0.7829652070216091) [Z11]
+ (0.7829652070216092) [Z10]
+ (0.808459100524366) [Z13]
+ (0.8084591005243662) [Z12]
+ (1.2034393392168443) [Z5]
+ (1.2034393392168448) [Z4]
+ (1.3096876619519762) [Z6]
+ (1.3096876619519766) [Z7]
+ (1.3693525712744405) [Z8]
+ (1.3693525712744408) [Z9]
+ (1.653893830563256) [Z3]
+ (1.6538938305632567) [Z2]
+ (12.41263071395131) [Z1]
+ (12.412630713951312) [Z0]
+ (-7.95422474472729e-06) [Y11 Y13]
+ (-7.95422474472729e-06) [X11 X13]
+ (-1.6021751095710788e-06) [Y2 Y4]
+ (-1.6021751095710788e-06) [X2 X4]
+ (5.929280616496623e-07) [Y4 Y6]
+ (5.929280616496623e-07) [X4 X6]
+ (7.765082230419594e-07) [Y3 Y5]
+ (7.765082230419594e-07) [X3 X5]
+ (1.8540565444416816e-06) [Y5 Y7]
+ (1.8540565444416816e-06) [X5 X7]
+ (8.194105008157981e-06) [Y10 Y12]
+ (8.194105008157981e-06) [X10 X12]
+ (0.003276965063178576) [Y1 Y3]
+ (0.003276965063178576) [X1 X3]
+ (0.1043306148395636) [Y0 Y2]
+ (0.1043306148395636) [X0 X2]
+ (0.11270381857118623) [Z10 Z12]
+ (0.11270381857118623) [Z11 Z13]
+ (0.11383573684195247) [Z4 Z12]
+ (0.11383573684195247) [Z5 Z13]
+ (0.11952441015514684) [Z6 Z10]
+ (0.11952441015514684) [Z7 Z11]
+ (0.12489977361700835) [Z4 Z10]
+ (0.12489977361700835) [Z5 Z11]
+ (0.12495799327864181) [Z2 Z4]
+ (0.12495799327864181) [Z3 Z5]
+ (0.12799492799734785) [Z2 Z10]
+ (0.12799492799734785) [Z3 Z11]
+ (0.13401737371852698) [Z6 Z12]
+ (0.13401737371852698) [Z7 Z13]
+ (0.1370119191332551) [Z4 Z6]
+ (0.1370119191332551) [Z5 Z7]
+ (0.13734942208662523) [Z6 Z11]
+ (0.13734942208662523) [Z7 Z10]
+ (0.1373911237474765) [Z2 Z6]
+ (0.1373911237474765) [Z3 Z7]
+ (0.13766859132455367) [Z8 Z10]
+ (0.13766859132455367) [Z9 Z11]
+ (0.14011294748300746) [Z2 Z12]
+ (0.14011294748300746) [Z3 Z13]
+ (0.14138903587614793) [Z10 Z13]
+ (0.14138903587614793) [Z11 Z12]
+ (0.14257991127040043) [Z4 Z11]
+ (0.14257991127040043) [Z5 Z10]
+ (0.1472293078198935) [Z8 Z11]
+ (0.1472293078198935) [Z9 Z10]
+ (0.14899426171508628) [Z4 Z7]
+ (0.14899426171508628) [Z5 Z6]
+ (0.14926347056922684) [Z10 Z11]
+ (0.1496069255771393) [Z4 Z8]
+ (0.1496069255771393) [Z5 Z9]
+ (0.14973497004652603) [Z8 Z12]
+ (0.14973497004652603) [Z9 Z13]
+ (0.1507140548214972) [Z2 Z8]
+ (0.1507140548214972) [Z3 Z9]
+ (0.15138342698018414) [Z6 Z13]
+ (0.15138342698018414) [Z7 Z12]
+ (0.1521504062146353) [Z4 Z13]
+ (0.1521504062146353) [Z5 Z12]
+ (0.1533795916912814) [Z2 Z11]
+ (0.1533795916912814) [Z3 Z10]
+ (0.15435760063007065) [Z12 Z13]
+ (0.15569017455665016) [Z2 Z13]
+ (0.15569017455665016) [Z3 Z12]
+ (0.15582280685050842) [Z8 Z13]
+ (0.15582280685050842) [Z9 Z12]
+ (0.15676384610699085) [Z4 Z9]
+ (0.15676384610699085) [Z5 Z8]
+ (0.15755303804215298) [Z4 Z5]
+ (0.1607975504633259) [Z2 Z5]
+ (0.1607975504633259) [Z3 Z4]
+ (0.16756669356734996) [Z6 Z8]
+ (0.16756669356734996) [Z7 Z9]
+ (0.1685349279412667) [Z2 Z7]
+ (0.1685349279412667) [Z3 Z6]
+ (0.18144009363568056) [Z6 Z9]
+ (0.18144009363568056) [Z7 Z8]
+ (0.18189081242782154) [Z2 Z3]
+ (0.18690814830799424) [Z2 Z9]
+ (0.18690814830799424) [Z3 Z8]
+ (0.19299700266826592) [Z0 Z10]
+ (0.19299700266826592) [Z1 Z11]
+ (0.19392574335596263) [Z6 Z7]
+ (0.19661749959336497) [Z0 Z4]
+ (0.19661749959336497) [Z1 Z5]
+ (0.199363326909141) [Z0 Z5]
+ (0.199363326909141) [Z1 Z4]
+ (0.2007284355142902) [Z0 Z11]
+ (0.2007284355142902) [Z1 Z10]
+ (0.21102681231760256) [Z0 Z12]
+ (0.21102681231760256) [Z1 Z13]
+ (0.2163105980709883) [Z0 Z13]
+ (0.2163105980709883) [Z1 Z12]
+ (0.22003977241164707) [Z8 Z9]
+ (0.23671071738476968) [Z0 Z2]
+ (0.23671071738476968) [Z1 Z3]
+ (0.2416469683111976) [Z0 Z6]
+ (0.2416469683111976) [Z1 Z7]
+ (0.2485351728535221) [Z0 Z7]
+ (0.2485351728535221) [Z1 Z6]
+ (0.2512943557103505) [Z0 Z3]
+ (0.2512943557103505) [Z1 Z2]
+ (0.2723251844957634) [Z0 Z8]
+ (0.2723251844957634) [Z1 Z9]
+ (0.278834545729906) [Z0 Z9]
+ (0.278834545729906) [Z1 Z8]
+ (1.1861764482920354) [Z0 Z1]
+ (3.886639603758476e-06) [Y2 Z3 Y4]
+ (3.886639603758476e-06) [X2 Z3 X4]
+ (3.886639603758476e-06) [Y3 Z4 Y5]
+ (3.886639603758476e-06) [X3 Z4 X5]
+ (1.0722748309045513e-05) [Y10 Z11 Y12]
+ (1.0722748309045513e-05) [X10 Z11 X12]
+ (1.0722748309045513e-05) [Y11 Z12 Y13]
+ (1.0722748309045513e-05) [X11 Z12 X13]
+ (1.2260276937984505e-05) [Y4 Z5 Y6]
+ (1.2260276937984505e-05) [X4 Z5 X6]
+ (1.2260276937984505e-05) [Y5 Z6 Y7]
+ (1.2260276937984505e-05) [X5 Z6 X7]
+ (0.12507036882189712) [Y1 Z2 Y3]
+ (0.12507036882189712) [X1 Z2 X3]
+ (0.12507036882189718) [Y0 Z1 Y2]
+ (0.12507036882189718) [X0 Z1 X2]
+ (-0.038314669372682825) [Y4 Y5 X12 X13]
+ (-0.038314669372682825) [X4 X5 Y12 Y13]
+ (-0.03619409348649705) [Y2 Y3 X8 X9]
+ (-0.03619409348649705) [X2 X3 Y8 Y9]
+ (-0.035839557184684075) [Y2 Y3 X4 X5]
+ (-0.035839557184684075) [X2 X3 Y4 Y5]
+ (-0.031143804193790196) [Y2 Y3 X6 X7]
+ (-0.031143804193790196) [X2 X3 Y6 Y7]
+ (-0.030787440727002863) [Y6 Z8 Z9 Y10]
+ (-0.030787440727002863) [X6 Z8 Z9 X10]
+ (-0.030104525280110178) [Y6 Z7 Z9 Y10]
+ (-0.030104525280110178) [X6 Z7 Z9 X10]
+ (-0.030104525280110178) [Y7 Z8 Z10 Y11]
+ (-0.030104525280110178) [X7 Z8 Z10 X11]
+ (-0.029812299608009185) [Y6 Z7 Z8 Y10]
+ (-0.029812299608009185) [X6 Z7 Z8 X10]
+ (-0.029812299608009185) [Y7 Z9 Z10 Y11]
+ (-0.029812299608009185) [X7 Z9 Z10 X11]
+ (-0.02868521730496169) [Y10 Y11 X12 X13]
+ (-0.02868521730496169) [X10 X11 Y12 Y13]
+ (-0.02599620626467622) [Y3 Z4 Z5 Y7]
+ (-0.02599620626467622) [X3 Z4 Z5 X7]
+ (-0.025384663693933537) [Y2 Y3 X10 X11]
+ (-0.025384663693933537) [X2 X3 Y10 Y11]
+ (-0.019028318717604294) [Y3 Y4 X11 X12]
+ (-0.019028318717604294) [X3 X4 Y11 Y12]
+ (-0.017825011931478407) [Y6 Y7 X10 X11]
+ (-0.017825011931478407) [X6 X7 Y10 Y11]
+ (-0.017680137653392103) [Y4 Y5 X10 X11]
+ (-0.017680137653392103) [X4 X5 Y10 Y11]
+ (-0.01736605326165715) [Y6 Y7 X12 X13]
+ (-0.01736605326165715) [X6 X7 Y12 Y13]
+ (-0.01557722707364272) [Y2 Y3 X12 X13]
+ (-0.01557722707364272) [X2 X3 Y12 Y13]
+ (-0.014583638325580841) [Y0 Y1 X2 X3]
+ (-0.014583638325580841) [X0 X1 Y2 Y3]
+ (-0.013873400068330589) [Y6 Y7 X8 X9]
+ (-0.013873400068330589) [X6 X7 Y8 Y9]
+ (-0.011982342581831161) [Y4 Y5 X6 X7]
+ (-0.011982342581831161) [X4 X5 Y6 Y7]
+ (-0.011307208035953573) [Y7 Z8 Z9 Y11]
+ (-0.011307208035953573) [X7 Z8 Z9 X11]
+ (-0.011285144615351716) [Y5 X6 X11 Y12]
+ (-0.011285144615351716) [X5 Y6 Y11 X12]
+ (-0.009560716495339834) [Y8 Y9 X10 X11]
+ (-0.009560716495339834) [X8 X9 Y10 Y11]
+ (-0.008125248410396928) [Y1 X2 X8 Y9]
+ (-0.008125248410396928) [Y1 Y2 Y8 Y9]
+ (-0.008125248410396928) [X1 X2 X8 X9]
+ (-0.008125248410396928) [X1 Y2 Y8 X9]
+ (-0.007731432846024306) [Y0 Y1 X10 X11]
+ (-0.007731432846024306) [X0 X1 Y10 Y11]
+ (-0.007156920529851534) [Y4 Y5 X8 X9]
+ (-0.007156920529851534) [X4 X5 Y8 Y9]
+ (-0.006888204542324463) [Y0 Y1 X6 X7]
+ (-0.006888204542324463) [X0 X1 Y6 Y7]
+ (-0.006509361234142669) [Y0 Y1 X8 X9]
+ (-0.006509361234142669) [X0 X1 Y8 Y9]
+ (-0.006087836803982384) [Y8 Y9 X12 X13]
+ (-0.006087836803982384) [X8 X9 Y12 Y13]
+ (-0.005283785753385714) [Y0 Y1 X12 X13]
+ (-0.005283785753385714) [X0 X1 Y12 Y13]
+ (-0.005143382384868528) [Y3 X4 X5 Y6]
+ (-0.005143382384868528) [X3 Y4 Y5 X6]
+ (-0.004684920227379422) [Y1 X2 X6 Y7]
+ (-0.004684920227379422) [Y1 Y2 Y6 Y7]
+ (-0.004684920227379422) [X1 X2 X6 X7]
+ (-0.004684920227379422) [X1 Y2 Y6 X7]
+ (-0.004575015188516512) [Y1 X2 X12 Y13]
+ (-0.004575015188516512) [Y1 Y2 Y12 Y13]
+ (-0.004575015188516512) [X1 X2 X12 X13]
+ (-0.004575015188516512) [X1 Y2 Y12 X13]
+ (-0.004424843669061684) [Y1 X2 X4 Y5]
+ (-0.004424843669061684) [Y1 Y2 Y4 Y5]
+ (-0.004424843669061684) [X1 X2 X4 X5]
+ (-0.004424843669061684) [X1 Y2 Y4 X5]
+ (-0.003479421726098688) [Y2 Z3 Z5 Y6]
+ (-0.003479421726098688) [X2 Z3 Z5 X6]
+ (-0.003479421726098688) [Y3 Z4 Z6 Y7]
+ (-0.003479421726098688) [X3 Z4 Z6 X7]
+ (-0.002745827315776045) [Y0 Y1 X4 X5]
+ (-0.002745827315776045) [X0 X1 Y4 Y5]
+ (-0.001799193008530659) [Y1 X2 X10 Y11]
+ (-0.001799193008530659) [Y1 Y2 Y10 Y11]
+ (-0.001799193008530659) [X1 X2 X10 X11]
+ (-0.001799193008530659) [X1 Y2 Y10 X11]
+ (-0.0002922256721009919) [Y7 X8 X9 Y10]
+ (-0.0002922256721009919) [X7 Y8 Y9 X10]
+ (-8.814793509347423e-06) [Y2 Z3 Y4 Z13]
+ (-8.814793509347423e-06) [X2 Z3 X4 Z13]
+ (-8.814793509347423e-06) [Y3 Z4 Y5 Z12]
+ (-8.814793509347423e-06) [X3 Z4 X5 Z12]
+ (-7.95422474472729e-06) [Y10 Z11 Y12 Z13]
+ (-7.95422474472729e-06) [X10 Z11 X12 Z13]
+ (-5.974176919175227e-06) [Y5 X6 X10 Y11]
+ (-5.974176919175227e-06) [Y5 Y6 Y10 Y11]
+ (-5.974176919175227e-06) [X5 X6 X10 X11]
+ (-5.974176919175227e-06) [X5 Y6 Y10 X11]
+ (-5.275783490635115e-06) [Y3 X4 X12 Y13]
+ (-5.275783490635115e-06) [Y3 Y4 Y12 Y13]
+ (-5.275783490635115e-06) [X3 X4 X12 X13]
+ (-5.275783490635115e-06) [X3 Y4 Y12 X13]
+ (-4.281812071103581e-06) [Y4 Z5 Y6 Z11]
+ (-4.281812071103581e-06) [X4 Z5 X6 Z11]
+ (-4.281812071103581e-06) [Y5 Z6 Y7 Z10]
+ (-4.281812071103581e-06) [X5 Z6 X7 Z10]
+ (-3.539010018711007e-06) [Y2 Z3 Y4 Z12]
+ (-3.539010018711007e-06) [X2 Z3 X4 Z12]
+ (-3.539010018711007e-06) [Y3 Z4 Y5 Z13]
+ (-3.539010018711007e-06) [X3 Z4 X5 Z13]
+ (-3.1173664103805502e-06) [Y0 Z2 Z3 Y4]
+ (-3.1173664103805502e-06) [X0 Z2 Z3 X4]
+ (-2.1726380340216043e-06) [Y2 X3 X11 Y12]
+ (-2.1726380340216043e-06) [Y2 Y3 Y11 Y12]
+ (-2.1726380340216043e-06) [X2 X3 X11 X12]
+ (-2.1726380340216043e-06) [X2 Y3 Y11 X12]
+ (-1.8781495717797372e-06) [Z4 Y10 Z11 Y12]
+ (-1.8781495717797372e-06) [Z4 X10 Z11 X12]
+ (-1.8781495717797372e-06) [Z5 Y11 Z12 Y13]
+ (-1.8781495717797372e-06) [Z5 X11 Z12 X13]
+ (-1.6021751095710788e-06) [Z2 Y3 Z4 Y5]
+ (-1.6021751095710788e-06) [Z2 X3 Z4 X5]
+ (-1.1094125271682471e-06) [Z2 Y11 Z12 Y13]
+ (-1.1094125271682471e-06) [Z2 X11 Z12 X13]
+ (-1.1094125271682471e-06) [Z3 Y10 Z11 Y12]
+ (-1.1094125271682471e-06) [Z3 X10 Z11 X12]
+ (-6.628427324606532e-07) [Y8 X9 X11 Y12]
+ (-6.628427324606532e-07) [Y8 Y9 Y11 Y12]
+ (-6.628427324606532e-07) [X8 X9 X11 X12]
+ (-6.628427324606532e-07) [X8 Y9 Y11 X12]
+ (-5.627722068843211e-07) [Y0 X1 X11 Y12]
+ (-5.627722068843211e-07) [Y0 Y1 Y11 Y12]
+ (-5.627722068843211e-07) [X0 X1 X11 X12]
+ (-5.627722068843211e-07) [X0 Y1 Y11 X12]
+ (-5.471606081122012e-07) [Y1 X2 X11 Y12]
+ (-5.471606081122012e-07) [X1 Y2 Y11 X12]
+ (-3.606069291470204e-07) [Y0 Z1 Z2 Y4]
+ (-3.606069291470204e-07) [X0 Z1 Z2 X4]
+ (-3.606069291470204e-07) [Y1 Z3 Z4 Y5]
+ (-3.606069291470204e-07) [X1 Z3 Z4 X5]
+ (-2.594146142250722e-07) [Y2 Z3 Y4 Z6]
+ (-2.594146142250722e-07) [X2 Z3 X4 Z6]
+ (-2.594146142250722e-07) [Y3 Z4 Y5 Z7]
+ (-2.594146142250722e-07) [X3 Z4 X5 Z7]
+ (-2.1872303970622256e-07) [Y2 Z3 Y4 Z8]
+ (-2.1872303970622256e-07) [X2 Z3 X4 Z8]
+ (-2.1872303970622256e-07) [Y3 Z4 Y5 Z9]
+ (-2.1872303970622256e-07) [X3 Z4 X5 Z9]
+ (-1.9332121396673574e-07) [Y1 Y2 X3 X4]
+ (-1.9332121396673574e-07) [X1 X2 Y3 Y4]
+ (-1.6728571518031344e-07) [Y0 Z1 Z3 Y4]
+ (-1.6728571518031344e-07) [X0 Z1 Z3 X4]
+ (-1.6728571518031344e-07) [Y1 Z2 Z4 Y5]
+ (-1.6728571518031344e-07) [X1 Z2 Z4 X5]
+ (-3.2261055728479213e-09) [Y1 Y2 X5 X6]
+ (-3.2261055728479213e-09) [X1 X2 Y5 Y6]
+ (3.2261055728479213e-09) [Y1 X2 X5 Y6]
+ (3.2261055728479213e-09) [X1 Y2 Y5 X6]
+ (3.2284036577748254e-08) [Y4 Z5 Y6 Z12]
+ (3.2284036577748254e-08) [X4 Z5 X6 Z12]
+ (3.2284036577748254e-08) [Y5 Z6 Y7 Z13]
+ (3.2284036577748254e-08) [X5 Z6 X7 Z13]
+ (1.2919458287171507e-07) [Y1 Z2 Z3 Y5]
+ (1.2919458287171507e-07) [X1 Z2 Z3 X5]
+ (1.9332121396673574e-07) [Y1 X2 X3 Y4]
+ (1.9332121396673574e-07) [X1 Y2 Y3 X4]
+ (2.1989637793217256e-07) [Y2 X3 X5 Y6]
+ (2.1989637793217256e-07) [Y2 Y3 Y5 Y6]
+ (2.1989637793217256e-07) [X2 X3 X5 X6]
+ (2.1989637793217256e-07) [X2 Y3 Y5 X6]
+ (2.447264819861061e-07) [Y0 X1 X5 Y6]
+ (2.447264819861061e-07) [Y0 Y1 Y5 Y6]
+ (2.447264819861061e-07) [X0 X1 X5 X6]
+ (2.447264819861061e-07) [X0 Y1 Y5 X6]
+ (3.5706355164863455e-07) [Y0 X1 X3 Y4]
+ (3.5706355164863455e-07) [Y0 Y1 Y3 Y4]
+ (3.5706355164863455e-07) [X0 X1 X3 X4]
+ (3.5706355164863455e-07) [X0 Y1 Y3 X4]
+ (4.837953352744616e-07) [Y5 X6 X8 Y9]
+ (4.837953352744616e-07) [Y5 Y6 Y8 Y9]
+ (4.837953352744616e-07) [X5 X6 X8 X9]
+ (4.837953352744616e-07) [X5 Y6 Y8 X9]
+ (5.471606081122012e-07) [Y1 Y2 X11 X12]
+ (5.471606081122012e-07) [X1 X2 Y11 Y12]
+ (5.769436604956268e-07) [Y2 Z3 Y4 Z9]
+ (5.769436604956268e-07) [X2 Z3 X4 Z9]
+ (5.769436604956268e-07) [Y3 Z4 Y5 Z8]
+ (5.769436604956268e-07) [X3 Z4 X5 Z8]
+ (5.929280616496623e-07) [Z4 Y5 Z6 Y7]
+ (5.929280616496623e-07) [Z4 X5 Z6 X7]
+ (7.765082230419594e-07) [Y2 Z3 Y4 Z5]
+ (7.765082230419594e-07) [X2 Z3 X4 Z5]
+ (7.956667002018494e-07) [Y3 X4 X8 Y9]
+ (7.956667002018494e-07) [Y3 Y4 Y8 Y9]
+ (7.956667002018494e-07) [X3 X4 X8 X9]
+ (7.956667002018494e-07) [X3 Y4 Y8 X9]
+ (8.336695633596248e-07) [Z0 Y2 Z3 Y4]
+ (8.336695633596248e-07) [Z0 X2 Z3 X4]
+ (8.336695633596248e-07) [Z1 Y3 Z4 Y5]
+ (8.336695633596248e-07) [Z1 X3 Z4 X5]
+ (9.344969833027794e-07) [Z8 Y11 Z12 Y13]
+ (9.344969833027794e-07) [Z8 X11 Z12 X13]
+ (9.344969833027794e-07) [Z9 Y10 Z11 Y12]
+ (9.344969833027794e-07) [Z9 X10 Z11 X12]
+ (9.509134533153437e-07) [Z2 Y4 Z5 Y6]
+ (9.509134533153437e-07) [Z2 X4 Z5 X6]
+ (9.509134533153437e-07) [Z3 Y5 Z6 Y7]
+ (9.509134533153437e-07) [Z3 X5 Z6 X7]
+ (1.0357924794509102e-06) [Y6 X7 X11 Y12]
+ (1.0357924794509102e-06) [Y6 Y7 Y11 Y12]
+ (1.0357924794509102e-06) [X6 X7 X11 X12]
+ (1.0357924794509102e-06) [X6 Y7 Y11 X12]
+ (1.0632255068545498e-06) [Z2 Y10 Z11 Y12]
+ (1.0632255068545498e-06) [Z2 X10 Z11 X12]
+ (1.0632255068545498e-06) [Z3 Y11 Z12 Y13]
+ (1.0632255068545498e-06) [Z3 X11 Z12 X13]
+ (1.17080983124692e-06) [Z2 Y5 Z6 Y7]
+ (1.17080983124692e-06) [Z2 X5 Z6 X7]
+ (1.17080983124692e-06) [Z3 Y4 Z5 Y6]
+ (1.17080983124692e-06) [Z3 X4 Z5 X6]
+ (1.1907331150082986e-06) [Z0 Y3 Z4 Y5]
+ (1.1907331150082986e-06) [Z0 X3 Z4 X5]
+ (1.1907331150082986e-06) [Z1 Y2 Z3 Y4]
+ (1.1907331150082986e-06) [Z1 X2 Z3 X4]
+ (1.1953920764770296e-06) [Y2 Z3 Y4 Z7]
+ (1.1953920764770296e-06) [X2 Z3 X4 Z7]
+ (1.1953920764770296e-06) [Y3 Z4 Y5 Z6]
+ (1.1953920764770296e-06) [X3 Z4 X5 Z6]
+ (1.398024297283128e-06) [Y4 Z5 Y6 Z8]
+ (1.398024297283128e-06) [X4 Z5 X6 Z8]
+ (1.398024297283128e-06) [Y5 Z6 Y7 Z9]
+ (1.398024297283128e-06) [X5 Z6 X7 Z9]
+ (1.4548066907012886e-06) [Y3 X4 X6 Y7]
+ (1.4548066907012886e-06) [Y3 Y4 Y6 Y7]
+ (1.4548066907012886e-06) [X3 X4 X6 X7]
+ (1.4548066907012886e-06) [X3 Y4 Y6 X7]
+ (1.5973397157634326e-06) [Z8 Y10 Z11 Y12]
+ (1.5973397157634326e-06) [Z8 X10 Z11 X12]
+ (1.5973397157634326e-06) [Z9 Y11 Z12 Y13]
+ (1.5973397157634326e-06) [Z9 X11 Z12 X13]
+ (1.6149607777666937e-06) [Z0 Y11 Z12 Y13]
+ (1.6149607777666937e-06) [Z0 X11 Z12 X13]
+ (1.6149607777666937e-06) [Z1 Y10 Z11 Y12]
+ (1.6149607777666937e-06) [Z1 X10 Z11 X12]
+ (1.6923648480694775e-06) [Y4 Z5 Y6 Z10]
+ (1.6923648480694775e-06) [X4 Z5 X6 Z10]
+ (1.6923648480694775e-06) [Y5 Z6 Y7 Z11]
+ (1.6923648480694775e-06) [X5 Z6 X7 Z11]
+ (1.8163673523568374e-06) [Z4 Y11 Z12 Y13]
+ (1.8163673523568374e-06) [Z4 X11 Z12 X13]
+ (1.8163673523568374e-06) [Z5 Y10 Z11 Y12]
+ (1.8163673523568374e-06) [Z5 X10 Z11 X12]
+ (1.8540565444416816e-06) [Y4 Z5 Y6 Z7]
+ (1.8540565444416816e-06) [X4 Z5 X6 Z7]
+ (1.8551374669465055e-06) [Z6 Y10 Z11 Y12]
+ (1.8551374669465055e-06) [Z6 X10 Z11 X12]
+ (1.8551374669465055e-06) [Z7 Y11 Z12 Y13]
+ (1.8551374669465055e-06) [Z7 X11 Z12 X13]
+ (1.8818196325575895e-06) [Y4 Z5 Y6 Z9]
+ (1.8818196325575895e-06) [X4 Z5 X6 Z9]
+ (1.8818196325575895e-06) [Y5 Z6 Y7 Z8]
+ (1.8818196325575895e-06) [X5 Z6 X7 Z8]
+ (2.177732984651205e-06) [Z0 Y10 Z11 Y12]
+ (2.177732984651205e-06) [Z0 X10 Z11 X12]
+ (2.177732984651205e-06) [Z1 Y11 Z12 Y13]
+ (2.177732984651205e-06) [Z1 X11 Z12 X13]
+ (2.8909299463965484e-06) [Z6 Y11 Z12 Y13]
+ (2.8909299463965484e-06) [Z6 X11 Z12 X13]
+ (2.8909299463965484e-06) [Z7 Y10 Z11 Y12]
+ (2.8909299463965484e-06) [Z7 X10 Z11 X12]
+ (3.0992966492277034e-06) [Z0 Y4 Z5 Y6]
+ (3.0992966492277034e-06) [Z0 X4 Z5 X6]
+ (3.0992966492277034e-06) [Z1 Y5 Z6 Y7]
+ (3.0992966492277034e-06) [Z1 X5 Z6 X7]
+ (3.158559397086879e-06) [Y2 Z3 Y4 Z10]
+ (3.158559397086879e-06) [X2 Z3 X4 Z10]
+ (3.158559397086879e-06) [Y3 Z4 Y5 Z11]
+ (3.158559397086879e-06) [X3 Z4 X5 Z11]
+ (3.344023131213777e-06) [Z0 Y5 Z6 Y7]
+ (3.344023131213777e-06) [Z0 X5 Z6 X7]
+ (3.344023131213777e-06) [Z1 Y4 Z5 Y6]
+ (3.344023131213777e-06) [Z1 X4 Z5 X6]
+ (3.69451692413484e-06) [Y4 X5 X11 Y12]
+ (3.69451692413484e-06) [Y4 Y5 Y11 Y12]
+ (3.69451692413484e-06) [X4 X5 X11 X12]
+ (3.69451692413484e-06) [X4 Y5 Y11 X12]
+ (4.556473804160147e-06) [Y5 X6 X12 Y13]
+ (4.556473804160147e-06) [Y5 Y6 Y12 Y13]
+ (4.556473804160147e-06) [X5 X6 X12 X13]
+ (4.556473804160147e-06) [X5 Y6 Y12 X13]
+ (4.58875784073616e-06) [Y4 Z5 Y6 Z13]
+ (4.58875784073616e-06) [X4 Z5 X6 Z13]
+ (4.58875784073616e-06) [Y5 Z6 Y7 Z12]
+ (4.58875784073616e-06) [X5 Z6 X7 Z12]
+ (4.642979014260083e-06) [Y3 X4 X10 Y11]
+ (4.642979014260083e-06) [Y3 Y4 Y10 Y11]
+ (4.642979014260083e-06) [X3 X4 X10 X11]
+ (4.642979014260083e-06) [X3 Y4 Y10 X11]
+ (7.801538411349347e-06) [Y2 Z3 Y4 Z11]
+ (7.801538411349347e-06) [X2 Z3 X4 Z11]
+ (7.801538411349347e-06) [Y3 Z4 Y5 Z10]
+ (7.801538411349347e-06) [X3 Z4 X5 Z10]
+ (8.194105008157981e-06) [Z10 Y11 Z12 Y13]
+ (8.194105008157981e-06) [Z10 X11 Z12 X13]
+ (0.0002922256721009919) [Y7 Y8 X9 X10]
+ (0.0002922256721009919) [X7 X8 Y9 Y10]
+ (0.0004957972947277078) [Y2 Z4 Z5 Y6]
+ (0.0004957972947277078) [X2 Z4 Z5 X6]
+ (0.0011058984792014777) [Y0 Z1 Y2 Z5]
+ (0.0011058984792014777) [X0 Z1 X2 Z5]
+ (0.0011058984792014777) [Y1 Z2 Y3 Z4]
+ (0.0011058984792014777) [X1 Z2 X3 Z4]
+ (0.0016639606587698396) [Y2 Z3 Z4 Y6]
+ (0.0016639606587698396) [X2 Z3 Z4 X6]
+ (0.0016639606587698396) [Y3 Z5 Z6 Y7]
+ (0.0016639606587698396) [X3 Z5 Z6 X7]
+ (0.0017560659607176207) [Y0 Z1 Y2 Z11]
+ (0.0017560659607176207) [X0 Z1 X2 Z11]
+ (0.0017560659607176207) [Y1 Z2 Y3 Z10]
+ (0.0017560659607176207) [X1 Z2 X3 Z10]
+ (0.002326234845611509) [Y0 Z1 Y2 Z13]
+ (0.002326234845611509) [X0 Z1 X2 Z13]
+ (0.002326234845611509) [Y1 Z2 Y3 Z12]
+ (0.002326234845611509) [X1 Z2 X3 Z12]
+ (0.002745827315776045) [Y0 X1 X4 Y5]
+ (0.002745827315776045) [X0 Y1 Y4 X5]
+ (0.0029297682762827) [Y0 Z1 Y2 Z9]
+ (0.0029297682762827) [X0 Z1 X2 Z9]
+ (0.0029297682762827) [Y1 Z2 Y3 Z8]
+ (0.0029297682762827) [X1 Z2 X3 Z8]
+ (0.003276965063178576) [Y0 Z1 Y2 Z3]
+ (0.003276965063178576) [X0 Z1 X2 Z3]
+ (0.003347626468870265) [Y0 Z1 Y2 Z7]
+ (0.003347626468870265) [X0 Z1 X2 Z7]
+ (0.003347626468870265) [Y1 Z2 Y3 Z6]
+ (0.003347626468870265) [X1 Z2 X3 Z6]
+ (0.003555258969248279) [Y0 Z1 Y2 Z10]
+ (0.003555258969248279) [X0 Z1 X2 Z10]
+ (0.003555258969248279) [Y1 Z2 Y3 Z11]
+ (0.003555258969248279) [X1 Z2 X3 Z11]
+ (0.005143382384868528) [Y3 Y4 X5 X6]
+ (0.005143382384868528) [X3 X4 Y5 Y6]
+ (0.005283785753385714) [Y0 X1 X12 Y13]
+ (0.005283785753385714) [X0 Y1 Y12 X13]
+ (0.005530742148263161) [Y0 Z1 Y2 Z4]
+ (0.005530742148263161) [X0 Z1 X2 Z4]
+ (0.005530742148263161) [Y1 Z2 Y3 Z5]
+ (0.005530742148263161) [X1 Z2 X3 Z5]
+ (0.006087836803982384) [Y8 X9 X12 Y13]
+ (0.006087836803982384) [X8 Y9 Y12 X13]
+ (0.006509361234142669) [Y0 X1 X8 Y9]
+ (0.006509361234142669) [X0 Y1 Y8 X9]
+ (0.006888204542324463) [Y0 X1 X6 Y7]
+ (0.006888204542324463) [X0 Y1 Y6 X7]
+ (0.006901250034128021) [Y0 Z1 Y2 Z12]
+ (0.006901250034128021) [X0 Z1 X2 Z12]
+ (0.006901250034128021) [Y1 Z2 Y3 Z13]
+ (0.006901250034128021) [X1 Z2 X3 Z13]
+ (0.007156920529851534) [Y4 X5 X8 Y9]
+ (0.007156920529851534) [X4 Y5 Y8 X9]
+ (0.007731432846024306) [Y0 X1 X10 Y11]
+ (0.007731432846024306) [X0 Y1 Y10 X11]
+ (0.008032546696249686) [Y0 Z1 Y2 Z6]
+ (0.008032546696249686) [X0 Z1 X2 Z6]
+ (0.008032546696249686) [Y1 Z2 Y3 Z7]
+ (0.008032546696249686) [X1 Z2 X3 Z7]
+ (0.009560716495339834) [Y8 X9 X10 Y11]
+ (0.009560716495339834) [X8 Y9 Y10 X11]
+ (0.011055016686679627) [Y0 Z1 Y2 Z8]
+ (0.011055016686679627) [X0 Z1 X2 Z8]
+ (0.011055016686679627) [Y1 Z2 Y3 Z9]
+ (0.011055016686679627) [X1 Z2 X3 Z9]
+ (0.011285144615351716) [Y5 Y6 X11 X12]
+ (0.011285144615351716) [X5 X6 Y11 Y12]
+ (0.011982342581831161) [Y4 X5 X6 Y7]
+ (0.011982342581831161) [X4 Y5 Y6 X7]
+ (0.013873400068330589) [Y6 X7 X8 Y9]
+ (0.013873400068330589) [X6 Y7 Y8 X9]
+ (0.014583638325580841) [Y0 X1 X2 Y3]
+ (0.014583638325580841) [X0 Y1 Y2 X3]
+ (0.01557722707364272) [Y2 X3 X12 Y13]
+ (0.01557722707364272) [X2 Y3 Y12 X13]
+ (0.01736605326165715) [Y6 X7 X12 Y13]
+ (0.01736605326165715) [X6 Y7 Y12 X13]
+ (0.017680137653392103) [Y4 X5 X10 Y11]
+ (0.017680137653392103) [X4 Y5 Y10 X11]
+ (0.017825011931478407) [Y6 X7 X10 Y11]
+ (0.017825011931478407) [X6 Y7 Y10 X11]
+ (0.019028318717604294) [Y3 X4 X11 Y12]
+ (0.019028318717604294) [X3 Y4 Y11 X12]
+ (0.025384663693933537) [Y2 X3 X10 Y11]
+ (0.025384663693933537) [X2 Y3 Y10 X11]
+ (0.02868521730496169) [Y10 X11 X12 Y13]
+ (0.02868521730496169) [X10 Y11 Y12 X13]
+ (0.031143804193790196) [Y2 X3 X6 Y7]
+ (0.031143804193790196) [X2 Y3 Y6 X7]
+ (0.035839557184684075) [Y2 X3 X4 Y5]
+ (0.035839557184684075) [X2 Y3 Y4 X5]
+ (0.03619409348649705) [Y2 X3 X8 Y9]
+ (0.03619409348649705) [X2 Y3 Y8 X9]
+ (0.038314669372682825) [Y4 X5 X12 Y13]
+ (0.038314669372682825) [X4 Y5 Y12 X13]
+ (0.1043306148395636) [Z0 Y1 Z2 Y3]
+ (0.1043306148395636) [Z0 X1 Z2 X3]
+ (-0.22847946315128354) [Y6 Z7 Z8 Z9 Y10]
+ (-0.22847946315128354) [X6 Z7 Z8 Z9 X10]
+ (-0.22847946315128354) [Y7 Z8 Z9 Z10 Y11]
+ (-0.22847946315128354) [X7 Z8 Z9 Z10 X11]
+ (-0.12133242245186002) [Y3 Z4 Z5 Z6 Y7]
+ (-0.12133242245186002) [X3 Z4 Z5 Z6 X7]
+ (-0.12133242245185999) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133242245185999) [X2 Z3 Z4 Z5 X6]
+ (-3.2041422508012704e-06) [Y0 Z1 Z2 Z3 Y4]
+ (-3.2041422508012704e-06) [X0 Z1 Z2 Z3 X4]
+ (-3.2041422508012704e-06) [Y1 Z2 Z3 Z4 Y5]
+ (-3.2041422508012704e-06) [X1 Z2 Z3 Z4 X5]
+ (-0.05608449432705888) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (-0.05608449432705888) [Z0 X6 Z7 Z8 Z9 X10]
+ (-0.05608449432705888) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (-0.05608449432705888) [Z1 X7 Z8 Z9 Z10 X11]
+ (-0.0560071356214792) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (-0.0560071356214792) [Z0 X7 Z8 Z9 Z10 X11]
+ (-0.0560071356214792) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (-0.0560071356214792) [Z1 X6 Z7 Z8 Z9 X10]
+ (-0.03276748588586587) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.03276748588586587) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.03276748588586587) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.03276748588586587) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.030787440727002863) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (-0.030787440727002863) [Z6 X7 Z8 Z9 Z10 X11]
+ (-0.027114878571486287) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.027114878571486287) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.027114878571486287) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.027114878571486287) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.02599620626467622) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.02599620626467622) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.025104907973031887) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (-0.025104907973031887) [X6 Z7 Z8 Z9 X10 Z12]
+ (-0.025104907973031887) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (-0.025104907973031887) [X7 Z8 Z9 Z10 X11 Z13]
+ (-0.02438898999224113) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (-0.02438898999224113) [Z2 X7 Z8 Z9 Z10 X11]
+ (-0.02438898999224113) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (-0.02438898999224113) [Z3 X6 Z7 Z8 Z9 X10]
+ (-0.01902037388001724) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (-0.01902037388001724) [Z2 X6 Z7 Z8 Z9 X10]
+ (-0.01902037388001724) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (-0.01902037388001724) [Z3 X7 Z8 Z9 Z10 X11]
+ (-0.018266758584798395) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (-0.018266758584798395) [Z4 X6 Z7 Z8 Z9 X10]
+ (-0.018266758584798395) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (-0.018266758584798395) [Z5 X7 Z8 Z9 Z10 X11]
+ (-0.01756111644834318) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.01756111644834318) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.01756111644834318) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.01756111644834318) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.01221498531918814) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.01221498531918814) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.01221498531918814) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.01221498531918814) [X4 Z5 X6 X11 Z12 X13]
+ (-0.01221498531918814) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.01221498531918814) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.01221498531918814) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.01221498531918814) [X5 Z6 X7 X10 Z11 X12]
+ (-0.011755995239949121) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.011755995239949121) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.011755995239949121) [X3 Z4 Z5 X6 X8 X9]
+ (-0.011755995239949121) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.011307208035953573) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (-0.011307208035953573) [X6 Z7 Z8 Z9 X10 Z11]
+ (-0.010959994618283944) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (-0.010959994618283944) [Z4 X7 Z8 Z9 Z10 X11]
+ (-0.010959994618283944) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (-0.010959994618283944) [Z5 X6 Z7 Z8 Z9 X10]
+ (-0.010540434336110655) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (-0.010540434336110655) [X6 Z7 Z8 Z9 X10 Z13]
+ (-0.010540434336110655) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (-0.010540434336110655) [X7 Z8 Z9 Z10 X11 Z12]
+ (-0.00876485821837431) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.00876485821837431) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.00876485821837431) [X2 Z3 Z4 X5 X11 X12]
+ (-0.00876485821837431) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.00876485821837431) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.00876485821837431) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.00876485821837431) [X3 X4 X10 Z11 Z12 X13]
+ (-0.00876485821837431) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.008125248410396928) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.008125248410396928) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.005805121208394059) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.005805121208394059) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.005805121208394059) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.005805121208394059) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.0056526073143795775) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.0056526073143795775) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.0056526073143795775) [X0 X1 X3 Z4 Z5 X6]
+ (-0.0056526073143795775) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.005368616112223888) [Y2 X3 X7 Z8 Z9 Y10]
+ (-0.005368616112223888) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (-0.005368616112223888) [X2 X3 X7 Z8 Z9 X10]
+ (-0.005368616112223888) [X2 Y3 Y7 Z8 Z9 X10]
+ (-0.005143382384868528) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.005143382384868528) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.005143382384868528) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.005143382384868528) [X2 Z3 X4 X5 Z6 X7]
+ (-0.004684920227379422) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.004684920227379422) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668615265773195) [Y1 Y2 X7 Z8 Z9 X10]
+ (-0.004668615265773195) [X1 X2 Y7 Z8 Z9 Y10]
+ (-0.004575015188516512) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.004575015188516512) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.004424843669061684) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.004424843669061684) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.004158830716208697) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.004158830716208697) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.004158830716208697) [X3 Z4 Z5 X6 X12 X13]
+ (-0.004158830716208697) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.0034938003687743358) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.0034938003687743358) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.0034938003687743358) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.0034938003687743358) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.002779040764112858) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.002779040764112858) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.0022939556229128752) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.0022939556229128752) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.001799193008530659) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.001799193008530659) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.0017278745819633567) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.0017278745819633567) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.0009298407038364259) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298407038364259) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298407038364259) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298407038364259) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533831053519735) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.0008533831053519735) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.0008144692855833737) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.0008144692855833737) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.0008144692855833737) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.0008144692855833737) [X3 Z4 Z5 Z6 X7 Z11]
+ (-0.0002922256721009919) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (-0.0002922256721009919) [Y6 Z7 Y8 X9 Z10 X11]
+ (-0.0002922256721009919) [X6 Z7 X8 Y9 Z10 Y11]
+ (-0.0002922256721009919) [X6 Z7 X8 X9 Z10 X11]
+ (-8.774724347562421e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.774724347562421e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.774724347562421e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.774724347562421e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.518288881196069e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.518288881196069e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.518288881196069e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.518288881196069e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.444267484090655e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.444267484090655e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.444267484090655e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.444267484090655e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.524204553450506e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.524204553450506e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.524204553450506e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.524204553450506e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-6.2900197593787534e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.2900197593787534e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.2900197593787534e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.2900197593787534e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.974176919175227e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (-5.974176919175227e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (-5.275783490635115e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (-5.275783490635115e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (-4.642979014260083e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (-4.642979014260083e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (-4.556473804160147e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (-4.556473804160147e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (-4.253118794052789e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.253118794052789e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-3.769583630561669e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.769583630561669e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.69451692413484e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (-3.69451692413484e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (-3.610242272637798e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.610242272637798e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.610242272637798e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.610242272637798e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.3130170976241116e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.3130170976241116e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.2774383038674734e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.2774383038674734e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.2774383038674734e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.2774383038674734e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.2111874558229253e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.2111874558229253e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.2111874558229253e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.2111874558229253e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.151295989683024e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.151295989683024e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.1173664103805502e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (-3.1173664103805502e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (-3.0882457400985665e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (-3.0882457400985665e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (-2.1726380340216043e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (-2.1726380340216043e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (-1.4548066907012886e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (-1.4548066907012886e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (-1.3304568634717661e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.3304568634717661e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.2393113953097312e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (-1.2393113953097312e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (-1.2393113953097312e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (-1.2393113953097312e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (-1.2282691218173155e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.2282691218173155e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.0357924794509102e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (-1.0357924794509102e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (-9.306343034142699e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (-9.306343034142699e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (-9.306343034142699e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (-9.306343034142699e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (-7.956667002018494e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (-7.956667002018494e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (-6.628427324606532e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (-6.628427324606532e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (-5.627722068843211e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (-5.627722068843211e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (-4.837953352744616e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (-4.837953352744616e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (-3.5706355164863455e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (-3.5706355164863455e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (-3.328039687703244e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.328039687703244e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.236183435035144e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (-3.236183435035144e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (-3.236183435035144e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (-3.236183435035144e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (-2.447264819861061e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (-2.447264819861061e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (-2.1989637793217256e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (-2.1989637793217256e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (-1.8290428645143729e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (-1.8290428645143729e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (-1.8290428645143729e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (-1.8290428645143729e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (-8.649129635125664e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (-8.649129635125664e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (-8.649129635125664e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (-8.649129635125664e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (-8.057465371514227e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (-8.057465371514227e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (-8.057465371514227e-08) [X1 Z2 Z3 X4 X10 X11]
+ (-8.057465371514227e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (-1.839565802319243e-08) [Y0 Z1 X2 X10 Z11 Y12]
+ (-1.839565802319243e-08) [X0 Z1 Y2 Y10 Z11 X12]
+ (-1.839565802319243e-08) [Y1 Z2 X3 X11 Z12 Y13]
+ (-1.839565802319243e-08) [X1 Z2 Y3 Y11 Z12 X13]
+ (1.0351498017488654e-09) [Y0 Z1 Z2 Z3 Y4 Z7]
+ (1.0351498017488654e-09) [X0 Z1 Z2 Z3 X4 Z7]
+ (1.0351498017488654e-09) [Y1 Z2 Z3 Z4 Y5 Z6]
+ (1.0351498017488654e-09) [X1 Z2 Z3 Z4 X5 Z6]
+ (2.2702082925926807e-08) [Y0 Z1 Z2 X3 X5 Y6]
+ (2.2702082925926807e-08) [Y0 Z1 Z2 Y3 Y5 Y6]
+ (2.2702082925926807e-08) [X0 Z1 Z2 X3 X5 X6]
+ (2.2702082925926807e-08) [X0 Z1 Z2 Y3 Y5 X6]
+ (2.2702082925926807e-08) [Y1 X2 X4 Z5 Z6 Y7]
+ (2.2702082925926807e-08) [Y1 Y2 Y4 Z5 Z6 Y7]
+ (2.2702082925926807e-08) [X1 X2 X4 Z5 Z6 X7]
+ (2.2702082925926807e-08) [X1 Y2 Y4 Z5 Z6 X7]
+ (2.5928188498386787e-08) [Y0 Z1 X2 X4 Z5 Y6]
+ (2.5928188498386787e-08) [X0 Z1 Y2 Y4 Z5 X6]
+ (2.5928188498386787e-08) [Y1 Z2 X3 X5 Z6 Y7]
+ (2.5928188498386787e-08) [X1 Z2 Y3 Y5 Z6 X7]
+ (1.1076529196140807e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (1.1076529196140807e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (1.1076529196140807e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (1.1076529196140807e-07) [X0 Z1 X2 X11 Z12 X13]
+ (1.1076529196140807e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (1.1076529196140807e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (1.1076529196140807e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (1.1076529196140807e-07) [X1 Z2 X3 X10 Z11 X12]
+ (1.2919458287171507e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (1.2919458287171507e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (1.348496888265591e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (1.348496888265591e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (1.348496888265591e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (1.348496888265591e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (1.3807579439931893e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (1.3807579439931893e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (1.3807579439931893e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (1.3807579439931893e-07) [X0 Z1 X2 X5 Z6 X7]
+ (1.3807579439931893e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (1.3807579439931893e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (1.3807579439931893e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (1.3807579439931893e-07) [X1 Z2 X3 X4 Z5 X6]
+ (1.6077787732539143e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (1.6077787732539143e-07) [X0 Z1 X2 X4 Z5 X6]
+ (1.6077787732539143e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (1.6077787732539143e-07) [X1 Z2 X3 X5 Z6 X7]
+ (1.839394362535114e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (1.839394362535114e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (1.839394362535114e-07) [X1 Z2 Z3 X4 X6 X7]
+ (1.839394362535114e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (1.9332121396673574e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (1.9332121396673574e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (1.9332121396673574e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (1.9332121396673574e-07) [X0 Z1 X2 X3 Z4 X5]
+ (2.1989637793217256e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (2.1989637793217256e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (2.3712704715225776e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (2.3712704715225776e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (2.3712704715225776e-07) [X1 Z2 Z3 X4 X8 X9]
+ (2.3712704715225776e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (2.447264819861061e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (2.447264819861061e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (3.086770918954478e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (3.086770918954478e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (3.086770918954478e-07) [X1 Z2 Z3 X4 X12 X13]
+ (3.086770918954478e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (3.328039687703244e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.328039687703244e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.5706355164863455e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (3.5706355164863455e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (4.837953352744616e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (4.837953352744616e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (5.287649500890257e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (5.287649500890257e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (5.287649500890257e-07) [X0 Z1 Z2 X3 X11 X12]
+ (5.287649500890257e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (5.287649500890257e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (5.287649500890257e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (5.287649500890257e-07) [X1 X2 X10 Z11 Z12 X13]
+ (5.287649500890257e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (5.627722068843211e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (5.627722068843211e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (5.927350275905069e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (5.927350275905069e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (5.927350275905069e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (5.927350275905069e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (6.395302420505151e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (6.395302420505151e-07) [X0 Z1 X2 X10 Z11 X12]
+ (6.395302420505151e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (6.395302420505151e-07) [X1 Z2 X3 X11 Z12 X13]
+ (6.579259000738329e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (6.579259000738329e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (6.579259000738329e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (6.579259000738329e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (6.628427324606532e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (6.628427324606532e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (6.733096813056356e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (6.733096813056356e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (6.733096813056356e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (6.733096813056356e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (7.956667002018494e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (7.956667002018494e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (1.0357924794509102e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (1.0357924794509102e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (1.2282691218173155e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.2282691218173155e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.3304568634717661e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.3304568634717661e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.4548066907012886e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (1.4548066907012886e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (2.1726380340216043e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (2.1726380340216043e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (3.0882457400985665e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (3.0882457400985665e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (3.151295989683024e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.151295989683024e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.3130170976241116e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.3130170976241116e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.3342618798708895e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.3342618798708895e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.69451692413484e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (3.69451692413484e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (4.18380885849369e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.18380885849369e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.556473804160147e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (4.556473804160147e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (4.642979014260083e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (4.642979014260083e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (5.275783490635115e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (5.275783490635115e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (5.974176919175227e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (5.974176919175227e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (7.73587055796807e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (7.73587055796807e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (7.73587055796807e-05) [X0 X1 X7 Z8 Z9 X10]
+ (7.73587055796807e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (0.0004957972947277078) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.0004957972947277078) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.0006650303474343604) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.0006650303474343604) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.0006650303474343604) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.0006650303474343604) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.0008533831053519735) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.0008533831053519735) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.001609533515665922) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.001609533515665922) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.001609533515665922) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.001609533515665922) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.001667613749769031) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.001667613749769031) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.001667613749769031) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.001667613749769031) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.0017278745819633567) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.0017278745819633567) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.001799193008530659) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.001799193008530659) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.0022939556229128752) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.0022939556229128752) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.0024629166210178954) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.0024629166210178954) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.0024629166210178954) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.0024629166210178954) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.003961569372681906) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.003961569372681906) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.003961569372681906) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.003961569372681906) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.004424843669061684) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.004424843669061684) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.004575015188516512) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.004575015188516512) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.004668615265773195) [Y1 X2 X7 Z8 Z9 Y10]
+ (0.004668615265773195) [X1 Y2 Y7 Z8 Z9 X10]
+ (0.004684920227379422) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.004684920227379422) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.005324817364137306) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.005324817364137306) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.005324817364137306) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.005324817364137306) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.007306763966514453) [Y4 X5 X7 Z8 Z9 Y10]
+ (0.007306763966514453) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (0.007306763966514453) [X4 X5 X7 Z8 Z9 X10]
+ (0.007306763966514453) [X4 Y5 Y7 Z8 Z9 X10]
+ (0.00796083963673744) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.00796083963673744) [X4 Z5 X6 X10 Z11 X12]
+ (0.00796083963673744) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.00796083963673744) [X5 Z6 X7 X11 Z12 X13]
+ (0.008125248410396928) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.008125248410396928) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.008890680340573864) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.008890680340573864) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.008890680340573864) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.008890680340573864) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.010263460499229988) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.010263460499229988) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.010263460499229988) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.010263460499229988) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.014411189770874319) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.014411189770874319) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.014411189770874319) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.014411189770874319) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.014564473636921238) [Y7 Z8 Z9 X10 X12 Y13]
+ (0.014564473636921238) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (0.014564473636921238) [X7 Z8 Z9 X10 X12 X13]
+ (0.014564473636921238) [X7 Z8 Z9 Y10 Y12 X13]
+ (0.015225659056457697) [Y3 Z4 Z5 X6 X10 Y11]
+ (0.015225659056457697) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (0.015225659056457697) [X3 Z4 Z5 X6 X10 X11]
+ (0.015225659056457697) [X3 Z4 Z5 Y6 Y10 X11]
+ (0.015588277863367294) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (0.015588277863367294) [X2 Z3 X4 X10 Z11 X12]
+ (0.015588277863367294) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (0.015588277863367294) [X3 Z4 X5 X11 Z12 X13]
+ (0.02017582495592558) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.02017582495592558) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.02017582495592558) [X4 Z5 Z6 X7 X11 X12]
+ (0.02017582495592558) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.02017582495592558) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.02017582495592558) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.02017582495592558) [X5 X6 X10 Z11 Z12 X13]
+ (0.02017582495592558) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.024353136081741607) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.024353136081741607) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.024353136081741607) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.024353136081741607) [X2 Z3 X4 X11 Z12 X13]
+ (0.024353136081741607) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.024353136081741607) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.024353136081741607) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.024353136081741607) [X3 Z4 X5 X10 Z11 X12]
+ (0.045879424024966364) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.045879424024966364) [X0 Z2 Z3 Z4 Z5 X6]
+ (-6.631261997064272e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.631261997064272e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.631261997064272e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.631261997064272e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.595081377448922e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.595081377448922e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.595081377448922e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.595081377448922e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.042743260056719266) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.042743260056719266) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (0.04274326005671927) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.04274326005671927) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (-0.019257452998958195) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019257452998958195) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.019028318717604294) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.019028318717604294) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.01602466609174238) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (-0.01602466609174238) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (-0.015225659056457697) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.015225659056457697) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.014603742409695385) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (-0.014603742409695385) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (-0.014564473636921238) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.014564473636921238) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.011755995239949121) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.011755995239949121) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.011285144615351716) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-0.011285144615351716) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-0.009841802921695154) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.009841802921695154) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008469833338918565) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.008469833338918565) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.007306763966514453) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.007306763966514453) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.005923799555839632) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-0.005923799555839632) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-0.0057084798531198385) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (-0.0057084798531198385) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (-0.0057084798531198385) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (-0.0057084798531198385) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (-0.0056526073143795775) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.0056526073143795775) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.005368616112223888) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.005368616112223888) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005262631032834302) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (-0.005262631032834302) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (-0.005262631032834302) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (-0.005262631032834302) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (-0.005114464086058734) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (-0.005114464086058734) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (-0.005114464086058734) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (-0.005114464086058734) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (-0.005114464086058734) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005114464086058734) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005114464086058734) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.005114464086058734) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.004158830716208697) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.004158830716208697) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0033566679215093834) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.0033566679215093834) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.0033566679215093834) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.0033566679215093834) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.0032675148969670237) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.0032675148969670237) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.0032675148969670237) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.0032675148969670237) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.002779040764112858) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.002779040764112858) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.0026860422746872174) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.0026860422746872174) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.0026860422746872174) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.0026860422746872174) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.0022939556229128752) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939556229128752) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.0022939556229128752) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939556229128752) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.0009581676927238608) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.0009581676927238608) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.0009581676927238608) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.0009581676927238608) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.0009581676927238608) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.0009581676927238608) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.0009581676927238608) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.0009581676927238608) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.0005940157670611062) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (-0.0005940157670611062) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (-0.0005940157670611062) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (-0.0005940157670611062) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (-0.0005940157670611062) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (-0.0005940157670611062) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (-0.0005940157670611062) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (-0.0005940157670611062) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (-0.00044584882028553713) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (-0.00044584882028553713) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (-0.00044584882028553713) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (-0.00044584882028553713) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (-0.0002464408136808765) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0002464408136808765) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.00013838603680294765) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.00013838603680294765) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.00013838603680294765) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.00013838603680294765) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.73587055796807e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (-7.73587055796807e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.6103375121580992e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103375121580992e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.6103375121580992e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.6103375121580992e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.531661425434164e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.531661425434164e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.531661425434164e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.531661425434164e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.805982136025168e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.805982136025168e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.805982136025168e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.805982136025168e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.089728703606245e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.089728703606245e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.089728703606245e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.089728703606245e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.652106408843835e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.652106408843835e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.652106408843835e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.652106408843835e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481752063146013e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.481752063146013e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481752063146013e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.481752063146013e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.071403707897562e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.071403707897562e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.071403707897562e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.071403707897562e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.7345784281241365e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.7345784281241365e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.7345784281241365e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.7345784281241365e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.7287814761039475e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.7287814761039475e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.7287814761039475e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.7287814761039475e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.253118794052789e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.253118794052789e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.769583630561669e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.769583630561669e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.7455106563394407e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (-2.7455106563394407e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (-2.7455106563394407e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (-2.7455106563394407e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (-2.7455106563394407e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (-2.7455106563394407e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (-2.7455106563394407e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (-2.7455106563394407e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (-2.3609472275035984e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609472275035984e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609472275035984e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.3609472275035984e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-2.1031634925449045e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.1031634925449045e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.1031634925449045e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.1031634925449045e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.011074040811356e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.011074040811356e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.011074040811356e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.011074040811356e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.942946544435716e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.942946544435716e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.942946544435716e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.942946544435716e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6540900744203835e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.6540900744203835e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6540900744203835e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.6540900744203835e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.5224582060591445e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.5224582060591445e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.5224582060591445e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.5224582060591445e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.5224582060591445e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224582060591445e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224582060591445e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.5224582060591445e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.2282691218173155e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2282691218173155e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.2282691218173155e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2282691218173155e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.867608672393008e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867608672393008e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867608672393008e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.867608672393008e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.189870518431925e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.189870518431925e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.175164937123327e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (-6.175164937123327e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (-5.471606081122012e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (-5.471606081122012e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (-4.5233394061654827e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.5233394061654827e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.328039687703244e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328039687703244e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.328039687703244e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328039687703244e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.086770918954478e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (-3.086770918954478e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (-2.8885647001489906e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8885647001489906e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8885647001489906e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.8885647001489906e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.3712704715225776e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (-2.3712704715225776e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (-1.839394362535114e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (-1.839394362535114e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (-8.057465371514227e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (-8.057465371514227e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (-6.772951835322876e-08) [Y1 Z2 Z3 X4 X7 Z8 Z9 Y10]
+ (-6.772951835322876e-08) [X1 Z2 Z3 Y4 Y7 Z8 Z9 X10]
+ (-3.2261055728479213e-09) [Y0 Z1 Z2 Y3 X4 Z5 Z6 X7]
+ (-3.2261055728479213e-09) [X0 Z1 Z2 X3 Y4 Z5 Z6 Y7]
+ (3.2261055728479213e-09) [Y0 Z1 Z2 X3 X4 Z5 Z6 Y7]
+ (3.2261055728479213e-09) [X0 Z1 Z2 Y3 Y4 Z5 Z6 X7]
+ (6.04679041371712e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (6.04679041371712e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (6.04679041371712e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (6.04679041371712e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (6.772951835322876e-08) [Y1 Z2 Z3 Y4 X7 Z8 Z9 X10]
+ (6.772951835322876e-08) [X1 Z2 Z3 X4 Y7 Z8 Z9 Y10]
+ (8.057465371514227e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (8.057465371514227e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (9.208945173452441e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.208945173452441e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.208945173452441e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.208945173452441e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.7035434569788487e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035434569788487e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035434569788487e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.7035434569788487e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.839394362535114e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (1.839394362535114e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (2.3712704715225776e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (2.3712704715225776e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (3.086770918954478e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (3.086770918954478e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (3.427350837608605e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (3.427350837608605e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (3.427350837608605e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (3.427350837608605e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (4.5233394061654827e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.5233394061654827e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (4.56111704402371e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (4.56111704402371e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (4.56111704402371e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (4.56111704402371e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (5.471606081122012e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (5.471606081122012e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (6.175164937123327e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (6.175164937123327e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (7.189870518431925e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.189870518431925e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (7.988467881631231e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (7.988467881631231e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (7.988467881631231e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (7.988467881631231e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (1.3304568634717661e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.3304568634717661e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.3304568634717661e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.3304568634717661e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.6288377836244216e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.6288377836244216e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.6288377836244216e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.6288377836244216e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.6893056877579065e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.6893056877579065e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.6893056877579065e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.6893056877579065e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (3.211763893818027e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.211763893818027e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.211763893818027e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.211763893818027e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.211763893818027e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.211763893818027e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.211763893818027e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.211763893818027e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.3130170976241116e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.3130170976241116e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.3130170976241116e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.3130170976241116e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (3.3342618798708895e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.3342618798708895e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (3.5443574445022386e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (3.5443574445022386e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (3.5443574445022386e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (3.5443574445022386e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (3.5443574445022386e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (3.5443574445022386e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (3.5443574445022386e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (3.5443574445022386e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (4.18380885849369e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.18380885849369e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (7.73587055796807e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (7.73587055796807e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.0002464408136808765) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (0.0002464408136808765) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (0.0008533831053519734) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.0008533831053519734) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.0008533831053519734) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.0008533831053519734) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.0010435237096517262) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.0010435237096517262) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.0010435237096517262) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.0010435237096517262) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.0012803055942721608) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.0012803055942721608) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.0012803055942721608) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.0012803055942721608) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.0013038029824963852) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (0.0013038029824963852) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (0.0013038029824963852) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (0.0013038029824963852) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (0.002261970675220248) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (0.002261970675220248) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (0.002261970675220248) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (0.002261970675220248) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (0.002261970675220248) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (0.002261970675220248) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (0.002261970675220248) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (0.002261970675220248) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (0.003989845257183604) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (0.003989845257183604) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (0.003989845257183604) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (0.003989845257183604) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (0.004158830716208697) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.004158830716208697) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.004311038606618749) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.004311038606618749) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.004311038606618749) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.004311038606618749) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.004636973515781544) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.004636973515781544) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.004636973515781544) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.004636973515781544) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.005241543596060072) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (0.005241543596060072) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (0.005241543596060072) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (0.005241543596060072) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (0.005368616112223888) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.005368616112223888) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.005379929632863018) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.005379929632863018) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.005379929632863018) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.005379929632863018) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.0056526073143795775) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.0056526073143795775) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.005923799555839632) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (0.005923799555839632) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (0.007306763966514453) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (0.007306763966514453) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.008469833338918565) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.008469833338918565) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.009612546721639666) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (0.009612546721639666) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (0.009612546721639666) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (0.009612546721639666) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (0.009841802921695154) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.009841802921695154) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.011285144615351716) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (0.011285144615351716) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (0.011755995239949121) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.011755995239949121) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.014564473636921238) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.014564473636921238) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.014603742409695385) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (0.014603742409695385) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (0.015225659056457697) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.015225659056457697) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.01602466609174238) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (0.01602466609174238) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (0.018888995077297854) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (0.018888995077297854) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (0.018888995077297854) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.018888995077297854) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.019028318717604294) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.019028318717604294) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.019257452998958195) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (0.019257452998958195) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (0.02143398011675084) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.02143398011675084) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.02143398011675084) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.02143398011675084) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.022528354253362432) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.022528354253362432) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.023145221660809023) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (0.023145221660809023) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (0.024282031618294043) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (0.024282031618294043) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (0.024755507979807363) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.024755507979807363) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.024755507979807363) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.024755507979807363) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.02563721281338205) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (0.02563721281338205) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (0.02563721281338205) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (0.02563721281338205) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (0.02873079799899301) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.02873079799899301) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.02873079799899301) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.02873079799899301) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.02990381345566941) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.02990381345566941) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.02990381345566941) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.02990381345566941) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.03560840034928923) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.03560840034928923) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0393181072386012) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0393181072386012) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0393181072386012) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0393181072386012) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.03935925038950275) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.03935925038950275) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.03935925038950275) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.03935925038950275) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.03956454805228207) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (0.03956454805228207) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (0.03956454805228207) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.03956454805228207) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.04171881405292486) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (0.04171881405292486) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (0.04171881405292486) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (0.04171881405292486) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (0.045879424024966364) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.045879424024966364) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (0.04764261360876449) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (0.04764261360876449) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (0.04764261360876449) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (0.04764261360876449) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (0.2816433575178901) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.2816433575178901) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.28164335751789016) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.28164335751789016) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.3693713755963822) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.3693713755963822) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.3693713755963823) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.3693713755963823) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0585921517848148) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0585921517848148) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.019299499855671975) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.019299499855671975) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019299499855671975) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.019299499855671975) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019299499855671975) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019299499855671975) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.019299499855671975) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019299499855671975) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.014603742409695385) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (-0.014603742409695385) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (-0.014603742409695385) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (-0.014603742409695385) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (-0.010757524199588267) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010757524199588267) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010757524199588267) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010757524199588267) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010715477342874483) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010715477342874483) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010715477342874483) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010715477342874483) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005923799555839632) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-0.005923799555839632) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-0.005923799555839632) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-0.005923799555839632) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-0.005408970757079541) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.005408970757079541) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.005408970757079541) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.005408970757079541) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.005286569055736535) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.005286569055736535) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.005286569055736535) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.005286569055736535) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.004767276643731287) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.004767276643731287) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.004767276643731287) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.004767276643731287) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.004668615265773195) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.004668615265773195) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.003876482195629466) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (-0.003876482195629466) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (-0.003484154579450169) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003484154579450169) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0033566679215093834) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.0033566679215093834) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.003267514896967024) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.003267514896967024) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0024464634226264927) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.0024464634226264927) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.0024464634226264927) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.0024464634226264927) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.0017278745819633567) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.0017278745819633567) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.0016407591158912969) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0016407591158912969) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0015324885614500755) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.0015324885614500755) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.0015324885614500755) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.0015324885614500755) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.0007870893706153517) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0007870893706153517) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007156737063136168) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0007156737063136168) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007156737063136168) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0007156737063136168) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0005192924120052474) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.0005192924120052474) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.0002464408136808765) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0002464408136808765) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0002464408136808765) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0002464408136808765) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.000194010306325131) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.000194010306325131) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.00018787485984257608) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.00018787485984257608) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.00018787485984257608) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.00018787485984257608) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.00013838603680294765) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.00013838603680294765) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-4.204685671377632e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-4.204685671377632e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-4.204685671377632e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.204685671377632e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-5.071403707897562e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.071403707897562e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.151295989683024e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.151295989683024e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0882457400985665e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-3.0882457400985665e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-2.9884125928293604e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.9884125928293604e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.874248583991133e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.874248583991133e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.3609472275035984e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.3609472275035984e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.300195899752235e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.300195899752235e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.1468226308471853e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.1468226308471853e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.1468226308471853e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.1468226308471853e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.398527747839529e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.398527747839529e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.398527747839529e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.398527747839529e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.091539912431428e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091539912431428e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091539912431428e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.091539912431428e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.091539912431428e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091539912431428e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091539912431428e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.091539912431428e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.027844263299983e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.027844263299983e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.027844263299983e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.027844263299983e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.900025802203638e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.900025802203638e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.900025802203638e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900025802203638e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867608672393008e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.867608672393008e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.560554002970149e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560554002970149e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560554002970149e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.560554002970149e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.560554002970149e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560554002970149e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560554002970149e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.560554002970149e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.246849474695668e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (-7.246849474695668e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (-7.246849474695668e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (-7.246849474695668e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (-7.246849474695668e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (-7.246849474695668e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (-7.246849474695668e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (-7.246849474695668e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (-3.5682005062616425e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.5682005062616425e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.5682005062616425e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.5682005062616425e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.376686396040459e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.376686396040459e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.376686396040459e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.376686396040459e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.376686396040459e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.376686396040459e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.376686396040459e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.376686396040459e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.8885647001489906e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.8885647001489906e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6863216669804945e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.6863216669804945e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.2498976416634043e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-2.2498976416634043e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-2.2498976416634043e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-2.2498976416634043e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-1.7035434569788487e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7035434569788487e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1782131041006175e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-1.1782131041006175e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-1.1782131041006175e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-1.1782131041006175e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-1.0716845375764072e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-1.0716845375764072e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-1.0716845375764072e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-1.0716845375764072e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-9.208945173452441e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.208945173452441e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379737448704982e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379737448704982e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379737448704982e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379737448704982e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379737448704982e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379737448704982e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.379737448704982e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379737448704982e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.706834845402913e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.706834845402913e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.706834845402913e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.706834845402913e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.204004404938442e-08) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (-3.204004404938442e-08) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (-3.204004404938442e-08) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (-3.204004404938442e-08) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (3.568947430367494e-08) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (3.568947430367494e-08) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (3.568947430367494e-08) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (3.568947430367494e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (3.568947430367494e-08) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.568947430367494e-08) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (3.568947430367494e-08) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (3.568947430367494e-08) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (9.208945173452441e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.208945173452441e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.7035434569788487e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7035434569788487e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863216669804945e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863216669804945e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.8885647001489906e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.8885647001489906e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.092161962627887e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.092161962627887e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.092161962627887e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.092161962627887e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.092161962627887e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.092161962627887e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.092161962627887e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.092161962627887e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.449056705663722e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.449056705663722e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.449056705663722e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.449056705663722e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.76945714616397e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.76945714616397e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.76945714616397e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.76945714616397e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (4.996951833027385e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (4.996951833027385e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (4.996951833027385e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (4.996951833027385e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (4.996951833027385e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (4.996951833027385e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (4.996951833027385e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (4.996951833027385e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (7.867608672393008e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.867608672393008e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.300195899752235e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.300195899752235e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.3609472275035984e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.3609472275035984e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.874248583991133e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.874248583991133e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.883653178234989e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.883653178234989e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.947331467316591e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.947331467316591e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.947331467316591e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.947331467316591e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9884125928293604e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9884125928293604e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.0882457400985665e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (3.0882457400985665e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (3.151295989683024e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.151295989683024e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.846190896890367e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.846190896890367e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.846190896890367e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.846190896890367e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.071403707897562e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.071403707897562e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.10546244784121e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.10546244784121e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.10546244784121e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.10546244784121e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.146386796646939e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.146386796646939e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.146386796646939e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.146386796646939e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.159294503114735e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.159294503114735e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.159294503114735e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.159294503114735e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.427926669816742e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.427926669816742e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.427926669816742e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.427926669816742e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.93574406014671e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.93574406014671e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.93574406014671e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.93574406014671e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253185198207525e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.253185198207525e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.979711031832343e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.979711031832343e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.979711031832343e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.979711031832343e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (7.141566430173471e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (7.141566430173471e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (7.141566430173471e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.141566430173471e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00013838603680294765) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.00013838603680294765) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.000194010306325131) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (0.000194010306325131) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (0.0005192924120052474) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.0005192924120052474) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.0007870893706153517) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.0007870893706153517) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.0014528842560487217) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0014528842560487217) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0014528842560487217) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0014528842560487217) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0016407591158912969) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0016407591158912969) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0017278745819633567) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.0017278745819633567) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.0021413489653396124) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (0.0021413489653396124) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (0.003267514896967024) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.003267514896967024) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.0033566679215093834) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.0033566679215093834) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.003484154579450169) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.003484154579450169) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.003804063154452629) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.003804063154452629) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.003804063154452629) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.003804063154452629) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.003876482195629466) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (0.003876482195629466) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (0.004668615265773195) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.004668615265773195) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.008469833338918565) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.008469833338918565) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.008469833338918565) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.008469833338918565) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.00854197565608371) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00854197565608371) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (0.00854197565608371) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (0.00854197565608371) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00854197565608371) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00854197565608371) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00854197565608371) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00854197565608371) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.008826387566621035) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.008826387566621035) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.008826387566621035) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.008826387566621035) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.009841802921695154) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.009841802921695154) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.009841802921695154) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.009841802921695154) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.01031147218194645) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01031147218194645) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01031147218194645) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.01031147218194645) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.01602466609174238) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (0.01602466609174238) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (0.01602466609174238) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (0.01602466609174238) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (0.01709162191951193) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.01709162191951193) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.01709162191951193) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.01709162191951193) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.019538085342138422) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.019538085342138422) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.019538085342138422) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.019538085342138422) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.022528354253362432) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.022528354253362432) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.023145221660809023) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (0.023145221660809023) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (0.024282031618294043) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (0.024282031618294043) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (0.024591832094857033) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.024591832094857033) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.024591832094857033) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.024591832094857033) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.03490330427680348) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.03490330427680348) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.03490330427680348) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.03490330427680348) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.03560840034928923) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.03560840034928923) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.06752398179237257) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.06752398179237257) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.06752398179237257) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.06752398179237257) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0763503693589936) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0763503693589936) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0763503693589936) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0763503693589936) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.08684736029994462) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.08684736029994462) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.08684736029994462) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.08684736029994462) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.09065142345439725) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.09065142345439725) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.09065142345439725) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.09065142345439725) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.07165056248463472) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07165056248463472) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07165056248463471) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07165056248463471) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.775872136010202e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.775872136010202e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.775872136010202e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.775872136010202e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0585921517848148) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0585921517848148) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.019257452998958195) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.019257452998958195) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.01031147218194645) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.01031147218194645) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008826387566621035) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.008826387566621035) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.004220835996963658) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.004220835996963658) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.004220835996963658) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.004220835996963658) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.003876482195629466) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (-0.003876482195629466) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (-0.003876482195629466) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (-0.003876482195629466) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (-0.003804063154452629) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.003804063154452629) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.002446463422626493) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.002446463422626493) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.0023949671541869107) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0023949671541869107) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0023949671541869107) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0023949671541869107) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0023949671541869107) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0023949671541869107) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0023949671541869107) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0023949671541869107) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0022009568478617795) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0022009568478617795) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0022009568478617795) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0022009568478617795) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012366559224875106) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.0012366559224875106) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.0012366559224875106) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.0012366559224875106) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.0011726297841432392) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0011726297841432392) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0011726297841432392) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0011726297841432392) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0007870893706153517) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0007870893706153517) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870893706153517) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0007870893706153517) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0005192924120052473) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.0005192924120052473) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.0005192924120052473) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.0005192924120052473) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-1.1462851075235078e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1462851075235078e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.874248583991133e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.874248583991133e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.874248583991133e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.874248583991133e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.300195899752235e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.300195899752235e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.300195899752235e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.300195899752235e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.04447417066924e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.04447417066924e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.04447417066924e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.04447417066924e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.955903647028836e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.955903647028836e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.955903647028836e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.955903647028836e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105341166159571e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105341166159571e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105341166159571e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105341166159571e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661200383002624e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661200383002624e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661200383002624e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661200383002624e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540204313439809e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540204313439809e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.189870518431925e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.189870518431925e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.876530430456672e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876530430456672e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876530430456672e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876530430456672e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175164937123327e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-6.175164937123327e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-4.5233394061654827e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.5233394061654827e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.076662720183807e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.076662720183807e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.076662720183807e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.076662720183807e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.013398869370751e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.013398869370751e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.90453739325259e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.90453739325259e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.90453739325259e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.90453739325259e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.6666797726817795e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.6666797726817795e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.6666797726817795e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6666797726817795e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505624808681647e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.8505624808681647e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.846699525452746e-08) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-7.846699525452746e-08) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-6.772951835322876e-08) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (-6.772951835322876e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (-4.0998294749687754e-08) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-4.0998294749687754e-08) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.0998294749687754e-08) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.0998294749687754e-08) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (6.772951835322876e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (6.772951835322876e-08) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (7.846699525452746e-08) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (7.846699525452746e-08) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (1.6570092837207694e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.6570092837207694e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.6570092837207694e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.6570092837207694e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.8505624808681647e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.8505624808681647e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863216669804945e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863216669804945e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.6863216669804945e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863216669804945e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013398869370751e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.013398869370751e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.5233394061654827e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.5233394061654827e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.670408153096349e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.670408153096349e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.670408153096349e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.670408153096349e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175164937123327e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (6.175164937123327e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (7.189870518431925e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.189870518431925e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540204313439809e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540204313439809e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949307788579683e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949307788579683e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.792463843079109e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.792463843079109e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.792463843079109e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.792463843079109e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.883653178234989e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.883653178234989e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.9884125928293604e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9884125928293604e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9884125928293604e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9884125928293604e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253185198207525e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.253185198207525e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4016916521664631e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4016916521664631e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4016916521664631e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4016916521664631e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809380364743605e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.5809380364743605e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809380364743605e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.5809380364743605e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0010283270637185416) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0010283270637185416) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0010283270637185416) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0010283270637185416) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012223373700436723) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.0012223373700436723) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (0.0012223373700436723) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (0.0012223373700436723) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0012223373700436723) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0012223373700436723) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0012223373700436723) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012223373700436723) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0016407591158912967) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0016407591158912967) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0016407591158912967) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0016407591158912967) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0018638931385461987) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0018638931385461987) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0018638931385461987) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0018638931385461987) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0018638931385461987) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0018638931385461987) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0018638931385461987) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0018638931385461987) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0021413489653396124) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (0.0021413489653396124) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (0.0022494140606299103) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0022494140606299103) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0022494140606299103) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0022494140606299103) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002446463422626493) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.002446463422626493) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.0029841800744761474) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.0029841800744761474) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.0029841800744761474) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.0029841800744761474) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.003804063154452629) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.003804063154452629) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.005348047717996368) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.005348047717996368) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.005348047717996368) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.005348047717996368) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.005733568640080078) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.005733568640080078) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.005733568640080078) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.005733568640080078) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.005733568640080078) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.005733568640080078) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.005733568640080078) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.005733568640080078) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.007597461778626277) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.007597461778626277) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.007597461778626277) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.007597461778626277) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.008826387566621035) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.008826387566621035) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.01031147218194645) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01031147218194645) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019257452998958195) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019257452998958195) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.3986653674269652e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.3986653674269652e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.3986653674269652e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.3986653674269652e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003484154579450169) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.003484154579450169) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.002984180074476147) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.002984180074476147) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.000194010306325131) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.000194010306325131) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1462851075235078e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1462851075235078e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.7924638430791093e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7924638430791093e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540204313439809e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540204313439809e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540204313439809e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540204313439809e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.8505624808681647e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505624808681647e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505624808681647e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505624808681647e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.846699525452746e-08) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-7.846699525452746e-08) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-7.846699525452746e-08) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-7.846699525452746e-08) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-4.0998294749687754e-08) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-4.0998294749687754e-08) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.0998294749687754e-08) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-4.0998294749687754e-08) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013398869370751e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013398869370751e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013398869370751e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013398869370751e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.949307788579683e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.949307788579683e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.7924638430791093e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7924638430791093e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.000194010306325131) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.000194010306325131) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002984180074476147) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.002984180074476147) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.003484154579450169) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.003484154579450169) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
List of core orbitals: [0, 1, 2]
List of active orbitals: [3, 4, 5, 6]
Number of qubits: 8
Number of qubits required to perform quantum simulations: 8
Hamiltonian of the water molecule
  (-73.13873149397193) [I0]
+ (-0.18066757496078878) [Z7]
+ (-0.18066757496078867) [Z6]
+ (-0.15961443573725723) [Z4]
+ (-0.15961443573725712) [Z5]
+ (0.17419986623485142) [Z3]
+ (0.17419986623485145) [Z2]
+ (0.2275732882496832) [Z1]
+ (0.22757328824968337) [Z0]
+ (-7.954224698583645e-06) [Y5 Y7]
+ (-7.954224698583645e-06) [X5 X7]
+ (8.194104990491557e-06) [Y4 Y6]
+ (8.194104990491557e-06) [X4 X6]
+ (0.11270381857120643) [Z4 Z6]
+ (0.11270381857120643) [Z5 Z7]
+ (0.11952441015517291) [Z0 Z4]
+ (0.11952441015517291) [Z1 Z5]
+ (0.1340173737184341) [Z0 Z6]
+ (0.1340173737184341) [Z1 Z7]
+ (0.1373494220866454) [Z0 Z5]
+ (0.1373494220866454) [Z1 Z4]
+ (0.13766859132458983) [Z2 Z4]
+ (0.13766859132458983) [Z3 Z5]
+ (0.14138903587619087) [Z4 Z7]
+ (0.14138903587619087) [Z5 Z6]
+ (0.1472293078199331) [Z2 Z5]
+ (0.1472293078199331) [Z3 Z4]
+ (0.14926347056935071) [Z4 Z5]
+ (0.14973497004647976) [Z2 Z6]
+ (0.14973497004647976) [Z3 Z7]
+ (0.15138342698010207) [Z0 Z7]
+ (0.15138342698010207) [Z1 Z6]
+ (0.15435760063002435) [Z6 Z7]
+ (0.15582280685045835) [Z2 Z7]
+ (0.15582280685045835) [Z3 Z6]
+ (0.16756669356724688) [Z0 Z2]
+ (0.16756669356724688) [Z1 Z3]
+ (0.1814400936355602) [Z0 Z3]
+ (0.1814400936355602) [Z1 Z2]
+ (0.19392574335573182) [Z0 Z1]
+ (0.2200397724116021) [Z2 Z3]
+ (7.038023769181928e-06) [Y4 Z5 Y6]
+ (7.038023769181928e-06) [X4 Z5 X6]
+ (7.038023769181928e-06) [Y5 Z6 Y7]
+ (7.038023769181928e-06) [X5 Z6 X7]
+ (-0.028685217304984443) [Y4 Y5 X6 X7]
+ (-0.028685217304984443) [X4 X5 Y6 Y7]
+ (-0.01782501193147254) [Y0 Y1 X4 X5]
+ (-0.01782501193147254) [X0 X1 Y4 Y5]
+ (-0.01736605326166795) [Y0 Y1 X6 X7]
+ (-0.01736605326166795) [X0 X1 Y6 Y7]
+ (-0.013873400068313318) [Y0 Y1 X2 X3]
+ (-0.013873400068313318) [X0 X1 Y2 Y3]
+ (-0.009560716495343263) [Y2 Y3 X4 X5]
+ (-0.009560716495343263) [X2 X3 Y4 Y5]
+ (-0.006087836803978588) [Y2 Y3 X6 X7]
+ (-0.006087836803978588) [X2 X3 Y6 Y7]
+ (-0.00029222567210206575) [Y1 Y2 X3 X4]
+ (-0.00029222567210206575) [X1 X2 Y3 Y4]
+ (-7.954224698583645e-06) [Y4 Z5 Y6 Z7]
+ (-7.954224698583645e-06) [X4 Z5 X6 Z7]
+ (-6.628427296067078e-07) [Y2 X3 X5 Y6]
+ (-6.628427296067078e-07) [Y2 Y3 Y5 Y6]
+ (-6.628427296067078e-07) [X2 X3 X5 X6]
+ (-6.628427296067078e-07) [X2 Y3 Y5 X6]
+ (9.344969884757248e-07) [Z2 Y5 Z6 Y7]
+ (9.344969884757248e-07) [Z2 X5 Z6 X7]
+ (9.344969884757248e-07) [Z3 Y4 Z5 Y6]
+ (9.344969884757248e-07) [Z3 X4 Z5 X6]
+ (1.035792482927296e-06) [Y0 X1 X5 Y6]
+ (1.035792482927296e-06) [Y0 Y1 Y5 Y6]
+ (1.035792482927296e-06) [X0 X1 X5 X6]
+ (1.035792482927296e-06) [X0 Y1 Y5 X6]
+ (1.5973397180824326e-06) [Z2 Y4 Z5 Y6]
+ (1.5973397180824326e-06) [Z2 X4 Z5 X6]
+ (1.5973397180824326e-06) [Z3 Y5 Z6 Y7]
+ (1.5973397180824326e-06) [Z3 X5 Z6 X7]
+ (1.8551374678199388e-06) [Z0 Y4 Z5 Y6]
+ (1.8551374678199388e-06) [Z0 X4 Z5 X6]
+ (1.8551374678199388e-06) [Z1 Y5 Z6 Y7]
+ (1.8551374678199388e-06) [Z1 X5 Z6 X7]
+ (2.8909299507450664e-06) [Z0 Y5 Z6 Y7]
+ (2.8909299507450664e-06) [Z0 X5 Z6 X7]
+ (2.8909299507450664e-06) [Z1 Y4 Z5 Y6]
+ (2.8909299507450664e-06) [Z1 X4 Z5 X6]
+ (8.194104990491557e-06) [Z4 Y5 Z6 Y7]
+ (8.194104990491557e-06) [Z4 X5 Z6 X7]
+ (0.00029222567210206575) [Y1 X2 X3 Y4]
+ (0.00029222567210206575) [X1 Y2 Y3 X4]
+ (0.006087836803978588) [Y2 X3 X6 Y7]
+ (0.006087836803978588) [X2 Y3 Y6 X7]
+ (0.009560716495343263) [Y2 X3 X4 Y5]
+ (0.009560716495343263) [X2 Y3 Y4 X5]
+ (0.011307208035836482) [Y1 Z2 Z3 Y5]
+ (0.011307208035836482) [X1 Z2 Z3 X5]
+ (0.013873400068313318) [Y0 X1 X2 Y3]
+ (0.013873400068313318) [X0 Y1 Y2 X3]
+ (0.01736605326166795) [Y0 X1 X6 Y7]
+ (0.01736605326166795) [X0 Y1 Y6 X7]
+ (0.01782501193147254) [Y0 X1 X4 Y5]
+ (0.01782501193147254) [X0 Y1 Y4 X5]
+ (0.028685217304984443) [Y4 X5 X6 Y7]
+ (0.028685217304984443) [X4 Y5 Y6 X7]
+ (0.02981229960789451) [Y0 Z1 Z2 Y4]
+ (0.02981229960789451) [X0 Z1 Z2 X4]
+ (0.02981229960789451) [Y1 Z3 Z4 Y5]
+ (0.02981229960789451) [X1 Z3 Z4 X5]
+ (0.030104525279996575) [Y0 Z1 Z3 Y4]
+ (0.030104525279996575) [X0 Z1 Z3 X4]
+ (0.030104525279996575) [Y1 Z2 Z4 Y5]
+ (0.030104525279996575) [X1 Z2 Z4 X5]
+ (0.03078744072684436) [Y0 Z2 Z3 Y4]
+ (0.03078744072684436) [X0 Z2 Z3 X4]
+ (0.04375171612735752) [Y1 Z2 Z3 Z4 Y5]
+ (0.04375171612735752) [X1 Z2 Z3 Z4 X5]
+ (0.043751716127357544) [Y0 Z1 Z2 Z3 Y4]
+ (0.043751716127357544) [X0 Z1 Z2 Z3 X4]
+ (-0.014564473636951946) [Y1 Z2 Z3 X4 X6 Y7]
+ (-0.014564473636951946) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-0.014564473636951946) [X1 Z2 Z3 X4 X6 X7]
+ (-0.014564473636951946) [X1 Z2 Z3 Y4 Y6 X7]
+ (-4.183808832672331e-06) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-4.183808832672331e-06) [X1 Z2 Z3 Z4 Z5 X7]
+ (-3.3130170911960938e-06) [Y1 Z2 Z3 X4 X5 Y6]
+ (-3.3130170911960938e-06) [X1 Z2 Z3 Y4 Y5 X6]
+ (-1.035792482927296e-06) [Y0 X1 X4 Z5 Z6 Y7]
+ (-1.035792482927296e-06) [X0 Y1 Y4 Z5 Z6 X7]
+ (-6.628427296067078e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-6.628427296067078e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-3.328039682594483e-07) [Y1 Y2 X3 Z4 Z5 X6]
+ (-3.328039682594483e-07) [X1 X2 Y3 Z4 Z5 Y6]
+ (3.328039682594483e-07) [Y1 X2 X3 Z4 Z5 Y6]
+ (3.328039682594483e-07) [X1 Y2 Y3 Z4 Z5 X6]
+ (6.628427296067078e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (6.628427296067078e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (1.035792482927296e-06) [Y0 Y1 X4 Z5 Z6 X7]
+ (1.035792482927296e-06) [X0 X1 Y4 Z5 Z6 Y7]
+ (3.2111874473305865e-06) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (3.2111874473305865e-06) [X0 Z1 Z2 Z3 Z5 X6]
+ (3.2111874473305865e-06) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (3.2111874473305865e-06) [X1 Z2 Z3 Z4 Z6 X7]
+ (3.277438292467738e-06) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (3.277438292467738e-06) [X0 Z1 Z2 Z4 Z5 X6]
+ (3.277438292467738e-06) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (3.277438292467738e-06) [X1 Z3 Z4 Z5 Z6 X7]
+ (3.3130170911960938e-06) [Y1 Z2 Z3 Y4 X5 X6]
+ (3.3130170911960938e-06) [X1 Z2 Z3 X4 Y5 Y6]
+ (3.6102422607271864e-06) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (3.6102422607271864e-06) [X0 Z1 Z3 Z4 Z5 X6]
+ (3.6102422607271864e-06) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (3.6102422607271864e-06) [X1 Z2 Z4 Z5 Z6 X7]
+ (3.769583620936447e-06) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (3.769583620936447e-06) [X0 Z2 Z3 Z4 Z5 X6]
+ (6.524204538527548e-06) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (6.524204538527548e-06) [X0 Z1 Z2 Z3 Z4 X6]
+ (6.524204538527548e-06) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (6.524204538527548e-06) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.0002922256721020657) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (0.0002922256721020657) [Y0 Z1 Y2 X3 Z4 X5]
+ (0.0002922256721020657) [X0 Z1 X2 Y3 Z4 Y5]
+ (0.0002922256721020657) [X0 Z1 X2 X3 Z4 X5]
+ (0.010540434335983721) [Y0 Z1 Z2 Z3 Y4 Z7]
+ (0.010540434335983721) [X0 Z1 Z2 Z3 X4 Z7]
+ (0.010540434335983721) [Y1 Z2 Z3 Z4 Y5 Z6]
+ (0.010540434335983721) [X1 Z2 Z3 Z4 X5 Z6]
+ (0.01130720803583648) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (0.01130720803583648) [X0 Z1 Z2 Z3 X4 Z5]
+ (0.025104907972935672) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (0.025104907972935672) [X0 Z1 Z2 Z3 X4 Z6]
+ (0.025104907972935672) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (0.025104907972935672) [X1 Z2 Z3 Z4 X5 Z7]
+ (0.03078744072684436) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (0.03078744072684436) [Z0 X1 Z2 Z3 Z4 X5]
+ (5.1056811148544716e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (5.1056811148544716e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (5.1056811148544716e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (5.1056811148544716e-06) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.014564473636951947) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-0.014564473636951947) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-4.183808832672331e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-4.183808832672331e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-3.3130170911960938e-06) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (-3.3130170911960938e-06) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (-3.3130170911960938e-06) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (-3.3130170911960938e-06) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (3.328039682594483e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (3.328039682594483e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (3.328039682594483e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (3.328039682594483e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (3.769583620936447e-06) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (3.769583620936447e-06) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (0.014564473636951947) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (0.014564473636951947) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pyscf/lib/misc.py:46: H5pyDeprecationWarning: Using default_file_mode other than 'r' is deprecated. Pass the mode to h5py.File() instead.
  h5py.get_config().default_file_mode = 'a'
  (-46.46390678868891) [I0]
+ (0.78296617259502) [Z10]
+ (0.7829661725950201) [Z11]
+ (0.8084581961720505) [Z13]
+ (0.8084581961720506) [Z12]
+ (1.203440228914563) [Z4]
+ (1.203440228914563) [Z5]
+ (1.3096862988615428) [Z7]
+ (1.309686298861543) [Z6]
+ (1.3693525634718169) [Z9]
+ (1.3693525634718173) [Z8]
+ (1.6538942226831694) [Z2]
+ (1.65389422268317) [Z3]
+ (12.412630742111762) [Z0]
+ (12.412630742111762) [Z1]
+ (-8.194261371595923e-06) [Y10 Y12]
+ (-8.194261371595923e-06) [X10 X12]
+ (-1.8540608579572954e-06) [Y5 Y7]
+ (-1.8540608579572954e-06) [X5 X7]
+ (-7.764994119620164e-07) [Y3 Y5]
+ (-7.764994119620164e-07) [X3 X5]
+ (-5.929765815528812e-07) [Y4 Y6]
+ (-5.929765815528812e-07) [X4 X6]
+ (1.6021167406578304e-06) [Y2 Y4]
+ (1.6021167406578304e-06) [X2 X4]
+ (7.95441317576663e-06) [Y11 Y13]
+ (7.95441317576663e-06) [X11 X13]
+ (0.0032769719312317307) [Y1 Y3]
+ (0.0032769719312317307) [X1 X3]
+ (0.10433064780651422) [Y0 Y2]
+ (0.10433064780651422) [X0 X2]
+ (0.11270386920332234) [Z10 Z12]
+ (0.11270386920332234) [Z11 Z13]
+ (0.11383573679388667) [Z4 Z12]
+ (0.11383573679388667) [Z5 Z13]
+ (0.11952438964682682) [Z6 Z10]
+ (0.11952438964682682) [Z7 Z11]
+ (0.124899909172376) [Z4 Z10]
+ (0.124899909172376) [Z5 Z11]
+ (0.12495807739503215) [Z2 Z4]
+ (0.12495807739503215) [Z3 Z5]
+ (0.1279950249246842) [Z2 Z10]
+ (0.1279950249246842) [Z3 Z11]
+ (0.13401715261963718) [Z6 Z12]
+ (0.13401715261963718) [Z7 Z13]
+ (0.13701191674040736) [Z4 Z6]
+ (0.13701191674040736) [Z5 Z7]
+ (0.13734953064261324) [Z6 Z11]
+ (0.13734953064261324) [Z7 Z10]
+ (0.13739104762683232) [Z2 Z6]
+ (0.13739104762683232) [Z3 Z7]
+ (0.13766872645852582) [Z8 Z10]
+ (0.13766872645852582) [Z9 Z11]
+ (0.14011289865354834) [Z2 Z12]
+ (0.14011289865354834) [Z3 Z13]
+ (0.14138905291942827) [Z10 Z13]
+ (0.14138905291942827) [Z11 Z12]
+ (0.14257997712485754) [Z4 Z11]
+ (0.14257997712485754) [Z5 Z10]
+ (0.14722943218766177) [Z8 Z11]
+ (0.14722943218766177) [Z9 Z10]
+ (0.1489943057506553) [Z4 Z7]
+ (0.1489943057506553) [Z5 Z6]
+ (0.14926355147388917) [Z10 Z11]
+ (0.14960702684445287) [Z4 Z8]
+ (0.14960702684445287) [Z5 Z9]
+ (0.14973486803496944) [Z8 Z12]
+ (0.14973486803496944) [Z9 Z13]
+ (0.15071408121008295) [Z2 Z8]
+ (0.15071408121008295) [Z3 Z9]
+ (0.15138327161428858) [Z6 Z13]
+ (0.15138327161428858) [Z7 Z12]
+ (0.15215040708869057) [Z4 Z13]
+ (0.15215040708869057) [Z5 Z12]
+ (0.1533796824331417) [Z2 Z11]
+ (0.1533796824331417) [Z3 Z10]
+ (0.1543574865722366) [Z12 Z13]
+ (0.15569010671752484) [Z2 Z13]
+ (0.15569010671752484) [Z3 Z12]
+ (0.15582269051553133) [Z8 Z13]
+ (0.15582269051553133) [Z9 Z12]
+ (0.1567639617643098) [Z4 Z9]
+ (0.1567639617643098) [Z5 Z8]
+ (0.1575531479798565) [Z4 Z5]
+ (0.16079764534838564) [Z2 Z5]
+ (0.16079764534838564) [Z3 Z4]
+ (0.1675665326546127) [Z6 Z8]
+ (0.1675665326546127) [Z7 Z9]
+ (0.16853486561579942) [Z2 Z7]
+ (0.16853486561579942) [Z3 Z6]
+ (0.1814399144030388) [Z6 Z9]
+ (0.1814399144030388) [Z7 Z8]
+ (0.1818908579075138) [Z2 Z3]
+ (0.18690820476912556) [Z2 Z9]
+ (0.18690820476912556) [Z3 Z8]
+ (0.19299723935364246) [Z0 Z10]
+ (0.19299723935364246) [Z1 Z11]
+ (0.1939253461327021) [Z6 Z7]
+ (0.19661770890342117) [Z0 Z4]
+ (0.19661770890342117) [Z1 Z5]
+ (0.19936354537360795) [Z0 Z5]
+ (0.19936354537360795) [Z1 Z4]
+ (0.20072866460441777) [Z0 Z11]
+ (0.20072866460441777) [Z1 Z10]
+ (0.21102659849791544) [Z0 Z12]
+ (0.21102659849791544) [Z1 Z13]
+ (0.2163103749863184) [Z0 Z13]
+ (0.2163103749863184) [Z1 Z12]
+ (0.2200397733437609) [Z8 Z9]
+ (0.23671080783830417) [Z0 Z2]
+ (0.23671080783830417) [Z1 Z3]
+ (0.24164663936017197) [Z0 Z6]
+ (0.24164663936017197) [Z1 Z7]
+ (0.24853483371314253) [Z0 Z7]
+ (0.24853483371314253) [Z1 Z6]
+ (0.2512944567459169) [Z0 Z3]
+ (0.2512944567459169) [Z1 Z2]
+ (0.2723251830660567) [Z0 Z8]
+ (0.2723251830660567) [Z1 Z9]
+ (0.27883454426723386) [Z0 Z9]
+ (0.27883454426723386) [Z1 Z8]
+ (1.1861763734860487) [Z0 Z1]
+ (-1.2260484988755717e-05) [Y4 Z5 Y6]
+ (-1.2260484988755717e-05) [X4 Z5 X6]
+ (-1.2260484988755717e-05) [Y5 Z6 Y7]
+ (-1.2260484988755717e-05) [X5 Z6 X7]
+ (-1.0722312156674751e-05) [Y11 Z12 Y13]
+ (-1.0722312156674751e-05) [X11 Z12 X13]
+ (-1.0722312156674748e-05) [Y10 Z11 Y12]
+ (-1.0722312156674748e-05) [X10 Z11 X12]
+ (-3.88705167387878e-06) [Y2 Z3 Y4]
+ (-3.88705167387878e-06) [X2 Z3 X4]
+ (-3.887051673878778e-06) [Y3 Z4 Y5]
+ (-3.887051673878778e-06) [X3 Z4 X5]
+ (0.12507032579772331) [Y0 Z1 Y2]
+ (0.12507032579772331) [X0 Z1 X2]
+ (0.1250703257977234) [Y1 Z2 Y3]
+ (0.1250703257977234) [X1 Z2 X3]
+ (-0.03831467029480389) [Y4 Y5 X12 X13]
+ (-0.03831467029480389) [X4 X5 Y12 Y13]
+ (-0.036194123559042606) [Y2 Y3 X8 X9]
+ (-0.036194123559042606) [X2 X3 Y8 Y9]
+ (-0.035839567953353475) [Y2 Y3 X4 X5]
+ (-0.035839567953353475) [X2 X3 Y4 Y5]
+ (-0.031143817988967096) [Y2 Y3 X6 X7]
+ (-0.031143817988967096) [X2 X3 Y6 Y7]
+ (-0.028685183716105903) [Y10 Y11 X12 X13]
+ (-0.028685183716105903) [X10 X11 Y12 Y13]
+ (-0.025996177598021253) [Y3 Z4 Z5 Y7]
+ (-0.025996177598021253) [X3 Z4 Z5 X7]
+ (-0.025384657508457482) [Y2 Y3 X10 X11]
+ (-0.025384657508457482) [X2 X3 Y10 Y11]
+ (-0.01902824244384734) [Y3 Y4 X11 X12]
+ (-0.01902824244384734) [X3 X4 Y11 Y12]
+ (-0.017825140995786404) [Y6 Y7 X10 X11]
+ (-0.017825140995786404) [X6 X7 Y10 Y11]
+ (-0.01768006795248155) [Y4 Y5 X10 X11]
+ (-0.01768006795248155) [X4 X5 Y10 Y11]
+ (-0.017366118994651406) [Y6 Y7 X12 X13]
+ (-0.017366118994651406) [X6 X7 Y12 Y13]
+ (-0.015577208063976484) [Y2 Y3 X12 X13]
+ (-0.015577208063976484) [X2 X3 Y12 Y13]
+ (-0.014583648907612698) [Y0 Y1 X2 X3]
+ (-0.014583648907612698) [X0 X1 Y2 Y3]
+ (-0.01387338174842612) [Y6 Y7 X8 X9]
+ (-0.01387338174842612) [X6 X7 Y8 Y9]
+ (-0.01198238901024794) [Y4 Y5 X6 X7]
+ (-0.01198238901024794) [X4 X5 Y6 Y7]
+ (-0.011285190200840893) [Y5 X6 X11 Y12]
+ (-0.011285190200840893) [X5 Y6 Y11 X12]
+ (-0.009560705729135964) [Y8 Y9 X10 X11]
+ (-0.009560705729135964) [X8 X9 Y10 Y11]
+ (-0.00812525192138102) [Y1 X2 X8 Y9]
+ (-0.00812525192138102) [Y1 Y2 Y8 Y9]
+ (-0.00812525192138102) [X1 X2 X8 X9]
+ (-0.00812525192138102) [X1 Y2 Y8 X9]
+ (-0.007731425250775303) [Y0 Y1 X10 X11]
+ (-0.007731425250775303) [X0 X1 Y10 Y11]
+ (-0.0071569349198569296) [Y4 Y5 X8 X9]
+ (-0.0071569349198569296) [X4 X5 Y8 Y9]
+ (-0.006888194352970576) [Y0 Y1 X6 X7]
+ (-0.006888194352970576) [X0 X1 Y6 Y7]
+ (-0.006509361201177232) [Y0 Y1 X8 X9]
+ (-0.006509361201177232) [X0 X1 Y8 Y9]
+ (-0.006087822480561874) [Y8 Y9 X12 X13]
+ (-0.006087822480561874) [X8 X9 Y12 Y13]
+ (-0.005283776488402967) [Y0 Y1 X12 X13]
+ (-0.005283776488402967) [X0 X1 Y12 Y13]
+ (-0.005143391768825129) [Y3 X4 X5 Y6]
+ (-0.005143391768825129) [X3 Y4 Y5 X6]
+ (-0.004684903388155197) [Y1 X2 X6 Y7]
+ (-0.004684903388155197) [Y1 Y2 Y6 Y7]
+ (-0.004684903388155197) [X1 X2 X6 X7]
+ (-0.004684903388155197) [X1 Y2 Y6 X7]
+ (-0.004575007626639211) [Y1 X2 X12 Y13]
+ (-0.004575007626639211) [Y1 Y2 Y12 Y13]
+ (-0.004575007626639211) [X1 X2 X12 X13]
+ (-0.004575007626639211) [X1 Y2 Y12 X13]
+ (-0.004424855449441841) [Y1 X2 X4 Y5]
+ (-0.004424855449441841) [Y1 Y2 Y4 Y5]
+ (-0.004424855449441841) [X1 X2 X4 X5]
+ (-0.004424855449441841) [X1 Y2 Y4 X5]
+ (-0.003479511890334331) [Y2 Z3 Z5 Y6]
+ (-0.003479511890334331) [X2 Z3 Z5 X6]
+ (-0.003479511890334331) [Y3 Z4 Z6 Y7]
+ (-0.003479511890334331) [X3 Z4 Z6 X7]
+ (-0.0027458364701868003) [Y0 Y1 X4 X5]
+ (-0.0027458364701868003) [X0 X1 Y4 Y5]
+ (-0.0017992194936630017) [Y1 X2 X10 Y11]
+ (-0.0017992194936630017) [Y1 Y2 Y10 Y11]
+ (-0.0017992194936630017) [X1 X2 X10 X11]
+ (-0.0017992194936630017) [X1 Y2 Y10 X11]
+ (-0.0002921986261110839) [Y7 Y8 X9 X10]
+ (-0.0002921986261110839) [X7 X8 Y9 Y10]
+ (-8.194261371595923e-06) [Z10 Y11 Z12 Y13]
+ (-8.194261371595923e-06) [Z10 X11 Z12 X13]
+ (-7.80170749993665e-06) [Y2 Z3 Y4 Z11]
+ (-7.80170749993665e-06) [X2 Z3 X4 Z11]
+ (-7.80170749993665e-06) [Y3 Z4 Y5 Z10]
+ (-7.80170749993665e-06) [X3 Z4 X5 Z10]
+ (-4.643051068146651e-06) [Y3 X4 X10 Y11]
+ (-4.643051068146651e-06) [Y3 Y4 Y10 Y11]
+ (-4.643051068146651e-06) [X3 X4 X10 X11]
+ (-4.643051068146651e-06) [X3 Y4 Y10 X11]
+ (-4.588855155360685e-06) [Y4 Z5 Y6 Z13]
+ (-4.588855155360685e-06) [X4 Z5 X6 Z13]
+ (-4.588855155360685e-06) [Y5 Z6 Y7 Z12]
+ (-4.588855155360685e-06) [X5 Z6 X7 Z12]
+ (-4.55656921776986e-06) [Y5 X6 X12 Y13]
+ (-4.55656921776986e-06) [Y5 Y6 Y12 Y13]
+ (-4.55656921776986e-06) [X5 X6 X12 X13]
+ (-4.55656921776986e-06) [X5 Y6 Y12 X13]
+ (-3.6945132942591346e-06) [Y4 X5 X11 Y12]
+ (-3.6945132942591346e-06) [Y4 Y5 Y11 Y12]
+ (-3.6945132942591346e-06) [X4 X5 X11 X12]
+ (-3.6945132942591346e-06) [X4 Y5 Y11 X12]
+ (-3.344081556458242e-06) [Z0 Y5 Z6 Y7]
+ (-3.344081556458242e-06) [Z0 X5 Z6 X7]
+ (-3.344081556458242e-06) [Z1 Y4 Z5 Y6]
+ (-3.344081556458242e-06) [Z1 X4 Z5 X6]
+ (-3.1586564317899996e-06) [Y2 Z3 Y4 Z10]
+ (-3.1586564317899996e-06) [X2 Z3 X4 Z10]
+ (-3.1586564317899996e-06) [Y3 Z4 Y5 Z11]
+ (-3.1586564317899996e-06) [X3 Z4 X5 Z11]
+ (-3.099349243575037e-06) [Z0 Y4 Z5 Y6]
+ (-3.099349243575037e-06) [Z0 X4 Z5 X6]
+ (-3.099349243575037e-06) [Z1 Y5 Z6 Y7]
+ (-3.099349243575037e-06) [Z1 X5 Z6 X7]
+ (-2.890967881412624e-06) [Z6 Y11 Z12 Y13]
+ (-2.890967881412624e-06) [Z6 X11 Z12 X13]
+ (-2.890967881412624e-06) [Z7 Y10 Z11 Y12]
+ (-2.890967881412624e-06) [Z7 X10 Z11 X12]
+ (-2.177664604637626e-06) [Z0 Y10 Z11 Y12]
+ (-2.177664604637626e-06) [Z0 X10 Z11 X12]
+ (-2.177664604637626e-06) [Z1 Y11 Z12 Y13]
+ (-2.177664604637626e-06) [Z1 X11 Z12 X13]
+ (-1.881850183190604e-06) [Y4 Z5 Y6 Z9]
+ (-1.881850183190604e-06) [X4 Z5 X6 Z9]
+ (-1.881850183190604e-06) [Y5 Z6 Y7 Z8]
+ (-1.881850183190604e-06) [X5 Z6 X7 Z8]
+ (-1.8551201212773364e-06) [Z6 Y10 Z11 Y12]
+ (-1.8551201212773364e-06) [Z6 X10 Z11 X12]
+ (-1.8551201212773364e-06) [Z7 Y11 Z12 Y13]
+ (-1.8551201212773364e-06) [Z7 X11 Z12 X13]
+ (-1.8540608579572954e-06) [Y4 Z5 Y6 Z7]
+ (-1.8540608579572954e-06) [X4 Z5 X6 Z7]
+ (-1.8163031695938426e-06) [Z4 Y11 Z12 Y13]
+ (-1.8163031695938426e-06) [Z4 X11 Z12 X13]
+ (-1.8163031695938426e-06) [Z5 Y10 Z11 Y12]
+ (-1.8163031695938426e-06) [Z5 X10 Z11 X12]
+ (-1.6923978284660816e-06) [Y4 Z5 Y6 Z10]
+ (-1.6923978284660816e-06) [X4 Z5 X6 Z10]
+ (-1.6923978284660816e-06) [Y5 Z6 Y7 Z11]
+ (-1.6923978284660816e-06) [X5 Z6 X7 Z11]
+ (-1.6148794135137072e-06) [Z0 Y11 Z12 Y13]
+ (-1.6148794135137072e-06) [Z0 X11 Z12 X13]
+ (-1.6148794135137072e-06) [Z1 Y10 Z11 Y12]
+ (-1.6148794135137072e-06) [Z1 X10 Z11 X12]
+ (-1.5973171975866097e-06) [Z8 Y10 Z11 Y12]
+ (-1.5973171975866097e-06) [Z8 X10 Z11 X12]
+ (-1.5973171975866097e-06) [Z9 Y11 Z12 Y13]
+ (-1.5973171975866097e-06) [Z9 X11 Z12 X13]
+ (-1.4548424490573079e-06) [Y3 X4 X6 Y7]
+ (-1.4548424490573079e-06) [Y3 Y4 Y6 Y7]
+ (-1.4548424490573079e-06) [X3 X4 X6 X7]
+ (-1.4548424490573079e-06) [X3 Y4 Y6 X7]
+ (-1.398044908091774e-06) [Y4 Z5 Y6 Z8]
+ (-1.398044908091774e-06) [X4 Z5 X6 Z8]
+ (-1.398044908091774e-06) [Y5 Z6 Y7 Z9]
+ (-1.398044908091774e-06) [X5 Z6 X7 Z9]
+ (-1.1954890099718503e-06) [Y2 Z3 Y4 Z7]
+ (-1.1954890099718503e-06) [X2 Z3 X4 Z7]
+ (-1.1954890099718503e-06) [Y3 Z4 Y5 Z6]
+ (-1.1954890099718503e-06) [X3 Z4 X5 Z6]
+ (-1.1908508084640527e-06) [Z0 Y3 Z4 Y5]
+ (-1.1908508084640527e-06) [Z0 X3 Z4 X5]
+ (-1.1908508084640527e-06) [Z1 Y2 Z3 Y4]
+ (-1.1908508084640527e-06) [Z1 X2 Z3 X4]
+ (-1.1708301370226586e-06) [Z2 Y5 Z6 Y7]
+ (-1.1708301370226586e-06) [Z2 X5 Z6 X7]
+ (-1.1708301370226586e-06) [Z3 Y4 Z5 Y6]
+ (-1.1708301370226586e-06) [Z3 X4 Z5 X6]
+ (-1.0632283420830098e-06) [Z2 Y10 Z11 Y12]
+ (-1.0632283420830098e-06) [Z2 X10 Z11 X12]
+ (-1.0632283420830098e-06) [Z3 Y11 Z12 Y13]
+ (-1.0632283420830098e-06) [Z3 X11 Z12 X13]
+ (-1.0358477601352874e-06) [Y6 X7 X11 Y12]
+ (-1.0358477601352874e-06) [Y6 Y7 Y11 Y12]
+ (-1.0358477601352874e-06) [X6 X7 X11 X12]
+ (-1.0358477601352874e-06) [X6 Y7 Y11 X12]
+ (-9.50924975139167e-07) [Z2 Y4 Z5 Y6]
+ (-9.50924975139167e-07) [Z2 X4 Z5 X6]
+ (-9.50924975139167e-07) [Z3 Y5 Z6 Y7]
+ (-9.50924975139167e-07) [Z3 X5 Z6 X7]
+ (-9.344557774503286e-07) [Z8 Y11 Z12 Y13]
+ (-9.344557774503286e-07) [Z8 X11 Z12 X13]
+ (-9.344557774503286e-07) [Z9 Y10 Z11 Y12]
+ (-9.344557774503286e-07) [Z9 X10 Z11 X12]
+ (-8.337746755398009e-07) [Z0 Y2 Z3 Y4]
+ (-8.337746755398009e-07) [Z0 X2 Z3 X4]
+ (-8.337746755398009e-07) [Z1 Y3 Z4 Y5]
+ (-8.337746755398009e-07) [Z1 X3 Z4 X5]
+ (-7.956895372900573e-07) [Y3 X4 X8 Y9]
+ (-7.956895372900573e-07) [Y3 Y4 Y8 Y9]
+ (-7.956895372900573e-07) [X3 X4 X8 X9]
+ (-7.956895372900573e-07) [X3 Y4 Y8 X9]
+ (-7.764994119620165e-07) [Y2 Z3 Y4 Z5]
+ (-7.764994119620165e-07) [X2 Z3 X4 Z5]
+ (-5.929765815528812e-07) [Z4 Y5 Z6 Y7]
+ (-5.929765815528812e-07) [Z4 X5 Z6 X7]
+ (-5.77005299559363e-07) [Y2 Z3 Y4 Z9]
+ (-5.77005299559363e-07) [X2 Z3 X4 Z9]
+ (-5.77005299559363e-07) [Y3 Z4 Y5 Z8]
+ (-5.77005299559363e-07) [X3 Z4 X5 Z8]
+ (-5.47164774426425e-07) [Y1 Y2 X11 X12]
+ (-5.47164774426425e-07) [X1 X2 Y11 Y12]
+ (-4.838052750988301e-07) [Y5 X6 X8 Y9]
+ (-4.838052750988301e-07) [Y5 Y6 Y8 Y9]
+ (-4.838052750988301e-07) [X5 X6 X8 X9]
+ (-4.838052750988301e-07) [X5 Y6 Y8 X9]
+ (-3.5707613292425203e-07) [Y0 X1 X3 Y4]
+ (-3.5707613292425203e-07) [Y0 Y1 Y3 Y4]
+ (-3.5707613292425203e-07) [X0 X1 X3 X4]
+ (-3.5707613292425203e-07) [X0 Y1 Y3 X4]
+ (-2.447323128832056e-07) [Y0 X1 X5 Y6]
+ (-2.447323128832056e-07) [Y0 Y1 Y5 Y6]
+ (-2.447323128832056e-07) [X0 X1 X5 X6]
+ (-2.447323128832056e-07) [X0 Y1 Y5 X6]
+ (-2.199051618834916e-07) [Y2 X3 X5 Y6]
+ (-2.199051618834916e-07) [Y2 Y3 Y5 Y6]
+ (-2.199051618834916e-07) [X2 X3 X5 X6]
+ (-2.199051618834916e-07) [X2 Y3 Y5 X6]
+ (-1.9332412771118596e-07) [Y1 X2 X3 Y4]
+ (-1.9332412771118596e-07) [X1 Y2 Y3 X4]
+ (-1.2919694865350102e-07) [Y1 Z2 Z3 Y5]
+ (-1.2919694865350102e-07) [X1 Z2 Z3 X5]
+ (1.7379332622789548e-07) [Y0 Z1 Z3 Y4]
+ (1.7379332622789548e-07) [X0 Z1 Z3 X4]
+ (1.7379332622789548e-07) [Y1 Z2 Z4 Y5]
+ (1.7379332622789548e-07) [X1 Z2 Z4 X5]
+ (1.9332412771118596e-07) [Y1 Y2 X3 X4]
+ (1.9332412771118596e-07) [X1 X2 Y3 Y4]
+ (2.1868423773069423e-07) [Y2 Z3 Y4 Z8]
+ (2.1868423773069423e-07) [X2 Z3 X4 Z8]
+ (2.1868423773069423e-07) [Y3 Z4 Y5 Z9]
+ (2.1868423773069423e-07) [X3 Z4 X5 Z9]
+ (2.5935343908545746e-07) [Y2 Z3 Y4 Z6]
+ (2.5935343908545746e-07) [X2 Z3 X4 Z6]
+ (2.5935343908545746e-07) [Y3 Z4 Y5 Z7]
+ (2.5935343908545746e-07) [X3 Z4 X5 Z7]
+ (3.6060718677803603e-07) [Y0 Z1 Z2 Y4]
+ (3.6060718677803603e-07) [X0 Z1 Z2 X4]
+ (3.6060718677803603e-07) [Y1 Z3 Z4 Y5]
+ (3.6060718677803603e-07) [X1 Z3 Z4 X5]
+ (5.47164774426425e-07) [Y1 X2 X11 Y12]
+ (5.47164774426425e-07) [X1 Y2 Y11 X12]
+ (5.627851911239184e-07) [Y0 X1 X11 Y12]
+ (5.627851911239184e-07) [Y0 Y1 Y11 Y12]
+ (5.627851911239184e-07) [X0 X1 X11 X12]
+ (5.627851911239184e-07) [X0 Y1 Y11 X12]
+ (6.628614201362812e-07) [Y8 X9 X11 Y12]
+ (6.628614201362812e-07) [Y8 Y9 Y11 Y12]
+ (6.628614201362812e-07) [X8 X9 X11 X12]
+ (6.628614201362812e-07) [X8 Y9 Y11 X12]
+ (1.1094407593519738e-06) [Z2 Y11 Z12 Y13]
+ (1.1094407593519738e-06) [Z2 X11 Z12 X13]
+ (1.1094407593519738e-06) [Z3 Y10 Z11 Y12]
+ (1.1094407593519738e-06) [Z3 X10 Z11 X12]
+ (1.6021167406578306e-06) [Z2 Y3 Z4 Y5]
+ (1.6021167406578306e-06) [Z2 X3 Z4 X5]
+ (1.8782101246652918e-06) [Z4 Y10 Z11 Y12]
+ (1.8782101246652918e-06) [Z4 X10 Z11 X12]
+ (1.8782101246652918e-06) [Z5 Y11 Z12 Y13]
+ (1.8782101246652918e-06) [Z5 X11 Z12 X13]
+ (2.172669101434983e-06) [Y2 X3 X11 Y12]
+ (2.172669101434983e-06) [Y2 Y3 Y11 Y12]
+ (2.172669101434983e-06) [X2 X3 X11 X12]
+ (2.172669101434983e-06) [X2 Y3 Y11 X12]
+ (3.117447946137541e-06) [Y0 Z2 Z3 Y4]
+ (3.117447946137541e-06) [X0 Z2 Z3 X4]
+ (3.5390541842202535e-06) [Y2 Z3 Y4 Z12]
+ (3.5390541842202535e-06) [X2 Z3 X4 Z12]
+ (3.5390541842202535e-06) [Y3 Z4 Y5 Z13]
+ (3.5390541842202535e-06) [X3 Z4 X5 Z13]
+ (4.281913884591407e-06) [Y4 Z5 Y6 Z11]
+ (4.281913884591407e-06) [X4 Z5 X6 Z11]
+ (4.281913884591407e-06) [Y5 Z6 Y7 Z10]
+ (4.281913884591407e-06) [X5 Z6 X7 Z10]
+ (5.2758831218345475e-06) [Y3 X4 X12 Y13]
+ (5.2758831218345475e-06) [Y3 Y4 Y12 Y13]
+ (5.2758831218345475e-06) [X3 X4 X12 X13]
+ (5.2758831218345475e-06) [X3 Y4 Y12 X13]
+ (5.974311713057489e-06) [Y5 X6 X10 Y11]
+ (5.974311713057489e-06) [Y5 Y6 Y10 Y11]
+ (5.974311713057489e-06) [X5 X6 X10 X11]
+ (5.974311713057489e-06) [X5 Y6 Y10 X11]
+ (7.95441317576663e-06) [Y10 Z11 Y12 Z13]
+ (7.95441317576663e-06) [X10 Z11 X12 Z13]
+ (8.8149373060548e-06) [Y2 Z3 Y4 Z13]
+ (8.8149373060548e-06) [X2 Z3 X4 Z13]
+ (8.8149373060548e-06) [Y3 Z4 Y5 Z12]
+ (8.8149373060548e-06) [X3 Z4 X5 Z12]
+ (0.0002921986261110839) [Y7 X8 X9 Y10]
+ (0.0002921986261110839) [X7 Y8 Y9 X10]
+ (0.0004956762314916488) [Y2 Z4 Z5 Y6]
+ (0.0004956762314916488) [X2 Z4 Z5 X6]
+ (0.0011059037691897291) [Y0 Z1 Y2 Z5]
+ (0.0011059037691897291) [X0 Z1 X2 Z5]
+ (0.0011059037691897291) [Y1 Z2 Y3 Z4]
+ (0.0011059037691897291) [X1 Z2 X3 Z4]
+ (0.001663879878490798) [Y2 Z3 Z4 Y6]
+ (0.001663879878490798) [X2 Z3 Z4 X6]
+ (0.001663879878490798) [Y3 Z5 Z6 Y7]
+ (0.001663879878490798) [X3 Z5 Z6 X7]
+ (0.001756070701841304) [Y0 Z1 Y2 Z11]
+ (0.001756070701841304) [X0 Z1 X2 Z11]
+ (0.001756070701841304) [Y1 Z2 Y3 Z10]
+ (0.001756070701841304) [X1 Z2 X3 Z10]
+ (0.002326230623158148) [Y0 Z1 Y2 Z13]
+ (0.002326230623158148) [X0 Z1 X2 Z13]
+ (0.002326230623158148) [Y1 Z2 Y3 Z12]
+ (0.002326230623158148) [X1 Z2 X3 Z12]
+ (0.0027458364701868003) [Y0 X1 X4 Y5]
+ (0.0027458364701868003) [X0 Y1 Y4 X5]
+ (0.0029297686747511284) [Y0 Z1 Y2 Z9]
+ (0.0029297686747511284) [X0 Z1 X2 Z9]
+ (0.0029297686747511284) [Y1 Z2 Y3 Z8]
+ (0.0029297686747511284) [X1 Z2 X3 Z8]
+ (0.0032769719312317307) [Y0 Z1 Y2 Z3]
+ (0.0032769719312317307) [X0 Z1 X2 Z3]
+ (0.0033476175306662572) [Y0 Z1 Y2 Z7]
+ (0.0033476175306662572) [X0 Z1 X2 Z7]
+ (0.0033476175306662572) [Y1 Z2 Y3 Z6]
+ (0.0033476175306662572) [X1 Z2 X3 Z6]
+ (0.003555290195504306) [Y0 Z1 Y2 Z10]
+ (0.003555290195504306) [X0 Z1 X2 Z10]
+ (0.003555290195504306) [Y1 Z2 Y3 Z11]
+ (0.003555290195504306) [X1 Z2 X3 Z11]
+ (0.005143391768825129) [Y3 Y4 X5 X6]
+ (0.005143391768825129) [X3 X4 Y5 Y6]
+ (0.005283776488402967) [Y0 X1 X12 Y13]
+ (0.005283776488402967) [X0 Y1 Y12 X13]
+ (0.005530759218631571) [Y0 Z1 Y2 Z4]
+ (0.005530759218631571) [X0 Z1 X2 Z4]
+ (0.005530759218631571) [Y1 Z2 Y3 Z5]
+ (0.005530759218631571) [X1 Z2 X3 Z5]
+ (0.006087822480561874) [Y8 X9 X12 Y13]
+ (0.006087822480561874) [X8 Y9 Y12 X13]
+ (0.006509361201177232) [Y0 X1 X8 Y9]
+ (0.006509361201177232) [X0 Y1 Y8 X9]
+ (0.006888194352970576) [Y0 X1 X6 Y7]
+ (0.006888194352970576) [X0 Y1 Y6 X7]
+ (0.006901238249797359) [Y0 Z1 Y2 Z12]
+ (0.006901238249797359) [X0 Z1 X2 Z12]
+ (0.006901238249797359) [Y1 Z2 Y3 Z13]
+ (0.006901238249797359) [X1 Z2 X3 Z13]
+ (0.0071569349198569296) [Y4 X5 X8 Y9]
+ (0.0071569349198569296) [X4 Y5 Y8 X9]
+ (0.007731425250775303) [Y0 X1 X10 Y11]
+ (0.007731425250775303) [X0 Y1 Y10 X11]
+ (0.008032520918821454) [Y0 Z1 Y2 Z6]
+ (0.008032520918821454) [X0 Z1 X2 Z6]
+ (0.008032520918821454) [Y1 Z2 Y3 Z7]
+ (0.008032520918821454) [X1 Z2 X3 Z7]
+ (0.009560705729135964) [Y8 X9 X10 Y11]
+ (0.009560705729135964) [X8 Y9 Y10 X11]
+ (0.01105502059613215) [Y0 Z1 Y2 Z8]
+ (0.01105502059613215) [X0 Z1 X2 Z8]
+ (0.01105502059613215) [Y1 Z2 Y3 Z9]
+ (0.01105502059613215) [X1 Z2 X3 Z9]
+ (0.011285190200840893) [Y5 Y6 X11 X12]
+ (0.011285190200840893) [X5 X6 Y11 Y12]
+ (0.011307274008848154) [Y7 Z8 Z9 Y11]
+ (0.011307274008848154) [X7 Z8 Z9 X11]
+ (0.01198238901024794) [Y4 X5 X6 Y7]
+ (0.01198238901024794) [X4 Y5 Y6 X7]
+ (0.01387338174842612) [Y6 X7 X8 Y9]
+ (0.01387338174842612) [X6 Y7 Y8 X9]
+ (0.014583648907612698) [Y0 X1 X2 Y3]
+ (0.014583648907612698) [X0 Y1 Y2 X3]
+ (0.015577208063976484) [Y2 X3 X12 Y13]
+ (0.015577208063976484) [X2 Y3 Y12 X13]
+ (0.017366118994651406) [Y6 X7 X12 Y13]
+ (0.017366118994651406) [X6 Y7 Y12 X13]
+ (0.01768006795248155) [Y4 X5 X10 Y11]
+ (0.01768006795248155) [X4 Y5 Y10 X11]
+ (0.017825140995786404) [Y6 X7 X10 Y11]
+ (0.017825140995786404) [X6 Y7 Y10 X11]
+ (0.01902824244384734) [Y3 X4 X11 Y12]
+ (0.01902824244384734) [X3 Y4 Y11 X12]
+ (0.025384657508457482) [Y2 X3 X10 Y11]
+ (0.025384657508457482) [X2 Y3 Y10 X11]
+ (0.028685183716105903) [Y10 X11 X12 Y13]
+ (0.028685183716105903) [X10 Y11 Y12 X13]
+ (0.029812424517345677) [Y6 Z7 Z8 Y10]
+ (0.029812424517345677) [X6 Z7 Z8 X10]
+ (0.029812424517345677) [Y7 Z9 Z10 Y11]
+ (0.029812424517345677) [X7 Z9 Z10 X11]
+ (0.030104623143456757) [Y6 Z7 Z9 Y10]
+ (0.030104623143456757) [X6 Z7 Z9 X10]
+ (0.030104623143456757) [Y7 Z8 Z10 Y11]
+ (0.030104623143456757) [X7 Z8 Z10 X11]
+ (0.030787505389143863) [Y6 Z8 Z9 Y10]
+ (0.030787505389143863) [X6 Z8 Z9 X10]
+ (0.031143817988967096) [Y2 X3 X6 Y7]
+ (0.031143817988967096) [X2 Y3 Y6 X7]
+ (0.035839567953353475) [Y2 X3 X4 Y5]
+ (0.035839567953353475) [X2 Y3 Y4 X5]
+ (0.036194123559042606) [Y2 X3 X8 Y9]
+ (0.036194123559042606) [X2 Y3 Y8 X9]
+ (0.03831467029480389) [Y4 X5 X12 Y13]
+ (0.03831467029480389) [X4 Y5 Y12 X13]
+ (0.10433064780651422) [Z0 Y1 Z2 Y3]
+ (0.10433064780651422) [Z0 X1 Z2 X3]
+ (-0.12133276911042408) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133276911042408) [X2 Z3 Z4 Z5 X6]
+ (-0.12133276911042408) [Y3 Z4 Z5 Z6 Y7]
+ (-0.12133276911042408) [X3 Z4 Z5 Z6 X7]
+ (3.2020768797081216e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.2020768797081216e-06) [X0 Z1 Z2 Z3 X4]
+ (3.2020768797081232e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.2020768797081232e-06) [X1 Z2 Z3 Z4 X5]
+ (0.22848106564918716) [Y7 Z8 Z9 Z10 Y11]
+ (0.22848106564918716) [X7 Z8 Z9 Z10 X11]
+ (0.2284810656491872) [Y6 Z7 Z8 Z9 Y10]
+ (0.2284810656491872) [X6 Z7 Z8 Z9 X10]
+ (-0.03276765782329055) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.03276765782329055) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.03276765782329055) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.03276765782329055) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845273183) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.027115036845273183) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845273183) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.027115036845273183) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.025996177598021253) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.025996177598021253) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.0175612024096462) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.0175612024096462) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.0175612024096462) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.0175612024096462) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.014564531231172982) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.014564531231172982) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.014564531231172982) [X7 Z8 Z9 X10 X12 X13]
+ (-0.014564531231172982) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.012215040997613925) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.012215040997613925) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.012215040997613925) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.012215040997613925) [X4 Z5 X6 X11 Z12 X13]
+ (-0.012215040997613925) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.012215040997613925) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.012215040997613925) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.012215040997613925) [X5 Z6 X7 X10 Z11 X12]
+ (-0.011756013419819272) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.011756013419819272) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.011756013419819272) [X3 Z4 Z5 X6 X8 X9]
+ (-0.011756013419819272) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.008764827575688798) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.008764827575688798) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.008764827575688798) [X2 Z3 Z4 X5 X11 X12]
+ (-0.008764827575688798) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.008764827575688798) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.008764827575688798) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.008764827575688798) [X3 X4 X10 Z11 Z12 X13]
+ (-0.008764827575688798) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.00812525192138102) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.00812525192138102) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.007306759928832983) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.007306759928832983) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.007306759928832983) [X4 X5 X7 Z8 Z9 X10]
+ (-0.007306759928832983) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.005805188989826926) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.005805188989826926) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.005805188989826926) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.005805188989826926) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.005652620978017369) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.005652620978017369) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.005652620978017369) [X0 X1 X3 Z4 Z5 X6]
+ (-0.005652620978017369) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.005143391768825129) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.005143391768825129) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.005143391768825129) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.005143391768825129) [X2 Z3 X4 X5 Z6 X7]
+ (-0.004684903388155197) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.004684903388155197) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668620318776302) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.004668620318776302) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.004575007626639211) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.004575007626639211) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.004424855449441841) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.004424855449441841) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.004158797381840073) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.004158797381840073) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.004158797381840073) [X3 Z4 Z5 X6 X12 X13]
+ (-0.004158797381840073) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.0034937903598902115) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.0034937903598902115) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.0034937903598902115) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.0034937903598902115) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.002779026799025555) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.002779026799025555) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.0022939566113524762) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.0022939566113524762) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.0017992194936630014) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.0017992194936630014) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.0017278753941369677) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.0017278753941369677) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.0009298507967730285) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298507967730285) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298507967730285) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298507967730285) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533856254125472) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.0008533856254125472) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.0008145313270956889) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.0008145313270956889) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.0008145313270956889) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.0008145313270956889) [X3 Z4 Z5 Z6 X7 Z11]
+ (-7.735036880592475e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.735036880592475e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.735036880592475e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.735036880592475e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-8.774817864051353e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.774817864051353e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.774817864051353e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.774817864051353e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.518362215235501e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.518362215235501e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.518362215235501e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.518362215235501e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.444344675437563e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.444344675437563e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.444344675437563e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.444344675437563e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.524373848100062e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.524373848100062e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.524373848100062e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.524373848100062e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-6.290028432753968e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.290028432753968e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.290028432753968e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.290028432753968e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.974311713057489e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (-5.974311713057489e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (-5.2758831218345475e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.2758831218345475e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.64305106814665e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.64305106814665e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.55656921776986e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (-4.55656921776986e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (-4.253224225302965e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.253224225302965e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-3.7696594515937515e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.7696594515937515e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.6945132942591346e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.6945132942591346e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.6102971302203094e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.6102971302203094e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.6102971302203094e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.6102971302203094e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.3131454999910333e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.3131454999910333e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.277483195161045e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.277483195161045e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.277483195161045e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.277483195161045e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.2112283481090286e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.2112283481090286e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.2112283481090286e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.2112283481090286e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.151346310893532e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.151346310893532e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.088250711090746e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.088250711090746e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.172669101434983e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.172669101434983e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.4548424490573079e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.4548424490573079e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.3304731886137888e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.3304731886137888e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.2283337824815328e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.2283337824815328e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.0358477601352874e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.0358477601352874e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-7.956895372900573e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.956895372900573e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.733197741843018e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.733197741843018e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.733197741843018e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.733197741843018e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.628614201362812e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.628614201362812e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.556281914112998e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.556281914112998e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.556281914112998e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.556281914112998e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.418291574217039e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.418291574217039e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.418291574217039e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.418291574217039e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.927453082391738e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.927453082391738e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.927453082391738e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.927453082391738e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.627851911239184e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.627851911239184e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.287660624402284e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.287660624402284e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.287660624402284e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.287660624402284e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.287660624402284e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.287660624402284e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.287660624402284e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.287660624402284e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.838052750988301e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (-4.838052750988301e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (-3.5707613292425203e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.5707613292425203e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.328139350592643e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.328139350592643e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.086826565198475e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.086826565198475e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.086826565198475e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.086826565198475e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.447323128832056e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (-2.447323128832056e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (-2.3713289480110134e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.3713289480110134e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.3713289480110134e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.3713289480110134e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.199051618834916e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-2.199051618834916e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-1.9332412771118596e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.9332412771118596e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.9332412771118596e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.9332412771118596e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.8394209155175266e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.8394209155175266e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.8394209155175266e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.8394209155175266e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.5510539176126766e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (-1.5510539176126766e-07) [X0 Z1 X2 X4 Z5 X6]
+ (-1.5510539176126766e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (-1.5510539176126766e-07) [X1 Z2 X3 X5 Z6 X7]
+ (-1.3807781480675324e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (-1.3807781480675324e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (-1.3807781480675324e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (-1.3807781480675324e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (-1.3807781480675324e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (-1.3807781480675324e-07) [X0 Z1 X2 X5 Z6 X7]
+ (-1.3807781480675324e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (-1.3807781480675324e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (-1.3807781480675324e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (-1.3807781480675324e-07) [X1 Z2 X3 X4 Z5 X6]
+ (-1.3807781480675324e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (-1.3807781480675324e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (-1.2919694865350102e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.2919694865350102e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.1076325598317528e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.1076325598317528e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.1076325598317528e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.1076325598317528e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.1076325598317528e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.1076325598317528e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.1076325598317528e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.1076325598317528e-07) [X1 Z2 X3 X10 Z11 X12]
+ (8.057446594512792e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.057446594512792e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.057446594512792e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.057446594512792e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.649310132521708e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.649310132521708e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.649310132521708e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.649310132521708e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.8394209155175266e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.8394209155175266e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.8394209155175266e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.8394209155175266e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.199051618834916e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (2.199051618834916e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (2.447323128832056e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (2.447323128832056e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.236259961263184e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.236259961263184e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.236259961263184e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.236259961263184e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.328139350592643e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.328139350592643e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.5707613292425203e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.5707613292425203e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.838052750988301e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (4.838052750988301e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (5.627851911239184e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.627851911239184e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (6.628614201362812e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.628614201362812e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (7.956895372900573e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.956895372900573e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.306536651221527e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.306536651221527e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.306536651221527e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.306536651221527e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.0358477601352874e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.0358477601352874e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.2283337824815328e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.2283337824815328e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.2393363216420002e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.2393363216420002e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.2393363216420002e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.2393363216420002e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.3304731886137888e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.3304731886137888e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.4548424490573079e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.4548424490573079e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.172669101434983e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.172669101434983e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.088250711090746e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.088250711090746e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.117447946137541e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.117447946137541e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.151346310893532e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.151346310893532e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.3131454999910333e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.3131454999910333e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.3343312891416712e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.3343312891416712e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.6945132942591346e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.6945132942591346e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (4.183932559166532e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.183932559166532e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.55656921776986e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (4.55656921776986e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (4.64305106814665e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.64305106814665e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.2758831218345475e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.2758831218345475e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.974311713057489e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (5.974311713057489e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (0.00029219862611108385) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.00029219862611108385) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.00029219862611108385) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.00029219862611108385) [X6 Z7 X8 X9 Z10 X11]
+ (0.0004956762314916487) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.0004956762314916487) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.0006650070219498614) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.0006650070219498614) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.0006650070219498614) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.0006650070219498614) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.0008533856254125472) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.0008533856254125472) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.001609531381721372) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.001609531381721372) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.001609531381721372) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.001609531381721372) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.001667604181144058) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.001667604181144058) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.001667604181144058) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.001667604181144058) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.0017278753941369677) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.0017278753941369677) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.0017992194936630014) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.0017992194936630014) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.0022939566113524762) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.0022939566113524762) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.002462917007133919) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.002462917007133919) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.002462917007133919) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.002462917007133919) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.003961560792496535) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.003961560792496535) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.003961560792496535) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.003961560792496535) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.004424855449441841) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.004424855449441841) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.004575007626639211) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.004575007626639211) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.004668620318776302) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.004668620318776302) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.004684903388155197) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.004684903388155197) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.005324835234221673) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.005324835234221673) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.005324835234221673) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.005324835234221673) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.005368659358109495) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.005368659358109495) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.005368659358109495) [X2 X3 X7 Z8 Z9 X10]
+ (0.005368659358109495) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.007960880725921542) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.007960880725921542) [X4 Z5 X6 X10 Z11 X12]
+ (0.007960880725921542) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.007960880725921542) [X5 Z6 X7 X11 Z12 X13]
+ (0.00812525192138102) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.00812525192138102) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.00889073152269457) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.00889073152269457) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.00889073152269457) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.00889073152269457) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.010263414868158544) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.010263414868158544) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.010263414868158544) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.010263414868158544) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.010540425907671508) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.010540425907671508) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.010540425907671508) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.010540425907671508) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.010960074940542472) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.010960074940542472) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.010960074940542472) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.010960074940542472) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.011307274008848154) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.011307274008848154) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.014411099430130907) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.014411099430130907) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.014411099430130907) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.014411099430130907) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.015225630757226596) [Y3 Z4 Z5 X6 X10 Y11]
+ (0.015225630757226596) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (0.015225630757226596) [X3 Z4 Z5 X6 X10 X11]
+ (0.015225630757226596) [X3 Z4 Z5 Y6 Y10 X11]
+ (0.015588250102380219) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (0.015588250102380219) [X2 Z3 X4 X10 Z11 X12]
+ (0.015588250102380219) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (0.015588250102380219) [X3 Z4 X5 X11 Z12 X13]
+ (0.018266834869375456) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.018266834869375456) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.018266834869375456) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.018266834869375456) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.019020423173039903) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.019020423173039903) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.019020423173039903) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.019020423173039903) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.020175921723535464) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.020175921723535464) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.020175921723535464) [X4 Z5 Z6 X7 X11 X12]
+ (0.020175921723535464) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.020175921723535464) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.020175921723535464) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.020175921723535464) [X5 X6 X10 Z11 Z12 X13]
+ (0.020175921723535464) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.02435307767806902) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.02435307767806902) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.02435307767806902) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.02435307767806902) [X2 Z3 X4 X11 Z12 X13]
+ (0.02435307767806902) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.02435307767806902) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.02435307767806902) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.02435307767806902) [X3 Z4 X5 X10 Z11 X12]
+ (0.024389082531149398) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.024389082531149398) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.024389082531149398) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.024389082531149398) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.025104957138844492) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.025104957138844492) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.025104957138844492) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.025104957138844492) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.030787505389143863) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.030787505389143863) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.04587947078129805) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.04587947078129805) [X0 Z2 Z3 Z4 Z5 X6]
+ (0.05600733087780746) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.05600733087780746) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.05600733087780746) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.05600733087780746) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.056084681246613394) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.056084681246613394) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.056084681246613394) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.056084681246613394) [Z1 X7 Z8 Z9 Z10 X11]
+ (-6.631277927964914e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.631277927964914e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.631277927964912e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.631277927964912e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.5950860067262673e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.5950860067262673e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.5950860067262662e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.5950860067262662e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.042743277013783186) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.042743277013783186) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (0.04274327701378319) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.04274327701378319) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (-0.04764261217638309) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.04764261217638309) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.04764261217638309) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.04764261217638309) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.04171881383982175) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.04171881383982175) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.04171881383982175) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.04171881383982175) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.03956441632289337) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.03956441632289337) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.03956441632289337) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03956441632289337) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03935916802205318) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.03935916802205318) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.03935916802205318) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.03935916802205318) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.039318051947197556) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.039318051947197556) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.039318051947197556) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.039318051947197556) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.035608378988312664) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.035608378988312664) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0299037895126249) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.0299037895126249) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.0299037895126249) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.0299037895126249) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.028730779551905554) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.028730779551905554) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.028730779551905554) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.028730779551905554) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.025637238296026835) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.025637238296026835) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.025637238296026835) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.025637238296026835) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.02475546329289104) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.02475546329289104) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.02475546329289104) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.02475546329289104) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.02428211735469306) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.02428211735469306) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.023145130929529113) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.023145130929529113) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.022528440196012897) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.022528440196012897) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.021433810721601) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.021433810721601) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.021433810721601) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.021433810721601) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251606) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.019257505095251606) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.01902824244384734) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.01902824244384734) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.018889030304942905) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-0.018889030304942905) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-0.018889030304942905) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.018889030304942905) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.01602460368917955) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.01602460368917955) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.015225630757226596) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.015225630757226596) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.014603704729162146) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.014603704729162146) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.014564531231172982) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.014564531231172982) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.011756013419819274) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.011756013419819274) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.011285190200840893) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-0.011285190200840893) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-0.00984174924696265) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.00984174924696265) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.009612634606847283) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.009612634606847283) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.009612634606847283) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.009612634606847283) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.008469978791023906) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.008469978791023906) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.007306759928832983) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.007306759928832983) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005923798336561347) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.005923798336561347) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.005652620978017369) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.005652620978017369) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.005368659358109495) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005368659358109495) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.004158797381840073) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.004158797381840073) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0033566705638328944) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.0033566705638328944) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.0033566705638328944) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.0033566705638328944) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.003267513854423557) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.003267513854423557) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.003267513854423557) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.003267513854423557) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.0027790267990255545) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.0027790267990255545) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.002686040977806622) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.002686040977806622) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.002686040977806622) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.002686040977806622) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.0022939566113524767) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939566113524767) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.0022939566113524767) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939566113524767) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.0009581655836696554) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.0009581655836696554) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.0009581655836696554) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.0009581655836696554) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.0009581655836696554) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.0009581655836696554) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.0009581655836696554) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.0009581655836696554) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.000246364375695814) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.000246364375695814) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303550328) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.00013840177303550328) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.00013840177303550328) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.00013840177303550328) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.735036880592475e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.735036880592475e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585304708656e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585304708656e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.6103585304708656e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.6103585304708656e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808794326665e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.5316808794326665e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808794326665e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5316808794326665e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.806102774489722e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.806102774489722e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.806102774489722e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.806102774489722e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.0897994671756765e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.0897994671756765e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.0897994671756765e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.0897994671756765e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.652209668791919e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.652209668791919e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.652209668791919e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.652209668791919e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851833245475e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.481851833245475e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851833245475e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.481851833245475e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.0714807360281715e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.0714807360281715e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.0714807360281715e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.0714807360281715e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.7346220384615475e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.7346220384615475e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.7346220384615475e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.7346220384615475e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.728843146939049e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.728843146939049e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.728843146939049e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.728843146939049e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.253224225302965e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.253224225302965e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.7696594515937515e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.7696594515937515e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.5443954290885104e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.5443954290885104e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.5443954290885104e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.5443954290885104e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.5443954290885104e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.5443954290885104e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.5443954290885104e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.5443954290885104e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.360956320236628e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.360956320236628e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.360956320236628e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.360956320236628e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-2.1032156044294735e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.1032156044294735e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.1032156044294735e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.1032156044294735e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220978858564e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.0111220978858564e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220978858564e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.0111220978858564e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9429468365564106e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.9429468365564106e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9429468365564106e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9429468365564106e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.654117476927522e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.654117476927522e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.654117476927522e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.654117476927522e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.5224930674911664e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.5224930674911664e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.5224930674911664e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.5224930674911664e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.5224930674911664e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930674911664e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930674911664e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.5224930674911664e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.2283337824815328e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337824815328e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.2283337824815328e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337824815328e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.988770288602579e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.988770288602579e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.988770288602579e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.988770288602579e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.867765103819956e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765103819956e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765103819956e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.867765103819956e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.189990974755708e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.189990974755708e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.175246206674518e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175246206674518e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.47164774426425e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.47164774426425e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.561447179977648e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.561447179977648e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.561447179977648e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.561447179977648e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.523389677465775e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.523389677465775e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.427323108624932e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.427323108624932e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.427323108624932e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.427323108624932e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.328139350592643e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350592643e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.328139350592643e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350592643e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.086826565198475e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.086826565198475e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.888293596288883e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.888293596288883e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.888293596288883e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.888293596288883e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.3713289480110134e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.3713289480110134e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.8394209155175266e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.8394209155175266e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-8.057446594512792e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.057446594512792e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (4.537178095426712e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.537178095426712e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.537178095426712e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (4.537178095426712e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (8.057446594512792e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.057446594512792e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (9.209350654361721e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350654361721e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350654361721e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.209350654361721e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783554644402e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783554644402e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783554644402e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783554644402e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.8394209155175266e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.8394209155175266e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.3713289480110134e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.3713289480110134e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (3.086826565198475e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.086826565198475e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (4.523389677465775e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.523389677465775e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (5.47164774426425e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.47164774426425e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.175246206674518e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175246206674518e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.189990974755708e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.189990974755708e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.3304731886137888e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.3304731886137888e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.3304731886137888e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.3304731886137888e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.6288532434023657e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.6288532434023657e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.6288532434023657e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.6288532434023657e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.6893489513418401e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.6893489513418401e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.6893489513418401e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.6893489513418401e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (2.7455184002282526e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.7455184002282526e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.7455184002282526e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.7455184002282526e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.7455184002282526e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.7455184002282526e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.7455184002282526e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.7455184002282526e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.2118420188330066e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420188330066e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.2118420188330066e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420188330066e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.2118420188330066e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420188330066e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.2118420188330066e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420188330066e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.313145499991033e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.313145499991033e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.313145499991033e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.313145499991033e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (3.334331289141671e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.334331289141671e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (4.183932559166532e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.183932559166532e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (7.735036880592475e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.735036880592475e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.000246364375695814) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.000246364375695814) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0004458535128840802) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.0004458535128840802) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.0004458535128840802) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.0004458535128840802) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.0005940221543005544) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005544) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005544) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005544) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005544) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005544) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.0005940221543005544) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005544) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.0008533856254125473) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.0008533856254125473) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.0008533856254125473) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.0008533856254125473) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.0010435246534907584) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.0010435246534907584) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.0010435246534907584) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.0010435246534907584) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.0012803060973496701) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.0012803060973496701) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.0012803060973496701) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.0012803060973496701) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.0013038004788126943) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (0.0013038004788126943) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (0.0013038004788126943) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (0.0013038004788126943) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (0.00226196606248235) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (0.00226196606248235) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (0.00226196606248235) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (0.00226196606248235) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (0.00226196606248235) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (0.00226196606248235) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (0.00226196606248235) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (0.00226196606248235) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (0.003989841456619317) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (0.003989841456619317) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (0.003989841456619317) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (0.003989841456619317) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (0.004158797381840073) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.004158797381840073) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.004311038507914315) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.004311038507914315) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.004311038507914315) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.004311038507914315) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.004636976661182564) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.004636976661182564) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.004636976661182564) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.004636976661182564) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.005114473831660382) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.005114473831660382) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.005114473831660382) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.005114473831660382) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.005114473831660382) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660382) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660382) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005114473831660382) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.005241535382803876) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (0.005241535382803876) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (0.005241535382803876) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (0.005241535382803876) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (0.005262642473076855) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.005262642473076855) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.005262642473076855) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.005262642473076855) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.005368659358109495) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005368659358109495) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005379937155839379) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.005379937155839379) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.005379937155839379) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.005379937155839379) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.005652620978017369) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.005652620978017369) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.005708495985960935) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.005708495985960935) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.005708495985960935) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.005708495985960935) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.005923798336561347) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.005923798336561347) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.007306759928832983) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.007306759928832983) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.008469978791023906) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.008469978791023906) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.00984174924696265) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.00984174924696265) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.011285190200840893) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (0.011285190200840893) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (0.011756013419819274) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.011756013419819274) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.014564531231172982) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.014564531231172982) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.014603704729162146) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.014603704729162146) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.015225630757226596) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.015225630757226596) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.01602460368917955) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.01602460368917955) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.01902824244384734) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.01902824244384734) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.019257505095251606) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.019257505095251606) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.04587947078129805) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04587947078129805) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.3693708936615624) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.3693708936615624) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.3693708936615624) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.3693708936615624) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.28164257767023065) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.28164257767023065) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.28164257767023054) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.28164257767023054) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.09065144207036469) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.09065144207036469) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.09065144207036469) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.09065144207036469) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863618) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.08684737589863618) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863618) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.08684737589863618) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.07635021950635029) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07635021950635029) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07635021950635029) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07635021950635029) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214042) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.06752385099214042) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214042) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.06752385099214042) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03560837898831266) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.03560837898831266) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.034903343373661744) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.034903343373661744) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.034903343373661744) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.034903343373661744) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883829964) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.024591860883829964) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883829964) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.024591860883829964) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.02428211735469306) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.02428211735469306) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.023145130929529113) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.023145130929529113) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.022528440196012897) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.022528440196012897) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019538050311314805) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.019538050311314805) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.019538050311314805) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.019538050311314805) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.017091553155898945) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.017091553155898945) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.017091553155898945) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.017091553155898945) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.016024603689179545) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.016024603689179545) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.016024603689179545) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.016024603689179545) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.010311482489831778) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831778) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831778) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010311482489831778) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.00984174924696265) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.00984174924696265) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.00984174924696265) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.00984174924696265) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209874) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209874) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209874) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008826368514209874) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008541996625454844) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454844) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454844) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454844) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454844) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454844) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454844) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008541996625454844) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008469978791023906) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023906) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.008469978791023906) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023906) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.004668620318776302) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.004668620318776302) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.003876470899336951) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.003876470899336951) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.003804066171728536) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728536) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728536) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003804066171728536) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003484157300217878) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003484157300217878) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003356670563832894) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.003356670563832894) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.003267513854423557) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.003267513854423557) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0021413612231015585) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.0021413612231015585) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.0017278753941369677) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.0017278753941369677) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.0016407548553124436) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0016407548553124436) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.001452884321416903) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.001452884321416903) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.001452884321416903) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.001452884321416903) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870896771024431) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007870896771024431) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0005192743499487691) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.0005192743499487691) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.0001940085702975649) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0001940085702975649) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303550328) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.00013840177303550328) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.141625221153381e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.141625221153381e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.141625221153381e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.141625221153381e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-5.071480736028172e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.071480736028172e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.151346310893532e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.151346310893532e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.088250711090746e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.088250711090746e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.9885117062387915e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.9885117062387915e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.874299071224402e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.874299071224402e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.3609563202366283e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.3609563202366283e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.300294656131589e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.300294656131589e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.1468376506598585e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.1468376506598585e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.1468376506598585e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.1468376506598585e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.35233210242304e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.35233210242304e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.35233210242304e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.35233210242304e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.091637198338709e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637198338709e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637198338709e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637198338709e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637198338709e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637198338709e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637198338709e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.091637198338709e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.07430598527014e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.07430598527014e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.07430598527014e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.07430598527014e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.900128985725649e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.900128985725649e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.900128985725649e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900128985725649e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867765103819956e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.867765103819956e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.560692464297741e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692464297741e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692464297741e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692464297741e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692464297741e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692464297741e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692464297741e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.560692464297741e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.997018421675544e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.997018421675544e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.997018421675544e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.997018421675544e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.997018421675544e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.997018421675544e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.997018421675544e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.997018421675544e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.5682475208729344e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.5682475208729344e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.5682475208729344e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.5682475208729344e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393082598746e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393082598746e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393082598746e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393082598746e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393082598746e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393082598746e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.3767393082598746e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393082598746e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.888293596288883e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.888293596288883e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6863815444737853e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.6863815444737853e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035783554644396e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7035783554644396e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.209350654361721e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.209350654361721e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773243198647e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773243198647e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773243198647e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773243198647e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773243198647e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773243198647e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.379773243198647e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773243198647e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253793283145e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253793283145e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9742253793283145e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9742253793283145e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0474716554332859e-08) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.0474716554332859e-08) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.0474716554332859e-08) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.0474716554332859e-08) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.209350654361721e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.209350654361721e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0717282182567258e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.0717282182567258e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.0717282182567258e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.0717282182567258e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.2004287493541528e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.2004287493541528e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.2004287493541528e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.2004287493541528e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.7035783554644396e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7035783554644396e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.3120943051420206e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.3120943051420206e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.3120943051420206e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.3120943051420206e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.6863815444737853e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815444737853e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.888293596288883e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.888293596288883e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.092250615741314e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.092250615741314e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.092250615741314e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.092250615741314e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.092250615741314e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.092250615741314e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.092250615741314e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.092250615741314e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.4445978539252117e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.4445978539252117e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.4445978539252117e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.4445978539252117e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.6849150949283104e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.6849150949283104e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.6849150949283104e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.6849150949283104e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.246974424931244e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.246974424931244e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.246974424931244e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.246974424931244e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.246974424931244e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.246974424931244e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.246974424931244e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.246974424931244e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.867765103819956e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.867765103819956e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.300294656131589e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.300294656131589e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.3609563202366283e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.3609563202366283e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.874299071224402e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.874299071224402e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.8836765759265984e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.8836765759265984e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.9473560114557034e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9473560114557034e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.9473560114557034e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9473560114557034e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117062387915e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9885117062387915e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.088250711090746e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.088250711090746e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.151346310893532e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.151346310893532e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.846201671021108e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.846201671021108e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.846201671021108e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.846201671021108e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.071480736028172e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.071480736028172e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.10552672171925e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.10552672171925e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.10552672171925e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.10552672171925e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.146496327152698e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.146496327152698e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.146496327152698e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.146496327152698e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.159350501692646e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.159350501692646e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.159350501692646e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.159350501692646e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.427988656140024e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.427988656140024e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.427988656140024e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.427988656140024e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.935867717694495e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.935867717694495e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.935867717694495e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.935867717694495e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273347712183e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.253273347712183e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.979825792943652e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.979825792943652e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.979825792943652e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.979825792943652e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (4.2055484112183544e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.2055484112183544e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.2055484112183544e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.2055484112183544e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00013840177303550328) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.00013840177303550328) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.0001878705338955407) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0001878705338955407) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0001878705338955407) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0001878705338955407) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0001940085702975649) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.0001940085702975649) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.000246364375695814) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.000246364375695814) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.000246364375695814) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.000246364375695814) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192743499487691) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.0005192743499487691) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.0007156734248909091) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007156734248909091) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0007156734248909091) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007156734248909091) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024431) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007870896771024431) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.001532483523073075) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.001532483523073075) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.001532483523073075) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.001532483523073075) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.0016407548553124436) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0016407548553124436) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0017278753941369677) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.0017278753941369677) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.0024464971554158583) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.0024464971554158583) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.0024464971554158583) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.0024464971554158583) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.003267513854423557) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.003267513854423557) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.003356670563832894) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.003356670563832894) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.003484157300217878) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.003484157300217878) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.003876470899336951) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.003876470899336951) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.004668620318776302) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.004668620318776302) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278155) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.004767272188278155) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.004767272188278155) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278155) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.005286546538226924) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.005286546538226924) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.005286546538226924) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.005286546538226924) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.005408954422410026) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.005408954422410026) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.005408954422410026) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.005408954422410026) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.005923798336561347) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561347) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.005923798336561347) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561347) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.010715508469796766) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010715508469796766) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010715508469796766) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010715508469796766) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01075756395390895) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01075756395390895) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.01075756395390895) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01075756395390895) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.014603704729162146) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.014603704729162146) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.014603704729162146) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.014603704729162146) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.019299560579363793) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363793) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363793) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363793) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363793) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363793) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.019299560579363793) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363793) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.058591988733862024) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.058591988733862024) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.7759505269753304e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.7759505269753304e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.775950526975332e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.775950526975332e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.07165035181002737) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.07165035181002737) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.07165035181002738) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.07165035181002738) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251606) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019257505095251606) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010311482489831778) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311482489831778) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008826368514209874) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209874) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.007597464029770627) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.007597464029770627) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.007597464029770627) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.007597464029770627) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0057335697473118695) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0057335697473118695) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0057335697473118695) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0057335697473118695) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0057335697473118695) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0057335697473118695) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0057335697473118695) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0057335697473118695) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676637) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005348051582676637) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005348051582676637) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676637) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.003804066171728536) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003804066171728536) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0029841661681219373) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.0029841661681219373) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.0029841661681219373) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.0029841661681219373) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.0024464971554158583) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.0024464971554158583) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.0022494124470939904) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0022494124470939904) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0022494124470939904) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0022494124470939904) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0021413612231015585) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.0021413612231015585) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.0018638942824587587) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587587) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587587) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587587) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587587) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587587) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0018638942824587587) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587587) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0016407548553124436) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124436) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0016407548553124436) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124436) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0012223378081538377) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538377) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538377) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538377) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538377) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538377) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538377) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0012223378081538377) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0010283292378562726) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0010283292378562726) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0010283292378562726) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0010283292378562726) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1463061452457945e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1463061452457945e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.874299071224402e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.874299071224402e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.874299071224402e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.874299071224402e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.300294656131589e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.300294656131589e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.300294656131589e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.300294656131589e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.0444941297729537e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.0444941297729537e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.0444941297729537e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.0444941297729537e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.956079229678416e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.956079229678416e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.956079229678416e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.956079229678416e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105515036898032e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105515036898032e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105515036898032e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105515036898032e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661347212839333e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661347212839333e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661347212839333e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661347212839333e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540341413240216e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540341413240216e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.189990974755708e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.189990974755708e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.87662165802578e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.87662165802578e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.87662165802578e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.87662165802578e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175246206674518e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175246206674518e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.523389677465775e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.523389677465775e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0767325321580573e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.0767325321580573e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0767325321580573e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.0767325321580573e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.013471458736137e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.013471458736137e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.9045998844893215e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.9045998844893215e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.9045998844893215e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.9045998844893215e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.666731755093001e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.666731755093001e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.666731755093001e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.666731755093001e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641927803864e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.8505641927803864e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309313458506e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.6569309313458506e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309313458506e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.6569309313458506e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.8505641927803864e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.8505641927803864e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815444737853e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815444737853e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.6863815444737853e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815444737853e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458736137e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.013471458736137e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.523389677465775e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.523389677465775e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.6704023900819875e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.6704023900819875e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.6704023900819875e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.6704023900819875e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175246206674518e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175246206674518e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.189990974755708e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.189990974755708e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540341413240216e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540341413240216e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949476486467515e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949476486467515e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.7924939576044865e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939576044865e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939576044865e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.7924939576044865e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.883676575926599e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.883676575926599e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.9885117062387915e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117062387915e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117062387915e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117062387915e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273347712183e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.253273347712183e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4017109734481867e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4017109734481867e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4017109734481867e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4017109734481867e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603692086352e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.5809603692086352e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603692086352e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.5809603692086352e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0005192743499487691) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487691) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.0005192743499487691) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487691) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.0007870896771024431) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024431) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024431) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024431) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.001172634831644184) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.001172634831644184) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.001172634831644184) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.001172634831644184) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012366478019245745) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.0012366478019245745) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.0012366478019245745) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.0012366478019245745) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.0022009640695004563) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0022009640695004563) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0022009640695004563) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0022009640695004563) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980214) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980214) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0023949726397980214) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980214) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0023949726397980214) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
 </code>
 </pre>
 </details>

---

## 12. tutorial_chemical_reactions.html <a name="demo11"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_chemical_reactions.html):

```
Ratio of reaction rates is 1948918
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_chemical_reactions.html):

```
Ratio of reaction rates is 1949082
```

---

## 13. tutorial_jax_transformations.html <a name="demo12"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_jax_transformations.html):

```
No jit time: 0.0085 seconds
First run time: 0.0665 seconds
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_jax_transformations.html):

```
No jit time: 0.0098 seconds
First run time: 0.0748 seconds
```

---

## 14. tutorial_vqe_spin_sectors.html <a name="demo13"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_vqe_spin_sectors.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
The Hamiltonian is    (-0.2427450172749822) [Z2]
+ (-0.2427450172749822) [Z3]
+ (-0.04207254303152995) [I0]
+ (0.17771358191549907) [Z0]
+ (0.17771358191549919) [Z1]
+ (0.12293330460167415) [Z0 Z2]
+ (0.12293330460167415) [Z1 Z3]
+ (0.16768338881432715) [Z0 Z3]
+ (0.16768338881432715) [Z1 Z2]
+ (0.17059759240560826) [Z0 Z1]
+ (0.17627661476093917) [Z2 Z3]
+ (-0.04475008421265302) [Y0 Y1 X2 X3]
+ (-0.04475008421265302) [X0 X1 Y2 Y3]
+ (0.04475008421265302) [Y0 X1 X2 Y3]
+ (0.04475008421265302) [X0 Y1 Y2 X3]
  (0.375) [Z1]
+ (0.375) [Z0]
+ (0.375) [Z2]
+ (0.375) [Z3]
+ (0.75) [I0]
+ (-0.375) [Z0 Z1]
+ (-0.375) [Z2 Z3]
+ (-0.125) [Z0 Z3]
+ (-0.125) [Z1 Z2]
+ (0.125) [Z0 Z2]
+ (0.125) [Z1 Z3]
+ (-0.125) [Y0 X1 X2 Y3]
+ (-0.125) [X0 Y1 Y2 X3]
+ (0.125) [Y0 X1 Y2 X3]
+ (0.125) [Y0 Y1 X2 X3]
+ (0.125) [Y0 Y1 Y2 Y3]
+ (0.125) [X0 X1 X2 X3]
+ (0.125) [X0 X1 Y2 Y3]
+ (0.125) [X0 Y1 X2 Y3]
Step = 0, Energy = -0.09929556 Ha, S = 0.1014
Optimal value of the circuit parameters = [3.14350662 3.14087516 2.93185887]
Step = 0, Energy = 0.31463320 Ha, S = 0.3539
Step = 8, Energy = -0.47698617 Ha, S = 0.9991
Step = 12, Energy = -0.47842742 Ha, S = 1.0000
Step = 16, Energy = -0.47844666 Ha, S = 1.0000
Final value of the energy = -0.47844666 Ha
 </code>
 </pre>
 </details>

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_vqe_spin_sectors.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
The Hamiltonian is    (-0.2427450125036239) [Z2]
+ (-0.2427450125036239) [Z3]
+ (-0.04207255204119781) [I0]
+ (0.17771358235560275) [Z0]
+ (0.17771358235560275) [Z1]
+ (0.12293330446038843) [Z0 Z2]
+ (0.12293330446038843) [Z1 Z3]
+ (0.16768338851153938) [Z0 Z3]
+ (0.16768338851153938) [Z1 Z2]
+ (0.17059759275416475) [Z0 Z1]
+ (0.1762766138629876) [Z2 Z3]
+ (-0.04475008405115097) [Y0 Y1 X2 X3]
+ (-0.04475008405115097) [X0 X1 Y2 Y3]
+ (0.04475008405115097) [Y0 X1 X2 Y3]
+ (0.04475008405115097) [X0 Y1 Y2 X3]
  ((0.375+0j)) [Z0]
+ ((0.375+0j)) [Z1]
+ ((0.375+0j)) [Z2]
+ ((0.375+0j)) [Z3]
+ ((0.75+0j)) [I0]
+ ((-0.375+0j)) [Z0 Z1]
+ ((-0.375+0j)) [Z2 Z3]
+ ((-0.125+0j)) [Z0 Z3]
+ ((-0.125+0j)) [Z1 Z2]
+ ((0.125+0j)) [Z0 Z2]
+ ((0.125+0j)) [Z1 Z3]
+ ((-0.125+0j)) [Y0 X1 X2 Y3]
+ ((-0.125+0j)) [X0 Y1 Y2 X3]
+ ((0.125+0j)) [Y0 X1 Y2 X3]
+ ((0.125+0j)) [Y0 Y1 X2 X3]
+ ((0.125+0j)) [Y0 Y1 Y2 Y3]
+ ((0.125+0j)) [X0 X1 X2 X3]
+ ((0.125+0j)) [X0 X1 Y2 Y3]
+ ((0.125+0j)) [X0 Y1 X2 Y3]
Step = 0, Energy = -0.09929557 Ha, S = 0.1014
Optimal value of the circuit parameters = [3.14350662 3.14087516 2.93185886]
Step = 0, Energy = 0.31463319 Ha, S = 0.3539
Step = 8, Energy = -0.47698618 Ha, S = 0.9991
Step = 12, Energy = -0.47842743 Ha, S = 1.0000
Step = 16, Energy = -0.47844667 Ha, S = 1.0000
Final value of the energy = -0.47844667 Ha
 </code>
 </pre>
 </details>

---

## 15. tutorial_sc_qubits.html <a name="demo14"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_sc_qubits.html):

```
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/transforms/get_unitary_matrix.py:107: UserWarning: get_unitary_matrix is deprecated, and will be removed in an upcoming release. For extracting matrices of operations and quantum functions, please use qml.matrix().
array([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
       [-0.+0.j,  1.-0.j,  0.+0.j,  0.+0.j],
       [ 0.+0.j,  0.+0.j, -0.+0.j,  1.+0.j],
       [ 0.+0.j,  0.+0.j,  1.-0.j,  0.+0.j]])
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_sc_qubits.html):

```
array([[ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j],
       [-0.+0.j,  1.-0.j,  0.+0.j,  0.+0.j],
       [ 0.+0.j,  0.+0.j, -0.+0.j,  1.+0.j],
       [ 0.+0.j,  0.+0.j,  1.-0.j,  0.+0.j]])
tensor([0.707+0.j, 0.   +0.j, 0.   -0.j, 0.707+0.j], requires_grad=True)
```

---

## 16. tutorial_quantum_transfer_learning.html <a name="demo15"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_quantum_transfer_learning.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
 14%|#4        | 6.28M/44.7M [00:00<00:00, 65.8MB/s]
 58%|#####7    | 25.9M/44.7M [00:00<00:00, 148MB/s]
100%|##########| 44.7M/44.7M [00:00<00:00, 159MB/s]
Training started:
Phase: train Epoch: 1/3 Iter: 1/62 Batch time: 0.2531
Phase: train Epoch: 1/3 Iter: 2/62 Batch time: 0.2263
Phase: train Epoch: 1/3 Iter: 3/62 Batch time: 0.2331
Phase: train Epoch: 1/3 Iter: 4/62 Batch time: 0.2277
Phase: train Epoch: 1/3 Iter: 5/62 Batch time: 0.2280
Phase: train Epoch: 1/3 Iter: 6/62 Batch time: 0.2292
Phase: train Epoch: 1/3 Iter: 7/62 Batch time: 0.2330
Phase: train Epoch: 1/3 Iter: 8/62 Batch time: 0.2314
Phase: train Epoch: 1/3 Iter: 9/62 Batch time: 0.2323
Phase: train Epoch: 1/3 Iter: 10/62 Batch time: 0.2301
Phase: train Epoch: 1/3 Iter: 11/62 Batch time: 0.2391
Phase: train Epoch: 1/3 Iter: 12/62 Batch time: 0.2270
Phase: train Epoch: 1/3 Iter: 13/62 Batch time: 0.2268
Phase: train Epoch: 1/3 Iter: 14/62 Batch time: 0.2256
Phase: train Epoch: 1/3 Iter: 15/62 Batch time: 0.2301
Phase: train Epoch: 1/3 Iter: 16/62 Batch time: 0.2263
Phase: train Epoch: 1/3 Iter: 17/62 Batch time: 0.2276
Phase: train Epoch: 1/3 Iter: 18/62 Batch time: 0.2267
Phase: train Epoch: 1/3 Iter: 19/62 Batch time: 0.2278
Phase: train Epoch: 1/3 Iter: 20/62 Batch time: 0.2254
Phase: train Epoch: 1/3 Iter: 21/62 Batch time: 0.2291
Phase: train Epoch: 1/3 Iter: 22/62 Batch time: 0.2286
Phase: train Epoch: 1/3 Iter: 23/62 Batch time: 0.2305
Phase: train Epoch: 1/3 Iter: 24/62 Batch time: 0.2289
Phase: train Epoch: 1/3 Iter: 25/62 Batch time: 0.2258
Phase: train Epoch: 1/3 Iter: 26/62 Batch time: 0.2276
Phase: train Epoch: 1/3 Iter: 27/62 Batch time: 0.2253
Phase: train Epoch: 1/3 Iter: 28/62 Batch time: 0.2243
Phase: train Epoch: 1/3 Iter: 29/62 Batch time: 0.2239
Phase: train Epoch: 1/3 Iter: 30/62 Batch time: 0.2268
Phase: train Epoch: 1/3 Iter: 31/62 Batch time: 0.2268
Phase: train Epoch: 1/3 Iter: 32/62 Batch time: 0.2377
Phase: train Epoch: 1/3 Iter: 33/62 Batch time: 0.2318
Phase: train Epoch: 1/3 Iter: 34/62 Batch time: 0.2301
Phase: train Epoch: 1/3 Iter: 35/62 Batch time: 0.2347
Phase: train Epoch: 1/3 Iter: 36/62 Batch time: 0.2248
Phase: train Epoch: 1/3 Iter: 37/62 Batch time: 0.2228
Phase: train Epoch: 1/3 Iter: 38/62 Batch time: 0.2238
Phase: train Epoch: 1/3 Iter: 39/62 Batch time: 0.2235
Phase: train Epoch: 1/3 Iter: 40/62 Batch time: 0.2232
Phase: train Epoch: 1/3 Iter: 41/62 Batch time: 0.2239
Phase: train Epoch: 1/3 Iter: 42/62 Batch time: 0.2258
Phase: train Epoch: 1/3 Iter: 43/62 Batch time: 0.2252
Phase: train Epoch: 1/3 Iter: 44/62 Batch time: 0.2253
Phase: train Epoch: 1/3 Iter: 45/62 Batch time: 0.2232
Phase: train Epoch: 1/3 Iter: 46/62 Batch time: 0.2257
Phase: train Epoch: 1/3 Iter: 47/62 Batch time: 0.2266
Phase: train Epoch: 1/3 Iter: 48/62 Batch time: 0.2230
Phase: train Epoch: 1/3 Iter: 49/62 Batch time: 0.2242
Phase: train Epoch: 1/3 Iter: 50/62 Batch time: 0.2275
Phase: train Epoch: 1/3 Iter: 51/62 Batch time: 0.2245
Phase: train Epoch: 1/3 Iter: 52/62 Batch time: 0.2255
Phase: train Epoch: 1/3 Iter: 53/62 Batch time: 0.2243
Phase: train Epoch: 1/3 Iter: 54/62 Batch time: 0.2291
Phase: train Epoch: 1/3 Iter: 55/62 Batch time: 0.2238
Phase: train Epoch: 1/3 Iter: 56/62 Batch time: 0.2261
Phase: train Epoch: 1/3 Iter: 57/62 Batch time: 0.2239
Phase: train Epoch: 1/3 Iter: 58/62 Batch time: 0.2264
Phase: train Epoch: 1/3 Iter: 59/62 Batch time: 0.2258
Phase: train Epoch: 1/3 Iter: 60/62 Batch time: 0.2285
Phase: train Epoch: 1/3 Iter: 61/62 Batch time: 0.2245
Phase: train Epoch: 1/3 Loss: 0.6993 Acc: 0.5246
Phase: validation Epoch: 1/3 Iter: 1/39 Batch time: 0.1783
Phase: validation Epoch: 1/3 Iter: 2/39 Batch time: 0.1763
Phase: validation Epoch: 1/3 Iter: 3/39 Batch time: 0.1746
Phase: validation Epoch: 1/3 Iter: 4/39 Batch time: 0.1755
Phase: validation Epoch: 1/3 Iter: 5/39 Batch time: 0.1736
Phase: validation Epoch: 1/3 Iter: 6/39 Batch time: 0.1748
Phase: validation Epoch: 1/3 Iter: 7/39 Batch time: 0.1727
Phase: validation Epoch: 1/3 Iter: 8/39 Batch time: 0.1743
Phase: validation Epoch: 1/3 Iter: 9/39 Batch time: 0.1735
Phase: validation Epoch: 1/3 Iter: 10/39 Batch time: 0.1741
Phase: validation Epoch: 1/3 Iter: 11/39 Batch time: 0.1763
Phase: validation Epoch: 1/3 Iter: 12/39 Batch time: 0.1760
Phase: validation Epoch: 1/3 Iter: 13/39 Batch time: 0.1739
Phase: validation Epoch: 1/3 Iter: 14/39 Batch time: 0.1742
Phase: validation Epoch: 1/3 Iter: 15/39 Batch time: 0.1740
Phase: validation Epoch: 1/3 Iter: 16/39 Batch time: 0.1777
Phase: validation Epoch: 1/3 Iter: 17/39 Batch time: 0.1738
Phase: validation Epoch: 1/3 Iter: 18/39 Batch time: 0.1749
Phase: validation Epoch: 1/3 Iter: 19/39 Batch time: 0.1755
Phase: validation Epoch: 1/3 Iter: 20/39 Batch time: 0.1749
Phase: validation Epoch: 1/3 Iter: 21/39 Batch time: 0.1756
Phase: validation Epoch: 1/3 Iter: 22/39 Batch time: 0.1743
Phase: validation Epoch: 1/3 Iter: 23/39 Batch time: 0.1729
Phase: validation Epoch: 1/3 Iter: 24/39 Batch time: 0.1744
Phase: validation Epoch: 1/3 Iter: 25/39 Batch time: 0.1716
Phase: validation Epoch: 1/3 Iter: 26/39 Batch time: 0.1744
Phase: validation Epoch: 1/3 Iter: 27/39 Batch time: 0.1735
Phase: validation Epoch: 1/3 Iter: 28/39 Batch time: 0.1733
Phase: validation Epoch: 1/3 Iter: 29/39 Batch time: 0.1708
Phase: validation Epoch: 1/3 Iter: 30/39 Batch time: 0.1726
Phase: validation Epoch: 1/3 Iter: 31/39 Batch time: 0.1753
Phase: validation Epoch: 1/3 Iter: 32/39 Batch time: 0.1700
Phase: validation Epoch: 1/3 Iter: 33/39 Batch time: 0.1708
Phase: validation Epoch: 1/3 Iter: 34/39 Batch time: 0.1696
Phase: validation Epoch: 1/3 Iter: 35/39 Batch time: 0.1700
Phase: validation Epoch: 1/3 Iter: 36/39 Batch time: 0.1710
Phase: validation Epoch: 1/3 Iter: 37/39 Batch time: 0.1696
Phase: validation Epoch: 1/3 Iter: 38/39 Batch time: 0.1702
Phase: validation Epoch: 1/3 Iter: 39/39 Batch time: 0.0603
Phase: validation   Epoch: 1/3 Loss: 0.6432 Acc: 0.6536
Phase: train Epoch: 2/3 Iter: 1/62 Batch time: 0.2209
Phase: train Epoch: 2/3 Iter: 2/62 Batch time: 0.2270
Phase: train Epoch: 2/3 Iter: 3/62 Batch time: 0.2243
Phase: train Epoch: 2/3 Iter: 4/62 Batch time: 0.2239
Phase: train Epoch: 2/3 Iter: 5/62 Batch time: 0.2241
Phase: train Epoch: 2/3 Iter: 6/62 Batch time: 0.2251
Phase: train Epoch: 2/3 Iter: 7/62 Batch time: 0.2244
Phase: train Epoch: 2/3 Iter: 8/62 Batch time: 0.2241
Phase: train Epoch: 2/3 Iter: 9/62 Batch time: 0.2259
Phase: train Epoch: 2/3 Iter: 10/62 Batch time: 0.2256
Phase: train Epoch: 2/3 Iter: 11/62 Batch time: 0.2295
Phase: train Epoch: 2/3 Iter: 12/62 Batch time: 0.2251
Phase: train Epoch: 2/3 Iter: 13/62 Batch time: 0.2240
Phase: train Epoch: 2/3 Iter: 14/62 Batch time: 0.2279
Phase: train Epoch: 2/3 Iter: 15/62 Batch time: 0.2238
Phase: train Epoch: 2/3 Iter: 16/62 Batch time: 0.2239
Phase: train Epoch: 2/3 Iter: 17/62 Batch time: 0.2285
Phase: train Epoch: 2/3 Iter: 18/62 Batch time: 0.2292
Phase: train Epoch: 2/3 Iter: 19/62 Batch time: 0.2254
Phase: train Epoch: 2/3 Iter: 20/62 Batch time: 0.2260
Phase: train Epoch: 2/3 Iter: 21/62 Batch time: 0.2250
Phase: train Epoch: 2/3 Iter: 22/62 Batch time: 0.2347
Phase: train Epoch: 2/3 Iter: 23/62 Batch time: 0.2221
Phase: train Epoch: 2/3 Iter: 24/62 Batch time: 0.2246
Phase: train Epoch: 2/3 Iter: 25/62 Batch time: 0.2235
Phase: train Epoch: 2/3 Iter: 26/62 Batch time: 0.2238
Phase: train Epoch: 2/3 Iter: 27/62 Batch time: 0.2230
Phase: train Epoch: 2/3 Iter: 28/62 Batch time: 0.2227
Phase: train Epoch: 2/3 Iter: 29/62 Batch time: 0.2250
Phase: train Epoch: 2/3 Iter: 30/62 Batch time: 0.2231
Phase: train Epoch: 2/3 Iter: 31/62 Batch time: 0.2235
Phase: train Epoch: 2/3 Iter: 32/62 Batch time: 0.2229
Phase: train Epoch: 2/3 Iter: 33/62 Batch time: 0.2253
Phase: train Epoch: 2/3 Iter: 34/62 Batch time: 0.2371
Phase: train Epoch: 2/3 Iter: 35/62 Batch time: 0.2275
Phase: train Epoch: 2/3 Iter: 36/62 Batch time: 0.2232
Phase: train Epoch: 2/3 Iter: 37/62 Batch time: 0.2251
Phase: train Epoch: 2/3 Iter: 38/62 Batch time: 0.2231
Phase: train Epoch: 2/3 Iter: 39/62 Batch time: 0.2231
Phase: train Epoch: 2/3 Iter: 40/62 Batch time: 0.2234
Phase: train Epoch: 2/3 Iter: 41/62 Batch time: 0.2254
Phase: train Epoch: 2/3 Iter: 42/62 Batch time: 0.2250
Phase: train Epoch: 2/3 Iter: 43/62 Batch time: 0.2234
Phase: train Epoch: 2/3 Iter: 44/62 Batch time: 0.2242
Phase: train Epoch: 2/3 Iter: 45/62 Batch time: 0.2238
Phase: train Epoch: 2/3 Iter: 46/62 Batch time: 0.2241
Phase: train Epoch: 2/3 Iter: 47/62 Batch time: 0.2243
Phase: train Epoch: 2/3 Iter: 48/62 Batch time: 0.2255
Phase: train Epoch: 2/3 Iter: 49/62 Batch time: 0.2270
Phase: train Epoch: 2/3 Iter: 50/62 Batch time: 0.2282
Phase: train Epoch: 2/3 Iter: 51/62 Batch time: 0.2277
Phase: train Epoch: 2/3 Iter: 52/62 Batch time: 0.2248
Phase: train Epoch: 2/3 Iter: 53/62 Batch time: 0.2267
Phase: train Epoch: 2/3 Iter: 54/62 Batch time: 0.2259
Phase: train Epoch: 2/3 Iter: 55/62 Batch time: 0.2253
Phase: train Epoch: 2/3 Iter: 56/62 Batch time: 0.2236
Phase: train Epoch: 2/3 Iter: 57/62 Batch time: 0.2284
Phase: train Epoch: 2/3 Iter: 58/62 Batch time: 0.2248
Phase: train Epoch: 2/3 Iter: 59/62 Batch time: 0.2223
Phase: train Epoch: 2/3 Iter: 60/62 Batch time: 0.2238
Phase: train Epoch: 2/3 Iter: 61/62 Batch time: 0.2277
Phase: train Epoch: 2/3 Loss: 0.6141 Acc: 0.7049
Phase: validation Epoch: 2/3 Iter: 1/39 Batch time: 0.1778
Phase: validation Epoch: 2/3 Iter: 2/39 Batch time: 0.1685
Phase: validation Epoch: 2/3 Iter: 3/39 Batch time: 0.1700
Phase: validation Epoch: 2/3 Iter: 4/39 Batch time: 0.1696
Phase: validation Epoch: 2/3 Iter: 5/39 Batch time: 0.1683
Phase: validation Epoch: 2/3 Iter: 6/39 Batch time: 0.1679
Phase: validation Epoch: 2/3 Iter: 7/39 Batch time: 0.1692
Phase: validation Epoch: 2/3 Iter: 8/39 Batch time: 0.1686
Phase: validation Epoch: 2/3 Iter: 9/39 Batch time: 0.1677
Phase: validation Epoch: 2/3 Iter: 10/39 Batch time: 0.1703
Phase: validation Epoch: 2/3 Iter: 11/39 Batch time: 0.1682
Phase: validation Epoch: 2/3 Iter: 12/39 Batch time: 0.1680
Phase: validation Epoch: 2/3 Iter: 13/39 Batch time: 0.1677
Phase: validation Epoch: 2/3 Iter: 14/39 Batch time: 0.1752
Phase: validation Epoch: 2/3 Iter: 15/39 Batch time: 0.1707
Phase: validation Epoch: 2/3 Iter: 16/39 Batch time: 0.1707
Phase: validation Epoch: 2/3 Iter: 17/39 Batch time: 0.1694
Phase: validation Epoch: 2/3 Iter: 18/39 Batch time: 0.1688
Phase: validation Epoch: 2/3 Iter: 19/39 Batch time: 0.1682
Phase: validation Epoch: 2/3 Iter: 20/39 Batch time: 0.1709
Phase: validation Epoch: 2/3 Iter: 21/39 Batch time: 0.1674
Phase: validation Epoch: 2/3 Iter: 22/39 Batch time: 0.1691
Phase: validation Epoch: 2/3 Iter: 23/39 Batch time: 0.1710
Phase: validation Epoch: 2/3 Iter: 24/39 Batch time: 0.1679
Phase: validation Epoch: 2/3 Iter: 25/39 Batch time: 0.1732
Phase: validation Epoch: 2/3 Iter: 26/39 Batch time: 0.1694
Phase: validation Epoch: 2/3 Iter: 27/39 Batch time: 0.1712
Phase: validation Epoch: 2/3 Iter: 28/39 Batch time: 0.1723
Phase: validation Epoch: 2/3 Iter: 29/39 Batch time: 0.1694
Phase: validation Epoch: 2/3 Iter: 30/39 Batch time: 0.1710
Phase: validation Epoch: 2/3 Iter: 31/39 Batch time: 0.1689
Phase: validation Epoch: 2/3 Iter: 32/39 Batch time: 0.1702
Phase: validation Epoch: 2/3 Iter: 33/39 Batch time: 0.1690
Phase: validation Epoch: 2/3 Iter: 34/39 Batch time: 0.1693
Phase: validation Epoch: 2/3 Iter: 35/39 Batch time: 0.1715
Phase: validation Epoch: 2/3 Iter: 36/39 Batch time: 0.1683
Phase: validation Epoch: 2/3 Iter: 37/39 Batch time: 0.1698
Phase: validation Epoch: 2/3 Iter: 38/39 Batch time: 0.1676
Phase: validation Epoch: 2/3 Iter: 39/39 Batch time: 0.0529
Phase: validation   Epoch: 2/3 Loss: 0.5392 Acc: 0.8235
Phase: train Epoch: 3/3 Iter: 1/62 Batch time: 0.2178
Phase: train Epoch: 3/3 Iter: 2/62 Batch time: 0.2226
Phase: train Epoch: 3/3 Iter: 3/62 Batch time: 0.2221
Phase: train Epoch: 3/3 Iter: 4/62 Batch time: 0.2236
Phase: train Epoch: 3/3 Iter: 5/62 Batch time: 0.2233
Phase: train Epoch: 3/3 Iter: 6/62 Batch time: 0.2241
Phase: train Epoch: 3/3 Iter: 7/62 Batch time: 0.2234
Phase: train Epoch: 3/3 Iter: 8/62 Batch time: 0.2239
Phase: train Epoch: 3/3 Iter: 9/62 Batch time: 0.2344
Phase: train Epoch: 3/3 Iter: 10/62 Batch time: 0.2253
Phase: train Epoch: 3/3 Iter: 11/62 Batch time: 0.2239
Phase: train Epoch: 3/3 Iter: 12/62 Batch time: 0.2232
Phase: train Epoch: 3/3 Iter: 13/62 Batch time: 0.2267
Phase: train Epoch: 3/3 Iter: 14/62 Batch time: 0.2235
Phase: train Epoch: 3/3 Iter: 15/62 Batch time: 0.2300
Phase: train Epoch: 3/3 Iter: 16/62 Batch time: 0.2242
Phase: train Epoch: 3/3 Iter: 17/62 Batch time: 0.2249
Phase: train Epoch: 3/3 Iter: 18/62 Batch time: 0.2242
Phase: train Epoch: 3/3 Iter: 19/62 Batch time: 0.2250
Phase: train Epoch: 3/3 Iter: 20/62 Batch time: 0.2241
Phase: train Epoch: 3/3 Iter: 21/62 Batch time: 0.2270
Phase: train Epoch: 3/3 Iter: 22/62 Batch time: 0.2279
Phase: train Epoch: 3/3 Iter: 23/62 Batch time: 0.2251
Phase: train Epoch: 3/3 Iter: 24/62 Batch time: 0.2249
Phase: train Epoch: 3/3 Iter: 25/62 Batch time: 0.2276
Phase: train Epoch: 3/3 Iter: 26/62 Batch time: 0.2243
Phase: train Epoch: 3/3 Iter: 27/62 Batch time: 0.2245
Phase: train Epoch: 3/3 Iter: 28/62 Batch time: 0.2235
Phase: train Epoch: 3/3 Iter: 29/62 Batch time: 0.2297
Phase: train Epoch: 3/3 Iter: 30/62 Batch time: 0.2241
Phase: train Epoch: 3/3 Iter: 31/62 Batch time: 0.2247
Phase: train Epoch: 3/3 Iter: 32/62 Batch time: 0.2249
Phase: train Epoch: 3/3 Iter: 33/62 Batch time: 0.2244
Phase: train Epoch: 3/3 Iter: 34/62 Batch time: 0.2227
Phase: train Epoch: 3/3 Iter: 35/62 Batch time: 0.2240
Phase: train Epoch: 3/3 Iter: 36/62 Batch time: 0.2242
Phase: train Epoch: 3/3 Iter: 37/62 Batch time: 0.2262
Phase: train Epoch: 3/3 Iter: 38/62 Batch time: 0.2242
Phase: train Epoch: 3/3 Iter: 39/62 Batch time: 0.2245
Phase: train Epoch: 3/3 Iter: 40/62 Batch time: 0.2246
Phase: train Epoch: 3/3 Iter: 41/62 Batch time: 0.2257
Phase: train Epoch: 3/3 Iter: 42/62 Batch time: 0.2234
Phase: train Epoch: 3/3 Iter: 43/62 Batch time: 0.2242
Phase: train Epoch: 3/3 Iter: 44/62 Batch time: 0.2231
Phase: train Epoch: 3/3 Iter: 45/62 Batch time: 0.2259
Phase: train Epoch: 3/3 Iter: 46/62 Batch time: 0.2234
Phase: train Epoch: 3/3 Iter: 47/62 Batch time: 0.2272
Phase: train Epoch: 3/3 Iter: 48/62 Batch time: 0.2223
Phase: train Epoch: 3/3 Iter: 49/62 Batch time: 0.2307
Phase: train Epoch: 3/3 Iter: 50/62 Batch time: 0.2232
Phase: train Epoch: 3/3 Iter: 51/62 Batch time: 0.2246
Phase: train Epoch: 3/3 Iter: 52/62 Batch time: 0.2243
Phase: train Epoch: 3/3 Iter: 53/62 Batch time: 0.2245
Phase: train Epoch: 3/3 Iter: 54/62 Batch time: 0.2230
Phase: train Epoch: 3/3 Iter: 55/62 Batch time: 0.2251
Phase: train Epoch: 3/3 Iter: 56/62 Batch time: 0.2228
Phase: train Epoch: 3/3 Iter: 57/62 Batch time: 0.2290
Phase: train Epoch: 3/3 Iter: 58/62 Batch time: 0.2266
Phase: train Epoch: 3/3 Iter: 59/62 Batch time: 0.2401
Phase: train Epoch: 3/3 Iter: 60/62 Batch time: 0.2268
Phase: train Epoch: 3/3 Iter: 61/62 Batch time: 0.2289
Phase: train Epoch: 3/3 Loss: 0.5652 Acc: 0.7336
Phase: validation Epoch: 3/3 Iter: 1/39 Batch time: 0.1738
Phase: validation Epoch: 3/3 Iter: 2/39 Batch time: 0.1732
Phase: validation Epoch: 3/3 Iter: 3/39 Batch time: 0.1715
Phase: validation Epoch: 3/3 Iter: 4/39 Batch time: 0.1702
Phase: validation Epoch: 3/3 Iter: 5/39 Batch time: 0.1705
Phase: validation Epoch: 3/3 Iter: 6/39 Batch time: 0.1691
Phase: validation Epoch: 3/3 Iter: 7/39 Batch time: 0.1710
Phase: validation Epoch: 3/3 Iter: 8/39 Batch time: 0.1704
Phase: validation Epoch: 3/3 Iter: 9/39 Batch time: 0.1713
Phase: validation Epoch: 3/3 Iter: 10/39 Batch time: 0.1697
Phase: validation Epoch: 3/3 Iter: 11/39 Batch time: 0.1697
Phase: validation Epoch: 3/3 Iter: 12/39 Batch time: 0.1705
Phase: validation Epoch: 3/3 Iter: 13/39 Batch time: 0.1704
Phase: validation Epoch: 3/3 Iter: 14/39 Batch time: 0.1690
Phase: validation Epoch: 3/3 Iter: 15/39 Batch time: 0.1707
Phase: validation Epoch: 3/3 Iter: 16/39 Batch time: 0.1704
Phase: validation Epoch: 3/3 Iter: 17/39 Batch time: 0.1728
Phase: validation Epoch: 3/3 Iter: 18/39 Batch time: 0.1721
Phase: validation Epoch: 3/3 Iter: 19/39 Batch time: 0.1717
Phase: validation Epoch: 3/3 Iter: 20/39 Batch time: 0.1763
Phase: validation Epoch: 3/3 Iter: 21/39 Batch time: 0.1737
Phase: validation Epoch: 3/3 Iter: 22/39 Batch time: 0.1697
Phase: validation Epoch: 3/3 Iter: 23/39 Batch time: 0.1704
Phase: validation Epoch: 3/3 Iter: 24/39 Batch time: 0.1696
Phase: validation Epoch: 3/3 Iter: 25/39 Batch time: 0.1696
Phase: validation Epoch: 3/3 Iter: 26/39 Batch time: 0.1704
Phase: validation Epoch: 3/3 Iter: 27/39 Batch time: 0.1713
Phase: validation Epoch: 3/3 Iter: 28/39 Batch time: 0.1705
Phase: validation Epoch: 3/3 Iter: 29/39 Batch time: 0.1747
Phase: validation Epoch: 3/3 Iter: 30/39 Batch time: 0.1711
Phase: validation Epoch: 3/3 Iter: 31/39 Batch time: 0.1737
Phase: validation Epoch: 3/3 Iter: 32/39 Batch time: 0.1705
Phase: validation Epoch: 3/3 Iter: 33/39 Batch time: 0.1709
Phase: validation Epoch: 3/3 Iter: 34/39 Batch time: 0.1705
Phase: validation Epoch: 3/3 Iter: 35/39 Batch time: 0.1749
Phase: validation Epoch: 3/3 Iter: 36/39 Batch time: 0.1710
Phase: validation Epoch: 3/3 Iter: 37/39 Batch time: 0.1712
Phase: validation Epoch: 3/3 Iter: 38/39 Batch time: 0.1702
Phase: validation Epoch: 3/3 Iter: 39/39 Batch time: 0.0544
Phase: validation   Epoch: 3/3 Loss: 0.4484 Acc: 0.8497
Training completed in 1m 10s
Best test loss: 0.4484 | Best test accuracy: 0.8497
 </code>
 </pre>
 </details>

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_quantum_transfer_learning.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
  7%|7         | 3.20M/44.7M [00:00<00:01, 33.5MB/s]
 49%|####8     | 21.8M/44.7M [00:00<00:00, 128MB/s]
 93%|#########2| 41.5M/44.7M [00:00<00:00, 164MB/s]
100%|##########| 44.7M/44.7M [00:00<00:00, 147MB/s]
Training started:
Phase: train Epoch: 1/3 Iter: 1/62 Batch time: 0.3048
Phase: train Epoch: 1/3 Iter: 2/62 Batch time: 0.2678
Phase: train Epoch: 1/3 Iter: 3/62 Batch time: 0.2643
Phase: train Epoch: 1/3 Iter: 4/62 Batch time: 0.2655
Phase: train Epoch: 1/3 Iter: 5/62 Batch time: 0.2551
Phase: train Epoch: 1/3 Iter: 6/62 Batch time: 0.2718
Phase: train Epoch: 1/3 Iter: 7/62 Batch time: 0.2679
Phase: train Epoch: 1/3 Iter: 8/62 Batch time: 0.2669
Phase: train Epoch: 1/3 Iter: 9/62 Batch time: 0.2562
Phase: train Epoch: 1/3 Iter: 10/62 Batch time: 0.2630
Phase: train Epoch: 1/3 Iter: 11/62 Batch time: 0.2728
Phase: train Epoch: 1/3 Iter: 12/62 Batch time: 0.2549
Phase: train Epoch: 1/3 Iter: 13/62 Batch time: 0.2612
Phase: train Epoch: 1/3 Iter: 14/62 Batch time: 0.2688
Phase: train Epoch: 1/3 Iter: 15/62 Batch time: 0.2642
Phase: train Epoch: 1/3 Iter: 16/62 Batch time: 0.2725
Phase: train Epoch: 1/3 Iter: 17/62 Batch time: 0.2790
Phase: train Epoch: 1/3 Iter: 18/62 Batch time: 0.2653
Phase: train Epoch: 1/3 Iter: 19/62 Batch time: 0.2544
Phase: train Epoch: 1/3 Iter: 20/62 Batch time: 0.2766
Phase: train Epoch: 1/3 Iter: 21/62 Batch time: 0.2632
Phase: train Epoch: 1/3 Iter: 22/62 Batch time: 0.2548
Phase: train Epoch: 1/3 Iter: 23/62 Batch time: 0.2529
Phase: train Epoch: 1/3 Iter: 24/62 Batch time: 0.2528
Phase: train Epoch: 1/3 Iter: 25/62 Batch time: 0.2632
Phase: train Epoch: 1/3 Iter: 26/62 Batch time: 0.2622
Phase: train Epoch: 1/3 Iter: 27/62 Batch time: 0.2608
Phase: train Epoch: 1/3 Iter: 28/62 Batch time: 0.2639
Phase: train Epoch: 1/3 Iter: 29/62 Batch time: 0.2666
Phase: train Epoch: 1/3 Iter: 30/62 Batch time: 0.2612
Phase: train Epoch: 1/3 Iter: 31/62 Batch time: 0.2550
Phase: train Epoch: 1/3 Iter: 32/62 Batch time: 0.2605
Phase: train Epoch: 1/3 Iter: 33/62 Batch time: 0.2653
Phase: train Epoch: 1/3 Iter: 34/62 Batch time: 0.2554
Phase: train Epoch: 1/3 Iter: 35/62 Batch time: 0.2794
Phase: train Epoch: 1/3 Iter: 36/62 Batch time: 0.2860
Phase: train Epoch: 1/3 Iter: 37/62 Batch time: 0.3043
Phase: train Epoch: 1/3 Iter: 38/62 Batch time: 0.2693
Phase: train Epoch: 1/3 Iter: 39/62 Batch time: 0.2611
Phase: train Epoch: 1/3 Iter: 40/62 Batch time: 0.2608
Phase: train Epoch: 1/3 Iter: 41/62 Batch time: 0.2565
Phase: train Epoch: 1/3 Iter: 42/62 Batch time: 0.2704
Phase: train Epoch: 1/3 Iter: 43/62 Batch time: 0.2717
Phase: train Epoch: 1/3 Iter: 44/62 Batch time: 0.2822
Phase: train Epoch: 1/3 Iter: 45/62 Batch time: 0.2628
Phase: train Epoch: 1/3 Iter: 46/62 Batch time: 0.2606
Phase: train Epoch: 1/3 Iter: 47/62 Batch time: 0.2631
Phase: train Epoch: 1/3 Iter: 48/62 Batch time: 0.2627
Phase: train Epoch: 1/3 Iter: 49/62 Batch time: 0.2579
Phase: train Epoch: 1/3 Iter: 50/62 Batch time: 0.2560
Phase: train Epoch: 1/3 Iter: 51/62 Batch time: 0.2564
Phase: train Epoch: 1/3 Iter: 52/62 Batch time: 0.2540
Phase: train Epoch: 1/3 Iter: 53/62 Batch time: 0.2536
Phase: train Epoch: 1/3 Iter: 54/62 Batch time: 0.2589
Phase: train Epoch: 1/3 Iter: 55/62 Batch time: 0.2600
Phase: train Epoch: 1/3 Iter: 56/62 Batch time: 0.2492
Phase: train Epoch: 1/3 Iter: 57/62 Batch time: 0.2569
Phase: train Epoch: 1/3 Iter: 58/62 Batch time: 0.2604
Phase: train Epoch: 1/3 Iter: 59/62 Batch time: 0.2597
Phase: train Epoch: 1/3 Iter: 60/62 Batch time: 0.2574
Phase: train Epoch: 1/3 Iter: 61/62 Batch time: 0.2662
Phase: train Epoch: 1/3 Loss: 0.6993 Acc: 0.5246
Phase: validation Epoch: 1/3 Iter: 1/39 Batch time: 0.2007
Phase: validation Epoch: 1/3 Iter: 2/39 Batch time: 0.2022
Phase: validation Epoch: 1/3 Iter: 3/39 Batch time: 0.2271
Phase: validation Epoch: 1/3 Iter: 4/39 Batch time: 0.2332
Phase: validation Epoch: 1/3 Iter: 5/39 Batch time: 0.2164
Phase: validation Epoch: 1/3 Iter: 6/39 Batch time: 0.1953
Phase: validation Epoch: 1/3 Iter: 7/39 Batch time: 0.1998
Phase: validation Epoch: 1/3 Iter: 8/39 Batch time: 0.2015
Phase: validation Epoch: 1/3 Iter: 9/39 Batch time: 0.2094
Phase: validation Epoch: 1/3 Iter: 10/39 Batch time: 0.1991
Phase: validation Epoch: 1/3 Iter: 11/39 Batch time: 0.1948
Phase: validation Epoch: 1/3 Iter: 12/39 Batch time: 0.2134
Phase: validation Epoch: 1/3 Iter: 13/39 Batch time: 0.2063
Phase: validation Epoch: 1/3 Iter: 14/39 Batch time: 0.2150
Phase: validation Epoch: 1/3 Iter: 15/39 Batch time: 0.1964
Phase: validation Epoch: 1/3 Iter: 16/39 Batch time: 0.2059
Phase: validation Epoch: 1/3 Iter: 17/39 Batch time: 0.2146
Phase: validation Epoch: 1/3 Iter: 18/39 Batch time: 0.2241
Phase: validation Epoch: 1/3 Iter: 19/39 Batch time: 0.1971
Phase: validation Epoch: 1/3 Iter: 20/39 Batch time: 0.2289
Phase: validation Epoch: 1/3 Iter: 21/39 Batch time: 0.2027
Phase: validation Epoch: 1/3 Iter: 22/39 Batch time: 0.1960
Phase: validation Epoch: 1/3 Iter: 23/39 Batch time: 0.2043
Phase: validation Epoch: 1/3 Iter: 24/39 Batch time: 0.2096
Phase: validation Epoch: 1/3 Iter: 25/39 Batch time: 0.2018
Phase: validation Epoch: 1/3 Iter: 26/39 Batch time: 0.2054
Phase: validation Epoch: 1/3 Iter: 27/39 Batch time: 0.2031
Phase: validation Epoch: 1/3 Iter: 28/39 Batch time: 0.2156
Phase: validation Epoch: 1/3 Iter: 29/39 Batch time: 0.2080
Phase: validation Epoch: 1/3 Iter: 30/39 Batch time: 0.2175
Phase: validation Epoch: 1/3 Iter: 31/39 Batch time: 0.2283
Phase: validation Epoch: 1/3 Iter: 32/39 Batch time: 0.2215
Phase: validation Epoch: 1/3 Iter: 33/39 Batch time: 0.2138
Phase: validation Epoch: 1/3 Iter: 34/39 Batch time: 0.2043
Phase: validation Epoch: 1/3 Iter: 35/39 Batch time: 0.2051
Phase: validation Epoch: 1/3 Iter: 36/39 Batch time: 0.2008
Phase: validation Epoch: 1/3 Iter: 37/39 Batch time: 0.1997
Phase: validation Epoch: 1/3 Iter: 38/39 Batch time: 0.1938
Phase: validation Epoch: 1/3 Iter: 39/39 Batch time: 0.0662
Phase: validation   Epoch: 1/3 Loss: 0.6432 Acc: 0.6536
Phase: train Epoch: 2/3 Iter: 1/62 Batch time: 0.2569
Phase: train Epoch: 2/3 Iter: 2/62 Batch time: 0.2708
Phase: train Epoch: 2/3 Iter: 3/62 Batch time: 0.2655
Phase: train Epoch: 2/3 Iter: 4/62 Batch time: 0.2675
Phase: train Epoch: 2/3 Iter: 5/62 Batch time: 0.2649
Phase: train Epoch: 2/3 Iter: 6/62 Batch time: 0.2571
Phase: train Epoch: 2/3 Iter: 7/62 Batch time: 0.2576
Phase: train Epoch: 2/3 Iter: 8/62 Batch time: 0.2686
Phase: train Epoch: 2/3 Iter: 9/62 Batch time: 0.2645
Phase: train Epoch: 2/3 Iter: 10/62 Batch time: 0.2601
Phase: train Epoch: 2/3 Iter: 11/62 Batch time: 0.2670
Phase: train Epoch: 2/3 Iter: 12/62 Batch time: 0.2652
Phase: train Epoch: 2/3 Iter: 13/62 Batch time: 0.2666
Phase: train Epoch: 2/3 Iter: 14/62 Batch time: 0.2647
Phase: train Epoch: 2/3 Iter: 15/62 Batch time: 0.2725
Phase: train Epoch: 2/3 Iter: 16/62 Batch time: 0.2907
Phase: train Epoch: 2/3 Iter: 17/62 Batch time: 0.2863
Phase: train Epoch: 2/3 Iter: 18/62 Batch time: 0.2689
Phase: train Epoch: 2/3 Iter: 19/62 Batch time: 0.2633
Phase: train Epoch: 2/3 Iter: 20/62 Batch time: 0.2579
Phase: train Epoch: 2/3 Iter: 21/62 Batch time: 0.2576
Phase: train Epoch: 2/3 Iter: 22/62 Batch time: 0.2620
Phase: train Epoch: 2/3 Iter: 23/62 Batch time: 0.2667
Phase: train Epoch: 2/3 Iter: 24/62 Batch time: 0.2864
Phase: train Epoch: 2/3 Iter: 25/62 Batch time: 0.2684
Phase: train Epoch: 2/3 Iter: 26/62 Batch time: 0.3107
Phase: train Epoch: 2/3 Iter: 27/62 Batch time: 0.2709
Phase: train Epoch: 2/3 Iter: 28/62 Batch time: 0.2722
Phase: train Epoch: 2/3 Iter: 29/62 Batch time: 0.2737
Phase: train Epoch: 2/3 Iter: 30/62 Batch time: 0.2694
Phase: train Epoch: 2/3 Iter: 31/62 Batch time: 0.2693
Phase: train Epoch: 2/3 Iter: 32/62 Batch time: 0.2667
Phase: train Epoch: 2/3 Iter: 33/62 Batch time: 0.2667
Phase: train Epoch: 2/3 Iter: 34/62 Batch time: 0.2619
Phase: train Epoch: 2/3 Iter: 35/62 Batch time: 0.2621
Phase: train Epoch: 2/3 Iter: 36/62 Batch time: 0.2819
Phase: train Epoch: 2/3 Iter: 37/62 Batch time: 0.2599
Phase: train Epoch: 2/3 Iter: 38/62 Batch time: 0.2768
Phase: train Epoch: 2/3 Iter: 39/62 Batch time: 0.2862
Phase: train Epoch: 2/3 Iter: 40/62 Batch time: 0.2526
Phase: train Epoch: 2/3 Iter: 41/62 Batch time: 0.2616
Phase: train Epoch: 2/3 Iter: 42/62 Batch time: 0.2607
Phase: train Epoch: 2/3 Iter: 43/62 Batch time: 0.2922
Phase: train Epoch: 2/3 Iter: 44/62 Batch time: 0.2565
Phase: train Epoch: 2/3 Iter: 45/62 Batch time: 0.2836
Phase: train Epoch: 2/3 Iter: 46/62 Batch time: 0.2683
Phase: train Epoch: 2/3 Iter: 47/62 Batch time: 0.2580
Phase: train Epoch: 2/3 Iter: 48/62 Batch time: 0.2895
Phase: train Epoch: 2/3 Iter: 49/62 Batch time: 0.2624
Phase: train Epoch: 2/3 Iter: 50/62 Batch time: 0.2662
Phase: train Epoch: 2/3 Iter: 51/62 Batch time: 0.2779
Phase: train Epoch: 2/3 Iter: 52/62 Batch time: 0.2770
Phase: train Epoch: 2/3 Iter: 53/62 Batch time: 0.2707
Phase: train Epoch: 2/3 Iter: 54/62 Batch time: 0.2575
Phase: train Epoch: 2/3 Iter: 55/62 Batch time: 0.2869
Phase: train Epoch: 2/3 Iter: 56/62 Batch time: 0.2612
Phase: train Epoch: 2/3 Iter: 57/62 Batch time: 0.2657
Phase: train Epoch: 2/3 Iter: 58/62 Batch time: 0.2654
Phase: train Epoch: 2/3 Iter: 59/62 Batch time: 0.2566
Phase: train Epoch: 2/3 Iter: 60/62 Batch time: 0.2628
Phase: train Epoch: 2/3 Iter: 61/62 Batch time: 0.2705
Phase: train Epoch: 2/3 Loss: 0.6141 Acc: 0.7049
Phase: validation Epoch: 2/3 Iter: 1/39 Batch time: 0.1978
Phase: validation Epoch: 2/3 Iter: 2/39 Batch time: 0.2066
Phase: validation Epoch: 2/3 Iter: 3/39 Batch time: 0.1988
Phase: validation Epoch: 2/3 Iter: 4/39 Batch time: 0.1982
Phase: validation Epoch: 2/3 Iter: 5/39 Batch time: 0.2040
Phase: validation Epoch: 2/3 Iter: 6/39 Batch time: 0.2131
Phase: validation Epoch: 2/3 Iter: 7/39 Batch time: 0.2051
Phase: validation Epoch: 2/3 Iter: 8/39 Batch time: 0.1972
Phase: validation Epoch: 2/3 Iter: 9/39 Batch time: 0.1983
Phase: validation Epoch: 2/3 Iter: 10/39 Batch time: 0.2077
Phase: validation Epoch: 2/3 Iter: 11/39 Batch time: 0.2102
Phase: validation Epoch: 2/3 Iter: 12/39 Batch time: 0.1952
Phase: validation Epoch: 2/3 Iter: 13/39 Batch time: 0.2015
Phase: validation Epoch: 2/3 Iter: 14/39 Batch time: 0.2198
Phase: validation Epoch: 2/3 Iter: 15/39 Batch time: 0.1980
Phase: validation Epoch: 2/3 Iter: 16/39 Batch time: 0.1962
Phase: validation Epoch: 2/3 Iter: 17/39 Batch time: 0.1987
Phase: validation Epoch: 2/3 Iter: 18/39 Batch time: 0.1961
Phase: validation Epoch: 2/3 Iter: 19/39 Batch time: 0.1963
Phase: validation Epoch: 2/3 Iter: 20/39 Batch time: 0.1936
Phase: validation Epoch: 2/3 Iter: 21/39 Batch time: 0.1951
Phase: validation Epoch: 2/3 Iter: 22/39 Batch time: 0.1956
Phase: validation Epoch: 2/3 Iter: 23/39 Batch time: 0.1933
Phase: validation Epoch: 2/3 Iter: 24/39 Batch time: 0.1941
Phase: validation Epoch: 2/3 Iter: 25/39 Batch time: 0.1962
Phase: validation Epoch: 2/3 Iter: 26/39 Batch time: 0.1902
Phase: validation Epoch: 2/3 Iter: 27/39 Batch time: 0.1911
Phase: validation Epoch: 2/3 Iter: 28/39 Batch time: 0.1909
Phase: validation Epoch: 2/3 Iter: 29/39 Batch time: 0.2004
Phase: validation Epoch: 2/3 Iter: 30/39 Batch time: 0.1999
Phase: validation Epoch: 2/3 Iter: 31/39 Batch time: 0.1992
Phase: validation Epoch: 2/3 Iter: 32/39 Batch time: 0.1961
Phase: validation Epoch: 2/3 Iter: 33/39 Batch time: 0.1960
Phase: validation Epoch: 2/3 Iter: 34/39 Batch time: 0.2015
Phase: validation Epoch: 2/3 Iter: 35/39 Batch time: 0.2018
Phase: validation Epoch: 2/3 Iter: 36/39 Batch time: 0.1976
Phase: validation Epoch: 2/3 Iter: 37/39 Batch time: 0.2013
Phase: validation Epoch: 2/3 Iter: 38/39 Batch time: 0.1997
Phase: validation Epoch: 2/3 Iter: 39/39 Batch time: 0.0581
Phase: validation   Epoch: 2/3 Loss: 0.5392 Acc: 0.8235
Phase: train Epoch: 3/3 Iter: 1/62 Batch time: 0.2507
Phase: train Epoch: 3/3 Iter: 2/62 Batch time: 0.2533
Phase: train Epoch: 3/3 Iter: 3/62 Batch time: 0.2518
Phase: train Epoch: 3/3 Iter: 4/62 Batch time: 0.2592
Phase: train Epoch: 3/3 Iter: 5/62 Batch time: 0.2538
Phase: train Epoch: 3/3 Iter: 6/62 Batch time: 0.2523
Phase: train Epoch: 3/3 Iter: 7/62 Batch time: 0.2682
Phase: train Epoch: 3/3 Iter: 8/62 Batch time: 0.2533
Phase: train Epoch: 3/3 Iter: 9/62 Batch time: 0.2514
Phase: train Epoch: 3/3 Iter: 10/62 Batch time: 0.2578
Phase: train Epoch: 3/3 Iter: 11/62 Batch time: 0.2741
Phase: train Epoch: 3/3 Iter: 12/62 Batch time: 0.2627
Phase: train Epoch: 3/3 Iter: 13/62 Batch time: 0.2667
Phase: train Epoch: 3/3 Iter: 14/62 Batch time: 0.2698
Phase: train Epoch: 3/3 Iter: 15/62 Batch time: 0.2570
Phase: train Epoch: 3/3 Iter: 16/62 Batch time: 0.2567
Phase: train Epoch: 3/3 Iter: 17/62 Batch time: 0.2606
Phase: train Epoch: 3/3 Iter: 18/62 Batch time: 0.2497
Phase: train Epoch: 3/3 Iter: 19/62 Batch time: 0.2587
Phase: train Epoch: 3/3 Iter: 20/62 Batch time: 0.2632
Phase: train Epoch: 3/3 Iter: 21/62 Batch time: 0.2569
Phase: train Epoch: 3/3 Iter: 22/62 Batch time: 0.2541
Phase: train Epoch: 3/3 Iter: 23/62 Batch time: 0.2522
Phase: train Epoch: 3/3 Iter: 24/62 Batch time: 0.2687
Phase: train Epoch: 3/3 Iter: 25/62 Batch time: 0.2584
Phase: train Epoch: 3/3 Iter: 26/62 Batch time: 0.2578
Phase: train Epoch: 3/3 Iter: 27/62 Batch time: 0.2561
Phase: train Epoch: 3/3 Iter: 28/62 Batch time: 0.2576
Phase: train Epoch: 3/3 Iter: 29/62 Batch time: 0.2598
Phase: train Epoch: 3/3 Iter: 30/62 Batch time: 0.2563
Phase: train Epoch: 3/3 Iter: 31/62 Batch time: 0.2748
Phase: train Epoch: 3/3 Iter: 32/62 Batch time: 0.2718
Phase: train Epoch: 3/3 Iter: 33/62 Batch time: 0.2826
Phase: train Epoch: 3/3 Iter: 34/62 Batch time: 0.3205
Phase: train Epoch: 3/3 Iter: 35/62 Batch time: 0.2579
Phase: train Epoch: 3/3 Iter: 36/62 Batch time: 0.2612
Phase: train Epoch: 3/3 Iter: 37/62 Batch time: 0.2598
Phase: train Epoch: 3/3 Iter: 38/62 Batch time: 0.2632
Phase: train Epoch: 3/3 Iter: 39/62 Batch time: 0.2646
Phase: train Epoch: 3/3 Iter: 40/62 Batch time: 0.3235
Phase: train Epoch: 3/3 Iter: 41/62 Batch time: 0.2907
Phase: train Epoch: 3/3 Iter: 42/62 Batch time: 0.2612
Phase: train Epoch: 3/3 Iter: 43/62 Batch time: 0.2571
Phase: train Epoch: 3/3 Iter: 44/62 Batch time: 0.2502
Phase: train Epoch: 3/3 Iter: 45/62 Batch time: 0.2734
Phase: train Epoch: 3/3 Iter: 46/62 Batch time: 0.2890
Phase: train Epoch: 3/3 Iter: 47/62 Batch time: 0.2579
Phase: train Epoch: 3/3 Iter: 48/62 Batch time: 0.2910
Phase: train Epoch: 3/3 Iter: 49/62 Batch time: 0.2554
Phase: train Epoch: 3/3 Iter: 50/62 Batch time: 0.3090
Phase: train Epoch: 3/3 Iter: 51/62 Batch time: 0.2636
Phase: train Epoch: 3/3 Iter: 52/62 Batch time: 0.2549
Phase: train Epoch: 3/3 Iter: 53/62 Batch time: 0.2686
Phase: train Epoch: 3/3 Iter: 54/62 Batch time: 0.2623
Phase: train Epoch: 3/3 Iter: 55/62 Batch time: 0.2549
Phase: train Epoch: 3/3 Iter: 56/62 Batch time: 0.2528
Phase: train Epoch: 3/3 Iter: 57/62 Batch time: 0.2652
Phase: train Epoch: 3/3 Iter: 58/62 Batch time: 0.2601
Phase: train Epoch: 3/3 Iter: 59/62 Batch time: 0.2595
Phase: train Epoch: 3/3 Iter: 60/62 Batch time: 0.2549
Phase: train Epoch: 3/3 Iter: 61/62 Batch time: 0.2613
Phase: train Epoch: 3/3 Loss: 0.5652 Acc: 0.7336
Phase: validation Epoch: 3/3 Iter: 1/39 Batch time: 0.2074
Phase: validation Epoch: 3/3 Iter: 2/39 Batch time: 0.1949
Phase: validation Epoch: 3/3 Iter: 3/39 Batch time: 0.2041
Phase: validation Epoch: 3/3 Iter: 4/39 Batch time: 0.2240
Phase: validation Epoch: 3/3 Iter: 5/39 Batch time: 0.2117
Phase: validation Epoch: 3/3 Iter: 6/39 Batch time: 0.2199
Phase: validation Epoch: 3/3 Iter: 7/39 Batch time: 0.2224
Phase: validation Epoch: 3/3 Iter: 8/39 Batch time: 0.2065
Phase: validation Epoch: 3/3 Iter: 9/39 Batch time: 0.1973
Phase: validation Epoch: 3/3 Iter: 10/39 Batch time: 0.1938
Phase: validation Epoch: 3/3 Iter: 11/39 Batch time: 0.2015
Phase: validation Epoch: 3/3 Iter: 12/39 Batch time: 0.2138
Phase: validation Epoch: 3/3 Iter: 13/39 Batch time: 0.2196
Phase: validation Epoch: 3/3 Iter: 14/39 Batch time: 0.2140
Phase: validation Epoch: 3/3 Iter: 15/39 Batch time: 0.2222
Phase: validation Epoch: 3/3 Iter: 16/39 Batch time: 0.2083
Phase: validation Epoch: 3/3 Iter: 17/39 Batch time: 0.2048
Phase: validation Epoch: 3/3 Iter: 18/39 Batch time: 0.1924
Phase: validation Epoch: 3/3 Iter: 19/39 Batch time: 0.1950
Phase: validation Epoch: 3/3 Iter: 20/39 Batch time: 0.2106
Phase: validation Epoch: 3/3 Iter: 21/39 Batch time: 0.2114
Phase: validation Epoch: 3/3 Iter: 22/39 Batch time: 0.2028
Phase: validation Epoch: 3/3 Iter: 23/39 Batch time: 0.2078
Phase: validation Epoch: 3/3 Iter: 24/39 Batch time: 0.2034
Phase: validation Epoch: 3/3 Iter: 25/39 Batch time: 0.1971
Phase: validation Epoch: 3/3 Iter: 26/39 Batch time: 0.1972
Phase: validation Epoch: 3/3 Iter: 27/39 Batch time: 0.2262
Phase: validation Epoch: 3/3 Iter: 28/39 Batch time: 0.2129
Phase: validation Epoch: 3/3 Iter: 29/39 Batch time: 0.1935
Phase: validation Epoch: 3/3 Iter: 30/39 Batch time: 0.1983
Phase: validation Epoch: 3/3 Iter: 31/39 Batch time: 0.2102
Phase: validation Epoch: 3/3 Iter: 32/39 Batch time: 0.2004
Phase: validation Epoch: 3/3 Iter: 33/39 Batch time: 0.2120
Phase: validation Epoch: 3/3 Iter: 34/39 Batch time: 0.1968
Phase: validation Epoch: 3/3 Iter: 35/39 Batch time: 0.1938
Phase: validation Epoch: 3/3 Iter: 36/39 Batch time: 0.2026
Phase: validation Epoch: 3/3 Iter: 37/39 Batch time: 0.1993
Phase: validation Epoch: 3/3 Iter: 38/39 Batch time: 0.1982
Phase: validation Epoch: 3/3 Iter: 39/39 Batch time: 0.0595
Phase: validation   Epoch: 3/3 Loss: 0.4484 Acc: 0.8497
Training completed in 1m 22s
 </code>
 </pre>
 </details>

---

## 17. tutorial_vqe_qng.html <a name="demo16"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_vqe_qng.html):

```
Iteration = 0,  Energy = -0.09424332 Ha
Iteration = 20,  Energy = -0.55156842 Ha
Accuracy with respect to the FCI energy: 0.00002547 Ha (0.01598216 kcal/mol)
 5.09234513e-08 4.05827240e+00 2.74944154e+00 6.07360302e+00
Iteration = 0,  Energy = -0.32164518 Ha
Iteration = 8,  Energy = -0.85091055 Ha
Accuracy with respect to the FCI energy: 0.00000008 Ha (0.00004854 kcal/mol)
 4.03252161e-04 4.05827240e+00 2.74944154e+00 6.07375181e+00
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_vqe_qng.html):

```
Iteration = 0,  Energy = -0.09424333 Ha
Iteration = 20,  Energy = -0.55156841 Ha
Accuracy with respect to the FCI energy: 0.00002547 Ha (0.01598212 kcal/mol)
 5.09234601e-08 4.05827240e+00 2.74944154e+00 6.07360302e+00
Iteration = 0,  Energy = -0.32164519 Ha
Iteration = 8,  Energy = -0.85091050 Ha
Accuracy with respect to the FCI energy: 0.00000008 Ha (0.00004850 kcal/mol)
 4.03252278e-04 4.05827240e+00 2.74944154e+00 6.07375181e+00
```

---

## 18. tutorial_quanvolution.html <a name="demo17"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_quanvolution.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
   16384/11490434 [..............................] - ETA: 3s
   73728/11490434 [..............................] - ETA: 8s
  286720/11490434 [..............................] - ETA: 4s
 1081344/11490434 [=>............................] - ETA: 1s
 4177920/11490434 [=========>....................] - ETA: 0s
10174464/11490434 [=========================>....] - ETA: 0s
11493376/11490434 [==============================] - 0s 0us/step

11501568/11490434 [==============================] - 0s 0us/step
Quantum pre-processing of train images:
1/50
2/50
3/50
4/50
5/50
6/50
7/50
8/50
9/50
10/50
11/50
12/50
13/50
14/50
15/50
16/50
17/50
18/50
19/50
20/50
21/50
22/50
23/50
24/50
25/50
26/50
27/50
28/50
29/50
30/50
31/50
32/50
33/50
34/50
35/50
36/50
37/50
38/50
39/50
40/50
41/50
42/50
43/50
44/50
45/50
46/50
47/50
48/50
49/50
50/50
Quantum pre-processing of test images:
1/30
2/30
3/30
4/30
5/30
6/30
7/30
8/30
9/30
10/30
11/30
12/30
13/30
14/30
15/30
16/30
17/30
18/30
19/30
20/30
21/30
22/30
23/30
24/30
25/30
26/30
27/30
28/30
29/30
30/30
Epoch 1/30
13/13 - 0s - loss: 3.0160 - accuracy: 0.1000 - val_loss: 2.0646 - val_accuracy: 0.2000 - 444ms/epoch - 34ms/step
Epoch 2/30
13/13 - 0s - loss: 2.2510 - accuracy: 0.1800 - val_loss: 1.9801 - val_accuracy: 0.3333 - 36ms/epoch - 3ms/step
Epoch 3/30
13/13 - 0s - loss: 1.7851 - accuracy: 0.4000 - val_loss: 1.8177 - val_accuracy: 0.2667 - 48ms/epoch - 4ms/step
Epoch 4/30
13/13 - 0s - loss: 1.3652 - accuracy: 0.5400 - val_loss: 1.6107 - val_accuracy: 0.4667 - 47ms/epoch - 4ms/step
Epoch 5/30
13/13 - 0s - loss: 1.1317 - accuracy: 0.7800 - val_loss: 1.4723 - val_accuracy: 0.6000 - 48ms/epoch - 4ms/step
Epoch 6/30
13/13 - 0s - loss: 0.9360 - accuracy: 0.8600 - val_loss: 1.4686 - val_accuracy: 0.5333 - 50ms/epoch - 4ms/step
Epoch 7/30
13/13 - 0s - loss: 0.7383 - accuracy: 0.9400 - val_loss: 1.3536 - val_accuracy: 0.5667 - 34ms/epoch - 3ms/step
Epoch 8/30
13/13 - 0s - loss: 0.5846 - accuracy: 0.9800 - val_loss: 1.2785 - val_accuracy: 0.6667 - 46ms/epoch - 4ms/step
Epoch 9/30
13/13 - 0s - loss: 0.4987 - accuracy: 0.9800 - val_loss: 1.2253 - val_accuracy: 0.6333 - 35ms/epoch - 3ms/step
Epoch 10/30
13/13 - 0s - loss: 0.3921 - accuracy: 1.0000 - val_loss: 1.2655 - val_accuracy: 0.6333 - 48ms/epoch - 4ms/step
Epoch 11/30
13/13 - 0s - loss: 0.3617 - accuracy: 1.0000 - val_loss: 1.1555 - val_accuracy: 0.7000 - 34ms/epoch - 3ms/step
Epoch 12/30
13/13 - 0s - loss: 0.3078 - accuracy: 1.0000 - val_loss: 1.2107 - val_accuracy: 0.6667 - 47ms/epoch - 4ms/step
Epoch 13/30
13/13 - 0s - loss: 0.2618 - accuracy: 1.0000 - val_loss: 1.1166 - val_accuracy: 0.7333 - 48ms/epoch - 4ms/step
Epoch 14/30
13/13 - 0s - loss: 0.2463 - accuracy: 1.0000 - val_loss: 1.0624 - val_accuracy: 0.7000 - 34ms/epoch - 3ms/step
Epoch 15/30
13/13 - 0s - loss: 0.2033 - accuracy: 1.0000 - val_loss: 1.0904 - val_accuracy: 0.7333 - 33ms/epoch - 3ms/step
Epoch 16/30
13/13 - 0s - loss: 0.1799 - accuracy: 1.0000 - val_loss: 1.0865 - val_accuracy: 0.7333 - 47ms/epoch - 4ms/step
Epoch 17/30
13/13 - 0s - loss: 0.1682 - accuracy: 1.0000 - val_loss: 1.0385 - val_accuracy: 0.7333 - 47ms/epoch - 4ms/step
Epoch 18/30
13/13 - 0s - loss: 0.1484 - accuracy: 1.0000 - val_loss: 1.0676 - val_accuracy: 0.7000 - 33ms/epoch - 3ms/step
Epoch 19/30
13/13 - 0s - loss: 0.1349 - accuracy: 1.0000 - val_loss: 1.0447 - val_accuracy: 0.7000 - 47ms/epoch - 4ms/step
Epoch 20/30
13/13 - 0s - loss: 0.1255 - accuracy: 1.0000 - val_loss: 0.9935 - val_accuracy: 0.7333 - 50ms/epoch - 4ms/step
Epoch 21/30
13/13 - 0s - loss: 0.1135 - accuracy: 1.0000 - val_loss: 1.0451 - val_accuracy: 0.7333 - 35ms/epoch - 3ms/step
Epoch 22/30
13/13 - 0s - loss: 0.1041 - accuracy: 1.0000 - val_loss: 1.0142 - val_accuracy: 0.7333 - 49ms/epoch - 4ms/step
Epoch 23/30
13/13 - 0s - loss: 0.0983 - accuracy: 1.0000 - val_loss: 0.9893 - val_accuracy: 0.7333 - 48ms/epoch - 4ms/step
Epoch 24/30
13/13 - 0s - loss: 0.0913 - accuracy: 1.0000 - val_loss: 0.9807 - val_accuracy: 0.7000 - 34ms/epoch - 3ms/step
Epoch 25/30
13/13 - 0s - loss: 0.0868 - accuracy: 1.0000 - val_loss: 0.9715 - val_accuracy: 0.7333 - 48ms/epoch - 4ms/step
Epoch 26/30
13/13 - 0s - loss: 0.0788 - accuracy: 1.0000 - val_loss: 0.9850 - val_accuracy: 0.7333 - 34ms/epoch - 3ms/step
Epoch 27/30
13/13 - 0s - loss: 0.0749 - accuracy: 1.0000 - val_loss: 0.9750 - val_accuracy: 0.7333 - 47ms/epoch - 4ms/step
Epoch 28/30
13/13 - 0s - loss: 0.0730 - accuracy: 1.0000 - val_loss: 0.9570 - val_accuracy: 0.7667 - 47ms/epoch - 4ms/step
Epoch 29/30
13/13 - 0s - loss: 0.0681 - accuracy: 1.0000 - val_loss: 0.9895 - val_accuracy: 0.7333 - 33ms/epoch - 3ms/step
Epoch 30/30
13/13 - 0s - loss: 0.0635 - accuracy: 1.0000 - val_loss: 0.9560 - val_accuracy: 0.7333 - 47ms/epoch - 4ms/step
Epoch 1/30
13/13 - 0s - loss: 2.3619 - accuracy: 0.1400 - val_loss: 2.0567 - val_accuracy: 0.3667 - 379ms/epoch - 29ms/step
Epoch 2/30
13/13 - 0s - loss: 1.9696 - accuracy: 0.4200 - val_loss: 1.9381 - val_accuracy: 0.4667 - 35ms/epoch - 3ms/step
Epoch 3/30
13/13 - 0s - loss: 1.6671 - accuracy: 0.6400 - val_loss: 1.8300 - val_accuracy: 0.4333 - 36ms/epoch - 3ms/step
Epoch 4/30
13/13 - 0s - loss: 1.4340 - accuracy: 0.7400 - val_loss: 1.7113 - val_accuracy: 0.4333 - 49ms/epoch - 4ms/step
Epoch 5/30
13/13 - 0s - loss: 1.2342 - accuracy: 0.7600 - val_loss: 1.6044 - val_accuracy: 0.5000 - 34ms/epoch - 3ms/step
Epoch 6/30
13/13 - 0s - loss: 1.0721 - accuracy: 0.8600 - val_loss: 1.5232 - val_accuracy: 0.5333 - 47ms/epoch - 4ms/step
Epoch 7/30
13/13 - 0s - loss: 0.9348 - accuracy: 0.9000 - val_loss: 1.4596 - val_accuracy: 0.5667 - 33ms/epoch - 3ms/step
Epoch 8/30
13/13 - 0s - loss: 0.8178 - accuracy: 0.9200 - val_loss: 1.3921 - val_accuracy: 0.6000 - 47ms/epoch - 4ms/step
Epoch 9/30
13/13 - 0s - loss: 0.7223 - accuracy: 0.9400 - val_loss: 1.3404 - val_accuracy: 0.6333 - 34ms/epoch - 3ms/step
Epoch 10/30
13/13 - 0s - loss: 0.6404 - accuracy: 0.9600 - val_loss: 1.3065 - val_accuracy: 0.6667 - 48ms/epoch - 4ms/step
Epoch 11/30
13/13 - 0s - loss: 0.5772 - accuracy: 1.0000 - val_loss: 1.2644 - val_accuracy: 0.6333 - 34ms/epoch - 3ms/step
Epoch 12/30
13/13 - 0s - loss: 0.5199 - accuracy: 1.0000 - val_loss: 1.2558 - val_accuracy: 0.6667 - 33ms/epoch - 3ms/step
Epoch 13/30
13/13 - 0s - loss: 0.4695 - accuracy: 1.0000 - val_loss: 1.2258 - val_accuracy: 0.6667 - 33ms/epoch - 3ms/step
Epoch 14/30
13/13 - 0s - loss: 0.4238 - accuracy: 1.0000 - val_loss: 1.1897 - val_accuracy: 0.6667 - 47ms/epoch - 4ms/step
Epoch 15/30
13/13 - 0s - loss: 0.3848 - accuracy: 1.0000 - val_loss: 1.1651 - val_accuracy: 0.6667 - 48ms/epoch - 4ms/step
Epoch 16/30
13/13 - 0s - loss: 0.3525 - accuracy: 1.0000 - val_loss: 1.1503 - val_accuracy: 0.7333 - 48ms/epoch - 4ms/step
Epoch 17/30
13/13 - 0s - loss: 0.3245 - accuracy: 1.0000 - val_loss: 1.1374 - val_accuracy: 0.7000 - 48ms/epoch - 4ms/step
Epoch 18/30
13/13 - 0s - loss: 0.2992 - accuracy: 1.0000 - val_loss: 1.1174 - val_accuracy: 0.6667 - 49ms/epoch - 4ms/step
Epoch 19/30
13/13 - 0s - loss: 0.2745 - accuracy: 1.0000 - val_loss: 1.1119 - val_accuracy: 0.6667 - 35ms/epoch - 3ms/step
Epoch 20/30
13/13 - 0s - loss: 0.2551 - accuracy: 1.0000 - val_loss: 1.0903 - val_accuracy: 0.7000 - 33ms/epoch - 3ms/step
Epoch 21/30
13/13 - 0s - loss: 0.2370 - accuracy: 1.0000 - val_loss: 1.0877 - val_accuracy: 0.6667 - 46ms/epoch - 4ms/step
Epoch 22/30
13/13 - 0s - loss: 0.2199 - accuracy: 1.0000 - val_loss: 1.0776 - val_accuracy: 0.6667 - 48ms/epoch - 4ms/step
Epoch 23/30
13/13 - 0s - loss: 0.2050 - accuracy: 1.0000 - val_loss: 1.0675 - val_accuracy: 0.7000 - 33ms/epoch - 3ms/step
Epoch 24/30
13/13 - 0s - loss: 0.1919 - accuracy: 1.0000 - val_loss: 1.0592 - val_accuracy: 0.7000 - 48ms/epoch - 4ms/step
Epoch 25/30
13/13 - 0s - loss: 0.1812 - accuracy: 1.0000 - val_loss: 1.0554 - val_accuracy: 0.6667 - 35ms/epoch - 3ms/step
Epoch 26/30
13/13 - 0s - loss: 0.1691 - accuracy: 1.0000 - val_loss: 1.0477 - val_accuracy: 0.7000 - 49ms/epoch - 4ms/step
Epoch 27/30
13/13 - 0s - loss: 0.1599 - accuracy: 1.0000 - val_loss: 1.0377 - val_accuracy: 0.7000 - 50ms/epoch - 4ms/step
Epoch 28/30
13/13 - 0s - loss: 0.1515 - accuracy: 1.0000 - val_loss: 1.0341 - val_accuracy: 0.6667 - 48ms/epoch - 4ms/step
 </code>
 </pre>
 </details>

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_quanvolution.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
   16384/11490434 [..............................] - ETA: 0s
 5382144/11490434 [=============>................] - ETA: 0s
11493376/11490434 [==============================] - 0s 0us/step

11501568/11490434 [==============================] - 0s 0us/step
Quantum pre-processing of train images:
1/50
2/50
3/50
4/50
5/50
6/50
7/50
8/50
9/50
10/50
11/50
12/50
13/50
14/50
15/50
16/50
17/50
18/50
19/50
20/50
21/50
22/50
23/50
24/50
25/50
26/50
27/50
28/50
29/50
30/50
31/50
32/50
33/50
34/50
35/50
36/50
37/50
38/50
39/50
40/50
41/50
42/50
43/50
44/50
45/50
46/50
47/50
48/50
49/50
50/50
Quantum pre-processing of test images:
1/30
2/30
3/30
4/30
5/30
6/30
7/30
8/30
9/30
10/30
11/30
12/30
13/30
14/30
15/30
16/30
17/30
18/30
19/30
20/30
21/30
22/30
23/30
24/30
25/30
26/30
27/30
28/30
29/30
30/30
Epoch 1/30
13/13 - 1s - loss: 3.0160 - accuracy: 0.1000 - val_loss: 2.0646 - val_accuracy: 0.2000 - 654ms/epoch - 50ms/step
Epoch 2/30
13/13 - 0s - loss: 2.2510 - accuracy: 0.1800 - val_loss: 1.9801 - val_accuracy: 0.3333 - 43ms/epoch - 3ms/step
Epoch 3/30
13/13 - 0s - loss: 1.7851 - accuracy: 0.4000 - val_loss: 1.8177 - val_accuracy: 0.2667 - 54ms/epoch - 4ms/step
Epoch 4/30
13/13 - 0s - loss: 1.3652 - accuracy: 0.5400 - val_loss: 1.6107 - val_accuracy: 0.4667 - 43ms/epoch - 3ms/step
Epoch 5/30
13/13 - 0s - loss: 1.1317 - accuracy: 0.7800 - val_loss: 1.4723 - val_accuracy: 0.6000 - 42ms/epoch - 3ms/step
Epoch 6/30
13/13 - 0s - loss: 0.9360 - accuracy: 0.8600 - val_loss: 1.4686 - val_accuracy: 0.5333 - 52ms/epoch - 4ms/step
Epoch 7/30
13/13 - 0s - loss: 0.7383 - accuracy: 0.9400 - val_loss: 1.3536 - val_accuracy: 0.5667 - 39ms/epoch - 3ms/step
Epoch 8/30
13/13 - 0s - loss: 0.5846 - accuracy: 0.9800 - val_loss: 1.2785 - val_accuracy: 0.6667 - 53ms/epoch - 4ms/step
Epoch 9/30
13/13 - 0s - loss: 0.4987 - accuracy: 0.9800 - val_loss: 1.2253 - val_accuracy: 0.6333 - 57ms/epoch - 4ms/step
Epoch 10/30
13/13 - 0s - loss: 0.3921 - accuracy: 1.0000 - val_loss: 1.2655 - val_accuracy: 0.6333 - 41ms/epoch - 3ms/step
Epoch 11/30
13/13 - 0s - loss: 0.3617 - accuracy: 1.0000 - val_loss: 1.1555 - val_accuracy: 0.7000 - 52ms/epoch - 4ms/step
Epoch 12/30
13/13 - 0s - loss: 0.3078 - accuracy: 1.0000 - val_loss: 1.2107 - val_accuracy: 0.6667 - 54ms/epoch - 4ms/step
Epoch 13/30
13/13 - 0s - loss: 0.2618 - accuracy: 1.0000 - val_loss: 1.1166 - val_accuracy: 0.7333 - 40ms/epoch - 3ms/step
Epoch 14/30
13/13 - 0s - loss: 0.2463 - accuracy: 1.0000 - val_loss: 1.0624 - val_accuracy: 0.7000 - 55ms/epoch - 4ms/step
Epoch 15/30
13/13 - 0s - loss: 0.2033 - accuracy: 1.0000 - val_loss: 1.0904 - val_accuracy: 0.7333 - 38ms/epoch - 3ms/step
Epoch 16/30
13/13 - 0s - loss: 0.1799 - accuracy: 1.0000 - val_loss: 1.0865 - val_accuracy: 0.7333 - 38ms/epoch - 3ms/step
Epoch 17/30
13/13 - 0s - loss: 0.1682 - accuracy: 1.0000 - val_loss: 1.0385 - val_accuracy: 0.7333 - 38ms/epoch - 3ms/step
Epoch 18/30
13/13 - 0s - loss: 0.1484 - accuracy: 1.0000 - val_loss: 1.0676 - val_accuracy: 0.7000 - 51ms/epoch - 4ms/step
Epoch 19/30
13/13 - 0s - loss: 0.1349 - accuracy: 1.0000 - val_loss: 1.0447 - val_accuracy: 0.7000 - 39ms/epoch - 3ms/step
Epoch 20/30
13/13 - 0s - loss: 0.1255 - accuracy: 1.0000 - val_loss: 0.9935 - val_accuracy: 0.7333 - 52ms/epoch - 4ms/step
Epoch 21/30
13/13 - 0s - loss: 0.1135 - accuracy: 1.0000 - val_loss: 1.0451 - val_accuracy: 0.7333 - 52ms/epoch - 4ms/step
Epoch 22/30
13/13 - 0s - loss: 0.1041 - accuracy: 1.0000 - val_loss: 1.0142 - val_accuracy: 0.7333 - 39ms/epoch - 3ms/step
Epoch 23/30
13/13 - 0s - loss: 0.0983 - accuracy: 1.0000 - val_loss: 0.9893 - val_accuracy: 0.7333 - 42ms/epoch - 3ms/step
Epoch 24/30
13/13 - 0s - loss: 0.0913 - accuracy: 1.0000 - val_loss: 0.9807 - val_accuracy: 0.7000 - 53ms/epoch - 4ms/step
Epoch 25/30
13/13 - 0s - loss: 0.0868 - accuracy: 1.0000 - val_loss: 0.9715 - val_accuracy: 0.7333 - 44ms/epoch - 3ms/step
Epoch 26/30
13/13 - 0s - loss: 0.0788 - accuracy: 1.0000 - val_loss: 0.9850 - val_accuracy: 0.7333 - 41ms/epoch - 3ms/step
Epoch 27/30
13/13 - 0s - loss: 0.0749 - accuracy: 1.0000 - val_loss: 0.9750 - val_accuracy: 0.7333 - 41ms/epoch - 3ms/step
Epoch 28/30
13/13 - 0s - loss: 0.0730 - accuracy: 1.0000 - val_loss: 0.9570 - val_accuracy: 0.7667 - 40ms/epoch - 3ms/step
Epoch 29/30
13/13 - 0s - loss: 0.0681 - accuracy: 1.0000 - val_loss: 0.9895 - val_accuracy: 0.7333 - 64ms/epoch - 5ms/step
Epoch 30/30
13/13 - 0s - loss: 0.0635 - accuracy: 1.0000 - val_loss: 0.9560 - val_accuracy: 0.7333 - 53ms/epoch - 4ms/step
Epoch 1/30
13/13 - 1s - loss: 2.3619 - accuracy: 0.1400 - val_loss: 2.0567 - val_accuracy: 0.3667 - 577ms/epoch - 44ms/step
Epoch 2/30
13/13 - 0s - loss: 1.9696 - accuracy: 0.4200 - val_loss: 1.9381 - val_accuracy: 0.4667 - 39ms/epoch - 3ms/step
Epoch 3/30
13/13 - 0s - loss: 1.6671 - accuracy: 0.6400 - val_loss: 1.8300 - val_accuracy: 0.4333 - 67ms/epoch - 5ms/step
Epoch 4/30
13/13 - 0s - loss: 1.4340 - accuracy: 0.7400 - val_loss: 1.7113 - val_accuracy: 0.4333 - 55ms/epoch - 4ms/step
Epoch 5/30
13/13 - 0s - loss: 1.2342 - accuracy: 0.7600 - val_loss: 1.6044 - val_accuracy: 0.5000 - 54ms/epoch - 4ms/step
Epoch 6/30
13/13 - 0s - loss: 1.0721 - accuracy: 0.8600 - val_loss: 1.5232 - val_accuracy: 0.5333 - 54ms/epoch - 4ms/step
Epoch 7/30
13/13 - 0s - loss: 0.9348 - accuracy: 0.9000 - val_loss: 1.4596 - val_accuracy: 0.5667 - 58ms/epoch - 4ms/step
Epoch 8/30
13/13 - 0s - loss: 0.8178 - accuracy: 0.9200 - val_loss: 1.3921 - val_accuracy: 0.6000 - 58ms/epoch - 4ms/step
Epoch 9/30
13/13 - 0s - loss: 0.7223 - accuracy: 0.9400 - val_loss: 1.3404 - val_accuracy: 0.6333 - 49ms/epoch - 4ms/step
Epoch 10/30
13/13 - 0s - loss: 0.6404 - accuracy: 0.9600 - val_loss: 1.3065 - val_accuracy: 0.6667 - 56ms/epoch - 4ms/step
Epoch 11/30
13/13 - 0s - loss: 0.5772 - accuracy: 1.0000 - val_loss: 1.2644 - val_accuracy: 0.6333 - 53ms/epoch - 4ms/step
Epoch 12/30
13/13 - 0s - loss: 0.5199 - accuracy: 1.0000 - val_loss: 1.2558 - val_accuracy: 0.6667 - 53ms/epoch - 4ms/step
Epoch 13/30
13/13 - 0s - loss: 0.4695 - accuracy: 1.0000 - val_loss: 1.2258 - val_accuracy: 0.6667 - 54ms/epoch - 4ms/step
Epoch 14/30
13/13 - 0s - loss: 0.4238 - accuracy: 1.0000 - val_loss: 1.1897 - val_accuracy: 0.6667 - 45ms/epoch - 3ms/step
Epoch 15/30
13/13 - 0s - loss: 0.3848 - accuracy: 1.0000 - val_loss: 1.1651 - val_accuracy: 0.6667 - 53ms/epoch - 4ms/step
Epoch 16/30
13/13 - 0s - loss: 0.3525 - accuracy: 1.0000 - val_loss: 1.1503 - val_accuracy: 0.7333 - 48ms/epoch - 4ms/step
Epoch 17/30
13/13 - 0s - loss: 0.3245 - accuracy: 1.0000 - val_loss: 1.1374 - val_accuracy: 0.7000 - 55ms/epoch - 4ms/step
Epoch 18/30
13/13 - 0s - loss: 0.2992 - accuracy: 1.0000 - val_loss: 1.1174 - val_accuracy: 0.6667 - 45ms/epoch - 3ms/step
Epoch 19/30
13/13 - 0s - loss: 0.2745 - accuracy: 1.0000 - val_loss: 1.1119 - val_accuracy: 0.6667 - 42ms/epoch - 3ms/step
Epoch 20/30
13/13 - 0s - loss: 0.2551 - accuracy: 1.0000 - val_loss: 1.0903 - val_accuracy: 0.7000 - 54ms/epoch - 4ms/step
Epoch 21/30
13/13 - 0s - loss: 0.2370 - accuracy: 1.0000 - val_loss: 1.0877 - val_accuracy: 0.6667 - 62ms/epoch - 5ms/step
Epoch 22/30
13/13 - 0s - loss: 0.2199 - accuracy: 1.0000 - val_loss: 1.0776 - val_accuracy: 0.6667 - 45ms/epoch - 3ms/step
Epoch 23/30
13/13 - 0s - loss: 0.2050 - accuracy: 1.0000 - val_loss: 1.0675 - val_accuracy: 0.7000 - 61ms/epoch - 5ms/step
Epoch 24/30
13/13 - 0s - loss: 0.1919 - accuracy: 1.0000 - val_loss: 1.0592 - val_accuracy: 0.7000 - 44ms/epoch - 3ms/step
Epoch 25/30
13/13 - 0s - loss: 0.1812 - accuracy: 1.0000 - val_loss: 1.0554 - val_accuracy: 0.6667 - 52ms/epoch - 4ms/step
Epoch 26/30
13/13 - 0s - loss: 0.1691 - accuracy: 1.0000 - val_loss: 1.0477 - val_accuracy: 0.7000 - 54ms/epoch - 4ms/step
Epoch 27/30
13/13 - 0s - loss: 0.1599 - accuracy: 1.0000 - val_loss: 1.0377 - val_accuracy: 0.7000 - 60ms/epoch - 5ms/step
Epoch 28/30
13/13 - 0s - loss: 0.1515 - accuracy: 1.0000 - val_loss: 1.0341 - val_accuracy: 0.6667 - 42ms/epoch - 3ms/step
Epoch 29/30
13/13 - 0s - loss: 0.1426 - accuracy: 1.0000 - val_loss: 1.0291 - val_accuracy: 0.7000 - 39ms/epoch - 3ms/step
Epoch 30/30
13/13 - 0s - loss: 0.1344 - accuracy: 1.0000 - val_loss: 1.0264 - val_accuracy: 0.7000 - 53ms/epoch - 4ms/step
 </code>
 </pre>
 </details>

---

