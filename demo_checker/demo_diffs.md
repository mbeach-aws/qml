Last update: 2022-11-06  00:30:38 (All times shown in Eastern time)
# List of differences in demonstration outputs

# Table of contents

1. [tutorial_qnn_module_tf.html](#demo0)
2. [tutorial_quantum_circuit_cutting.html](#demo1)
3. [tutorial_adaptive_circuits.html](#demo2)
4. [tutorial_jax_transformations.html](#demo3)
5. [tutorial_tn_circuits.html](#demo4)
6. [tutorial_quantum_transfer_learning.html](#demo5)
7. [tutorial_noisy_circuit_optimization.html](#demo6)
8. [tutorial_quanvolution.html](#demo7)
9. [tutorial_quantum_chemistry.html](#demo8)
10. [tutorial_error_mitigation.html](#demo9)
11. [tutorial_backprop.html](#demo10)
12. [tutorial_local_cost_functions.html](#demo11)
13. [tutorial_measurement_optimize.html](#demo12)


Number of demos different/all demos: 13/71

## 1. tutorial_qnn_module_tf.html <a name="demo0"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_qnn_module_tf.html):

```
30/30 - 11s - loss: 0.4997 - accuracy: 0.5000 - val_loss: 0.5081 - val_accuracy: 0.4400 - 11s/epoch - 370ms/step
30/30 - 11s - loss: 0.4673 - accuracy: 0.6200 - val_loss: 0.4488 - val_accuracy: 0.6200 - 11s/epoch - 366ms/step
30/30 - 11s - loss: 0.3230 - accuracy: 0.8267 - val_loss: 0.2562 - val_accuracy: 0.8400 - 11s/epoch - 376ms/step
30/30 - 11s - loss: 0.2124 - accuracy: 0.8867 - val_loss: 0.1997 - val_accuracy: 0.8400 - 11s/epoch - 370ms/step
30/30 - 11s - loss: 0.1800 - accuracy: 0.8933 - val_loss: 0.1841 - val_accuracy: 0.8400 - 11s/epoch - 369ms/step
30/30 - 11s - loss: 0.1593 - accuracy: 0.8667 - val_loss: 0.2177 - val_accuracy: 0.8400 - 11s/epoch - 372ms/step
30/30 - 22s - loss: 0.5189 - accuracy: 0.4000 - val_loss: 0.4945 - val_accuracy: 0.5400 - 22s/epoch - 744ms/step
30/30 - 22s - loss: 0.4822 - accuracy: 0.6200 - val_loss: 0.4412 - val_accuracy: 0.7200 - 22s/epoch - 739ms/step
30/30 - 22s - loss: 0.3850 - accuracy: 0.7133 - val_loss: 0.2898 - val_accuracy: 0.7800 - 22s/epoch - 739ms/step
30/30 - 22s - loss: 0.2720 - accuracy: 0.7867 - val_loss: 0.2185 - val_accuracy: 0.8200 - 22s/epoch - 735ms/step
30/30 - 22s - loss: 0.2056 - accuracy: 0.8400 - val_loss: 0.1893 - val_accuracy: 0.8400 - 22s/epoch - 728ms/step
30/30 - 22s - loss: 0.1753 - accuracy: 0.8400 - val_loss: 0.1856 - val_accuracy: 0.8400 - 22s/epoch - 738ms/step
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_qnn_module_tf.html):

```
30/30 - 9s - loss: 0.4997 - accuracy: 0.5000 - val_loss: 0.5081 - val_accuracy: 0.4400 - 9s/epoch - 310ms/step
30/30 - 9s - loss: 0.4673 - accuracy: 0.6200 - val_loss: 0.4488 - val_accuracy: 0.6200 - 9s/epoch - 307ms/step
30/30 - 9s - loss: 0.3230 - accuracy: 0.8267 - val_loss: 0.2562 - val_accuracy: 0.8400 - 9s/epoch - 313ms/step
30/30 - 9s - loss: 0.2124 - accuracy: 0.8867 - val_loss: 0.1997 - val_accuracy: 0.8400 - 9s/epoch - 309ms/step
30/30 - 9s - loss: 0.1800 - accuracy: 0.8933 - val_loss: 0.1841 - val_accuracy: 0.8400 - 9s/epoch - 309ms/step
30/30 - 9s - loss: 0.1593 - accuracy: 0.8667 - val_loss: 0.2177 - val_accuracy: 0.8400 - 9s/epoch - 309ms/step
30/30 - 19s - loss: 0.5189 - accuracy: 0.4000 - val_loss: 0.4945 - val_accuracy: 0.5400 - 19s/epoch - 617ms/step
30/30 - 18s - loss: 0.4822 - accuracy: 0.6200 - val_loss: 0.4412 - val_accuracy: 0.7200 - 18s/epoch - 616ms/step
30/30 - 18s - loss: 0.3850 - accuracy: 0.7133 - val_loss: 0.2898 - val_accuracy: 0.7800 - 18s/epoch - 614ms/step
30/30 - 19s - loss: 0.2720 - accuracy: 0.7867 - val_loss: 0.2185 - val_accuracy: 0.8200 - 19s/epoch - 621ms/step
30/30 - 18s - loss: 0.2056 - accuracy: 0.8400 - val_loss: 0.1893 - val_accuracy: 0.8400 - 18s/epoch - 609ms/step
30/30 - 18s - loss: 0.1753 - accuracy: 0.8400 - val_loss: 0.1856 - val_accuracy: 0.8400 - 18s/epoch - 615ms/step
```

---

## 2. tutorial_quantum_circuit_cutting.html <a name="demo1"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_quantum_circuit_cutting.html):

```
0.47165198882111165
0.47165198882111165
Cut size: k=3
Channel probabilities: p0=0.53; p1=0.47
Which channel to run: [1 0 1 ... 0 1 0]
Channel 0: 5305 times
Channel 1: 4695 times.
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_quantum_circuit_cutting.html):

```
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The animation.avconv_args rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The animation.avconv_path rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The animation.html_args rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
```

---

## 3. tutorial_adaptive_circuits.html <a name="demo2"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_adaptive_circuits.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
n = 0,  E = -7.86266588 H, t = 0.77 s
n = 1,  E = -7.87094622 H, t = 0.77 s
n = 2,  E = -7.87563101 H, t = 1.02 s
n = 3,  E = -7.87829148 H, t = 0.77 s
n = 4,  E = -7.87981707 H, t = 0.77 s
n = 5,  E = -7.88070478 H, t = 0.77 s
n = 6,  E = -7.88123144 H, t = 0.77 s
n = 7,  E = -7.88155162 H, t = 1.03 s
n = 8,  E = -7.88175219 H, t = 0.77 s
n = 9,  E = -7.88188238 H, t = 0.77 s
n = 10,  E = -7.88197042 H, t = 0.77 s
n = 11,  E = -7.88203269 H, t = 0.77 s
n = 12,  E = -7.88207881 H, t = 1.02 s
n = 13,  E = -7.88211453 H, t = 0.77 s
n = 14,  E = -7.88214336 H, t = 0.77 s
n = 15,  E = -7.88216745 H, t = 0.77 s
n = 16,  E = -7.88218815 H, t = 1.01 s
n = 17,  E = -7.88220635 H, t = 0.77 s
n = 18,  E = -7.88222262 H, t = 0.77 s
n = 19,  E = -7.88223735 H, t = 0.77 s
n = 0,  E = -7.86266588 H, t = 0.15 s
n = 1,  E = -7.87094622 H, t = 0.15 s
n = 2,  E = -7.87563101 H, t = 0.15 s
n = 3,  E = -7.87829148 H, t = 0.15 s
n = 4,  E = -7.87981707 H, t = 0.15 s
n = 5,  E = -7.88070478 H, t = 0.15 s
n = 6,  E = -7.88123144 H, t = 0.15 s
n = 7,  E = -7.88155162 H, t = 0.15 s
n = 10,  E = -7.88197042 H, t = 0.15 s
n = 12,  E = -7.88207881 H, t = 0.15 s
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
n = 0,  E = -7.86266588 H, t = 0.74 s
n = 1,  E = -7.87094622 H, t = 0.74 s
n = 2,  E = -7.87563101 H, t = 0.92 s
n = 3,  E = -7.87829148 H, t = 0.74 s
n = 4,  E = -7.87981707 H, t = 0.74 s
n = 5,  E = -7.88070478 H, t = 0.74 s
n = 6,  E = -7.88123144 H, t = 0.74 s
n = 7,  E = -7.88155162 H, t = 0.91 s
n = 8,  E = -7.88175219 H, t = 0.74 s
n = 9,  E = -7.88188238 H, t = 0.74 s
n = 10,  E = -7.88197042 H, t = 0.74 s
n = 11,  E = -7.88203269 H, t = 0.74 s
n = 12,  E = -7.88207881 H, t = 0.91 s
n = 13,  E = -7.88211453 H, t = 0.74 s
n = 14,  E = -7.88214336 H, t = 0.74 s
n = 15,  E = -7.88216745 H, t = 0.74 s
n = 16,  E = -7.88218815 H, t = 0.92 s
n = 17,  E = -7.88220635 H, t = 0.74 s
n = 18,  E = -7.88222262 H, t = 0.74 s
n = 19,  E = -7.88223735 H, t = 0.74 s
n = 0,  E = -7.86266588 H, t = 0.14 s
n = 1,  E = -7.87094622 H, t = 0.14 s
n = 2,  E = -7.87563101 H, t = 0.14 s
n = 3,  E = -7.87829148 H, t = 0.14 s
n = 4,  E = -7.87981707 H, t = 0.14 s
n = 5,  E = -7.88070478 H, t = 0.14 s
n = 6,  E = -7.88123144 H, t = 0.14 s
n = 7,  E = -7.88155162 H, t = 0.14 s
n = 10,  E = -7.88197042 H, t = 0.14 s
n = 12,  E = -7.88207881 H, t = 0.14 s
 </code>
 </pre>
 </details>

---

## 4. tutorial_jax_transformations.html <a name="demo3"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_jax_transformations.html):

```
No jit time: 0.0059 seconds
First run time: 0.0879 seconds
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_jax_transformations.html):

```
No jit time: 0.0056 seconds
First run time: 0.0820 seconds
```

---

## 5. tutorial_tn_circuits.html <a name="demo4"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_tn_circuits.html):

```
Step 0, cost: -0.01795177455549668
Step 20, cost: -2.961936236524546
Step 40, cost: -3.9999999973819755
Step 60, cost: -4.0
Step 80, cost: -4.000000000000001
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_tn_circuits.html):

```
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The animation.avconv_args rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The animation.avconv_path rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
```

---

## 6. tutorial_quantum_transfer_learning.html <a name="demo5"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_quantum_transfer_learning.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
 38%|###8      | 17.1M/44.7M [00:00<00:00, 180MB/s]
 94%|#########4| 42.1M/44.7M [00:00<00:00, 228MB/s]
100%|##########| 44.7M/44.7M [00:00<00:00, 224MB/s]
Training started:
Phase: train Epoch: 1/3 Iter: 1/62 Batch time: 0.3592
Phase: train Epoch: 1/3 Iter: 2/62 Batch time: 0.3026
Phase: train Epoch: 1/3 Iter: 3/62 Batch time: 0.2984
Phase: train Epoch: 1/3 Iter: 4/62 Batch time: 0.2987
Phase: train Epoch: 1/3 Iter: 5/62 Batch time: 0.2969
Phase: train Epoch: 1/3 Iter: 6/62 Batch time: 0.2997
Phase: train Epoch: 1/3 Iter: 7/62 Batch time: 0.2998
Phase: train Epoch: 1/3 Iter: 8/62 Batch time: 0.2997
Phase: train Epoch: 1/3 Iter: 9/62 Batch time: 0.2976
Phase: train Epoch: 1/3 Iter: 10/62 Batch time: 0.2988
Phase: train Epoch: 1/3 Iter: 11/62 Batch time: 0.2963
Phase: train Epoch: 1/3 Iter: 12/62 Batch time: 0.2994
Phase: train Epoch: 1/3 Iter: 13/62 Batch time: 0.2992
Phase: train Epoch: 1/3 Iter: 14/62 Batch time: 0.2992
Phase: train Epoch: 1/3 Iter: 15/62 Batch time: 0.2973
Phase: train Epoch: 1/3 Iter: 16/62 Batch time: 0.3000
Phase: train Epoch: 1/3 Iter: 17/62 Batch time: 0.2983
Phase: train Epoch: 1/3 Iter: 18/62 Batch time: 0.3017
Phase: train Epoch: 1/3 Iter: 19/62 Batch time: 0.2976
Phase: train Epoch: 1/3 Iter: 20/62 Batch time: 0.3017
Phase: train Epoch: 1/3 Iter: 21/62 Batch time: 0.2992
Phase: train Epoch: 1/3 Iter: 22/62 Batch time: 0.2999
Phase: train Epoch: 1/3 Iter: 23/62 Batch time: 0.3018
Phase: train Epoch: 1/3 Iter: 24/62 Batch time: 0.3003
Phase: train Epoch: 1/3 Iter: 25/62 Batch time: 0.2994
Phase: train Epoch: 1/3 Iter: 26/62 Batch time: 0.2971
Phase: train Epoch: 1/3 Iter: 27/62 Batch time: 0.3004
Phase: train Epoch: 1/3 Iter: 28/62 Batch time: 0.3043
Phase: train Epoch: 1/3 Iter: 29/62 Batch time: 0.3005
Phase: train Epoch: 1/3 Iter: 30/62 Batch time: 0.2993
Phase: train Epoch: 1/3 Iter: 31/62 Batch time: 0.2995
Phase: train Epoch: 1/3 Iter: 32/62 Batch time: 0.3009
Phase: train Epoch: 1/3 Iter: 33/62 Batch time: 0.3025
Phase: train Epoch: 1/3 Iter: 34/62 Batch time: 0.2977
Phase: train Epoch: 1/3 Iter: 35/62 Batch time: 0.2996
Phase: train Epoch: 1/3 Iter: 36/62 Batch time: 0.2999
Phase: train Epoch: 1/3 Iter: 37/62 Batch time: 0.3000
Phase: train Epoch: 1/3 Iter: 38/62 Batch time: 0.3025
Phase: train Epoch: 1/3 Iter: 39/62 Batch time: 0.3001
Phase: train Epoch: 1/3 Iter: 40/62 Batch time: 0.3007
Phase: train Epoch: 1/3 Iter: 41/62 Batch time: 0.2983
Phase: train Epoch: 1/3 Iter: 42/62 Batch time: 0.2973
Phase: train Epoch: 1/3 Iter: 43/62 Batch time: 0.3007
Phase: train Epoch: 1/3 Iter: 44/62 Batch time: 0.2994
Phase: train Epoch: 1/3 Iter: 45/62 Batch time: 0.2982
Phase: train Epoch: 1/3 Iter: 46/62 Batch time: 0.3004
Phase: train Epoch: 1/3 Iter: 47/62 Batch time: 0.2964
Phase: train Epoch: 1/3 Iter: 48/62 Batch time: 0.2971
Phase: train Epoch: 1/3 Iter: 49/62 Batch time: 0.2974
Phase: train Epoch: 1/3 Iter: 50/62 Batch time: 0.2976
Phase: train Epoch: 1/3 Iter: 51/62 Batch time: 0.2999
Phase: train Epoch: 1/3 Iter: 52/62 Batch time: 0.3001
Phase: train Epoch: 1/3 Iter: 53/62 Batch time: 0.3004
Phase: train Epoch: 1/3 Iter: 54/62 Batch time: 0.3003
Phase: train Epoch: 1/3 Iter: 55/62 Batch time: 0.2989
Phase: train Epoch: 1/3 Iter: 56/62 Batch time: 0.2975
Phase: train Epoch: 1/3 Iter: 57/62 Batch time: 0.2999
Phase: train Epoch: 1/3 Iter: 58/62 Batch time: 0.2979
Phase: train Epoch: 1/3 Iter: 59/62 Batch time: 0.2968
Phase: train Epoch: 1/3 Iter: 60/62 Batch time: 0.3001
Phase: train Epoch: 1/3 Iter: 61/62 Batch time: 0.2989
Phase: train Epoch: 1/3 Loss: 0.6990 Acc: 0.5246
Phase: validation Epoch: 1/3 Iter: 1/39 Batch time: 0.2258
Phase: validation Epoch: 1/3 Iter: 2/39 Batch time: 0.2245
Phase: validation Epoch: 1/3 Iter: 3/39 Batch time: 0.2247
Phase: validation Epoch: 1/3 Iter: 4/39 Batch time: 0.2226
Phase: validation Epoch: 1/3 Iter: 5/39 Batch time: 0.2239
Phase: validation Epoch: 1/3 Iter: 6/39 Batch time: 0.2223
Phase: validation Epoch: 1/3 Iter: 7/39 Batch time: 0.2225
Phase: validation Epoch: 1/3 Iter: 8/39 Batch time: 0.2227
Phase: validation Epoch: 1/3 Iter: 9/39 Batch time: 0.2220
Phase: validation Epoch: 1/3 Iter: 10/39 Batch time: 0.2222
Phase: validation Epoch: 1/3 Iter: 11/39 Batch time: 0.2238
Phase: validation Epoch: 1/3 Iter: 12/39 Batch time: 0.2257
Phase: validation Epoch: 1/3 Iter: 13/39 Batch time: 0.2213
Phase: validation Epoch: 1/3 Iter: 14/39 Batch time: 0.2229
Phase: validation Epoch: 1/3 Iter: 15/39 Batch time: 0.2236
Phase: validation Epoch: 1/3 Iter: 16/39 Batch time: 0.2237
Phase: validation Epoch: 1/3 Iter: 17/39 Batch time: 0.2223
Phase: validation Epoch: 1/3 Iter: 18/39 Batch time: 0.2231
Phase: validation Epoch: 1/3 Iter: 19/39 Batch time: 0.2236
Phase: validation Epoch: 1/3 Iter: 20/39 Batch time: 0.2231
Phase: validation Epoch: 1/3 Iter: 21/39 Batch time: 0.2232
Phase: validation Epoch: 1/3 Iter: 22/39 Batch time: 0.2237
Phase: validation Epoch: 1/3 Iter: 23/39 Batch time: 0.2242
Phase: validation Epoch: 1/3 Iter: 24/39 Batch time: 0.2241
Phase: validation Epoch: 1/3 Iter: 25/39 Batch time: 0.2214
Phase: validation Epoch: 1/3 Iter: 26/39 Batch time: 0.2230
Phase: validation Epoch: 1/3 Iter: 27/39 Batch time: 0.2234
Phase: validation Epoch: 1/3 Iter: 28/39 Batch time: 0.2223
Phase: validation Epoch: 1/3 Iter: 29/39 Batch time: 0.2226
Phase: validation Epoch: 1/3 Iter: 30/39 Batch time: 0.2248
Phase: validation Epoch: 1/3 Iter: 31/39 Batch time: 0.2234
Phase: validation Epoch: 1/3 Iter: 32/39 Batch time: 0.2234
Phase: validation Epoch: 1/3 Iter: 33/39 Batch time: 0.2251
Phase: validation Epoch: 1/3 Iter: 34/39 Batch time: 0.2214
Phase: validation Epoch: 1/3 Iter: 35/39 Batch time: 0.2215
Phase: validation Epoch: 1/3 Iter: 36/39 Batch time: 0.2234
Phase: validation Epoch: 1/3 Iter: 37/39 Batch time: 0.2237
Phase: validation Epoch: 1/3 Iter: 38/39 Batch time: 0.2253
Phase: validation Epoch: 1/3 Iter: 39/39 Batch time: 0.0742
Phase: validation   Epoch: 1/3 Loss: 0.6429 Acc: 0.6536
Phase: train Epoch: 2/3 Iter: 1/62 Batch time: 0.2947
Phase: train Epoch: 2/3 Iter: 2/62 Batch time: 0.2973
Phase: train Epoch: 2/3 Iter: 3/62 Batch time: 0.3009
Phase: train Epoch: 2/3 Iter: 4/62 Batch time: 0.3012
Phase: train Epoch: 2/3 Iter: 5/62 Batch time: 0.3003
Phase: train Epoch: 2/3 Iter: 6/62 Batch time: 0.2976
Phase: train Epoch: 2/3 Iter: 7/62 Batch time: 0.2992
Phase: train Epoch: 2/3 Iter: 8/62 Batch time: 0.2975
Phase: train Epoch: 2/3 Iter: 9/62 Batch time: 0.2992
Phase: train Epoch: 2/3 Iter: 10/62 Batch time: 0.2998
Phase: train Epoch: 2/3 Iter: 11/62 Batch time: 0.2982
Phase: train Epoch: 2/3 Iter: 12/62 Batch time: 0.3008
Phase: train Epoch: 2/3 Iter: 13/62 Batch time: 0.2990
Phase: train Epoch: 2/3 Iter: 14/62 Batch time: 0.3044
Phase: train Epoch: 2/3 Iter: 15/62 Batch time: 0.3004
Phase: train Epoch: 2/3 Iter: 16/62 Batch time: 0.2973
Phase: train Epoch: 2/3 Iter: 17/62 Batch time: 0.2992
Phase: train Epoch: 2/3 Iter: 18/62 Batch time: 0.2969
Phase: train Epoch: 2/3 Iter: 19/62 Batch time: 0.2990
Phase: train Epoch: 2/3 Iter: 20/62 Batch time: 0.2980
Phase: train Epoch: 2/3 Iter: 21/62 Batch time: 0.2977
Phase: train Epoch: 2/3 Iter: 22/62 Batch time: 0.2998
Phase: train Epoch: 2/3 Iter: 23/62 Batch time: 0.3008
Phase: train Epoch: 2/3 Iter: 24/62 Batch time: 0.3047
Phase: train Epoch: 2/3 Iter: 25/62 Batch time: 0.2978
Phase: train Epoch: 2/3 Iter: 26/62 Batch time: 0.2999
Phase: train Epoch: 2/3 Iter: 27/62 Batch time: 0.2968
Phase: train Epoch: 2/3 Iter: 28/62 Batch time: 0.2983
Phase: train Epoch: 2/3 Iter: 29/62 Batch time: 0.2988
Phase: train Epoch: 2/3 Iter: 30/62 Batch time: 0.3017
Phase: train Epoch: 2/3 Iter: 31/62 Batch time: 0.2984
Phase: train Epoch: 2/3 Iter: 32/62 Batch time: 0.2983
Phase: train Epoch: 2/3 Iter: 33/62 Batch time: 0.2983
Phase: train Epoch: 2/3 Iter: 34/62 Batch time: 0.3006
Phase: train Epoch: 2/3 Iter: 35/62 Batch time: 0.3020
Phase: train Epoch: 2/3 Iter: 36/62 Batch time: 0.3001
Phase: train Epoch: 2/3 Iter: 37/62 Batch time: 0.3001
Phase: train Epoch: 2/3 Iter: 38/62 Batch time: 0.3036
Phase: train Epoch: 2/3 Iter: 39/62 Batch time: 0.3049
Phase: train Epoch: 2/3 Iter: 40/62 Batch time: 0.3023
Phase: train Epoch: 2/3 Iter: 41/62 Batch time: 0.3027
Phase: train Epoch: 2/3 Iter: 42/62 Batch time: 0.3046
Phase: train Epoch: 2/3 Iter: 43/62 Batch time: 0.3031
Phase: train Epoch: 2/3 Iter: 44/62 Batch time: 0.3052
Phase: train Epoch: 2/3 Iter: 45/62 Batch time: 0.3020
Phase: train Epoch: 2/3 Iter: 46/62 Batch time: 0.2998
Phase: train Epoch: 2/3 Iter: 47/62 Batch time: 0.2987
Phase: train Epoch: 2/3 Iter: 48/62 Batch time: 0.3005
Phase: train Epoch: 2/3 Iter: 49/62 Batch time: 0.2996
Phase: train Epoch: 2/3 Iter: 50/62 Batch time: 0.3009
Phase: train Epoch: 2/3 Iter: 51/62 Batch time: 0.2988
Phase: train Epoch: 2/3 Iter: 52/62 Batch time: 0.2978
Phase: train Epoch: 2/3 Iter: 53/62 Batch time: 0.2994
Phase: train Epoch: 2/3 Iter: 54/62 Batch time: 0.3013
Phase: train Epoch: 2/3 Iter: 55/62 Batch time: 0.2991
Phase: train Epoch: 2/3 Iter: 56/62 Batch time: 0.2989
Phase: train Epoch: 2/3 Iter: 57/62 Batch time: 0.3003
Phase: train Epoch: 2/3 Iter: 58/62 Batch time: 0.3021
Phase: train Epoch: 2/3 Iter: 59/62 Batch time: 0.3012
Phase: train Epoch: 2/3 Iter: 60/62 Batch time: 0.3046
Phase: train Epoch: 2/3 Iter: 61/62 Batch time: 0.3023
Phase: train Epoch: 2/3 Loss: 0.6134 Acc: 0.7008
Phase: validation Epoch: 2/3 Iter: 1/39 Batch time: 0.2283
Phase: validation Epoch: 2/3 Iter: 2/39 Batch time: 0.2245
Phase: validation Epoch: 2/3 Iter: 3/39 Batch time: 0.2231
Phase: validation Epoch: 2/3 Iter: 4/39 Batch time: 0.2231
Phase: validation Epoch: 2/3 Iter: 5/39 Batch time: 0.2222
Phase: validation Epoch: 2/3 Iter: 6/39 Batch time: 0.2237
Phase: validation Epoch: 2/3 Iter: 7/39 Batch time: 0.2249
Phase: validation Epoch: 2/3 Iter: 8/39 Batch time: 0.2207
Phase: validation Epoch: 2/3 Iter: 9/39 Batch time: 0.2232
Phase: validation Epoch: 2/3 Iter: 10/39 Batch time: 0.2218
Phase: validation Epoch: 2/3 Iter: 11/39 Batch time: 0.2220
Phase: validation Epoch: 2/3 Iter: 12/39 Batch time: 0.2218
Phase: validation Epoch: 2/3 Iter: 13/39 Batch time: 0.2203
Phase: validation Epoch: 2/3 Iter: 14/39 Batch time: 0.2222
Phase: validation Epoch: 2/3 Iter: 15/39 Batch time: 0.2222
Phase: validation Epoch: 2/3 Iter: 16/39 Batch time: 0.2234
Phase: validation Epoch: 2/3 Iter: 17/39 Batch time: 0.2249
Phase: validation Epoch: 2/3 Iter: 18/39 Batch time: 0.2230
Phase: validation Epoch: 2/3 Iter: 19/39 Batch time: 0.2224
Phase: validation Epoch: 2/3 Iter: 20/39 Batch time: 0.2225
Phase: validation Epoch: 2/3 Iter: 21/39 Batch time: 0.2244
Phase: validation Epoch: 2/3 Iter: 22/39 Batch time: 0.2214
Phase: validation Epoch: 2/3 Iter: 23/39 Batch time: 0.2220
Phase: validation Epoch: 2/3 Iter: 24/39 Batch time: 0.2229
Phase: validation Epoch: 2/3 Iter: 25/39 Batch time: 0.2229
Phase: validation Epoch: 2/3 Iter: 26/39 Batch time: 0.2231
Phase: validation Epoch: 2/3 Iter: 27/39 Batch time: 0.2232
Phase: validation Epoch: 2/3 Iter: 28/39 Batch time: 0.2211
Phase: validation Epoch: 2/3 Iter: 29/39 Batch time: 0.2206
Phase: validation Epoch: 2/3 Iter: 30/39 Batch time: 0.2237
Phase: validation Epoch: 2/3 Iter: 31/39 Batch time: 0.2219
Phase: validation Epoch: 2/3 Iter: 32/39 Batch time: 0.2225
Phase: validation Epoch: 2/3 Iter: 33/39 Batch time: 0.2223
Phase: validation Epoch: 2/3 Iter: 34/39 Batch time: 0.2210
Phase: validation Epoch: 2/3 Iter: 35/39 Batch time: 0.2215
Phase: validation Epoch: 2/3 Iter: 36/39 Batch time: 0.2239
Phase: validation Epoch: 2/3 Iter: 37/39 Batch time: 0.2222
Phase: validation Epoch: 2/3 Iter: 38/39 Batch time: 0.2232
Phase: validation Epoch: 2/3 Iter: 39/39 Batch time: 0.0671
Phase: validation   Epoch: 2/3 Loss: 0.5389 Acc: 0.8235
Phase: train Epoch: 3/3 Iter: 1/62 Batch time: 0.2924
Phase: train Epoch: 3/3 Iter: 2/62 Batch time: 0.2975
Phase: train Epoch: 3/3 Iter: 3/62 Batch time: 0.3001
Phase: train Epoch: 3/3 Iter: 4/62 Batch time: 0.2997
Phase: train Epoch: 3/3 Iter: 5/62 Batch time: 0.3002
Phase: train Epoch: 3/3 Iter: 6/62 Batch time: 0.3005
Phase: train Epoch: 3/3 Iter: 7/62 Batch time: 0.3018
Phase: train Epoch: 3/3 Iter: 8/62 Batch time: 0.2979
Phase: train Epoch: 3/3 Iter: 9/62 Batch time: 0.3020
Phase: train Epoch: 3/3 Iter: 10/62 Batch time: 0.2987
Phase: train Epoch: 3/3 Iter: 11/62 Batch time: 0.2992
Phase: train Epoch: 3/3 Iter: 12/62 Batch time: 0.2999
Phase: train Epoch: 3/3 Iter: 13/62 Batch time: 0.3013
Phase: train Epoch: 3/3 Iter: 14/62 Batch time: 0.3021
Phase: train Epoch: 3/3 Iter: 15/62 Batch time: 0.2997
Phase: train Epoch: 3/3 Iter: 16/62 Batch time: 0.3000
Phase: train Epoch: 3/3 Iter: 17/62 Batch time: 0.3012
Phase: train Epoch: 3/3 Iter: 18/62 Batch time: 0.2986
Phase: train Epoch: 3/3 Iter: 19/62 Batch time: 0.3012
Phase: train Epoch: 3/3 Iter: 20/62 Batch time: 0.3009
Phase: train Epoch: 3/3 Iter: 21/62 Batch time: 0.3012
Phase: train Epoch: 3/3 Iter: 22/62 Batch time: 0.2998
Phase: train Epoch: 3/3 Iter: 23/62 Batch time: 0.3002
Phase: train Epoch: 3/3 Iter: 24/62 Batch time: 0.3010
Phase: train Epoch: 3/3 Iter: 25/62 Batch time: 0.2994
Phase: train Epoch: 3/3 Iter: 26/62 Batch time: 0.2998
Phase: train Epoch: 3/3 Iter: 27/62 Batch time: 0.3005
Phase: train Epoch: 3/3 Iter: 28/62 Batch time: 0.2987
Phase: train Epoch: 3/3 Iter: 29/62 Batch time: 0.3005
Phase: train Epoch: 3/3 Iter: 30/62 Batch time: 0.3007
Phase: train Epoch: 3/3 Iter: 31/62 Batch time: 0.3005
Phase: train Epoch: 3/3 Iter: 32/62 Batch time: 0.2978
Phase: train Epoch: 3/3 Iter: 33/62 Batch time: 0.2983
Phase: train Epoch: 3/3 Iter: 34/62 Batch time: 0.3009
Phase: train Epoch: 3/3 Iter: 35/62 Batch time: 0.2968
Phase: train Epoch: 3/3 Iter: 36/62 Batch time: 0.2982
Phase: train Epoch: 3/3 Iter: 37/62 Batch time: 0.2972
Phase: train Epoch: 3/3 Iter: 38/62 Batch time: 0.2999
Phase: train Epoch: 3/3 Iter: 39/62 Batch time: 0.3024
Phase: train Epoch: 3/3 Iter: 40/62 Batch time: 0.2993
Phase: train Epoch: 3/3 Iter: 41/62 Batch time: 0.3011
Phase: train Epoch: 3/3 Iter: 42/62 Batch time: 0.2982
Phase: train Epoch: 3/3 Iter: 43/62 Batch time: 0.2985
Phase: train Epoch: 3/3 Iter: 44/62 Batch time: 0.2993
Phase: train Epoch: 3/3 Iter: 45/62 Batch time: 0.2995
Phase: train Epoch: 3/3 Iter: 46/62 Batch time: 0.2993
Phase: train Epoch: 3/3 Iter: 47/62 Batch time: 0.2983
Phase: train Epoch: 3/3 Iter: 48/62 Batch time: 0.2993
Phase: train Epoch: 3/3 Iter: 49/62 Batch time: 0.2990
Phase: train Epoch: 3/3 Iter: 50/62 Batch time: 0.2979
Phase: train Epoch: 3/3 Iter: 51/62 Batch time: 0.2988
Phase: train Epoch: 3/3 Iter: 52/62 Batch time: 0.2999
Phase: train Epoch: 3/3 Iter: 53/62 Batch time: 0.3012
Phase: train Epoch: 3/3 Iter: 54/62 Batch time: 0.3013
Phase: train Epoch: 3/3 Iter: 55/62 Batch time: 0.2989
Phase: train Epoch: 3/3 Iter: 56/62 Batch time: 0.3039
Phase: train Epoch: 3/3 Iter: 57/62 Batch time: 0.3004
Phase: train Epoch: 3/3 Iter: 58/62 Batch time: 0.3004
Phase: train Epoch: 3/3 Iter: 59/62 Batch time: 0.2985
Phase: train Epoch: 3/3 Iter: 60/62 Batch time: 0.2981
Phase: train Epoch: 3/3 Iter: 61/62 Batch time: 0.2991
Phase: train Epoch: 3/3 Loss: 0.5652 Acc: 0.7418
Phase: validation Epoch: 3/3 Iter: 1/39 Batch time: 0.2272
Phase: validation Epoch: 3/3 Iter: 2/39 Batch time: 0.2229
Phase: validation Epoch: 3/3 Iter: 3/39 Batch time: 0.2252
Phase: validation Epoch: 3/3 Iter: 4/39 Batch time: 0.2240
Phase: validation Epoch: 3/3 Iter: 5/39 Batch time: 0.2234
Phase: validation Epoch: 3/3 Iter: 6/39 Batch time: 0.2227
Phase: validation Epoch: 3/3 Iter: 7/39 Batch time: 0.2224
Phase: validation Epoch: 3/3 Iter: 8/39 Batch time: 0.2222
Phase: validation Epoch: 3/3 Iter: 9/39 Batch time: 0.2253
Phase: validation Epoch: 3/3 Iter: 10/39 Batch time: 0.2240
Phase: validation Epoch: 3/3 Iter: 11/39 Batch time: 0.2236
Phase: validation Epoch: 3/3 Iter: 12/39 Batch time: 0.2216
Phase: validation Epoch: 3/3 Iter: 13/39 Batch time: 0.2217
Phase: validation Epoch: 3/3 Iter: 14/39 Batch time: 0.2230
Phase: validation Epoch: 3/3 Iter: 15/39 Batch time: 0.2230
Phase: validation Epoch: 3/3 Iter: 16/39 Batch time: 0.2209
Phase: validation Epoch: 3/3 Iter: 17/39 Batch time: 0.2239
Phase: validation Epoch: 3/3 Iter: 18/39 Batch time: 0.2252
Phase: validation Epoch: 3/3 Iter: 19/39 Batch time: 0.2226
Phase: validation Epoch: 3/3 Iter: 20/39 Batch time: 0.2219
Phase: validation Epoch: 3/3 Iter: 21/39 Batch time: 0.2226
Phase: validation Epoch: 3/3 Iter: 22/39 Batch time: 0.2241
Phase: validation Epoch: 3/3 Iter: 23/39 Batch time: 0.2230
Phase: validation Epoch: 3/3 Iter: 24/39 Batch time: 0.2234
Phase: validation Epoch: 3/3 Iter: 25/39 Batch time: 0.2224
Phase: validation Epoch: 3/3 Iter: 26/39 Batch time: 0.2226
Phase: validation Epoch: 3/3 Iter: 27/39 Batch time: 0.2225
Phase: validation Epoch: 3/3 Iter: 28/39 Batch time: 0.2235
Phase: validation Epoch: 3/3 Iter: 29/39 Batch time: 0.2225
Phase: validation Epoch: 3/3 Iter: 30/39 Batch time: 0.2268
Phase: validation Epoch: 3/3 Iter: 31/39 Batch time: 0.2229
Phase: validation Epoch: 3/3 Iter: 32/39 Batch time: 0.2240
Phase: validation Epoch: 3/3 Iter: 33/39 Batch time: 0.2238
Phase: validation Epoch: 3/3 Iter: 34/39 Batch time: 0.2229
Phase: validation Epoch: 3/3 Iter: 35/39 Batch time: 0.2234
Phase: validation Epoch: 3/3 Iter: 36/39 Batch time: 0.2227
Phase: validation Epoch: 3/3 Iter: 37/39 Batch time: 0.2213
Phase: validation Epoch: 3/3 Iter: 38/39 Batch time: 0.2224
Phase: validation Epoch: 3/3 Iter: 39/39 Batch time: 0.0679
Phase: validation   Epoch: 3/3 Loss: 0.4484 Acc: 0.8497
Training completed in 1m 27s
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
 43%|####3     | 19.4M/44.7M [00:00<00:00, 203MB/s]
100%|##########| 44.7M/44.7M [00:00<00:00, 243MB/s]
Training started:
Phase: train Epoch: 1/3 Iter: 1/62 Batch time: 0.4363
Phase: train Epoch: 1/3 Iter: 2/62 Batch time: 0.3632
Phase: train Epoch: 1/3 Iter: 3/62 Batch time: 0.3612
Phase: train Epoch: 1/3 Iter: 4/62 Batch time: 0.3630
Phase: train Epoch: 1/3 Iter: 5/62 Batch time: 0.3679
Phase: train Epoch: 1/3 Iter: 6/62 Batch time: 0.3652
Phase: train Epoch: 1/3 Iter: 7/62 Batch time: 0.3642
Phase: train Epoch: 1/3 Iter: 8/62 Batch time: 0.3640
Phase: train Epoch: 1/3 Iter: 9/62 Batch time: 0.3640
Phase: train Epoch: 1/3 Iter: 10/62 Batch time: 0.3641
Phase: train Epoch: 1/3 Iter: 11/62 Batch time: 0.3636
Phase: train Epoch: 1/3 Iter: 12/62 Batch time: 0.3636
Phase: train Epoch: 1/3 Iter: 13/62 Batch time: 0.3647
Phase: train Epoch: 1/3 Iter: 14/62 Batch time: 0.3648
Phase: train Epoch: 1/3 Iter: 15/62 Batch time: 0.3670
Phase: train Epoch: 1/3 Iter: 16/62 Batch time: 0.3634
Phase: train Epoch: 1/3 Iter: 17/62 Batch time: 0.3599
Phase: train Epoch: 1/3 Iter: 18/62 Batch time: 0.3595
Phase: train Epoch: 1/3 Iter: 19/62 Batch time: 0.3582
Phase: train Epoch: 1/3 Iter: 20/62 Batch time: 0.3601
Phase: train Epoch: 1/3 Iter: 21/62 Batch time: 0.3611
Phase: train Epoch: 1/3 Iter: 22/62 Batch time: 0.3618
Phase: train Epoch: 1/3 Iter: 23/62 Batch time: 0.3620
Phase: train Epoch: 1/3 Iter: 24/62 Batch time: 0.3610
Phase: train Epoch: 1/3 Iter: 25/62 Batch time: 0.3685
Phase: train Epoch: 1/3 Iter: 26/62 Batch time: 0.3719
Phase: train Epoch: 1/3 Iter: 27/62 Batch time: 0.3639
Phase: train Epoch: 1/3 Iter: 28/62 Batch time: 0.3646
Phase: train Epoch: 1/3 Iter: 29/62 Batch time: 0.3597
Phase: train Epoch: 1/3 Iter: 30/62 Batch time: 0.3631
Phase: train Epoch: 1/3 Iter: 31/62 Batch time: 0.3612
Phase: train Epoch: 1/3 Iter: 32/62 Batch time: 0.3601
Phase: train Epoch: 1/3 Iter: 33/62 Batch time: 0.3592
Phase: train Epoch: 1/3 Iter: 34/62 Batch time: 0.3600
Phase: train Epoch: 1/3 Iter: 35/62 Batch time: 0.3592
Phase: train Epoch: 1/3 Iter: 36/62 Batch time: 0.3589
Phase: train Epoch: 1/3 Iter: 37/62 Batch time: 0.3596
Phase: train Epoch: 1/3 Iter: 38/62 Batch time: 0.3600
Phase: train Epoch: 1/3 Iter: 39/62 Batch time: 0.3618
Phase: train Epoch: 1/3 Iter: 40/62 Batch time: 0.3615
Phase: train Epoch: 1/3 Iter: 41/62 Batch time: 0.3603
Phase: train Epoch: 1/3 Iter: 42/62 Batch time: 0.3635
Phase: train Epoch: 1/3 Iter: 43/62 Batch time: 0.3605
Phase: train Epoch: 1/3 Iter: 44/62 Batch time: 0.3597
Phase: train Epoch: 1/3 Iter: 45/62 Batch time: 0.3616
Phase: train Epoch: 1/3 Iter: 46/62 Batch time: 0.3612
Phase: train Epoch: 1/3 Iter: 47/62 Batch time: 0.3598
Phase: train Epoch: 1/3 Iter: 48/62 Batch time: 0.3635
Phase: train Epoch: 1/3 Iter: 49/62 Batch time: 0.3572
Phase: train Epoch: 1/3 Iter: 50/62 Batch time: 0.3596
Phase: train Epoch: 1/3 Iter: 51/62 Batch time: 0.3596
Phase: train Epoch: 1/3 Iter: 52/62 Batch time: 0.3628
Phase: train Epoch: 1/3 Iter: 53/62 Batch time: 0.3634
Phase: train Epoch: 1/3 Iter: 54/62 Batch time: 0.3630
Phase: train Epoch: 1/3 Iter: 55/62 Batch time: 0.3652
Phase: train Epoch: 1/3 Iter: 56/62 Batch time: 0.3626
Phase: train Epoch: 1/3 Iter: 57/62 Batch time: 0.3599
Phase: train Epoch: 1/3 Iter: 58/62 Batch time: 0.3582
Phase: train Epoch: 1/3 Iter: 59/62 Batch time: 0.3605
Phase: train Epoch: 1/3 Iter: 60/62 Batch time: 0.3597
Phase: train Epoch: 1/3 Iter: 61/62 Batch time: 0.3611
Phase: train Epoch: 1/3 Loss: 0.6990 Acc: 0.5246
Phase: validation Epoch: 1/3 Iter: 1/39 Batch time: 0.2831
Phase: validation Epoch: 1/3 Iter: 2/39 Batch time: 0.2810
Phase: validation Epoch: 1/3 Iter: 3/39 Batch time: 0.2825
Phase: validation Epoch: 1/3 Iter: 4/39 Batch time: 0.2845
Phase: validation Epoch: 1/3 Iter: 5/39 Batch time: 0.2802
Phase: validation Epoch: 1/3 Iter: 6/39 Batch time: 0.2831
Phase: validation Epoch: 1/3 Iter: 7/39 Batch time: 0.2806
Phase: validation Epoch: 1/3 Iter: 8/39 Batch time: 0.2801
Phase: validation Epoch: 1/3 Iter: 9/39 Batch time: 0.2791
Phase: validation Epoch: 1/3 Iter: 10/39 Batch time: 0.2814
Phase: validation Epoch: 1/3 Iter: 11/39 Batch time: 0.2823
Phase: validation Epoch: 1/3 Iter: 12/39 Batch time: 0.2817
Phase: validation Epoch: 1/3 Iter: 13/39 Batch time: 0.2828
Phase: validation Epoch: 1/3 Iter: 14/39 Batch time: 0.2829
Phase: validation Epoch: 1/3 Iter: 15/39 Batch time: 0.2823
Phase: validation Epoch: 1/3 Iter: 16/39 Batch time: 0.2836
Phase: validation Epoch: 1/3 Iter: 17/39 Batch time: 0.2799
Phase: validation Epoch: 1/3 Iter: 18/39 Batch time: 0.2809
Phase: validation Epoch: 1/3 Iter: 19/39 Batch time: 0.2818
Phase: validation Epoch: 1/3 Iter: 20/39 Batch time: 0.2810
Phase: validation Epoch: 1/3 Iter: 21/39 Batch time: 0.2807
Phase: validation Epoch: 1/3 Iter: 22/39 Batch time: 0.2822
Phase: validation Epoch: 1/3 Iter: 23/39 Batch time: 0.2816
Phase: validation Epoch: 1/3 Iter: 24/39 Batch time: 0.2824
Phase: validation Epoch: 1/3 Iter: 25/39 Batch time: 0.2833
Phase: validation Epoch: 1/3 Iter: 26/39 Batch time: 0.2837
Phase: validation Epoch: 1/3 Iter: 27/39 Batch time: 0.2854
Phase: validation Epoch: 1/3 Iter: 28/39 Batch time: 0.2828
Phase: validation Epoch: 1/3 Iter: 29/39 Batch time: 0.2833
Phase: validation Epoch: 1/3 Iter: 30/39 Batch time: 0.2860
Phase: validation Epoch: 1/3 Iter: 31/39 Batch time: 0.2865
Phase: validation Epoch: 1/3 Iter: 32/39 Batch time: 0.2835
Phase: validation Epoch: 1/3 Iter: 33/39 Batch time: 0.2856
Phase: validation Epoch: 1/3 Iter: 34/39 Batch time: 0.2832
Phase: validation Epoch: 1/3 Iter: 35/39 Batch time: 0.2817
Phase: validation Epoch: 1/3 Iter: 36/39 Batch time: 0.2812
Phase: validation Epoch: 1/3 Iter: 37/39 Batch time: 0.2799
Phase: validation Epoch: 1/3 Iter: 38/39 Batch time: 0.2830
Phase: validation Epoch: 1/3 Iter: 39/39 Batch time: 0.0886
Phase: validation   Epoch: 1/3 Loss: 0.6429 Acc: 0.6536
Phase: train Epoch: 2/3 Iter: 1/62 Batch time: 0.3573
Phase: train Epoch: 2/3 Iter: 2/62 Batch time: 0.3600
Phase: train Epoch: 2/3 Iter: 3/62 Batch time: 0.3586
Phase: train Epoch: 2/3 Iter: 4/62 Batch time: 0.3584
Phase: train Epoch: 2/3 Iter: 5/62 Batch time: 0.3631
Phase: train Epoch: 2/3 Iter: 6/62 Batch time: 0.3604
Phase: train Epoch: 2/3 Iter: 7/62 Batch time: 0.3602
Phase: train Epoch: 2/3 Iter: 8/62 Batch time: 0.3610
Phase: train Epoch: 2/3 Iter: 9/62 Batch time: 0.3647
Phase: train Epoch: 2/3 Iter: 10/62 Batch time: 0.3622
Phase: train Epoch: 2/3 Iter: 11/62 Batch time: 0.3609
Phase: train Epoch: 2/3 Iter: 12/62 Batch time: 0.3643
Phase: train Epoch: 2/3 Iter: 13/62 Batch time: 0.3639
Phase: train Epoch: 2/3 Iter: 14/62 Batch time: 0.3632
Phase: train Epoch: 2/3 Iter: 15/62 Batch time: 0.3628
Phase: train Epoch: 2/3 Iter: 16/62 Batch time: 0.3648
Phase: train Epoch: 2/3 Iter: 17/62 Batch time: 0.3639
Phase: train Epoch: 2/3 Iter: 18/62 Batch time: 0.3642
Phase: train Epoch: 2/3 Iter: 19/62 Batch time: 0.3630
Phase: train Epoch: 2/3 Iter: 20/62 Batch time: 0.3645
Phase: train Epoch: 2/3 Iter: 21/62 Batch time: 0.3669
Phase: train Epoch: 2/3 Iter: 22/62 Batch time: 0.3638
Phase: train Epoch: 2/3 Iter: 23/62 Batch time: 0.3634
Phase: train Epoch: 2/3 Iter: 24/62 Batch time: 0.3679
Phase: train Epoch: 2/3 Iter: 25/62 Batch time: 0.3620
Phase: train Epoch: 2/3 Iter: 26/62 Batch time: 0.3623
Phase: train Epoch: 2/3 Iter: 27/62 Batch time: 0.3612
Phase: train Epoch: 2/3 Iter: 28/62 Batch time: 0.3630
Phase: train Epoch: 2/3 Iter: 29/62 Batch time: 0.3611
Phase: train Epoch: 2/3 Iter: 30/62 Batch time: 0.3623
Phase: train Epoch: 2/3 Iter: 31/62 Batch time: 0.3629
Phase: train Epoch: 2/3 Iter: 32/62 Batch time: 0.3629
Phase: train Epoch: 2/3 Iter: 33/62 Batch time: 0.3678
Phase: train Epoch: 2/3 Iter: 34/62 Batch time: 0.3600
Phase: train Epoch: 2/3 Iter: 35/62 Batch time: 0.3581
Phase: train Epoch: 2/3 Iter: 36/62 Batch time: 0.3576
Phase: train Epoch: 2/3 Iter: 37/62 Batch time: 0.3599
Phase: train Epoch: 2/3 Iter: 38/62 Batch time: 0.3587
Phase: train Epoch: 2/3 Iter: 39/62 Batch time: 0.3581
Phase: train Epoch: 2/3 Iter: 40/62 Batch time: 0.3605
Phase: train Epoch: 2/3 Iter: 41/62 Batch time: 0.3599
Phase: train Epoch: 2/3 Iter: 42/62 Batch time: 0.3584
Phase: train Epoch: 2/3 Iter: 43/62 Batch time: 0.3598
Phase: train Epoch: 2/3 Iter: 44/62 Batch time: 0.3591
Phase: train Epoch: 2/3 Iter: 45/62 Batch time: 0.3628
Phase: train Epoch: 2/3 Iter: 46/62 Batch time: 0.3604
Phase: train Epoch: 2/3 Iter: 47/62 Batch time: 0.3593
Phase: train Epoch: 2/3 Iter: 48/62 Batch time: 0.3599
Phase: train Epoch: 2/3 Iter: 49/62 Batch time: 0.3579
Phase: train Epoch: 2/3 Iter: 50/62 Batch time: 0.3598
Phase: train Epoch: 2/3 Iter: 51/62 Batch time: 0.3592
Phase: train Epoch: 2/3 Iter: 52/62 Batch time: 0.3599
Phase: train Epoch: 2/3 Iter: 53/62 Batch time: 0.3612
Phase: train Epoch: 2/3 Iter: 54/62 Batch time: 0.3593
Phase: train Epoch: 2/3 Iter: 55/62 Batch time: 0.3609
Phase: train Epoch: 2/3 Iter: 56/62 Batch time: 0.3599
Phase: train Epoch: 2/3 Iter: 57/62 Batch time: 0.3605
Phase: train Epoch: 2/3 Iter: 58/62 Batch time: 0.3610
Phase: train Epoch: 2/3 Iter: 59/62 Batch time: 0.3604
Phase: train Epoch: 2/3 Iter: 60/62 Batch time: 0.3593
Phase: train Epoch: 2/3 Iter: 61/62 Batch time: 0.3587
Phase: train Epoch: 2/3 Loss: 0.6134 Acc: 0.7008
Phase: validation Epoch: 2/3 Iter: 1/39 Batch time: 0.2834
Phase: validation Epoch: 2/3 Iter: 2/39 Batch time: 0.2809
Phase: validation Epoch: 2/3 Iter: 3/39 Batch time: 0.2815
Phase: validation Epoch: 2/3 Iter: 4/39 Batch time: 0.2802
Phase: validation Epoch: 2/3 Iter: 5/39 Batch time: 0.2819
Phase: validation Epoch: 2/3 Iter: 6/39 Batch time: 0.2820
Phase: validation Epoch: 2/3 Iter: 7/39 Batch time: 0.2815
Phase: validation Epoch: 2/3 Iter: 8/39 Batch time: 0.2816
Phase: validation Epoch: 2/3 Iter: 9/39 Batch time: 0.2832
Phase: validation Epoch: 2/3 Iter: 10/39 Batch time: 0.2817
Phase: validation Epoch: 2/3 Iter: 11/39 Batch time: 0.2826
Phase: validation Epoch: 2/3 Iter: 12/39 Batch time: 0.2839
Phase: validation Epoch: 2/3 Iter: 13/39 Batch time: 0.2827
Phase: validation Epoch: 2/3 Iter: 14/39 Batch time: 0.2799
Phase: validation Epoch: 2/3 Iter: 15/39 Batch time: 0.2811
Phase: validation Epoch: 2/3 Iter: 16/39 Batch time: 0.2844
Phase: validation Epoch: 2/3 Iter: 17/39 Batch time: 0.2833
Phase: validation Epoch: 2/3 Iter: 18/39 Batch time: 0.2831
Phase: validation Epoch: 2/3 Iter: 19/39 Batch time: 0.2818
Phase: validation Epoch: 2/3 Iter: 20/39 Batch time: 0.2818
Phase: validation Epoch: 2/3 Iter: 21/39 Batch time: 0.2814
Phase: validation Epoch: 2/3 Iter: 22/39 Batch time: 0.2805
Phase: validation Epoch: 2/3 Iter: 23/39 Batch time: 0.2818
Phase: validation Epoch: 2/3 Iter: 24/39 Batch time: 0.2823
Phase: validation Epoch: 2/3 Iter: 25/39 Batch time: 0.2796
Phase: validation Epoch: 2/3 Iter: 26/39 Batch time: 0.2794
Phase: validation Epoch: 2/3 Iter: 27/39 Batch time: 0.2823
Phase: validation Epoch: 2/3 Iter: 28/39 Batch time: 0.2817
Phase: validation Epoch: 2/3 Iter: 29/39 Batch time: 0.2810
Phase: validation Epoch: 2/3 Iter: 30/39 Batch time: 0.2815
Phase: validation Epoch: 2/3 Iter: 31/39 Batch time: 0.2826
Phase: validation Epoch: 2/3 Iter: 32/39 Batch time: 0.2828
Phase: validation Epoch: 2/3 Iter: 33/39 Batch time: 0.2831
Phase: validation Epoch: 2/3 Iter: 34/39 Batch time: 0.2829
Phase: validation Epoch: 2/3 Iter: 35/39 Batch time: 0.2833
Phase: validation Epoch: 2/3 Iter: 36/39 Batch time: 0.2841
Phase: validation Epoch: 2/3 Iter: 37/39 Batch time: 0.2839
Phase: validation Epoch: 2/3 Iter: 38/39 Batch time: 0.2809
Phase: validation Epoch: 2/3 Iter: 39/39 Batch time: 0.0823
Phase: validation   Epoch: 2/3 Loss: 0.5389 Acc: 0.8235
Phase: train Epoch: 3/3 Iter: 1/62 Batch time: 0.3556
Phase: train Epoch: 3/3 Iter: 2/62 Batch time: 0.3604
Phase: train Epoch: 3/3 Iter: 3/62 Batch time: 0.3610
Phase: train Epoch: 3/3 Iter: 4/62 Batch time: 0.3588
Phase: train Epoch: 3/3 Iter: 5/62 Batch time: 0.3608
Phase: train Epoch: 3/3 Iter: 6/62 Batch time: 0.3611
Phase: train Epoch: 3/3 Iter: 7/62 Batch time: 0.3598
Phase: train Epoch: 3/3 Iter: 8/62 Batch time: 0.3592
Phase: train Epoch: 3/3 Iter: 9/62 Batch time: 0.3604
Phase: train Epoch: 3/3 Iter: 10/62 Batch time: 0.3619
Phase: train Epoch: 3/3 Iter: 11/62 Batch time: 0.3594
Phase: train Epoch: 3/3 Iter: 12/62 Batch time: 0.3635
Phase: train Epoch: 3/3 Iter: 13/62 Batch time: 0.3640
Phase: train Epoch: 3/3 Iter: 14/62 Batch time: 0.3637
Phase: train Epoch: 3/3 Iter: 15/62 Batch time: 0.3630
Phase: train Epoch: 3/3 Iter: 16/62 Batch time: 0.3630
Phase: train Epoch: 3/3 Iter: 17/62 Batch time: 0.3637
Phase: train Epoch: 3/3 Iter: 18/62 Batch time: 0.3594
Phase: train Epoch: 3/3 Iter: 19/62 Batch time: 0.3591
Phase: train Epoch: 3/3 Iter: 20/62 Batch time: 0.3593
Phase: train Epoch: 3/3 Iter: 21/62 Batch time: 0.3586
Phase: train Epoch: 3/3 Iter: 22/62 Batch time: 0.3594
Phase: train Epoch: 3/3 Iter: 23/62 Batch time: 0.3598
Phase: train Epoch: 3/3 Iter: 24/62 Batch time: 0.3613
Phase: train Epoch: 3/3 Iter: 25/62 Batch time: 0.3597
Phase: train Epoch: 3/3 Iter: 26/62 Batch time: 0.3607
Phase: train Epoch: 3/3 Iter: 27/62 Batch time: 0.3579
Phase: train Epoch: 3/3 Iter: 28/62 Batch time: 0.3620
Phase: train Epoch: 3/3 Iter: 29/62 Batch time: 0.3604
Phase: train Epoch: 3/3 Iter: 30/62 Batch time: 0.3584
Phase: train Epoch: 3/3 Iter: 31/62 Batch time: 0.3551
Phase: train Epoch: 3/3 Iter: 32/62 Batch time: 0.3580
Phase: train Epoch: 3/3 Iter: 33/62 Batch time: 0.3554
Phase: train Epoch: 3/3 Iter: 34/62 Batch time: 0.3573
Phase: train Epoch: 3/3 Iter: 35/62 Batch time: 0.3589
Phase: train Epoch: 3/3 Iter: 36/62 Batch time: 0.3596
Phase: train Epoch: 3/3 Iter: 37/62 Batch time: 0.3593
Phase: train Epoch: 3/3 Iter: 38/62 Batch time: 0.3593
Phase: train Epoch: 3/3 Iter: 39/62 Batch time: 0.3559
Phase: train Epoch: 3/3 Iter: 40/62 Batch time: 0.3628
Phase: train Epoch: 3/3 Iter: 41/62 Batch time: 0.3587
Phase: train Epoch: 3/3 Iter: 42/62 Batch time: 0.3609
Phase: train Epoch: 3/3 Iter: 43/62 Batch time: 0.3602
Phase: train Epoch: 3/3 Iter: 44/62 Batch time: 0.3605
Phase: train Epoch: 3/3 Iter: 45/62 Batch time: 0.3615
Phase: train Epoch: 3/3 Iter: 46/62 Batch time: 0.3609
Phase: train Epoch: 3/3 Iter: 47/62 Batch time: 0.3602
Phase: train Epoch: 3/3 Iter: 48/62 Batch time: 0.3607
Phase: train Epoch: 3/3 Iter: 49/62 Batch time: 0.3624
Phase: train Epoch: 3/3 Iter: 50/62 Batch time: 0.3612
Phase: train Epoch: 3/3 Iter: 51/62 Batch time: 0.3606
Phase: train Epoch: 3/3 Iter: 52/62 Batch time: 0.3638
Phase: train Epoch: 3/3 Iter: 53/62 Batch time: 0.3615
Phase: train Epoch: 3/3 Iter: 54/62 Batch time: 0.3628
Phase: train Epoch: 3/3 Iter: 55/62 Batch time: 0.3605
Phase: train Epoch: 3/3 Iter: 56/62 Batch time: 0.3619
Phase: train Epoch: 3/3 Iter: 57/62 Batch time: 0.3629
Phase: train Epoch: 3/3 Iter: 58/62 Batch time: 0.3652
Phase: train Epoch: 3/3 Iter: 59/62 Batch time: 0.3602
Phase: train Epoch: 3/3 Iter: 60/62 Batch time: 0.3682
Phase: train Epoch: 3/3 Iter: 61/62 Batch time: 0.3618
Phase: train Epoch: 3/3 Loss: 0.5652 Acc: 0.7418
Phase: validation Epoch: 3/3 Iter: 1/39 Batch time: 0.2869
Phase: validation Epoch: 3/3 Iter: 2/39 Batch time: 0.2828
Phase: validation Epoch: 3/3 Iter: 3/39 Batch time: 0.2807
Phase: validation Epoch: 3/3 Iter: 4/39 Batch time: 0.2810
Phase: validation Epoch: 3/3 Iter: 5/39 Batch time: 0.2833
Phase: validation Epoch: 3/3 Iter: 6/39 Batch time: 0.2834
Phase: validation Epoch: 3/3 Iter: 7/39 Batch time: 0.2821
Phase: validation Epoch: 3/3 Iter: 8/39 Batch time: 0.2836
Phase: validation Epoch: 3/3 Iter: 9/39 Batch time: 0.2832
Phase: validation Epoch: 3/3 Iter: 10/39 Batch time: 0.2818
Phase: validation Epoch: 3/3 Iter: 11/39 Batch time: 0.2812
Phase: validation Epoch: 3/3 Iter: 12/39 Batch time: 0.2830
Phase: validation Epoch: 3/3 Iter: 13/39 Batch time: 0.2829
Phase: validation Epoch: 3/3 Iter: 14/39 Batch time: 0.2826
Phase: validation Epoch: 3/3 Iter: 15/39 Batch time: 0.2798
Phase: validation Epoch: 3/3 Iter: 16/39 Batch time: 0.2818
Phase: validation Epoch: 3/3 Iter: 17/39 Batch time: 0.2839
Phase: validation Epoch: 3/3 Iter: 18/39 Batch time: 0.2841
Phase: validation Epoch: 3/3 Iter: 19/39 Batch time: 0.2829
Phase: validation Epoch: 3/3 Iter: 20/39 Batch time: 0.2844
Phase: validation Epoch: 3/3 Iter: 21/39 Batch time: 0.2821
Phase: validation Epoch: 3/3 Iter: 22/39 Batch time: 0.2838
Phase: validation Epoch: 3/3 Iter: 23/39 Batch time: 0.2842
Phase: validation Epoch: 3/3 Iter: 24/39 Batch time: 0.2809
Phase: validation Epoch: 3/3 Iter: 25/39 Batch time: 0.2829
Phase: validation Epoch: 3/3 Iter: 26/39 Batch time: 0.2830
Phase: validation Epoch: 3/3 Iter: 27/39 Batch time: 0.2833
Phase: validation Epoch: 3/3 Iter: 28/39 Batch time: 0.2837
Phase: validation Epoch: 3/3 Iter: 29/39 Batch time: 0.2814
Phase: validation Epoch: 3/3 Iter: 30/39 Batch time: 0.2826
Phase: validation Epoch: 3/3 Iter: 31/39 Batch time: 0.2846
Phase: validation Epoch: 3/3 Iter: 32/39 Batch time: 0.2834
Phase: validation Epoch: 3/3 Iter: 33/39 Batch time: 0.2838
Phase: validation Epoch: 3/3 Iter: 34/39 Batch time: 0.2842
Phase: validation Epoch: 3/3 Iter: 35/39 Batch time: 0.2839
Phase: validation Epoch: 3/3 Iter: 36/39 Batch time: 0.2887
Phase: validation Epoch: 3/3 Iter: 37/39 Batch time: 0.2849
Phase: validation Epoch: 3/3 Iter: 38/39 Batch time: 0.2837
Phase: validation Epoch: 3/3 Iter: 39/39 Batch time: 0.0825
Phase: validation   Epoch: 3/3 Loss: 0.4484 Acc: 0.8497
Training completed in 1m 46s
Best test loss: 0.4484 | Best test accuracy: 0.8497
 </code>
 </pre>
 </details>

---

## 7. tutorial_noisy_circuit_optimization.html <a name="demo6"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_noisy_circuit_optimization.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
CPhase
Depolarize
ISWAP
Operation
PhaseDamp
PhaseFlip
Expectation value: 0.862
Step     5. Cost:  0.9560000; Noisy Cost:  0.5460000
Step    10. Cost:  0.2620000; Noisy Cost:  0.5400000
Step    15. Cost: -0.9440000; Noisy Cost:  0.2740000
Step    20. Cost: -1.0000000; Noisy Cost: -0.2720000
Step    25. Cost: -1.0000000; Noisy Cost: -0.5520000
Step    30. Cost: -1.0000000; Noisy Cost: -0.5900000
Step    35. Cost: -1.0000000; Noisy Cost: -0.6060000
Step    40. Cost: -1.0000000; Noisy Cost: -0.6040000
Step    45. Cost: -1.0000000; Noisy Cost: -0.6300000
Step    50. Cost: -1.0000000; Noisy Cost: -0.6340000
Step    55. Cost: -1.0000000; Noisy Cost: -0.5700000
Step    60. Cost: -1.0000000; Noisy Cost: -0.6000000
Step    65. Cost: -1.0000000; Noisy Cost: -0.6120000
Step    70. Cost: -1.0000000; Noisy Cost: -0.5860000
Step    75. Cost: -0.9980000; Noisy Cost: -0.5960000
Step    80. Cost: -1.0000000; Noisy Cost: -0.6060000
Step    85. Cost: -0.9980000; Noisy Cost: -0.5960000
Step    90. Cost: -1.0000000; Noisy Cost: -0.5940000
Step    95. Cost: -1.0000000; Noisy Cost: -0.6000000
Step   100. Cost: -1.0000000; Noisy Cost: -0.5520000
Optimized rotation angles (noise-free case):
( 0.0034000,  3.1410000)
 </code>
 </pre>
 </details>

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_noisy_circuit_optimization.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Depolarize
Operation
PhaseDamp
PhaseFlip
Expectation value: 0.862
Step     5. Cost:  0.9560000; Noisy Cost:  0.5460000
Step    10. Cost:  0.2620000; Noisy Cost:  0.5400000
Step    15. Cost: -0.9440000; Noisy Cost:  0.2740000
Step    20. Cost: -1.0000000; Noisy Cost: -0.2720000
Step    25. Cost: -1.0000000; Noisy Cost: -0.5520000
Step    30. Cost: -1.0000000; Noisy Cost: -0.5900000
Step    35. Cost: -1.0000000; Noisy Cost: -0.6060000
Step    40. Cost: -1.0000000; Noisy Cost: -0.6040000
Step    45. Cost: -1.0000000; Noisy Cost: -0.6300000
Step    50. Cost: -1.0000000; Noisy Cost: -0.6340000
Step    55. Cost: -1.0000000; Noisy Cost: -0.5700000
Step    60. Cost: -1.0000000; Noisy Cost: -0.6000000
Step    65. Cost: -1.0000000; Noisy Cost: -0.6120000
Step    70. Cost: -1.0000000; Noisy Cost: -0.5860000
Step    75. Cost: -0.9980000; Noisy Cost: -0.5960000
Step    80. Cost: -1.0000000; Noisy Cost: -0.6060000
Step    85. Cost: -0.9980000; Noisy Cost: -0.5960000
Step    90. Cost: -1.0000000; Noisy Cost: -0.5940000
Step    95. Cost: -1.0000000; Noisy Cost: -0.6000000
Step   100. Cost: -1.0000000; Noisy Cost: -0.5520000
Optimized rotation angles (noise-free case):
( 0.0034000,  3.1410000)
Optimized rotation angles (noisy case):
( 0.0074000,  3.1554000)
 </code>
 </pre>
 </details>

---

## 8. tutorial_quanvolution.html <a name="demo7"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_quanvolution.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
   16384/11490434 [..............................] - ETA: 0s
 3915776/11490434 [=========>....................] - ETA: 0s
13/13 - 0s - loss: 0.1135 - accuracy: 1.0000 - val_loss: 1.0451 - val_accuracy: 0.7333 - 47ms/epoch - 4ms/step
13/13 - 0s - loss: 0.1041 - accuracy: 1.0000 - val_loss: 1.0142 - val_accuracy: 0.7333 - 49ms/epoch - 4ms/step
13/13 - 0s - loss: 0.0983 - accuracy: 1.0000 - val_loss: 0.9893 - val_accuracy: 0.7333 - 35ms/epoch - 3ms/step
13/13 - 0s - loss: 0.0913 - accuracy: 1.0000 - val_loss: 0.9807 - val_accuracy: 0.7000 - 49ms/epoch - 4ms/step
13/13 - 0s - loss: 0.0868 - accuracy: 1.0000 - val_loss: 0.9715 - val_accuracy: 0.7333 - 49ms/epoch - 4ms/step
13/13 - 0s - loss: 0.0788 - accuracy: 1.0000 - val_loss: 0.9850 - val_accuracy: 0.7333 - 36ms/epoch - 3ms/step
13/13 - 0s - loss: 0.0749 - accuracy: 1.0000 - val_loss: 0.9750 - val_accuracy: 0.7333 - 48ms/epoch - 4ms/step
13/13 - 0s - loss: 0.0730 - accuracy: 1.0000 - val_loss: 0.9570 - val_accuracy: 0.7667 - 35ms/epoch - 3ms/step
13/13 - 0s - loss: 0.0681 - accuracy: 1.0000 - val_loss: 0.9895 - val_accuracy: 0.7333 - 50ms/epoch - 4ms/step
13/13 - 0s - loss: 0.0635 - accuracy: 1.0000 - val_loss: 0.9560 - val_accuracy: 0.7333 - 49ms/epoch - 4ms/step
13/13 - 0s - loss: 2.3619 - accuracy: 0.1400 - val_loss: 2.0567 - val_accuracy: 0.3667 - 371ms/epoch - 29ms/step
13/13 - 0s - loss: 1.9696 - accuracy: 0.4200 - val_loss: 1.9381 - val_accuracy: 0.4667 - 47ms/epoch - 4ms/step
13/13 - 0s - loss: 1.6671 - accuracy: 0.6400 - val_loss: 1.8300 - val_accuracy: 0.4333 - 48ms/epoch - 4ms/step
13/13 - 0s - loss: 1.4340 - accuracy: 0.7400 - val_loss: 1.7113 - val_accuracy: 0.4333 - 47ms/epoch - 4ms/step
13/13 - 0s - loss: 1.2342 - accuracy: 0.7600 - val_loss: 1.6044 - val_accuracy: 0.5000 - 32ms/epoch - 2ms/step
13/13 - 0s - loss: 1.0721 - accuracy: 0.8600 - val_loss: 1.5232 - val_accuracy: 0.5333 - 32ms/epoch - 2ms/step
13/13 - 0s - loss: 0.9348 - accuracy: 0.9000 - val_loss: 1.4596 - val_accuracy: 0.5667 - 46ms/epoch - 4ms/step
13/13 - 0s - loss: 0.8178 - accuracy: 0.9200 - val_loss: 1.3921 - val_accuracy: 0.6000 - 33ms/epoch - 3ms/step
13/13 - 0s - loss: 0.7223 - accuracy: 0.9400 - val_loss: 1.3404 - val_accuracy: 0.6333 - 32ms/epoch - 2ms/step
13/13 - 0s - loss: 0.5772 - accuracy: 1.0000 - val_loss: 1.2644 - val_accuracy: 0.6333 - 34ms/epoch - 3ms/step
13/13 - 0s - loss: 0.5199 - accuracy: 1.0000 - val_loss: 1.2558 - val_accuracy: 0.6667 - 47ms/epoch - 4ms/step
13/13 - 0s - loss: 0.4695 - accuracy: 1.0000 - val_loss: 1.2258 - val_accuracy: 0.6667 - 34ms/epoch - 3ms/step
13/13 - 0s - loss: 0.4238 - accuracy: 1.0000 - val_loss: 1.1897 - val_accuracy: 0.6667 - 48ms/epoch - 4ms/step
13/13 - 0s - loss: 0.3848 - accuracy: 1.0000 - val_loss: 1.1651 - val_accuracy: 0.6667 - 33ms/epoch - 3ms/step
13/13 - 0s - loss: 0.3525 - accuracy: 1.0000 - val_loss: 1.1503 - val_accuracy: 0.7333 - 47ms/epoch - 4ms/step
13/13 - 0s - loss: 0.3245 - accuracy: 1.0000 - val_loss: 1.1374 - val_accuracy: 0.7000 - 48ms/epoch - 4ms/step
13/13 - 0s - loss: 0.2992 - accuracy: 1.0000 - val_loss: 1.1174 - val_accuracy: 0.6667 - 47ms/epoch - 4ms/step
13/13 - 0s - loss: 0.2745 - accuracy: 1.0000 - val_loss: 1.1119 - val_accuracy: 0.6667 - 33ms/epoch - 3ms/step
13/13 - 0s - loss: 0.2551 - accuracy: 1.0000 - val_loss: 1.0903 - val_accuracy: 0.7000 - 46ms/epoch - 4ms/step
13/13 - 0s - loss: 0.2370 - accuracy: 1.0000 - val_loss: 1.0877 - val_accuracy: 0.6667 - 33ms/epoch - 3ms/step
13/13 - 0s - loss: 0.2199 - accuracy: 1.0000 - val_loss: 1.0776 - val_accuracy: 0.6667 - 34ms/epoch - 3ms/step
13/13 - 0s - loss: 0.2050 - accuracy: 1.0000 - val_loss: 1.0675 - val_accuracy: 0.7000 - 48ms/epoch - 4ms/step
13/13 - 0s - loss: 0.1919 - accuracy: 1.0000 - val_loss: 1.0592 - val_accuracy: 0.7000 - 33ms/epoch - 3ms/step
13/13 - 0s - loss: 0.1812 - accuracy: 1.0000 - val_loss: 1.0554 - val_accuracy: 0.6667 - 34ms/epoch - 3ms/step
13/13 - 0s - loss: 0.1599 - accuracy: 1.0000 - val_loss: 1.0377 - val_accuracy: 0.7000 - 32ms/epoch - 2ms/step
13/13 - 0s - loss: 0.1515 - accuracy: 1.0000 - val_loss: 1.0341 - val_accuracy: 0.6667 - 32ms/epoch - 2ms/step
13/13 - 0s - loss: 0.1426 - accuracy: 1.0000 - val_loss: 1.0291 - val_accuracy: 0.7000 - 48ms/epoch - 4ms/step
13/13 - 0s - loss: 0.1344 - accuracy: 1.0000 - val_loss: 1.0264 - val_accuracy: 0.7000 - 47ms/epoch - 4ms/step
13/13 - 0s - loss: 3.0160 - accuracy: 0.1000 - val_loss: 2.0646 - val_accuracy: 0.2000 - 434ms/epoch - 33ms/step
13/13 - 0s - loss: 2.2510 - accuracy: 0.1800 - val_loss: 1.9801 - val_accuracy: 0.3333 - 47ms/epoch - 4ms/step
13/13 - 0s - loss: 1.7851 - accuracy: 0.4000 - val_loss: 1.8177 - val_accuracy: 0.2667 - 33ms/epoch - 3ms/step
13/13 - 0s - loss: 1.3652 - accuracy: 0.5400 - val_loss: 1.6107 - val_accuracy: 0.4667 - 34ms/epoch - 3ms/step
13/13 - 0s - loss: 0.7383 - accuracy: 0.9400 - val_loss: 1.3536 - val_accuracy: 0.5667 - 33ms/epoch - 3ms/step
13/13 - 0s - loss: 0.5846 - accuracy: 0.9800 - val_loss: 1.2785 - val_accuracy: 0.6667 - 32ms/epoch - 2ms/step
13/13 - 0s - loss: 0.4987 - accuracy: 0.9800 - val_loss: 1.2253 - val_accuracy: 0.6333 - 33ms/epoch - 3ms/step
13/13 - 0s - loss: 0.3921 - accuracy: 1.0000 - val_loss: 1.2655 - val_accuracy: 0.6333 - 48ms/epoch - 4ms/step
13/13 - 0s - loss: 0.3617 - accuracy: 1.0000 - val_loss: 1.1555 - val_accuracy: 0.7000 - 49ms/epoch - 4ms/step
13/13 - 0s - loss: 0.2618 - accuracy: 1.0000 - val_loss: 1.1166 - val_accuracy: 0.7333 - 47ms/epoch - 4ms/step
13/13 - 0s - loss: 0.2463 - accuracy: 1.0000 - val_loss: 1.0624 - val_accuracy: 0.7000 - 33ms/epoch - 3ms/step
13/13 - 0s - loss: 0.2033 - accuracy: 1.0000 - val_loss: 1.0904 - val_accuracy: 0.7333 - 47ms/epoch - 4ms/step
13/13 - 0s - loss: 0.1682 - accuracy: 1.0000 - val_loss: 1.0385 - val_accuracy: 0.7333 - 48ms/epoch - 4ms/step
13/13 - 0s - loss: 0.1484 - accuracy: 1.0000 - val_loss: 1.0676 - val_accuracy: 0.7000 - 35ms/epoch - 3ms/step
13/13 - 0s - loss: 0.1349 - accuracy: 1.0000 - val_loss: 1.0447 - val_accuracy: 0.7000 - 35ms/epoch - 3ms/step
13/13 - 0s - loss: 0.1255 - accuracy: 1.0000 - val_loss: 0.9935 - val_accuracy: 0.7333 - 34ms/epoch - 3ms/step
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
   16384/11490434 [..............................] - ETA: 1s
 4202496/11490434 [=========>....................] - ETA: 0s
13/13 - 0s - loss: 0.1135 - accuracy: 1.0000 - val_loss: 1.0451 - val_accuracy: 0.7333 - 34ms/epoch - 3ms/step
13/13 - 0s - loss: 0.1041 - accuracy: 1.0000 - val_loss: 1.0142 - val_accuracy: 0.7333 - 47ms/epoch - 4ms/step
13/13 - 0s - loss: 0.0983 - accuracy: 1.0000 - val_loss: 0.9893 - val_accuracy: 0.7333 - 49ms/epoch - 4ms/step
13/13 - 0s - loss: 0.0913 - accuracy: 1.0000 - val_loss: 0.9807 - val_accuracy: 0.7000 - 33ms/epoch - 3ms/step
13/13 - 0s - loss: 0.0868 - accuracy: 1.0000 - val_loss: 0.9715 - val_accuracy: 0.7333 - 48ms/epoch - 4ms/step
13/13 - 0s - loss: 0.0788 - accuracy: 1.0000 - val_loss: 0.9850 - val_accuracy: 0.7333 - 47ms/epoch - 4ms/step
13/13 - 0s - loss: 0.0749 - accuracy: 1.0000 - val_loss: 0.9750 - val_accuracy: 0.7333 - 49ms/epoch - 4ms/step
13/13 - 0s - loss: 0.0730 - accuracy: 1.0000 - val_loss: 0.9570 - val_accuracy: 0.7667 - 34ms/epoch - 3ms/step
13/13 - 0s - loss: 0.0681 - accuracy: 1.0000 - val_loss: 0.9895 - val_accuracy: 0.7333 - 49ms/epoch - 4ms/step
13/13 - 0s - loss: 0.0635 - accuracy: 1.0000 - val_loss: 0.9560 - val_accuracy: 0.7333 - 33ms/epoch - 3ms/step
13/13 - 0s - loss: 2.3619 - accuracy: 0.1400 - val_loss: 2.0567 - val_accuracy: 0.3667 - 355ms/epoch - 27ms/step
13/13 - 0s - loss: 1.9696 - accuracy: 0.4200 - val_loss: 1.9381 - val_accuracy: 0.4667 - 46ms/epoch - 4ms/step
13/13 - 0s - loss: 1.6671 - accuracy: 0.6400 - val_loss: 1.8300 - val_accuracy: 0.4333 - 46ms/epoch - 4ms/step
13/13 - 0s - loss: 1.4340 - accuracy: 0.7400 - val_loss: 1.7113 - val_accuracy: 0.4333 - 32ms/epoch - 2ms/step
13/13 - 0s - loss: 1.2342 - accuracy: 0.7600 - val_loss: 1.6044 - val_accuracy: 0.5000 - 45ms/epoch - 3ms/step
13/13 - 0s - loss: 1.0721 - accuracy: 0.8600 - val_loss: 1.5232 - val_accuracy: 0.5333 - 33ms/epoch - 3ms/step
13/13 - 0s - loss: 0.9348 - accuracy: 0.9000 - val_loss: 1.4596 - val_accuracy: 0.5667 - 31ms/epoch - 2ms/step
13/13 - 0s - loss: 0.8178 - accuracy: 0.9200 - val_loss: 1.3921 - val_accuracy: 0.6000 - 31ms/epoch - 2ms/step
13/13 - 0s - loss: 0.7223 - accuracy: 0.9400 - val_loss: 1.3404 - val_accuracy: 0.6333 - 47ms/epoch - 4ms/step
13/13 - 0s - loss: 0.5772 - accuracy: 1.0000 - val_loss: 1.2644 - val_accuracy: 0.6333 - 46ms/epoch - 4ms/step
13/13 - 0s - loss: 0.5199 - accuracy: 1.0000 - val_loss: 1.2558 - val_accuracy: 0.6667 - 32ms/epoch - 2ms/step
13/13 - 0s - loss: 0.4695 - accuracy: 1.0000 - val_loss: 1.2258 - val_accuracy: 0.6667 - 32ms/epoch - 2ms/step
13/13 - 0s - loss: 0.4238 - accuracy: 1.0000 - val_loss: 1.1897 - val_accuracy: 0.6667 - 31ms/epoch - 2ms/step
13/13 - 0s - loss: 0.3848 - accuracy: 1.0000 - val_loss: 1.1651 - val_accuracy: 0.6667 - 31ms/epoch - 2ms/step
13/13 - 0s - loss: 0.3525 - accuracy: 1.0000 - val_loss: 1.1503 - val_accuracy: 0.7333 - 45ms/epoch - 3ms/step
13/13 - 0s - loss: 0.3245 - accuracy: 1.0000 - val_loss: 1.1374 - val_accuracy: 0.7000 - 32ms/epoch - 2ms/step
13/13 - 0s - loss: 0.2992 - accuracy: 1.0000 - val_loss: 1.1174 - val_accuracy: 0.6667 - 32ms/epoch - 2ms/step
13/13 - 0s - loss: 0.2745 - accuracy: 1.0000 - val_loss: 1.1119 - val_accuracy: 0.6667 - 47ms/epoch - 4ms/step
13/13 - 0s - loss: 0.2551 - accuracy: 1.0000 - val_loss: 1.0903 - val_accuracy: 0.7000 - 47ms/epoch - 4ms/step
13/13 - 0s - loss: 0.2370 - accuracy: 1.0000 - val_loss: 1.0877 - val_accuracy: 0.6667 - 32ms/epoch - 2ms/step
13/13 - 0s - loss: 0.2199 - accuracy: 1.0000 - val_loss: 1.0776 - val_accuracy: 0.6667 - 46ms/epoch - 4ms/step
13/13 - 0s - loss: 0.2050 - accuracy: 1.0000 - val_loss: 1.0675 - val_accuracy: 0.7000 - 31ms/epoch - 2ms/step
13/13 - 0s - loss: 0.1919 - accuracy: 1.0000 - val_loss: 1.0592 - val_accuracy: 0.7000 - 47ms/epoch - 4ms/step
13/13 - 0s - loss: 0.1812 - accuracy: 1.0000 - val_loss: 1.0554 - val_accuracy: 0.6667 - 33ms/epoch - 3ms/step
13/13 - 0s - loss: 0.1599 - accuracy: 1.0000 - val_loss: 1.0377 - val_accuracy: 0.7000 - 33ms/epoch - 3ms/step
13/13 - 0s - loss: 0.1515 - accuracy: 1.0000 - val_loss: 1.0341 - val_accuracy: 0.6667 - 33ms/epoch - 3ms/step
13/13 - 0s - loss: 0.1426 - accuracy: 1.0000 - val_loss: 1.0291 - val_accuracy: 0.7000 - 46ms/epoch - 4ms/step
13/13 - 0s - loss: 0.1344 - accuracy: 1.0000 - val_loss: 1.0264 - val_accuracy: 0.7000 - 32ms/epoch - 2ms/step
13/13 - 0s - loss: 3.0160 - accuracy: 0.1000 - val_loss: 2.0646 - val_accuracy: 0.2000 - 411ms/epoch - 32ms/step
13/13 - 0s - loss: 2.2510 - accuracy: 0.1800 - val_loss: 1.9801 - val_accuracy: 0.3333 - 48ms/epoch - 4ms/step
13/13 - 0s - loss: 1.7851 - accuracy: 0.4000 - val_loss: 1.8177 - val_accuracy: 0.2667 - 34ms/epoch - 3ms/step
13/13 - 0s - loss: 1.3652 - accuracy: 0.5400 - val_loss: 1.6107 - val_accuracy: 0.4667 - 32ms/epoch - 2ms/step
13/13 - 0s - loss: 0.7383 - accuracy: 0.9400 - val_loss: 1.3536 - val_accuracy: 0.5667 - 32ms/epoch - 2ms/step
13/13 - 0s - loss: 0.5846 - accuracy: 0.9800 - val_loss: 1.2785 - val_accuracy: 0.6667 - 46ms/epoch - 4ms/step
13/13 - 0s - loss: 0.4987 - accuracy: 0.9800 - val_loss: 1.2253 - val_accuracy: 0.6333 - 47ms/epoch - 4ms/step
13/13 - 0s - loss: 0.3921 - accuracy: 1.0000 - val_loss: 1.2655 - val_accuracy: 0.6333 - 32ms/epoch - 2ms/step
13/13 - 0s - loss: 0.3617 - accuracy: 1.0000 - val_loss: 1.1555 - val_accuracy: 0.7000 - 32ms/epoch - 2ms/step
13/13 - 0s - loss: 0.2618 - accuracy: 1.0000 - val_loss: 1.1166 - val_accuracy: 0.7333 - 48ms/epoch - 4ms/step
13/13 - 0s - loss: 0.2463 - accuracy: 1.0000 - val_loss: 1.0624 - val_accuracy: 0.7000 - 34ms/epoch - 3ms/step
13/13 - 0s - loss: 0.2033 - accuracy: 1.0000 - val_loss: 1.0904 - val_accuracy: 0.7333 - 33ms/epoch - 3ms/step
13/13 - 0s - loss: 0.1682 - accuracy: 1.0000 - val_loss: 1.0385 - val_accuracy: 0.7333 - 33ms/epoch - 3ms/step
13/13 - 0s - loss: 0.1484 - accuracy: 1.0000 - val_loss: 1.0676 - val_accuracy: 0.7000 - 48ms/epoch - 4ms/step
13/13 - 0s - loss: 0.1349 - accuracy: 1.0000 - val_loss: 1.0447 - val_accuracy: 0.7000 - 47ms/epoch - 4ms/step
13/13 - 0s - loss: 0.1255 - accuracy: 1.0000 - val_loss: 0.9935 - val_accuracy: 0.7333 - 48ms/epoch - 4ms/step
 </code>
 </pre>
 </details>

---

## 9. tutorial_quantum_chemistry.html <a name="demo8"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_quantum_chemistry.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
  (-46.46390691340522) [I0]
+ (0.7829652070485842) [Z10]
+ (0.7829652070485844) [Z11]
+ (0.8084591005183819) [Z12]
+ (0.8084591005183821) [Z13]
+ (1.2034393391311744) [Z5]
+ (1.2034393391311746) [Z4]
+ (1.3096876618623208) [Z6]
+ (1.3096876618623208) [Z7]
+ (1.3693525711710477) [Z8]
+ (1.3693525711710481) [Z9]
+ (1.653893830547394) [Z3]
+ (1.653893830547395) [Z2]
+ (12.412630714436522) [Z1]
+ (12.412630714436524) [Z0]
+ (-7.95422471060181e-06) [Y11 Y13]
+ (-7.95422471060181e-06) [X11 X13]
+ (-7.76508223335453e-07) [Y3 Y5]
+ (-7.76508223335453e-07) [X3 X5]
+ (5.929280673072461e-07) [Y4 Y6]
+ (5.929280673072461e-07) [X4 X6]
+ (1.6021751134475217e-06) [Y2 Y4]
+ (1.6021751134475217e-06) [X2 X4]
+ (1.8540565561410904e-06) [Y5 Y7]
+ (1.8540565561410904e-06) [X5 X7]
+ (8.194104991636475e-06) [Y10 Y12]
+ (8.194104991636475e-06) [X10 X12]
+ (0.0032769650657487027) [Y1 Y3]
+ (0.0032769650657487027) [X1 X3]
+ (0.10433061485309597) [Y0 Y2]
+ (0.10433061485309597) [X0 X2]
+ (0.11270381859114653) [Z10 Z12]
+ (0.11270381859114653) [Z11 Z13]
+ (0.1138357368529725) [Z4 Z12]
+ (0.1138357368529725) [Z5 Z13]
+ (0.11952441016887765) [Z6 Z10]
+ (0.11952441016887765) [Z7 Z11]
+ (0.12489977362986582) [Z4 Z10]
+ (0.12489977362986582) [Z5 Z11]
+ (0.12495799328196168) [Z2 Z4]
+ (0.12495799328196168) [Z3 Z5]
+ (0.1279949280139447) [Z2 Z10]
+ (0.1279949280139447) [Z3 Z11]
+ (0.1340173737264463) [Z6 Z12]
+ (0.1340173737264463) [Z7 Z13]
+ (0.13701191913053434) [Z4 Z6]
+ (0.13701191913053434) [Z5 Z7]
+ (0.13734942210149265) [Z6 Z11]
+ (0.13734942210149265) [Z7 Z10]
+ (0.13739112375009896) [Z2 Z6]
+ (0.13739112375009896) [Z3 Z7]
+ (0.13766859133621442) [Z8 Z10]
+ (0.13766859133621442) [Z9 Z11]
+ (0.14011294749711087) [Z2 Z12]
+ (0.14011294749711087) [Z3 Z13]
+ (0.1413890359014121) [Z10 Z13]
+ (0.1413890359014121) [Z11 Z12]
+ (0.14257991128698025) [Z4 Z11]
+ (0.14257991128698025) [Z5 Z10]
+ (0.1472293078326632) [Z8 Z11]
+ (0.1472293078326632) [Z9 Z10]
+ (0.148994261713793) [Z4 Z7]
+ (0.148994261713793) [Z5 Z6]
+ (0.14926347060033884) [Z10 Z11]
+ (0.14960692557255753) [Z4 Z8]
+ (0.14960692557255753) [Z5 Z9]
+ (0.14973497005437506) [Z8 Z12]
+ (0.14973497005437506) [Z9 Z13]
+ (0.15071405482302613) [Z2 Z8]
+ (0.15071405482302613) [Z3 Z9]
+ (0.15138342699106816) [Z6 Z13]
+ (0.15138342699106816) [Z7 Z12]
+ (0.15215040622641263) [Z4 Z13]
+ (0.15215040622641263) [Z5 Z12]
+ (0.15337959171059234) [Z2 Z11]
+ (0.15337959171059234) [Z3 Z10]
+ (0.15435760065295295) [Z12 Z13]
+ (0.15569017457256934) [Z2 Z13]
+ (0.15569017457256934) [Z3 Z12]
+ (0.15582280685860975) [Z8 Z13]
+ (0.15582280685860975) [Z9 Z12]
+ (0.1567638461017493) [Z4 Z9]
+ (0.1567638461017493) [Z5 Z8]
+ (0.1575530380435019) [Z4 Z5]
+ (0.1607975504669914) [Z2 Z5]
+ (0.1607975504669914) [Z3 Z4]
+ (0.1675666935617351) [Z6 Z8]
+ (0.1675666935617351) [Z7 Z9]
+ (0.16853492794646802) [Z2 Z7]
+ (0.16853492794646802) [Z3 Z6]
+ (0.18144009362936325) [Z6 Z9]
+ (0.18144009362936325) [Z7 Z8]
+ (0.18189081243745908) [Z2 Z3]
+ (0.18690814831056377) [Z2 Z9]
+ (0.18690814831056377) [Z3 Z8]
+ (0.19299700269853698) [Z0 Z10]
+ (0.19299700269853698) [Z1 Z11]
+ (0.1939257433496167) [Z6 Z7]
+ (0.19661749959722186) [Z0 Z4]
+ (0.19661749959722186) [Z1 Z5]
+ (0.19936332691264544) [Z0 Z5]
+ (0.19936332691264544) [Z1 Z4]
+ (0.20072843554589154) [Z0 Z11]
+ (0.20072843554589154) [Z1 Z10]
+ (0.21102681234274556) [Z0 Z12]
+ (0.21102681234274556) [Z1 Z13]
+ (0.21631059809657172) [Z0 Z13]
+ (0.21631059809657172) [Z1 Z12]
+ (0.22003977240299175) [Z8 Z9]
+ (0.2367107174042222) [Z0 Z2]
+ (0.2367107174042222) [Z1 Z3]
+ (0.2416469683172905) [Z0 Z6]
+ (0.2416469683172905) [Z1 Z7]
+ (0.2485351728596039) [Z0 Z7]
+ (0.2485351728596039) [Z1 Z6]
+ (0.2512943557312408) [Z0 Z3]
+ (0.2512943557312408) [Z1 Z2]
+ (0.2723251845038348) [Z0 Z8]
+ (0.2723251845038348) [Z1 Z9]
+ (0.2788345457379162) [Z0 Z9]
+ (0.2788345457379162) [Z1 Z8]
+ (1.1861764484126038) [Z0 Z1]
+ (-3.8866396090695755e-06) [Y2 Z3 Y4]
+ (-3.8866396090695755e-06) [X2 Z3 X4]
+ (-3.886639609069575e-06) [Y3 Z4 Y5]
+ (-3.886639609069575e-06) [X3 Z4 X5]
+ (1.0722748307982262e-05) [Y10 Z11 Y12]
+ (1.0722748307982262e-05) [X10 Z11 X12]
+ (1.0722748307982262e-05) [Y11 Z12 Y13]
+ (1.0722748307982262e-05) [X11 Z12 X13]
+ (1.2260277027681307e-05) [Y4 Z5 Y6]
+ (1.2260277027681307e-05) [X4 Z5 X6]
+ (1.2260277027681307e-05) [Y5 Z6 Y7]
+ (1.2260277027681307e-05) [X5 Z6 X7]
+ (0.125070368839738) [Y0 Z1 Y2]
+ (0.125070368839738) [X0 Z1 X2]
+ (0.12507036883973802) [Y1 Z2 Y3]
+ (0.12507036883973802) [X1 Z2 X3]
+ (-0.03831466937344012) [Y4 Y5 X12 X13]
+ (-0.03831466937344012) [X4 X5 Y12 Y13]
+ (-0.036194093487537646) [Y2 Y3 X8 X9]
+ (-0.036194093487537646) [X2 X3 Y8 Y9]
+ (-0.035839557185029736) [Y2 Y3 X4 X5]
+ (-0.035839557185029736) [X2 X3 Y4 Y5]
+ (-0.03114380419636905) [Y2 Y3 X6 X7]
+ (-0.03114380419636905) [X2 X3 Y6 Y7]
+ (-0.028685217310265555) [Y10 Y11 X12 X13]
+ (-0.028685217310265555) [X10 X11 Y12 Y13]
+ (-0.025384663696647613) [Y2 Y3 X10 X11]
+ (-0.025384663696647613) [X2 X3 Y10 Y11]
+ (-0.019028318718278044) [Y3 X4 X11 Y12]
+ (-0.019028318718278044) [X3 Y4 Y11 X12]
+ (-0.017825011932615018) [Y6 Y7 X10 X11]
+ (-0.017825011932615018) [X6 X7 Y10 Y11]
+ (-0.017680137657114424) [Y4 Y5 X10 X11]
+ (-0.017680137657114424) [X4 X5 Y10 Y11]
+ (-0.017366053264621886) [Y6 Y7 X12 X13]
+ (-0.017366053264621886) [X6 X7 Y12 Y13]
+ (-0.015577227075458482) [Y2 Y3 X12 X13]
+ (-0.015577227075458482) [X2 X3 Y12 Y13]
+ (-0.014583638327018651) [Y0 Y1 X2 X3]
+ (-0.014583638327018651) [X0 X1 Y2 Y3]
+ (-0.013873400067628133) [Y6 Y7 X8 X9]
+ (-0.013873400067628133) [X6 X7 Y8 Y9]
+ (-0.011982342583258684) [Y4 Y5 X6 X7]
+ (-0.011982342583258684) [X4 X5 Y6 Y7]
+ (-0.011285144618318447) [Y5 X6 X11 Y12]
+ (-0.011285144618318447) [X5 Y6 Y11 X12]
+ (-0.009560716496448757) [Y8 Y9 X10 X11]
+ (-0.009560716496448757) [X8 X9 Y10 Y11]
+ (-0.008125248410131977) [Y1 X2 X8 Y9]
+ (-0.008125248410131977) [Y1 Y2 Y8 Y9]
+ (-0.008125248410131977) [X1 X2 X8 X9]
+ (-0.008125248410131977) [X1 Y2 Y8 X9]
+ (-0.00773143284735457) [Y0 Y1 X10 X11]
+ (-0.00773143284735457) [X0 X1 Y10 Y11]
+ (-0.007156920529191778) [Y4 Y5 X8 X9]
+ (-0.007156920529191778) [X4 X5 Y8 Y9]
+ (-0.006888204542313399) [Y0 Y1 X6 X7]
+ (-0.006888204542313399) [X0 X1 Y6 Y7]
+ (-0.006509361234081384) [Y0 Y1 X8 X9]
+ (-0.006509361234081384) [X0 X1 Y8 Y9]
+ (-0.006087836804234693) [Y8 Y9 X12 X13]
+ (-0.006087836804234693) [X8 X9 Y12 Y13]
+ (-0.005283785753826158) [Y0 Y1 X12 X13]
+ (-0.005283785753826158) [X0 X1 Y12 Y13]
+ (-0.005143382387697844) [Y3 Y4 X5 X6]
+ (-0.005143382387697844) [X3 X4 Y5 Y6]
+ (-0.0046849202268703346) [Y1 X2 X6 Y7]
+ (-0.0046849202268703346) [Y1 Y2 Y6 Y7]
+ (-0.0046849202268703346) [X1 X2 X6 X7]
+ (-0.0046849202268703346) [X1 Y2 Y6 X7]
+ (-0.0045750151888949145) [Y1 X2 X12 Y13]
+ (-0.0045750151888949145) [Y1 Y2 Y12 Y13]
+ (-0.0045750151888949145) [X1 X2 X12 X13]
+ (-0.0045750151888949145) [X1 Y2 Y12 X13]
+ (-0.004424843668499126) [Y1 X2 X4 Y5]
+ (-0.004424843668499126) [Y1 Y2 Y4 Y5]
+ (-0.004424843668499126) [X1 X2 X4 X5]
+ (-0.004424843668499126) [X1 Y2 Y4 X5]
+ (-0.002745827315423585) [Y0 Y1 X4 X5]
+ (-0.002745827315423585) [X0 X1 Y4 Y5]
+ (-0.0017991930083861738) [Y1 X2 X10 Y11]
+ (-0.0017991930083861738) [Y1 Y2 Y10 Y11]
+ (-0.0017991930083861738) [X1 X2 X10 X11]
+ (-0.0017991930083861738) [X1 Y2 Y10 X11]
+ (-0.0016639606583792057) [Y2 Z3 Z4 Y6]
+ (-0.0016639606583792057) [X2 Z3 Z4 X6]
+ (-0.0016639606583792057) [Y3 Z5 Z6 Y7]
+ (-0.0016639606583792057) [X3 Z5 Z6 X7]
+ (-0.0004957972885579184) [Y2 Z4 Z5 Y6]
+ (-0.0004957972885579184) [X2 Z4 Z5 X6]
+ (-0.000292225672451927) [Y7 Y8 X9 X10]
+ (-0.000292225672451927) [X7 X8 Y9 Y10]
+ (-7.95422471060181e-06) [Y10 Z11 Y12 Z13]
+ (-7.95422471060181e-06) [X10 Z11 X12 Z13]
+ (-7.801538383020445e-06) [Y2 Z3 Y4 Z11]
+ (-7.801538383020445e-06) [X2 Z3 X4 Z11]
+ (-7.801538383020445e-06) [Y3 Z4 Y5 Z10]
+ (-7.801538383020445e-06) [X3 Z4 X5 Z10]
+ (-5.974176903285593e-06) [Y5 X6 X10 Y11]
+ (-5.974176903285593e-06) [Y5 Y6 Y10 Y11]
+ (-5.974176903285593e-06) [X5 X6 X10 X11]
+ (-5.974176903285593e-06) [X5 Y6 Y10 X11]
+ (-4.642978995234286e-06) [Y3 X4 X10 Y11]
+ (-4.642978995234286e-06) [Y3 Y4 Y10 Y11]
+ (-4.642978995234286e-06) [X3 X4 X10 X11]
+ (-4.642978995234286e-06) [X3 Y4 Y10 X11]
+ (-4.28181205790884e-06) [Y4 Z5 Y6 Z11]
+ (-4.28181205790884e-06) [X4 Z5 X6 Z11]
+ (-4.28181205790884e-06) [Y5 Z6 Y7 Z10]
+ (-4.28181205790884e-06) [X5 Z6 X7 Z10]
+ (-3.158559387786159e-06) [Y2 Z3 Y4 Z10]
+ (-3.158559387786159e-06) [X2 Z3 X4 Z10]
+ (-3.158559387786159e-06) [Y3 Z4 Y5 Z11]
+ (-3.158559387786159e-06) [X3 Z4 X5 Z11]
+ (-2.172638029106916e-06) [Y2 X3 X11 Y12]
+ (-2.172638029106916e-06) [Y2 Y3 Y11 Y12]
+ (-2.172638029106916e-06) [X2 X3 X11 X12]
+ (-2.172638029106916e-06) [X2 Y3 Y11 X12]
+ (-1.8781495552330774e-06) [Z4 Y10 Z11 Y12]
+ (-1.8781495552330774e-06) [Z4 X10 Z11 X12]
+ (-1.8781495552330774e-06) [Z5 Y11 Z12 Y13]
+ (-1.8781495552330774e-06) [Z5 X11 Z12 X13]
+ (-1.4548067005396102e-06) [Y3 X4 X6 Y7]
+ (-1.4548067005396102e-06) [Y3 Y4 Y6 Y7]
+ (-1.4548067005396102e-06) [X3 X4 X6 X7]
+ (-1.4548067005396102e-06) [X3 Y4 Y6 X7]
+ (-1.1953920855127705e-06) [Y2 Z3 Y4 Z7]
+ (-1.1953920855127705e-06) [X2 Z3 X4 Z7]
+ (-1.1953920855127705e-06) [Y3 Z4 Y5 Z6]
+ (-1.1953920855127705e-06) [X3 Z4 X5 Z6]
+ (-1.1907331173370887e-06) [Z0 Y3 Z4 Y5]
+ (-1.1907331173370887e-06) [Z0 X3 Z4 X5]
+ (-1.1907331173370887e-06) [Z1 Y2 Z3 Y4]
+ (-1.1907331173370887e-06) [Z1 X2 Z3 X4]
+ (-1.10941252276682e-06) [Z2 Y11 Z12 Y13]
+ (-1.10941252276682e-06) [Z2 X11 Z12 X13]
+ (-1.10941252276682e-06) [Z3 Y10 Z11 Y12]
+ (-1.10941252276682e-06) [Z3 X10 Z11 X12]
+ (-8.33669564825354e-07) [Z0 Y2 Z3 Y4]
+ (-8.33669564825354e-07) [Z0 X2 Z3 X4]
+ (-8.33669564825354e-07) [Z1 Y3 Z4 Y5]
+ (-8.33669564825354e-07) [Z1 X3 Z4 X5]
+ (-7.956667024294647e-07) [Y3 X4 X8 Y9]
+ (-7.956667024294647e-07) [Y3 Y4 Y8 Y9]
+ (-7.956667024294647e-07) [X3 X4 X8 X9]
+ (-7.956667024294647e-07) [X3 Y4 Y8 X9]
+ (-7.76508223335453e-07) [Y2 Z3 Y4 Z5]
+ (-7.76508223335453e-07) [X2 Z3 X4 Z5]
+ (-6.628427303686851e-07) [Y8 X9 X11 Y12]
+ (-6.628427303686851e-07) [Y8 Y9 Y11 Y12]
+ (-6.628427303686851e-07) [X8 X9 X11 X12]
+ (-6.628427303686851e-07) [X8 Y9 Y11 X12]
+ (-5.769436622980045e-07) [Y2 Z3 Y4 Z9]
+ (-5.769436622980045e-07) [X2 Z3 X4 Z9]
+ (-5.769436622980045e-07) [Y3 Z4 Y5 Z8]
+ (-5.769436622980045e-07) [X3 Z4 X5 Z8]
+ (-5.627722054083482e-07) [Y0 X1 X11 Y12]
+ (-5.627722054083482e-07) [Y0 Y1 Y11 Y12]
+ (-5.627722054083482e-07) [X0 X1 X11 X12]
+ (-5.627722054083482e-07) [X0 Y1 Y11 X12]
+ (-5.471606065934441e-07) [Y1 X2 X11 Y12]
+ (-5.471606065934441e-07) [X1 Y2 Y11 X12]
+ (-3.5706355251168246e-07) [Y0 X1 X3 Y4]
+ (-3.5706355251168246e-07) [Y0 Y1 Y3 Y4]
+ (-3.5706355251168246e-07) [X0 X1 X3 X4]
+ (-3.5706355251168246e-07) [X0 Y1 Y3 X4]
+ (-1.933212146127449e-07) [Y1 X2 X3 Y4]
+ (-1.933212146127449e-07) [X1 Y2 Y3 X4]
+ (-1.2919458298947298e-07) [Y1 Z2 Z3 Y5]
+ (-1.2919458298947298e-07) [X1 Z2 Z3 X5]
+ (-3.2261050206086977e-09) [Y1 Y2 X5 X6]
+ (-3.2261050206086977e-09) [X1 X2 Y5 Y6]
+ (3.2261050206086977e-09) [Y1 X2 X5 Y6]
+ (3.2261050206086977e-09) [X1 Y2 Y5 X6]
+ (3.228404718731703e-08) [Y4 Z5 Y6 Z12]
+ (3.228404718731703e-08) [X4 Z5 X6 Z12]
+ (3.228404718731703e-08) [Y5 Z6 Y7 Z13]
+ (3.228404718731703e-08) [X5 Z6 X7 Z13]
+ (1.6728571581176803e-07) [Y0 Z1 Z3 Y4]
+ (1.6728571581176803e-07) [X0 Z1 Z3 X4]
+ (1.6728571581176803e-07) [Y1 Z2 Z4 Y5]
+ (1.6728571581176803e-07) [X1 Z2 Z4 X5]
+ (1.933212146127449e-07) [Y1 Y2 X3 X4]
+ (1.933212146127449e-07) [X1 X2 Y3 Y4]
+ (2.187230401314602e-07) [Y2 Z3 Y4 Z8]
+ (2.187230401314602e-07) [X2 Z3 X4 Z8]
+ (2.187230401314602e-07) [Y3 Z4 Y5 Z9]
+ (2.187230401314602e-07) [X3 Z4 X5 Z9]
+ (2.1989637768272475e-07) [Y2 X3 X5 Y6]
+ (2.1989637768272475e-07) [Y2 Y3 Y5 Y6]
+ (2.1989637768272475e-07) [X2 X3 X5 X6]
+ (2.1989637768272475e-07) [X2 Y3 Y5 X6]
+ (2.447264832973169e-07) [Y0 X1 X5 Y6]
+ (2.447264832973169e-07) [Y0 Y1 Y5 Y6]
+ (2.447264832973169e-07) [X0 X1 X5 X6]
+ (2.447264832973169e-07) [X0 Y1 Y5 X6]
+ (2.5941461502819495e-07) [Y2 Z3 Y4 Z6]
+ (2.5941461502819495e-07) [X2 Z3 X4 Z6]
+ (2.5941461502819495e-07) [Y3 Z4 Y5 Z7]
+ (2.5941461502819495e-07) [X3 Z4 X5 Z7]
+ (3.6060693042456376e-07) [Y0 Z1 Z2 Y4]
+ (3.6060693042456376e-07) [X0 Z1 Z2 X4]
+ (3.6060693042456376e-07) [Y1 Z3 Z4 Y5]
+ (3.6060693042456376e-07) [X1 Z3 Z4 X5]
+ (4.837953374670979e-07) [Y5 X6 X8 Y9]
+ (4.837953374670979e-07) [Y5 Y6 Y8 Y9]
+ (4.837953374670979e-07) [X5 X6 X8 X9]
+ (4.837953374670979e-07) [X5 Y6 Y8 X9]
+ (5.471606065934441e-07) [Y1 Y2 X11 X12]
+ (5.471606065934441e-07) [X1 X2 Y11 Y12]
+ (5.929280673072461e-07) [Z4 Y5 Z6 Y7]
+ (5.929280673072461e-07) [Z4 X5 Z6 X7]
+ (9.344969841944273e-07) [Z8 Y11 Z12 Y13]
+ (9.344969841944273e-07) [Z8 X11 Z12 X13]
+ (9.344969841944273e-07) [Z9 Y10 Z11 Y12]
+ (9.344969841944273e-07) [Z9 X10 Z11 X12]
+ (9.509134610911874e-07) [Z2 Y4 Z5 Y6]
+ (9.509134610911874e-07) [Z2 X4 Z5 X6]
+ (9.509134610911874e-07) [Z3 Y5 Z6 Y7]
+ (9.509134610911874e-07) [Z3 X5 Z6 X7]
+ (1.0357924863529412e-06) [Y6 X7 X11 Y12]
+ (1.0357924863529412e-06) [Y6 Y7 Y11 Y12]
+ (1.0357924863529412e-06) [X6 X7 X11 X12]
+ (1.0357924863529412e-06) [X6 Y7 Y11 X12]
+ (1.0632255063368432e-06) [Z2 Y10 Z11 Y12]
+ (1.0632255063368432e-06) [Z2 X10 Z11 X12]
+ (1.0632255063368432e-06) [Z3 Y11 Z12 Y13]
+ (1.0632255063368432e-06) [Z3 X11 Z12 X13]
+ (1.1708098387736411e-06) [Z2 Y5 Z6 Y7]
+ (1.1708098387736411e-06) [Z2 X5 Z6 X7]
+ (1.1708098387736411e-06) [Z3 Y4 Z5 Y6]
+ (1.1708098387736411e-06) [Z3 X4 Z5 X6]
+ (1.398024307783897e-06) [Y4 Z5 Y6 Z8]
+ (1.398024307783897e-06) [X4 Z5 X6 Z8]
+ (1.398024307783897e-06) [Y5 Z6 Y7 Z9]
+ (1.398024307783897e-06) [X5 Z6 X7 Z9]
+ (1.5973397145631124e-06) [Z8 Y10 Z11 Y12]
+ (1.5973397145631124e-06) [Z8 X10 Z11 X12]
+ (1.5973397145631124e-06) [Z9 Y11 Z12 Y13]
+ (1.5973397145631124e-06) [Z9 X11 Z12 X13]
+ (1.6021751134475217e-06) [Z2 Y3 Z4 Y5]
+ (1.6021751134475217e-06) [Z2 X3 Z4 X5]
+ (1.6149607753844156e-06) [Z0 Y11 Z12 Y13]
+ (1.6149607753844156e-06) [Z0 X11 Z12 X13]
+ (1.6149607753844156e-06) [Z1 Y10 Z11 Y12]
+ (1.6149607753844156e-06) [Z1 X10 Z11 X12]
+ (1.6923648453776204e-06) [Y4 Z5 Y6 Z10]
+ (1.6923648453776204e-06) [X4 Z5 X6 Z10]
+ (1.6923648453776204e-06) [Y5 Z6 Y7 Z11]
+ (1.6923648453776204e-06) [X5 Z6 X7 Z11]
+ (1.8163673528477642e-06) [Z4 Y11 Z12 Y13]
+ (1.8163673528477642e-06) [Z4 X11 Z12 X13]
+ (1.8163673528477642e-06) [Z5 Y10 Z11 Y12]
+ (1.8163673528477642e-06) [Z5 X10 Z11 X12]
+ (1.8540565561410904e-06) [Y4 Z5 Y6 Z7]
+ (1.8540565561410904e-06) [X4 Z5 X6 Z7]
+ (1.8551374645248316e-06) [Z6 Y10 Z11 Y12]
+ (1.8551374645248316e-06) [Z6 X10 Z11 X12]
+ (1.8551374645248316e-06) [Z7 Y11 Z12 Y13]
+ (1.8551374645248316e-06) [Z7 X11 Z12 X13]
+ (1.8818196452509948e-06) [Y4 Z5 Y6 Z9]
+ (1.8818196452509948e-06) [X4 Z5 X6 Z9]
+ (1.8818196452509948e-06) [Y5 Z6 Y7 Z8]
+ (1.8818196452509948e-06) [X5 Z6 X7 Z8]
+ (2.1777329807928983e-06) [Z0 Y10 Z11 Y12]
+ (2.1777329807928983e-06) [Z0 X10 Z11 X12]
+ (2.1777329807928983e-06) [Z1 Y11 Z12 Y13]
+ (2.1777329807928983e-06) [Z1 X11 Z12 X13]
+ (2.890929950879074e-06) [Z6 Y11 Z12 Y13]
+ (2.890929950879074e-06) [Z6 X11 Z12 X13]
+ (2.890929950879074e-06) [Z7 Y10 Z11 Y12]
+ (2.890929950879074e-06) [Z7 X10 Z11 X12]
+ (3.0992966713922013e-06) [Z0 Y4 Z5 Y6]
+ (3.0992966713922013e-06) [Z0 X4 Z5 X6]
+ (3.0992966713922013e-06) [Z1 Y5 Z6 Y7]
+ (3.0992966713922013e-06) [Z1 X5 Z6 X7]
+ (3.117366419074471e-06) [Y0 Z2 Z3 Y4]
+ (3.117366419074471e-06) [X0 Z2 Z3 X4]
+ (3.3440231546895615e-06) [Z0 Y5 Z6 Y7]
+ (3.3440231546895615e-06) [Z0 X5 Z6 X7]
+ (3.3440231546895615e-06) [Z1 Y4 Z5 Y6]
+ (3.3440231546895615e-06) [Z1 X4 Z5 X6]
+ (3.5390100060167344e-06) [Y2 Z3 Y4 Z12]
+ (3.5390100060167344e-06) [X2 Z3 X4 Z12]
+ (3.5390100060167344e-06) [Y3 Z4 Y5 Z13]
+ (3.5390100060167344e-06) [X3 Z4 X5 Z13]
+ (3.694516908079974e-06) [Y4 X5 X11 Y12]
+ (3.694516908079974e-06) [Y4 Y5 Y11 Y12]
+ (3.694516908079974e-06) [X4 X5 X11 X12]
+ (3.694516908079974e-06) [X4 Y5 Y11 X12]
+ (4.5564737840833244e-06) [Y5 X6 X12 Y13]
+ (4.5564737840833244e-06) [Y5 Y6 Y12 Y13]
+ (4.5564737840833244e-06) [X5 X6 X12 X13]
+ (4.5564737840833244e-06) [X5 Y6 Y12 X13]
+ (4.5887578312706415e-06) [Y4 Z5 Y6 Z13]
+ (4.5887578312706415e-06) [X4 Z5 X6 Z13]
+ (4.5887578312706415e-06) [Y5 Z6 Y7 Z12]
+ (4.5887578312706415e-06) [X5 Z6 X7 Z12]
+ (5.275783473266196e-06) [Y3 X4 X12 Y13]
+ (5.275783473266196e-06) [Y3 Y4 Y12 Y13]
+ (5.275783473266196e-06) [X3 X4 X12 X13]
+ (5.275783473266196e-06) [X3 Y4 Y12 X13]
+ (8.194104991636475e-06) [Z10 Y11 Z12 Y13]
+ (8.194104991636475e-06) [Z10 X11 Z12 X13]
+ (8.81479347928293e-06) [Y2 Z3 Y4 Z13]
+ (8.81479347928293e-06) [X2 Z3 X4 Z13]
+ (8.81479347928293e-06) [Y3 Z4 Y5 Z12]
+ (8.81479347928293e-06) [X3 Z4 X5 Z12]
+ (0.000292225672451927) [Y7 X8 X9 Y10]
+ (0.000292225672451927) [X7 Y8 Y9 X10]
+ (0.0011058984808954508) [Y0 Z1 Y2 Z5]
+ (0.0011058984808954508) [X0 Z1 X2 Z5]
+ (0.0011058984808954508) [Y1 Z2 Y3 Z4]
+ (0.0011058984808954508) [X1 Z2 X3 Z4]
+ (0.0017560659628748875) [Y0 Z1 Y2 Z11]
+ (0.0017560659628748875) [X0 Z1 X2 Z11]
+ (0.0017560659628748875) [Y1 Z2 Y3 Z10]
+ (0.0017560659628748875) [X1 Z2 X3 Z10]
+ (0.002326234847606732) [Y0 Z1 Y2 Z13]
+ (0.002326234847606732) [X0 Z1 X2 Z13]
+ (0.002326234847606732) [Y1 Z2 Y3 Z12]
+ (0.002326234847606732) [X1 Z2 X3 Z12]
+ (0.002745827315423585) [Y0 X1 X4 Y5]
+ (0.002745827315423585) [X0 Y1 Y4 X5]
+ (0.002929768278581767) [Y0 Z1 Y2 Z9]
+ (0.002929768278581767) [X0 Z1 X2 Z9]
+ (0.002929768278581767) [Y1 Z2 Y3 Z8]
+ (0.002929768278581767) [X1 Z2 X3 Z8]
+ (0.003276965065748703) [Y0 Z1 Y2 Z3]
+ (0.003276965065748703) [X0 Z1 X2 Z3]
+ (0.00334762647068453) [Y0 Z1 Y2 Z7]
+ (0.00334762647068453) [X0 Z1 X2 Z7]
+ (0.00334762647068453) [Y1 Z2 Y3 Z6]
+ (0.00334762647068453) [X1 Z2 X3 Z6]
+ (0.003479421729318637) [Y2 Z3 Z5 Y6]
+ (0.003479421729318637) [X2 Z3 Z5 X6]
+ (0.003479421729318637) [Y3 Z4 Z6 Y7]
+ (0.003479421729318637) [X3 Z4 Z6 X7]
+ (0.003555258971261062) [Y0 Z1 Y2 Z10]
+ (0.003555258971261062) [X0 Z1 X2 Z10]
+ (0.003555258971261062) [Y1 Z2 Y3 Z11]
+ (0.003555258971261062) [X1 Z2 X3 Z11]
+ (0.005143382387697844) [Y3 X4 X5 Y6]
+ (0.005143382387697844) [X3 Y4 Y5 X6]
+ (0.005283785753826158) [Y0 X1 X12 Y13]
+ (0.005283785753826158) [X0 Y1 Y12 X13]
+ (0.005530742149394577) [Y0 Z1 Y2 Z4]
+ (0.005530742149394577) [X0 Z1 X2 Z4]
+ (0.005530742149394577) [Y1 Z2 Y3 Z5]
+ (0.005530742149394577) [X1 Z2 X3 Z5]
+ (0.006087836804234693) [Y8 X9 X12 Y13]
+ (0.006087836804234693) [X8 Y9 Y12 X13]
+ (0.006509361234081384) [Y0 X1 X8 Y9]
+ (0.006509361234081384) [X0 Y1 Y8 X9]
+ (0.006888204542313399) [Y0 X1 X6 Y7]
+ (0.006888204542313399) [X0 Y1 Y6 X7]
+ (0.006901250036501646) [Y0 Z1 Y2 Z12]
+ (0.006901250036501646) [X0 Z1 X2 Z12]
+ (0.006901250036501646) [Y1 Z2 Y3 Z13]
+ (0.006901250036501646) [X1 Z2 X3 Z13]
+ (0.007156920529191778) [Y4 X5 X8 Y9]
+ (0.007156920529191778) [X4 Y5 Y8 X9]
+ (0.00773143284735457) [Y0 X1 X10 Y11]
+ (0.00773143284735457) [X0 Y1 Y10 X11]
+ (0.008032546697554864) [Y0 Z1 Y2 Z6]
+ (0.008032546697554864) [X0 Z1 X2 Z6]
+ (0.008032546697554864) [Y1 Z2 Y3 Z7]
+ (0.008032546697554864) [X1 Z2 X3 Z7]
+ (0.009560716496448757) [Y8 X9 X10 Y11]
+ (0.009560716496448757) [X8 Y9 Y10 X11]
+ (0.011055016688713743) [Y0 Z1 Y2 Z8]
+ (0.011055016688713743) [X0 Z1 X2 Z8]
+ (0.011055016688713743) [Y1 Z2 Y3 Z9]
+ (0.011055016688713743) [X1 Z2 X3 Z9]
+ (0.011285144618318447) [Y5 Y6 X11 X12]
+ (0.011285144618318447) [X5 X6 Y11 Y12]
+ (0.011307208029975837) [Y7 Z8 Z9 Y11]
+ (0.011307208029975837) [X7 Z8 Z9 X11]
+ (0.011982342583258684) [Y4 X5 X6 Y7]
+ (0.011982342583258684) [X4 Y5 Y6 X7]
+ (0.013873400067628133) [Y6 X7 X8 Y9]
+ (0.013873400067628133) [X6 Y7 Y8 X9]
+ (0.014583638327018651) [Y0 X1 X2 Y3]
+ (0.014583638327018651) [X0 Y1 Y2 X3]
+ (0.015577227075458482) [Y2 X3 X12 Y13]
+ (0.015577227075458482) [X2 Y3 Y12 X13]
+ (0.017366053264621886) [Y6 X7 X12 Y13]
+ (0.017366053264621886) [X6 Y7 Y12 X13]
+ (0.017680137657114424) [Y4 X5 X10 Y11]
+ (0.017680137657114424) [X4 Y5 Y10 X11]
+ (0.017825011932615018) [Y6 X7 X10 Y11]
+ (0.017825011932615018) [X6 Y7 Y10 X11]
+ (0.019028318718278044) [Y3 Y4 X11 X12]
+ (0.019028318718278044) [X3 X4 Y11 Y12]
+ (0.025384663696647613) [Y2 X3 X10 Y11]
+ (0.025384663696647613) [X2 Y3 Y10 X11]
+ (0.025996206267188907) [Y3 Z4 Z5 Y7]
+ (0.025996206267188907) [X3 Z4 Z5 X7]
+ (0.028685217310265555) [Y10 X11 X12 Y13]
+ (0.028685217310265555) [X10 Y11 Y12 X13]
+ (0.029812299601086244) [Y6 Z7 Z8 Y10]
+ (0.029812299601086244) [X6 Z7 Z8 X10]
+ (0.029812299601086244) [Y7 Z9 Z10 Y11]
+ (0.029812299601086244) [X7 Z9 Z10 X11]
+ (0.03010452527353817) [Y6 Z7 Z9 Y10]
+ (0.03010452527353817) [X6 Z7 Z9 X10]
+ (0.03010452527353817) [Y7 Z8 Z10 Y11]
+ (0.03010452527353817) [X7 Z8 Z10 X11]
+ (0.0307874407185145) [Y6 Z8 Z9 Y10]
+ (0.0307874407185145) [X6 Z8 Z9 X10]
+ (0.03114380419636905) [Y2 X3 X6 Y7]
+ (0.03114380419636905) [X2 Y3 Y6 X7]
+ (0.035839557185029736) [Y2 X3 X4 Y5]
+ (0.035839557185029736) [X2 Y3 Y4 X5]
+ (0.036194093487537646) [Y2 X3 X8 Y9]
+ (0.036194093487537646) [X2 Y3 Y8 X9]
+ (0.03831466937344012) [Y4 X5 X12 Y13]
+ (0.03831466937344012) [X4 Y5 Y12 X13]
+ (0.10433061485309596) [Z0 Y1 Z2 Y3]
+ (0.10433061485309596) [Z0 X1 Z2 X3]
+ (3.2041422572821185e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.2041422572821185e-06) [X0 Z1 Z2 Z3 X4]
+ (3.2041422572821185e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.2041422572821185e-06) [X1 Z2 Z3 Z4 X5]
+ (0.12133242248380338) [Y3 Z4 Z5 Z6 Y7]
+ (0.12133242248380338) [X3 Z4 Z5 Z6 X7]
+ (0.12133242248380341) [Y2 Z3 Z4 Z5 Y6]
+ (0.12133242248380341) [X2 Z3 Z4 Z5 X6]
+ (0.22847946311012254) [Y7 Z8 Z9 Z10 Y11]
+ (0.22847946311012254) [X7 Z8 Z9 Z10 X11]
+ (0.22847946311012257) [Y6 Z7 Z8 Z9 Y10]
+ (0.22847946311012257) [X6 Z7 Z8 Z9 X10]
+ (-0.045879424030657236) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (-0.045879424030657236) [X0 Z2 Z3 Z4 Z5 X6]
+ (-0.024353136084491928) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (-0.024353136084491928) [Y2 Z3 Y4 X11 Z12 X13]
+ (-0.024353136084491928) [X2 Z3 X4 Y11 Z12 Y13]
+ (-0.024353136084491928) [X2 Z3 X4 X11 Z12 X13]
+ (-0.024353136084491928) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (-0.024353136084491928) [Y3 Z4 Y5 X10 Z11 X12]
+ (-0.024353136084491928) [X3 Z4 X5 Y10 Z11 Y12]
+ (-0.024353136084491928) [X3 Z4 X5 X10 Z11 X12]
+ (-0.015588277865107775) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (-0.015588277865107775) [X2 Z3 X4 X10 Z11 X12]
+ (-0.015588277865107775) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (-0.015588277865107775) [X3 Z4 X5 X11 Z12 X13]
+ (-0.015225659057108532) [Y3 Z4 Z5 X6 X10 Y11]
+ (-0.015225659057108532) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (-0.015225659057108532) [X3 Z4 Z5 X6 X10 X11]
+ (-0.015225659057108532) [X3 Z4 Z5 Y6 Y10 X11]
+ (-0.014564473640811556) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.014564473640811556) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.014564473640811556) [X7 Z8 Z9 X10 X12 X13]
+ (-0.014564473640811556) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.014411189770045546) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (-0.014411189770045546) [X2 Z3 Z4 Z5 X6 Z11]
+ (-0.014411189770045546) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (-0.014411189770045546) [X3 Z4 Z5 Z6 X7 Z10]
+ (-0.012214985322762559) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.012214985322762559) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.012214985322762559) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.012214985322762559) [X4 Z5 X6 X11 Z12 X13]
+ (-0.012214985322762559) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.012214985322762559) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.012214985322762559) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.012214985322762559) [X5 Z6 X7 X10 Z11 X12]
+ (-0.010263460498893894) [Y2 Z3 X4 X10 Z11 Y12]
+ (-0.010263460498893894) [X2 Z3 Y4 Y10 Z11 X12]
+ (-0.010263460498893894) [Y3 Z4 X5 X11 Z12 Y13]
+ (-0.010263460498893894) [X3 Z4 Y5 Y11 Z12 X13]
+ (-0.008125248410131975) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.008125248410131975) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.007306763969606398) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.007306763969606398) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.007306763969606398) [X4 X5 X7 Z8 Z9 X10]
+ (-0.007306763969606398) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.005324817366213881) [Y2 Z3 Y4 X10 Z11 X12]
+ (-0.005324817366213881) [X2 Z3 X4 Y10 Z11 Y12]
+ (-0.005324817366213881) [Y3 Z4 Y5 X11 Z12 X13]
+ (-0.005324817366213881) [X3 Z4 X5 Y11 Z12 Y13]
+ (-0.004684920226870334) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.004684920226870334) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668615266018511) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.004668615266018511) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.0045750151888949145) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.0045750151888949145) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.004424843668499126) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.004424843668499126) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.00396156937303177) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (-0.00396156937303177) [X0 Z1 Z2 Z4 Z5 X6]
+ (-0.00396156937303177) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (-0.00396156937303177) [X1 Z3 Z4 Z5 Z6 X7]
+ (-0.002462916621394921) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (-0.002462916621394921) [X0 Z1 Z2 Z3 Z5 X6]
+ (-0.002462916621394921) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (-0.002462916621394921) [X1 Z2 Z3 Z4 Z6 X7]
+ (-0.002293955623064501) [Y1 Y2 X3 Z4 Z5 X6]
+ (-0.002293955623064501) [X1 X2 Y3 Z4 Z5 Y6]
+ (-0.001799193008386174) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.001799193008386174) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.0017278745823308406) [Y1 Z2 Z3 Y4 X11 X12]
+ (-0.0017278745823308406) [X1 Z2 Z3 X4 Y11 Y12]
+ (-0.0016676137499672694) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (-0.0016676137499672694) [X0 Z1 Z3 Z4 Z5 X6]
+ (-0.0016676137499672694) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (-0.0016676137499672694) [X1 Z2 Z4 Z5 Z6 X7]
+ (-0.0016095335162951978) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (-0.0016095335162951978) [X0 Z1 Z2 Z3 Z4 X6]
+ (-0.0016095335162951978) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (-0.0016095335162951978) [X1 Z2 Z3 Z5 Z6 X7]
+ (-0.0009298407044441078) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298407044441078) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298407044441078) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298407044441078) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533831050997234) [Y1 Z2 Z3 X4 X5 Y6]
+ (-0.0008533831050997234) [X1 Z2 Z3 Y4 Y5 X6]
+ (-0.0006650303448843551) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (-0.0006650303448843551) [X2 Z3 Z4 Z5 X6 Z12]
+ (-0.0006650303448843551) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (-0.0006650303448843551) [X3 Z4 Z5 Z6 X7 Z13]
+ (-0.0004957972885579183) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0004957972885579183) [Z2 X3 Z4 Z5 Z6 X7]
+ (-7.735870605754389e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.735870605754389e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.735870605754389e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.735870605754389e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-5.974176903285593e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (-5.974176903285593e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (-5.275783473266196e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.275783473266196e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.642978995234286e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.642978995234286e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.5564737840833244e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (-4.5564737840833244e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (-4.183808837220776e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (-4.183808837220776e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (-3.694516908079974e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (-3.694516908079974e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (-3.3342618876112257e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (-3.3342618876112257e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (-3.3130170961591376e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (-3.3130170961591376e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (-3.15129598151703e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.15129598151703e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.0882457264049847e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.0882457264049847e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.172638029106916e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (-2.172638029106916e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (-1.4548067005396102e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.4548067005396102e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.3304568598916221e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (-1.3304568598916221e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (-1.2282691321655916e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (-1.2282691321655916e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (-1.0357924863529412e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (-1.0357924863529412e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (-7.956667024294647e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.956667024294647e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.733096772522644e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.733096772522644e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.733096772522644e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.733096772522644e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.628427303686851e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (-6.628427303686851e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (-5.927350237838324e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.927350237838324e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.927350237838324e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.927350237838324e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.627722054083482e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (-5.627722054083482e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (-4.837953374670979e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (-4.837953374670979e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (-3.5706355251168246e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.5706355251168246e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.32803968902326e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (-3.32803968902326e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (-3.0867709220731853e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.0867709220731853e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.0867709220731853e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.0867709220731853e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.447264832973169e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (-2.447264832973169e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (-2.3712704782553376e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.3712704782553376e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.3712704782553376e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.3712704782553376e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.1989637768272475e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (-2.1989637768272475e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (-1.933212146127449e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.933212146127449e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.933212146127449e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.933212146127449e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.8393943714832888e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.8393943714832888e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.8393943714832888e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.8393943714832888e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.2919458298947298e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.2919458298947298e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.8395658226829314e-08) [Y0 Z1 X2 X10 Z11 Y12]
+ (-1.8395658226829314e-08) [X0 Z1 Y2 Y10 Z11 X12]
+ (-1.8395658226829314e-08) [Y1 Z2 X3 X11 Z12 Y13]
+ (-1.8395658226829314e-08) [X1 Z2 Y3 Y11 Z12 X13]
+ (-1.0351505866434756e-09) [Y0 Z1 Z2 Z3 Y4 Z7]
+ (-1.0351505866434756e-09) [X0 Z1 Z2 Z3 X4 Z7]
+ (-1.0351505866434756e-09) [Y1 Z2 Z3 Z4 Y5 Z6]
+ (-1.0351505866434756e-09) [X1 Z2 Z3 Z4 X5 Z6]
+ (2.2702083648268034e-08) [Y0 Z1 Z2 X3 X5 Y6]
+ (2.2702083648268034e-08) [Y0 Z1 Z2 Y3 Y5 Y6]
+ (2.2702083648268034e-08) [X0 Z1 Z2 X3 X5 X6]
+ (2.2702083648268034e-08) [X0 Z1 Z2 Y3 Y5 X6]
+ (2.2702083648268034e-08) [Y1 X2 X4 Z5 Z6 Y7]
+ (2.2702083648268034e-08) [Y1 Y2 Y4 Z5 Z6 Y7]
+ (2.2702083648268034e-08) [X1 X2 X4 Z5 Z6 X7]
+ (2.2702083648268034e-08) [X1 Y2 Y4 Z5 Z6 X7]
+ (2.5928188668818287e-08) [Y0 Z1 X2 X4 Z5 Y6]
+ (2.5928188668818287e-08) [X0 Z1 Y2 Y4 Z5 X6]
+ (2.5928188668818287e-08) [Y1 Z2 X3 X5 Z6 Y7]
+ (2.5928188668818287e-08) [X1 Z2 Y3 Y5 Z6 X7]
+ (8.057465346845917e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.057465346845917e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.057465346845917e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.057465346845917e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.649129656460052e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.649129656460052e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.649129656460052e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.649129656460052e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.1076529177912997e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (1.1076529177912997e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (1.1076529177912997e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (1.1076529177912997e-07) [X0 Z1 X2 X11 Z12 X13]
+ (1.1076529177912997e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (1.1076529177912997e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (1.1076529177912997e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (1.1076529177912997e-07) [X1 Z2 X3 X10 Z11 X12]
+ (1.3484969013802614e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (1.3484969013802614e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (1.3484969013802614e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (1.3484969013802614e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (1.3807579515877884e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (1.3807579515877884e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (1.3807579515877884e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (1.3807579515877884e-07) [X0 Z1 X2 X5 Z6 X7]
+ (1.3807579515877884e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (1.3807579515877884e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (1.3807579515877884e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (1.3807579515877884e-07) [X1 Z2 X3 X4 Z5 X6]
+ (1.6077787880698588e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (1.6077787880698588e-07) [X0 Z1 X2 X4 Z5 X6]
+ (1.6077787880698588e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (1.6077787880698588e-07) [X1 Z2 X3 X5 Z6 X7]
+ (1.8290428656181584e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.8290428656181584e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.8290428656181584e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.8290428656181584e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.1989637768272475e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (2.1989637768272475e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (2.447264832973169e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (2.447264832973169e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (3.236183443901343e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.236183443901343e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.236183443901343e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.236183443901343e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.32803968902326e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (3.32803968902326e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (3.5706355251168246e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.5706355251168246e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.837953374670979e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (4.837953374670979e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (5.287649483665944e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (5.287649483665944e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (5.287649483665944e-07) [X0 Z1 Z2 X3 X11 X12]
+ (5.287649483665944e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (5.287649483665944e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (5.287649483665944e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (5.287649483665944e-07) [X1 X2 X10 Z11 Z12 X13]
+ (5.287649483665944e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (5.627722054083482e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (5.627722054083482e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (6.395302401460158e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (6.395302401460158e-07) [X0 Z1 X2 X10 Z11 X12]
+ (6.395302401460158e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (6.395302401460158e-07) [X1 Z2 X3 X11 Z12 X13]
+ (6.579258983732381e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (6.579258983732381e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (6.579258983732381e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (6.579258983732381e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (6.628427303686851e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (6.628427303686851e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (7.956667024294647e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.956667024294647e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.306343007108524e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.306343007108524e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.306343007108524e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.306343007108524e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.0357924863529412e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (1.0357924863529412e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (1.2282691321655916e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (1.2282691321655916e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (1.2393113929178322e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.2393113929178322e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.2393113929178322e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.2393113929178322e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.3304568598916221e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (1.3304568598916221e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (1.4548067005396102e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.4548067005396102e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.172638029106916e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (2.172638029106916e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (3.0882457264049847e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.0882457264049847e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.117366419074471e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.117366419074471e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.15129598151703e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.15129598151703e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.2111874482994296e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (3.2111874482994296e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (3.2111874482994296e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (3.2111874482994296e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (3.2774382925189124e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (3.2774382925189124e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (3.2774382925189124e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (3.2774382925189124e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (3.3130170961591376e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (3.3130170961591376e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (3.6102422614212384e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (3.6102422614212384e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (3.6102422614212384e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (3.6102422614212384e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (3.694516908079974e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (3.694516908079974e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (3.7695836224123716e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (3.7695836224123716e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (4.253118763408899e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (4.253118763408899e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (4.5564737840833244e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (4.5564737840833244e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (4.642978995234286e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.642978995234286e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.275783473266196e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.275783473266196e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.974176903285593e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (5.974176903285593e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (6.290019712139631e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (6.290019712139631e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (6.290019712139631e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (6.290019712139631e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (6.5242045444568325e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (6.5242045444568325e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (6.5242045444568325e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (6.5242045444568325e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (7.444267440651119e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (7.444267440651119e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (7.444267440651119e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (7.444267440651119e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (7.518288844303922e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (7.518288844303922e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (7.518288844303922e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (7.518288844303922e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (8.774724300542741e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (8.774724300542741e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (8.774724300542741e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (8.774724300542741e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (0.000292225672451927) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.000292225672451927) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.000292225672451927) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.000292225672451927) [X6 Z7 X8 X9 Z10 X11]
+ (0.0008144692870629832) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (0.0008144692870629832) [X2 Z3 Z4 Z5 X6 Z10]
+ (0.0008144692870629832) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (0.0008144692870629832) [X3 Z4 Z5 Z6 X7 Z11]
+ (0.0008533831050997234) [Y1 Z2 Z3 Y4 X5 X6]
+ (0.0008533831050997234) [X1 Z2 Z3 X4 Y5 Y6]
+ (0.0017278745823308406) [Y1 Z2 Z3 X4 X11 Y12]
+ (0.0017278745823308406) [X1 Z2 Z3 Y4 Y11 X12]
+ (0.001799193008386174) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.001799193008386174) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.002293955623064501) [Y1 X2 X3 Z4 Z5 Y6]
+ (0.002293955623064501) [X1 Y2 Y3 Z4 Z5 X6]
+ (0.002779040762880847) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (0.002779040762880847) [X1 Z2 Z3 Z4 Z5 X7]
+ (0.0034938003715269308) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (0.0034938003715269308) [X2 Z3 Z4 Z5 X6 Z13]
+ (0.0034938003715269308) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (0.0034938003715269308) [X3 Z4 Z5 Z6 X7 Z12]
+ (0.004158830716411285) [Y3 Z4 Z5 X6 X12 Y13]
+ (0.004158830716411285) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (0.004158830716411285) [X3 Z4 Z5 X6 X12 X13]
+ (0.004158830716411285) [X3 Z4 Z5 Y6 Y12 X13]
+ (0.004424843668499126) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.004424843668499126) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.0045750151888949145) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.0045750151888949145) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.004668615266018511) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.004668615266018511) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.004684920226870334) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.004684920226870334) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.005143382387697844) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (0.005143382387697844) [Y2 Z3 Y4 X5 Z6 X7]
+ (0.005143382387697844) [X2 Z3 X4 Y5 Z6 Y7]
+ (0.005143382387697844) [X2 Z3 X4 X5 Z6 X7]
+ (0.005368616111420641) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.005368616111420641) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.005368616111420641) [X2 X3 X7 Z8 Z9 X10]
+ (0.005368616111420641) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.005652607315223128) [Y0 X1 X3 Z4 Z5 Y6]
+ (0.005652607315223128) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (0.005652607315223128) [X0 X1 X3 Z4 Z5 X6]
+ (0.005652607315223128) [X0 Y1 Y3 Z4 Z5 X6]
+ (0.0058051212123838986) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (0.0058051212123838986) [X2 Z3 Z4 Z5 X6 Z8]
+ (0.0058051212123838986) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (0.0058051212123838986) [X3 Z4 Z5 Z6 X7 Z9]
+ (0.007960839634219279) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.007960839634219279) [X4 Z5 X6 X10 Z11 X12]
+ (0.007960839634219279) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.007960839634219279) [X5 Z6 X7 X11 Z12 X13]
+ (0.008125248410131975) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.008125248410131975) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.00876485821938415) [Y2 Z3 Z4 X5 X11 Y12]
+ (0.00876485821938415) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (0.00876485821938415) [X2 Z3 Z4 X5 X11 X12]
+ (0.00876485821938415) [X2 Z3 Z4 Y5 Y11 X12]
+ (0.00876485821938415) [Y3 X4 X10 Z11 Z12 Y13]
+ (0.00876485821938415) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (0.00876485821938415) [X3 X4 X10 Z11 Z12 X13]
+ (0.00876485821938415) [X3 Y4 Y10 Z11 Z12 X13]
+ (0.008890680338663392) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.008890680338663392) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.008890680338663392) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.008890680338663392) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.010540434329184292) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.010540434329184292) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.010540434329184292) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.010540434329184292) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.010959994608890192) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.010959994608890192) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.010959994608890192) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.010959994608890192) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.011307208029975837) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.011307208029975837) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.011755995240396343) [Y3 Z4 Z5 X6 X8 Y9]
+ (0.011755995240396343) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (0.011755995240396343) [X3 Z4 Z5 X6 X8 X9]
+ (0.011755995240396343) [X3 Z4 Z5 Y6 Y8 X9]
+ (0.017561116452780245) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (0.017561116452780245) [X2 Z3 Z4 Z5 X6 Z9]
+ (0.017561116452780245) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (0.017561116452780245) [X3 Z4 Z5 Z6 X7 Z8]
+ (0.018266758578496592) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.018266758578496592) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.018266758578496592) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.018266758578496592) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.01902037387511046) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.01902037387511046) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.01902037387511046) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.01902037387511046) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.02017582495698184) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.02017582495698184) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.02017582495698184) [X4 Z5 Z6 X7 X11 X12]
+ (0.02017582495698184) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.02017582495698184) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.02017582495698184) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.02017582495698184) [X5 X6 X10 Z11 Z12 X13]
+ (0.02017582495698184) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.0243889899865311) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.0243889899865311) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.0243889899865311) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.0243889899865311) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.025104907969995847) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.025104907969995847) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.025104907969995847) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.025104907969995847) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.02599620626718891) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (0.02599620626718891) [X2 Z3 Z4 Z5 X6 Z7]
+ (0.027114878580481193) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (0.027114878580481193) [Z0 X2 Z3 Z4 Z5 X6]
+ (0.027114878580481193) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (0.027114878580481193) [Z1 X3 Z4 Z5 Z6 X7]
+ (0.0307874407185145) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.0307874407185145) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.03276748589570432) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (0.03276748589570432) [Z0 X3 Z4 Z5 Z6 X7]
+ (0.03276748589570432) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (0.03276748589570432) [Z1 X2 Z3 Z4 Z5 X6]
+ (0.05600713561683751) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.05600713561683751) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.05600713561683751) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.05600713561683751) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.05608449432289506) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.05608449432289506) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.05608449432289506) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.05608449432289506) [Z1 X7 Z8 Z9 Z10 X11]
+ (-0.04274326006288902) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (-0.04274326006288902) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (-0.042743260062889) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-0.042743260062889) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (2.5950813668686533e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.5950813668686533e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.5950813668686536e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.5950813668686536e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (6.631261962916301e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (6.631261962916301e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (6.631261962916301e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (6.631261962916301e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.04764261360015648) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.04764261360015648) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.04764261360015648) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.04764261360015648) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.045879424030657236) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-0.045879424030657236) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.04171881404454458) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.04171881404454458) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.04171881404454458) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.04171881404454458) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.0395645480415713) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.0395645480415713) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.0395645480415713) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0395645480415713) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03931810723100555) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03931810723100555) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.03931810723100555) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03931810723100555) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.02563721280993811) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.02563721280993811) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.02563721280993811) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.02563721280993811) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.02314522165323584) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.02314522165323584) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.022528354240836008) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.022528354240836008) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.019257453001871646) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019257453001871646) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.019028318718278044) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.019028318718278044) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.01602466609552088) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.01602466609552088) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.015225659057108532) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.015225659057108532) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-0.014603742410739428) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (-0.014603742410739428) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (-0.014564473640811556) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.014564473640811556) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.011755995240396343) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (-0.011755995240396343) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (-0.011285144618318447) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-0.011285144618318447) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-0.009841802923813819) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.009841802923813819) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.009612546714417233) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.009612546714417233) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.009612546714417233) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.009612546714417233) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.008469833341369658) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.008469833341369658) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.007306763969606397) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.007306763969606397) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005923799555611908) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.005923799555611908) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.005652607315223128) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (-0.005652607315223128) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (-0.005379929634053252) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (-0.005379929634053252) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (-0.005379929634053252) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (-0.005379929634053252) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (-0.005368616111420641) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005368616111420641) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.005241543597042509) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (-0.005241543597042509) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (-0.005241543597042509) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (-0.005241543597042509) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (-0.00463697351649715) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (-0.00463697351649715) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (-0.00463697351649715) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (-0.00463697351649715) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (-0.004311038607563142) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (-0.004311038607563142) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (-0.004311038607563142) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (-0.004311038607563142) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (-0.004158830716411285) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (-0.004158830716411285) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (-0.003989845257549345) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (-0.003989845257549345) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (-0.003989845257549345) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (-0.003989845257549345) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (-0.0022619706752185047) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (-0.0022619706752185047) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (-0.0022619706752185047) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (-0.0022619706752185047) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (-0.0022619706752185047) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (-0.0022619706752185047) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (-0.0022619706752185047) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (-0.0022619706752185047) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (-0.0013038029824609314) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (-0.0013038029824609314) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (-0.0013038029824609314) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (-0.0013038029824609314) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (-0.0012803055951264297) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (-0.0012803055951264297) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (-0.0012803055951264297) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (-0.0012803055951264297) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (-0.0010435237104093097) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (-0.0010435237104093097) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (-0.0010435237104093097) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (-0.0010435237104093097) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (-0.0008533831050997234) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (-0.0008533831050997234) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (-0.0008533831050997234) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (-0.0008533831050997234) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (-0.00024644081056574723) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00024644081056574723) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.735870605754389e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.735870605754389e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-4.183808837220776e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-4.183808837220776e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-3.5443574316889204e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.5443574316889204e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.5443574316889204e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.5443574316889204e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.5443574316889204e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.5443574316889204e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.5443574316889204e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.5443574316889204e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.3342618876112257e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-3.3342618876112257e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-3.3130170961591376e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-3.3130170961591376e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-3.3130170961591376e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-3.3130170961591376e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.5224582046927245e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.5224582046927245e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.5224582046927245e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.5224582046927245e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.5224582046927245e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224582046927245e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224582046927245e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.5224582046927245e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.3304568598916221e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (-1.3304568598916221e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (-1.3304568598916221e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (-1.3304568598916221e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (-7.988467914937921e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.988467914937921e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.988467914937921e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.988467914937921e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.189870489301581e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.189870489301581e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (-6.175164920705592e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175164920705592e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.471606065934441e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (-5.471606065934441e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (-4.561117052846947e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.561117052846947e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.561117052846947e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.561117052846947e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.523339373666387e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (-4.523339373666387e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (-3.4273508620812163e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.4273508620812163e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.4273508620812163e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.4273508620812163e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.0867709220731853e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.0867709220731853e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.3712704782553376e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.3712704782553376e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.8393943714832888e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.8393943714832888e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-1.7035434590496648e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035434590496648e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035434590496648e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (-1.7035434590496648e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (-9.208945586105209e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-9.208945586105209e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-9.208945586105209e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-9.208945586105209e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-8.057465346845917e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.057465346845917e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (-6.772951823274002e-08) [Y1 Z2 Z3 X4 X7 Z8 Z9 Y10]
+ (-6.772951823274002e-08) [X1 Z2 Z3 Y4 Y7 Z8 Z9 X10]
+ (-3.2261050206086977e-09) [Y0 Z1 Z2 Y3 X4 Z5 Z6 X7]
+ (-3.2261050206086977e-09) [X0 Z1 Z2 X3 Y4 Z5 Z6 Y7]
+ (3.2261050206086977e-09) [Y0 Z1 Z2 X3 X4 Z5 Z6 Y7]
+ (3.2261050206086977e-09) [X0 Z1 Z2 Y3 Y4 Z5 Z6 X7]
+ (6.046790514200977e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (6.046790514200977e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (6.046790514200977e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (6.046790514200977e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (6.772951823274002e-08) [Y1 Z2 Z3 Y4 X7 Z8 Z9 X10]
+ (6.772951823274002e-08) [X1 Z2 Z3 X4 Y7 Z8 Z9 Y10]
+ (8.057465346845917e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.057465346845917e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (1.8393943714832888e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.8393943714832888e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.3712704782553376e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.3712704782553376e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (2.888564838371757e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.888564838371757e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.888564838371757e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (2.888564838371757e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (3.0867709220731853e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.0867709220731853e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (3.32803968902326e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (3.32803968902326e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (3.32803968902326e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (3.32803968902326e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (4.523339373666387e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (4.523339373666387e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (5.471606065934441e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (5.471606065934441e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (6.175164920705592e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175164920705592e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.189870489301581e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (7.189870489301581e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (7.867608648565853e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (7.867608648565853e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (7.867608648565853e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (7.867608648565853e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (1.2282691321655916e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (1.2282691321655916e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (1.2282691321655916e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (1.2282691321655916e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (1.6288377768247393e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.6288377768247393e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.6288377768247393e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.6288377768247393e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.6540900665950459e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.6540900665950459e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.6540900665950459e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.6540900665950459e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.6893056819638217e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.6893056819638217e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.6893056819638217e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.6893056819638217e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (1.942946550431788e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.942946550431788e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.942946550431788e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.942946550431788e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.011074033623746e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.011074033623746e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.011074033623746e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.011074033623746e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.1031634894846355e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.1031634894846355e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.1031634894846355e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.1031634894846355e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.360947217808988e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (2.360947217808988e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (2.360947217808988e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (2.360947217808988e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (2.7455106401972967e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.7455106401972967e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.7455106401972967e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.7455106401972967e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.7455106401972967e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.7455106401972967e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.7455106401972967e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.7455106401972967e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.211763886657739e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.211763886657739e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.211763886657739e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.211763886657739e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.211763886657739e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.211763886657739e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.211763886657739e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.211763886657739e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.7695836224123716e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.7695836224123716e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.253118763408899e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.253118763408899e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.7287814466183105e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (4.7287814466183105e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (4.7287814466183105e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.7287814466183105e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.734578396859215e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (4.734578396859215e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (4.734578396859215e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (4.734578396859215e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (5.071403690415019e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (5.071403690415019e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (5.071403690415019e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (5.071403690415019e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (6.481752034236745e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (6.481752034236745e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.481752034236745e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (6.481752034236745e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (6.652106380141645e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (6.652106380141645e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (6.652106380141645e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (6.652106380141645e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.089728664424696e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.089728664424696e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (7.089728664424696e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (7.089728664424696e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (9.8059820872725e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (9.8059820872725e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (9.8059820872725e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (9.8059820872725e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (1.5316614177890407e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (1.5316614177890407e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (1.5316614177890407e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (1.5316614177890407e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.6103375042746965e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (1.6103375042746965e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.6103375042746965e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (1.6103375042746965e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (7.735870605754389e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.735870605754389e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.00013838603701074467) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (0.00013838603701074467) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (0.00013838603701074467) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (0.00013838603701074467) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (0.00024644081056574723) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00024644081056574723) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0004458488204531645) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.0004458488204531645) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.0004458488204531645) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.0004458488204531645) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.00059401576739196) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.00059401576739196) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.00059401576739196) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.00059401576739196) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.00059401576739196) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.00059401576739196) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.00059401576739196) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.00059401576739196) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.0009581676927575732) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (0.0009581676927575732) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (0.0009581676927575732) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (0.0009581676927575732) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (0.0009581676927575732) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (0.0009581676927575732) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (0.0009581676927575732) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (0.0009581676927575732) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (0.002293955623064501) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (0.002293955623064501) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (0.002293955623064501) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (0.002293955623064501) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (0.0026860422750884135) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (0.0026860422750884135) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (0.0026860422750884135) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (0.0026860422750884135) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (0.002779040762880847) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (0.002779040762880847) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (0.0032675148971538317) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (0.0032675148971538317) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (0.0032675148971538317) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (0.0032675148971538317) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (0.00335666792137072) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (0.00335666792137072) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (0.00335666792137072) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (0.00335666792137072) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (0.004158830716411285) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (0.004158830716411285) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (0.005114464086471674) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.005114464086471674) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.005114464086471674) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.005114464086471674) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.005114464086471674) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114464086471674) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114464086471674) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005114464086471674) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.005262631033410468) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.005262631033410468) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.005262631033410468) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.005262631033410468) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.005368616111420641) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005368616111420641) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005652607315223128) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (0.005652607315223128) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (0.005708479853863634) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.005708479853863634) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.005708479853863634) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.005708479853863634) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.005923799555611908) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.005923799555611908) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.007306763969606397) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.007306763969606397) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.008469833341369658) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.008469833341369658) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.009841802923813819) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.009841802923813819) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.011285144618318447) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (0.011285144618318447) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (0.011755995240396343) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (0.011755995240396343) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (0.014564473640811556) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.014564473640811556) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.014603742410739428) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (0.014603742410739428) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (0.015225659057108532) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.015225659057108532) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.01602466609552088) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.01602466609552088) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.018888995077423636) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (0.018888995077423636) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (0.018888995077423636) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.018888995077423636) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.019028318718278044) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.019028318718278044) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.019257453001871646) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (0.019257453001871646) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (0.02143398011689322) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.02143398011689322) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.02143398011689322) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.02143398011689322) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.024282031623370524) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (0.024282031623370524) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (0.024755507980794123) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.024755507980794123) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.024755507980794123) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.024755507980794123) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.02873079800123745) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.02873079800123745) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.02873079800123745) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.02873079800123745) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.029903813458262873) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.029903813458262873) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.029903813458262873) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.029903813458262873) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.035608400352290494) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.035608400352290494) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.039359250391533554) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.039359250391533554) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.039359250391533554) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.039359250391533554) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.3693713755443826) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.3693713755443826) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.36937137554438254) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.36937137554438254) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.2816433575341089) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.2816433575341089) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.2816433575341089) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.2816433575341089) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.09065142344949306) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.09065142344949306) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.09065142344949306) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.09065142344949306) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684736029512481) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.08684736029512481) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684736029512481) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.08684736029512481) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.058592151795446724) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.058592151795446724) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.034903304270696336) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.034903304270696336) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.034903304270696336) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.034903304270696336) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591832088270153) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.024591832088270153) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591832088270153) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.024591832088270153) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.02314522165323584) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.02314522165323584) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.022528354240836008) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.022528354240836008) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019299499858011) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.019299499858011) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019299499858011) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.019299499858011) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019299499858011) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019299499858011) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.019299499858011) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019299499858011) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.01602466609552088) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.01602466609552088) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.01602466609552088) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.01602466609552088) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.014603742410739426) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (-0.014603742410739426) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (-0.014603742410739426) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (-0.014603742410739426) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (-0.010757524201214109) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010757524201214109) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010757524201214109) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010757524201214109) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.01071547734507476) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.01071547734507476) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.01071547734507476) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.01071547734507476) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.01031147218242618) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.01031147218242618) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.01031147218242618) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.01031147218242618) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005408970758387174) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.005408970758387174) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.005408970758387174) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.005408970758387174) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.005286569056773387) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.005286569056773387) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.005286569056773387) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.005286569056773387) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.004767276644723243) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.004767276644723243) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.004767276644723243) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.004767276644723243) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.004668615266018511) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.004668615266018511) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.0038764821957270826) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (-0.0038764821957270826) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (-0.0038040631543682555) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0038040631543682555) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0038040631543682555) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0038040631543682555) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003484154579346415) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003484154579346415) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00335666792137072) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (-0.00335666792137072) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (-0.0032675148971538317) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (-0.0032675148971538317) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (-0.002446463422677062) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.002446463422677062) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.002446463422677062) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.002446463422677062) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.0017278745823308406) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.0017278745823308406) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.0016407591167052525) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0016407591167052525) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0015324885626600909) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.0015324885626600909) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.0015324885626600909) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.0015324885626600909) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.0007870893705577073) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0007870893705577073) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007156737069630855) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0007156737069630855) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007156737069630855) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0007156737069630855) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0005192924120501438) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.0005192924120501438) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.00019401030606668883) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00019401030606668883) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.00018787486138094185) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.00018787486138094185) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.00018787486138094185) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.00018787486138094185) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.00013838603701074467) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.00013838603701074467) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-4.2046856139353215e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-4.2046856139353215e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-4.2046856139353215e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.2046856139353215e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-5.071403690415019e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-5.071403690415019e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-3.15129598151703e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.15129598151703e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0882457264049847e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.0882457264049847e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.988412591826582e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.988412591826582e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8742485753621304e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.8742485753621304e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.360947217808988e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-2.360947217808988e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.3001958958404336e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.3001958958404336e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-7.867608648565853e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867608648565853e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.996951815662803e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.996951815662803e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.996951815662803e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.996951815662803e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.996951815662803e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.996951815662803e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.996951815662803e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.996951815662803e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.888564838371757e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.888564838371757e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.686321553351228e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.686321553351228e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035434590496648e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.7035434590496648e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.208945586105209e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.208945586105209e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.2040041898185006e-08) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (-3.2040041898185006e-08) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (-3.2040041898185006e-08) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (-3.2040041898185006e-08) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (3.56894763347312e-08) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (3.56894763347312e-08) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (3.56894763347312e-08) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (3.56894763347312e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (3.56894763347312e-08) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.56894763347312e-08) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (3.56894763347312e-08) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (3.56894763347312e-08) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.706834762489992e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.706834762489992e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.706834762489992e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.706834762489992e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.379737421932558e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (8.379737421932558e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.379737421932558e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (8.379737421932558e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.379737421932558e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (8.379737421932558e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (8.379737421932558e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (8.379737421932558e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (9.208945586105209e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.208945586105209e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.0716845264030261e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.0716845264030261e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.0716845264030261e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.0716845264030261e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.1782131050401464e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.1782131050401464e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.1782131050401464e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.1782131050401464e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.7035434590496648e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.7035434590496648e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.2498976314431725e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.2498976314431725e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.2498976314431725e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.2498976314431725e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.686321553351228e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.686321553351228e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.888564838371757e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.888564838371757e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.376686385101164e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (3.376686385101164e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (3.376686385101164e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (3.376686385101164e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (3.376686385101164e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (3.376686385101164e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (3.376686385101164e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (3.376686385101164e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (3.568200494966831e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (3.568200494966831e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (3.568200494966831e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (3.568200494966831e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.092161949993408e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.092161949993408e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.092161949993408e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.092161949993408e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.092161949993408e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.092161949993408e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.092161949993408e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.092161949993408e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.4490567133414994e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.4490567133414994e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.4490567133414994e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.4490567133414994e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.769457132318572e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.769457132318572e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.769457132318572e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.769457132318572e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.246849447106857e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.246849447106857e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.246849447106857e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.246849447106857e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.246849447106857e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.246849447106857e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.246849447106857e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.246849447106857e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.560553965547065e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.560553965547065e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (7.560553965547065e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (7.560553965547065e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (7.560553965547065e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.560553965547065e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.560553965547065e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.560553965547065e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.867608648565853e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (7.867608648565853e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.900025758764231e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (7.900025758764231e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (7.900025758764231e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (7.900025758764231e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (8.027844231492194e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (8.027844231492194e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (8.027844231492194e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.027844231492194e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (8.091539868635725e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (8.091539868635725e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (8.091539868635725e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (8.091539868635725e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (8.091539868635725e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (8.091539868635725e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (8.091539868635725e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (8.091539868635725e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (8.398527707738084e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (8.398527707738084e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (8.398527707738084e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (8.398527707738084e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.1468226253734585e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (1.1468226253734585e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (1.1468226253734585e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (1.1468226253734585e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.3001958958404336e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.3001958958404336e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.360947217808988e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.360947217808988e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.8742485753621304e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.8742485753621304e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.8836531766416457e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.8836531766416457e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.947331458469501e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.947331458469501e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.947331458469501e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.947331458469501e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.988412591826582e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.988412591826582e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.0882457264049847e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.0882457264049847e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.15129598151703e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.15129598151703e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.846190890424185e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.846190890424185e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.846190890424185e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.846190890424185e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.071403690415019e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (5.071403690415019e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (5.10546243871445e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.10546243871445e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.10546243871445e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.10546243871445e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.146386786268088e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.146386786268088e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.146386786268088e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.146386786268088e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.159294497886387e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.159294497886387e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.159294497886387e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.159294497886387e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.4279266532201e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.4279266532201e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.4279266532201e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.4279266532201e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.935744050295866e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.935744050295866e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.935744050295866e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.935744050295866e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253185184707446e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.253185184707446e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.97971101407658e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.97971101407658e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.97971101407658e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.97971101407658e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (7.14156635946223e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (7.14156635946223e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (7.14156635946223e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.14156635946223e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00013838603701074467) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.00013838603701074467) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.00019401030606668883) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (0.00019401030606668883) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (0.00024644081056574723) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024644081056574723) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00024644081056574723) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024644081056574723) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192924120501438) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.0005192924120501438) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.0007870893705577073) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.0007870893705577073) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.0014528842553243092) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0014528842553243092) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0014528842553243092) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0014528842553243092) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0016407591167052525) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0016407591167052525) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0017278745823308406) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.0017278745823308406) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.0021413489647418437) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (0.0021413489647418437) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (0.0032675148971538317) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (0.0032675148971538317) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (0.00335666792137072) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (0.00335666792137072) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (0.003484154579346415) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.003484154579346415) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0038764821957270826) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (0.0038764821957270826) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (0.004668615266018511) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.004668615266018511) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.0059237995556119085) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.0059237995556119085) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.0059237995556119085) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.0059237995556119085) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.008469833341369658) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.008469833341369658) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.008469833341369658) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.008469833341369658) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.008541975656796886) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.008541975656796886) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (0.008541975656796886) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (0.008541975656796886) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.008541975656796886) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.008541975656796886) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.008541975656796886) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.008541975656796886) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.008826387567793205) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.008826387567793205) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.008826387567793205) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.008826387567793205) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.009841802923813819) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.009841802923813819) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.009841802923813819) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.009841802923813819) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.017091621922246163) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.017091621922246163) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.017091621922246163) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.017091621922246163) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.019538085344923226) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.019538085344923226) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.019538085344923226) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.019538085344923226) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.024282031623370524) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (0.024282031623370524) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (0.035608400352290494) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.035608400352290494) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.06752398179963201) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.06752398179963201) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.06752398179963201) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.06752398179963201) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0763503693674252) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0763503693674252) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0763503693674252) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0763503693674252) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.07165056250040007) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07165056250040007) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.07165056250040006) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07165056250040006) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (5.775872123090838e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.775872123090838e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.775872123090838e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.775872123090838e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.058592151795446724) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.058592151795446724) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.019257453001871646) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.019257453001871646) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.01031147218242618) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.01031147218242618) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008826387567793205) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.008826387567793205) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.007597461779083911) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.007597461779083911) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.007597461779083911) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.007597461779083911) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0057335686400186185) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0057335686400186185) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0057335686400186185) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0057335686400186185) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0057335686400186185) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0057335686400186185) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0057335686400186185) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0057335686400186185) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00534804771841171) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.00534804771841171) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00534804771841171) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00534804771841171) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.004220835998337692) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.004220835998337692) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.004220835998337692) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.004220835998337692) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.0038764821957270826) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (-0.0038764821957270826) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (-0.0038764821957270826) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (-0.0038764821957270826) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (-0.0038040631543682555) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0038040631543682555) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002446463422677062) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.002446463422677062) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.0023949671540609585) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0023949671540609585) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0023949671540609585) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0023949671540609585) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0023949671540609585) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0023949671540609585) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0023949671540609585) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0023949671540609585) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0022494140606722024) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0022494140606722024) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0022494140606722024) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0022494140606722024) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0022009568479942694) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0022009568479942694) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0022009568479942694) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0022009568479942694) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638931390652935) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638931390652935) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638931390652935) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638931390652935) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638931390652935) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638931390652935) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0018638931390652935) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638931390652935) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012366559235495501) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.0012366559235495501) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.0012366559235495501) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.0012366559235495501) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.0011726297842404002) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0011726297842404002) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0011726297842404002) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0011726297842404002) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0007870893705577074) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0007870893705577074) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870893705577074) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0007870893705577074) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0005192924120501438) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.0005192924120501438) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.0005192924120501438) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.0005192924120501438) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-1.1462851040955387e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1462851040955387e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.8742485753621304e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742485753621304e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.8742485753621304e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742485753621304e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.3001958958404336e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.3001958958404336e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.3001958958404336e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.3001958958404336e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.0444741677284923e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.0444741677284923e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.0444741677284923e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.0444741677284923e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.95590362227135e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.95590362227135e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.95590362227135e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.95590362227135e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105341142090791e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105341142090791e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105341142090791e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105341142090791e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661200361849568e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661200361849568e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661200361849568e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661200361849568e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540204289106399e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540204289106399e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.189870489301581e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.189870489301581e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.876530412657596e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876530412657596e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876530412657596e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876530412657596e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175164920705592e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175164920705592e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.523339373666387e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.523339373666387e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.0766627195141766e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.0766627195141766e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0766627195141766e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.0766627195141766e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.0133988594157424e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0133988594157424e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.9045373881785236e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.9045373881785236e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.9045373881785236e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.9045373881785236e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.6666797829653693e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.6666797829653693e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.6666797829653693e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6666797829653693e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505624801765284e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.8505624801765284e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.846699491827571e-08) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-7.846699491827571e-08) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-6.772951823274002e-08) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (-6.772951823274002e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (-4.0998293655050136e-08) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-4.0998293655050136e-08) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.0998293655050136e-08) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.0998293655050136e-08) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (6.772951823274002e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (6.772951823274002e-08) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (7.846699491827571e-08) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (7.846699491827571e-08) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (1.6570092868269324e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.6570092868269324e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.6570092868269324e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.6570092868269324e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.8505624801765284e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.8505624801765284e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.686321553351228e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.686321553351228e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.686321553351228e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.686321553351228e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0133988594157424e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.0133988594157424e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.523339373666387e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.523339373666387e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.6704081462453515e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.6704081462453515e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.6704081462453515e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.6704081462453515e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175164920705592e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175164920705592e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.189870489301581e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.189870489301581e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.540204289106399e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540204289106399e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949307755240195e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949307755240195e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.792463838069044e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.792463838069044e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.792463838069044e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.792463838069044e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.8836531766416457e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.8836531766416457e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.988412591826582e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.988412591826582e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.988412591826582e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.988412591826582e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253185184707446e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.253185184707446e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4016916490390175e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4016916490390175e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4016916490390175e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4016916490390175e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809380328459116e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.5809380328459116e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809380328459116e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.5809380328459116e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0010283270637538695) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0010283270637538695) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0010283270637538695) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0010283270637538695) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.001222337369820558) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.001222337369820558) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (0.001222337369820558) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (0.001222337369820558) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.001222337369820558) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.001222337369820558) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.001222337369820558) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.001222337369820558) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0016407591167052525) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0016407591167052525) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0016407591167052525) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0016407591167052525) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0021413489647418432) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (0.0021413489647418432) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (0.002446463422677062) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.002446463422677062) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.0029841800747881413) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.0029841800747881413) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.0029841800747881413) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.0029841800747881413) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.0038040631543682555) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0038040631543682555) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.008826387567793205) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.008826387567793205) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.01031147218242618) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01031147218242618) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019257453001871646) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019257453001871646) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.3986653635132954e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.3986653635132954e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.3986653635132954e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.3986653635132954e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003484154579346415) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003484154579346415) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0029841800747881417) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.0029841800747881417) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.00019401030606668883) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00019401030606668883) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1462851040955387e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1462851040955387e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.792463838069044e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.792463838069044e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540204289106399e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540204289106399e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540204289106399e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540204289106399e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.8505624801765284e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505624801765284e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505624801765284e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505624801765284e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.846699491827571e-08) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-7.846699491827571e-08) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-7.846699491827571e-08) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-7.846699491827571e-08) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-4.0998293655050136e-08) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-4.0998293655050136e-08) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.0998293655050136e-08) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-4.0998293655050136e-08) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0133988594157424e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.0133988594157424e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0133988594157424e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.0133988594157424e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.949307755240195e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.949307755240195e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.792463838069044e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.792463838069044e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019401030606668883) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00019401030606668883) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0029841800747881417) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.0029841800747881417) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.003484154579346415) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.003484154579346415) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
  (-73.13873149597572) [I0]
+ (-0.18066757507019904) [Z7]
+ (-0.18066757507019895) [Z6]
+ (-0.15961443583741797) [Z4]
+ (-0.15961443583741797) [Z5]
+ (0.17419986612092292) [Z2]
+ (0.17419986612092297) [Z3]
+ (0.2275732881443361) [Z0]
+ (0.2275732881443361) [Z1]
+ (-7.954224451177383e-06) [Y5 Y7]
+ (-7.954224451177383e-06) [X5 X7]
+ (8.194104912529615e-06) [Y4 Y6]
+ (8.194104912529615e-06) [X4 X6]
+ (0.11270381859116556) [Z4 Z6]
+ (0.11270381859116556) [Z5 Z7]
+ (0.11952441016886267) [Z0 Z4]
+ (0.11952441016886267) [Z1 Z5]
+ (0.13401737372649042) [Z0 Z6]
+ (0.13401737372649042) [Z1 Z7]
+ (0.1373494221014767) [Z0 Z5]
+ (0.1373494221014767) [Z1 Z4]
+ (0.1376685913361254) [Z2 Z4]
+ (0.1376685913361254) [Z3 Z5]
+ (0.14138903590142135) [Z4 Z7]
+ (0.14138903590142135) [Z5 Z6]
+ (0.14722930783256924) [Z2 Z5]
+ (0.14722930783256924) [Z3 Z4]
+ (0.14926347060030007) [Z4 Z5]
+ (0.14973497005433192) [Z2 Z6]
+ (0.14973497005433192) [Z3 Z7]
+ (0.15138342699111423) [Z0 Z7]
+ (0.15138342699111423) [Z1 Z6]
+ (0.15435760065302265) [Z6 Z7]
+ (0.15582280685856634) [Z2 Z7]
+ (0.15582280685856634) [Z3 Z6]
+ (0.16756669356165346) [Z0 Z2]
+ (0.16756669356165346) [Z1 Z3]
+ (0.18144009362928004) [Z0 Z3]
+ (0.18144009362928004) [Z1 Z2]
+ (0.1939257433496631) [Z0 Z1]
+ (0.22003977240274578) [Z2 Z3]
+ (7.038023697217582e-06) [Y4 Z5 Y6]
+ (7.038023697217582e-06) [X4 Z5 X6]
+ (7.038023697217582e-06) [Y5 Z6 Y7]
+ (7.038023697217582e-06) [X5 Z6 X7]
+ (-0.028685217310255782) [Y4 Y5 X6 X7]
+ (-0.028685217310255782) [X4 X5 Y6 Y7]
+ (-0.01782501193261402) [Y0 Y1 X4 X5]
+ (-0.01782501193261402) [X0 X1 Y4 Y5]
+ (-0.0173660532646238) [Y0 Y1 X6 X7]
+ (-0.0173660532646238) [X0 X1 Y6 Y7]
+ (-0.01387340006762657) [Y0 Y1 X2 X3]
+ (-0.01387340006762657) [X0 X1 Y2 Y3]
+ (-0.009560716496443829) [Y2 Y3 X4 X5]
+ (-0.009560716496443829) [X2 X3 Y4 Y5]
+ (-0.006087836804234402) [Y2 Y3 X6 X7]
+ (-0.006087836804234402) [X2 X3 Y6 Y7]
+ (-0.00029222567245334357) [Y1 Y2 X3 X4]
+ (-0.00029222567245334357) [X1 X2 Y3 Y4]
+ (-7.954224451177383e-06) [Y4 Z5 Y6 Z7]
+ (-7.954224451177383e-06) [X4 Z5 X6 Z7]
+ (-6.628427142990742e-07) [Y2 X3 X5 Y6]
+ (-6.628427142990742e-07) [Y2 Y3 Y5 Y6]
+ (-6.628427142990742e-07) [X2 X3 X5 X6]
+ (-6.628427142990742e-07) [X2 Y3 Y5 X6]
+ (9.344970287317178e-07) [Z2 Y5 Z6 Y7]
+ (9.344970287317178e-07) [Z2 X5 Z6 X7]
+ (9.344970287317178e-07) [Z3 Y4 Z5 Y6]
+ (9.344970287317178e-07) [Z3 X4 Z5 X6]
+ (1.0357924821644514e-06) [Y0 X1 X5 Y6]
+ (1.0357924821644514e-06) [Y0 Y1 Y5 Y6]
+ (1.0357924821644514e-06) [X0 X1 X5 X6]
+ (1.0357924821644514e-06) [X0 Y1 Y5 X6]
+ (1.597339743030792e-06) [Z2 Y4 Z5 Y6]
+ (1.597339743030792e-06) [Z2 X4 Z5 X6]
+ (1.597339743030792e-06) [Z3 Y5 Z6 Y7]
+ (1.597339743030792e-06) [Z3 X5 Z6 X7]
+ (1.8551374835864068e-06) [Z0 Y4 Z5 Y6]
+ (1.8551374835864068e-06) [Z0 X4 Z5 X6]
+ (1.8551374835864068e-06) [Z1 Y5 Z6 Y7]
+ (1.8551374835864068e-06) [Z1 X5 Z6 X7]
+ (2.8909299657521592e-06) [Z0 Y5 Z6 Y7]
+ (2.8909299657521592e-06) [Z0 X5 Z6 X7]
+ (2.8909299657521592e-06) [Z1 Y4 Z5 Y6]
+ (2.8909299657521592e-06) [Z1 X4 Z5 X6]
+ (8.194104912529615e-06) [Z4 Y5 Z6 Y7]
+ (8.194104912529615e-06) [Z4 X5 Z6 X7]
+ (0.00029222567245334357) [Y1 X2 X3 Y4]
+ (0.00029222567245334357) [X1 Y2 Y3 X4]
+ (0.006087836804234402) [Y2 X3 X6 Y7]
+ (0.006087836804234402) [X2 Y3 Y6 X7]
+ (0.009560716496443829) [Y2 X3 X4 Y5]
+ (0.009560716496443829) [X2 Y3 Y4 X5]
+ (0.01130720803001204) [Y1 Z2 Z3 Y5]
+ (0.01130720803001204) [X1 Z2 Z3 X5]
+ (0.01387340006762657) [Y0 X1 X2 Y3]
+ (0.01387340006762657) [X0 Y1 Y2 X3]
+ (0.0173660532646238) [Y0 X1 X6 Y7]
+ (0.0173660532646238) [X0 Y1 Y6 X7]
+ (0.01782501193261402) [Y0 X1 X4 Y5]
+ (0.01782501193261402) [X0 Y1 Y4 X5]
+ (0.028685217310255782) [Y4 X5 X6 Y7]
+ (0.028685217310255782) [X4 Y5 Y6 X7]
+ (0.029812299601098526) [Y0 Z1 Z2 Y4]
+ (0.029812299601098526) [X0 Z1 Z2 X4]
+ (0.029812299601098526) [Y1 Z3 Z4 Y5]
+ (0.029812299601098526) [X1 Z3 Z4 X5]
+ (0.03010452527355187) [Y0 Z1 Z3 Y4]
+ (0.03010452527355187) [X0 Z1 Z3 X4]
+ (0.03010452527355187) [Y1 Z2 Z4 Y5]
+ (0.03010452527355187) [X1 Z2 Z4 X5]
+ (0.030787440718556946) [Y0 Z2 Z3 Y4]
+ (0.030787440718556946) [X0 Z2 Z3 X4]
+ (0.04375171612132008) [Y0 Z1 Z2 Z3 Y4]
+ (0.04375171612132008) [X0 Z1 Z2 Z3 X4]
+ (0.04375171612132008) [Y1 Z2 Z3 Z4 Y5]
+ (0.04375171612132008) [X1 Z2 Z3 Z4 X5]
+ (-0.014564473640802622) [Y1 Z2 Z3 X4 X6 Y7]
+ (-0.014564473640802622) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-0.014564473640802622) [X1 Z2 Z3 X4 X6 X7]
+ (-0.014564473640802622) [X1 Z2 Z3 Y4 Y6 X7]
+ (-4.183808719408766e-06) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-4.183808719408766e-06) [X1 Z2 Z3 Z4 Z5 X7]
+ (-3.3130170413895807e-06) [Y1 Z2 Z3 X4 X5 Y6]
+ (-3.3130170413895807e-06) [X1 Z2 Z3 Y4 Y5 X6]
+ (-1.0357924821644514e-06) [Y0 X1 X4 Z5 Z6 Y7]
+ (-1.0357924821644514e-06) [X0 Y1 Y4 Z5 Z6 X7]
+ (-6.628427142990742e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-6.628427142990742e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-3.3280396332968946e-07) [Y1 Y2 X3 Z4 Z5 X6]
+ (-3.3280396332968946e-07) [X1 X2 Y3 Z4 Z5 Y6]
+ (3.3280396332968946e-07) [Y1 X2 X3 Z4 Z5 Y6]
+ (3.3280396332968946e-07) [X1 Y2 Y3 Z4 Z5 X6]
+ (6.628427142990742e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (6.628427142990742e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (1.0357924821644514e-06) [Y0 Y1 X4 Z5 Z6 X7]
+ (1.0357924821644514e-06) [X0 X1 Y4 Z5 Z6 Y7]
+ (3.2111873983801595e-06) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (3.2111873983801595e-06) [X0 Z1 Z2 Z3 Z5 X6]
+ (3.2111873983801595e-06) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (3.2111873983801595e-06) [X1 Z2 Z3 Z4 Z6 X7]
+ (3.2774382311010278e-06) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (3.2774382311010278e-06) [X0 Z1 Z2 Z4 Z5 X6]
+ (3.2774382311010278e-06) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (3.2774382311010278e-06) [X1 Z3 Z4 Z5 Z6 X7]
+ (3.3130170413895807e-06) [Y1 Z2 Z3 Y4 X5 X6]
+ (3.3130170413895807e-06) [X1 Z2 Z3 X4 Y5 Y6]
+ (3.6102421944307172e-06) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (3.6102421944307172e-06) [X0 Z1 Z3 Z4 Z5 X6]
+ (3.6102421944307172e-06) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (3.6102421944307172e-06) [X1 Z2 Z4 Z5 Z6 X7]
+ (3.7695835582091716e-06) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (3.7695835582091716e-06) [X0 Z2 Z3 Z4 Z5 X6]
+ (6.524204439771475e-06) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (6.524204439771475e-06) [X0 Z1 Z2 Z3 Z4 X6]
+ (6.524204439771475e-06) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (6.524204439771475e-06) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.0002922256724533436) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (0.0002922256724533436) [Y0 Z1 Y2 X3 Z4 X5]
+ (0.0002922256724533436) [X0 Z1 X2 Y3 Z4 Y5]
+ (0.0002922256724533436) [X0 Z1 X2 X3 Z4 X5]
+ (0.010540434329225636) [Y0 Z1 Z2 Z3 Y4 Z7]
+ (0.010540434329225636) [X0 Z1 Z2 Z3 X4 Z7]
+ (0.010540434329225636) [Y1 Z2 Z3 Z4 Y5 Z6]
+ (0.010540434329225636) [X1 Z2 Z3 Z4 X5 Z6]
+ (0.01130720803001204) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (0.01130720803001204) [X0 Z1 Z2 Z3 X4 Z5]
+ (0.025104907970028262) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (0.025104907970028262) [X0 Z1 Z2 Z3 X4 Z6]
+ (0.025104907970028262) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (0.025104907970028262) [X1 Z2 Z3 Z4 X5 Z7]
+ (0.030787440718556942) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (0.030787440718556942) [Z0 X1 Z2 Z3 Z4 X5]
+ (5.105680958886812e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (5.105680958886812e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (5.105680958886812e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (5.105680958886812e-06) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.014564473640802624) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-0.014564473640802624) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-4.183808719408766e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-4.183808719408766e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-3.3130170413895807e-06) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (-3.3130170413895807e-06) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (-3.3130170413895807e-06) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (-3.3130170413895807e-06) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (3.3280396332968946e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (3.3280396332968946e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (3.3280396332968946e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (3.3280396332968946e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (3.7695835582091716e-06) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (3.7695835582091716e-06) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (0.014564473640802624) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (0.014564473640802624) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
  (-46.46390678868895) [I0]
+ (0.7829661725950194) [Z10]
+ (0.7829661725950198) [Z11]
+ (0.8084581961720565) [Z12]
+ (0.8084581961720565) [Z13]
+ (1.2034402289145656) [Z4]
+ (1.2034402289145658) [Z5]
+ (1.3096862988615527) [Z7]
+ (1.3096862988615534) [Z6]
+ (1.3693525634718244) [Z8]
+ (1.3693525634718247) [Z9]
+ (1.653894222683162) [Z2]
+ (1.653894222683162) [Z3]
+ (12.412630742111766) [Z0]
+ (12.412630742111766) [Z1]
+ (-8.194261372992256e-06) [Y10 Y12]
+ (-8.194261372992256e-06) [X10 X12]
+ (-1.8540608579617089e-06) [Y5 Y7]
+ (-1.8540608579617089e-06) [X5 X7]
+ (-7.764994119884017e-07) [Y3 Y5]
+ (-7.764994119884017e-07) [X3 X5]
+ (-5.929765814613117e-07) [Y4 Y6]
+ (-5.929765814613117e-07) [X4 X6]
+ (1.6021167409676577e-06) [Y2 Y4]
+ (1.6021167409676577e-06) [X2 X4]
+ (7.95441317675942e-06) [Y11 Y13]
+ (7.95441317675942e-06) [X11 X13]
+ (0.003276971931231578) [Y1 Y3]
+ (0.003276971931231578) [X1 X3]
+ (0.10433064780651319) [Y0 Y2]
+ (0.10433064780651319) [X0 X2]
+ (0.1127038692033219) [Z10 Z12]
+ (0.1127038692033219) [Z11 Z13]
+ (0.11383573679388642) [Z4 Z12]
+ (0.11383573679388642) [Z5 Z13]
+ (0.11952438964682735) [Z6 Z10]
+ (0.11952438964682735) [Z7 Z11]
+ (0.1248999091723759) [Z4 Z10]
+ (0.1248999091723759) [Z5 Z11]
+ (0.1249580773950319) [Z2 Z4]
+ (0.1249580773950319) [Z3 Z5]
+ (0.12799502492468365) [Z2 Z10]
+ (0.12799502492468365) [Z3 Z11]
+ (0.13401715261963756) [Z6 Z12]
+ (0.13401715261963756) [Z7 Z13]
+ (0.13701191674040797) [Z4 Z6]
+ (0.13701191674040797) [Z5 Z7]
+ (0.1373495306426129) [Z6 Z11]
+ (0.1373495306426129) [Z7 Z10]
+ (0.13739104762683255) [Z2 Z6]
+ (0.13739104762683255) [Z3 Z7]
+ (0.1376687264585255) [Z8 Z10]
+ (0.1376687264585255) [Z9 Z11]
+ (0.14011289865354803) [Z2 Z12]
+ (0.14011289865354803) [Z3 Z13]
+ (0.1413890529194283) [Z10 Z13]
+ (0.1413890529194283) [Z11 Z12]
+ (0.1425799771248575) [Z4 Z11]
+ (0.1425799771248575) [Z5 Z10]
+ (0.14722943218766166) [Z8 Z11]
+ (0.14722943218766166) [Z9 Z10]
+ (0.1489943057506557) [Z4 Z7]
+ (0.1489943057506557) [Z5 Z6]
+ (0.14926355147388953) [Z10 Z11]
+ (0.14960702684445296) [Z4 Z8]
+ (0.14960702684445296) [Z5 Z9]
+ (0.1497348680349691) [Z8 Z12]
+ (0.1497348680349691) [Z9 Z13]
+ (0.1507140812100827) [Z2 Z8]
+ (0.1507140812100827) [Z3 Z9]
+ (0.15138327161428874) [Z6 Z13]
+ (0.15138327161428874) [Z7 Z12]
+ (0.15215040708869043) [Z4 Z13]
+ (0.15215040708869043) [Z5 Z12]
+ (0.15337968243314137) [Z2 Z11]
+ (0.15337968243314137) [Z3 Z10]
+ (0.15435748657223655) [Z12 Z13]
+ (0.15569010671752448) [Z2 Z13]
+ (0.15569010671752448) [Z3 Z12]
+ (0.15582269051553105) [Z8 Z13]
+ (0.15582269051553105) [Z9 Z12]
+ (0.15676396176430993) [Z4 Z9]
+ (0.15676396176430993) [Z5 Z8]
+ (0.1575531479798566) [Z4 Z5]
+ (0.16079764534838553) [Z2 Z5]
+ (0.16079764534838553) [Z3 Z4]
+ (0.1675665326546135) [Z6 Z8]
+ (0.1675665326546135) [Z7 Z9]
+ (0.16853486561579936) [Z2 Z7]
+ (0.16853486561579936) [Z3 Z6]
+ (0.1814399144030399) [Z6 Z9]
+ (0.1814399144030399) [Z7 Z8]
+ (0.18189085790751297) [Z2 Z3]
+ (0.18690820476912504) [Z2 Z9]
+ (0.18690820476912504) [Z3 Z8]
+ (0.19299723935364152) [Z0 Z10]
+ (0.19299723935364152) [Z1 Z11]
+ (0.19392534613270457) [Z6 Z7]
+ (0.19661770890342128) [Z0 Z4]
+ (0.19661770890342128) [Z1 Z5]
+ (0.19936354537360812) [Z0 Z5]
+ (0.19936354537360812) [Z1 Z4]
+ (0.20072866460441688) [Z0 Z11]
+ (0.20072866460441688) [Z1 Z10]
+ (0.21102659849791494) [Z0 Z12]
+ (0.21102659849791494) [Z1 Z13]
+ (0.2163103749863179) [Z0 Z13]
+ (0.2163103749863179) [Z1 Z12]
+ (0.220039773343761) [Z8 Z9]
+ (0.23671080783830264) [Z0 Z2]
+ (0.23671080783830264) [Z1 Z3]
+ (0.24164663936017358) [Z0 Z6]
+ (0.24164663936017358) [Z1 Z7]
+ (0.24853483371314422) [Z0 Z7]
+ (0.24853483371314422) [Z1 Z6]
+ (0.2512944567459151) [Z0 Z3]
+ (0.2512944567459151) [Z1 Z2]
+ (0.2723251830660567) [Z0 Z8]
+ (0.2723251830660567) [Z1 Z9]
+ (0.278834544267234) [Z0 Z9]
+ (0.278834544267234) [Z1 Z8]
+ (1.1861763734860453) [Z0 Z1]
+ (-1.2260484988915952e-05) [Y4 Z5 Y6]
+ (-1.2260484988915952e-05) [X4 Z5 X6]
+ (-1.2260484988915952e-05) [Y5 Z6 Y7]
+ (-1.2260484988915952e-05) [X5 Z6 X7]
+ (-1.072231215985018e-05) [Y11 Z12 Y13]
+ (-1.072231215985018e-05) [X11 Z12 X13]
+ (-1.0722312159850178e-05) [Y10 Z11 Y12]
+ (-1.0722312159850178e-05) [X10 Z11 X12]
+ (-3.887051671711167e-06) [Y2 Z3 Y4]
+ (-3.887051671711167e-06) [X2 Z3 X4]
+ (-3.887051671711167e-06) [Y3 Z4 Y5]
+ (-3.887051671711167e-06) [X3 Z4 X5]
+ (0.1250703257977216) [Y1 Z2 Y3]
+ (0.1250703257977216) [X1 Z2 X3]
+ (0.12507032579772165) [Y0 Z1 Y2]
+ (0.12507032579772165) [X0 Z1 X2]
+ (-0.038314670294804) [Y4 Y5 X12 X13]
+ (-0.038314670294804) [X4 X5 Y12 Y13]
+ (-0.036194123559042314) [Y2 Y3 X8 X9]
+ (-0.036194123559042314) [X2 X3 Y8 Y9]
+ (-0.03583956795335364) [Y2 Y3 X4 X5]
+ (-0.03583956795335364) [X2 X3 Y4 Y5]
+ (-0.031143817988966822) [Y2 Y3 X6 X7]
+ (-0.031143817988966822) [X2 X3 Y6 Y7]
+ (-0.028685183716106403) [Y10 Y11 X12 X13]
+ (-0.028685183716106403) [X10 X11 Y12 Y13]
+ (-0.025996177598021676) [Y3 Z4 Z5 Y7]
+ (-0.025996177598021676) [X3 Z4 Z5 X7]
+ (-0.025384657508457732) [Y2 Y3 X10 X11]
+ (-0.025384657508457732) [X2 X3 Y10 Y11]
+ (-0.01902824244384799) [Y3 Y4 X11 X12]
+ (-0.01902824244384799) [X3 X4 Y11 Y12]
+ (-0.01782514099578553) [Y6 Y7 X10 X11]
+ (-0.01782514099578553) [X6 X7 Y10 Y11]
+ (-0.017680067952481626) [Y4 Y5 X10 X11]
+ (-0.017680067952481626) [X4 X5 Y10 Y11]
+ (-0.01736611899465119) [Y6 Y7 X12 X13]
+ (-0.01736611899465119) [X6 X7 Y12 Y13]
+ (-0.01557720806397644) [Y2 Y3 X12 X13]
+ (-0.01557720806397644) [X2 X3 Y12 Y13]
+ (-0.014583648907612476) [Y0 Y1 X2 X3]
+ (-0.014583648907612476) [X0 X1 Y2 Y3]
+ (-0.013873381748426379) [Y6 Y7 X8 X9]
+ (-0.013873381748426379) [X6 X7 Y8 Y9]
+ (-0.011982389010247708) [Y4 Y5 X6 X7]
+ (-0.011982389010247708) [X4 X5 Y6 Y7]
+ (-0.011285190200840543) [Y5 X6 X11 Y12]
+ (-0.011285190200840543) [X5 Y6 Y11 X12]
+ (-0.009560705729136197) [Y8 Y9 X10 X11]
+ (-0.009560705729136197) [X8 X9 Y10 Y11]
+ (-0.008125251921380987) [Y1 X2 X8 Y9]
+ (-0.008125251921380987) [Y1 Y2 Y8 Y9]
+ (-0.008125251921380987) [X1 X2 X8 X9]
+ (-0.008125251921380987) [X1 Y2 Y8 X9]
+ (-0.007731425250775374) [Y0 Y1 X10 X11]
+ (-0.007731425250775374) [X0 X1 Y10 Y11]
+ (-0.007156934919856985) [Y4 Y5 X8 X9]
+ (-0.007156934919856985) [X4 X5 Y8 Y9]
+ (-0.006888194352970668) [Y0 Y1 X6 X7]
+ (-0.006888194352970668) [X0 X1 Y6 Y7]
+ (-0.006509361201177238) [Y0 Y1 X8 X9]
+ (-0.006509361201177238) [X0 X1 Y8 Y9]
+ (-0.006087822480561936) [Y8 Y9 X12 X13]
+ (-0.006087822480561936) [X8 X9 Y12 Y13]
+ (-0.005283776488402955) [Y0 Y1 X12 X13]
+ (-0.005283776488402955) [X0 X1 Y12 Y13]
+ (-0.005143391768824774) [Y3 X4 X5 Y6]
+ (-0.005143391768824774) [X3 Y4 Y5 X6]
+ (-0.004684903388155237) [Y1 X2 X6 Y7]
+ (-0.004684903388155237) [Y1 Y2 Y6 Y7]
+ (-0.004684903388155237) [X1 X2 X6 X7]
+ (-0.004684903388155237) [X1 Y2 Y6 X7]
+ (-0.004575007626639159) [Y1 X2 X12 Y13]
+ (-0.004575007626639159) [Y1 Y2 Y12 Y13]
+ (-0.004575007626639159) [X1 X2 X12 X13]
+ (-0.004575007626639159) [X1 Y2 Y12 X13]
+ (-0.00442485544944185) [Y1 X2 X4 Y5]
+ (-0.00442485544944185) [Y1 Y2 Y4 Y5]
+ (-0.00442485544944185) [X1 X2 X4 X5]
+ (-0.00442485544944185) [X1 Y2 Y4 X5]
+ (-0.003479511890334004) [Y2 Z3 Z5 Y6]
+ (-0.003479511890334004) [X2 Z3 Z5 X6]
+ (-0.003479511890334004) [Y3 Z4 Z6 Y7]
+ (-0.003479511890334004) [X3 Z4 Z6 X7]
+ (-0.002745836470186817) [Y0 Y1 X4 X5]
+ (-0.002745836470186817) [X0 X1 Y4 Y5]
+ (-0.0017992194936628824) [Y1 X2 X10 Y11]
+ (-0.0017992194936628824) [Y1 Y2 Y10 Y11]
+ (-0.0017992194936628824) [X1 X2 X10 X11]
+ (-0.0017992194936628824) [X1 Y2 Y10 X11]
+ (-0.00029219862611128274) [Y7 Y8 X9 X10]
+ (-0.00029219862611128274) [X7 X8 Y9 Y10]
+ (-8.194261372992256e-06) [Z10 Y11 Z12 Y13]
+ (-8.194261372992256e-06) [Z10 X11 Z12 X13]
+ (-7.801707501157475e-06) [Y2 Z3 Y4 Z11]
+ (-7.801707501157475e-06) [X2 Z3 X4 Z11]
+ (-7.801707501157475e-06) [Y3 Z4 Y5 Z10]
+ (-7.801707501157475e-06) [X3 Z4 X5 Z10]
+ (-4.643051068956277e-06) [Y3 X4 X10 Y11]
+ (-4.643051068956277e-06) [Y3 Y4 Y10 Y11]
+ (-4.643051068956277e-06) [X3 X4 X10 X11]
+ (-4.643051068956277e-06) [X3 Y4 Y10 X11]
+ (-4.588855156065566e-06) [Y4 Z5 Y6 Z13]
+ (-4.588855156065566e-06) [X4 Z5 X6 Z13]
+ (-4.588855156065566e-06) [Y5 Z6 Y7 Z12]
+ (-4.588855156065566e-06) [X5 Z6 X7 Z12]
+ (-4.556569218720731e-06) [Y5 X6 X12 Y13]
+ (-4.556569218720731e-06) [Y5 Y6 Y12 Y13]
+ (-4.556569218720731e-06) [X5 X6 X12 X13]
+ (-4.556569218720731e-06) [X5 Y6 Y12 X13]
+ (-3.6945132950781494e-06) [Y4 X5 X11 Y12]
+ (-3.6945132950781494e-06) [Y4 Y5 Y11 Y12]
+ (-3.6945132950781494e-06) [X4 X5 X11 X12]
+ (-3.6945132950781494e-06) [X4 Y5 Y11 X12]
+ (-3.344081556381499e-06) [Z0 Y5 Z6 Y7]
+ (-3.344081556381499e-06) [Z0 X5 Z6 X7]
+ (-3.344081556381499e-06) [Z1 Y4 Z5 Y6]
+ (-3.344081556381499e-06) [Z1 X4 Z5 X6]
+ (-3.1586564322011977e-06) [Y2 Z3 Y4 Z10]
+ (-3.1586564322011977e-06) [X2 Z3 X4 Z10]
+ (-3.1586564322011977e-06) [Y3 Z4 Y5 Z11]
+ (-3.1586564322011977e-06) [X3 Z4 X5 Z11]
+ (-3.099349243512085e-06) [Z0 Y4 Z5 Y6]
+ (-3.099349243512085e-06) [Z0 X4 Z5 X6]
+ (-3.099349243512085e-06) [Z1 Y5 Z6 Y7]
+ (-3.099349243512085e-06) [Z1 X5 Z6 X7]
+ (-2.890967881868803e-06) [Z6 Y11 Z12 Y13]
+ (-2.890967881868803e-06) [Z6 X11 Z12 X13]
+ (-2.890967881868803e-06) [Z7 Y10 Z11 Y12]
+ (-2.890967881868803e-06) [Z7 X10 Z11 X12]
+ (-2.1776646055265396e-06) [Z0 Y10 Z11 Y12]
+ (-2.1776646055265396e-06) [Z0 X10 Z11 X12]
+ (-2.1776646055265396e-06) [Z1 Y11 Z12 Y13]
+ (-2.1776646055265396e-06) [Z1 X11 Z12 X13]
+ (-1.8818501831587005e-06) [Y4 Z5 Y6 Z9]
+ (-1.8818501831587005e-06) [X4 Z5 X6 Z9]
+ (-1.8818501831587005e-06) [Y5 Z6 Y7 Z8]
+ (-1.8818501831587005e-06) [X5 Z6 X7 Z8]
+ (-1.8551201219022678e-06) [Z6 Y10 Z11 Y12]
+ (-1.8551201219022678e-06) [Z6 X10 Z11 X12]
+ (-1.8551201219022678e-06) [Z7 Y11 Z12 Y13]
+ (-1.8551201219022678e-06) [Z7 X11 Z12 X13]
+ (-1.8540608579617089e-06) [Y4 Z5 Y6 Z7]
+ (-1.8540608579617089e-06) [X4 Z5 X6 Z7]
+ (-1.8163031703082727e-06) [Z4 Y11 Z12 Y13]
+ (-1.8163031703082727e-06) [Z4 X11 Z12 X13]
+ (-1.8163031703082727e-06) [Z5 Y10 Z11 Y12]
+ (-1.8163031703082727e-06) [Z5 X10 Z11 X12]
+ (-1.6923978287414904e-06) [Y4 Z5 Y6 Z10]
+ (-1.6923978287414904e-06) [X4 Z5 X6 Z10]
+ (-1.6923978287414904e-06) [Y5 Z6 Y7 Z11]
+ (-1.6923978287414904e-06) [X5 Z6 X7 Z11]
+ (-1.614879414345991e-06) [Z0 Y11 Z12 Y13]
+ (-1.614879414345991e-06) [Z0 X11 Z12 X13]
+ (-1.614879414345991e-06) [Z1 Y10 Z11 Y12]
+ (-1.614879414345991e-06) [Z1 X10 Z11 X12]
+ (-1.5973171981129822e-06) [Z8 Y10 Z11 Y12]
+ (-1.5973171981129822e-06) [Z8 X10 Z11 X12]
+ (-1.5973171981129822e-06) [Z9 Y11 Z12 Y13]
+ (-1.5973171981129822e-06) [Z9 X11 Z12 X13]
+ (-1.4548424489223256e-06) [Y3 X4 X6 Y7]
+ (-1.4548424489223256e-06) [Y3 Y4 Y6 Y7]
+ (-1.4548424489223256e-06) [X3 X4 X6 X7]
+ (-1.4548424489223256e-06) [X3 Y4 Y6 X7]
+ (-1.3980449080828426e-06) [Y4 Z5 Y6 Z8]
+ (-1.3980449080828426e-06) [X4 Z5 X6 Z8]
+ (-1.3980449080828426e-06) [Y5 Z6 Y7 Z9]
+ (-1.3980449080828426e-06) [X5 Z6 X7 Z9]
+ (-1.1954890096446853e-06) [Y2 Z3 Y4 Z7]
+ (-1.1954890096446853e-06) [X2 Z3 X4 Z7]
+ (-1.1954890096446853e-06) [Y3 Z4 Y5 Z6]
+ (-1.1954890096446853e-06) [X3 Z4 X5 Z6]
+ (-1.1908508079988916e-06) [Z0 Y3 Z4 Y5]
+ (-1.1908508079988916e-06) [Z0 X3 Z4 X5]
+ (-1.1908508079988916e-06) [Z1 Y2 Z3 Y4]
+ (-1.1908508079988916e-06) [Z1 X2 Z3 X4]
+ (-1.1708301370462796e-06) [Z2 Y5 Z6 Y7]
+ (-1.1708301370462796e-06) [Z2 X5 Z6 X7]
+ (-1.1708301370462796e-06) [Z3 Y4 Z5 Y6]
+ (-1.1708301370462796e-06) [Z3 X4 Z5 X6]
+ (-1.0632283425835972e-06) [Z2 Y10 Z11 Y12]
+ (-1.0632283425835972e-06) [Z2 X10 Z11 X12]
+ (-1.0632283425835972e-06) [Z3 Y11 Z12 Y13]
+ (-1.0632283425835972e-06) [Z3 X11 Z12 X13]
+ (-1.0358477599665353e-06) [Y6 X7 X11 Y12]
+ (-1.0358477599665353e-06) [Y6 Y7 Y11 Y12]
+ (-1.0358477599665353e-06) [X6 X7 X11 X12]
+ (-1.0358477599665353e-06) [X6 Y7 Y11 X12]
+ (-9.509249751529455e-07) [Z2 Y4 Z5 Y6]
+ (-9.509249751529455e-07) [Z2 X4 Z5 X6]
+ (-9.509249751529455e-07) [Z3 Y5 Z6 Y7]
+ (-9.509249751529455e-07) [Z3 X5 Z6 X7]
+ (-9.34455777916454e-07) [Z8 Y11 Z12 Y13]
+ (-9.34455777916454e-07) [Z8 X11 Z12 X13]
+ (-9.34455777916454e-07) [Z9 Y10 Z11 Y12]
+ (-9.34455777916454e-07) [Z9 X10 Z11 X12]
+ (-8.337746751113972e-07) [Z0 Y2 Z3 Y4]
+ (-8.337746751113972e-07) [Z0 X2 Z3 X4]
+ (-8.337746751113972e-07) [Z1 Y3 Z4 Y5]
+ (-8.337746751113972e-07) [Z1 X3 Z4 X5]
+ (-7.956895372169788e-07) [Y3 X4 X8 Y9]
+ (-7.956895372169788e-07) [Y3 Y4 Y8 Y9]
+ (-7.956895372169788e-07) [X3 X4 X8 X9]
+ (-7.956895372169788e-07) [X3 Y4 Y8 X9]
+ (-7.764994119884017e-07) [Y2 Z3 Y4 Z5]
+ (-7.764994119884017e-07) [X2 Z3 X4 Z5]
+ (-5.929765814613117e-07) [Z4 Y5 Z6 Y7]
+ (-5.929765814613117e-07) [Z4 X5 Z6 X7]
+ (-5.770052992860527e-07) [Y2 Z3 Y4 Z9]
+ (-5.770052992860527e-07) [X2 Z3 X4 Z9]
+ (-5.770052992860527e-07) [Y3 Z4 Y5 Z8]
+ (-5.770052992860527e-07) [X3 Z4 X5 Z8]
+ (-5.471647745194209e-07) [Y1 Y2 X11 X12]
+ (-5.471647745194209e-07) [X1 X2 Y11 Y12]
+ (-4.838052750758578e-07) [Y5 X6 X8 Y9]
+ (-4.838052750758578e-07) [Y5 Y6 Y8 Y9]
+ (-4.838052750758578e-07) [X5 X6 X8 X9]
+ (-4.838052750758578e-07) [X5 Y6 Y8 X9]
+ (-3.570761328874944e-07) [Y0 X1 X3 Y4]
+ (-3.570761328874944e-07) [Y0 Y1 Y3 Y4]
+ (-3.570761328874944e-07) [X0 X1 X3 X4]
+ (-3.570761328874944e-07) [X0 Y1 Y3 X4]
+ (-2.4473231286941343e-07) [Y0 X1 X5 Y6]
+ (-2.4473231286941343e-07) [Y0 Y1 Y5 Y6]
+ (-2.4473231286941343e-07) [X0 X1 X5 X6]
+ (-2.4473231286941343e-07) [X0 Y1 Y5 X6]
+ (-2.1990516189333408e-07) [Y2 X3 X5 Y6]
+ (-2.1990516189333408e-07) [Y2 Y3 Y5 Y6]
+ (-2.1990516189333408e-07) [X2 X3 X5 X6]
+ (-2.1990516189333408e-07) [X2 Y3 Y5 X6]
+ (-1.933241277107054e-07) [Y1 X2 X3 Y4]
+ (-1.933241277107054e-07) [X1 Y2 Y3 X4]
+ (-1.2919694863578368e-07) [Y1 Z2 Z3 Y5]
+ (-1.2919694863578368e-07) [X1 Z2 Z3 X5]
+ (1.7379332624937822e-07) [Y0 Z1 Z3 Y4]
+ (1.7379332624937822e-07) [X0 Z1 Z3 X4]
+ (1.7379332624937822e-07) [Y1 Z2 Z4 Y5]
+ (1.7379332624937822e-07) [X1 Z2 Z4 X5]
+ (1.933241277107054e-07) [Y1 Y2 X3 X4]
+ (1.933241277107054e-07) [X1 X2 Y3 Y4]
+ (2.1868423793092617e-07) [Y2 Z3 Y4 Z8]
+ (2.1868423793092617e-07) [X2 Z3 X4 Z8]
+ (2.1868423793092617e-07) [Y3 Z4 Y5 Z9]
+ (2.1868423793092617e-07) [X3 Z4 X5 Z9]
+ (2.5935343927764023e-07) [Y2 Z3 Y4 Z6]
+ (2.5935343927764023e-07) [X2 Z3 X4 Z6]
+ (2.5935343927764023e-07) [Y3 Z4 Y5 Z7]
+ (2.5935343927764023e-07) [X3 Z4 X5 Z7]
+ (3.606071868062085e-07) [Y0 Z1 Z2 Y4]
+ (3.606071868062085e-07) [X0 Z1 Z2 X4]
+ (3.606071868062085e-07) [Y1 Z3 Z4 Y5]
+ (3.606071868062085e-07) [X1 Z3 Z4 X5]
+ (5.471647745194209e-07) [Y1 X2 X11 Y12]
+ (5.471647745194209e-07) [X1 Y2 Y11 X12]
+ (5.627851911805488e-07) [Y0 X1 X11 Y12]
+ (5.627851911805488e-07) [Y0 Y1 Y11 Y12]
+ (5.627851911805488e-07) [X0 X1 X11 X12]
+ (5.627851911805488e-07) [X0 Y1 Y11 X12]
+ (6.628614201965279e-07) [Y8 X9 X11 Y12]
+ (6.628614201965279e-07) [Y8 Y9 Y11 Y12]
+ (6.628614201965279e-07) [X8 X9 X11 X12]
+ (6.628614201965279e-07) [X8 Y9 Y11 X12]
+ (1.1094407591054345e-06) [Z2 Y11 Z12 Y13]
+ (1.1094407591054345e-06) [Z2 X11 Z12 X13]
+ (1.1094407591054345e-06) [Z3 Y10 Z11 Y12]
+ (1.1094407591054345e-06) [Z3 X10 Z11 X12]
+ (1.6021167409676577e-06) [Z2 Y3 Z4 Y5]
+ (1.6021167409676577e-06) [Z2 X3 Z4 X5]
+ (1.8782101247698767e-06) [Z4 Y10 Z11 Y12]
+ (1.8782101247698767e-06) [Z4 X10 Z11 X12]
+ (1.8782101247698767e-06) [Z5 Y11 Z12 Y13]
+ (1.8782101247698767e-06) [Z5 X11 Z12 X13]
+ (2.1726691016890317e-06) [Y2 X3 X11 Y12]
+ (2.1726691016890317e-06) [Y2 Y3 Y11 Y12]
+ (2.1726691016890317e-06) [X2 X3 X11 X12]
+ (2.1726691016890317e-06) [X2 Y3 Y11 X12]
+ (3.1174479459651303e-06) [Y0 Z2 Z3 Y4]
+ (3.1174479459651303e-06) [X0 Z2 Z3 X4]
+ (3.539054184866549e-06) [Y2 Z3 Y4 Z12]
+ (3.539054184866549e-06) [X2 Z3 X4 Z12]
+ (3.539054184866549e-06) [Y3 Z4 Y5 Z13]
+ (3.539054184866549e-06) [X3 Z4 X5 Z13]
+ (4.281913885172475e-06) [Y4 Z5 Y6 Z11]
+ (4.281913885172475e-06) [X4 Z5 X6 Z11]
+ (4.281913885172475e-06) [Y5 Z6 Y7 Z10]
+ (4.281913885172475e-06) [X5 Z6 X7 Z10]
+ (5.275883122598385e-06) [Y3 X4 X12 Y13]
+ (5.275883122598385e-06) [Y3 Y4 Y12 Y13]
+ (5.275883122598385e-06) [X3 X4 X12 X13]
+ (5.275883122598385e-06) [X3 Y4 Y12 X13]
+ (5.974311713913966e-06) [Y5 X6 X10 Y11]
+ (5.974311713913966e-06) [Y5 Y6 Y10 Y11]
+ (5.974311713913966e-06) [X5 X6 X10 X11]
+ (5.974311713913966e-06) [X5 Y6 Y10 X11]
+ (7.95441317675942e-06) [Y10 Z11 Y12 Z13]
+ (7.95441317675942e-06) [X10 Z11 X12 Z13]
+ (8.814937307464934e-06) [Y2 Z3 Y4 Z13]
+ (8.814937307464934e-06) [X2 Z3 X4 Z13]
+ (8.814937307464934e-06) [Y3 Z4 Y5 Z12]
+ (8.814937307464934e-06) [X3 Z4 X5 Z12]
+ (0.00029219862611128274) [Y7 X8 X9 Y10]
+ (0.00029219862611128274) [X7 Y8 Y9 X10]
+ (0.0004956762314924203) [Y2 Z4 Z5 Y6]
+ (0.0004956762314924203) [X2 Z4 Z5 X6]
+ (0.0011059037691896906) [Y0 Z1 Y2 Z5]
+ (0.0011059037691896906) [X0 Z1 X2 Z5]
+ (0.0011059037691896906) [Y1 Z2 Y3 Z4]
+ (0.0011059037691896906) [X1 Z2 X3 Z4]
+ (0.0016638798784907698) [Y2 Z3 Z4 Y6]
+ (0.0016638798784907698) [X2 Z3 Z4 X6]
+ (0.0016638798784907698) [Y3 Z5 Z6 Y7]
+ (0.0016638798784907698) [X3 Z5 Z6 X7]
+ (0.0017560707018412273) [Y0 Z1 Y2 Z11]
+ (0.0017560707018412273) [X0 Z1 X2 Z11]
+ (0.0017560707018412273) [Y1 Z2 Y3 Z10]
+ (0.0017560707018412273) [X1 Z2 X3 Z10]
+ (0.0023262306231581066) [Y0 Z1 Y2 Z13]
+ (0.0023262306231581066) [X0 Z1 X2 Z13]
+ (0.0023262306231581066) [Y1 Z2 Y3 Z12]
+ (0.0023262306231581066) [X1 Z2 X3 Z12]
+ (0.002745836470186817) [Y0 X1 X4 Y5]
+ (0.002745836470186817) [X0 Y1 Y4 X5]
+ (0.0029297686747510807) [Y0 Z1 Y2 Z9]
+ (0.0029297686747510807) [X0 Z1 X2 Z9]
+ (0.0029297686747510807) [Y1 Z2 Y3 Z8]
+ (0.0029297686747510807) [X1 Z2 X3 Z8]
+ (0.003276971931231578) [Y0 Z1 Y2 Z3]
+ (0.003276971931231578) [X0 Z1 X2 Z3]
+ (0.003347617530666272) [Y0 Z1 Y2 Z7]
+ (0.003347617530666272) [X0 Z1 X2 Z7]
+ (0.003347617530666272) [Y1 Z2 Y3 Z6]
+ (0.003347617530666272) [X1 Z2 X3 Z6]
+ (0.00355529019550411) [Y0 Z1 Y2 Z10]
+ (0.00355529019550411) [X0 Z1 X2 Z10]
+ (0.00355529019550411) [Y1 Z2 Y3 Z11]
+ (0.00355529019550411) [X1 Z2 X3 Z11]
+ (0.005143391768824774) [Y3 Y4 X5 X6]
+ (0.005143391768824774) [X3 X4 Y5 Y6]
+ (0.005283776488402955) [Y0 X1 X12 Y13]
+ (0.005283776488402955) [X0 Y1 Y12 X13]
+ (0.005530759218631541) [Y0 Z1 Y2 Z4]
+ (0.005530759218631541) [X0 Z1 X2 Z4]
+ (0.005530759218631541) [Y1 Z2 Y3 Z5]
+ (0.005530759218631541) [X1 Z2 X3 Z5]
+ (0.006087822480561936) [Y8 X9 X12 Y13]
+ (0.006087822480561936) [X8 Y9 Y12 X13]
+ (0.006509361201177238) [Y0 X1 X8 Y9]
+ (0.006509361201177238) [X0 Y1 Y8 X9]
+ (0.006888194352970668) [Y0 X1 X6 Y7]
+ (0.006888194352970668) [X0 Y1 Y6 X7]
+ (0.006901238249797266) [Y0 Z1 Y2 Z12]
+ (0.006901238249797266) [X0 Z1 X2 Z12]
+ (0.006901238249797266) [Y1 Z2 Y3 Z13]
+ (0.006901238249797266) [X1 Z2 X3 Z13]
+ (0.007156934919856985) [Y4 X5 X8 Y9]
+ (0.007156934919856985) [X4 Y5 Y8 X9]
+ (0.007731425250775374) [Y0 X1 X10 Y11]
+ (0.007731425250775374) [X0 Y1 Y10 X11]
+ (0.00803252091882151) [Y0 Z1 Y2 Z6]
+ (0.00803252091882151) [X0 Z1 X2 Z6]
+ (0.00803252091882151) [Y1 Z2 Y3 Z7]
+ (0.00803252091882151) [X1 Z2 X3 Z7]
+ (0.009560705729136197) [Y8 X9 X10 Y11]
+ (0.009560705729136197) [X8 Y9 Y10 X11]
+ (0.011055020596132068) [Y0 Z1 Y2 Z8]
+ (0.011055020596132068) [X0 Z1 X2 Z8]
+ (0.011055020596132068) [Y1 Z2 Y3 Z9]
+ (0.011055020596132068) [X1 Z2 X3 Z9]
+ (0.011285190200840543) [Y5 Y6 X11 X12]
+ (0.011285190200840543) [X5 X6 Y11 Y12]
+ (0.011307274008848074) [Y7 Z8 Z9 Y11]
+ (0.011307274008848074) [X7 Z8 Z9 X11]
+ (0.011982389010247708) [Y4 X5 X6 Y7]
+ (0.011982389010247708) [X4 Y5 Y6 X7]
+ (0.013873381748426379) [Y6 X7 X8 Y9]
+ (0.013873381748426379) [X6 Y7 Y8 X9]
+ (0.014583648907612476) [Y0 X1 X2 Y3]
+ (0.014583648907612476) [X0 Y1 Y2 X3]
+ (0.01557720806397644) [Y2 X3 X12 Y13]
+ (0.01557720806397644) [X2 Y3 Y12 X13]
+ (0.01736611899465119) [Y6 X7 X12 Y13]
+ (0.01736611899465119) [X6 Y7 Y12 X13]
+ (0.017680067952481626) [Y4 X5 X10 Y11]
+ (0.017680067952481626) [X4 Y5 Y10 X11]
+ (0.01782514099578553) [Y6 X7 X10 Y11]
+ (0.01782514099578553) [X6 Y7 Y10 X11]
+ (0.01902824244384799) [Y3 X4 X11 Y12]
+ (0.01902824244384799) [X3 Y4 Y11 X12]
+ (0.025384657508457732) [Y2 X3 X10 Y11]
+ (0.025384657508457732) [X2 Y3 Y10 X11]
+ (0.028685183716106403) [Y10 X11 X12 Y13]
+ (0.028685183716106403) [X10 Y11 Y12 X13]
+ (0.0298124245173448) [Y6 Z7 Z8 Y10]
+ (0.0298124245173448) [X6 Z7 Z8 X10]
+ (0.0298124245173448) [Y7 Z9 Z10 Y11]
+ (0.0298124245173448) [X7 Z9 Z10 X11]
+ (0.03010462314345608) [Y6 Z7 Z9 Y10]
+ (0.03010462314345608) [X6 Z7 Z9 X10]
+ (0.03010462314345608) [Y7 Z8 Z10 Y11]
+ (0.03010462314345608) [X7 Z8 Z10 X11]
+ (0.030787505389143526) [Y6 Z8 Z9 Y10]
+ (0.030787505389143526) [X6 Z8 Z9 X10]
+ (0.031143817988966822) [Y2 X3 X6 Y7]
+ (0.031143817988966822) [X2 Y3 Y6 X7]
+ (0.03583956795335364) [Y2 X3 X4 Y5]
+ (0.03583956795335364) [X2 Y3 Y4 X5]
+ (0.036194123559042314) [Y2 X3 X8 Y9]
+ (0.036194123559042314) [X2 Y3 Y8 X9]
+ (0.038314670294804) [Y4 X5 X12 Y13]
+ (0.038314670294804) [X4 Y5 Y12 X13]
+ (0.10433064780651319) [Z0 Y1 Z2 Y3]
+ (0.10433064780651319) [Z0 X1 Z2 X3]
+ (-0.12133276911042254) [Y3 Z4 Z5 Z6 Y7]
+ (-0.12133276911042254) [X3 Z4 Z5 Z6 X7]
+ (-0.12133276911042253) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133276911042253) [X2 Z3 Z4 Z5 X6]
+ (3.2020768797395206e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.2020768797395206e-06) [X0 Z1 Z2 Z3 X4]
+ (3.2020768797395206e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.2020768797395206e-06) [X1 Z2 Z3 Z4 X5]
+ (0.22848106564918402) [Y6 Z7 Z8 Z9 Y10]
+ (0.22848106564918402) [X6 Z7 Z8 Z9 X10]
+ (0.22848106564918408) [Y7 Z8 Z9 Z10 Y11]
+ (0.22848106564918408) [X7 Z8 Z9 Z10 X11]
+ (-0.032767657823289685) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.032767657823289685) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.032767657823289685) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.032767657823289685) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845272375) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.027115036845272375) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845272375) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.027115036845272375) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.025996177598021673) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.025996177598021673) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.01756120240964599) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.01756120240964599) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.01756120240964599) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.01756120240964599) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.014564531231172854) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.014564531231172854) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.014564531231172854) [X7 Z8 Z9 X10 X12 X13]
+ (-0.014564531231172854) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.012215040997613478) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.012215040997613478) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.012215040997613478) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.012215040997613478) [X4 Z5 X6 X11 Z12 X13]
+ (-0.012215040997613478) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.012215040997613478) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.012215040997613478) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.012215040997613478) [X5 Z6 X7 X10 Z11 X12]
+ (-0.011756013419819297) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.011756013419819297) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.011756013419819297) [X3 Z4 Z5 X6 X8 X9]
+ (-0.011756013419819297) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.008764827575689136) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.008764827575689136) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.008764827575689136) [X2 Z3 Z4 X5 X11 X12]
+ (-0.008764827575689136) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.008764827575689136) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.008764827575689136) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.008764827575689136) [X3 X4 X10 Z11 Z12 X13]
+ (-0.008764827575689136) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.008125251921380987) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.008125251921380987) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.007306759928832761) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.007306759928832761) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.007306759928832761) [X4 X5 X7 Z8 Z9 X10]
+ (-0.007306759928832761) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.005805188989826695) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.005805188989826695) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.005805188989826695) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.005805188989826695) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.00565262097801731) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.00565262097801731) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.00565262097801731) [X0 X1 X3 Z4 Z5 X6]
+ (-0.00565262097801731) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.005143391768824774) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.005143391768824774) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.005143391768824774) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.005143391768824774) [X2 Z3 X4 X5 Z6 X7]
+ (-0.004684903388155236) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.004684903388155236) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668620318776289) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.004668620318776289) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.00457500762663916) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.00457500762663916) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.00442485544944185) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.00442485544944185) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.00415879738184) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.00415879738184) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.00415879738184) [X3 Z4 Z5 X6 X12 X13]
+ (-0.00415879738184) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.003493790359890122) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.003493790359890122) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.003493790359890122) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.003493790359890122) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.002779026799025637) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.002779026799025637) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.002293956611352465) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.002293956611352465) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.0017992194936628824) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.0017992194936628824) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.0017278753941369646) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.0017278753941369646) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.000929850796772937) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.000929850796772937) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.000929850796772937) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.000929850796772937) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.000853385625412585) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.000853385625412585) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.0008145313270956291) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.0008145313270956291) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.0008145313270956291) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.0008145313270956291) [X3 Z4 Z5 Z6 X7 Z11]
+ (-7.735036880599938e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.735036880599938e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.735036880599938e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.735036880599938e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-8.774817865578667e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.774817865578667e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.774817865578667e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.774817865578667e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.518362216486925e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.518362216486925e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.518362216486925e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.518362216486925e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.444344676794922e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.444344676794922e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.444344676794922e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.444344676794922e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.524373849164239e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.524373849164239e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.524373849164239e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.524373849164239e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-6.290028434132811e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.290028434132811e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.290028434132811e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.290028434132811e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.974311713913966e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (-5.974311713913966e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (-5.275883122598385e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.275883122598385e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.643051068956277e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.643051068956277e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.556569218720731e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (-4.556569218720731e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (-4.25322422585887e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.25322422585887e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-3.7696594526380424e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.7696594526380424e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.6945132950781494e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.6945132950781494e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.610297131235198e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.610297131235198e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.610297131235198e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.610297131235198e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.313145500229934e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.313145500229934e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.277483196210354e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.277483196210354e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.277483196210354e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.277483196210354e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.211228348934305e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.211228348934305e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.211228348934305e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.211228348934305e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.1513463114313115e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.1513463114313115e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.0882507117494786e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.0882507117494786e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.1726691016890317e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.1726691016890317e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.4548424489223256e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.4548424489223256e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.3304731887837451e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.3304731887837451e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.2283337823541145e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.2283337823541145e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.0358477599665353e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.0358477599665353e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-7.956895372169788e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.956895372169788e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.733197743335911e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.733197743335911e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.733197743335911e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.733197743335911e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.628614201965279e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.628614201965279e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.55628191530664e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.55628191530664e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.55628191530664e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.55628191530664e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.418291575417878e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.418291575417878e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.418291575417878e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.418291575417878e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.927453083697865e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.927453083697865e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.927453083697865e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.927453083697865e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.627851911805488e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.627851911805488e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.287660625342769e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.287660625342769e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.287660625342769e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.287660625342769e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.287660625342769e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.287660625342769e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.287660625342769e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.287660625342769e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.838052750758578e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (-4.838052750758578e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (-3.5707613288749435e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.5707613288749435e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.3281393502484417e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.3281393502484417e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.08682656539022e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.08682656539022e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.08682656539022e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.08682656539022e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.4473231286941343e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (-2.4473231286941343e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (-2.3713289478172916e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.3713289478172916e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.3713289478172916e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.3713289478172916e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.1990516189333408e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-2.1990516189333408e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-1.933241277107054e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.933241277107054e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.933241277107054e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.933241277107054e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.839420915360672e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.839420915360672e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.839420915360672e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.839420915360672e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.5510539176080348e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (-1.5510539176080348e-07) [X0 Z1 X2 X4 Z5 X6]
+ (-1.5510539176080348e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (-1.5510539176080348e-07) [X1 Z2 X3 X5 Z6 X7]
+ (-1.380778148042567e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (-1.380778148042567e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (-1.380778148042567e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (-1.380778148042567e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (-1.380778148042567e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (-1.380778148042567e-07) [X0 Z1 X2 X5 Z6 X7]
+ (-1.380778148042567e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (-1.380778148042567e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (-1.380778148042567e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (-1.380778148042567e-07) [X1 Z2 X3 X4 Z5 X6]
+ (-1.380778148042567e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (-1.380778148042567e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (-1.2919694863578368e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.2919694863578368e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.1076325600937705e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.1076325600937705e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.1076325600937705e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.1076325600937705e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.1076325600937705e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.1076325600937705e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.1076325600937705e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.1076325600937705e-07) [X1 Z2 X3 X10 Z11 X12]
+ (8.057446596380463e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.057446596380463e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.057446596380463e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.057446596380463e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.649310134797404e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.649310134797404e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.649310134797404e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.649310134797404e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.839420915360672e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.839420915360672e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.839420915360672e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.839420915360672e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.1990516189333408e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (2.1990516189333408e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (2.4473231286941343e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (2.4473231286941343e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.236259961297032e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.236259961297032e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.236259961297032e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.236259961297032e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.3281393502484417e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.3281393502484417e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.5707613288749435e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.5707613288749435e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.838052750758578e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (4.838052750758578e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (5.627851911805488e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.627851911805488e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (6.628614201965279e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.628614201965279e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (7.956895372169788e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.956895372169788e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.306536652490508e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.306536652490508e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.306536652490508e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.306536652490508e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.0358477599665353e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.0358477599665353e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.2283337823541145e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.2283337823541145e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.2393363217880728e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.2393363217880728e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.2393363217880728e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.2393363217880728e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.3304731887837451e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.3304731887837451e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.4548424489223256e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.4548424489223256e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.1726691016890317e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.1726691016890317e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.0882507117494786e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.0882507117494786e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.1174479459651303e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.1174479459651303e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.1513463114313115e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.1513463114313115e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.313145500229934e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.313145500229934e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.334331289601457e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.334331289601457e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.6945132950781494e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.6945132950781494e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (4.183932559518808e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.183932559518808e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.556569218720731e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (4.556569218720731e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (4.643051068956277e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.643051068956277e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.275883122598385e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.275883122598385e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.974311713913966e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (5.974311713913966e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (0.00029219862611128274) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.00029219862611128274) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.00029219862611128274) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.00029219862611128274) [X6 Z7 X8 X9 Z10 X11]
+ (0.0004956762314924203) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.0004956762314924203) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.000665007021949879) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.000665007021949879) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.000665007021949879) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.000665007021949879) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.000853385625412585) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.000853385625412585) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.0016095313817213928) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.0016095313817213928) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.0016095313817213928) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.0016095313817213928) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.001667604181144148) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.001667604181144148) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.001667604181144148) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.001667604181144148) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.0017278753941369646) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.0017278753941369646) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.0017992194936628824) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.0017992194936628824) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.002293956611352465) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.002293956611352465) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.0024629170071339777) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.0024629170071339777) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.0024629170071339777) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.0024629170071339777) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.003961560792496613) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.003961560792496613) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.003961560792496613) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.003961560792496613) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.00442485544944185) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.00442485544944185) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.00457500762663916) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.00457500762663916) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.004668620318776289) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.004668620318776289) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.004684903388155236) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.004684903388155236) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.005324835234221465) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.005324835234221465) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.005324835234221465) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.005324835234221465) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.005368659358109184) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.005368659358109184) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.005368659358109184) [X2 X3 X7 Z8 Z9 X10]
+ (0.005368659358109184) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.007960880725921346) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.007960880725921346) [X4 Z5 X6 X10 Z11 X12]
+ (0.007960880725921346) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.007960880725921346) [X5 Z6 X7 X11 Z12 X13]
+ (0.008125251921380987) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.008125251921380987) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.008890731522694281) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.008890731522694281) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.008890731522694281) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.008890731522694281) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.010263414868158856) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.010263414868158856) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.010263414868158856) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.010263414868158856) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.01054042590767108) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.01054042590767108) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.01054042590767108) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.01054042590767108) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.010960074940542294) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.010960074940542294) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.010960074940542294) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.010960074940542294) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.011307274008848076) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.011307274008848076) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.014411099430131295) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.014411099430131295) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.014411099430131295) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.014411099430131295) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.015225630757226924) [Y3 Z4 Z5 X6 X10 Y11]
+ (0.015225630757226924) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (0.015225630757226924) [X3 Z4 Z5 X6 X10 X11]
+ (0.015225630757226924) [X3 Z4 Z5 Y6 Y10 X11]
+ (0.01558825010238032) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (0.01558825010238032) [X2 Z3 X4 X10 Z11 X12]
+ (0.01558825010238032) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (0.01558825010238032) [X3 Z4 X5 X11 Z12 X13]
+ (0.018266834869375054) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.018266834869375054) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.018266834869375054) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.018266834869375054) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.019020423173039647) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.019020423173039647) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.019020423173039647) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.019020423173039647) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.020175921723534825) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.020175921723534825) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.020175921723534825) [X4 Z5 Z6 X7 X11 X12]
+ (0.020175921723534825) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.020175921723534825) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.020175921723534825) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.020175921723534825) [X5 X6 X10 Z11 Z12 X13]
+ (0.020175921723534825) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.024353077678069456) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.024353077678069456) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.024353077678069456) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.024353077678069456) [X2 Z3 X4 X11 Z12 X13]
+ (0.024353077678069456) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.024353077678069456) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.024353077678069456) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.024353077678069456) [X3 Z4 X5 X10 Z11 X12]
+ (0.02438908253114883) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.02438908253114883) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.02438908253114883) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.02438908253114883) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.02510495713884394) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.02510495713884394) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.02510495713884394) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.02510495713884394) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.030787505389143526) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.030787505389143526) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.04587947078129819) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.04587947078129819) [X0 Z2 Z3 Z4 Z5 X6]
+ (0.05600733087780638) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.05600733087780638) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.05600733087780638) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.05600733087780638) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.056084681246612374) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.056084681246612374) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.056084681246612374) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.056084681246612374) [Z1 X7 Z8 Z9 Z10 X11]
+ (-6.631277929188417e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.631277929188417e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.631277929188417e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.631277929188417e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.595086007500835e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.595086007500835e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.5950860075008345e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.5950860075008345e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.042743277013784) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.042743277013784) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (0.042743277013784) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.042743277013784) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.04764261217638315) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.04764261217638315) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.04764261217638315) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.04764261217638315) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.04171881383982176) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.04171881383982176) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.04171881383982176) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.04171881383982176) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.039564416322894265) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.039564416322894265) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.039564416322894265) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.039564416322894265) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.039359168022053984) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.039359168022053984) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.039359168022053984) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.039359168022053984) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.03931805194719789) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03931805194719789) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.03931805194719789) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03931805194719789) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03560837898831276) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.03560837898831276) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.029903789512625296) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.029903789512625296) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.029903789512625296) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.029903789512625296) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.028730779551905783) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.028730779551905783) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.028730779551905783) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.028730779551905783) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.025637238296026783) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.025637238296026783) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.025637238296026783) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.025637238296026783) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.024755463292891557) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.024755463292891557) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.024755463292891557) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.024755463292891557) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.024282117354693034) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.024282117354693034) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.023145130929529165) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.023145130929529165) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.02252844019601312) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.02252844019601312) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.021433810721602026) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.021433810721602026) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.021433810721602026) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.021433810721602026) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251502) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.019257505095251502) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.01902824244384799) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.01902824244384799) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.01888903030494318) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-0.01888903030494318) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-0.01888903030494318) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.01888903030494318) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.016024603689179594) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.016024603689179594) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.015225630757226924) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.015225630757226924) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.01460370472916243) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.01460370472916243) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.014564531231172854) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.014564531231172854) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.011756013419819297) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.011756013419819297) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.011285190200840543) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-0.011285190200840543) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-0.009841749246962602) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.009841749246962602) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.009612634606847189) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.009612634606847189) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.009612634606847189) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.009612634606847189) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.008469978791023269) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.008469978791023269) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.00730675992883276) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.00730675992883276) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005923798336561392) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.005923798336561392) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.00565262097801731) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.00565262097801731) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.005368659358109183) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005368659358109183) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.00415879738184) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.00415879738184) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0033566705638329196) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.0033566705638329196) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.0033566705638329196) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.0033566705638329196) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.0032675138544235585) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.0032675138544235585) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.0032675138544235585) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.0032675138544235585) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.002779026799025637) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.002779026799025637) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.0026860409778066653) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.0026860409778066653) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.0026860409778066653) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.0026860409778066653) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.002293956611352465) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.002293956611352465) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.002293956611352465) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.002293956611352465) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.0009581655836697007) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.0009581655836697007) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.0009581655836697007) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.0009581655836697007) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.0009581655836697007) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.0009581655836697007) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.0009581655836697007) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.0009581655836697007) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.0002463643756963746) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0002463643756963746) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303540898) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.00013840177303540898) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.00013840177303540898) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.00013840177303540898) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.735036880599938e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.735036880599938e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585307563865e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585307563865e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.6103585307563865e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.6103585307563865e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.531680879707829e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.531680879707829e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.531680879707829e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.531680879707829e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.80610277609862e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.80610277609862e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.80610277609862e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.80610277609862e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.08979946833165e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.08979946833165e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.08979946833165e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.08979946833165e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.652209670747312e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.652209670747312e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.652209670747312e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.652209670747312e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.48185183521211e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.48185183521211e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.48185183521211e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.48185183521211e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.0714807368263705e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.0714807368263705e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.0714807368263705e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.0714807368263705e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.7346220392722495e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.7346220392722495e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.7346220392722495e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.7346220392722495e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.72884314777416e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.72884314777416e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.72884314777416e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.72884314777416e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.25322422585887e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.25322422585887e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.7696594526380424e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.7696594526380424e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.5443954296829946e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.5443954296829946e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.5443954296829946e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.5443954296829946e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.5443954296829946e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.5443954296829946e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.5443954296829946e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.5443954296829946e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.3609563205574907e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563205574907e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563205574907e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.3609563205574907e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-2.1032156050627645e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.1032156050627645e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.1032156050627645e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.1032156050627645e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220986687457e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.0111220986687457e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220986687457e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.0111220986687457e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9429468370536426e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.9429468370536426e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9429468370536426e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9429468370536426e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.65411747760991e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.65411747760991e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.65411747760991e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.65411747760991e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.5224930677279263e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.5224930677279263e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.5224930677279263e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.5224930677279263e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.5224930677279263e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930677279263e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930677279263e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.5224930677279263e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.2283337823541145e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337823541145e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.2283337823541145e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337823541145e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.988770288009164e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.988770288009164e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.988770288009164e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.988770288009164e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.867765104855758e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104855758e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104855758e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.867765104855758e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.189990976158817e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.189990976158817e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.175246207417844e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175246207417844e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.471647745194209e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.471647745194209e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.5614471793351593e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.5614471793351593e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.5614471793351593e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.5614471793351593e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.52338967858187e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.52338967858187e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.427323108674005e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.427323108674005e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.427323108674005e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.427323108674005e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.3281393502484417e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.3281393502484417e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.3281393502484417e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.3281393502484417e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.08682656539022e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.08682656539022e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.888293594437326e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.888293594437326e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.888293594437326e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.888293594437326e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.3713289478172916e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.3713289478172916e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.839420915360672e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.839420915360672e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-8.057446596380463e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.057446596380463e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (4.53717809627361e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.53717809627361e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.53717809627361e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (4.53717809627361e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (8.057446596380463e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.057446596380463e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (9.209350639401885e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350639401885e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350639401885e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.209350639401885e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783553520087e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783553520087e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783553520087e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783553520087e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.839420915360672e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.839420915360672e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.3713289478172916e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.3713289478172916e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (3.08682656539022e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.08682656539022e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (4.52338967858187e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.52338967858187e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (5.471647745194209e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.471647745194209e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.175246207417844e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175246207417844e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.189990976158817e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.189990976158817e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.3304731887837451e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.3304731887837451e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.3304731887837451e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.3304731887837451e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.6288532437033851e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.6288532437033851e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.6288532437033851e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.6288532437033851e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.6893489516527603e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.6893489516527603e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.6893489516527603e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.6893489516527603e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (2.745518400882078e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.745518400882078e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.745518400882078e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.745518400882078e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.745518400882078e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.745518400882078e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.745518400882078e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.745518400882078e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.2118420193806866e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420193806866e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.2118420193806866e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420193806866e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.2118420193806866e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420193806866e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.2118420193806866e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420193806866e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.313145500229934e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.313145500229934e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.313145500229934e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.313145500229934e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (3.334331289601457e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.334331289601457e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (4.183932559518808e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.183932559518808e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (7.735036880599938e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.735036880599938e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.0002463643756963746) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.0002463643756963746) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00044585351288404) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.00044585351288404) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.00044585351288404) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.00044585351288404) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.0005940221543005165) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005165) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005165) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005165) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005165) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005165) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.0005940221543005165) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005165) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.0008533856254125851) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.0008533856254125851) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.0008533856254125851) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.0008533856254125851) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.00104352465349077) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.00104352465349077) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.00104352465349077) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.00104352465349077) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.0012803060973496884) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.0012803060973496884) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.0012803060973496884) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.0012803060973496884) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.0013038004788126995) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (0.0013038004788126995) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (0.0013038004788126995) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (0.0013038004788126995) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (0.0022619660624824006) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (0.0022619660624824006) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (0.0022619660624824006) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (0.0022619660624824006) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (0.0022619660624824006) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (0.0022619660624824006) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (0.0022619660624824006) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (0.0022619660624824006) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (0.003989841456619366) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (0.003989841456619366) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (0.003989841456619366) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (0.003989841456619366) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (0.00415879738184) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.00415879738184) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.004311038507914328) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.004311038507914328) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.004311038507914328) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.004311038507914328) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.004636976661182608) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.004636976661182608) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.004636976661182608) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.004636976661182608) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.005114473831660329) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.005114473831660329) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.005114473831660329) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.005114473831660329) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.005114473831660329) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660329) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660329) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005114473831660329) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.005241535382803964) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (0.005241535382803964) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (0.005241535382803964) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (0.005241535382803964) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (0.0052626424730768066) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.0052626424730768066) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.0052626424730768066) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.0052626424730768066) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.005368659358109183) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005368659358109183) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005379937155839372) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.005379937155839372) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.005379937155839372) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.005379937155839372) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.00565262097801731) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.00565262097801731) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.005708495985960845) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.005708495985960845) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.005708495985960845) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.005708495985960845) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.005923798336561392) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.005923798336561392) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.00730675992883276) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.00730675992883276) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.008469978791023269) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.008469978791023269) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.009841749246962602) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.009841749246962602) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.011285190200840543) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (0.011285190200840543) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (0.011756013419819297) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.011756013419819297) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.014564531231172854) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.014564531231172854) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.01460370472916243) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.01460370472916243) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.015225630757226924) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.015225630757226924) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.016024603689179594) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.016024603689179594) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.01902824244384799) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.01902824244384799) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.019257505095251502) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.019257505095251502) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.04587947078129819) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04587947078129819) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.3693708936615634) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.3693708936615634) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.3693708936615634) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.3693708936615634) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.28164257767023504) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.28164257767023504) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.2816425776702349) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.2816425776702349) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.09065144207036487) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.09065144207036487) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.09065144207036487) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.09065144207036487) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863634) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.08684737589863634) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863634) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.08684737589863634) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.07635021950635111) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07635021950635111) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07635021950635111) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07635021950635111) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214116) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.06752385099214116) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214116) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.06752385099214116) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03560837898831276) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.03560837898831276) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03490334337366141) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03490334337366141) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03490334337366141) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03490334337366141) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883830016) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.024591860883830016) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883830016) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.024591860883830016) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.024282117354693038) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.024282117354693038) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.023145130929529162) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.023145130929529162) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.02252844019601312) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.02252844019601312) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.01953805031131525) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.01953805031131525) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.01953805031131525) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.01953805031131525) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.017091553155899275) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.017091553155899275) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.017091553155899275) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.017091553155899275) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.016024603689179594) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.016024603689179594) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.016024603689179594) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.016024603689179594) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.010311482489831396) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831396) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831396) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010311482489831396) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.009841749246962602) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962602) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.009841749246962602) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962602) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209948) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209948) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209948) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008826368514209948) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.00854199662545489) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00854199662545489) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00854199662545489) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00854199662545489) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.00854199662545489) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00854199662545489) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00854199662545489) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00854199662545489) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008469978791023269) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023269) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.008469978791023269) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023269) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.004668620318776289) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.004668620318776289) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.0038764708993370353) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.0038764708993370353) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.003804066171728543) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728543) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728543) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003804066171728543) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178557) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178557) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0033566705638329196) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.0033566705638329196) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.0032675138544235585) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.0032675138544235585) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.002141361223101557) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.002141361223101557) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.0017278753941369646) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.0017278753941369646) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.0016407548553124215) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0016407548553124215) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214169068) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0014528843214169068) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214169068) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0014528843214169068) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870896771024844) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007870896771024844) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0005192743499488553) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.0005192743499488553) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.00019400857029756342) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00019400857029756342) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303540898) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.00013840177303540898) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.141625221152606e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.141625221152606e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.141625221152606e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.141625221152606e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-5.0714807368263705e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.0714807368263705e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.1513463114313115e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.1513463114313115e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0882507117494786e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.0882507117494786e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.9885117062166915e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.9885117062166915e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8742990715923605e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.8742990715923605e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.3609563205574907e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.3609563205574907e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.3002946562573788e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.3002946562573788e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.1468376508985958e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.1468376508985958e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.1468376508985958e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.1468376508985958e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.352332104368793e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.352332104368793e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.352332104368793e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.352332104368793e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.091637200169596e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637200169596e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637200169596e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637200169596e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637200169596e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637200169596e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637200169596e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.091637200169596e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.07430598704487e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.07430598704487e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.07430598704487e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.07430598704487e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.900128987398232e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.900128987398232e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.900128987398232e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900128987398232e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867765104855756e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.867765104855756e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.560692465927374e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692465927374e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692465927374e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692465927374e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692465927374e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692465927374e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692465927374e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.560692465927374e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.997018422510811e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.997018422510811e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.997018422510811e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.997018422510811e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.997018422510811e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.997018422510811e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.997018422510811e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.997018422510811e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.5682475215877264e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.5682475215877264e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.5682475215877264e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.5682475215877264e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.376739308816362e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.376739308816362e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.376739308816362e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.376739308816362e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.376739308816362e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.376739308816362e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.376739308816362e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.376739308816362e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.888293594437326e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.888293594437326e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6863815487957127e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.6863815487957127e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035783553520087e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7035783553520087e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.209350639401885e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.209350639401885e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773246637356e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773246637356e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773246637356e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773246637356e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773246637356e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773246637356e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.379773246637356e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773246637356e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253802331482e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253802331482e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9742253802331482e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9742253802331482e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0474716558397492e-08) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.0474716558397492e-08) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.0474716558397492e-08) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.0474716558397492e-08) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.209350639401885e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.209350639401885e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0717282186784369e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.0717282186784369e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.0717282186784369e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.0717282186784369e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.2004287494811956e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.2004287494811956e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.2004287494811956e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.2004287494811956e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.7035783553520087e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7035783553520087e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.3120943054927134e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.3120943054927134e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.3120943054927134e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.3120943054927134e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.6863815487957127e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815487957127e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.888293594437326e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.888293594437326e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.092250616513476e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.092250616513476e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.092250616513476e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.092250616513476e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.092250616513476e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.092250616513476e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.092250616513476e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.092250616513476e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.444597854270713e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.444597854270713e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.444597854270713e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.444597854270713e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.6849150955853633e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.6849150955853633e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.6849150955853633e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.6849150955853633e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.246974426096281e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.246974426096281e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.246974426096281e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.246974426096281e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.246974426096281e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.246974426096281e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.246974426096281e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.246974426096281e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.867765104855756e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.867765104855756e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.3002946562573788e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.3002946562573788e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.3609563205574907e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.3609563205574907e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.8742990715923605e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.8742990715923605e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.8836765762715356e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.8836765762715356e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.9473560121016728e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9473560121016728e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.9473560121016728e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9473560121016728e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117062166915e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9885117062166915e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.0882507117494786e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.0882507117494786e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.1513463114313115e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.1513463114313115e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.846201671667271e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.846201671667271e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.846201671667271e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.846201671667271e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.0714807368263705e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.0714807368263705e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.105526722447096e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.105526722447096e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.105526722447096e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.105526722447096e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.14649632792465e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.14649632792465e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.14649632792465e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.14649632792465e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.159350502220153e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.159350502220153e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.159350502220153e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.159350502220153e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.427988657099724e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.427988657099724e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.427988657099724e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.427988657099724e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.935867718318364e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.935867718318364e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.935867718318364e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.935867718318364e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.25327334873453e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.25327334873453e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.979825794039457e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.979825794039457e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.979825794039457e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.979825794039457e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (4.205548411212803e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.205548411212803e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.205548411212803e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.205548411212803e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00013840177303540898) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.00013840177303540898) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.0001878705338955148) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0001878705338955148) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0001878705338955148) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0001878705338955148) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00019400857029756342) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00019400857029756342) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0002463643756963746) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0002463643756963746) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0002463643756963746) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0002463643756963746) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192743499488553) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.0005192743499488553) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.0007156734248909582) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007156734248909582) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0007156734248909582) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007156734248909582) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024844) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007870896771024844) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.001532483523073067) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.001532483523073067) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.001532483523073067) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.001532483523073067) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.0016407548553124215) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0016407548553124215) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0017278753941369646) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.0017278753941369646) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.002446497155415974) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.002446497155415974) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.002446497155415974) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.002446497155415974) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.0032675138544235585) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.0032675138544235585) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.0033566705638329196) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.0033566705638329196) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.0034841573002178557) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0034841573002178557) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0038764708993370353) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.0038764708993370353) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.004668620318776289) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.004668620318776289) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278195) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.004767272188278195) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.004767272188278195) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278195) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.00528654653822705) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.00528654653822705) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.00528654653822705) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.00528654653822705) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.005408954422410102) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.005408954422410102) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.005408954422410102) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.005408954422410102) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.005923798336561392) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561392) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.005923798336561392) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561392) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.01071550846979661) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01071550846979661) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.01071550846979661) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01071550846979661) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.010757563953908736) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010757563953908736) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010757563953908736) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010757563953908736) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01460370472916243) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.01460370472916243) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.01460370472916243) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.01460370472916243) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.01929956057936363) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01929956057936363) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01929956057936363) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01929956057936363) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01929956057936363) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01929956057936363) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.01929956057936363) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01929956057936363) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.058591988733862885) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.058591988733862885) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.775950527728342e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.775950527728342e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.775950527728345e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.775950527728345e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.07165035181002799) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.07165035181002799) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.07165035181002805) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.07165035181002805) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.019257505095251502) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019257505095251502) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010311482489831396) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311482489831396) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008826368514209948) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209948) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.007597464029770613) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.007597464029770613) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.007597464029770613) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.007597464029770613) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311859) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311859) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311859) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311859) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311859) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311859) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311859) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311859) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00534805158267661) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.00534805158267661) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00534805158267661) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00534805158267661) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0038040661717285433) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0038040661717285433) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002984166168122004) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.002984166168122004) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.002984166168122004) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.002984166168122004) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.0024464971554159737) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.0024464971554159737) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.0022494124470940043) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0022494124470940043) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0022494124470940043) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0022494124470940043) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.002141361223101557) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.002141361223101557) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.001863894282458754) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.001863894282458754) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.001863894282458754) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.001863894282458754) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.001863894282458754) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.001863894282458754) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.001863894282458754) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.001863894282458754) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0016407548553124215) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124215) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0016407548553124215) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124215) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0012223378081538795) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538795) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538795) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538795) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538795) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538795) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538795) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0012223378081538795) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.001028329237856316) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.001028329237856316) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.001028329237856316) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.001028329237856316) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1463061454101287e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1463061454101287e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.8742990715923605e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990715923605e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.8742990715923605e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990715923605e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.3002946562573788e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.3002946562573788e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.3002946562573788e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.3002946562573788e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.0444941299256803e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.0444941299256803e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.0444941299256803e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.0444941299256803e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.956079231141312e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.956079231141312e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.956079231141312e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.956079231141312e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105515038371399e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105515038371399e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105515038371399e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105515038371399e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661347214048752e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661347214048752e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661347214048752e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661347214048752e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540341414201439e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540341414201439e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.189990976158817e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.189990976158817e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.876621659138724e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876621659138724e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876621659138724e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876621659138724e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175246207417844e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175246207417844e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.52338967858187e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.52338967858187e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0767325325559293e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.0767325325559293e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0767325325559293e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.0767325325559293e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.013471459102309e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.013471459102309e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.9045998850553645e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.9045998850553645e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.9045998850553645e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.9045998850553645e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.6667317550577705e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.6667317550577705e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.6667317550577705e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6667317550577705e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641927699134e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.8505641927699134e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309314426093e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.6569309314426093e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309314426093e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.6569309314426093e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.8505641927699134e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.8505641927699134e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815487957127e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815487957127e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.6863815487957127e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815487957127e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471459102309e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.013471459102309e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.52338967858187e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.52338967858187e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.6704023905449185e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.6704023905449185e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.6704023905449185e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.6704023905449185e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175246207417844e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175246207417844e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.189990976158817e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.189990976158817e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540341414201439e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540341414201439e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949476487569194e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949476487569194e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.7924939578577481e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939578577481e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939578577481e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.7924939578577481e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.8836765762715356e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.8836765762715356e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.9885117062166915e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117062166915e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117062166915e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117062166915e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.25327334873453e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.25327334873453e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4017109736392057e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4017109736392057e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4017109736392057e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4017109736392057e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603694249806e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.5809603694249806e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603694249806e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.5809603694249806e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0005192743499488553) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499488553) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.0005192743499488553) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499488553) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.0007870896771024844) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024844) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024844) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024844) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00117263483164415) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.00117263483164415) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.00117263483164415) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00117263483164415) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.001236647801924552) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.001236647801924552) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.001236647801924552) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.001236647801924552) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.0022009640695004663) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0022009640695004663) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0022009640695004663) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0022009640695004663) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798029) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798029) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798029) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798029) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798029) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798029) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002394972639798029) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798029) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0024464971554159737) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.0024464971554159737) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.0038040661717285433) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0038040661717285433) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0038764708993370353) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.0038764708993370353) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.0038764708993370353) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.0038764708993370353) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.004220813970046556) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.004220813970046556) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.004220813970046556) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.004220813970046556) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.008826368514209948) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.008826368514209948) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.010311482489831396) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010311482489831396) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019257505095251502) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019257505095251502) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.058591988733862885) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.058591988733862885) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.3987009017900383e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.3987009017900383e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.3987009017900383e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.3987009017900383e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841573002178557) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841573002178557) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002984166168122004) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.002984166168122004) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.00019400857029756342) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00019400857029756342) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061454101287e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061454101287e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.7924939578577481e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7924939578577481e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540341414201439e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341414201439e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540341414201439e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341414201439e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.8505641927699134e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641927699134e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641927699134e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641927699134e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471459102309e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471459102309e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471459102309e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471459102309e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.949476487569194e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.949476487569194e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.7924939578577481e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7924939578577481e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029756342) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029756342) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002984166168122004) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.002984166168122004) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.0034841573002178557) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0034841573002178557) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
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
  (-46.463906913420516) [I0]
+ (0.7829652070487763) [Z11]
+ (0.7829652070487764) [Z10]
+ (0.8084591005185322) [Z12]
+ (0.8084591005185322) [Z13]
+ (1.2034393391308191) [Z4]
+ (1.20343933913082) [Z5]
+ (1.309687661862785) [Z7]
+ (1.3096876618627855) [Z6]
+ (1.3693525711695096) [Z8]
+ (1.36935257116951) [Z9]
+ (1.653893830546219) [Z3]
+ (1.6538938305462192) [Z2]
+ (12.41263071444327) [Z0]
+ (12.41263071444327) [Z1]
+ (-7.954224644512314e-06) [Y11 Y13]
+ (-7.954224644512314e-06) [X11 X13]
+ (-7.765082298255956e-07) [Y3 Y5]
+ (-7.765082298255956e-07) [X3 X5]
+ (5.929280463290183e-07) [Y4 Y6]
+ (5.929280463290183e-07) [X4 X6]
+ (1.6021751061174477e-06) [Y2 Y4]
+ (1.6021751061174477e-06) [X2 X4]
+ (1.8540565275789768e-06) [Y5 Y7]
+ (1.8540565275789768e-06) [X5 X7]
+ (8.194104981214256e-06) [Y10 Y12]
+ (8.194104981214256e-06) [X10 X12]
+ (0.0032769650657712246) [Y1 Y3]
+ (0.0032769650657712246) [X1 X3]
+ (0.10433061485317607) [Y0 Y2]
+ (0.10433061485317607) [X0 X2]
+ (0.1127038185912678) [Z10 Z12]
+ (0.1127038185912678) [Z11 Z13]
+ (0.11383573685303822) [Z4 Z12]
+ (0.11383573685303822) [Z5 Z13]
+ (0.11952441016898577) [Z6 Z10]
+ (0.11952441016898577) [Z7 Z11]
+ (0.12489977362993819) [Z4 Z10]
+ (0.12489977362993819) [Z5 Z11]
+ (0.12495799328191462) [Z2 Z4]
+ (0.12495799328191462) [Z3 Z5]
+ (0.12799492801397844) [Z2 Z10]
+ (0.12799492801397844) [Z3 Z11]
+ (0.13401737372659003) [Z6 Z12]
+ (0.13401737372659003) [Z7 Z13]
+ (0.13701191913060518) [Z4 Z6]
+ (0.13701191913060518) [Z5 Z7]
+ (0.1373494221016206) [Z6 Z11]
+ (0.1373494221016206) [Z7 Z10]
+ (0.13739112375011658) [Z2 Z6]
+ (0.13739112375011658) [Z3 Z7]
+ (0.13766859133619272) [Z8 Z10]
+ (0.13766859133619272) [Z9 Z11]
+ (0.14011294749713782) [Z2 Z12]
+ (0.14011294749713782) [Z3 Z13]
+ (0.14138903590154728) [Z10 Z13]
+ (0.14138903590154728) [Z11 Z12]
+ (0.14257991128706593) [Z4 Z11]
+ (0.14257991128706593) [Z5 Z10]
+ (0.14722930783264196) [Z8 Z11]
+ (0.14722930783264196) [Z9 Z10]
+ (0.14899426171386093) [Z4 Z7]
+ (0.14899426171386093) [Z5 Z6]
+ (0.1492634706004885) [Z10 Z11]
+ (0.14960692557245064) [Z4 Z8]
+ (0.14960692557245064) [Z5 Z9]
+ (0.14973497005434805) [Z8 Z12]
+ (0.14973497005434805) [Z9 Z13]
+ (0.1507140548228603) [Z2 Z8]
+ (0.1507140548228603) [Z3 Z9]
+ (0.15138342699121976) [Z6 Z13]
+ (0.15138342699121976) [Z7 Z12]
+ (0.15215040622649662) [Z4 Z13]
+ (0.15215040622649662) [Z5 Z12]
+ (0.15337959171064447) [Z2 Z11]
+ (0.15337959171064447) [Z3 Z10]
+ (0.15435760065310425) [Z12 Z13]
+ (0.15569017457259954) [Z2 Z13]
+ (0.15569017457259954) [Z3 Z12]
+ (0.15582280685858307) [Z8 Z13]
+ (0.15582280685858307) [Z9 Z12]
+ (0.15676384610163535) [Z4 Z9]
+ (0.15676384610163535) [Z5 Z8]
+ (0.1575530380435236) [Z4 Z5]
+ (0.16079755046695032) [Z2 Z5]
+ (0.16079755046695032) [Z3 Z4]
+ (0.1675666935617093) [Z6 Z8]
+ (0.1675666935617093) [Z7 Z9]
+ (0.16853492794646652) [Z2 Z7]
+ (0.16853492794646652) [Z3 Z6]
+ (0.1814400936293424) [Z6 Z9]
+ (0.1814400936293424) [Z7 Z8]
+ (0.18189081243733157) [Z2 Z3]
+ (0.1869081483103383) [Z2 Z9]
+ (0.1869081483103383) [Z3 Z8]
+ (0.1929970026987758) [Z0 Z10]
+ (0.1929970026987758) [Z1 Z11]
+ (0.19392574334985946) [Z6 Z7]
+ (0.1966174995973278) [Z0 Z4]
+ (0.1966174995973278) [Z1 Z5]
+ (0.19936332691274994) [Z0 Z5]
+ (0.19936332691274994) [Z1 Z4]
+ (0.20072843554614012) [Z0 Z11]
+ (0.20072843554614012) [Z1 Z10]
+ (0.21102681234299953) [Z0 Z12]
+ (0.21102681234299953) [Z1 Z13]
+ (0.21631059809683018) [Z0 Z13]
+ (0.21631059809683018) [Z1 Z12]
+ (0.22003977240267739) [Z8 Z9]
+ (0.23671071740423064) [Z0 Z2]
+ (0.23671071740423064) [Z1 Z3]
+ (0.24164696831759996) [Z0 Z6]
+ (0.24164696831759996) [Z1 Z7]
+ (0.24853517285992294) [Z0 Z7]
+ (0.24853517285992294) [Z1 Z6]
+ (0.25129435573124165) [Z0 Z3]
+ (0.25129435573124165) [Z1 Z2]
+ (0.2723251845038109) [Z0 Z8]
+ (0.2723251845038109) [Z1 Z9]
+ (0.27883454573788796) [Z0 Z9]
+ (0.27883454573788796) [Z1 Z8]
+ (1.186176448414148) [Z0 Z1]
+ (-3.8866395392500735e-06) [Y2 Z3 Y4]
+ (-3.8866395392500735e-06) [X2 Z3 X4]
+ (-3.8866395392500735e-06) [Y3 Z4 Y5]
+ (-3.8866395392500735e-06) [X3 Z4 X5]
+ (1.072274843365805e-05) [Y10 Z11 Y12]
+ (1.072274843365805e-05) [X10 Z11 X12]
+ (1.072274843365805e-05) [Y11 Z12 Y13]
+ (1.072274843365805e-05) [X11 Z12 X13]
+ (1.2260276840805483e-05) [Y4 Z5 Y6]
+ (1.2260276840805483e-05) [X4 Z5 X6]
+ (1.2260276840805483e-05) [Y5 Z6 Y7]
+ (1.2260276840805483e-05) [X5 Z6 X7]
+ (0.12507036883989628) [Y1 Z2 Y3]
+ (0.12507036883989628) [X1 Z2 X3]
+ (0.1250703688398963) [Y0 Z1 Y2]
+ (0.1250703688398963) [X0 Z1 X2]
+ (-0.038314669373458385) [Y4 Y5 X12 X13]
+ (-0.038314669373458385) [X4 X5 Y12 Y13]
+ (-0.03619409348747799) [Y2 Y3 X8 X9]
+ (-0.03619409348747799) [X2 X3 Y8 Y9]
+ (-0.03583955718503571) [Y2 Y3 X4 X5]
+ (-0.03583955718503571) [X2 X3 Y4 Y5]
+ (-0.031143804196349937) [Y2 Y3 X6 X7]
+ (-0.031143804196349937) [X2 X3 Y6 Y7]
+ (-0.02868521731027948) [Y10 Y11 X12 X13]
+ (-0.02868521731027948) [X10 X11 Y12 Y13]
+ (-0.025384663696666032) [Y2 Y3 X10 X11]
+ (-0.025384663696666032) [X2 X3 Y10 Y11]
+ (-0.01902831871829496) [Y3 X4 X11 Y12]
+ (-0.01902831871829496) [X3 Y4 Y11 X12]
+ (-0.017825011932634815) [Y6 Y7 X10 X11]
+ (-0.017825011932634815) [X6 X7 Y10 Y11]
+ (-0.01768013765712775) [Y4 Y5 X10 X11]
+ (-0.01768013765712775) [X4 X5 Y10 Y11]
+ (-0.017366053264629745) [Y6 Y7 X12 X13]
+ (-0.017366053264629745) [X6 X7 Y12 Y13]
+ (-0.015577227075461728) [Y2 Y3 X12 X13]
+ (-0.015577227075461728) [X2 X3 Y12 Y13]
+ (-0.014583638327011018) [Y0 Y1 X2 X3]
+ (-0.014583638327011018) [X0 X1 Y2 Y3]
+ (-0.013873400067633128) [Y6 Y7 X8 X9]
+ (-0.013873400067633128) [X6 X7 Y8 Y9]
+ (-0.011982342583255765) [Y4 Y5 X6 X7]
+ (-0.011982342583255765) [X4 X5 Y6 Y7]
+ (-0.01128514461831756) [Y5 X6 X11 Y12]
+ (-0.01128514461831756) [X5 Y6 Y11 X12]
+ (-0.009560716496449257) [Y8 Y9 X10 X11]
+ (-0.009560716496449257) [X8 X9 Y10 Y11]
+ (-0.008125248410116595) [Y1 X2 X8 Y9]
+ (-0.008125248410116595) [Y1 Y2 Y8 Y9]
+ (-0.008125248410116595) [X1 X2 X8 X9]
+ (-0.008125248410116595) [X1 Y2 Y8 X9]
+ (-0.007731432847364331) [Y0 Y1 X10 X11]
+ (-0.007731432847364331) [X0 X1 Y10 Y11]
+ (-0.007156920529184707) [Y4 Y5 X8 X9]
+ (-0.007156920529184707) [X4 X5 Y8 Y9]
+ (-0.00688820454232299) [Y0 Y1 X6 X7]
+ (-0.00688820454232299) [X0 X1 Y6 Y7]
+ (-0.00650936123407709) [Y0 Y1 X8 X9]
+ (-0.00650936123407709) [X0 X1 Y8 Y9]
+ (-0.006087836804235032) [Y8 Y9 X12 X13]
+ (-0.006087836804235032) [X8 X9 Y12 Y13]
+ (-0.005283785753830626) [Y0 Y1 X12 X13]
+ (-0.005283785753830626) [X0 X1 Y12 Y13]
+ (-0.005143382387698603) [Y3 Y4 X5 X6]
+ (-0.005143382387698603) [X3 X4 Y5 Y6]
+ (-0.004684920226866116) [Y1 X2 X6 Y7]
+ (-0.004684920226866116) [Y1 Y2 Y6 Y7]
+ (-0.004684920226866116) [X1 X2 X6 X7]
+ (-0.004684920226866116) [X1 Y2 Y6 X7]
+ (-0.00457501518889368) [Y1 X2 X12 Y13]
+ (-0.00457501518889368) [Y1 Y2 Y12 Y13]
+ (-0.00457501518889368) [X1 X2 X12 X13]
+ (-0.00457501518889368) [X1 Y2 Y12 X13]
+ (-0.004424843668493054) [Y1 X2 X4 Y5]
+ (-0.004424843668493054) [Y1 Y2 Y4 Y5]
+ (-0.004424843668493054) [X1 X2 X4 X5]
+ (-0.004424843668493054) [X1 Y2 Y4 X5]
+ (-0.0027458273154221265) [Y0 Y1 X4 X5]
+ (-0.0027458273154221265) [X0 X1 Y4 Y5]
+ (-0.001799193008382776) [Y1 X2 X10 Y11]
+ (-0.001799193008382776) [Y1 Y2 Y10 Y11]
+ (-0.001799193008382776) [X1 X2 X10 X11]
+ (-0.001799193008382776) [X1 Y2 Y10 X11]
+ (-0.0016639606584258372) [Y2 Z3 Z4 Y6]
+ (-0.0016639606584258372) [X2 Z3 Z4 X6]
+ (-0.0016639606584258372) [Y3 Z5 Z6 Y7]
+ (-0.0016639606584258372) [X3 Z5 Z6 X7]
+ (-0.0004957972886374948) [Y2 Z4 Z5 Y6]
+ (-0.0004957972886374948) [X2 Z4 Z5 X6]
+ (-0.00029222567245317606) [Y7 Y8 X9 X10]
+ (-0.00029222567245317606) [X7 X8 Y9 Y10]
+ (-7.954224644512314e-06) [Y10 Z11 Y12 Z13]
+ (-7.954224644512314e-06) [X10 Z11 X12 Z13]
+ (-7.801538358394311e-06) [Y2 Z3 Y4 Z11]
+ (-7.801538358394311e-06) [X2 Z3 X4 Z11]
+ (-7.801538358394311e-06) [Y3 Z4 Y5 Z10]
+ (-7.801538358394311e-06) [X3 Z4 X5 Z10]
+ (-5.974176880323059e-06) [Y5 X6 X10 Y11]
+ (-5.974176880323059e-06) [Y5 Y6 Y10 Y11]
+ (-5.974176880323059e-06) [X5 X6 X10 X11]
+ (-5.974176880323059e-06) [X5 Y6 Y10 X11]
+ (-4.642978984005855e-06) [Y3 X4 X10 Y11]
+ (-4.642978984005855e-06) [Y3 Y4 Y10 Y11]
+ (-4.642978984005855e-06) [X3 X4 X10 X11]
+ (-4.642978984005855e-06) [X3 Y4 Y10 X11]
+ (-4.281812047748999e-06) [Y4 Z5 Y6 Z11]
+ (-4.281812047748999e-06) [X4 Z5 X6 Z11]
+ (-4.281812047748999e-06) [Y5 Z6 Y7 Z10]
+ (-4.281812047748999e-06) [X5 Z6 X7 Z10]
+ (-3.1585593743888897e-06) [Y2 Z3 Y4 Z10]
+ (-3.1585593743888897e-06) [X2 Z3 X4 Z10]
+ (-3.1585593743888897e-06) [Y3 Z4 Y5 Z11]
+ (-3.1585593743888897e-06) [X3 Z4 X5 Z11]
+ (-2.172638015956194e-06) [Y2 X3 X11 Y12]
+ (-2.172638015956194e-06) [Y2 Y3 Y11 Y12]
+ (-2.172638015956194e-06) [X2 X3 X11 X12]
+ (-2.172638015956194e-06) [X2 Y3 Y11 X12]
+ (-1.8781495309963883e-06) [Z4 Y10 Z11 Y12]
+ (-1.8781495309963883e-06) [Z4 X10 Z11 X12]
+ (-1.8781495309963883e-06) [Z5 Y11 Z12 Y13]
+ (-1.8781495309963883e-06) [Z5 X11 Z12 X13]
+ (-1.4548066800557785e-06) [Y3 X4 X6 Y7]
+ (-1.4548066800557785e-06) [Y3 Y4 Y6 Y7]
+ (-1.4548066800557785e-06) [X3 X4 X6 X7]
+ (-1.4548066800557785e-06) [X3 Y4 Y6 X7]
+ (-1.1953920657620221e-06) [Y2 Z3 Y4 Z7]
+ (-1.1953920657620221e-06) [X2 Z3 X4 Z7]
+ (-1.1953920657620221e-06) [Y3 Z4 Y5 Z6]
+ (-1.1953920657620221e-06) [X3 Z4 X5 Z6]
+ (-1.1907331014359382e-06) [Z0 Y3 Z4 Y5]
+ (-1.1907331014359382e-06) [Z0 X3 Z4 X5]
+ (-1.1907331014359382e-06) [Z1 Y2 Z3 Y4]
+ (-1.1907331014359382e-06) [Z1 X2 Z3 X4]
+ (-1.1094124952885833e-06) [Z2 Y11 Z12 Y13]
+ (-1.1094124952885833e-06) [Z2 X11 Z12 X13]
+ (-1.1094124952885833e-06) [Z3 Y10 Z11 Y12]
+ (-1.1094124952885833e-06) [Z3 X10 Z11 X12]
+ (-8.336695524443844e-07) [Z0 Y2 Z3 Y4]
+ (-8.336695524443844e-07) [Z0 X2 Z3 X4]
+ (-8.336695524443844e-07) [Z1 Y3 Z4 Y5]
+ (-8.336695524443844e-07) [Z1 X3 Z4 X5]
+ (-7.956666948644712e-07) [Y3 X4 X8 Y9]
+ (-7.956666948644712e-07) [Y3 Y4 Y8 Y9]
+ (-7.956666948644712e-07) [X3 X4 X8 X9]
+ (-7.956666948644712e-07) [X3 Y4 Y8 X9]
+ (-7.765082298255956e-07) [Y2 Z3 Y4 Z5]
+ (-7.765082298255956e-07) [X2 Z3 X4 Z5]
+ (-6.6284272624644e-07) [Y8 X9 X11 Y12]
+ (-6.6284272624644e-07) [Y8 Y9 Y11 Y12]
+ (-6.6284272624644e-07) [X8 X9 X11 X12]
+ (-6.6284272624644e-07) [X8 Y9 Y11 X12]
+ (-5.769436545515969e-07) [Y2 Z3 Y4 Z9]
+ (-5.769436545515969e-07) [X2 Z3 X4 Z9]
+ (-5.769436545515969e-07) [Y3 Z4 Y5 Z8]
+ (-5.769436545515969e-07) [X3 Z4 X5 Z8]
+ (-5.627722023294061e-07) [Y0 X1 X11 Y12]
+ (-5.627722023294061e-07) [Y0 Y1 Y11 Y12]
+ (-5.627722023294061e-07) [X0 X1 X11 X12]
+ (-5.627722023294061e-07) [X0 Y1 Y11 X12]
+ (-5.471606047754267e-07) [Y1 X2 X11 Y12]
+ (-5.471606047754267e-07) [X1 Y2 Y11 X12]
+ (-3.570635489915134e-07) [Y0 X1 X3 Y4]
+ (-3.570635489915134e-07) [Y0 Y1 Y3 Y4]
+ (-3.570635489915134e-07) [X0 X1 X3 X4]
+ (-3.570635489915134e-07) [X0 Y1 Y3 X4]
+ (-1.9332121299963122e-07) [Y1 X2 X3 Y4]
+ (-1.9332121299963122e-07) [X1 Y2 Y3 X4]
+ (-1.2919458259252962e-07) [Y1 Z2 Z3 Y5]
+ (-1.2919458259252962e-07) [X1 Z2 Z3 X5]
+ (-3.2261056233124502e-09) [Y1 Y2 X5 X6]
+ (-3.2261056233124502e-09) [X1 X2 Y5 Y6]
+ (3.2261056233124502e-09) [Y1 X2 X5 Y6]
+ (3.2261056233124502e-09) [X1 Y2 Y5 X6]
+ (3.2284033528104383e-08) [Y4 Z5 Y6 Z12]
+ (3.2284033528104383e-08) [X4 Z5 X6 Z12]
+ (3.2284033528104383e-08) [Y5 Z6 Y7 Z13]
+ (3.2284033528104383e-08) [X5 Z6 X7 Z13]
+ (1.6728571451953965e-07) [Y0 Z1 Z3 Y4]
+ (1.6728571451953965e-07) [X0 Z1 Z3 X4]
+ (1.6728571451953965e-07) [Y1 Z2 Z4 Y5]
+ (1.6728571451953965e-07) [X1 Z2 Z4 X5]
+ (1.9332121299963122e-07) [Y1 Y2 X3 X4]
+ (1.9332121299963122e-07) [X1 X2 Y3 Y4]
+ (2.1872304031287434e-07) [Y2 Z3 Y4 Z8]
+ (2.1872304031287434e-07) [X2 Z3 X4 Z8]
+ (2.1872304031287434e-07) [Y3 Z4 Y5 Z9]
+ (2.1872304031287434e-07) [X3 Z4 X5 Z9]
+ (2.1989637660229018e-07) [Y2 X3 X5 Y6]
+ (2.1989637660229018e-07) [Y2 Y3 Y5 Y6]
+ (2.1989637660229018e-07) [X2 X3 X5 X6]
+ (2.1989637660229018e-07) [X2 Y3 Y5 X6]
+ (2.447264802856554e-07) [Y0 X1 X5 Y6]
+ (2.447264802856554e-07) [Y0 Y1 Y5 Y6]
+ (2.447264802856554e-07) [X0 X1 X5 X6]
+ (2.447264802856554e-07) [X0 Y1 Y5 X6]
+ (2.5941461429348535e-07) [Y2 Z3 Y4 Z6]
+ (2.5941461429348535e-07) [X2 Z3 X4 Z6]
+ (2.5941461429348535e-07) [Y3 Z4 Y5 Z7]
+ (2.5941461429348535e-07) [X3 Z4 X5 Z7]
+ (3.606069275187846e-07) [Y0 Z1 Z2 Y4]
+ (3.606069275187846e-07) [X0 Z1 Z2 X4]
+ (3.606069275187846e-07) [Y1 Z3 Z4 Y5]
+ (3.606069275187846e-07) [X1 Z3 Z4 X5]
+ (4.837953322849909e-07) [Y5 X6 X8 Y9]
+ (4.837953322849909e-07) [Y5 Y6 Y8 Y9]
+ (4.837953322849909e-07) [X5 X6 X8 X9]
+ (4.837953322849909e-07) [X5 Y6 Y8 X9]
+ (5.471606047754267e-07) [Y1 Y2 X11 X12]
+ (5.471606047754267e-07) [X1 X2 Y11 Y12]
+ (5.929280463290183e-07) [Z4 Y5 Z6 Y7]
+ (5.929280463290183e-07) [Z4 X5 Z6 X7]
+ (9.344970029641353e-07) [Z8 Y11 Z12 Y13]
+ (9.344970029641353e-07) [Z8 X11 Z12 X13]
+ (9.344970029641353e-07) [Z9 Y10 Z11 Y12]
+ (9.344970029641353e-07) [Z9 X10 Z11 X12]
+ (9.509134429048886e-07) [Z2 Y4 Z5 Y6]
+ (9.509134429048886e-07) [Z2 X4 Z5 X6]
+ (9.509134429048886e-07) [Z3 Y5 Z6 Y7]
+ (9.509134429048886e-07) [Z3 X5 Z6 X7]
+ (1.0357924741639067e-06) [Y6 X7 X11 Y12]
+ (1.0357924741639067e-06) [Y6 Y7 Y11 Y12]
+ (1.0357924741639067e-06) [X6 X7 X11 X12]
+ (1.0357924741639067e-06) [X6 Y7 Y11 X12]
+ (1.0632255206691286e-06) [Z2 Y10 Z11 Y12]
+ (1.0632255206691286e-06) [Z2 X10 Z11 X12]
+ (1.0632255206691286e-06) [Z3 Y11 Z12 Y13]
+ (1.0632255206691286e-06) [Z3 X11 Z12 X13]
+ (1.1708098195074498e-06) [Z2 Y5 Z6 Y7]
+ (1.1708098195074498e-06) [Z2 X5 Z6 X7]
+ (1.1708098195074498e-06) [Z3 Y4 Z5 Y6]
+ (1.1708098195074498e-06) [Z3 X4 Z5 X6]
+ (1.3980242834488166e-06) [Y4 Z5 Y6 Z8]
+ (1.3980242834488166e-06) [X4 Z5 X6 Z8]
+ (1.3980242834488166e-06) [Y5 Z6 Y7 Z9]
+ (1.3980242834488166e-06) [X5 Z6 X7 Z9]
+ (1.5973397292105753e-06) [Z8 Y10 Z11 Y12]
+ (1.5973397292105753e-06) [Z8 X10 Z11 X12]
+ (1.5973397292105753e-06) [Z9 Y11 Z12 Y13]
+ (1.5973397292105753e-06) [Z9 X11 Z12 X13]
+ (1.6021751061174477e-06) [Z2 Y3 Z4 Y5]
+ (1.6021751061174477e-06) [Z2 X3 Z4 X5]
+ (1.614960798053671e-06) [Z0 Y11 Z12 Y13]
+ (1.614960798053671e-06) [Z0 X11 Z12 X13]
+ (1.614960798053671e-06) [Z1 Y10 Z11 Y12]
+ (1.614960798053671e-06) [Z1 X10 Z11 X12]
+ (1.6923648325736264e-06) [Y4 Z5 Y6 Z10]
+ (1.6923648325736264e-06) [X4 Z5 X6 Z10]
+ (1.6923648325736264e-06) [Y5 Z6 Y7 Z11]
+ (1.6923648325736264e-06) [X5 Z6 X7 Z11]
+ (1.8163673813978432e-06) [Z4 Y11 Z12 Y13]
+ (1.8163673813978432e-06) [Z4 X11 Z12 X13]
+ (1.8163673813978432e-06) [Z5 Y10 Z11 Y12]
+ (1.8163673813978432e-06) [Z5 X10 Z11 X12]
+ (1.8540565275789768e-06) [Y4 Z5 Y6 Z7]
+ (1.8540565275789768e-06) [X4 Z5 X6 Z7]
+ (1.8551374761678618e-06) [Z6 Y10 Z11 Y12]
+ (1.8551374761678618e-06) [Z6 X10 Z11 X12]
+ (1.8551374761678618e-06) [Z7 Y11 Z12 Y13]
+ (1.8551374761678618e-06) [Z7 X11 Z12 X13]
+ (1.8818196157338075e-06) [Y4 Z5 Y6 Z9]
+ (1.8818196157338075e-06) [X4 Z5 X6 Z9]
+ (1.8818196157338075e-06) [Y5 Z6 Y7 Z8]
+ (1.8818196157338075e-06) [X5 Z6 X7 Z8]
+ (2.1777330003828587e-06) [Z0 Y10 Z11 Y12]
+ (2.1777330003828587e-06) [Z0 X10 Z11 X12]
+ (2.1777330003828587e-06) [Z1 Y11 Z12 Y13]
+ (2.1777330003828587e-06) [Z1 X11 Z12 X13]
+ (2.8909299503291665e-06) [Z6 Y11 Z12 Y13]
+ (2.8909299503291665e-06) [Z6 X11 Z12 X13]
+ (2.8909299503291665e-06) [Z7 Y10 Z11 Y12]
+ (2.8909299503291665e-06) [Z7 X10 Z11 X12]
+ (3.0992966242962398e-06) [Z0 Y4 Z5 Y6]
+ (3.0992966242962398e-06) [Z0 X4 Z5 X6]
+ (3.0992966242962398e-06) [Z1 Y5 Z6 Y7]
+ (3.0992966242962398e-06) [Z1 X5 Z6 X7]
+ (3.117366389552674e-06) [Y0 Z2 Z3 Y4]
+ (3.117366389552674e-06) [X0 Z2 Z3 X4]
+ (3.344023104582028e-06) [Z0 Y5 Z6 Y7]
+ (3.344023104582028e-06) [Z0 X5 Z6 X7]
+ (3.344023104582028e-06) [Z1 Y4 Z5 Y6]
+ (3.344023104582028e-06) [Z1 X4 Z5 X6]
+ (3.539009985481078e-06) [Y2 Z3 Y4 Z12]
+ (3.539009985481078e-06) [X2 Z3 X4 Z12]
+ (3.539009985481078e-06) [Y3 Z4 Y5 Z13]
+ (3.539009985481078e-06) [X3 Z4 X5 Z13]
+ (3.6945169123942315e-06) [Y4 X5 X11 Y12]
+ (3.6945169123942315e-06) [Y4 Y5 Y11 Y12]
+ (3.6945169123942315e-06) [X4 X5 X11 X12]
+ (3.6945169123942315e-06) [X4 Y5 Y11 X12]
+ (4.556473763267944e-06) [Y5 X6 X12 Y13]
+ (4.556473763267944e-06) [Y5 Y6 Y12 Y13]
+ (4.556473763267944e-06) [X5 X6 X12 X13]
+ (4.556473763267944e-06) [X5 Y6 Y12 X13]
+ (4.5887577967956145e-06) [Y4 Z5 Y6 Z13]
+ (4.5887577967956145e-06) [X4 Z5 X6 Z13]
+ (4.5887577967956145e-06) [Y5 Z6 Y7 Z12]
+ (4.5887577967956145e-06) [X5 Z6 X7 Z12]
+ (5.275783449201895e-06) [Y3 X4 X12 Y13]
+ (5.275783449201895e-06) [Y3 Y4 Y12 Y13]
+ (5.275783449201895e-06) [X3 X4 X12 X13]
+ (5.275783449201895e-06) [X3 Y4 Y12 X13]
+ (8.194104981214256e-06) [Z10 Y11 Z12 Y13]
+ (8.194104981214256e-06) [Z10 X11 Z12 X13]
+ (8.81479343468319e-06) [Y2 Z3 Y4 Z13]
+ (8.81479343468319e-06) [X2 Z3 X4 Z13]
+ (8.81479343468319e-06) [Y3 Z4 Y5 Z12]
+ (8.81479343468319e-06) [X3 Z4 X5 Z12]
+ (0.00029222567245317606) [Y7 X8 X9 Y10]
+ (0.00029222567245317606) [X7 Y8 Y9 X10]
+ (0.0011058984809191148) [Y0 Z1 Y2 Z5]
+ (0.0011058984809191148) [X0 Z1 X2 Z5]
+ (0.0011058984809191148) [Y1 Z2 Y3 Z4]
+ (0.0011058984809191148) [X1 Z2 X3 Z4]
+ (0.0017560659628976416) [Y0 Z1 Y2 Z11]
+ (0.0017560659628976416) [X0 Z1 X2 Z11]
+ (0.0017560659628976416) [Y1 Z2 Y3 Z10]
+ (0.0017560659628976416) [X1 Z2 X3 Z10]
+ (0.0023262348476322872) [Y0 Z1 Y2 Z13]
+ (0.0023262348476322872) [X0 Z1 X2 Z13]
+ (0.0023262348476322872) [Y1 Z2 Y3 Z12]
+ (0.0023262348476322872) [X1 Z2 X3 Z12]
+ (0.0027458273154221265) [Y0 X1 X4 Y5]
+ (0.0027458273154221265) [X0 Y1 Y4 X5]
+ (0.0029297682786106815) [Y0 Z1 Y2 Z9]
+ (0.0029297682786106815) [X0 Z1 X2 Z9]
+ (0.0029297682786106815) [Y1 Z2 Y3 Z8]
+ (0.0029297682786106815) [X1 Z2 X3 Z8]
+ (0.0032769650657712246) [Y0 Z1 Y2 Z3]
+ (0.0032769650657712246) [X0 Z1 X2 Z3]
+ (0.003347626470717248) [Y0 Z1 Y2 Z7]
+ (0.003347626470717248) [X0 Z1 X2 Z7]
+ (0.003347626470717248) [Y1 Z2 Y3 Z6]
+ (0.003347626470717248) [X1 Z2 X3 Z6]
+ (0.0034794217292727645) [Y2 Z3 Z5 Y6]
+ (0.0034794217292727645) [X2 Z3 Z5 X6]
+ (0.0034794217292727645) [Y3 Z4 Z6 Y7]
+ (0.0034794217292727645) [X3 Z4 Z6 X7]
+ (0.003555258971280418) [Y0 Z1 Y2 Z10]
+ (0.003555258971280418) [X0 Z1 X2 Z10]
+ (0.003555258971280418) [Y1 Z2 Y3 Z11]
+ (0.003555258971280418) [X1 Z2 X3 Z11]
+ (0.005143382387698603) [Y3 X4 X5 Y6]
+ (0.005143382387698603) [X3 Y4 Y5 X6]
+ (0.005283785753830626) [Y0 X1 X12 Y13]
+ (0.005283785753830626) [X0 Y1 Y12 X13]
+ (0.00553074214941217) [Y0 Z1 Y2 Z4]
+ (0.00553074214941217) [X0 Z1 X2 Z4]
+ (0.00553074214941217) [Y1 Z2 Y3 Z5]
+ (0.00553074214941217) [X1 Z2 X3 Z5]
+ (0.006087836804235032) [Y8 X9 X12 Y13]
+ (0.006087836804235032) [X8 Y9 Y12 X13]
+ (0.00650936123407709) [Y0 X1 X8 Y9]
+ (0.00650936123407709) [X0 Y1 Y8 X9]
+ (0.00688820454232299) [Y0 X1 X6 Y7]
+ (0.00688820454232299) [X0 Y1 Y6 X7]
+ (0.006901250036525967) [Y0 Z1 Y2 Z12]
+ (0.006901250036525967) [X0 Z1 X2 Z12]
+ (0.006901250036525967) [Y1 Z2 Y3 Z13]
+ (0.006901250036525967) [X1 Z2 X3 Z13]
+ (0.007156920529184707) [Y4 X5 X8 Y9]
+ (0.007156920529184707) [X4 Y5 Y8 X9]
+ (0.007731432847364331) [Y0 X1 X10 Y11]
+ (0.007731432847364331) [X0 Y1 Y10 X11]
+ (0.008032546697583364) [Y0 Z1 Y2 Z6]
+ (0.008032546697583364) [X0 Z1 X2 Z6]
+ (0.008032546697583364) [Y1 Z2 Y3 Z7]
+ (0.008032546697583364) [X1 Z2 X3 Z7]
+ (0.009560716496449257) [Y8 X9 X10 Y11]
+ (0.009560716496449257) [X8 Y9 Y10 X11]
+ (0.011055016688727276) [Y0 Z1 Y2 Z8]
+ (0.011055016688727276) [X0 Z1 X2 Z8]
+ (0.011055016688727276) [Y1 Z2 Y3 Z9]
+ (0.011055016688727276) [X1 Z2 X3 Z9]
+ (0.01128514461831756) [Y5 Y6 X11 X12]
+ (0.01128514461831756) [X5 X6 Y11 Y12]
+ (0.011307208030064578) [Y7 Z8 Z9 Y11]
+ (0.011307208030064578) [X7 Z8 Z9 X11]
+ (0.011982342583255765) [Y4 X5 X6 Y7]
+ (0.011982342583255765) [X4 Y5 Y6 X7]
+ (0.013873400067633128) [Y6 X7 X8 Y9]
+ (0.013873400067633128) [X6 Y7 Y8 X9]
+ (0.014583638327011018) [Y0 X1 X2 Y3]
+ (0.014583638327011018) [X0 Y1 Y2 X3]
+ (0.015577227075461728) [Y2 X3 X12 Y13]
+ (0.015577227075461728) [X2 Y3 Y12 X13]
+ (0.017366053264629745) [Y6 X7 X12 Y13]
+ (0.017366053264629745) [X6 Y7 Y12 X13]
+ (0.01768013765712775) [Y4 X5 X10 Y11]
+ (0.01768013765712775) [X4 Y5 Y10 X11]
+ (0.017825011932634815) [Y6 X7 X10 Y11]
+ (0.017825011932634815) [X6 Y7 Y10 X11]
+ (0.01902831871829496) [Y3 Y4 X11 X12]
+ (0.01902831871829496) [X3 X4 Y11 Y12]
+ (0.025384663696666032) [Y2 X3 X10 Y11]
+ (0.025384663696666032) [X2 Y3 Y10 X11]
+ (0.025996206267175942) [Y3 Z4 Z5 Y7]
+ (0.025996206267175942) [X3 Z4 Z5 X7]
+ (0.02868521731027948) [Y10 X11 X12 Y13]
+ (0.02868521731027948) [X10 Y11 Y12 X13]
+ (0.029812299601144253) [Y6 Z7 Z8 Y10]
+ (0.029812299601144253) [X6 Z7 Z8 X10]
+ (0.029812299601144253) [Y7 Z9 Z10 Y11]
+ (0.029812299601144253) [X7 Z9 Z10 X11]
+ (0.03010452527359743) [Y6 Z7 Z9 Y10]
+ (0.03010452527359743) [X6 Z7 Z9 X10]
+ (0.03010452527359743) [Y7 Z8 Z10 Y11]
+ (0.03010452527359743) [X7 Z8 Z10 X11]
+ (0.030787440718631116) [Y6 Z8 Z9 Y10]
+ (0.030787440718631116) [X6 Z8 Z9 X10]
+ (0.031143804196349937) [Y2 X3 X6 Y7]
+ (0.031143804196349937) [X2 Y3 Y6 X7]
+ (0.03583955718503571) [Y2 X3 X4 Y5]
+ (0.03583955718503571) [X2 Y3 Y4 X5]
+ (0.03619409348747799) [Y2 X3 X8 Y9]
+ (0.03619409348747799) [X2 Y3 Y8 X9]
+ (0.038314669373458385) [Y4 X5 X12 Y13]
+ (0.038314669373458385) [X4 Y5 Y12 X13]
+ (0.1043306148531761) [Z0 Y1 Z2 Y3]
+ (0.1043306148531761) [Z0 X1 Z2 X3]
+ (3.204142230604051e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.204142230604051e-06) [X0 Z1 Z2 Z3 X4]
+ (3.204142230604052e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.204142230604052e-06) [X1 Z2 Z3 Z4 X5]
+ (0.12133242248353078) [Y2 Z3 Z4 Z5 Y6]
+ (0.12133242248353078) [X2 Z3 Z4 Z5 X6]
+ (0.12133242248353084) [Y3 Z4 Z5 Z6 Y7]
+ (0.12133242248353084) [X3 Z4 Z5 Z6 X7]
+ (0.22847946311061676) [Y6 Z7 Z8 Z9 Y10]
+ (0.22847946311061676) [X6 Z7 Z8 Z9 X10]
+ (0.22847946311061682) [Y7 Z8 Z9 Z10 Y11]
+ (0.22847946311061682) [X7 Z8 Z9 Z10 X11]
+ (-0.04587942403075064) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (-0.04587942403075064) [X0 Z2 Z3 Z4 Z5 X6]
+ (-0.024353136084512124) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (-0.024353136084512124) [Y2 Z3 Y4 X11 Z12 X13]
+ (-0.024353136084512124) [X2 Z3 X4 Y11 Z12 Y13]
+ (-0.024353136084512124) [X2 Z3 X4 X11 Z12 X13]
+ (-0.024353136084512124) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (-0.024353136084512124) [Y3 Z4 Y5 X10 Z11 X12]
+ (-0.024353136084512124) [X3 Z4 X5 Y10 Z11 Y12]
+ (-0.024353136084512124) [X3 Z4 X5 X10 Z11 X12]
+ (-0.01558827786511269) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (-0.01558827786511269) [X2 Z3 X4 X10 Z11 X12]
+ (-0.01558827786511269) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (-0.01558827786511269) [X3 Z4 X5 X11 Z12 X13]
+ (-0.015225659057128414) [Y3 Z4 Z5 X6 X10 Y11]
+ (-0.015225659057128414) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (-0.015225659057128414) [X3 Z4 Z5 X6 X10 X11]
+ (-0.015225659057128414) [X3 Z4 Z5 Y6 Y10 X11]
+ (-0.014564473640806342) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.014564473640806342) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.014564473640806342) [X7 Z8 Z9 X10 X12 X13]
+ (-0.014564473640806342) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.01441118977008507) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (-0.01441118977008507) [X2 Z3 Z4 Z5 X6 Z11]
+ (-0.01441118977008507) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (-0.01441118977008507) [X3 Z4 Z5 Z6 X7 Z10]
+ (-0.012214985322757266) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.012214985322757266) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.012214985322757266) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.012214985322757266) [X4 Z5 X6 X11 Z12 X13]
+ (-0.012214985322757266) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.012214985322757266) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.012214985322757266) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.012214985322757266) [X5 Z6 X7 X10 Z11 X12]
+ (-0.010263460498895517) [Y2 Z3 X4 X10 Z11 Y12]
+ (-0.010263460498895517) [X2 Z3 Y4 Y10 Z11 X12]
+ (-0.010263460498895517) [Y3 Z4 X5 X11 Z12 Y13]
+ (-0.010263460498895517) [X3 Z4 Y5 Y11 Z12 X13]
+ (-0.008125248410116595) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.008125248410116595) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.007306763969603394) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.007306763969603394) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.007306763969603394) [X4 X5 X7 Z8 Z9 X10]
+ (-0.007306763969603394) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.005324817366217166) [Y2 Z3 Y4 X10 Z11 X12]
+ (-0.005324817366217166) [X2 Z3 X4 Y10 Z11 Y12]
+ (-0.005324817366217166) [Y3 Z4 Y5 X11 Z12 X13]
+ (-0.005324817366217166) [X3 Z4 X5 Y11 Z12 Y13]
+ (-0.004684920226866115) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.004684920226866115) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668615266021916) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.004668615266021916) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.00457501518889368) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.00457501518889368) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.004424843668493054) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.004424843668493054) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.0039615693730466715) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (-0.0039615693730466715) [X0 Z1 Z2 Z4 Z5 X6]
+ (-0.0039615693730466715) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (-0.0039615693730466715) [X1 Z3 Z4 Z5 Z6 X7]
+ (-0.0024629166214062788) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (-0.0024629166214062788) [X0 Z1 Z2 Z3 Z5 X6]
+ (-0.0024629166214062788) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (-0.0024629166214062788) [X1 Z2 Z3 Z4 Z6 X7]
+ (-0.0022939556230666194) [Y1 Y2 X3 Z4 Z5 X6]
+ (-0.0022939556230666194) [X1 X2 Y3 Z4 Z5 Y6]
+ (-0.001799193008382776) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.001799193008382776) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.0017278745823349807) [Y1 Z2 Z3 Y4 X11 X12]
+ (-0.0017278745823349807) [X1 Z2 Z3 X4 Y11 Y12]
+ (-0.0016676137499800512) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (-0.0016676137499800512) [X0 Z1 Z3 Z4 Z5 X6]
+ (-0.0016676137499800512) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (-0.0016676137499800512) [X1 Z2 Z4 Z5 Z6 X7]
+ (-0.0016095335163060116) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (-0.0016095335163060116) [X0 Z1 Z2 Z3 Z4 X6]
+ (-0.0016095335163060116) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (-0.0016095335163060116) [X1 Z2 Z3 Z5 Z6 X7]
+ (-0.0009298407044397048) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298407044397048) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298407044397048) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298407044397048) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533831051002677) [Y1 Z2 Z3 X4 X5 Y6]
+ (-0.0008533831051002677) [X1 Z2 Z3 Y4 Y5 X6]
+ (-0.0006650303449131571) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (-0.0006650303449131571) [X2 Z3 Z4 Z5 X6 Z12]
+ (-0.0006650303449131571) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (-0.0006650303449131571) [X3 Z4 Z5 Z6 X7 Z13]
+ (-0.0004957972886374947) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0004957972886374947) [Z2 X3 Z4 Z5 Z6 X7]
+ (-7.735870606033008e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.735870606033008e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.735870606033008e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.735870606033008e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-5.974176880323059e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (-5.974176880323059e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (-5.275783449201895e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.275783449201895e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.642978984005855e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.642978984005855e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.556473763267944e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (-4.556473763267944e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (-4.183808818884749e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (-4.183808818884749e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (-3.6945169123942315e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (-3.6945169123942315e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (-3.3342618642618477e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (-3.3342618642618477e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (-3.3130170736294165e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (-3.3130170736294165e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (-3.151295967334473e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.151295967334473e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.0882457170869175e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.0882457170869175e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.172638015956194e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (-2.172638015956194e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (-1.4548066800557785e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.4548066800557785e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.3304568548832586e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (-1.3304568548832586e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (-1.228269114239718e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (-1.228269114239718e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (-1.0357924741639067e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (-1.0357924741639067e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (-7.956666948644712e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.956666948644712e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.733096765708163e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.733096765708163e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.733096765708163e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.733096765708163e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.6284272624644e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (-6.6284272624644e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (-5.9273502352895e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.9273502352895e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.9273502352895e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.9273502352895e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.627722023294061e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (-5.627722023294061e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (-4.837953322849909e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (-4.837953322849909e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (-3.570635489915134e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.570635489915134e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.3280396615208456e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (-3.3280396615208456e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (-3.0867708949349273e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.0867708949349273e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.0867708949349273e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.0867708949349273e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.447264802856554e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (-2.447264802856554e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (-2.3712704563263618e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.3712704563263618e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.3712704563263618e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.3712704563263618e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.1989637660229018e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (-2.1989637660229018e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (-1.9332121299963122e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.9332121299963122e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.9332121299963122e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.9332121299963122e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.839394349121924e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.839394349121924e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.839394349121924e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.839394349121924e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.2919458259252962e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.2919458259252962e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.83956580805535e-08) [Y0 Z1 X2 X10 Z11 Y12]
+ (-1.83956580805535e-08) [X0 Z1 Y2 Y10 Z11 X12]
+ (-1.83956580805535e-08) [Y1 Z2 X3 X11 Z12 Y13]
+ (-1.83956580805535e-08) [X1 Z2 Y3 Y11 Z12 X13]
+ (-1.035149618408584e-09) [Y0 Z1 Z2 Z3 Y4 Z7]
+ (-1.035149618408584e-09) [X0 Z1 Z2 Z3 X4 Z7]
+ (-1.035149618408584e-09) [Y1 Z2 Z3 Z4 Y5 Z6]
+ (-1.035149618408584e-09) [X1 Z2 Z3 Z4 X5 Z6]
+ (2.270208285726801e-08) [Y0 Z1 Z2 X3 X5 Y6]
+ (2.270208285726801e-08) [Y0 Z1 Z2 Y3 Y5 Y6]
+ (2.270208285726801e-08) [X0 Z1 Z2 X3 X5 X6]
+ (2.270208285726801e-08) [X0 Z1 Z2 Y3 Y5 X6]
+ (2.270208285726801e-08) [Y1 X2 X4 Z5 Z6 Y7]
+ (2.270208285726801e-08) [Y1 Y2 Y4 Z5 Z6 Y7]
+ (2.270208285726801e-08) [X1 X2 X4 Z5 Z6 X7]
+ (2.270208285726801e-08) [X1 Y2 Y4 Z5 Z6 X7]
+ (2.5928188480687187e-08) [Y0 Z1 X2 X4 Z5 Y6]
+ (2.5928188480687187e-08) [X0 Z1 Y2 Y4 Z5 X6]
+ (2.5928188480687187e-08) [Y1 Z2 X3 X5 Z6 Y7]
+ (2.5928188480687187e-08) [X1 Z2 Y3 Y5 Z6 X7]
+ (8.057465304121575e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.057465304121575e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.057465304121575e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.057465304121575e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.649129576822692e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.649129576822692e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.649129576822692e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.649129576822692e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.1076529152075442e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (1.1076529152075442e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (1.1076529152075442e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (1.1076529152075442e-07) [X0 Z1 X2 X11 Z12 X13]
+ (1.1076529152075442e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (1.1076529152075442e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (1.1076529152075442e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (1.1076529152075442e-07) [X1 Z2 X3 X10 Z11 X12]
+ (1.3484968790782227e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (1.3484968790782227e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (1.3484968790782227e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (1.3484968790782227e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (1.3807579353100936e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (1.3807579353100936e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (1.3807579353100936e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (1.3807579353100936e-07) [X0 Z1 X2 X5 Z6 X7]
+ (1.3807579353100936e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (1.3807579353100936e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (1.3807579353100936e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (1.3807579353100936e-07) [X1 Z2 X3 X4 Z5 X6]
+ (1.6077787638827399e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (1.6077787638827399e-07) [X0 Z1 X2 X4 Z5 X6]
+ (1.6077787638827399e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (1.6077787638827399e-07) [X1 Z2 X3 X5 Z6 X7]
+ (1.8290428529417005e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.8290428529417005e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.8290428529417005e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.8290428529417005e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.1989637660229018e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (2.1989637660229018e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (2.447264802856554e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (2.447264802856554e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (3.236183414008631e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.236183414008631e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.236183414008631e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.236183414008631e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.3280396615208456e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (3.3280396615208456e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (3.570635489915134e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.570635489915134e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.837953322849909e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (4.837953322849909e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (5.287649466957542e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (5.287649466957542e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (5.287649466957542e-07) [X0 Z1 Z2 X3 X11 X12]
+ (5.287649466957542e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (5.287649466957542e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (5.287649466957542e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (5.287649466957542e-07) [X1 X2 X10 Z11 Z12 X13]
+ (5.287649466957542e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (5.627722023294061e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (5.627722023294061e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (6.395302382167525e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (6.395302382167525e-07) [X0 Z1 X2 X10 Z11 X12]
+ (6.395302382167525e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (6.395302382167525e-07) [X1 Z2 X3 X11 Z12 X13]
+ (6.579258962959237e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (6.579258962959237e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (6.579258962959237e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (6.579258962959237e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (6.6284272624644e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (6.6284272624644e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (7.956666948644712e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.956666948644712e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.30634296037914e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.30634296037914e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.30634296037914e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.30634296037914e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.0357924741639067e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (1.0357924741639067e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (1.228269114239718e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (1.228269114239718e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (1.2393113855319217e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.2393113855319217e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.2393113855319217e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.2393113855319217e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.3304568548832586e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (1.3304568548832586e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (1.4548066800557785e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.4548066800557785e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.172638015956194e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (2.172638015956194e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (3.0882457170869175e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.0882457170869175e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.117366389552674e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.117366389552674e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.151295967334473e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.151295967334473e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.211187435479823e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (3.211187435479823e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (3.211187435479823e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (3.211187435479823e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (3.2774382792595536e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (3.2774382792595536e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (3.2774382792595536e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (3.2774382792595536e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (3.3130170736294165e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (3.3130170736294165e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (3.610242245411638e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (3.610242245411638e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (3.610242245411638e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (3.610242245411638e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (3.6945169123942315e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (3.6945169123942315e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (3.769583602763701e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (3.769583602763701e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (4.253118746984103e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (4.253118746984103e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (4.556473763267944e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (4.556473763267944e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (4.642978984005855e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.642978984005855e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.275783449201895e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.275783449201895e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.974176880323059e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (5.974176880323059e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (6.2900197097908155e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (6.2900197097908155e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (6.2900197097908155e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (6.2900197097908155e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (6.5242045091066375e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (6.5242045091066375e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (6.5242045091066375e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (6.5242045091066375e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (7.4442674276860125e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (7.4442674276860125e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (7.4442674276860125e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (7.4442674276860125e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (7.518288824029558e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (7.518288824029558e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (7.518288824029558e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (7.518288824029558e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (8.774724282569271e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (8.774724282569271e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (8.774724282569271e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (8.774724282569271e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (0.00029222567245317606) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.00029222567245317606) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.00029222567245317606) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.00029222567245317606) [X6 Z7 X8 X9 Z10 X11]
+ (0.0008144692870433404) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (0.0008144692870433404) [X2 Z3 Z4 Z5 X6 Z10]
+ (0.0008144692870433404) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (0.0008144692870433404) [X3 Z4 Z5 Z6 X7 Z11]
+ (0.0008533831051002677) [Y1 Z2 Z3 Y4 X5 X6]
+ (0.0008533831051002677) [X1 Z2 Z3 X4 Y5 Y6]
+ (0.0017278745823349807) [Y1 Z2 Z3 X4 X11 Y12]
+ (0.0017278745823349807) [X1 Z2 Z3 Y4 Y11 X12]
+ (0.001799193008382776) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.001799193008382776) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.0022939556230666194) [Y1 X2 X3 Z4 Z5 Y6]
+ (0.0022939556230666194) [X1 Y2 Y3 Z4 Z5 X6]
+ (0.0027790407628720806) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (0.0027790407628720806) [X1 Z2 Z3 Z4 Z5 X7]
+ (0.003493800371495497) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (0.003493800371495497) [X2 Z3 Z4 Z5 X6 Z13]
+ (0.003493800371495497) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (0.003493800371495497) [X3 Z4 Z5 Z6 X7 Z12]
+ (0.0041588307164086525) [Y3 Z4 Z5 X6 X12 Y13]
+ (0.0041588307164086525) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (0.0041588307164086525) [X3 Z4 Z5 X6 X12 X13]
+ (0.0041588307164086525) [X3 Z4 Z5 Y6 Y12 X13]
+ (0.004424843668493054) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.004424843668493054) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.00457501518889368) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.00457501518889368) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.004668615266021916) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.004668615266021916) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.004684920226866115) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.004684920226866115) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.005143382387698603) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (0.005143382387698603) [Y2 Z3 Y4 X5 Z6 X7]
+ (0.005143382387698603) [X2 Z3 X4 Y5 Z6 Y7]
+ (0.005143382387698603) [X2 Z3 X4 X5 Z6 X7]
+ (0.005368616111419997) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.005368616111419997) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.005368616111419997) [X2 X3 X7 Z8 Z9 X10]
+ (0.005368616111419997) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.005652607315226246) [Y0 X1 X3 Z4 Z5 Y6]
+ (0.005652607315226246) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (0.005652607315226246) [X0 X1 X3 Z4 Z5 X6]
+ (0.005652607315226246) [X0 Y1 Y3 Z4 Z5 X6]
+ (0.005805121212332079) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (0.005805121212332079) [X2 Z3 Z4 Z5 X6 Z8]
+ (0.005805121212332079) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (0.005805121212332079) [X3 Z4 Z5 Z6 X7 Z9]
+ (0.007960839634238779) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.007960839634238779) [X4 Z5 X6 X10 Z11 X12]
+ (0.007960839634238779) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.007960839634238779) [X5 Z6 X7 X11 Z12 X13]
+ (0.008125248410116595) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.008125248410116595) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.008764858219399445) [Y2 Z3 Z4 X5 X11 Y12]
+ (0.008764858219399445) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (0.008764858219399445) [X2 Z3 Z4 X5 X11 X12]
+ (0.008764858219399445) [X2 Z3 Z4 Y5 Y11 X12]
+ (0.008764858219399445) [Y3 X4 X10 Z11 Z12 Y13]
+ (0.008764858219399445) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (0.008764858219399445) [X3 X4 X10 Z11 Z12 X13]
+ (0.008764858219399445) [X3 Y4 Y10 Z11 Z12 X13]
+ (0.008890680338678484) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.008890680338678484) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.008890680338678484) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.008890680338678484) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.010540434329270043) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.010540434329270043) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.010540434329270043) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.010540434329270043) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.010959994608955949) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.010959994608955949) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.010959994608955949) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.010959994608955949) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.011307208030064578) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.011307208030064578) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.011755995240389214) [Y3 Z4 Z5 X6 X8 Y9]
+ (0.011755995240389214) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (0.011755995240389214) [X3 Z4 Z5 X6 X8 X9]
+ (0.011755995240389214) [X3 Z4 Z5 Y6 Y8 X9]
+ (0.017561116452721293) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (0.017561116452721293) [X2 Z3 Z4 Z5 X6 Z9]
+ (0.017561116452721293) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (0.017561116452721293) [X3 Z4 Z5 Z6 X7 Z8]
+ (0.018266758578559347) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.018266758578559347) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.018266758578559347) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.018266758578559347) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.019020373875163973) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.019020373875163973) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.019020373875163973) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.019020373875163973) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.020175824956996047) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.020175824956996047) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.020175824956996047) [X4 Z5 Z6 X7 X11 X12]
+ (0.020175824956996047) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.020175824956996047) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.020175824956996047) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.020175824956996047) [X5 X6 X10 Z11 Z12 X13]
+ (0.020175824956996047) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.02438898998658397) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.02438898998658397) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.02438898998658397) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.02438898998658397) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.025104907970076386) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.025104907970076386) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.025104907970076386) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.025104907970076386) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.025996206267175942) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (0.025996206267175942) [X2 Z3 Z4 Z5 X6 Z7]
+ (0.027114878580431497) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (0.027114878580431497) [Z0 X2 Z3 Z4 Z5 X6]
+ (0.027114878580431497) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (0.027114878580431497) [Z1 X3 Z4 Z5 Z6 X7]
+ (0.030787440718631116) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.030787440718631116) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.03276748589565774) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (0.03276748589565774) [Z0 X3 Z4 Z5 Z6 X7]
+ (0.03276748589565774) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (0.03276748589565774) [Z1 X2 Z3 Z4 Z5 X6]
+ (0.056007135616976185) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.056007135616976185) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.056007135616976185) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.056007135616976185) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.056084494323036506) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.056084494323036506) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.056084494323036506) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.056084494323036506) [Z1 X7 Z8 Z9 Z10 X11]
+ (-0.04274326006300962) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (-0.04274326006300962) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (-0.0427432600630096) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-0.0427432600630096) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (2.5950813571612905e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.5950813571612905e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.5950813571612905e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.5950813571612905e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.631261949600617e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (6.631261949600617e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (6.631261949600618e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (6.631261949600618e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.04764261360012388) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.04764261360012388) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.04764261360012388) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.04764261360012388) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.04587942403075064) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-0.04587942403075064) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.04171881404451479) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.04171881404451479) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.04171881404451479) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.04171881404451479) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.03956454804161601) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.03956454804161601) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.03956454804161601) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03956454804161601) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03931810723103286) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03931810723103286) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.03931810723103286) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03931810723103286) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.025637212809953323) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.025637212809953323) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.025637212809953323) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.025637212809953323) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.02314522165325282) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.02314522165325282) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.022528354240815132) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.022528354240815132) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.019257453001866234) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019257453001866234) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.01902831871829496) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.01902831871829496) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.01602466609552875) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.01602466609552875) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.015225659057128414) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.015225659057128414) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-0.014603742410728936) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (-0.014603742410728936) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (-0.014564473640806342) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.014564473640806342) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.011755995240389214) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (-0.011755995240389214) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (-0.01128514461831756) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-0.01128514461831756) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-0.009841802923830786) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.009841802923830786) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.009612546714424571) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.009612546714424571) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.009612546714424571) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.009612546714424571) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.008469833341362707) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.008469833341362707) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.007306763969603394) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.007306763969603394) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005923799555609087) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.005923799555609087) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.005652607315226246) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (-0.005652607315226246) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (-0.005379929634070006) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (-0.005379929634070006) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (-0.005379929634070006) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (-0.005379929634070006) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (-0.005368616111419997) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005368616111419997) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.005241543597056575) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (-0.005241543597056575) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (-0.005241543597056575) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (-0.005241543597056575) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (-0.004636973516508874) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (-0.004636973516508874) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (-0.004636973516508874) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (-0.004636973516508874) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (-0.004311038607576276) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (-0.004311038607576276) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (-0.004311038607576276) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (-0.004311038607576276) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (-0.004158830716408653) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (-0.004158830716408653) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (-0.003989845257552995) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (-0.003989845257552995) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (-0.003989845257552995) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (-0.003989845257552995) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (-0.0022619706752180137) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (-0.0022619706752180137) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (-0.0022619706752180137) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (-0.0022619706752180137) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (-0.0022619706752180137) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (-0.0022619706752180137) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (-0.0022619706752180137) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (-0.0022619706752180137) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (-0.0013038029824591758) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (-0.0013038029824591758) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (-0.0013038029824591758) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (-0.0013038029824591758) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (-0.0012803055951397468) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (-0.0012803055951397468) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (-0.0012803055951397468) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (-0.0012803055951397468) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (-0.0010435237104209484) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (-0.0010435237104209484) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (-0.0010435237104209484) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (-0.0010435237104209484) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (-0.0008533831051002676) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (-0.0008533831051002676) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (-0.0008533831051002676) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (-0.0008533831051002676) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (-0.00024644081058314933) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00024644081058314933) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.735870606033008e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.735870606033008e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-4.183808818884749e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-4.183808818884749e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-3.5443574181906034e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.5443574181906034e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.5443574181906034e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.5443574181906034e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.5443574181906034e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.5443574181906034e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.5443574181906034e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.5443574181906034e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.3342618642618477e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-3.3342618642618477e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-3.3130170736294165e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-3.3130170736294165e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-3.3130170736294165e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-3.3130170736294165e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.5224581950395305e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.5224581950395305e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.5224581950395305e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.5224581950395305e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.5224581950395305e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224581950395305e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224581950395305e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.5224581950395305e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.3304568548832586e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (-1.3304568548832586e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (-1.3304568548832586e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (-1.3304568548832586e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (-7.988467821804955e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.988467821804955e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.988467821804955e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.988467821804955e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.189870461740873e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.189870461740873e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (-6.175164890501956e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175164890501956e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.471606047754267e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (-5.471606047754267e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (-4.561117011009754e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.561117011009754e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.561117011009754e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.561117011009754e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.523339375987698e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (-4.523339375987698e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (-3.427350810746412e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.427350810746412e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.427350810746412e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.427350810746412e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.0867708949349273e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.0867708949349273e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.3712704563263618e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.3712704563263618e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.839394349121924e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.839394349121924e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-1.7035434429343172e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035434429343172e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035434429343172e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (-1.7035434429343172e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (-9.208945499672612e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-9.208945499672612e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-9.208945499672612e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-9.208945499672612e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-8.057465304121575e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.057465304121575e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (-6.772951822521159e-08) [Y1 Z2 Z3 X4 X7 Z8 Z9 Y10]
+ (-6.772951822521159e-08) [X1 Z2 Z3 Y4 Y7 Z8 Z9 X10]
+ (-3.2261056233124502e-09) [Y0 Z1 Z2 Y3 X4 Z5 Z6 X7]
+ (-3.2261056233124502e-09) [X0 Z1 Z2 X3 Y4 Z5 Z6 Y7]
+ (3.2261056233124502e-09) [Y0 Z1 Z2 X3 X4 Z5 Z6 Y7]
+ (3.2261056233124502e-09) [X0 Z1 Z2 Y3 Y4 Z5 Z6 X7]
+ (6.046790399568281e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (6.046790399568281e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (6.046790399568281e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (6.046790399568281e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (6.772951822521159e-08) [Y1 Z2 Z3 Y4 X7 Z8 Z9 X10]
+ (6.772951822521159e-08) [X1 Z2 Z3 X4 Y7 Z8 Z9 Y10]
+ (8.057465304121575e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.057465304121575e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (1.839394349121924e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.839394349121924e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.3712704563263618e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.3712704563263618e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (2.888564749187455e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.888564749187455e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.888564749187455e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (2.888564749187455e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (3.0867708949349273e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.0867708949349273e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (3.3280396615208456e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (3.3280396615208456e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (3.3280396615208456e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (3.3280396615208456e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (4.523339375987698e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (4.523339375987698e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (5.471606047754267e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (5.471606047754267e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (6.175164890501956e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175164890501956e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.189870461740873e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (7.189870461740873e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (7.867608617758405e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (7.867608617758405e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (7.867608617758405e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (7.867608617758405e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (1.228269114239718e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (1.228269114239718e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (1.228269114239718e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (1.228269114239718e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (1.6288377722947257e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.6288377722947257e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.6288377722947257e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.6288377722947257e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.654090064856853e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.654090064856853e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.654090064856853e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.654090064856853e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.6893056762900833e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.6893056762900833e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.6893056762900833e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.6893056762900833e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (1.9429465397758153e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.9429465397758153e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.9429465397758153e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.9429465397758153e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.0110740213885787e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.0110740213885787e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.0110740213885787e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.0110740213885787e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.103163476384871e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.103163476384871e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.103163476384871e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.103163476384871e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.3609472124708642e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (2.3609472124708642e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (2.3609472124708642e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (2.3609472124708642e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (2.745510636012927e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.745510636012927e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.745510636012927e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.745510636012927e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.745510636012927e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.745510636012927e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.745510636012927e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.745510636012927e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.2117638713292885e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.2117638713292885e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.2117638713292885e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.2117638713292885e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.2117638713292885e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.2117638713292885e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.2117638713292885e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.2117638713292885e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.769583602763701e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.769583602763701e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.253118746984103e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.253118746984103e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.728781439200199e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (4.728781439200199e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (4.728781439200199e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.728781439200199e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.734578388381622e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (4.734578388381622e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (4.734578388381622e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (4.734578388381622e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (5.071403660008786e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (5.071403660008786e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (5.071403660008786e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (5.071403660008786e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (6.481752014837781e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (6.481752014837781e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.481752014837781e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (6.481752014837781e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (6.652106359131218e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (6.652106359131218e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (6.652106359131218e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (6.652106359131218e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.0897286516703045e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.0897286516703045e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (7.0897286516703045e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (7.0897286516703045e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (9.805982048388673e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (9.805982048388673e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (9.805982048388673e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (9.805982048388673e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (1.53166141455817e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (1.53166141455817e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (1.53166141455817e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (1.53166141455817e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.6103375007357823e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (1.6103375007357823e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.6103375007357823e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (1.6103375007357823e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (7.735870606033008e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.735870606033008e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.00013838603701343236) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (0.00013838603701343236) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (0.00013838603701343236) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (0.00013838603701343236) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (0.00024644081058314933) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00024644081058314933) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0004458488204492246) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.0004458488204492246) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.0004458488204492246) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.0004458488204492246) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.0005940157673974844) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940157673974844) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.0005940157673974844) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940157673974844) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.0005940157673974844) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940157673974844) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.0005940157673974844) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940157673974844) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.0009581676927588375) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (0.0009581676927588375) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (0.0009581676927588375) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (0.0009581676927588375) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (0.0009581676927588375) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (0.0009581676927588375) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (0.0009581676927588375) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (0.0009581676927588375) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (0.0022939556230666194) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (0.0022939556230666194) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (0.0022939556230666194) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (0.0022939556230666194) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (0.002686042275093818) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (0.002686042275093818) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (0.002686042275093818) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (0.002686042275093818) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (0.0027790407628720806) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (0.0027790407628720806) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (0.0032675148971553275) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (0.0032675148971553275) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (0.0032675148971553275) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (0.0032675148971553275) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (0.0033566679213691275) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (0.0033566679213691275) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (0.0033566679213691275) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (0.0033566679213691275) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (0.004158830716408653) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (0.004158830716408653) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (0.00511446408647114) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.00511446408647114) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.00511446408647114) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.00511446408647114) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.00511446408647114) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.00511446408647114) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.00511446408647114) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.00511446408647114) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.005262631033419401) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.005262631033419401) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.005262631033419401) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.005262631033419401) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.005368616111419997) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005368616111419997) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005652607315226246) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (0.005652607315226246) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (0.005708479853868626) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.005708479853868626) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.005708479853868626) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.005708479853868626) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.005923799555609087) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.005923799555609087) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.007306763969603394) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.007306763969603394) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.008469833341362707) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.008469833341362707) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.009841802923830786) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.009841802923830786) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.01128514461831756) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (0.01128514461831756) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (0.011755995240389214) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (0.011755995240389214) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (0.014564473640806342) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.014564473640806342) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.014603742410728936) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (0.014603742410728936) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (0.015225659057128414) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.015225659057128414) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.01602466609552875) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.01602466609552875) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.018888995077393583) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (0.018888995077393583) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (0.018888995077393583) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.018888995077393583) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.01902831871829496) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.01902831871829496) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.019257453001866234) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (0.019257453001866234) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (0.021433980116896404) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.021433980116896404) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.021433980116896404) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.021433980116896404) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.024282031623368935) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (0.024282031623368935) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (0.024755507980765812) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.024755507980765812) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.024755507980765812) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.024755507980765812) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.02873079800122437) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.02873079800122437) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.02873079800122437) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.02873079800122437) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.029903813458259113) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.029903813458259113) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.029903813458259113) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.029903813458259113) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.035608400352257465) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.035608400352257465) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.03935925039149475) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.03935925039149475) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.03935925039149475) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.03935925039149475) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.36937137554438576) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.36937137554438576) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.36937137554438576) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.36937137554438576) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.2816433575340578) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.2816433575340578) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.28164335753405784) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.28164335753405784) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.09065142344955501) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.09065142344955501) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.09065142344955501) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.09065142344955501) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684736029518614) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.08684736029518614) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684736029518614) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.08684736029518614) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.05859215179555093) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.05859215179555093) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.034903304270648035) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.034903304270648035) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.034903304270648035) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.034903304270648035) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591832088249197) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.024591832088249197) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591832088249197) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.024591832088249197) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.02314522165325282) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.02314522165325282) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.022528354240815132) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.022528354240815132) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019299499858015958) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.019299499858015958) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019299499858015958) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.019299499858015958) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019299499858015958) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019299499858015958) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.019299499858015958) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019299499858015958) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.01602466609552875) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.01602466609552875) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.01602466609552875) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.01602466609552875) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.014603742410728938) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (-0.014603742410728938) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (-0.014603742410728938) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (-0.014603742410728938) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (-0.01075752420121241) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.01075752420121241) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.01075752420121241) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.01075752420121241) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010715477345062686) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010715477345062686) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010715477345062686) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010715477345062686) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311472182398845) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311472182398845) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311472182398845) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010311472182398845) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005408970758400719) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.005408970758400719) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.005408970758400719) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.005408970758400719) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.00528656905679907) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.00528656905679907) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.00528656905679907) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.00528656905679907) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.00476727664474625) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.00476727664474625) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.00476727664474625) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.00476727664474625) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.004668615266021916) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.004668615266021916) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.0038764821957235716) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (-0.0038764821957235716) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (-0.0038040631543688765) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0038040631543688765) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0038040631543688765) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0038040631543688765) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841545793424483) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841545793424483) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0033566679213691275) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (-0.0033566679213691275) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (-0.0032675148971553275) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (-0.0032675148971553275) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (-0.002446463422691723) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.002446463422691723) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.002446463422691723) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.002446463422691723) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.0017278745823349807) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.0017278745823349807) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.001640759116709813) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.001640759116709813) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0015324885626771469) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.0015324885626771469) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.0015324885626771469) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.0015324885626771469) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.0007870893705573733) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0007870893705573733) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007156737069762657) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0007156737069762657) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007156737069762657) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0007156737069762657) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0005192924120528198) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.0005192924120528198) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.0001940103060625518) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0001940103060625518) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.00018787486139245816) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.00018787486139245816) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.00018787486139245816) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.00018787486139245816) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.00013838603701343236) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.00013838603701343236) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-4.204685614972469e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-4.204685614972469e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-4.204685614972469e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.204685614972469e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-5.071403660008786e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-5.071403660008786e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-3.151295967334473e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.151295967334473e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0882457170869175e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.0882457170869175e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.9884125611914737e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.9884125611914737e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.874248561946267e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.874248561946267e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.3609472124708642e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-2.3609472124708642e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.3001958859121775e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.3001958859121775e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-7.867608617758405e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867608617758405e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.996951796018957e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.996951796018957e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.996951796018957e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.996951796018957e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.996951796018957e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.996951796018957e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.996951796018957e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.996951796018957e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.888564749187455e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.888564749187455e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.6863216618533024e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.6863216618533024e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035434429343172e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.7035434429343172e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.208945499672612e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.208945499672612e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.2040044053799154e-08) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (-3.2040044053799154e-08) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (-3.2040044053799154e-08) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (-3.2040044053799154e-08) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (3.568947417104652e-08) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (3.568947417104652e-08) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (3.568947417104652e-08) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (3.568947417104652e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (3.568947417104652e-08) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.568947417104652e-08) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (3.568947417104652e-08) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (3.568947417104652e-08) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.706834839174341e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.706834839174341e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.706834839174341e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.706834839174341e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.379737430819881e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (8.379737430819881e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.379737430819881e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (8.379737430819881e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.379737430819881e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (8.379737430819881e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (8.379737430819881e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (8.379737430819881e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (9.208945499672612e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.208945499672612e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.0716845266730602e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.0716845266730602e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.0716845266730602e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.0716845266730602e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.1782130944817113e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.1782130944817113e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.1782130944817113e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.1782130944817113e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.7035434429343172e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.7035434429343172e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.2498976211576176e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.2498976211576176e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.2498976211576176e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.2498976211576176e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.6863216618533024e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863216618533024e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.888564749187455e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.888564749187455e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.3766863738776047e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (3.3766863738776047e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (3.3766863738776047e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (3.3766863738776047e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (3.3766863738776047e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (3.3766863738776047e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (3.3766863738776047e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (3.3766863738776047e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (3.5682004801596107e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (3.5682004801596107e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (3.5682004801596107e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (3.5682004801596107e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.0921619318585344e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.0921619318585344e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.0921619318585344e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.0921619318585344e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.0921619318585344e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.0921619318585344e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.0921619318585344e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.0921619318585344e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.4490566735719134e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.4490566735719134e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.4490566735719134e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.4490566735719134e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.769457114111192e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.769457114111192e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.769457114111192e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.769457114111192e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.246849417176304e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.246849417176304e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.246849417176304e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.246849417176304e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.246849417176304e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.246849417176304e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.246849417176304e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.246849417176304e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.560553945658189e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.560553945658189e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (7.560553945658189e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (7.560553945658189e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (7.560553945658189e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.560553945658189e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.560553945658189e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.560553945658189e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.867608617758405e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (7.867608617758405e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.900025749865574e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (7.900025749865574e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (7.900025749865574e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (7.900025749865574e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (8.027844204825979e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (8.027844204825979e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (8.027844204825979e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.027844204825979e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (8.091539856149816e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (8.091539856149816e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (8.091539856149816e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (8.091539856149816e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (8.091539856149816e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (8.091539856149816e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (8.091539856149816e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (8.091539856149816e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (8.398527688735603e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (8.398527688735603e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (8.398527688735603e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (8.398527688735603e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.146822623002383e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (1.146822623002383e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (1.146822623002383e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (1.146822623002383e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.3001958859121775e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.3001958859121775e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.3609472124708642e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.3609472124708642e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.874248561946267e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.874248561946267e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.883653164579246e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.883653164579246e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.9473314506252984e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9473314506252984e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.9473314506252984e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9473314506252984e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9884125611914737e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9884125611914737e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.0882457170869175e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.0882457170869175e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.151295967334473e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.151295967334473e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.846190874604374e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.846190874604374e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.846190874604374e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.846190874604374e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.071403660008786e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (5.071403660008786e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (5.105462414173697e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.105462414173697e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.105462414173697e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.105462414173697e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.146386760521322e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.146386760521322e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.146386760521322e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.146386760521322e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.159294468599809e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.159294468599809e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.159294468599809e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.159294468599809e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.42792663478476e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.42792663478476e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.42792663478476e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.42792663478476e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.9357440118168805e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.9357440118168805e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.9357440118168805e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.9357440118168805e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.2531851495939595e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.2531851495939595e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.979710976119964e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.979710976119964e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.979710976119964e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.979710976119964e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (7.141566358110756e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (7.141566358110756e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (7.141566358110756e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.141566358110756e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00013838603701343236) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.00013838603701343236) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.0001940103060625518) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (0.0001940103060625518) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (0.00024644081058314933) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024644081058314933) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00024644081058314933) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024644081058314933) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192924120528198) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.0005192924120528198) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.0007870893705573733) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.0007870893705573733) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.0014528842553173556) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0014528842553173556) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0014528842553173556) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0014528842553173556) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.001640759116709813) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.001640759116709813) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0017278745823349807) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.0017278745823349807) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.002141348964728975) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (0.002141348964728975) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (0.0032675148971553275) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (0.0032675148971553275) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (0.0033566679213691275) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (0.0033566679213691275) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (0.0034841545793424483) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0034841545793424483) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0038764821957235716) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (0.0038764821957235716) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (0.004668615266021916) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.004668615266021916) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005923799555609087) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923799555609087) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.005923799555609087) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923799555609087) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.008469833341362708) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.008469833341362708) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.008469833341362708) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.008469833341362708) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.00854197565680355) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00854197565680355) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (0.00854197565680355) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (0.00854197565680355) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00854197565680355) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00854197565680355) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00854197565680355) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00854197565680355) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00882638756779817) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.00882638756779817) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.00882638756779817) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.00882638756779817) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.009841802923830786) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.009841802923830786) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.009841802923830786) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.009841802923830786) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.01709162192223535) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.01709162192223535) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.01709162192223535) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.01709162192223535) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.019538085344927073) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.019538085344927073) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.019538085344927073) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.019538085344927073) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.024282031623368935) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (0.024282031623368935) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (0.035608400352257465) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.035608400352257465) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.06752398179967045) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.06752398179967045) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.06752398179967045) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.06752398179967045) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.07635036936746863) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.07635036936746863) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.07635036936746863) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.07635036936746863) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.07165056250058312) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07165056250058312) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0716505625005831) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0716505625005831) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (5.775872095258355e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.775872095258355e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.775872095258355e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.775872095258355e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.05859215179555093) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.05859215179555093) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.019257453001866234) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.019257453001866234) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311472182398845) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311472182398845) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00882638756779817) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.00882638756779817) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.007597461779088092) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.007597461779088092) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.007597461779088092) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.007597461779088092) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733568640015007) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733568640015007) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733568640015007) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733568640015007) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733568640015007) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733568640015007) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733568640015007) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733568640015007) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348047718415534) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005348047718415534) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005348047718415534) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348047718415534) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.004220835998354303) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.004220835998354303) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.004220835998354303) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.004220835998354303) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.003876482195723572) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (-0.003876482195723572) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (-0.003876482195723572) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (-0.003876482195723572) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (-0.0038040631543688765) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0038040631543688765) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002446463422691723) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.002446463422691723) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.0023949671540606844) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0023949671540606844) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0023949671540606844) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0023949671540606844) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0023949671540606844) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0023949671540606844) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0023949671540606844) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0023949671540606844) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0022494140606725585) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0022494140606725585) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0022494140606725585) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0022494140606725585) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.002200956847998133) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.002200956847998133) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.002200956847998133) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.002200956847998133) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638931390730846) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638931390730846) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638931390730846) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638931390730846) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638931390730846) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638931390730846) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0018638931390730846) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638931390730846) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012366559235657496) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.0012366559235657496) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.0012366559235657496) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.0012366559235657496) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.0011726297842382196) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0011726297842382196) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0011726297842382196) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0011726297842382196) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0007870893705573731) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0007870893705573731) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870893705573731) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0007870893705573731) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0005192924120528197) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.0005192924120528197) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.0005192924120528197) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.0005192924120528197) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-1.1462850988594931e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1462850988594931e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.874248561946267e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.874248561946267e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.874248561946267e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.874248561946267e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.3001958859121775e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.3001958859121775e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.3001958859121775e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.3001958859121775e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.044474162943388e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.044474162943388e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.044474162943388e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.044474162943388e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.955903571950885e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.955903571950885e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.955903571950885e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.955903571950885e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105341107946655e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105341107946655e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105341107946655e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105341107946655e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661200331801447e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661200331801447e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661200331801447e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661200331801447e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540204254426245e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540204254426245e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.189870461740873e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.189870461740873e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.876530389315265e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876530389315265e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876530389315265e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876530389315265e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175164890501956e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175164890501956e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.523339375987698e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.523339375987698e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.076662710624261e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.076662710624261e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.076662710624261e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.076662710624261e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.013398845816747e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.013398845816747e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.9045373750076356e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.9045373750076356e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.9045373750076356e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.9045373750076356e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.666679765786999e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.666679765786999e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.666679765786999e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.666679765786999e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505624640037557e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.8505624640037557e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.846699424825232e-08) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-7.846699424825232e-08) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-6.772951822521159e-08) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (-6.772951822521159e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (-4.099829448368553e-08) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-4.099829448368553e-08) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.099829448368553e-08) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.099829448368553e-08) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (6.772951822521159e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (6.772951822521159e-08) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (7.846699424825232e-08) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (7.846699424825232e-08) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (1.6570092805184004e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.6570092805184004e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.6570092805184004e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.6570092805184004e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.8505624640037557e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.8505624640037557e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863216618533024e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863216618533024e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.6863216618533024e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863216618533024e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013398845816747e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.013398845816747e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.523339375987698e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.523339375987698e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.670408126339027e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.670408126339027e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.670408126339027e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.670408126339027e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175164890501956e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175164890501956e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.189870461740873e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.189870461740873e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.540204254426245e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540204254426245e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949307712335334e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949307712335334e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.7924638299332196e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924638299332196e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924638299332196e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.7924638299332196e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.883653164579246e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.883653164579246e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.9884125611914737e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9884125611914737e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9884125611914737e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9884125611914737e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.2531851495939595e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.2531851495939595e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.401691642481972e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.401691642481972e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.401691642481972e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.401691642481972e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809380254752935e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.5809380254752935e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809380254752935e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.5809380254752935e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0010283270637599122) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0010283270637599122) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0010283270637599122) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0010283270637599122) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012223373698224643) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.0012223373698224643) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (0.0012223373698224643) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (0.0012223373698224643) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0012223373698224643) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0012223373698224643) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0012223373698224643) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012223373698224643) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.001640759116709813) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.001640759116709813) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.001640759116709813) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.001640759116709813) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.002141348964728975) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (0.002141348964728975) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (0.002446463422691723) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.002446463422691723) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.0029841800747885533) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.0029841800747885533) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.0029841800747885533) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.0029841800747885533) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.0038040631543688765) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0038040631543688765) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00882638756779817) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00882638756779817) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.010311472182398845) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010311472182398845) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019257453001866234) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019257453001866234) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.3986653569555166e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.3986653569555166e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.3986653569555166e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.3986653569555166e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841545793424483) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841545793424483) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002984180074788553) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.002984180074788553) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.0001940103060625518) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0001940103060625518) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.146285098859493e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.146285098859493e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.7924638299332196e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7924638299332196e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540204254426245e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540204254426245e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540204254426245e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540204254426245e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.8505624640037557e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505624640037557e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505624640037557e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505624640037557e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.846699424825232e-08) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-7.846699424825232e-08) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-7.846699424825232e-08) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-7.846699424825232e-08) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-4.099829448368553e-08) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-4.099829448368553e-08) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.099829448368553e-08) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-4.099829448368553e-08) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013398845816747e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013398845816747e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013398845816747e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013398845816747e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.949307712335334e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.949307712335334e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.7924638299332196e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7924638299332196e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0001940103060625518) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0001940103060625518) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002984180074788553) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.002984180074788553) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.0034841545793424483) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0034841545793424483) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
  (-73.13873149597158) [I0]
+ (-0.18066757507038633) [Z7]
+ (-0.18066757507038628) [Z6]
+ (-0.15961443583758816) [Z4]
+ (-0.15961443583758803) [Z5]
+ (0.17419986612078633) [Z3]
+ (0.1741998661207864) [Z2]
+ (0.2275732881443664) [Z1]
+ (0.22757328814436642) [Z0]
+ (-8.194104951689263e-06) [Y4 Y6]
+ (-8.194104951689263e-06) [X4 X6]
+ (7.954224558247985e-06) [Y5 Y7]
+ (7.954224558247985e-06) [X5 X7]
+ (0.11270381859123513) [Z4 Z6]
+ (0.11270381859123513) [Z5 Z7]
+ (0.11952441016895) [Z0 Z4]
+ (0.11952441016895) [Z1 Z5]
+ (0.13401737372655423) [Z0 Z6]
+ (0.13401737372655423) [Z1 Z7]
+ (0.13734942210157508) [Z0 Z5]
+ (0.13734942210157508) [Z1 Z4]
+ (0.13766859133616852) [Z2 Z4]
+ (0.13766859133616852) [Z3 Z5]
+ (0.14138903590151874) [Z4 Z7]
+ (0.14138903590151874) [Z5 Z6]
+ (0.14722930783261495) [Z2 Z5]
+ (0.14722930783261495) [Z3 Z4]
+ (0.14926347060044154) [Z4 Z5]
+ (0.14973497005433511) [Z2 Z6]
+ (0.14973497005433511) [Z3 Z7]
+ (0.15138342699118112) [Z0 Z7]
+ (0.15138342699118112) [Z1 Z6]
+ (0.15435760065308218) [Z6 Z7]
+ (0.15582280685856847) [Z2 Z7]
+ (0.15582280685856847) [Z3 Z6]
+ (0.16756669356168516) [Z0 Z2]
+ (0.16756669356168516) [Z1 Z3]
+ (0.1814400936293149) [Z0 Z3]
+ (0.1814400936293149) [Z1 Z2]
+ (0.19392574334979157) [Z0 Z1]
+ (0.220039772402686) [Z2 Z3]
+ (-7.038023733047854e-06) [Y4 Z5 Y6]
+ (-7.038023733047854e-06) [X4 Z5 X6]
+ (-7.038023733047854e-06) [Y5 Z6 Y7]
+ (-7.038023733047854e-06) [X5 Z6 X7]
+ (-0.03078744071859541) [Y0 Z2 Z3 Y4]
+ (-0.03078744071859541) [X0 Z2 Z3 X4]
+ (-0.030104525273571246) [Y0 Z1 Z3 Y4]
+ (-0.030104525273571246) [X0 Z1 Z3 X4]
+ (-0.030104525273571246) [Y1 Z2 Z4 Y5]
+ (-0.030104525273571246) [X1 Z2 Z4 X5]
+ (-0.029812299601117885) [Y0 Z1 Z2 Y4]
+ (-0.029812299601117885) [X0 Z1 Z2 X4]
+ (-0.029812299601117885) [Y1 Z3 Z4 Y5]
+ (-0.029812299601117885) [X1 Z3 Z4 X5]
+ (-0.028685217310283596) [Y4 Y5 X6 X7]
+ (-0.028685217310283596) [X4 X5 Y6 Y7]
+ (-0.017825011932625062) [Y0 Y1 X4 X5]
+ (-0.017825011932625062) [X0 X1 Y4 Y5]
+ (-0.017366053264626896) [Y0 Y1 X6 X7]
+ (-0.017366053264626896) [X0 X1 Y6 Y7]
+ (-0.013873400067629755) [Y0 Y1 X2 X3]
+ (-0.013873400067629755) [X0 X1 Y2 Y3]
+ (-0.01130720803003451) [Y1 Z2 Z3 Y5]
+ (-0.01130720803003451) [X1 Z2 Z3 X5]
+ (-0.009560716496446447) [Y2 Y3 X4 X5]
+ (-0.009560716496446447) [X2 X3 Y4 Y5]
+ (-0.006087836804233367) [Y2 Y3 X6 X7]
+ (-0.006087836804233367) [X2 X3 Y6 Y7]
+ (-0.0002922256724533579) [Y1 X2 X3 Y4]
+ (-0.0002922256724533579) [X1 Y2 Y3 X4]
+ (-8.194104951689263e-06) [Z4 Y5 Z6 Y7]
+ (-8.194104951689263e-06) [Z4 X5 Z6 X7]
+ (-2.890929956435393e-06) [Z0 Y5 Z6 Y7]
+ (-2.890929956435393e-06) [Z0 X5 Z6 X7]
+ (-2.890929956435393e-06) [Z1 Y4 Z5 Y6]
+ (-2.890929956435393e-06) [Z1 X4 Z5 X6]
+ (-1.8551374803216572e-06) [Z0 Y4 Z5 Y6]
+ (-1.8551374803216572e-06) [Z0 X4 Z5 X6]
+ (-1.8551374803216572e-06) [Z1 Y5 Z6 Y7]
+ (-1.8551374803216572e-06) [Z1 X5 Z6 X7]
+ (-1.5973397363848495e-06) [Z2 Y4 Z5 Y6]
+ (-1.5973397363848495e-06) [Z2 X4 Z5 X6]
+ (-1.5973397363848495e-06) [Z3 Y5 Z6 Y7]
+ (-1.5973397363848495e-06) [Z3 X5 Z6 X7]
+ (-1.0357924761107001e-06) [Y0 X1 X5 Y6]
+ (-1.0357924761107001e-06) [Y0 Y1 Y5 Y6]
+ (-1.0357924761107001e-06) [X0 X1 X5 X6]
+ (-1.0357924761107001e-06) [X0 Y1 Y5 X6]
+ (-9.344970154853693e-07) [Z2 Y5 Z6 Y7]
+ (-9.344970154853693e-07) [Z2 X5 Z6 X7]
+ (-9.344970154853693e-07) [Z3 Y4 Z5 Y6]
+ (-9.344970154853693e-07) [Z3 X4 Z5 X6]
+ (6.628427208994802e-07) [Y2 X3 X5 Y6]
+ (6.628427208994802e-07) [Y2 Y3 Y5 Y6]
+ (6.628427208994802e-07) [X2 X3 X5 X6]
+ (6.628427208994802e-07) [X2 Y3 Y5 X6]
+ (7.954224558247985e-06) [Y4 Z5 Y6 Z7]
+ (7.954224558247985e-06) [X4 Z5 X6 Z7]
+ (0.0002922256724533579) [Y1 Y2 X3 X4]
+ (0.0002922256724533579) [X1 X2 Y3 Y4]
+ (0.006087836804233367) [Y2 X3 X6 Y7]
+ (0.006087836804233367) [X2 Y3 Y6 X7]
+ (0.009560716496446447) [Y2 X3 X4 Y5]
+ (0.009560716496446447) [X2 Y3 Y4 X5]
+ (0.013873400067629755) [Y0 X1 X2 Y3]
+ (0.013873400067629755) [X0 Y1 Y2 X3]
+ (0.017366053264626896) [Y0 X1 X6 Y7]
+ (0.017366053264626896) [X0 Y1 Y6 X7]
+ (0.017825011932625062) [Y0 X1 X4 Y5]
+ (0.017825011932625062) [X0 Y1 Y4 X5]
+ (0.028685217310283596) [Y4 X5 X6 Y7]
+ (0.028685217310283596) [X4 Y5 Y6 X7]
+ (-0.04375171612133564) [Y0 Z1 Z2 Z3 Y4]
+ (-0.04375171612133564) [X0 Z1 Z2 Z3 X4]
+ (-0.04375171612133563) [Y1 Z2 Z3 Z4 Y5]
+ (-0.04375171612133563) [X1 Z2 Z3 Z4 X5]
+ (-0.03078744071859541) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (-0.03078744071859541) [Z0 X1 Z2 Z3 Z4 X5]
+ (-0.025104907970051406) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (-0.025104907970051406) [X0 Z1 Z2 Z3 X4 Z6]
+ (-0.025104907970051406) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (-0.025104907970051406) [X1 Z2 Z3 Z4 X5 Z7]
+ (-0.01130720803003451) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-0.01130720803003451) [X0 Z1 Z2 Z3 X4 Z5]
+ (-0.010540434329241354) [Y0 Z1 Z2 Z3 Y4 Z7]
+ (-0.010540434329241354) [X0 Z1 Z2 Z3 X4 Z7]
+ (-0.010540434329241354) [Y1 Z2 Z3 Z4 Y5 Z6]
+ (-0.010540434329241354) [X1 Z2 Z3 Z4 X5 Z6]
+ (-0.0002922256724533579) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-0.0002922256724533579) [Y0 Z1 Y2 X3 Z4 X5]
+ (-0.0002922256724533579) [X0 Z1 X2 Y3 Z4 Y5]
+ (-0.0002922256724533579) [X0 Z1 X2 X3 Z4 X5]
+ (-4.1838087766615795e-06) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-4.1838087766615795e-06) [X1 Z2 Z3 Z4 Z5 X7]
+ (-3.313017057851239e-06) [Y1 Z2 Z3 X4 X5 Y6]
+ (-3.313017057851239e-06) [X1 Z2 Z3 Y4 Y5 X6]
+ (-1.0357924761107001e-06) [Y0 Y1 X4 Z5 Z6 X7]
+ (-1.0357924761107001e-06) [X0 X1 Y4 Z5 Z6 Y7]
+ (-6.628427208994802e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (-6.628427208994802e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (-3.328039647226724e-07) [Y1 Y2 X3 Z4 Z5 X6]
+ (-3.328039647226724e-07) [X1 X2 Y3 Z4 Z5 Y6]
+ (3.328039647226724e-07) [Y1 X2 X3 Z4 Z5 Y6]
+ (3.328039647226724e-07) [X1 Y2 Y3 Z4 Z5 X6]
+ (6.628427208994802e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (6.628427208994802e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (1.0357924761107001e-06) [Y0 X1 X4 Z5 Z6 Y7]
+ (1.0357924761107001e-06) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.2111874184300934e-06) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (3.2111874184300934e-06) [X0 Z1 Z2 Z3 Z5 X6]
+ (3.2111874184300934e-06) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (3.2111874184300934e-06) [X1 Z2 Z3 Z4 Z6 X7]
+ (3.2774382575425504e-06) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (3.2774382575425504e-06) [X0 Z1 Z2 Z4 Z5 X6]
+ (3.2774382575425504e-06) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (3.2774382575425504e-06) [X1 Z3 Z4 Z5 Z6 X7]
+ (3.313017057851239e-06) [Y1 Z2 Z3 Y4 X5 X6]
+ (3.313017057851239e-06) [X1 Z2 Z3 X4 Y5 Y6]
+ (3.6102422222652228e-06) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (3.6102422222652228e-06) [X0 Z1 Z3 Z4 Z5 X6]
+ (3.6102422222652228e-06) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (3.6102422222652228e-06) [X1 Z2 Z4 Z5 Z6 X7]
+ (3.7695835817322387e-06) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (3.7695835817322387e-06) [X0 Z2 Z3 Z4 Z5 X6]
+ (6.524204476277863e-06) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (6.524204476277863e-06) [X0 Z1 Z2 Z3 Z4 X6]
+ (6.524204476277863e-06) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (6.524204476277863e-06) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.01456447364081005) [Y1 Z2 Z3 X4 X6 Y7]
+ (0.01456447364081005) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (0.01456447364081005) [X1 Z2 Z3 X4 X6 X7]
+ (0.01456447364081005) [X1 Z2 Z3 Y4 Y6 X7]
+ (5.105681037358153e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (5.105681037358153e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (5.105681037358153e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (5.105681037358153e-06) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.01456447364081005) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (-0.01456447364081005) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (-4.1838087766615795e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-4.1838087766615795e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-3.313017057851239e-06) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (-3.313017057851239e-06) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (-3.313017057851239e-06) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (-3.313017057851239e-06) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (3.328039647226724e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (3.328039647226724e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (3.328039647226724e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (3.328039647226724e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (3.7695835817322387e-06) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (3.7695835817322387e-06) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (0.01456447364081005) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (0.01456447364081005) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/site-packages/pyscf/lib/misc.py:46: H5pyDeprecationWarning: Using default_file_mode other than 'r' is deprecated. Pass the mode to h5py.File() instead.
  h5py.get_config().default_file_mode = 'a'
  (-46.46390678868898) [I0]
+ (0.782966172595019) [Z11]
+ (0.7829661725950191) [Z10]
+ (0.8084581961720485) [Z12]
+ (0.8084581961720487) [Z13]
+ (1.2034402289145631) [Z4]
+ (1.2034402289145634) [Z5]
+ (1.3096862988615423) [Z7]
+ (1.3096862988615425) [Z6]
+ (1.369352563471818) [Z8]
+ (1.369352563471818) [Z9]
+ (1.6538942226831719) [Z3]
+ (1.653894222683172) [Z2]
+ (12.412630742111766) [Z0]
+ (12.412630742111766) [Z1]
+ (-8.194261372461431e-06) [Y10 Y12]
+ (-8.194261372461431e-06) [X10 X12]
+ (-1.8540608578102458e-06) [Y5 Y7]
+ (-1.8540608578102458e-06) [X5 X7]
+ (-7.764994118426849e-07) [Y3 Y5]
+ (-7.764994118426849e-07) [X3 X5]
+ (-5.929765814877662e-07) [Y4 Y6]
+ (-5.929765814877662e-07) [X4 X6]
+ (1.602116740518798e-06) [Y2 Y4]
+ (1.602116740518798e-06) [X2 X4]
+ (7.95441317652263e-06) [Y11 Y13]
+ (7.95441317652263e-06) [X11 X13]
+ (0.0032769719312316916) [Y1 Y3]
+ (0.0032769719312316916) [X1 X3]
+ (0.10433064780651403) [Y0 Y2]
+ (0.10433064780651403) [X0 X2]
+ (0.11270386920332219) [Z10 Z12]
+ (0.11270386920332219) [Z11 Z13]
+ (0.11383573679388656) [Z4 Z12]
+ (0.11383573679388656) [Z5 Z13]
+ (0.11952438964682682) [Z6 Z10]
+ (0.11952438964682682) [Z7 Z11]
+ (0.12489990917237606) [Z4 Z10]
+ (0.12489990917237606) [Z5 Z11]
+ (0.12495807739503226) [Z2 Z4]
+ (0.12495807739503226) [Z3 Z5]
+ (0.1279950249246842) [Z2 Z10]
+ (0.1279950249246842) [Z3 Z11]
+ (0.13401715261963704) [Z6 Z12]
+ (0.13401715261963704) [Z7 Z13]
+ (0.1370119167404076) [Z4 Z6]
+ (0.1370119167404076) [Z5 Z7]
+ (0.1373495306426133) [Z6 Z11]
+ (0.1373495306426133) [Z7 Z10]
+ (0.13739104762683238) [Z2 Z6]
+ (0.13739104762683238) [Z3 Z7]
+ (0.1376687264585258) [Z8 Z10]
+ (0.1376687264585258) [Z9 Z11]
+ (0.1401128986535482) [Z2 Z12]
+ (0.1401128986535482) [Z3 Z13]
+ (0.14138905291942816) [Z10 Z13]
+ (0.14138905291942816) [Z11 Z12]
+ (0.1425799771248576) [Z4 Z11]
+ (0.1425799771248576) [Z5 Z10]
+ (0.1472294321876617) [Z8 Z11]
+ (0.1472294321876617) [Z9 Z10]
+ (0.14899430575065556) [Z4 Z7]
+ (0.14899430575065556) [Z5 Z6]
+ (0.1492635514738891) [Z10 Z11]
+ (0.149607026844453) [Z4 Z8]
+ (0.149607026844453) [Z5 Z9]
+ (0.14973486803496916) [Z8 Z12]
+ (0.14973486803496916) [Z9 Z13]
+ (0.15071408121008298) [Z2 Z8]
+ (0.15071408121008298) [Z3 Z9]
+ (0.15138327161428844) [Z6 Z13]
+ (0.15138327161428844) [Z7 Z12]
+ (0.15215040708869043) [Z4 Z13]
+ (0.15215040708869043) [Z5 Z12]
+ (0.15337968243314162) [Z2 Z11]
+ (0.15337968243314162) [Z3 Z10]
+ (0.15435748657223625) [Z12 Z13]
+ (0.15569010671752465) [Z2 Z13]
+ (0.15569010671752465) [Z3 Z12]
+ (0.15582269051553102) [Z8 Z13]
+ (0.15582269051553102) [Z9 Z12]
+ (0.15676396176430996) [Z4 Z9]
+ (0.15676396176430996) [Z5 Z8]
+ (0.15755314797985662) [Z4 Z5]
+ (0.16079764534838575) [Z2 Z5]
+ (0.16079764534838575) [Z3 Z4]
+ (0.16756653265461272) [Z6 Z8]
+ (0.16756653265461272) [Z7 Z9]
+ (0.16853486561579956) [Z2 Z7]
+ (0.16853486561579956) [Z3 Z6]
+ (0.18143991440303883) [Z6 Z9]
+ (0.18143991440303883) [Z7 Z8]
+ (0.1818908579075139) [Z2 Z3]
+ (0.18690820476912565) [Z2 Z9]
+ (0.18690820476912565) [Z3 Z8]
+ (0.19299723935364216) [Z0 Z10]
+ (0.19299723935364216) [Z1 Z11]
+ (0.19392534613270213) [Z6 Z7]
+ (0.1966177089034214) [Z0 Z4]
+ (0.1966177089034214) [Z1 Z5]
+ (0.19936354537360823) [Z0 Z5]
+ (0.19936354537360823) [Z1 Z4]
+ (0.20072866460441746) [Z0 Z11]
+ (0.20072866460441746) [Z1 Z10]
+ (0.21102659849791483) [Z0 Z12]
+ (0.21102659849791483) [Z1 Z13]
+ (0.21631037498631778) [Z0 Z13]
+ (0.21631037498631778) [Z1 Z12]
+ (0.2200397733437609) [Z8 Z9]
+ (0.23671080783830428) [Z0 Z2]
+ (0.23671080783830428) [Z1 Z3]
+ (0.24164663936017192) [Z0 Z6]
+ (0.24164663936017192) [Z1 Z7]
+ (0.2485348337131425) [Z0 Z7]
+ (0.2485348337131425) [Z1 Z6]
+ (0.251294456745917) [Z0 Z3]
+ (0.251294456745917) [Z1 Z2]
+ (0.2723251830660566) [Z0 Z8]
+ (0.2723251830660566) [Z1 Z9]
+ (0.27883454426723375) [Z0 Z9]
+ (0.27883454426723375) [Z1 Z8]
+ (1.1861763734860482) [Z0 Z1]
+ (-1.226048498815306e-05) [Y4 Z5 Y6]
+ (-1.226048498815306e-05) [X4 Z5 X6]
+ (-1.226048498815306e-05) [Y5 Z6 Y7]
+ (-1.226048498815306e-05) [X5 Z6 X7]
+ (-1.0722312157800475e-05) [Y11 Z12 Y13]
+ (-1.0722312157800475e-05) [X11 Z12 X13]
+ (-1.072231215780047e-05) [Y10 Z11 Y12]
+ (-1.072231215780047e-05) [X10 Z11 X12]
+ (-3.887051671858037e-06) [Y2 Z3 Y4]
+ (-3.887051671858037e-06) [X2 Z3 X4]
+ (-3.887051671858037e-06) [Y3 Z4 Y5]
+ (-3.887051671858037e-06) [X3 Z4 X5]
+ (0.12507032579771982) [Y0 Z1 Y2]
+ (0.12507032579771982) [X0 Z1 X2]
+ (0.12507032579771984) [Y1 Z2 Y3]
+ (0.12507032579771984) [X1 Z2 X3]
+ (-0.03831467029480389) [Y4 Y5 X12 X13]
+ (-0.03831467029480389) [X4 X5 Y12 Y13]
+ (-0.03619412355904267) [Y2 Y3 X8 X9]
+ (-0.03619412355904267) [X2 X3 Y8 Y9]
+ (-0.035839567953353475) [Y2 Y3 X4 X5]
+ (-0.035839567953353475) [X2 X3 Y4 Y5]
+ (-0.03114381798896718) [Y2 Y3 X6 X7]
+ (-0.03114381798896718) [X2 X3 Y6 Y7]
+ (-0.028685183716105987) [Y10 Y11 X12 X13]
+ (-0.028685183716105987) [X10 X11 Y12 Y13]
+ (-0.025996177598021218) [Y3 Z4 Z5 Y7]
+ (-0.025996177598021218) [X3 Z4 Z5 X7]
+ (-0.025384657508457413) [Y2 Y3 X10 X11]
+ (-0.025384657508457413) [X2 X3 Y10 Y11]
+ (-0.019028242443847272) [Y3 Y4 X11 X12]
+ (-0.019028242443847272) [X3 X4 Y11 Y12]
+ (-0.017825140995786456) [Y6 Y7 X10 X11]
+ (-0.017825140995786456) [X6 X7 Y10 Y11]
+ (-0.017680067952481508) [Y4 Y5 X10 X11]
+ (-0.017680067952481508) [X4 X5 Y10 Y11]
+ (-0.017366118994651392) [Y6 Y7 X12 X13]
+ (-0.017366118994651392) [X6 X7 Y12 Y13]
+ (-0.015577208063976455) [Y2 Y3 X12 X13]
+ (-0.015577208063976455) [X2 X3 Y12 Y13]
+ (-0.01458364890761268) [Y0 Y1 X2 X3]
+ (-0.01458364890761268) [X0 X1 Y2 Y3]
+ (-0.013873381748426103) [Y6 Y7 X8 X9]
+ (-0.013873381748426103) [X6 X7 Y8 Y9]
+ (-0.011982389010247953) [Y4 Y5 X6 X7]
+ (-0.011982389010247953) [X4 X5 Y6 Y7]
+ (-0.011285190200840907) [Y5 X6 X11 Y12]
+ (-0.011285190200840907) [X5 Y6 Y11 X12]
+ (-0.009560705729135937) [Y8 Y9 X10 X11]
+ (-0.009560705729135937) [X8 X9 Y10 Y11]
+ (-0.00812525192138102) [Y1 X2 X8 Y9]
+ (-0.00812525192138102) [Y1 Y2 Y8 Y9]
+ (-0.00812525192138102) [X1 X2 X8 X9]
+ (-0.00812525192138102) [X1 Y2 Y8 X9]
+ (-0.007731425250775277) [Y0 Y1 X10 X11]
+ (-0.007731425250775277) [X0 X1 Y10 Y11]
+ (-0.0071569349198569564) [Y4 Y5 X8 X9]
+ (-0.0071569349198569564) [X4 X5 Y8 Y9]
+ (-0.0068881943529705576) [Y0 Y1 X6 X7]
+ (-0.0068881943529705576) [X0 X1 Y6 Y7]
+ (-0.006509361201177225) [Y0 Y1 X8 X9]
+ (-0.006509361201177225) [X0 X1 Y8 Y9]
+ (-0.006087822480561848) [Y8 Y9 X12 X13]
+ (-0.006087822480561848) [X8 X9 Y12 Y13]
+ (-0.005283776488402942) [Y0 Y1 X12 X13]
+ (-0.005283776488402942) [X0 X1 Y12 Y13]
+ (-0.005143391768825084) [Y3 X4 X5 Y6]
+ (-0.005143391768825084) [X3 Y4 Y5 X6]
+ (-0.004684903388155206) [Y1 X2 X6 Y7]
+ (-0.004684903388155206) [Y1 Y2 Y6 Y7]
+ (-0.004684903388155206) [X1 X2 X6 X7]
+ (-0.004684903388155206) [X1 Y2 Y6 X7]
+ (-0.0045750076266391935) [Y1 X2 X12 Y13]
+ (-0.0045750076266391935) [Y1 Y2 Y12 Y13]
+ (-0.0045750076266391935) [X1 X2 X12 X13]
+ (-0.0045750076266391935) [X1 Y2 Y12 X13]
+ (-0.004424855449441857) [Y1 X2 X4 Y5]
+ (-0.004424855449441857) [Y1 Y2 Y4 Y5]
+ (-0.004424855449441857) [X1 X2 X4 X5]
+ (-0.004424855449441857) [X1 Y2 Y4 X5]
+ (-0.003479511890334345) [Y2 Z3 Z5 Y6]
+ (-0.003479511890334345) [X2 Z3 Z5 X6]
+ (-0.003479511890334345) [Y3 Z4 Z6 Y7]
+ (-0.003479511890334345) [X3 Z4 Z6 X7]
+ (-0.0027458364701868116) [Y0 Y1 X4 X5]
+ (-0.0027458364701868116) [X0 X1 Y4 Y5]
+ (-0.001799219493663007) [Y1 X2 X10 Y11]
+ (-0.001799219493663007) [Y1 Y2 Y10 Y11]
+ (-0.001799219493663007) [X1 X2 X10 X11]
+ (-0.001799219493663007) [X1 Y2 Y10 X11]
+ (-0.0002921986261110529) [Y7 Y8 X9 X10]
+ (-0.0002921986261110529) [X7 X8 Y9 Y10]
+ (-8.194261372461431e-06) [Z10 Y11 Z12 Y13]
+ (-8.194261372461431e-06) [Z10 X11 Z12 X13]
+ (-7.801707500839153e-06) [Y2 Z3 Y4 Z11]
+ (-7.801707500839153e-06) [X2 Z3 X4 Z11]
+ (-7.801707500839153e-06) [Y3 Z4 Y5 Z10]
+ (-7.801707500839153e-06) [X3 Z4 X5 Z10]
+ (-4.6430510686919484e-06) [Y3 X4 X10 Y11]
+ (-4.6430510686919484e-06) [Y3 Y4 Y10 Y11]
+ (-4.6430510686919484e-06) [X3 X4 X10 X11]
+ (-4.6430510686919484e-06) [X3 Y4 Y10 X11]
+ (-4.588855155801888e-06) [Y4 Z5 Y6 Z13]
+ (-4.588855155801888e-06) [X4 Z5 X6 Z13]
+ (-4.588855155801888e-06) [Y5 Z6 Y7 Z12]
+ (-4.588855155801888e-06) [X5 Z6 X7 Z12]
+ (-4.556569218414335e-06) [Y5 X6 X12 Y13]
+ (-4.556569218414335e-06) [Y5 Y6 Y12 Y13]
+ (-4.556569218414335e-06) [X5 X6 X12 X13]
+ (-4.556569218414335e-06) [X5 Y6 Y12 X13]
+ (-3.694513294725133e-06) [Y4 X5 X11 Y12]
+ (-3.694513294725133e-06) [Y4 Y5 Y11 Y12]
+ (-3.694513294725133e-06) [X4 X5 X11 X12]
+ (-3.694513294725133e-06) [X4 Y5 Y11 X12]
+ (-3.3440815561904226e-06) [Z0 Y5 Z6 Y7]
+ (-3.3440815561904226e-06) [Z0 X5 Z6 X7]
+ (-3.3440815561904226e-06) [Z1 Y4 Z5 Y6]
+ (-3.3440815561904226e-06) [Z1 X4 Z5 X6]
+ (-3.1586564321472044e-06) [Y2 Z3 Y4 Z10]
+ (-3.1586564321472044e-06) [X2 Z3 X4 Z10]
+ (-3.1586564321472044e-06) [Y3 Z4 Y5 Z11]
+ (-3.1586564321472044e-06) [X3 Z4 X5 Z11]
+ (-3.0993492433408928e-06) [Z0 Y4 Z5 Y6]
+ (-3.0993492433408928e-06) [Z0 X4 Z5 X6]
+ (-3.0993492433408928e-06) [Z1 Y5 Z6 Y7]
+ (-3.0993492433408928e-06) [Z1 X5 Z6 X7]
+ (-2.890967881646108e-06) [Z6 Y11 Z12 Y13]
+ (-2.890967881646108e-06) [Z6 X11 Z12 X13]
+ (-2.890967881646108e-06) [Z7 Y10 Z11 Y12]
+ (-2.890967881646108e-06) [Z7 X10 Z11 X12]
+ (-2.1776646051346894e-06) [Z0 Y10 Z11 Y12]
+ (-2.1776646051346894e-06) [Z0 X10 Z11 X12]
+ (-2.1776646051346894e-06) [Z1 Y11 Z12 Y13]
+ (-2.1776646051346894e-06) [Z1 X11 Z12 X13]
+ (-1.881850183060689e-06) [Y4 Z5 Y6 Z9]
+ (-1.881850183060689e-06) [X4 Z5 X6 Z9]
+ (-1.881850183060689e-06) [Y5 Z6 Y7 Z8]
+ (-1.881850183060689e-06) [X5 Z6 X7 Z8]
+ (-1.8551201216607076e-06) [Z6 Y10 Z11 Y12]
+ (-1.8551201216607076e-06) [Z6 X10 Z11 X12]
+ (-1.8551201216607076e-06) [Z7 Y11 Z12 Y13]
+ (-1.8551201216607076e-06) [Z7 X11 Z12 X13]
+ (-1.8540608578102458e-06) [Y4 Z5 Y6 Z7]
+ (-1.8540608578102458e-06) [X4 Z5 X6 Z7]
+ (-1.8163031698641835e-06) [Z4 Y11 Z12 Y13]
+ (-1.8163031698641835e-06) [Z4 X11 Z12 X13]
+ (-1.8163031698641835e-06) [Z5 Y10 Z11 Y12]
+ (-1.8163031698641835e-06) [Z5 X10 Z11 X12]
+ (-1.6923978286395754e-06) [Y4 Z5 Y6 Z10]
+ (-1.6923978286395754e-06) [X4 Z5 X6 Z10]
+ (-1.6923978286395754e-06) [Y5 Z6 Y7 Z11]
+ (-1.6923978286395754e-06) [X5 Z6 X7 Z11]
+ (-1.6148794139861028e-06) [Z0 Y11 Z12 Y13]
+ (-1.6148794139861028e-06) [Z0 X11 Z12 X13]
+ (-1.6148794139861028e-06) [Z1 Y10 Z11 Y12]
+ (-1.6148794139861028e-06) [Z1 X10 Z11 X12]
+ (-1.597317197897768e-06) [Z8 Y10 Z11 Y12]
+ (-1.597317197897768e-06) [Z8 X10 Z11 X12]
+ (-1.597317197897768e-06) [Z9 Y11 Z12 Y13]
+ (-1.597317197897768e-06) [Z9 X11 Z12 X13]
+ (-1.4548424488661097e-06) [Y3 X4 X6 Y7]
+ (-1.4548424488661097e-06) [Y3 Y4 Y6 Y7]
+ (-1.4548424488661097e-06) [X3 X4 X6 X7]
+ (-1.4548424488661097e-06) [X3 Y4 Y6 X7]
+ (-1.398044908026844e-06) [Y4 Z5 Y6 Z8]
+ (-1.398044908026844e-06) [X4 Z5 X6 Z8]
+ (-1.398044908026844e-06) [Y5 Z6 Y7 Z9]
+ (-1.398044908026844e-06) [X5 Z6 X7 Z9]
+ (-1.1954890097570087e-06) [Y2 Z3 Y4 Z7]
+ (-1.1954890097570087e-06) [X2 Z3 X4 Z7]
+ (-1.1954890097570087e-06) [Y3 Z4 Y5 Z6]
+ (-1.1954890097570087e-06) [X3 Z4 X5 Z6]
+ (-1.1908508080898373e-06) [Z0 Y3 Z4 Y5]
+ (-1.1908508080898373e-06) [Z0 X3 Z4 X5]
+ (-1.1908508080898373e-06) [Z1 Y2 Z3 Y4]
+ (-1.1908508080898373e-06) [Z1 X2 Z3 X4]
+ (-1.1708301369593808e-06) [Z2 Y5 Z6 Y7]
+ (-1.1708301369593808e-06) [Z2 X5 Z6 X7]
+ (-1.1708301369593808e-06) [Z3 Y4 Z5 Y6]
+ (-1.1708301369593808e-06) [Z3 X4 Z5 X6]
+ (-1.0632283424711112e-06) [Z2 Y10 Z11 Y12]
+ (-1.0632283424711112e-06) [Z2 X10 Z11 X12]
+ (-1.0632283424711112e-06) [Z3 Y11 Z12 Y13]
+ (-1.0632283424711112e-06) [Z3 X11 Z12 X13]
+ (-1.0358477599854005e-06) [Y6 X7 X11 Y12]
+ (-1.0358477599854005e-06) [Y6 Y7 Y11 Y12]
+ (-1.0358477599854005e-06) [X6 X7 X11 X12]
+ (-1.0358477599854005e-06) [X6 Y7 Y11 X12]
+ (-9.509249750884897e-07) [Z2 Y4 Z5 Y6]
+ (-9.509249750884897e-07) [Z2 X4 Z5 X6]
+ (-9.509249750884897e-07) [Z3 Y5 Z6 Y7]
+ (-9.509249750884897e-07) [Z3 X5 Z6 X7]
+ (-9.344557777152263e-07) [Z8 Y11 Z12 Y13]
+ (-9.344557777152263e-07) [Z8 X11 Z12 X13]
+ (-9.344557777152263e-07) [Z9 Y10 Z11 Y12]
+ (-9.344557777152263e-07) [Z9 X10 Z11 X12]
+ (-8.337746752347712e-07) [Z0 Y2 Z3 Y4]
+ (-8.337746752347712e-07) [Z0 X2 Z3 X4]
+ (-8.337746752347712e-07) [Z1 Y3 Z4 Y5]
+ (-8.337746752347712e-07) [Z1 X3 Z4 X5]
+ (-7.956895371345524e-07) [Y3 X4 X8 Y9]
+ (-7.956895371345524e-07) [Y3 Y4 Y8 Y9]
+ (-7.956895371345524e-07) [X3 X4 X8 X9]
+ (-7.956895371345524e-07) [X3 Y4 Y8 X9]
+ (-7.764994118426849e-07) [Y2 Z3 Y4 Z5]
+ (-7.764994118426849e-07) [X2 Z3 X4 Z5]
+ (-5.929765814877662e-07) [Z4 Y5 Z6 Y7]
+ (-5.929765814877662e-07) [Z4 X5 Z6 X7]
+ (-5.770052993727889e-07) [Y2 Z3 Y4 Z9]
+ (-5.770052993727889e-07) [X2 Z3 X4 Z9]
+ (-5.770052993727889e-07) [Y3 Z4 Y5 Z8]
+ (-5.770052993727889e-07) [X3 Z4 X5 Z8]
+ (-5.47164774480149e-07) [Y1 Y2 X11 X12]
+ (-5.47164774480149e-07) [X1 X2 Y11 Y12]
+ (-4.83805275033845e-07) [Y5 X6 X8 Y9]
+ (-4.83805275033845e-07) [Y5 Y6 Y8 Y9]
+ (-4.83805275033845e-07) [X5 X6 X8 X9]
+ (-4.83805275033845e-07) [X5 Y6 Y8 X9]
+ (-3.5707613285506617e-07) [Y0 X1 X3 Y4]
+ (-3.5707613285506617e-07) [Y0 Y1 Y3 Y4]
+ (-3.5707613285506617e-07) [X0 X1 X3 X4]
+ (-3.5707613285506617e-07) [X0 Y1 Y3 X4]
+ (-2.447323128495302e-07) [Y0 X1 X5 Y6]
+ (-2.447323128495302e-07) [Y0 Y1 Y5 Y6]
+ (-2.447323128495302e-07) [X0 X1 X5 X6]
+ (-2.447323128495302e-07) [X0 Y1 Y5 X6]
+ (-2.199051618708911e-07) [Y2 X3 X5 Y6]
+ (-2.199051618708911e-07) [Y2 Y3 Y5 Y6]
+ (-2.199051618708911e-07) [X2 X3 X5 X6]
+ (-2.199051618708911e-07) [X2 Y3 Y5 X6]
+ (-1.933241276813248e-07) [Y1 X2 X3 Y4]
+ (-1.933241276813248e-07) [X1 Y2 Y3 X4]
+ (-1.2919694861892434e-07) [Y1 Z2 Z3 Y5]
+ (-1.2919694861892434e-07) [X1 Z2 Z3 X5]
+ (1.7379332621248083e-07) [Y0 Z1 Z3 Y4]
+ (1.7379332621248083e-07) [X0 Z1 Z3 X4]
+ (1.7379332621248083e-07) [Y1 Z2 Z4 Y5]
+ (1.7379332621248083e-07) [X1 Z2 Z4 X5]
+ (1.933241276813248e-07) [Y1 Y2 X3 X4]
+ (1.933241276813248e-07) [X1 X2 Y3 Y4]
+ (2.1868423776176342e-07) [Y2 Z3 Y4 Z8]
+ (2.1868423776176342e-07) [X2 Z3 X4 Z8]
+ (2.1868423776176342e-07) [Y3 Z4 Y5 Z9]
+ (2.1868423776176342e-07) [X3 Z4 X5 Z9]
+ (2.59353439109101e-07) [Y2 Z3 Y4 Z6]
+ (2.59353439109101e-07) [X2 Z3 X4 Z6]
+ (2.59353439109101e-07) [Y3 Z4 Y5 Z7]
+ (2.59353439109101e-07) [X3 Z4 X5 Z7]
+ (3.6060718673753785e-07) [Y0 Z1 Z2 Y4]
+ (3.6060718673753785e-07) [X0 Z1 Z2 X4]
+ (3.6060718673753785e-07) [Y1 Z3 Z4 Y5]
+ (3.6060718673753785e-07) [X1 Z3 Z4 X5]
+ (5.47164774480149e-07) [Y1 X2 X11 Y12]
+ (5.47164774480149e-07) [X1 Y2 Y11 X12]
+ (5.627851911485864e-07) [Y0 X1 X11 Y12]
+ (5.627851911485864e-07) [Y0 Y1 Y11 Y12]
+ (5.627851911485864e-07) [X0 X1 X11 X12]
+ (5.627851911485864e-07) [X0 Y1 Y11 X12]
+ (6.628614201825418e-07) [Y8 X9 X11 Y12]
+ (6.628614201825418e-07) [Y8 Y9 Y11 Y12]
+ (6.628614201825418e-07) [X8 X9 X11 X12]
+ (6.628614201825418e-07) [X8 Y9 Y11 X12]
+ (1.1094407590226014e-06) [Z2 Y11 Z12 Y13]
+ (1.1094407590226014e-06) [Z2 X11 Z12 X13]
+ (1.1094407590226014e-06) [Z3 Y10 Z11 Y12]
+ (1.1094407590226014e-06) [Z3 X10 Z11 X12]
+ (1.602116740518798e-06) [Z2 Y3 Z4 Y5]
+ (1.602116740518798e-06) [Z2 X3 Z4 X5]
+ (1.8782101248609497e-06) [Z4 Y10 Z11 Y12]
+ (1.8782101248609497e-06) [Z4 X10 Z11 X12]
+ (1.8782101248609497e-06) [Z5 Y11 Z12 Y13]
+ (1.8782101248609497e-06) [Z5 X11 Z12 X13]
+ (2.1726691014937126e-06) [Y2 X3 X11 Y12]
+ (2.1726691014937126e-06) [Y2 Y3 Y11 Y12]
+ (2.1726691014937126e-06) [X2 X3 X11 X12]
+ (2.1726691014937126e-06) [X2 Y3 Y11 X12]
+ (3.1174479455876873e-06) [Y0 Z2 Z3 Y4]
+ (3.1174479455876873e-06) [X0 Z2 Z3 X4]
+ (3.5390541846599e-06) [Y2 Z3 Y4 Z12]
+ (3.5390541846599e-06) [X2 Z3 X4 Z12]
+ (3.5390541846599e-06) [Y3 Z4 Y5 Z13]
+ (3.5390541846599e-06) [X3 Z4 X5 Z13]
+ (4.281913884938721e-06) [Y4 Z5 Y6 Z11]
+ (4.281913884938721e-06) [X4 Z5 X6 Z11]
+ (4.281913884938721e-06) [Y5 Z6 Y7 Z10]
+ (4.281913884938721e-06) [X5 Z6 X7 Z10]
+ (5.275883122262065e-06) [Y3 X4 X12 Y13]
+ (5.275883122262065e-06) [Y3 Y4 Y12 Y13]
+ (5.275883122262065e-06) [X3 X4 X12 X13]
+ (5.275883122262065e-06) [X3 Y4 Y12 X13]
+ (5.974311713578297e-06) [Y5 X6 X10 Y11]
+ (5.974311713578297e-06) [Y5 Y6 Y10 Y11]
+ (5.974311713578297e-06) [X5 X6 X10 X11]
+ (5.974311713578297e-06) [X5 Y6 Y10 X11]
+ (7.95441317652263e-06) [Y10 Z11 Y12 Z13]
+ (7.95441317652263e-06) [X10 Z11 X12 Z13]
+ (8.814937306921965e-06) [Y2 Z3 Y4 Z13]
+ (8.814937306921965e-06) [X2 Z3 X4 Z13]
+ (8.814937306921965e-06) [Y3 Z4 Y5 Z12]
+ (8.814937306921965e-06) [X3 Z4 X5 Z12]
+ (0.0002921986261110529) [Y7 X8 X9 Y10]
+ (0.0002921986261110529) [X7 Y8 Y9 X10]
+ (0.0004956762314916195) [Y2 Z4 Z5 Y6]
+ (0.0004956762314916195) [X2 Z4 Z5 X6]
+ (0.0011059037691897105) [Y0 Z1 Y2 Z5]
+ (0.0011059037691897105) [X0 Z1 X2 Z5]
+ (0.0011059037691897105) [Y1 Z2 Y3 Z4]
+ (0.0011059037691897105) [X1 Z2 X3 Z4]
+ (0.001663879878490739) [Y2 Z3 Z4 Y6]
+ (0.001663879878490739) [X2 Z3 Z4 X6]
+ (0.001663879878490739) [Y3 Z5 Z6 Y7]
+ (0.001663879878490739) [X3 Z5 Z6 X7]
+ (0.0017560707018412613) [Y0 Z1 Y2 Z11]
+ (0.0017560707018412613) [X0 Z1 X2 Z11]
+ (0.0017560707018412613) [Y1 Z2 Y3 Z10]
+ (0.0017560707018412613) [X1 Z2 X3 Z10]
+ (0.002326230623158095) [Y0 Z1 Y2 Z13]
+ (0.002326230623158095) [X0 Z1 X2 Z13]
+ (0.002326230623158095) [Y1 Z2 Y3 Z12]
+ (0.002326230623158095) [X1 Z2 X3 Z12]
+ (0.0027458364701868116) [Y0 X1 X4 Y5]
+ (0.0027458364701868116) [X0 Y1 Y4 X5]
+ (0.002929768674751088) [Y0 Z1 Y2 Z9]
+ (0.002929768674751088) [X0 Z1 X2 Z9]
+ (0.002929768674751088) [Y1 Z2 Y3 Z8]
+ (0.002929768674751088) [X1 Z2 X3 Z8]
+ (0.003276971931231692) [Y0 Z1 Y2 Z3]
+ (0.003276971931231692) [X0 Z1 X2 Z3]
+ (0.0033476175306662095) [Y0 Z1 Y2 Z7]
+ (0.0033476175306662095) [X0 Z1 X2 Z7]
+ (0.0033476175306662095) [Y1 Z2 Y3 Z6]
+ (0.0033476175306662095) [X1 Z2 X3 Z6]
+ (0.003555290195504269) [Y0 Z1 Y2 Z10]
+ (0.003555290195504269) [X0 Z1 X2 Z10]
+ (0.003555290195504269) [Y1 Z2 Y3 Z11]
+ (0.003555290195504269) [X1 Z2 X3 Z11]
+ (0.005143391768825084) [Y3 Y4 X5 X6]
+ (0.005143391768825084) [X3 X4 Y5 Y6]
+ (0.005283776488402942) [Y0 X1 X12 Y13]
+ (0.005283776488402942) [X0 Y1 Y12 X13]
+ (0.005530759218631567) [Y0 Z1 Y2 Z4]
+ (0.005530759218631567) [X0 Z1 X2 Z4]
+ (0.005530759218631567) [Y1 Z2 Y3 Z5]
+ (0.005530759218631567) [X1 Z2 X3 Z5]
+ (0.006087822480561848) [Y8 X9 X12 Y13]
+ (0.006087822480561848) [X8 Y9 Y12 X13]
+ (0.006509361201177225) [Y0 X1 X8 Y9]
+ (0.006509361201177225) [X0 Y1 Y8 X9]
+ (0.0068881943529705576) [Y0 X1 X6 Y7]
+ (0.0068881943529705576) [X0 Y1 Y6 X7]
+ (0.006901238249797289) [Y0 Z1 Y2 Z12]
+ (0.006901238249797289) [X0 Z1 X2 Z12]
+ (0.006901238249797289) [Y1 Z2 Y3 Z13]
+ (0.006901238249797289) [X1 Z2 X3 Z13]
+ (0.0071569349198569564) [Y4 X5 X8 Y9]
+ (0.0071569349198569564) [X4 Y5 Y8 X9]
+ (0.007731425250775277) [Y0 X1 X10 Y11]
+ (0.007731425250775277) [X0 Y1 Y10 X11]
+ (0.008032520918821416) [Y0 Z1 Y2 Z6]
+ (0.008032520918821416) [X0 Z1 X2 Z6]
+ (0.008032520918821416) [Y1 Z2 Y3 Z7]
+ (0.008032520918821416) [X1 Z2 X3 Z7]
+ (0.009560705729135937) [Y8 X9 X10 Y11]
+ (0.009560705729135937) [X8 Y9 Y10 X11]
+ (0.011055020596132108) [Y0 Z1 Y2 Z8]
+ (0.011055020596132108) [X0 Z1 X2 Z8]
+ (0.011055020596132108) [Y1 Z2 Y3 Z9]
+ (0.011055020596132108) [X1 Z2 X3 Z9]
+ (0.011285190200840907) [Y5 Y6 X11 X12]
+ (0.011285190200840907) [X5 X6 Y11 Y12]
+ (0.01130727400884813) [Y7 Z8 Z9 Y11]
+ (0.01130727400884813) [X7 Z8 Z9 X11]
+ (0.011982389010247953) [Y4 X5 X6 Y7]
+ (0.011982389010247953) [X4 Y5 Y6 X7]
+ (0.013873381748426103) [Y6 X7 X8 Y9]
+ (0.013873381748426103) [X6 Y7 Y8 X9]
+ (0.01458364890761268) [Y0 X1 X2 Y3]
+ (0.01458364890761268) [X0 Y1 Y2 X3]
+ (0.015577208063976455) [Y2 X3 X12 Y13]
+ (0.015577208063976455) [X2 Y3 Y12 X13]
+ (0.017366118994651392) [Y6 X7 X12 Y13]
+ (0.017366118994651392) [X6 Y7 Y12 X13]
+ (0.017680067952481508) [Y4 X5 X10 Y11]
+ (0.017680067952481508) [X4 Y5 Y10 X11]
+ (0.017825140995786456) [Y6 X7 X10 Y11]
+ (0.017825140995786456) [X6 Y7 Y10 X11]
+ (0.019028242443847272) [Y3 X4 X11 Y12]
+ (0.019028242443847272) [X3 Y4 Y11 X12]
+ (0.025384657508457413) [Y2 X3 X10 Y11]
+ (0.025384657508457413) [X2 Y3 Y10 X11]
+ (0.028685183716105987) [Y10 X11 X12 Y13]
+ (0.028685183716105987) [X10 Y11 Y12 X13]
+ (0.029812424517345767) [Y6 Z7 Z8 Y10]
+ (0.029812424517345767) [X6 Z7 Z8 X10]
+ (0.029812424517345767) [Y7 Z9 Z10 Y11]
+ (0.029812424517345767) [X7 Z9 Z10 X11]
+ (0.030104623143456813) [Y6 Z7 Z9 Y10]
+ (0.030104623143456813) [X6 Z7 Z9 X10]
+ (0.030104623143456813) [Y7 Z8 Z10 Y11]
+ (0.030104623143456813) [X7 Z8 Z10 X11]
+ (0.030787505389143918) [Y6 Z8 Z9 Y10]
+ (0.030787505389143918) [X6 Z8 Z9 X10]
+ (0.03114381798896718) [Y2 X3 X6 Y7]
+ (0.03114381798896718) [X2 Y3 Y6 X7]
+ (0.035839567953353475) [Y2 X3 X4 Y5]
+ (0.035839567953353475) [X2 Y3 Y4 X5]
+ (0.03619412355904267) [Y2 X3 X8 Y9]
+ (0.03619412355904267) [X2 Y3 Y8 X9]
+ (0.03831467029480389) [Y4 X5 X12 Y13]
+ (0.03831467029480389) [X4 Y5 Y12 X13]
+ (0.10433064780651403) [Z0 Y1 Z2 Y3]
+ (0.10433064780651403) [Z0 X1 Z2 X3]
+ (-0.12133276911042361) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133276911042361) [X2 Z3 Z4 Z5 X6]
+ (-0.12133276911042355) [Y3 Z4 Z5 Z6 Y7]
+ (-0.12133276911042355) [X3 Z4 Z5 Z6 X7]
+ (3.20207687930842e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.20207687930842e-06) [X0 Z1 Z2 Z3 X4]
+ (3.20207687930842e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.20207687930842e-06) [X1 Z2 Z3 Z4 X5]
+ (0.2284810656491881) [Y6 Z7 Z8 Z9 Y10]
+ (0.2284810656491881) [X6 Z7 Z8 Z9 X10]
+ (0.2284810656491882) [Y7 Z8 Z9 Z10 Y11]
+ (0.2284810656491882) [X7 Z8 Z9 Z10 X11]
+ (-0.03276765782329046) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.03276765782329046) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.03276765782329046) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.03276765782329046) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.02711503684527312) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.02711503684527312) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.02711503684527312) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.02711503684527312) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.025996177598021218) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.025996177598021218) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.017561202409646162) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.017561202409646162) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.017561202409646162) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.017561202409646162) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.014564531231173034) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.014564531231173034) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.014564531231173034) [X7 Z8 Z9 X10 X12 X13]
+ (-0.014564531231173034) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.01221504099761396) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.01221504099761396) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.01221504099761396) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.01221504099761396) [X4 Z5 X6 X11 Z12 X13]
+ (-0.01221504099761396) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.01221504099761396) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.01221504099761396) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.01221504099761396) [X5 Z6 X7 X10 Z11 X12]
+ (-0.011756013419819245) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.011756013419819245) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.011756013419819245) [X3 Z4 Z5 X6 X8 X9]
+ (-0.011756013419819245) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.008764827575688767) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.008764827575688767) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.008764827575688767) [X2 Z3 Z4 X5 X11 X12]
+ (-0.008764827575688767) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.008764827575688767) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.008764827575688767) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.008764827575688767) [X3 X4 X10 Z11 Z12 X13]
+ (-0.008764827575688767) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.00812525192138102) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.00812525192138102) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.007306759928832951) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.007306759928832951) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.007306759928832951) [X4 X5 X7 Z8 Z9 X10]
+ (-0.007306759928832951) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.005805188989826918) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.005805188989826918) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.005805188989826918) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.005805188989826918) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.005652620978017345) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.005652620978017345) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.005652620978017345) [X0 X1 X3 Z4 Z5 X6]
+ (-0.005652620978017345) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.005143391768825083) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.005143391768825083) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.005143391768825083) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.005143391768825083) [X2 Z3 X4 X5 Z6 X7]
+ (-0.004684903388155206) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.004684903388155206) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668620318776296) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.004668620318776296) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.004575007626639194) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.004575007626639194) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.004424855449441857) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.004424855449441857) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.0041587973818400445) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.0041587973818400445) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.0041587973818400445) [X3 Z4 Z5 X6 X12 X13]
+ (-0.0041587973818400445) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.0034937903598901083) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.0034937903598901083) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.0034937903598901083) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.0034937903598901083) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.0027790267990255337) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.0027790267990255337) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.0022939566113524676) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.0022939566113524676) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.001799219493663007) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.001799219493663007) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.001727875394136968) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.001727875394136968) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.0009298507967730497) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298507967730497) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298507967730497) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298507967730497) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533856254125488) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.0008533856254125488) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.0008145313270956629) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.0008145313270956629) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.0008145313270956629) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.0008145313270956629) [X3 Z4 Z5 Z6 X7 Z11]
+ (-7.735036880590775e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.735036880590775e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.735036880590775e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.735036880590775e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-8.774817864928147e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.774817864928147e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.774817864928147e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.774817864928147e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.518362215949595e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.518362215949595e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.518362215949595e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.518362215949595e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.44434467625434e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.44434467625434e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.44434467625434e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.44434467625434e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.5243738488181616e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.5243738488181616e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.5243738488181616e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.5243738488181616e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-6.290028433610009e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.290028433610009e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.290028433610009e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.290028433610009e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.974311713578297e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (-5.974311713578297e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (-5.275883122262065e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.275883122262065e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.6430510686919484e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.6430510686919484e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.556569218414335e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (-4.556569218414335e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (-4.2532242256472334e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.2532242256472334e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-3.7696594523449825e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.7696594523449825e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.694513294725133e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.694513294725133e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.61029713093642e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.61029713093642e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.61029713093642e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.61029713093642e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.313145500142764e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.313145500142764e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.2774831959037423e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.2774831959037423e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.2774831959037423e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.2774831959037423e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.2112283486753976e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.2112283486753976e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.2112283486753976e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.2112283486753976e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.1513463112865705e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.1513463112865705e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.0882507114944743e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.0882507114944743e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.1726691014937126e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.1726691014937126e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.4548424488661097e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.4548424488661097e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.330473188673807e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.330473188673807e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.2283337823395862e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.2283337823395862e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.0358477599854005e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.0358477599854005e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-7.956895371345524e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.956895371345524e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.733197743057746e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.733197743057746e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.733197743057746e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.733197743057746e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.628614201825418e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.628614201825418e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.556281914731657e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.556281914731657e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.556281914731657e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.556281914731657e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.418291574752551e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.418291574752551e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.418291574752551e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.418291574752551e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.927453083398625e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.927453083398625e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.927453083398625e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.927453083398625e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.627851911485864e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.627851911485864e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.287660624841173e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.287660624841173e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.287660624841173e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.287660624841173e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.287660624841173e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.287660624841173e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.287660624841173e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.287660624841173e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.83805275033845e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (-4.83805275033845e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (-3.570761328550662e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.570761328550662e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.3281393503267753e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.3281393503267753e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.0868265649047007e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.0868265649047007e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.0868265649047007e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.0868265649047007e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.447323128495302e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (-2.447323128495302e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (-2.371328947576226e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.371328947576226e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.371328947576226e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.371328947576226e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.199051618708911e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-2.199051618708911e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-1.933241276813248e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.933241276813248e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.933241276813248e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.933241276813248e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.839420915204479e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.839420915204479e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.839420915204479e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.839420915204479e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.5510539175085593e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (-1.5510539175085593e-07) [X0 Z1 X2 X4 Z5 X6]
+ (-1.5510539175085593e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (-1.5510539175085593e-07) [X1 Z2 X3 X5 Z6 X7]
+ (-1.3807781478685866e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (-1.3807781478685866e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (-1.3807781478685866e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (-1.3807781478685866e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (-1.3807781478685866e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (-1.3807781478685866e-07) [X0 Z1 X2 X5 Z6 X7]
+ (-1.3807781478685866e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (-1.3807781478685866e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (-1.3807781478685866e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (-1.3807781478685866e-07) [X1 Z2 X3 X4 Z5 X6]
+ (-1.3807781478685866e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (-1.3807781478685866e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (-1.2919694861892434e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.2919694861892434e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.1076325599207725e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.1076325599207725e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.1076325599207725e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.1076325599207725e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.1076325599207725e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.1076325599207725e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.1076325599207725e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.1076325599207725e-07) [X1 Z2 X3 X10 Z11 X12]
+ (8.057446596591205e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.057446596591205e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.057446596591205e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.057446596591205e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.64931013203269e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.64931013203269e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.64931013203269e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.64931013203269e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.839420915204479e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.839420915204479e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.839420915204479e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.839420915204479e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.199051618708911e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (2.199051618708911e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (2.447323128495302e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (2.447323128495302e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.236259960779495e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.236259960779495e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.236259960779495e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.236259960779495e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.3281393503267753e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.3281393503267753e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.570761328550662e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.570761328550662e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.83805275033845e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (4.83805275033845e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (5.627851911485864e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.627851911485864e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (6.628614201825418e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.628614201825418e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (7.956895371345524e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.956895371345524e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.306536651987438e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.306536651987438e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.306536651987438e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.306536651987438e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.0358477599854005e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.0358477599854005e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.2283337823395862e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.2283337823395862e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.2393363216892138e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.2393363216892138e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.2393363216892138e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.2393363216892138e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.330473188673807e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.330473188673807e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.4548424488661097e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.4548424488661097e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.1726691014937126e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.1726691014937126e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.0882507114944743e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.0882507114944743e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.1174479455876873e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.1174479455876873e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.1513463112865705e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.1513463112865705e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.313145500142764e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.313145500142764e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.3343312895329352e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.3343312895329352e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.694513294725133e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.694513294725133e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (4.183932559410821e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.183932559410821e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.556569218414335e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (4.556569218414335e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (4.6430510686919484e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.6430510686919484e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.275883122262065e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.275883122262065e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.974311713578297e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (5.974311713578297e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (0.0002921986261110529) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.0002921986261110529) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.0002921986261110529) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.0002921986261110529) [X6 Z7 X8 X9 Z10 X11]
+ (0.0004956762314916195) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.0004956762314916195) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.0006650070219499367) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.0006650070219499367) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.0006650070219499367) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.0006650070219499367) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.0008533856254125488) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.0008533856254125488) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.0016095313817213828) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.0016095313817213828) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.0016095313817213828) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.0016095313817213828) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.001667604181144063) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.001667604181144063) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.001667604181144063) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.001667604181144063) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.001727875394136968) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.001727875394136968) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.001799219493663007) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.001799219493663007) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.0022939566113524676) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.0022939566113524676) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.0024629170071339313) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.0024629170071339313) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.0024629170071339313) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.0024629170071339313) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.003961560792496531) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.003961560792496531) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.003961560792496531) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.003961560792496531) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.004424855449441857) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.004424855449441857) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.004575007626639194) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.004575007626639194) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.004668620318776296) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.004668620318776296) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.004684903388155206) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.004684903388155206) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.005324835234221691) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.005324835234221691) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.005324835234221691) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.005324835234221691) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.005368659358109563) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.005368659358109563) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.005368659358109563) [X2 X3 X7 Z8 Z9 X10]
+ (0.005368659358109563) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.007960880725921545) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.007960880725921545) [X4 Z5 X6 X10 Z11 X12]
+ (0.007960880725921545) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.007960880725921545) [X5 Z6 X7 X11 Z12 X13]
+ (0.00812525192138102) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.00812525192138102) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.008890731522694595) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.008890731522694595) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.008890731522694595) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.008890731522694595) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.010263414868158507) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.010263414868158507) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.010263414868158507) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.010263414868158507) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.010540425907671458) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.010540425907671458) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.010540425907671458) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.010540425907671458) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.01096007494054261) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.01096007494054261) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.01096007494054261) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.01096007494054261) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.01130727400884813) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.01130727400884813) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.01441109943013093) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.01441109943013093) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.01441109943013093) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.01441109943013093) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.015225630757226594) [Y3 Z4 Z5 X6 X10 Y11]
+ (0.015225630757226594) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (0.015225630757226594) [X3 Z4 Z5 X6 X10 X11]
+ (0.015225630757226594) [X3 Z4 Z5 Y6 Y10 X11]
+ (0.015588250102380194) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (0.015588250102380194) [X2 Z3 X4 X10 Z11 X12]
+ (0.015588250102380194) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (0.015588250102380194) [X3 Z4 X5 X11 Z12 X13]
+ (0.01826683486937556) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.01826683486937556) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.01826683486937556) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.01826683486937556) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.019020423173039976) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.019020423173039976) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.019020423173039976) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.019020423173039976) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.020175921723535502) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.020175921723535502) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.020175921723535502) [X4 Z5 Z6 X7 X11 X12]
+ (0.020175921723535502) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.020175921723535502) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.020175921723535502) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.020175921723535502) [X5 X6 X10 Z11 Z12 X13]
+ (0.020175921723535502) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.024353077678068963) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.024353077678068963) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.024353077678068963) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.024353077678068963) [X2 Z3 X4 X11 Z12 X13]
+ (0.024353077678068963) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.024353077678068963) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.024353077678068963) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.024353077678068963) [X3 Z4 X5 X10 Z11 X12]
+ (0.02438908253114954) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.02438908253114954) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.02438908253114954) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.02438908253114954) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.02510495713884449) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.02510495713884449) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.02510495713884449) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.02510495713884449) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.030787505389143918) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.030787505389143918) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.045879470781297935) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.045879470781297935) [X0 Z2 Z3 Z4 Z5 X6]
+ (0.056007330877807605) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.056007330877807605) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.056007330877807605) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.056007330877807605) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661351) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.05608468124661351) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661351) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.05608468124661351) [Z1 X7 Z8 Z9 Z10 X11]
+ (-6.631277928708412e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.631277928708412e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.631277928708412e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.631277928708412e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.595086007265158e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.595086007265158e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.5950860072651574e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.5950860072651574e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.04274327701378304) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.04274327701378304) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (0.04274327701378304) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04274327701378304) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.04764261217638307) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.04764261217638307) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.04764261217638307) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.04764261217638307) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.041718813839821726) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.041718813839821726) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.041718813839821726) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.041718813839821726) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.03956441632289334) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.03956441632289334) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.03956441632289334) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03956441632289334) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.039359168022053116) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.039359168022053116) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.039359168022053116) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.039359168022053116) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.03931805194719755) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03931805194719755) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.03931805194719755) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03931805194719755) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03560837898831262) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.03560837898831262) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.02990378951262487) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.02990378951262487) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.02990378951262487) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.02990378951262487) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.028730779551905554) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.028730779551905554) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.028730779551905554) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.028730779551905554) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.025637238296026793) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.025637238296026793) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.025637238296026793) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.025637238296026793) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.024755463292891) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.024755463292891) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.024755463292891) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.024755463292891) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.024282117354693003) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.024282117354693003) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.02314513092952893) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.02314513092952893) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.022528440196013008) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.022528440196013008) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.021433810721600964) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.021433810721600964) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.021433810721600964) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.021433810721600964) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251596) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.019257505095251596) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019028242443847272) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.019028242443847272) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.018889030304942975) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-0.018889030304942975) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-0.018889030304942975) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.018889030304942975) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.01602460368917958) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.01602460368917958) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.015225630757226594) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.015225630757226594) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.014603704729162115) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.014603704729162115) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.014564531231173034) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.014564531231173034) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.011756013419819245) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.011756013419819245) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.011285190200840907) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-0.011285190200840907) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-0.009841749246962581) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.009841749246962581) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.009612634606847217) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.009612634606847217) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.009612634606847217) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.009612634606847217) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.008469978791023906) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.008469978791023906) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.007306759928832951) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.007306759928832951) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005923798336561342) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.005923798336561342) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.005652620978017345) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.005652620978017345) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.005368659358109563) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005368659358109563) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.0041587973818400445) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.0041587973818400445) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0033566705638328805) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.0033566705638328805) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.0033566705638328805) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.0033566705638328805) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.003267513854423538) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.003267513854423538) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.003267513854423538) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.003267513854423538) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.0027790267990255337) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.0027790267990255337) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.0026860409778066228) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.0026860409778066228) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.0026860409778066228) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.0026860409778066228) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.0022939566113524676) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939566113524676) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.0022939566113524676) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939566113524676) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.0009581655836696545) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.0009581655836696545) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.0009581655836696545) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.0009581655836696545) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.0009581655836696545) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.0009581655836696545) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.0009581655836696545) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.0009581655836696545) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.0002463643756957924) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0002463643756957924) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303550474) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.00013840177303550474) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.00013840177303550474) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.00013840177303550474) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.735036880590775e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.735036880590775e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.610358530637115e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.610358530637115e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.610358530637115e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.610358530637115e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808795951186e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.5316808795951186e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808795951186e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5316808795951186e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.806102775507947e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.806102775507947e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.806102775507947e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.806102775507947e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.0897994677408685e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.0897994677408685e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.0897994677408685e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.0897994677408685e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.652209670177278e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.652209670177278e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.652209670177278e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.652209670177278e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.4818518346554215e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.4818518346554215e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.4818518346554215e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.4818518346554215e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.071480736540141e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.071480736540141e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.071480736540141e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.071480736540141e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.7346220389678055e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.7346220389678055e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.7346220389678055e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.7346220389678055e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.728843147385473e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.728843147385473e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.728843147385473e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.728843147385473e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.2532242256472334e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.2532242256472334e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.7696594523449825e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.7696594523449825e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.5443954294089083e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.5443954294089083e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.5443954294089083e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.5443954294089083e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.5443954294089083e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.5443954294089083e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.5443954294089083e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.5443954294089083e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.3609563203553954e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563203553954e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563203553954e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.3609563203553954e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-2.1032156049334734e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.1032156049334734e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.1032156049334734e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.1032156049334734e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220986055367e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.0111220986055367e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220986055367e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.0111220986055367e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.94294683675527e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.94294683675527e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.94294683675527e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.94294683675527e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174773811434e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.6541174773811434e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174773811434e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.6541174773811434e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.5224930676893287e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.5224930676893287e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.5224930676893287e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.5224930676893287e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.5224930676893287e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930676893287e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930676893287e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.5224930676893287e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.2283337823395862e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337823395862e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.2283337823395862e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337823395862e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.988770287727272e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.988770287727272e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.988770287727272e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.988770287727272e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.867765104199631e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104199631e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104199631e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.867765104199631e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.189990975753969e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.189990975753969e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.175246207169291e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175246207169291e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.47164774480149e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.47164774480149e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.5614471791443397e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.5614471791443397e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.5614471791443397e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.5614471791443397e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.523389678384596e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.523389678384596e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.427323108582932e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.427323108582932e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.427323108582932e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.427323108582932e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.3281393503267753e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.3281393503267753e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.3281393503267753e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.3281393503267753e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.0868265649047007e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.0868265649047007e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.888293593741268e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.888293593741268e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.888293593741268e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.888293593741268e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.371328947576226e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.371328947576226e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.839420915204479e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.839420915204479e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-8.057446596591205e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.057446596591205e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (4.5371780959754544e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.5371780959754544e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.5371780959754544e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (4.5371780959754544e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (8.057446596591205e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.057446596591205e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (9.209350632793673e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350632793673e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350632793673e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.209350632793673e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783552185693e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783552185693e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783552185693e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783552185693e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.839420915204479e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.839420915204479e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.371328947576226e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.371328947576226e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (3.0868265649047007e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.0868265649047007e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (4.523389678384596e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.523389678384596e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (5.47164774480149e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.47164774480149e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.175246207169291e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175246207169291e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.189990975753969e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.189990975753969e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.330473188673807e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.330473188673807e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.330473188673807e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.330473188673807e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.6288532435972418e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.6288532435972418e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.6288532435972418e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.6288532435972418e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.6893489515438522e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.6893489515438522e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.6893489515438522e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.6893489515438522e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (2.745518400636181e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.745518400636181e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.745518400636181e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.745518400636181e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.745518400636181e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.745518400636181e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.745518400636181e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.745518400636181e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.211842019233181e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.211842019233181e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.211842019233181e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.211842019233181e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.211842019233181e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.211842019233181e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.211842019233181e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.211842019233181e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.313145500142764e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.313145500142764e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.313145500142764e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.313145500142764e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (3.3343312895329352e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.3343312895329352e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (4.183932559410821e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.183932559410821e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (7.735036880590775e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.735036880590775e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.0002463643756957924) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.0002463643756957924) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0004458535128840862) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.0004458535128840862) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.0004458535128840862) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.0004458535128840862) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.000594022154300549) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.000594022154300549) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.000594022154300549) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.000594022154300549) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.000594022154300549) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.000594022154300549) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.000594022154300549) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.000594022154300549) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.0008533856254125488) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.0008533856254125488) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.0008533856254125488) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.0008533856254125488) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.0010435246534907625) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.0010435246534907625) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.0010435246534907625) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.0010435246534907625) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.0012803060973496788) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.0012803060973496788) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.0012803060973496788) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.0012803060973496788) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.001303800478812693) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (0.001303800478812693) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (0.001303800478812693) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (0.001303800478812693) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (0.0022619660624823477) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (0.0022619660624823477) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (0.0022619660624823477) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (0.0022619660624823477) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (0.0022619660624823477) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (0.0022619660624823477) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (0.0022619660624823477) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (0.0022619660624823477) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (0.003989841456619315) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (0.003989841456619315) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (0.003989841456619315) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (0.003989841456619315) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (0.0041587973818400445) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.0041587973818400445) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.004311038507914301) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.004311038507914301) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.004311038507914301) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.004311038507914301) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.004636976661182559) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.004636976661182559) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.004636976661182559) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.004636976661182559) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.0051144738316603825) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.0051144738316603825) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.0051144738316603825) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.0051144738316603825) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.0051144738316603825) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.0051144738316603825) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.0051144738316603825) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.0051144738316603825) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.0052415353828038636) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (0.0052415353828038636) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (0.0052415353828038636) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (0.0052415353828038636) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (0.005262642473076846) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.005262642473076846) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.005262642473076846) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.005262642473076846) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.005368659358109563) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005368659358109563) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005379937155839368) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.005379937155839368) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.005379937155839368) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.005379937155839368) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.005652620978017345) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.005652620978017345) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.005708495985960932) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.005708495985960932) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.005708495985960932) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.005708495985960932) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.005923798336561342) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.005923798336561342) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.007306759928832951) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.007306759928832951) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.008469978791023906) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.008469978791023906) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.009841749246962581) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.009841749246962581) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.011285190200840907) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (0.011285190200840907) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (0.011756013419819245) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.011756013419819245) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.014564531231173034) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.014564531231173034) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.014603704729162115) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.014603704729162115) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.015225630757226594) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.015225630757226594) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.01602460368917958) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.01602460368917958) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.019028242443847272) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.019028242443847272) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.019257505095251596) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.019257505095251596) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.045879470781297935) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.045879470781297935) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.36937089366156073) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.36937089366156073) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.36937089366156073) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.36937089366156073) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.2816425776702293) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.2816425776702293) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.2816425776702292) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.2816425776702292) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.09065144207036469) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.09065144207036469) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.09065144207036469) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.09065144207036469) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863616) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.08684737589863616) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863616) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.08684737589863616) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.07635021950635013) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07635021950635013) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07635021950635013) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07635021950635013) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214027) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.06752385099214027) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214027) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.06752385099214027) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03560837898831263) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.03560837898831263) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.034903343373661806) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.034903343373661806) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.034903343373661806) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.034903343373661806) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883829992) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.024591860883829992) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883829992) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.024591860883829992) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.024282117354693) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.024282117354693) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.023145130929528926) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.023145130929528926) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.02252844019601301) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.02252844019601301) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019538050311314736) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.019538050311314736) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.019538050311314736) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.019538050311314736) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.01709155315589885) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.01709155315589885) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.01709155315589885) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.01709155315589885) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.01602460368917958) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.01602460368917958) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.01602460368917958) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.01602460368917958) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.010311482489831814) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831814) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831814) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010311482489831814) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.009841749246962581) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962581) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.009841749246962581) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962581) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.00882636851420985) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.00882636851420985) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.00882636851420985) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.00882636851420985) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.00854199662545483) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00854199662545483) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00854199662545483) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00854199662545483) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.00854199662545483) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00854199662545483) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00854199662545483) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00854199662545483) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008469978791023906) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023906) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.008469978791023906) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023906) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.004668620318776296) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.004668620318776296) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.003876470899336938) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.003876470899336938) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.0038040661717285364) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0038040661717285364) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0038040661717285364) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0038040661717285364) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178813) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178813) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00335667056383288) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.00335667056383288) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.0032675138544235385) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.0032675138544235385) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0021413612231015546) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.0021413612231015546) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.001727875394136968) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.001727875394136968) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.0016407548553124254) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0016407548553124254) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214168723) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0014528843214168723) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214168723) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0014528843214168723) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.000787089677102445) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.000787089677102445) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0005192743499487741) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.0005192743499487741) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.00019400857029756323) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00019400857029756323) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303550474) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.00013840177303550474) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.14162522115147e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.14162522115147e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.14162522115147e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.14162522115147e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-5.071480736540141e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.071480736540141e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.1513463112865705e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.1513463112865705e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0882507114944743e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.0882507114944743e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.9885117061252255e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.9885117061252255e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8742990714474166e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.8742990714474166e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.3609563203553954e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.3609563203553954e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.3002946562983616e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.3002946562983616e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.1468376508172332e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.1468376508172332e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.1468376508172332e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.1468376508172332e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.352332103680968e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.352332103680968e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.352332103680968e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.352332103680968e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.09163719965765e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.09163719965765e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.09163719965765e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.09163719965765e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.09163719965765e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.09163719965765e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.09163719965765e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.09163719965765e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.074305986436226e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.074305986436226e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.074305986436226e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.074305986436226e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.900128986899279e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.900128986899279e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.900128986899279e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900128986899279e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867765104199631e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.867765104199631e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.560692465422068e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692465422068e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692465422068e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692465422068e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692465422068e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692465422068e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692465422068e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.560692465422068e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.997018422368475e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.997018422368475e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.997018422368475e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.997018422368475e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.997018422368475e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.997018422368475e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.997018422368475e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.997018422368475e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.5682475212730537e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.5682475212730537e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.5682475212730537e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.5682475212730537e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393085146827e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393085146827e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393085146827e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393085146827e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393085146827e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393085146827e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.3767393085146827e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393085146827e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.888293593741268e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.888293593741268e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6863815478302307e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.6863815478302307e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035783552185693e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7035783552185693e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.209350632793673e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.209350632793673e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244705782e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773244705782e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244705782e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773244705782e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244705782e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773244705782e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.379773244705782e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773244705782e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253796669067e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253796669067e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9742253796669067e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9742253796669067e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0474716555579414e-08) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.0474716555579414e-08) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.0474716555579414e-08) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.0474716555579414e-08) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.209350632793673e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.209350632793673e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0717282186190768e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.0717282186190768e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.0717282186190768e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.0717282186190768e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.2004287494176342e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.2004287494176342e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.2004287494176342e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.2004287494176342e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.7035783552185693e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7035783552185693e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.3120943053544692e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.3120943053544692e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.3120943053544692e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.3120943053544692e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.6863815478302307e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815478302307e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.888293593741268e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.888293593741268e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.092250616191506e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.092250616191506e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.092250616191506e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.092250616191506e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.092250616191506e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.092250616191506e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.092250616191506e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.092250616191506e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.444597853907171e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.444597853907171e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.444597853907171e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.444597853907171e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.684915095186449e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.684915095186449e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.684915095186449e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.684915095186449e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.246974425788368e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.246974425788368e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.246974425788368e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.246974425788368e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.246974425788368e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.246974425788368e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.246974425788368e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.246974425788368e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.867765104199631e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.867765104199631e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.3002946562983616e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.3002946562983616e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.3609563203553954e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.3609563203553954e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.8742990714474166e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.8742990714474166e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.8836765760499247e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.8836765760499247e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.9473560118570768e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9473560118570768e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.9473560118570768e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9473560118570768e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117061252255e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9885117061252255e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.0882507114944743e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.0882507114944743e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.1513463112865705e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.1513463112865705e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.8462016713131705e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.8462016713131705e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.8462016713131705e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.8462016713131705e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.071480736540141e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.071480736540141e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.1055267220096086e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.1055267220096086e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.1055267220096086e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.1055267220096086e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.146496327611532e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.146496327611532e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.146496327611532e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.146496327611532e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.159350501844043e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.159350501844043e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.159350501844043e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.159350501844043e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.427988656627066e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.427988656627066e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.427988656627066e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.427988656627066e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.935867717982302e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.935867717982302e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.935867717982302e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.935867717982302e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.2532733481430875e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.2532733481430875e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.979825793457025e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.979825793457025e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.979825793457025e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.979825793457025e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (4.205548411217357e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.205548411217357e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.205548411217357e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.205548411217357e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00013840177303550474) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.00013840177303550474) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.00018787053389555274) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.00018787053389555274) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.00018787053389555274) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.00018787053389555274) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00019400857029756323) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00019400857029756323) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0002463643756957924) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0002463643756957924) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0002463643756957924) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0002463643756957924) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192743499487741) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.0005192743499487741) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.0007156734248909302) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007156734248909302) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0007156734248909302) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007156734248909302) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.000787089677102445) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.000787089677102445) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.0015324835230730838) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.0015324835230730838) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.0015324835230730838) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.0015324835230730838) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.0016407548553124254) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0016407548553124254) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.001727875394136968) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.001727875394136968) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.002446497155415886) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.002446497155415886) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.002446497155415886) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.002446497155415886) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.0032675138544235385) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.0032675138544235385) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.00335667056383288) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.00335667056383288) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.0034841573002178813) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0034841573002178813) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.003876470899336938) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.003876470899336938) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.004668620318776296) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.004668620318776296) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.0047672721882781426) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.0047672721882781426) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.0047672721882781426) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.0047672721882781426) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.005286546538226917) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.005286546538226917) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.005286546538226917) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.005286546538226917) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.0054089544224100216) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.0054089544224100216) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.0054089544224100216) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.0054089544224100216) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.005923798336561342) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561342) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.005923798336561342) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561342) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.010715508469796768) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010715508469796768) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010715508469796768) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010715508469796768) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.010757563953908941) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010757563953908941) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010757563953908941) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010757563953908941) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.014603704729162115) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.014603704729162115) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.014603704729162115) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.014603704729162115) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.01929956057936377) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01929956057936377) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01929956057936377) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01929956057936377) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01929956057936377) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01929956057936377) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.01929956057936377) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01929956057936377) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.058591988733861906) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.058591988733861906) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.775950527283265e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.775950527283265e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.775950527283265e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.775950527283265e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.07165035181002678) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.07165035181002678) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.07165035181002684) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.07165035181002684) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.019257505095251596) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019257505095251596) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010311482489831814) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311482489831814) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00882636851420985) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.00882636851420985) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0075974640297706095) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0075974640297706095) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0075974640297706095) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0075974640297706095) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311866) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311866) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311866) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311866) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311866) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311866) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311866) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311866) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0053480515826766235) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0053480515826766235) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0053480515826766235) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0053480515826766235) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0038040661717285364) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0038040661717285364) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002984166168121921) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.002984166168121921) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.002984166168121921) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.002984166168121921) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.002446497155415886) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.002446497155415886) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.0022494124470939843) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0022494124470939843) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0022494124470939843) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0022494124470939843) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0021413612231015546) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.0021413612231015546) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.0018638942824587442) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587442) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587442) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587442) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587442) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587442) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0018638942824587442) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587442) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0016407548553124252) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124252) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0016407548553124252) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124252) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0012223378081538349) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538349) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538349) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538349) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538349) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538349) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538349) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0012223378081538349) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0010283292378562717) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0010283292378562717) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0010283292378562717) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0010283292378562717) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1463061453287492e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1463061453287492e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.8742990714474166e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990714474166e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.8742990714474166e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990714474166e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.3002946562983616e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.3002946562983616e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.3002946562983616e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.3002946562983616e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.0444941298301619e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.0444941298301619e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.0444941298301619e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.0444941298301619e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.95607923024471e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.95607923024471e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.95607923024471e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.95607923024471e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105515037534124e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105515037534124e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105515037534124e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105515037534124e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661347213318e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661347213318e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661347213318e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661347213318e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540341413893793e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540341413893793e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.189990975753969e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.189990975753969e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.87662165847858e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.87662165847858e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.87662165847858e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.87662165847858e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175246207169291e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175246207169291e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.523389678384596e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.523389678384596e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0767325317909484e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.0767325317909484e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0767325317909484e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.0767325317909484e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.0134714590794775e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0134714590794775e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.9045998844078253e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.9045998844078253e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.9045998844078253e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.9045998844078253e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.666731754405623e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.666731754405623e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.666731754405623e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.666731754405623e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641927105872e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.8505641927105872e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309312938025e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.6569309312938025e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309312938025e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.6569309312938025e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.8505641927105872e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.8505641927105872e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815478302307e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815478302307e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.6863815478302307e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815478302307e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714590794775e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.0134714590794775e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.523389678384596e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.523389678384596e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.67040239037328e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.67040239037328e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.67040239037328e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.67040239037328e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175246207169291e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175246207169291e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.189990975753969e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.189990975753969e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540341413893793e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540341413893793e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949476487520405e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949476487520405e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.7924939577303844e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939577303844e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939577303844e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.7924939577303844e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.8836765760499247e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.8836765760499247e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.9885117061252255e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117061252255e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117061252255e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117061252255e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.2532733481430875e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.2532733481430875e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4017109735380291e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4017109735380291e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4017109735380291e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4017109735380291e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603693110676e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.5809603693110676e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603693110676e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.5809603693110676e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0005192743499487741) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487741) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.0005192743499487741) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487741) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.000787089677102445) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.000787089677102445) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.000787089677102445) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.000787089677102445) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.001172634831644184) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.001172634831644184) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.001172634831644184) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.001172634831644184) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012366478019245715) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.0012366478019245715) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.0012366478019245715) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.0012366478019245715) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.002200964069500456) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002200964069500456) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002200964069500456) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002200964069500456) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798019) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798019) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798019) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798019) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798019) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798019) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002394972639798019) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798019) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002446497155415886) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.002446497155415886) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.0038040661717285364) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0038040661717285364) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0038764708993369377) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.0038764708993369377) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.0038764708993369377) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.0038764708993369377) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.004220813970046492) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.004220813970046492) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.004220813970046492) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.004220813970046492) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.00882636851420985) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.00882636851420985) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.010311482489831814) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010311482489831814) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019257505095251596) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019257505095251596) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.058591988733861906) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.058591988733861906) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.3987009014845734e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.3987009014845734e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.3987009014845734e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.3987009014845734e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841573002178813) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841573002178813) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002984166168121921) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.002984166168121921) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.00019400857029756323) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00019400857029756323) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453287492e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453287492e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.7924939577303844e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7924939577303844e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540341413893793e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413893793e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540341413893793e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413893793e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.8505641927105872e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641927105872e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641927105872e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641927105872e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714590794775e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.0134714590794775e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714590794775e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.0134714590794775e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.949476487520405e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.949476487520405e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.7924939577303844e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7924939577303844e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029756323) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029756323) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002984166168121921) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.002984166168121921) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
 </code>
 </pre>
 </details>

---

## 10. tutorial_error_mitigation.html <a name="demo9"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_error_mitigation.html):

```
──╭●─────────╭●──RY(-4.05)─┤
──╰Z─────────╰Z──RY(-3.51)─┤
0: ──RY(4.56)─╭●──────────RY(5.93)──RY(-5.93)───────────────────────────────
1: ──RY(3.60)─╰Z──────────RY(5.90)──────────────────────╭●──────────RY(5.18)
2: ──RY(4.05)──RY(-4.05)──RY(4.05)─╭●──────────RY(3.32)─╰Z──────────RY(1.07)
3: ──RY(3.51)──────────────────────╰Z──────────RY(3.66)──RY(-3.66)──────────
───────────────────────────╭●──RY(-4.56)─┤
───RY(-5.18)─╭●──RY(-5.90)─╰Z──RY(-3.60)─┤
───RY(-1.07)─╰Z──RY(-3.32)─╭●──RY(-4.05)─┤
───────────────────────────╰Z──RY(-3.51)─┤
0.9721946076757156
0: ──RY(4.56)─╭●──RY(5.93)──RY(-5.93)──RY(5.93)──RY(-5.93)───────────────╭●
───RY(-4.56)─┤
───RY(-3.60)─┤
───RY(-4.05)─┤
───RY(-3.51)─┤
───RY(-4.56)───────────────┤
───RY(-3.60)───────────────┤
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_error_mitigation.html):

```
───RY(-4.05)───────────────┤
───RY(-3.51)───────────────┤
0: ──RY(4.56)─╭●──RY(5.93)──RY(-5.93)───────────────────────────────────
1: ──RY(3.60)─╰Z──RY(5.90)─╭●──────────RY(5.18)──RY(-5.18)─╭●──RY(-5.90)
2: ──RY(4.05)─╭●──RY(3.32)─╰Z──────────RY(1.07)──RY(-1.07)─╰Z──RY(-3.32)
3: ──RY(3.51)─╰Z──RY(3.66)──RY(-3.66)───────────────────────────────────
───────────────────────╭●──RY(-4.56)─┤
───RY(5.90)──RY(-5.90)─╰Z──RY(-3.60)─┤
──╭●─────────RY(-4.05)───────────────┤
──╰Z─────────RY(-3.51)───────────────┤
0.9625290387961026
0: ──RY(4.56)─╭●──RY(5.93)──RY(-5.93)────────────────────────────────────╭●
───RY(-4.56)───────────────┤
───RY(-3.60)───────────────┤
──╭●─────────╭●──RY(-4.05)─┤
──╰Z─────────╰Z──RY(-3.51)─┤
──╭●─────────╭●──RY(-4.56)─┤
──╰Z─────────╰Z──RY(-3.60)─┤
```

---

## 11. tutorial_backprop.html <a name="demo10"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_backprop.html):

```
-0.06518877224958124
[-6.51888e-02 -2.72892e-02  0.00000e+00 -9.33935e-02 -7.61068e-01
  4.16334e-17]
-0.06518877224958125
[-6.51888e-02 -2.72892e-02 -2.83425e-17 -9.33935e-02 -7.61068e-01
  4.10465e-17]
180
0.8947771876917631
Forward pass (best of 3): 0.02812477289999151 sec per loop
Gradient computation (best of 3): 11.47715549809991 sec per loop
10.124918243996945
0.9358535378025424
Forward pass (best of 3): 0.06274951479999799 sec per loop
Backward pass (best of 3): 0.17320888879994528 sec per loop
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_backprop.html):

```
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The animation.avconv_args rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The animation.avconv_path rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The animation.html_args rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The keymap.all_axes rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The savefig.jpeg_quality rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The text.latex.preview rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
-0.06518877224958124
[-6.51888e-02 -2.72892e-02  0.00000e+00 -9.33935e-02 -7.61068e-01
```

---

## 12. tutorial_local_cost_functions.html <a name="demo11"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_local_cost_functions.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Cost after step     5:  1.0000000
Cost after step    10:  1.0000000
Cost after step    15:  1.0000000
Cost after step    20:  1.0000000
Cost after step    25:  1.0000000
Cost after step    30:  1.0000000
Cost after step    35:  1.0000000
Cost after step    40:  1.0000000
Cost after step    45:  1.0000000
Cost after step    50:  1.0000000
Cost after step    55:  1.0000000
Cost after step    60:  1.0000000
Cost after step    65:  1.0000000
Cost after step    70:  1.0000000
Cost after step    75:  1.0000000
Cost after step    80:  1.0000000
Cost after step    85:  1.0000000
Cost after step    90:  1.0000000
Cost after step    95:  1.0000000
Cost after step   100:  1.0000000
Cost after step     5:  0.9871000
Cost after step    10:  0.9651000
Cost after step    15:  0.9173000
Cost after step    20:  0.8059000
Cost after step    25:  0.6213000
Cost after step    30:  0.3703000
Cost after step    35:  0.1821000
Cost after step    40:  0.0684000
tensor(1., requires_grad=True)
Current cost: 0.9999999999972213.
Initial cost: 0.9999999999999843.
Difference: 2.763012041384627e-12
0.9957
Cost after step    10:  0.9909000. Locality: 2
Cost after step    20:  0.9753000. Locality: 2
Cost after step    30:  0.9275000. Locality: 2
Cost after step    40:  0.8386000. Locality: 2
Cost after step    50:  0.6821000. Locality: 2
Cost after step    60:  0.4353000. Locality: 2
Cost after step    70:  0.2264000. Locality: 2
Cost after step    80:  0.0923000. Locality: 2
---Switching Locality---
Cost after step    90:  0.9901000. Locality: 3
Cost after step   100:  0.9737000. Locality: 3
Cost after step   110:  0.9400000. Locality: 3
Cost after step   120:  0.8711000. Locality: 3
Cost after step   130:  0.7228000. Locality: 3
Cost after step   140:  0.5156000. Locality: 3
Cost after step   150:  0.2846000. Locality: 3
Cost after step   160:  0.1285000. Locality: 3
---Switching Locality---
Cost after step   170:  0.9899000. Locality: 4
Cost after step   180:  0.9799000. Locality: 4
Cost after step   190:  0.9512000. Locality: 4
Cost after step   200:  0.8964000. Locality: 4
Cost after step   210:  0.7683000. Locality: 4
Cost after step   220:  0.5752000. Locality: 4
Cost after step   230:  0.3314000. Locality: 4
Cost after step   240:  0.1575000. Locality: 4
---Switching Locality---
Cost after step   250:  0.9942000. Locality: 5
Cost after step   260:  0.9866000. Locality: 5
Cost after step   270:  0.9641000. Locality: 5
Cost after step   280:  0.9120000. Locality: 5
Cost after step   290:  0.8136000. Locality: 5
Cost after step   300:  0.6380000. Locality: 5
Cost after step   310:  0.4004000. Locality: 5
Cost after step   320:  0.1996000. Locality: 5
---Switching Locality---
Cost after step   330:  0.9945000. Locality: 6
Cost after step   340:  0.9873000. Locality: 6
Cost after step   350:  0.9689000. Locality: 6
Cost after step   360:  0.9288000. Locality: 6
Cost after step   370:  0.8476000. Locality: 6
Cost after step   380:  0.6711000. Locality: 6
Cost after step   390:  0.4527000. Locality: 6
Cost after step   400:  0.2342000. Locality: 6
Cost after step   410:  0.1014000. Locality: 6
--- New run! ---
Cost after step    20:  1.0000000
Cost after step    40:  1.0000000
Cost after step    60:  1.0000000
Cost after step    80:  1.0000000
Cost after step   100:  1.0000000
Cost after step   120:  1.0000000
Cost after step   140:  1.0000000
Cost after step   160:  1.0000000
Cost after step   180:  1.0000000
Cost after step   200:  1.0000000
Cost after step   220:  1.0000000
Cost after step   240:  1.0000000
Cost after step   260:  1.0000000
Cost after step   280:  1.0000000
Cost after step   300:  1.0000000
Cost after step   320:  1.0000000
Cost after step   340:  1.0000000
Cost after step   360:  1.0000000
Cost after step   380:  1.0000000
Cost after step   400:  1.0000000
Trained:     0
Plateau'd:     1
--- New run! ---
Cost after step    20:  0.9993000
Cost after step    40:  0.9994000
Cost after step    60:  0.9995000
Cost after step    80:  0.9994000
Cost after step   100:  0.9996000
Cost after step   120:  0.9995000
Cost after step   140:  0.9986000
Cost after step   160:  0.9993000
Cost after step   180:  0.9991000
Cost after step   200:  0.9996000
Cost after step   220:  0.9985000
Cost after step   240:  0.9989000
Cost after step   260:  0.9993000
Cost after step   280:  0.9991000
Cost after step   300:  0.9990000
Cost after step   320:  0.9987000
Cost after step   340:  0.9987000
Cost after step   360:  0.9994000
Cost after step   380:  0.9985000
Cost after step   400:  0.9991000
Trained:     0
Plateau'd:     2
--- New run! ---
Cost after step    20:  0.9976000
Cost after step    40:  0.9967000
Cost after step    60:  0.9970000
Cost after step    80:  0.9964000
Cost after step   100:  0.9956000
Cost after step   120:  0.9965000
Cost after step   140:  0.9958000
Cost after step   160:  0.9931000
Cost after step   180:  0.9940000
Cost after step   200:  0.9908000
Cost after step   220:  0.9887000
Cost after step   240:  0.9834000
Cost after step   260:  0.9821000
Cost after step   280:  0.9742000
Cost after step   300:  0.9576000
Cost after step   320:  0.9330000
Trained:     1
Plateau'd:     2
--- New run! ---
Cost after step    20:  0.9998000
Cost after step    40:  0.9997000
Cost after step    60:  0.9996000
Cost after step    80:  0.9994000
Cost after step   100:  0.9994000
Cost after step   120:  0.9996000
Cost after step   140:  0.9997000
Cost after step   160:  0.9992000
Cost after step   180:  0.9991000
Cost after step   200:  0.9992000
Cost after step   220:  0.9990000
Cost after step   240:  0.9986000
Cost after step   260:  0.9989000
Cost after step   280:  0.9988000
Cost after step   300:  0.9972000
Cost after step   320:  0.9982000
Cost after step   340:  0.9981000
Cost after step   360:  0.9979000
Cost after step   380:  0.9977000
Cost after step   400:  0.9972000
Trained:     1
Plateau'd:     3
--- New run! ---
Cost after step    20:  1.0000000
Cost after step    40:  1.0000000
Cost after step    60:  0.9999000
Cost after step    80:  1.0000000
Cost after step   100:  1.0000000
Cost after step   120:  1.0000000
Cost after step   140:  1.0000000
Cost after step   160:  0.9999000
Cost after step   180:  1.0000000
Cost after step   200:  1.0000000
Cost after step   220:  1.0000000
Cost after step   240:  1.0000000
Cost after step   260:  1.0000000
Cost after step   280:  1.0000000
Cost after step   300:  1.0000000
Cost after step   320:  1.0000000
Cost after step   340:  1.0000000
Cost after step   360:  0.9998000
Cost after step   380:  1.0000000
Cost after step   400:  1.0000000
Trained:     1
Plateau'd:     4
--- New run! ---
Cost after step    20:  0.9950000
Cost after step    40:  0.9957000
Cost after step    60:  0.9931000
Cost after step    80:  0.9920000
Cost after step   100:  0.9925000
Cost after step   120:  0.9908000
Cost after step   140:  0.9865000
Cost after step   160:  0.9861000
Cost after step   180:  0.9846000
Cost after step   200:  0.9767000
Cost after step   220:  0.9696000
Cost after step   240:  0.9560000
Cost after step   260:  0.9276000
Trained:     2
Plateau'd:     4
--- New run! ---
Cost after step    20:  0.9989000
Cost after step    40:  0.9979000
Cost after step    60:  0.9979000
Cost after step    80:  0.9982000
Cost after step   100:  0.9984000
Cost after step   120:  0.9986000
Cost after step   140:  0.9978000
Cost after step   160:  0.9976000
Cost after step   180:  0.9967000
Cost after step   200:  0.9972000
Cost after step   220:  0.9958000
Cost after step   240:  0.9966000
Cost after step   260:  0.9966000
Cost after step   280:  0.9952000
Cost after step   300:  0.9958000
Cost after step   320:  0.9972000
Cost after step   340:  0.9953000
Cost after step   360:  0.9934000
Cost after step   380:  0.9929000
Cost after step   400:  0.9916000
Trained:     2
Plateau'd:     5
--- New run! ---
Cost after step    20:  0.9988000
Cost after step    40:  0.9980000
Cost after step    60:  0.9978000
Cost after step    80:  0.9986000
Cost after step   100:  0.9978000
Cost after step   120:  0.9977000
Cost after step   140:  0.9974000
Cost after step   160:  0.9976000
Cost after step   180:  0.9975000
Cost after step   200:  0.9970000
Cost after step   220:  0.9972000
Cost after step   240:  0.9961000
Cost after step   260:  0.9960000
Cost after step   280:  0.9953000
Cost after step   300:  0.9947000
Cost after step   320:  0.9945000
Cost after step   340:  0.9912000
Cost after step   360:  0.9916000
Cost after step   380:  0.9866000
Cost after step   400:  0.9822000
Trained:     2
Plateau'd:     6
--- New run! ---
Cost after step    20:  1.0000000
Cost after step    40:  1.0000000
Cost after step    60:  1.0000000
Cost after step    80:  1.0000000
Cost after step   100:  1.0000000
Cost after step   120:  1.0000000
Cost after step   140:  1.0000000
Cost after step   160:  1.0000000
Cost after step   180:  1.0000000
Cost after step   200:  1.0000000
Cost after step   220:  1.0000000
Cost after step   240:  1.0000000
Cost after step   260:  1.0000000
Cost after step   280:  1.0000000
Cost after step   300:  1.0000000
Cost after step   320:  1.0000000
Cost after step   340:  1.0000000
Cost after step   360:  1.0000000
Cost after step   380:  1.0000000
Cost after step   400:  1.0000000
Trained:     2
Plateau'd:     7
--- New run! ---
Cost after step    20:  1.0000000
Cost after step    40:  1.0000000
Cost after step    60:  1.0000000
Cost after step    80:  1.0000000
Cost after step   100:  0.9999000
Cost after step   120:  1.0000000
Cost after step   140:  1.0000000
Cost after step   160:  1.0000000
Cost after step   180:  1.0000000
Cost after step   200:  1.0000000
Cost after step   220:  1.0000000
Cost after step   240:  1.0000000
Cost after step   260:  1.0000000
Cost after step   280:  1.0000000
Cost after step   300:  1.0000000
Cost after step   320:  1.0000000
Cost after step   340:  1.0000000
Cost after step   360:  1.0000000
Cost after step   380:  1.0000000
Cost after step   400:  1.0000000
Trained:     2
Plateau'd:     8
--- New run! ---
---Switching Locality---
Cost after step    10:  0.6137000. Locality: 2
---Switching Locality---
Cost after step    20:  0.5159000. Locality: 3
---Switching Locality---
Cost after step    30:  0.7040000. Locality: 4
Cost after step    40:  0.5823000. Locality: 4
---Switching Locality---
Cost after step    50:  0.5055000. Locality: 5
---Switching Locality---
Cost after step    60:  0.6965000. Locality: 6
Cost after step    70:  0.5823000. Locality: 6
---Switching Locality---
Cost after step    80:  0.8792000. Locality: 7
Cost after step    90:  0.7172000. Locality: 7
---Switching Locality---
Cost after step   100:  0.9741000. Locality: 8
Cost after step   110:  0.9329000. Locality: 8
Cost after step   120:  0.8278000. Locality: 8
Cost after step   130:  0.5973000. Locality: 8
Cost after step   140:  0.2649000. Locality: 8
Trained:     1
Plateau'd:     0
--- New run! ---
---Switching Locality---
Cost after step    10:  0.6273000. Locality: 2
---Switching Locality---
---Switching Locality---
Cost after step    20:  0.9218000. Locality: 4
Cost after step    30:  0.8247000. Locality: 4
Cost after step    40:  0.5992000. Locality: 4
---Switching Locality---
Cost after step    50:  0.6852000. Locality: 5
Cost after step    60:  0.6003000. Locality: 5
Cost after step    70:  0.5220000. Locality: 5
---Switching Locality---
Cost after step    80:  0.5100000. Locality: 6
---Switching Locality---
---Switching Locality---
Cost after step    90:  0.4677000. Locality: 8
Cost after step   100:  0.1562000. Locality: 8
Trained:     2
Plateau'd:     0
--- New run! ---
---Switching Locality---
---Switching Locality---
---Switching Locality---
---Switching Locality---
Cost after step    10:  0.5405000. Locality: 5
---Switching Locality---
Cost after step    20:  0.6024000. Locality: 6
---Switching Locality---
Cost after step    30:  0.6060000. Locality: 7
---Switching Locality---
Cost after step    40:  0.5009000. Locality: 8
Cost after step    50:  0.1676000. Locality: 8
Trained:     3
Plateau'd:     0
--- New run! ---
Cost after step    10:  0.4871000. Locality: 1
---Switching Locality---
Cost after step    20:  0.4948000. Locality: 2
---Switching Locality---
Cost after step    30:  0.6627000. Locality: 3
---Switching Locality---
Cost after step    40:  0.5826000. Locality: 4
---Switching Locality---
Cost after step    50:  0.5234000. Locality: 5
---Switching Locality---
---Switching Locality---
Cost after step    60:  0.6394000. Locality: 7
---Switching Locality---
Cost after step    70:  0.3126000. Locality: 8
Trained:     4
Plateau'd:     0
--- New run! ---
Cost after step    10:  0.5485000. Locality: 1
---Switching Locality---
Cost after step    20:  0.6393000. Locality: 2
Cost after step    30:  0.8061000. Locality: 3
Cost after step    40:  0.7073000. Locality: 3
Cost after step    50:  0.6144000. Locality: 3
Cost after step    60:  0.4888000. Locality: 3
Cost after step    70:  0.6102000. Locality: 4
---Switching Locality---
Cost after step    80:  0.4909000. Locality: 5
---Switching Locality---
Cost after step    90:  0.7897000. Locality: 6
Cost after step   100:  0.4939000. Locality: 6
---Switching Locality---
Cost after step   110:  0.7063000. Locality: 7
Cost after step   120:  0.5323000. Locality: 7
---Switching Locality---
Cost after step   130:  0.2921000. Locality: 8
Trained:     5
Plateau'd:     0
--- New run! ---
Cost after step    10:  0.7449000. Locality: 2
Cost after step    20:  0.5005000. Locality: 2
---Switching Locality---
---Switching Locality---
Cost after step    30:  0.7292000. Locality: 5
Cost after step    40:  0.4696000. Locality: 5
---Switching Locality---
Cost after step    50:  0.5099000. Locality: 7
Cost after step    60:  0.6587000. Locality: 8
Cost after step    70:  0.4912000. Locality: 8
Cost after step    80:  0.2440000. Locality: 8
Trained:     6
Plateau'd:     0
--- New run! ---
Cost after step    10:  0.8510000. Locality: 3
Cost after step    20:  0.7255000. Locality: 3
Cost after step    30:  0.5811000. Locality: 3
---Switching Locality---
Cost after step    40:  0.6300000. Locality: 4
---Switching Locality---
Cost after step    50:  0.8801000. Locality: 5
Cost after step    60:  0.7395000. Locality: 5
---Switching Locality---
Cost after step    70:  0.8948000. Locality: 6
Cost after step    80:  0.7399000. Locality: 6
---Switching Locality---
Cost after step    90:  0.7139000. Locality: 7
Cost after step   100:  0.5959000. Locality: 7
---Switching Locality---
Cost after step   110:  0.5939000. Locality: 8
Cost after step   120:  0.2906000. Locality: 8
Trained:     7
Plateau'd:     0
--- New run! ---
---Switching Locality---
Cost after step    10:  0.6986000. Locality: 2
Cost after step    20:  0.6273000. Locality: 2
Cost after step    30:  0.5645000. Locality: 2
Cost after step    40:  0.5016000. Locality: 2
---Switching Locality---
Cost after step    50:  0.9048000. Locality: 3
Cost after step    60:  0.7903000. Locality: 3
Cost after step    70:  0.5948000. Locality: 3
---Switching Locality---
Cost after step    80:  0.5600000. Locality: 5
Cost after step    90:  0.8307000. Locality: 6
Cost after step   100:  0.6442000. Locality: 6
---Switching Locality---
Cost after step   110:  0.7075000. Locality: 7
Cost after step   120:  0.5936000. Locality: 7
---Switching Locality---
Cost after step   130:  0.7649000. Locality: 8
Cost after step   140:  0.6810000. Locality: 8
Cost after step   150:  0.5936000. Locality: 8
Cost after step   160:  0.4526000. Locality: 8
Cost after step   170:  0.1974000. Locality: 8
Trained:     8
Plateau'd:     0
--- New run! ---
Cost after step    10:  0.5488000. Locality: 1
---Switching Locality---
Cost after step    20:  0.8358000. Locality: 2
Cost after step    30:  0.7614000. Locality: 2
Cost after step    40:  0.6718000. Locality: 2
Cost after step    50:  0.4921000. Locality: 2
---Switching Locality---
---Switching Locality---
Cost after step    60:  0.6433000. Locality: 4
Cost after step    70:  0.7787000. Locality: 5
Cost after step    80:  0.5250000. Locality: 5
---Switching Locality---
Cost after step    90:  0.5270000. Locality: 6
---Switching Locality---
Cost after step   100:  0.9690000. Locality: 8
Cost after step   110:  0.9106000. Locality: 8
Cost after step   120:  0.7561000. Locality: 8
Cost after step   130:  0.4537000. Locality: 8
Cost after step   140:  0.1197000. Locality: 8
Trained:     9
Plateau'd:     0
--- New run! ---
Cost after step    10:  0.6759000. Locality: 2
---Switching Locality---
Cost after step    20:  0.8020000. Locality: 3
Cost after step    30:  0.6412000. Locality: 3
---Switching Locality---
Cost after step    40:  0.6920000. Locality: 4
---Switching Locality---
Cost after step    50:  0.7176000. Locality: 5
Cost after step    60:  0.5973000. Locality: 5
---Switching Locality---
Cost after step    70:  0.5469000. Locality: 6
Cost after step    80:  0.7019000. Locality: 7
Cost after step    90:  0.6006000. Locality: 7
Cost after step   100:  0.8612000. Locality: 8
Cost after step   110:  0.7023000. Locality: 8
Cost after step   120:  0.4108000. Locality: 8
Cost after step   130:  0.1081000. Locality: 8
Trained:    10
Plateau'd:     0
 </code>
 </pre>
 </details>

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_local_cost_functions.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The animation.avconv_args rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The animation.avconv_path rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The animation.html_args rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The keymap.all_axes rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The savefig.jpeg_quality rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The text.latex.preview rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The animation.avconv_args rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The animation.avconv_path rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The animation.html_args rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The keymap.all_axes rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The savefig.jpeg_quality rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The text.latex.preview rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
Cost after step     5:  1.0000000
Cost after step    10:  1.0000000
Cost after step    15:  1.0000000
Cost after step    20:  1.0000000
Cost after step    25:  1.0000000
Cost after step    30:  1.0000000
Cost after step    35:  1.0000000
Cost after step    40:  1.0000000
Cost after step    45:  1.0000000
Cost after step    50:  1.0000000
Cost after step    55:  1.0000000
Cost after step    60:  1.0000000
Cost after step    65:  1.0000000
Cost after step    70:  1.0000000
Cost after step    75:  1.0000000
Cost after step    80:  1.0000000
Cost after step    85:  1.0000000
Cost after step    90:  1.0000000
Cost after step    95:  1.0000000
Cost after step   100:  1.0000000
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The animation.avconv_args rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The animation.avconv_path rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The animation.html_args rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The keymap.all_axes rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The savefig.jpeg_quality rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The text.latex.preview rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
Cost after step     5:  0.9871000
Cost after step    10:  0.9651000
Cost after step    15:  0.9173000
Cost after step    20:  0.8059000
Cost after step    25:  0.6213000
Cost after step    30:  0.3703000
Cost after step    35:  0.1821000
Cost after step    40:  0.0684000
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The animation.avconv_args rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The animation.avconv_path rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The animation.html_args rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The keymap.all_axes rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The savefig.jpeg_quality rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The text.latex.preview rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
tensor(1., requires_grad=True)
Current cost: 0.9999999999972213.
Initial cost: 0.9999999999999843.
Difference: 2.763012041384627e-12
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The animation.avconv_args rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The animation.avconv_path rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The animation.html_args rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The keymap.all_axes rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The savefig.jpeg_quality rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The text.latex.preview rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
0.9957
Cost after step    10:  0.9909000. Locality: 2
Cost after step    20:  0.9753000. Locality: 2
Cost after step    30:  0.9275000. Locality: 2
Cost after step    40:  0.8386000. Locality: 2
Cost after step    50:  0.6821000. Locality: 2
Cost after step    60:  0.4353000. Locality: 2
Cost after step    70:  0.2264000. Locality: 2
Cost after step    80:  0.0923000. Locality: 2
---Switching Locality---
Cost after step    90:  0.9901000. Locality: 3
Cost after step   100:  0.9737000. Locality: 3
Cost after step   110:  0.9400000. Locality: 3
Cost after step   120:  0.8711000. Locality: 3
Cost after step   130:  0.7228000. Locality: 3
Cost after step   140:  0.5156000. Locality: 3
Cost after step   150:  0.2846000. Locality: 3
Cost after step   160:  0.1285000. Locality: 3
---Switching Locality---
Cost after step   170:  0.9899000. Locality: 4
Cost after step   180:  0.9799000. Locality: 4
Cost after step   190:  0.9512000. Locality: 4
Cost after step   200:  0.8964000. Locality: 4
Cost after step   210:  0.7683000. Locality: 4
Cost after step   220:  0.5752000. Locality: 4
Cost after step   230:  0.3314000. Locality: 4
Cost after step   240:  0.1575000. Locality: 4
---Switching Locality---
Cost after step   250:  0.9942000. Locality: 5
Cost after step   260:  0.9866000. Locality: 5
Cost after step   270:  0.9641000. Locality: 5
Cost after step   280:  0.9120000. Locality: 5
Cost after step   290:  0.8136000. Locality: 5
Cost after step   300:  0.6380000. Locality: 5
Cost after step   310:  0.4004000. Locality: 5
Cost after step   320:  0.1996000. Locality: 5
---Switching Locality---
Cost after step   330:  0.9945000. Locality: 6
Cost after step   340:  0.9873000. Locality: 6
Cost after step   350:  0.9689000. Locality: 6
Cost after step   360:  0.9288000. Locality: 6
Cost after step   370:  0.8476000. Locality: 6
Cost after step   380:  0.6711000. Locality: 6
Cost after step   390:  0.4527000. Locality: 6
Cost after step   400:  0.2342000. Locality: 6
Cost after step   410:  0.1014000. Locality: 6
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The animation.avconv_args rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The animation.avconv_path rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The animation.html_args rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The keymap.all_axes rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The savefig.jpeg_quality rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The text.latex.preview rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
--- New run! ---
Cost after step    20:  1.0000000
Cost after step    40:  1.0000000
Cost after step    60:  1.0000000
Cost after step    80:  1.0000000
Cost after step   100:  1.0000000
Cost after step   120:  1.0000000
Cost after step   140:  1.0000000
Cost after step   160:  1.0000000
Cost after step   180:  1.0000000
Cost after step   200:  1.0000000
Cost after step   220:  1.0000000
Cost after step   240:  1.0000000
Cost after step   260:  1.0000000
Cost after step   280:  1.0000000
Cost after step   300:  1.0000000
Cost after step   320:  1.0000000
Cost after step   340:  1.0000000
Cost after step   360:  1.0000000
Cost after step   380:  1.0000000
Cost after step   400:  1.0000000
Trained:     0
Plateau'd:     1
--- New run! ---
Cost after step    20:  0.9993000
Cost after step    40:  0.9994000
Cost after step    60:  0.9995000
Cost after step    80:  0.9994000
Cost after step   100:  0.9996000
Cost after step   120:  0.9995000
Cost after step   140:  0.9986000
Cost after step   160:  0.9993000
Cost after step   180:  0.9991000
Cost after step   200:  0.9996000
Cost after step   220:  0.9985000
Cost after step   240:  0.9989000
Cost after step   260:  0.9993000
Cost after step   280:  0.9991000
Cost after step   300:  0.9990000
Cost after step   320:  0.9987000
Cost after step   340:  0.9987000
Cost after step   360:  0.9994000
Cost after step   380:  0.9985000
Cost after step   400:  0.9991000
Trained:     0
Plateau'd:     2
--- New run! ---
Cost after step    20:  0.9976000
Cost after step    40:  0.9967000
Cost after step    60:  0.9970000
Cost after step    80:  0.9964000
Cost after step   100:  0.9956000
Cost after step   120:  0.9965000
Cost after step   140:  0.9958000
Cost after step   160:  0.9931000
Cost after step   180:  0.9940000
Cost after step   200:  0.9908000
Cost after step   220:  0.9887000
Cost after step   240:  0.9834000
Cost after step   260:  0.9821000
Cost after step   280:  0.9742000
Cost after step   300:  0.9576000
Cost after step   320:  0.9330000
Trained:     1
Plateau'd:     2
--- New run! ---
Cost after step    20:  0.9998000
Cost after step    40:  0.9997000
Cost after step    60:  0.9996000
Cost after step    80:  0.9994000
Cost after step   100:  0.9994000
Cost after step   120:  0.9996000
Cost after step   140:  0.9997000
Cost after step   160:  0.9992000
Cost after step   180:  0.9991000
Cost after step   200:  0.9992000
Cost after step   220:  0.9990000
Cost after step   240:  0.9986000
Cost after step   260:  0.9989000
Cost after step   280:  0.9988000
Cost after step   300:  0.9972000
Cost after step   320:  0.9982000
Cost after step   340:  0.9981000
Cost after step   360:  0.9979000
Cost after step   380:  0.9977000
Cost after step   400:  0.9972000
Trained:     1
Plateau'd:     3
--- New run! ---
Cost after step    20:  1.0000000
Cost after step    40:  1.0000000
Cost after step    60:  0.9999000
Cost after step    80:  1.0000000
Cost after step   100:  1.0000000
Cost after step   120:  1.0000000
Cost after step   140:  1.0000000
Cost after step   160:  0.9999000
Cost after step   180:  1.0000000
Cost after step   200:  1.0000000
Cost after step   220:  1.0000000
Cost after step   240:  1.0000000
Cost after step   260:  1.0000000
Cost after step   280:  1.0000000
Cost after step   300:  1.0000000
Cost after step   320:  1.0000000
Cost after step   340:  1.0000000
Cost after step   360:  0.9998000
Cost after step   380:  1.0000000
Cost after step   400:  1.0000000
Trained:     1
Plateau'd:     4
--- New run! ---
Cost after step    20:  0.9950000
Cost after step    40:  0.9957000
Cost after step    60:  0.9931000
Cost after step    80:  0.9920000
Cost after step   100:  0.9925000
Cost after step   120:  0.9908000
Cost after step   140:  0.9865000
Cost after step   160:  0.9861000
Cost after step   180:  0.9846000
Cost after step   200:  0.9767000
Cost after step   220:  0.9696000
Cost after step   240:  0.9560000
Cost after step   260:  0.9276000
Trained:     2
Plateau'd:     4
--- New run! ---
Cost after step    20:  0.9989000
Cost after step    40:  0.9979000
Cost after step    60:  0.9979000
Cost after step    80:  0.9982000
Cost after step   100:  0.9984000
Cost after step   120:  0.9986000
Cost after step   140:  0.9978000
Cost after step   160:  0.9976000
Cost after step   180:  0.9967000
Cost after step   200:  0.9972000
Cost after step   220:  0.9958000
Cost after step   240:  0.9966000
Cost after step   260:  0.9966000
Cost after step   280:  0.9952000
Cost after step   300:  0.9958000
Cost after step   320:  0.9972000
Cost after step   340:  0.9953000
Cost after step   360:  0.9934000
Cost after step   380:  0.9929000
Cost after step   400:  0.9916000
Trained:     2
Plateau'd:     5
--- New run! ---
Cost after step    20:  0.9988000
Cost after step    40:  0.9980000
Cost after step    60:  0.9978000
Cost after step    80:  0.9986000
Cost after step   100:  0.9978000
Cost after step   120:  0.9977000
Cost after step   140:  0.9974000
Cost after step   160:  0.9976000
Cost after step   180:  0.9975000
Cost after step   200:  0.9970000
Cost after step   220:  0.9972000
Cost after step   240:  0.9961000
Cost after step   260:  0.9960000
Cost after step   280:  0.9953000
Cost after step   300:  0.9947000
Cost after step   320:  0.9945000
Cost after step   340:  0.9912000
Cost after step   360:  0.9916000
Cost after step   380:  0.9866000
Cost after step   400:  0.9822000
Trained:     2
Plateau'd:     6
--- New run! ---
Cost after step    20:  1.0000000
Cost after step    40:  1.0000000
Cost after step    60:  1.0000000
Cost after step    80:  1.0000000
Cost after step   100:  1.0000000
Cost after step   120:  1.0000000
Cost after step   140:  1.0000000
Cost after step   160:  1.0000000
Cost after step   180:  1.0000000
Cost after step   200:  1.0000000
Cost after step   220:  1.0000000
Cost after step   240:  1.0000000
Cost after step   260:  1.0000000
Cost after step   280:  1.0000000
Cost after step   300:  1.0000000
Cost after step   320:  1.0000000
Cost after step   340:  1.0000000
Cost after step   360:  1.0000000
Cost after step   380:  1.0000000
Cost after step   400:  1.0000000
Trained:     2
Plateau'd:     7
--- New run! ---
Cost after step    20:  1.0000000
Cost after step    40:  1.0000000
Cost after step    60:  1.0000000
Cost after step    80:  1.0000000
Cost after step   100:  0.9999000
Cost after step   120:  1.0000000
Cost after step   140:  1.0000000
Cost after step   160:  1.0000000
Cost after step   180:  1.0000000
Cost after step   200:  1.0000000
Cost after step   220:  1.0000000
Cost after step   240:  1.0000000
Cost after step   260:  1.0000000
Cost after step   280:  1.0000000
Cost after step   300:  1.0000000
Cost after step   320:  1.0000000
Cost after step   340:  1.0000000
Cost after step   360:  1.0000000
Cost after step   380:  1.0000000
Cost after step   400:  1.0000000
Trained:     2
Plateau'd:     8
--- New run! ---
---Switching Locality---
Cost after step    10:  0.6137000. Locality: 2
---Switching Locality---
Cost after step    20:  0.5159000. Locality: 3
---Switching Locality---
Cost after step    30:  0.7040000. Locality: 4
Cost after step    40:  0.5823000. Locality: 4
Cost after step    50:  0.5055000. Locality: 5
---Switching Locality---
Cost after step    60:  0.6965000. Locality: 6
Cost after step    70:  0.5823000. Locality: 6
Cost after step    80:  0.8792000. Locality: 7
Cost after step    90:  0.7172000. Locality: 7
---Switching Locality---
Cost after step   100:  0.9741000. Locality: 8
Cost after step   110:  0.9329000. Locality: 8
Cost after step   120:  0.8278000. Locality: 8
Cost after step   130:  0.5973000. Locality: 8
Cost after step   140:  0.2649000. Locality: 8
Trained:     1
Plateau'd:     0
--- New run! ---
---Switching Locality---
Cost after step    10:  0.6273000. Locality: 2
---Switching Locality---
Cost after step    20:  0.9218000. Locality: 4
Cost after step    30:  0.8247000. Locality: 4
Cost after step    40:  0.5992000. Locality: 4
Cost after step    50:  0.6852000. Locality: 5
Cost after step    60:  0.6003000. Locality: 5
Cost after step    70:  0.5220000. Locality: 5
Cost after step    80:  0.5100000. Locality: 6
---Switching Locality---
Cost after step    90:  0.4677000. Locality: 8
Cost after step   100:  0.1562000. Locality: 8
Trained:     2
Plateau'd:     0
--- New run! ---
---Switching Locality---
---Switching Locality---
Cost after step    10:  0.5405000. Locality: 5
---Switching Locality---
Cost after step    20:  0.6024000. Locality: 6
---Switching Locality---
Cost after step    30:  0.6060000. Locality: 7
---Switching Locality---
Cost after step    40:  0.5009000. Locality: 8
Cost after step    50:  0.1676000. Locality: 8
Trained:     3
Plateau'd:     0
--- New run! ---
Cost after step    10:  0.4871000. Locality: 1
---Switching Locality---
Cost after step    20:  0.4948000. Locality: 2
---Switching Locality---
Cost after step    30:  0.6627000. Locality: 3
---Switching Locality---
Cost after step    40:  0.5826000. Locality: 4
---Switching Locality---
Cost after step    50:  0.5234000. Locality: 5
---Switching Locality---
---Switching Locality---
Cost after step    60:  0.6394000. Locality: 7
---Switching Locality---
Cost after step    70:  0.3126000. Locality: 8
Trained:     4
Plateau'd:     0
--- New run! ---
Cost after step    10:  0.5485000. Locality: 1
Cost after step    20:  0.6393000. Locality: 2
Cost after step    30:  0.8061000. Locality: 3
Cost after step    40:  0.7073000. Locality: 3
Cost after step    50:  0.6144000. Locality: 3
Cost after step    60:  0.4888000. Locality: 3
---Switching Locality---
Cost after step    70:  0.6102000. Locality: 4
---Switching Locality---
Cost after step    80:  0.4909000. Locality: 5
---Switching Locality---
Cost after step    90:  0.7897000. Locality: 6
Cost after step   100:  0.4939000. Locality: 6
---Switching Locality---
Cost after step   110:  0.7063000. Locality: 7
Cost after step   120:  0.5323000. Locality: 7
---Switching Locality---
Cost after step   130:  0.2921000. Locality: 8
Trained:     5
Plateau'd:     0
--- New run! ---
---Switching Locality---
Cost after step    10:  0.7449000. Locality: 2
Cost after step    20:  0.5005000. Locality: 2
---Switching Locality---
---Switching Locality---
Cost after step    30:  0.7292000. Locality: 5
Cost after step    40:  0.4696000. Locality: 5
---Switching Locality---
Cost after step    50:  0.5099000. Locality: 7
---Switching Locality---
Cost after step    60:  0.6587000. Locality: 8
Cost after step    70:  0.4912000. Locality: 8
Cost after step    80:  0.2440000. Locality: 8
Trained:     6
Plateau'd:     0
--- New run! ---
---Switching Locality---
Cost after step    10:  0.8510000. Locality: 3
Cost after step    20:  0.7255000. Locality: 3
Cost after step    30:  0.5811000. Locality: 3
---Switching Locality---
Cost after step    40:  0.6300000. Locality: 4
---Switching Locality---
Cost after step    50:  0.8801000. Locality: 5
Cost after step    60:  0.7395000. Locality: 5
---Switching Locality---
Cost after step    70:  0.8948000. Locality: 6
Cost after step    80:  0.7399000. Locality: 6
Cost after step    90:  0.7139000. Locality: 7
Cost after step   100:  0.5959000. Locality: 7
Cost after step   110:  0.5939000. Locality: 8
Cost after step   120:  0.2906000. Locality: 8
Trained:     7
Plateau'd:     0
--- New run! ---
---Switching Locality---
 </code>
 </pre>
 </details>

---

## 13. tutorial_measurement_optimize.html <a name="demo12"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_measurement_optimize.html):

```
Cost function value: -0.5657291586837982
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_measurement_optimize.html):

```
Cost function value: -0.33214936351527113
```

---

