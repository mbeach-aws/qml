Last update: 2022-11-12  23:26:41 (All times shown in Eastern time)
# List of differences in demonstration outputs

# Table of contents

1. [tutorial_adaptive_circuits.html](#demo0)
2. [tutorial_tn_circuits.html](#demo1)
3. [tutorial_qnn_module_tf.html](#demo2)
4. [tutorial_quantum_circuit_cutting.html](#demo3)
5. [tutorial_quantum_transfer_learning.html](#demo4)
6. [tutorial_local_cost_functions.html](#demo5)
7. [tutorial_noisy_circuit_optimization.html](#demo6)
8. [tutorial_vqe_spin_sectors.html](#demo7)
9. [tutorial_backprop.html](#demo8)
10. [tutorial_measurement_optimize.html](#demo9)
11. [tutorial_error_mitigation.html](#demo10)
12. [tutorial_qft_arithmetics.html](#demo11)
13. [tutorial_jax_transformations.html](#demo12)
14. [tutorial_quanvolution.html](#demo13)
15. [tutorial_qubit_tapering.html](#demo14)
16. [tutorial_quantum_chemistry.html](#demo15)


Number of demos different/all demos: 16/71

## 1. tutorial_adaptive_circuits.html <a name="demo0"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_adaptive_circuits.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
n = 0,  E = -7.86266588 H, t = 0.74 s
n = 1,  E = -7.87094622 H, t = 0.74 s
n = 2,  E = -7.87563101 H, t = 0.74 s
n = 3,  E = -7.87829148 H, t = 0.74 s
n = 4,  E = -7.87981707 H, t = 0.91 s
n = 5,  E = -7.88070478 H, t = 0.74 s
n = 6,  E = -7.88123144 H, t = 0.74 s
n = 7,  E = -7.88155162 H, t = 0.74 s
n = 8,  E = -7.88175219 H, t = 0.74 s
n = 9,  E = -7.88188238 H, t = 0.90 s
n = 10,  E = -7.88197042 H, t = 0.74 s
n = 11,  E = -7.88203269 H, t = 0.74 s
n = 12,  E = -7.88207881 H, t = 0.74 s
n = 13,  E = -7.88211453 H, t = 0.74 s
n = 14,  E = -7.88214336 H, t = 0.90 s
n = 15,  E = -7.88216745 H, t = 0.74 s
n = 16,  E = -7.88218815 H, t = 0.74 s
n = 17,  E = -7.88220635 H, t = 0.74 s
n = 18,  E = -7.88222262 H, t = 0.90 s
n = 19,  E = -7.88223735 H, t = 0.74 s
n = 0,  E = -7.86266588 H, t = 0.14 s
n = 1,  E = -7.87094622 H, t = 0.14 s
n = 2,  E = -7.87563101 H, t = 0.14 s
n = 3,  E = -7.87829148 H, t = 0.14 s
n = 4,  E = -7.87981707 H, t = 0.14 s
n = 5,  E = -7.88070478 H, t = 0.14 s
n = 6,  E = -7.88123144 H, t = 0.14 s
n = 7,  E = -7.88155162 H, t = 0.14 s
n = 8,  E = -7.88175219 H, t = 0.14 s
n = 9,  E = -7.88188238 H, t = 0.14 s
n = 10,  E = -7.88197042 H, t = 0.14 s
n = 11,  E = -7.88203269 H, t = 0.14 s
n = 12,  E = -7.88207881 H, t = 0.14 s
n = 13,  E = -7.88211453 H, t = 0.14 s
n = 14,  E = -7.88214336 H, t = 0.14 s
n = 15,  E = -7.88216745 H, t = 0.14 s
n = 16,  E = -7.88218815 H, t = 0.14 s
n = 17,  E = -7.88220635 H, t = 0.14 s
n = 18,  E = -7.88222262 H, t = 0.14 s
n = 19,  E = -7.88223735 H, t = 0.14 s
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
n = 0,  E = -7.86266588 H, t = 0.79 s
n = 1,  E = -7.87094622 H, t = 0.78 s
n = 2,  E = -7.87563101 H, t = 0.78 s
n = 3,  E = -7.87829148 H, t = 1.04 s
n = 4,  E = -7.87981707 H, t = 0.78 s
n = 5,  E = -7.88070478 H, t = 0.78 s
n = 6,  E = -7.88123144 H, t = 0.79 s
n = 7,  E = -7.88155162 H, t = 1.06 s
n = 8,  E = -7.88175219 H, t = 0.78 s
n = 9,  E = -7.88188238 H, t = 0.78 s
n = 10,  E = -7.88197042 H, t = 0.78 s
n = 11,  E = -7.88203269 H, t = 0.78 s
n = 12,  E = -7.88207881 H, t = 1.04 s
n = 13,  E = -7.88211453 H, t = 0.79 s
n = 14,  E = -7.88214336 H, t = 0.79 s
n = 15,  E = -7.88216745 H, t = 0.79 s
n = 16,  E = -7.88218815 H, t = 0.79 s
n = 17,  E = -7.88220635 H, t = 1.04 s
n = 18,  E = -7.88222262 H, t = 0.79 s
n = 19,  E = -7.88223735 H, t = 0.79 s
n = 0,  E = -7.86266588 H, t = 0.16 s
n = 1,  E = -7.87094622 H, t = 0.16 s
n = 2,  E = -7.87563101 H, t = 0.16 s
n = 3,  E = -7.87829148 H, t = 0.16 s
n = 4,  E = -7.87981707 H, t = 0.16 s
n = 5,  E = -7.88070478 H, t = 0.16 s
n = 6,  E = -7.88123144 H, t = 0.16 s
n = 7,  E = -7.88155162 H, t = 0.15 s
n = 8,  E = -7.88175219 H, t = 0.16 s
n = 9,  E = -7.88188238 H, t = 0.16 s
n = 10,  E = -7.88197042 H, t = 0.16 s
n = 11,  E = -7.88203269 H, t = 0.16 s
n = 12,  E = -7.88207881 H, t = 0.16 s
n = 13,  E = -7.88211453 H, t = 0.15 s
n = 14,  E = -7.88214336 H, t = 0.16 s
n = 15,  E = -7.88216745 H, t = 0.16 s
n = 16,  E = -7.88218815 H, t = 0.16 s
n = 17,  E = -7.88220635 H, t = 0.16 s
n = 18,  E = -7.88222262 H, t = 0.15 s
n = 19,  E = -7.88223735 H, t = 0.16 s
 </code>
 </pre>
 </details>

---

## 2. tutorial_tn_circuits.html <a name="demo1"></a>

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

## 3. tutorial_qnn_module_tf.html <a name="demo2"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_qnn_module_tf.html):

```
30/30 - 9s - loss: 0.4997 - accuracy: 0.5000 - val_loss: 0.5081 - val_accuracy: 0.4400 - 9s/epoch - 298ms/step
30/30 - 9s - loss: 0.4673 - accuracy: 0.6200 - val_loss: 0.4488 - val_accuracy: 0.6200 - 9s/epoch - 298ms/step
30/30 - 9s - loss: 0.3230 - accuracy: 0.8267 - val_loss: 0.2562 - val_accuracy: 0.8400 - 9s/epoch - 302ms/step
30/30 - 9s - loss: 0.2124 - accuracy: 0.8867 - val_loss: 0.1997 - val_accuracy: 0.8400 - 9s/epoch - 297ms/step
30/30 - 9s - loss: 0.1800 - accuracy: 0.8933 - val_loss: 0.1841 - val_accuracy: 0.8400 - 9s/epoch - 297ms/step
30/30 - 9s - loss: 0.1593 - accuracy: 0.8667 - val_loss: 0.2177 - val_accuracy: 0.8400 - 9s/epoch - 297ms/step
30/30 - 18s - loss: 0.5189 - accuracy: 0.4000 - val_loss: 0.4945 - val_accuracy: 0.5400 - 18s/epoch - 596ms/step
30/30 - 18s - loss: 0.4822 - accuracy: 0.6200 - val_loss: 0.4412 - val_accuracy: 0.7200 - 18s/epoch - 597ms/step
30/30 - 18s - loss: 0.3850 - accuracy: 0.7133 - val_loss: 0.2898 - val_accuracy: 0.7800 - 18s/epoch - 600ms/step
30/30 - 18s - loss: 0.2720 - accuracy: 0.7867 - val_loss: 0.2185 - val_accuracy: 0.8200 - 18s/epoch - 589ms/step
30/30 - 18s - loss: 0.2056 - accuracy: 0.8400 - val_loss: 0.1893 - val_accuracy: 0.8400 - 18s/epoch - 600ms/step
30/30 - 18s - loss: 0.1753 - accuracy: 0.8400 - val_loss: 0.1856 - val_accuracy: 0.8400 - 18s/epoch - 595ms/step
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_qnn_module_tf.html):

```
30/30 - 11s - loss: 0.4997 - accuracy: 0.5000 - val_loss: 0.5081 - val_accuracy: 0.4400 - 11s/epoch - 373ms/step
30/30 - 11s - loss: 0.4673 - accuracy: 0.6200 - val_loss: 0.4488 - val_accuracy: 0.6200 - 11s/epoch - 374ms/step
30/30 - 11s - loss: 0.3230 - accuracy: 0.8267 - val_loss: 0.2562 - val_accuracy: 0.8400 - 11s/epoch - 371ms/step
30/30 - 11s - loss: 0.2124 - accuracy: 0.8867 - val_loss: 0.1997 - val_accuracy: 0.8400 - 11s/epoch - 379ms/step
30/30 - 11s - loss: 0.1800 - accuracy: 0.8933 - val_loss: 0.1841 - val_accuracy: 0.8400 - 11s/epoch - 371ms/step
30/30 - 12s - loss: 0.1593 - accuracy: 0.8667 - val_loss: 0.2177 - val_accuracy: 0.8400 - 12s/epoch - 404ms/step
30/30 - 22s - loss: 0.5189 - accuracy: 0.4000 - val_loss: 0.4945 - val_accuracy: 0.5400 - 22s/epoch - 744ms/step
30/30 - 22s - loss: 0.4822 - accuracy: 0.6200 - val_loss: 0.4412 - val_accuracy: 0.7200 - 22s/epoch - 743ms/step
30/30 - 22s - loss: 0.3850 - accuracy: 0.7133 - val_loss: 0.2898 - val_accuracy: 0.7800 - 22s/epoch - 748ms/step
30/30 - 22s - loss: 0.2720 - accuracy: 0.7867 - val_loss: 0.2185 - val_accuracy: 0.8200 - 22s/epoch - 748ms/step
30/30 - 22s - loss: 0.2056 - accuracy: 0.8400 - val_loss: 0.1893 - val_accuracy: 0.8400 - 22s/epoch - 732ms/step
30/30 - 22s - loss: 0.1753 - accuracy: 0.8400 - val_loss: 0.1856 - val_accuracy: 0.8400 - 22s/epoch - 746ms/step
```

---

## 4. tutorial_quantum_circuit_cutting.html <a name="demo3"></a>

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

## 5. tutorial_quantum_transfer_learning.html <a name="demo4"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_quantum_transfer_learning.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
  0%|          | 160k/44.7M [00:00<00:29, 1.57MB/s]
  2%|2         | 928k/44.7M [00:00<00:08, 5.21MB/s]
 11%|#1        | 5.13M/44.7M [00:00<00:01, 22.9MB/s]
 28%|##7       | 12.4M/44.7M [00:00<00:00, 43.7MB/s]
 44%|####3     | 19.5M/44.7M [00:00<00:00, 55.0MB/s]
 60%|######    | 26.9M/44.7M [00:00<00:00, 62.3MB/s]
 76%|#######6  | 34.2M/44.7M [00:00<00:00, 66.8MB/s]
 93%|#########3| 41.6M/44.7M [00:00<00:00, 70.2MB/s]
100%|##########| 44.7M/44.7M [00:00<00:00, 55.2MB/s]
Training started:
Phase: train Epoch: 1/3 Iter: 1/62 Batch time: 0.3690
Phase: train Epoch: 1/3 Iter: 2/62 Batch time: 0.3405
Phase: train Epoch: 1/3 Iter: 3/62 Batch time: 0.3410
Phase: train Epoch: 1/3 Iter: 4/62 Batch time: 0.3414
Phase: train Epoch: 1/3 Iter: 5/62 Batch time: 0.3412
Phase: train Epoch: 1/3 Iter: 6/62 Batch time: 0.3421
Phase: train Epoch: 1/3 Iter: 7/62 Batch time: 0.3414
Phase: train Epoch: 1/3 Iter: 8/62 Batch time: 0.3426
Phase: train Epoch: 1/3 Iter: 9/62 Batch time: 0.3435
Phase: train Epoch: 1/3 Iter: 10/62 Batch time: 0.3426
Phase: train Epoch: 1/3 Iter: 11/62 Batch time: 0.3419
Phase: train Epoch: 1/3 Iter: 12/62 Batch time: 0.3415
Phase: train Epoch: 1/3 Iter: 13/62 Batch time: 0.3423
Phase: train Epoch: 1/3 Iter: 14/62 Batch time: 0.3422
Phase: train Epoch: 1/3 Iter: 15/62 Batch time: 0.3421
Phase: train Epoch: 1/3 Iter: 16/62 Batch time: 0.3419
Phase: train Epoch: 1/3 Iter: 17/62 Batch time: 0.3431
Phase: train Epoch: 1/3 Iter: 18/62 Batch time: 0.3425
Phase: train Epoch: 1/3 Iter: 19/62 Batch time: 0.3424
Phase: train Epoch: 1/3 Iter: 20/62 Batch time: 0.3429
Phase: train Epoch: 1/3 Iter: 21/62 Batch time: 0.3417
Phase: train Epoch: 1/3 Iter: 22/62 Batch time: 0.3475
Phase: train Epoch: 1/3 Iter: 23/62 Batch time: 0.3424
Phase: train Epoch: 1/3 Iter: 24/62 Batch time: 0.3419
Phase: train Epoch: 1/3 Iter: 25/62 Batch time: 0.3427
Phase: train Epoch: 1/3 Iter: 26/62 Batch time: 0.3416
Phase: train Epoch: 1/3 Iter: 27/62 Batch time: 0.3423
Phase: train Epoch: 1/3 Iter: 28/62 Batch time: 0.3431
Phase: train Epoch: 1/3 Iter: 29/62 Batch time: 0.3421
Phase: train Epoch: 1/3 Iter: 30/62 Batch time: 0.3426
Phase: train Epoch: 1/3 Iter: 31/62 Batch time: 0.3426
Phase: train Epoch: 1/3 Iter: 32/62 Batch time: 0.3434
Phase: train Epoch: 1/3 Iter: 33/62 Batch time: 0.3426
Phase: train Epoch: 1/3 Iter: 34/62 Batch time: 0.3432
Phase: train Epoch: 1/3 Iter: 35/62 Batch time: 0.3422
Phase: train Epoch: 1/3 Iter: 36/62 Batch time: 0.3421
Phase: train Epoch: 1/3 Iter: 37/62 Batch time: 0.3424
Phase: train Epoch: 1/3 Iter: 38/62 Batch time: 0.3420
Phase: train Epoch: 1/3 Iter: 39/62 Batch time: 0.3436
Phase: train Epoch: 1/3 Iter: 40/62 Batch time: 0.3424
Phase: train Epoch: 1/3 Iter: 41/62 Batch time: 0.3427
Phase: train Epoch: 1/3 Iter: 42/62 Batch time: 0.3422
Phase: train Epoch: 1/3 Iter: 43/62 Batch time: 0.3417
Phase: train Epoch: 1/3 Iter: 44/62 Batch time: 0.3420
Phase: train Epoch: 1/3 Iter: 45/62 Batch time: 0.3418
Phase: train Epoch: 1/3 Iter: 46/62 Batch time: 0.3430
Phase: train Epoch: 1/3 Iter: 47/62 Batch time: 0.3421
Phase: train Epoch: 1/3 Iter: 48/62 Batch time: 0.3420
Phase: train Epoch: 1/3 Iter: 49/62 Batch time: 0.3413
Phase: train Epoch: 1/3 Iter: 50/62 Batch time: 0.3413
Phase: train Epoch: 1/3 Iter: 51/62 Batch time: 0.3405
Phase: train Epoch: 1/3 Iter: 52/62 Batch time: 0.3421
Phase: train Epoch: 1/3 Iter: 53/62 Batch time: 0.3429
Phase: train Epoch: 1/3 Iter: 54/62 Batch time: 0.3423
Phase: train Epoch: 1/3 Iter: 55/62 Batch time: 0.3427
Phase: train Epoch: 1/3 Iter: 56/62 Batch time: 0.3422
Phase: train Epoch: 1/3 Iter: 57/62 Batch time: 0.3416
Phase: train Epoch: 1/3 Iter: 58/62 Batch time: 0.3429
Phase: train Epoch: 1/3 Iter: 59/62 Batch time: 0.3422
Phase: train Epoch: 1/3 Iter: 60/62 Batch time: 0.3422
Phase: train Epoch: 1/3 Iter: 61/62 Batch time: 0.3431
Phase: train Epoch: 1/3 Loss: 0.6990 Acc: 0.5246
Phase: validation Epoch: 1/3 Iter: 1/39 Batch time: 0.2713
Phase: validation Epoch: 1/3 Iter: 2/39 Batch time: 0.2688
Phase: validation Epoch: 1/3 Iter: 3/39 Batch time: 0.2691
Phase: validation Epoch: 1/3 Iter: 4/39 Batch time: 0.2696
Phase: validation Epoch: 1/3 Iter: 5/39 Batch time: 0.2686
Phase: validation Epoch: 1/3 Iter: 6/39 Batch time: 0.2687
Phase: validation Epoch: 1/3 Iter: 7/39 Batch time: 0.2689
Phase: validation Epoch: 1/3 Iter: 8/39 Batch time: 0.2694
Phase: validation Epoch: 1/3 Iter: 9/39 Batch time: 0.2694
Phase: validation Epoch: 1/3 Iter: 10/39 Batch time: 0.2685
Phase: validation Epoch: 1/3 Iter: 11/39 Batch time: 0.2694
Phase: validation Epoch: 1/3 Iter: 12/39 Batch time: 0.2687
Phase: validation Epoch: 1/3 Iter: 13/39 Batch time: 0.2693
Phase: validation Epoch: 1/3 Iter: 14/39 Batch time: 0.2689
Phase: validation Epoch: 1/3 Iter: 15/39 Batch time: 0.2691
Phase: validation Epoch: 1/3 Iter: 16/39 Batch time: 0.2688
Phase: validation Epoch: 1/3 Iter: 17/39 Batch time: 0.2691
Phase: validation Epoch: 1/3 Iter: 18/39 Batch time: 0.2692
Phase: validation Epoch: 1/3 Iter: 19/39 Batch time: 0.2695
Phase: validation Epoch: 1/3 Iter: 20/39 Batch time: 0.2687
Phase: validation Epoch: 1/3 Iter: 21/39 Batch time: 0.2688
Phase: validation Epoch: 1/3 Iter: 22/39 Batch time: 0.2686
Phase: validation Epoch: 1/3 Iter: 23/39 Batch time: 0.2672
Phase: validation Epoch: 1/3 Iter: 24/39 Batch time: 0.2688
Phase: validation Epoch: 1/3 Iter: 25/39 Batch time: 0.2682
Phase: validation Epoch: 1/3 Iter: 26/39 Batch time: 0.2683
Phase: validation Epoch: 1/3 Iter: 27/39 Batch time: 0.2685
Phase: validation Epoch: 1/3 Iter: 28/39 Batch time: 0.2690
Phase: validation Epoch: 1/3 Iter: 29/39 Batch time: 0.2686
Phase: validation Epoch: 1/3 Iter: 30/39 Batch time: 0.2684
Phase: validation Epoch: 1/3 Iter: 31/39 Batch time: 0.2687
Phase: validation Epoch: 1/3 Iter: 32/39 Batch time: 0.2688
Phase: validation Epoch: 1/3 Iter: 33/39 Batch time: 0.2691
Phase: validation Epoch: 1/3 Iter: 34/39 Batch time: 0.2689
Phase: validation Epoch: 1/3 Iter: 35/39 Batch time: 0.2689
Phase: validation Epoch: 1/3 Iter: 36/39 Batch time: 0.2677
Phase: validation Epoch: 1/3 Iter: 37/39 Batch time: 0.2688
Phase: validation Epoch: 1/3 Iter: 38/39 Batch time: 0.2693
Phase: validation Epoch: 1/3 Iter: 39/39 Batch time: 0.0831
Phase: validation   Epoch: 1/3 Loss: 0.6429 Acc: 0.6536
Phase: train Epoch: 2/3 Iter: 1/62 Batch time: 0.3399
Phase: train Epoch: 2/3 Iter: 2/62 Batch time: 0.3418
Phase: train Epoch: 2/3 Iter: 3/62 Batch time: 0.3422
Phase: train Epoch: 2/3 Iter: 4/62 Batch time: 0.3435
Phase: train Epoch: 2/3 Iter: 5/62 Batch time: 0.3436
Phase: train Epoch: 2/3 Iter: 6/62 Batch time: 0.3420
Phase: train Epoch: 2/3 Iter: 7/62 Batch time: 0.3449
Phase: train Epoch: 2/3 Iter: 8/62 Batch time: 0.3434
Phase: train Epoch: 2/3 Iter: 9/62 Batch time: 0.3440
Phase: train Epoch: 2/3 Iter: 10/62 Batch time: 0.3430
Phase: train Epoch: 2/3 Iter: 11/62 Batch time: 0.3441
Phase: train Epoch: 2/3 Iter: 12/62 Batch time: 0.3439
Phase: train Epoch: 2/3 Iter: 13/62 Batch time: 0.3436
Phase: train Epoch: 2/3 Iter: 14/62 Batch time: 0.3436
Phase: train Epoch: 2/3 Iter: 15/62 Batch time: 0.3437
Phase: train Epoch: 2/3 Iter: 16/62 Batch time: 0.3446
Phase: train Epoch: 2/3 Iter: 17/62 Batch time: 0.3436
Phase: train Epoch: 2/3 Iter: 18/62 Batch time: 0.3441
Phase: train Epoch: 2/3 Iter: 19/62 Batch time: 0.3441
Phase: train Epoch: 2/3 Iter: 20/62 Batch time: 0.3444
Phase: train Epoch: 2/3 Iter: 21/62 Batch time: 0.3438
Phase: train Epoch: 2/3 Iter: 22/62 Batch time: 0.3446
Phase: train Epoch: 2/3 Iter: 23/62 Batch time: 0.3427
Phase: train Epoch: 2/3 Iter: 24/62 Batch time: 0.3441
Phase: train Epoch: 2/3 Iter: 25/62 Batch time: 0.3440
Phase: train Epoch: 2/3 Iter: 26/62 Batch time: 0.3440
Phase: train Epoch: 2/3 Iter: 27/62 Batch time: 0.3424
Phase: train Epoch: 2/3 Iter: 28/62 Batch time: 0.3440
Phase: train Epoch: 2/3 Iter: 29/62 Batch time: 0.3435
Phase: train Epoch: 2/3 Iter: 30/62 Batch time: 0.3442
Phase: train Epoch: 2/3 Iter: 31/62 Batch time: 0.3444
Phase: train Epoch: 2/3 Iter: 32/62 Batch time: 0.3442
Phase: train Epoch: 2/3 Iter: 33/62 Batch time: 0.3433
Phase: train Epoch: 2/3 Iter: 34/62 Batch time: 0.3432
Phase: train Epoch: 2/3 Iter: 35/62 Batch time: 0.3433
Phase: train Epoch: 2/3 Iter: 36/62 Batch time: 0.3435
Phase: train Epoch: 2/3 Iter: 37/62 Batch time: 0.3441
Phase: train Epoch: 2/3 Iter: 38/62 Batch time: 0.3459
Phase: train Epoch: 2/3 Iter: 39/62 Batch time: 0.3448
Phase: train Epoch: 2/3 Iter: 40/62 Batch time: 0.3434
Phase: train Epoch: 2/3 Iter: 41/62 Batch time: 0.3449
Phase: train Epoch: 2/3 Iter: 42/62 Batch time: 0.3438
Phase: train Epoch: 2/3 Iter: 43/62 Batch time: 0.3440
Phase: train Epoch: 2/3 Iter: 44/62 Batch time: 0.3442
Phase: train Epoch: 2/3 Iter: 45/62 Batch time: 0.3434
Phase: train Epoch: 2/3 Iter: 46/62 Batch time: 0.3439
Phase: train Epoch: 2/3 Iter: 47/62 Batch time: 0.3443
Phase: train Epoch: 2/3 Iter: 48/62 Batch time: 0.3433
Phase: train Epoch: 2/3 Iter: 49/62 Batch time: 0.3438
Phase: train Epoch: 2/3 Iter: 50/62 Batch time: 0.3440
Phase: train Epoch: 2/3 Iter: 51/62 Batch time: 0.3436
Phase: train Epoch: 2/3 Iter: 52/62 Batch time: 0.3453
Phase: train Epoch: 2/3 Iter: 53/62 Batch time: 0.3441
Phase: train Epoch: 2/3 Iter: 54/62 Batch time: 0.3453
Phase: train Epoch: 2/3 Iter: 55/62 Batch time: 0.3440
Phase: train Epoch: 2/3 Iter: 56/62 Batch time: 0.3457
Phase: train Epoch: 2/3 Iter: 57/62 Batch time: 0.3443
Phase: train Epoch: 2/3 Iter: 58/62 Batch time: 0.3440
Phase: train Epoch: 2/3 Iter: 59/62 Batch time: 0.3441
Phase: train Epoch: 2/3 Iter: 60/62 Batch time: 0.3445
Phase: train Epoch: 2/3 Iter: 61/62 Batch time: 0.3439
Phase: train Epoch: 2/3 Loss: 0.6134 Acc: 0.7008
Phase: validation Epoch: 2/3 Iter: 1/39 Batch time: 0.2731
Phase: validation Epoch: 2/3 Iter: 2/39 Batch time: 0.2701
Phase: validation Epoch: 2/3 Iter: 3/39 Batch time: 0.2700
Phase: validation Epoch: 2/3 Iter: 4/39 Batch time: 0.2706
Phase: validation Epoch: 2/3 Iter: 5/39 Batch time: 0.2706
Phase: validation Epoch: 2/3 Iter: 6/39 Batch time: 0.2697
Phase: validation Epoch: 2/3 Iter: 7/39 Batch time: 0.2697
Phase: validation Epoch: 2/3 Iter: 8/39 Batch time: 0.2705
Phase: validation Epoch: 2/3 Iter: 9/39 Batch time: 0.2706
Phase: validation Epoch: 2/3 Iter: 10/39 Batch time: 0.2698
Phase: validation Epoch: 2/3 Iter: 11/39 Batch time: 0.2695
Phase: validation Epoch: 2/3 Iter: 12/39 Batch time: 0.2710
Phase: validation Epoch: 2/3 Iter: 13/39 Batch time: 0.2700
Phase: validation Epoch: 2/3 Iter: 14/39 Batch time: 0.2701
Phase: validation Epoch: 2/3 Iter: 15/39 Batch time: 0.2695
Phase: validation Epoch: 2/3 Iter: 16/39 Batch time: 0.2700
Phase: validation Epoch: 2/3 Iter: 17/39 Batch time: 0.2713
Phase: validation Epoch: 2/3 Iter: 18/39 Batch time: 0.2702
Phase: validation Epoch: 2/3 Iter: 19/39 Batch time: 0.2696
Phase: validation Epoch: 2/3 Iter: 20/39 Batch time: 0.2699
Phase: validation Epoch: 2/3 Iter: 21/39 Batch time: 0.2697
Phase: validation Epoch: 2/3 Iter: 22/39 Batch time: 0.2697
Phase: validation Epoch: 2/3 Iter: 23/39 Batch time: 0.2705
Phase: validation Epoch: 2/3 Iter: 24/39 Batch time: 0.2698
Phase: validation Epoch: 2/3 Iter: 25/39 Batch time: 0.2695
Phase: validation Epoch: 2/3 Iter: 26/39 Batch time: 0.2706
Phase: validation Epoch: 2/3 Iter: 27/39 Batch time: 0.2701
Phase: validation Epoch: 2/3 Iter: 28/39 Batch time: 0.2697
Phase: validation Epoch: 2/3 Iter: 29/39 Batch time: 0.2704
Phase: validation Epoch: 2/3 Iter: 30/39 Batch time: 0.2706
Phase: validation Epoch: 2/3 Iter: 31/39 Batch time: 0.2703
Phase: validation Epoch: 2/3 Iter: 32/39 Batch time: 0.2698
Phase: validation Epoch: 2/3 Iter: 33/39 Batch time: 0.2695
Phase: validation Epoch: 2/3 Iter: 34/39 Batch time: 0.2699
Phase: validation Epoch: 2/3 Iter: 35/39 Batch time: 0.2691
Phase: validation Epoch: 2/3 Iter: 36/39 Batch time: 0.2708
Phase: validation Epoch: 2/3 Iter: 37/39 Batch time: 0.2698
Phase: validation Epoch: 2/3 Iter: 38/39 Batch time: 0.2707
Phase: validation Epoch: 2/3 Iter: 39/39 Batch time: 0.0784
Phase: validation   Epoch: 2/3 Loss: 0.5389 Acc: 0.8235
Phase: train Epoch: 3/3 Iter: 1/62 Batch time: 0.3392
Phase: train Epoch: 3/3 Iter: 2/62 Batch time: 0.3419
Phase: train Epoch: 3/3 Iter: 3/62 Batch time: 0.3425
Phase: train Epoch: 3/3 Iter: 4/62 Batch time: 0.3436
Phase: train Epoch: 3/3 Iter: 5/62 Batch time: 0.3437
Phase: train Epoch: 3/3 Iter: 6/62 Batch time: 0.3444
Phase: train Epoch: 3/3 Iter: 7/62 Batch time: 0.3440
Phase: train Epoch: 3/3 Iter: 8/62 Batch time: 0.3424
Phase: train Epoch: 3/3 Iter: 9/62 Batch time: 0.3440
Phase: train Epoch: 3/3 Iter: 10/62 Batch time: 0.3442
Phase: train Epoch: 3/3 Iter: 11/62 Batch time: 0.3440
Phase: train Epoch: 3/3 Iter: 12/62 Batch time: 0.3436
Phase: train Epoch: 3/3 Iter: 13/62 Batch time: 0.3450
Phase: train Epoch: 3/3 Iter: 14/62 Batch time: 0.3437
Phase: train Epoch: 3/3 Iter: 15/62 Batch time: 0.3448
Phase: train Epoch: 3/3 Iter: 16/62 Batch time: 0.3442
Phase: train Epoch: 3/3 Iter: 17/62 Batch time: 0.3444
Phase: train Epoch: 3/3 Iter: 18/62 Batch time: 0.3428
Phase: train Epoch: 3/3 Iter: 19/62 Batch time: 0.3437
Phase: train Epoch: 3/3 Iter: 20/62 Batch time: 0.3442
Phase: train Epoch: 3/3 Iter: 21/62 Batch time: 0.3451
Phase: train Epoch: 3/3 Iter: 22/62 Batch time: 0.3432
Phase: train Epoch: 3/3 Iter: 23/62 Batch time: 0.3437
Phase: train Epoch: 3/3 Iter: 24/62 Batch time: 0.3437
Phase: train Epoch: 3/3 Iter: 25/62 Batch time: 0.3442
Phase: train Epoch: 3/3 Iter: 26/62 Batch time: 0.3443
Phase: train Epoch: 3/3 Iter: 27/62 Batch time: 0.3453
Phase: train Epoch: 3/3 Iter: 28/62 Batch time: 0.3459
Phase: train Epoch: 3/3 Iter: 29/62 Batch time: 0.3453
Phase: train Epoch: 3/3 Iter: 30/62 Batch time: 0.3439
Phase: train Epoch: 3/3 Iter: 31/62 Batch time: 0.3440
Phase: train Epoch: 3/3 Iter: 32/62 Batch time: 0.3430
Phase: train Epoch: 3/3 Iter: 33/62 Batch time: 0.3444
Phase: train Epoch: 3/3 Iter: 34/62 Batch time: 0.3447
Phase: train Epoch: 3/3 Iter: 35/62 Batch time: 0.3438
Phase: train Epoch: 3/3 Iter: 36/62 Batch time: 0.3434
Phase: train Epoch: 3/3 Iter: 37/62 Batch time: 0.3444
Phase: train Epoch: 3/3 Iter: 38/62 Batch time: 0.3437
Phase: train Epoch: 3/3 Iter: 39/62 Batch time: 0.3429
Phase: train Epoch: 3/3 Iter: 40/62 Batch time: 0.3430
Phase: train Epoch: 3/3 Iter: 41/62 Batch time: 0.3440
Phase: train Epoch: 3/3 Iter: 42/62 Batch time: 0.3443
Phase: train Epoch: 3/3 Iter: 43/62 Batch time: 0.3448
Phase: train Epoch: 3/3 Iter: 44/62 Batch time: 0.3439
Phase: train Epoch: 3/3 Iter: 45/62 Batch time: 0.3435
Phase: train Epoch: 3/3 Iter: 46/62 Batch time: 0.3435
Phase: train Epoch: 3/3 Iter: 47/62 Batch time: 0.3436
Phase: train Epoch: 3/3 Iter: 48/62 Batch time: 0.3436
Phase: train Epoch: 3/3 Iter: 49/62 Batch time: 0.3429
Phase: train Epoch: 3/3 Iter: 50/62 Batch time: 0.3456
Phase: train Epoch: 3/3 Iter: 51/62 Batch time: 0.3437
Phase: train Epoch: 3/3 Iter: 52/62 Batch time: 0.3443
Phase: train Epoch: 3/3 Iter: 53/62 Batch time: 0.3444
Phase: train Epoch: 3/3 Iter: 54/62 Batch time: 0.3442
Phase: train Epoch: 3/3 Iter: 55/62 Batch time: 0.3435
Phase: train Epoch: 3/3 Iter: 56/62 Batch time: 0.3440
Phase: train Epoch: 3/3 Iter: 57/62 Batch time: 0.3442
Phase: train Epoch: 3/3 Iter: 58/62 Batch time: 0.3440
Phase: train Epoch: 3/3 Iter: 59/62 Batch time: 0.3426
Phase: train Epoch: 3/3 Iter: 60/62 Batch time: 0.3447
Phase: train Epoch: 3/3 Iter: 61/62 Batch time: 0.3432
Phase: train Epoch: 3/3 Loss: 0.5652 Acc: 0.7418
Phase: validation Epoch: 3/3 Iter: 1/39 Batch time: 0.2733
Phase: validation Epoch: 3/3 Iter: 2/39 Batch time: 0.2697
Phase: validation Epoch: 3/3 Iter: 3/39 Batch time: 0.2704
Phase: validation Epoch: 3/3 Iter: 4/39 Batch time: 0.2702
Phase: validation Epoch: 3/3 Iter: 5/39 Batch time: 0.2687
Phase: validation Epoch: 3/3 Iter: 6/39 Batch time: 0.2691
Phase: validation Epoch: 3/3 Iter: 7/39 Batch time: 0.2696
Phase: validation Epoch: 3/3 Iter: 8/39 Batch time: 0.2701
Phase: validation Epoch: 3/3 Iter: 9/39 Batch time: 0.2709
Phase: validation Epoch: 3/3 Iter: 10/39 Batch time: 0.2708
Phase: validation Epoch: 3/3 Iter: 11/39 Batch time: 0.2698
Phase: validation Epoch: 3/3 Iter: 12/39 Batch time: 0.2698
Phase: validation Epoch: 3/3 Iter: 13/39 Batch time: 0.2690
Phase: validation Epoch: 3/3 Iter: 14/39 Batch time: 0.2702
Phase: validation Epoch: 3/3 Iter: 15/39 Batch time: 0.2701
Phase: validation Epoch: 3/3 Iter: 16/39 Batch time: 0.2699
Phase: validation Epoch: 3/3 Iter: 17/39 Batch time: 0.2702
Phase: validation Epoch: 3/3 Iter: 18/39 Batch time: 0.2699
Phase: validation Epoch: 3/3 Iter: 19/39 Batch time: 0.2698
Phase: validation Epoch: 3/3 Iter: 20/39 Batch time: 0.2701
Phase: validation Epoch: 3/3 Iter: 21/39 Batch time: 0.2701
Phase: validation Epoch: 3/3 Iter: 22/39 Batch time: 0.2709
Phase: validation Epoch: 3/3 Iter: 23/39 Batch time: 0.2701
Phase: validation Epoch: 3/3 Iter: 24/39 Batch time: 0.2691
Phase: validation Epoch: 3/3 Iter: 25/39 Batch time: 0.2696
Phase: validation Epoch: 3/3 Iter: 26/39 Batch time: 0.2689
Phase: validation Epoch: 3/3 Iter: 27/39 Batch time: 0.2696
Phase: validation Epoch: 3/3 Iter: 28/39 Batch time: 0.2699
Phase: validation Epoch: 3/3 Iter: 29/39 Batch time: 0.2686
Phase: validation Epoch: 3/3 Iter: 30/39 Batch time: 0.2692
Phase: validation Epoch: 3/3 Iter: 31/39 Batch time: 0.2697
Phase: validation Epoch: 3/3 Iter: 32/39 Batch time: 0.2700
Phase: validation Epoch: 3/3 Iter: 33/39 Batch time: 0.2701
Phase: validation Epoch: 3/3 Iter: 34/39 Batch time: 0.2716
Phase: validation Epoch: 3/3 Iter: 35/39 Batch time: 0.2703
Phase: validation Epoch: 3/3 Iter: 36/39 Batch time: 0.2695
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
 39%|###8      | 17.2M/44.7M [00:00<00:00, 180MB/s]
 94%|#########3| 41.9M/44.7M [00:00<00:00, 227MB/s]
100%|##########| 44.7M/44.7M [00:00<00:00, 222MB/s]
Training started:
Phase: train Epoch: 1/3 Iter: 1/62 Batch time: 0.3492
Phase: train Epoch: 1/3 Iter: 2/62 Batch time: 0.3050
Phase: train Epoch: 1/3 Iter: 3/62 Batch time: 0.3070
Phase: train Epoch: 1/3 Iter: 4/62 Batch time: 0.3050
Phase: train Epoch: 1/3 Iter: 5/62 Batch time: 0.3082
Phase: train Epoch: 1/3 Iter: 6/62 Batch time: 0.3118
Phase: train Epoch: 1/3 Iter: 7/62 Batch time: 0.3062
Phase: train Epoch: 1/3 Iter: 8/62 Batch time: 0.3127
Phase: train Epoch: 1/3 Iter: 9/62 Batch time: 0.3074
Phase: train Epoch: 1/3 Iter: 10/62 Batch time: 0.3099
Phase: train Epoch: 1/3 Iter: 11/62 Batch time: 0.3067
Phase: train Epoch: 1/3 Iter: 12/62 Batch time: 0.3129
Phase: train Epoch: 1/3 Iter: 13/62 Batch time: 0.3092
Phase: train Epoch: 1/3 Iter: 14/62 Batch time: 0.3063
Phase: train Epoch: 1/3 Iter: 15/62 Batch time: 0.3036
Phase: train Epoch: 1/3 Iter: 16/62 Batch time: 0.3025
Phase: train Epoch: 1/3 Iter: 17/62 Batch time: 0.3068
Phase: train Epoch: 1/3 Iter: 18/62 Batch time: 0.3050
Phase: train Epoch: 1/3 Iter: 19/62 Batch time: 0.3086
Phase: train Epoch: 1/3 Iter: 20/62 Batch time: 0.3123
Phase: train Epoch: 1/3 Iter: 21/62 Batch time: 0.3143
Phase: train Epoch: 1/3 Iter: 22/62 Batch time: 0.3051
Phase: train Epoch: 1/3 Iter: 23/62 Batch time: 0.3054
Phase: train Epoch: 1/3 Iter: 24/62 Batch time: 0.3055
Phase: train Epoch: 1/3 Iter: 25/62 Batch time: 0.3043
Phase: train Epoch: 1/3 Iter: 26/62 Batch time: 0.3144
Phase: train Epoch: 1/3 Iter: 27/62 Batch time: 0.3066
Phase: train Epoch: 1/3 Iter: 28/62 Batch time: 0.3050
Phase: train Epoch: 1/3 Iter: 29/62 Batch time: 0.3062
Phase: train Epoch: 1/3 Iter: 30/62 Batch time: 0.3034
Phase: train Epoch: 1/3 Iter: 31/62 Batch time: 0.3043
Phase: train Epoch: 1/3 Iter: 32/62 Batch time: 0.3048
Phase: train Epoch: 1/3 Iter: 33/62 Batch time: 0.3058
Phase: train Epoch: 1/3 Iter: 34/62 Batch time: 0.3108
Phase: train Epoch: 1/3 Iter: 35/62 Batch time: 0.3041
Phase: train Epoch: 1/3 Iter: 36/62 Batch time: 0.3061
Phase: train Epoch: 1/3 Iter: 37/62 Batch time: 0.3064
Phase: train Epoch: 1/3 Iter: 38/62 Batch time: 0.3035
Phase: train Epoch: 1/3 Iter: 39/62 Batch time: 0.3062
Phase: train Epoch: 1/3 Iter: 40/62 Batch time: 0.3078
Phase: train Epoch: 1/3 Iter: 41/62 Batch time: 0.3022
Phase: train Epoch: 1/3 Iter: 42/62 Batch time: 0.3048
Phase: train Epoch: 1/3 Iter: 43/62 Batch time: 0.3040
Phase: train Epoch: 1/3 Iter: 44/62 Batch time: 0.3055
Phase: train Epoch: 1/3 Iter: 45/62 Batch time: 0.3067
Phase: train Epoch: 1/3 Iter: 46/62 Batch time: 0.3065
Phase: train Epoch: 1/3 Iter: 47/62 Batch time: 0.3057
Phase: train Epoch: 1/3 Iter: 48/62 Batch time: 0.3082
Phase: train Epoch: 1/3 Iter: 49/62 Batch time: 0.3101
Phase: train Epoch: 1/3 Iter: 50/62 Batch time: 0.3094
Phase: train Epoch: 1/3 Iter: 51/62 Batch time: 0.3080
Phase: train Epoch: 1/3 Iter: 52/62 Batch time: 0.3039
Phase: train Epoch: 1/3 Iter: 53/62 Batch time: 0.3089
Phase: train Epoch: 1/3 Iter: 54/62 Batch time: 0.3035
Phase: train Epoch: 1/3 Iter: 55/62 Batch time: 0.3035
Phase: train Epoch: 1/3 Iter: 56/62 Batch time: 0.3053
Phase: train Epoch: 1/3 Iter: 57/62 Batch time: 0.3045
Phase: train Epoch: 1/3 Iter: 58/62 Batch time: 0.3044
Phase: train Epoch: 1/3 Iter: 59/62 Batch time: 0.3109
Phase: train Epoch: 1/3 Iter: 60/62 Batch time: 0.3048
Phase: train Epoch: 1/3 Iter: 61/62 Batch time: 0.3036
Phase: train Epoch: 1/3 Loss: 0.6990 Acc: 0.5246
Phase: validation Epoch: 1/3 Iter: 1/39 Batch time: 0.2286
Phase: validation Epoch: 1/3 Iter: 2/39 Batch time: 0.2264
Phase: validation Epoch: 1/3 Iter: 3/39 Batch time: 0.2255
Phase: validation Epoch: 1/3 Iter: 4/39 Batch time: 0.2264
Phase: validation Epoch: 1/3 Iter: 5/39 Batch time: 0.2256
Phase: validation Epoch: 1/3 Iter: 6/39 Batch time: 0.2267
Phase: validation Epoch: 1/3 Iter: 7/39 Batch time: 0.2283
Phase: validation Epoch: 1/3 Iter: 8/39 Batch time: 0.2257
Phase: validation Epoch: 1/3 Iter: 9/39 Batch time: 0.2278
Phase: validation Epoch: 1/3 Iter: 10/39 Batch time: 0.2267
Phase: validation Epoch: 1/3 Iter: 11/39 Batch time: 0.2260
Phase: validation Epoch: 1/3 Iter: 12/39 Batch time: 0.2250
Phase: validation Epoch: 1/3 Iter: 13/39 Batch time: 0.2273
Phase: validation Epoch: 1/3 Iter: 14/39 Batch time: 0.2254
Phase: validation Epoch: 1/3 Iter: 15/39 Batch time: 0.2273
Phase: validation Epoch: 1/3 Iter: 16/39 Batch time: 0.2271
Phase: validation Epoch: 1/3 Iter: 17/39 Batch time: 0.2273
Phase: validation Epoch: 1/3 Iter: 18/39 Batch time: 0.2292
Phase: validation Epoch: 1/3 Iter: 19/39 Batch time: 0.2308
Phase: validation Epoch: 1/3 Iter: 20/39 Batch time: 0.2253
Phase: validation Epoch: 1/3 Iter: 21/39 Batch time: 0.2270
Phase: validation Epoch: 1/3 Iter: 22/39 Batch time: 0.2250
Phase: validation Epoch: 1/3 Iter: 23/39 Batch time: 0.2269
Phase: validation Epoch: 1/3 Iter: 24/39 Batch time: 0.2259
Phase: validation Epoch: 1/3 Iter: 25/39 Batch time: 0.2261
Phase: validation Epoch: 1/3 Iter: 26/39 Batch time: 0.2258
Phase: validation Epoch: 1/3 Iter: 27/39 Batch time: 0.2289
Phase: validation Epoch: 1/3 Iter: 28/39 Batch time: 0.2288
Phase: validation Epoch: 1/3 Iter: 29/39 Batch time: 0.2305
Phase: validation Epoch: 1/3 Iter: 30/39 Batch time: 0.2340
Phase: validation Epoch: 1/3 Iter: 31/39 Batch time: 0.2325
Phase: validation Epoch: 1/3 Iter: 32/39 Batch time: 0.2270
Phase: validation Epoch: 1/3 Iter: 33/39 Batch time: 0.2332
Phase: validation Epoch: 1/3 Iter: 34/39 Batch time: 0.2280
Phase: validation Epoch: 1/3 Iter: 35/39 Batch time: 0.2296
Phase: validation Epoch: 1/3 Iter: 36/39 Batch time: 0.2305
Phase: validation Epoch: 1/3 Iter: 37/39 Batch time: 0.2286
Phase: validation Epoch: 1/3 Iter: 38/39 Batch time: 0.2242
Phase: validation Epoch: 1/3 Iter: 39/39 Batch time: 0.0749
Phase: validation   Epoch: 1/3 Loss: 0.6429 Acc: 0.6536
Phase: train Epoch: 2/3 Iter: 1/62 Batch time: 0.3007
Phase: train Epoch: 2/3 Iter: 2/62 Batch time: 0.3074
Phase: train Epoch: 2/3 Iter: 3/62 Batch time: 0.3055
Phase: train Epoch: 2/3 Iter: 4/62 Batch time: 0.3089
Phase: train Epoch: 2/3 Iter: 5/62 Batch time: 0.3058
Phase: train Epoch: 2/3 Iter: 6/62 Batch time: 0.3112
Phase: train Epoch: 2/3 Iter: 7/62 Batch time: 0.3089
Phase: train Epoch: 2/3 Iter: 8/62 Batch time: 0.3126
Phase: train Epoch: 2/3 Iter: 9/62 Batch time: 0.3169
Phase: train Epoch: 2/3 Iter: 10/62 Batch time: 0.3077
Phase: train Epoch: 2/3 Iter: 11/62 Batch time: 0.3085
Phase: train Epoch: 2/3 Iter: 12/62 Batch time: 0.3117
Phase: train Epoch: 2/3 Iter: 13/62 Batch time: 0.3064
Phase: train Epoch: 2/3 Iter: 14/62 Batch time: 0.3075
Phase: train Epoch: 2/3 Iter: 15/62 Batch time: 0.3083
Phase: train Epoch: 2/3 Iter: 16/62 Batch time: 0.3108
Phase: train Epoch: 2/3 Iter: 17/62 Batch time: 0.3064
Phase: train Epoch: 2/3 Iter: 18/62 Batch time: 0.3147
Phase: train Epoch: 2/3 Iter: 19/62 Batch time: 0.3111
Phase: train Epoch: 2/3 Iter: 20/62 Batch time: 0.3083
Phase: train Epoch: 2/3 Iter: 21/62 Batch time: 0.3110
Phase: train Epoch: 2/3 Iter: 22/62 Batch time: 0.3071
Phase: train Epoch: 2/3 Iter: 23/62 Batch time: 0.3066
Phase: train Epoch: 2/3 Iter: 24/62 Batch time: 0.3077
Phase: train Epoch: 2/3 Iter: 25/62 Batch time: 0.3114
Phase: train Epoch: 2/3 Iter: 26/62 Batch time: 0.3121
Phase: train Epoch: 2/3 Iter: 27/62 Batch time: 0.3116
Phase: train Epoch: 2/3 Iter: 28/62 Batch time: 0.3104
Phase: train Epoch: 2/3 Iter: 29/62 Batch time: 0.3111
Phase: train Epoch: 2/3 Iter: 30/62 Batch time: 0.3080
Phase: train Epoch: 2/3 Iter: 31/62 Batch time: 0.3117
Phase: train Epoch: 2/3 Iter: 32/62 Batch time: 0.3082
Phase: train Epoch: 2/3 Iter: 33/62 Batch time: 0.3083
Phase: train Epoch: 2/3 Iter: 34/62 Batch time: 0.3117
Phase: train Epoch: 2/3 Iter: 35/62 Batch time: 0.3100
Phase: train Epoch: 2/3 Iter: 36/62 Batch time: 0.3092
Phase: train Epoch: 2/3 Iter: 37/62 Batch time: 0.3115
Phase: train Epoch: 2/3 Iter: 38/62 Batch time: 0.3087
Phase: train Epoch: 2/3 Iter: 39/62 Batch time: 0.3051
Phase: train Epoch: 2/3 Iter: 40/62 Batch time: 0.3065
Phase: train Epoch: 2/3 Iter: 41/62 Batch time: 0.3106
Phase: train Epoch: 2/3 Iter: 42/62 Batch time: 0.3091
Phase: train Epoch: 2/3 Iter: 43/62 Batch time: 0.3097
Phase: train Epoch: 2/3 Iter: 44/62 Batch time: 0.3126
Phase: train Epoch: 2/3 Iter: 45/62 Batch time: 0.3057
Phase: train Epoch: 2/3 Iter: 46/62 Batch time: 0.3068
Phase: train Epoch: 2/3 Iter: 47/62 Batch time: 0.3083
Phase: train Epoch: 2/3 Iter: 48/62 Batch time: 0.3121
Phase: train Epoch: 2/3 Iter: 49/62 Batch time: 0.3073
Phase: train Epoch: 2/3 Iter: 50/62 Batch time: 0.3079
Phase: train Epoch: 2/3 Iter: 51/62 Batch time: 0.3087
Phase: train Epoch: 2/3 Iter: 52/62 Batch time: 0.3079
Phase: train Epoch: 2/3 Iter: 53/62 Batch time: 0.3069
Phase: train Epoch: 2/3 Iter: 54/62 Batch time: 0.3073
Phase: train Epoch: 2/3 Iter: 55/62 Batch time: 0.3060
Phase: train Epoch: 2/3 Iter: 56/62 Batch time: 0.3161
Phase: train Epoch: 2/3 Iter: 57/62 Batch time: 0.3075
Phase: train Epoch: 2/3 Iter: 58/62 Batch time: 0.3064
Phase: train Epoch: 2/3 Iter: 59/62 Batch time: 0.3079
Phase: train Epoch: 2/3 Iter: 60/62 Batch time: 0.3076
Phase: train Epoch: 2/3 Iter: 61/62 Batch time: 0.3050
Phase: train Epoch: 2/3 Loss: 0.6134 Acc: 0.7008
Phase: validation Epoch: 2/3 Iter: 1/39 Batch time: 0.2313
Phase: validation Epoch: 2/3 Iter: 2/39 Batch time: 0.2309
Phase: validation Epoch: 2/3 Iter: 3/39 Batch time: 0.2292
Phase: validation Epoch: 2/3 Iter: 4/39 Batch time: 0.2269
Phase: validation Epoch: 2/3 Iter: 5/39 Batch time: 0.2291
Phase: validation Epoch: 2/3 Iter: 6/39 Batch time: 0.2301
Phase: validation Epoch: 2/3 Iter: 7/39 Batch time: 0.2319
Phase: validation Epoch: 2/3 Iter: 8/39 Batch time: 0.2284
Phase: validation Epoch: 2/3 Iter: 9/39 Batch time: 0.2317
Phase: validation Epoch: 2/3 Iter: 10/39 Batch time: 0.2289
Phase: validation Epoch: 2/3 Iter: 11/39 Batch time: 0.2292
Phase: validation Epoch: 2/3 Iter: 12/39 Batch time: 0.2274
Phase: validation Epoch: 2/3 Iter: 13/39 Batch time: 0.2306
Phase: validation Epoch: 2/3 Iter: 14/39 Batch time: 0.2296
Phase: validation Epoch: 2/3 Iter: 15/39 Batch time: 0.2319
Phase: validation Epoch: 2/3 Iter: 16/39 Batch time: 0.2267
Phase: validation Epoch: 2/3 Iter: 17/39 Batch time: 0.2275
Phase: validation Epoch: 2/3 Iter: 18/39 Batch time: 0.2246
Phase: validation Epoch: 2/3 Iter: 19/39 Batch time: 0.2276
Phase: validation Epoch: 2/3 Iter: 20/39 Batch time: 0.2312
Phase: validation Epoch: 2/3 Iter: 21/39 Batch time: 0.2289
Phase: validation Epoch: 2/3 Iter: 22/39 Batch time: 0.2276
Phase: validation Epoch: 2/3 Iter: 23/39 Batch time: 0.2291
Phase: validation Epoch: 2/3 Iter: 24/39 Batch time: 0.2292
Phase: validation Epoch: 2/3 Iter: 25/39 Batch time: 0.2278
Phase: validation Epoch: 2/3 Iter: 26/39 Batch time: 0.2291
Phase: validation Epoch: 2/3 Iter: 27/39 Batch time: 0.2307
Phase: validation Epoch: 2/3 Iter: 28/39 Batch time: 0.2294
Phase: validation Epoch: 2/3 Iter: 29/39 Batch time: 0.2298
Phase: validation Epoch: 2/3 Iter: 30/39 Batch time: 0.2263
Phase: validation Epoch: 2/3 Iter: 31/39 Batch time: 0.2290
Phase: validation Epoch: 2/3 Iter: 32/39 Batch time: 0.2305
Phase: validation Epoch: 2/3 Iter: 33/39 Batch time: 0.2329
Phase: validation Epoch: 2/3 Iter: 34/39 Batch time: 0.2278
Phase: validation Epoch: 2/3 Iter: 35/39 Batch time: 0.2287
Phase: validation Epoch: 2/3 Iter: 36/39 Batch time: 0.2277
Phase: validation Epoch: 2/3 Iter: 37/39 Batch time: 0.2306
Phase: validation Epoch: 2/3 Iter: 38/39 Batch time: 0.2279
Phase: validation Epoch: 2/3 Iter: 39/39 Batch time: 0.0667
Phase: validation   Epoch: 2/3 Loss: 0.5389 Acc: 0.8235
Phase: train Epoch: 3/3 Iter: 1/62 Batch time: 0.3021
Phase: train Epoch: 3/3 Iter: 2/62 Batch time: 0.3101
Phase: train Epoch: 3/3 Iter: 3/62 Batch time: 0.3096
Phase: train Epoch: 3/3 Iter: 4/62 Batch time: 0.3077
Phase: train Epoch: 3/3 Iter: 5/62 Batch time: 0.3080
Phase: train Epoch: 3/3 Iter: 6/62 Batch time: 0.3101
Phase: train Epoch: 3/3 Iter: 7/62 Batch time: 0.3100
Phase: train Epoch: 3/3 Iter: 8/62 Batch time: 0.3057
Phase: train Epoch: 3/3 Iter: 9/62 Batch time: 0.3097
Phase: train Epoch: 3/3 Iter: 10/62 Batch time: 0.3099
Phase: train Epoch: 3/3 Iter: 11/62 Batch time: 0.3112
Phase: train Epoch: 3/3 Iter: 12/62 Batch time: 0.3180
Phase: train Epoch: 3/3 Iter: 13/62 Batch time: 0.3198
Phase: train Epoch: 3/3 Iter: 14/62 Batch time: 0.3132
Phase: train Epoch: 3/3 Iter: 15/62 Batch time: 0.3181
Phase: train Epoch: 3/3 Iter: 16/62 Batch time: 0.3132
Phase: train Epoch: 3/3 Iter: 17/62 Batch time: 0.3110
Phase: train Epoch: 3/3 Iter: 18/62 Batch time: 0.3135
Phase: train Epoch: 3/3 Iter: 19/62 Batch time: 0.3142
Phase: train Epoch: 3/3 Iter: 20/62 Batch time: 0.3126
Phase: train Epoch: 3/3 Iter: 21/62 Batch time: 0.3160
Phase: train Epoch: 3/3 Iter: 22/62 Batch time: 0.3091
Phase: train Epoch: 3/3 Iter: 23/62 Batch time: 0.3090
Phase: train Epoch: 3/3 Iter: 24/62 Batch time: 0.3117
Phase: train Epoch: 3/3 Iter: 25/62 Batch time: 0.3198
Phase: train Epoch: 3/3 Iter: 26/62 Batch time: 0.3115
Phase: train Epoch: 3/3 Iter: 27/62 Batch time: 0.3168
Phase: train Epoch: 3/3 Iter: 28/62 Batch time: 0.3162
Phase: train Epoch: 3/3 Iter: 29/62 Batch time: 0.3113
Phase: train Epoch: 3/3 Iter: 30/62 Batch time: 0.3104
Phase: train Epoch: 3/3 Iter: 31/62 Batch time: 0.3097
Phase: train Epoch: 3/3 Iter: 32/62 Batch time: 0.3096
Phase: train Epoch: 3/3 Iter: 33/62 Batch time: 0.3088
Phase: train Epoch: 3/3 Iter: 34/62 Batch time: 0.3107
Phase: train Epoch: 3/3 Iter: 35/62 Batch time: 0.3087
Phase: train Epoch: 3/3 Iter: 36/62 Batch time: 0.3078
Phase: train Epoch: 3/3 Iter: 37/62 Batch time: 0.3094
Phase: train Epoch: 3/3 Iter: 38/62 Batch time: 0.3056
Phase: train Epoch: 3/3 Iter: 39/62 Batch time: 0.3073
Phase: train Epoch: 3/3 Iter: 40/62 Batch time: 0.3088
Phase: train Epoch: 3/3 Iter: 41/62 Batch time: 0.3103
Phase: train Epoch: 3/3 Iter: 42/62 Batch time: 0.3102
Phase: train Epoch: 3/3 Iter: 43/62 Batch time: 0.3122
Phase: train Epoch: 3/3 Iter: 44/62 Batch time: 0.3079
Phase: train Epoch: 3/3 Iter: 45/62 Batch time: 0.3082
Phase: train Epoch: 3/3 Iter: 46/62 Batch time: 0.3088
Phase: train Epoch: 3/3 Iter: 47/62 Batch time: 0.3076
Phase: train Epoch: 3/3 Iter: 48/62 Batch time: 0.3103
Phase: train Epoch: 3/3 Iter: 49/62 Batch time: 0.3099
Phase: train Epoch: 3/3 Iter: 50/62 Batch time: 0.3106
Phase: train Epoch: 3/3 Iter: 51/62 Batch time: 0.3113
Phase: train Epoch: 3/3 Iter: 52/62 Batch time: 0.3106
Phase: train Epoch: 3/3 Iter: 53/62 Batch time: 0.3127
Phase: train Epoch: 3/3 Iter: 54/62 Batch time: 0.3148
Phase: train Epoch: 3/3 Iter: 55/62 Batch time: 0.3097
Phase: train Epoch: 3/3 Iter: 56/62 Batch time: 0.3106
Phase: train Epoch: 3/3 Iter: 57/62 Batch time: 0.3139
Phase: train Epoch: 3/3 Iter: 58/62 Batch time: 0.3083
Phase: train Epoch: 3/3 Iter: 59/62 Batch time: 0.3078
Phase: train Epoch: 3/3 Iter: 60/62 Batch time: 0.3126
Phase: train Epoch: 3/3 Iter: 61/62 Batch time: 0.3113
Phase: train Epoch: 3/3 Loss: 0.5652 Acc: 0.7418
Phase: validation Epoch: 3/3 Iter: 1/39 Batch time: 0.2388
Phase: validation Epoch: 3/3 Iter: 2/39 Batch time: 0.2297
Phase: validation Epoch: 3/3 Iter: 3/39 Batch time: 0.2294
Phase: validation Epoch: 3/3 Iter: 4/39 Batch time: 0.2306
Phase: validation Epoch: 3/3 Iter: 5/39 Batch time: 0.2297
Phase: validation Epoch: 3/3 Iter: 6/39 Batch time: 0.2289
Phase: validation Epoch: 3/3 Iter: 7/39 Batch time: 0.2283
Phase: validation Epoch: 3/3 Iter: 8/39 Batch time: 0.2298
Phase: validation Epoch: 3/3 Iter: 9/39 Batch time: 0.2278
Phase: validation Epoch: 3/3 Iter: 10/39 Batch time: 0.2295
Phase: validation Epoch: 3/3 Iter: 11/39 Batch time: 0.2299
Phase: validation Epoch: 3/3 Iter: 12/39 Batch time: 0.2284
Phase: validation Epoch: 3/3 Iter: 13/39 Batch time: 0.2301
Phase: validation Epoch: 3/3 Iter: 14/39 Batch time: 0.2293
Phase: validation Epoch: 3/3 Iter: 15/39 Batch time: 0.2364
Phase: validation Epoch: 3/3 Iter: 16/39 Batch time: 0.2302
Phase: validation Epoch: 3/3 Iter: 17/39 Batch time: 0.2314
Phase: validation Epoch: 3/3 Iter: 18/39 Batch time: 0.2289
Phase: validation Epoch: 3/3 Iter: 19/39 Batch time: 0.2300
Phase: validation Epoch: 3/3 Iter: 20/39 Batch time: 0.2296
Phase: validation Epoch: 3/3 Iter: 21/39 Batch time: 0.2284
Phase: validation Epoch: 3/3 Iter: 22/39 Batch time: 0.2304
Phase: validation Epoch: 3/3 Iter: 23/39 Batch time: 0.2284
Phase: validation Epoch: 3/3 Iter: 24/39 Batch time: 0.2302
Phase: validation Epoch: 3/3 Iter: 25/39 Batch time: 0.2290
Phase: validation Epoch: 3/3 Iter: 26/39 Batch time: 0.2289
Phase: validation Epoch: 3/3 Iter: 27/39 Batch time: 0.2306
Phase: validation Epoch: 3/3 Iter: 28/39 Batch time: 0.2287
Phase: validation Epoch: 3/3 Iter: 29/39 Batch time: 0.2344
Phase: validation Epoch: 3/3 Iter: 30/39 Batch time: 0.2332
Phase: validation Epoch: 3/3 Iter: 31/39 Batch time: 0.2348
Phase: validation Epoch: 3/3 Iter: 32/39 Batch time: 0.2357
Phase: validation Epoch: 3/3 Iter: 33/39 Batch time: 0.2319
Phase: validation Epoch: 3/3 Iter: 34/39 Batch time: 0.2301
Phase: validation Epoch: 3/3 Iter: 35/39 Batch time: 0.2294
Phase: validation Epoch: 3/3 Iter: 36/39 Batch time: 0.2310
Phase: validation Epoch: 3/3 Iter: 37/39 Batch time: 0.2326
Phase: validation Epoch: 3/3 Iter: 38/39 Batch time: 0.2317
Phase: validation Epoch: 3/3 Iter: 39/39 Batch time: 0.0704
Phase: validation   Epoch: 3/3 Loss: 0.4484 Acc: 0.8497
Training completed in 1m 30s
Best test loss: 0.4484 | Best test accuracy: 0.8497
 </code>
 </pre>
 </details>

---

## 6. tutorial_local_cost_functions.html <a name="demo5"></a>

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

## 8. tutorial_vqe_spin_sectors.html <a name="demo7"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_vqe_spin_sectors.html):

```
The Hamiltonian is    (-0.24274501260947007) [Z2]
+ (-0.24274501260947) [Z3]
+ (-0.042072551947389875) [I0]
+ (0.17771358229088333) [Z1]
+ (0.1777135822908834) [Z0]
+ (0.12293330449301082) [Z0 Z2]
+ (0.12293330449301082) [Z1 Z3]
+ (0.16768338855603704) [Z0 Z3]
+ (0.16768338855603704) [Z1 Z2]
+ (0.1705975927683755) [Z0 Z1]
+ (0.17627661394185995) [Z2 Z3]
+ (-0.04475008406302621) [Y0 Y1 X2 X3]
+ (-0.04475008406302621) [X0 X1 Y2 Y3]
+ (0.04475008406302621) [Y0 X1 X2 Y3]
+ (0.04475008406302621) [X0 Y1 Y2 X3]
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_vqe_spin_sectors.html):

```
The Hamiltonian is    (-0.24274501260911585) [Z3]
+ (-0.2427450126091158) [Z2]
+ (-0.04207255194770376) [I0]
+ (0.17771358229109968) [Z0]
+ (0.17771358229109968) [Z1]
+ (0.12293330449290178) [Z0 Z2]
+ (0.12293330449290178) [Z1 Z3]
+ (0.1676833885558882) [Z0 Z3]
+ (0.1676833885558882) [Z1 Z2]
+ (0.17059759276832803) [Z0 Z1]
+ (0.1762766139415956) [Z2 Z3]
+ (-0.044750084062986466) [Y0 Y1 X2 X3]
+ (-0.044750084062986466) [X0 X1 Y2 Y3]
+ (0.044750084062986466) [Y0 X1 X2 Y3]
+ (0.044750084062986466) [X0 Y1 Y2 X3]
```

---

## 9. tutorial_backprop.html <a name="demo8"></a>

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
Forward pass (best of 3): 0.028114071800064266 sec per loop
Gradient computation (best of 3): 11.467769472800047 sec per loop
10.121065848023136
0.9358535378025424
Forward pass (best of 3): 0.059971427799973755 sec per loop
Backward pass (best of 3): 0.166022428999986 sec per loop
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

## 10. tutorial_measurement_optimize.html <a name="demo9"></a>

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

## 11. tutorial_error_mitigation.html <a name="demo10"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_error_mitigation.html):

```
0.988612932346176
1: ──RY(3.60)─╰Z──RY(5.90)─╭●──────────RY(5.18)──RY(-5.18)─────────────────────
2: ──RY(4.05)─╭●──RY(3.32)─╰Z──────────RY(1.07)──RY(-1.07)──RY(1.07)──RY(-1.07)
0: ──RY(4.56)─╭●──RY(5.93)──RY(-5.93)─────────────────────────────────────
1: ──RY(3.60)─╰Z──RY(5.90)──RY(-5.90)──RY(5.90)─╭●──RY(5.18)──RY(-5.18)─╭●
2: ──RY(4.05)─╭●──RY(3.32)──────────────────────╰Z──RY(1.07)──RY(-1.07)─╰Z
3: ──RY(3.51)─╰Z──RY(3.66)──RY(-3.66)─────────────────────────────────────
─────────────╭●──RY(-4.56)─┤
───RY(-5.90)─╰Z──RY(-3.60)─┤
───RY(-3.32)─╭●──RY(-4.05)─┤
─────────────╰Z──RY(-3.51)─┤
0: ──RY(4.56)─╭●──RY(5.93)──RY(-5.93)──────────────────────────────────────────
1: ──RY(3.60)─╰Z──RY(5.90)─╭●──────────RY(5.18)──RY(-5.18)─────────────────────
2: ──RY(4.05)─╭●──RY(3.32)─╰Z──────────RY(1.07)──RY(-1.07)──RY(1.07)──RY(-1.07)
3: ──RY(3.51)─╰Z──RY(3.66)──RY(-3.66)──────────────────────────────────────────
────────────────╭●──RY(-4.56)─┤
──╭●──RY(-5.90)─╰Z──RY(-3.60)─┤
──╰Z──RY(-3.32)─╭●──RY(-4.05)─┤
────────────────╰Z──RY(-3.51)─┤
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_error_mitigation.html):

```
0.9652500393119565
1: ──RY(3.60)─╰Z──RY(5.90)─╭●──────────RY(5.18)──RY(-5.18)──RY(5.18)──RY(-5.18)
2: ──RY(4.05)─╭●──RY(3.32)─╰Z──────────RY(1.07)──RY(-1.07)─────────────────────
0: ──RY(4.56)─╭●──RY(5.93)──RY(-5.93)────────────────────────────────────╭●
1: ──RY(3.60)─╰Z──RY(5.90)─╭●──────────RY(5.18)──RY(-5.18)─╭●──RY(-5.90)─╰Z
2: ──RY(4.05)─╭●──RY(3.32)─╰Z──────────RY(1.07)──RY(-1.07)─╰Z──RY(-3.32)─╭●
3: ──RY(3.51)─╰Z──RY(3.66)──RY(-3.66)────────────────────────────────────╰Z
───RY(-4.56)───────────────┤
───RY(-3.60)───────────────┤
──╭●─────────╭●──RY(-4.05)─┤
──╰Z─────────╰Z──RY(-3.51)─┤
0: ──RY(4.56)─╭●──RY(5.93)──RY(-5.93)──RY(5.93)──RY(-5.93)───────────────╭●
1: ──RY(3.60)─╰Z──RY(5.90)─╭●──────────RY(5.18)──RY(-5.18)─╭●──RY(-5.90)─╰Z
2: ──RY(4.05)─╭●──RY(3.32)─╰Z──────────RY(1.07)──RY(-1.07)─╰Z──RY(-3.32)─╭●
3: ──RY(3.51)─╰Z──RY(3.66)──RY(-3.66)────────────────────────────────────╰Z
───RY(-4.56)─┤
───RY(-3.60)─┤
───RY(-4.05)─┤
───RY(-3.51)─┤
```

---

## 12. tutorial_qft_arithmetics.html <a name="demo11"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_qft_arithmetics.html):

```
The ket representation of the sum of 3 and 4 is [0 1 1 1]
The ket representation of the sum of 7 and 3 is [1 0 1 0]
The ket representation of the multiplication of 3 and 7 is [1 0 1 0 1]
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_qft_arithmetics.html):

```
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
The animation.avconv_args rcparam was deprecated in Matplotlib 3.3 and will be removed two minor releases later.
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/_collections_abc.py:841: MatplotlibDeprecationWarning:
```

---

## 13. tutorial_jax_transformations.html <a name="demo12"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_jax_transformations.html):

```
No jit time: 0.0053 seconds
First run time: 0.0767 seconds
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_jax_transformations.html):

```
No jit time: 0.0058 seconds
First run time: 0.0882 seconds
```

---

## 14. tutorial_quanvolution.html <a name="demo13"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_quanvolution.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
13/13 - 0s - loss: 0.1135 - accuracy: 1.0000 - val_loss: 1.0451 - val_accuracy: 0.7333 - 29ms/epoch - 2ms/step
 1794048/11490434 [===>..........................] - ETA: 0s
13/13 - 0s - loss: 0.1041 - accuracy: 1.0000 - val_loss: 1.0142 - val_accuracy: 0.7333 - 30ms/epoch - 2ms/step
13/13 - 0s - loss: 0.0983 - accuracy: 1.0000 - val_loss: 0.9893 - val_accuracy: 0.7333 - 44ms/epoch - 3ms/step
13/13 - 0s - loss: 0.0913 - accuracy: 1.0000 - val_loss: 0.9807 - val_accuracy: 0.7000 - 45ms/epoch - 3ms/step
13/13 - 0s - loss: 0.0868 - accuracy: 1.0000 - val_loss: 0.9715 - val_accuracy: 0.7333 - 29ms/epoch - 2ms/step
13/13 - 0s - loss: 0.0788 - accuracy: 1.0000 - val_loss: 0.9850 - val_accuracy: 0.7333 - 44ms/epoch - 3ms/step
13/13 - 0s - loss: 0.0749 - accuracy: 1.0000 - val_loss: 0.9750 - val_accuracy: 0.7333 - 43ms/epoch - 3ms/step
13/13 - 0s - loss: 0.0730 - accuracy: 1.0000 - val_loss: 0.9570 - val_accuracy: 0.7667 - 30ms/epoch - 2ms/step
13/13 - 0s - loss: 0.0681 - accuracy: 1.0000 - val_loss: 0.9895 - val_accuracy: 0.7333 - 29ms/epoch - 2ms/step
13/13 - 0s - loss: 0.0635 - accuracy: 1.0000 - val_loss: 0.9560 - val_accuracy: 0.7333 - 45ms/epoch - 3ms/step
13/13 - 0s - loss: 2.3619 - accuracy: 0.1400 - val_loss: 2.0567 - val_accuracy: 0.3667 - 325ms/epoch - 25ms/step
13/13 - 0s - loss: 1.9696 - accuracy: 0.4200 - val_loss: 1.9381 - val_accuracy: 0.4667 - 29ms/epoch - 2ms/step
13/13 - 0s - loss: 1.6671 - accuracy: 0.6400 - val_loss: 1.8300 - val_accuracy: 0.4333 - 44ms/epoch - 3ms/step
13/13 - 0s - loss: 1.4340 - accuracy: 0.7400 - val_loss: 1.7113 - val_accuracy: 0.4333 - 30ms/epoch - 2ms/step
13/13 - 0s - loss: 1.2342 - accuracy: 0.7600 - val_loss: 1.6044 - val_accuracy: 0.5000 - 29ms/epoch - 2ms/step
13/13 - 0s - loss: 1.0721 - accuracy: 0.8600 - val_loss: 1.5232 - val_accuracy: 0.5333 - 45ms/epoch - 3ms/step
13/13 - 0s - loss: 0.9348 - accuracy: 0.9000 - val_loss: 1.4596 - val_accuracy: 0.5667 - 44ms/epoch - 3ms/step
13/13 - 0s - loss: 0.8178 - accuracy: 0.9200 - val_loss: 1.3921 - val_accuracy: 0.6000 - 44ms/epoch - 3ms/step
13/13 - 0s - loss: 0.7223 - accuracy: 0.9400 - val_loss: 1.3404 - val_accuracy: 0.6333 - 28ms/epoch - 2ms/step
13/13 - 0s - loss: 0.6404 - accuracy: 0.9600 - val_loss: 1.3065 - val_accuracy: 0.6667 - 28ms/epoch - 2ms/step
13/13 - 0s - loss: 0.5772 - accuracy: 1.0000 - val_loss: 1.2644 - val_accuracy: 0.6333 - 29ms/epoch - 2ms/step
13/13 - 0s - loss: 0.5199 - accuracy: 1.0000 - val_loss: 1.2558 - val_accuracy: 0.6667 - 29ms/epoch - 2ms/step
13/13 - 0s - loss: 0.4695 - accuracy: 1.0000 - val_loss: 1.2258 - val_accuracy: 0.6667 - 45ms/epoch - 3ms/step
13/13 - 0s - loss: 0.4238 - accuracy: 1.0000 - val_loss: 1.1897 - val_accuracy: 0.6667 - 29ms/epoch - 2ms/step
13/13 - 0s - loss: 0.3848 - accuracy: 1.0000 - val_loss: 1.1651 - val_accuracy: 0.6667 - 29ms/epoch - 2ms/step
13/13 - 0s - loss: 0.3525 - accuracy: 1.0000 - val_loss: 1.1503 - val_accuracy: 0.7333 - 29ms/epoch - 2ms/step
13/13 - 0s - loss: 0.3245 - accuracy: 1.0000 - val_loss: 1.1374 - val_accuracy: 0.7000 - 44ms/epoch - 3ms/step
13/13 - 0s - loss: 0.2992 - accuracy: 1.0000 - val_loss: 1.1174 - val_accuracy: 0.6667 - 29ms/epoch - 2ms/step
13/13 - 0s - loss: 0.2745 - accuracy: 1.0000 - val_loss: 1.1119 - val_accuracy: 0.6667 - 29ms/epoch - 2ms/step
13/13 - 0s - loss: 0.2551 - accuracy: 1.0000 - val_loss: 1.0903 - val_accuracy: 0.7000 - 44ms/epoch - 3ms/step
13/13 - 0s - loss: 0.2370 - accuracy: 1.0000 - val_loss: 1.0877 - val_accuracy: 0.6667 - 29ms/epoch - 2ms/step
13/13 - 0s - loss: 0.2199 - accuracy: 1.0000 - val_loss: 1.0776 - val_accuracy: 0.6667 - 29ms/epoch - 2ms/step
13/13 - 0s - loss: 0.2050 - accuracy: 1.0000 - val_loss: 1.0675 - val_accuracy: 0.7000 - 30ms/epoch - 2ms/step
13/13 - 0s - loss: 0.1919 - accuracy: 1.0000 - val_loss: 1.0592 - val_accuracy: 0.7000 - 44ms/epoch - 3ms/step
13/13 - 0s - loss: 0.1812 - accuracy: 1.0000 - val_loss: 1.0554 - val_accuracy: 0.6667 - 29ms/epoch - 2ms/step
13/13 - 0s - loss: 0.1691 - accuracy: 1.0000 - val_loss: 1.0477 - val_accuracy: 0.7000 - 30ms/epoch - 2ms/step
13/13 - 0s - loss: 0.1599 - accuracy: 1.0000 - val_loss: 1.0377 - val_accuracy: 0.7000 - 45ms/epoch - 3ms/step
13/13 - 0s - loss: 0.1515 - accuracy: 1.0000 - val_loss: 1.0341 - val_accuracy: 0.6667 - 30ms/epoch - 2ms/step
13/13 - 0s - loss: 0.1426 - accuracy: 1.0000 - val_loss: 1.0291 - val_accuracy: 0.7000 - 29ms/epoch - 2ms/step
13/13 - 0s - loss: 0.1344 - accuracy: 1.0000 - val_loss: 1.0264 - val_accuracy: 0.7000 - 30ms/epoch - 2ms/step
13/13 - 0s - loss: 3.0160 - accuracy: 0.1000 - val_loss: 2.0646 - val_accuracy: 0.2000 - 379ms/epoch - 29ms/step
13/13 - 0s - loss: 2.2510 - accuracy: 0.1800 - val_loss: 1.9801 - val_accuracy: 0.3333 - 44ms/epoch - 3ms/step
13/13 - 0s - loss: 1.7851 - accuracy: 0.4000 - val_loss: 1.8177 - val_accuracy: 0.2667 - 29ms/epoch - 2ms/step
13/13 - 0s - loss: 1.3652 - accuracy: 0.5400 - val_loss: 1.6107 - val_accuracy: 0.4667 - 29ms/epoch - 2ms/step
13/13 - 0s - loss: 1.1317 - accuracy: 0.7800 - val_loss: 1.4723 - val_accuracy: 0.6000 - 44ms/epoch - 3ms/step
13/13 - 0s - loss: 0.9360 - accuracy: 0.8600 - val_loss: 1.4686 - val_accuracy: 0.5333 - 44ms/epoch - 3ms/step
13/13 - 0s - loss: 0.7383 - accuracy: 0.9400 - val_loss: 1.3536 - val_accuracy: 0.5667 - 29ms/epoch - 2ms/step
13/13 - 0s - loss: 0.5846 - accuracy: 0.9800 - val_loss: 1.2785 - val_accuracy: 0.6667 - 44ms/epoch - 3ms/step
13/13 - 0s - loss: 0.4987 - accuracy: 0.9800 - val_loss: 1.2253 - val_accuracy: 0.6333 - 45ms/epoch - 3ms/step
13/13 - 0s - loss: 0.3921 - accuracy: 1.0000 - val_loss: 1.2655 - val_accuracy: 0.6333 - 45ms/epoch - 3ms/step
13/13 - 0s - loss: 0.3617 - accuracy: 1.0000 - val_loss: 1.1555 - val_accuracy: 0.7000 - 29ms/epoch - 2ms/step
13/13 - 0s - loss: 0.3078 - accuracy: 1.0000 - val_loss: 1.2107 - val_accuracy: 0.6667 - 30ms/epoch - 2ms/step
13/13 - 0s - loss: 0.2618 - accuracy: 1.0000 - val_loss: 1.1166 - val_accuracy: 0.7333 - 44ms/epoch - 3ms/step
13/13 - 0s - loss: 0.2463 - accuracy: 1.0000 - val_loss: 1.0624 - val_accuracy: 0.7000 - 29ms/epoch - 2ms/step
13/13 - 0s - loss: 0.2033 - accuracy: 1.0000 - val_loss: 1.0904 - val_accuracy: 0.7333 - 29ms/epoch - 2ms/step
13/13 - 0s - loss: 0.1799 - accuracy: 1.0000 - val_loss: 1.0865 - val_accuracy: 0.7333 - 45ms/epoch - 3ms/step
13/13 - 0s - loss: 0.1682 - accuracy: 1.0000 - val_loss: 1.0385 - val_accuracy: 0.7333 - 29ms/epoch - 2ms/step
13/13 - 0s - loss: 0.1484 - accuracy: 1.0000 - val_loss: 1.0676 - val_accuracy: 0.7000 - 45ms/epoch - 3ms/step
13/13 - 0s - loss: 0.1349 - accuracy: 1.0000 - val_loss: 1.0447 - val_accuracy: 0.7000 - 30ms/epoch - 2ms/step
13/13 - 0s - loss: 0.1255 - accuracy: 1.0000 - val_loss: 0.9935 - val_accuracy: 0.7333 - 29ms/epoch - 2ms/step
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
13/13 - 0s - loss: 0.1135 - accuracy: 1.0000 - val_loss: 1.0451 - val_accuracy: 0.7333 - 47ms/epoch - 4ms/step
 5087232/11490434 [============>.................] - ETA: 0s
13/13 - 0s - loss: 0.1041 - accuracy: 1.0000 - val_loss: 1.0142 - val_accuracy: 0.7333 - 34ms/epoch - 3ms/step
13/13 - 0s - loss: 0.0983 - accuracy: 1.0000 - val_loss: 0.9893 - val_accuracy: 0.7333 - 34ms/epoch - 3ms/step
13/13 - 0s - loss: 0.0913 - accuracy: 1.0000 - val_loss: 0.9807 - val_accuracy: 0.7000 - 32ms/epoch - 2ms/step
13/13 - 0s - loss: 0.0868 - accuracy: 1.0000 - val_loss: 0.9715 - val_accuracy: 0.7333 - 34ms/epoch - 3ms/step
13/13 - 0s - loss: 0.0788 - accuracy: 1.0000 - val_loss: 0.9850 - val_accuracy: 0.7333 - 47ms/epoch - 4ms/step
13/13 - 0s - loss: 0.0749 - accuracy: 1.0000 - val_loss: 0.9750 - val_accuracy: 0.7333 - 47ms/epoch - 4ms/step
13/13 - 0s - loss: 0.0730 - accuracy: 1.0000 - val_loss: 0.9570 - val_accuracy: 0.7667 - 50ms/epoch - 4ms/step
13/13 - 0s - loss: 0.0681 - accuracy: 1.0000 - val_loss: 0.9895 - val_accuracy: 0.7333 - 48ms/epoch - 4ms/step
13/13 - 0s - loss: 0.0635 - accuracy: 1.0000 - val_loss: 0.9560 - val_accuracy: 0.7333 - 48ms/epoch - 4ms/step
13/13 - 0s - loss: 2.3619 - accuracy: 0.1400 - val_loss: 2.0567 - val_accuracy: 0.3667 - 378ms/epoch - 29ms/step
13/13 - 0s - loss: 1.9696 - accuracy: 0.4200 - val_loss: 1.9381 - val_accuracy: 0.4667 - 47ms/epoch - 4ms/step
13/13 - 0s - loss: 1.6671 - accuracy: 0.6400 - val_loss: 1.8300 - val_accuracy: 0.4333 - 34ms/epoch - 3ms/step
13/13 - 0s - loss: 1.4340 - accuracy: 0.7400 - val_loss: 1.7113 - val_accuracy: 0.4333 - 48ms/epoch - 4ms/step
13/13 - 0s - loss: 1.2342 - accuracy: 0.7600 - val_loss: 1.6044 - val_accuracy: 0.5000 - 35ms/epoch - 3ms/step
13/13 - 0s - loss: 1.0721 - accuracy: 0.8600 - val_loss: 1.5232 - val_accuracy: 0.5333 - 36ms/epoch - 3ms/step
13/13 - 0s - loss: 0.9348 - accuracy: 0.9000 - val_loss: 1.4596 - val_accuracy: 0.5667 - 48ms/epoch - 4ms/step
13/13 - 0s - loss: 0.8178 - accuracy: 0.9200 - val_loss: 1.3921 - val_accuracy: 0.6000 - 35ms/epoch - 3ms/step
13/13 - 0s - loss: 0.7223 - accuracy: 0.9400 - val_loss: 1.3404 - val_accuracy: 0.6333 - 48ms/epoch - 4ms/step
13/13 - 0s - loss: 0.6404 - accuracy: 0.9600 - val_loss: 1.3065 - val_accuracy: 0.6667 - 47ms/epoch - 4ms/step
13/13 - 0s - loss: 0.5772 - accuracy: 1.0000 - val_loss: 1.2644 - val_accuracy: 0.6333 - 34ms/epoch - 3ms/step
13/13 - 0s - loss: 0.5199 - accuracy: 1.0000 - val_loss: 1.2558 - val_accuracy: 0.6667 - 33ms/epoch - 3ms/step
13/13 - 0s - loss: 0.4695 - accuracy: 1.0000 - val_loss: 1.2258 - val_accuracy: 0.6667 - 48ms/epoch - 4ms/step
13/13 - 0s - loss: 0.4238 - accuracy: 1.0000 - val_loss: 1.1897 - val_accuracy: 0.6667 - 34ms/epoch - 3ms/step
13/13 - 0s - loss: 0.3848 - accuracy: 1.0000 - val_loss: 1.1651 - val_accuracy: 0.6667 - 34ms/epoch - 3ms/step
13/13 - 0s - loss: 0.3525 - accuracy: 1.0000 - val_loss: 1.1503 - val_accuracy: 0.7333 - 48ms/epoch - 4ms/step
13/13 - 0s - loss: 0.3245 - accuracy: 1.0000 - val_loss: 1.1374 - val_accuracy: 0.7000 - 48ms/epoch - 4ms/step
13/13 - 0s - loss: 0.2992 - accuracy: 1.0000 - val_loss: 1.1174 - val_accuracy: 0.6667 - 34ms/epoch - 3ms/step
13/13 - 0s - loss: 0.2745 - accuracy: 1.0000 - val_loss: 1.1119 - val_accuracy: 0.6667 - 48ms/epoch - 4ms/step
13/13 - 0s - loss: 0.2551 - accuracy: 1.0000 - val_loss: 1.0903 - val_accuracy: 0.7000 - 34ms/epoch - 3ms/step
13/13 - 0s - loss: 0.2370 - accuracy: 1.0000 - val_loss: 1.0877 - val_accuracy: 0.6667 - 47ms/epoch - 4ms/step
13/13 - 0s - loss: 0.2199 - accuracy: 1.0000 - val_loss: 1.0776 - val_accuracy: 0.6667 - 34ms/epoch - 3ms/step
13/13 - 0s - loss: 0.2050 - accuracy: 1.0000 - val_loss: 1.0675 - val_accuracy: 0.7000 - 47ms/epoch - 4ms/step
13/13 - 0s - loss: 0.1919 - accuracy: 1.0000 - val_loss: 1.0592 - val_accuracy: 0.7000 - 48ms/epoch - 4ms/step
13/13 - 0s - loss: 0.1812 - accuracy: 1.0000 - val_loss: 1.0554 - val_accuracy: 0.6667 - 49ms/epoch - 4ms/step
13/13 - 0s - loss: 0.1691 - accuracy: 1.0000 - val_loss: 1.0477 - val_accuracy: 0.7000 - 49ms/epoch - 4ms/step
13/13 - 0s - loss: 0.1599 - accuracy: 1.0000 - val_loss: 1.0377 - val_accuracy: 0.7000 - 35ms/epoch - 3ms/step
13/13 - 0s - loss: 0.1515 - accuracy: 1.0000 - val_loss: 1.0341 - val_accuracy: 0.6667 - 47ms/epoch - 4ms/step
13/13 - 0s - loss: 0.1426 - accuracy: 1.0000 - val_loss: 1.0291 - val_accuracy: 0.7000 - 47ms/epoch - 4ms/step
13/13 - 0s - loss: 0.1344 - accuracy: 1.0000 - val_loss: 1.0264 - val_accuracy: 0.7000 - 48ms/epoch - 4ms/step
13/13 - 0s - loss: 3.0160 - accuracy: 0.1000 - val_loss: 2.0646 - val_accuracy: 0.2000 - 440ms/epoch - 34ms/step
13/13 - 0s - loss: 2.2510 - accuracy: 0.1800 - val_loss: 1.9801 - val_accuracy: 0.3333 - 48ms/epoch - 4ms/step
13/13 - 0s - loss: 1.7851 - accuracy: 0.4000 - val_loss: 1.8177 - val_accuracy: 0.2667 - 33ms/epoch - 3ms/step
13/13 - 0s - loss: 1.3652 - accuracy: 0.5400 - val_loss: 1.6107 - val_accuracy: 0.4667 - 34ms/epoch - 3ms/step
13/13 - 0s - loss: 1.1317 - accuracy: 0.7800 - val_loss: 1.4723 - val_accuracy: 0.6000 - 47ms/epoch - 4ms/step
13/13 - 0s - loss: 0.9360 - accuracy: 0.8600 - val_loss: 1.4686 - val_accuracy: 0.5333 - 49ms/epoch - 4ms/step
13/13 - 0s - loss: 0.7383 - accuracy: 0.9400 - val_loss: 1.3536 - val_accuracy: 0.5667 - 48ms/epoch - 4ms/step
13/13 - 0s - loss: 0.5846 - accuracy: 0.9800 - val_loss: 1.2785 - val_accuracy: 0.6667 - 48ms/epoch - 4ms/step
13/13 - 0s - loss: 0.4987 - accuracy: 0.9800 - val_loss: 1.2253 - val_accuracy: 0.6333 - 47ms/epoch - 4ms/step
13/13 - 0s - loss: 0.3921 - accuracy: 1.0000 - val_loss: 1.2655 - val_accuracy: 0.6333 - 47ms/epoch - 4ms/step
13/13 - 0s - loss: 0.3617 - accuracy: 1.0000 - val_loss: 1.1555 - val_accuracy: 0.7000 - 48ms/epoch - 4ms/step
13/13 - 0s - loss: 0.3078 - accuracy: 1.0000 - val_loss: 1.2107 - val_accuracy: 0.6667 - 46ms/epoch - 4ms/step
13/13 - 0s - loss: 0.2618 - accuracy: 1.0000 - val_loss: 1.1166 - val_accuracy: 0.7333 - 47ms/epoch - 4ms/step
13/13 - 0s - loss: 0.2463 - accuracy: 1.0000 - val_loss: 1.0624 - val_accuracy: 0.7000 - 49ms/epoch - 4ms/step
13/13 - 0s - loss: 0.2033 - accuracy: 1.0000 - val_loss: 1.0904 - val_accuracy: 0.7333 - 34ms/epoch - 3ms/step
13/13 - 0s - loss: 0.1799 - accuracy: 1.0000 - val_loss: 1.0865 - val_accuracy: 0.7333 - 50ms/epoch - 4ms/step
13/13 - 0s - loss: 0.1682 - accuracy: 1.0000 - val_loss: 1.0385 - val_accuracy: 0.7333 - 35ms/epoch - 3ms/step
13/13 - 0s - loss: 0.1484 - accuracy: 1.0000 - val_loss: 1.0676 - val_accuracy: 0.7000 - 34ms/epoch - 3ms/step
13/13 - 0s - loss: 0.1349 - accuracy: 1.0000 - val_loss: 1.0447 - val_accuracy: 0.7000 - 36ms/epoch - 3ms/step
13/13 - 0s - loss: 0.1255 - accuracy: 1.0000 - val_loss: 0.9935 - val_accuracy: 0.7333 - 34ms/epoch - 3ms/step
 </code>
 </pre>
 </details>

---

## 15. tutorial_qubit_tapering.html <a name="demo14"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_qubit_tapering.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
  (-1.7101641607053724) [I0]
+ (0.2145954888959534) [Z2]
+ (0.21459548889595348) [Z3]
+ (0.7306880119480534) [Z0]
+ (0.7306880119480537) [Z1]
+ (-0.014203509377566538) [Y1 Y3]
+ (-0.014203509377566538) [X1 X3]
+ (0.04383965196882916) [Y0 Y2]
+ (0.04383965196882916) [X0 X2]
+ (0.11793755332548975) [Z0 Z2]
+ (0.11793755332548975) [Z1 Z3]
+ (0.14964510367723513) [Z0 Z3]
+ (0.14964510367723513) [Z1 Z2]
+ (0.1867894262579603) [Z2 Z3]
+ (0.2363564449772685) [Z0 Z1]
+ (0.058043160912742364) [Y0 Z1 Y2]
+ (0.058043160912742364) [X0 Z1 X2]
+ (0.058043160912742364) [Y1 Z2 Y3]
+ (0.058043160912742364) [X1 Z2 X3]
+ (-0.03170755035174541) [Y0 Y1 X2 X3]
+ (-0.03170755035174541) [X0 X1 Y2 Y3]
+ (-0.014203509377566538) [Y0 Z1 Y2 Z3]
+ (-0.014203509377566538) [X0 Z1 X2 Z3]
+ (0.03170755035174541) [Y0 X1 X2 Y3]
+ (0.03170755035174541) [X0 Y1 Y2 X3]
+ (0.04383965196882916) [Z0 Y1 Z2 Y3]
+ (0.04383965196882916) [Z0 X1 Z2 X3]
  ((-1.9460392673563507+0j)) [I0]
+ ((-0.11608632269279137+0j)) [X0]
+ ((0.11608632269279137+0j)) [X1]
+ ((0.5160925230521+0j)) [Z0]
+ ((0.5160925230521001+0j)) [Z1]
+ ((-0.1268302014069816+0j)) [Y0 Y1]
+ ((-0.11608632182548467+0j)) [X0 Z1]
+ ((0.11608632182548467+0j)) [Z0 X1]
+ ((0.12385566388075847+0j)) [Z0 Z1]
n: 1, E: -2.85436865 Ha, Params: [ 6.34151007e-02  4.33653349e-10 -4.33653349e-10]
n: 3, E: -2.86179046 Ha, Params: [ 0.10911501  0.00865135 -0.007968  ]
n: 5, E: -2.86238874 Ha, Params: [ 0.12112298  0.01747361 -0.01577759]
n: 7, E: -2.86251958 Ha, Params: [ 0.1250903   0.02371963 -0.02121041]
n: 9, E: -2.86256599 Ha, Params: [ 0.12679592  0.02778521 -0.02470784]
n: 11, E: -2.86258387 Ha, Params: [ 0.12768633  0.03036515 -0.02690639]
n: 13, E: -2.86259082 Ha, Params: [ 0.12820072  0.03198859 -0.02827676]
n: 15, E: -2.86259352 Ha, Params: [ 0.12851121  0.03300724 -0.02912787]
 </code>
 </pre>
 </details>

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_qubit_tapering.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
  (-1.710164160706292) [I0]
+ (0.2145954888958696) [Z3]
+ (0.21459548889586966) [Z2]
+ (0.7306880119480709) [Z0]
+ (0.7306880119480709) [Z1]
+ (-0.014203509377549904) [Y1 Y3]
+ (-0.014203509377549904) [X1 X3]
+ (0.04383965196890963) [Y0 Y2]
+ (0.04383965196890963) [X0 X2]
+ (0.11793755332561838) [Z0 Z2]
+ (0.11793755332561838) [Z1 Z3]
+ (0.14964510367739073) [Z0 Z3]
+ (0.14964510367739073) [Z1 Z2]
+ (0.18678942625822093) [Z2 Z3]
+ (0.23635644497749186) [Z0 Z1]
+ (0.05804316091280627) [Y1 Z2 Y3]
+ (0.05804316091280627) [X1 Z2 X3]
+ (0.05804316091280628) [Y0 Z1 Y2]
+ (0.05804316091280628) [X0 Z1 X2]
+ (-0.03170755035177234) [Y0 Y1 X2 X3]
+ (-0.03170755035177234) [X0 X1 Y2 Y3]
+ (-0.014203509377549904) [Y0 Z1 Y2 Z3]
+ (-0.014203509377549904) [X0 Z1 X2 Z3]
+ (0.03170755035177234) [Y0 X1 X2 Y3]
+ (0.03170755035177234) [X0 Y1 Y2 X3]
+ (0.04383965196890963) [Z0 Y1 Z2 Y3]
+ (0.04383965196890963) [Z0 X1 Z2 X3]
  ((-1.946039267357528+0j)) [I0]
+ ((-0.11608632269291903+0j)) [X0]
+ ((0.11608632269291903+0j)) [X1]
+ ((0.5160925230522011+0j)) [Z0]
+ ((0.5160925230522011+0j)) [Z1]
+ ((-0.1268302014070893+0j)) [Y0 Y1]
+ ((-0.1160863218256125+0j)) [X0 Z1]
+ ((0.11608632182561249+0j)) [Z0 X1]
+ ((0.12385566388093128+0j)) [Z0 Z1]
n: 5, E: -2.86236975 Ha, Params: [ 0.12087368  0.01627666 -0.01627666]
n: 10, E: -2.86256772 Ha, Params: [ 0.12687479  0.02808328 -0.02808328]
n: 15, E: -2.86259160 Ha, Params: [ 0.12820847  0.03248734 -0.03248734]
n: 20, E: -2.86259476 Ha, Params: [ 0.12867238  0.03409235 -0.03409235]
n: 25, E: -2.86259518 Ha, Params: [ 0.12884045  0.03467702 -0.03467702]
n: 30, E: -2.86259523 Ha, Params: [ 0.12890162  0.03489005 -0.03489005]
n: 35, E: -2.86259524 Ha, Params: [ 0.1289239   0.03496769 -0.03496769]
n: 40, E: -2.86259524 Ha, Params: [ 0.12893201  0.03499598 -0.03499598]
 </code>
 </pre>
 </details>

---

## 16. tutorial_quantum_chemistry.html <a name="demo15"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_quantum_chemistry.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
  (-46.46390691340334) [I0]
+ (0.7829652070485085) [Z11]
+ (0.7829652070485086) [Z10]
+ (0.8084591005186499) [Z13]
+ (0.8084591005186501) [Z12]
+ (1.2034393391316274) [Z4]
+ (1.2034393391316274) [Z5]
+ (1.3096876618627467) [Z6]
+ (1.3096876618627469) [Z7]
+ (1.369352571170564) [Z8]
+ (1.3693525711705643) [Z9]
+ (1.653893830546777) [Z3]
+ (1.6538938305467772) [Z2]
+ (12.412630714437382) [Z1]
+ (12.412630714437384) [Z0]
+ (-7.954224637309742e-06) [Y11 Y13]
+ (-7.954224637309742e-06) [X11 X13]
+ (-7.76508227384406e-07) [Y3 Y5]
+ (-7.76508227384406e-07) [X3 X5]
+ (5.92928059644322e-07) [Y4 Y6]
+ (5.92928059644322e-07) [X4 X6]
+ (1.6021751126734014e-06) [Y2 Y4]
+ (1.6021751126734014e-06) [X2 X4]
+ (1.854056548945349e-06) [Y5 Y7]
+ (1.854056548945349e-06) [X5 X7]
+ (8.194104970257743e-06) [Y10 Y12]
+ (8.194104970257743e-06) [X10 X12]
+ (0.0032769650657454683) [Y1 Y3]
+ (0.0032769650657454683) [X1 X3]
+ (0.10433061485305034) [Y0 Y2]
+ (0.10433061485305034) [X0 X2]
+ (0.11270381859113379) [Z10 Z12]
+ (0.11270381859113379) [Z11 Z13]
+ (0.11383573685301082) [Z4 Z12]
+ (0.11383573685301082) [Z5 Z13]
+ (0.11952441016884777) [Z6 Z10]
+ (0.11952441016884777) [Z7 Z11]
+ (0.12489977362985362) [Z4 Z10]
+ (0.12489977362985362) [Z5 Z11]
+ (0.12495799328192063) [Z2 Z4]
+ (0.12495799328192063) [Z3 Z5]
+ (0.1279949280138784) [Z2 Z10]
+ (0.1279949280138784) [Z3 Z11]
+ (0.1340173737264723) [Z6 Z12]
+ (0.1340173737264723) [Z7 Z13]
+ (0.13701191913056504) [Z4 Z6]
+ (0.13701191913056504) [Z5 Z7]
+ (0.1373494221014489) [Z6 Z11]
+ (0.1373494221014489) [Z7 Z10]
+ (0.13739112375004509) [Z2 Z6]
+ (0.13739112375004509) [Z3 Z7]
+ (0.13766859133610862) [Z8 Z10]
+ (0.13766859133610862) [Z9 Z11]
+ (0.14011294749708048) [Z2 Z12]
+ (0.14011294749708048) [Z3 Z13]
+ (0.141389035901418) [Z10 Z13]
+ (0.141389035901418) [Z11 Z12]
+ (0.14257991128696756) [Z4 Z11]
+ (0.14257991128696756) [Z5 Z10]
+ (0.14722930783254784) [Z8 Z11]
+ (0.14722930783254784) [Z9 Z10]
+ (0.14899426171381508) [Z4 Z7]
+ (0.14899426171381508) [Z5 Z6]
+ (0.14926347060028072) [Z10 Z11]
+ (0.14960692557250652) [Z4 Z8]
+ (0.14960692557250652) [Z5 Z9]
+ (0.14973497005431669) [Z8 Z12]
+ (0.14973497005431669) [Z9 Z13]
+ (0.15071405482288183) [Z2 Z8]
+ (0.15071405482288183) [Z3 Z9]
+ (0.15138342699108792) [Z6 Z13]
+ (0.15138342699108792) [Z7 Z12]
+ (0.15215040622646006) [Z4 Z13]
+ (0.15215040622646006) [Z5 Z12]
+ (0.15337959171050725) [Z2 Z11]
+ (0.15337959171050725) [Z3 Z10]
+ (0.1543576006530095) [Z12 Z13]
+ (0.15569017457252965) [Z2 Z13]
+ (0.15569017457252965) [Z3 Z12]
+ (0.15582280685854688) [Z8 Z13]
+ (0.15582280685854688) [Z9 Z12]
+ (0.15676384610169716) [Z4 Z9]
+ (0.15676384610169716) [Z5 Z8]
+ (0.15755303804354095) [Z4 Z5]
+ (0.1607975504669552) [Z2 Z5]
+ (0.1607975504669552) [Z3 Z4]
+ (0.16756669356166604) [Z6 Z8]
+ (0.16756669356166604) [Z7 Z9]
+ (0.16853492794638303) [Z2 Z7]
+ (0.16853492794638303) [Z3 Z6]
+ (0.18144009362929264) [Z6 Z9]
+ (0.18144009362929264) [Z7 Z8]
+ (0.18189081243730704) [Z2 Z3]
+ (0.1869081483103697) [Z2 Z9]
+ (0.1869081483103697) [Z3 Z8]
+ (0.19299700269847894) [Z0 Z10]
+ (0.19299700269847894) [Z1 Z11]
+ (0.19392574334966958) [Z6 Z7]
+ (0.1966174995972715) [Z0 Z4]
+ (0.1966174995972715) [Z1 Z5]
+ (0.19936332691269643) [Z0 Z5]
+ (0.19936332691269643) [Z1 Z4]
+ (0.20072843554583047) [Z0 Z11]
+ (0.20072843554583047) [Z1 Z10]
+ (0.21102681234276385) [Z0 Z12]
+ (0.21102681234276385) [Z1 Z13]
+ (0.21631059809658926) [Z0 Z13]
+ (0.21631059809658926) [Z1 Z12]
+ (0.22003977240276795) [Z8 Z9]
+ (0.23671071740409028) [Z0 Z2]
+ (0.23671071740409028) [Z1 Z3]
+ (0.2416469683173428) [Z0 Z6]
+ (0.2416469683173428) [Z1 Z7]
+ (0.24853517285965882) [Z0 Z7]
+ (0.24853517285965882) [Z1 Z6]
+ (0.25129435573109415) [Z0 Z3]
+ (0.25129435573109415) [Z1 Z2]
+ (0.2723251845037055) [Z0 Z8]
+ (0.2723251845037055) [Z1 Z9]
+ (0.27883454573778327) [Z0 Z9]
+ (0.27883454573778327) [Z1 Z8]
+ (1.1861764484126927) [Z0 Z1]
+ (-3.886639565415012e-06) [Y2 Z3 Y4]
+ (-3.886639565415012e-06) [X2 Z3 X4]
+ (-3.886639565415012e-06) [Y3 Z4 Y5]
+ (-3.886639565415012e-06) [X3 Z4 X5]
+ (1.072274838943606e-05) [Y10 Z11 Y12]
+ (1.072274838943606e-05) [X10 Z11 X12]
+ (1.072274838943606e-05) [Y11 Z12 Y13]
+ (1.072274838943606e-05) [X11 Z12 X13]
+ (1.2260276990928063e-05) [Y4 Z5 Y6]
+ (1.2260276990928063e-05) [X4 Z5 X6]
+ (1.2260276990928063e-05) [Y5 Z6 Y7]
+ (1.2260276990928063e-05) [X5 Z6 X7]
+ (0.1250703688397039) [Y1 Z2 Y3]
+ (0.1250703688397039) [X1 Z2 X3]
+ (0.12507036883970393) [Y0 Z1 Y2]
+ (0.12507036883970393) [X0 Z1 X2]
+ (-0.03831466937344923) [Y4 Y5 X12 X13]
+ (-0.03831466937344923) [X4 X5 Y12 Y13]
+ (-0.03619409348748788) [Y2 Y3 X8 X9]
+ (-0.03619409348748788) [X2 X3 Y8 Y9]
+ (-0.03583955718503457) [Y2 Y3 X4 X5]
+ (-0.03583955718503457) [X2 X3 Y4 Y5]
+ (-0.031143804196337957) [Y2 Y3 X6 X7]
+ (-0.031143804196337957) [X2 X3 Y6 Y7]
+ (-0.028685217310284214) [Y10 Y11 X12 X13]
+ (-0.028685217310284214) [X10 X11 Y12 Y13]
+ (-0.025384663696628798) [Y2 Y3 X10 X11]
+ (-0.025384663696628798) [X2 X3 Y10 Y11]
+ (-0.019028318718282793) [Y3 X4 X11 Y12]
+ (-0.019028318718282793) [X3 Y4 Y11 X12]
+ (-0.01782501193260116) [Y6 Y7 X10 X11]
+ (-0.01782501193260116) [X6 X7 Y10 Y11]
+ (-0.017680137657113928) [Y4 Y5 X10 X11]
+ (-0.017680137657113928) [X4 X5 Y10 Y11]
+ (-0.01736605326461565) [Y6 Y7 X12 X13]
+ (-0.01736605326461565) [X6 X7 Y12 Y13]
+ (-0.01557722707544916) [Y2 Y3 X12 X13]
+ (-0.01557722707544916) [X2 X3 Y12 Y13]
+ (-0.014583638327003835) [Y0 Y1 X2 X3]
+ (-0.014583638327003835) [X0 X1 Y2 Y3]
+ (-0.013873400067626593) [Y6 Y7 X8 X9]
+ (-0.013873400067626593) [X6 X7 Y8 Y9]
+ (-0.011982342583250026) [Y4 Y5 X6 X7]
+ (-0.011982342583250026) [X4 X5 Y6 Y7]
+ (-0.011285144618312079) [Y5 X6 X11 Y12]
+ (-0.011285144618312079) [X5 Y6 Y11 X12]
+ (-0.009560716496439222) [Y8 Y9 X10 X11]
+ (-0.009560716496439222) [X8 X9 Y10 Y11]
+ (-0.008125248410122873) [Y1 X2 X8 Y9]
+ (-0.008125248410122873) [Y1 Y2 Y8 Y9]
+ (-0.008125248410122873) [X1 X2 X8 X9]
+ (-0.008125248410122873) [X1 Y2 Y8 X9]
+ (-0.007731432847351531) [Y0 Y1 X10 X11]
+ (-0.007731432847351531) [X0 X1 Y10 Y11]
+ (-0.00715692052919062) [Y4 Y5 X8 X9]
+ (-0.00715692052919062) [X4 X5 Y8 Y9]
+ (-0.0068882045423160065) [Y0 Y1 X6 X7]
+ (-0.0068882045423160065) [X0 X1 Y6 Y7]
+ (-0.006509361234077764) [Y0 Y1 X8 X9]
+ (-0.006509361234077764) [X0 X1 Y8 Y9]
+ (-0.00608783680423021) [Y8 Y9 X12 X13]
+ (-0.00608783680423021) [X8 X9 Y12 Y13]
+ (-0.005283785753825449) [Y0 Y1 X12 X13]
+ (-0.005283785753825449) [X0 X1 Y12 Y13]
+ (-0.005143382387686593) [Y3 Y4 X5 X6]
+ (-0.005143382387686593) [X3 X4 Y5 Y6]
+ (-0.004684920226869187) [Y1 X2 X6 Y7]
+ (-0.004684920226869187) [Y1 Y2 Y6 Y7]
+ (-0.004684920226869187) [X1 X2 X6 X7]
+ (-0.004684920226869187) [X1 Y2 Y6 X7]
+ (-0.004575015188890368) [Y1 X2 X12 Y13]
+ (-0.004575015188890368) [Y1 Y2 Y12 Y13]
+ (-0.004575015188890368) [X1 X2 X12 X13]
+ (-0.004575015188890368) [X1 Y2 Y12 X13]
+ (-0.004424843668499158) [Y1 X2 X4 Y5]
+ (-0.004424843668499158) [Y1 Y2 Y4 Y5]
+ (-0.004424843668499158) [X1 X2 X4 X5]
+ (-0.004424843668499158) [X1 Y2 Y4 X5]
+ (-0.00274582731542491) [Y0 Y1 X4 X5]
+ (-0.00274582731542491) [X0 X1 Y4 Y5]
+ (-0.0017991930083825537) [Y1 X2 X10 Y11]
+ (-0.0017991930083825537) [Y1 Y2 Y10 Y11]
+ (-0.0017991930083825537) [X1 X2 X10 X11]
+ (-0.0017991930083825537) [X1 Y2 Y10 X11]
+ (-0.0016639606584095258) [Y2 Z3 Z4 Y6]
+ (-0.0016639606584095258) [X2 Z3 Z4 X6]
+ (-0.0016639606584095258) [Y3 Z5 Z6 Y7]
+ (-0.0016639606584095258) [X3 Z5 Z6 X7]
+ (-0.0004957972886317744) [Y2 Z4 Z5 Y6]
+ (-0.0004957972886317744) [X2 Z4 Z5 X6]
+ (-0.0002922256724535163) [Y7 Y8 X9 X10]
+ (-0.0002922256724535163) [X7 X8 Y9 Y10]
+ (-7.954224637309742e-06) [Y10 Z11 Y12 Z13]
+ (-7.954224637309742e-06) [X10 Z11 X12 Z13]
+ (-7.801538341811656e-06) [Y2 Z3 Y4 Z11]
+ (-7.801538341811656e-06) [X2 Z3 X4 Z11]
+ (-7.801538341811656e-06) [Y3 Z4 Y5 Z10]
+ (-7.801538341811656e-06) [X3 Z4 X5 Z10]
+ (-5.974176874278415e-06) [Y5 X6 X10 Y11]
+ (-5.974176874278415e-06) [Y5 Y6 Y10 Y11]
+ (-5.974176874278415e-06) [X5 X6 X10 X11]
+ (-5.974176874278415e-06) [X5 Y6 Y10 X11]
+ (-4.642978971311799e-06) [Y3 X4 X10 Y11]
+ (-4.642978971311799e-06) [Y3 Y4 Y10 Y11]
+ (-4.642978971311799e-06) [X3 X4 X10 X11]
+ (-4.642978971311799e-06) [X3 Y4 Y10 X11]
+ (-4.281812038809101e-06) [Y4 Z5 Y6 Z11]
+ (-4.281812038809101e-06) [X4 Z5 X6 Z11]
+ (-4.281812038809101e-06) [Y5 Z6 Y7 Z10]
+ (-4.281812038809101e-06) [X5 Z6 X7 Z10]
+ (-3.158559370497905e-06) [Y2 Z3 Y4 Z10]
+ (-3.158559370497905e-06) [X2 Z3 X4 Z10]
+ (-3.158559370497905e-06) [Y3 Z4 Y5 Z11]
+ (-3.158559370497905e-06) [X3 Z4 X5 Z11]
+ (-2.172638016389875e-06) [Y2 X3 X11 Y12]
+ (-2.172638016389875e-06) [Y2 Y3 Y11 Y12]
+ (-2.172638016389875e-06) [X2 X3 X11 X12]
+ (-2.172638016389875e-06) [X2 Y3 Y11 X12]
+ (-1.8781495254895086e-06) [Z4 Y10 Z11 Y12]
+ (-1.8781495254895086e-06) [Z4 X10 Z11 X12]
+ (-1.8781495254895086e-06) [Z5 Y11 Z12 Y13]
+ (-1.8781495254895086e-06) [Z5 X11 Z12 X13]
+ (-1.4548066961249558e-06) [Y3 X4 X6 Y7]
+ (-1.4548066961249558e-06) [Y3 Y4 Y6 Y7]
+ (-1.4548066961249558e-06) [X3 X4 X6 X7]
+ (-1.4548066961249558e-06) [X3 Y4 Y6 X7]
+ (-1.1953920805599723e-06) [Y2 Z3 Y4 Z7]
+ (-1.1953920805599723e-06) [X2 Z3 X4 Z7]
+ (-1.1953920805599723e-06) [Y3 Z4 Y5 Z6]
+ (-1.1953920805599723e-06) [X3 Z4 X5 Z6]
+ (-1.190733108592262e-06) [Z0 Y3 Z4 Y5]
+ (-1.190733108592262e-06) [Z0 X3 Z4 X5]
+ (-1.190733108592262e-06) [Z1 Y2 Z3 Y4]
+ (-1.190733108592262e-06) [Z1 X2 Z3 X4]
+ (-1.1094125010309517e-06) [Z2 Y11 Z12 Y13]
+ (-1.1094125010309517e-06) [Z2 X11 Z12 X13]
+ (-1.1094125010309517e-06) [Z3 Y10 Z11 Y12]
+ (-1.1094125010309517e-06) [Z3 X10 Z11 X12]
+ (-8.336695575657873e-07) [Z0 Y2 Z3 Y4]
+ (-8.336695575657873e-07) [Z0 X2 Z3 X4]
+ (-8.336695575657873e-07) [Z1 Y3 Z4 Y5]
+ (-8.336695575657873e-07) [Z1 X3 Z4 X5]
+ (-7.956666995379789e-07) [Y3 X4 X8 Y9]
+ (-7.956666995379789e-07) [Y3 Y4 Y8 Y9]
+ (-7.956666995379789e-07) [X3 X4 X8 X9]
+ (-7.956666995379789e-07) [X3 Y4 Y8 X9]
+ (-7.76508227384406e-07) [Y2 Z3 Y4 Z5]
+ (-7.76508227384406e-07) [X2 Z3 X4 Z5]
+ (-6.628427258131928e-07) [Y8 X9 X11 Y12]
+ (-6.628427258131928e-07) [Y8 Y9 Y11 Y12]
+ (-6.628427258131928e-07) [X8 X9 X11 X12]
+ (-6.628427258131928e-07) [X8 Y9 Y11 X12]
+ (-5.769436585232462e-07) [Y2 Z3 Y4 Z9]
+ (-5.769436585232462e-07) [X2 Z3 X4 Z9]
+ (-5.769436585232462e-07) [Y3 Z4 Y5 Z8]
+ (-5.769436585232462e-07) [X3 Z4 X5 Z8]
+ (-5.627722020314877e-07) [Y0 X1 X11 Y12]
+ (-5.627722020314877e-07) [Y0 Y1 Y11 Y12]
+ (-5.627722020314877e-07) [X0 X1 X11 X12]
+ (-5.627722020314877e-07) [X0 Y1 Y11 X12]
+ (-5.471606040103832e-07) [Y1 X2 X11 Y12]
+ (-5.471606040103832e-07) [X1 Y2 Y11 X12]
+ (-3.5706355102636894e-07) [Y0 X1 X3 Y4]
+ (-3.5706355102636894e-07) [Y0 Y1 Y3 Y4]
+ (-3.5706355102636894e-07) [X0 X1 X3 X4]
+ (-3.5706355102636894e-07) [X0 Y1 Y3 X4]
+ (-1.933212141894207e-07) [Y1 X2 X3 Y4]
+ (-1.933212141894207e-07) [X1 Y2 Y3 X4]
+ (-1.2919458272498863e-07) [Y1 Z2 Z3 Y5]
+ (-1.2919458272498863e-07) [X1 Z2 Z3 X5]
+ (-3.2261049427570525e-09) [Y1 Y2 X5 X6]
+ (-3.2261049427570525e-09) [X1 X2 Y5 Y6]
+ (3.2261049427570525e-09) [Y1 X2 X5 Y6]
+ (3.2261049427570525e-09) [X1 Y2 Y5 X6]
+ (3.2284047463138066e-08) [Y4 Z5 Y6 Z12]
+ (3.2284047463138066e-08) [X4 Z5 X6 Z12]
+ (3.2284047463138066e-08) [Y5 Z6 Y7 Z13]
+ (3.2284047463138066e-08) [X5 Z6 X7 Z13]
+ (1.6728571570357397e-07) [Y0 Z1 Z3 Y4]
+ (1.6728571570357397e-07) [X0 Z1 Z3 X4]
+ (1.6728571570357397e-07) [Y1 Z2 Z4 Y5]
+ (1.6728571570357397e-07) [X1 Z2 Z4 X5]
+ (1.933212141894207e-07) [Y1 Y2 X3 X4]
+ (1.933212141894207e-07) [X1 X2 Y3 Y4]
+ (2.187230410147326e-07) [Y2 Z3 Y4 Z8]
+ (2.187230410147326e-07) [X2 Z3 X4 Z8]
+ (2.187230410147326e-07) [Y3 Z4 Y5 Z9]
+ (2.187230410147326e-07) [X3 Z4 X5 Z9]
+ (2.198963765514682e-07) [Y2 X3 X5 Y6]
+ (2.198963765514682e-07) [Y2 Y3 Y5 Y6]
+ (2.198963765514682e-07) [X2 X3 X5 X6]
+ (2.198963765514682e-07) [X2 Y3 Y5 X6]
+ (2.4472648254715786e-07) [Y0 X1 X5 Y6]
+ (2.4472648254715786e-07) [Y0 Y1 Y5 Y6]
+ (2.4472648254715786e-07) [X0 X1 X5 X6]
+ (2.4472648254715786e-07) [X0 Y1 Y5 X6]
+ (2.594146155663929e-07) [Y2 Z3 Y4 Z6]
+ (2.594146155663929e-07) [X2 Z3 X4 Z6]
+ (2.594146155663929e-07) [Y3 Z4 Y5 Z7]
+ (2.594146155663929e-07) [X3 Z4 X5 Z7]
+ (3.606069298927541e-07) [Y0 Z1 Z2 Y4]
+ (3.606069298927541e-07) [X0 Z1 Z2 X4]
+ (3.606069298927541e-07) [Y1 Z3 Z4 Y5]
+ (3.606069298927541e-07) [X1 Z3 Z4 X5]
+ (4.837953361198953e-07) [Y5 X6 X8 Y9]
+ (4.837953361198953e-07) [Y5 Y6 Y8 Y9]
+ (4.837953361198953e-07) [X5 X6 X8 X9]
+ (4.837953361198953e-07) [X5 Y6 Y8 X9]
+ (5.471606040103832e-07) [Y1 Y2 X11 X12]
+ (5.471606040103832e-07) [X1 X2 Y11 Y12]
+ (5.92928059644322e-07) [Z4 Y5 Z6 Y7]
+ (5.92928059644322e-07) [Z4 X5 Z6 X7]
+ (9.344969974164896e-07) [Z8 Y11 Z12 Y13]
+ (9.344969974164896e-07) [Z8 X11 Z12 X13]
+ (9.344969974164896e-07) [Z9 Y10 Z11 Y12]
+ (9.344969974164896e-07) [Z9 X10 Z11 X12]
+ (9.509134566513795e-07) [Z2 Y4 Z5 Y6]
+ (9.509134566513795e-07) [Z2 X4 Z5 X6]
+ (9.509134566513795e-07) [Z3 Y5 Z6 Y7]
+ (9.509134566513795e-07) [Z3 X5 Z6 X7]
+ (1.035792484215545e-06) [Y6 X7 X11 Y12]
+ (1.035792484215545e-06) [Y6 Y7 Y11 Y12]
+ (1.035792484215545e-06) [X6 X7 X11 X12]
+ (1.035792484215545e-06) [X6 Y7 Y11 X12]
+ (1.0632255153595737e-06) [Z2 Y10 Z11 Y12]
+ (1.0632255153595737e-06) [Z2 X10 Z11 X12]
+ (1.0632255153595737e-06) [Z3 Y11 Z12 Y13]
+ (1.0632255153595737e-06) [Z3 X11 Z12 X13]
+ (1.170809833203715e-06) [Z2 Y5 Z6 Y7]
+ (1.170809833203715e-06) [Z2 X5 Z6 X7]
+ (1.170809833203715e-06) [Z3 Y4 Z5 Y6]
+ (1.170809833203715e-06) [Z3 X4 Z5 X6]
+ (1.3980243019051089e-06) [Y4 Z5 Y6 Z8]
+ (1.3980243019051089e-06) [X4 Z5 X6 Z8]
+ (1.3980243019051089e-06) [Y5 Z6 Y7 Z9]
+ (1.3980243019051089e-06) [X5 Z6 X7 Z9]
+ (1.5973397232296824e-06) [Z8 Y10 Z11 Y12]
+ (1.5973397232296824e-06) [Z8 X10 Z11 X12]
+ (1.5973397232296824e-06) [Z9 Y11 Z12 Y13]
+ (1.5973397232296824e-06) [Z9 X11 Z12 X13]
+ (1.6021751126734014e-06) [Z2 Y3 Z4 Y5]
+ (1.6021751126734014e-06) [Z2 X3 Z4 X5]
+ (1.614960788346088e-06) [Z0 Y11 Z12 Y13]
+ (1.614960788346088e-06) [Z0 X11 Z12 X13]
+ (1.614960788346088e-06) [Z1 Y10 Z11 Y12]
+ (1.614960788346088e-06) [Z1 X10 Z11 X12]
+ (1.6923648354697472e-06) [Y4 Z5 Y6 Z10]
+ (1.6923648354697472e-06) [X4 Z5 X6 Z10]
+ (1.6923648354697472e-06) [Y5 Z6 Y7 Z11]
+ (1.6923648354697472e-06) [X5 Z6 X7 Z11]
+ (1.816367371621809e-06) [Z4 Y11 Z12 Y13]
+ (1.816367371621809e-06) [Z4 X11 Z12 X13]
+ (1.816367371621809e-06) [Z5 Y10 Z11 Y12]
+ (1.816367371621809e-06) [Z5 X10 Z11 X12]
+ (1.854056548945349e-06) [Y4 Z5 Y6 Z7]
+ (1.854056548945349e-06) [X4 Z5 X6 Z7]
+ (1.8551374705031223e-06) [Z6 Y10 Z11 Y12]
+ (1.8551374705031223e-06) [Z6 X10 Z11 X12]
+ (1.8551374705031223e-06) [Z7 Y11 Z12 Y13]
+ (1.8551374705031223e-06) [Z7 X11 Z12 X13]
+ (1.8818196380250042e-06) [Y4 Z5 Y6 Z9]
+ (1.8818196380250042e-06) [X4 Z5 X6 Z9]
+ (1.8818196380250042e-06) [Y5 Z6 Y7 Z8]
+ (1.8818196380250042e-06) [X5 Z6 X7 Z8]
+ (2.1777329903774607e-06) [Z0 Y10 Z11 Y12]
+ (2.1777329903774607e-06) [Z0 X10 Z11 X12]
+ (2.1777329903774607e-06) [Z1 Y11 Z12 Y13]
+ (2.1777329903774607e-06) [Z1 X11 Z12 X13]
+ (2.8909299547201853e-06) [Z6 Y11 Z12 Y13]
+ (2.8909299547201853e-06) [Z6 X11 Z12 X13]
+ (2.8909299547201853e-06) [Z7 Y10 Z11 Y12]
+ (2.8909299547201853e-06) [Z7 X10 Z11 X12]
+ (3.0992966612892624e-06) [Z0 Y4 Z5 Y6]
+ (3.0992966612892624e-06) [Z0 X4 Z5 X6]
+ (3.0992966612892624e-06) [Z1 Y5 Z6 Y7]
+ (3.0992966612892624e-06) [Z1 X5 Z6 X7]
+ (3.1173664083424102e-06) [Y0 Z2 Z3 Y4]
+ (3.1173664083424102e-06) [X0 Z2 Z3 X4]
+ (3.344023143836347e-06) [Z0 Y5 Z6 Y7]
+ (3.344023143836347e-06) [Z0 X5 Z6 X7]
+ (3.344023143836347e-06) [Z1 Y4 Z5 Y6]
+ (3.344023143836347e-06) [Z1 X4 Z5 X6]
+ (3.5390099817349426e-06) [Y2 Z3 Y4 Z12]
+ (3.5390099817349426e-06) [X2 Z3 X4 Z12]
+ (3.5390099817349426e-06) [Y3 Z4 Y5 Z13]
+ (3.5390099817349426e-06) [X3 Z4 X5 Z13]
+ (3.6945168971113176e-06) [Y4 X5 X11 Y12]
+ (3.6945168971113176e-06) [Y4 Y5 Y11 Y12]
+ (3.6945168971113176e-06) [X4 X5 X11 X12]
+ (3.6945168971113176e-06) [X4 Y5 Y11 X12]
+ (4.556473753115041e-06) [Y5 X6 X12 Y13]
+ (4.556473753115041e-06) [Y5 Y6 Y12 Y13]
+ (4.556473753115041e-06) [X5 X6 X12 X13]
+ (4.556473753115041e-06) [X5 Y6 Y12 X13]
+ (4.588757800579046e-06) [Y4 Z5 Y6 Z13]
+ (4.588757800579046e-06) [X4 Z5 X6 Z13]
+ (4.588757800579046e-06) [Y5 Z6 Y7 Z12]
+ (4.588757800579046e-06) [X5 Z6 X7 Z12]
+ (5.275783442645291e-06) [Y3 X4 X12 Y13]
+ (5.275783442645291e-06) [Y3 Y4 Y12 Y13]
+ (5.275783442645291e-06) [X3 X4 X12 X13]
+ (5.275783442645291e-06) [X3 Y4 Y12 X13]
+ (8.194104970257743e-06) [Z10 Y11 Z12 Y13]
+ (8.194104970257743e-06) [Z10 X11 Z12 X13]
+ (8.814793424380667e-06) [Y2 Z3 Y4 Z13]
+ (8.814793424380667e-06) [X2 Z3 X4 Z13]
+ (8.814793424380667e-06) [Y3 Z4 Y5 Z12]
+ (8.814793424380667e-06) [X3 Z4 X5 Z12]
+ (0.0002922256724535163) [Y7 X8 X9 Y10]
+ (0.0002922256724535163) [X7 Y8 Y9 X10]
+ (0.0011058984808982794) [Y0 Z1 Y2 Z5]
+ (0.0011058984808982794) [X0 Z1 X2 Z5]
+ (0.0011058984808982794) [Y1 Z2 Y3 Z4]
+ (0.0011058984808982794) [X1 Z2 X3 Z4]
+ (0.0017560659628744746) [Y0 Z1 Y2 Z11]
+ (0.0017560659628744746) [X0 Z1 X2 Z11]
+ (0.0017560659628744746) [Y1 Z2 Y3 Z10]
+ (0.0017560659628744746) [X1 Z2 X3 Z10]
+ (0.0023262348476082014) [Y0 Z1 Y2 Z13]
+ (0.0023262348476082014) [X0 Z1 X2 Z13]
+ (0.0023262348476082014) [Y1 Z2 Y3 Z12]
+ (0.0023262348476082014) [X1 Z2 X3 Z12]
+ (0.00274582731542491) [Y0 X1 X4 Y5]
+ (0.00274582731542491) [X0 Y1 Y4 X5]
+ (0.0029297682785827854) [Y0 Z1 Y2 Z9]
+ (0.0029297682785827854) [X0 Z1 X2 Z9]
+ (0.0029297682785827854) [Y1 Z2 Y3 Z8]
+ (0.0029297682785827854) [X1 Z2 X3 Z8]
+ (0.0032769650657454683) [Y0 Z1 Y2 Z3]
+ (0.0032769650657454683) [X0 Z1 X2 Z3]
+ (0.0033476264706893267) [Y0 Z1 Y2 Z7]
+ (0.0033476264706893267) [X0 Z1 X2 Z7]
+ (0.0033476264706893267) [Y1 Z2 Y3 Z6]
+ (0.0033476264706893267) [X1 Z2 X3 Z6]
+ (0.0034794217292770683) [Y2 Z3 Z5 Y6]
+ (0.0034794217292770683) [X2 Z3 Z5 X6]
+ (0.0034794217292770683) [Y3 Z4 Z6 Y7]
+ (0.0034794217292770683) [X3 Z4 Z6 X7]
+ (0.003555258971257029) [Y0 Z1 Y2 Z10]
+ (0.003555258971257029) [X0 Z1 X2 Z10]
+ (0.003555258971257029) [Y1 Z2 Y3 Z11]
+ (0.003555258971257029) [X1 Z2 X3 Z11]
+ (0.005143382387686593) [Y3 X4 X5 Y6]
+ (0.005143382387686593) [X3 Y4 Y5 X6]
+ (0.005283785753825449) [Y0 X1 X12 Y13]
+ (0.005283785753825449) [X0 Y1 Y12 X13]
+ (0.005530742149397438) [Y0 Z1 Y2 Z4]
+ (0.005530742149397438) [X0 Z1 X2 Z4]
+ (0.005530742149397438) [Y1 Z2 Y3 Z5]
+ (0.005530742149397438) [X1 Z2 X3 Z5]
+ (0.00608783680423021) [Y8 X9 X12 Y13]
+ (0.00608783680423021) [X8 Y9 Y12 X13]
+ (0.006509361234077764) [Y0 X1 X8 Y9]
+ (0.006509361234077764) [X0 Y1 Y8 X9]
+ (0.0068882045423160065) [Y0 X1 X6 Y7]
+ (0.0068882045423160065) [X0 Y1 Y6 X7]
+ (0.006901250036498568) [Y0 Z1 Y2 Z12]
+ (0.006901250036498568) [X0 Z1 X2 Z12]
+ (0.006901250036498568) [Y1 Z2 Y3 Z13]
+ (0.006901250036498568) [X1 Z2 X3 Z13]
+ (0.00715692052919062) [Y4 X5 X8 Y9]
+ (0.00715692052919062) [X4 Y5 Y8 X9]
+ (0.007731432847351531) [Y0 X1 X10 Y11]
+ (0.007731432847351531) [X0 Y1 Y10 X11]
+ (0.008032546697558514) [Y0 Z1 Y2 Z6]
+ (0.008032546697558514) [X0 Z1 X2 Z6]
+ (0.008032546697558514) [Y1 Z2 Y3 Z7]
+ (0.008032546697558514) [X1 Z2 X3 Z7]
+ (0.009560716496439222) [Y8 X9 X10 Y11]
+ (0.009560716496439222) [X8 Y9 Y10 X11]
+ (0.01105501668870566) [Y0 Z1 Y2 Z8]
+ (0.01105501668870566) [X0 Z1 X2 Z8]
+ (0.01105501668870566) [Y1 Z2 Y3 Z9]
+ (0.01105501668870566) [X1 Z2 X3 Z9]
+ (0.011285144618312079) [Y5 Y6 X11 X12]
+ (0.011285144618312079) [X5 X6 Y11 Y12]
+ (0.011307208029984764) [Y7 Z8 Z9 Y11]
+ (0.011307208029984764) [X7 Z8 Z9 X11]
+ (0.011982342583250026) [Y4 X5 X6 Y7]
+ (0.011982342583250026) [X4 Y5 Y6 X7]
+ (0.013873400067626593) [Y6 X7 X8 Y9]
+ (0.013873400067626593) [X6 Y7 Y8 X9]
+ (0.014583638327003835) [Y0 X1 X2 Y3]
+ (0.014583638327003835) [X0 Y1 Y2 X3]
+ (0.01557722707544916) [Y2 X3 X12 Y13]
+ (0.01557722707544916) [X2 Y3 Y12 X13]
+ (0.01736605326461565) [Y6 X7 X12 Y13]
+ (0.01736605326461565) [X6 Y7 Y12 X13]
+ (0.017680137657113928) [Y4 X5 X10 Y11]
+ (0.017680137657113928) [X4 Y5 Y10 X11]
+ (0.01782501193260116) [Y6 X7 X10 Y11]
+ (0.01782501193260116) [X6 Y7 Y10 X11]
+ (0.019028318718282793) [Y3 Y4 X11 X12]
+ (0.019028318718282793) [X3 X4 Y11 Y12]
+ (0.025384663696628798) [Y2 X3 X10 Y11]
+ (0.025384663696628798) [X2 Y3 Y10 X11]
+ (0.025996206267152683) [Y3 Z4 Z5 Y7]
+ (0.025996206267152683) [X3 Z4 Z5 X7]
+ (0.028685217310284214) [Y10 X11 X12 Y13]
+ (0.028685217310284214) [X10 Y11 Y12 X13]
+ (0.029812299601080512) [Y6 Z7 Z8 Y10]
+ (0.029812299601080512) [X6 Z7 Z8 X10]
+ (0.029812299601080512) [Y7 Z9 Z10 Y11]
+ (0.029812299601080512) [X7 Z9 Z10 X11]
+ (0.03010452527353403) [Y6 Z7 Z9 Y10]
+ (0.03010452527353403) [X6 Z7 Z9 X10]
+ (0.03010452527353403) [Y7 Z8 Z10 Y11]
+ (0.03010452527353403) [X7 Z8 Z10 X11]
+ (0.030787440718540504) [Y6 Z8 Z9 Y10]
+ (0.030787440718540504) [X6 Z8 Z9 X10]
+ (0.031143804196337957) [Y2 X3 X6 Y7]
+ (0.031143804196337957) [X2 Y3 Y6 X7]
+ (0.03583955718503457) [Y2 X3 X4 Y5]
+ (0.03583955718503457) [X2 Y3 Y4 X5]
+ (0.03619409348748788) [Y2 X3 X8 Y9]
+ (0.03619409348748788) [X2 Y3 Y8 X9]
+ (0.03831466937344923) [Y4 X5 X12 Y13]
+ (0.03831466937344923) [X4 Y5 Y12 X13]
+ (0.10433061485305033) [Z0 Y1 Z2 Y3]
+ (0.10433061485305033) [Z0 X1 Z2 X3]
+ (3.2041422518157804e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.2041422518157804e-06) [X0 Z1 Z2 Z3 X4]
+ (3.2041422518157804e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.2041422518157804e-06) [X1 Z2 Z3 Z4 X5]
+ (0.12133242248350022) [Y3 Z4 Z5 Z6 Y7]
+ (0.12133242248350022) [X3 Z4 Z5 Z6 X7]
+ (0.12133242248350024) [Y2 Z3 Z4 Z5 Y6]
+ (0.12133242248350024) [X2 Z3 Z4 Z5 X6]
+ (0.22847946311015263) [Y7 Z8 Z9 Z10 Y11]
+ (0.22847946311015263) [X7 Z8 Z9 Z10 X11]
+ (0.22847946311015266) [Y6 Z7 Z8 Z9 Y10]
+ (0.22847946311015266) [X6 Z7 Z8 Z9 X10]
+ (-0.045879424030665986) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (-0.045879424030665986) [X0 Z2 Z3 Z4 Z5 X6]
+ (-0.024353136084505806) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (-0.024353136084505806) [Y2 Z3 Y4 X11 Z12 X13]
+ (-0.024353136084505806) [X2 Z3 X4 Y11 Z12 Y13]
+ (-0.024353136084505806) [X2 Z3 X4 X11 Z12 X13]
+ (-0.024353136084505806) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (-0.024353136084505806) [Y3 Z4 Y5 X10 Z11 X12]
+ (-0.024353136084505806) [X3 Z4 X5 Y10 Z11 Y12]
+ (-0.024353136084505806) [X3 Z4 X5 X10 Z11 X12]
+ (-0.015588277865109666) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (-0.015588277865109666) [X2 Z3 X4 X10 Z11 X12]
+ (-0.015588277865109666) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (-0.015588277865109666) [X3 Z4 X5 X11 Z12 X13]
+ (-0.015225659057106267) [Y3 Z4 Z5 X6 X10 Y11]
+ (-0.015225659057106267) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (-0.015225659057106267) [X3 Z4 Z5 X6 X10 X11]
+ (-0.015225659057106267) [X3 Z4 Z5 Y6 Y10 X11]
+ (-0.014564473640810644) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.014564473640810644) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.014564473640810644) [X7 Z8 Z9 X10 X12 X13]
+ (-0.014564473640810644) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.014411189770078117) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (-0.014411189770078117) [X2 Z3 Z4 Z5 X6 Z11]
+ (-0.014411189770078117) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (-0.014411189770078117) [X3 Z4 Z5 Z6 X7 Z10]
+ (-0.012214985322750487) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.012214985322750487) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.012214985322750487) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.012214985322750487) [X4 Z5 X6 X11 Z12 X13]
+ (-0.012214985322750487) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.012214985322750487) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.012214985322750487) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.012214985322750487) [X5 Z6 X7 X10 Z11 X12]
+ (-0.010263460498886646) [Y2 Z3 X4 X10 Z11 Y12]
+ (-0.010263460498886646) [X2 Z3 Y4 Y10 Z11 X12]
+ (-0.010263460498886646) [Y3 Z4 X5 X11 Z12 Y13]
+ (-0.010263460498886646) [X3 Z4 Y5 Y11 Z12 X13]
+ (-0.008125248410122873) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.008125248410122873) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.007306763969595647) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.007306763969595647) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.007306763969595647) [X4 X5 X7 Z8 Z9 X10]
+ (-0.007306763969595647) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.005324817366223016) [Y2 Z3 Y4 X10 Z11 X12]
+ (-0.005324817366223016) [X2 Z3 X4 Y10 Z11 Y12]
+ (-0.005324817366223016) [Y3 Z4 Y5 X11 Z12 X13]
+ (-0.005324817366223016) [X3 Z4 X5 Y11 Z12 Y13]
+ (-0.004684920226869188) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.004684920226869188) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668615266015887) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.004668615266015887) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.004575015188890368) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.004575015188890368) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.004424843668499158) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.004424843668499158) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.00396156937303498) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (-0.00396156937303498) [X0 Z1 Z2 Z4 Z5 X6]
+ (-0.00396156937303498) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (-0.00396156937303498) [X1 Z3 Z4 Z5 Z6 X7]
+ (-0.002462916621397943) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (-0.002462916621397943) [X0 Z1 Z2 Z3 Z5 X6]
+ (-0.002462916621397943) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (-0.002462916621397943) [X1 Z2 Z3 Z4 Z6 X7]
+ (-0.002293955623063685) [Y1 Y2 X3 Z4 Z5 X6]
+ (-0.002293955623063685) [X1 X2 Y3 Z4 Z5 Y6]
+ (-0.0017991930083825537) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.0017991930083825537) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.0017278745823312985) [Y1 Z2 Z3 Y4 X11 X12]
+ (-0.0017278745823312985) [X1 Z2 Z3 X4 Y11 Y12]
+ (-0.0016676137499712952) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (-0.0016676137499712952) [X0 Z1 Z3 Z4 Z5 X6]
+ (-0.0016676137499712952) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (-0.0016676137499712952) [X1 Z2 Z4 Z5 Z6 X7]
+ (-0.0016095335162967612) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (-0.0016095335162967612) [X0 Z1 Z2 Z3 Z4 X6]
+ (-0.0016095335162967612) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (-0.0016095335162967612) [X1 Z2 Z3 Z5 Z6 X7]
+ (-0.0009298407044384138) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298407044384138) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298407044384138) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298407044384138) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533831051011819) [Y1 Z2 Z3 X4 X5 Y6]
+ (-0.0008533831051011819) [X1 Z2 Z3 Y4 Y5 X6]
+ (-0.0006650303449186705) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (-0.0006650303449186705) [X2 Z3 Z4 Z5 X6 Z12]
+ (-0.0006650303449186705) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (-0.0006650303449186705) [X3 Z4 Z5 Z6 X7 Z13]
+ (-0.0004957972886317744) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0004957972886317744) [Z2 X3 Z4 Z5 Z6 X7]
+ (-7.735870605795914e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.735870605795914e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.735870605795914e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.735870605795914e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-5.974176874278415e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (-5.974176874278415e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (-5.275783442645291e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.275783442645291e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.642978971311799e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.642978971311799e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.556473753115041e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (-4.556473753115041e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (-4.183808804366848e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (-4.183808804366848e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (-3.6945168971113176e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (-3.6945168971113176e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (-3.3342618779644284e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (-3.3342618779644284e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (-3.3130170800747816e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (-3.3130170800747816e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (-3.15129596513441e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.15129596513441e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.0882457084887606e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.0882457084887606e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.172638016389875e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (-2.172638016389875e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (-1.4548066961249558e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.4548066961249558e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.3304568536037916e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (-1.3304568536037916e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (-1.2282691293681331e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (-1.2282691293681331e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (-1.035792484215545e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (-1.035792484215545e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (-7.956666995379789e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.956666995379789e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.733096731428672e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.733096731428672e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.733096731428672e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.733096731428672e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.628427258131928e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (-6.628427258131928e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (-5.927350202088112e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.927350202088112e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.927350202088112e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.927350202088112e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.627722020314877e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (-5.627722020314877e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (-4.837953361198953e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (-4.837953361198953e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (-3.5706355102636894e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.5706355102636894e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.328039671803961e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (-3.328039671803961e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (-3.08677090669764e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.08677090669764e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.08677090669764e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.08677090669764e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.4472648254715786e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (-2.4472648254715786e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (-2.3712704702404027e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.3712704702404027e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.3712704702404027e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.3712704702404027e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.198963765514682e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (-2.198963765514682e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (-1.933212141894207e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.933212141894207e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.933212141894207e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.933212141894207e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.8393943653691678e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.8393943653691678e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.8393943653691678e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.8393943653691678e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.2919458272498863e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.2919458272498863e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.8395658333697767e-08) [Y0 Z1 X2 X10 Z11 Y12]
+ (-1.8395658333697767e-08) [X0 Z1 Y2 Y10 Z11 X12]
+ (-1.8395658333697767e-08) [Y1 Z2 X3 X11 Z12 Y13]
+ (-1.8395658333697767e-08) [X1 Z2 Y3 Y11 Z12 X13]
+ (-1.0351505326197143e-09) [Y0 Z1 Z2 Z3 Y4 Z7]
+ (-1.0351505326197143e-09) [X0 Z1 Z2 Z3 X4 Z7]
+ (-1.0351505326197143e-09) [Y1 Z2 Z3 Z4 Y5 Z6]
+ (-1.0351505326197143e-09) [X1 Z2 Z3 Z4 X5 Z6]
+ (2.270208376835613e-08) [Y0 Z1 Z2 X3 X5 Y6]
+ (2.270208376835613e-08) [Y0 Z1 Z2 Y3 Y5 Y6]
+ (2.270208376835613e-08) [X0 Z1 Z2 X3 X5 X6]
+ (2.270208376835613e-08) [X0 Z1 Z2 Y3 Y5 X6]
+ (2.270208376835613e-08) [Y1 X2 X4 Z5 Z6 Y7]
+ (2.270208376835613e-08) [Y1 Y2 Y4 Z5 Z6 Y7]
+ (2.270208376835613e-08) [X1 X2 X4 Z5 Z6 X7]
+ (2.270208376835613e-08) [X1 Y2 Y4 Z5 Z6 X7]
+ (2.5928188711269884e-08) [Y0 Z1 X2 X4 Z5 Y6]
+ (2.5928188711269884e-08) [X0 Z1 Y2 Y4 Z5 X6]
+ (2.5928188711269884e-08) [Y1 Z2 X3 X5 Z6 Y7]
+ (2.5928188711269884e-08) [X1 Z2 Y3 Y5 Z6 X7]
+ (8.057465293402204e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.057465293402204e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.057465293402204e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.057465293402204e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.649129641348985e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.649129641348985e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.649129641348985e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.649129641348985e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.1076529149865364e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (1.1076529149865364e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (1.1076529149865364e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (1.1076529149865364e-07) [X0 Z1 X2 X11 Z12 X13]
+ (1.1076529149865364e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (1.1076529149865364e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (1.1076529149865364e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (1.1076529149865364e-07) [X1 Z2 X3 X10 Z11 X12]
+ (1.348496898605661e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (1.348496898605661e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (1.348496898605661e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (1.348496898605661e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (1.3807579480366536e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (1.3807579480366536e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (1.3807579480366536e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (1.3807579480366536e-07) [X0 Z1 X2 X5 Z6 X7]
+ (1.3807579480366536e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (1.3807579480366536e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (1.3807579480366536e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (1.3807579480366536e-07) [X1 Z2 X3 X4 Z5 X6]
+ (1.607778785719334e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (1.607778785719334e-07) [X0 Z1 X2 X4 Z5 X6]
+ (1.607778785719334e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (1.607778785719334e-07) [X1 Z2 X3 X5 Z6 X7]
+ (1.82904286004087e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.82904286004087e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.82904286004087e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.82904286004087e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.198963765514682e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (2.198963765514682e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (2.4472648254715786e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (2.4472648254715786e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (3.236183434375301e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.236183434375301e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.236183434375301e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.236183434375301e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.328039671803961e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (3.328039671803961e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (3.5706355102636894e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.5706355102636894e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.837953361198953e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (4.837953361198953e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (5.28764945675337e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (5.28764945675337e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (5.28764945675337e-07) [X0 Z1 Z2 X3 X11 X12]
+ (5.28764945675337e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (5.28764945675337e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (5.28764945675337e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (5.28764945675337e-07) [X1 X2 X10 Z11 Z12 X13]
+ (5.28764945675337e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (5.627722020314877e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (5.627722020314877e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (6.395302371742752e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (6.395302371742752e-07) [X0 Z1 X2 X10 Z11 X12]
+ (6.395302371742752e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (6.395302371742752e-07) [X1 Z2 X3 X11 Z12 X13]
+ (6.579258955086777e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (6.579258955086777e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (6.579258955086777e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (6.579258955086777e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (6.628427258131928e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (6.628427258131928e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (7.956666995379789e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.956666995379789e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.306342955225791e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.306342955225791e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.306342955225791e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.306342955225791e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.035792484215545e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (1.035792484215545e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (1.2282691293681331e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (1.2282691293681331e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (1.239311386192133e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.239311386192133e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.239311386192133e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.239311386192133e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.3304568536037916e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (1.3304568536037916e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (1.4548066961249558e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.4548066961249558e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.172638016389875e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (2.172638016389875e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (3.0882457084887606e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.0882457084887606e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.1173664083424102e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.1173664083424102e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.15129596513441e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.15129596513441e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.2111874344953675e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (3.2111874344953675e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (3.2111874344953675e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (3.2111874344953675e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (3.2774382758837817e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (3.2774382758837817e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (3.2774382758837817e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (3.2774382758837817e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (3.3130170800747816e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (3.3130170800747816e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (3.6102422430641778e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (3.6102422430641778e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (3.6102422430641778e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (3.6102422430641778e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (3.6945168971113176e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (3.6945168971113176e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (3.7695836046025078e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (3.7695836046025078e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (4.253118727961559e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (4.253118727961559e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (4.556473753115041e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (4.556473753115041e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (4.642978971311799e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.642978971311799e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.275783442645291e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.275783442645291e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.974176874278415e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (5.974176874278415e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (6.290019670861019e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (6.290019670861019e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (6.290019670861019e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (6.290019670861019e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (6.524204514572751e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (6.524204514572751e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (6.524204514572751e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (6.524204514572751e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (7.444267395521746e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (7.444267395521746e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (7.444267395521746e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (7.444267395521746e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (7.518288800230236e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (7.518288800230236e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (7.518288800230236e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (7.518288800230236e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (8.774724249125537e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (8.774724249125537e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (8.774724249125537e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (8.774724249125537e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (0.0002922256724535163) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.0002922256724535163) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.0002922256724535163) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.0002922256724535163) [X6 Z7 X8 X9 Z10 X11]
+ (0.0008144692870281581) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (0.0008144692870281581) [X2 Z3 Z4 Z5 X6 Z10]
+ (0.0008144692870281581) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (0.0008144692870281581) [X3 Z4 Z5 Z6 X7 Z11]
+ (0.0008533831051011819) [Y1 Z2 Z3 Y4 X5 X6]
+ (0.0008533831051011819) [X1 Z2 Z3 X4 Y5 Y6]
+ (0.0017278745823312985) [Y1 Z2 Z3 X4 X11 Y12]
+ (0.0017278745823312985) [X1 Z2 Z3 Y4 Y11 X12]
+ (0.0017991930083825537) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.0017991930083825537) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.002293955623063685) [Y1 X2 X3 Z4 Z5 Y6]
+ (0.002293955623063685) [X1 Y2 Y3 Z4 Z5 X6]
+ (0.0027790407628819447) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (0.0027790407628819447) [X1 Z2 Z3 Z4 Z5 X7]
+ (0.0034938003714786252) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (0.0034938003714786252) [X2 Z3 Z4 Z5 X6 Z13]
+ (0.0034938003714786252) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (0.0034938003714786252) [X3 Z4 Z5 Z6 X7 Z12]
+ (0.004158830716397298) [Y3 Z4 Z5 X6 X12 Y13]
+ (0.004158830716397298) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (0.004158830716397298) [X3 Z4 Z5 X6 X12 X13]
+ (0.004158830716397298) [X3 Z4 Z5 Y6 Y12 X13]
+ (0.004424843668499158) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.004424843668499158) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.004575015188890368) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.004575015188890368) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.004668615266015887) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.004668615266015887) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.004684920226869188) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.004684920226869188) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.005143382387686592) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (0.005143382387686592) [Y2 Z3 Y4 X5 Z6 X7]
+ (0.005143382387686592) [X2 Z3 X4 Y5 Z6 Y7]
+ (0.005143382387686592) [X2 Z3 X4 X5 Z6 X7]
+ (0.005368616111411193) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.005368616111411193) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.005368616111411193) [X2 X3 X7 Z8 Z9 X10]
+ (0.005368616111411193) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.0056526073152192114) [Y0 X1 X3 Z4 Z5 Y6]
+ (0.0056526073152192114) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (0.0056526073152192114) [X0 X1 X3 Z4 Z5 X6]
+ (0.0056526073152192114) [X0 Y1 Y3 Z4 Z5 X6]
+ (0.0058051212123328925) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (0.0058051212123328925) [X2 Z3 Z4 Z5 X6 Z8]
+ (0.0058051212123328925) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (0.0058051212123328925) [X3 Z4 Z5 Z6 X7 Z9]
+ (0.007960839634224306) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.007960839634224306) [X4 Z5 X6 X10 Z11 X12]
+ (0.007960839634224306) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.007960839634224306) [X5 Z6 X7 X11 Z12 X13]
+ (0.008125248410122873) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.008125248410122873) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.00876485821939615) [Y2 Z3 Z4 X5 X11 Y12]
+ (0.00876485821939615) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (0.00876485821939615) [X2 Z3 Z4 X5 X11 X12]
+ (0.00876485821939615) [X2 Z3 Z4 Y5 Y11 X12]
+ (0.00876485821939615) [Y3 X4 X10 Z11 Z12 Y13]
+ (0.00876485821939615) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (0.00876485821939615) [X3 X4 X10 Z11 Z12 X13]
+ (0.00876485821939615) [X3 Y4 Y10 Z11 Z12 X13]
+ (0.008890680338662714) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.008890680338662714) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.008890680338662714) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.008890680338662714) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.010540434329196836) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.010540434329196836) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.010540434329196836) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.010540434329196836) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.010959994608917254) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.010959994608917254) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.010959994608917254) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.010959994608917254) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.011307208029984766) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.011307208029984766) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.01175599524038515) [Y3 Z4 Z5 X6 X8 Y9]
+ (0.01175599524038515) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (0.01175599524038515) [X3 Z4 Z5 X6 X8 X9]
+ (0.01175599524038515) [X3 Z4 Z5 Y6 Y8 X9]
+ (0.017561116452718042) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (0.017561116452718042) [X2 Z3 Z4 Z5 X6 Z9]
+ (0.017561116452718042) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (0.017561116452718042) [X3 Z4 Z5 Z6 X7 Z8]
+ (0.0182667585785129) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.0182667585785129) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.0182667585785129) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.0182667585785129) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.01902037387510943) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.01902037387510943) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.01902037387510943) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.01902037387510943) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.020175824956974793) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.020175824956974793) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.020175824956974793) [X4 Z5 Z6 X7 X11 X12]
+ (0.020175824956974793) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.020175824956974793) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.020175824956974793) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.020175824956974793) [X5 X6 X10 Z11 Z12 X13]
+ (0.020175824956974793) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.024388989986520622) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.024388989986520622) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.024388989986520622) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.024388989986520622) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.025104907970007483) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.025104907970007483) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.025104907970007483) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.025104907970007483) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.025996206267152683) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (0.025996206267152683) [X2 Z3 Z4 Z5 X6 Z7]
+ (0.027114878580404907) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (0.027114878580404907) [Z0 X2 Z3 Z4 Z5 X6]
+ (0.027114878580404907) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (0.027114878580404907) [Z1 X3 Z4 Z5 Z6 X7]
+ (0.030787440718540504) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.030787440718540504) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.03276748589562412) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (0.03276748589562412) [Z0 X3 Z4 Z5 Z6 X7]
+ (0.03276748589562412) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (0.03276748589562412) [Z1 X2 Z3 Z4 Z5 X6]
+ (0.05600713561684509) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.05600713561684509) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.05600713561684509) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.05600713561684509) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.05608449432290305) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.05608449432290305) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.05608449432290305) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.05608449432290305) [Z1 X7 Z8 Z9 Z10 X11]
+ (-0.04274326006290702) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-0.04274326006290702) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.04274326006290701) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (-0.04274326006290701) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (2.5950813526228324e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.5950813526228324e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.5950813526228327e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.5950813526228327e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (6.631261925268854e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (6.631261925268854e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (6.631261925268854e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (6.631261925268854e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.047642613600126456) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.047642613600126456) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.047642613600126456) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.047642613600126456) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.045879424030665986) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-0.045879424030665986) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.04171881404451736) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.04171881404451736) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.04171881404451736) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.04171881404451736) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.03956454804158996) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.03956454804158996) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.03956454804158996) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03956454804158996) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.039318107231006906) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.039318107231006906) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.039318107231006906) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.039318107231006906) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.025637212809922226) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.025637212809922226) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.025637212809922226) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.025637212809922226) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.023145221653213602) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.023145221653213602) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.022528354240839318) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.022528354240839318) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.019257453001858722) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019257453001858722) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.019028318718282793) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.019028318718282793) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.016024666095530417) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.016024666095530417) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.015225659057106268) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.015225659057106268) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-0.014603742410724168) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (-0.014603742410724168) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (-0.014564473640810642) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.014564473640810642) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.01175599524038515) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (-0.01175599524038515) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (-0.011285144618312079) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-0.011285144618312079) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-0.00984180292382079) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.00984180292382079) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.009612546714391804) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.009612546714391804) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.009612546714391804) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.009612546714391804) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.008469833341350463) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.008469833341350463) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.007306763969595647) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.007306763969595647) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005923799555609095) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.005923799555609095) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.0056526073152192114) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (-0.0056526073152192114) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (-0.005379929634052915) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (-0.005379929634052915) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (-0.005379929634052915) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (-0.005379929634052915) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (-0.005368616111411193) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005368616111411193) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.005241543597042598) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (-0.005241543597042598) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (-0.005241543597042598) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (-0.005241543597042598) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (-0.0046369735164968765) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (-0.0046369735164968765) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (-0.0046369735164968765) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (-0.0046369735164968765) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (-0.004311038607564498) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (-0.004311038607564498) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (-0.004311038607564498) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (-0.004311038607564498) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (-0.004158830716397298) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (-0.004158830716397298) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (-0.003989845257550309) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (-0.003989845257550309) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (-0.003989845257550309) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (-0.003989845257550309) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (-0.0022619706752190103) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (-0.0022619706752190103) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (-0.0022619706752190103) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (-0.0022619706752190103) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (-0.0022619706752190103) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (-0.0022619706752190103) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (-0.0022619706752190103) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (-0.0022619706752190103) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (-0.0013038029824613139) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (-0.0013038029824613139) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (-0.0013038029824613139) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (-0.0013038029824613139) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (-0.0012803055951275538) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (-0.0012803055951275538) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (-0.0012803055951275538) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (-0.0012803055951275538) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (-0.0010435237104111548) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (-0.0010435237104111548) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (-0.0010435237104111548) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (-0.0010435237104111548) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (-0.0008533831051011819) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (-0.0008533831051011819) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (-0.0008533831051011819) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (-0.0008533831051011819) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (-0.00024644081058304633) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00024644081058304633) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.735870605795914e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.735870605795914e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-4.183808804366848e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-4.183808804366848e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-3.5443574118377208e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.5443574118377208e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.5443574118377208e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.5443574118377208e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.5443574118377208e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.5443574118377208e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.5443574118377208e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.5443574118377208e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.3342618779644284e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-3.3342618779644284e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-3.3130170800747816e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-3.3130170800747816e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-3.3130170800747816e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-3.3130170800747816e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.5224581974430983e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.5224581974430983e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.5224581974430983e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.5224581974430983e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.5224581974430983e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224581974430983e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224581974430983e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.5224581974430983e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.3304568536037916e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (-1.3304568536037916e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (-1.3304568536037916e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (-1.3304568536037916e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (-7.988467887227882e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.988467887227882e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.988467887227882e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.988467887227882e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.189870446171476e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.189870446171476e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (-6.175164887089226e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175164887089226e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.471606040103832e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (-5.471606040103832e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (-4.561117033517791e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.561117033517791e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.561117033517791e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.561117033517791e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.5233393464651437e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (-4.5233393464651437e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (-3.427350853730691e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.427350853730691e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.427350853730691e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.427350853730691e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.08677090669764e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.08677090669764e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.3712704702404027e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.3712704702404027e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.8393943653691678e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.8393943653691678e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-1.7035434496992046e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035434496992046e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035434496992046e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (-1.7035434496992046e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (-9.2089458536592e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-9.2089458536592e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-9.2089458536592e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-9.2089458536592e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-8.057465293402204e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.057465293402204e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (-6.77295180595455e-08) [Y1 Z2 Z3 X4 X7 Z8 Z9 Y10]
+ (-6.77295180595455e-08) [X1 Z2 Z3 Y4 Y7 Z8 Z9 X10]
+ (-3.2261049427570525e-09) [Y0 Z1 Z2 Y3 X4 Z5 Z6 X7]
+ (-3.2261049427570525e-09) [X0 Z1 Z2 X3 Y4 Z5 Z6 Y7]
+ (3.2261049427570525e-09) [Y0 Z1 Z2 X3 X4 Z5 Z6 Y7]
+ (3.2261049427570525e-09) [X0 Z1 Z2 Y3 Y4 Z5 Z6 X7]
+ (6.04679054484053e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (6.04679054484053e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (6.04679054484053e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (6.04679054484053e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (6.77295180595455e-08) [Y1 Z2 Z3 Y4 X7 Z8 Z9 X10]
+ (6.77295180595455e-08) [X1 Z2 Z3 X4 Y7 Z8 Z9 Y10]
+ (8.057465293402204e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.057465293402204e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (1.8393943653691678e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.8393943653691678e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.3712704702404027e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.3712704702404027e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (2.8885649071340273e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.8885649071340273e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.8885649071340273e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (2.8885649071340273e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (3.08677090669764e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.08677090669764e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (3.328039671803961e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (3.328039671803961e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (3.328039671803961e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (3.328039671803961e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (4.5233393464651437e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (4.5233393464651437e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (5.471606040103832e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (5.471606040103832e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (6.175164887089226e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175164887089226e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.189870446171476e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (7.189870446171476e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (7.867608608332041e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (7.867608608332041e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (7.867608608332041e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (7.867608608332041e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (1.2282691293681331e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (1.2282691293681331e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (1.2282691293681331e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (1.2282691293681331e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (1.6288377676927212e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.6288377676927212e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.6288377676927212e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.6288377676927212e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.6540900598582473e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.6540900598582473e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.6540900598582473e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.6540900598582473e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.689305673142536e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.689305673142536e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.689305673142536e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.689305673142536e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (1.9429465505723005e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.9429465505723005e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.9429465505723005e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.9429465505723005e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.011074021183285e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.011074021183285e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.011074021183285e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.011074021183285e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.103163479720853e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.103163479720853e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.103163479720853e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.103163479720853e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.36094720566446e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (2.36094720566446e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (2.36094720566446e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (2.36094720566446e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (2.745510623112981e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.745510623112981e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.745510623112981e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.745510623112981e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.745510623112981e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.745510623112981e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.745510623112981e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.745510623112981e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.2117638705843332e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.2117638705843332e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.2117638705843332e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.2117638705843332e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.2117638705843332e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.2117638705843332e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.2117638705843332e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.2117638705843332e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.7695836046025078e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.7695836046025078e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.253118727961559e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.253118727961559e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.728781416899624e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (4.728781416899624e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (4.728781416899624e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.728781416899624e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.7345783644250905e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (4.7345783644250905e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (4.7345783644250905e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (4.7345783644250905e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (5.07140365548376e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (5.07140365548376e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (5.07140365548376e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (5.07140365548376e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (6.481751998469115e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (6.481751998469115e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.481751998469115e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (6.481751998469115e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (6.652106343439038e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (6.652106343439038e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (6.652106343439038e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (6.652106343439038e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.0897286225654935e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.0897286225654935e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (7.0897286225654935e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (7.0897286225654935e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (9.805982019909718e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (9.805982019909718e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (9.805982019909718e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (9.805982019909718e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (1.53166140910774e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (1.53166140910774e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (1.53166140910774e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (1.53166140910774e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.6103374951910694e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (1.6103374951910694e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.6103374951910694e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (1.6103374951910694e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (7.735870605795914e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.735870605795914e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.00013838603701031818) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (0.00013838603701031818) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (0.00013838603701031818) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (0.00013838603701031818) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (0.00024644081058304633) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00024644081058304633) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0004458488204511857) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.0004458488204511857) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.0004458488204511857) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.0004458488204511857) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.0005940157673920291) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940157673920291) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.0005940157673920291) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940157673920291) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.0005940157673920291) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940157673920291) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.0005940157673920291) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940157673920291) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.0009581676927576982) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (0.0009581676927576982) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (0.0009581676927576982) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (0.0009581676927576982) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (0.0009581676927576982) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (0.0009581676927576982) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (0.0009581676927576982) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (0.0009581676927576982) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (0.002293955623063685) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (0.002293955623063685) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (0.002293955623063685) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (0.002293955623063685) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (0.002686042275088996) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (0.002686042275088996) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (0.002686042275088996) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (0.002686042275088996) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (0.0027790407628819447) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (0.0027790407628819447) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (0.003267514897153343) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (0.003267514897153343) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (0.003267514897153343) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (0.003267514897153343) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (0.0033566679213693222) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (0.0033566679213693222) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (0.0033566679213693222) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (0.0033566679213693222) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (0.004158830716397298) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (0.004158830716397298) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (0.005114464086467073) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.005114464086467073) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.005114464086467073) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.005114464086467073) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.005114464086467073) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114464086467073) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114464086467073) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005114464086467073) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.005262631033407916) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.005262631033407916) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.005262631033407916) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.005262631033407916) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.005368616111411193) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005368616111411193) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.0056526073152192114) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (0.0056526073152192114) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (0.005708479853859101) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.005708479853859101) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.005708479853859101) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.005708479853859101) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.005923799555609095) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.005923799555609095) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.007306763969595647) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.007306763969595647) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.008469833341350463) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.008469833341350463) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.00984180292382079) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.00984180292382079) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.011285144618312079) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (0.011285144618312079) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (0.01175599524038515) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (0.01175599524038515) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (0.014564473640810642) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.014564473640810642) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.014603742410724168) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (0.014603742410724168) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (0.015225659057106268) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.015225659057106268) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.016024666095530417) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.016024666095530417) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.018888995077376562) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (0.018888995077376562) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (0.018888995077376562) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.018888995077376562) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.019028318718282793) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.019028318718282793) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.019257453001858722) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (0.019257453001858722) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (0.021433980116862292) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.021433980116862292) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.021433980116862292) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.021433980116862292) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.02428203162329563) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (0.02428203162329563) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (0.024755507980739576) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.024755507980739576) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.024755507980739576) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.024755507980739576) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.028730798001197357) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.028730798001197357) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.028730798001197357) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.028730798001197357) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.02990381345821276) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.02990381345821276) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.02990381345821276) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.02990381345821276) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.035608400352215235) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.035608400352215235) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.03935925039146375) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.03935925039146375) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.03935925039146375) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.03935925039146375) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.36937137554435007) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.36937137554435007) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.36937137554435007) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.36937137554435007) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.28164335753379216) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.28164335753379216) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.2816433575337922) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.2816433575337922) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.09065142344949657) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.09065142344949657) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.09065142344949657) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.09065142344949657) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684736029512764) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.08684736029512764) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684736029512764) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.08684736029512764) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.05859215179544686) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.05859215179544686) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.034903304270646224) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.034903304270646224) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.034903304270646224) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.034903304270646224) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591832088250935) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.024591832088250935) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591832088250935) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.024591832088250935) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.023145221653213602) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.023145221653213602) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.022528354240839318) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.022528354240839318) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019299499858008193) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.019299499858008193) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019299499858008193) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.019299499858008193) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019299499858008193) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019299499858008193) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.019299499858008193) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019299499858008193) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.016024666095530417) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.016024666095530417) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.016024666095530417) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.016024666095530417) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.014603742410724166) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (-0.014603742410724166) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (-0.014603742410724166) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (-0.014603742410724166) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (-0.010757524201207255) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010757524201207255) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010757524201207255) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010757524201207255) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.01071547734505778) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.01071547734505778) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.01071547734505778) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.01071547734505778) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311472182395285) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311472182395285) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311472182395285) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010311472182395285) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005408970758385956) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.005408970758385956) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.005408970758385956) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.005408970758385956) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.005286569056778024) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.005286569056778024) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.005286569056778024) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.005286569056778024) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.004767276644726839) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.004767276644726839) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.004767276644726839) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.004767276644726839) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.004668615266015887) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.004668615266015887) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.003876482195724198) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (-0.003876482195724198) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (-0.003804063154368915) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804063154368915) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804063154368915) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003804063154368915) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003484154579343894) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003484154579343894) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0033566679213693222) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (-0.0033566679213693222) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (-0.003267514897153343) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (-0.003267514897153343) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (-0.0024464634227012996) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.0024464634227012996) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.0024464634227012996) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.0024464634227012996) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.0017278745823312985) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.0017278745823312985) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.0016407591167034768) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0016407591167034768) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0015324885626617582) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.0015324885626617582) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.0015324885626617582) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.0015324885626617582) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.0007870893705577512) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0007870893705577512) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007156737069662402) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0007156737069662402) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007156737069662402) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0007156737069662402) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0005192924120511841) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.0005192924120511841) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.00019401030606666604) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00019401030606666604) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.0001878748613795221) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0001878748613795221) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0001878748613795221) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0001878748613795221) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.00013838603701031818) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.00013838603701031818) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-4.204685614947641e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-4.204685614947641e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-4.204685614947641e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.204685614947641e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-5.07140365548376e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-5.07140365548376e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-3.15129596513441e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.15129596513441e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0882457084887606e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.0882457084887606e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.9884125712050565e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.9884125712050565e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.874248559182961e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.874248559182961e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.36094720566446e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-2.36094720566446e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.3001958860657005e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.3001958860657005e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-7.867608608332041e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867608608332041e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.99695178856893e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.99695178856893e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.99695178856893e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.99695178856893e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.99695178856893e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.99695178856893e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.99695178856893e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.99695178856893e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.8885649071340273e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.8885649071340273e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.68632152015621e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.68632152015621e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035434496992046e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.7035434496992046e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.2089458536592e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.2089458536592e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.204004136943316e-08) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (-3.204004136943316e-08) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (-3.204004136943316e-08) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (-3.204004136943316e-08) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (3.568947669020043e-08) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (3.568947669020043e-08) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (3.568947669020043e-08) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (3.568947669020043e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (3.568947669020043e-08) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.568947669020043e-08) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (3.568947669020043e-08) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (3.568947669020043e-08) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.706834734899164e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.706834734899164e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.706834734899164e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.706834734899164e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.379737403741593e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (8.379737403741593e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.379737403741593e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (8.379737403741593e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.379737403741593e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (8.379737403741593e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (8.379737403741593e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (8.379737403741593e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (9.2089458536592e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.2089458536592e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.071684516665942e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.071684516665942e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.071684516665942e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.071684516665942e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.178213098516773e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.178213098516773e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.178213098516773e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.178213098516773e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.7035434496992046e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.7035434496992046e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.249897615193557e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.249897615193557e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.249897615193557e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.249897615193557e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.68632152015621e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.68632152015621e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.8885649071340273e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.8885649071340273e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.376686367630059e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (3.376686367630059e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (3.376686367630059e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (3.376686367630059e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (3.376686367630059e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (3.376686367630059e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (3.376686367630059e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (3.376686367630059e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (3.5682004761463347e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (3.5682004761463347e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (3.5682004761463347e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (3.5682004761463347e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.092161928157407e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.092161928157407e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.092161928157407e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.092161928157407e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.092161928157407e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.092161928157407e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.092161928157407e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.092161928157407e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.4490566950625284e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.4490566950625284e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.4490566950625284e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.4490566950625284e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.769457108752184e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.769457108752184e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.769457108752184e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.769457108752184e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.246849403757337e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.246849403757337e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.246849403757337e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.246849403757337e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.246849403757337e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.246849403757337e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.246849403757337e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.246849403757337e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.560553919662426e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.560553919662426e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (7.560553919662426e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (7.560553919662426e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (7.560553919662426e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.560553919662426e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.560553919662426e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.560553919662426e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.867608608332041e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (7.867608608332041e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.900025714095745e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (7.900025714095745e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (7.900025714095745e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (7.900025714095745e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (8.027844186545635e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (8.027844186545635e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (8.027844186545635e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.027844186545635e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (8.091539822613613e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (8.091539822613613e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (8.091539822613613e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (8.091539822613613e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (8.091539822613613e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (8.091539822613613e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (8.091539822613613e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (8.091539822613613e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (8.398527660036534e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (8.398527660036534e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (8.398527660036534e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (8.398527660036534e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.1468226190245027e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (1.1468226190245027e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (1.1468226190245027e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (1.1468226190245027e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.3001958860657005e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.3001958860657005e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.36094720566446e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.36094720566446e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.874248559182961e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.874248559182961e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.883653167245516e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.883653167245516e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.9473314461774675e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9473314461774675e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.9473314461774675e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9473314461774675e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9884125712050565e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9884125712050565e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.0882457084887606e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.0882457084887606e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.15129596513441e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.15129596513441e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.846190874519373e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.846190874519373e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.846190874519373e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.846190874519373e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.07140365548376e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (5.07140365548376e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (5.105462415326312e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.105462415326312e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.105462415326312e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.105462415326312e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.146386760591579e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.146386760591579e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.146386760591579e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.146386760591579e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.159294474959414e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.159294474959414e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.159294474959414e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.159294474959414e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.427926626972541e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.427926626972541e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.427926626972541e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.427926626972541e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.935744017387078e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.935744017387078e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.935744017387078e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.935744017387078e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253185150589908e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.253185150589908e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.979710974509273e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.979710974509273e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.979710974509273e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.979710974509273e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (7.141566359151094e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (7.141566359151094e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (7.141566359151094e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.141566359151094e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00013838603701031818) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.00013838603701031818) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.00019401030606666604) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (0.00019401030606666604) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (0.00024644081058304633) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024644081058304633) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00024644081058304633) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024644081058304633) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192924120511841) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.0005192924120511841) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.0007870893705577512) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.0007870893705577512) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.0014528842553239558) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0014528842553239558) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0014528842553239558) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0014528842553239558) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0016407591167034768) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0016407591167034768) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0017278745823312985) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.0017278745823312985) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.002141348964737671) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (0.002141348964737671) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (0.003267514897153343) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (0.003267514897153343) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (0.0033566679213693222) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (0.0033566679213693222) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (0.003484154579343894) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.003484154579343894) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.003876482195724198) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (0.003876482195724198) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (0.004668615266015887) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.004668615266015887) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005923799555609095) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923799555609095) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.005923799555609095) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923799555609095) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.008469833341350463) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.008469833341350463) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.008469833341350463) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.008469833341350463) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.008541975656800942) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.008541975656800942) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (0.008541975656800942) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (0.008541975656800942) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.008541975656800942) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.008541975656800942) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.008541975656800942) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.008541975656800942) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.008826387567788835) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.008826387567788835) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.008826387567788835) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.008826387567788835) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.00984180292382079) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.00984180292382079) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00984180292382079) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.00984180292382079) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.01709162192218861) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.01709162192218861) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.01709162192218861) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.01709162192218861) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.019538085344889912) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.019538085344889912) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.019538085344889912) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.019538085344889912) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.02428203162329563) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (0.02428203162329563) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (0.035608400352215235) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.035608400352215235) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0675239817995759) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0675239817995759) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0675239817995759) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0675239817995759) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.07635036936736472) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.07635036936736472) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.07635036936736472) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.07635036936736472) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.07165056250040594) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07165056250040594) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07165056250040593) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07165056250040593) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.775872094091676e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.775872094091676e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.775872094091676e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.775872094091676e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.05859215179544686) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.05859215179544686) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.019257453001858722) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.019257453001858722) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311472182395285) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311472182395285) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008826387567788835) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.008826387567788835) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0075974617790828475) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0075974617790828475) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0075974617790828475) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0075974617790828475) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0057335686400167485) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0057335686400167485) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0057335686400167485) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0057335686400167485) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0057335686400167485) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0057335686400167485) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0057335686400167485) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0057335686400167485) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348047718409991) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005348047718409991) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005348047718409991) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348047718409991) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.004220835998338751) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.004220835998338751) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.004220835998338751) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.004220835998338751) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.003876482195724198) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (-0.003876482195724198) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (-0.003876482195724198) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (-0.003876482195724198) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (-0.003804063154368915) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003804063154368915) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002446463422701299) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.002446463422701299) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.00239496715406149) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00239496715406149) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00239496715406149) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00239496715406149) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00239496715406149) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00239496715406149) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.00239496715406149) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00239496715406149) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0022494140606728564) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0022494140606728564) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0022494140606728564) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0022494140606728564) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0022009568479948245) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0022009568479948245) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0022009568479948245) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0022009568479948245) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.001863893139066098) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.001863893139066098) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.001863893139066098) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.001863893139066098) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.001863893139066098) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.001863893139066098) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.001863893139066098) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.001863893139066098) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012366559235511766) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.0012366559235511766) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.0012366559235511766) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.0012366559235511766) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.0011726297842391746) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0011726297842391746) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0011726297842391746) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0011726297842391746) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0007870893705577512) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0007870893705577512) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870893705577512) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0007870893705577512) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0005192924120511841) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.0005192924120511841) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.0005192924120511841) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.0005192924120511841) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-1.1462850975895056e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1462850975895056e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.874248559182961e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.874248559182961e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.874248559182961e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.874248559182961e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.3001958860657005e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.3001958860657005e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.3001958860657005e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.3001958860657005e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.044474161737883e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.044474161737883e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.044474161737883e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.044474161737883e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.955903565735086e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.955903565735086e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.955903565735086e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.955903565735086e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105341096136543e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105341096136543e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105341096136543e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105341096136543e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.66120032078514e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.66120032078514e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.66120032078514e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.66120032078514e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540204245901629e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540204245901629e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.189870446171476e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.189870446171476e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.876530378729657e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876530378729657e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876530378729657e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876530378729657e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175164887089226e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175164887089226e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.5233393464651437e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.5233393464651437e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.076662710988485e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.076662710988485e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.076662710988485e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.076662710988485e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.0133988404493195e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0133988404493195e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.9045373714772023e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.9045373714772023e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.9045373714772023e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.9045373714772023e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.666679778099741e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.666679778099741e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.666679778099741e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.666679778099741e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505624695966126e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.8505624695966126e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.846699420613107e-08) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-7.846699420613107e-08) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-6.77295180595455e-08) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (-6.77295180595455e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (-4.099829328900993e-08) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-4.099829328900993e-08) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.099829328900993e-08) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.099829328900993e-08) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (6.77295180595455e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (6.77295180595455e-08) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (7.846699420613107e-08) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (7.846699420613107e-08) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (1.6570092900642585e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.6570092900642585e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.6570092900642585e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.6570092900642585e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.8505624695966126e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.8505624695966126e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.68632152015621e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.68632152015621e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.68632152015621e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.68632152015621e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0133988404493195e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.0133988404493195e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.5233393464651437e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.5233393464651437e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.6704081305167797e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.6704081305167797e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.6704081305167797e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.6704081305167797e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175164887089226e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175164887089226e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.189870446171476e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.189870446171476e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.540204245901629e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540204245901629e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949307701207896e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949307701207896e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.7924638283745843e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924638283745843e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924638283745843e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.7924638283745843e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.883653167245516e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.883653167245516e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.9884125712050565e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9884125712050565e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9884125712050565e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9884125712050565e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253185150589908e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.253185150589908e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4016916420662938e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4016916420662938e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4016916420662938e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4016916420662938e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809380249037848e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.5809380249037848e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809380249037848e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.5809380249037848e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.00102832706375565) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.00102832706375565) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.00102832706375565) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00102832706375565) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.001222337369822316) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.001222337369822316) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (0.001222337369822316) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (0.001222337369822316) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.001222337369822316) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.001222337369822316) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.001222337369822316) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.001222337369822316) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.001640759116703477) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.001640759116703477) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.001640759116703477) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.001640759116703477) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.002141348964737671) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (0.002141348964737671) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (0.002446463422701299) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.002446463422701299) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.0029841800747875753) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.0029841800747875753) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.0029841800747875753) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.0029841800747875753) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.003804063154368915) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.003804063154368915) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.008826387567788835) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.008826387567788835) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.010311472182395285) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010311472182395285) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019257453001858722) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019257453001858722) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.3986653555126069e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.3986653555126069e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.3986653555126069e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.3986653555126069e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003484154579343894) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003484154579343894) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002984180074787575) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.002984180074787575) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.00019401030606666604) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00019401030606666604) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1462850975895056e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1462850975895056e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.7924638283745843e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7924638283745843e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540204245901629e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540204245901629e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540204245901629e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540204245901629e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.8505624695966126e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505624695966126e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505624695966126e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505624695966126e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.846699420613107e-08) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-7.846699420613107e-08) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-7.846699420613107e-08) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-7.846699420613107e-08) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-4.099829328900993e-08) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-4.099829328900993e-08) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.099829328900993e-08) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-4.099829328900993e-08) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0133988404493195e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.0133988404493195e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0133988404493195e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.0133988404493195e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.949307701207896e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.949307701207896e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.7924638283745843e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7924638283745843e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019401030606666604) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00019401030606666604) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002984180074787575) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.002984180074787575) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.003484154579343894) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.003484154579343894) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
  (-73.13873149596549) [I0]
+ (-0.18066757506990258) [Z6]
+ (-0.1806675750699025) [Z7]
+ (-0.15961443583713672) [Z5]
+ (-0.1596144358371366) [Z4]
+ (0.1741998661213093) [Z2]
+ (0.17419986612130942) [Z3]
+ (0.2275732881450001) [Z0]
+ (0.22757328814500033) [Z1]
+ (-8.194104893031323e-06) [Y4 Y6]
+ (-8.194104893031323e-06) [X4 X6]
+ (7.95422432648546e-06) [Y5 Y7]
+ (7.95422432648546e-06) [X5 X7]
+ (0.11270381859114276) [Z4 Z6]
+ (0.11270381859114276) [Z5 Z7]
+ (0.11952441016894674) [Z0 Z4]
+ (0.11952441016894674) [Z1 Z5]
+ (0.13401737372650185) [Z0 Z6]
+ (0.13401737372650185) [Z1 Z7]
+ (0.13734942210157353) [Z0 Z5]
+ (0.13734942210157353) [Z1 Z4]
+ (0.13766859133612516) [Z2 Z4]
+ (0.13766859133612516) [Z3 Z5]
+ (0.1413890359014125) [Z4 Z7]
+ (0.1413890359014125) [Z5 Z6]
+ (0.1472293078325672) [Z2 Z5]
+ (0.1472293078325672) [Z3 Z4]
+ (0.14926347060039544) [Z4 Z5]
+ (0.14973497005421602) [Z2 Z6]
+ (0.14973497005421602) [Z3 Z7]
+ (0.15138342699111035) [Z0 Z7]
+ (0.15138342699111035) [Z1 Z6]
+ (0.15435760065288903) [Z6 Z7]
+ (0.15582280685844346) [Z2 Z7]
+ (0.15582280685844346) [Z3 Z6]
+ (0.16756669356170792) [Z0 Z2]
+ (0.16756669356170792) [Z1 Z3]
+ (0.18144009362934027) [Z0 Z3]
+ (0.18144009362934027) [Z1 Z2]
+ (0.1939257433499018) [Z0 Z1]
+ (0.22003977240262504) [Z2 Z3]
+ (-7.0380236753115605e-06) [Y4 Z5 Y6]
+ (-7.0380236753115605e-06) [X4 Z5 X6]
+ (-7.0380236753115605e-06) [Y5 Z6 Y7]
+ (-7.0380236753115605e-06) [X5 Z6 X7]
+ (-0.03078744071865401) [Y0 Z2 Z3 Y4]
+ (-0.03078744071865401) [X0 Z2 Z3 X4]
+ (-0.030104525273606013) [Y0 Z1 Z3 Y4]
+ (-0.030104525273606013) [X0 Z1 Z3 X4]
+ (-0.030104525273606013) [Y1 Z2 Z4 Y5]
+ (-0.030104525273606013) [X1 Z2 Z4 X5]
+ (-0.029812299601154356) [Y0 Z1 Z2 Y4]
+ (-0.029812299601154356) [X0 Z1 Z2 X4]
+ (-0.029812299601154356) [Y1 Z3 Z4 Y5]
+ (-0.029812299601154356) [X1 Z3 Z4 X5]
+ (-0.02868521731026974) [Y4 Y5 X6 X7]
+ (-0.02868521731026974) [X4 X5 Y6 Y7]
+ (-0.01782501193262679) [Y0 Y1 X4 X5]
+ (-0.01782501193262679) [X0 X1 Y4 Y5]
+ (-0.0173660532646085) [Y0 Y1 X6 X7]
+ (-0.0173660532646085) [X0 X1 Y6 Y7]
+ (-0.013873400067632356) [Y0 Y1 X2 X3]
+ (-0.013873400067632356) [X0 X1 Y2 Y3]
+ (-0.011307208030072777) [Y1 Z2 Z3 Y5]
+ (-0.011307208030072777) [X1 Z2 Z3 X5]
+ (-0.009560716496442013) [Y2 Y3 X4 X5]
+ (-0.009560716496442013) [X2 X3 Y4 Y5]
+ (-0.006087836804227481) [Y2 Y3 X6 X7]
+ (-0.006087836804227481) [X2 X3 Y6 Y7]
+ (-0.00029222567245165904) [Y1 X2 X3 Y4]
+ (-0.00029222567245165904) [X1 Y2 Y3 X4]
+ (-8.194104893031323e-06) [Z4 Y5 Z6 Y7]
+ (-8.194104893031323e-06) [Z4 X5 Z6 X7]
+ (-2.8909299660180056e-06) [Z0 Y5 Z6 Y7]
+ (-2.8909299660180056e-06) [Z0 X5 Z6 X7]
+ (-2.8909299660180056e-06) [Z1 Y4 Z5 Y6]
+ (-2.8909299660180056e-06) [Z1 X4 Z5 X6]
+ (-1.8551375052834607e-06) [Z0 Y4 Z5 Y6]
+ (-1.8551375052834607e-06) [Z0 X4 Z5 X6]
+ (-1.8551375052834607e-06) [Z1 Y5 Z6 Y7]
+ (-1.8551375052834607e-06) [Z1 X5 Z6 X7]
+ (-1.5973397705928375e-06) [Z2 Y4 Z5 Y6]
+ (-1.5973397705928375e-06) [Z2 X4 Z5 X6]
+ (-1.5973397705928375e-06) [Z3 Y5 Z6 Y7]
+ (-1.5973397705928375e-06) [Z3 X5 Z6 X7]
+ (-1.0357924607354123e-06) [Y0 X1 X5 Y6]
+ (-1.0357924607354123e-06) [Y0 Y1 Y5 Y6]
+ (-1.0357924607354123e-06) [X0 X1 X5 X6]
+ (-1.0357924607354123e-06) [X0 Y1 Y5 X6]
+ (-9.344970640264016e-07) [Z2 Y5 Z6 Y7]
+ (-9.344970640264016e-07) [Z2 X5 Z6 X7]
+ (-9.344970640264016e-07) [Z3 Y4 Z5 Y6]
+ (-9.344970640264016e-07) [Z3 X4 Z5 X6]
+ (6.628427065664359e-07) [Y2 X3 X5 Y6]
+ (6.628427065664359e-07) [Y2 Y3 Y5 Y6]
+ (6.628427065664359e-07) [X2 X3 X5 X6]
+ (6.628427065664359e-07) [X2 Y3 Y5 X6]
+ (7.95422432648546e-06) [Y4 Z5 Y6 Z7]
+ (7.95422432648546e-06) [X4 Z5 X6 Z7]
+ (0.00029222567245165904) [Y1 Y2 X3 X4]
+ (0.00029222567245165904) [X1 X2 Y3 Y4]
+ (0.006087836804227481) [Y2 X3 X6 Y7]
+ (0.006087836804227481) [X2 Y3 Y6 X7]
+ (0.009560716496442013) [Y2 X3 X4 Y5]
+ (0.009560716496442013) [X2 Y3 Y4 X5]
+ (0.013873400067632356) [Y0 X1 X2 Y3]
+ (0.013873400067632356) [X0 Y1 Y2 X3]
+ (0.0173660532646085) [Y0 X1 X6 Y7]
+ (0.0173660532646085) [X0 Y1 Y6 X7]
+ (0.01782501193262679) [Y0 X1 X4 Y5]
+ (0.01782501193262679) [X0 Y1 Y4 X5]
+ (0.02868521731026974) [Y4 X5 X6 Y7]
+ (0.02868521731026974) [X4 Y5 Y6 X7]
+ (-0.04375171612138067) [Y1 Z2 Z3 Z4 Y5]
+ (-0.04375171612138067) [X1 Z2 Z3 Z4 X5]
+ (-0.04375171612138066) [Y0 Z1 Z2 Z3 Y4]
+ (-0.04375171612138066) [X0 Z1 Z2 Z3 X4]
+ (-0.03078744071865401) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (-0.03078744071865401) [Z0 X1 Z2 Z3 Z4 X5]
+ (-0.02510490797006635) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (-0.02510490797006635) [X0 Z1 Z2 Z3 X4 Z6]
+ (-0.02510490797006635) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (-0.02510490797006635) [X1 Z2 Z3 Z4 X5 Z7]
+ (-0.011307208030072777) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-0.011307208030072777) [X0 Z1 Z2 Z3 X4 Z5]
+ (-0.010540434329272414) [Y0 Z1 Z2 Z3 Y4 Z7]
+ (-0.010540434329272414) [X0 Z1 Z2 Z3 X4 Z7]
+ (-0.010540434329272414) [Y1 Z2 Z3 Z4 Y5 Z6]
+ (-0.010540434329272414) [X1 Z2 Z3 Z4 X5 Z6]
+ (-0.00029222567245165904) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-0.00029222567245165904) [Y0 Z1 Y2 X3 Z4 X5]
+ (-0.00029222567245165904) [X0 Z1 X2 Y3 Z4 Y5]
+ (-0.00029222567245165904) [X0 Z1 X2 X3 Z4 X5]
+ (-4.183808685248591e-06) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-4.183808685248591e-06) [X1 Z2 Z3 Z4 Z5 X7]
+ (-3.31301699988025e-06) [Y1 Z2 Z3 X4 X5 Y6]
+ (-3.31301699988025e-06) [X1 Z2 Z3 Y4 Y5 X6]
+ (-1.0357924607354123e-06) [Y0 Y1 X4 Z5 Z6 X7]
+ (-1.0357924607354123e-06) [X0 X1 Y4 Z5 Z6 Y7]
+ (-6.628427065664359e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (-6.628427065664359e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (-3.328039584246504e-07) [Y1 Y2 X3 Z4 Z5 X6]
+ (-3.328039584246504e-07) [X1 X2 Y3 Z4 Z5 Y6]
+ (3.328039584246504e-07) [Y1 X2 X3 Z4 Z5 Y6]
+ (3.328039584246504e-07) [X1 Y2 Y3 Z4 Z5 X6]
+ (6.628427065664359e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (6.628427065664359e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (1.0357924607354123e-06) [Y0 X1 X4 Z5 Z6 Y7]
+ (1.0357924607354123e-06) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.21118737339407e-06) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (3.21118737339407e-06) [X0 Z1 Z2 Z3 Z5 X6]
+ (3.21118737339407e-06) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (3.21118737339407e-06) [X1 Z2 Z3 Z4 Z6 X7]
+ (3.2774382044487363e-06) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (3.2774382044487363e-06) [X0 Z1 Z2 Z4 Z5 X6]
+ (3.2774382044487363e-06) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (3.2774382044487363e-06) [X1 Z3 Z4 Z5 Z6 X7]
+ (3.31301699988025e-06) [Y1 Z2 Z3 Y4 X5 X6]
+ (3.31301699988025e-06) [X1 Z2 Z3 X4 Y5 Y6]
+ (3.6102421628733867e-06) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (3.6102421628733867e-06) [X0 Z1 Z3 Z4 Z5 X6]
+ (3.6102421628733867e-06) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (3.6102421628733867e-06) [X1 Z2 Z4 Z5 Z6 X7]
+ (3.769583519976625e-06) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (3.769583519976625e-06) [X0 Z2 Z3 Z4 Z5 X6]
+ (6.524204373272585e-06) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (6.524204373272585e-06) [X0 Z1 Z2 Z3 Z4 X6]
+ (6.524204373272585e-06) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (6.524204373272585e-06) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.014564473640793928) [Y1 Z2 Z3 X4 X6 Y7]
+ (0.014564473640793928) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (0.014564473640793928) [X1 Z2 Z3 X4 X6 X7]
+ (0.014564473640793928) [X1 Z2 Z3 Y4 Y6 X7]
+ (5.1056809195201596e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (5.1056809195201596e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (5.1056809195201596e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (5.1056809195201596e-06) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.014564473640793928) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (-0.014564473640793928) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (-4.183808685248591e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-4.183808685248591e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-3.31301699988025e-06) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (-3.31301699988025e-06) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (-3.31301699988025e-06) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (-3.31301699988025e-06) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (3.328039584246504e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (3.328039584246504e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (3.328039584246504e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (3.328039584246504e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (3.769583519976625e-06) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (3.769583519976625e-06) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (0.014564473640793928) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (0.014564473640793928) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
  (-46.46390678868893) [I0]
+ (0.7829661725950184) [Z10]
+ (0.7829661725950188) [Z11]
+ (0.8084581961720543) [Z13]
+ (0.8084581961720545) [Z12]
+ (1.2034402289145647) [Z4]
+ (1.2034402289145651) [Z5]
+ (1.3096862988615532) [Z7]
+ (1.3096862988615536) [Z6]
+ (1.3693525634718229) [Z8]
+ (1.3693525634718229) [Z9]
+ (1.653894222683164) [Z2]
+ (1.653894222683164) [Z3]
+ (12.412630742111773) [Z0]
+ (12.412630742111773) [Z1]
+ (-8.194261372724242e-06) [Y10 Y12]
+ (-8.194261372724242e-06) [X10 X12]
+ (-1.854060858047903e-06) [Y5 Y7]
+ (-1.854060858047903e-06) [X5 X7]
+ (-7.764994121636087e-07) [Y3 Y5]
+ (-7.764994121636087e-07) [X3 X5]
+ (-5.929765814773579e-07) [Y4 Y6]
+ (-5.929765814773579e-07) [X4 X6]
+ (1.6021167409472747e-06) [Y2 Y4]
+ (1.6021167409472747e-06) [X2 X4]
+ (7.954413176439364e-06) [Y11 Y13]
+ (7.954413176439364e-06) [X11 X13]
+ (0.003276971931231564) [Y1 Y3]
+ (0.003276971931231564) [X1 X3]
+ (0.10433064780651312) [Y0 Y2]
+ (0.10433064780651312) [X0 X2]
+ (0.11270386920332208) [Z10 Z12]
+ (0.11270386920332208) [Z11 Z13]
+ (0.11383573679388659) [Z4 Z12]
+ (0.11383573679388659) [Z5 Z13]
+ (0.1195243896468274) [Z6 Z10]
+ (0.1195243896468274) [Z7 Z11]
+ (0.12489990917237608) [Z4 Z10]
+ (0.12489990917237608) [Z5 Z11]
+ (0.12495807739503201) [Z2 Z4]
+ (0.12495807739503201) [Z3 Z5]
+ (0.1279950249246839) [Z2 Z10]
+ (0.1279950249246839) [Z3 Z11]
+ (0.13401715261963765) [Z6 Z12]
+ (0.13401715261963765) [Z7 Z13]
+ (0.13701191674040808) [Z4 Z6]
+ (0.13701191674040808) [Z5 Z7]
+ (0.13734953064261296) [Z6 Z11]
+ (0.13734953064261296) [Z7 Z10]
+ (0.13739104762683263) [Z2 Z6]
+ (0.13739104762683263) [Z3 Z7]
+ (0.13766872645852557) [Z8 Z10]
+ (0.13766872645852557) [Z9 Z11]
+ (0.1401128986535482) [Z2 Z12]
+ (0.1401128986535482) [Z3 Z13]
+ (0.1413890529194285) [Z10 Z13]
+ (0.1413890529194285) [Z11 Z12]
+ (0.1425799771248577) [Z4 Z11]
+ (0.1425799771248577) [Z5 Z10]
+ (0.14722943218766182) [Z8 Z11]
+ (0.14722943218766182) [Z9 Z10]
+ (0.1489943057506558) [Z4 Z7]
+ (0.1489943057506558) [Z5 Z6]
+ (0.14926355147388973) [Z10 Z11]
+ (0.149607026844453) [Z4 Z8]
+ (0.149607026844453) [Z5 Z9]
+ (0.14973486803496924) [Z8 Z12]
+ (0.14973486803496924) [Z9 Z13]
+ (0.15071408121008284) [Z2 Z8]
+ (0.15071408121008284) [Z3 Z9]
+ (0.15138327161428888) [Z6 Z13]
+ (0.15138327161428888) [Z7 Z12]
+ (0.1521504070886906) [Z4 Z13]
+ (0.1521504070886906) [Z5 Z12]
+ (0.15337968243314176) [Z2 Z11]
+ (0.15337968243314176) [Z3 Z10]
+ (0.15435748657223666) [Z12 Z13]
+ (0.15569010671752465) [Z2 Z13]
+ (0.15569010671752465) [Z3 Z12]
+ (0.1558226905155312) [Z8 Z13]
+ (0.1558226905155312) [Z9 Z12]
+ (0.15676396176430998) [Z4 Z9]
+ (0.15676396176430998) [Z5 Z8]
+ (0.1575531479798568) [Z4 Z5]
+ (0.16079764534838575) [Z2 Z5]
+ (0.16079764534838575) [Z3 Z4]
+ (0.16756653265461355) [Z6 Z8]
+ (0.16756653265461355) [Z7 Z9]
+ (0.16853486561579945) [Z2 Z7]
+ (0.16853486561579945) [Z3 Z6]
+ (0.18143991440303991) [Z6 Z9]
+ (0.18143991440303991) [Z7 Z8]
+ (0.18189085790751328) [Z2 Z3]
+ (0.18690820476912517) [Z2 Z9]
+ (0.18690820476912517) [Z3 Z8]
+ (0.19299723935364185) [Z0 Z10]
+ (0.19299723935364185) [Z1 Z11]
+ (0.19392534613270465) [Z6 Z7]
+ (0.19661770890342148) [Z0 Z4]
+ (0.19661770890342148) [Z1 Z5]
+ (0.1993635453736083) [Z0 Z5]
+ (0.1993635453736083) [Z1 Z4]
+ (0.20072866460441727) [Z0 Z11]
+ (0.20072866460441727) [Z1 Z10]
+ (0.21102659849791522) [Z0 Z12]
+ (0.21102659849791522) [Z1 Z13]
+ (0.21631037498631817) [Z0 Z13]
+ (0.21631037498631817) [Z1 Z12]
+ (0.23671080783830287) [Z0 Z2]
+ (0.23671080783830287) [Z1 Z3]
+ (0.24164663936017375) [Z0 Z6]
+ (0.24164663936017375) [Z1 Z7]
+ (0.24853483371314442) [Z0 Z7]
+ (0.24853483371314442) [Z1 Z6]
+ (0.25129445674591533) [Z0 Z3]
+ (0.25129445674591533) [Z1 Z2]
+ (0.27232518306605685) [Z0 Z8]
+ (0.27232518306605685) [Z1 Z9]
+ (0.2788345442672341) [Z0 Z9]
+ (0.2788345442672341) [Z1 Z8]
+ (1.186176373486046) [Z0 Z1]
+ (-1.2260484989158237e-05) [Y4 Z5 Y6]
+ (-1.2260484989158237e-05) [X4 Z5 X6]
+ (-1.2260484989158237e-05) [Y5 Z6 Y7]
+ (-1.2260484989158237e-05) [X5 Z6 X7]
+ (-1.072231215981411e-05) [Y11 Z12 Y13]
+ (-1.072231215981411e-05) [X11 Z12 X13]
+ (-1.0722312159814106e-05) [Y10 Z11 Y12]
+ (-1.0722312159814106e-05) [X10 Z11 X12]
+ (-3.887051673802145e-06) [Y2 Z3 Y4]
+ (-3.887051673802145e-06) [X2 Z3 X4]
+ (-3.887051673802145e-06) [Y3 Z4 Y5]
+ (-3.887051673802145e-06) [X3 Z4 X5]
+ (0.1250703257977204) [Y1 Z2 Y3]
+ (0.1250703257977204) [X1 Z2 X3]
+ (0.12507032579772046) [Y0 Z1 Y2]
+ (0.12507032579772046) [X0 Z1 X2]
+ (-0.03619412355904233) [Y2 Y3 X8 X9]
+ (-0.03619412355904233) [X2 X3 Y8 Y9]
+ (-0.03583956795335374) [Y2 Y3 X4 X5]
+ (-0.03583956795335374) [X2 X3 Y4 Y5]
+ (-0.03114381798896682) [Y2 Y3 X6 X7]
+ (-0.03114381798896682) [X2 X3 Y6 Y7]
+ (-0.028685183716106424) [Y10 Y11 X12 X13]
+ (-0.028685183716106424) [X10 X11 Y12 Y13]
+ (-0.025996177598021725) [Y3 Z4 Z5 Y7]
+ (-0.025996177598021725) [X3 Z4 Z5 X7]
+ (-0.02538465750845784) [Y2 Y3 X10 X11]
+ (-0.02538465750845784) [X2 X3 Y10 Y11]
+ (-0.01902824244384804) [Y3 Y4 X11 X12]
+ (-0.01902824244384804) [X3 X4 Y11 Y12]
+ (-0.017680067952481643) [Y4 Y5 X10 X11]
+ (-0.017680067952481643) [X4 X5 Y10 Y11]
+ (-0.01736611899465121) [Y6 Y7 X12 X13]
+ (-0.01736611899465121) [X6 X7 Y12 Y13]
+ (-0.015577208063976448) [Y2 Y3 X12 X13]
+ (-0.015577208063976448) [X2 X3 Y12 Y13]
+ (-0.014583648907612472) [Y0 Y1 X2 X3]
+ (-0.014583648907612472) [X0 X1 Y2 Y3]
+ (-0.013873381748426398) [Y6 Y7 X8 X9]
+ (-0.013873381748426398) [X6 X7 Y8 Y9]
+ (-0.011982389010247717) [Y4 Y5 X6 X7]
+ (-0.011982389010247717) [X4 X5 Y6 Y7]
+ (-0.009560705729136228) [Y8 Y9 X10 X11]
+ (-0.009560705729136228) [X8 X9 Y10 Y11]
+ (-0.0077314252507754205) [Y0 Y1 X10 X11]
+ (-0.0077314252507754205) [X0 X1 Y10 Y11]
+ (-0.007156934919856984) [Y4 Y5 X8 X9]
+ (-0.007156934919856984) [X4 X5 Y8 Y9]
+ (-0.006888194352970669) [Y0 Y1 X6 X7]
+ (-0.006888194352970669) [X0 X1 Y6 Y7]
+ (-0.006509361201177241) [Y0 Y1 X8 X9]
+ (-0.006509361201177241) [X0 X1 Y8 Y9]
+ (-0.006087822480561943) [Y8 Y9 X12 X13]
+ (-0.006087822480561943) [X8 X9 Y12 Y13]
+ (-0.005283776488402962) [Y0 Y1 X12 X13]
+ (-0.005283776488402962) [X0 X1 Y12 Y13]
+ (-0.005143391768824783) [Y3 X4 X5 Y6]
+ (-0.005143391768824783) [X3 Y4 Y5 X6]
+ (-0.004575007626639164) [Y1 X2 X12 Y13]
+ (-0.004575007626639164) [Y1 Y2 Y12 Y13]
+ (-0.004575007626639164) [X1 X2 X12 X13]
+ (-0.004575007626639164) [X1 Y2 Y12 X13]
+ (-0.004424855449441849) [Y1 X2 X4 Y5]
+ (-0.004424855449441849) [Y1 Y2 Y4 Y5]
+ (-0.004424855449441849) [X1 X2 X4 X5]
+ (-0.004424855449441849) [X1 Y2 Y4 X5]
+ (-0.003479511890333995) [Y2 Z3 Z5 Y6]
+ (-0.003479511890333995) [X2 Z3 Z5 X6]
+ (-0.003479511890333995) [Y3 Z4 Z6 Y7]
+ (-0.003479511890333995) [X3 Z4 Z6 X7]
+ (-0.0017992194936628614) [Y1 X2 X10 Y11]
+ (-0.0017992194936628614) [Y1 Y2 Y10 Y11]
+ (-0.0017992194936628614) [X1 X2 X10 X11]
+ (-0.0017992194936628614) [X1 Y2 Y10 X11]
+ (-0.00029219862611130345) [Y7 Y8 X9 X10]
+ (-0.00029219862611130345) [X7 X8 Y9 Y10]
+ (-8.194261372724242e-06) [Z10 Y11 Z12 Y13]
+ (-8.194261372724242e-06) [Z10 X11 Z12 X13]
+ (-7.801707500919818e-06) [Y2 Z3 Y4 Z11]
+ (-7.801707500919818e-06) [X2 Z3 X4 Z11]
+ (-7.801707500919818e-06) [Y3 Z4 Y5 Z10]
+ (-7.801707500919818e-06) [X3 Z4 X5 Z10]
+ (-4.643051068760036e-06) [Y3 X4 X10 Y11]
+ (-4.643051068760036e-06) [Y3 Y4 Y10 Y11]
+ (-4.643051068760036e-06) [X3 X4 X10 X11]
+ (-4.643051068760036e-06) [X3 Y4 Y10 X11]
+ (-4.5888551558773485e-06) [Y4 Z5 Y6 Z13]
+ (-4.5888551558773485e-06) [X4 Z5 X6 Z13]
+ (-4.5888551558773485e-06) [Y5 Z6 Y7 Z12]
+ (-4.5888551558773485e-06) [X5 Z6 X7 Z12]
+ (-4.556569218453366e-06) [Y5 X6 X12 Y13]
+ (-4.556569218453366e-06) [Y5 Y6 Y12 Y13]
+ (-4.556569218453366e-06) [X5 X6 X12 X13]
+ (-4.556569218453366e-06) [X5 Y6 Y12 X13]
+ (-3.6945132948955697e-06) [Y4 X5 X11 Y12]
+ (-3.6945132948955697e-06) [Y4 Y5 Y11 Y12]
+ (-3.6945132948955697e-06) [X4 X5 X11 X12]
+ (-3.6945132948955697e-06) [X4 Y5 Y11 X12]
+ (-3.344081556562569e-06) [Z0 Y5 Z6 Y7]
+ (-3.344081556562569e-06) [Z0 X5 Z6 X7]
+ (-3.344081556562569e-06) [Z1 Y4 Z5 Y6]
+ (-3.344081556562569e-06) [Z1 X4 Z5 X6]
+ (-3.158656432159781e-06) [Y2 Z3 Y4 Z10]
+ (-3.158656432159781e-06) [X2 Z3 X4 Z10]
+ (-3.158656432159781e-06) [Y3 Z4 Y5 Z11]
+ (-3.158656432159781e-06) [X3 Z4 X5 Z11]
+ (-3.0993492436652463e-06) [Z0 Y4 Z5 Y6]
+ (-3.0993492436652463e-06) [Z0 X4 Z5 X6]
+ (-3.0993492436652463e-06) [Z1 Y5 Z6 Y7]
+ (-3.0993492436652463e-06) [Z1 X5 Z6 X7]
+ (-2.8909678818627316e-06) [Z6 Y11 Z12 Y13]
+ (-2.8909678818627316e-06) [Z6 X11 Z12 X13]
+ (-2.8909678818627316e-06) [Z7 Y10 Z11 Y12]
+ (-2.8909678818627316e-06) [Z7 X10 Z11 X12]
+ (-2.1776646053879696e-06) [Z0 Y10 Z11 Y12]
+ (-2.1776646053879696e-06) [Z0 X10 Z11 X12]
+ (-2.1776646053879696e-06) [Z1 Y11 Z12 Y13]
+ (-2.1776646053879696e-06) [Z1 X11 Z12 X13]
+ (-1.8818501832419672e-06) [Y4 Z5 Y6 Z9]
+ (-1.8818501832419672e-06) [X4 Z5 X6 Z9]
+ (-1.8818501832419672e-06) [Y5 Z6 Y7 Z8]
+ (-1.8818501832419672e-06) [X5 Z6 X7 Z8]
+ (-1.8551201217773677e-06) [Z6 Y10 Z11 Y12]
+ (-1.8551201217773677e-06) [Z6 X10 Z11 X12]
+ (-1.8551201217773677e-06) [Z7 Y11 Z12 Y13]
+ (-1.8551201217773677e-06) [Z7 X11 Z12 X13]
+ (-1.854060858047903e-06) [Y4 Z5 Y6 Z7]
+ (-1.854060858047903e-06) [X4 Z5 X6 Z7]
+ (-1.8163031702553636e-06) [Z4 Y11 Z12 Y13]
+ (-1.8163031702553636e-06) [Z4 X11 Z12 X13]
+ (-1.8163031702553636e-06) [Z5 Y10 Z11 Y12]
+ (-1.8163031702553636e-06) [Z5 X10 Z11 X12]
+ (-1.6923978286473816e-06) [Y4 Z5 Y6 Z10]
+ (-1.6923978286473816e-06) [X4 Z5 X6 Z10]
+ (-1.6923978286473816e-06) [Y5 Z6 Y7 Z11]
+ (-1.6923978286473816e-06) [X5 Z6 X7 Z11]
+ (-1.614879414212594e-06) [Z0 Y11 Z12 Y13]
+ (-1.614879414212594e-06) [Z0 X11 Z12 X13]
+ (-1.614879414212594e-06) [Z1 Y10 Z11 Y12]
+ (-1.614879414212594e-06) [Z1 X10 Z11 X12]
+ (-1.597317198037576e-06) [Z8 Y10 Z11 Y12]
+ (-1.597317198037576e-06) [Z8 X10 Z11 X12]
+ (-1.597317198037576e-06) [Z9 Y11 Z12 Y13]
+ (-1.597317198037576e-06) [Z9 X11 Z12 X13]
+ (-1.4548424490780712e-06) [Y3 X4 X6 Y7]
+ (-1.4548424490780712e-06) [Y3 Y4 Y6 Y7]
+ (-1.4548424490780712e-06) [X3 X4 X6 X7]
+ (-1.4548424490780712e-06) [X3 Y4 Y6 X7]
+ (-1.3980449081181876e-06) [Y4 Z5 Y6 Z8]
+ (-1.3980449081181876e-06) [X4 Z5 X6 Z8]
+ (-1.3980449081181876e-06) [Y5 Z6 Y7 Z9]
+ (-1.3980449081181876e-06) [X5 Z6 X7 Z9]
+ (-1.1954890099022918e-06) [Y2 Z3 Y4 Z7]
+ (-1.1954890099022918e-06) [X2 Z3 X4 Z7]
+ (-1.1954890099022918e-06) [Y3 Z4 Y5 Z6]
+ (-1.1954890099022918e-06) [X3 Z4 X5 Z6]
+ (-1.190850808401253e-06) [Z0 Y3 Z4 Y5]
+ (-1.190850808401253e-06) [Z0 X3 Z4 X5]
+ (-1.190850808401253e-06) [Z1 Y2 Z3 Y4]
+ (-1.190850808401253e-06) [Z1 X2 Z3 X4]
+ (-1.1708301370634642e-06) [Z2 Y5 Z6 Y7]
+ (-1.1708301370634642e-06) [Z2 X5 Z6 X7]
+ (-1.1708301370634642e-06) [Z3 Y4 Z5 Y6]
+ (-1.1708301370634642e-06) [Z3 X4 Z5 X6]
+ (-1.0632283424863984e-06) [Z2 Y10 Z11 Y12]
+ (-1.0632283424863984e-06) [Z2 X10 Z11 X12]
+ (-1.0632283424863984e-06) [Z3 Y11 Z12 Y13]
+ (-1.0632283424863984e-06) [Z3 X11 Z12 X13]
+ (-1.035847760085364e-06) [Y6 X7 X11 Y12]
+ (-1.035847760085364e-06) [Y6 Y7 Y11 Y12]
+ (-1.035847760085364e-06) [X6 X7 X11 X12]
+ (-1.035847760085364e-06) [X6 Y7 Y11 X12]
+ (-9.509249751597218e-07) [Z2 Y4 Z5 Y6]
+ (-9.509249751597218e-07) [Z2 X4 Z5 X6]
+ (-9.509249751597218e-07) [Z3 Y5 Z6 Y7]
+ (-9.509249751597218e-07) [Z3 X5 Z6 X7]
+ (-9.344557778557386e-07) [Z8 Y11 Z12 Y13]
+ (-9.344557778557386e-07) [Z8 X11 Z12 X13]
+ (-9.344557778557386e-07) [Z9 Y10 Z11 Y12]
+ (-9.344557778557386e-07) [Z9 X10 Z11 X12]
+ (-8.337746754558349e-07) [Z0 Y2 Z3 Y4]
+ (-8.337746754558349e-07) [Z0 X2 Z3 X4]
+ (-8.337746754558349e-07) [Z1 Y3 Z4 Y5]
+ (-8.337746754558349e-07) [Z1 X3 Z4 X5]
+ (-7.956895373372711e-07) [Y3 X4 X8 Y9]
+ (-7.956895373372711e-07) [Y3 Y4 Y8 Y9]
+ (-7.956895373372711e-07) [X3 X4 X8 X9]
+ (-7.956895373372711e-07) [X3 Y4 Y8 X9]
+ (-7.764994121636087e-07) [Y2 Z3 Y4 Z5]
+ (-7.764994121636087e-07) [X2 Z3 X4 Z5]
+ (-5.929765814773579e-07) [Z4 Y5 Z6 Y7]
+ (-5.929765814773579e-07) [Z4 X5 Z6 X7]
+ (-5.770052995167709e-07) [Y2 Z3 Y4 Z9]
+ (-5.770052995167709e-07) [X2 Z3 X4 Z9]
+ (-5.770052995167709e-07) [Y3 Z4 Y5 Z8]
+ (-5.770052995167709e-07) [X3 Z4 X5 Z8]
+ (-5.471647745009352e-07) [Y1 Y2 X11 X12]
+ (-5.471647745009352e-07) [X1 X2 Y11 Y12]
+ (-4.838052751237796e-07) [Y5 X6 X8 Y9]
+ (-4.838052751237796e-07) [Y5 Y6 Y8 Y9]
+ (-4.838052751237796e-07) [X5 X6 X8 X9]
+ (-4.838052751237796e-07) [X5 Y6 Y8 X9]
+ (-3.570761329454181e-07) [Y0 X1 X3 Y4]
+ (-3.570761329454181e-07) [Y0 Y1 Y3 Y4]
+ (-3.570761329454181e-07) [X0 X1 X3 X4]
+ (-3.570761329454181e-07) [X0 Y1 Y3 X4]
+ (-2.447323128973224e-07) [Y0 X1 X5 Y6]
+ (-2.447323128973224e-07) [Y0 Y1 Y5 Y6]
+ (-2.447323128973224e-07) [X0 X1 X5 X6]
+ (-2.447323128973224e-07) [X0 Y1 Y5 X6]
+ (-2.1990516190374242e-07) [Y2 X3 X5 Y6]
+ (-2.1990516190374242e-07) [Y2 Y3 Y5 Y6]
+ (-2.1990516190374242e-07) [X2 X3 X5 X6]
+ (-2.1990516190374242e-07) [X2 Y3 Y5 X6]
+ (-1.9332412772978313e-07) [Y1 X2 X3 Y4]
+ (-1.9332412772978313e-07) [X1 Y2 Y3 X4]
+ (-1.291969486530496e-07) [Y1 Z2 Z3 Y5]
+ (-1.291969486530496e-07) [X1 Z2 Z3 X5]
+ (1.7379332626569758e-07) [Y0 Z1 Z3 Y4]
+ (1.7379332626569758e-07) [X0 Z1 Z3 X4]
+ (1.7379332626569758e-07) [Y1 Z2 Z4 Y5]
+ (1.7379332626569758e-07) [X1 Z2 Z4 X5]
+ (1.9332412772978313e-07) [Y1 Y2 X3 X4]
+ (1.9332412772978313e-07) [X1 X2 Y3 Y4]
+ (2.1868423782050018e-07) [Y2 Z3 Y4 Z8]
+ (2.1868423782050018e-07) [X2 Z3 X4 Z8]
+ (2.1868423782050018e-07) [Y3 Z4 Y5 Z9]
+ (2.1868423782050018e-07) [X3 Z4 X5 Z9]
+ (2.5935343917577944e-07) [Y2 Z3 Y4 Z6]
+ (2.5935343917577944e-07) [X2 Z3 X4 Z6]
+ (2.5935343917577944e-07) [Y3 Z4 Y5 Z7]
+ (2.5935343917577944e-07) [X3 Z4 X5 Z7]
+ (3.6060718684076745e-07) [Y0 Z1 Z2 Y4]
+ (3.6060718684076745e-07) [X0 Z1 Z2 X4]
+ (3.6060718684076745e-07) [Y1 Z3 Z4 Y5]
+ (3.6060718684076745e-07) [X1 Z3 Z4 X5]
+ (5.471647745009352e-07) [Y1 X2 X11 Y12]
+ (5.471647745009352e-07) [X1 Y2 Y11 X12]
+ (5.627851911753755e-07) [Y0 X1 X11 Y12]
+ (5.627851911753755e-07) [Y0 Y1 Y11 Y12]
+ (5.627851911753755e-07) [X0 X1 X11 X12]
+ (5.627851911753755e-07) [X0 Y1 Y11 X12]
+ (6.62861420181837e-07) [Y8 X9 X11 Y12]
+ (6.62861420181837e-07) [Y8 Y9 Y11 Y12]
+ (6.62861420181837e-07) [X8 X9 X11 X12]
+ (6.62861420181837e-07) [X8 Y9 Y11 X12]
+ (1.109440759202579e-06) [Z2 Y11 Z12 Y13]
+ (1.109440759202579e-06) [Z2 X11 Z12 X13]
+ (1.109440759202579e-06) [Z3 Y10 Z11 Y12]
+ (1.109440759202579e-06) [Z3 X10 Z11 X12]
+ (1.6021167409472747e-06) [Z2 Y3 Z4 Y5]
+ (1.6021167409472747e-06) [Z2 X3 Z4 X5]
+ (1.878210124640206e-06) [Z4 Y10 Z11 Y12]
+ (1.878210124640206e-06) [Z4 X10 Z11 X12]
+ (1.878210124640206e-06) [Z5 Y11 Z12 Y13]
+ (1.878210124640206e-06) [Z5 X11 Z12 X13]
+ (2.1726691016889775e-06) [Y2 X3 X11 Y12]
+ (2.1726691016889775e-06) [Y2 Y3 Y11 Y12]
+ (2.1726691016889775e-06) [X2 X3 X11 X12]
+ (2.1726691016889775e-06) [X2 Y3 Y11 X12]
+ (3.11744794642372e-06) [Y0 Z2 Z3 Y4]
+ (3.11744794642372e-06) [X0 Z2 Z3 X4]
+ (3.5390541846078583e-06) [Y2 Z3 Y4 Z12]
+ (3.5390541846078583e-06) [X2 Z3 X4 Z12]
+ (3.5390541846078583e-06) [Y3 Z4 Y5 Z13]
+ (3.5390541846078583e-06) [X3 Z4 X5 Z13]
+ (4.281913885102219e-06) [Y4 Z5 Y6 Z11]
+ (4.281913885102219e-06) [X4 Z5 X6 Z11]
+ (4.281913885102219e-06) [Y5 Z6 Y7 Z10]
+ (4.281913885102219e-06) [X5 Z6 X7 Z10]
+ (5.2758831224411756e-06) [Y3 X4 X12 Y13]
+ (5.2758831224411756e-06) [Y3 Y4 Y12 Y13]
+ (5.2758831224411756e-06) [X3 X4 X12 X13]
+ (5.2758831224411756e-06) [X3 Y4 Y12 X13]
+ (5.974311713749601e-06) [Y5 X6 X10 Y11]
+ (5.974311713749601e-06) [Y5 Y6 Y10 Y11]
+ (5.974311713749601e-06) [X5 X6 X10 X11]
+ (5.974311713749601e-06) [X5 Y6 Y10 X11]
+ (7.954413176439364e-06) [Y10 Z11 Y12 Z13]
+ (7.954413176439364e-06) [X10 Z11 X12 Z13]
+ (8.814937307049034e-06) [Y2 Z3 Y4 Z13]
+ (8.814937307049034e-06) [X2 Z3 X4 Z13]
+ (8.814937307049034e-06) [Y3 Z4 Y5 Z12]
+ (8.814937307049034e-06) [X3 Z4 X5 Z12]
+ (0.00029219862611130345) [Y7 X8 X9 Y10]
+ (0.00029219862611130345) [X7 Y8 Y9 X10]
+ (0.0004956762314924487) [Y2 Z4 Z5 Y6]
+ (0.0004956762314924487) [X2 Z4 Z5 X6]
+ (0.0011059037691896869) [Y0 Z1 Y2 Z5]
+ (0.0011059037691896869) [X0 Z1 X2 Z5]
+ (0.0011059037691896869) [Y1 Z2 Y3 Z4]
+ (0.0011059037691896869) [X1 Z2 X3 Z4]
+ (0.0016638798784907875) [Y2 Z3 Z4 Y6]
+ (0.0016638798784907875) [X2 Z3 Z4 X6]
+ (0.0016638798784907875) [Y3 Z5 Z6 Y7]
+ (0.0016638798784907875) [X3 Z5 Z6 X7]
+ (0.0017560707018412253) [Y0 Z1 Y2 Z11]
+ (0.0017560707018412253) [X0 Z1 X2 Z11]
+ (0.0017560707018412253) [Y1 Z2 Y3 Z10]
+ (0.0017560707018412253) [X1 Z2 X3 Z10]
+ (0.0023262306231580936) [Y0 Z1 Y2 Z13]
+ (0.0023262306231580936) [X0 Z1 X2 Z13]
+ (0.0023262306231580936) [Y1 Z2 Y3 Z12]
+ (0.0023262306231580936) [X1 Z2 X3 Z12]
+ (0.002929768674751058) [Y0 Z1 Y2 Z9]
+ (0.002929768674751058) [X0 Z1 X2 Z9]
+ (0.002929768674751058) [Y1 Z2 Y3 Z8]
+ (0.002929768674751058) [X1 Z2 X3 Z8]
+ (0.0032769719312315646) [Y0 Z1 Y2 Z3]
+ (0.0032769719312315646) [X0 Z1 X2 Z3]
+ (0.003347617530666261) [Y0 Z1 Y2 Z7]
+ (0.003347617530666261) [X0 Z1 X2 Z7]
+ (0.003347617530666261) [Y1 Z2 Y3 Z6]
+ (0.003347617530666261) [X1 Z2 X3 Z6]
+ (0.003555290195504087) [Y0 Z1 Y2 Z10]
+ (0.003555290195504087) [X0 Z1 X2 Z10]
+ (0.003555290195504087) [Y1 Z2 Y3 Z11]
+ (0.003555290195504087) [X1 Z2 X3 Z11]
+ (0.005143391768824783) [Y3 Y4 X5 X6]
+ (0.005143391768824783) [X3 X4 Y5 Y6]
+ (0.005283776488402962) [Y0 X1 X12 Y13]
+ (0.005283776488402962) [X0 Y1 Y12 X13]
+ (0.005530759218631534) [Y0 Z1 Y2 Z4]
+ (0.005530759218631534) [X0 Z1 X2 Z4]
+ (0.005530759218631534) [Y1 Z2 Y3 Z5]
+ (0.005530759218631534) [X1 Z2 X3 Z5]
+ (0.006087822480561943) [Y8 X9 X12 Y13]
+ (0.006087822480561943) [X8 Y9 Y12 X13]
+ (0.006509361201177241) [Y0 X1 X8 Y9]
+ (0.006509361201177241) [X0 Y1 Y8 X9]
+ (0.006888194352970669) [Y0 X1 X6 Y7]
+ (0.006888194352970669) [X0 Y1 Y6 X7]
+ (0.006901238249797257) [Y0 Z1 Y2 Z12]
+ (0.006901238249797257) [X0 Z1 X2 Z12]
+ (0.006901238249797257) [Y1 Z2 Y3 Z13]
+ (0.006901238249797257) [X1 Z2 X3 Z13]
+ (0.007156934919856984) [Y4 X5 X8 Y9]
+ (0.007156934919856984) [X4 Y5 Y8 X9]
+ (0.0077314252507754205) [Y0 X1 X10 Y11]
+ (0.0077314252507754205) [X0 Y1 Y10 X11]
+ (0.008032520918821498) [Y0 Z1 Y2 Z6]
+ (0.008032520918821498) [X0 Z1 X2 Z6]
+ (0.008032520918821498) [Y1 Z2 Y3 Z7]
+ (0.008032520918821498) [X1 Z2 X3 Z7]
+ (0.009560705729136228) [Y8 X9 X10 Y11]
+ (0.009560705729136228) [X8 Y9 Y10 X11]
+ (0.011055020596132045) [Y0 Z1 Y2 Z8]
+ (0.011055020596132045) [X0 Z1 X2 Z8]
+ (0.011055020596132045) [Y1 Z2 Y3 Z9]
+ (0.011055020596132045) [X1 Z2 X3 Z9]
+ (0.011307274008848105) [Y7 Z8 Z9 Y11]
+ (0.011307274008848105) [X7 Z8 Z9 X11]
+ (0.011982389010247717) [Y4 X5 X6 Y7]
+ (0.011982389010247717) [X4 Y5 Y6 X7]
+ (0.013873381748426398) [Y6 X7 X8 Y9]
+ (0.013873381748426398) [X6 Y7 Y8 X9]
+ (0.014583648907612472) [Y0 X1 X2 Y3]
+ (0.014583648907612472) [X0 Y1 Y2 X3]
+ (0.015577208063976448) [Y2 X3 X12 Y13]
+ (0.015577208063976448) [X2 Y3 Y12 X13]
+ (0.01736611899465121) [Y6 X7 X12 Y13]
+ (0.01736611899465121) [X6 Y7 Y12 X13]
+ (0.017680067952481643) [Y4 X5 X10 Y11]
+ (0.017680067952481643) [X4 Y5 Y10 X11]
+ (0.01902824244384804) [Y3 X4 X11 Y12]
+ (0.01902824244384804) [X3 Y4 Y11 X12]
+ (0.02538465750845784) [Y2 X3 X10 Y11]
+ (0.02538465750845784) [X2 Y3 Y10 X11]
+ (0.028685183716106424) [Y10 X11 X12 Y13]
+ (0.028685183716106424) [X10 Y11 Y12 X13]
+ (0.02981242451734475) [Y6 Z7 Z8 Y10]
+ (0.02981242451734475) [X6 Z7 Z8 X10]
+ (0.02981242451734475) [Y7 Z9 Z10 Y11]
+ (0.02981242451734475) [X7 Z9 Z10 X11]
+ (0.030104623143456057) [Y6 Z7 Z9 Y10]
+ (0.030104623143456057) [X6 Z7 Z9 X10]
+ (0.030104623143456057) [Y7 Z8 Z10 Y11]
+ (0.030104623143456057) [X7 Z8 Z10 X11]
+ (0.030787505389143523) [Y6 Z8 Z9 Y10]
+ (0.030787505389143523) [X6 Z8 Z9 X10]
+ (0.03114381798896682) [Y2 X3 X6 Y7]
+ (0.03114381798896682) [X2 Y3 Y6 X7]
+ (0.03583956795335374) [Y2 X3 X4 Y5]
+ (0.03583956795335374) [X2 Y3 Y4 X5]
+ (0.03619412355904233) [Y2 X3 X8 Y9]
+ (0.03619412355904233) [X2 Y3 Y8 X9]
+ (0.10433064780651312) [Z0 Y1 Z2 Y3]
+ (0.10433064780651312) [Z0 X1 Z2 X3]
+ (-0.12133276911042257) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133276911042257) [X2 Z3 Z4 Z5 X6]
+ (-0.12133276911042257) [Y3 Z4 Z5 Z6 Y7]
+ (-0.12133276911042257) [X3 Z4 Z5 Z6 X7]
+ (3.202076880536128e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.202076880536128e-06) [X0 Z1 Z2 Z3 X4]
+ (3.202076880536128e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.202076880536128e-06) [X1 Z2 Z3 Z4 X5]
+ (0.2284810656491832) [Y6 Z7 Z8 Z9 Y10]
+ (0.2284810656491832) [X6 Z7 Z8 Z9 X10]
+ (0.22848106564918325) [Y7 Z8 Z9 Z10 Y11]
+ (0.22848106564918325) [X7 Z8 Z9 Z10 X11]
+ (-0.03276765782328977) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.03276765782328977) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.03276765782328977) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.03276765782328977) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845272465) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.027115036845272465) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845272465) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.027115036845272465) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.02599617759802172) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.02599617759802172) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.017561202409646023) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.017561202409646023) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.017561202409646023) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.017561202409646023) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.014564531231172868) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.014564531231172868) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.014564531231172868) [X7 Z8 Z9 X10 X12 X13]
+ (-0.014564531231172868) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.012215040997613473) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.012215040997613473) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.012215040997613473) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.012215040997613473) [X4 Z5 X6 X11 Z12 X13]
+ (-0.012215040997613473) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.012215040997613473) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.012215040997613473) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.012215040997613473) [X5 Z6 X7 X10 Z11 X12]
+ (-0.011756013419819324) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.011756013419819324) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.011756013419819324) [X3 Z4 Z5 X6 X8 X9]
+ (-0.011756013419819324) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.008764827575689166) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.008764827575689166) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.008764827575689166) [X2 Z3 Z4 X5 X11 X12]
+ (-0.008764827575689166) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.008764827575689166) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.008764827575689166) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.008764827575689166) [X3 X4 X10 Z11 Z12 X13]
+ (-0.008764827575689166) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.007306759928832772) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.007306759928832772) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.007306759928832772) [X4 X5 X7 Z8 Z9 X10]
+ (-0.007306759928832772) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.0058051889898267) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.0058051889898267) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.0058051889898267) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.0058051889898267) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.005143391768824782) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.005143391768824782) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.005143391768824782) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.005143391768824782) [X2 Z3 X4 X5 Z6 X7]
+ (-0.004684903388155237) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.004684903388155237) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668620318776292) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.004668620318776292) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.004575007626639163) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.004575007626639163) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.004424855449441849) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.004424855449441849) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.003493790359890154) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.003493790359890154) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.003493790359890154) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.003493790359890154) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.002779026799025665) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.002779026799025665) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.002293956611352467) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.002293956611352467) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.0017992194936628614) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.0017992194936628614) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.001727875394136986) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.001727875394136986) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.0009298507967729303) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298507967729303) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298507967729303) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298507967729303) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533856254125882) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.0008533856254125882) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.0008145313270956503) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.0008145313270956503) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.0008145313270956503) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.0008145313270956503) [X3 Z4 Z5 Z6 X7 Z11]
+ (-7.735036880601781e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.735036880601781e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.735036880601781e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.735036880601781e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-8.774817865257743e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.774817865257743e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.774817865257743e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.774817865257743e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.518362216223681e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.518362216223681e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.518362216223681e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.518362216223681e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.444344676477034e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.444344676477034e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.444344676477034e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.444344676477034e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.524373848927449e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.524373848927449e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.524373848927449e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.524373848927449e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-6.2900284337583275e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.2900284337583275e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.2900284337583275e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.2900284337583275e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.974311713749601e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (-5.974311713749601e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (-5.2758831224411756e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.2758831224411756e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.643051068760036e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.643051068760036e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.556569218453366e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (-4.556569218453366e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (-4.253224225758256e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.253224225758256e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-3.7696594523398868e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.7696594523398868e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.6945132948955697e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.6945132948955697e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.6102971309210233e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.6102971309210233e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.6102971309210233e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.6102971309210233e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.3131455002295e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.3131455002295e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.2774831958768533e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.2774831958768533e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.2774831958768533e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.2774831958768533e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.211228348697949e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.211228348697949e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.211228348697949e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.211228348697949e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.1513463112824505e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.1513463112824505e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.0882507115799094e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.0882507115799094e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.1726691016889775e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.1726691016889775e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.4548424490780712e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.4548424490780712e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.3304731887807094e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.3304731887807094e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.2283337824653537e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.2283337824653537e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.035847760085364e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.035847760085364e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-7.956895373372711e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.956895373372711e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.733197742744276e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.733197742744276e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.733197742744276e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.733197742744276e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.62861420181837e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.62861420181837e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.556281915085006e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.556281915085006e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.556281915085006e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.556281915085006e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.418291575234818e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.418291575234818e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.418291575234818e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.418291575234818e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.92745308319344e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.92745308319344e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.92745308319344e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.92745308319344e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.627851911753755e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.627851911753755e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.287660625223219e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.287660625223219e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.287660625223219e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.287660625223219e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.287660625223219e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.287660625223219e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.287660625223219e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.287660625223219e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.838052751237796e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (-4.838052751237796e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (-3.5707613294541817e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.5707613294541817e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.328139350441701e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.328139350441701e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.0868265656852585e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.0868265656852585e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.0868265656852585e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.0868265656852585e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.447323128973224e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (-2.447323128973224e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (-2.371328948139757e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.371328948139757e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.371328948139757e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.371328948139757e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.1990516190374242e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-2.1990516190374242e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-1.9332412772978313e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.9332412772978313e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.9332412772978313e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.9332412772978313e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.8394209155989254e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.8394209155989254e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.8394209155989254e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.8394209155989254e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.5510539176738662e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (-1.5510539176738662e-07) [X0 Z1 X2 X4 Z5 X6]
+ (-1.5510539176738662e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (-1.5510539176738662e-07) [X1 Z2 X3 X5 Z6 X7]
+ (-1.380778148177076e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (-1.380778148177076e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (-1.380778148177076e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (-1.380778148177076e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (-1.380778148177076e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (-1.380778148177076e-07) [X0 Z1 X2 X5 Z6 X7]
+ (-1.380778148177076e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (-1.380778148177076e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (-1.380778148177076e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (-1.380778148177076e-07) [X1 Z2 X3 X4 Z5 X6]
+ (-1.380778148177076e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (-1.380778148177076e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (-1.291969486530496e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.291969486530496e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.1076325600436262e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.1076325600436262e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.1076325600436262e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.1076325600436262e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.1076325600436262e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.1076325600436262e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.1076325600436262e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.1076325600436262e-07) [X1 Z2 X3 X10 Z11 X12]
+ (8.057446595508358e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.057446595508358e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.057446595508358e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.057446595508358e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.649310136258366e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.649310136258366e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.649310136258366e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.649310136258366e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.8394209155989254e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.8394209155989254e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.8394209155989254e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.8394209155989254e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.1990516190374242e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (2.1990516190374242e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (2.447323128973224e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (2.447323128973224e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.236259961765594e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.236259961765594e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.236259961765594e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.236259961765594e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.328139350441701e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.328139350441701e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.5707613294541817e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.5707613294541817e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.838052751237796e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (4.838052751237796e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (5.627851911753755e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.627851911753755e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (6.62861420181837e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.62861420181837e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (7.956895373372711e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.956895373372711e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.30653665228722e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.30653665228722e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.30653665228722e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.30653665228722e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.035847760085364e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.035847760085364e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.2283337824653537e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.2283337824653537e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.2393363217972478e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.2393363217972478e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.2393363217972478e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.2393363217972478e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.3304731887807094e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.3304731887807094e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.4548424490780712e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.4548424490780712e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.1726691016889775e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.1726691016889775e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.0882507115799094e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.0882507115799094e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.11744794642372e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.11744794642372e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.1513463112824505e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.1513463112824505e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.3131455002295e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.3131455002295e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.334331289458342e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.334331289458342e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.6945132948955697e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.6945132948955697e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (4.183932559451153e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.183932559451153e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.556569218453366e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (4.556569218453366e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (4.643051068760036e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.643051068760036e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.2758831224411756e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.2758831224411756e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.974311713749601e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (5.974311713749601e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (0.0002921986261113034) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.0002921986261113034) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.0002921986261113034) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.0002921986261113034) [X6 Z7 X8 X9 Z10 X11]
+ (0.0004956762314924487) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.0004956762314924487) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.0006650070219498465) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.0006650070219498465) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.0006650070219498465) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.0006650070219498465) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.0008533856254125882) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.0008533856254125882) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.001609531381721381) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.001609531381721381) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.001609531381721381) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.001609531381721381) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.0016676041811441332) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.0016676041811441332) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.0016676041811441332) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.0016676041811441332) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.001727875394136986) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.001727875394136986) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.0017992194936628614) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.0017992194936628614) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.002293956611352467) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.002293956611352467) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.002462917007133969) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.002462917007133969) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.002462917007133969) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.002462917007133969) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.003961560792496601) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.003961560792496601) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.003961560792496601) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.003961560792496601) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.004424855449441849) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.004424855449441849) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.004575007626639163) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.004575007626639163) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.004668620318776292) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.004668620318776292) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.004684903388155237) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.004684903388155237) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.005324835234221474) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.005324835234221474) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.005324835234221474) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.005324835234221474) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.005368659358109169) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.005368659358109169) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.005368659358109169) [X2 X3 X7 Z8 Z9 X10]
+ (0.005368659358109169) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.007960880725921335) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.007960880725921335) [X4 Z5 X6 X10 Z11 X12]
+ (0.007960880725921335) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.007960880725921335) [X5 Z6 X7 X11 Z12 X13]
+ (0.008890731522694265) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.008890731522694265) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.008890731522694265) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.008890731522694265) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.010263414868158873) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.010263414868158873) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.010263414868158873) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.010263414868158873) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.010540425907671087) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.010540425907671087) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.010540425907671087) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.010540425907671087) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.010960074940542262) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.010960074940542262) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.010960074940542262) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.010960074940542262) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.011307274008848105) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.011307274008848105) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.014411099430131297) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.014411099430131297) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.014411099430131297) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.014411099430131297) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.015225630757226946) [Y3 Z4 Z5 X6 X10 Y11]
+ (0.015225630757226946) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (0.015225630757226946) [X3 Z4 Z5 X6 X10 X11]
+ (0.015225630757226946) [X3 Z4 Z5 Y6 Y10 X11]
+ (0.015588250102380345) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (0.015588250102380345) [X2 Z3 X4 X10 Z11 X12]
+ (0.015588250102380345) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (0.015588250102380345) [X3 Z4 X5 X11 Z12 X13]
+ (0.018266834869375036) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.018266834869375036) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.018266834869375036) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.018266834869375036) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.01902042317303964) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.01902042317303964) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.01902042317303964) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.01902042317303964) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.020175921723534808) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.020175921723534808) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.020175921723534808) [X4 Z5 Z6 X7 X11 X12]
+ (0.020175921723534808) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.020175921723534808) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.020175921723534808) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.020175921723534808) [X5 X6 X10 Z11 Z12 X13]
+ (0.020175921723534808) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.024353077678069514) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.024353077678069514) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.024353077678069514) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.024353077678069514) [X2 Z3 X4 X11 Z12 X13]
+ (0.024353077678069514) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.024353077678069514) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.024353077678069514) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.024353077678069514) [X3 Z4 X5 X10 Z11 X12]
+ (0.024389082531148805) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.024389082531148805) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.024389082531148805) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.024389082531148805) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.025104957138843954) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.025104957138843954) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.025104957138843954) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.025104957138843954) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.030787505389143523) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.030787505389143523) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.04587947078129822) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.04587947078129822) [X0 Z2 Z3 Z4 Z5 X6]
+ (0.05600733087780635) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.05600733087780635) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.05600733087780635) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.05600733087780635) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661237) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.05608468124661237) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661237) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.05608468124661237) [Z1 X7 Z8 Z9 Z10 X11]
+ (-6.631277928899776e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.631277928899776e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.631277928899776e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.631277928899776e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.5950860072640034e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.5950860072640034e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.5950860072640027e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.5950860072640027e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.042743277013783304) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.042743277013783304) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (0.04274327701378332) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.04274327701378332) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (-0.047642612176383166) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.047642612176383166) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.047642612176383166) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.047642612176383166) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.04171881383982177) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.04171881383982177) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.04171881383982177) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.04171881383982177) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.03956441632289429) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.03956441632289429) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.03956441632289429) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03956441632289429) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.039359168022054136) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.039359168022054136) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.039359168022054136) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.039359168022054136) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.03931805194719791) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03931805194719791) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.03931805194719791) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03931805194719791) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03560837898831293) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.03560837898831293) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.029903789512625387) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.029903789512625387) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.029903789512625387) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.029903789512625387) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.02873077955190587) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.02873077955190587) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.02873077955190587) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.02873077955190587) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.02563723829602681) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.02563723829602681) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.02563723829602681) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.02563723829602681) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.024755463292891675) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.024755463292891675) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.024755463292891675) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.024755463292891675) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.024282117354693197) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.024282117354693197) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.02314513092952917) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.02314513092952917) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.022528440196013085) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.022528440196013085) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.021433810721602106) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.021433810721602106) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.021433810721602106) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.021433810721602106) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251527) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.019257505095251527) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.01902824244384804) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.01902824244384804) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.018889030304943252) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-0.018889030304943252) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-0.018889030304943252) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.018889030304943252) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.015225630757226946) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.015225630757226946) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.014603704729162462) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.014603704729162462) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.014564531231172868) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.014564531231172868) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.011756013419819324) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.011756013419819324) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.009841749246962617) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.009841749246962617) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.009612634606847213) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.009612634606847213) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.009612634606847213) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.009612634606847213) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.00846997879102328) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.00846997879102328) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.007306759928832772) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.007306759928832772) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005923798336561396) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.005923798336561396) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.005368659358109168) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005368659358109168) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.0033566705638329282) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.0033566705638329282) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.0033566705638329282) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.0033566705638329282) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.0032675138544235654) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.0032675138544235654) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.0032675138544235654) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.0032675138544235654) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.0027790267990256655) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.0027790267990256655) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.00268604097780669) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.00268604097780669) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.00268604097780669) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.00268604097780669) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.002293956611352467) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.002293956611352467) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.002293956611352467) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.002293956611352467) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.0009581655836697038) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.0009581655836697038) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.0009581655836697038) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.0009581655836697038) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.0009581655836697038) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.0009581655836697038) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.0009581655836697038) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.0009581655836697038) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.0002463643756963844) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0002463643756963844) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303542334) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.00013840177303542334) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.00013840177303542334) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.00013840177303542334) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.735036880601781e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.735036880601781e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585306957752e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585306957752e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.6103585306957752e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.6103585306957752e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808796471298e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.5316808796471298e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808796471298e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5316808796471298e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.80610277574994e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.80610277574994e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.80610277574994e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.80610277574994e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.089799468159045e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.089799468159045e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.089799468159045e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.089799468159045e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.65220967010742e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.65220967010742e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.65220967010742e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.65220967010742e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851834553181e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.481851834553181e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851834553181e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.481851834553181e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.071480736654199e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.071480736654199e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.071480736654199e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.071480736654199e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.734622039095741e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.734622039095741e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.734622039095741e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.734622039095741e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.728843147605566e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.728843147605566e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.728843147605566e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.728843147605566e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.253224225758256e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.253224225758256e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.7696594523398868e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.7696594523398868e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.5443954295508303e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.5443954295508303e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.5443954295508303e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.5443954295508303e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.5443954295508303e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.5443954295508303e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.5443954295508303e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.5443954295508303e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.360956320553479e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.360956320553479e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.360956320553479e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.360956320553479e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-2.103215604843322e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.103215604843322e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.103215604843322e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.103215604843322e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220983405577e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.0111220983405577e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220983405577e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.0111220983405577e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9429468369963968e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.9429468369963968e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9429468369963968e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9429468369963968e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174774043453e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.6541174774043453e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174774043453e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.6541174774043453e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.5224930676404312e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.5224930676404312e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.5224930676404312e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.5224930676404312e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.5224930676404312e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930676404312e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930676404312e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.5224930676404312e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.2283337824653537e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337824653537e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.2283337824653537e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337824653537e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.988770288649928e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.988770288649928e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.988770288649928e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.988770288649928e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.867765104864533e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104864533e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104864533e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.867765104864533e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.189990975757645e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.189990975757645e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.175246207259889e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175246207259889e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.471647745009352e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.471647745009352e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.561447179709209e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.561447179709209e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.561447179709209e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.561447179709209e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.5233896781365e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.5233896781365e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.427323108940719e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.427323108940719e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.427323108940719e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.427323108940719e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.328139350441701e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350441701e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.328139350441701e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350441701e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.0868265656852585e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.0868265656852585e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.8882935959205144e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8882935959205144e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8882935959205144e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.8882935959205144e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.371328948139757e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.371328948139757e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.8394209155989254e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.8394209155989254e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-8.057446595508358e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.057446595508358e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (4.537178094945462e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.537178094945462e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.537178094945462e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (4.537178094945462e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (8.057446595508358e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.057446595508358e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (9.209350650276433e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350650276433e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350650276433e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.209350650276433e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.70357835554239e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.70357835554239e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.70357835554239e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.70357835554239e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.8394209155989254e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.8394209155989254e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.371328948139757e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.371328948139757e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (3.0868265656852585e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.0868265656852585e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (4.5233896781365e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.5233896781365e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (5.471647745009352e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.471647745009352e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.175246207259889e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175246207259889e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.189990975757645e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.189990975757645e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.3304731887807094e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.3304731887807094e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.3304731887807094e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.3304731887807094e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.6288532436420193e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.6288532436420193e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.6288532436420193e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.6288532436420193e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.689348951575511e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.689348951575511e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.689348951575511e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.689348951575511e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (2.7455184006858376e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.7455184006858376e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.7455184006858376e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.7455184006858376e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.7455184006858376e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.7455184006858376e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.7455184006858376e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.7455184006858376e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.211842019215942e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.211842019215942e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.211842019215942e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.211842019215942e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.211842019215942e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.211842019215942e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.211842019215942e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.211842019215942e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.3131455002295e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.3131455002295e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.3131455002295e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.3131455002295e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (3.334331289458342e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.334331289458342e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (4.183932559451153e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.183932559451153e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (7.735036880601781e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.735036880601781e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.0002463643756963844) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.0002463643756963844) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00044585351288404737) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.00044585351288404737) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.00044585351288404737) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.00044585351288404737) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.0005940221543005084) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005084) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005084) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005084) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005084) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005084) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.0005940221543005084) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005084) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.0008533856254125882) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.0008533856254125882) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.0008533856254125882) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.0008533856254125882) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.001043524653490759) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.001043524653490759) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.001043524653490759) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.001043524653490759) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.0012803060973496719) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.0012803060973496719) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.0012803060973496719) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.0012803060973496719) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.0013038004788126967) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (0.0013038004788126967) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (0.0013038004788126967) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (0.0013038004788126967) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (0.0039898414566193864) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (0.0039898414566193864) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (0.0039898414566193864) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (0.0039898414566193864) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (0.004311038507914325) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.004311038507914325) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.004311038507914325) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.004311038507914325) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.0046369766611826) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.0046369766611826) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.0046369766611826) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.0046369766611826) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.005114473831660339) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.005114473831660339) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.005114473831660339) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.005114473831660339) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.005114473831660339) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660339) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660339) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005114473831660339) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.005241535382803957) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (0.005241535382803957) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (0.005241535382803957) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (0.005241535382803957) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (0.0052626424730768) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.0052626424730768) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.0052626424730768) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.0052626424730768) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.005368659358109168) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005368659358109168) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005379937155839381) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.005379937155839381) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.005379937155839381) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.005379937155839381) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.005708495985960849) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.005708495985960849) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.005708495985960849) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.005708495985960849) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.005923798336561396) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.005923798336561396) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.007306759928832772) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.007306759928832772) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.00846997879102328) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.00846997879102328) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.009841749246962617) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.009841749246962617) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.011756013419819324) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.011756013419819324) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.014564531231172868) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.014564531231172868) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.014603704729162462) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.014603704729162462) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.015225630757226946) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.015225630757226946) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.01902824244384804) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.01902824244384804) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.019257505095251527) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.019257505095251527) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.04587947078129822) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04587947078129822) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.3693708936615633) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.3693708936615633) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.3693708936615633) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.3693708936615633) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.28164257767023543) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.28164257767023543) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.28164257767023543) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.28164257767023543) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.09065144207036493) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.09065144207036493) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.09065144207036493) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.09065144207036493) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863639) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.08684737589863639) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863639) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.08684737589863639) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.07635021950635137) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07635021950635137) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07635021950635137) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07635021950635137) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214141) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.06752385099214141) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214141) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.06752385099214141) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03560837898831293) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.03560837898831293) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03490334337366137) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03490334337366137) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03490334337366137) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03490334337366137) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883829978) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.024591860883829978) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883829978) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.024591860883829978) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.024282117354693197) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.024282117354693197) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.023145130929529172) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.023145130929529172) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.022528440196013085) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.022528440196013085) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019538050311315374) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.019538050311315374) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.019538050311315374) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.019538050311315374) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.0170915531558994) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.0170915531558994) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.0170915531558994) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.0170915531558994) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.010311482489831394) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831394) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831394) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010311482489831394) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.009841749246962617) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962617) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.009841749246962617) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962617) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209986) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209986) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209986) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008826368514209986) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008541996625454913) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454913) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454913) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454913) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454913) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454913) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454913) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008541996625454913) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00846997879102328) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.00846997879102328) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.00846997879102328) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.00846997879102328) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.004668620318776292) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.004668620318776292) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.0038764708993370414) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.0038764708993370414) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.0038040661717285455) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0038040661717285455) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0038040661717285455) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0038040661717285455) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178574) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178574) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0033566705638329282) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.0033566705638329282) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.0032675138544235654) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.0032675138544235654) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0021413612231015065) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.0021413612231015065) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.001727875394136986) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.001727875394136986) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.0016407548553124282) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0016407548553124282) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214168634) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0014528843214168634) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214168634) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0014528843214168634) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870896771024877) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007870896771024877) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0005192743499488541) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.0005192743499488541) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.0001940085702975652) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0001940085702975652) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303542334) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.00013840177303542334) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.141625221148039e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.141625221148039e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.141625221148039e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.141625221148039e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-5.071480736654199e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.071480736654199e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.1513463112824505e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.1513463112824505e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0882507115799094e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.0882507115799094e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.9885117063868435e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.9885117063868435e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.874299071511452e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.874299071511452e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.360956320553479e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.360956320553479e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.3002946561886403e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.3002946561886403e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.146837650836261e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.146837650836261e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.146837650836261e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.146837650836261e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.352332103806312e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.352332103806312e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.352332103806312e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.352332103806312e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.091637199604591e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637199604591e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637199604591e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637199604591e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637199604591e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637199604591e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637199604591e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.091637199604591e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.074305986545696e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.074305986545696e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.074305986545696e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.074305986545696e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.900128986894518e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.900128986894518e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.900128986894518e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900128986894518e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867765104864533e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.867765104864533e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.560692465446124e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692465446124e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692465446124e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692465446124e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692465446124e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692465446124e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692465446124e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.560692465446124e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.997018422215874e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.997018422215874e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.997018422215874e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.997018422215874e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.997018422215874e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.997018422215874e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.997018422215874e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.997018422215874e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.5682475214680915e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.5682475214680915e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.5682475214680915e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.5682475214680915e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393087580183e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393087580183e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393087580183e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393087580183e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393087580183e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393087580183e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.3767393087580183e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393087580183e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.8882935959205144e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.8882935959205144e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6863815468701696e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.6863815468701696e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.70357835554239e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.70357835554239e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.209350650276433e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.209350650276433e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773245741195e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773245741195e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773245741195e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773245741195e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773245741195e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773245741195e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.379773245741195e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773245741195e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.974225379986831e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.974225379986831e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.974225379986831e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.974225379986831e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0474716557630927e-08) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.0474716557630927e-08) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.0474716557630927e-08) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.0474716557630927e-08) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.209350650276433e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.209350650276433e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0717282184820608e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.0717282184820608e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.0717282184820608e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.0717282184820608e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.2004287494620357e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.2004287494620357e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.2004287494620357e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.2004287494620357e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.70357835554239e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.70357835554239e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.3120943054119657e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.3120943054119657e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.3120943054119657e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.3120943054119657e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.6863815468701696e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815468701696e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.8882935959205144e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.8882935959205144e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.0922506163727243e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506163727243e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.0922506163727243e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506163727243e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.0922506163727243e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506163727243e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.0922506163727243e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506163727243e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.444597854410838e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.444597854410838e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.444597854410838e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.444597854410838e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.684915095572819e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.684915095572819e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.684915095572819e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.684915095572819e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.24697442574195e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.24697442574195e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.24697442574195e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.24697442574195e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.24697442574195e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.24697442574195e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.24697442574195e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.24697442574195e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.867765104864533e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.867765104864533e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.3002946561886403e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.3002946561886403e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.360956320553479e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.360956320553479e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.874299071511452e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.874299071511452e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.883676576215157e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.883676576215157e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.947356011857944e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.947356011857944e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.947356011857944e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.947356011857944e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117063868435e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9885117063868435e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.0882507115799094e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.0882507115799094e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.1513463112824505e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.1513463112824505e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.8462016715590675e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.8462016715590675e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.8462016715590675e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.8462016715590675e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.071480736654199e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.071480736654199e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.105526722333282e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.105526722333282e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.105526722333282e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.105526722333282e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.146496327747708e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.146496327747708e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.146496327747708e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.146496327747708e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.159350502182531e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.159350502182531e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.159350502182531e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.159350502182531e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.427988656869548e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.427988656869548e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.427988656869548e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.427988656869548e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.935867718244788e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.935867718244788e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.935867718244788e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.935867718244788e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.2532733485675425e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.2532733485675425e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.979825793844734e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.979825793844734e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.979825793844734e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.979825793844734e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (4.2055484112143646e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.2055484112143646e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.2055484112143646e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.2055484112143646e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00013840177303542334) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.00013840177303542334) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.00018787053389556497) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.00018787053389556497) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.00018787053389556497) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.00018787053389556497) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0001940085702975652) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.0001940085702975652) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0002463643756963844) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0002463643756963844) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0002463643756963844) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0002463643756963844) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192743499488541) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.0005192743499488541) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.0007156734248910074) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007156734248910074) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0007156734248910074) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007156734248910074) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024877) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007870896771024877) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.001532483523073131) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.001532483523073131) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.001532483523073131) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.001532483523073131) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.0016407548553124282) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0016407548553124282) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.001727875394136986) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.001727875394136986) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.0032675138544235654) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.0032675138544235654) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.0033566705638329282) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.0033566705638329282) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.0034841573002178574) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0034841573002178574) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0038764708993370414) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.0038764708993370414) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.004668620318776292) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.004668620318776292) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278271) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.004767272188278271) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.004767272188278271) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278271) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.005286546538227124) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.005286546538227124) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.005286546538227124) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.005286546538227124) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.0054089544224101725) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.0054089544224101725) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.0054089544224101725) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.0054089544224101725) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.005923798336561396) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561396) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.005923798336561396) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561396) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.010715508469796613) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010715508469796613) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010715508469796613) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010715508469796613) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.010757563953908759) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010757563953908759) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010757563953908759) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010757563953908759) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.014603704729162462) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.014603704729162462) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.014603704729162462) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.014603704729162462) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.01929956057936367) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01929956057936367) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01929956057936367) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01929956057936367) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01929956057936367) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01929956057936367) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.01929956057936367) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01929956057936367) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.058591988733863246) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.058591988733863246) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.7759505275778846e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.7759505275778846e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.7759505275778846e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.7759505275778846e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.07165035181003086) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.07165035181003086) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.07165035181003092) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.07165035181003092) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.019257505095251527) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019257505095251527) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010311482489831394) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311482489831394) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008826368514209986) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209986) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005733569747311865) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311865) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311865) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311865) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311865) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311865) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311865) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311865) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676606) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005348051582676606) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005348051582676606) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676606) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.003804066171728546) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003804066171728546) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0029841661681220084) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.0029841661681220084) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.0029841661681220084) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.0029841661681220084) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.002446497155415974) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.002446497155415974) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.0022494124470940078) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0022494124470940078) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0022494124470940078) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0022494124470940078) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0021413612231015065) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.0021413612231015065) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.0018638942824587479) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587479) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587479) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587479) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587479) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587479) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0018638942824587479) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587479) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0016407548553124282) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124282) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0016407548553124282) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124282) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0012223378081538832) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538832) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538832) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538832) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538832) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538832) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538832) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0012223378081538832) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0010283292378563181) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0010283292378563181) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0010283292378563181) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0010283292378563181) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1463061453879802e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1463061453879802e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.874299071511452e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.874299071511452e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.874299071511452e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.874299071511452e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.3002946561886403e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.3002946561886403e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.3002946561886403e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.3002946561886403e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.0444941299253813e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.0444941299253813e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.0444941299253813e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.0444941299253813e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.956079231150257e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.956079231150257e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.956079231150257e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.956079231150257e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105515038300045e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105515038300045e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105515038300045e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105515038300045e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661347214023409e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661347214023409e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661347214023409e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661347214023409e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540341413953012e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540341413953012e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.189990975757645e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.189990975757645e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.876621659123952e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876621659123952e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876621659123952e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876621659123952e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175246207259889e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175246207259889e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.5233896781365e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.5233896781365e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0767325329921174e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.0767325329921174e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0767325329921174e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.0767325329921174e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.0134714590076554e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0134714590076554e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.904599885300801e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.904599885300801e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.904599885300801e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.904599885300801e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.66673175565232e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.66673175565232e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.66673175565232e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.66673175565232e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.850564192850212e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.850564192850212e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309313007143e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.6569309313007143e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309313007143e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.6569309313007143e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.850564192850212e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.850564192850212e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815468701696e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815468701696e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.6863815468701696e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815468701696e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714590076554e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.0134714590076554e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.5233896781365e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.5233896781365e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.67040239030837e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.67040239030837e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.67040239030837e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.67040239030837e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175246207259889e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175246207259889e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.189990975757645e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.189990975757645e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540341413953012e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540341413953012e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949476487004596e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949476487004596e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.7924939578183302e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939578183302e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939578183302e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.7924939578183302e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.883676576215157e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.883676576215157e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.9885117063868435e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117063868435e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117063868435e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117063868435e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.2532733485675425e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.2532733485675425e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4017109736062578e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4017109736062578e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4017109736062578e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4017109736062578e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603693880907e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.5809603693880907e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603693880907e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.5809603693880907e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0005192743499488541) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499488541) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.0005192743499488541) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499488541) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.0007870896771024877) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024877) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024877) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024877) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0011726348316441486) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0011726348316441486) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0011726348316441486) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0011726348316441486) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012366478019246025) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.0012366478019246025) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.0012366478019246025) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.0012366478019246025) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.002200964069500467) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002200964069500467) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002200964069500467) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002200964069500467) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798032) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798032) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798032) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798032) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798032) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798032) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002394972639798032) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798032) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002446497155415974) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.002446497155415974) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.003804066171728546) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.003804066171728546) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0038764708993370414) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.0038764708993370414) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.0038764708993370414) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.0038764708993370414) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.00422081397004661) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.00422081397004661) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.00422081397004661) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.00422081397004661) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.008826368514209986) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.008826368514209986) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.010311482489831394) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010311482489831394) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019257505095251527) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019257505095251527) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.058591988733863246) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.058591988733863246) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.3987009018670224e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.3987009018670224e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.3987009018670224e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.3987009018670224e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841573002178574) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841573002178574) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002984166168122008) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.002984166168122008) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.0001940085702975652) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0001940085702975652) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453879802e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453879802e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.7924939578183302e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7924939578183302e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540341413953012e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413953012e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540341413953012e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413953012e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.850564192850212e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.850564192850212e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.850564192850212e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.850564192850212e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714590076554e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.0134714590076554e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714590076554e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.0134714590076554e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.949476487004596e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.949476487004596e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.7924939578183302e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7924939578183302e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0001940085702975652) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0001940085702975652) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002984166168122008) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.002984166168122008) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.0034841573002178574) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0034841573002178574) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
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
  (-46.46390691340476) [I0]
+ (0.7829652070486298) [Z10]
+ (0.7829652070486304) [Z11]
+ (0.8084591005188306) [Z12]
+ (0.8084591005188306) [Z13]
+ (1.2034393391316256) [Z5]
+ (1.203439339131626) [Z4]
+ (1.3096876618626048) [Z6]
+ (1.309687661862605) [Z7]
+ (1.369352571169661) [Z9]
+ (1.3693525711696617) [Z8]
+ (1.6538938305469626) [Z3]
+ (1.6538938305469628) [Z2]
+ (12.41263071443703) [Z0]
+ (12.412630714437032) [Z1]
+ (-8.194104935323882e-06) [Y10 Y12]
+ (-8.194104935323882e-06) [X10 X12]
+ (-1.6021751137634176e-06) [Y2 Y4]
+ (-1.6021751137634176e-06) [X2 X4]
+ (5.929280541051331e-07) [Y4 Y6]
+ (5.929280541051331e-07) [X4 X6]
+ (7.765082322642915e-07) [Y3 Y5]
+ (7.765082322642915e-07) [X3 X5]
+ (1.8540565472905312e-06) [Y5 Y7]
+ (1.8540565472905312e-06) [X5 X7]
+ (7.954224533243681e-06) [Y11 Y13]
+ (7.954224533243681e-06) [X11 X13]
+ (0.003276965065747487) [Y1 Y3]
+ (0.003276965065747487) [X1 X3]
+ (0.10433061485307377) [Y0 Y2]
+ (0.10433061485307377) [X0 X2]
+ (0.11270381859122955) [Z10 Z12]
+ (0.11270381859122955) [Z11 Z13]
+ (0.11383573685308228) [Z4 Z12]
+ (0.11383573685308228) [Z5 Z13]
+ (0.11952441016891156) [Z6 Z10]
+ (0.11952441016891156) [Z7 Z11]
+ (0.12489977362992502) [Z4 Z10]
+ (0.12489977362992502) [Z5 Z11]
+ (0.12495799328197146) [Z2 Z4]
+ (0.12495799328197146) [Z3 Z5]
+ (0.12799492801395224) [Z2 Z10]
+ (0.12799492801395224) [Z3 Z11]
+ (0.13401737372654254) [Z6 Z12]
+ (0.13401737372654254) [Z7 Z13]
+ (0.13701191913060035) [Z4 Z6]
+ (0.13701191913060035) [Z5 Z7]
+ (0.1373494221015196) [Z6 Z11]
+ (0.1373494221015196) [Z7 Z10]
+ (0.13739112375008727) [Z2 Z6]
+ (0.13739112375008727) [Z3 Z7]
+ (0.13766859133610349) [Z8 Z10]
+ (0.13766859133610349) [Z9 Z11]
+ (0.14011294749717298) [Z2 Z12]
+ (0.14011294749717298) [Z3 Z13]
+ (0.14138903590154284) [Z10 Z13]
+ (0.14138903590154284) [Z11 Z12]
+ (0.14257991128704897) [Z4 Z11]
+ (0.14257991128704897) [Z5 Z10]
+ (0.14722930783254268) [Z8 Z11]
+ (0.14722930783254268) [Z9 Z10]
+ (0.14899426171385638) [Z4 Z7]
+ (0.14899426171385638) [Z5 Z6]
+ (0.14926347060040387) [Z10 Z11]
+ (0.1496069255724685) [Z4 Z8]
+ (0.1496069255724685) [Z5 Z9]
+ (0.14973497005431963) [Z8 Z12]
+ (0.14973497005431963) [Z9 Z13]
+ (0.15071405482285102) [Z2 Z8]
+ (0.15071405482285102) [Z3 Z9]
+ (0.15138342699117047) [Z6 Z13]
+ (0.15138342699117047) [Z7 Z12]
+ (0.15215040622655546) [Z4 Z13]
+ (0.15215040622655546) [Z5 Z12]
+ (0.15337959171059495) [Z2 Z11]
+ (0.15337959171059495) [Z3 Z10]
+ (0.15435760065315168) [Z12 Z13]
+ (0.15569017457263243) [Z2 Z13]
+ (0.15569017457263243) [Z3 Z12]
+ (0.15582280685854957) [Z8 Z13]
+ (0.15582280685854957) [Z9 Z12]
+ (0.15676384610165758) [Z4 Z9]
+ (0.15676384610165758) [Z5 Z8]
+ (0.1575530380435935) [Z4 Z5]
+ (0.160797550467014) [Z2 Z5]
+ (0.160797550467014) [Z3 Z4]
+ (0.16756669356160547) [Z6 Z8]
+ (0.16756669356160547) [Z7 Z9]
+ (0.1685349279464417) [Z2 Z7]
+ (0.1685349279464417) [Z3 Z6]
+ (0.18144009362922492) [Z6 Z9]
+ (0.18144009362922492) [Z7 Z8]
+ (0.18189081243738486) [Z2 Z3]
+ (0.1869081483103378) [Z2 Z9]
+ (0.1869081483103378) [Z3 Z8]
+ (0.19299700269855277) [Z0 Z10]
+ (0.19299700269855277) [Z1 Z11]
+ (0.1939257433496893) [Z6 Z7]
+ (0.1966174995973087) [Z0 Z4]
+ (0.1966174995973087) [Z1 Z5]
+ (0.19936332691273434) [Z0 Z5]
+ (0.19936332691273434) [Z1 Z4]
+ (0.20072843554590758) [Z0 Z11]
+ (0.20072843554590758) [Z1 Z10]
+ (0.21102681234285642) [Z0 Z12]
+ (0.21102681234285642) [Z1 Z13]
+ (0.21631059809668413) [Z0 Z13]
+ (0.21631059809668413) [Z1 Z12]
+ (0.22003977240257852) [Z8 Z9]
+ (0.23671071740415772) [Z0 Z2]
+ (0.23671071740415772) [Z1 Z3]
+ (0.24164696831735213) [Z0 Z6]
+ (0.24164696831735213) [Z1 Z7]
+ (0.24853517285966767) [Z0 Z7]
+ (0.24853517285966767) [Z1 Z6]
+ (0.25129435573116843) [Z0 Z3]
+ (0.25129435573116843) [Z1 Z2]
+ (0.27232518450358784) [Z0 Z8]
+ (0.27232518450358784) [Z1 Z9]
+ (0.278834545737663) [Z0 Z9]
+ (0.278834545737663) [Z1 Z8]
+ (1.1861764484126864) [Z0 Z1]
+ (-1.072274848293996e-05) [Y10 Z11 Y12]
+ (-1.072274848293996e-05) [X10 Z11 X12]
+ (-1.0722748482939956e-05) [Y11 Z12 Y13]
+ (-1.0722748482939956e-05) [X11 Z12 X13]
+ (3.886639520878969e-06) [Y2 Z3 Y4]
+ (3.886639520878969e-06) [X2 Z3 X4]
+ (3.886639520878969e-06) [Y3 Z4 Y5]
+ (3.886639520878969e-06) [X3 Z4 X5]
+ (1.2260276997588577e-05) [Y4 Z5 Y6]
+ (1.2260276997588577e-05) [X4 Z5 X6]
+ (1.2260276997588577e-05) [Y5 Z6 Y7]
+ (1.2260276997588577e-05) [X5 Z6 X7]
+ (0.1250703688397222) [Y0 Z1 Y2]
+ (0.1250703688397222) [X0 Z1 X2]
+ (0.12507036883972222) [Y1 Z2 Y3]
+ (0.12507036883972222) [X1 Z2 X3]
+ (-0.03831466937347315) [Y4 Y5 X12 X13]
+ (-0.03831466937347315) [X4 X5 Y12 Y13]
+ (-0.03619409348748678) [Y2 Y3 X8 X9]
+ (-0.03619409348748678) [X2 X3 Y8 Y9]
+ (-0.03583955718504256) [Y2 Y3 X4 X5]
+ (-0.03583955718504256) [X2 X3 Y4 Y5]
+ (-0.031143804196354423) [Y2 Y3 X6 X7]
+ (-0.031143804196354423) [X2 X3 Y6 Y7]
+ (-0.028685217310313302) [Y10 Y11 X12 X13]
+ (-0.028685217310313302) [X10 X11 Y12 Y13]
+ (-0.02599620626716795) [Y3 Z4 Z5 Y7]
+ (-0.02599620626716795) [X3 Z4 Z5 X7]
+ (-0.025384663696642704) [Y2 Y3 X10 X11]
+ (-0.025384663696642704) [X2 X3 Y10 Y11]
+ (-0.01902831871829257) [Y3 X4 X11 Y12]
+ (-0.01902831871829257) [X3 Y4 Y11 X12]
+ (-0.017825011932608013) [Y6 Y7 X10 X11]
+ (-0.017825011932608013) [X6 X7 Y10 Y11]
+ (-0.01768013765712395) [Y4 Y5 X10 X11]
+ (-0.01768013765712395) [X4 X5 Y10 Y11]
+ (-0.01736605326462793) [Y6 Y7 X12 X13]
+ (-0.01736605326462793) [X6 X7 Y12 Y13]
+ (-0.015577227075459443) [Y2 Y3 X12 X13]
+ (-0.015577227075459443) [X2 X3 Y12 Y13]
+ (-0.014583638327010718) [Y0 Y1 X2 X3]
+ (-0.014583638327010718) [X0 X1 Y2 Y3]
+ (-0.013873400067619456) [Y6 Y7 X8 X9]
+ (-0.013873400067619456) [X6 X7 Y8 Y9]
+ (-0.011982342583256063) [Y4 Y5 X6 X7]
+ (-0.011982342583256063) [X4 X5 Y6 Y7]
+ (-0.01128514461832123) [Y5 Y6 X11 X12]
+ (-0.01128514461832123) [X5 X6 Y11 Y12]
+ (-0.009560716496439168) [Y8 Y9 X10 X11]
+ (-0.009560716496439168) [X8 X9 Y10 Y11]
+ (-0.008125248410121527) [Y1 X2 X8 Y9]
+ (-0.008125248410121527) [Y1 Y2 Y8 Y9]
+ (-0.008125248410121527) [X1 X2 X8 X9]
+ (-0.008125248410121527) [X1 Y2 Y8 X9]
+ (-0.007731432847354803) [Y0 Y1 X10 X11]
+ (-0.007731432847354803) [X0 X1 Y10 Y11]
+ (-0.007156920529189085) [Y4 Y5 X8 X9]
+ (-0.007156920529189085) [X4 X5 Y8 Y9]
+ (-0.006888204542315533) [Y0 Y1 X6 X7]
+ (-0.006888204542315533) [X0 X1 Y6 Y7]
+ (-0.006509361234075112) [Y0 Y1 X8 X9]
+ (-0.006509361234075112) [X0 X1 Y8 Y9]
+ (-0.006087836804229955) [Y8 Y9 X12 X13]
+ (-0.006087836804229955) [X8 X9 Y12 Y13]
+ (-0.005283785753827709) [Y0 Y1 X12 X13]
+ (-0.005283785753827709) [X0 X1 Y12 Y13]
+ (-0.00514338238768913) [Y3 X4 X5 Y6]
+ (-0.00514338238768913) [X3 Y4 Y5 X6]
+ (-0.004684920226870667) [Y1 X2 X6 Y7]
+ (-0.004684920226870667) [Y1 Y2 Y6 Y7]
+ (-0.004684920226870667) [X1 X2 X6 X7]
+ (-0.004684920226870667) [X1 Y2 Y6 X7]
+ (-0.004575015188893613) [Y1 X2 X12 Y13]
+ (-0.004575015188893613) [Y1 Y2 Y12 Y13]
+ (-0.004575015188893613) [X1 X2 X12 X13]
+ (-0.004575015188893613) [X1 Y2 Y12 X13]
+ (-0.004424843668500905) [Y1 X2 X4 Y5]
+ (-0.004424843668500905) [Y1 Y2 Y4 Y5]
+ (-0.004424843668500905) [X1 X2 X4 X5]
+ (-0.004424843668500905) [X1 Y2 Y4 X5]
+ (-0.003479421729292206) [Y2 Z3 Z5 Y6]
+ (-0.003479421729292206) [X2 Z3 Z5 X6]
+ (-0.003479421729292206) [Y3 Z4 Z6 Y7]
+ (-0.003479421729292206) [X3 Z4 Z6 X7]
+ (-0.002745827315425665) [Y0 Y1 X4 X5]
+ (-0.002745827315425665) [X0 X1 Y4 Y5]
+ (-0.0017991930083833454) [Y1 X2 X10 Y11]
+ (-0.0017991930083833454) [Y1 Y2 Y10 Y11]
+ (-0.0017991930083833454) [X1 X2 X10 X11]
+ (-0.0017991930083833454) [X1 Y2 Y10 X11]
+ (-0.000292225672453627) [Y7 Y8 X9 X10]
+ (-0.000292225672453627) [X7 X8 Y9 Y10]
+ (-8.814793343927228e-06) [Y2 Z3 Y4 Z13]
+ (-8.814793343927228e-06) [X2 Z3 X4 Z13]
+ (-8.814793343927228e-06) [Y3 Z4 Y5 Z12]
+ (-8.814793343927228e-06) [X3 Z4 X5 Z12]
+ (-8.194104935323882e-06) [Z10 Y11 Z12 Y13]
+ (-8.194104935323882e-06) [Z10 X11 Z12 X13]
+ (-5.97417683152442e-06) [Y5 X6 X10 Y11]
+ (-5.97417683152442e-06) [Y5 Y6 Y10 Y11]
+ (-5.97417683152442e-06) [X5 X6 X10 X11]
+ (-5.97417683152442e-06) [X5 Y6 Y10 X11]
+ (-5.275783397604714e-06) [Y3 X4 X12 Y13]
+ (-5.275783397604714e-06) [Y3 Y4 Y12 Y13]
+ (-5.275783397604714e-06) [X3 X4 X12 X13]
+ (-5.275783397604714e-06) [X3 Y4 Y12 X13]
+ (-4.2818120088508604e-06) [Y4 Z5 Y6 Z11]
+ (-4.2818120088508604e-06) [X4 Z5 X6 Z11]
+ (-4.2818120088508604e-06) [Y5 Z6 Y7 Z10]
+ (-4.2818120088508604e-06) [X5 Z6 X7 Z10]
+ (-3.6945168754689076e-06) [Y4 X5 X11 Y12]
+ (-3.6945168754689076e-06) [Y4 Y5 Y11 Y12]
+ (-3.6945168754689076e-06) [X4 X5 X11 X12]
+ (-3.6945168754689076e-06) [X4 Y5 Y11 X12]
+ (-3.5390099463209965e-06) [Y2 Z3 Y4 Z12]
+ (-3.5390099463209965e-06) [X2 Z3 X4 Z12]
+ (-3.5390099463209965e-06) [Y3 Z4 Y5 Z13]
+ (-3.5390099463209965e-06) [X3 Z4 X5 Z13]
+ (-3.1173664008472677e-06) [Y0 Z2 Z3 Y4]
+ (-3.1173664008472677e-06) [X0 Z2 Z3 X4]
+ (-2.890929961728468e-06) [Z6 Y11 Z12 Y13]
+ (-2.890929961728468e-06) [Z6 X11 Z12 X13]
+ (-2.890929961728468e-06) [Z7 Y10 Z11 Y12]
+ (-2.890929961728468e-06) [Z7 X10 Z11 X12]
+ (-2.1777329991894278e-06) [Z0 Y10 Z11 Y12]
+ (-2.1777329991894278e-06) [Z0 X10 Z11 X12]
+ (-2.1777329991894278e-06) [Z1 Y11 Z12 Y13]
+ (-2.1777329991894278e-06) [Z1 X11 Z12 X13]
+ (-1.8551374760824267e-06) [Z6 Y10 Z11 Y12]
+ (-1.8551374760824267e-06) [Z6 X10 Z11 X12]
+ (-1.8551374760824267e-06) [Z7 Y11 Z12 Y13]
+ (-1.8551374760824267e-06) [Z7 X11 Z12 X13]
+ (-1.816367393333608e-06) [Z4 Y11 Z12 Y13]
+ (-1.816367393333608e-06) [Z4 X11 Z12 X13]
+ (-1.816367393333608e-06) [Z5 Y10 Z11 Y12]
+ (-1.816367393333608e-06) [Z5 X10 Z11 X12]
+ (-1.6149608019507343e-06) [Z0 Y11 Z12 Y13]
+ (-1.6149608019507343e-06) [Z0 X11 Z12 X13]
+ (-1.6149608019507343e-06) [Z1 Y10 Z11 Y12]
+ (-1.6149608019507343e-06) [Z1 X10 Z11 X12]
+ (-1.6021751137634176e-06) [Z2 Y3 Z4 Y5]
+ (-1.6021751137634176e-06) [Z2 X3 Z4 X5]
+ (-1.5973397324235e-06) [Z8 Y10 Z11 Y12]
+ (-1.5973397324235e-06) [Z8 X10 Z11 X12]
+ (-1.5973397324235e-06) [Z9 Y11 Z12 Y13]
+ (-1.5973397324235e-06) [Z9 X11 Z12 X13]
+ (-1.0632255252581227e-06) [Z2 Y10 Z11 Y12]
+ (-1.0632255252581227e-06) [Z2 X10 Z11 X12]
+ (-1.0632255252581227e-06) [Z3 Y11 Z12 Y13]
+ (-1.0632255252581227e-06) [Z3 X11 Z12 X13]
+ (-1.0357924856447404e-06) [Y6 X7 X11 Y12]
+ (-1.0357924856447404e-06) [Y6 Y7 Y11 Y12]
+ (-1.0357924856447404e-06) [X6 X7 X11 X12]
+ (-1.0357924856447404e-06) [X6 Y7 Y11 X12]
+ (-9.344970130636954e-07) [Z8 Y11 Z12 Y13]
+ (-9.344970130636954e-07) [Z8 X11 Z12 X13]
+ (-9.344970130636954e-07) [Z9 Y10 Z11 Y12]
+ (-9.344970130636954e-07) [Z9 X10 Z11 X12]
+ (-5.471606001111822e-07) [Y1 Y2 X11 X12]
+ (-5.471606001111822e-07) [X1 X2 Y11 Y12]
+ (-3.6060693002437203e-07) [Y0 Z1 Z2 Y4]
+ (-3.6060693002437203e-07) [X0 Z1 Z2 X4]
+ (-3.6060693002437203e-07) [Y1 Z3 Z4 Y5]
+ (-3.6060693002437203e-07) [X1 Z3 Z4 X5]
+ (-2.5941461654309644e-07) [Y2 Z3 Y4 Z6]
+ (-2.5941461654309644e-07) [X2 Z3 X4 Z6]
+ (-2.5941461654309644e-07) [Y3 Z4 Y5 Z7]
+ (-2.5941461654309644e-07) [X3 Z4 X5 Z7]
+ (-2.1872304213360215e-07) [Y2 Z3 Y4 Z8]
+ (-2.1872304213360215e-07) [X2 Z3 X4 Z8]
+ (-2.1872304213360215e-07) [Y3 Z4 Y5 Z9]
+ (-2.1872304213360215e-07) [X3 Z4 X5 Z9]
+ (-1.9332121409916087e-07) [Y1 Y2 X3 X4]
+ (-1.9332121409916087e-07) [X1 X2 Y3 Y4]
+ (-1.6728571592538903e-07) [Y0 Z1 Z3 Y4]
+ (-1.6728571592538903e-07) [X0 Z1 Z3 X4]
+ (-1.6728571592538903e-07) [Y1 Z2 Z4 Y5]
+ (-1.6728571592538903e-07) [X1 Z2 Z4 X5]
+ (-3.226104524722576e-09) [Y1 Y2 X5 X6]
+ (-3.226104524722576e-09) [X1 X2 Y5 Y6]
+ (3.226104524722576e-09) [Y1 X2 X5 Y6]
+ (3.226104524722576e-09) [X1 Y2 Y5 X6]
+ (3.228405355375219e-08) [Y4 Z5 Y6 Z12]
+ (3.228405355375219e-08) [X4 Z5 X6 Z12]
+ (3.228405355375219e-08) [Y5 Z6 Y7 Z13]
+ (3.228405355375219e-08) [X5 Z6 X7 Z13]
+ (1.291945825221344e-07) [Y1 Z2 Z3 Y5]
+ (1.291945825221344e-07) [X1 Z2 Z3 X5]
+ (1.9332121409916087e-07) [Y1 X2 X3 Y4]
+ (1.9332121409916087e-07) [X1 Y2 Y3 X4]
+ (2.1989637490269485e-07) [Y2 X3 X5 Y6]
+ (2.1989637490269485e-07) [Y2 Y3 Y5 Y6]
+ (2.1989637490269485e-07) [X2 X3 X5 X6]
+ (2.1989637490269485e-07) [X2 Y3 Y5 X6]
+ (2.4472648242874843e-07) [Y0 X1 X5 Y6]
+ (2.4472648242874843e-07) [Y0 Y1 Y5 Y6]
+ (2.4472648242874843e-07) [X0 X1 X5 X6]
+ (2.4472648242874843e-07) [X0 Y1 Y5 X6]
+ (3.5706354981050476e-07) [Y0 X1 X3 Y4]
+ (3.5706354981050476e-07) [Y0 Y1 Y3 Y4]
+ (3.5706354981050476e-07) [X0 X1 X3 X4]
+ (3.5706354981050476e-07) [X0 Y1 Y3 X4]
+ (4.837953358557024e-07) [Y5 X6 X8 Y9]
+ (4.837953358557024e-07) [Y5 Y6 Y8 Y9]
+ (4.837953358557024e-07) [X5 X6 X8 X9]
+ (4.837953358557024e-07) [X5 Y6 Y8 X9]
+ (5.471606001111822e-07) [Y1 X2 X11 Y12]
+ (5.471606001111822e-07) [X1 Y2 Y11 X12]
+ (5.627721972384872e-07) [Y0 X1 X11 Y12]
+ (5.627721972384872e-07) [Y0 Y1 Y11 Y12]
+ (5.627721972384872e-07) [X0 X1 X11 X12]
+ (5.627721972384872e-07) [X0 Y1 Y11 X12]
+ (5.769436554744697e-07) [Y2 Z3 Y4 Z9]
+ (5.769436554744697e-07) [X2 Z3 X4 Z9]
+ (5.769436554744697e-07) [Y3 Z4 Y5 Z8]
+ (5.769436554744697e-07) [X3 Z4 X5 Z8]
+ (5.929280541051331e-07) [Z4 Y5 Z6 Y7]
+ (5.929280541051331e-07) [Z4 X5 Z6 X7]
+ (6.628427193598047e-07) [Y8 X9 X11 Y12]
+ (6.628427193598047e-07) [Y8 Y9 Y11 Y12]
+ (6.628427193598047e-07) [X8 X9 X11 X12]
+ (6.628427193598047e-07) [X8 Y9 Y11 X12]
+ (7.765082322642915e-07) [Y2 Z3 Y4 Z5]
+ (7.765082322642915e-07) [X2 Z3 X4 Z5]
+ (7.956666976080719e-07) [Y3 X4 X8 Y9]
+ (7.956666976080719e-07) [Y3 Y4 Y8 Y9]
+ (7.956666976080719e-07) [X3 X4 X8 X9]
+ (7.956666976080719e-07) [X3 Y4 Y8 X9]
+ (8.336695503692684e-07) [Z0 Y2 Z3 Y4]
+ (8.336695503692684e-07) [Z0 X2 Z3 X4]
+ (8.336695503692684e-07) [Z1 Y3 Z4 Y5]
+ (8.336695503692684e-07) [Z1 X3 Z4 X5]
+ (9.509134558930614e-07) [Z2 Y4 Z5 Y6]
+ (9.509134558930614e-07) [Z2 X4 Z5 X6]
+ (9.509134558930614e-07) [Z3 Y5 Z6 Y7]
+ (9.509134558930614e-07) [Z3 X5 Z6 X7]
+ (1.109412473734861e-06) [Z2 Y11 Z12 Y13]
+ (1.109412473734861e-06) [Z2 X11 Z12 X13]
+ (1.109412473734861e-06) [Z3 Y10 Z11 Y12]
+ (1.109412473734861e-06) [Z3 X10 Z11 X12]
+ (1.1708098307937505e-06) [Z2 Y5 Z6 Y7]
+ (1.1708098307937505e-06) [Z2 X5 Z6 X7]
+ (1.1708098307937505e-06) [Z3 Y4 Z5 Y6]
+ (1.1708098307937505e-06) [Z3 X4 Z5 X6]
+ (1.190733100179833e-06) [Z0 Y3 Z4 Y5]
+ (1.190733100179833e-06) [Z0 X3 Z4 X5]
+ (1.190733100179833e-06) [Z1 Y2 Z3 Y4]
+ (1.190733100179833e-06) [Z1 X2 Z3 X4]
+ (1.19539208018576e-06) [Y2 Z3 Y4 Z7]
+ (1.19539208018576e-06) [X2 Z3 X4 Z7]
+ (1.19539208018576e-06) [Y3 Z4 Y5 Z6]
+ (1.19539208018576e-06) [X3 Z4 X5 Z6]
+ (1.3980243009810163e-06) [Y4 Z5 Y6 Z8]
+ (1.3980243009810163e-06) [X4 Z5 X6 Z8]
+ (1.3980243009810163e-06) [Y5 Z6 Y7 Z9]
+ (1.3980243009810163e-06) [X5 Z6 X7 Z9]
+ (1.4548066967282058e-06) [Y3 X4 X6 Y7]
+ (1.4548066967282058e-06) [Y3 Y4 Y6 Y7]
+ (1.4548066967282058e-06) [X3 X4 X6 X7]
+ (1.4548066967282058e-06) [X3 Y4 Y6 X7]
+ (1.6923648226744269e-06) [Y4 Z5 Y6 Z10]
+ (1.6923648226744269e-06) [X4 Z5 X6 Z10]
+ (1.6923648226744269e-06) [Y5 Z6 Y7 Z11]
+ (1.6923648226744269e-06) [X5 Z6 X7 Z11]
+ (1.8540565472905312e-06) [Y4 Z5 Y6 Z7]
+ (1.8540565472905312e-06) [X4 Z5 X6 Z7]
+ (1.8781494821370343e-06) [Z4 Y10 Z11 Y12]
+ (1.8781494821370343e-06) [Z4 X10 Z11 X12]
+ (1.8781494821370343e-06) [Z5 Y11 Z12 Y13]
+ (1.8781494821370343e-06) [Z5 X11 Z12 X13]
+ (1.8818196368367186e-06) [Y4 Z5 Y6 Z9]
+ (1.8818196368367186e-06) [X4 Z5 X6 Z9]
+ (1.8818196368367186e-06) [Y5 Z6 Y7 Z8]
+ (1.8818196368367186e-06) [X5 Z6 X7 Z8]
+ (2.1726379989954773e-06) [Y2 X3 X11 Y12]
+ (2.1726379989954773e-06) [Y2 Y3 Y11 Y12]
+ (2.1726379989954773e-06) [X2 X3 X11 X12]
+ (2.1726379989954773e-06) [X2 Y3 Y11 X12]
+ (3.0992966620178958e-06) [Z0 Y4 Z5 Y6]
+ (3.0992966620178958e-06) [Z0 X4 Z5 X6]
+ (3.0992966620178958e-06) [Z1 Y5 Z6 Y7]
+ (3.0992966620178958e-06) [Z1 X5 Z6 X7]
+ (3.1585593452303565e-06) [Y2 Z3 Y4 Z10]
+ (3.1585593452303565e-06) [X2 Z3 X4 Z10]
+ (3.1585593452303565e-06) [Y3 Z4 Y5 Z11]
+ (3.1585593452303565e-06) [X3 Z4 X5 Z11]
+ (3.344023144446711e-06) [Z0 Y5 Z6 Y7]
+ (3.344023144446711e-06) [Z0 X5 Z6 X7]
+ (3.344023144446711e-06) [Z1 Y4 Z5 Y6]
+ (3.344023144446711e-06) [Z1 X4 Z5 X6]
+ (4.556473705684232e-06) [Y5 X6 X12 Y13]
+ (4.556473705684232e-06) [Y5 Y6 Y12 Y13]
+ (4.556473705684232e-06) [X5 X6 X12 X13]
+ (4.556473705684232e-06) [X5 Y6 Y12 X13]
+ (4.588757759235382e-06) [Y4 Z5 Y6 Z13]
+ (4.588757759235382e-06) [X4 Z5 X6 Z13]
+ (4.588757759235382e-06) [Y5 Z6 Y7 Z12]
+ (4.588757759235382e-06) [X5 Z6 X7 Z12]
+ (4.642978932872495e-06) [Y3 X4 X10 Y11]
+ (4.642978932872495e-06) [Y3 Y4 Y10 Y11]
+ (4.642978932872495e-06) [X3 X4 X10 X11]
+ (4.642978932872495e-06) [X3 Y4 Y10 X11]
+ (7.801538278099599e-06) [Y2 Z3 Y4 Z11]
+ (7.801538278099599e-06) [X2 Z3 X4 Z11]
+ (7.801538278099599e-06) [Y3 Z4 Y5 Z10]
+ (7.801538278099599e-06) [X3 Z4 X5 Z10]
+ (7.954224533243681e-06) [Y10 Z11 Y12 Z13]
+ (7.954224533243681e-06) [X10 Z11 X12 Z13]
+ (0.000292225672453627) [Y7 X8 X9 Y10]
+ (0.000292225672453627) [X7 Y8 Y9 X10]
+ (0.0004957972886071685) [Y2 Z4 Z5 Y6]
+ (0.0004957972886071685) [X2 Z4 Z5 X6]
+ (0.0011058984808981914) [Y0 Z1 Y2 Z5]
+ (0.0011058984808981914) [X0 Z1 X2 Z5]
+ (0.0011058984808981914) [Y1 Z2 Y3 Z4]
+ (0.0011058984808981914) [X1 Z2 X3 Z4]
+ (0.001663960658396925) [Y2 Z3 Z4 Y6]
+ (0.001663960658396925) [X2 Z3 Z4 X6]
+ (0.001663960658396925) [Y3 Z5 Z6 Y7]
+ (0.001663960658396925) [X3 Z5 Z6 X7]
+ (0.0017560659628753548) [Y0 Z1 Y2 Z11]
+ (0.0017560659628753548) [X0 Z1 X2 Z11]
+ (0.0017560659628753548) [Y1 Z2 Y3 Z10]
+ (0.0017560659628753548) [X1 Z2 X3 Z10]
+ (0.002326234847608931) [Y0 Z1 Y2 Z13]
+ (0.002326234847608931) [X0 Z1 X2 Z13]
+ (0.002326234847608931) [Y1 Z2 Y3 Z12]
+ (0.002326234847608931) [X1 Z2 X3 Z12]
+ (0.002745827315425665) [Y0 X1 X4 Y5]
+ (0.002745827315425665) [X0 Y1 Y4 X5]
+ (0.0029297682785811158) [Y0 Z1 Y2 Z9]
+ (0.0029297682785811158) [X0 Z1 X2 Z9]
+ (0.0029297682785811158) [Y1 Z2 Y3 Z8]
+ (0.0029297682785811158) [X1 Z2 X3 Z8]
+ (0.003276965065747487) [Y0 Z1 Y2 Z3]
+ (0.003276965065747487) [X0 Z1 X2 Z3]
+ (0.0033476264706883887) [Y0 Z1 Y2 Z7]
+ (0.0033476264706883887) [X0 Z1 X2 Z7]
+ (0.0033476264706883887) [Y1 Z2 Y3 Z6]
+ (0.0033476264706883887) [X1 Z2 X3 Z6]
+ (0.0035552589712586997) [Y0 Z1 Y2 Z10]
+ (0.0035552589712586997) [X0 Z1 X2 Z10]
+ (0.0035552589712586997) [Y1 Z2 Y3 Z11]
+ (0.0035552589712586997) [X1 Z2 X3 Z11]
+ (0.00514338238768913) [Y3 Y4 X5 X6]
+ (0.00514338238768913) [X3 X4 Y5 Y6]
+ (0.005283785753827709) [Y0 X1 X12 Y13]
+ (0.005283785753827709) [X0 Y1 Y12 X13]
+ (0.0055307421493990955) [Y0 Z1 Y2 Z4]
+ (0.0055307421493990955) [X0 Z1 X2 Z4]
+ (0.0055307421493990955) [Y1 Z2 Y3 Z5]
+ (0.0055307421493990955) [X1 Z2 X3 Z5]
+ (0.006087836804229955) [Y8 X9 X12 Y13]
+ (0.006087836804229955) [X8 Y9 Y12 X13]
+ (0.006509361234075112) [Y0 X1 X8 Y9]
+ (0.006509361234075112) [X0 Y1 Y8 X9]
+ (0.006888204542315533) [Y0 X1 X6 Y7]
+ (0.006888204542315533) [X0 Y1 Y6 X7]
+ (0.006901250036502545) [Y0 Z1 Y2 Z12]
+ (0.006901250036502545) [X0 Z1 X2 Z12]
+ (0.006901250036502545) [Y1 Z2 Y3 Z13]
+ (0.006901250036502545) [X1 Z2 X3 Z13]
+ (0.007156920529189085) [Y4 X5 X8 Y9]
+ (0.007156920529189085) [X4 Y5 Y8 X9]
+ (0.007731432847354803) [Y0 X1 X10 Y11]
+ (0.007731432847354803) [X0 Y1 Y10 X11]
+ (0.008032546697559055) [Y0 Z1 Y2 Z6]
+ (0.008032546697559055) [X0 Z1 X2 Z6]
+ (0.008032546697559055) [Y1 Z2 Y3 Z7]
+ (0.008032546697559055) [X1 Z2 X3 Z7]
+ (0.009560716496439168) [Y8 X9 X10 Y11]
+ (0.009560716496439168) [X8 Y9 Y10 X11]
+ (0.011055016688702641) [Y0 Z1 Y2 Z8]
+ (0.011055016688702641) [X0 Z1 X2 Z8]
+ (0.011055016688702641) [Y1 Z2 Y3 Z9]
+ (0.011055016688702641) [X1 Z2 X3 Z9]
+ (0.01128514461832123) [Y5 X6 X11 Y12]
+ (0.01128514461832123) [X5 Y6 Y11 X12]
+ (0.011307208029968412) [Y7 Z8 Z9 Y11]
+ (0.011307208029968412) [X7 Z8 Z9 X11]
+ (0.011982342583256063) [Y4 X5 X6 Y7]
+ (0.011982342583256063) [X4 Y5 Y6 X7]
+ (0.013873400067619456) [Y6 X7 X8 Y9]
+ (0.013873400067619456) [X6 Y7 Y8 X9]
+ (0.014583638327010718) [Y0 X1 X2 Y3]
+ (0.014583638327010718) [X0 Y1 Y2 X3]
+ (0.015577227075459443) [Y2 X3 X12 Y13]
+ (0.015577227075459443) [X2 Y3 Y12 X13]
+ (0.01736605326462793) [Y6 X7 X12 Y13]
+ (0.01736605326462793) [X6 Y7 Y12 X13]
+ (0.01768013765712395) [Y4 X5 X10 Y11]
+ (0.01768013765712395) [X4 Y5 Y10 X11]
+ (0.017825011932608013) [Y6 X7 X10 Y11]
+ (0.017825011932608013) [X6 Y7 Y10 X11]
+ (0.01902831871829257) [Y3 Y4 X11 X12]
+ (0.01902831871829257) [X3 X4 Y11 Y12]
+ (0.025384663696642704) [Y2 X3 X10 Y11]
+ (0.025384663696642704) [X2 Y3 Y10 X11]
+ (0.028685217310313302) [Y10 X11 X12 Y13]
+ (0.028685217310313302) [X10 Y11 Y12 X13]
+ (0.02981229960105299) [Y6 Z7 Z8 Y10]
+ (0.02981229960105299) [X6 Z7 Z8 X10]
+ (0.02981229960105299) [Y7 Z9 Z10 Y11]
+ (0.02981229960105299) [X7 Z9 Z10 X11]
+ (0.030104525273506617) [Y6 Z7 Z9 Y10]
+ (0.030104525273506617) [X6 Z7 Z9 X10]
+ (0.030104525273506617) [Y7 Z8 Z10 Y11]
+ (0.030104525273506617) [X7 Z8 Z10 X11]
+ (0.03078744071852189) [Y6 Z8 Z9 Y10]
+ (0.03078744071852189) [X6 Z8 Z9 X10]
+ (0.031143804196354423) [Y2 X3 X6 Y7]
+ (0.031143804196354423) [X2 Y3 Y6 X7]
+ (0.03583955718504256) [Y2 X3 X4 Y5]
+ (0.03583955718504256) [X2 Y3 Y4 X5]
+ (0.03619409348748678) [Y2 X3 X8 Y9]
+ (0.03619409348748678) [X2 Y3 Y8 X9]
+ (0.03831466937347315) [Y4 X5 X12 Y13]
+ (0.03831466937347315) [X4 Y5 Y12 X13]
+ (0.10433061485307377) [Z0 Y1 Z2 Y3]
+ (0.10433061485307377) [Z0 X1 Z2 X3]
+ (-0.12133242248361376) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133242248361376) [X2 Z3 Z4 Z5 X6]
+ (-0.12133242248361376) [Y3 Z4 Z5 Z6 Y7]
+ (-0.12133242248361376) [X3 Z4 Z5 Z6 X7]
+ (-3.2041422455258916e-06) [Y0 Z1 Z2 Z3 Y4]
+ (-3.2041422455258916e-06) [X0 Z1 Z2 Z3 X4]
+ (-3.2041422455258916e-06) [Y1 Z2 Z3 Z4 Y5]
+ (-3.2041422455258916e-06) [X1 Z2 Z3 Z4 X5]
+ (0.22847946311004724) [Y6 Z7 Z8 Z9 Y10]
+ (0.22847946311004724) [X6 Z7 Z8 Z9 X10]
+ (0.22847946311004724) [Y7 Z8 Z9 Z10 Y11]
+ (0.22847946311004724) [X7 Z8 Z9 Z10 X11]
+ (-0.03276748589565045) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.03276748589565045) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.03276748589565045) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.03276748589565045) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.02711487858042988) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.02711487858042988) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.02711487858042988) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.02711487858042988) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.02599620626716795) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.02599620626716795) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.024353136084517914) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (-0.024353136084517914) [Y2 Z3 Y4 X11 Z12 X13]
+ (-0.024353136084517914) [X2 Z3 X4 Y11 Z12 Y13]
+ (-0.024353136084517914) [X2 Z3 X4 X11 Z12 X13]
+ (-0.024353136084517914) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (-0.024353136084517914) [Y3 Z4 Y5 X10 Z11 X12]
+ (-0.024353136084517914) [X3 Z4 X5 Y10 Z11 Y12]
+ (-0.024353136084517914) [X3 Z4 X5 X10 Z11 X12]
+ (-0.020175824956985194) [Y4 Z5 Z6 X7 X11 Y12]
+ (-0.020175824956985194) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (-0.020175824956985194) [X4 Z5 Z6 X7 X11 X12]
+ (-0.020175824956985194) [X4 Z5 Z6 Y7 Y11 X12]
+ (-0.020175824956985194) [Y5 X6 X10 Z11 Z12 Y13]
+ (-0.020175824956985194) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (-0.020175824956985194) [X5 X6 X10 Z11 Z12 X13]
+ (-0.020175824956985194) [X5 Y6 Y10 Z11 Z12 X13]
+ (-0.01756111645272811) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.01756111645272811) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.01756111645272811) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.01756111645272811) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.015588277865119689) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (-0.015588277865119689) [X2 Z3 X4 X10 Z11 X12]
+ (-0.015588277865119689) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (-0.015588277865119689) [X3 Z4 X5 X11 Z12 X13]
+ (-0.014564473640826997) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.014564473640826997) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.014564473640826997) [X7 Z8 Z9 X10 X12 X13]
+ (-0.014564473640826997) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.01175599524038182) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.01175599524038182) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.01175599524038182) [X3 Z4 Z5 X6 X8 X9]
+ (-0.01175599524038182) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.010263460498894358) [Y2 Z3 X4 X10 Z11 Y12]
+ (-0.010263460498894358) [X2 Z3 Y4 Y10 Z11 X12]
+ (-0.010263460498894358) [Y3 Z4 X5 X11 Z12 Y13]
+ (-0.010263460498894358) [X3 Z4 Y5 Y11 Z12 X13]
+ (-0.008890680338663961) [Y4 Z5 X6 X10 Z11 Y12]
+ (-0.008890680338663961) [X4 Z5 Y6 Y10 Z11 X12]
+ (-0.008890680338663961) [Y5 Z6 X7 X11 Z12 Y13]
+ (-0.008890680338663961) [X5 Z6 Y7 Y11 Z12 X13]
+ (-0.008125248410121527) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.008125248410121527) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.007960839634223161) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (-0.007960839634223161) [X4 Z5 X6 X10 Z11 X12]
+ (-0.007960839634223161) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (-0.007960839634223161) [X5 Z6 X7 X11 Z12 X13]
+ (-0.007306763969601731) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.007306763969601731) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.007306763969601731) [X4 X5 X7 Z8 Z9 X10]
+ (-0.007306763969601731) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.005805121212346289) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.005805121212346289) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.005805121212346289) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.005805121212346289) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.005652607315220576) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.005652607315220576) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.005652607315220576) [X0 X1 X3 Z4 Z5 X6]
+ (-0.005652607315220576) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.00532481736622534) [Y2 Z3 Y4 X10 Z11 X12]
+ (-0.00532481736622534) [X2 Z3 X4 Y10 Z11 Y12]
+ (-0.00532481736622534) [Y3 Z4 Y5 X11 Z12 X13]
+ (-0.00532481736622534) [X3 Z4 X5 Y11 Z12 Y13]
+ (-0.00514338238768913) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.00514338238768913) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.00514338238768913) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.00514338238768913) [X2 Z3 X4 X5 Z6 X7]
+ (-0.004684920226870667) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.004684920226870667) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668615266017398) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.004668615266017398) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.004575015188893614) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.004575015188893614) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.004424843668500904) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.004424843668500904) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.0041588307164017995) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.0041588307164017995) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.0041588307164017995) [X3 Z4 Z5 X6 X12 X13]
+ (-0.0041588307164017995) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.003493800371491816) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.003493800371491816) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.003493800371491816) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.003493800371491816) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.0027790407628820344) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.0027790407628820344) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.0022939556230641665) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.0022939556230641665) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.0017991930083833454) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.0017991930083833454) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.0017278745823321462) [Y1 Z2 Z3 Y4 X11 X12]
+ (-0.0017278745823321462) [X1 Z2 Z3 X4 Y11 Y12]
+ (-0.0008533831051011209) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.0008533831051011209) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.0008144692870356062) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.0008144692870356062) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.0008144692870356062) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.0008144692870356062) [X3 Z4 Z5 Z6 X7 Z11]
+ (-7.735870605807998e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.735870605807998e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.735870605807998e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.735870605807998e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-6.5242044745596195e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.5242044745596195e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.5242044745596195e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.5242044745596195e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-5.97417683152442e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (-5.97417683152442e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (-5.275783397604714e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (-5.275783397604714e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (-4.642978932872495e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (-4.642978932872495e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (-4.556473705684232e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (-4.556473705684232e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (-3.7695835792481147e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.7695835792481147e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.6945168754689076e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.6945168754689076e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.6102422154620168e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.6102422154620168e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.6102422154620168e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.6102422154620168e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.3342618698580656e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (-3.3342618698580656e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (-3.3130170603318937e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.3130170603318937e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.2774382501292096e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.2774382501292096e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.2774382501292096e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.2774382501292096e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.211187414228593e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.211187414228593e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.211187414228593e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.211187414228593e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.151295941433859e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (-3.151295941433859e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (-3.1173664008472677e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (-3.1173664008472677e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (-3.088245679912444e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.088245679912444e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.1726379989954773e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.1726379989954773e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.4548066967282058e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (-1.4548066967282058e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (-1.3304568444028183e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (-1.3304568444028183e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (-1.2393113771137017e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (-1.2393113771137017e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (-1.2393113771137017e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (-1.2393113771137017e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (-1.2282691317143466e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (-1.2282691317143466e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (-1.0357924856447404e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.0357924856447404e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-9.306342880644744e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (-9.306342880644744e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (-9.306342880644744e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (-9.306342880644744e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (-7.956666976080719e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (-7.956666976080719e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (-6.628427193598047e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.628427193598047e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.579258912269261e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.579258912269261e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.579258912269261e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.579258912269261e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.395302326619138e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.395302326619138e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.395302326619138e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.395302326619138e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.627721972384872e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.627721972384872e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.287649415457837e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.287649415457837e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.287649415457837e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.287649415457837e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.287649415457837e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.287649415457837e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.287649415457837e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.287649415457837e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.837953358557024e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (-4.837953358557024e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (-3.5706354981050476e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (-3.5706354981050476e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (-3.328039653328072e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.328039653328072e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.236183428433297e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (-3.236183428433297e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (-3.236183428433297e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (-3.236183428433297e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (-2.4472648242874843e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (-2.4472648242874843e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (-2.1989637490269485e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (-2.1989637490269485e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (-1.8290428543575839e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (-1.8290428543575839e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (-1.8290428543575839e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (-1.8290428543575839e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (-1.1076529111595399e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.1076529111595399e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.1076529111595399e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.1076529111595399e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.1076529111595399e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.1076529111595399e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.1076529111595399e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.1076529111595399e-07) [X1 Z2 X3 X10 Z11 X12]
+ (-8.649129631237444e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (-8.649129631237444e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (-8.649129631237444e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (-8.649129631237444e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (-8.057465217180758e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (-8.057465217180758e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (-8.057465217180758e-08) [X1 Z2 Z3 X4 X10 X11]
+ (-8.057465217180758e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (1.035150942102546e-09) [Y0 Z1 Z2 Z3 Y4 Z7]
+ (1.035150942102546e-09) [X0 Z1 Z2 Z3 X4 Z7]
+ (1.035150942102546e-09) [Y1 Z2 Z3 Z4 Y5 Z6]
+ (1.035150942102546e-09) [X1 Z2 Z3 Z4 X5 Z6]
+ (1.8395658564934373e-08) [Y0 Z1 X2 X10 Z11 Y12]
+ (1.8395658564934373e-08) [X0 Z1 Y2 Y10 Z11 X12]
+ (1.8395658564934373e-08) [Y1 Z2 X3 X11 Z12 Y13]
+ (1.8395658564934373e-08) [X1 Z2 Y3 Y11 Z12 X13]
+ (2.2702084314539915e-08) [Y0 Z1 Z2 X3 X5 Y6]
+ (2.2702084314539915e-08) [Y0 Z1 Z2 Y3 Y5 Y6]
+ (2.2702084314539915e-08) [X0 Z1 Z2 X3 X5 X6]
+ (2.2702084314539915e-08) [X0 Z1 Z2 Y3 Y5 X6]
+ (2.2702084314539915e-08) [Y1 X2 X4 Z5 Z6 Y7]
+ (2.2702084314539915e-08) [Y1 Y2 Y4 Z5 Z6 Y7]
+ (2.2702084314539915e-08) [X1 X2 X4 Z5 Z6 X7]
+ (2.2702084314539915e-08) [X1 Y2 Y4 Z5 Z6 X7]
+ (2.5928188839881672e-08) [Y0 Z1 X2 X4 Z5 Y6]
+ (2.5928188839881672e-08) [X0 Z1 Y2 Y4 Z5 X6]
+ (2.5928188839881672e-08) [Y1 Z2 X3 X5 Z6 Y7]
+ (2.5928188839881672e-08) [X1 Z2 Y3 Y5 Z6 X7]
+ (1.291945825221344e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (1.291945825221344e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (1.3484969029013208e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (1.3484969029013208e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (1.3484969029013208e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (1.3484969029013208e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (1.380757948147564e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (1.380757948147564e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (1.380757948147564e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (1.380757948147564e-07) [X0 Z1 X2 X5 Z6 X7]
+ (1.380757948147564e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (1.380757948147564e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (1.380757948147564e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (1.380757948147564e-07) [X1 Z2 X3 X4 Z5 X6]
+ (1.6077787912926244e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (1.6077787912926244e-07) [X0 Z1 X2 X4 Z5 X6]
+ (1.6077787912926244e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (1.6077787912926244e-07) [X1 Z2 X3 X5 Z6 X7]
+ (1.839394363779575e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (1.839394363779575e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (1.839394363779575e-07) [X1 Z2 Z3 X4 X6 X7]
+ (1.839394363779575e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (1.9332121409916087e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (1.9332121409916087e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (1.9332121409916087e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (1.9332121409916087e-07) [X0 Z1 X2 X3 Z4 X5]
+ (2.1989637490269485e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (2.1989637490269485e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (2.3712704653095528e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (2.3712704653095528e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (2.3712704653095528e-07) [X1 Z2 Z3 X4 X8 X9]
+ (2.3712704653095528e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (2.4472648242874843e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (2.4472648242874843e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (3.0867708904914603e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (3.0867708904914603e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (3.0867708904914603e-07) [X1 Z2 Z3 X4 X12 X13]
+ (3.0867708904914603e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (3.328039653328072e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.328039653328072e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.5706354981050476e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (3.5706354981050476e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (4.837953358557024e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (4.837953358557024e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (5.627721972384872e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.627721972384872e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (5.927350139202625e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (5.927350139202625e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (5.927350139202625e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (5.927350139202625e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (6.628427193598047e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.628427193598047e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (6.733096660913789e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (6.733096660913789e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (6.733096660913789e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (6.733096660913789e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (7.956666976080719e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (7.956666976080719e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (1.0357924856447404e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.0357924856447404e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.2282691317143466e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (1.2282691317143466e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (1.3304568444028183e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (1.3304568444028183e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (1.4548066967282058e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (1.4548066967282058e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (2.1726379989954773e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.1726379989954773e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.088245679912444e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.088245679912444e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.151295941433859e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (3.151295941433859e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (3.3130170603318937e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.3130170603318937e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.6945168754689076e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.6945168754689076e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (4.183808754039051e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.183808754039051e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.253118671347124e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (4.253118671347124e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (4.556473705684232e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (4.556473705684232e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (4.642978932872495e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (4.642978932872495e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (5.275783397604714e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (5.275783397604714e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (5.97417683152442e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (5.97417683152442e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (6.2900195976509175e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (6.2900195976509175e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (6.2900195976509175e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (6.2900195976509175e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (7.444267319834026e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (7.444267319834026e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (7.444267319834026e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (7.444267319834026e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (7.518288729364397e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (7.518288729364397e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (7.518288729364397e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (7.518288729364397e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (8.774724164236844e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (8.774724164236844e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (8.774724164236844e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (8.774724164236844e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (0.000292225672453627) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.000292225672453627) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.000292225672453627) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.000292225672453627) [X6 Z7 X8 X9 Z10 X11]
+ (0.0004957972886071685) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.0004957972886071685) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.0006650303449099839) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.0006650303449099839) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.0006650303449099839) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.0006650303449099839) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.0008533831051011209) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.0008533831051011209) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.0009298407044408025) [Y4 Z5 Y6 X10 Z11 X12]
+ (0.0009298407044408025) [X4 Z5 X6 Y10 Z11 Y12]
+ (0.0009298407044408025) [Y5 Z6 Y7 X11 Z12 X13]
+ (0.0009298407044408025) [X5 Z6 X7 Y11 Z12 Y13]
+ (0.001609533516296407) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.001609533516296407) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.001609533516296407) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.001609533516296407) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.0016676137499700098) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.0016676137499700098) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.0016676137499700098) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.0016676137499700098) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.0017278745823321462) [Y1 Z2 Z3 X4 X11 Y12]
+ (0.0017278745823321462) [X1 Z2 Z3 Y4 Y11 X12]
+ (0.0017991930083833454) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.0017991930083833454) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.0022939556230641665) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.0022939556230641665) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.0024629166213975275) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.0024629166213975275) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.0024629166213975275) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.0024629166213975275) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.003961569373034176) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.003961569373034176) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.003961569373034176) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.003961569373034176) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.004424843668500904) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.004424843668500904) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.004575015188893614) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.004575015188893614) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.004668615266017398) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.004668615266017398) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.004684920226870667) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.004684920226870667) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.005368616111414019) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.005368616111414019) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.005368616111414019) [X2 X3 X7 Z8 Z9 X10]
+ (0.005368616111414019) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.008125248410121527) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.008125248410121527) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.00876485821939822) [Y2 Z3 Z4 X5 X11 Y12]
+ (0.00876485821939822) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (0.00876485821939822) [X2 Z3 Z4 X5 X11 X12]
+ (0.00876485821939822) [X2 Z3 Z4 Y5 Y11 X12]
+ (0.00876485821939822) [Y3 X4 X10 Z11 Z12 Y13]
+ (0.00876485821939822) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (0.00876485821939822) [X3 X4 X10 Z11 Z12 X13]
+ (0.00876485821939822) [X3 Y4 Y10 Z11 Z12 X13]
+ (0.010540434329179964) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.010540434329179964) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.010540434329179964) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.010540434329179964) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.010959994608900792) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.010959994608900792) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.010959994608900792) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.010959994608900792) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.011307208029968412) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.011307208029968412) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.012214985322762037) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (0.012214985322762037) [Y4 Z5 Y6 X11 Z12 X13]
+ (0.012214985322762037) [X4 Z5 X6 Y11 Z12 Y13]
+ (0.012214985322762037) [X4 Z5 X6 X11 Z12 X13]
+ (0.012214985322762037) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (0.012214985322762037) [Y5 Z6 Y7 X10 Z11 X12]
+ (0.012214985322762037) [X5 Z6 X7 Y10 Z11 Y12]
+ (0.012214985322762037) [X5 Z6 X7 X10 Z11 X12]
+ (0.014411189770076643) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.014411189770076643) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.014411189770076643) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.014411189770076643) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.015225659057112248) [Y3 Z4 Z5 X6 X10 Y11]
+ (0.015225659057112248) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (0.015225659057112248) [X3 Z4 Z5 X6 X10 X11]
+ (0.015225659057112248) [X3 Z4 Z5 Y6 Y10 X11]
+ (0.018266758578502525) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.018266758578502525) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.018266758578502525) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.018266758578502525) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.01902037387510145) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.01902037387510145) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.01902037387510145) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.01902037387510145) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.024388989986515466) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.024388989986515466) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.024388989986515466) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.024388989986515466) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.025104907970006966) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.025104907970006966) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.025104907970006966) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.025104907970006966) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.03078744071852189) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.03078744071852189) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.045879424030661344) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.045879424030661344) [X0 Z2 Z3 Z4 Z5 X6]
+ (0.05600713561683179) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.05600713561683179) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.05600713561683179) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.05600713561683179) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.056084494322889866) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.056084494322889866) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.056084494322889866) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.056084494322889866) [Z1 X7 Z8 Z9 Z10 X11]
+ (-2.595081330031322e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.595081330031322e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.595081330031322e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.595081330031322e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.631261863244803e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (6.631261863244803e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (6.631261863244803e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (6.631261863244803e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.04274326006289241) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04274326006289241) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (0.042743260062892446) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.042743260062892446) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (-0.039359250391471146) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.039359250391471146) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.039359250391471146) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.039359250391471146) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.03560840035224899) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.03560840035224899) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.029903813458237106) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.029903813458237106) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.029903813458237106) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.029903813458237106) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.028730798001222087) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.028730798001222087) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.028730798001222087) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.028730798001222087) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.024755507980746962) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.024755507980746962) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.024755507980746962) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.024755507980746962) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.02428203162332565) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.02428203162332565) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.021433980116881156) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.021433980116881156) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.021433980116881156) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.021433980116881156) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.019257453001871226) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019257453001871226) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.01902831871829257) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.01902831871829257) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.0188889950774013) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-0.0188889950774013) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-0.0188889950774013) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0188889950774013) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.016024666095543556) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (-0.016024666095543556) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (-0.015225659057112246) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.015225659057112246) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.01460374241072418) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.01460374241072418) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.014564473640826997) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.014564473640826997) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.01175599524038182) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.01175599524038182) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.01128514461832123) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-0.01128514461832123) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-0.009841802923820782) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.009841802923820782) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.008469833341355948) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.008469833341355948) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.007306763969601731) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.007306763969601731) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005923799555608317) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-0.005923799555608317) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-0.005652607315220576) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.005652607315220576) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.00536861611141402) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.00536861611141402) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.004158830716401799) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.004158830716401799) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.003989845257552537) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (-0.003989845257552537) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (-0.003989845257552537) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (-0.003989845257552537) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (-0.0033566679213676183) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.0033566679213676183) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.0033566679213676183) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.0033566679213676183) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.003267514897154872) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.003267514897154872) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.003267514897154872) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.003267514897154872) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.0027790407628820344) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.0027790407628820344) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.002293955623064167) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.002293955623064167) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.002293955623064167) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.002293955623064167) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.002261970675220391) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (-0.002261970675220391) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (-0.002261970675220391) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (-0.002261970675220391) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (-0.002261970675220391) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (-0.002261970675220391) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (-0.002261970675220391) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (-0.002261970675220391) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (-0.001303802982462151) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (-0.001303802982462151) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (-0.001303802982462151) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (-0.001303802982462151) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (-0.0002464408105774295) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0002464408105774295) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.00013838603701003027) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.00013838603701003027) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.00013838603701003027) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.00013838603701003027) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.735870605807998e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.735870605807998e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-6.652106284324108e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.652106284324108e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.652106284324108e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.652106284324108e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481751940322436e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.481751940322436e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481751940322436e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.481751940322436e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.7695835792481147e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.7695835792481147e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.5443573817176086e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.5443573817176086e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.5443573817176086e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.5443573817176086e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.5443573817176086e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.5443573817176086e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.5443573817176086e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.5443573817176086e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.3342618698580656e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-3.3342618698580656e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-3.2117638478351695e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (-3.2117638478351695e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (-3.2117638478351695e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (-3.2117638478351695e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (-3.2117638478351695e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (-3.2117638478351695e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (-3.2117638478351695e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (-3.2117638478351695e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (-2.1031634666332857e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.1031634666332857e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.1031634666332857e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.1031634666332857e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0110740026985053e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.0110740026985053e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0110740026985053e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.0110740026985053e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9429465542737667e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.9429465542737667e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9429465542737667e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9429465542737667e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6893056595971645e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (-1.6893056595971645e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (-1.6893056595971645e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (-1.6893056595971645e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (-1.654090047613268e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.654090047613268e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.654090047613268e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.654090047613268e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.628837753190216e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (-1.628837753190216e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (-1.628837753190216e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (-1.628837753190216e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (-1.3304568444028183e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (-1.3304568444028183e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (-1.3304568444028183e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (-1.3304568444028183e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (-7.988467876997351e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.988467876997351e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.988467876997351e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.988467876997351e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.18987037964334e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.18987037964334e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.175164839620145e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175164839620145e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.471606001111822e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.471606001111822e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.5611170180505627e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.5611170180505627e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.5611170180505627e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.5611170180505627e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.5233392971006395e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (-4.5233392971006395e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (-3.4273508589391984e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.4273508589391984e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.4273508589391984e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.4273508589391984e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.328039653328072e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328039653328072e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.328039653328072e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328039653328072e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.0867708904914603e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (-3.0867708904914603e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (-2.8885650666245037e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8885650666245037e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8885650666245037e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.8885650666245037e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.3712704653095528e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (-2.3712704653095528e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (-1.839394363779575e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (-1.839394363779575e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (-8.057465217180758e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (-8.057465217180758e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (-6.772951779512892e-08) [Y1 Z2 Z3 Y4 X7 Z8 Z9 X10]
+ (-6.772951779512892e-08) [X1 Z2 Z3 X4 Y7 Z8 Z9 Y10]
+ (-6.046790640185269e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (-6.046790640185269e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (-6.046790640185269e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (-6.046790640185269e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (-3.226104524722576e-09) [Y0 Z1 Z2 Y3 X4 Z5 Z6 X7]
+ (-3.226104524722576e-09) [X0 Z1 Z2 X3 Y4 Z5 Z6 Y7]
+ (3.226104524722576e-09) [Y0 Z1 Z2 X3 X4 Z5 Z6 Y7]
+ (3.226104524722576e-09) [X0 Z1 Z2 Y3 Y4 Z5 Z6 X7]
+ (6.772951779512892e-08) [Y1 Z2 Z3 X4 X7 Z8 Z9 Y10]
+ (6.772951779512892e-08) [X1 Z2 Z3 Y4 Y7 Z8 Z9 X10]
+ (8.057465217180758e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (8.057465217180758e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (9.208946393559356e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.208946393559356e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.208946393559356e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.208946393559356e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.7035434400169234e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035434400169234e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035434400169234e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.7035434400169234e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.839394363779575e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (1.839394363779575e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (2.3712704653095528e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (2.3712704653095528e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (3.0867708904914603e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (3.0867708904914603e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (4.5233392971006395e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (4.5233392971006395e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (5.471606001111822e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.471606001111822e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.175164839620145e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175164839620145e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.18987037964334e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.18987037964334e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (7.867608548588759e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (7.867608548588759e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (7.867608548588759e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (7.867608548588759e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (1.2282691317143466e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (1.2282691317143466e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (1.2282691317143466e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (1.2282691317143466e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (1.5224581882384387e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (1.5224581882384387e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (1.5224581882384387e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (1.5224581882384387e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (1.5224581882384387e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (1.5224581882384387e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (1.5224581882384387e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (1.5224581882384387e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (2.3609471861882315e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (2.3609471861882315e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (2.3609471861882315e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (2.3609471861882315e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (2.745510594017006e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.745510594017006e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.745510594017006e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.745510594017006e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.745510594017006e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.745510594017006e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.745510594017006e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.745510594017006e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.3130170603318937e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.3130170603318937e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.3130170603318937e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.3130170603318937e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (4.183808754039051e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.183808754039051e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (4.253118671347124e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.253118671347124e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.728781366677482e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (4.728781366677482e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (4.728781366677482e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.728781366677482e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.7345783097969135e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (4.7345783097969135e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (4.7345783097969135e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (4.7345783097969135e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (5.0714036053511186e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (5.0714036053511186e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (5.0714036053511186e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (5.0714036053511186e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (7.089728552864846e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.089728552864846e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (7.089728552864846e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (7.089728552864846e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (9.805981915153236e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (9.805981915153236e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (9.805981915153236e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (9.805981915153236e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (1.5316613948397513e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (1.5316613948397513e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (1.5316613948397513e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (1.5316613948397513e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.61033748032563e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (1.61033748032563e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.61033748032563e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (1.61033748032563e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (7.735870605807998e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.735870605807998e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.0002464408105774295) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (0.0002464408105774295) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (0.0004458488204520667) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.0004458488204520667) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.0004458488204520667) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.0004458488204520667) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.0005940157673919247) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940157673919247) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.0005940157673919247) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940157673919247) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.0005940157673919247) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940157673919247) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.0005940157673919247) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940157673919247) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.0008533831051011209) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.0008533831051011209) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.0008533831051011209) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.0008533831051011209) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.0009581676927582414) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (0.0009581676927582414) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (0.0009581676927582414) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (0.0009581676927582414) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (0.0009581676927582414) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (0.0009581676927582414) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (0.0009581676927582414) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (0.0009581676927582414) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (0.0010435237104109841) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.0010435237104109841) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.0010435237104109841) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.0010435237104109841) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.0012803055951262439) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.0012803055951262439) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.0012803055951262439) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.0012803055951262439) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.0026860422750903867) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (0.0026860422750903867) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (0.0026860422750903867) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (0.0026860422750903867) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (0.004158830716401799) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.004158830716401799) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.004311038607565856) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.004311038607565856) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.004311038607565856) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.004311038607565856) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.004636973516493862) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.004636973516493862) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.004636973516493862) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.004636973516493862) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.005114464086469465) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.005114464086469465) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.005114464086469465) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.005114464086469465) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.005114464086469465) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114464086469465) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114464086469465) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005114464086469465) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.005241543597044163) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (0.005241543597044163) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (0.005241543597044163) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (0.005241543597044163) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (0.0052626310334093224) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.0052626310334093224) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.0052626310334093224) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.0052626310334093224) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.00536861611141402) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.00536861611141402) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005379929634054193) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.005379929634054193) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.005379929634054193) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.005379929634054193) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.005652607315220576) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.005652607315220576) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.00570847985386139) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.00570847985386139) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.00570847985386139) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.00570847985386139) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.005923799555608317) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (0.005923799555608317) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (0.007306763969601731) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.007306763969601731) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.008469833341355948) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.008469833341355948) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.009612546714394977) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (0.009612546714394977) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (0.009612546714394977) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (0.009612546714394977) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (0.009841802923820782) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.009841802923820782) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.01128514461832123) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (0.01128514461832123) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (0.01175599524038182) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.01175599524038182) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.014564473640826997) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.014564473640826997) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.01460374241072418) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.01460374241072418) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.015225659057112246) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.015225659057112246) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.016024666095543556) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (0.016024666095543556) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (0.01902831871829257) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.01902831871829257) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.019257453001871226) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (0.019257453001871226) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (0.022528354240851336) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.022528354240851336) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.02314522165322748) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (0.02314522165322748) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (0.025637212809938532) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (0.025637212809938532) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (0.025637212809938532) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (0.025637212809938532) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (0.03931810723101963) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.03931810723101963) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.03931810723101963) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.03931810723101963) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.039564548041597064) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (0.039564548041597064) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (0.039564548041597064) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.039564548041597064) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.041718814044511415) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (0.041718814044511415) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (0.041718814044511415) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (0.041718814044511415) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (0.04587942403066135) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04587942403066135) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (0.04764261360011973) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (0.04764261360011973) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (0.04764261360011973) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (0.04764261360011973) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.2816433575339251) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.2816433575339251) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.28164335753392505) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.28164335753392505) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.36937137554443256) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.36937137554443256) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.36937137554443256) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.36937137554443256) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0763503693674021) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0763503693674021) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0763503693674021) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0763503693674021) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0675239817996094) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0675239817996094) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0675239817996094) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0675239817996094) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03560840035224899) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.03560840035224899) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.024282031623325654) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.024282031623325654) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.019538085344913032) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.019538085344913032) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.019538085344913032) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.019538085344913032) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.019299499858017436) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.019299499858017436) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019299499858017436) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.019299499858017436) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019299499858017436) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019299499858017436) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.019299499858017436) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019299499858017436) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.017091621922215063) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.017091621922215063) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.017091621922215063) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.017091621922215063) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.010757524201213923) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010757524201213923) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010757524201213923) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010757524201213923) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010715477345067713) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010715477345067713) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010715477345067713) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010715477345067713) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.009841802923820782) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841802923820782) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.009841802923820782) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841802923820782) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826387567792703) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826387567792703) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826387567792703) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008826387567792703) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008469833341355948) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469833341355948) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.008469833341355948) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469833341355948) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.005923799555608317) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-0.005923799555608317) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-0.005923799555608317) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-0.005923799555608317) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-0.004668615266017398) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.004668615266017398) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.0038764821957236965) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.0038764821957236965) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.0034841545793461615) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0034841545793461615) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0033566679213676183) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.0033566679213676183) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.003267514897154872) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.003267514897154872) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0021413489647397984) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.0021413489647397984) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.0017278745823321462) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.0017278745823321462) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.0016407591167047868) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0016407591167047868) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.001452884255324411) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.001452884255324411) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.001452884255324411) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.001452884255324411) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870893705583118) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007870893705583118) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.000519292412051434) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.000519292412051434) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.0002464408105774295) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0002464408105774295) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0002464408105774295) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0002464408105774295) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00019401030606694015) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00019401030606694015) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.0001383860370100303) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.0001383860370100303) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.141566359233591e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.141566359233591e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.141566359233591e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.141566359233591e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.2046856146210143e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-4.2046856146210143e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-4.2046856146210143e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.2046856146210143e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-5.0714036053511186e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-5.0714036053511186e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-3.151295941433859e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (-3.151295941433859e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (-3.088245679912444e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.088245679912444e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.9884125470766793e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.9884125470766793e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.874248535710824e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.874248535710824e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.3609471861882315e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-2.3609471861882315e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.3001958724979945e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.3001958724979945e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-8.398527581243429e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.398527581243429e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.398527581243429e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.398527581243429e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.02784411614229e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.02784411614229e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.02784411614229e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.02784411614229e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.867608548588759e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867608548588759e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.560553844752052e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560553844752052e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560553844752052e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.560553844752052e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.560553844752052e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560553844752052e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560553844752052e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.560553844752052e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.996951747716463e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.996951747716463e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.996951747716463e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.996951747716463e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.996951747716463e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.996951747716463e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.996951747716463e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.996951747716463e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-4.769457073608957e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (-4.769457073608957e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (-4.769457073608957e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (-4.769457073608957e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (-4.4490566783548374e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (-4.4490566783548374e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (-4.4490566783548374e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (-4.4490566783548374e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (-4.0921618956573966e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (-4.0921618956573966e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (-4.0921618956573966e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (-4.0921618956573966e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (-4.0921618956573966e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (-4.0921618956573966e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (-4.0921618956573966e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (-4.0921618956573966e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (-2.8885650666245037e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.8885650666245037e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6863214124634924e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.6863214124634924e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035434400169234e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7035434400169234e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.208946393559356e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.208946393559356e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379737364938675e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379737364938675e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379737364938675e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379737364938675e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379737364938675e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379737364938675e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.379737364938675e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379737364938675e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.706834651053067e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.706834651053067e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.706834651053067e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.706834651053067e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.5689478269649216e-08) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-3.5689478269649216e-08) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-3.5689478269649216e-08) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-3.5689478269649216e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-3.5689478269649216e-08) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-3.5689478269649216e-08) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.5689478269649216e-08) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.5689478269649216e-08) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.204003952541533e-08) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (3.204003952541533e-08) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (3.204003952541533e-08) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (3.204003952541533e-08) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (9.208946393559356e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.208946393559356e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0716844987141967e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.0716844987141967e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.0716844987141967e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.0716844987141967e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.1782130919011746e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.1782130919011746e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.1782130919011746e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.1782130919011746e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.7035434400169234e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7035434400169234e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.249897590622893e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.249897590622893e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.249897590622893e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.249897590622893e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.6863214124634924e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863214124634924e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.8885650666245037e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.8885650666245037e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.37668634058489e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (3.37668634058489e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (3.37668634058489e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (3.37668634058489e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (3.37668634058489e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (3.37668634058489e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (3.37668634058489e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (3.37668634058489e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (3.5682004482142544e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (3.5682004482142544e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (3.5682004482142544e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (3.5682004482142544e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (7.24684933833529e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.24684933833529e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.24684933833529e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.24684933833529e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.24684933833529e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.24684933833529e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.24684933833529e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.24684933833529e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.867608548588759e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (7.867608548588759e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.900025637683429e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (7.900025637683429e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (7.900025637683429e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (7.900025637683429e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (8.091539745310913e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (8.091539745310913e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (8.091539745310913e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (8.091539745310913e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (8.091539745310913e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (8.091539745310913e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (8.091539745310913e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (8.091539745310913e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.1468226085894448e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (1.1468226085894448e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (1.1468226085894448e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (1.1468226085894448e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.3001958724979945e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.3001958724979945e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.3609471861882315e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.3609471861882315e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.874248535710824e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.874248535710824e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.8836531554546005e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.8836531554546005e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.9473314269675734e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9473314269675734e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.9473314269675734e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9473314269675734e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9884125470766793e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9884125470766793e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.088245679912444e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.088245679912444e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.151295941433859e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (3.151295941433859e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (3.846190852263738e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.846190852263738e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.846190852263738e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.846190852263738e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.0714036053511186e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (5.0714036053511186e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (5.10546238338184e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.10546238338184e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.10546238338184e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.10546238338184e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.146386724764335e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.146386724764335e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.146386724764335e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.146386724764335e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.1592944461047825e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.1592944461047825e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.1592944461047825e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.1592944461047825e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.427926587349831e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.427926587349831e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.427926587349831e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.427926587349831e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.9357439740465295e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.9357439740465295e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.9357439740465295e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.9357439740465295e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253185103659539e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.253185103659539e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.979710919092664e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.979710919092664e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.979710919092664e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.979710919092664e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (0.0001383860370100303) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.0001383860370100303) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.00018787486138037538) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.00018787486138037538) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.00018787486138037538) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.00018787486138037538) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00019401030606694015) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (0.00019401030606694015) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (0.000519292412051434) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.000519292412051434) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.0007156737069659755) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007156737069659755) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0007156737069659755) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007156737069659755) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870893705583118) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007870893705583118) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.0015324885626604163) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.0015324885626604163) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.0015324885626604163) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.0015324885626604163) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.0016407591167047868) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0016407591167047868) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0017278745823321462) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.0017278745823321462) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.0024464634226979724) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.0024464634226979724) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.0024464634226979724) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.0024464634226979724) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.003267514897154872) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.003267514897154872) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.0033566679213676183) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.0033566679213676183) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.0034841545793461615) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0034841545793461615) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0038040631543702656) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0038040631543702656) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0038040631543702656) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0038040631543702656) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0038764821957236965) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.0038764821957236965) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.004668615266017398) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.004668615266017398) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.0047672766447260225) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.0047672766447260225) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.0047672766447260225) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.0047672766447260225) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.0052865690567774565) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.0052865690567774565) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.0052865690567774565) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.0052865690567774565) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.0054089707583841115) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.0054089707583841115) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.0054089707583841115) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.0054089707583841115) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.008541975656803511) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.008541975656803511) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (0.008541975656803511) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (0.008541975656803511) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.008541975656803511) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.008541975656803511) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.008541975656803511) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.008541975656803511) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.010311472182408958) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010311472182408958) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010311472182408958) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010311472182408958) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.01460374241072418) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.01460374241072418) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.01460374241072418) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.01460374241072418) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.016024666095543556) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (0.016024666095543556) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (0.016024666095543556) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (0.016024666095543556) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (0.022528354240851336) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.022528354240851336) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.02314522165322748) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (0.02314522165322748) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (0.024591832088265323) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.024591832088265323) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.024591832088265323) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.024591832088265323) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.034903304270674285) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.034903304270674285) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.034903304270674285) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.034903304270674285) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.05859215179546026) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.05859215179546026) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.08684736029515393) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.08684736029515393) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.08684736029515393) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.08684736029515393) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.09065142344952419) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.09065142344952419) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.09065142344952419) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.09065142344952419) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.775872053471593e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.775872053471593e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.775872053471593e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.775872053471593e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.07165056250041449) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.07165056250041449) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.07165056250041452) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.07165056250041452) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.019257453001871226) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.019257453001871226) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.01031147218240896) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.01031147218240896) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008826387567792702) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826387567792702) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0038040631543702656) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0038040631543702656) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.002984180074789936) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.002984180074789936) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.002984180074789936) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.002984180074789936) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.0024464634226979724) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.0024464634226979724) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.002394967154062301) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002394967154062301) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.002394967154062301) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002394967154062301) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.002394967154062301) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.002394967154062301) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.002394967154062301) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.002394967154062301) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0022009568479953606) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0022009568479953606) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0022009568479953606) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0022009568479953606) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002141348964739799) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.002141348964739799) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.001640759116704787) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.001640759116704787) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.001640759116704787) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.001640759116704787) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0011726297842400738) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0011726297842400738) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0011726297842400738) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0011726297842400738) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1462850880839782e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1462850880839782e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.874248535710824e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.874248535710824e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.874248535710824e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.874248535710824e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.3001958724979945e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.3001958724979945e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.3001958724979945e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.3001958724979945e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.0444741531636727e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.0444741531636727e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.0444741531636727e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.0444741531636727e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.955903486549127e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.955903486549127e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.955903486549127e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.955903486549127e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105341029274134e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105341029274134e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105341029274134e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105341029274134e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661200260774008e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661200260774008e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661200260774008e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661200260774008e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540204182680843e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540204182680843e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.18987037964334e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.18987037964334e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.876530329363222e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876530329363222e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876530329363222e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876530329363222e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175164839620145e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175164839620145e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.5233392971006395e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.5233392971006395e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.0766627009480953e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.0766627009480953e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0766627009480953e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.0766627009480953e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.0133988123873805e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0133988123873805e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.9045373489558837e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.9045373489558837e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.9045373489558837e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.9045373489558837e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.666679779213149e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.666679779213149e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.666679779213149e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.666679779213149e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505624572758407e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.8505624572758407e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.846699314010283e-08) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-7.846699314010283e-08) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-6.772951779512892e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-6.772951779512892e-08) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-4.099829217343366e-08) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-4.099829217343366e-08) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.099829217343366e-08) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.099829217343366e-08) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (6.772951779512892e-08) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (6.772951779512892e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (7.846699314010283e-08) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (7.846699314010283e-08) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (1.657009296169672e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.657009296169672e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.657009296169672e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.657009296169672e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.8505624572758407e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.8505624572758407e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863214124634924e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863214124634924e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.6863214124634924e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863214124634924e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0133988123873805e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.0133988123873805e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.5233392971006395e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.5233392971006395e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.6704081085499374e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.6704081085499374e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.6704081085499374e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.6704081085499374e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175164839620145e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175164839620145e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.18987037964334e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.18987037964334e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540204182680843e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540204182680843e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949307620274101e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949307620274101e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.7924638143067034e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924638143067034e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924638143067034e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.7924638143067034e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.8836531554546005e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.8836531554546005e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.9884125470766793e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9884125470766793e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9884125470766793e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9884125470766793e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253185103659539e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.253185103659539e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4016916322469282e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4016916322469282e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4016916322469282e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4016916322469282e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809380136775648e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.5809380136775648e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809380136775648e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.5809380136775648e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.000519292412051434) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.000519292412051434) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.000519292412051434) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.000519292412051434) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.0007870893705583118) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870893705583118) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870893705583118) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870893705583118) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0010283270637552878) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0010283270637552878) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0010283270637552878) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0010283270637552878) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012223373698222275) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.0012223373698222275) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (0.0012223373698222275) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (0.0012223373698222275) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0012223373698222275) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0012223373698222275) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0012223373698222275) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012223373698222275) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012366559235509147) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.0012366559235509147) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.0012366559235509147) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.0012366559235509147) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.001863893139066711) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.001863893139066711) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.001863893139066711) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.001863893139066711) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.001863893139066711) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.001863893139066711) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.001863893139066711) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.001863893139066711) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0022494140606735048) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0022494140606735048) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0022494140606735048) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0022494140606735048) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0024464634226979724) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.0024464634226979724) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.0038040631543702656) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0038040631543702656) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0038764821957236965) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.0038764821957236965) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.0038764821957236965) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.0038764821957236965) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.004220835998340851) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.004220835998340851) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.004220835998340851) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.004220835998340851) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.0053480477184128725) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0053480477184128725) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0053480477184128725) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0053480477184128725) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.005733568640019667) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.005733568640019667) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.005733568640019667) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.005733568640019667) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.005733568640019667) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.005733568640019667) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.005733568640019667) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.005733568640019667) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.007597461779086377) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.007597461779086377) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.007597461779086377) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.007597461779086377) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.008826387567792702) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.008826387567792702) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.01031147218240896) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01031147218240896) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019257453001871226) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019257453001871226) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.05859215179546026) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.05859215179546026) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.3986653437980436e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.3986653437980436e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.3986653437980436e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.3986653437980436e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841545793461615) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0034841545793461615) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0029841800747899354) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.0029841800747899354) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.00019401030606694015) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00019401030606694015) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1462850880839782e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1462850880839782e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.7924638143067034e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7924638143067034e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540204182680843e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540204182680843e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540204182680843e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540204182680843e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.8505624572758407e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505624572758407e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505624572758407e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505624572758407e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.846699314010283e-08) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-7.846699314010283e-08) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-7.846699314010283e-08) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-7.846699314010283e-08) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-4.099829217343366e-08) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-4.099829217343366e-08) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.099829217343366e-08) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-4.099829217343366e-08) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0133988123873805e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.0133988123873805e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0133988123873805e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.0133988123873805e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.949307620274101e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.949307620274101e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.7924638143067034e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7924638143067034e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019401030606694015) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00019401030606694015) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0029841800747899354) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.0029841800747899354) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.0034841545793461615) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0034841545793461615) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
  (-73.13873149596827) [I0]
+ (-0.1806675750701966) [Z7]
+ (-0.18066757507019654) [Z6]
+ (-0.15961443583742488) [Z5]
+ (-0.15961443583742482) [Z4]
+ (0.1741998661208775) [Z3]
+ (0.17419986612087757) [Z2]
+ (0.22757328814457198) [Z1]
+ (0.2275732881445721) [Z0]
+ (-7.954224821794115e-06) [Y5 Y7]
+ (-7.954224821794115e-06) [X5 X7]
+ (8.194105036048865e-06) [Y4 Y6]
+ (8.194105036048865e-06) [X4 X6]
+ (0.11270381859119916) [Z4 Z6]
+ (0.11270381859119916) [Z5 Z7]
+ (0.1195244101689615) [Z0 Z4]
+ (0.1195244101689615) [Z1 Z5]
+ (0.1340173737265387) [Z0 Z6]
+ (0.1340173737265387) [Z1 Z7]
+ (0.13734942210159334) [Z0 Z5]
+ (0.13734942210159334) [Z1 Z4]
+ (0.13766859133614112) [Z2 Z4]
+ (0.13766859133614112) [Z3 Z5]
+ (0.14138903590147123) [Z4 Z7]
+ (0.14138903590147123) [Z5 Z6]
+ (0.14722930783258542) [Z2 Z5]
+ (0.14722930783258542) [Z3 Z4]
+ (0.14926347060043316) [Z4 Z5]
+ (0.1497349700542645) [Z2 Z6]
+ (0.1497349700542645) [Z3 Z7]
+ (0.1513834269911583) [Z0 Z7]
+ (0.1513834269911583) [Z1 Z6]
+ (0.15435760065298848) [Z6 Z7]
+ (0.15582280685849526) [Z2 Z7]
+ (0.15582280685849526) [Z3 Z6]
+ (0.16756669356168535) [Z0 Z2]
+ (0.16756669356168535) [Z1 Z3]
+ (0.18144009362931396) [Z0 Z3]
+ (0.18144009362931396) [Z1 Z2]
+ (0.19392574334985652) [Z0 Z1]
+ (0.2200397724026084) [Z2 Z3]
+ (7.0380238092858004e-06) [Y4 Z5 Y6]
+ (7.0380238092858004e-06) [X4 Z5 X6]
+ (7.0380238092858004e-06) [Y5 Z6 Y7]
+ (7.0380238092858004e-06) [X5 Z6 X7]
+ (-0.028685217310272064) [Y4 Y5 X6 X7]
+ (-0.028685217310272064) [X4 X5 Y6 Y7]
+ (-0.017825011932631835) [Y0 Y1 X4 X5]
+ (-0.017825011932631835) [X0 X1 Y4 Y5]
+ (-0.017366053264619593) [Y0 Y1 X6 X7]
+ (-0.017366053264619593) [X0 X1 Y6 Y7]
+ (-0.013873400067628602) [Y0 Y1 X2 X3]
+ (-0.013873400067628602) [X0 X1 Y2 Y3]
+ (-0.009560716496444289) [Y2 Y3 X4 X5]
+ (-0.009560716496444289) [X2 X3 Y4 Y5]
+ (-0.006087836804230773) [Y2 Y3 X6 X7]
+ (-0.006087836804230773) [X2 X3 Y6 Y7]
+ (-0.0002922256724514113) [Y1 Y2 X3 X4]
+ (-0.0002922256724514113) [X1 X2 Y3 Y4]
+ (-7.954224821794115e-06) [Y4 Z5 Y6 Z7]
+ (-7.954224821794115e-06) [X4 Z5 X6 Z7]
+ (-6.628427372148797e-07) [Y2 X3 X5 Y6]
+ (-6.628427372148797e-07) [Y2 Y3 Y5 Y6]
+ (-6.628427372148797e-07) [X2 X3 X5 X6]
+ (-6.628427372148797e-07) [X2 Y3 Y5 X6]
+ (9.344969730470942e-07) [Z2 Y5 Z6 Y7]
+ (9.344969730470942e-07) [Z2 X5 Z6 X7]
+ (9.344969730470942e-07) [Z3 Y4 Z5 Y6]
+ (9.344969730470942e-07) [Z3 X4 Z5 X6]
+ (1.035792476106797e-06) [Y0 X1 X5 Y6]
+ (1.035792476106797e-06) [Y0 Y1 Y5 Y6]
+ (1.035792476106797e-06) [X0 X1 X5 X6]
+ (1.035792476106797e-06) [X0 Y1 Y5 X6]
+ (1.597339710261974e-06) [Z2 Y4 Z5 Y6]
+ (1.597339710261974e-06) [Z2 X4 Z5 X6]
+ (1.597339710261974e-06) [Z3 Y5 Z6 Y7]
+ (1.597339710261974e-06) [Z3 X5 Z6 X7]
+ (1.8551374636522656e-06) [Z0 Y4 Z5 Y6]
+ (1.8551374636522656e-06) [Z0 X4 Z5 X6]
+ (1.8551374636522656e-06) [Z1 Y5 Z6 Y7]
+ (1.8551374636522656e-06) [Z1 X5 Z6 X7]
+ (2.890929939756027e-06) [Z0 Y5 Z6 Y7]
+ (2.890929939756027e-06) [Z0 X5 Z6 X7]
+ (2.890929939756027e-06) [Z1 Y4 Z5 Y6]
+ (2.890929939756027e-06) [Z1 X4 Z5 X6]
+ (8.194105036048865e-06) [Z4 Y5 Z6 Y7]
+ (8.194105036048865e-06) [Z4 X5 Z6 X7]
+ (0.0002922256724514113) [Y1 X2 X3 Y4]
+ (0.0002922256724514113) [X1 Y2 Y3 X4]
+ (0.006087836804230773) [Y2 X3 X6 Y7]
+ (0.006087836804230773) [X2 Y3 Y6 X7]
+ (0.009560716496444289) [Y2 X3 X4 Y5]
+ (0.009560716496444289) [X2 Y3 Y4 X5]
+ (0.011307208030058757) [Y1 Z2 Z3 Y5]
+ (0.011307208030058757) [X1 Z2 Z3 X5]
+ (0.013873400067628602) [Y0 X1 X2 Y3]
+ (0.013873400067628602) [X0 Y1 Y2 X3]
+ (0.017366053264619593) [Y0 X1 X6 Y7]
+ (0.017366053264619593) [X0 Y1 Y6 X7]
+ (0.017825011932631835) [Y0 X1 X4 Y5]
+ (0.017825011932631835) [X0 Y1 Y4 X5]
+ (0.028685217310272064) [Y4 X5 X6 Y7]
+ (0.028685217310272064) [X4 Y5 Y6 X7]
+ (0.029812299601138584) [Y0 Z1 Z2 Y4]
+ (0.029812299601138584) [X0 Z1 Z2 X4]
+ (0.029812299601138584) [Y1 Z3 Z4 Y5]
+ (0.029812299601138584) [X1 Z3 Z4 X5]
+ (0.03010452527358999) [Y0 Z1 Z3 Y4]
+ (0.03010452527358999) [X0 Z1 Z3 X4]
+ (0.03010452527358999) [Y1 Z2 Z4 Y5]
+ (0.03010452527358999) [X1 Z2 Z4 X5]
+ (0.030787440718628628) [Y0 Z2 Z3 Y4]
+ (0.030787440718628628) [X0 Z2 Z3 X4]
+ (0.04375171612135065) [Y1 Z2 Z3 Z4 Y5]
+ (0.04375171612135065) [X1 Z2 Z3 Z4 X5]
+ (0.043751716121350655) [Y0 Z1 Z2 Z3 Y4]
+ (0.043751716121350655) [X0 Z1 Z2 Z3 X4]
+ (-0.014564473640801191) [Y1 Z2 Z3 X4 X6 Y7]
+ (-0.014564473640801191) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-0.014564473640801191) [X1 Z2 Z3 X4 X6 X7]
+ (-0.014564473640801191) [X1 Z2 Z3 Y4 Y6 X7]
+ (-4.183808900231137e-06) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-4.183808900231137e-06) [X1 Z2 Z3 Z4 Z5 X7]
+ (-3.3130171103847375e-06) [Y1 Z2 Z3 X4 X5 Y6]
+ (-3.3130171103847375e-06) [X1 Z2 Z3 Y4 Y5 X6]
+ (-1.035792476106797e-06) [Y0 X1 X4 Z5 Z6 Y7]
+ (-1.035792476106797e-06) [X0 Y1 Y4 Z5 Z6 X7]
+ (-6.628427372148797e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-6.628427372148797e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-3.3280396983745037e-07) [Y1 Y2 X3 Z4 Z5 X6]
+ (-3.3280396983745037e-07) [X1 X2 Y3 Z4 Z5 Y6]
+ (3.3280396983745037e-07) [Y1 X2 X3 Z4 Z5 Y6]
+ (3.3280396983745037e-07) [X1 Y2 Y3 Z4 Z5 X6]
+ (6.628427372148797e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (6.628427372148797e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (1.035792476106797e-06) [Y0 Y1 X4 Z5 Z6 X7]
+ (1.035792476106797e-06) [X0 X1 Y4 Z5 Z6 Y7]
+ (3.211187469581017e-06) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (3.211187469581017e-06) [X0 Z1 Z2 Z3 Z5 X6]
+ (3.211187469581017e-06) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (3.211187469581017e-06) [X1 Z2 Z3 Z4 Z6 X7]
+ (3.2774383214671104e-06) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (3.2774383214671104e-06) [X0 Z1 Z2 Z4 Z5 X6]
+ (3.2774383214671104e-06) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (3.2774383214671104e-06) [X1 Z3 Z4 Z5 Z6 X7]
+ (3.3130171103847375e-06) [Y1 Z2 Z3 Y4 X5 X6]
+ (3.3130171103847375e-06) [X1 Z2 Z3 X4 Y5 Y6]
+ (3.610242291304561e-06) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (3.610242291304561e-06) [X0 Z1 Z3 Z4 Z5 X6]
+ (3.610242291304561e-06) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (3.610242291304561e-06) [X1 Z2 Z4 Z5 Z6 X7]
+ (3.7695836464031636e-06) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (3.7695836464031636e-06) [X0 Z2 Z3 Z4 Z5 X6]
+ (6.524204579964887e-06) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (6.524204579964887e-06) [X0 Z1 Z2 Z3 Z4 X6]
+ (6.524204579964887e-06) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (6.524204579964887e-06) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.0002922256724514113) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (0.0002922256724514113) [Y0 Z1 Y2 X3 Z4 X5]
+ (0.0002922256724514113) [X0 Z1 X2 Y3 Z4 Y5]
+ (0.0002922256724514113) [X0 Z1 X2 X3 Z4 X5]
+ (0.010540434329262462) [Y0 Z1 Z2 Z3 Y4 Z7]
+ (0.010540434329262462) [X0 Z1 Z2 Z3 X4 Z7]
+ (0.010540434329262462) [Y1 Z2 Z3 Z4 Y5 Z6]
+ (0.010540434329262462) [X1 Z2 Z3 Z4 X5 Z6]
+ (0.011307208030058757) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (0.011307208030058757) [X0 Z1 Z2 Z3 X4 Z5]
+ (0.025104907970063647) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (0.025104907970063647) [X0 Z1 Z2 Z3 X4 Z6]
+ (0.025104907970063647) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (0.025104907970063647) [X1 Z2 Z3 Z4 X5 Z7]
+ (0.030787440718628628) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (0.030787440718628628) [Z0 X1 Z2 Z3 Z4 X5]
+ (5.105681203723776e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (5.105681203723776e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (5.105681203723776e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (5.105681203723776e-06) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.014564473640801191) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-0.014564473640801191) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-4.183808900231137e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-4.183808900231137e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-3.3130171103847375e-06) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (-3.3130171103847375e-06) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (-3.3130171103847375e-06) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (-3.3130171103847375e-06) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (3.3280396983745037e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (3.3280396983745037e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (3.3280396983745037e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (3.3280396983745037e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (3.7695836464031636e-06) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (3.7695836464031636e-06) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (0.014564473640801191) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (0.014564473640801191) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
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
+ (-0.009560705729136197) [Y8 Y9 X10 X11]
+ (-0.009560705729136197) [X8 X9 Y10 Y11]
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
+ (-0.007306759928832761) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.007306759928832761) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.007306759928832761) [X4 X5 X7 Z8 Z9 X10]
+ (-0.007306759928832761) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.005805188989826695) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.005805188989826695) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.005805188989826695) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.005805188989826695) [X3 Z4 Z5 Z6 X7 Z9]
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
+ (-0.015225630757226924) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.015225630757226924) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.01460370472916243) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.01460370472916243) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.014564531231172854) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.014564531231172854) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.011756013419819297) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.011756013419819297) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
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
+ (-0.005368659358109183) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005368659358109183) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
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
+ (0.003989841456619366) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (0.003989841456619366) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (0.003989841456619366) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (0.003989841456619366) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
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
+ (0.011756013419819297) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.011756013419819297) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.014564531231172854) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.014564531231172854) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.01460370472916243) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.01460370472916243) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.015225630757226924) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.015225630757226924) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
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

---

