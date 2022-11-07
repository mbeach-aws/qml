Last update: 2022-11-07  15:09:28 (All times shown in Eastern time)
# List of differences in demonstration outputs

# Table of contents

1. [tutorial_backprop.html](#demo0)
2. [tutorial_measurement_optimize.html](#demo1)
3. [tutorial_quanvolution.html](#demo2)
4. [tutorial_adaptive_circuits.html](#demo3)
5. [tutorial_noisy_circuit_optimization.html](#demo4)
6. [tutorial_error_mitigation.html](#demo5)
7. [tutorial_jax_transformations.html](#demo6)
8. [tutorial_quantum_circuit_cutting.html](#demo7)
9. [tutorial_quantum_chemistry.html](#demo8)
10. [tutorial_local_cost_functions.html](#demo9)
11. [tutorial_tn_circuits.html](#demo10)
12. [tutorial_quantum_transfer_learning.html](#demo11)
13. [tutorial_qnn_module_tf.html](#demo12)


Number of demos different/all demos: 13/71

## 1. tutorial_backprop.html <a name="demo0"></a>

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
Forward pass (best of 3): 0.028115254400017876 sec per loop
Gradient computation (best of 3): 11.408574439099993 sec per loop
10.121491584006435
0.9358535378025424
Forward pass (best of 3): 0.06057506490005835 sec per loop
Backward pass (best of 3): 0.16933104960007767 sec per loop
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

## 2. tutorial_measurement_optimize.html <a name="demo1"></a>

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

## 3. tutorial_quanvolution.html <a name="demo2"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_quanvolution.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
  630784/11490434 [>.............................] - ETA: 0s
11059200/11490434 [===========================>..] - ETA: 0s
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
13/13 - 0s - loss: 3.0160 - accuracy: 0.1000 - val_loss: 2.0646 - val_accuracy: 0.2000 - 396ms/epoch - 30ms/step
Epoch 2/30
13/13 - 0s - loss: 2.2510 - accuracy: 0.1800 - val_loss: 1.9801 - val_accuracy: 0.3333 - 46ms/epoch - 4ms/step
Epoch 3/30
13/13 - 0s - loss: 1.7851 - accuracy: 0.4000 - val_loss: 1.8177 - val_accuracy: 0.2667 - 46ms/epoch - 4ms/step
Epoch 4/30
13/13 - 0s - loss: 1.3652 - accuracy: 0.5400 - val_loss: 1.6107 - val_accuracy: 0.4667 - 31ms/epoch - 2ms/step
Epoch 5/30
13/13 - 0s - loss: 1.1317 - accuracy: 0.7800 - val_loss: 1.4723 - val_accuracy: 0.6000 - 46ms/epoch - 4ms/step
Epoch 6/30
13/13 - 0s - loss: 0.9360 - accuracy: 0.8600 - val_loss: 1.4686 - val_accuracy: 0.5333 - 31ms/epoch - 2ms/step
Epoch 7/30
13/13 - 0s - loss: 0.7383 - accuracy: 0.9400 - val_loss: 1.3536 - val_accuracy: 0.5667 - 31ms/epoch - 2ms/step
Epoch 8/30
13/13 - 0s - loss: 0.5846 - accuracy: 0.9800 - val_loss: 1.2785 - val_accuracy: 0.6667 - 45ms/epoch - 3ms/step
Epoch 9/30
13/13 - 0s - loss: 0.4987 - accuracy: 0.9800 - val_loss: 1.2253 - val_accuracy: 0.6333 - 32ms/epoch - 2ms/step
Epoch 10/30
13/13 - 0s - loss: 0.3921 - accuracy: 1.0000 - val_loss: 1.2655 - val_accuracy: 0.6333 - 47ms/epoch - 4ms/step
Epoch 11/30
13/13 - 0s - loss: 0.3617 - accuracy: 1.0000 - val_loss: 1.1555 - val_accuracy: 0.7000 - 32ms/epoch - 2ms/step
Epoch 12/30
13/13 - 0s - loss: 0.3078 - accuracy: 1.0000 - val_loss: 1.2107 - val_accuracy: 0.6667 - 46ms/epoch - 4ms/step
Epoch 13/30
13/13 - 0s - loss: 0.2618 - accuracy: 1.0000 - val_loss: 1.1166 - val_accuracy: 0.7333 - 31ms/epoch - 2ms/step
Epoch 14/30
13/13 - 0s - loss: 0.2463 - accuracy: 1.0000 - val_loss: 1.0624 - val_accuracy: 0.7000 - 47ms/epoch - 4ms/step
Epoch 15/30
13/13 - 0s - loss: 0.2033 - accuracy: 1.0000 - val_loss: 1.0904 - val_accuracy: 0.7333 - 32ms/epoch - 2ms/step
Epoch 16/30
13/13 - 0s - loss: 0.1799 - accuracy: 1.0000 - val_loss: 1.0865 - val_accuracy: 0.7333 - 51ms/epoch - 4ms/step
Epoch 17/30
13/13 - 0s - loss: 0.1682 - accuracy: 1.0000 - val_loss: 1.0385 - val_accuracy: 0.7333 - 32ms/epoch - 2ms/step
Epoch 18/30
13/13 - 0s - loss: 0.1484 - accuracy: 1.0000 - val_loss: 1.0676 - val_accuracy: 0.7000 - 31ms/epoch - 2ms/step
Epoch 19/30
13/13 - 0s - loss: 0.1349 - accuracy: 1.0000 - val_loss: 1.0447 - val_accuracy: 0.7000 - 46ms/epoch - 4ms/step
Epoch 20/30
13/13 - 0s - loss: 0.1255 - accuracy: 1.0000 - val_loss: 0.9935 - val_accuracy: 0.7333 - 31ms/epoch - 2ms/step
Epoch 21/30
13/13 - 0s - loss: 0.1135 - accuracy: 1.0000 - val_loss: 1.0451 - val_accuracy: 0.7333 - 46ms/epoch - 4ms/step
Epoch 22/30
13/13 - 0s - loss: 0.1041 - accuracy: 1.0000 - val_loss: 1.0142 - val_accuracy: 0.7333 - 31ms/epoch - 2ms/step
Epoch 23/30
13/13 - 0s - loss: 0.0983 - accuracy: 1.0000 - val_loss: 0.9893 - val_accuracy: 0.7333 - 46ms/epoch - 4ms/step
Epoch 24/30
13/13 - 0s - loss: 0.0913 - accuracy: 1.0000 - val_loss: 0.9807 - val_accuracy: 0.7000 - 46ms/epoch - 4ms/step
Epoch 25/30
13/13 - 0s - loss: 0.0868 - accuracy: 1.0000 - val_loss: 0.9715 - val_accuracy: 0.7333 - 32ms/epoch - 2ms/step
Epoch 26/30
13/13 - 0s - loss: 0.0788 - accuracy: 1.0000 - val_loss: 0.9850 - val_accuracy: 0.7333 - 31ms/epoch - 2ms/step
Epoch 27/30
13/13 - 0s - loss: 0.0749 - accuracy: 1.0000 - val_loss: 0.9750 - val_accuracy: 0.7333 - 30ms/epoch - 2ms/step
Epoch 28/30
13/13 - 0s - loss: 0.0730 - accuracy: 1.0000 - val_loss: 0.9570 - val_accuracy: 0.7667 - 31ms/epoch - 2ms/step
Epoch 29/30
13/13 - 0s - loss: 0.0681 - accuracy: 1.0000 - val_loss: 0.9895 - val_accuracy: 0.7333 - 45ms/epoch - 3ms/step
Epoch 30/30
13/13 - 0s - loss: 0.0635 - accuracy: 1.0000 - val_loss: 0.9560 - val_accuracy: 0.7333 - 32ms/epoch - 2ms/step
Epoch 1/30
13/13 - 0s - loss: 2.3619 - accuracy: 0.1400 - val_loss: 2.0567 - val_accuracy: 0.3667 - 334ms/epoch - 26ms/step
Epoch 2/30
13/13 - 0s - loss: 1.9696 - accuracy: 0.4200 - val_loss: 1.9381 - val_accuracy: 0.4667 - 47ms/epoch - 4ms/step
Epoch 3/30
13/13 - 0s - loss: 1.6671 - accuracy: 0.6400 - val_loss: 1.8300 - val_accuracy: 0.4333 - 31ms/epoch - 2ms/step
Epoch 4/30
13/13 - 0s - loss: 1.4340 - accuracy: 0.7400 - val_loss: 1.7113 - val_accuracy: 0.4333 - 46ms/epoch - 4ms/step
Epoch 5/30
13/13 - 0s - loss: 1.2342 - accuracy: 0.7600 - val_loss: 1.6044 - val_accuracy: 0.5000 - 31ms/epoch - 2ms/step
Epoch 6/30
13/13 - 0s - loss: 1.0721 - accuracy: 0.8600 - val_loss: 1.5232 - val_accuracy: 0.5333 - 47ms/epoch - 4ms/step
Epoch 7/30
13/13 - 0s - loss: 0.9348 - accuracy: 0.9000 - val_loss: 1.4596 - val_accuracy: 0.5667 - 46ms/epoch - 4ms/step
Epoch 8/30
13/13 - 0s - loss: 0.8178 - accuracy: 0.9200 - val_loss: 1.3921 - val_accuracy: 0.6000 - 46ms/epoch - 4ms/step
Epoch 9/30
13/13 - 0s - loss: 0.7223 - accuracy: 0.9400 - val_loss: 1.3404 - val_accuracy: 0.6333 - 46ms/epoch - 4ms/step
Epoch 10/30
13/13 - 0s - loss: 0.6404 - accuracy: 0.9600 - val_loss: 1.3065 - val_accuracy: 0.6667 - 31ms/epoch - 2ms/step
Epoch 11/30
13/13 - 0s - loss: 0.5772 - accuracy: 1.0000 - val_loss: 1.2644 - val_accuracy: 0.6333 - 31ms/epoch - 2ms/step
Epoch 12/30
13/13 - 0s - loss: 0.5199 - accuracy: 1.0000 - val_loss: 1.2558 - val_accuracy: 0.6667 - 31ms/epoch - 2ms/step
Epoch 13/30
13/13 - 0s - loss: 0.4695 - accuracy: 1.0000 - val_loss: 1.2258 - val_accuracy: 0.6667 - 45ms/epoch - 3ms/step
Epoch 14/30
13/13 - 0s - loss: 0.4238 - accuracy: 1.0000 - val_loss: 1.1897 - val_accuracy: 0.6667 - 30ms/epoch - 2ms/step
Epoch 15/30
13/13 - 0s - loss: 0.3848 - accuracy: 1.0000 - val_loss: 1.1651 - val_accuracy: 0.6667 - 45ms/epoch - 3ms/step
Epoch 16/30
13/13 - 0s - loss: 0.3525 - accuracy: 1.0000 - val_loss: 1.1503 - val_accuracy: 0.7333 - 31ms/epoch - 2ms/step
Epoch 17/30
13/13 - 0s - loss: 0.3245 - accuracy: 1.0000 - val_loss: 1.1374 - val_accuracy: 0.7000 - 31ms/epoch - 2ms/step
Epoch 18/30
13/13 - 0s - loss: 0.2992 - accuracy: 1.0000 - val_loss: 1.1174 - val_accuracy: 0.6667 - 31ms/epoch - 2ms/step
Epoch 19/30
13/13 - 0s - loss: 0.2745 - accuracy: 1.0000 - val_loss: 1.1119 - val_accuracy: 0.6667 - 30ms/epoch - 2ms/step
Epoch 20/30
13/13 - 0s - loss: 0.2551 - accuracy: 1.0000 - val_loss: 1.0903 - val_accuracy: 0.7000 - 31ms/epoch - 2ms/step
Epoch 21/30
13/13 - 0s - loss: 0.2370 - accuracy: 1.0000 - val_loss: 1.0877 - val_accuracy: 0.6667 - 30ms/epoch - 2ms/step
Epoch 22/30
13/13 - 0s - loss: 0.2199 - accuracy: 1.0000 - val_loss: 1.0776 - val_accuracy: 0.6667 - 30ms/epoch - 2ms/step
Epoch 23/30
13/13 - 0s - loss: 0.2050 - accuracy: 1.0000 - val_loss: 1.0675 - val_accuracy: 0.7000 - 30ms/epoch - 2ms/step
Epoch 24/30
13/13 - 0s - loss: 0.1919 - accuracy: 1.0000 - val_loss: 1.0592 - val_accuracy: 0.7000 - 30ms/epoch - 2ms/step
Epoch 25/30
13/13 - 0s - loss: 0.1812 - accuracy: 1.0000 - val_loss: 1.0554 - val_accuracy: 0.6667 - 45ms/epoch - 3ms/step
Epoch 26/30
13/13 - 0s - loss: 0.1691 - accuracy: 1.0000 - val_loss: 1.0477 - val_accuracy: 0.7000 - 31ms/epoch - 2ms/step
Epoch 27/30
13/13 - 0s - loss: 0.1599 - accuracy: 1.0000 - val_loss: 1.0377 - val_accuracy: 0.7000 - 30ms/epoch - 2ms/step
Epoch 28/30
13/13 - 0s - loss: 0.1515 - accuracy: 1.0000 - val_loss: 1.0341 - val_accuracy: 0.6667 - 45ms/epoch - 3ms/step
Epoch 29/30
13/13 - 0s - loss: 0.1426 - accuracy: 1.0000 - val_loss: 1.0291 - val_accuracy: 0.7000 - 30ms/epoch - 2ms/step
Epoch 30/30
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
 3489792/11490434 [========>.....................] - ETA: 0s
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
13/13 - 0s - loss: 3.0160 - accuracy: 0.1000 - val_loss: 2.0646 - val_accuracy: 0.2000 - 436ms/epoch - 34ms/step
Epoch 2/30
13/13 - 0s - loss: 2.2510 - accuracy: 0.1800 - val_loss: 1.9801 - val_accuracy: 0.3333 - 48ms/epoch - 4ms/step
Epoch 3/30
13/13 - 0s - loss: 1.7851 - accuracy: 0.4000 - val_loss: 1.8177 - val_accuracy: 0.2667 - 34ms/epoch - 3ms/step
Epoch 4/30
13/13 - 0s - loss: 1.3652 - accuracy: 0.5400 - val_loss: 1.6107 - val_accuracy: 0.4667 - 34ms/epoch - 3ms/step
Epoch 5/30
13/13 - 0s - loss: 1.1317 - accuracy: 0.7800 - val_loss: 1.4723 - val_accuracy: 0.6000 - 52ms/epoch - 4ms/step
Epoch 6/30
13/13 - 0s - loss: 0.9360 - accuracy: 0.8600 - val_loss: 1.4686 - val_accuracy: 0.5333 - 47ms/epoch - 4ms/step
Epoch 7/30
13/13 - 0s - loss: 0.7383 - accuracy: 0.9400 - val_loss: 1.3536 - val_accuracy: 0.5667 - 34ms/epoch - 3ms/step
Epoch 8/30
13/13 - 0s - loss: 0.5846 - accuracy: 0.9800 - val_loss: 1.2785 - val_accuracy: 0.6667 - 47ms/epoch - 4ms/step
Epoch 9/30
13/13 - 0s - loss: 0.4987 - accuracy: 0.9800 - val_loss: 1.2253 - val_accuracy: 0.6333 - 33ms/epoch - 3ms/step
Epoch 10/30
13/13 - 0s - loss: 0.3921 - accuracy: 1.0000 - val_loss: 1.2655 - val_accuracy: 0.6333 - 49ms/epoch - 4ms/step
Epoch 11/30
13/13 - 0s - loss: 0.3617 - accuracy: 1.0000 - val_loss: 1.1555 - val_accuracy: 0.7000 - 47ms/epoch - 4ms/step
Epoch 12/30
13/13 - 0s - loss: 0.3078 - accuracy: 1.0000 - val_loss: 1.2107 - val_accuracy: 0.6667 - 48ms/epoch - 4ms/step
Epoch 13/30
13/13 - 0s - loss: 0.2618 - accuracy: 1.0000 - val_loss: 1.1166 - val_accuracy: 0.7333 - 47ms/epoch - 4ms/step
Epoch 14/30
13/13 - 0s - loss: 0.2463 - accuracy: 1.0000 - val_loss: 1.0624 - val_accuracy: 0.7000 - 32ms/epoch - 2ms/step
Epoch 15/30
13/13 - 0s - loss: 0.2033 - accuracy: 1.0000 - val_loss: 1.0904 - val_accuracy: 0.7333 - 32ms/epoch - 2ms/step
Epoch 16/30
13/13 - 0s - loss: 0.1799 - accuracy: 1.0000 - val_loss: 1.0865 - val_accuracy: 0.7333 - 47ms/epoch - 4ms/step
Epoch 17/30
13/13 - 0s - loss: 0.1682 - accuracy: 1.0000 - val_loss: 1.0385 - val_accuracy: 0.7333 - 48ms/epoch - 4ms/step
Epoch 18/30
13/13 - 0s - loss: 0.1484 - accuracy: 1.0000 - val_loss: 1.0676 - val_accuracy: 0.7000 - 47ms/epoch - 4ms/step
Epoch 19/30
13/13 - 0s - loss: 0.1349 - accuracy: 1.0000 - val_loss: 1.0447 - val_accuracy: 0.7000 - 47ms/epoch - 4ms/step
Epoch 20/30
13/13 - 0s - loss: 0.1255 - accuracy: 1.0000 - val_loss: 0.9935 - val_accuracy: 0.7333 - 49ms/epoch - 4ms/step
Epoch 21/30
13/13 - 0s - loss: 0.1135 - accuracy: 1.0000 - val_loss: 1.0451 - val_accuracy: 0.7333 - 48ms/epoch - 4ms/step
Epoch 22/30
13/13 - 0s - loss: 0.1041 - accuracy: 1.0000 - val_loss: 1.0142 - val_accuracy: 0.7333 - 47ms/epoch - 4ms/step
Epoch 23/30
13/13 - 0s - loss: 0.0983 - accuracy: 1.0000 - val_loss: 0.9893 - val_accuracy: 0.7333 - 33ms/epoch - 3ms/step
Epoch 24/30
13/13 - 0s - loss: 0.0913 - accuracy: 1.0000 - val_loss: 0.9807 - val_accuracy: 0.7000 - 47ms/epoch - 4ms/step
Epoch 25/30
13/13 - 0s - loss: 0.0868 - accuracy: 1.0000 - val_loss: 0.9715 - val_accuracy: 0.7333 - 47ms/epoch - 4ms/step
Epoch 26/30
13/13 - 0s - loss: 0.0788 - accuracy: 1.0000 - val_loss: 0.9850 - val_accuracy: 0.7333 - 49ms/epoch - 4ms/step
Epoch 27/30
13/13 - 0s - loss: 0.0749 - accuracy: 1.0000 - val_loss: 0.9750 - val_accuracy: 0.7333 - 47ms/epoch - 4ms/step
Epoch 28/30
13/13 - 0s - loss: 0.0730 - accuracy: 1.0000 - val_loss: 0.9570 - val_accuracy: 0.7667 - 48ms/epoch - 4ms/step
Epoch 29/30
13/13 - 0s - loss: 0.0681 - accuracy: 1.0000 - val_loss: 0.9895 - val_accuracy: 0.7333 - 47ms/epoch - 4ms/step
Epoch 30/30
13/13 - 0s - loss: 0.0635 - accuracy: 1.0000 - val_loss: 0.9560 - val_accuracy: 0.7333 - 48ms/epoch - 4ms/step
Epoch 1/30
13/13 - 0s - loss: 2.3619 - accuracy: 0.1400 - val_loss: 2.0567 - val_accuracy: 0.3667 - 366ms/epoch - 28ms/step
Epoch 2/30
13/13 - 0s - loss: 1.9696 - accuracy: 0.4200 - val_loss: 1.9381 - val_accuracy: 0.4667 - 48ms/epoch - 4ms/step
Epoch 3/30
13/13 - 0s - loss: 1.6671 - accuracy: 0.6400 - val_loss: 1.8300 - val_accuracy: 0.4333 - 49ms/epoch - 4ms/step
Epoch 4/30
13/13 - 0s - loss: 1.4340 - accuracy: 0.7400 - val_loss: 1.7113 - val_accuracy: 0.4333 - 48ms/epoch - 4ms/step
Epoch 5/30
13/13 - 0s - loss: 1.2342 - accuracy: 0.7600 - val_loss: 1.6044 - val_accuracy: 0.5000 - 35ms/epoch - 3ms/step
Epoch 6/30
13/13 - 0s - loss: 1.0721 - accuracy: 0.8600 - val_loss: 1.5232 - val_accuracy: 0.5333 - 34ms/epoch - 3ms/step
Epoch 7/30
13/13 - 0s - loss: 0.9348 - accuracy: 0.9000 - val_loss: 1.4596 - val_accuracy: 0.5667 - 33ms/epoch - 3ms/step
Epoch 8/30
13/13 - 0s - loss: 0.8178 - accuracy: 0.9200 - val_loss: 1.3921 - val_accuracy: 0.6000 - 35ms/epoch - 3ms/step
Epoch 9/30
13/13 - 0s - loss: 0.7223 - accuracy: 0.9400 - val_loss: 1.3404 - val_accuracy: 0.6333 - 35ms/epoch - 3ms/step
Epoch 10/30
13/13 - 0s - loss: 0.6404 - accuracy: 0.9600 - val_loss: 1.3065 - val_accuracy: 0.6667 - 49ms/epoch - 4ms/step
Epoch 11/30
13/13 - 0s - loss: 0.5772 - accuracy: 1.0000 - val_loss: 1.2644 - val_accuracy: 0.6333 - 49ms/epoch - 4ms/step
Epoch 12/30
13/13 - 0s - loss: 0.5199 - accuracy: 1.0000 - val_loss: 1.2558 - val_accuracy: 0.6667 - 49ms/epoch - 4ms/step
Epoch 13/30
13/13 - 0s - loss: 0.4695 - accuracy: 1.0000 - val_loss: 1.2258 - val_accuracy: 0.6667 - 49ms/epoch - 4ms/step
Epoch 14/30
13/13 - 0s - loss: 0.4238 - accuracy: 1.0000 - val_loss: 1.1897 - val_accuracy: 0.6667 - 36ms/epoch - 3ms/step
Epoch 15/30
13/13 - 0s - loss: 0.3848 - accuracy: 1.0000 - val_loss: 1.1651 - val_accuracy: 0.6667 - 35ms/epoch - 3ms/step
Epoch 16/30
13/13 - 0s - loss: 0.3525 - accuracy: 1.0000 - val_loss: 1.1503 - val_accuracy: 0.7333 - 34ms/epoch - 3ms/step
Epoch 17/30
13/13 - 0s - loss: 0.3245 - accuracy: 1.0000 - val_loss: 1.1374 - val_accuracy: 0.7000 - 49ms/epoch - 4ms/step
Epoch 18/30
13/13 - 0s - loss: 0.2992 - accuracy: 1.0000 - val_loss: 1.1174 - val_accuracy: 0.6667 - 35ms/epoch - 3ms/step
Epoch 19/30
13/13 - 0s - loss: 0.2745 - accuracy: 1.0000 - val_loss: 1.1119 - val_accuracy: 0.6667 - 48ms/epoch - 4ms/step
Epoch 20/30
13/13 - 0s - loss: 0.2551 - accuracy: 1.0000 - val_loss: 1.0903 - val_accuracy: 0.7000 - 49ms/epoch - 4ms/step
Epoch 21/30
13/13 - 0s - loss: 0.2370 - accuracy: 1.0000 - val_loss: 1.0877 - val_accuracy: 0.6667 - 35ms/epoch - 3ms/step
Epoch 22/30
13/13 - 0s - loss: 0.2199 - accuracy: 1.0000 - val_loss: 1.0776 - val_accuracy: 0.6667 - 49ms/epoch - 4ms/step
Epoch 23/30
13/13 - 0s - loss: 0.2050 - accuracy: 1.0000 - val_loss: 1.0675 - val_accuracy: 0.7000 - 49ms/epoch - 4ms/step
Epoch 24/30
13/13 - 0s - loss: 0.1919 - accuracy: 1.0000 - val_loss: 1.0592 - val_accuracy: 0.7000 - 34ms/epoch - 3ms/step
Epoch 25/30
13/13 - 0s - loss: 0.1812 - accuracy: 1.0000 - val_loss: 1.0554 - val_accuracy: 0.6667 - 47ms/epoch - 4ms/step
Epoch 26/30
13/13 - 0s - loss: 0.1691 - accuracy: 1.0000 - val_loss: 1.0477 - val_accuracy: 0.7000 - 34ms/epoch - 3ms/step
Epoch 27/30
13/13 - 0s - loss: 0.1599 - accuracy: 1.0000 - val_loss: 1.0377 - val_accuracy: 0.7000 - 36ms/epoch - 3ms/step
Epoch 28/30
13/13 - 0s - loss: 0.1515 - accuracy: 1.0000 - val_loss: 1.0341 - val_accuracy: 0.6667 - 55ms/epoch - 4ms/step
Epoch 29/30
13/13 - 0s - loss: 0.1426 - accuracy: 1.0000 - val_loss: 1.0291 - val_accuracy: 0.7000 - 48ms/epoch - 4ms/step
Epoch 30/30
13/13 - 0s - loss: 0.1344 - accuracy: 1.0000 - val_loss: 1.0264 - val_accuracy: 0.7000 - 35ms/epoch - 3ms/step
 </code>
 </pre>
 </details>

---

## 4. tutorial_adaptive_circuits.html <a name="demo3"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_adaptive_circuits.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
n = 0,  E = -7.86266588 H, t = 0.75 s
n = 1,  E = -7.87094622 H, t = 0.75 s
n = 2,  E = -7.87563101 H, t = 0.94 s
n = 3,  E = -7.87829148 H, t = 0.75 s
n = 4,  E = -7.87981707 H, t = 0.75 s
n = 5,  E = -7.88070478 H, t = 0.75 s
n = 6,  E = -7.88123144 H, t = 0.75 s
n = 7,  E = -7.88155162 H, t = 0.94 s
n = 8,  E = -7.88175219 H, t = 0.75 s
n = 9,  E = -7.88188238 H, t = 0.75 s
n = 10,  E = -7.88197042 H, t = 0.75 s
n = 11,  E = -7.88203269 H, t = 0.75 s
n = 12,  E = -7.88207881 H, t = 0.93 s
n = 13,  E = -7.88211453 H, t = 0.75 s
n = 14,  E = -7.88214336 H, t = 0.75 s
n = 15,  E = -7.88216745 H, t = 0.75 s
n = 16,  E = -7.88218815 H, t = 0.94 s
n = 17,  E = -7.88220635 H, t = 0.75 s
n = 18,  E = -7.88222262 H, t = 0.75 s
n = 19,  E = -7.88223735 H, t = 0.75 s
n = 0,  E = -7.86266588 H, t = 0.14 s
n = 1,  E = -7.87094622 H, t = 0.14 s
n = 2,  E = -7.87563101 H, t = 0.14 s
n = 3,  E = -7.87829148 H, t = 0.14 s
n = 4,  E = -7.87981707 H, t = 0.14 s
n = 5,  E = -7.88070478 H, t = 0.14 s
n = 6,  E = -7.88123144 H, t = 0.14 s
n = 7,  E = -7.88155162 H, t = 0.14 s
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
n = 0,  E = -7.86266588 H, t = 0.77 s
n = 1,  E = -7.87094622 H, t = 0.76 s
n = 2,  E = -7.87563101 H, t = 1.00 s
n = 3,  E = -7.87829148 H, t = 0.77 s
n = 4,  E = -7.87981707 H, t = 0.77 s
n = 5,  E = -7.88070478 H, t = 0.77 s
n = 6,  E = -7.88123144 H, t = 0.77 s
n = 7,  E = -7.88155162 H, t = 0.98 s
n = 8,  E = -7.88175219 H, t = 0.76 s
n = 9,  E = -7.88188238 H, t = 0.77 s
n = 10,  E = -7.88197042 H, t = 0.77 s
n = 11,  E = -7.88203269 H, t = 0.77 s
n = 12,  E = -7.88207881 H, t = 0.99 s
n = 13,  E = -7.88211453 H, t = 0.77 s
n = 14,  E = -7.88214336 H, t = 0.77 s
n = 15,  E = -7.88216745 H, t = 0.77 s
n = 16,  E = -7.88218815 H, t = 0.99 s
n = 17,  E = -7.88220635 H, t = 0.77 s
n = 18,  E = -7.88222262 H, t = 0.77 s
n = 19,  E = -7.88223735 H, t = 0.76 s
n = 0,  E = -7.86266588 H, t = 0.15 s
n = 1,  E = -7.87094622 H, t = 0.15 s
n = 2,  E = -7.87563101 H, t = 0.15 s
n = 3,  E = -7.87829148 H, t = 0.15 s
n = 4,  E = -7.87981707 H, t = 0.15 s
n = 5,  E = -7.88070478 H, t = 0.15 s
n = 6,  E = -7.88123144 H, t = 0.15 s
n = 7,  E = -7.88155162 H, t = 0.15 s
n = 9,  E = -7.88188238 H, t = 0.15 s
n = 10,  E = -7.88197042 H, t = 0.15 s
n = 11,  E = -7.88203269 H, t = 0.16 s
n = 12,  E = -7.88207881 H, t = 0.15 s
n = 13,  E = -7.88211453 H, t = 0.15 s
n = 14,  E = -7.88214336 H, t = 0.15 s
n = 15,  E = -7.88216745 H, t = 0.15 s
n = 16,  E = -7.88218815 H, t = 0.15 s
n = 17,  E = -7.88220635 H, t = 0.15 s
n = 18,  E = -7.88222262 H, t = 0.16 s
n = 19,  E = -7.88223735 H, t = 0.15 s
 </code>
 </pre>
 </details>

---

## 5. tutorial_noisy_circuit_optimization.html <a name="demo4"></a>

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

## 6. tutorial_error_mitigation.html <a name="demo5"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_error_mitigation.html):

```
──╰Z──RY(-3.32)─╭●──RY(-4.05)─┤
────────────────╰Z──RY(-3.51)─┤
3: ──RY(3.51)─╰Z──RY(3.66)──RY(-3.66)──RY(3.66)──RY(-3.66)───────────────╰Z
───RY(-4.56)─┤
───RY(-3.60)─┤
───RY(-4.05)─┤
───RY(-3.51)─┤
0.9517658614811262
───RY(-4.56)──────────────────────┤
───RY(-4.05)──RY(4.05)──RY(-4.05)─┤
0: ──RY(4.56)─╭●──RY(5.93)──RY(-5.93)──────────────────────────────────────────
1: ──RY(3.60)─╰Z──RY(5.90)─╭●──────────RY(5.18)──RY(-5.18)─────────────────────
2: ──RY(4.05)─╭●──RY(3.32)─╰Z──────────RY(1.07)──RY(-1.07)──RY(1.07)──RY(-1.07)
3: ──RY(3.51)─╰Z──RY(3.66)──RY(-3.66)──────────────────────────────────────────
────────────────╭●──RY(-4.56)─┤
──╭●──RY(-5.90)─╰Z──RY(-3.60)─┤
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_error_mitigation.html):

```
───RY(3.32)──RY(-3.32)─╭●──RY(-4.05)─┤
───────────────────────╰Z──RY(-3.51)─┤
3: ──RY(3.51)─╰Z──RY(3.66)──RY(-3.66)────────────────────────────────────╰Z
───RY(-4.56)──────────────────────┤
───RY(-3.60)──RY(3.60)──RY(-3.60)─┤
───RY(-4.05)──────────────────────┤
───RY(-3.51)──────────────────────┤
0.9804679082805065
───RY(-4.56)──RY(4.56)──RY(-4.56)─┤
───RY(-4.05)──────────────────────┤
0: ──RY(4.56)─╭●──RY(5.93)──RY(-5.93)───────────────────────────────────
1: ──RY(3.60)─╰Z──RY(5.90)─╭●──────────RY(5.18)──RY(-5.18)─╭●──RY(-5.90)
2: ──RY(4.05)─╭●──RY(3.32)─╰Z──────────RY(1.07)──RY(-1.07)─╰Z──RY(-3.32)
3: ──RY(3.51)─╰Z──RY(3.66)──RY(-3.66)───────────────────────────────────
──╭●─────────RY(-4.56)───────────────┤
──╰Z─────────RY(-3.60)───────────────┤
```

---

## 7. tutorial_jax_transformations.html <a name="demo6"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_jax_transformations.html):

```
No jit time: 0.0054 seconds
First run time: 0.0784 seconds
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_jax_transformations.html):

```
No jit time: 0.0059 seconds
First run time: 0.0869 seconds
```

---

## 8. tutorial_quantum_circuit_cutting.html <a name="demo7"></a>

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

## 9. tutorial_quantum_chemistry.html <a name="demo8"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_quantum_chemistry.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
  (-46.46390691341527) [I0]
+ (0.7829652070484762) [Z10]
+ (0.7829652070484763) [Z11]
+ (0.8084591005187078) [Z12]
+ (0.8084591005187078) [Z13]
+ (1.2034393391312599) [Z4]
+ (1.2034393391312603) [Z5]
+ (1.309687661862195) [Z6]
+ (1.3096876618621955) [Z7]
+ (1.3693525711701853) [Z8]
+ (1.3693525711701857) [Z9]
+ (1.65389383054678) [Z2]
+ (1.65389383054678) [Z3]
+ (12.412630714441441) [Z0]
+ (12.412630714441441) [Z1]
+ (-8.19410489543565e-06) [Y10 Y12]
+ (-8.19410489543565e-06) [X10 X12]
+ (-1.6021751140412444e-06) [Y2 Y4]
+ (-1.6021751140412444e-06) [X2 X4]
+ (5.929280441508561e-07) [Y4 Y6]
+ (5.929280441508561e-07) [X4 X6]
+ (7.765082391203524e-07) [Y3 Y5]
+ (7.765082391203524e-07) [X3 X5]
+ (1.8540565404753448e-06) [Y5 Y7]
+ (1.8540565404753448e-06) [X5 X7]
+ (7.954224404280869e-06) [Y11 Y13]
+ (7.954224404280869e-06) [X11 X13]
+ (0.0032769650657663235) [Y1 Y3]
+ (0.0032769650657663235) [X1 X3]
+ (0.10433061485316272) [Y0 Y2]
+ (0.10433061485316272) [X0 X2]
+ (0.11270381859119778) [Z10 Z12]
+ (0.11270381859119778) [Z11 Z13]
+ (0.11383573685303344) [Z4 Z12]
+ (0.11383573685303344) [Z5 Z13]
+ (0.11952441016887197) [Z6 Z10]
+ (0.11952441016887197) [Z7 Z11]
+ (0.12489977362988341) [Z4 Z10]
+ (0.12489977362988341) [Z5 Z11]
+ (0.12495799328194812) [Z2 Z4]
+ (0.12495799328194812) [Z3 Z5]
+ (0.12799492801392184) [Z2 Z10]
+ (0.12799492801392184) [Z3 Z11]
+ (0.13401737372650885) [Z6 Z12]
+ (0.13401737372650885) [Z7 Z13]
+ (0.1370119191305552) [Z4 Z6]
+ (0.1370119191305552) [Z5 Z7]
+ (0.13734942210148537) [Z6 Z11]
+ (0.13734942210148537) [Z7 Z10]
+ (0.1373911237500648) [Z2 Z6]
+ (0.1373911237500648) [Z3 Z7]
+ (0.13766859133615283) [Z8 Z10]
+ (0.13766859133615283) [Z9 Z11]
+ (0.1401129474971454) [Z2 Z12]
+ (0.1401129474971454) [Z3 Z13]
+ (0.14138903590146557) [Z10 Z13]
+ (0.14138903590146557) [Z11 Z12]
+ (0.14257991128699615) [Z4 Z11]
+ (0.14257991128699615) [Z5 Z10]
+ (0.1472293078325984) [Z8 Z11]
+ (0.1472293078325984) [Z9 Z10]
+ (0.14899426171381325) [Z4 Z7]
+ (0.14899426171381325) [Z5 Z6]
+ (0.1492634706003302) [Z10 Z11]
+ (0.1496069255725133) [Z4 Z8]
+ (0.1496069255725133) [Z5 Z9]
+ (0.14973497005437825) [Z8 Z12]
+ (0.14973497005437825) [Z9 Z13]
+ (0.15071405482292327) [Z2 Z8]
+ (0.15071405482292327) [Z3 Z9]
+ (0.15138342699113821) [Z6 Z13]
+ (0.15138342699113821) [Z7 Z12]
+ (0.1521504062265009) [Z4 Z13]
+ (0.1521504062265009) [Z5 Z12]
+ (0.15337959171056728) [Z2 Z11]
+ (0.15337959171056728) [Z3 Z10]
+ (0.15435760065308654) [Z12 Z13]
+ (0.15569017457260936) [Z2 Z13]
+ (0.15569017457260936) [Z3 Z12]
+ (0.15582280685861422) [Z8 Z13]
+ (0.15582280685861422) [Z9 Z12]
+ (0.15676384610170307) [Z4 Z9]
+ (0.15676384610170307) [Z5 Z8]
+ (0.15755303804354884) [Z4 Z5]
+ (0.16079755046698577) [Z2 Z5]
+ (0.16079755046698577) [Z3 Z4]
+ (0.16756669356166093) [Z6 Z8]
+ (0.16756669356166093) [Z7 Z9]
+ (0.16853492794641817) [Z2 Z7]
+ (0.16853492794641817) [Z3 Z6]
+ (0.18144009362928676) [Z6 Z9]
+ (0.18144009362928676) [Z7 Z8]
+ (0.1818908124373772) [Z2 Z3]
+ (0.18690814831042646) [Z2 Z9]
+ (0.18690814831042646) [Z3 Z8]
+ (0.19299700269862718) [Z0 Z10]
+ (0.19299700269862718) [Z1 Z11]
+ (0.19392574334963214) [Z6 Z7]
+ (0.19661749959733849) [Z0 Z4]
+ (0.19661749959733849) [Z1 Z5]
+ (0.1993633269127626) [Z0 Z5]
+ (0.1993633269127626) [Z1 Z4]
+ (0.2007284355459854) [Z0 Z11]
+ (0.2007284355459854) [Z1 Z10]
+ (0.21102681234294474) [Z0 Z12]
+ (0.21102681234294474) [Z1 Z13]
+ (0.2163105980967746) [Z0 Z13]
+ (0.2163105980967746) [Z1 Z12]
+ (0.22003977240279504) [Z8 Z9]
+ (0.2367107174042449) [Z0 Z2]
+ (0.2367107174042449) [Z1 Z3]
+ (0.24164696831740481) [Z0 Z6]
+ (0.24164696831740481) [Z1 Z7]
+ (0.24853517285972174) [Z0 Z7]
+ (0.24853517285972174) [Z1 Z6]
+ (0.2512943557312608) [Z0 Z3]
+ (0.2512943557312608) [Z1 Z2]
+ (0.2723251845038335) [Z0 Z8]
+ (0.2723251845038335) [Z1 Z9]
+ (0.2788345457379122) [Z0 Z9]
+ (0.2788345457379122) [Z1 Z8]
+ (1.1861764484136939) [Z0 Z1]
+ (-1.0722748612460347e-05) [Y10 Z11 Y12]
+ (-1.0722748612460347e-05) [X10 Z11 X12]
+ (-1.0722748612460347e-05) [Y11 Z12 Y13]
+ (-1.0722748612460347e-05) [X11 Z12 X13]
+ (3.886639458202496e-06) [Y2 Z3 Y4]
+ (3.886639458202496e-06) [X2 Z3 X4]
+ (3.886639458202496e-06) [Y3 Z4 Y5]
+ (3.886639458202496e-06) [X3 Z4 X5]
+ (1.22602769720406e-05) [Y4 Z5 Y6]
+ (1.22602769720406e-05) [X4 Z5 X6]
+ (1.22602769720406e-05) [Y5 Z6 Y7]
+ (1.22602769720406e-05) [X5 Z6 X7]
+ (0.1250703688398574) [Y0 Z1 Y2]
+ (0.1250703688398574) [X0 Z1 X2]
+ (0.12507036883985742) [Y1 Z2 Y3]
+ (0.12507036883985742) [X1 Z2 X3]
+ (-0.03831466937346745) [Y4 Y5 X12 X13]
+ (-0.03831466937346745) [X4 X5 Y12 Y13]
+ (-0.03619409348750321) [Y2 Y3 X8 X9]
+ (-0.03619409348750321) [X2 X3 Y8 Y9]
+ (-0.035839557185037646) [Y2 Y3 X4 X5]
+ (-0.035839557185037646) [X2 X3 Y4 Y5]
+ (-0.031143804196353403) [Y2 Y3 X6 X7]
+ (-0.031143804196353403) [X2 X3 Y6 Y7]
+ (-0.028685217310267793) [Y10 Y11 X12 X13]
+ (-0.028685217310267793) [X10 X11 Y12 Y13]
+ (-0.02599620626717901) [Y3 Z4 Z5 Y7]
+ (-0.02599620626717901) [X3 Z4 Z5 X7]
+ (-0.025384663696645434) [Y2 Y3 X10 X11]
+ (-0.025384663696645434) [X2 X3 Y10 Y11]
+ (-0.01902831871828794) [Y3 X4 X11 Y12]
+ (-0.01902831871828794) [X3 Y4 Y11 X12]
+ (-0.01782501193261339) [Y6 Y7 X10 X11]
+ (-0.01782501193261339) [X6 X7 Y10 Y11]
+ (-0.017680137657112755) [Y4 Y5 X10 X11]
+ (-0.017680137657112755) [X4 X5 Y10 Y11]
+ (-0.017366053264629366) [Y6 Y7 X12 X13]
+ (-0.017366053264629366) [X6 X7 Y12 Y13]
+ (-0.01557722707546399) [Y2 Y3 X12 X13]
+ (-0.01557722707546399) [X2 X3 Y12 Y13]
+ (-0.014583638327015865) [Y0 Y1 X2 X3]
+ (-0.014583638327015865) [X0 X1 Y2 Y3]
+ (-0.01387340006762584) [Y6 Y7 X8 X9]
+ (-0.01387340006762584) [X6 X7 Y8 Y9]
+ (-0.011982342583258088) [Y4 Y5 X6 X7]
+ (-0.011982342583258088) [X4 X5 Y6 Y7]
+ (-0.011285144618316839) [Y5 Y6 X11 X12]
+ (-0.011285144618316839) [X5 X6 Y11 Y12]
+ (-0.009560716496445583) [Y8 Y9 X10 X11]
+ (-0.009560716496445583) [X8 X9 Y10 Y11]
+ (-0.008125248410122344) [Y1 X2 X8 Y9]
+ (-0.008125248410122344) [Y1 Y2 Y8 Y9]
+ (-0.008125248410122344) [X1 X2 X8 X9]
+ (-0.008125248410122344) [X1 Y2 Y8 X9]
+ (-0.007731432847358214) [Y0 Y1 X10 X11]
+ (-0.007731432847358214) [X0 X1 Y10 Y11]
+ (-0.0071569205291897855) [Y4 Y5 X8 X9]
+ (-0.0071569205291897855) [X4 X5 Y8 Y9]
+ (-0.006888204542316916) [Y0 Y1 X6 X7]
+ (-0.006888204542316916) [X0 X1 Y6 Y7]
+ (-0.006509361234078718) [Y0 Y1 X8 X9]
+ (-0.006509361234078718) [X0 X1 Y8 Y9]
+ (-0.0060878368042359635) [Y8 Y9 X12 X13]
+ (-0.0060878368042359635) [X8 X9 Y12 Y13]
+ (-0.00528378575382988) [Y0 Y1 X12 X13]
+ (-0.00528378575382988) [X0 X1 Y12 Y13]
+ (-0.005143382387696295) [Y3 X4 X5 Y6]
+ (-0.005143382387696295) [X3 Y4 Y5 X6]
+ (-0.004684920226865016) [Y1 X2 X6 Y7]
+ (-0.004684920226865016) [Y1 Y2 Y6 Y7]
+ (-0.004684920226865016) [X1 X2 X6 X7]
+ (-0.004684920226865016) [X1 Y2 Y6 X7]
+ (-0.004575015188895283) [Y1 X2 X12 Y13]
+ (-0.004575015188895283) [Y1 Y2 Y12 Y13]
+ (-0.004575015188895283) [X1 X2 X12 X13]
+ (-0.004575015188895283) [X1 Y2 Y12 X13]
+ (-0.004424843668496997) [Y1 X2 X4 Y5]
+ (-0.004424843668496997) [Y1 Y2 Y4 Y5]
+ (-0.004424843668496997) [X1 X2 X4 X5]
+ (-0.004424843668496997) [X1 Y2 Y4 X5]
+ (-0.00347942172930562) [Y2 Z3 Z5 Y6]
+ (-0.00347942172930562) [X2 Z3 Z5 X6]
+ (-0.00347942172930562) [Y3 Z4 Z6 Y7]
+ (-0.00347942172930562) [X3 Z4 Z6 X7]
+ (-0.0027458273154241345) [Y0 Y1 X4 X5]
+ (-0.0027458273154241345) [X0 X1 Y4 Y5]
+ (-0.0017991930083828414) [Y1 X2 X10 Y11]
+ (-0.0017991930083828414) [Y1 Y2 Y10 Y11]
+ (-0.0017991930083828414) [X1 X2 X10 X11]
+ (-0.0017991930083828414) [X1 Y2 Y10 X11]
+ (-0.0002922256724535683) [Y7 Y8 X9 X10]
+ (-0.0002922256724535683) [X7 X8 Y9 Y10]
+ (-8.814793246107473e-06) [Y2 Z3 Y4 Z13]
+ (-8.814793246107473e-06) [X2 Z3 X4 Z13]
+ (-8.814793246107473e-06) [Y3 Z4 Y5 Z12]
+ (-8.814793246107473e-06) [X3 Z4 X5 Z12]
+ (-8.19410489543565e-06) [Z10 Y11 Z12 Y13]
+ (-8.19410489543565e-06) [Z10 X11 Z12 X13]
+ (-5.974176779876715e-06) [Y5 X6 X10 Y11]
+ (-5.974176779876715e-06) [Y5 Y6 Y10 Y11]
+ (-5.974176779876715e-06) [X5 X6 X10 X11]
+ (-5.974176779876715e-06) [X5 Y6 Y10 X11]
+ (-5.275783343121386e-06) [Y3 X4 X12 Y13]
+ (-5.275783343121386e-06) [Y3 Y4 Y12 Y13]
+ (-5.275783343121386e-06) [X3 X4 X12 X13]
+ (-5.275783343121386e-06) [X3 Y4 Y12 X13]
+ (-4.281811973731384e-06) [Y4 Z5 Y6 Z11]
+ (-4.281811973731384e-06) [X4 Z5 X6 Z11]
+ (-4.281811973731384e-06) [Y5 Z6 Y7 Z10]
+ (-4.281811973731384e-06) [X5 Z6 X7 Z10]
+ (-3.694516852413565e-06) [Y4 X5 X11 Y12]
+ (-3.694516852413565e-06) [Y4 Y5 Y11 Y12]
+ (-3.694516852413565e-06) [X4 X5 X11 X12]
+ (-3.694516852413565e-06) [X4 Y5 Y11 X12]
+ (-3.539009902981966e-06) [Y2 Z3 Y4 Z12]
+ (-3.539009902981966e-06) [X2 Z3 X4 Z12]
+ (-3.539009902981966e-06) [Y3 Z4 Y5 Z13]
+ (-3.539009902981966e-06) [X3 Z4 X5 Z13]
+ (-3.1173663876224656e-06) [Y0 Z2 Z3 Y4]
+ (-3.1173663876224656e-06) [X0 Z2 Z3 X4]
+ (-2.8909299700894016e-06) [Z6 Y11 Z12 Y13]
+ (-2.8909299700894016e-06) [Z6 X11 Z12 X13]
+ (-2.8909299700894016e-06) [Z7 Y10 Z11 Y12]
+ (-2.8909299700894016e-06) [Z7 X10 Z11 X12]
+ (-2.1777330133110336e-06) [Z0 Y10 Z11 Y12]
+ (-2.1777330133110336e-06) [Z0 X10 Z11 X12]
+ (-2.1777330133110336e-06) [Z1 Y11 Z12 Y13]
+ (-2.1777330133110336e-06) [Z1 X11 Z12 X13]
+ (-1.8551374850739322e-06) [Z6 Y10 Z11 Y12]
+ (-1.8551374850739322e-06) [Z6 X10 Z11 X12]
+ (-1.8551374850739322e-06) [Z7 Y11 Z12 Y13]
+ (-1.8551374850739322e-06) [Z7 X11 Z12 X13]
+ (-1.8163674236418292e-06) [Z4 Y11 Z12 Y13]
+ (-1.8163674236418292e-06) [Z4 X11 Z12 X13]
+ (-1.8163674236418292e-06) [Z5 Y10 Z11 Y12]
+ (-1.8163674236418292e-06) [Z5 X10 Z11 X12]
+ (-1.6149608219861889e-06) [Z0 Y11 Z12 Y13]
+ (-1.6149608219861889e-06) [Z0 X11 Z12 X13]
+ (-1.6149608219861889e-06) [Z1 Y10 Z11 Y12]
+ (-1.6149608219861889e-06) [Z1 X10 Z11 X12]
+ (-1.6021751140412444e-06) [Z2 Y3 Z4 Y5]
+ (-1.6021751140412444e-06) [Z2 X3 Z4 X5]
+ (-1.5973397459964102e-06) [Z8 Y10 Z11 Y12]
+ (-1.5973397459964102e-06) [Z8 X10 Z11 X12]
+ (-1.5973397459964102e-06) [Z9 Y11 Z12 Y13]
+ (-1.5973397459964102e-06) [Z9 X11 Z12 X13]
+ (-1.0632255395301264e-06) [Z2 Y10 Z11 Y12]
+ (-1.0632255395301264e-06) [Z2 X10 Z11 X12]
+ (-1.0632255395301264e-06) [Z3 Y11 Z12 Y13]
+ (-1.0632255395301264e-06) [Z3 X11 Z12 X13]
+ (-1.035792485014602e-06) [Y6 X7 X11 Y12]
+ (-1.035792485014602e-06) [Y6 Y7 Y11 Y12]
+ (-1.035792485014602e-06) [X6 X7 X11 X12]
+ (-1.035792485014602e-06) [X6 Y7 Y11 X12]
+ (-9.344970345812054e-07) [Z8 Y11 Z12 Y13]
+ (-9.344970345812054e-07) [Z8 X11 Z12 X13]
+ (-9.344970345812054e-07) [Z9 Y10 Z11 Y12]
+ (-9.344970345812054e-07) [Z9 X10 Z11 X12]
+ (-5.471605954546118e-07) [Y1 Y2 X11 X12]
+ (-5.471605954546118e-07) [X1 X2 Y11 Y12]
+ (-3.6060692969167654e-07) [Y0 Z1 Z2 Y4]
+ (-3.6060692969167654e-07) [X0 Z1 Z2 X4]
+ (-3.6060692969167654e-07) [Y1 Z3 Z4 Y5]
+ (-3.6060692969167654e-07) [X1 Z3 Z4 X5]
+ (-2.5941461764844055e-07) [Y2 Z3 Y4 Z6]
+ (-2.5941461764844055e-07) [X2 Z3 X4 Z6]
+ (-2.5941461764844055e-07) [Y3 Z4 Y5 Z7]
+ (-2.5941461764844055e-07) [X3 Z4 X5 Z7]
+ (-2.1872304352269552e-07) [Y2 Z3 Y4 Z8]
+ (-2.1872304352269552e-07) [X2 Z3 X4 Z8]
+ (-2.1872304352269552e-07) [Y3 Z4 Y5 Z9]
+ (-2.1872304352269552e-07) [X3 Z4 X5 Z9]
+ (-1.933212137190972e-07) [Y1 Y2 X3 X4]
+ (-1.933212137190972e-07) [X1 X2 Y3 Y4]
+ (-1.672857159725404e-07) [Y0 Z1 Z3 Y4]
+ (-1.672857159725404e-07) [X0 Z1 Z3 X4]
+ (-1.672857159725404e-07) [Y1 Z2 Z4 Y5]
+ (-1.672857159725404e-07) [X1 Z2 Z4 X5]
+ (-3.226104183868895e-09) [Y1 Y2 X5 X6]
+ (-3.226104183868895e-09) [X1 X2 Y5 Y6]
+ (3.226104183868895e-09) [Y1 X2 X5 Y6]
+ (3.226104183868895e-09) [X1 Y2 Y5 X6]
+ (3.2284057809028877e-08) [Y4 Z5 Y6 Z12]
+ (3.2284057809028877e-08) [X4 Z5 X6 Z12]
+ (3.2284057809028877e-08) [Y5 Z6 Y7 Z13]
+ (3.2284057809028877e-08) [X5 Z6 X7 Z13]
+ (1.291945822423662e-07) [Y1 Z2 Z3 Y5]
+ (1.291945822423662e-07) [X1 Z2 Z3 X5]
+ (1.933212137190972e-07) [Y1 X2 X3 Y4]
+ (1.933212137190972e-07) [X1 Y2 Y3 X4]
+ (2.1989637299401114e-07) [Y2 X3 X5 Y6]
+ (2.1989637299401114e-07) [Y2 Y3 Y5 Y6]
+ (2.1989637299401114e-07) [X2 X3 X5 X6]
+ (2.1989637299401114e-07) [X2 Y3 Y5 X6]
+ (2.4472648177162583e-07) [Y0 X1 X5 Y6]
+ (2.4472648177162583e-07) [Y0 Y1 Y5 Y6]
+ (2.4472648177162583e-07) [X0 X1 X5 X6]
+ (2.4472648177162583e-07) [X0 Y1 Y5 X6]
+ (3.5706354787241953e-07) [Y0 X1 X3 Y4]
+ (3.5706354787241953e-07) [Y0 Y1 Y3 Y4]
+ (3.5706354787241953e-07) [X0 X1 X3 X4]
+ (3.5706354787241953e-07) [X0 Y1 Y3 X4]
+ (4.837953346510182e-07) [Y5 X6 X8 Y9]
+ (4.837953346510182e-07) [Y5 Y6 Y8 Y9]
+ (4.837953346510182e-07) [X5 X6 X8 X9]
+ (4.837953346510182e-07) [X5 Y6 Y8 X9]
+ (5.471605954546118e-07) [Y1 X2 X11 Y12]
+ (5.471605954546118e-07) [X1 Y2 Y11 X12]
+ (5.627721913248176e-07) [Y0 X1 X11 Y12]
+ (5.627721913248176e-07) [Y0 Y1 Y11 Y12]
+ (5.627721913248176e-07) [X0 X1 X11 X12]
+ (5.627721913248176e-07) [X0 Y1 Y11 X12]
+ (5.769436506493364e-07) [Y2 Z3 Y4 Z9]
+ (5.769436506493364e-07) [X2 Z3 X4 Z9]
+ (5.769436506493364e-07) [Y3 Z4 Y5 Z8]
+ (5.769436506493364e-07) [X3 Z4 X5 Z8]
+ (5.929280441508561e-07) [Z4 Y5 Z6 Y7]
+ (5.929280441508561e-07) [Z4 X5 Z6 X7]
+ (6.628427114152048e-07) [Y8 X9 X11 Y12]
+ (6.628427114152048e-07) [Y8 Y9 Y11 Y12]
+ (6.628427114152048e-07) [X8 X9 X11 X12]
+ (6.628427114152048e-07) [X8 Y9 Y11 X12]
+ (7.765082391203524e-07) [Y2 Z3 Y4 Z5]
+ (7.765082391203524e-07) [X2 Z3 X4 Z5]
+ (7.956666941720319e-07) [Y3 X4 X8 Y9]
+ (7.956666941720319e-07) [Y3 Y4 Y8 Y9]
+ (7.956666941720319e-07) [X3 X4 X8 X9]
+ (7.956666941720319e-07) [X3 Y4 Y8 X9]
+ (8.336695401112596e-07) [Z0 Y2 Z3 Y4]
+ (8.336695401112596e-07) [Z0 X2 Z3 X4]
+ (8.336695401112596e-07) [Z1 Y3 Z4 Y5]
+ (8.336695401112596e-07) [Z1 X3 Z4 X5]
+ (9.509134517222441e-07) [Z2 Y4 Z5 Y6]
+ (9.509134517222441e-07) [Z2 X4 Z5 X6]
+ (9.509134517222441e-07) [Z3 Y5 Z6 Y7]
+ (9.509134517222441e-07) [Z3 X5 Z6 X7]
+ (1.1094124377864032e-06) [Z2 Y11 Z12 Y13]
+ (1.1094124377864032e-06) [Z2 X11 Z12 X13]
+ (1.1094124377864032e-06) [Z3 Y10 Z11 Y12]
+ (1.1094124377864032e-06) [Z3 X10 Z11 X12]
+ (1.1708098247149542e-06) [Z2 Y5 Z6 Y7]
+ (1.1708098247149542e-06) [Z2 X5 Z6 X7]
+ (1.1708098247149542e-06) [Z3 Y4 Z5 Y6]
+ (1.1708098247149542e-06) [Z3 X4 Z5 X6]
+ (1.190733087983696e-06) [Z0 Y3 Z4 Y5]
+ (1.190733087983696e-06) [Z0 X3 Z4 X5]
+ (1.190733087983696e-06) [Z1 Y2 Z3 Y4]
+ (1.190733087983696e-06) [Z1 X2 Z3 X4]
+ (1.1953920759999806e-06) [Y2 Z3 Y4 Z7]
+ (1.1953920759999806e-06) [X2 Z3 X4 Z7]
+ (1.1953920759999806e-06) [Y3 Z4 Y5 Z6]
+ (1.1953920759999806e-06) [X3 Z4 X5 Z6]
+ (1.3980242955551536e-06) [Y4 Z5 Y6 Z8]
+ (1.3980242955551536e-06) [X4 Z5 X6 Z8]
+ (1.3980242955551536e-06) [Y5 Z6 Y7 Z9]
+ (1.3980242955551536e-06) [X5 Z6 X7 Z9]
+ (1.4548066936480959e-06) [Y3 X4 X6 Y7]
+ (1.4548066936480959e-06) [Y3 Y4 Y6 Y7]
+ (1.4548066936480959e-06) [X3 X4 X6 X7]
+ (1.4548066936480959e-06) [X3 Y4 Y6 X7]
+ (1.6923648061442469e-06) [Y4 Z5 Y6 Z10]
+ (1.6923648061442469e-06) [X4 Z5 X6 Z10]
+ (1.6923648061442469e-06) [Y5 Z6 Y7 Z11]
+ (1.6923648061442469e-06) [X5 Z6 X7 Z11]
+ (1.8540565404753448e-06) [Y4 Z5 Y6 Z7]
+ (1.8540565404753448e-06) [X4 Z5 X6 Z7]
+ (1.8781494287673992e-06) [Z4 Y10 Z11 Y12]
+ (1.8781494287673992e-06) [Z4 X10 Z11 X12]
+ (1.8781494287673992e-06) [Z5 Y11 Z12 Y13]
+ (1.8781494287673992e-06) [Z5 X11 Z12 X13]
+ (1.8818196302061718e-06) [Y4 Z5 Y6 Z9]
+ (1.8818196302061718e-06) [X4 Z5 X6 Z9]
+ (1.8818196302061718e-06) [Y5 Z6 Y7 Z8]
+ (1.8818196302061718e-06) [X5 Z6 X7 Z8]
+ (2.17263797731718e-06) [Y2 X3 X11 Y12]
+ (2.17263797731718e-06) [Y2 Y3 Y11 Y12]
+ (2.17263797731718e-06) [X2 X3 X11 X12]
+ (2.17263797731718e-06) [X2 Y3 Y11 X12]
+ (3.0992966542934157e-06) [Z0 Y4 Z5 Y6]
+ (3.0992966542934157e-06) [Z0 X4 Z5 X6]
+ (3.0992966542934157e-06) [Z1 Y5 Z6 Y7]
+ (3.0992966542934157e-06) [Z1 X5 Z6 X7]
+ (3.1585593146484826e-06) [Y2 Z3 Y4 Z10]
+ (3.1585593146484826e-06) [X2 Z3 X4 Z10]
+ (3.1585593146484826e-06) [Y3 Z4 Y5 Z11]
+ (3.1585593146484826e-06) [X3 Z4 X5 Z11]
+ (3.344023136064965e-06) [Z0 Y5 Z6 Y7]
+ (3.344023136064965e-06) [Z0 X5 Z6 X7]
+ (3.344023136064965e-06) [Z1 Y4 Z5 Y6]
+ (3.344023136064965e-06) [Z1 X4 Z5 X6]
+ (4.55647364903727e-06) [Y5 X6 X12 Y13]
+ (4.55647364903727e-06) [Y5 Y6 Y12 Y13]
+ (4.55647364903727e-06) [X5 X6 X12 X13]
+ (4.55647364903727e-06) [X5 Y6 Y12 X13]
+ (4.5887577068476e-06) [Y4 Z5 Y6 Z13]
+ (4.5887577068476e-06) [X4 Z5 X6 Z13]
+ (4.5887577068476e-06) [Y5 Z6 Y7 Z12]
+ (4.5887577068476e-06) [X5 Z6 X7 Z12]
+ (4.6429788880990655e-06) [Y3 X4 X10 Y11]
+ (4.6429788880990655e-06) [Y3 Y4 Y10 Y11]
+ (4.6429788880990655e-06) [X3 X4 X10 X11]
+ (4.6429788880990655e-06) [X3 Y4 Y10 X11]
+ (7.801538202749283e-06) [Y2 Z3 Y4 Z11]
+ (7.801538202749283e-06) [X2 Z3 X4 Z11]
+ (7.801538202749283e-06) [Y3 Z4 Y5 Z10]
+ (7.801538202749283e-06) [X3 Z4 X5 Z10]
+ (7.954224404280869e-06) [Y10 Z11 Y12 Z13]
+ (7.954224404280869e-06) [X10 Z11 X12 Z13]
+ (0.0002922256724535683) [Y7 X8 X9 Y10]
+ (0.0002922256724535683) [X7 Y8 Y9 X10]
+ (0.0004957972885841321) [Y2 Z4 Z5 Y6]
+ (0.0004957972885841321) [X2 Z4 Z5 X6]
+ (0.0011058984809132154) [Y0 Z1 Y2 Z5]
+ (0.0011058984809132154) [X0 Z1 X2 Z5]
+ (0.0011058984809132154) [Y1 Z2 Y3 Z4]
+ (0.0011058984809132154) [X1 Z2 X3 Z4]
+ (0.001663960658390677) [Y2 Z3 Z4 Y6]
+ (0.001663960658390677) [X2 Z3 Z4 X6]
+ (0.001663960658390677) [Y3 Z5 Z6 Y7]
+ (0.001663960658390677) [X3 Z5 Z6 X7]
+ (0.0017560659628911351) [Y0 Z1 Y2 Z11]
+ (0.0017560659628911351) [X0 Z1 X2 Z11]
+ (0.0017560659628911351) [Y1 Z2 Y3 Z10]
+ (0.0017560659628911351) [X1 Z2 X3 Z10]
+ (0.0023262348476252664) [Y0 Z1 Y2 Z13]
+ (0.0023262348476252664) [X0 Z1 X2 Z13]
+ (0.0023262348476252664) [Y1 Z2 Y3 Z12]
+ (0.0023262348476252664) [X1 Z2 X3 Z12]
+ (0.0027458273154241345) [Y0 X1 X4 Y5]
+ (0.0027458273154241345) [X0 Y1 Y4 X5]
+ (0.0029297682786026055) [Y0 Z1 Y2 Z9]
+ (0.0029297682786026055) [X0 Z1 X2 Z9]
+ (0.0029297682786026055) [Y1 Z2 Y3 Z8]
+ (0.0029297682786026055) [X1 Z2 X3 Z8]
+ (0.0032769650657663235) [Y0 Z1 Y2 Z3]
+ (0.0032769650657663235) [X0 Z1 X2 Z3]
+ (0.003347626470705396) [Y0 Z1 Y2 Z7]
+ (0.003347626470705396) [X0 Z1 X2 Z7]
+ (0.003347626470705396) [Y1 Z2 Y3 Z6]
+ (0.003347626470705396) [X1 Z2 X3 Z6]
+ (0.0035552589712739735) [Y0 Z1 Y2 Z10]
+ (0.0035552589712739735) [X0 Z1 X2 Z10]
+ (0.0035552589712739735) [Y1 Z2 Y3 Z11]
+ (0.0035552589712739735) [X1 Z2 X3 Z11]
+ (0.005143382387696295) [Y3 Y4 X5 X6]
+ (0.005143382387696295) [X3 X4 Y5 Y6]
+ (0.00528378575382988) [Y0 X1 X12 Y13]
+ (0.00528378575382988) [X0 Y1 Y12 X13]
+ (0.005530742149410212) [Y0 Z1 Y2 Z4]
+ (0.005530742149410212) [X0 Z1 X2 Z4]
+ (0.005530742149410212) [Y1 Z2 Y3 Z5]
+ (0.005530742149410212) [X1 Z2 X3 Z5]
+ (0.0060878368042359635) [Y8 X9 X12 Y13]
+ (0.0060878368042359635) [X8 Y9 Y12 X13]
+ (0.006509361234078718) [Y0 X1 X8 Y9]
+ (0.006509361234078718) [X0 Y1 Y8 X9]
+ (0.006888204542316916) [Y0 X1 X6 Y7]
+ (0.006888204542316916) [X0 Y1 Y6 X7]
+ (0.00690125003652055) [Y0 Z1 Y2 Z12]
+ (0.00690125003652055) [X0 Z1 X2 Z12]
+ (0.00690125003652055) [Y1 Z2 Y3 Z13]
+ (0.00690125003652055) [X1 Z2 X3 Z13]
+ (0.0071569205291897855) [Y4 X5 X8 Y9]
+ (0.0071569205291897855) [X4 Y5 Y8 X9]
+ (0.007731432847358214) [Y0 X1 X10 Y11]
+ (0.007731432847358214) [X0 Y1 Y10 X11]
+ (0.008032546697570413) [Y0 Z1 Y2 Z6]
+ (0.008032546697570413) [X0 Z1 X2 Z6]
+ (0.008032546697570413) [Y1 Z2 Y3 Z7]
+ (0.008032546697570413) [X1 Z2 X3 Z7]
+ (0.009560716496445583) [Y8 X9 X10 Y11]
+ (0.009560716496445583) [X8 Y9 Y10 X11]
+ (0.011055016688724951) [Y0 Z1 Y2 Z8]
+ (0.011055016688724951) [X0 Z1 X2 Z8]
+ (0.011055016688724951) [Y1 Z2 Y3 Z9]
+ (0.011055016688724951) [X1 Z2 X3 Z9]
+ (0.011285144618316839) [Y5 X6 X11 Y12]
+ (0.011285144618316839) [X5 Y6 Y11 X12]
+ (0.011307208029992369) [Y7 Z8 Z9 Y11]
+ (0.011307208029992369) [X7 Z8 Z9 X11]
+ (0.011982342583258088) [Y4 X5 X6 Y7]
+ (0.011982342583258088) [X4 Y5 Y6 X7]
+ (0.01387340006762584) [Y6 X7 X8 Y9]
+ (0.01387340006762584) [X6 Y7 Y8 X9]
+ (0.014583638327015865) [Y0 X1 X2 Y3]
+ (0.014583638327015865) [X0 Y1 Y2 X3]
+ (0.01557722707546399) [Y2 X3 X12 Y13]
+ (0.01557722707546399) [X2 Y3 Y12 X13]
+ (0.017366053264629366) [Y6 X7 X12 Y13]
+ (0.017366053264629366) [X6 Y7 Y12 X13]
+ (0.017680137657112755) [Y4 X5 X10 Y11]
+ (0.017680137657112755) [X4 Y5 Y10 X11]
+ (0.01782501193261339) [Y6 X7 X10 Y11]
+ (0.01782501193261339) [X6 Y7 Y10 X11]
+ (0.01902831871828794) [Y3 Y4 X11 X12]
+ (0.01902831871828794) [X3 X4 Y11 Y12]
+ (0.025384663696645434) [Y2 X3 X10 Y11]
+ (0.025384663696645434) [X2 Y3 Y10 X11]
+ (0.028685217310267793) [Y10 X11 X12 Y13]
+ (0.028685217310267793) [X10 Y11 Y12 X13]
+ (0.02981229960108375) [Y6 Z7 Z8 Y10]
+ (0.02981229960108375) [X6 Z7 Z8 X10]
+ (0.02981229960108375) [Y7 Z9 Z10 Y11]
+ (0.02981229960108375) [X7 Z9 Z10 X11]
+ (0.030104525273537318) [Y6 Z7 Z9 Y10]
+ (0.030104525273537318) [X6 Z7 Z9 X10]
+ (0.030104525273537318) [Y7 Z8 Z10 Y11]
+ (0.030104525273537318) [X7 Z8 Z10 X11]
+ (0.030787440718532143) [Y6 Z8 Z9 Y10]
+ (0.030787440718532143) [X6 Z8 Z9 X10]
+ (0.031143804196353403) [Y2 X3 X6 Y7]
+ (0.031143804196353403) [X2 Y3 Y6 X7]
+ (0.035839557185037646) [Y2 X3 X4 Y5]
+ (0.035839557185037646) [X2 Y3 Y4 X5]
+ (0.03619409348750321) [Y2 X3 X8 Y9]
+ (0.03619409348750321) [X2 Y3 Y8 X9]
+ (0.03831466937346745) [Y4 X5 X12 Y13]
+ (0.03831466937346745) [X4 Y5 Y12 X13]
+ (0.10433061485316271) [Z0 Y1 Z2 Y3]
+ (0.10433061485316271) [Z0 X1 Z2 X3]
+ (-0.12133242248372145) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133242248372145) [X2 Z3 Z4 Z5 X6]
+ (-0.12133242248372145) [Y3 Z4 Z5 Z6 Y7]
+ (-0.12133242248372145) [X3 Z4 Z5 Z6 X7]
+ (-3.204142235856605e-06) [Y1 Z2 Z3 Z4 Y5]
+ (-3.204142235856605e-06) [X1 Z2 Z3 Z4 X5]
+ (-3.204142235856603e-06) [Y0 Z1 Z2 Z3 Y4]
+ (-3.204142235856603e-06) [X0 Z1 Z2 Z3 X4]
+ (0.22847946311017167) [Y6 Z7 Z8 Z9 Y10]
+ (0.22847946311017167) [X6 Z7 Z8 Z9 X10]
+ (0.22847946311017167) [Y7 Z8 Z9 Z10 Y11]
+ (0.22847946311017167) [X7 Z8 Z9 Z10 X11]
+ (-0.03276748589570107) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.03276748589570107) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.03276748589570107) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.03276748589570107) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.027114878580474868) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.027114878580474868) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.027114878580474868) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.027114878580474868) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.02599620626717901) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.02599620626717901) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.024353136084500668) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (-0.024353136084500668) [Y2 Z3 Y4 X11 Z12 X13]
+ (-0.024353136084500668) [X2 Z3 X4 Y11 Z12 Y13]
+ (-0.024353136084500668) [X2 Z3 X4 X11 Z12 X13]
+ (-0.024353136084500668) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (-0.024353136084500668) [Y3 Z4 Y5 X10 Z11 X12]
+ (-0.024353136084500668) [X3 Z4 X5 Y10 Z11 Y12]
+ (-0.024353136084500668) [X3 Z4 X5 X10 Z11 X12]
+ (-0.02017582495698751) [Y4 Z5 Z6 X7 X11 Y12]
+ (-0.02017582495698751) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (-0.02017582495698751) [X4 Z5 Z6 X7 X11 X12]
+ (-0.02017582495698751) [X4 Z5 Z6 Y7 Y11 X12]
+ (-0.02017582495698751) [Y5 X6 X10 Z11 Z12 Y13]
+ (-0.02017582495698751) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (-0.02017582495698751) [X5 X6 X10 Z11 Z12 X13]
+ (-0.02017582495698751) [X5 Y6 Y10 Z11 Z12 X13]
+ (-0.017561116452758326) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.017561116452758326) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.017561116452758326) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.017561116452758326) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.015588277865109698) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (-0.015588277865109698) [X2 Z3 X4 X10 Z11 X12]
+ (-0.015588277865109698) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (-0.015588277865109698) [X3 Z4 X5 X11 Z12 X13]
+ (-0.014564473640811188) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.014564473640811188) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.014564473640811188) [X7 Z8 Z9 X10 X12 X13]
+ (-0.014564473640811188) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.011755995240392239) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.011755995240392239) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.011755995240392239) [X3 Z4 Z5 X6 X8 X9]
+ (-0.011755995240392239) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.010263460498896967) [Y2 Z3 X4 X10 Z11 Y12]
+ (-0.010263460498896967) [X2 Z3 Y4 Y10 Z11 X12]
+ (-0.010263460498896967) [Y3 Z4 X5 X11 Z12 Y13]
+ (-0.010263460498896967) [X3 Z4 Y5 Y11 Z12 X13]
+ (-0.008890680338670664) [Y4 Z5 X6 X10 Z11 Y12]
+ (-0.008890680338670664) [X4 Z5 Y6 Y10 Z11 X12]
+ (-0.008890680338670664) [Y5 Z6 X7 X11 Z12 Y13]
+ (-0.008890680338670664) [X5 Z6 Y7 Y11 Z12 X13]
+ (-0.008125248410122344) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.008125248410122344) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.00796083963422844) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (-0.00796083963422844) [X4 Z5 X6 X10 Z11 X12]
+ (-0.00796083963422844) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (-0.00796083963422844) [X5 Z6 X7 X11 Z12 X13]
+ (-0.007306763969603241) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.007306763969603241) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.007306763969603241) [X4 X5 X7 Z8 Z9 X10]
+ (-0.007306763969603241) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.00580512121236609) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.00580512121236609) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.00580512121236609) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.00580512121236609) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.005652607315226203) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.005652607315226203) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.005652607315226203) [X0 X1 X3 Z4 Z5 X6]
+ (-0.005652607315226203) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.005324817366212726) [Y2 Z3 Y4 X10 Z11 X12]
+ (-0.005324817366212726) [X2 Z3 X4 Y10 Z11 Y12]
+ (-0.005324817366212726) [Y3 Z4 Y5 X11 Z12 X13]
+ (-0.005324817366212726) [X3 Z4 X5 Y11 Z12 Y13]
+ (-0.005143382387696295) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.005143382387696295) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.005143382387696295) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.005143382387696295) [X2 Z3 X4 X5 Z6 X7]
+ (-0.004684920226865015) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.004684920226865015) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668615266018521) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.004668615266018521) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.004575015188895283) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.004575015188895283) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.004424843668496996) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.004424843668496996) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.0041588307164109484) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.0041588307164109484) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.0041588307164109484) [X3 Z4 Z5 X6 X12 X13]
+ (-0.0041588307164109484) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.0034938003715196475) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.0034938003715196475) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.0034938003715196475) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.0034938003715196475) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.002779040762872333) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.002779040762872333) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.0022939556230654697) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.0022939556230654697) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.0017991930083828414) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.0017991930083828414) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.0017278745823337723) [Y1 Z2 Z3 Y4 X11 X12]
+ (-0.0017278745823337723) [X1 Z2 Z3 X4 Y11 Y12]
+ (-0.000853383105100422) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.000853383105100422) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.0008144692870553296) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.0008144692870553296) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.0008144692870553296) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.0008144692870553296) [X3 Z4 Z5 Z6 X7 Z11]
+ (-7.735870606040646e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.735870606040646e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.735870606040646e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.735870606040646e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-6.52420442390049e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.52420442390049e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.52420442390049e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.52420442390049e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-5.974176779876715e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (-5.974176779876715e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (-5.275783343121386e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (-5.275783343121386e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (-4.6429788880990655e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (-4.6429788880990655e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (-4.55647364903727e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (-4.55647364903727e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (-3.7695835478614376e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.7695835478614376e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.694516852413565e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.694516852413565e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.6102421821804253e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.6102421821804253e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.6102421821804253e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.6102421821804253e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.334261856681106e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (-3.334261856681106e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (-3.3130170343587464e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.3130170343587464e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.2774382194237367e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.2774382194237367e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.2774382194237367e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.2774382194237367e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.2111873895400087e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.2111873895400087e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.2111873895400087e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.2111873895400087e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.1512959123694345e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (-3.1512959123694345e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (-3.1173663876224656e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (-3.1173663876224656e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (-3.0882456464704448e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.0882456464704448e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.17263797731718e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.17263797731718e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.4548066936480959e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (-1.4548066936480959e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (-1.3304568332952755e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (-1.3304568332952755e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (-1.2393113656699749e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (-1.2393113656699749e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (-1.2393113656699749e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (-1.2393113656699749e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (-1.2282691310729326e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (-1.2282691310729326e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (-1.035792485014602e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.035792485014602e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-9.306342789241215e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (-9.306342789241215e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (-9.306342789241215e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (-9.306342789241215e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (-7.956666941720319e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (-7.956666941720319e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (-6.628427114152048e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.628427114152048e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.579258861090108e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.579258861090108e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.579258861090108e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.579258861090108e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.395302272962752e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.395302272962752e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.395302272962752e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.395302272962752e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.627721913248176e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.627721913248176e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.287649366406294e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.287649366406294e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.287649366406294e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.287649366406294e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.287649366406294e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.287649366406294e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.287649366406294e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.287649366406294e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.837953346510182e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (-4.837953346510182e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (-3.5706354787241953e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (-3.5706354787241953e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (-3.328039627566886e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.328039627566886e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.2361834169954725e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (-3.2361834169954725e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (-3.2361834169954725e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (-3.2361834169954725e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (-2.4472648177162583e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (-2.4472648177162583e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (-2.1989637299401114e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (-2.1989637299401114e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (-1.8290428461134122e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (-1.8290428461134122e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (-1.8290428461134122e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (-1.8290428461134122e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (-1.107652906549004e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.107652906549004e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.107652906549004e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.107652906549004e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.107652906549004e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.107652906549004e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.107652906549004e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.107652906549004e-07) [X1 Z2 X3 X10 Z11 X12]
+ (-8.649129609041115e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (-8.649129609041115e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (-8.649129609041115e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (-8.649129609041115e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (-8.057465124343237e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (-8.057465124343237e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (-8.057465124343237e-08) [X1 Z2 Z3 X4 X10 X11]
+ (-8.057465124343237e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (1.0351511905152866e-09) [Y0 Z1 Z2 Z3 Y4 Z7]
+ (1.0351511905152866e-09) [X0 Z1 Z2 Z3 X4 Z7]
+ (1.0351511905152866e-09) [Y1 Z2 Z3 Z4 Y5 Z6]
+ (1.0351511905152866e-09) [X1 Z2 Z3 Z4 X5 Z6]
+ (1.8395658813558872e-08) [Y0 Z1 X2 X10 Z11 Y12]
+ (1.8395658813558872e-08) [X0 Z1 Y2 Y10 Z11 X12]
+ (1.8395658813558872e-08) [Y1 Z2 X3 X11 Z12 Y13]
+ (1.8395658813558872e-08) [X1 Z2 Y3 Y11 Z12 X13]
+ (2.270208477750356e-08) [Y0 Z1 Z2 X3 X5 Y6]
+ (2.270208477750356e-08) [Y0 Z1 Z2 Y3 Y5 Y6]
+ (2.270208477750356e-08) [X0 Z1 Z2 X3 X5 X6]
+ (2.270208477750356e-08) [X0 Z1 Z2 Y3 Y5 X6]
+ (2.270208477750356e-08) [Y1 X2 X4 Z5 Z6 Y7]
+ (2.270208477750356e-08) [Y1 Y2 Y4 Z5 Z6 Y7]
+ (2.270208477750356e-08) [X1 X2 X4 Z5 Z6 X7]
+ (2.270208477750356e-08) [X1 Y2 Y4 Z5 Z6 X7]
+ (2.5928188961402948e-08) [Y0 Z1 X2 X4 Z5 Y6]
+ (2.5928188961402948e-08) [X0 Z1 Y2 Y4 Z5 X6]
+ (2.5928188961402948e-08) [Y1 Z2 X3 X5 Z6 Y7]
+ (2.5928188961402948e-08) [X1 Z2 Y3 Y5 Z6 X7]
+ (1.291945822423662e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (1.291945822423662e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (1.3484969035398736e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (1.3484969035398736e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (1.3484969035398736e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (1.3484969035398736e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (1.3807579453733448e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (1.3807579453733448e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (1.3807579453733448e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (1.3807579453733448e-07) [X0 Z1 X2 X5 Z6 X7]
+ (1.3807579453733448e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (1.3807579453733448e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (1.3807579453733448e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (1.3807579453733448e-07) [X1 Z2 X3 X4 Z5 X6]
+ (1.6077787931489225e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (1.6077787931489225e-07) [X0 Z1 X2 X4 Z5 X6]
+ (1.6077787931489225e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (1.6077787931489225e-07) [X1 Z2 X3 X5 Z6 X7]
+ (1.839394358018294e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (1.839394358018294e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (1.839394358018294e-07) [X1 Z2 Z3 X4 X6 X7]
+ (1.839394358018294e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (1.933212137190972e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (1.933212137190972e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (1.933212137190972e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (1.933212137190972e-07) [X0 Z1 X2 X3 Z4 X5]
+ (2.1989637299401114e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (2.1989637299401114e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (2.371270456091361e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (2.371270456091361e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (2.371270456091361e-07) [X1 Z2 Z3 X4 X8 X9]
+ (2.371270456091361e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (2.4472648177162583e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (2.4472648177162583e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (3.0867708674611765e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (3.0867708674611765e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (3.0867708674611765e-07) [X1 Z2 Z3 X4 X12 X13]
+ (3.0867708674611765e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (3.328039627566886e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.328039627566886e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.5706354787241953e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (3.5706354787241953e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (4.837953346510182e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (4.837953346510182e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (5.627721913248176e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.627721913248176e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (5.927350068632854e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (5.927350068632854e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (5.927350068632854e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (5.927350068632854e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (6.628427114152048e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.628427114152048e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (6.733096581065145e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (6.733096581065145e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (6.733096581065145e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (6.733096581065145e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (7.956666941720319e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (7.956666941720319e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (1.035792485014602e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.035792485014602e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.2282691310729326e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (1.2282691310729326e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (1.3304568332952755e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (1.3304568332952755e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (1.4548066936480959e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (1.4548066936480959e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (2.17263797731718e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.17263797731718e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.0882456464704448e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.0882456464704448e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.1512959123694345e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (3.1512959123694345e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (3.3130170343587464e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.3130170343587464e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.694516852413565e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.694516852413565e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (4.183808694172009e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.183808694172009e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.253118605114948e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (4.253118605114948e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (4.55647364903727e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (4.55647364903727e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (4.6429788880990655e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (4.6429788880990655e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (5.275783343121386e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (5.275783343121386e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (5.974176779876715e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (5.974176779876715e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (6.290019515683282e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (6.290019515683282e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (6.290019515683282e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (6.290019515683282e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (7.444267233188166e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (7.444267233188166e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (7.444267233188166e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (7.444267233188166e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (7.518288646756648e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (7.518288646756648e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (7.518288646756648e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (7.518288646756648e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (8.774724066483441e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (8.774724066483441e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (8.774724066483441e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (8.774724066483441e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (0.0002922256724535683) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.0002922256724535683) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.0002922256724535683) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.0002922256724535683) [X6 Z7 X8 X9 Z10 X11]
+ (0.0004957972885841321) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.0004957972885841321) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.0006650303448913022) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.0006650303448913022) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.0006650303448913022) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.0006650303448913022) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.000853383105100422) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.000853383105100422) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.0009298407044422185) [Y4 Z5 Y6 X10 Z11 X12]
+ (0.0009298407044422185) [X4 Z5 X6 Y10 Z11 Y12]
+ (0.0009298407044422185) [Y5 Z6 Y7 X11 Z12 X13]
+ (0.0009298407044422185) [X5 Z6 X7 Y11 Z12 Y13]
+ (0.0016095335163021148) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.0016095335163021148) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.0016095335163021148) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.0016095335163021148) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.001667613749974078) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.001667613749974078) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.001667613749974078) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.001667613749974078) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.0017278745823337723) [Y1 Z2 Z3 X4 X11 Y12]
+ (0.0017278745823337723) [X1 Z2 Z3 Y4 Y11 X12]
+ (0.0017991930083828414) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.0017991930083828414) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.0022939556230654697) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.0022939556230654697) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.0024629166214025365) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.0024629166214025365) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.0024629166214025365) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.0024629166214025365) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.003961569373039547) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.003961569373039547) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.003961569373039547) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.003961569373039547) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.004424843668496996) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.004424843668496996) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.004575015188895283) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.004575015188895283) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.004668615266018521) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.004668615266018521) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.004684920226865015) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.004684920226865015) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.005368616111413834) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.005368616111413834) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.005368616111413834) [X2 X3 X7 Z8 Z9 X10]
+ (0.005368616111413834) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.008125248410122344) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.008125248410122344) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.008764858219390968) [Y2 Z3 Z4 X5 X11 Y12]
+ (0.008764858219390968) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (0.008764858219390968) [X2 Z3 Z4 X5 X11 X12]
+ (0.008764858219390968) [X2 Z3 Z4 Y5 Y11 X12]
+ (0.008764858219390968) [Y3 X4 X10 Z11 Z12 Y13]
+ (0.008764858219390968) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (0.008764858219390968) [X3 X4 X10 Z11 Z12 X13]
+ (0.008764858219390968) [X3 Y4 Y10 Z11 Z12 X13]
+ (0.010540434329207277) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.010540434329207277) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.010540434329207277) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.010540434329207277) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.010959994608905595) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.010959994608905595) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.010959994608905595) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.010959994608905595) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.011307208029992369) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.011307208029992369) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.012214985322759064) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (0.012214985322759064) [Y4 Z5 Y6 X11 Z12 X13]
+ (0.012214985322759064) [X4 Z5 X6 Y11 Z12 Y13]
+ (0.012214985322759064) [X4 Z5 X6 X11 Z12 X13]
+ (0.012214985322759064) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (0.012214985322759064) [Y5 Z6 Y7 X10 Z11 X12]
+ (0.012214985322759064) [X5 Z6 X7 Y10 Z11 Y12]
+ (0.012214985322759064) [X5 Z6 X7 X10 Z11 X12]
+ (0.014411189770052613) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.014411189770052613) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.014411189770052613) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.014411189770052613) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.015225659057107953) [Y3 Z4 Z5 X6 X10 Y11]
+ (0.015225659057107953) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (0.015225659057107953) [X3 Z4 Z5 X6 X10 X11]
+ (0.015225659057107953) [X3 Z4 Z5 Y6 Y10 X11]
+ (0.018266758578508836) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.018266758578508836) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.018266758578508836) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.018266758578508836) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.019020373875113836) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.019020373875113836) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.019020373875113836) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.019020373875113836) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.024388989986527672) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.024388989986527672) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.024388989986527672) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.024388989986527672) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.025104907970018464) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.025104907970018464) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.025104907970018464) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.025104907970018464) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.030787440718532143) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.030787440718532143) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.045879424030717646) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.045879424030717646) [X0 Z2 Z3 Z4 Z5 X6]
+ (0.0560071356168713) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.0560071356168713) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.0560071356168713) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.0560071356168713) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.0560844943229317) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.0560844943229317) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.0560844943229317) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.0560844943229317) [Z1 X7 Z8 Z9 Z10 X11]
+ (-2.5950813032859262e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.5950813032859262e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.5950813032859262e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.5950813032859262e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.631261791628445e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (6.631261791628445e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (6.631261791628445e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (6.631261791628445e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.04274326006297237) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04274326006297237) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (0.04274326006297238) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.04274326006297238) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (-0.03935925039149668) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.03935925039149668) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.03935925039149668) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.03935925039149668) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.03560840035225662) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.03560840035225662) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.02990381345824404) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.02990381345824404) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.02990381345824404) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.02990381345824404) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.028730798001222805) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.028730798001222805) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.028730798001222805) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.028730798001222805) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.02475550798076639) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.02475550798076639) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.02475550798076639) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.02475550798076639) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.024282031623352362) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.024282031623352362) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.02143398011687494) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.02143398011687494) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.02143398011687494) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.02143398011687494) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.01925745300187425) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.01925745300187425) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.01902831871828794) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.01902831871828794) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.01888899507740559) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-0.01888899507740559) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-0.01888899507740559) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.01888899507740559) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.01602466609551723) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (-0.01602466609551723) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (-0.015225659057107953) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.015225659057107953) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.014603742410730286) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.014603742410730286) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.014564473640811188) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.014564473640811188) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.011755995240392239) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.011755995240392239) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.011285144618316839) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-0.011285144618316839) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-0.009841802923817214) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.009841802923817214) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.008469833341369103) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.008469833341369103) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.007306763969603241) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.007306763969603241) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005923799555611916) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-0.005923799555611916) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-0.005652607315226203) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.005652607315226203) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.005368616111413834) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005368616111413834) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.0041588307164109484) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.0041588307164109484) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.003989845257552255) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (-0.003989845257552255) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (-0.003989845257552255) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (-0.003989845257552255) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (-0.003356667921369303) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.003356667921369303) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.003356667921369303) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.003356667921369303) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.003267514897154917) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.003267514897154917) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.003267514897154917) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.003267514897154917) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.002779040762872333) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.002779040762872333) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.0022939556230654697) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939556230654697) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.0022939556230654697) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939556230654697) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.002261970675218483) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (-0.002261970675218483) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (-0.002261970675218483) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (-0.002261970675218483) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (-0.002261970675218483) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (-0.002261970675218483) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (-0.002261970675218483) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (-0.002261970675218483) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (-0.001303802982459615) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (-0.001303802982459615) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (-0.001303802982459615) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (-0.001303802982459615) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (-0.0002464408105730951) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0002464408105730951) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.0001383860370128315) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.0001383860370128315) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.0001383860370128315) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.0001383860370128315) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.735870606040646e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.735870606040646e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-6.652106215153923e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.652106215153923e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.652106215153923e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.652106215153923e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481751872537706e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.481751872537706e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481751872537706e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.481751872537706e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.7695835478614376e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.7695835478614376e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.5443573457548394e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.5443573457548394e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.5443573457548394e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.5443573457548394e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.5443573457548394e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.5443573457548394e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.5443573457548394e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.5443573457548394e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.334261856681106e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-3.334261856681106e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-3.2117638196904738e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (-3.2117638196904738e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (-3.2117638196904738e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (-3.2117638196904738e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (-3.2117638196904738e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (-3.2117638196904738e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (-3.2117638196904738e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (-3.2117638196904738e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (-2.103163449961509e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.103163449961509e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.103163449961509e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.103163449961509e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0110739801157667e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.0110739801157667e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0110739801157667e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.0110739801157667e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9429465569348325e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.9429465569348325e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9429465569348325e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9429465569348325e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.689305643508797e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (-1.689305643508797e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (-1.689305643508797e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (-1.689305643508797e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (-1.6540900340334189e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.6540900340334189e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6540900340334189e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.6540900340334189e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.628837736186023e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (-1.628837736186023e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (-1.628837736186023e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (-1.628837736186023e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (-1.3304568332952755e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (-1.3304568332952755e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (-1.3304568332952755e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (-1.3304568332952755e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (-7.988467848300687e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.988467848300687e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.988467848300687e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.988467848300687e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.189870300618198e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.189870300618198e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.175164781266148e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175164781266148e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.471605954546118e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.471605954546118e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.5611169928883987e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.5611169928883987e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.5611169928883987e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.5611169928883987e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.523339242506452e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (-4.523339242506452e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (-3.4273508554697515e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.4273508554697515e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.4273508554697515e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.4273508554697515e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.328039627566886e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328039627566886e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.328039627566886e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328039627566886e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.0867708674611765e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (-3.0867708674611765e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (-2.8885652290228103e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8885652290228103e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8885652290228103e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.8885652290228103e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.371270456091361e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (-2.371270456091361e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (-1.839394358018294e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (-1.839394358018294e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (-8.057465124343237e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (-8.057465124343237e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (-6.772951752439347e-08) [Y1 Z2 Z3 Y4 X7 Z8 Z9 X10]
+ (-6.772951752439347e-08) [X1 Z2 Z3 X4 Y7 Z8 Z9 Y10]
+ (-6.046790732147297e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (-6.046790732147297e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (-6.046790732147297e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (-6.046790732147297e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (-3.226104183868895e-09) [Y0 Z1 Z2 Y3 X4 Z5 Z6 X7]
+ (-3.226104183868895e-09) [X0 Z1 Z2 X3 Y4 Z5 Z6 Y7]
+ (3.226104183868895e-09) [Y0 Z1 Z2 X3 X4 Z5 Z6 Y7]
+ (3.226104183868895e-09) [X0 Z1 Z2 Y3 Y4 Z5 Z6 X7]
+ (6.772951752439347e-08) [Y1 Z2 Z3 X4 X7 Z8 Z9 Y10]
+ (6.772951752439347e-08) [X1 Z2 Z3 Y4 Y7 Z8 Z9 X10]
+ (8.057465124343237e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (8.057465124343237e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (9.208946984585065e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.208946984585065e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.208946984585065e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.208946984585065e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.703543426162214e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.703543426162214e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.703543426162214e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.703543426162214e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.839394358018294e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (1.839394358018294e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (2.371270456091361e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (2.371270456091361e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (3.0867708674611765e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (3.0867708674611765e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (4.523339242506452e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (4.523339242506452e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (5.471605954546118e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.471605954546118e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.175164781266148e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175164781266148e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.189870300618198e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.189870300618198e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (7.867608476930313e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (7.867608476930313e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (7.867608476930313e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (7.867608476930313e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (1.2282691310729326e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (1.2282691310729326e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (1.2282691310729326e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (1.2282691310729326e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (1.5224581761824358e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (1.5224581761824358e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (1.5224581761824358e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (1.5224581761824358e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (1.5224581761824358e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (1.5224581761824358e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (1.5224581761824358e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (1.5224581761824358e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (2.360947163619208e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (2.360947163619208e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (2.360947163619208e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (2.360947163619208e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (2.7455105609267222e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.7455105609267222e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.7455105609267222e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.7455105609267222e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.7455105609267222e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.7455105609267222e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.7455105609267222e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.7455105609267222e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.3130170343587464e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.3130170343587464e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.3130170343587464e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.3130170343587464e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (4.183808694172009e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.183808694172009e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (4.253118605114948e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.253118605114948e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.728781309392956e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (4.728781309392956e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (4.728781309392956e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.728781309392956e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.73457824733646e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (4.73457824733646e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (4.73457824733646e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (4.73457824733646e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (5.071403543719863e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (5.071403543719863e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (5.071403543719863e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (5.071403543719863e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (7.0897284730123806e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.0897284730123806e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (7.0897284730123806e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (7.0897284730123806e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (9.805981791059792e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (9.805981791059792e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (9.805981791059792e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (9.805981791059792e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (1.5316613783800424e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (1.5316613783800424e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (1.5316613783800424e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (1.5316613783800424e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.6103374631493496e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (1.6103374631493496e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.6103374631493496e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (1.6103374631493496e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (7.735870606040646e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.735870606040646e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.0002464408105730951) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (0.0002464408105730951) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (0.0004458488204513146) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.0004458488204513146) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.0004458488204513146) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.0004458488204513146) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.0005940157673954444) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940157673954444) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.0005940157673954444) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940157673954444) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.0005940157673954444) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940157673954444) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.0005940157673954444) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940157673954444) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.000853383105100422) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.000853383105100422) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.000853383105100422) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.000853383105100422) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.0009581676927588676) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (0.0009581676927588676) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (0.0009581676927588676) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (0.0009581676927588676) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (0.0009581676927588676) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (0.0009581676927588676) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (0.0009581676927588676) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (0.0009581676927588676) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (0.0010435237104171355) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.0010435237104171355) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.0010435237104171355) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.0010435237104171355) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.0012803055951353239) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.0012803055951353239) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.0012803055951353239) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.0012803055951353239) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.00268604227509264) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (0.00268604227509264) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (0.00268604227509264) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (0.00268604227509264) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (0.0041588307164109484) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.0041588307164109484) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.004311038607572053) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.004311038607572053) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.004311038607572053) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.004311038607572053) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.004636973516504626) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.004636973516504626) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.004636973516504626) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.004636973516504626) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.005114464086469835) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.005114464086469835) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.005114464086469835) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.005114464086469835) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.005114464086469835) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114464086469835) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114464086469835) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005114464086469835) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.005241543597048513) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (0.005241543597048513) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (0.005241543597048513) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (0.005241543597048513) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (0.005262631033413966) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.005262631033413966) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.005262631033413966) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.005262631033413966) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.005368616111413834) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005368616111413834) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005379929634061343) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.005379929634061343) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.005379929634061343) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.005379929634061343) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.005652607315226203) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.005652607315226203) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.005708479853865279) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.005708479853865279) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.005708479853865279) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.005708479853865279) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.005923799555611916) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (0.005923799555611916) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (0.007306763969603241) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.007306763969603241) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.008469833341369103) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.008469833341369103) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.009612546714446673) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (0.009612546714446673) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (0.009612546714446673) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (0.009612546714446673) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (0.009841802923817214) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.009841802923817214) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.011285144618316839) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (0.011285144618316839) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (0.011755995240392239) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.011755995240392239) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.014564473640811188) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.014564473640811188) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.014603742410730286) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.014603742410730286) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.015225659057107953) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.015225659057107953) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.01602466609551723) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (0.01602466609551723) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (0.01902831871828794) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.01902831871828794) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.01925745300187425) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (0.01925745300187425) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (0.022528354240870095) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.022528354240870095) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.02314522165328449) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (0.02314522165328449) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (0.025637212809963904) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (0.025637212809963904) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (0.025637212809963904) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (0.025637212809963904) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (0.03931810723103826) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.03931810723103826) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.03931810723103826) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.03931810723103826) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.03956454804161136) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (0.03956454804161136) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (0.03956454804161136) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.03956454804161136) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0417188140445597) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (0.0417188140445597) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (0.0417188140445597) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (0.0417188140445597) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (0.045879424030717646) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.045879424030717646) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (0.04764261360017163) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (0.04764261360017163) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (0.04764261360017163) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (0.04764261360017163) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.2816433575339883) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.2816433575339883) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.2816433575339882) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.2816433575339882) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.3693713755446293) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.3693713755446293) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.36937137554462934) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.36937137554462934) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.07635036936743313) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07635036936743313) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07635036936743313) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07635036936743313) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752398179963794) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.06752398179963794) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752398179963794) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.06752398179963794) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03560840035225662) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.03560840035225662) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.024282031623352362) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.024282031623352362) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.019538085344919805) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.019538085344919805) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.019538085344919805) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.019538085344919805) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.01929949985801787) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.01929949985801787) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.01929949985801787) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.01929949985801787) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.01929949985801787) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.01929949985801787) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.01929949985801787) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.01929949985801787) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.017091621922236375) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.017091621922236375) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.017091621922236375) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.017091621922236375) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.01075752420121564) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.01075752420121564) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.01075752420121564) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.01075752420121564) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010715477345072026) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010715477345072026) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010715477345072026) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010715477345072026) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.009841802923817214) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841802923817214) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.009841802923817214) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841802923817214) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826387567795186) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826387567795186) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826387567795186) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008826387567795186) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008469833341369103) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469833341369103) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.008469833341369103) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469833341369103) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.005923799555611916) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-0.005923799555611916) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-0.005923799555611916) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-0.005923799555611916) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-0.004668615266018521) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.004668615266018521) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.003876482195724052) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.003876482195724052) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.003484154579345313) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003484154579345313) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003356667921369303) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.003356667921369303) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.003267514897154917) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.003267514897154917) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.002141348964731147) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.002141348964731147) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.0017278745823337723) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.0017278745823337723) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.0016407591167089798) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0016407591167089798) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528842553178521) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0014528842553178521) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528842553178521) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0014528842553178521) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870893705579123) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007870893705579123) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.000519292412050978) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.000519292412050978) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.0002464408105730951) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0002464408105730951) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0002464408105730951) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0002464408105730951) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00019401030606343647) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00019401030606343647) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.0001383860370128315) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.0001383860370128315) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.141566358416856e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.141566358416856e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.141566358416856e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.141566358416856e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.204685614361565e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-4.204685614361565e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-4.204685614361565e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.204685614361565e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-5.071403543719863e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-5.071403543719863e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-3.1512959123694345e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (-3.1512959123694345e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (-3.0882456464704448e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.0882456464704448e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.9884125145179796e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.9884125145179796e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8742485071453494e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.8742485071453494e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.360947163619208e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-2.360947163619208e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.3001958555523482e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.3001958555523482e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-8.398527490255489e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.398527490255489e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.398527490255489e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.398527490255489e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.027844033078529e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.027844033078529e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.027844033078529e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.027844033078529e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.867608476930313e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867608476930313e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.5605537577998e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.5605537577998e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.5605537577998e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.5605537577998e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.5605537577998e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.5605537577998e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.5605537577998e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.5605537577998e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.996951698647234e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.996951698647234e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.996951698647234e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.996951698647234e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.996951698647234e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.996951698647234e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.996951698647234e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.996951698647234e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-4.769457031508573e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (-4.769457031508573e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (-4.769457031508573e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (-4.769457031508573e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (-4.4490566527130834e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (-4.4490566527130834e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (-4.4490566527130834e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (-4.4490566527130834e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (-4.092161856267349e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (-4.092161856267349e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (-4.092161856267349e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (-4.092161856267349e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (-4.092161856267349e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (-4.092161856267349e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (-4.092161856267349e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (-4.092161856267349e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (-2.8885652290228103e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.8885652290228103e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.686321313373377e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.686321313373377e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.703543426162214e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.703543426162214e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.208946984585065e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.208946984585065e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379737324591108e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379737324591108e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379737324591108e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379737324591108e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379737324591108e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379737324591108e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.379737324591108e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379737324591108e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.7068345717626515e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.7068345717626515e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.7068345717626515e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.7068345717626515e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.568947964514941e-08) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-3.568947964514941e-08) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-3.568947964514941e-08) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-3.568947964514941e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-3.568947964514941e-08) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-3.568947964514941e-08) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.568947964514941e-08) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.568947964514941e-08) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.2040037879261005e-08) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (3.2040037879261005e-08) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (3.2040037879261005e-08) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (3.2040037879261005e-08) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (9.208946984585065e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.208946984585065e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.071684478751663e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.071684478751663e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.071684478751663e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.071684478751663e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.1782130826139666e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.1782130826139666e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.1782130826139666e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.1782130826139666e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.703543426162214e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.703543426162214e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.249897561366646e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.249897561366646e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.249897561366646e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.249897561366646e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.686321313373377e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.686321313373377e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.8885652290228103e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.8885652290228103e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.3766863085345694e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (3.3766863085345694e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (3.3766863085345694e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (3.3766863085345694e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (3.3766863085345694e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (3.3766863085345694e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (3.3766863085345694e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (3.3766863085345694e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (3.568200414554334e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (3.568200414554334e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (3.568200414554334e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (3.568200414554334e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (7.246849260020115e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.246849260020115e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.246849260020115e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.246849260020115e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.246849260020115e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.246849260020115e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.246849260020115e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.246849260020115e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.867608476930313e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (7.867608476930313e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.900025551037972e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (7.900025551037972e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (7.900025551037972e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (7.900025551037972e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (8.091539657058245e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (8.091539657058245e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (8.091539657058245e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (8.091539657058245e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (8.091539657058245e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (8.091539657058245e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (8.091539657058245e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (8.091539657058245e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.1468225965593695e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (1.1468225965593695e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (1.1468225965593695e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (1.1468225965593695e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.3001958555523482e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.3001958555523482e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.360947163619208e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.360947163619208e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.8742485071453494e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.8742485071453494e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.883653140352098e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.883653140352098e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.9473314043636928e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9473314043636928e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.9473314043636928e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9473314043636928e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9884125145179796e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9884125145179796e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.0882456464704448e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.0882456464704448e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.1512959123694345e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (3.1512959123694345e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (3.846190824844699e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.846190824844699e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.846190824844699e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.846190824844699e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.071403543719863e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (5.071403543719863e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (5.105462343542831e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.105462343542831e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.105462343542831e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.105462343542831e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.146386680398782e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.146386680398782e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.146386680398782e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.146386680398782e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.159294408706204e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.159294408706204e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.159294408706204e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.159294408706204e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.427926540046957e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.427926540046957e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.427926540046957e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.427926540046957e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.935743918881239e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.935743918881239e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.935743918881239e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.935743918881239e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253185045218657e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.253185045218657e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.97971085068818e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.97971085068818e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.97971085068818e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.97971085068818e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (0.0001383860370128315) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.0001383860370128315) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.00018787486139112784) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.00018787486139112784) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.00018787486139112784) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.00018787486139112784) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00019401030606343647) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (0.00019401030606343647) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (0.000519292412050978) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.000519292412050978) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.0007156737069737439) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007156737069737439) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0007156737069737439) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007156737069737439) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870893705579123) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007870893705579123) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.0015324885626721556) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.0015324885626721556) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.0015324885626721556) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.0015324885626721556) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.0016407591167089798) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0016407591167089798) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0017278745823337723) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.0017278745823337723) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.0024464634226834285) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.0024464634226834285) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.0024464634226834285) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.0024464634226834285) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.003267514897154917) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.003267514897154917) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.003356667921369303) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.003356667921369303) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.003484154579345313) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.003484154579345313) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.003804063154369996) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.003804063154369996) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.003804063154369996) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.003804063154369996) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.003876482195724052) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.003876482195724052) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.004668615266018521) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.004668615266018521) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.004767276644736062) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.004767276644736062) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.004767276644736062) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.004767276644736062) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.00528656905678704) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.00528656905678704) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.00528656905678704) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.00528656905678704) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.005408970758396208) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.005408970758396208) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.005408970758396208) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.005408970758396208) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.008541975656802234) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.008541975656802234) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (0.008541975656802234) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (0.008541975656802234) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.008541975656802234) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.008541975656802234) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.008541975656802234) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.008541975656802234) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01031147218241722) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01031147218241722) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01031147218241722) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.01031147218241722) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.014603742410730286) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.014603742410730286) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.014603742410730286) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.014603742410730286) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.016024666095517233) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (0.016024666095517233) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (0.016024666095517233) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (0.016024666095517233) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (0.022528354240870095) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.022528354240870095) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.02314522165328449) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (0.02314522165328449) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (0.02459183208828822) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.02459183208828822) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.02459183208828822) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.02459183208828822) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.03490330427070544) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.03490330427070544) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.03490330427070544) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.03490330427070544) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.05859215179550917) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.05859215179550917) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.08684736029521722) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.08684736029521722) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.08684736029521722) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.08684736029521722) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.09065142344958721) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.09065142344958721) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.09065142344958721) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.09065142344958721) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.775872003458119e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.775872003458119e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.775872003458119e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.775872003458119e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.07165056250051129) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.07165056250051129) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.07165056250051131) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.07165056250051131) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.01925745300187425) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.01925745300187425) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.01031147218241722) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.01031147218241722) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008826387567795184) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826387567795184) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.003804063154369996) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.003804063154369996) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0029841800747881946) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.0029841800747881946) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.0029841800747881946) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.0029841800747881946) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.0024464634226834285) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.0024464634226834285) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.0023949671540613423) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0023949671540613423) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0023949671540613423) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0023949671540613423) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0023949671540613423) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0023949671540613423) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0023949671540613423) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0023949671540613423) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0022009568479979054) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0022009568479979054) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0022009568479979054) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0022009568479979054) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002141348964731147) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.002141348964731147) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.0016407591167089798) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407591167089798) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0016407591167089798) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407591167089798) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.001172629784239294) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.001172629784239294) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.001172629784239294) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.001172629784239294) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1462850765425207e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1462850765425207e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.8742485071453494e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742485071453494e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.8742485071453494e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742485071453494e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.3001958555523482e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.3001958555523482e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.3001958555523482e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.3001958555523482e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.0444741427588512e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.0444741427588512e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.0444741427588512e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.0444741427588512e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.95590338953719e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.95590338953719e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.95590338953719e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.95590338953719e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105340949055777e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105340949055777e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105340949055777e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105340949055777e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661200188873785e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661200188873785e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661200188873785e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661200188873785e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.54020410589679e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.54020410589679e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.189870300618198e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.189870300618198e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.876530270620606e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876530270620606e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876530270620606e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876530270620606e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175164781266148e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175164781266148e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.523339242506452e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.523339242506452e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.076662688867508e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.076662688867508e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.076662688867508e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.076662688867508e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.013398778643917e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.013398778643917e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.904537321691723e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.904537321691723e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.904537321691723e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.904537321691723e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.666679777955e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.666679777955e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.666679777955e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.666679777955e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505624404840902e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.8505624404840902e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.846699182575165e-08) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-7.846699182575165e-08) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-6.772951752439347e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-6.772951752439347e-08) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-4.0998291091206765e-08) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-4.0998291091206765e-08) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.0998291091206765e-08) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.0998291091206765e-08) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (6.772951752439347e-08) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (6.772951752439347e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (7.846699182575165e-08) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (7.846699182575165e-08) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (1.657009301970018e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.657009301970018e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.657009301970018e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.657009301970018e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.8505624404840902e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.8505624404840902e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.686321313373377e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.686321313373377e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.686321313373377e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.686321313373377e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013398778643917e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.013398778643917e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.523339242506452e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.523339242506452e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.670408080615087e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.670408080615087e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.670408080615087e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.670408080615087e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175164781266148e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175164781266148e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.189870300618198e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.189870300618198e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.54020410589679e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.54020410589679e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949307521847788e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949307521847788e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.7924637971542174e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924637971542174e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924637971542174e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.7924637971542174e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.883653140352098e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.883653140352098e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.9884125145179796e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9884125145179796e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9884125145179796e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9884125145179796e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253185045218657e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.253185045218657e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4016916201534368e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4016916201534368e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4016916201534368e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4016916201534368e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.580937999868857e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.580937999868857e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.580937999868857e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.580937999868857e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.000519292412050978) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.000519292412050978) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.000519292412050978) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.000519292412050978) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.0007870893705579124) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870893705579124) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870893705579124) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870893705579124) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0010283270637586124) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0010283270637586124) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0010283270637586124) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0010283270637586124) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012223373698220486) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.0012223373698220486) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (0.0012223373698220486) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (0.0012223373698220486) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0012223373698220486) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0012223373698220486) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0012223373698220486) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012223373698220486) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012366559235610919) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.0012366559235610919) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.0012366559235610919) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.0012366559235610919) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.0018638931390713486) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0018638931390713486) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0018638931390713486) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0018638931390713486) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0018638931390713486) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0018638931390713486) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0018638931390713486) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0018638931390713486) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0022494140606723586) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0022494140606723586) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0022494140606723586) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0022494140606723586) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0024464634226834285) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.0024464634226834285) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.003804063154369996) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.003804063154369996) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0038764821957240517) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.0038764821957240517) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.0038764821957240517) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.0038764821957240517) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.004220835998349285) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.004220835998349285) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.004220835998349285) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.004220835998349285) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.00534804771841666) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.00534804771841666) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.00534804771841666) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00534804771841666) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.005733568640017672) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.005733568640017672) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.005733568640017672) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.005733568640017672) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.005733568640017672) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.005733568640017672) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.005733568640017672) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.005733568640017672) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.007597461779089021) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.007597461779089021) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.007597461779089021) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.007597461779089021) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.008826387567795184) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.008826387567795184) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.01031147218241722) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01031147218241722) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01925745300187425) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01925745300187425) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.05859215179550918) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.05859215179550918) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.3986653295385059e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.3986653295385059e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.3986653295385055e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.3986653295385055e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003484154579345313) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.003484154579345313) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.002984180074788194) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.002984180074788194) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.00019401030606343647) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00019401030606343647) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1462850765425207e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1462850765425207e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.7924637971542174e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7924637971542174e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.54020410589679e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.54020410589679e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.54020410589679e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.54020410589679e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.8505624404840902e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505624404840902e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505624404840902e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505624404840902e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.846699182575165e-08) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-7.846699182575165e-08) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-7.846699182575165e-08) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-7.846699182575165e-08) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-4.0998291091206765e-08) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-4.0998291091206765e-08) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.0998291091206765e-08) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-4.0998291091206765e-08) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013398778643917e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013398778643917e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013398778643917e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013398778643917e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.949307521847788e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.949307521847788e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.7924637971542174e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7924637971542174e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019401030606343647) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00019401030606343647) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002984180074788194) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.002984180074788194) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.003484154579345313) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.003484154579345313) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
  (-73.13873149596098) [I0]
+ (-0.180667575069893) [Z6]
+ (-0.18066757506989284) [Z7]
+ (-0.15961443583713358) [Z4]
+ (-0.1596144358371335) [Z5]
+ (0.1741998661212935) [Z2]
+ (0.17419986612129354) [Z3]
+ (0.2275732881450031) [Z0]
+ (0.22757328814500333) [Z1]
+ (-7.95422441813437e-06) [Y5 Y7]
+ (-7.95422441813437e-06) [X5 X7]
+ (8.194104917598477e-06) [Y4 Y6]
+ (8.194104917598477e-06) [X4 X6]
+ (0.11270381859115759) [Z4 Z6]
+ (0.11270381859115759) [Z5 Z7]
+ (0.11952441016894327) [Z0 Z4]
+ (0.11952441016894327) [Z1 Z5]
+ (0.13401737372652692) [Z0 Z6]
+ (0.13401737372652692) [Z1 Z7]
+ (0.13734942210156964) [Z0 Z5]
+ (0.13734942210156964) [Z1 Z4]
+ (0.13766859133611448) [Z2 Z4]
+ (0.13766859133611448) [Z3 Z5]
+ (0.14138903590143162) [Z4 Z7]
+ (0.14138903590143162) [Z5 Z6]
+ (0.14722930783255536) [Z2 Z5]
+ (0.14722930783255536) [Z3 Z4]
+ (0.14926347060038597) [Z4 Z5]
+ (0.14973497005423547) [Z2 Z6]
+ (0.14973497005423547) [Z3 Z7]
+ (0.15138342699113738) [Z0 Z7]
+ (0.15138342699113738) [Z1 Z6]
+ (0.15435760065294096) [Z6 Z7]
+ (0.15582280685846384) [Z2 Z7]
+ (0.15582280685846384) [Z3 Z6]
+ (0.1675666935617041) [Z0 Z2]
+ (0.1675666935617041) [Z1 Z3]
+ (0.18144009362933605) [Z0 Z3]
+ (0.18144009362933605) [Z1 Z2]
+ (0.19392574334990828) [Z0 Z1]
+ (0.22003977240260741) [Z2 Z3]
+ (7.0380236993958125e-06) [Y4 Z5 Y6]
+ (7.0380236993958125e-06) [X4 Z5 X6]
+ (7.0380236993958125e-06) [Y5 Z6 Y7]
+ (7.0380236993958125e-06) [X5 Z6 X7]
+ (-0.028685217310274038) [Y4 Y5 X6 X7]
+ (-0.028685217310274038) [X4 X5 Y6 Y7]
+ (-0.017825011932626346) [Y0 Y1 X4 X5]
+ (-0.017825011932626346) [X0 X1 Y4 Y5]
+ (-0.01736605326461045) [Y0 Y1 X6 X7]
+ (-0.01736605326461045) [X0 X1 Y6 Y7]
+ (-0.013873400067631991) [Y0 Y1 X2 X3]
+ (-0.013873400067631991) [X0 X1 Y2 Y3]
+ (-0.00956071649644088) [Y2 Y3 X4 X5]
+ (-0.00956071649644088) [X2 X3 Y4 Y5]
+ (-0.006087836804228357) [Y2 Y3 X6 X7]
+ (-0.006087836804228357) [X2 X3 Y6 Y7]
+ (-0.0002922256724514197) [Y1 Y2 X3 X4]
+ (-0.0002922256724514197) [X1 X2 Y3 Y4]
+ (-7.95422441813437e-06) [Y4 Z5 Y6 Z7]
+ (-7.95422441813437e-06) [X4 Z5 X6 Z7]
+ (-6.628427122204418e-07) [Y2 X3 X5 Y6]
+ (-6.628427122204418e-07) [Y2 Y3 Y5 Y6]
+ (-6.628427122204418e-07) [X2 X3 X5 X6]
+ (-6.628427122204418e-07) [X2 Y3 Y5 X6]
+ (9.344970458534385e-07) [Z2 Y5 Z6 Y7]
+ (9.344970458534385e-07) [Z2 X5 Z6 X7]
+ (9.344970458534385e-07) [Z3 Y4 Z5 Y6]
+ (9.344970458534385e-07) [Z3 X4 Z5 X6]
+ (1.0357924654330435e-06) [Y0 X1 X5 Y6]
+ (1.0357924654330435e-06) [Y0 Y1 Y5 Y6]
+ (1.0357924654330435e-06) [X0 X1 X5 X6]
+ (1.0357924654330435e-06) [X0 Y1 Y5 X6]
+ (1.5973397580738803e-06) [Z2 Y4 Z5 Y6]
+ (1.5973397580738803e-06) [Z2 X4 Z5 X6]
+ (1.5973397580738803e-06) [Z3 Y5 Z6 Y7]
+ (1.5973397580738803e-06) [Z3 X5 Z6 X7]
+ (1.8551374962633323e-06) [Z0 Y4 Z5 Y6]
+ (1.8551374962633323e-06) [Z0 X4 Z5 X6]
+ (1.8551374962633323e-06) [Z1 Y5 Z6 Y7]
+ (1.8551374962633323e-06) [Z1 X5 Z6 X7]
+ (2.890929961695942e-06) [Z0 Y5 Z6 Y7]
+ (2.890929961695942e-06) [Z0 X5 Z6 X7]
+ (2.890929961695942e-06) [Z1 Y4 Z5 Y6]
+ (2.890929961695942e-06) [Z1 X4 Z5 X6]
+ (8.194104917598477e-06) [Z4 Y5 Z6 Y7]
+ (8.194104917598477e-06) [Z4 X5 Z6 X7]
+ (0.0002922256724514197) [Y1 X2 X3 Y4]
+ (0.0002922256724514197) [X1 Y2 Y3 X4]
+ (0.006087836804228357) [Y2 X3 X6 Y7]
+ (0.006087836804228357) [X2 Y3 Y6 X7]
+ (0.00956071649644088) [Y2 X3 X4 Y5]
+ (0.00956071649644088) [X2 Y3 Y4 X5]
+ (0.011307208030073658) [Y1 Z2 Z3 Y5]
+ (0.011307208030073658) [X1 Z2 Z3 X5]
+ (0.013873400067631991) [Y0 X1 X2 Y3]
+ (0.013873400067631991) [X0 Y1 Y2 X3]
+ (0.01736605326461045) [Y0 X1 X6 Y7]
+ (0.01736605326461045) [X0 Y1 Y6 X7]
+ (0.017825011932626346) [Y0 X1 X4 Y5]
+ (0.017825011932626346) [X0 Y1 Y4 X5]
+ (0.028685217310274038) [Y4 X5 X6 Y7]
+ (0.028685217310274038) [X4 Y5 Y6 X7]
+ (0.029812299601155026) [Y0 Z1 Z2 Y4]
+ (0.029812299601155026) [X0 Z1 Z2 X4]
+ (0.029812299601155026) [Y1 Z3 Z4 Y5]
+ (0.029812299601155026) [X1 Z3 Z4 X5]
+ (0.030104525273606443) [Y0 Z1 Z3 Y4]
+ (0.030104525273606443) [X0 Z1 Z3 X4]
+ (0.030104525273606443) [Y1 Z2 Z4 Y5]
+ (0.030104525273606443) [X1 Z2 Z4 X5]
+ (0.030787440718657147) [Y0 Z2 Z3 Y4]
+ (0.030787440718657147) [X0 Z2 Z3 X4]
+ (0.04375171612137438) [Y1 Z2 Z3 Z4 Y5]
+ (0.04375171612137438) [X1 Z2 Z3 Z4 X5]
+ (0.04375171612137441) [Y0 Z1 Z2 Z3 Y4]
+ (0.04375171612137441) [X0 Z1 Z2 Z3 X4]
+ (-0.014564473640795116) [Y1 Z2 Z3 X4 X6 Y7]
+ (-0.014564473640795116) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-0.014564473640795116) [X1 Z2 Z3 X4 X6 X7]
+ (-0.014564473640795116) [X1 Z2 Z3 Y4 Y6 X7]
+ (-4.183808722819232e-06) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-4.183808722819232e-06) [X1 Z2 Z3 Z4 Z5 X7]
+ (-3.313017021738633e-06) [Y1 Z2 Z3 X4 X5 Y6]
+ (-3.313017021738633e-06) [X1 Z2 Z3 Y4 Y5 X6]
+ (-1.0357924654330435e-06) [Y0 X1 X4 Z5 Z6 Y7]
+ (-1.0357924654330435e-06) [X0 Y1 Y4 Z5 Z6 X7]
+ (-6.628427122204418e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-6.628427122204418e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-3.328039607427831e-07) [Y1 Y2 X3 Z4 Z5 X6]
+ (-3.328039607427831e-07) [X1 X2 Y3 Z4 Z5 Y6]
+ (3.328039607427831e-07) [Y1 X2 X3 Z4 Z5 Y6]
+ (3.328039607427831e-07) [X1 Y2 Y3 Z4 Z5 X6]
+ (6.628427122204418e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (6.628427122204418e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (1.0357924654330435e-06) [Y0 Y1 X4 Z5 Z6 X7]
+ (1.0357924654330435e-06) [X0 X1 Y4 Z5 Z6 Y7]
+ (3.2111873912825384e-06) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (3.2111873912825384e-06) [X0 Z1 Z2 Z3 Z5 X6]
+ (3.2111873912825384e-06) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (3.2111873912825384e-06) [X1 Z2 Z3 Z4 Z6 X7]
+ (3.2774382258378768e-06) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (3.2774382258378768e-06) [X0 Z1 Z2 Z4 Z5 X6]
+ (3.2774382258378768e-06) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (3.2774382258378768e-06) [X1 Z3 Z4 Z5 Z6 X7]
+ (3.313017021738633e-06) [Y1 Z2 Z3 Y4 X5 X6]
+ (3.313017021738633e-06) [X1 Z2 Z3 X4 Y5 Y6]
+ (3.61024218658066e-06) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (3.61024218658066e-06) [X0 Z1 Z3 Z4 Z5 X6]
+ (3.61024218658066e-06) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (3.61024218658066e-06) [X1 Z2 Z4 Z5 Z6 X7]
+ (3.769583544105978e-06) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (3.769583544105978e-06) [X0 Z2 Z3 Z4 Z5 X6]
+ (6.524204413015967e-06) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (6.524204413015967e-06) [X0 Z1 Z2 Z3 Z4 X6]
+ (6.524204413015967e-06) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (6.524204413015967e-06) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.0002922256724514197) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (0.0002922256724514197) [Y0 Z1 Y2 X3 Z4 X5]
+ (0.0002922256724514197) [X0 Z1 X2 Y3 Z4 Y5]
+ (0.0002922256724514197) [X0 Z1 X2 X3 Z4 X5]
+ (0.010540434329276753) [Y0 Z1 Z2 Z3 Y4 Z7]
+ (0.010540434329276753) [X0 Z1 Z2 Z3 X4 Z7]
+ (0.010540434329276753) [Y1 Z2 Z3 Z4 Y5 Z6]
+ (0.010540434329276753) [X1 Z2 Z3 Z4 X5 Z6]
+ (0.011307208030073656) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (0.011307208030073656) [X0 Z1 Z2 Z3 X4 Z5]
+ (0.02510490797007187) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (0.02510490797007187) [X0 Z1 Z2 Z3 X4 Z6]
+ (0.02510490797007187) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (0.02510490797007187) [X1 Z2 Z3 Z4 X5 Z7]
+ (0.030787440718657147) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (0.030787440718657147) [Z0 X1 Z2 Z3 Z4 X5]
+ (5.105680968470451e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (5.105680968470451e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (5.105680968470451e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (5.105680968470451e-06) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.014564473640795116) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-0.014564473640795116) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-4.183808722819232e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-4.183808722819232e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-3.313017021738633e-06) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (-3.313017021738633e-06) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (-3.313017021738633e-06) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (-3.313017021738633e-06) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (3.328039607427831e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (3.328039607427831e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (3.328039607427831e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (3.328039607427831e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (3.769583544105978e-06) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (3.769583544105978e-06) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (0.014564473640795116) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (0.014564473640795116) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
  (-46.46390678868897) [I0]
+ (0.7829661725950168) [Z10]
+ (0.7829661725950172) [Z11]
+ (0.8084581961720531) [Z12]
+ (0.8084581961720532) [Z13]
+ (1.2034402289145676) [Z4]
+ (1.203440228914568) [Z5]
+ (1.3096862988615523) [Z7]
+ (1.309686298861553) [Z6]
+ (1.3693525634718218) [Z8]
+ (1.3693525634718218) [Z9]
+ (1.6538942226831628) [Z2]
+ (1.653894222683163) [Z3]
+ (12.412630742111778) [Z0]
+ (12.412630742111778) [Z1]
+ (-8.194261372687812e-06) [Y10 Y12]
+ (-8.194261372687812e-06) [X10 X12]
+ (-1.854060858026002e-06) [Y5 Y7]
+ (-1.854060858026002e-06) [X5 X7]
+ (-7.764994120890156e-07) [Y3 Y5]
+ (-7.764994120890156e-07) [X3 X5]
+ (-5.929765815116186e-07) [Y4 Y6]
+ (-5.929765815116186e-07) [X4 X6]
+ (1.602116740734771e-06) [Y2 Y4]
+ (1.602116740734771e-06) [X2 X4]
+ (7.954413176532171e-06) [Y11 Y13]
+ (7.954413176532171e-06) [X11 X13]
+ (0.0032769719312315264) [Y1 Y3]
+ (0.0032769719312315264) [X1 X3]
+ (0.1043306478065131) [Y0 Y2]
+ (0.1043306478065131) [X0 X2]
+ (0.11270386920332195) [Z10 Z12]
+ (0.11270386920332195) [Z11 Z13]
+ (0.11383573679388659) [Z4 Z12]
+ (0.11383573679388659) [Z5 Z13]
+ (0.11952438964682742) [Z6 Z10]
+ (0.11952438964682742) [Z7 Z11]
+ (0.12489990917237613) [Z4 Z10]
+ (0.12489990917237613) [Z5 Z11]
+ (0.12495807739503212) [Z2 Z4]
+ (0.12495807739503212) [Z3 Z5]
+ (0.12799502492468373) [Z2 Z10]
+ (0.12799502492468373) [Z3 Z11]
+ (0.1340171526196376) [Z6 Z12]
+ (0.1340171526196376) [Z7 Z13]
+ (0.13701191674040825) [Z4 Z6]
+ (0.13701191674040825) [Z5 Z7]
+ (0.13734953064261302) [Z6 Z11]
+ (0.13734953064261302) [Z7 Z10]
+ (0.13739104762683257) [Z2 Z6]
+ (0.13739104762683257) [Z3 Z7]
+ (0.13766872645852565) [Z8 Z10]
+ (0.13766872645852565) [Z9 Z11]
+ (0.14011289865354803) [Z2 Z12]
+ (0.14011289865354803) [Z3 Z13]
+ (0.1413890529194284) [Z10 Z13]
+ (0.1413890529194284) [Z11 Z12]
+ (0.14257997712485776) [Z4 Z11]
+ (0.14257997712485776) [Z5 Z10]
+ (0.14722943218766182) [Z8 Z11]
+ (0.14722943218766182) [Z9 Z10]
+ (0.14899430575065598) [Z4 Z7]
+ (0.14899430575065598) [Z5 Z6]
+ (0.1492635514738896) [Z10 Z11]
+ (0.14960702684445323) [Z4 Z8]
+ (0.14960702684445323) [Z5 Z9]
+ (0.1497348680349692) [Z8 Z12]
+ (0.1497348680349692) [Z9 Z13]
+ (0.1507140812100828) [Z2 Z8]
+ (0.1507140812100828) [Z3 Z9]
+ (0.15138327161428877) [Z6 Z13]
+ (0.15138327161428877) [Z7 Z12]
+ (0.15215040708869065) [Z4 Z13]
+ (0.15215040708869065) [Z5 Z12]
+ (0.15337968243314148) [Z2 Z11]
+ (0.15337968243314148) [Z3 Z10]
+ (0.15435748657223652) [Z12 Z13]
+ (0.15569010671752448) [Z2 Z13]
+ (0.15569010671752448) [Z3 Z12]
+ (0.15582269051553113) [Z8 Z13]
+ (0.15582269051553113) [Z9 Z12]
+ (0.15676396176431023) [Z4 Z9]
+ (0.15676396176431023) [Z5 Z8]
+ (0.1575531479798571) [Z4 Z5]
+ (0.1607976453483858) [Z2 Z5]
+ (0.1607976453483858) [Z3 Z4]
+ (0.16756653265461358) [Z6 Z8]
+ (0.16756653265461358) [Z7 Z9]
+ (0.16853486561579945) [Z2 Z7]
+ (0.16853486561579945) [Z3 Z6]
+ (0.18143991440303997) [Z6 Z9]
+ (0.18143991440303997) [Z7 Z8]
+ (0.18189085790751305) [Z2 Z3]
+ (0.1869082047691252) [Z2 Z9]
+ (0.1869082047691252) [Z3 Z8]
+ (0.19299723935364188) [Z0 Z10]
+ (0.19299723935364188) [Z1 Z11]
+ (0.19392534613270457) [Z6 Z7]
+ (0.19661770890342192) [Z0 Z4]
+ (0.19661770890342192) [Z1 Z5]
+ (0.19936354537360873) [Z0 Z5]
+ (0.19936354537360873) [Z1 Z4]
+ (0.2007286646044173) [Z0 Z11]
+ (0.2007286646044173) [Z1 Z10]
+ (0.21102659849791522) [Z0 Z12]
+ (0.21102659849791522) [Z1 Z13]
+ (0.21631037498631817) [Z0 Z13]
+ (0.21631037498631817) [Z1 Z12]
+ (0.22003977334376124) [Z8 Z9]
+ (0.23671080783830312) [Z0 Z2]
+ (0.23671080783830312) [Z1 Z3]
+ (0.24164663936017386) [Z0 Z6]
+ (0.24164663936017386) [Z1 Z7]
+ (0.24853483371314455) [Z0 Z7]
+ (0.24853483371314455) [Z1 Z6]
+ (0.25129445674591555) [Z0 Z3]
+ (0.25129445674591555) [Z1 Z2]
+ (0.2723251830660572) [Z0 Z8]
+ (0.2723251830660572) [Z1 Z9]
+ (0.2788345442672344) [Z0 Z9]
+ (0.2788345442672344) [Z1 Z8]
+ (1.1861763734860484) [Z0 Z1]
+ (-1.2260484989069807e-05) [Y4 Z5 Y6]
+ (-1.2260484989069807e-05) [X4 Z5 X6]
+ (-1.2260484989069807e-05) [Y5 Z6 Y7]
+ (-1.2260484989069807e-05) [X5 Z6 X7]
+ (-1.0722312159850488e-05) [Y10 Z11 Y12]
+ (-1.0722312159850488e-05) [X10 Z11 X12]
+ (-1.0722312159850488e-05) [Y11 Z12 Y13]
+ (-1.0722312159850488e-05) [X11 Z12 X13]
+ (-3.887051673148853e-06) [Y2 Z3 Y4]
+ (-3.887051673148853e-06) [X2 Z3 X4]
+ (-3.887051673148853e-06) [Y3 Z4 Y5]
+ (-3.887051673148853e-06) [X3 Z4 X5]
+ (0.12507032579771826) [Y0 Z1 Y2]
+ (0.12507032579771826) [X0 Z1 X2]
+ (0.12507032579771826) [Y1 Z2 Y3]
+ (0.12507032579771826) [X1 Z2 X3]
+ (-0.03831467029480406) [Y4 Y5 X12 X13]
+ (-0.03831467029480406) [X4 X5 Y12 Y13]
+ (-0.036194123559042383) [Y2 Y3 X8 X9]
+ (-0.036194123559042383) [X2 X3 Y8 Y9]
+ (-0.0358395679533537) [Y2 Y3 X4 X5]
+ (-0.0358395679533537) [X2 X3 Y4 Y5]
+ (-0.031143817988966878) [Y2 Y3 X6 X7]
+ (-0.031143817988966878) [X2 X3 Y6 Y7]
+ (-0.028685183716106448) [Y10 Y11 X12 X13]
+ (-0.028685183716106448) [X10 X11 Y12 Y13]
+ (-0.025996177598021676) [Y3 Z4 Z5 Y7]
+ (-0.025996177598021676) [X3 Z4 Z5 X7]
+ (-0.02538465750845775) [Y2 Y3 X10 X11]
+ (-0.02538465750845775) [X2 X3 Y10 Y11]
+ (-0.019028242443847977) [Y3 Y4 X11 X12]
+ (-0.019028242443847977) [X3 X4 Y11 Y12]
+ (-0.017825140995785606) [Y6 Y7 X10 X11]
+ (-0.017825140995785606) [X6 X7 Y10 Y11]
+ (-0.01768006795248164) [Y4 Y5 X10 X11]
+ (-0.01768006795248164) [X4 X5 Y10 Y11]
+ (-0.017366118994651205) [Y6 Y7 X12 X13]
+ (-0.017366118994651205) [X6 X7 Y12 Y13]
+ (-0.015577208063976434) [Y2 Y3 X12 X13]
+ (-0.015577208063976434) [X2 X3 Y12 Y13]
+ (-0.01458364890761246) [Y0 Y1 X2 X3]
+ (-0.01458364890761246) [X0 X1 Y2 Y3]
+ (-0.013873381748426367) [Y6 Y7 X8 X9]
+ (-0.013873381748426367) [X6 X7 Y8 Y9]
+ (-0.011982389010247747) [Y4 Y5 X6 X7]
+ (-0.011982389010247747) [X4 X5 Y6 Y7]
+ (-0.011285190200840569) [Y5 X6 X11 Y12]
+ (-0.011285190200840569) [X5 Y6 Y11 X12]
+ (-0.009560705729136205) [Y8 Y9 X10 X11]
+ (-0.009560705729136205) [X8 X9 Y10 Y11]
+ (-0.008125251921381) [Y1 X2 X8 Y9]
+ (-0.008125251921381) [Y1 Y2 Y8 Y9]
+ (-0.008125251921381) [X1 X2 X8 X9]
+ (-0.008125251921381) [X1 Y2 Y8 X9]
+ (-0.007731425250775413) [Y0 Y1 X10 X11]
+ (-0.007731425250775413) [X0 X1 Y10 Y11]
+ (-0.007156934919857001) [Y4 Y5 X8 X9]
+ (-0.007156934919857001) [X4 X5 Y8 Y9]
+ (-0.006888194352970665) [Y0 Y1 X6 X7]
+ (-0.006888194352970665) [X0 X1 Y6 Y7]
+ (-0.0065093612011772484) [Y0 Y1 X8 X9]
+ (-0.0065093612011772484) [X0 X1 Y8 Y9]
+ (-0.006087822480561935) [Y8 Y9 X12 X13]
+ (-0.006087822480561935) [X8 X9 Y12 Y13]
+ (-0.00528377648840296) [Y0 Y1 X12 X13]
+ (-0.00528377648840296) [X0 X1 Y12 Y13]
+ (-0.0051433917688248) [Y3 X4 X5 Y6]
+ (-0.0051433917688248) [X3 Y4 Y5 X6]
+ (-0.004684903388155249) [Y1 X2 X6 Y7]
+ (-0.004684903388155249) [Y1 Y2 Y6 Y7]
+ (-0.004684903388155249) [X1 X2 X6 X7]
+ (-0.004684903388155249) [X1 Y2 Y6 X7]
+ (-0.004575007626639163) [Y1 X2 X12 Y13]
+ (-0.004575007626639163) [Y1 Y2 Y12 Y13]
+ (-0.004575007626639163) [X1 X2 X12 X13]
+ (-0.004575007626639163) [X1 Y2 Y12 X13]
+ (-0.004424855449441861) [Y1 X2 X4 Y5]
+ (-0.004424855449441861) [Y1 Y2 Y4 Y5]
+ (-0.004424855449441861) [X1 X2 X4 X5]
+ (-0.004424855449441861) [X1 Y2 Y4 X5]
+ (-0.003479511890334035) [Y2 Z3 Z5 Y6]
+ (-0.003479511890334035) [X2 Z3 Z5 X6]
+ (-0.003479511890334035) [Y3 Z4 Z6 Y7]
+ (-0.003479511890334035) [X3 Z4 Z6 X7]
+ (-0.0027458364701868254) [Y0 Y1 X4 X5]
+ (-0.0027458364701868254) [X0 X1 Y4 Y5]
+ (-0.001799219493662879) [Y1 X2 X10 Y11]
+ (-0.001799219493662879) [Y1 Y2 Y10 Y11]
+ (-0.001799219493662879) [X1 X2 X10 X11]
+ (-0.001799219493662879) [X1 Y2 Y10 X11]
+ (-0.0002921986261112657) [Y7 Y8 X9 X10]
+ (-0.0002921986261112657) [X7 X8 Y9 Y10]
+ (-8.194261372687812e-06) [Z10 Y11 Z12 Y13]
+ (-8.194261372687812e-06) [Z10 X11 Z12 X13]
+ (-7.80170750096492e-06) [Y2 Z3 Y4 Z11]
+ (-7.80170750096492e-06) [X2 Z3 X4 Z11]
+ (-7.80170750096492e-06) [Y3 Z4 Y5 Z10]
+ (-7.80170750096492e-06) [X3 Z4 X5 Z10]
+ (-4.643051068758952e-06) [Y3 X4 X10 Y11]
+ (-4.643051068758952e-06) [Y3 Y4 Y10 Y11]
+ (-4.643051068758952e-06) [X3 X4 X10 X11]
+ (-4.643051068758952e-06) [X3 Y4 Y10 X11]
+ (-4.588855155933727e-06) [Y4 Z5 Y6 Z13]
+ (-4.588855155933727e-06) [X4 Z5 X6 Z13]
+ (-4.588855155933727e-06) [Y5 Z6 Y7 Z12]
+ (-4.588855155933727e-06) [X5 Z6 X7 Z12]
+ (-4.556569218493699e-06) [Y5 X6 X12 Y13]
+ (-4.556569218493699e-06) [Y5 Y6 Y12 Y13]
+ (-4.556569218493699e-06) [X5 X6 X12 X13]
+ (-4.556569218493699e-06) [X5 Y6 Y12 X13]
+ (-3.6945132948647784e-06) [Y4 X5 X11 Y12]
+ (-3.6945132948647784e-06) [Y4 Y5 Y11 Y12]
+ (-3.6945132948647784e-06) [X4 X5 X11 X12]
+ (-3.6945132948647784e-06) [X4 Y5 Y11 X12]
+ (-3.344081556534839e-06) [Z0 Y5 Z6 Y7]
+ (-3.344081556534839e-06) [Z0 X5 Z6 X7]
+ (-3.344081556534839e-06) [Z1 Y4 Z5 Y6]
+ (-3.344081556534839e-06) [Z1 X4 Z5 X6]
+ (-3.158656432205968e-06) [Y2 Z3 Y4 Z10]
+ (-3.158656432205968e-06) [X2 Z3 X4 Z10]
+ (-3.158656432205968e-06) [Y3 Z4 Y5 Z11]
+ (-3.158656432205968e-06) [X3 Z4 X5 Z11]
+ (-3.0993492436522252e-06) [Z0 Y4 Z5 Y6]
+ (-3.0993492436522252e-06) [Z0 X4 Z5 X6]
+ (-3.0993492436522252e-06) [Z1 Y5 Z6 Y7]
+ (-3.0993492436522252e-06) [Z1 X5 Z6 X7]
+ (-2.8909678818462518e-06) [Z6 Y11 Z12 Y13]
+ (-2.8909678818462518e-06) [Z6 X11 Z12 X13]
+ (-2.8909678818462518e-06) [Z7 Y10 Z11 Y12]
+ (-2.8909678818462518e-06) [Z7 X10 Z11 X12]
+ (-2.1776646054170203e-06) [Z0 Y10 Z11 Y12]
+ (-2.1776646054170203e-06) [Z0 X10 Z11 X12]
+ (-2.1776646054170203e-06) [Z1 Y11 Z12 Y13]
+ (-2.1776646054170203e-06) [Z1 X11 Z12 X13]
+ (-1.8818501832506412e-06) [Y4 Z5 Y6 Z9]
+ (-1.8818501832506412e-06) [X4 Z5 X6 Z9]
+ (-1.8818501832506412e-06) [Y5 Z6 Y7 Z8]
+ (-1.8818501832506412e-06) [X5 Z6 X7 Z8]
+ (-1.8551201217955823e-06) [Z6 Y10 Z11 Y12]
+ (-1.8551201217955823e-06) [Z6 X10 Z11 X12]
+ (-1.8551201217955823e-06) [Z7 Y11 Z12 Y13]
+ (-1.8551201217955823e-06) [Z7 X11 Z12 X13]
+ (-1.854060858026002e-06) [Y4 Z5 Y6 Z7]
+ (-1.854060858026002e-06) [X4 Z5 X6 Z7]
+ (-1.816303170143474e-06) [Z4 Y11 Z12 Y13]
+ (-1.816303170143474e-06) [Z4 X11 Z12 X13]
+ (-1.816303170143474e-06) [Z5 Y10 Z11 Y12]
+ (-1.816303170143474e-06) [Z5 X10 Z11 X12]
+ (-1.6923978286994233e-06) [Y4 Z5 Y6 Z10]
+ (-1.6923978286994233e-06) [X4 Z5 X6 Z10]
+ (-1.6923978286994233e-06) [Y5 Z6 Y7 Z11]
+ (-1.6923978286994233e-06) [X5 Z6 X7 Z11]
+ (-1.614879414264232e-06) [Z0 Y11 Z12 Y13]
+ (-1.614879414264232e-06) [Z0 X11 Z12 X13]
+ (-1.614879414264232e-06) [Z1 Y10 Z11 Y12]
+ (-1.614879414264232e-06) [Z1 X10 Z11 X12]
+ (-1.5973171980403406e-06) [Z8 Y10 Z11 Y12]
+ (-1.5973171980403406e-06) [Z8 X10 Z11 X12]
+ (-1.5973171980403406e-06) [Z9 Y11 Z12 Y13]
+ (-1.5973171980403406e-06) [Z9 X11 Z12 X13]
+ (-1.4548424490255958e-06) [Y3 X4 X6 Y7]
+ (-1.4548424490255958e-06) [Y3 Y4 Y6 Y7]
+ (-1.4548424490255958e-06) [X3 X4 X6 X7]
+ (-1.4548424490255958e-06) [X3 Y4 Y6 X7]
+ (-1.3980449081570025e-06) [Y4 Z5 Y6 Z8]
+ (-1.3980449081570025e-06) [X4 Z5 X6 Z8]
+ (-1.3980449081570025e-06) [Y5 Z6 Y7 Z9]
+ (-1.3980449081570025e-06) [X5 Z6 X7 Z9]
+ (-1.195489009892534e-06) [Y2 Z3 Y4 Z7]
+ (-1.195489009892534e-06) [X2 Z3 X4 Z7]
+ (-1.195489009892534e-06) [Y3 Z4 Y5 Z6]
+ (-1.195489009892534e-06) [X3 Z4 X5 Z6]
+ (-1.1908508082933027e-06) [Z0 Y3 Z4 Y5]
+ (-1.1908508082933027e-06) [Z0 X3 Z4 X5]
+ (-1.1908508082933027e-06) [Z1 Y2 Z3 Y4]
+ (-1.1908508082933027e-06) [Z1 X2 Z3 X4]
+ (-1.1708301370681805e-06) [Z2 Y5 Z6 Y7]
+ (-1.1708301370681805e-06) [Z2 X5 Z6 X7]
+ (-1.1708301370681805e-06) [Z3 Y4 Z5 Y6]
+ (-1.1708301370681805e-06) [Z3 X4 Z5 X6]
+ (-1.0632283425039625e-06) [Z2 Y10 Z11 Y12]
+ (-1.0632283425039625e-06) [Z2 X10 Z11 X12]
+ (-1.0632283425039625e-06) [Z3 Y11 Z12 Y13]
+ (-1.0632283425039625e-06) [Z3 X11 Z12 X13]
+ (-1.0358477600506694e-06) [Y6 X7 X11 Y12]
+ (-1.0358477600506694e-06) [Y6 Y7 Y11 Y12]
+ (-1.0358477600506694e-06) [X6 X7 X11 X12]
+ (-1.0358477600506694e-06) [X6 Y7 Y11 X12]
+ (-9.509249751795085e-07) [Z2 Y4 Z5 Y6]
+ (-9.509249751795085e-07) [Z2 X4 Z5 X6]
+ (-9.509249751795085e-07) [Z3 Y5 Z6 Y7]
+ (-9.509249751795085e-07) [Z3 X5 Z6 X7]
+ (-9.344557778661472e-07) [Z8 Y11 Z12 Y13]
+ (-9.344557778661472e-07) [Z8 X11 Z12 X13]
+ (-9.344557778661472e-07) [Z9 Y10 Z11 Y12]
+ (-9.344557778661472e-07) [Z9 X10 Z11 X12]
+ (-8.337746753849749e-07) [Z0 Y2 Z3 Y4]
+ (-8.337746753849749e-07) [Z0 X2 Z3 X4]
+ (-8.337746753849749e-07) [Z1 Y3 Z4 Y5]
+ (-8.337746753849749e-07) [Z1 X3 Z4 X5]
+ (-7.956895372550072e-07) [Y3 X4 X8 Y9]
+ (-7.956895372550072e-07) [Y3 Y4 Y8 Y9]
+ (-7.956895372550072e-07) [X3 X4 X8 X9]
+ (-7.956895372550072e-07) [X3 Y4 Y8 X9]
+ (-7.764994120890156e-07) [Y2 Z3 Y4 Z5]
+ (-7.764994120890156e-07) [X2 Z3 X4 Z5]
+ (-5.929765815116186e-07) [Z4 Y5 Z6 Y7]
+ (-5.929765815116186e-07) [Z4 X5 Z6 X7]
+ (-5.770052994838112e-07) [Y2 Z3 Y4 Z9]
+ (-5.770052994838112e-07) [X2 Z3 X4 Z9]
+ (-5.770052994838112e-07) [Y3 Z4 Y5 Z8]
+ (-5.770052994838112e-07) [X3 Z4 X5 Z8]
+ (-5.471647744988075e-07) [Y1 Y2 X11 X12]
+ (-5.471647744988075e-07) [X1 X2 Y11 Y12]
+ (-4.838052750936388e-07) [Y5 X6 X8 Y9]
+ (-4.838052750936388e-07) [Y5 Y6 Y8 Y9]
+ (-4.838052750936388e-07) [X5 X6 X8 X9]
+ (-4.838052750936388e-07) [X5 Y6 Y8 X9]
+ (-3.570761329083278e-07) [Y0 X1 X3 Y4]
+ (-3.570761329083278e-07) [Y0 Y1 Y3 Y4]
+ (-3.570761329083278e-07) [X0 X1 X3 X4]
+ (-3.570761329083278e-07) [X0 Y1 Y3 X4]
+ (-2.447323128826137e-07) [Y0 X1 X5 Y6]
+ (-2.447323128826137e-07) [Y0 Y1 Y5 Y6]
+ (-2.447323128826137e-07) [X0 X1 X5 X6]
+ (-2.447323128826137e-07) [X0 Y1 Y5 X6]
+ (-2.19905161888672e-07) [Y2 X3 X5 Y6]
+ (-2.19905161888672e-07) [Y2 Y3 Y5 Y6]
+ (-2.19905161888672e-07) [X2 X3 X5 X6]
+ (-2.19905161888672e-07) [X2 Y3 Y5 X6]
+ (-1.9332412771358574e-07) [Y1 X2 X3 Y4]
+ (-1.9332412771358574e-07) [X1 Y2 Y3 X4]
+ (-1.291969486262427e-07) [Y1 Z2 Z3 Y5]
+ (-1.291969486262427e-07) [X1 Z2 Z3 X5]
+ (1.737933262565168e-07) [Y0 Z1 Z3 Y4]
+ (1.737933262565168e-07) [X0 Z1 Z3 X4]
+ (1.737933262565168e-07) [Y1 Z2 Z4 Y5]
+ (1.737933262565168e-07) [X1 Z2 Z4 X5]
+ (1.9332412771358574e-07) [Y1 Y2 X3 X4]
+ (1.9332412771358574e-07) [X1 X2 Y3 Y4]
+ (2.1868423777119598e-07) [Y2 Z3 Y4 Z8]
+ (2.1868423777119598e-07) [X2 Z3 X4 Z8]
+ (2.1868423777119598e-07) [Y3 Z4 Y5 Z9]
+ (2.1868423777119598e-07) [X3 Z4 X5 Z9]
+ (2.5935343913306187e-07) [Y2 Z3 Y4 Z6]
+ (2.5935343913306187e-07) [X2 Z3 X4 Z6]
+ (2.5935343913306187e-07) [Y3 Z4 Y5 Z7]
+ (2.5935343913306187e-07) [X3 Z4 X5 Z7]
+ (3.6060718681767395e-07) [Y0 Z1 Z2 Y4]
+ (3.6060718681767395e-07) [X0 Z1 Z2 X4]
+ (3.6060718681767395e-07) [Y1 Z3 Z4 Y5]
+ (3.6060718681767395e-07) [X1 Z3 Z4 X5]
+ (5.471647744988075e-07) [Y1 X2 X11 Y12]
+ (5.471647744988075e-07) [X1 Y2 Y11 X12]
+ (5.627851911527883e-07) [Y0 X1 X11 Y12]
+ (5.627851911527883e-07) [Y0 Y1 Y11 Y12]
+ (5.627851911527883e-07) [X0 X1 X11 X12]
+ (5.627851911527883e-07) [X0 Y1 Y11 X12]
+ (6.628614201741934e-07) [Y8 X9 X11 Y12]
+ (6.628614201741934e-07) [Y8 Y9 Y11 Y12]
+ (6.628614201741934e-07) [X8 X9 X11 X12]
+ (6.628614201741934e-07) [X8 Y9 Y11 X12]
+ (1.1094407590826662e-06) [Z2 Y11 Z12 Y13]
+ (1.1094407590826662e-06) [Z2 X11 Z12 X13]
+ (1.1094407590826662e-06) [Z3 Y10 Z11 Y12]
+ (1.1094407590826662e-06) [Z3 X10 Z11 X12]
+ (1.602116740734771e-06) [Z2 Y3 Z4 Y5]
+ (1.602116740734771e-06) [Z2 X3 Z4 X5]
+ (1.8782101247213044e-06) [Z4 Y10 Z11 Y12]
+ (1.8782101247213044e-06) [Z4 X10 Z11 X12]
+ (1.8782101247213044e-06) [Z5 Y11 Z12 Y13]
+ (1.8782101247213044e-06) [Z5 X11 Z12 X13]
+ (2.1726691015866288e-06) [Y2 X3 X11 Y12]
+ (2.1726691015866288e-06) [Y2 Y3 Y11 Y12]
+ (2.1726691015866288e-06) [X2 X3 X11 X12]
+ (2.1726691015866288e-06) [X2 Y3 Y11 X12]
+ (3.1174479461229833e-06) [Y0 Z2 Z3 Y4]
+ (3.1174479461229833e-06) [X0 Z2 Z3 X4]
+ (3.539054184657081e-06) [Y2 Z3 Y4 Z12]
+ (3.539054184657081e-06) [X2 Z3 X4 Z12]
+ (3.539054184657081e-06) [Y3 Z4 Y5 Z13]
+ (3.539054184657081e-06) [X3 Z4 X5 Z13]
+ (4.281913885022855e-06) [Y4 Z5 Y6 Z11]
+ (4.281913885022855e-06) [X4 Z5 X6 Z11]
+ (4.281913885022855e-06) [Y5 Z6 Y7 Z10]
+ (4.281913885022855e-06) [X5 Z6 X7 Z10]
+ (5.2758831223746055e-06) [Y3 X4 X12 Y13]
+ (5.2758831223746055e-06) [Y3 Y4 Y12 Y13]
+ (5.2758831223746055e-06) [X3 X4 X12 X13]
+ (5.2758831223746055e-06) [X3 Y4 Y12 X13]
+ (5.974311713722279e-06) [Y5 X6 X10 Y11]
+ (5.974311713722279e-06) [Y5 Y6 Y10 Y11]
+ (5.974311713722279e-06) [X5 X6 X10 X11]
+ (5.974311713722279e-06) [X5 Y6 Y10 X11]
+ (7.954413176532171e-06) [Y10 Z11 Y12 Z13]
+ (7.954413176532171e-06) [X10 Z11 X12 Z13]
+ (8.814937307031687e-06) [Y2 Z3 Y4 Z13]
+ (8.814937307031687e-06) [X2 Z3 X4 Z13]
+ (8.814937307031687e-06) [Y3 Z4 Y5 Z12]
+ (8.814937307031687e-06) [X3 Z4 X5 Z12]
+ (0.0002921986261112657) [Y7 X8 X9 Y10]
+ (0.0002921986261112657) [X7 Y8 Y9 X10]
+ (0.0004956762314923447) [Y2 Z4 Z5 Y6]
+ (0.0004956762314923447) [X2 Z4 Z5 X6]
+ (0.0011059037691896548) [Y0 Z1 Y2 Z5]
+ (0.0011059037691896548) [X0 Z1 X2 Z5]
+ (0.0011059037691896548) [Y1 Z2 Y3 Z4]
+ (0.0011059037691896548) [X1 Z2 X3 Z4]
+ (0.0016638798784907646) [Y2 Z3 Z4 Y6]
+ (0.0016638798784907646) [X2 Z3 Z4 X6]
+ (0.0016638798784907646) [Y3 Z5 Z6 Y7]
+ (0.0016638798784907646) [X3 Z5 Z6 X7]
+ (0.0017560707018411826) [Y0 Z1 Y2 Z11]
+ (0.0017560707018411826) [X0 Z1 X2 Z11]
+ (0.0017560707018411826) [Y1 Z2 Y3 Z10]
+ (0.0017560707018411826) [X1 Z2 X3 Z10]
+ (0.002326230623158051) [Y0 Z1 Y2 Z13]
+ (0.002326230623158051) [X0 Z1 X2 Z13]
+ (0.002326230623158051) [Y1 Z2 Y3 Z12]
+ (0.002326230623158051) [X1 Z2 X3 Z12]
+ (0.0027458364701868254) [Y0 X1 X4 Y5]
+ (0.0027458364701868254) [X0 Y1 Y4 X5]
+ (0.0029297686747510147) [Y0 Z1 Y2 Z9]
+ (0.0029297686747510147) [X0 Z1 X2 Z9]
+ (0.0029297686747510147) [Y1 Z2 Y3 Z8]
+ (0.0029297686747510147) [X1 Z2 X3 Z8]
+ (0.0032769719312315264) [Y0 Z1 Y2 Z3]
+ (0.0032769719312315264) [X0 Z1 X2 Z3]
+ (0.003347617530666205) [Y0 Z1 Y2 Z7]
+ (0.003347617530666205) [X0 Z1 X2 Z7]
+ (0.003347617530666205) [Y1 Z2 Y3 Z6]
+ (0.003347617530666205) [X1 Z2 X3 Z6]
+ (0.0035552901955040617) [Y0 Z1 Y2 Z10]
+ (0.0035552901955040617) [X0 Z1 X2 Z10]
+ (0.0035552901955040617) [Y1 Z2 Y3 Z11]
+ (0.0035552901955040617) [X1 Z2 X3 Z11]
+ (0.0051433917688248) [Y3 Y4 X5 X6]
+ (0.0051433917688248) [X3 X4 Y5 Y6]
+ (0.00528377648840296) [Y0 X1 X12 Y13]
+ (0.00528377648840296) [X0 Y1 Y12 X13]
+ (0.005530759218631516) [Y0 Z1 Y2 Z4]
+ (0.005530759218631516) [X0 Z1 X2 Z4]
+ (0.005530759218631516) [Y1 Z2 Y3 Z5]
+ (0.005530759218631516) [X1 Z2 X3 Z5]
+ (0.006087822480561935) [Y8 X9 X12 Y13]
+ (0.006087822480561935) [X8 Y9 Y12 X13]
+ (0.0065093612011772484) [Y0 X1 X8 Y9]
+ (0.0065093612011772484) [X0 Y1 Y8 X9]
+ (0.006888194352970665) [Y0 X1 X6 Y7]
+ (0.006888194352970665) [X0 Y1 Y6 X7]
+ (0.006901238249797214) [Y0 Z1 Y2 Z12]
+ (0.006901238249797214) [X0 Z1 X2 Z12]
+ (0.006901238249797214) [Y1 Z2 Y3 Z13]
+ (0.006901238249797214) [X1 Z2 X3 Z13]
+ (0.007156934919857001) [Y4 X5 X8 Y9]
+ (0.007156934919857001) [X4 Y5 Y8 X9]
+ (0.007731425250775413) [Y0 X1 X10 Y11]
+ (0.007731425250775413) [X0 Y1 Y10 X11]
+ (0.008032520918821454) [Y0 Z1 Y2 Z6]
+ (0.008032520918821454) [X0 Z1 X2 Z6]
+ (0.008032520918821454) [Y1 Z2 Y3 Z7]
+ (0.008032520918821454) [X1 Z2 X3 Z7]
+ (0.009560705729136205) [Y8 X9 X10 Y11]
+ (0.009560705729136205) [X8 Y9 Y10 X11]
+ (0.011055020596132014) [Y0 Z1 Y2 Z8]
+ (0.011055020596132014) [X0 Z1 X2 Z8]
+ (0.011055020596132014) [Y1 Z2 Y3 Z9]
+ (0.011055020596132014) [X1 Z2 X3 Z9]
+ (0.011285190200840569) [Y5 Y6 X11 X12]
+ (0.011285190200840569) [X5 X6 Y11 Y12]
+ (0.011307274008848126) [Y7 Z8 Z9 Y11]
+ (0.011307274008848126) [X7 Z8 Z9 X11]
+ (0.011982389010247747) [Y4 X5 X6 Y7]
+ (0.011982389010247747) [X4 Y5 Y6 X7]
+ (0.013873381748426367) [Y6 X7 X8 Y9]
+ (0.013873381748426367) [X6 Y7 Y8 X9]
+ (0.01458364890761246) [Y0 X1 X2 Y3]
+ (0.01458364890761246) [X0 Y1 Y2 X3]
+ (0.015577208063976434) [Y2 X3 X12 Y13]
+ (0.015577208063976434) [X2 Y3 Y12 X13]
+ (0.017366118994651205) [Y6 X7 X12 Y13]
+ (0.017366118994651205) [X6 Y7 Y12 X13]
+ (0.01768006795248164) [Y4 X5 X10 Y11]
+ (0.01768006795248164) [X4 Y5 Y10 X11]
+ (0.017825140995785606) [Y6 X7 X10 Y11]
+ (0.017825140995785606) [X6 Y7 Y10 X11]
+ (0.019028242443847977) [Y3 X4 X11 Y12]
+ (0.019028242443847977) [X3 Y4 Y11 X12]
+ (0.02538465750845775) [Y2 X3 X10 Y11]
+ (0.02538465750845775) [X2 Y3 Y10 X11]
+ (0.028685183716106448) [Y10 X11 X12 Y13]
+ (0.028685183716106448) [X10 Y11 Y12 X13]
+ (0.02981242451734489) [Y6 Z7 Z8 Y10]
+ (0.02981242451734489) [X6 Z7 Z8 X10]
+ (0.02981242451734489) [Y7 Z9 Z10 Y11]
+ (0.02981242451734489) [X7 Z9 Z10 X11]
+ (0.030104623143456154) [Y6 Z7 Z9 Y10]
+ (0.030104623143456154) [X6 Z7 Z9 X10]
+ (0.030104623143456154) [Y7 Z8 Z10 Y11]
+ (0.030104623143456154) [X7 Z8 Z10 X11]
+ (0.03078750538914359) [Y6 Z8 Z9 Y10]
+ (0.03078750538914359) [X6 Z8 Z9 X10]
+ (0.031143817988966878) [Y2 X3 X6 Y7]
+ (0.031143817988966878) [X2 Y3 Y6 X7]
+ (0.0358395679533537) [Y2 X3 X4 Y5]
+ (0.0358395679533537) [X2 Y3 Y4 X5]
+ (0.036194123559042383) [Y2 X3 X8 Y9]
+ (0.036194123559042383) [X2 Y3 Y8 X9]
+ (0.03831467029480406) [Y4 X5 X12 Y13]
+ (0.03831467029480406) [X4 Y5 Y12 X13]
+ (0.1043306478065131) [Z0 Y1 Z2 Y3]
+ (0.1043306478065131) [Z0 X1 Z2 X3]
+ (-0.12133276911042223) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133276911042223) [X2 Z3 Z4 Z5 X6]
+ (-0.12133276911042223) [Y3 Z4 Z5 Z6 Y7]
+ (-0.12133276911042223) [X3 Z4 Z5 Z6 X7]
+ (3.2020768802827127e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.2020768802827127e-06) [X0 Z1 Z2 Z3 X4]
+ (3.2020768802827127e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.2020768802827127e-06) [X1 Z2 Z3 Z4 X5]
+ (0.2284810656491839) [Y7 Z8 Z9 Z10 Y11]
+ (0.2284810656491839) [X7 Z8 Z9 Z10 X11]
+ (0.228481065649184) [Y6 Z7 Z8 Z9 Y10]
+ (0.228481065649184) [X6 Z7 Z8 Z9 X10]
+ (-0.032767657823289775) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.032767657823289775) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.032767657823289775) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.032767657823289775) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.02711503684527248) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.02711503684527248) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.02711503684527248) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.02711503684527248) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.025996177598021676) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.025996177598021676) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.017561202409646023) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.017561202409646023) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.017561202409646023) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.017561202409646023) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.01456453123117288) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.01456453123117288) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.01456453123117288) [X7 Z8 Z9 X10 X12 X13]
+ (-0.01456453123117288) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.012215040997613516) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.012215040997613516) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.012215040997613516) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.012215040997613516) [X4 Z5 X6 X11 Z12 X13]
+ (-0.012215040997613516) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.012215040997613516) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.012215040997613516) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.012215040997613516) [X5 Z6 X7 X10 Z11 X12]
+ (-0.011756013419819293) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.011756013419819293) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.011756013419819293) [X3 Z4 Z5 X6 X8 X9]
+ (-0.011756013419819293) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.008764827575689133) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.008764827575689133) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.008764827575689133) [X2 Z3 Z4 X5 X11 X12]
+ (-0.008764827575689133) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.008764827575689133) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.008764827575689133) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.008764827575689133) [X3 X4 X10 Z11 Z12 X13]
+ (-0.008764827575689133) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.008125251921381) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.008125251921381) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.007306759928832776) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.007306759928832776) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.007306759928832776) [X4 X5 X7 Z8 Z9 X10]
+ (-0.007306759928832776) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.005805188989826731) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.005805188989826731) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.005805188989826731) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.005805188989826731) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.005652620978017297) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.005652620978017297) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.005652620978017297) [X0 X1 X3 Z4 Z5 X6]
+ (-0.005652620978017297) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.0051433917688248) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.0051433917688248) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.0051433917688248) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.0051433917688248) [X2 Z3 X4 X5 Z6 X7]
+ (-0.004684903388155249) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.004684903388155249) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.00466862031877629) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.00466862031877629) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.004575007626639163) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.004575007626639163) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.004424855449441861) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.004424855449441861) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.004158797381839992) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.004158797381839992) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.004158797381839992) [X3 Z4 Z5 X6 X12 X13]
+ (-0.004158797381839992) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.0034937903598901408) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.0034937903598901408) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.0034937903598901408) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.0034937903598901408) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.002779026799025657) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.002779026799025657) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.002293956611352454) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.002293956611352454) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.001799219493662879) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.001799219493662879) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.0017278753941369787) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.0017278753941369787) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.0009298507967729496) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298507967729496) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298507967729496) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298507967729496) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.000853385625412585) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.000853385625412585) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.000814531327095643) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.000814531327095643) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.000814531327095643) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.000814531327095643) [X3 Z4 Z5 Z6 X7 Z11]
+ (-7.73503688059987e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.73503688059987e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.73503688059987e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.73503688059987e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-8.774817865208305e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.774817865208305e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.774817865208305e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.774817865208305e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.5183622161933234e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.5183622161933234e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.5183622161933234e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.5183622161933234e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.444344676460881e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.444344676460881e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.444344676460881e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.444344676460881e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.5243738488970915e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.5243738488970915e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.5243738488970915e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.5243738488970915e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-6.2900284337663506e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.2900284337663506e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.2900284337663506e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.2900284337663506e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.974311713722279e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (-5.974311713722279e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (-5.2758831223746055e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.2758831223746055e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.643051068758952e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.643051068758952e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.556569218493699e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (-4.556569218493699e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (-4.253224225744378e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.253224225744378e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-3.769659452364498e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.769659452364498e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.6945132948647784e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.6945132948647784e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.610297130944931e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.610297130944931e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.610297130944931e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.610297130944931e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.3131455001774585e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.3131455001774585e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.2774831959072118e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.2774831959072118e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.2774831959072118e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.2774831959072118e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.211228348719633e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.211228348719633e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.211228348719633e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.211228348719633e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.151346311331131e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.151346311331131e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.0882507115853304e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.0882507115853304e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.1726691015866288e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.1726691015866288e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.4548424490255958e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.4548424490255958e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.3304731887474244e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.3304731887474244e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.2283337824269729e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.2283337824269729e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.0358477600506694e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.0358477600506694e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-7.956895372550072e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.956895372550072e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.733197742797876e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.733197742797876e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.733197742797876e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.733197742797876e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.628614201741934e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.628614201741934e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.556281915054936e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.556281915054936e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.556281915054936e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.556281915054936e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.418291575084181e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.418291575084181e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.418291575084181e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.418291575084181e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.927453083205095e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.927453083205095e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.927453083205095e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.927453083205095e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.627851911527883e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.627851911527883e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.287660625031942e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.287660625031942e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.287660625031942e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.287660625031942e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.287660625031942e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.287660625031942e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.287660625031942e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.287660625031942e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.838052750936388e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (-4.838052750936388e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (-3.570761329083278e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.570761329083278e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.3281393503771907e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.3281393503771907e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.0868265652170864e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.0868265652170864e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.0868265652170864e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.0868265652170864e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.447323128826137e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (-2.447323128826137e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (-2.3713289478951086e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.3713289478951086e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.3713289478951086e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.3713289478951086e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.19905161888672e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-2.19905161888672e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-1.9332412771358574e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.9332412771358574e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.9332412771358574e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.9332412771358574e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.8394209154542183e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.8394209154542183e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.8394209154542183e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.8394209154542183e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.5510539176529954e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (-1.5510539176529954e-07) [X0 Z1 X2 X4 Z5 X6]
+ (-1.5510539176529954e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (-1.5510539176529954e-07) [X1 Z2 X3 X5 Z6 X7]
+ (-1.380778148099115e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (-1.380778148099115e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (-1.380778148099115e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (-1.380778148099115e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (-1.380778148099115e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (-1.380778148099115e-07) [X0 Z1 X2 X5 Z6 X7]
+ (-1.380778148099115e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (-1.380778148099115e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (-1.380778148099115e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (-1.380778148099115e-07) [X1 Z2 X3 X4 Z5 X6]
+ (-1.380778148099115e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (-1.380778148099115e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (-1.291969486262427e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.291969486262427e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.1076325600595504e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.1076325600595504e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.1076325600595504e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.1076325600595504e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.1076325600595504e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.1076325600595504e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.1076325600595504e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.1076325600595504e-07) [X1 Z2 X3 X10 Z11 X12]
+ (8.057446595927808e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.057446595927808e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.057446595927808e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.057446595927808e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.649310136361367e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.649310136361367e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.649310136361367e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.649310136361367e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.8394209154542183e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.8394209154542183e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.8394209154542183e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.8394209154542183e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.19905161888672e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (2.19905161888672e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (2.447323128826137e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (2.447323128826137e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.236259961531245e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.236259961531245e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.236259961531245e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.236259961531245e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.3281393503771907e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.3281393503771907e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.570761329083278e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.570761329083278e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.838052750936388e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (4.838052750936388e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (5.627851911527883e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.627851911527883e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (6.628614201741934e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.628614201741934e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (7.956895372550072e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.956895372550072e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.306536652341159e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.306536652341159e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.306536652341159e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.306536652341159e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.0358477600506694e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.0358477600506694e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.2283337824269729e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.2283337824269729e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.2393363217558245e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.2393363217558245e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.2393363217558245e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.2393363217558245e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.3304731887474244e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.3304731887474244e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.4548424490255958e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.4548424490255958e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.1726691015866288e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.1726691015866288e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.0882507115853304e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.0882507115853304e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.1174479461229833e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.1174479461229833e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.151346311331131e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.151346311331131e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.3131455001774585e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.3131455001774585e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.3343312895043123e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.3343312895043123e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.6945132948647784e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.6945132948647784e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (4.183932559486715e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.183932559486715e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.556569218493699e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (4.556569218493699e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (4.643051068758952e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.643051068758952e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.2758831223746055e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.2758831223746055e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.974311713722279e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (5.974311713722279e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (0.0002921986261112657) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.0002921986261112657) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.0002921986261112657) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.0002921986261112657) [X6 Z7 X8 X9 Z10 X11]
+ (0.0004956762314923447) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.0004956762314923447) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.0006650070219498504) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.0006650070219498504) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.0006650070219498504) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.0006650070219498504) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.000853385625412585) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.000853385625412585) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.0016095313817213802) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.0016095313817213802) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.0016095313817213802) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.0016095313817213802) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.0016676041811441358) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.0016676041811441358) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.0016676041811441358) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.0016676041811441358) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.0017278753941369787) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.0017278753941369787) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.001799219493662879) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.001799219493662879) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.002293956611352454) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.002293956611352454) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.002462917007133965) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.002462917007133965) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.002462917007133965) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.002462917007133965) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.00396156079249659) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.00396156079249659) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.00396156079249659) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.00396156079249659) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.004424855449441861) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.004424855449441861) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.004575007626639163) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.004575007626639163) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.00466862031877629) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.00466862031877629) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.004684903388155249) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.004684903388155249) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.005324835234221485) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.005324835234221485) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.005324835234221485) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.005324835234221485) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.005368659358109228) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.005368659358109228) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.005368659358109228) [X2 X3 X7 Z8 Z9 X10]
+ (0.005368659358109228) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.007960880725921365) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.007960880725921365) [X4 Z5 X6 X10 Z11 X12]
+ (0.007960880725921365) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.007960880725921365) [X5 Z6 X7 X11 Z12 X13]
+ (0.008125251921381) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.008125251921381) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.008890731522694314) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.008890731522694314) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.008890731522694314) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.008890731522694314) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.010263414868158845) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.010263414868158845) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.010263414868158845) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.010263414868158845) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.010540425907671097) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.010540425907671097) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.010540425907671097) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.010540425907671097) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.010960074940542361) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.010960074940542361) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.010960074940542361) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.010960074940542361) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.011307274008848126) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.011307274008848126) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.014411099430131294) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.014411099430131294) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.014411099430131294) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.014411099430131294) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.015225630757226934) [Y3 Z4 Z5 X6 X10 Y11]
+ (0.015225630757226934) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (0.015225630757226934) [X3 Z4 Z5 X6 X10 X11]
+ (0.015225630757226934) [X3 Z4 Z5 Y6 Y10 X11]
+ (0.015588250102380326) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (0.015588250102380326) [X2 Z3 X4 X10 Z11 X12]
+ (0.015588250102380326) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (0.015588250102380326) [X3 Z4 X5 X11 Z12 X13]
+ (0.018266834869375137) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.018266834869375137) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.018266834869375137) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.018266834869375137) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.019020423173039716) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.019020423173039716) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.019020423173039716) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.019020423173039716) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.020175921723534884) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.020175921723534884) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.020175921723534884) [X4 Z5 Z6 X7 X11 X12]
+ (0.020175921723534884) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.020175921723534884) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.020175921723534884) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.020175921723534884) [X5 X6 X10 Z11 Z12 X13]
+ (0.020175921723534884) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.024353077678069462) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.024353077678069462) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.024353077678069462) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.024353077678069462) [X2 Z3 X4 X11 Z12 X13]
+ (0.024353077678069462) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.024353077678069462) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.024353077678069462) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.024353077678069462) [X3 Z4 X5 X10 Z11 X12]
+ (0.02438908253114895) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.02438908253114895) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.02438908253114895) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.02438908253114895) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.02510495713884398) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.02510495713884398) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.02510495713884398) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.02510495713884398) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.03078750538914359) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.03078750538914359) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.04587947078129813) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.04587947078129813) [X0 Z2 Z3 Z4 Z5 X6]
+ (0.056007330877806565) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.056007330877806565) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.056007330877806565) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.056007330877806565) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661256) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.05608468124661256) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661256) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.05608468124661256) [Z1 X7 Z8 Z9 Z10 X11]
+ (-6.63127792890837e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.63127792890837e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.63127792890837e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.63127792890837e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.5950860072658757e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.5950860072658757e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.595086007265874e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.595086007265874e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.042743277013783235) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.042743277013783235) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (0.042743277013783235) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.042743277013783235) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.047642612176383214) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.047642612176383214) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.047642612176383214) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.047642612176383214) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.04171881383982182) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.04171881383982182) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.04171881383982182) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.04171881383982182) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.03956441632289427) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.03956441632289427) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.03956441632289427) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03956441632289427) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03935916802205407) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.03935916802205407) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.03935916802205407) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.03935916802205407) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.03931805194719794) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03931805194719794) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.03931805194719794) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03931805194719794) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03560837898831284) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.03560837898831284) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.02990378951262536) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.02990378951262536) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.02990378951262536) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.02990378951262536) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.028730779551905873) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.028730779551905873) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.028730779551905873) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.028730779551905873) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.025637238296026828) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.025637238296026828) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.025637238296026828) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.025637238296026828) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.02475546329289162) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.02475546329289162) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.02475546329289162) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.02475546329289162) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.024282117354693135) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.024282117354693135) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.023145130929529162) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.023145130929529162) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.022528440196013185) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.022528440196013185) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.021433810721602053) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.021433810721602053) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.021433810721602053) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.021433810721602053) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.01925750509525153) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.01925750509525153) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019028242443847977) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.019028242443847977) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.01888903030494327) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-0.01888903030494327) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-0.01888903030494327) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.01888903030494327) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.016024603689179597) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.016024603689179597) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.015225630757226934) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.015225630757226934) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.014603704729162444) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.014603704729162444) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.01456453123117288) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.01456453123117288) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.011756013419819293) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.011756013419819293) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.011285190200840569) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-0.011285190200840569) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-0.0098417492469626) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0098417492469626) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.009612634606847229) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.009612634606847229) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.009612634606847229) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.009612634606847229) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.008469978791023307) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.008469978791023307) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.007306759928832776) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.007306759928832776) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0059237983365613995) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.0059237983365613995) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.005652620978017297) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.005652620978017297) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.005368659358109227) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005368659358109227) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.004158797381839992) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.004158797381839992) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.003356670563832918) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.003356670563832918) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.003356670563832918) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.003356670563832918) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.0032675138544235576) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.0032675138544235576) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.0032675138544235576) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.0032675138544235576) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.0027790267990256572) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.0027790267990256572) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.0026860409778066796) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.0026860409778066796) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.0026860409778066796) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.0026860409778066796) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.0022939566113524537) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939566113524537) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.0022939566113524537) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939566113524537) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.0009581655836697011) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.0009581655836697011) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.0009581655836697011) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.0009581655836697011) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.0009581655836697011) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.0009581655836697011) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.0009581655836697011) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.0009581655836697011) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.0002463643756963395) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0002463643756963395) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303542473) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.00013840177303542473) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.00013840177303542473) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.00013840177303542473) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.73503688059987e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.73503688059987e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.610358530685109e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.610358530685109e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.610358530685109e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.610358530685109e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808796387577e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.5316808796387577e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808796387577e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5316808796387577e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.806102775752543e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.806102775752543e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.806102775752543e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.806102775752543e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.089799468031868e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.089799468031868e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.089799468031868e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.089799468031868e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.652209670164516e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.652209670164516e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.652209670164516e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.652209670164516e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851834629174e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.481851834629174e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851834629174e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.481851834629174e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.071480736700169e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.071480736700169e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.071480736700169e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.071480736700169e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.734622039052373e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.734622039052373e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.734622039052373e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.734622039052373e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.728843147530539e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.728843147530539e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.728843147530539e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.728843147530539e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.253224225744378e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.253224225744378e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.769659452364498e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.769659452364498e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.544395429517437e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.544395429517437e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.544395429517437e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.544395429517437e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.544395429517437e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.544395429517437e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.544395429517437e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.544395429517437e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.360956320501329e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.360956320501329e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.360956320501329e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.360956320501329e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-2.1032156048873406e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.1032156048873406e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.1032156048873406e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.1032156048873406e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220984679514e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.0111220984679514e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220984679514e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.0111220984679514e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9429468369113953e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.9429468369113953e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9429468369113953e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9429468369113953e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174773785413e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.6541174773785413e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174773785413e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.6541174773785413e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.522493067691714e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.522493067691714e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.522493067691714e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.522493067691714e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.522493067691714e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.522493067691714e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.522493067691714e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.522493067691714e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.2283337824269729e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337824269729e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.2283337824269729e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337824269729e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.988770288203237e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.988770288203237e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.988770288203237e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.988770288203237e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.86776510463514e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.86776510463514e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.86776510463514e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.86776510463514e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.18999097576742e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.18999097576742e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.175246207257721e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175246207257721e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.471647744988075e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.471647744988075e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.5614471793210647e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.5614471793210647e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.5614471793210647e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.5614471793210647e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.5233896782270817e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.5233896782270817e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.427323108882172e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.427323108882172e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.427323108882172e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.427323108882172e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.3281393503771907e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.3281393503771907e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.3281393503771907e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.3281393503771907e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.0868265652170864e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.0868265652170864e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.88829359532854e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.88829359532854e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.88829359532854e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.88829359532854e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.3713289478951086e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.3713289478951086e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.8394209154542183e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.8394209154542183e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-8.057446595927808e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.057446595927808e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (4.537178094365414e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.537178094365414e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.537178094365414e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (4.537178094365414e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (8.057446595927808e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.057446595927808e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (9.209350641938919e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350641938919e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350641938919e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.209350641938919e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783553534296e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783553534296e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783553534296e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783553534296e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.8394209154542183e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.8394209154542183e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.3713289478951086e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.3713289478951086e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (3.0868265652170864e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.0868265652170864e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (4.5233896782270817e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.5233896782270817e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (5.471647744988075e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.471647744988075e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.175246207257721e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175246207257721e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.18999097576742e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.18999097576742e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.3304731887474244e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.3304731887474244e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.3304731887474244e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.3304731887474244e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.6288532436394172e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.6288532436394172e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.6288532436394172e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.6288532436394172e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.6893489515633136e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.6893489515633136e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.6893489515633136e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.6893489515633136e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (2.7455184006971133e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.7455184006971133e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.7455184006971133e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.7455184006971133e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.7455184006971133e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.7455184006971133e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.7455184006971133e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.7455184006971133e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.2118420192550276e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420192550276e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.2118420192550276e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420192550276e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.2118420192550276e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420192550276e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.2118420192550276e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420192550276e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.3131455001774585e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.3131455001774585e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.3131455001774585e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.3131455001774585e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (3.3343312895043123e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.3343312895043123e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (4.183932559486715e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.183932559486715e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (7.73503688059987e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.73503688059987e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.0002463643756963395) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.0002463643756963395) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00044585351288405734) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.00044585351288405734) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.00044585351288405734) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.00044585351288405734) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.0005940221543005049) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005049) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005049) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005049) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005049) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005049) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.0005940221543005049) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005049) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.000853385625412585) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.000853385625412585) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.000853385625412585) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.000853385625412585) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.001043524653490751) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.001043524653490751) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.001043524653490751) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.001043524653490751) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.0012803060973496675) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.0012803060973496675) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.0012803060973496675) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.0012803060973496675) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.0013038004788126986) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (0.0013038004788126986) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (0.0013038004788126986) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (0.0013038004788126986) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (0.0022619660624823997) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (0.0022619660624823997) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (0.0022619660624823997) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (0.0022619660624823997) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (0.0022619660624823997) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (0.0022619660624823997) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (0.0022619660624823997) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (0.0022619660624823997) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (0.003989841456619379) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (0.003989841456619379) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (0.003989841456619379) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (0.003989841456619379) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (0.004158797381839992) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.004158797381839992) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.004311038507914309) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.004311038507914309) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.004311038507914309) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.004311038507914309) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.004636976661182585) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.004636976661182585) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.004636976661182585) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.004636976661182585) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.005114473831660347) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.005114473831660347) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.005114473831660347) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.005114473831660347) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.005114473831660347) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660347) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660347) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005114473831660347) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.005241535382803944) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (0.005241535382803944) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (0.005241535382803944) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (0.005241535382803944) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (0.005262642473076794) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.005262642473076794) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.005262642473076794) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.005262642473076794) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.005368659358109227) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005368659358109227) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005379937155839369) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.005379937155839369) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.005379937155839369) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.005379937155839369) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.005652620978017297) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.005652620978017297) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.00570849598596085) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.00570849598596085) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.00570849598596085) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.00570849598596085) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.0059237983365613995) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.0059237983365613995) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.007306759928832776) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.007306759928832776) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.008469978791023307) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.008469978791023307) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.0098417492469626) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.0098417492469626) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.011285190200840569) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (0.011285190200840569) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (0.011756013419819293) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.011756013419819293) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.01456453123117288) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.01456453123117288) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.014603704729162444) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.014603704729162444) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.015225630757226934) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.015225630757226934) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.016024603689179597) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.016024603689179597) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.019028242443847977) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.019028242443847977) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.01925750509525153) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.01925750509525153) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.04587947078129812) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04587947078129812) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.3693708936615631) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.3693708936615631) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.3693708936615631) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.3693708936615631) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.2816425776702344) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.2816425776702344) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.2816425776702343) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.2816425776702343) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.09065144207036506) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.09065144207036506) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.09065144207036506) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.09065144207036506) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0868473758986365) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0868473758986365) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0868473758986365) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0868473758986365) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.07635021950635124) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07635021950635124) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07635021950635124) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07635021950635124) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214127) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.06752385099214127) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214127) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.06752385099214127) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03560837898831284) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.03560837898831284) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.034903343373661466) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.034903343373661466) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.034903343373661466) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.034903343373661466) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.02459186088383002) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.02459186088383002) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.02459186088383002) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.02459186088383002) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.024282117354693135) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.024282117354693135) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.023145130929529165) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.023145130929529165) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.022528440196013185) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.022528440196013185) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019538050311315322) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.019538050311315322) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.019538050311315322) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.019538050311315322) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.01709155315589935) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.01709155315589935) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.01709155315589935) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.01709155315589935) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.016024603689179597) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.016024603689179597) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.016024603689179597) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.016024603689179597) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.01031148248983145) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.01031148248983145) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.01031148248983145) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.01031148248983145) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0098417492469626) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0098417492469626) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0098417492469626) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0098417492469626) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209966) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209966) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209966) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008826368514209966) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008541996625454894) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454894) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454894) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454894) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454894) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454894) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454894) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008541996625454894) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008469978791023307) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023307) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.008469978791023307) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023307) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.00466862031877629) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.00466862031877629) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.003876470899337037) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.003876470899337037) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.0038040661717285524) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0038040661717285524) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0038040661717285524) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0038040661717285524) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178643) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178643) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003356670563832918) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.003356670563832918) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.003267513854423558) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.003267513854423558) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0021413612231015304) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.0021413612231015304) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.0017278753941369787) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.0017278753941369787) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.0016407548553124037) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0016407548553124037) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214168421) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0014528843214168421) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214168421) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0014528843214168421) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870896771024876) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007870896771024876) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0005192743499488531) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.0005192743499488531) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.00019400857029757172) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00019400857029757172) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303542473) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.00013840177303542473) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.141625221149132e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.141625221149132e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.141625221149132e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.141625221149132e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-5.071480736700169e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.071480736700169e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.151346311331131e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.151346311331131e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0882507115853304e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.0882507115853304e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.988511706305271e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.988511706305271e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8742990714927363e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.8742990714927363e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.360956320501329e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.360956320501329e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.3002946561992655e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.3002946561992655e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.1468376508360746e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.1468376508360746e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.1468376508360746e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.1468376508360746e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.352332103764418e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.352332103764418e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.352332103764418e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.352332103764418e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.091637199668966e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637199668966e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637199668966e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637199668966e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637199668966e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637199668966e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637199668966e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.091637199668966e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.074305986536006e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.074305986536006e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.074305986536006e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.074305986536006e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.900128986918862e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.900128986918862e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.900128986918862e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900128986918862e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.86776510463514e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.86776510463514e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.560692465413293e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692465413293e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692465413293e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692465413293e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692465413293e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692465413293e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692465413293e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.560692465413293e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.997018422433731e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.997018422433731e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.997018422433731e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.997018422433731e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.997018422433731e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.997018422433731e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.997018422433731e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.997018422433731e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.5682475214418843e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.5682475214418843e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.5682475214418843e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.5682475214418843e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393086917804e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393086917804e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393086917804e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393086917804e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393086917804e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393086917804e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.3767393086917804e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393086917804e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.88829359532854e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.88829359532854e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6863815473412555e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.6863815473412555e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035783553534296e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7035783553534296e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.209350641938919e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.209350641938919e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773245598555e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773245598555e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773245598555e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773245598555e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773245598555e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773245598555e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.379773245598555e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773245598555e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253798524916e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253798524916e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9742253798524916e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9742253798524916e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.047471655721334e-08) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.047471655721334e-08) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.047471655721334e-08) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.047471655721334e-08) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.209350641938919e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.209350641938919e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0717282185123507e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.0717282185123507e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.0717282185123507e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.0717282185123507e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.2004287494128908e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.2004287494128908e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.2004287494128908e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.2004287494128908e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.7035783553534296e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7035783553534296e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.3120943052949566e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.3120943052949566e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.3120943052949566e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.3120943052949566e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.6863815473412555e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815473412555e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.88829359532854e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.88829359532854e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.092250616392202e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.092250616392202e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.092250616392202e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.092250616392202e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.092250616392202e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.092250616392202e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.092250616392202e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.092250616392202e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.444597854291368e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.444597854291368e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.444597854291368e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.444597854291368e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.6849150954675114e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.6849150954675114e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.6849150954675114e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.6849150954675114e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.246974425770072e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.246974425770072e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.246974425770072e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.246974425770072e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.246974425770072e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.246974425770072e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.246974425770072e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.246974425770072e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.86776510463514e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.86776510463514e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.3002946561992655e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.3002946561992655e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.360956320501329e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.360956320501329e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.8742990714927363e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.8742990714927363e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.883676576075512e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.883676576075512e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.9473560118317607e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9473560118317607e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.9473560118317607e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9473560118317607e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.988511706305271e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.988511706305271e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.0882507115853304e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.0882507115853304e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.151346311331131e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.151346311331131e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.8462016714612725e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.8462016714612725e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.8462016714612725e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.8462016714612725e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.071480736700169e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.071480736700169e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.105526722165083e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.105526722165083e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.105526722165083e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.105526722165083e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.146496327660538e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.146496327660538e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.146496327660538e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.146496327660538e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.1593505020124195e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.1593505020124195e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.1593505020124195e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.1593505020124195e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.427988656746545e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.427988656746545e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.427988656746545e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.427988656746545e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.9358677181370315e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.9358677181370315e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.9358677181370315e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.9358677181370315e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273348367795e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.253273348367795e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.97982579365782e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.97982579365782e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.97982579365782e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.97982579365782e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (4.205548411212673e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.205548411212673e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.205548411212673e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.205548411212673e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00013840177303542473) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.00013840177303542473) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.00018787053389556174) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.00018787053389556174) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.00018787053389556174) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.00018787053389556174) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00019400857029757172) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00019400857029757172) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0002463643756963395) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0002463643756963395) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0002463643756963395) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0002463643756963395) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192743499488531) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.0005192743499488531) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.0007156734248909963) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007156734248909963) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0007156734248909963) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007156734248909963) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024876) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007870896771024876) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.0015324835230731094) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.0015324835230731094) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.0015324835230731094) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.0015324835230731094) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.0016407548553124037) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0016407548553124037) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0017278753941369787) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.0017278753941369787) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.002446497155415973) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.002446497155415973) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.002446497155415973) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.002446497155415973) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.003267513854423558) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.003267513854423558) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.003356670563832918) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.003356670563832918) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.0034841573002178643) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0034841573002178643) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.003876470899337037) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.003876470899337037) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.00466862031877629) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.00466862031877629) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278232) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.004767272188278232) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.004767272188278232) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278232) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.005286546538227086) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.005286546538227086) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.005286546538227086) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.005286546538227086) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.005408954422410146) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.005408954422410146) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.005408954422410146) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.005408954422410146) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.0059237983365613995) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.0059237983365613995) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.0059237983365613995) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.0059237983365613995) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.01071550846979664) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01071550846979664) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.01071550846979664) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01071550846979664) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.010757563953908764) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010757563953908764) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010757563953908764) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010757563953908764) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.014603704729162444) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.014603704729162444) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.014603704729162444) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.014603704729162444) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.019299560579363658) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363658) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363658) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363658) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363658) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363658) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.019299560579363658) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363658) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.05859198873386316) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.05859198873386316) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.775950527435544e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.775950527435544e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.7759505274355464e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.7759505274355464e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.07165035181003022) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.07165035181003022) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.07165035181003025) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.07165035181003025) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.01925750509525153) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.01925750509525153) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.01031148248983145) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.01031148248983145) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008826368514209966) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209966) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.007597464029770597) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.007597464029770597) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.007597464029770597) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.007597464029770597) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311871) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311871) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311871) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311871) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311871) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311871) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311871) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311871) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0053480515826765905) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0053480515826765905) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0053480515826765905) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0053480515826765905) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.003804066171728553) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003804066171728553) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002984166168122003) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.002984166168122003) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.002984166168122003) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.002984166168122003) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.002446497155415973) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.002446497155415973) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.002249412447094005) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.002249412447094005) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.002249412447094005) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002249412447094005) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0021413612231015304) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.0021413612231015304) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.0018638942824587275) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587275) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587275) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587275) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587275) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587275) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0018638942824587275) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587275) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0016407548553124037) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124037) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0016407548553124037) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124037) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0012223378081538767) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538767) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538767) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538767) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538767) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538767) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538767) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0012223378081538767) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.001028329237856305) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.001028329237856305) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.001028329237856305) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.001028329237856305) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1463061453472514e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1463061453472514e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.8742990714927363e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990714927363e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.8742990714927363e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990714927363e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.3002946561992655e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.3002946561992655e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.3002946561992655e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.3002946561992655e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.0444941298251445e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.0444941298251445e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.0444941298251445e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.0444941298251445e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.956079230246303e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.956079230246303e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.956079230246303e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.956079230246303e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105515037405985e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105515037405985e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105515037405985e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105515037405985e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661347213347883e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661347213347883e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661347213347883e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661347213347883e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540341413972775e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540341413972775e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.18999097576742e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.18999097576742e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.876621658415832e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876621658415832e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876621658415832e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876621658415832e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175246207257721e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175246207257721e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.5233896782270817e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.5233896782270817e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0767325320054383e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.0767325320054383e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0767325320054383e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.0767325320054383e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.013471458982922e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.013471458982922e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.9045998842786697e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.9045998842786697e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.9045998842786697e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.9045998842786697e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.6667317545850584e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.6667317545850584e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.6667317545850584e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6667317545850584e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641928403188e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.8505641928403188e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309319164056e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.6569309319164056e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309319164056e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.6569309319164056e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.8505641928403188e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.8505641928403188e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815473412555e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815473412555e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.6863815473412555e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815473412555e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458982922e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.013471458982922e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.5233896782270817e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.5233896782270817e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.6704023908993277e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.6704023908993277e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.6704023908993277e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.6704023908993277e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175246207257721e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175246207257721e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.18999097576742e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.18999097576742e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540341413972775e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540341413972775e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949476487861251e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949476487861251e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.792493957775984e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.792493957775984e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.792493957775984e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.792493957775984e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.883676576075512e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.883676576075512e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.988511706305271e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.988511706305271e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.988511706305271e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.988511706305271e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273348367795e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.253273348367795e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4017109735791383e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4017109735791383e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4017109735791383e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4017109735791383e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.580960369356737e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.580960369356737e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.580960369356737e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.580960369356737e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0005192743499488531) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499488531) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.0005192743499488531) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499488531) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.0007870896771024876) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024876) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024876) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024876) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0011726348316441527) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0011726348316441527) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0011726348316441527) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0011726348316441527) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012366478019245806) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.0012366478019245806) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.0012366478019245806) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.0012366478019245806) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.002200964069500457) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002200964069500457) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002200964069500457) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002200964069500457) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798029) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798029) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798029) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798029) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798029) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798029) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002394972639798029) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798029) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002446497155415973) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.002446497155415973) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.003804066171728553) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.003804066171728553) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.003876470899337037) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.003876470899337037) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.003876470899337037) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.003876470899337037) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.004220813970046584) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.004220813970046584) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.004220813970046584) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.004220813970046584) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.008826368514209966) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.008826368514209966) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.01031148248983145) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01031148248983145) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01925750509525153) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01925750509525153) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.05859198873386316) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.05859198873386316) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.398700901461617e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.398700901461617e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.398700901461617e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.398700901461617e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841573002178643) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841573002178643) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002984166168122004) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.002984166168122004) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.00019400857029757172) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00019400857029757172) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453472514e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453472514e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.792493957775984e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.792493957775984e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540341413972775e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413972775e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540341413972775e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413972775e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.8505641928403188e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641928403188e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641928403188e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641928403188e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458982922e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471458982922e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458982922e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.013471458982922e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.949476487861251e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.949476487861251e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.792493957775984e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.792493957775984e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029757172) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029757172) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002984166168122004) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.002984166168122004) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.0034841573002178643) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0034841573002178643) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
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
  (-46.4639069134128) [I0]
+ (0.7829652070488005) [Z11]
+ (0.7829652070488006) [Z10]
+ (0.808459100518435) [Z12]
+ (0.808459100518435) [Z13]
+ (1.2034393391311968) [Z4]
+ (1.203439339131197) [Z5]
+ (1.3096876618628746) [Z7]
+ (1.3096876618628748) [Z6]
+ (1.369352571169491) [Z8]
+ (1.3693525711694912) [Z9]
+ (1.6538938305476834) [Z2]
+ (1.6538938305476834) [Z3]
+ (12.412630714439707) [Z0]
+ (12.412630714439707) [Z1]
+ (-8.194104982744282e-06) [Y10 Y12]
+ (-8.194104982744282e-06) [X10 X12]
+ (-7.765082269611335e-07) [Y3 Y5]
+ (-7.765082269611335e-07) [X3 X5]
+ (5.929280564418056e-07) [Y4 Y6]
+ (5.929280564418056e-07) [X4 X6]
+ (1.6021751102587612e-06) [Y2 Y4]
+ (1.6021751102587612e-06) [X2 X4]
+ (1.8540565419691585e-06) [Y5 Y7]
+ (1.8540565419691585e-06) [X5 X7]
+ (7.954224667459237e-06) [Y11 Y13]
+ (7.954224667459237e-06) [X11 X13]
+ (0.0032769650657632843) [Y1 Y3]
+ (0.0032769650657632843) [X1 X3]
+ (0.10433061485318182) [Y0 Y2]
+ (0.10433061485318182) [X0 X2]
+ (0.11270381859117455) [Z10 Z12]
+ (0.11270381859117455) [Z11 Z13]
+ (0.11383573685295546) [Z4 Z12]
+ (0.11383573685295546) [Z5 Z13]
+ (0.11952441016894247) [Z6 Z10]
+ (0.11952441016894247) [Z7 Z11]
+ (0.12489977362989704) [Z4 Z10]
+ (0.12489977362989704) [Z5 Z11]
+ (0.12495799328199608) [Z2 Z4]
+ (0.12495799328199608) [Z3 Z5]
+ (0.1279949280139855) [Z2 Z10]
+ (0.1279949280139855) [Z3 Z11]
+ (0.13401737372651495) [Z6 Z12]
+ (0.13401737372651495) [Z7 Z13]
+ (0.13701191913060262) [Z4 Z6]
+ (0.13701191913060262) [Z5 Z7]
+ (0.13734942210158071) [Z6 Z11]
+ (0.13734942210158071) [Z7 Z10]
+ (0.13739112375019166) [Z2 Z6]
+ (0.13739112375019166) [Z3 Z7]
+ (0.13766859133612055) [Z8 Z10]
+ (0.13766859133612055) [Z9 Z11]
+ (0.14011294749713635) [Z2 Z12]
+ (0.14011294749713635) [Z3 Z13]
+ (0.14138903590142188) [Z10 Z13]
+ (0.14138903590142188) [Z11 Z12]
+ (0.14257991128700406) [Z4 Z11]
+ (0.14257991128700406) [Z5 Z10]
+ (0.14722930783256447) [Z8 Z11]
+ (0.14722930783256447) [Z9 Z10]
+ (0.14899426171386249) [Z4 Z7]
+ (0.14899426171386249) [Z5 Z6]
+ (0.14926347060038953) [Z10 Z11]
+ (0.149606925572428) [Z4 Z8]
+ (0.149606925572428) [Z5 Z9]
+ (0.1497349700542438) [Z8 Z12]
+ (0.1497349700542438) [Z9 Z13]
+ (0.15071405482292008) [Z2 Z8]
+ (0.15071405482292008) [Z3 Z9]
+ (0.15138342699113605) [Z6 Z13]
+ (0.15138342699113605) [Z7 Z12]
+ (0.15215040622640968) [Z4 Z13]
+ (0.15215040622640968) [Z5 Z12]
+ (0.15337959171065832) [Z2 Z11]
+ (0.15337959171065832) [Z3 Z10]
+ (0.15435760065293985) [Z12 Z13]
+ (0.15569017457260087) [Z2 Z13]
+ (0.15569017457260087) [Z3 Z12]
+ (0.15582280685847588) [Z8 Z13]
+ (0.15582280685847588) [Z9 Z12]
+ (0.15676384610161448) [Z4 Z9]
+ (0.15676384610161448) [Z5 Z8]
+ (0.15755303804350845) [Z4 Z5]
+ (0.16079755046702937) [Z2 Z5]
+ (0.16079755046702937) [Z3 Z4]
+ (0.16756669356165868) [Z6 Z8]
+ (0.16756669356165868) [Z7 Z9]
+ (0.16853492794657698) [Z2 Z7]
+ (0.16853492794657698) [Z3 Z6]
+ (0.1814400936292806) [Z6 Z9]
+ (0.1814400936292806) [Z7 Z8]
+ (0.18189081243754135) [Z2 Z3]
+ (0.1869081483104358) [Z2 Z9]
+ (0.1869081483104358) [Z3 Z8]
+ (0.1929970026986519) [Z0 Z10]
+ (0.1929970026986519) [Z1 Z11]
+ (0.19392574334979726) [Z6 Z7]
+ (0.1966174995972975) [Z0 Z4]
+ (0.1966174995972975) [Z1 Z5]
+ (0.19936332691272202) [Z0 Z5]
+ (0.19936332691272202) [Z1 Z4]
+ (0.20072843554601127) [Z0 Z11]
+ (0.20072843554601127) [Z1 Z10]
+ (0.21102681234283446) [Z0 Z12]
+ (0.21102681234283446) [Z1 Z13]
+ (0.21631059809666286) [Z0 Z13]
+ (0.21631059809666286) [Z1 Z12]
+ (0.22003977240258757) [Z8 Z9]
+ (0.23671071740435307) [Z0 Z2]
+ (0.23671071740435307) [Z1 Z3]
+ (0.24164696831748045) [Z0 Z6]
+ (0.24164696831748045) [Z1 Z7]
+ (0.24853517285979831) [Z0 Z7]
+ (0.24853517285979831) [Z1 Z6]
+ (0.251294355731382) [Z0 Z3]
+ (0.251294355731382) [Z1 Z2]
+ (0.27232518450366555) [Z0 Z8]
+ (0.27232518450366555) [Z1 Z9]
+ (0.2788345457377413) [Z0 Z9]
+ (0.2788345457377413) [Z1 Z8]
+ (1.186176448413332) [Z0 Z1]
+ (-1.072274838001921e-05) [Y10 Z11 Y12]
+ (-1.072274838001921e-05) [X10 Z11 X12]
+ (-1.0722748380019209e-05) [Y11 Z12 Y13]
+ (-1.0722748380019209e-05) [X11 Z12 X13]
+ (-3.886639568545297e-06) [Y3 Z4 Y5]
+ (-3.886639568545297e-06) [X3 Z4 X5]
+ (-3.886639568545296e-06) [Y2 Z3 Y4]
+ (-3.886639568545296e-06) [X2 Z3 X4]
+ (1.2260276937021385e-05) [Y4 Z5 Y6]
+ (1.2260276937021385e-05) [X4 Z5 X6]
+ (1.2260276937021385e-05) [Y5 Z6 Y7]
+ (1.2260276937021385e-05) [X5 Z6 X7]
+ (0.12507036883986286) [Y0 Z1 Y2]
+ (0.12507036883986286) [X0 Z1 X2]
+ (0.12507036883986286) [Y1 Z2 Y3]
+ (0.12507036883986286) [X1 Z2 X3]
+ (-0.03831466937345421) [Y4 Y5 X12 X13]
+ (-0.03831466937345421) [X4 X5 Y12 Y13]
+ (-0.03619409348751573) [Y2 Y3 X8 X9]
+ (-0.03619409348751573) [X2 X3 Y8 Y9]
+ (-0.03583955718503329) [Y2 Y3 X4 X5]
+ (-0.03583955718503329) [X2 X3 Y4 Y5]
+ (-0.031143804196385322) [Y2 Y3 X6 X7]
+ (-0.031143804196385322) [X2 X3 Y6 Y7]
+ (-0.030787440718607423) [Y6 Z8 Z9 Y10]
+ (-0.030787440718607423) [X6 Z8 Z9 X10]
+ (-0.0301045252735797) [Y6 Z7 Z9 Y10]
+ (-0.0301045252735797) [X6 Z7 Z9 X10]
+ (-0.0301045252735797) [Y7 Z8 Z10 Y11]
+ (-0.0301045252735797) [X7 Z8 Z10 X11]
+ (-0.029812299601130875) [Y6 Z7 Z8 Y10]
+ (-0.029812299601130875) [X6 Z7 Z8 X10]
+ (-0.029812299601130875) [Y7 Z9 Z10 Y11]
+ (-0.029812299601130875) [X7 Z9 Z10 X11]
+ (-0.02868521731024733) [Y10 Y11 X12 X13]
+ (-0.02868521731024733) [X10 X11 Y12 Y13]
+ (-0.02538466369667284) [Y2 Y3 X10 X11]
+ (-0.02538466369667284) [X2 X3 Y10 Y11]
+ (-0.019028318718288154) [Y3 Y4 X11 X12]
+ (-0.019028318718288154) [X3 X4 Y11 Y12]
+ (-0.017825011932638246) [Y6 Y7 X10 X11]
+ (-0.017825011932638246) [X6 X7 Y10 Y11]
+ (-0.017680137657107006) [Y4 Y5 X10 X11]
+ (-0.017680137657107006) [X4 X5 Y10 Y11]
+ (-0.017366053264621137) [Y6 Y7 X12 X13]
+ (-0.017366053264621137) [X6 X7 Y12 Y13]
+ (-0.015577227075464528) [Y2 Y3 X12 X13]
+ (-0.015577227075464528) [X2 X3 Y12 Y13]
+ (-0.014583638327028933) [Y0 Y1 X2 X3]
+ (-0.014583638327028933) [X0 X1 Y2 Y3]
+ (-0.013873400067621902) [Y6 Y7 X8 X9]
+ (-0.013873400067621902) [X6 X7 Y8 Y9]
+ (-0.011982342583259845) [Y4 Y5 X6 X7]
+ (-0.011982342583259845) [X4 X5 Y6 Y7]
+ (-0.011307208030050428) [Y7 Z8 Z9 Y11]
+ (-0.011307208030050428) [X7 Z8 Z9 X11]
+ (-0.011285144618310752) [Y5 Y6 X11 X12]
+ (-0.011285144618310752) [X5 X6 Y11 Y12]
+ (-0.009560716496443926) [Y8 Y9 X10 X11]
+ (-0.009560716496443926) [X8 X9 Y10 Y11]
+ (-0.008125248410123891) [Y1 X2 X8 Y9]
+ (-0.008125248410123891) [Y1 Y2 Y8 Y9]
+ (-0.008125248410123891) [X1 X2 X8 X9]
+ (-0.008125248410123891) [X1 Y2 Y8 X9]
+ (-0.007731432847359362) [Y0 Y1 X10 X11]
+ (-0.007731432847359362) [X0 X1 Y10 Y11]
+ (-0.007156920529186489) [Y4 Y5 X8 X9]
+ (-0.007156920529186489) [X4 X5 Y8 Y9]
+ (-0.0068882045423178575) [Y0 Y1 X6 X7]
+ (-0.0068882045423178575) [X0 X1 Y6 Y7]
+ (-0.00650936123407577) [Y0 Y1 X8 X9]
+ (-0.00650936123407577) [X0 X1 Y8 Y9]
+ (-0.006087836804232079) [Y8 Y9 X12 X13]
+ (-0.006087836804232079) [X8 X9 Y12 Y13]
+ (-0.005283785753828417) [Y0 Y1 X12 X13]
+ (-0.005283785753828417) [X0 X1 Y12 Y13]
+ (-0.005143382387690614) [Y3 Y4 X5 X6]
+ (-0.005143382387690614) [X3 X4 Y5 Y6]
+ (-0.004684920226872378) [Y1 X2 X6 Y7]
+ (-0.004684920226872378) [Y1 Y2 Y6 Y7]
+ (-0.004684920226872378) [X1 X2 X6 X7]
+ (-0.004684920226872378) [X1 Y2 Y6 X7]
+ (-0.004575015188897824) [Y1 X2 X12 Y13]
+ (-0.004575015188897824) [Y1 Y2 Y12 Y13]
+ (-0.004575015188897824) [X1 X2 X12 X13]
+ (-0.004575015188897824) [X1 Y2 Y12 X13]
+ (-0.00442484366849943) [Y1 X2 X4 Y5]
+ (-0.00442484366849943) [Y1 Y2 Y4 Y5]
+ (-0.00442484366849943) [X1 X2 X4 X5]
+ (-0.00442484366849943) [X1 Y2 Y4 X5]
+ (-0.0027458273154245512) [Y0 Y1 X4 X5]
+ (-0.0027458273154245512) [X0 X1 Y4 Y5]
+ (-0.0017991930083868836) [Y1 X2 X10 Y11]
+ (-0.0017991930083868836) [Y1 Y2 Y10 Y11]
+ (-0.0017991930083868836) [X1 X2 X10 X11]
+ (-0.0017991930083868836) [X1 Y2 Y10 X11]
+ (-0.0016639606583707493) [Y2 Z3 Z4 Y6]
+ (-0.0016639606583707493) [X2 Z3 Z4 X6]
+ (-0.0016639606583707493) [Y3 Z5 Z6 Y7]
+ (-0.0016639606583707493) [X3 Z5 Z6 X7]
+ (-0.0004957972885593658) [Y2 Z4 Z5 Y6]
+ (-0.0004957972885593658) [X2 Z4 Z5 X6]
+ (-0.0002922256724488298) [Y7 X8 X9 Y10]
+ (-0.0002922256724488298) [X7 Y8 Y9 X10]
+ (-8.194104982744282e-06) [Z10 Y11 Z12 Y13]
+ (-8.194104982744282e-06) [Z10 X11 Z12 X13]
+ (-7.8015383641215e-06) [Y2 Z3 Y4 Z11]
+ (-7.8015383641215e-06) [X2 Z3 X4 Z11]
+ (-7.8015383641215e-06) [Y3 Z4 Y5 Z10]
+ (-7.8015383641215e-06) [X3 Z4 X5 Z10]
+ (-5.974176887575504e-06) [Y5 X6 X10 Y11]
+ (-5.974176887575504e-06) [Y5 Y6 Y10 Y11]
+ (-5.974176887575504e-06) [X5 X6 X10 X11]
+ (-5.974176887575504e-06) [X5 Y6 Y10 X11]
+ (-4.642978985753806e-06) [Y3 X4 X10 Y11]
+ (-4.642978985753806e-06) [Y3 Y4 Y10 Y11]
+ (-4.642978985753806e-06) [X3 X4 X10 X11]
+ (-4.642978985753806e-06) [X3 Y4 Y10 X11]
+ (-4.281812049697527e-06) [Y4 Z5 Y6 Z11]
+ (-4.281812049697527e-06) [X4 Z5 X6 Z11]
+ (-4.281812049697527e-06) [Y5 Z6 Y7 Z10]
+ (-4.281812049697527e-06) [X5 Z6 X7 Z10]
+ (-3.6945169080444124e-06) [Y4 X5 X11 Y12]
+ (-3.6945169080444124e-06) [Y4 Y5 Y11 Y12]
+ (-3.6945169080444124e-06) [X4 X5 X11 X12]
+ (-3.6945169080444124e-06) [X4 Y5 Y11 X12]
+ (-3.1585593783657433e-06) [Y2 Z3 Y4 Z10]
+ (-3.1585593783657433e-06) [X2 Z3 X4 Z10]
+ (-3.1585593783657433e-06) [Y3 Z4 Y5 Z11]
+ (-3.1585593783657433e-06) [X3 Z4 X5 Z11]
+ (-2.890929951173977e-06) [Z6 Y11 Z12 Y13]
+ (-2.890929951173977e-06) [Z6 X11 Z12 X13]
+ (-2.890929951173977e-06) [Z7 Y10 Z11 Y12]
+ (-2.890929951173977e-06) [Z7 X10 Z11 X12]
+ (-2.177732991365732e-06) [Z0 Y10 Z11 Y12]
+ (-2.177732991365732e-06) [Z0 X10 Z11 X12]
+ (-2.177732991365732e-06) [Z1 Y11 Z12 Y13]
+ (-2.177732991365732e-06) [Z1 X11 Z12 X13]
+ (-1.8551374707694024e-06) [Z6 Y10 Z11 Y12]
+ (-1.8551374707694024e-06) [Z6 X10 Z11 X12]
+ (-1.8551374707694024e-06) [Z7 Y11 Z12 Y13]
+ (-1.8551374707694024e-06) [Z7 X11 Z12 X13]
+ (-1.8163673691029905e-06) [Z4 Y11 Z12 Y13]
+ (-1.8163673691029905e-06) [Z4 X11 Z12 X13]
+ (-1.8163673691029905e-06) [Z5 Y10 Z11 Y12]
+ (-1.8163673691029905e-06) [Z5 X10 Z11 X12]
+ (-1.6149607879714887e-06) [Z0 Y11 Z12 Y13]
+ (-1.6149607879714887e-06) [Z0 X11 Z12 X13]
+ (-1.6149607879714887e-06) [Z1 Y10 Z11 Y12]
+ (-1.6149607879714887e-06) [Z1 X10 Z11 X12]
+ (-1.5973397226870393e-06) [Z8 Y10 Z11 Y12]
+ (-1.5973397226870393e-06) [Z8 X10 Z11 X12]
+ (-1.5973397226870393e-06) [Z9 Y11 Z12 Y13]
+ (-1.5973397226870393e-06) [Z9 X11 Z12 X13]
+ (-1.4548066904412427e-06) [Y3 X4 X6 Y7]
+ (-1.4548066904412427e-06) [Y3 Y4 Y6 Y7]
+ (-1.4548066904412427e-06) [X3 X4 X6 X7]
+ (-1.4548066904412427e-06) [X3 Y4 Y6 X7]
+ (-1.195392075507048e-06) [Y2 Z3 Y4 Z7]
+ (-1.195392075507048e-06) [X2 Z3 X4 Z7]
+ (-1.195392075507048e-06) [Y3 Z4 Y5 Z6]
+ (-1.195392075507048e-06) [X3 Z4 X5 Z6]
+ (-1.1907331084223485e-06) [Z0 Y3 Z4 Y5]
+ (-1.1907331084223485e-06) [Z0 X3 Z4 X5]
+ (-1.1907331084223485e-06) [Z1 Y2 Z3 Y4]
+ (-1.1907331084223485e-06) [Z1 X2 Z3 X4]
+ (-1.0632255143294732e-06) [Z2 Y10 Z11 Y12]
+ (-1.0632255143294732e-06) [Z2 X10 Z11 X12]
+ (-1.0632255143294732e-06) [Z3 Y11 Z12 Y13]
+ (-1.0632255143294732e-06) [Z3 X11 Z12 X13]
+ (-1.0357924804058755e-06) [Y6 X7 X11 Y12]
+ (-1.0357924804058755e-06) [Y6 Y7 Y11 Y12]
+ (-1.0357924804058755e-06) [X6 X7 X11 X12]
+ (-1.0357924804058755e-06) [X6 Y7 Y11 X12]
+ (-9.344969950329796e-07) [Z8 Y11 Z12 Y13]
+ (-9.344969950329796e-07) [Z8 X11 Z12 X13]
+ (-9.344969950329796e-07) [Z9 Y10 Z11 Y12]
+ (-9.344969950329796e-07) [Z9 X10 Z11 X12]
+ (-8.336695577670859e-07) [Z0 Y2 Z3 Y4]
+ (-8.336695577670859e-07) [Z0 X2 Z3 X4]
+ (-8.336695577670859e-07) [Z1 Y3 Z4 Y5]
+ (-8.336695577670859e-07) [Z1 X3 Z4 X5]
+ (-7.956666985119848e-07) [Y3 X4 X8 Y9]
+ (-7.956666985119848e-07) [Y3 Y4 Y8 Y9]
+ (-7.956666985119848e-07) [X3 X4 X8 X9]
+ (-7.956666985119848e-07) [X3 Y4 Y8 X9]
+ (-7.765082269611335e-07) [Y2 Z3 Y4 Z5]
+ (-7.765082269611335e-07) [X2 Z3 X4 Z5]
+ (-5.769436579659663e-07) [Y2 Z3 Y4 Z9]
+ (-5.769436579659663e-07) [X2 Z3 X4 Z9]
+ (-5.769436579659663e-07) [Y3 Z4 Y5 Z8]
+ (-5.769436579659663e-07) [X3 Z4 X5 Z8]
+ (-5.471606053050798e-07) [Y1 Y2 X11 X12]
+ (-5.471606053050798e-07) [X1 X2 Y11 Y12]
+ (-3.5706355065514834e-07) [Y0 X1 X3 Y4]
+ (-3.5706355065514834e-07) [Y0 Y1 Y3 Y4]
+ (-3.5706355065514834e-07) [X0 X1 X3 X4]
+ (-3.5706355065514834e-07) [X0 Y1 Y3 X4]
+ (-1.933212138231048e-07) [Y1 X2 X3 Y4]
+ (-1.933212138231048e-07) [X1 Y2 Y3 X4]
+ (-1.29194582752629e-07) [Y1 Z2 Z3 Y5]
+ (-1.29194582752629e-07) [X1 Z2 Z3 X5]
+ (-3.2261052829136256e-09) [Y1 Y2 X5 X6]
+ (-3.2261052829136256e-09) [X1 X2 Y5 Y6]
+ (3.2261052829136256e-09) [Y1 X2 X5 Y6]
+ (3.2261052829136256e-09) [X1 Y2 Y5 X6]
+ (3.2284041085427206e-08) [Y4 Z5 Y6 Z12]
+ (3.2284041085427206e-08) [X4 Z5 X6 Z12]
+ (3.2284041085427206e-08) [Y5 Z6 Y7 Z13]
+ (3.2284041085427206e-08) [X5 Z6 X7 Z13]
+ (1.6728571523377731e-07) [Y0 Z1 Z3 Y4]
+ (1.6728571523377731e-07) [X0 Z1 Z3 X4]
+ (1.6728571523377731e-07) [Y1 Z2 Z4 Y5]
+ (1.6728571523377731e-07) [X1 Z2 Z4 X5]
+ (1.933212138231048e-07) [Y1 Y2 X3 X4]
+ (1.933212138231048e-07) [X1 X2 Y3 Y4]
+ (2.1872304054601846e-07) [Y2 Z3 Y4 Z8]
+ (2.1872304054601846e-07) [X2 Z3 X4 Z8]
+ (2.1872304054601846e-07) [Y3 Z4 Y5 Z9]
+ (2.1872304054601846e-07) [X3 Z4 X5 Z9]
+ (2.198963770008429e-07) [Y2 X3 X5 Y6]
+ (2.198963770008429e-07) [Y2 Y3 Y5 Y6]
+ (2.198963770008429e-07) [X2 X3 X5 X6]
+ (2.198963770008429e-07) [X2 Y3 Y5 X6]
+ (2.4472648179983764e-07) [Y0 X1 X5 Y6]
+ (2.4472648179983764e-07) [Y0 Y1 Y5 Y6]
+ (2.4472648179983764e-07) [X0 X1 X5 X6]
+ (2.4472648179983764e-07) [X0 Y1 Y5 X6]
+ (2.594146149352246e-07) [Y2 Z3 Y4 Z6]
+ (2.594146149352246e-07) [X2 Z3 X4 Z6]
+ (2.594146149352246e-07) [Y3 Z4 Y5 Z7]
+ (2.594146149352246e-07) [X3 Z4 X5 Z7]
+ (3.606069290568516e-07) [Y0 Z1 Z2 Y4]
+ (3.606069290568516e-07) [X0 Z1 Z2 X4]
+ (3.606069290568516e-07) [Y1 Z3 Z4 Y5]
+ (3.606069290568516e-07) [X1 Z3 Z4 X5]
+ (4.837953348879164e-07) [Y5 X6 X8 Y9]
+ (4.837953348879164e-07) [Y5 Y6 Y8 Y9]
+ (4.837953348879164e-07) [X5 X6 X8 X9]
+ (4.837953348879164e-07) [X5 Y6 Y8 X9]
+ (5.471606053050798e-07) [Y1 X2 X11 Y12]
+ (5.471606053050798e-07) [X1 Y2 Y11 X12]
+ (5.627722033941827e-07) [Y0 X1 X11 Y12]
+ (5.627722033941827e-07) [Y0 Y1 Y11 Y12]
+ (5.627722033941827e-07) [X0 X1 X11 X12]
+ (5.627722033941827e-07) [X0 Y1 Y11 X12]
+ (5.929280564418056e-07) [Z4 Y5 Z6 Y7]
+ (5.929280564418056e-07) [Z4 X5 Z6 X7]
+ (6.628427276540597e-07) [Y8 X9 X11 Y12]
+ (6.628427276540597e-07) [Y8 Y9 Y11 Y12]
+ (6.628427276540597e-07) [X8 X9 X11 X12]
+ (6.628427276540597e-07) [X8 Y9 Y11 X12]
+ (9.509134521006578e-07) [Z2 Y4 Z5 Y6]
+ (9.509134521006578e-07) [Z2 X4 Z5 X6]
+ (9.509134521006578e-07) [Z3 Y5 Z6 Y7]
+ (9.509134521006578e-07) [Z3 X5 Z6 X7]
+ (1.1094125065342535e-06) [Z2 Y11 Z12 Y13]
+ (1.1094125065342535e-06) [Z2 X11 Z12 X13]
+ (1.1094125065342535e-06) [Z3 Y10 Z11 Y12]
+ (1.1094125065342535e-06) [Z3 X10 Z11 X12]
+ (1.1708098291009585e-06) [Z2 Y5 Z6 Y7]
+ (1.1708098291009585e-06) [Z2 X5 Z6 X7]
+ (1.1708098291009585e-06) [Z3 Y4 Z5 Y6]
+ (1.1708098291009585e-06) [Z3 X4 Z5 X6]
+ (1.3980242957341554e-06) [Y4 Z5 Y6 Z8]
+ (1.3980242957341554e-06) [X4 Z5 X6 Z8]
+ (1.3980242957341554e-06) [Y5 Z6 Y7 Z9]
+ (1.3980242957341554e-06) [X5 Z6 X7 Z9]
+ (1.6021751102587612e-06) [Z2 Y3 Z4 Y5]
+ (1.6021751102587612e-06) [Z2 X3 Z4 X5]
+ (1.692364837883615e-06) [Y4 Z5 Y6 Z10]
+ (1.692364837883615e-06) [X4 Z5 X6 Z10]
+ (1.692364837883615e-06) [Y5 Z6 Y7 Z11]
+ (1.692364837883615e-06) [X5 Z6 X7 Z11]
+ (1.8540565419691585e-06) [Y4 Z5 Y6 Z7]
+ (1.8540565419691585e-06) [X4 Z5 X6 Z7]
+ (1.8781495389431566e-06) [Z4 Y10 Z11 Y12]
+ (1.8781495389431566e-06) [Z4 X10 Z11 X12]
+ (1.8781495389431566e-06) [Z5 Y11 Z12 Y13]
+ (1.8781495389431566e-06) [Z5 X11 Z12 X13]
+ (1.8818196306220718e-06) [Y4 Z5 Y6 Z9]
+ (1.8818196306220718e-06) [X4 Z5 X6 Z9]
+ (1.8818196306220718e-06) [Y5 Z6 Y7 Z8]
+ (1.8818196306220718e-06) [X5 Z6 X7 Z8]
+ (2.172638020863076e-06) [Y2 X3 X11 Y12]
+ (2.172638020863076e-06) [Y2 Y3 Y11 Y12]
+ (2.172638020863076e-06) [X2 X3 X11 X12]
+ (2.172638020863076e-06) [X2 Y3 Y11 X12]
+ (3.0992966482288584e-06) [Z0 Y4 Z5 Y6]
+ (3.0992966482288584e-06) [Z0 X4 Z5 X6]
+ (3.0992966482288584e-06) [Z1 Y5 Z6 Y7]
+ (3.0992966482288584e-06) [Z1 X5 Z6 X7]
+ (3.117366403871158e-06) [Y0 Z2 Z3 Y4]
+ (3.117366403871158e-06) [X0 Z2 Z3 X4]
+ (3.344023130028765e-06) [Z0 Y5 Z6 Y7]
+ (3.344023130028765e-06) [Z0 X5 Z6 X7]
+ (3.344023130028765e-06) [Z1 Y4 Z5 Y6]
+ (3.344023130028765e-06) [Z1 X4 Z5 X6]
+ (3.539009992426477e-06) [Y2 Z3 Y4 Z12]
+ (3.539009992426477e-06) [X2 Z3 X4 Z12]
+ (3.539009992426477e-06) [Y3 Z4 Y5 Z13]
+ (3.539009992426477e-06) [X3 Z4 X5 Z13]
+ (4.556473769064522e-06) [Y5 X6 X12 Y13]
+ (4.556473769064522e-06) [Y5 Y6 Y12 Y13]
+ (4.556473769064522e-06) [X5 X6 X12 X13]
+ (4.556473769064522e-06) [X5 Y6 Y12 X13]
+ (4.5887578101512505e-06) [Y4 Z5 Y6 Z13]
+ (4.5887578101512505e-06) [X4 Z5 X6 Z13]
+ (4.5887578101512505e-06) [Y5 Z6 Y7 Z12]
+ (4.5887578101512505e-06) [X5 Z6 X7 Z12]
+ (5.2757834568171146e-06) [Y3 X4 X12 Y13]
+ (5.2757834568171146e-06) [Y3 Y4 Y12 Y13]
+ (5.2757834568171146e-06) [X3 X4 X12 X13]
+ (5.2757834568171146e-06) [X3 Y4 Y12 X13]
+ (7.954224667459237e-06) [Y10 Z11 Y12 Z13]
+ (7.954224667459237e-06) [X10 Z11 X12 Z13]
+ (8.814793449241857e-06) [Y2 Z3 Y4 Z13]
+ (8.814793449241857e-06) [X2 Z3 X4 Z13]
+ (8.814793449241857e-06) [Y3 Z4 Y5 Z12]
+ (8.814793449241857e-06) [X3 Z4 X5 Z12]
+ (0.0002922256724488298) [Y7 Y8 X9 X10]
+ (0.0002922256724488298) [X7 X8 Y9 Y10]
+ (0.0011058984809068691) [Y0 Z1 Y2 Z5]
+ (0.0011058984809068691) [X0 Z1 X2 Z5]
+ (0.0011058984809068691) [Y1 Z2 Y3 Z4]
+ (0.0011058984809068691) [X1 Z2 X3 Z4]
+ (0.001756065962885826) [Y0 Z1 Y2 Z11]
+ (0.001756065962885826) [X0 Z1 X2 Z11]
+ (0.001756065962885826) [Y1 Z2 Y3 Z10]
+ (0.001756065962885826) [X1 Z2 X3 Z10]
+ (0.002326234847618847) [Y0 Z1 Y2 Z13]
+ (0.002326234847618847) [X0 Z1 X2 Z13]
+ (0.002326234847618847) [Y1 Z2 Y3 Z12]
+ (0.002326234847618847) [X1 Z2 X3 Z12]
+ (0.0027458273154245512) [Y0 X1 X4 Y5]
+ (0.0027458273154245512) [X0 Y1 Y4 X5]
+ (0.0029297682785936322) [Y0 Z1 Y2 Z9]
+ (0.0029297682785936322) [X0 Z1 X2 Z9]
+ (0.0029297682785936322) [Y1 Z2 Y3 Z8]
+ (0.0029297682785936322) [X1 Z2 X3 Z8]
+ (0.0032769650657632843) [Y0 Z1 Y2 Z3]
+ (0.0032769650657632843) [X0 Z1 X2 Z3]
+ (0.003347626470700019) [Y0 Z1 Y2 Z7]
+ (0.003347626470700019) [X0 Z1 X2 Z7]
+ (0.003347626470700019) [Y1 Z2 Y3 Z6]
+ (0.003347626470700019) [X1 Z2 X3 Z6]
+ (0.003479421729319864) [Y2 Z3 Z5 Y6]
+ (0.003479421729319864) [X2 Z3 Z5 X6]
+ (0.003479421729319864) [Y3 Z4 Z6 Y7]
+ (0.003479421729319864) [X3 Z4 Z6 X7]
+ (0.0035552589712727106) [Y0 Z1 Y2 Z10]
+ (0.0035552589712727106) [X0 Z1 X2 Z10]
+ (0.0035552589712727106) [Y1 Z2 Y3 Z11]
+ (0.0035552589712727106) [X1 Z2 X3 Z11]
+ (0.005143382387690614) [Y3 X4 X5 Y6]
+ (0.005143382387690614) [X3 Y4 Y5 X6]
+ (0.005283785753828417) [Y0 X1 X12 Y13]
+ (0.005283785753828417) [X0 Y1 Y12 X13]
+ (0.005530742149406301) [Y0 Z1 Y2 Z4]
+ (0.005530742149406301) [X0 Z1 X2 Z4]
+ (0.005530742149406301) [Y1 Z2 Y3 Z5]
+ (0.005530742149406301) [X1 Z2 X3 Z5]
+ (0.006087836804232079) [Y8 X9 X12 Y13]
+ (0.006087836804232079) [X8 Y9 Y12 X13]
+ (0.00650936123407577) [Y0 X1 X8 Y9]
+ (0.00650936123407577) [X0 Y1 Y8 X9]
+ (0.0068882045423178575) [Y0 X1 X6 Y7]
+ (0.0068882045423178575) [X0 Y1 Y6 X7]
+ (0.00690125003651667) [Y0 Z1 Y2 Z12]
+ (0.00690125003651667) [X0 Z1 X2 Z12]
+ (0.00690125003651667) [Y1 Z2 Y3 Z13]
+ (0.00690125003651667) [X1 Z2 X3 Z13]
+ (0.007156920529186489) [Y4 X5 X8 Y9]
+ (0.007156920529186489) [X4 Y5 Y8 X9]
+ (0.007731432847359362) [Y0 X1 X10 Y11]
+ (0.007731432847359362) [X0 Y1 Y10 X11]
+ (0.008032546697572399) [Y0 Z1 Y2 Z6]
+ (0.008032546697572399) [X0 Z1 X2 Z6]
+ (0.008032546697572399) [Y1 Z2 Y3 Z7]
+ (0.008032546697572399) [X1 Z2 X3 Z7]
+ (0.009560716496443926) [Y8 X9 X10 Y11]
+ (0.009560716496443926) [X8 Y9 Y10 X11]
+ (0.011055016688717525) [Y0 Z1 Y2 Z8]
+ (0.011055016688717525) [X0 Z1 X2 Z8]
+ (0.011055016688717525) [Y1 Z2 Y3 Z9]
+ (0.011055016688717525) [X1 Z2 X3 Z9]
+ (0.011285144618310752) [Y5 X6 X11 Y12]
+ (0.011285144618310752) [X5 Y6 Y11 X12]
+ (0.011982342583259845) [Y4 X5 X6 Y7]
+ (0.011982342583259845) [X4 Y5 Y6 X7]
+ (0.013873400067621902) [Y6 X7 X8 Y9]
+ (0.013873400067621902) [X6 Y7 Y8 X9]
+ (0.014583638327028933) [Y0 X1 X2 Y3]
+ (0.014583638327028933) [X0 Y1 Y2 X3]
+ (0.015577227075464528) [Y2 X3 X12 Y13]
+ (0.015577227075464528) [X2 Y3 Y12 X13]
+ (0.017366053264621137) [Y6 X7 X12 Y13]
+ (0.017366053264621137) [X6 Y7 Y12 X13]
+ (0.017680137657107006) [Y4 X5 X10 Y11]
+ (0.017680137657107006) [X4 Y5 Y10 X11]
+ (0.017825011932638246) [Y6 X7 X10 Y11]
+ (0.017825011932638246) [X6 Y7 Y10 X11]
+ (0.019028318718288154) [Y3 X4 X11 Y12]
+ (0.019028318718288154) [X3 Y4 Y11 X12]
+ (0.02538466369667284) [Y2 X3 X10 Y11]
+ (0.02538466369667284) [X2 Y3 Y10 X11]
+ (0.025996206267213842) [Y3 Z4 Z5 Y7]
+ (0.025996206267213842) [X3 Z4 Z5 X7]
+ (0.02868521731024733) [Y10 X11 X12 Y13]
+ (0.02868521731024733) [X10 Y11 Y12 X13]
+ (0.031143804196385322) [Y2 X3 X6 Y7]
+ (0.031143804196385322) [X2 Y3 Y6 X7]
+ (0.03583955718503329) [Y2 X3 X4 Y5]
+ (0.03583955718503329) [X2 Y3 Y4 X5]
+ (0.03619409348751573) [Y2 X3 X8 Y9]
+ (0.03619409348751573) [X2 Y3 Y8 X9]
+ (0.03831466937345421) [Y4 X5 X12 Y13]
+ (0.03831466937345421) [X4 Y5 Y12 X13]
+ (0.10433061485318182) [Z0 Y1 Z2 Y3]
+ (0.10433061485318182) [Z0 X1 Z2 X3]
+ (-0.22847946311059572) [Y6 Z7 Z8 Z9 Y10]
+ (-0.22847946311059572) [X6 Z7 Z8 Z9 X10]
+ (-0.22847946311059564) [Y7 Z8 Z9 Z10 Y11]
+ (-0.22847946311059564) [X7 Z8 Z9 Z10 X11]
+ (3.204142243921359e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.204142243921359e-06) [X0 Z1 Z2 Z3 X4]
+ (3.2041422439213605e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.2041422439213605e-06) [X1 Z2 Z3 Z4 X5]
+ (0.12133242248379796) [Y3 Z4 Z5 Z6 Y7]
+ (0.12133242248379796) [X3 Z4 Z5 Z6 X7]
+ (0.12133242248379798) [Y2 Z3 Z4 Z5 Y6]
+ (0.12133242248379798) [X2 Z3 Z4 Z5 X6]
+ (-0.056084494323010554) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (-0.056084494323010554) [Z0 X6 Z7 Z8 Z9 X10]
+ (-0.056084494323010554) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (-0.056084494323010554) [Z1 X7 Z8 Z9 Z10 X11]
+ (-0.05600713561695384) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (-0.05600713561695384) [Z0 X7 Z8 Z9 Z10 X11]
+ (-0.05600713561695384) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (-0.05600713561695384) [Z1 X6 Z7 Z8 Z9 X10]
+ (-0.04587942403068949) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (-0.04587942403068949) [X0 Z2 Z3 Z4 Z5 X6]
+ (-0.030787440718607423) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (-0.030787440718607423) [Z6 X7 Z8 Z9 Z10 X11]
+ (-0.02510490797005453) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (-0.02510490797005453) [X6 Z7 Z8 Z9 X10 Z12]
+ (-0.02510490797005453) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (-0.02510490797005453) [X7 Z8 Z9 Z10 X11 Z13]
+ (-0.024388989986611893) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (-0.024388989986611893) [Z2 X7 Z8 Z9 Z10 X11]
+ (-0.024388989986611893) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (-0.024388989986611893) [Z3 X6 Z7 Z8 Z9 X10]
+ (-0.02017582495699752) [Y4 Z5 Z6 X7 X11 Y12]
+ (-0.02017582495699752) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (-0.02017582495699752) [X4 Z5 Z6 X7 X11 X12]
+ (-0.02017582495699752) [X4 Z5 Z6 Y7 Y11 X12]
+ (-0.02017582495699752) [Y5 X6 X10 Z11 Z12 Y13]
+ (-0.02017582495699752) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (-0.02017582495699752) [X5 X6 X10 Z11 Z12 X13]
+ (-0.02017582495699752) [X5 Y6 Y10 Z11 Z12 X13]
+ (-0.01902037387517369) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (-0.01902037387517369) [Z2 X6 Z7 Z8 Z9 X10]
+ (-0.01902037387517369) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (-0.01902037387517369) [Z3 X7 Z8 Z9 Z10 X11]
+ (-0.018266758578555413) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (-0.018266758578555413) [Z4 X6 Z7 Z8 Z9 X10]
+ (-0.018266758578555413) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (-0.018266758578555413) [Z5 X7 Z8 Z9 Z10 X11]
+ (-0.01522565905712556) [Y3 Z4 Z5 X6 X10 Y11]
+ (-0.01522565905712556) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (-0.01522565905712556) [X3 Z4 Z5 X6 X10 X11]
+ (-0.01522565905712556) [X3 Z4 Z5 Y6 Y10 X11]
+ (-0.01441118977004547) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (-0.01441118977004547) [X2 Z3 Z4 Z5 X6 Z11]
+ (-0.01441118977004547) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (-0.01441118977004547) [X3 Z4 Z5 Z6 X7 Z10]
+ (-0.01130720803005043) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (-0.01130720803005043) [X6 Z7 Z8 Z9 X10 Z11]
+ (-0.010959994608957198) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (-0.010959994608957198) [Z4 X7 Z8 Z9 Z10 X11]
+ (-0.010959994608957198) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (-0.010959994608957198) [Z5 X6 Z7 Z8 Z9 X10]
+ (-0.010540434329258077) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (-0.010540434329258077) [X6 Z7 Z8 Z9 X10 Z13]
+ (-0.010540434329258077) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (-0.010540434329258077) [X7 Z8 Z9 Z10 X11 Z12]
+ (-0.008890680338686771) [Y4 Z5 X6 X10 Z11 Y12]
+ (-0.008890680338686771) [X4 Z5 Y6 Y10 Z11 X12]
+ (-0.008890680338686771) [Y5 Z6 X7 X11 Z12 Y13]
+ (-0.008890680338686771) [X5 Z6 Y7 Y11 Z12 X13]
+ (-0.00876485821937722) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.00876485821937722) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.00876485821937722) [X2 Z3 Z4 X5 X11 X12]
+ (-0.00876485821937722) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.00876485821937722) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.00876485821937722) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.00876485821937722) [X3 X4 X10 Z11 Z12 X13]
+ (-0.00876485821937722) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.008125248410123893) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.008125248410123893) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.007960839634242163) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (-0.007960839634242163) [X4 Z5 X6 X10 Z11 X12]
+ (-0.007960839634242163) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (-0.007960839634242163) [X5 Z6 X7 X11 Z12 X13]
+ (-0.005368616111438202) [Y2 X3 X7 Z8 Z9 Y10]
+ (-0.005368616111438202) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (-0.005368616111438202) [X2 X3 X7 Z8 Z9 X10]
+ (-0.005368616111438202) [X2 Y3 Y7 Z8 Z9 X10]
+ (-0.004684920226872379) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.004684920226872379) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.0046686152660233736) [Y1 Y2 X7 Z8 Z9 X10]
+ (-0.0046686152660233736) [X1 X2 Y7 Z8 Z9 Y10]
+ (-0.0045750151888978245) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.0045750151888978245) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.00442484366849943) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.00442484366849943) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.003961569373039791) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (-0.003961569373039791) [X0 Z1 Z2 Z4 Z5 X6]
+ (-0.003961569373039791) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (-0.003961569373039791) [X1 Z3 Z4 Z5 Z6 X7]
+ (-0.0024629166214000433) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (-0.0024629166214000433) [X0 Z1 Z2 Z3 Z5 X6]
+ (-0.0024629166214000433) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (-0.0024629166214000433) [X1 Z2 Z3 Z4 Z6 X7]
+ (-0.0022939556230669855) [Y1 Y2 X3 Z4 Z5 X6]
+ (-0.0022939556230669855) [X1 X2 Y3 Z4 Z5 Y6]
+ (-0.0017991930083868833) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.0017991930083868833) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.001727874582332599) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.001727874582332599) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.0016676137499728062) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (-0.0016676137499728062) [X0 Z1 Z3 Z4 Z5 X6]
+ (-0.0016676137499728062) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (-0.0016676137499728062) [X1 Z2 Z4 Z5 Z6 X7]
+ (-0.001609533516299819) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (-0.001609533516299819) [X0 Z1 Z2 Z3 Z4 X6]
+ (-0.001609533516299819) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (-0.001609533516299819) [X1 Z2 Z3 Z5 Z6 X7]
+ (-0.0008533831051002235) [Y1 Z2 Z3 X4 X5 Y6]
+ (-0.0008533831051002235) [X1 Z2 Z3 Y4 Y5 X6]
+ (-0.0006650303448806321) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (-0.0006650303448806321) [X2 Z3 Z4 Z5 X6 Z12]
+ (-0.0006650303448806321) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (-0.0006650303448806321) [X3 Z4 Z5 Z6 X7 Z13]
+ (-0.0004957972885593658) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0004957972885593658) [Z2 X3 Z4 Z5 Z6 X7]
+ (-0.0002922256724488298) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (-0.0002922256724488298) [Y6 Z7 Y8 X9 Z10 X11]
+ (-0.0002922256724488298) [X6 Z7 X8 Y9 Z10 Y11]
+ (-0.0002922256724488298) [X6 Z7 X8 X9 Z10 X11]
+ (-8.77472428330306e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.77472428330306e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.77472428330306e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.77472428330306e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.518288827298427e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.518288827298427e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.518288827298427e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.518288827298427e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.444267426800436e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.444267426800436e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.444267426800436e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.444267426800436e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.2900197037843355e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.2900197037843355e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.2900197037843355e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.2900197037843355e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.974176887575504e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (-5.974176887575504e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (-5.2757834568171146e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.2757834568171146e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.642978985753806e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.642978985753806e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.556473769064522e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (-4.556473769064522e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (-4.253118749854637e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.253118749854637e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-4.183808823153903e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (-4.183808823153903e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (-3.6945169080444124e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.6945169080444124e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.3130170829162586e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (-3.3130170829162586e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (-3.1512959720551978e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (-3.1512959720551978e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (-3.0882457189517452e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (-3.0882457189517452e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (-2.172638020863076e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.172638020863076e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.4548066904412427e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.4548066904412427e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.330456856502623e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.330456856502623e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.2282691235142003e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.2282691235142003e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.0357924804058755e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.0357924804058755e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-7.956666985119848e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.956666985119848e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.733096761705153e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.733096761705153e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.733096761705153e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.733096761705153e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.628427276540597e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.628427276540597e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.579258969265228e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.579258969265228e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.579258969265228e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.579258969265228e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.395302387488417e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.395302387488417e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.395302387488417e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.395302387488417e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.927350229926358e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.927350229926358e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.927350229926358e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.927350229926358e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.627722033941827e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.627722033941827e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.287649471262671e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.287649471262671e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.287649471262671e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.287649471262671e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.287649471262671e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.287649471262671e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.287649471262671e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.287649471262671e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.837953348879164e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (-4.837953348879164e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (-3.5706355065514834e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.5706355065514834e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.32803967336467e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (-3.32803967336467e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (-3.0867709071008954e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.0867709071008954e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.0867709071008954e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.0867709071008954e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.4472648179983764e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (-2.4472648179983764e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (-2.3712704669796307e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.3712704669796307e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.3712704669796307e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.3712704669796307e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.198963770008429e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (-2.198963770008429e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (-1.933212138231048e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.933212138231048e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.933212138231048e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.933212138231048e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.8393943602833787e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.8393943602833787e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.8393943602833787e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.8393943602833787e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.29194582752629e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.29194582752629e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.1076529162215443e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.1076529162215443e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.1076529162215443e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.1076529162215443e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.1076529162215443e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.1076529162215443e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.1076529162215443e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.1076529162215443e-07) [X1 Z2 X3 X10 Z11 X12]
+ (-1.0351501183308176e-09) [Y0 Z1 Z2 Z3 Y4 Z7]
+ (-1.0351501183308176e-09) [X0 Z1 Z2 Z3 X4 Z7]
+ (-1.0351501183308176e-09) [Y1 Z2 Z3 Z4 Y5 Z6]
+ (-1.0351501183308176e-09) [X1 Z2 Z3 Z4 X5 Z6]
+ (1.8395658178843203e-08) [Y0 Z1 X2 X10 Z11 Y12]
+ (1.8395658178843203e-08) [X0 Z1 Y2 Y10 Z11 X12]
+ (1.8395658178843203e-08) [Y1 Z2 X3 X11 Z12 Y13]
+ (1.8395658178843203e-08) [X1 Z2 Y3 Y11 Z12 X13]
+ (2.270208330932526e-08) [Y0 Z1 Z2 X3 X5 Y6]
+ (2.270208330932526e-08) [Y0 Z1 Z2 Y3 Y5 Y6]
+ (2.270208330932526e-08) [X0 Z1 Z2 X3 X5 X6]
+ (2.270208330932526e-08) [X0 Z1 Z2 Y3 Y5 X6]
+ (2.270208330932526e-08) [Y1 X2 X4 Z5 Z6 Y7]
+ (2.270208330932526e-08) [Y1 Y2 Y4 Z5 Z6 Y7]
+ (2.270208330932526e-08) [X1 X2 X4 Z5 Z6 X7]
+ (2.270208330932526e-08) [X1 Y2 Y4 Z5 Z6 X7]
+ (2.592818859251671e-08) [Y0 Z1 X2 X4 Z5 Y6]
+ (2.592818859251671e-08) [X0 Z1 Y2 Y4 Z5 X6]
+ (2.592818859251671e-08) [Y1 Z2 X3 X5 Z6 Y7]
+ (2.592818859251671e-08) [X1 Z2 Y3 Y5 Z6 X7]
+ (8.057465317812338e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.057465317812338e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.057465317812338e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.057465317812338e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.649129619444035e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.649129619444035e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.649129619444035e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.649129619444035e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.3484968907376482e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (1.3484968907376482e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (1.3484968907376482e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (1.3484968907376482e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (1.38075794356631e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (1.38075794356631e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (1.38075794356631e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (1.38075794356631e-07) [X0 Z1 X2 X5 Z6 X7]
+ (1.38075794356631e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (1.38075794356631e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (1.38075794356631e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (1.38075794356631e-07) [X1 Z2 X3 X4 Z5 X6]
+ (1.6077787766604436e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (1.6077787766604436e-07) [X0 Z1 X2 X4 Z5 X6]
+ (1.6077787766604436e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (1.6077787766604436e-07) [X1 Z2 X3 X5 Z6 X7]
+ (1.8290428591005787e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.8290428591005787e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.8290428591005787e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.8290428591005787e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.198963770008429e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (2.198963770008429e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (2.4472648179983764e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (2.4472648179983764e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (3.236183428924034e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.236183428924034e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.236183428924034e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.236183428924034e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.32803967336467e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (3.32803967336467e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (3.5706355065514834e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.5706355065514834e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.837953348879164e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (4.837953348879164e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (5.627722033941827e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.627722033941827e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (6.628427276540597e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.628427276540597e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (7.956666985119848e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.956666985119848e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.306342976635125e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.306342976635125e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.306342976635125e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.306342976635125e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.0357924804058755e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.0357924804058755e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.2282691235142003e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.2282691235142003e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.2393113883744423e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.2393113883744423e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.2393113883744423e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.2393113883744423e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.330456856502623e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.330456856502623e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.4548066904412427e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.4548066904412427e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.172638020863076e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.172638020863076e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.0882457189517452e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (3.0882457189517452e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (3.117366403871158e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.117366403871158e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.1512959720551978e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (3.1512959720551978e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (3.211187439839183e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (3.211187439839183e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (3.211187439839183e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (3.211187439839183e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (3.277438283308398e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (3.277438283308398e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (3.277438283308398e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (3.277438283308398e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (3.3130170829162586e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (3.3130170829162586e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (3.3342618748887637e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.3342618748887637e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.610242250644865e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (3.610242250644865e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (3.610242250644865e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (3.610242250644865e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (3.6945169080444124e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.6945169080444124e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (3.7695836100755603e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (3.7695836100755603e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (4.556473769064522e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (4.556473769064522e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (4.642978985753806e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.642978985753806e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.2757834568171146e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.2757834568171146e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.974176887575504e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (5.974176887575504e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (6.524204522761513e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (6.524204522761513e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (6.524204522761513e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (6.524204522761513e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (7.735870605671254e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (7.735870605671254e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (7.735870605671254e-05) [X0 X1 X7 Z8 Z9 X10]
+ (7.735870605671254e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (0.0008144692870800953) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (0.0008144692870800953) [X2 Z3 Z4 Z5 X6 Z10]
+ (0.0008144692870800953) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (0.0008144692870800953) [X3 Z4 Z5 Z6 X7 Z11]
+ (0.0008533831051002235) [Y1 Z2 Z3 Y4 X5 X6]
+ (0.0008533831051002235) [X1 Z2 Z3 X4 Y5 Y6]
+ (0.0009298407044446028) [Y4 Z5 Y6 X10 Z11 X12]
+ (0.0009298407044446028) [X4 Z5 X6 Y10 Z11 Y12]
+ (0.0009298407044446028) [Y5 Z6 Y7 X11 Z12 X13]
+ (0.0009298407044446028) [X5 Z6 X7 Y11 Z12 Y13]
+ (0.001727874582332599) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.001727874582332599) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.0017991930083868833) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.0017991930083868833) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.0022939556230669855) [Y1 X2 X3 Z4 Z5 Y6]
+ (0.0022939556230669855) [X1 Y2 Y3 Z4 Z5 X6]
+ (0.002779040762877347) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (0.002779040762877347) [X1 Z2 Z3 Z4 Z5 X7]
+ (0.0034938003715444194) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (0.0034938003715444194) [X2 Z3 Z4 Z5 X6 Z13]
+ (0.0034938003715444194) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (0.0034938003715444194) [X3 Z4 Z5 Z6 X7 Z12]
+ (0.004158830716425051) [Y3 Z4 Z5 X6 X12 Y13]
+ (0.004158830716425051) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (0.004158830716425051) [X3 Z4 Z5 X6 X12 X13]
+ (0.004158830716425051) [X3 Z4 Z5 Y6 Y12 X13]
+ (0.00442484366849943) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.00442484366849943) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.0045750151888978245) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.0045750151888978245) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.0046686152660233736) [Y1 X2 X7 Z8 Z9 Y10]
+ (0.0046686152660233736) [X1 Y2 Y7 Z8 Z9 X10]
+ (0.004684920226872379) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.004684920226872379) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.005143382387690614) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (0.005143382387690614) [Y2 Z3 Y4 X5 Z6 X7]
+ (0.005143382387690614) [X2 Z3 X4 Y5 Z6 Y7]
+ (0.005143382387690614) [X2 Z3 X4 X5 Z6 X7]
+ (0.005324817366195254) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.005324817366195254) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.005324817366195254) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.005324817366195254) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.005652607315225919) [Y0 X1 X3 Z4 Z5 Y6]
+ (0.005652607315225919) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (0.005652607315225919) [X0 X1 X3 Z4 Z5 X6]
+ (0.005652607315225919) [X0 Y1 Y3 Z4 Z5 X6]
+ (0.005805121212379543) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (0.005805121212379543) [X2 Z3 Z4 Z5 X6 Z8]
+ (0.005805121212379543) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (0.005805121212379543) [X3 Z4 Z5 Z6 X7 Z9]
+ (0.007306763969598216) [Y4 X5 X7 Z8 Z9 Y10]
+ (0.007306763969598216) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (0.007306763969598216) [X4 X5 X7 Z8 Z9 X10]
+ (0.007306763969598216) [X4 Y5 Y7 Z8 Z9 X10]
+ (0.008125248410123893) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.008125248410123893) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.010263460498910935) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.010263460498910935) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.010263460498910935) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.010263460498910935) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.011755995240387986) [Y3 Z4 Z5 X6 X8 Y9]
+ (0.011755995240387986) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (0.011755995240387986) [X3 Z4 Z5 X6 X8 X9]
+ (0.011755995240387986) [X3 Z4 Z5 Y6 Y8 X9]
+ (0.012214985322755358) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (0.012214985322755358) [Y4 Z5 Y6 X11 Z12 X13]
+ (0.012214985322755358) [X4 Z5 X6 Y11 Z12 Y13]
+ (0.012214985322755358) [X4 Z5 X6 X11 Z12 X13]
+ (0.012214985322755358) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (0.012214985322755358) [Y5 Z6 Y7 X10 Z11 X12]
+ (0.012214985322755358) [X5 Z6 X7 Y10 Z11 Y12]
+ (0.012214985322755358) [X5 Z6 X7 X10 Z11 X12]
+ (0.014564473640796443) [Y7 Z8 Z9 X10 X12 Y13]
+ (0.014564473640796443) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (0.014564473640796443) [X7 Z8 Z9 X10 X12 X13]
+ (0.014564473640796443) [X7 Z8 Z9 Y10 Y12 X13]
+ (0.01558827786510619) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (0.01558827786510619) [X2 Z3 X4 X10 Z11 X12]
+ (0.01558827786510619) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (0.01558827786510619) [X3 Z4 X5 X11 Z12 X13]
+ (0.017561116452767526) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (0.017561116452767526) [X2 Z3 Z4 Z5 X6 Z9]
+ (0.017561116452767526) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (0.017561116452767526) [X3 Z4 Z5 Z6 X7 Z8]
+ (0.024353136084483404) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.024353136084483404) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.024353136084483404) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.024353136084483404) [X2 Z3 X4 X11 Z12 X13]
+ (0.024353136084483404) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.024353136084483404) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.024353136084483404) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.024353136084483404) [X3 Z4 X5 X10 Z11 X12]
+ (0.025996206267213842) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (0.025996206267213842) [X2 Z3 Z4 Z5 X6 Z7]
+ (0.02711487858048771) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (0.02711487858048771) [Z0 X2 Z3 Z4 Z5 X6]
+ (0.02711487858048771) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (0.02711487858048771) [Z1 X3 Z4 Z5 Z6 X7]
+ (0.03276748589571363) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (0.03276748589571363) [Z0 X3 Z4 Z5 Z6 X7]
+ (0.03276748589571363) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (0.03276748589571363) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.042743260062932144) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (-0.042743260062932144) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (-0.04274326006293213) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-0.04274326006293213) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-6.631261950234372e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.631261950234372e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-6.631261950234371e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.631261950234371e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (2.5950813597915578e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.5950813597915578e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.595081359791558e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.595081359791558e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.04764261360017761) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.04764261360017761) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.04764261360017761) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.04764261360017761) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.04587942403068948) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-0.04587942403068948) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.04171881404456858) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.04171881404456858) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.04171881404456858) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.04171881404456858) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.03956454804165919) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.03956454804165919) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.03956454804165919) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03956454804165919) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.039359250391546044) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.039359250391546044) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.039359250391546044) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.039359250391546044) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.03931810723108144) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03931810723108144) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.03931810723108144) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03931810723108144) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03560840035235818) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.03560840035235818) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.029903813458316837) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.029903813458316837) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.029903813458316837) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.029903813458316837) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.028730798001277376) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.028730798001277376) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.028730798001277376) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.028730798001277376) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.025637212809996254) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.025637212809996254) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.025637212809996254) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.025637212809996254) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.0247555079808151) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.0247555079808151) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.0247555079808151) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.0247555079808151) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.02428203162343577) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.02428203162343577) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.023145221653322324) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.023145221653322324) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.022528354240911177) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.022528354240911177) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.021433980116947686) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.021433980116947686) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.021433980116947686) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.021433980116947686) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.019257453001870036) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019257453001870036) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.019028318718288154) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.019028318718288154) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.01888899507747462) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-0.01888899507747462) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-0.01888899507747462) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.01888899507747462) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.016024666095503626) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.016024666095503626) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.01522565905712556) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.01522565905712556) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-0.014603742410730952) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.014603742410730952) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.014564473640796445) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.014564473640796445) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.011755995240387986) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (-0.011755995240387986) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (-0.011285144618310752) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-0.011285144618310752) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-0.009841802923802762) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.009841802923802762) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.009612546714492633) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.009612546714492633) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.009612546714492633) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.009612546714492633) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.008469833341369148) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.008469833341369148) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.007306763969598216) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.007306763969598216) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.005923799555609032) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.005923799555609032) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.005708479853869989) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (-0.005708479853869989) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (-0.005708479853869989) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (-0.005708479853869989) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (-0.005652607315225919) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (-0.005652607315225919) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (-0.005379929634061358) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (-0.005379929634061358) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (-0.005379929634061358) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (-0.005379929634061358) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (-0.005368616111438202) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.005368616111438202) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005262631033418641) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (-0.005262631033418641) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (-0.005262631033418641) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (-0.005262631033418641) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (-0.005241543597049449) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (-0.005241543597049449) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (-0.005241543597049449) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (-0.005241543597049449) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (-0.005114464086474722) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (-0.005114464086474722) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (-0.005114464086474722) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (-0.005114464086474722) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (-0.005114464086474722) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005114464086474722) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005114464086474722) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.005114464086474722) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.004636973516498478) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (-0.004636973516498478) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (-0.004636973516498478) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (-0.004636973516498478) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (-0.004311038607567797) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (-0.004311038607567797) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (-0.004311038607567797) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (-0.004311038607567797) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (-0.004158830716425051) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (-0.004158830716425051) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (-0.0026860422750915325) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.0026860422750915325) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.0026860422750915325) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.0026860422750915325) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.0012803055951312236) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (-0.0012803055951312236) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (-0.0012803055951312236) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (-0.0012803055951312236) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (-0.001043523710413771) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (-0.001043523710413771) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (-0.001043523710413771) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (-0.001043523710413771) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (-0.000958167692758934) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.000958167692758934) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.000958167692758934) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.000958167692758934) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.000958167692758934) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.000958167692758934) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.000958167692758934) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.000958167692758934) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.0008533831051002235) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (-0.0008533831051002235) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (-0.0008533831051002235) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (-0.0008533831051002235) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (-0.0005940157673952677) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (-0.0005940157673952677) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (-0.0005940157673952677) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (-0.0005940157673952677) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (-0.0005940157673952677) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (-0.0005940157673952677) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (-0.0005940157673952677) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (-0.0005940157673952677) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (-0.00044584882045134916) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (-0.00044584882045134916) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (-0.00044584882045134916) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (-0.00044584882045134916) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (-0.00024644081057775043) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00024644081057775043) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.735870605671254e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (-7.735870605671254e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.6103375010526808e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103375010526808e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.6103375010526808e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.6103375010526808e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316614147789224e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.5316614147789224e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316614147789224e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5316614147789224e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.805982057648627e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.805982057648627e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.805982057648627e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.805982057648627e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.089728651356211e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.089728651356211e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.089728651356211e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.089728651356211e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-5.071403670420596e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.071403670420596e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.071403670420596e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.071403670420596e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.734578387226296e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.734578387226296e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.734578387226296e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.734578387226296e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.728781438112961e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.728781438112961e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.728781438112961e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.728781438112961e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.253118749854637e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.253118749854637e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.183808823153903e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-4.183808823153903e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-3.3130170829162586e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-3.3130170829162586e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-3.3130170829162586e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-3.3130170829162586e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-3.2117638767244954e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (-3.2117638767244954e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (-3.2117638767244954e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (-3.2117638767244954e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (-3.2117638767244954e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (-3.2117638767244954e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (-3.2117638767244954e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (-3.2117638767244954e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (-2.745510635248022e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (-2.745510635248022e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (-2.745510635248022e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (-2.745510635248022e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (-2.745510635248022e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (-2.745510635248022e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (-2.745510635248022e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (-2.745510635248022e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (-2.3609472132433583e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609472132433583e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609472132433583e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.3609472132433583e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-1.6893056777780423e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (-1.6893056777780423e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (-1.6893056777780423e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (-1.6893056777780423e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (-1.6288377731119973e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (-1.6288377731119973e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (-1.6288377731119973e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (-1.6288377731119973e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (-1.2282691235142003e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2282691235142003e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.2282691235142003e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2282691235142003e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.867608627375332e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867608627375332e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867608627375332e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.867608627375332e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.189870468852901e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.189870468852901e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (-6.175164900940587e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (-6.175164900940587e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (-5.471606053050798e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.471606053050798e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.523339369760515e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.523339369760515e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.0867709071008954e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.0867709071008954e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.3712704669796307e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.3712704669796307e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.8393943602833787e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.8393943602833787e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-1.7035434501016199e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035434501016199e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035434501016199e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (-1.7035434501016199e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (-9.208945604151754e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-9.208945604151754e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-9.208945604151754e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-9.208945604151754e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-8.057465317812338e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.057465317812338e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (-6.772951821023266e-08) [Y1 Z2 Z3 Y4 X7 Z8 Z9 X10]
+ (-6.772951821023266e-08) [X1 Z2 Z3 X4 Y7 Z8 Z9 Y10]
+ (-6.046790467016498e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (-6.046790467016498e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (-6.046790467016498e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (-6.046790467016498e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (-3.2261052829136256e-09) [Y0 Z1 Z2 Y3 X4 Z5 Z6 X7]
+ (-3.2261052829136256e-09) [X0 Z1 Z2 X3 Y4 Z5 Z6 Y7]
+ (3.2261052829136256e-09) [Y0 Z1 Z2 X3 X4 Z5 Z6 Y7]
+ (3.2261052829136256e-09) [X0 Z1 Z2 Y3 Y4 Z5 Z6 X7]
+ (6.772951821023266e-08) [Y1 Z2 Z3 X4 X7 Z8 Z9 Y10]
+ (6.772951821023266e-08) [X1 Z2 Z3 Y4 Y7 Z8 Z9 X10]
+ (8.057465317812338e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.057465317812338e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (1.8393943602833787e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.8393943602833787e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.3712704669796307e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.3712704669796307e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (2.8885648104383724e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.8885648104383724e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.8885648104383724e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (2.8885648104383724e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (3.0867709071008954e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.0867709071008954e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (3.32803967336467e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (3.32803967336467e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (3.32803967336467e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (3.32803967336467e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (3.4273508370805984e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (3.4273508370805984e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (3.4273508370805984e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (3.4273508370805984e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (4.523339369760515e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.523339369760515e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (4.561117030465762e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (4.561117030465762e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (4.561117030465762e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (4.561117030465762e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (5.471606053050798e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.471606053050798e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.175164900940587e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (6.175164900940587e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (7.189870468852901e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (7.189870468852901e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (7.988467867516003e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (7.988467867516003e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (7.988467867516003e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (7.988467867516003e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (1.330456856502623e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.330456856502623e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.330456856502623e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.330456856502623e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.5224581989453688e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (1.5224581989453688e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (1.5224581989453688e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (1.5224581989453688e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (1.5224581989453688e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (1.5224581989453688e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (1.5224581989453688e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (1.5224581989453688e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (1.654090064419269e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.654090064419269e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.654090064419269e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.654090064419269e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.942946545463106e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.942946545463106e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.942946545463106e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.942946545463106e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.0110740255520777e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.0110740255520777e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.0110740255520777e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.0110740255520777e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.1031634815926736e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.1031634815926736e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.1031634815926736e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.1031634815926736e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.3342618748887637e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.3342618748887637e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (3.5443574219994056e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (3.5443574219994056e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (3.5443574219994056e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (3.5443574219994056e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (3.5443574219994056e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (3.5443574219994056e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (3.5443574219994056e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (3.5443574219994056e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (3.7695836100755603e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.7695836100755603e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.481752018705062e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (6.481752018705062e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.481752018705062e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (6.481752018705062e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (6.652106363715258e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (6.652106363715258e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (6.652106363715258e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (6.652106363715258e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.735870605671254e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (7.735870605671254e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.00013838603701190876) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (0.00013838603701190876) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (0.00013838603701190876) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (0.00013838603701190876) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (0.00024644081057775043) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00024644081057775043) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0013038029824596108) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (0.0013038029824596108) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (0.0013038029824596108) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (0.0013038029824596108) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (0.002261970675218544) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (0.002261970675218544) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (0.002261970675218544) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (0.002261970675218544) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (0.002261970675218544) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (0.002261970675218544) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (0.002261970675218544) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (0.002261970675218544) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (0.0022939556230669855) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (0.0022939556230669855) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (0.0022939556230669855) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (0.0022939556230669855) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (0.002779040762877347) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (0.002779040762877347) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (0.0032675148971540264) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (0.0032675148971540264) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (0.0032675148971540264) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (0.0032675148971540264) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (0.0033566679213672536) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (0.0033566679213672536) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (0.0033566679213672536) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (0.0033566679213672536) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (0.003989845257551144) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (0.003989845257551144) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (0.003989845257551144) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (0.003989845257551144) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (0.004158830716425051) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (0.004158830716425051) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (0.005368616111438202) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.005368616111438202) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.005652607315225919) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (0.005652607315225919) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (0.005923799555609032) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.005923799555609032) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.007306763969598216) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (0.007306763969598216) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.008469833341369148) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.008469833341369148) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.009841802923802762) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.009841802923802762) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.011285144618310752) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (0.011285144618310752) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (0.011755995240387986) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (0.011755995240387986) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (0.014564473640796445) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.014564473640796445) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.014603742410730952) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.014603742410730952) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.01522565905712556) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.01522565905712556) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.016024666095503626) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.016024666095503626) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.019028318718288154) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.019028318718288154) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.019257453001870036) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (0.019257453001870036) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.36937137554480126) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.36937137554480126) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.3693713755448012) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.3693713755448012) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.28164335753442576) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.28164335753442576) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.2816433575344257) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.2816433575344257) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.09065142344960209) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.09065142344960209) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.09065142344960209) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.09065142344960209) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684736029523235) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.08684736029523235) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684736029523235) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.08684736029523235) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.07635036936751445) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07635036936751445) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07635036936751445) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07635036936751445) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752398179971536) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.06752398179971536) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752398179971536) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.06752398179971536) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03560840035235818) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.03560840035235818) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.034903304270776425) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.034903304270776425) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.034903304270776425) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.034903304270776425) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591832088329487) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.024591832088329487) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591832088329487) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.024591832088329487) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.024282031623435774) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.024282031623435774) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.023145221653322324) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.023145221653322324) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.022528354240911177) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.022528354240911177) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.01953808534496087) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.01953808534496087) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.01953808534496087) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.01953808534496087) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.01929949985800725) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.01929949985800725) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.01929949985800725) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.01929949985800725) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.01929949985800725) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.01929949985800725) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.01929949985800725) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.01929949985800725) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.017091621922300824) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.017091621922300824) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.017091621922300824) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.017091621922300824) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.016024666095503622) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.016024666095503622) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.016024666095503622) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.016024666095503622) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.010757524201213346) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010757524201213346) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010757524201213346) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010757524201213346) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010715477345076135) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010715477345076135) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010715477345076135) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010715477345076135) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311472182446947) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311472182446947) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311472182446947) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010311472182446947) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.00984180292380276) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.00984180292380276) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.00984180292380276) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.00984180292380276) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826387567799087) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826387567799087) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826387567799087) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008826387567799087) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008469833341369148) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469833341369148) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.008469833341369148) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469833341369148) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.0046686152660233736) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.0046686152660233736) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.003876482195723115) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.003876482195723115) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.003804063154369746) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804063154369746) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804063154369746) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003804063154369746) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003484154579348114) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003484154579348114) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003356667921367254) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (-0.003356667921367254) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (-0.0032675148971540264) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (-0.0032675148971540264) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (-0.002141348964736889) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.002141348964736889) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.001727874582332599) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.001727874582332599) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.001640759116708638) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.001640759116708638) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528842553212767) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0014528842553212767) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528842553212767) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0014528842553212767) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870893705583223) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007870893705583223) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0005192924120519969) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.0005192924120519969) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.00019401030606455106) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00019401030606455106) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.00013838603701190876) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.00013838603701190876) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-7.141566358820735e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.141566358820735e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.141566358820735e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.141566358820735e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.204685613721387e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-4.204685613721387e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-4.204685613721387e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.204685613721387e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-5.071403670420596e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.071403670420596e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.1512959720551978e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (-3.1512959720551978e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (-3.0882457189517452e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-3.0882457189517452e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-2.9884125742746497e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.9884125742746497e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.874248566315927e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.874248566315927e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.3609472132433583e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.3609472132433583e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.3001958894362682e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.3001958894362682e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.1468226231481132e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.1468226231481132e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.1468226231481132e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.1468226231481132e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.09153985465338e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.09153985465338e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.09153985465338e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.09153985465338e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.09153985465338e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.09153985465338e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.09153985465338e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.09153985465338e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900025746587014e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.900025746587014e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.900025746587014e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900025746587014e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867608627375332e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.867608627375332e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.246849425624068e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (-7.246849425624068e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (-7.246849425624068e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (-7.246849425624068e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (-7.246849425624068e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (-7.246849425624068e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (-7.246849425624068e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (-7.246849425624068e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (-4.769457119860344e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (-4.769457119860344e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (-4.769457119860344e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (-4.769457119860344e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (-4.4490566920306587e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (-4.4490566920306587e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (-4.4490566920306587e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (-4.4490566920306587e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (-4.0921619377628284e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (-4.0921619377628284e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (-4.0921619377628284e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (-4.0921619377628284e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (-4.0921619377628284e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (-4.0921619377628284e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (-4.0921619377628284e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (-4.0921619377628284e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (-3.5682004848860207e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.5682004848860207e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.5682004848860207e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.5682004848860207e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3766863768283964e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3766863768283964e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3766863768283964e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3766863768283964e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3766863768283964e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3766863768283964e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.3766863768283964e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3766863768283964e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.8885648104383724e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.8885648104383724e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.686321596796293e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.686321596796293e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.2498976238515212e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-2.2498976238515212e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-2.2498976238515212e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-2.2498976238515212e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-1.7035434501016199e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.7035434501016199e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1782130991786784e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-1.1782130991786784e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-1.1782130991786784e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-1.1782130991786784e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-1.0716845246830749e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-1.0716845246830749e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-1.0716845246830749e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-1.0716845246830749e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-9.208945604151754e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.208945604151754e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.568947542692194e-08) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-3.568947542692194e-08) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-3.568947542692194e-08) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-3.568947542692194e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-3.568947542692194e-08) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-3.568947542692194e-08) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.568947542692194e-08) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.568947542692194e-08) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.204004278305322e-08) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (3.204004278305322e-08) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (3.204004278305322e-08) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (3.204004278305322e-08) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (3.7068347922294034e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.7068347922294034e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.7068347922294034e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.7068347922294034e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.379737423698198e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (8.379737423698198e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.379737423698198e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (8.379737423698198e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.379737423698198e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (8.379737423698198e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (8.379737423698198e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (8.379737423698198e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (9.208945604151754e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.208945604151754e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7035434501016199e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.7035434501016199e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.686321596796293e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.686321596796293e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.8885648104383724e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.8885648104383724e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.99695180176428e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (4.99695180176428e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (4.99695180176428e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (4.99695180176428e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (4.99695180176428e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (4.99695180176428e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (4.99695180176428e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (4.99695180176428e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (7.560553948076908e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.560553948076908e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (7.560553948076908e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (7.560553948076908e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (7.560553948076908e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.560553948076908e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.560553948076908e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.560553948076908e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.867608627375332e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.867608627375332e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (8.027844211223263e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (8.027844211223263e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (8.027844211223263e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.027844211223263e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (8.398527690442408e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (8.398527690442408e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (8.398527690442408e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (8.398527690442408e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.3001958894362682e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.3001958894362682e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.3609472132433583e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.3609472132433583e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.874248566315927e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.874248566315927e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.8836531693202452e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.8836531693202452e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.94733145254065e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.94733145254065e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.94733145254065e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.94733145254065e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9884125742746497e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9884125742746497e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.0882457189517452e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (3.0882457189517452e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (3.1512959720551978e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (3.1512959720551978e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (3.846190880282124e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.846190880282124e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.846190880282124e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.846190880282124e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.071403670420596e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.071403670420596e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.105462423275899e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.105462423275899e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.105462423275899e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.105462423275899e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.146386769712755e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.146386769712755e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.146386769712755e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.146386769712755e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.159294480412734e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.159294480412734e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.159294480412734e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.159294480412734e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.427926640093664e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.427926640093664e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.427926640093664e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.427926640093664e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.935744026817468e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.935744026817468e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.935744026817468e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.935744026817468e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253185162558361e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.253185162558361e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.979710989591826e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.979710989591826e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.979710989591826e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.979710989591826e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (0.00013838603701190876) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.00013838603701190876) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.0001878748613873628) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0001878748613873628) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0001878748613873628) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0001878748613873628) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00019401030606455106) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (0.00019401030606455106) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (0.00024644081057775043) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024644081057775043) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00024644081057775043) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024644081057775043) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192924120519969) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.0005192924120519969) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.0007156737069701153) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007156737069701153) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0007156737069701153) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007156737069701153) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870893705583223) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007870893705583223) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.001532488562666406) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.001532488562666406) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.001532488562666406) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.001532488562666406) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.001640759116708638) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.001640759116708638) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.001727874582332599) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.001727874582332599) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.0024464634226600414) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.0024464634226600414) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.0024464634226600414) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.0024464634226600414) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.0032675148971540264) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (0.0032675148971540264) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (0.003356667921367254) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (0.003356667921367254) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (0.003484154579348114) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.003484154579348114) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.003876482195723115) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.003876482195723115) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.0046686152660233736) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.0046686152660233736) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.004767276644733376) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.004767276644733376) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.004767276644733376) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.004767276644733376) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.005286569056785371) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.005286569056785371) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.005286569056785371) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.005286569056785371) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.00540897075838952) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.00540897075838952) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.00540897075838952) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.00540897075838952) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.005923799555609031) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923799555609031) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.005923799555609031) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923799555609031) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.008541975656793903) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.008541975656793903) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (0.008541975656793903) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (0.008541975656793903) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.008541975656793903) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.008541975656793903) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.008541975656793903) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.008541975656793903) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.014603742410730952) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.014603742410730952) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.014603742410730952) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.014603742410730952) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.058592151795491126) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.058592151795491126) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.775872105110729e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.775872105110729e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.775872105110729e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.775872105110729e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.07165056250047723) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.07165056250047723) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.07165056250047724) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.07165056250047724) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.019257453001870036) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.019257453001870036) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311472182446947) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311472182446947) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008826387567799087) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826387567799087) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.007597461779089394) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.007597461779089394) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.007597461779089394) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.007597461779089394) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733568640019528) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733568640019528) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733568640019528) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733568640019528) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733568640019528) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733568640019528) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733568640019528) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733568640019528) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348047718417981) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005348047718417981) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005348047718417981) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348047718417981) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.003804063154369746) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003804063154369746) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002984180074787863) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.002984180074787863) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.002984180074787863) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.002984180074787863) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.0024464634226600414) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.0024464634226600414) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.002394967154061103) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002394967154061103) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.002394967154061103) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002394967154061103) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.002394967154061103) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.002394967154061103) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.002394967154061103) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.002394967154061103) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.002249414060671413) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.002249414060671413) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.002249414060671413) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002249414060671413) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.002200956847996551) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.002200956847996551) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.002200956847996551) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.002200956847996551) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002141348964736889) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.002141348964736889) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.0018638931390698664) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638931390698664) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638931390698664) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638931390698664) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638931390698664) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638931390698664) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0018638931390698664) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638931390698664) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.001640759116708638) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.001640759116708638) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.001640759116708638) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.001640759116708638) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0011726297842396024) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0011726297842396024) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0011726297842396024) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0011726297842396024) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1462851005429137e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1462851005429137e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.874248566315927e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.874248566315927e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.874248566315927e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.874248566315927e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.3001958894362682e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.3001958894362682e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.3001958894362682e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.3001958894362682e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.044474164497154e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.044474164497154e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.044474164497154e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.044474164497154e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.95590358925604e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.95590358925604e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.95590358925604e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.95590358925604e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105341118351405e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105341118351405e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105341118351405e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105341118351405e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661200340931785e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661200340931785e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661200340931785e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661200340931785e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540204265453623e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540204265453623e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.189870468852901e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.189870468852901e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.876530396171217e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876530396171217e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876530396171217e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876530396171217e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175164900940587e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-6.175164900940587e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-4.523339369760515e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.523339369760515e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.076662714154423e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.076662714154423e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.076662714154423e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.076662714154423e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.0133988497583217e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0133988497583217e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.9045373795179167e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.9045373795179167e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.9045373795179167e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.9045373795179167e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.6666797746278546e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.6666797746278546e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.6666797746278546e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6666797746278546e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.850562470909683e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.850562470909683e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.846699447579925e-08) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-7.846699447579925e-08) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-6.772951821023266e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-6.772951821023266e-08) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-4.099829395306343e-08) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-4.099829395306343e-08) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.099829395306343e-08) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.099829395306343e-08) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (6.772951821023266e-08) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (6.772951821023266e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (7.846699447579925e-08) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (7.846699447579925e-08) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (1.657009284584616e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.657009284584616e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.657009284584616e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.657009284584616e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.850562470909683e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.850562470909683e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.686321596796293e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.686321596796293e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.686321596796293e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.686321596796293e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0133988497583217e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.0133988497583217e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.523339369760515e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.523339369760515e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.6704081343472914e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.6704081343472914e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.6704081343472914e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.6704081343472914e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175164900940587e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (6.175164900940587e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (7.189870468852901e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.189870468852901e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.540204265453623e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540204265453623e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949307725608679e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949307725608679e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.7924638326242236e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924638326242236e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924638326242236e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.7924638326242236e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.8836531693202452e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.8836531693202452e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.9884125742746497e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9884125742746497e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9884125742746497e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9884125742746497e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253185162558361e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.253185162558361e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4016916447968917e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4016916447968917e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4016916447968917e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4016916447968917e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809380280593125e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.5809380280593125e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809380280593125e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.5809380280593125e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0005192924120519969) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192924120519969) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.0005192924120519969) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192924120519969) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.0007870893705583223) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870893705583223) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870893705583223) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870893705583223) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0010283270637569488) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0010283270637569488) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0010283270637569488) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0010283270637569488) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012223373698214996) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.0012223373698214996) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (0.0012223373698214996) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (0.0012223373698214996) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0012223373698214996) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0012223373698214996) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0012223373698214996) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012223373698214996) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.001236655923556919) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.001236655923556919) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.001236655923556919) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.001236655923556919) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.0024464634226600414) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.0024464634226600414) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.003804063154369746) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.003804063154369746) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.003876482195723115) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.003876482195723115) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.003876482195723115) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.003876482195723115) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.004220835998344783) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.004220835998344783) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.004220835998344783) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.004220835998344783) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.008826387567799087) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.008826387567799087) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.010311472182446947) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010311472182446947) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019257453001870036) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019257453001870036) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.058592151795491126) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.058592151795491126) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.3986653592310757e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.3986653592310757e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.3986653592310754e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.3986653592310754e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003484154579348114) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003484154579348114) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002984180074787863) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.002984180074787863) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.00019401030606455106) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00019401030606455106) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1462851005429137e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1462851005429137e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.7924638326242236e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7924638326242236e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540204265453623e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540204265453623e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540204265453623e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540204265453623e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.850562470909683e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.850562470909683e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.850562470909683e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.850562470909683e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.846699447579925e-08) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-7.846699447579925e-08) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-7.846699447579925e-08) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-7.846699447579925e-08) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-4.099829395306343e-08) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-4.099829395306343e-08) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.099829395306343e-08) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-4.099829395306343e-08) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0133988497583217e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.0133988497583217e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0133988497583217e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.0133988497583217e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.949307725608679e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.949307725608679e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.7924638326242236e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7924638326242236e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019401030606455106) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00019401030606455106) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002984180074787863) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.002984180074787863) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.003484154579348114) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.003484154579348114) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
  (-73.13873149595307) [I0]
+ (-0.18066757506995124) [Z6]
+ (-0.18066757506995113) [Z7]
+ (-0.15961443583716609) [Z5]
+ (-0.15961443583716603) [Z4]
+ (0.174199866121444) [Z3]
+ (0.17419986612144425) [Z2]
+ (0.2275732881450832) [Z0]
+ (0.22757328814508332) [Z1]
+ (-7.954224621284367e-06) [Y5 Y7]
+ (-7.954224621284367e-06) [X5 X7]
+ (8.194104979663414e-06) [Y4 Y6]
+ (8.194104979663414e-06) [X4 X6]
+ (0.11270381859119455) [Z4 Z6]
+ (0.11270381859119455) [Z5 Z7]
+ (0.11952441016896882) [Z0 Z4]
+ (0.11952441016896882) [Z1 Z5]
+ (0.1340173737265343) [Z0 Z6]
+ (0.1340173737265343) [Z1 Z7]
+ (0.13734942210158813) [Z0 Z5]
+ (0.13734942210158813) [Z1 Z4]
+ (0.1376685913361424) [Z2 Z4]
+ (0.1376685913361424) [Z3 Z5]
+ (0.14138903590150373) [Z4 Z7]
+ (0.14138903590150373) [Z5 Z6]
+ (0.14722930783258384) [Z2 Z5]
+ (0.14722930783258384) [Z3 Z4]
+ (0.1492634706004607) [Z4 Z5]
+ (0.14973497005425623) [Z2 Z6]
+ (0.14973497005425623) [Z3 Z7]
+ (0.15138342699114823) [Z0 Z7]
+ (0.15138342699114823) [Z1 Z6]
+ (0.15435760065300175) [Z6 Z7]
+ (0.15582280685848277) [Z2 Z7]
+ (0.15582280685848277) [Z3 Z6]
+ (0.16756669356169857) [Z0 Z2]
+ (0.16756669356169857) [Z1 Z3]
+ (0.18144009362932917) [Z0 Z3]
+ (0.18144009362932917) [Z1 Z2]
+ (0.19392574334988288) [Z0 Z1]
+ (0.22003977240261777) [Z2 Z3]
+ (7.0380237553372784e-06) [Y4 Z5 Y6]
+ (7.0380237553372784e-06) [X4 Z5 X6]
+ (7.0380237553372784e-06) [Y5 Z6 Y7]
+ (7.0380237553372784e-06) [X5 Z6 X7]
+ (-0.03078744071861659) [Y0 Z2 Z3 Y4]
+ (-0.03078744071861659) [X0 Z2 Z3 X4]
+ (-0.030104525273573227) [Y0 Z1 Z3 Y4]
+ (-0.030104525273573227) [X0 Z1 Z3 X4]
+ (-0.030104525273573227) [Y1 Z2 Z4 Y5]
+ (-0.030104525273573227) [X1 Z2 Z4 X5]
+ (-0.029812299601120223) [Y0 Z1 Z2 Y4]
+ (-0.029812299601120223) [X0 Z1 Z2 X4]
+ (-0.029812299601120223) [Y1 Z3 Z4 Y5]
+ (-0.029812299601120223) [X1 Z3 Z4 X5]
+ (-0.02868521731030917) [Y4 Y5 X6 X7]
+ (-0.02868521731030917) [X4 X5 Y6 Y7]
+ (-0.017825011932619296) [Y0 Y1 X4 X5]
+ (-0.017825011932619296) [X0 X1 Y4 Y5]
+ (-0.01736605326461394) [Y0 Y1 X6 X7]
+ (-0.01736605326461394) [X0 X1 Y6 Y7]
+ (-0.013873400067630604) [Y0 Y1 X2 X3]
+ (-0.013873400067630604) [X0 X1 Y2 Y3]
+ (-0.011307208030034632) [Y1 Z2 Z3 Y5]
+ (-0.011307208030034632) [X1 Z2 Z3 X5]
+ (-0.009560716496441435) [Y2 Y3 X4 X5]
+ (-0.009560716496441435) [X2 X3 Y4 Y5]
+ (-0.006087836804226511) [Y2 Y3 X6 X7]
+ (-0.006087836804226511) [X2 X3 Y6 Y7]
+ (-0.0002922256724530053) [Y1 X2 X3 Y4]
+ (-0.0002922256724530053) [X1 Y2 Y3 X4]
+ (-7.954224621284367e-06) [Y4 Z5 Y6 Z7]
+ (-7.954224621284367e-06) [X4 Z5 X6 Z7]
+ (-6.628427248246173e-07) [Y2 X3 X5 Y6]
+ (-6.628427248246173e-07) [Y2 Y3 Y5 Y6]
+ (-6.628427248246173e-07) [X2 X3 X5 X6]
+ (-6.628427248246173e-07) [X2 Y3 Y5 X6]
+ (9.34497010895291e-07) [Z2 Y5 Z6 Y7]
+ (9.34497010895291e-07) [Z2 X5 Z6 X7]
+ (9.34497010895291e-07) [Z3 Y4 Z5 Y6]
+ (9.34497010895291e-07) [Z3 X4 Z5 X6]
+ (1.0357924676135909e-06) [Y0 X1 X5 Y6]
+ (1.0357924676135909e-06) [Y0 Y1 Y5 Y6]
+ (1.0357924676135909e-06) [X0 X1 X5 X6]
+ (1.0357924676135909e-06) [X0 Y1 Y5 X6]
+ (1.5973397357199083e-06) [Z2 Y4 Z5 Y6]
+ (1.5973397357199083e-06) [Z2 X4 Z5 X6]
+ (1.5973397357199083e-06) [Z3 Y5 Z6 Y7]
+ (1.5973397357199083e-06) [Z3 X5 Z6 X7]
+ (1.8551374815823675e-06) [Z0 Y4 Z5 Y6]
+ (1.8551374815823675e-06) [Z0 X4 Z5 X6]
+ (1.8551374815823675e-06) [Z1 Y5 Z6 Y7]
+ (1.8551374815823675e-06) [Z1 X5 Z6 X7]
+ (2.890929949196392e-06) [Z0 Y5 Z6 Y7]
+ (2.890929949196392e-06) [Z0 X5 Z6 X7]
+ (2.890929949196392e-06) [Z1 Y4 Z5 Y6]
+ (2.890929949196392e-06) [Z1 X4 Z5 X6]
+ (8.194104979663414e-06) [Z4 Y5 Z6 Y7]
+ (8.194104979663414e-06) [Z4 X5 Z6 X7]
+ (0.0002922256724530053) [Y1 Y2 X3 X4]
+ (0.0002922256724530053) [X1 X2 Y3 Y4]
+ (0.006087836804226511) [Y2 X3 X6 Y7]
+ (0.006087836804226511) [X2 Y3 Y6 X7]
+ (0.009560716496441435) [Y2 X3 X4 Y5]
+ (0.009560716496441435) [X2 Y3 Y4 X5]
+ (0.013873400067630604) [Y0 X1 X2 Y3]
+ (0.013873400067630604) [X0 Y1 Y2 X3]
+ (0.01736605326461394) [Y0 X1 X6 Y7]
+ (0.01736605326461394) [X0 Y1 Y6 X7]
+ (0.017825011932619296) [Y0 X1 X4 Y5]
+ (0.017825011932619296) [X0 Y1 Y4 X5]
+ (0.02868521731030917) [Y4 X5 X6 Y7]
+ (0.02868521731030917) [X4 Y5 Y6 X7]
+ (-0.043751716121373636) [Y0 Z1 Z2 Z3 Y4]
+ (-0.043751716121373636) [X0 Z1 Z2 Z3 X4]
+ (-0.04375171612137361) [Y1 Z2 Z3 Z4 Y5]
+ (-0.04375171612137361) [X1 Z2 Z3 Z4 X5]
+ (-0.03078744071861659) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (-0.03078744071861659) [Z0 X1 Z2 Z3 Z4 X5]
+ (-0.02510490797004636) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (-0.02510490797004636) [X0 Z1 Z2 Z3 X4 Z6]
+ (-0.02510490797004636) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (-0.02510490797004636) [X1 Z2 Z3 Z4 X5 Z7]
+ (-0.011307208030034632) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-0.011307208030034632) [X0 Z1 Z2 Z3 X4 Z5]
+ (-0.010540434329233855) [Y0 Z1 Z2 Z3 Y4 Z7]
+ (-0.010540434329233855) [X0 Z1 Z2 Z3 X4 Z7]
+ (-0.010540434329233855) [Y1 Z2 Z3 Z4 Y5 Z6]
+ (-0.010540434329233855) [X1 Z2 Z3 Z4 X5 Z6]
+ (-0.00029222567245300524) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-0.00029222567245300524) [Y0 Z1 Y2 X3 Z4 X5]
+ (-0.00029222567245300524) [X0 Z1 X2 Y3 Z4 Y5]
+ (-0.00029222567245300524) [X0 Z1 X2 X3 Z4 X5]
+ (-6.524204495205431e-06) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (-6.524204495205431e-06) [X0 Z1 Z2 Z3 Z4 X6]
+ (-6.524204495205431e-06) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (-6.524204495205431e-06) [X1 Z2 Z3 Z5 Z6 X7]
+ (-3.769583595466695e-06) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (-3.769583595466695e-06) [X0 Z2 Z3 Z4 Z5 X6]
+ (-3.6102422402616236e-06) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (-3.6102422402616236e-06) [X0 Z1 Z3 Z4 Z5 X6]
+ (-3.6102422402616236e-06) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (-3.6102422402616236e-06) [X1 Z2 Z4 Z5 Z6 X7]
+ (-3.3130170641248666e-06) [Y1 Z2 Z3 Y4 X5 X6]
+ (-3.3130170641248666e-06) [X1 Z2 Z3 X4 Y5 Y6]
+ (-3.277438275350354e-06) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (-3.277438275350354e-06) [X0 Z1 Z2 Z4 Z5 X6]
+ (-3.277438275350354e-06) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (-3.277438275350354e-06) [X1 Z3 Z4 Z5 Z6 X7]
+ (-3.211187431077095e-06) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (-3.211187431077095e-06) [X0 Z1 Z2 Z3 Z5 X6]
+ (-3.211187431077095e-06) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (-3.211187431077095e-06) [X1 Z2 Z3 Z4 Z6 X7]
+ (-1.0357924676135909e-06) [Y0 X1 X4 Z5 Z6 Y7]
+ (-1.0357924676135909e-06) [X0 Y1 Y4 Z5 Z6 X7]
+ (-6.628427248246173e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-6.628427248246173e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-3.328039649112694e-07) [Y1 X2 X3 Z4 Z5 Y6]
+ (-3.328039649112694e-07) [X1 Y2 Y3 Z4 Z5 X6]
+ (3.328039649112694e-07) [Y1 Y2 X3 Z4 Z5 X6]
+ (3.328039649112694e-07) [X1 X2 Y3 Z4 Z5 Y6]
+ (6.628427248246173e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (6.628427248246173e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (1.0357924676135909e-06) [Y0 Y1 X4 Z5 Z6 X7]
+ (1.0357924676135909e-06) [X0 X1 Y4 Z5 Z6 Y7]
+ (3.3130170641248666e-06) [Y1 Z2 Z3 X4 X5 Y6]
+ (3.3130170641248666e-06) [X1 Z2 Z3 Y4 Y5 X6]
+ (4.183808814579165e-06) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (4.183808814579165e-06) [X1 Z2 Z3 Z4 Z5 X7]
+ (0.0145644736408125) [Y1 Z2 Z3 X4 X6 Y7]
+ (0.0145644736408125) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (0.0145644736408125) [X1 Z2 Z3 X4 X6 X7]
+ (0.0145644736408125) [X1 Z2 Z3 Y4 Y6 X7]
+ (-5.1056810933798485e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (-5.1056810933798485e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (-5.1056810933798485e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-5.1056810933798485e-06) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.014564473640812503) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (-0.014564473640812503) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (-3.769583595466695e-06) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-3.769583595466695e-06) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-3.328039649112694e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-3.328039649112694e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-3.328039649112694e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-3.328039649112694e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (3.3130170641248666e-06) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (3.3130170641248666e-06) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (3.3130170641248666e-06) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (3.3130170641248666e-06) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (4.183808814579165e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (4.183808814579165e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (0.014564473640812503) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (0.014564473640812503) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
/opt/hostedtoolcache/Python/3.7.15/x64/lib/python3.7/site-packages/pyscf/lib/misc.py:46: H5pyDeprecationWarning: Using default_file_mode other than 'r' is deprecated. Pass the mode to h5py.File() instead.
  h5py.get_config().default_file_mode = 'a'
  (-46.46390678868891) [I0]
+ (0.7829661725950168) [Z11]
+ (0.7829661725950169) [Z10]
+ (0.8084581961720478) [Z12]
+ (0.8084581961720478) [Z13]
+ (1.2034402289145623) [Z5]
+ (1.2034402289145627) [Z4]
+ (1.3096862988615434) [Z6]
+ (1.309686298861544) [Z7]
+ (1.3693525634718162) [Z8]
+ (1.3693525634718164) [Z9]
+ (1.6538942226831719) [Z3]
+ (1.653894222683172) [Z2]
+ (12.41263074211178) [Z0]
+ (12.41263074211178) [Z1]
+ (-8.194261372220718e-06) [Y10 Y12]
+ (-8.194261372220718e-06) [X10 X12]
+ (-1.8540608579495387e-06) [Y5 Y7]
+ (-1.8540608579495387e-06) [X5 X7]
+ (-7.764994117642362e-07) [Y3 Y5]
+ (-7.764994117642362e-07) [X3 X5]
+ (-5.929765815772743e-07) [Y4 Y6]
+ (-5.929765815772743e-07) [X4 X6]
+ (1.6021167406494994e-06) [Y2 Y4]
+ (1.6021167406494994e-06) [X2 X4]
+ (7.954413176529942e-06) [Y11 Y13]
+ (7.954413176529942e-06) [X11 X13]
+ (0.0032769719312316288) [Y1 Y3]
+ (0.0032769719312316288) [X1 X3]
+ (0.1043306478065143) [Y0 Y2]
+ (0.1043306478065143) [X0 X2]
+ (0.11270386920332218) [Z10 Z12]
+ (0.11270386920332218) [Z11 Z13]
+ (0.11383573679388667) [Z4 Z12]
+ (0.11383573679388667) [Z5 Z13]
+ (0.11952438964682682) [Z6 Z10]
+ (0.11952438964682682) [Z7 Z11]
+ (0.12489990917237599) [Z4 Z10]
+ (0.12489990917237599) [Z5 Z11]
+ (0.12495807739503226) [Z2 Z4]
+ (0.12495807739503226) [Z3 Z5]
+ (0.12799502492468418) [Z2 Z10]
+ (0.12799502492468418) [Z3 Z11]
+ (0.13401715261963723) [Z6 Z12]
+ (0.13401715261963723) [Z7 Z13]
+ (0.1370119167404076) [Z4 Z6]
+ (0.1370119167404076) [Z5 Z7]
+ (0.13734953064261318) [Z6 Z11]
+ (0.13734953064261318) [Z7 Z10]
+ (0.13739104762683246) [Z2 Z6]
+ (0.13739104762683246) [Z3 Z7]
+ (0.13766872645852568) [Z8 Z10]
+ (0.13766872645852568) [Z9 Z11]
+ (0.14011289865354837) [Z2 Z12]
+ (0.14011289865354837) [Z3 Z13]
+ (0.14138905291942822) [Z10 Z13]
+ (0.14138905291942822) [Z11 Z12]
+ (0.1425799771248575) [Z4 Z11]
+ (0.1425799771248575) [Z5 Z10]
+ (0.14722943218766166) [Z8 Z11]
+ (0.14722943218766166) [Z9 Z10]
+ (0.14899430575065553) [Z4 Z7]
+ (0.14899430575065553) [Z5 Z6]
+ (0.14926355147388903) [Z10 Z11]
+ (0.14960702684445296) [Z4 Z8]
+ (0.14960702684445296) [Z5 Z9]
+ (0.14973486803496938) [Z8 Z12]
+ (0.14973486803496938) [Z9 Z13]
+ (0.150714081210083) [Z2 Z8]
+ (0.150714081210083) [Z3 Z9]
+ (0.1513832716142886) [Z6 Z13]
+ (0.1513832716142886) [Z7 Z12]
+ (0.15215040708869054) [Z4 Z13]
+ (0.15215040708869054) [Z5 Z12]
+ (0.15337968243314165) [Z2 Z11]
+ (0.15337968243314165) [Z3 Z10]
+ (0.1543574865722366) [Z12 Z13]
+ (0.1556901067175248) [Z2 Z13]
+ (0.1556901067175248) [Z3 Z12]
+ (0.15582269051553121) [Z8 Z13]
+ (0.15582269051553121) [Z9 Z12]
+ (0.15676396176430993) [Z4 Z9]
+ (0.15676396176430993) [Z5 Z8]
+ (0.15755314797985662) [Z4 Z5]
+ (0.16079764534838578) [Z2 Z5]
+ (0.16079764534838578) [Z3 Z4]
+ (0.16756653265461285) [Z6 Z8]
+ (0.16756653265461285) [Z7 Z9]
+ (0.1685348656157996) [Z2 Z7]
+ (0.1685348656157996) [Z3 Z6]
+ (0.18143991440303897) [Z6 Z9]
+ (0.18143991440303897) [Z7 Z8]
+ (0.18189085790751403) [Z2 Z3]
+ (0.18690820476912562) [Z2 Z9]
+ (0.18690820476912562) [Z3 Z8]
+ (0.19299723935364244) [Z0 Z10]
+ (0.19299723935364244) [Z1 Z11]
+ (0.1939253461327024) [Z6 Z7]
+ (0.1966177089034217) [Z0 Z4]
+ (0.1966177089034217) [Z1 Z5]
+ (0.1993635453736085) [Z0 Z5]
+ (0.1993635453736085) [Z1 Z4]
+ (0.20072866460441774) [Z0 Z11]
+ (0.20072866460441774) [Z1 Z10]
+ (0.21102659849791552) [Z0 Z12]
+ (0.21102659849791552) [Z1 Z13]
+ (0.2163103749863185) [Z0 Z13]
+ (0.2163103749863185) [Z1 Z12]
+ (0.2200397733437609) [Z8 Z9]
+ (0.23671080783830462) [Z0 Z2]
+ (0.23671080783830462) [Z1 Z3]
+ (0.2416466393601726) [Z0 Z6]
+ (0.2416466393601726) [Z1 Z7]
+ (0.2485348337131432) [Z0 Z7]
+ (0.2485348337131432) [Z1 Z6]
+ (0.2512944567459173) [Z0 Z3]
+ (0.2512944567459173) [Z1 Z2]
+ (0.2723251830660572) [Z0 Z8]
+ (0.2723251830660572) [Z1 Z9]
+ (0.2788345442672344) [Z0 Z9]
+ (0.2788345442672344) [Z1 Z8]
+ (1.1861763734860526) [Z0 Z1]
+ (-1.2260484989042322e-05) [Y5 Z6 Y7]
+ (-1.2260484989042322e-05) [X5 Z6 X7]
+ (-1.2260484989042317e-05) [Y4 Z5 Y6]
+ (-1.2260484989042317e-05) [X4 Z5 X6]
+ (-1.0722312157476136e-05) [Y10 Z11 Y12]
+ (-1.0722312157476136e-05) [X10 Z11 X12]
+ (-1.072231215747613e-05) [Y11 Z12 Y13]
+ (-1.072231215747613e-05) [X11 Z12 X13]
+ (-3.887051671852945e-06) [Y3 Z4 Y5]
+ (-3.887051671852945e-06) [X3 Z4 X5]
+ (-3.887051671852942e-06) [Y2 Z3 Y4]
+ (-3.887051671852942e-06) [X2 Z3 X4]
+ (0.1250703257977208) [Y0 Z1 Y2]
+ (0.1250703257977208) [X0 Z1 X2]
+ (0.1250703257977208) [Y1 Z2 Y3]
+ (0.1250703257977208) [X1 Z2 X3]
+ (-0.0383146702948039) [Y4 Y5 X12 X13]
+ (-0.0383146702948039) [X4 X5 Y12 Y13]
+ (-0.03619412355904262) [Y2 Y3 X8 X9]
+ (-0.03619412355904262) [X2 X3 Y8 Y9]
+ (-0.03583956795335353) [Y2 Y3 X4 X5]
+ (-0.03583956795335353) [X2 X3 Y4 Y5]
+ (-0.031143817988967145) [Y2 Y3 X6 X7]
+ (-0.031143817988967145) [X2 X3 Y6 Y7]
+ (-0.028685183716106018) [Y10 Y11 X12 X13]
+ (-0.028685183716106018) [X10 X11 Y12 Y13]
+ (-0.02599617759802126) [Y3 Z4 Z5 Y7]
+ (-0.02599617759802126) [X3 Z4 Z5 X7]
+ (-0.025384657508457472) [Y2 Y3 X10 X11]
+ (-0.025384657508457472) [X2 X3 Y10 Y11]
+ (-0.01902824244384737) [Y3 Y4 X11 X12]
+ (-0.01902824244384737) [X3 X4 Y11 Y12]
+ (-0.017825140995786356) [Y6 Y7 X10 X11]
+ (-0.017825140995786356) [X6 X7 Y10 Y11]
+ (-0.01768006795248153) [Y4 Y5 X10 X11]
+ (-0.01768006795248153) [X4 X5 Y10 Y11]
+ (-0.01736611899465137) [Y6 Y7 X12 X13]
+ (-0.01736611899465137) [X6 X7 Y12 Y13]
+ (-0.01557720806397647) [Y2 Y3 X12 X13]
+ (-0.01557720806397647) [X2 X3 Y12 Y13]
+ (-0.014583648907612698) [Y0 Y1 X2 X3]
+ (-0.014583648907612698) [X0 X1 Y2 Y3]
+ (-0.013873381748426129) [Y6 Y7 X8 X9]
+ (-0.013873381748426129) [X6 X7 Y8 Y9]
+ (-0.011982389010247911) [Y4 Y5 X6 X7]
+ (-0.011982389010247911) [X4 X5 Y6 Y7]
+ (-0.011285190200840855) [Y5 X6 X11 Y12]
+ (-0.011285190200840855) [X5 Y6 Y11 X12]
+ (-0.009560705729135963) [Y8 Y9 X10 X11]
+ (-0.009560705729135963) [X8 X9 Y10 Y11]
+ (-0.008125251921381044) [Y1 X2 X8 Y9]
+ (-0.008125251921381044) [Y1 Y2 Y8 Y9]
+ (-0.008125251921381044) [X1 X2 X8 X9]
+ (-0.008125251921381044) [X1 Y2 Y8 X9]
+ (-0.007731425250775303) [Y0 Y1 X10 X11]
+ (-0.007731425250775303) [X0 X1 Y10 Y11]
+ (-0.007156934919856946) [Y4 Y5 X8 X9]
+ (-0.007156934919856946) [X4 X5 Y8 Y9]
+ (-0.006888194352970597) [Y0 Y1 X6 X7]
+ (-0.006888194352970597) [X0 X1 Y6 Y7]
+ (-0.0065093612011772484) [Y0 Y1 X8 X9]
+ (-0.0065093612011772484) [X0 X1 Y8 Y9]
+ (-0.006087822480561859) [Y8 Y9 X12 X13]
+ (-0.006087822480561859) [X8 X9 Y12 Y13]
+ (-0.0052837764884029696) [Y0 Y1 X12 X13]
+ (-0.0052837764884029696) [X0 X1 Y12 Y13]
+ (-0.005143391768825061) [Y3 X4 X5 Y6]
+ (-0.005143391768825061) [X3 Y4 Y5 X6]
+ (-0.004684903388155227) [Y1 X2 X6 Y7]
+ (-0.004684903388155227) [Y1 Y2 Y6 Y7]
+ (-0.004684903388155227) [X1 X2 X6 X7]
+ (-0.004684903388155227) [X1 Y2 Y6 X7]
+ (-0.004575007626639207) [Y1 X2 X12 Y13]
+ (-0.004575007626639207) [Y1 Y2 Y12 Y13]
+ (-0.004575007626639207) [X1 X2 X12 X13]
+ (-0.004575007626639207) [X1 Y2 Y12 X13]
+ (-0.004424855449441867) [Y1 X2 X4 Y5]
+ (-0.004424855449441867) [Y1 Y2 Y4 Y5]
+ (-0.004424855449441867) [X1 X2 X4 X5]
+ (-0.004424855449441867) [X1 Y2 Y4 X5]
+ (-0.0034795118903342796) [Y2 Z3 Z5 Y6]
+ (-0.0034795118903342796) [X2 Z3 Z5 X6]
+ (-0.0034795118903342796) [Y3 Z4 Z6 Y7]
+ (-0.0034795118903342796) [X3 Z4 Z6 X7]
+ (-0.002745836470186818) [Y0 Y1 X4 X5]
+ (-0.002745836470186818) [X0 X1 Y4 Y5]
+ (-0.0017992194936630062) [Y1 X2 X10 Y11]
+ (-0.0017992194936630062) [Y1 Y2 Y10 Y11]
+ (-0.0017992194936630062) [X1 X2 X10 X11]
+ (-0.0017992194936630062) [X1 Y2 Y10 X11]
+ (-0.00029219862611108856) [Y7 Y8 X9 X10]
+ (-0.00029219862611108856) [X7 X8 Y9 Y10]
+ (-8.194261372220718e-06) [Z10 Y11 Z12 Y13]
+ (-8.194261372220718e-06) [Z10 X11 Z12 X13]
+ (-7.801707500572277e-06) [Y2 Z3 Y4 Z11]
+ (-7.801707500572277e-06) [X2 Z3 X4 Z11]
+ (-7.801707500572277e-06) [Y3 Z4 Y5 Z10]
+ (-7.801707500572277e-06) [X3 Z4 X5 Z10]
+ (-4.64305106855627e-06) [Y3 X4 X10 Y11]
+ (-4.64305106855627e-06) [Y3 Y4 Y10 Y11]
+ (-4.64305106855627e-06) [X3 X4 X10 X11]
+ (-4.64305106855627e-06) [X3 Y4 Y10 X11]
+ (-4.588855155786482e-06) [Y4 Z5 Y6 Z13]
+ (-4.588855155786482e-06) [X4 Z5 X6 Z13]
+ (-4.588855155786482e-06) [Y5 Z6 Y7 Z12]
+ (-4.588855155786482e-06) [X5 Z6 X7 Z12]
+ (-4.5565692182674655e-06) [Y5 X6 X12 Y13]
+ (-4.5565692182674655e-06) [Y5 Y6 Y12 Y13]
+ (-4.5565692182674655e-06) [X5 X6 X12 X13]
+ (-4.5565692182674655e-06) [X5 Y6 Y12 X13]
+ (-3.694513294564377e-06) [Y4 X5 X11 Y12]
+ (-3.694513294564377e-06) [Y4 Y5 Y11 Y12]
+ (-3.694513294564377e-06) [X4 X5 X11 X12]
+ (-3.694513294564377e-06) [X4 Y5 Y11 X12]
+ (-3.344081556474469e-06) [Z0 Y5 Z6 Y7]
+ (-3.344081556474469e-06) [Z0 X5 Z6 X7]
+ (-3.344081556474469e-06) [Z1 Y4 Z5 Y6]
+ (-3.344081556474469e-06) [Z1 X4 Z5 X6]
+ (-3.1586564320160084e-06) [Y2 Z3 Y4 Z10]
+ (-3.1586564320160084e-06) [X2 Z3 X4 Z10]
+ (-3.1586564320160084e-06) [Y3 Z4 Y5 Z11]
+ (-3.1586564320160084e-06) [X3 Z4 X5 Z11]
+ (-3.099349243608753e-06) [Z0 Y4 Z5 Y6]
+ (-3.099349243608753e-06) [Z0 X4 Z5 X6]
+ (-3.099349243608753e-06) [Z1 Y5 Z6 Y7]
+ (-3.099349243608753e-06) [Z1 X5 Z6 X7]
+ (-2.8909678816447418e-06) [Z6 Y11 Z12 Y13]
+ (-2.8909678816447418e-06) [Z6 X11 Z12 X13]
+ (-2.8909678816447418e-06) [Z7 Y10 Z11 Y12]
+ (-2.8909678816447418e-06) [Z7 X10 Z11 X12]
+ (-2.1776646050422836e-06) [Z0 Y10 Z11 Y12]
+ (-2.1776646050422836e-06) [Z0 X10 Z11 X12]
+ (-2.1776646050422836e-06) [Z1 Y11 Z12 Y13]
+ (-2.1776646050422836e-06) [Z1 X11 Z12 X13]
+ (-1.8818501832017672e-06) [Y4 Z5 Y6 Z9]
+ (-1.8818501832017672e-06) [X4 Z5 X6 Z9]
+ (-1.8818501832017672e-06) [Y5 Z6 Y7 Z8]
+ (-1.8818501832017672e-06) [X5 Z6 X7 Z8]
+ (-1.8551201215458772e-06) [Z6 Y10 Z11 Y12]
+ (-1.8551201215458772e-06) [Z6 X10 Z11 X12]
+ (-1.8551201215458772e-06) [Z7 Y11 Z12 Y13]
+ (-1.8551201215458772e-06) [Z7 X11 Z12 X13]
+ (-1.8540608579495385e-06) [Y4 Z5 Y6 Z7]
+ (-1.8540608579495385e-06) [X4 Z5 X6 Z7]
+ (-1.8163031697010795e-06) [Z4 Y11 Z12 Y13]
+ (-1.8163031697010795e-06) [Z4 X11 Z12 X13]
+ (-1.8163031697010795e-06) [Z5 Y10 Z11 Y12]
+ (-1.8163031697010795e-06) [Z5 X10 Z11 X12]
+ (-1.692397828642373e-06) [Y4 Z5 Y6 Z10]
+ (-1.692397828642373e-06) [X4 Z5 X6 Z10]
+ (-1.692397828642373e-06) [Y5 Z6 Y7 Z11]
+ (-1.692397828642373e-06) [X5 Z6 X7 Z11]
+ (-1.6148794138892746e-06) [Z0 Y11 Z12 Y13]
+ (-1.6148794138892746e-06) [Z0 X11 Z12 X13]
+ (-1.6148794138892746e-06) [Z1 Y10 Z11 Y12]
+ (-1.6148794138892746e-06) [Z1 X10 Z11 X12]
+ (-1.5973171977925287e-06) [Z8 Y10 Z11 Y12]
+ (-1.5973171977925287e-06) [Z8 X10 Z11 X12]
+ (-1.5973171977925287e-06) [Z9 Y11 Z12 Y13]
+ (-1.5973171977925287e-06) [Z9 X11 Z12 X13]
+ (-1.4548424489964395e-06) [Y3 X4 X6 Y7]
+ (-1.4548424489964395e-06) [Y3 Y4 Y6 Y7]
+ (-1.4548424489964395e-06) [X3 X4 X6 X7]
+ (-1.4548424489964395e-06) [X3 Y4 Y6 X7]
+ (-1.3980449081375773e-06) [Y4 Z5 Y6 Z8]
+ (-1.3980449081375773e-06) [X4 Z5 X6 Z8]
+ (-1.3980449081375773e-06) [Y5 Z6 Y7 Z9]
+ (-1.3980449081375773e-06) [X5 Z6 X7 Z9]
+ (-1.195489009836766e-06) [Y2 Z3 Y4 Z7]
+ (-1.195489009836766e-06) [X2 Z3 X4 Z7]
+ (-1.195489009836766e-06) [Y3 Z4 Y5 Z6]
+ (-1.195489009836766e-06) [X3 Z4 X5 Z6]
+ (-1.1908508080805856e-06) [Z0 Y3 Z4 Y5]
+ (-1.1908508080805856e-06) [Z0 X3 Z4 X5]
+ (-1.1908508080805856e-06) [Z1 Y2 Z3 Y4]
+ (-1.1908508080805856e-06) [Z1 X2 Z3 X4]
+ (-1.1708301370352508e-06) [Z2 Y5 Z6 Y7]
+ (-1.1708301370352508e-06) [Z2 X5 Z6 X7]
+ (-1.1708301370352508e-06) [Z3 Y4 Z5 Y6]
+ (-1.1708301370352508e-06) [Z3 X4 Z5 X6]
+ (-1.0632283423658491e-06) [Z2 Y10 Z11 Y12]
+ (-1.0632283423658491e-06) [Z2 X10 Z11 X12]
+ (-1.0632283423658491e-06) [Z3 Y11 Z12 Y13]
+ (-1.0632283423658491e-06) [Z3 X11 Z12 X13]
+ (-1.0358477600988645e-06) [Y6 X7 X11 Y12]
+ (-1.0358477600988645e-06) [Y6 Y7 Y11 Y12]
+ (-1.0358477600988645e-06) [X6 X7 X11 X12]
+ (-1.0358477600988645e-06) [X6 Y7 Y11 X12]
+ (-9.509249751804851e-07) [Z2 Y4 Z5 Y6]
+ (-9.509249751804851e-07) [Z2 X4 Z5 X6]
+ (-9.509249751804851e-07) [Z3 Y5 Z6 Y7]
+ (-9.509249751804851e-07) [Z3 X5 Z6 X7]
+ (-9.344557776202969e-07) [Z8 Y11 Z12 Y13]
+ (-9.344557776202969e-07) [Z8 X11 Z12 X13]
+ (-9.344557776202969e-07) [Z9 Y10 Z11 Y12]
+ (-9.344557776202969e-07) [Z9 X10 Z11 X12]
+ (-8.337746752130682e-07) [Z0 Y2 Z3 Y4]
+ (-8.337746752130682e-07) [Z0 X2 Z3 X4]
+ (-8.337746752130682e-07) [Z1 Y3 Z4 Y5]
+ (-8.337746752130682e-07) [Z1 X3 Z4 X5]
+ (-7.956895371708537e-07) [Y3 X4 X8 Y9]
+ (-7.956895371708537e-07) [Y3 Y4 Y8 Y9]
+ (-7.956895371708537e-07) [X3 X4 X8 X9]
+ (-7.956895371708537e-07) [X3 Y4 Y8 X9]
+ (-7.764994117642363e-07) [Y2 Z3 Y4 Z5]
+ (-7.764994117642363e-07) [X2 Z3 X4 Z5]
+ (-5.929765815772743e-07) [Z4 Y5 Z6 Y7]
+ (-5.929765815772743e-07) [Z4 X5 Z6 X7]
+ (-5.770052993493652e-07) [Y2 Z3 Y4 Z9]
+ (-5.770052993493652e-07) [X2 Z3 X4 Z9]
+ (-5.770052993493652e-07) [Y3 Z4 Y5 Z8]
+ (-5.770052993493652e-07) [X3 Z4 X5 Z8]
+ (-5.47164774476234e-07) [Y1 Y2 X11 X12]
+ (-5.47164774476234e-07) [X1 X2 Y11 Y12]
+ (-4.838052750641899e-07) [Y5 X6 X8 Y9]
+ (-4.838052750641899e-07) [Y5 Y6 Y8 Y9]
+ (-4.838052750641899e-07) [X5 X6 X8 X9]
+ (-4.838052750641899e-07) [X5 Y6 Y8 X9]
+ (-3.5707613286751724e-07) [Y0 X1 X3 Y4]
+ (-3.5707613286751724e-07) [Y0 Y1 Y3 Y4]
+ (-3.5707613286751724e-07) [X0 X1 X3 X4]
+ (-3.5707613286751724e-07) [X0 Y1 Y3 X4]
+ (-2.447323128657165e-07) [Y0 X1 X5 Y6]
+ (-2.447323128657165e-07) [Y0 Y1 Y5 Y6]
+ (-2.447323128657165e-07) [X0 X1 X5 X6]
+ (-2.447323128657165e-07) [X0 Y1 Y5 X6]
+ (-2.199051618547657e-07) [Y2 X3 X5 Y6]
+ (-2.199051618547657e-07) [Y2 Y3 Y5 Y6]
+ (-2.199051618547657e-07) [X2 X3 X5 X6]
+ (-2.199051618547657e-07) [X2 Y3 Y5 X6]
+ (-1.933241276960496e-07) [Y1 X2 X3 Y4]
+ (-1.933241276960496e-07) [X1 Y2 Y3 X4]
+ (-1.2919694861459897e-07) [Y1 Z2 Z3 Y5]
+ (-1.2919694861459897e-07) [X1 Z2 Z3 X5]
+ (1.737933262281268e-07) [Y0 Z1 Z3 Y4]
+ (1.737933262281268e-07) [X0 Z1 Z3 X4]
+ (1.737933262281268e-07) [Y1 Z2 Z4 Y5]
+ (1.737933262281268e-07) [X1 Z2 Z4 X5]
+ (1.933241276960496e-07) [Y1 Y2 X3 X4]
+ (1.933241276960496e-07) [X1 X2 Y3 Y4]
+ (2.1868423782148843e-07) [Y2 Z3 Y4 Z8]
+ (2.1868423782148843e-07) [X2 Z3 X4 Z8]
+ (2.1868423782148843e-07) [Y3 Z4 Y5 Z9]
+ (2.1868423782148843e-07) [X3 Z4 X5 Z9]
+ (2.5935343915967374e-07) [Y2 Z3 Y4 Z6]
+ (2.5935343915967374e-07) [X2 Z3 X4 Z6]
+ (2.5935343915967374e-07) [Y3 Z4 Y5 Z7]
+ (2.5935343915967374e-07) [X3 Z4 X5 Z7]
+ (3.6060718676802617e-07) [Y0 Z1 Z2 Y4]
+ (3.6060718676802617e-07) [X0 Z1 Z2 X4]
+ (3.6060718676802617e-07) [Y1 Z3 Z4 Y5]
+ (3.6060718676802617e-07) [X1 Z3 Z4 X5]
+ (5.47164774476234e-07) [Y1 X2 X11 Y12]
+ (5.47164774476234e-07) [X1 Y2 Y11 X12]
+ (5.627851911530087e-07) [Y0 X1 X11 Y12]
+ (5.627851911530087e-07) [Y0 Y1 Y11 Y12]
+ (5.627851911530087e-07) [X0 X1 X11 X12]
+ (5.627851911530087e-07) [X0 Y1 Y11 X12]
+ (6.628614201722317e-07) [Y8 X9 X11 Y12]
+ (6.628614201722317e-07) [Y8 Y9 Y11 Y12]
+ (6.628614201722317e-07) [X8 X9 X11 X12]
+ (6.628614201722317e-07) [X8 Y9 Y11 X12]
+ (1.1094407591261965e-06) [Z2 Y11 Z12 Y13]
+ (1.1094407591261965e-06) [Z2 X11 Z12 X13]
+ (1.1094407591261965e-06) [Z3 Y10 Z11 Y12]
+ (1.1094407591261965e-06) [Z3 X10 Z11 X12]
+ (1.6021167406494994e-06) [Z2 Y3 Z4 Y5]
+ (1.6021167406494994e-06) [Z2 X3 Z4 X5]
+ (1.8782101248632972e-06) [Z4 Y10 Z11 Y12]
+ (1.8782101248632972e-06) [Z4 X10 Z11 X12]
+ (1.8782101248632972e-06) [Z5 Y11 Z12 Y13]
+ (1.8782101248632972e-06) [Z5 X11 Z12 X13]
+ (2.1726691014920457e-06) [Y2 X3 X11 Y12]
+ (2.1726691014920457e-06) [Y2 Y3 Y11 Y12]
+ (2.1726691014920457e-06) [X2 X3 X11 X12]
+ (2.1726691014920457e-06) [X2 Y3 Y11 X12]
+ (3.1174479457499144e-06) [Y0 Z2 Z3 Y4]
+ (3.1174479457499144e-06) [X0 Z2 Z3 X4]
+ (3.539054184666288e-06) [Y2 Z3 Y4 Z12]
+ (3.539054184666288e-06) [X2 Z3 X4 Z12]
+ (3.539054184666288e-06) [Y3 Z4 Y5 Z13]
+ (3.539054184666288e-06) [X3 Z4 X5 Z13]
+ (4.281913884847984e-06) [Y4 Z5 Y6 Z11]
+ (4.281913884847984e-06) [X4 Z5 X6 Z11]
+ (4.281913884847984e-06) [Y5 Z6 Y7 Z10]
+ (4.281913884847984e-06) [X5 Z6 X7 Z10]
+ (5.2758831222224395e-06) [Y3 X4 X12 Y13]
+ (5.2758831222224395e-06) [Y3 Y4 Y12 Y13]
+ (5.2758831222224395e-06) [X3 X4 X12 X13]
+ (5.2758831222224395e-06) [X3 Y4 Y12 X13]
+ (5.974311713490356e-06) [Y5 X6 X10 Y11]
+ (5.974311713490356e-06) [Y5 Y6 Y10 Y11]
+ (5.974311713490356e-06) [X5 X6 X10 X11]
+ (5.974311713490356e-06) [X5 Y6 Y10 X11]
+ (7.954413176529942e-06) [Y10 Z11 Y12 Z13]
+ (7.954413176529942e-06) [X10 Z11 X12 Z13]
+ (8.814937306888728e-06) [Y2 Z3 Y4 Z13]
+ (8.814937306888728e-06) [X2 Z3 X4 Z13]
+ (8.814937306888728e-06) [Y3 Z4 Y5 Z12]
+ (8.814937306888728e-06) [X3 Z4 X5 Z12]
+ (0.00029219862611108856) [Y7 X8 X9 Y10]
+ (0.00029219862611108856) [X7 Y8 Y9 X10]
+ (0.0004956762314917394) [Y2 Z4 Z5 Y6]
+ (0.0004956762314917394) [X2 Z4 Z5 X6]
+ (0.001105903769189661) [Y0 Z1 Y2 Z5]
+ (0.001105903769189661) [X0 Z1 X2 Z5]
+ (0.001105903769189661) [Y1 Z2 Y3 Z4]
+ (0.001105903769189661) [X1 Z2 X3 Z4]
+ (0.0016638798784907817) [Y2 Z3 Z4 Y6]
+ (0.0016638798784907817) [X2 Z3 Z4 X6]
+ (0.0016638798784907817) [Y3 Z5 Z6 Y7]
+ (0.0016638798784907817) [X3 Z5 Z6 X7]
+ (0.001756070701841219) [Y0 Z1 Y2 Z11]
+ (0.001756070701841219) [X0 Z1 X2 Z11]
+ (0.001756070701841219) [Y1 Z2 Y3 Z10]
+ (0.001756070701841219) [X1 Z2 X3 Z10]
+ (0.0023262306231580593) [Y0 Z1 Y2 Z13]
+ (0.0023262306231580593) [X0 Z1 X2 Z13]
+ (0.0023262306231580593) [Y1 Z2 Y3 Z12]
+ (0.0023262306231580593) [X1 Z2 X3 Z12]
+ (0.002745836470186818) [Y0 X1 X4 Y5]
+ (0.002745836470186818) [X0 Y1 Y4 X5]
+ (0.002929768674751023) [Y0 Z1 Y2 Z9]
+ (0.002929768674751023) [X0 Z1 X2 Z9]
+ (0.002929768674751023) [Y1 Z2 Y3 Z8]
+ (0.002929768674751023) [X1 Z2 X3 Z8]
+ (0.003276971931231629) [Y0 Z1 Y2 Z3]
+ (0.003276971931231629) [X0 Z1 X2 Z3]
+ (0.0033476175306661696) [Y0 Z1 Y2 Z7]
+ (0.0033476175306661696) [X0 Z1 X2 Z7]
+ (0.0033476175306661696) [Y1 Z2 Y3 Z6]
+ (0.0033476175306661696) [X1 Z2 X3 Z6]
+ (0.003555290195504225) [Y0 Z1 Y2 Z10]
+ (0.003555290195504225) [X0 Z1 X2 Z10]
+ (0.003555290195504225) [Y1 Z2 Y3 Z11]
+ (0.003555290195504225) [X1 Z2 X3 Z11]
+ (0.005143391768825061) [Y3 Y4 X5 X6]
+ (0.005143391768825061) [X3 X4 Y5 Y6]
+ (0.0052837764884029696) [Y0 X1 X12 Y13]
+ (0.0052837764884029696) [X0 Y1 Y12 X13]
+ (0.0055307592186315275) [Y0 Z1 Y2 Z4]
+ (0.0055307592186315275) [X0 Z1 X2 Z4]
+ (0.0055307592186315275) [Y1 Z2 Y3 Z5]
+ (0.0055307592186315275) [X1 Z2 X3 Z5]
+ (0.006087822480561859) [Y8 X9 X12 Y13]
+ (0.006087822480561859) [X8 Y9 Y12 X13]
+ (0.0065093612011772484) [Y0 X1 X8 Y9]
+ (0.0065093612011772484) [X0 Y1 Y8 X9]
+ (0.006888194352970597) [Y0 X1 X6 Y7]
+ (0.006888194352970597) [X0 Y1 Y6 X7]
+ (0.006901238249797268) [Y0 Z1 Y2 Z12]
+ (0.006901238249797268) [X0 Z1 X2 Z12]
+ (0.006901238249797268) [Y1 Z2 Y3 Z13]
+ (0.006901238249797268) [X1 Z2 X3 Z13]
+ (0.007156934919856946) [Y4 X5 X8 Y9]
+ (0.007156934919856946) [X4 Y5 Y8 X9]
+ (0.007731425250775303) [Y0 X1 X10 Y11]
+ (0.007731425250775303) [X0 Y1 Y10 X11]
+ (0.008032520918821397) [Y0 Z1 Y2 Z6]
+ (0.008032520918821397) [X0 Z1 X2 Z6]
+ (0.008032520918821397) [Y1 Z2 Y3 Z7]
+ (0.008032520918821397) [X1 Z2 X3 Z7]
+ (0.009560705729135963) [Y8 X9 X10 Y11]
+ (0.009560705729135963) [X8 Y9 Y10 X11]
+ (0.011055020596132068) [Y0 Z1 Y2 Z8]
+ (0.011055020596132068) [X0 Z1 X2 Z8]
+ (0.011055020596132068) [Y1 Z2 Y3 Z9]
+ (0.011055020596132068) [X1 Z2 X3 Z9]
+ (0.011285190200840855) [Y5 Y6 X11 X12]
+ (0.011285190200840855) [X5 X6 Y11 Y12]
+ (0.011307274008848126) [Y7 Z8 Z9 Y11]
+ (0.011307274008848126) [X7 Z8 Z9 X11]
+ (0.011982389010247911) [Y4 X5 X6 Y7]
+ (0.011982389010247911) [X4 Y5 Y6 X7]
+ (0.013873381748426129) [Y6 X7 X8 Y9]
+ (0.013873381748426129) [X6 Y7 Y8 X9]
+ (0.014583648907612698) [Y0 X1 X2 Y3]
+ (0.014583648907612698) [X0 Y1 Y2 X3]
+ (0.01557720806397647) [Y2 X3 X12 Y13]
+ (0.01557720806397647) [X2 Y3 Y12 X13]
+ (0.01736611899465137) [Y6 X7 X12 Y13]
+ (0.01736611899465137) [X6 Y7 Y12 X13]
+ (0.01768006795248153) [Y4 X5 X10 Y11]
+ (0.01768006795248153) [X4 Y5 Y10 X11]
+ (0.017825140995786356) [Y6 X7 X10 Y11]
+ (0.017825140995786356) [X6 Y7 Y10 X11]
+ (0.01902824244384737) [Y3 X4 X11 Y12]
+ (0.01902824244384737) [X3 Y4 Y11 X12]
+ (0.025384657508457472) [Y2 X3 X10 Y11]
+ (0.025384657508457472) [X2 Y3 Y10 X11]
+ (0.028685183716106018) [Y10 X11 X12 Y13]
+ (0.028685183716106018) [X10 Y11 Y12 X13]
+ (0.02981242451734566) [Y6 Z7 Z8 Y10]
+ (0.02981242451734566) [X6 Z7 Z8 X10]
+ (0.02981242451734566) [Y7 Z9 Z10 Y11]
+ (0.02981242451734566) [X7 Z9 Z10 X11]
+ (0.030104623143456747) [Y6 Z7 Z9 Y10]
+ (0.030104623143456747) [X6 Z7 Z9 X10]
+ (0.030104623143456747) [Y7 Z8 Z10 Y11]
+ (0.030104623143456747) [X7 Z8 Z10 X11]
+ (0.030787505389143887) [Y6 Z8 Z9 Y10]
+ (0.030787505389143887) [X6 Z8 Z9 X10]
+ (0.031143817988967145) [Y2 X3 X6 Y7]
+ (0.031143817988967145) [X2 Y3 Y6 X7]
+ (0.03583956795335353) [Y2 X3 X4 Y5]
+ (0.03583956795335353) [X2 Y3 Y4 X5]
+ (0.03619412355904262) [Y2 X3 X8 Y9]
+ (0.03619412355904262) [X2 Y3 Y8 X9]
+ (0.0383146702948039) [Y4 X5 X12 Y13]
+ (0.0383146702948039) [X4 Y5 Y12 X13]
+ (0.1043306478065143) [Z0 Y1 Z2 Y3]
+ (0.1043306478065143) [Z0 X1 Z2 X3]
+ (-0.12133276911042343) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133276911042343) [X2 Z3 Z4 Z5 X6]
+ (-0.1213327691104234) [Y3 Z4 Z5 Z6 Y7]
+ (-0.1213327691104234) [X3 Z4 Z5 Z6 X7]
+ (3.2020768794351275e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.2020768794351275e-06) [X0 Z1 Z2 Z3 X4]
+ (3.202076879435128e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.202076879435128e-06) [X1 Z2 Z3 Z4 X5]
+ (0.22848106564918677) [Y6 Z7 Z8 Z9 Y10]
+ (0.22848106564918677) [X6 Z7 Z8 Z9 X10]
+ (0.22848106564918685) [Y7 Z8 Z9 Z10 Y11]
+ (0.22848106564918685) [X7 Z8 Z9 Z10 X11]
+ (-0.03276765782329044) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.03276765782329044) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.03276765782329044) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.03276765782329044) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.02711503684527308) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.02711503684527308) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.02711503684527308) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.02711503684527308) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.025996177598021263) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.025996177598021263) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.01756120240964612) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.01756120240964612) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.01756120240964612) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.01756120240964612) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.014564531231173012) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.014564531231173012) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.014564531231173012) [X7 Z8 Z9 X10 X12 X13]
+ (-0.014564531231173012) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.012215040997613882) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.012215040997613882) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.012215040997613882) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.012215040997613882) [X4 Z5 X6 X11 Z12 X13]
+ (-0.012215040997613882) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.012215040997613882) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.012215040997613882) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.012215040997613882) [X5 Z6 X7 X10 Z11 X12]
+ (-0.011756013419819255) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.011756013419819255) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.011756013419819255) [X3 Z4 Z5 X6 X8 X9]
+ (-0.011756013419819255) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.008764827575688819) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.008764827575688819) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.008764827575688819) [X2 Z3 Z4 X5 X11 X12]
+ (-0.008764827575688819) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.008764827575688819) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.008764827575688819) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.008764827575688819) [X3 X4 X10 Z11 Z12 X13]
+ (-0.008764827575688819) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.008125251921381044) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.008125251921381044) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.007306759928832935) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.007306759928832935) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.007306759928832935) [X4 X5 X7 Z8 Z9 X10]
+ (-0.007306759928832935) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.005805188989826868) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.005805188989826868) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.005805188989826868) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.005805188989826868) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.005652620978017359) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.005652620978017359) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.005652620978017359) [X0 X1 X3 Z4 Z5 X6]
+ (-0.005652620978017359) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.005143391768825061) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.005143391768825061) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.005143391768825061) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.005143391768825061) [X2 Z3 X4 X5 Z6 X7]
+ (-0.0046849033881552265) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.0046849033881552265) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668620318776307) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.004668620318776307) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.004575007626639207) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.004575007626639207) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.004424855449441867) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.004424855449441867) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.00415879738184005) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.00415879738184005) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.00415879738184005) [X3 Z4 Z5 X6 X12 X13]
+ (-0.00415879738184005) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.0034937903598901256) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.0034937903598901256) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.0034937903598901256) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.0034937903598901256) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.0027790267990255727) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.0027790267990255727) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.002293956611352471) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.002293956611352471) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.0017992194936630062) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.0017992194936630062) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.0017278753941369594) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.0017278753941369594) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.0009298507967730272) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298507967730272) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298507967730272) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298507967730272) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533856254125546) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.0008533856254125546) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.0008145313270956378) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.0008145313270956378) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.0008145313270956378) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.0008145313270956378) [X3 Z4 Z5 Z6 X7 Z11]
+ (-7.735036880591424e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.735036880591424e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.735036880591424e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.735036880591424e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-8.774817864740077e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.774817864740077e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.774817864740077e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.774817864740077e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.518362215820897e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.518362215820897e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.518362215820897e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.518362215820897e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.444344676077458e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.444344676077458e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.444344676077458e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.444344676077458e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.524373848659816e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.524373848659816e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.524373848659816e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.524373848659816e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-6.2900284333697265e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.2900284333697265e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.2900284333697265e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.2900284333697265e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.974311713490356e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (-5.974311713490356e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (-5.2758831222224395e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.2758831222224395e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.643051068556269e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.643051068556269e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.5565692182674655e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (-4.5565692182674655e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (-4.2532242256252055e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.2532242256252055e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-3.7696594521248776e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.7696594521248776e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.694513294564377e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.694513294564377e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.610297130721139e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.610297130721139e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.610297130721139e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.610297130721139e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.313145500144856e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.313145500144856e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.2774831956673566e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.2774831956673566e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.2774831956673566e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.2774831956673566e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.2112283485149606e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.2112283485149606e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.2112283485149606e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.2112283485149606e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.1513463112464465e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.1513463112464465e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.0882507113955108e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.0882507113955108e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.1726691014920457e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.1726691014920457e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.4548424489964397e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.4548424489964397e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.330473188662619e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.330473188662619e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.2283337824511707e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.2283337824511707e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.0358477600988645e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.0358477600988645e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-7.956895371708537e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.956895371708537e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.733197742663681e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.733197742663681e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.733197742663681e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.733197742663681e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.628614201722317e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.628614201722317e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.556281914719921e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.556281914719921e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.556281914719921e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.556281914719921e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.418291574787692e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.418291574787692e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.418291574787692e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.418291574787692e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.927453083063789e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.927453083063789e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.927453083063789e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.927453083063789e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.627851911530087e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.627851911530087e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.287660624860823e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.287660624860823e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.287660624860823e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.287660624860823e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.287660624860823e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.287660624860823e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.287660624860823e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.287660624860823e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.838052750641899e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (-4.838052750641899e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (-3.5707613286751724e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.5707613286751724e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.3281393505378226e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.3281393505378226e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.0868265650556684e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.0868265650556684e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.0868265650556684e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.0868265650556684e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.447323128657165e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (-2.447323128657165e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (-2.371328947674745e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.371328947674745e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.371328947674745e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.371328947674745e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.199051618547657e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-2.199051618547657e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-1.933241276960496e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.933241276960496e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.933241276960496e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.933241276960496e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.8394209153399684e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.8394209153399684e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.8394209153399684e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.8394209153399684e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.5510539176454237e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (-1.5510539176454237e-07) [X0 Z1 X2 X4 Z5 X6]
+ (-1.5510539176454237e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (-1.5510539176454237e-07) [X1 Z2 X3 X5 Z6 X7]
+ (-1.380778148002668e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (-1.380778148002668e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (-1.380778148002668e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (-1.380778148002668e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (-1.380778148002668e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (-1.380778148002668e-07) [X0 Z1 X2 X5 Z6 X7]
+ (-1.380778148002668e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (-1.380778148002668e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (-1.380778148002668e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (-1.380778148002668e-07) [X1 Z2 X3 X4 Z5 X6]
+ (-1.380778148002668e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (-1.380778148002668e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (-1.2919694861459897e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.2919694861459897e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.1076325599422245e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.1076325599422245e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.1076325599422245e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.1076325599422245e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.1076325599422245e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.1076325599422245e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.1076325599422245e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.1076325599422245e-07) [X1 Z2 X3 X10 Z11 X12]
+ (8.057446595998925e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.057446595998925e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.057446595998925e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.057446595998925e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.649310133644022e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.649310133644022e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.649310133644022e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.649310133644022e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.8394209153399687e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.8394209153399687e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.8394209153399687e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.8394209153399687e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.199051618547657e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (2.199051618547657e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (2.447323128657165e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (2.447323128657165e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.2362599610391477e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.2362599610391477e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.2362599610391477e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.2362599610391477e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.3281393505378226e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.3281393505378226e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.5707613286751724e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.5707613286751724e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.838052750641899e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (4.838052750641899e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (5.627851911530087e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.627851911530087e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (6.628614201722317e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.628614201722317e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (7.956895371708537e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.956895371708537e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.306536651975466e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.306536651975466e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.306536651975466e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.306536651975466e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.0358477600988645e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.0358477600988645e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.2283337824511707e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.2283337824511707e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.2393363217031137e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.2393363217031137e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.2393363217031137e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.2393363217031137e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.330473188662619e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.330473188662619e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.4548424489964397e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.4548424489964397e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.1726691014920457e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.1726691014920457e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.0882507113955108e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.0882507113955108e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.1174479457499144e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.1174479457499144e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.1513463112464465e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.1513463112464465e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.313145500144856e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.313145500144856e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.334331289451436e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.334331289451436e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.694513294564377e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.694513294564377e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (4.183932559461352e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.183932559461352e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.5565692182674655e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (4.5565692182674655e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (4.643051068556269e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.643051068556269e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.2758831222224395e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.2758831222224395e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.974311713490356e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (5.974311713490356e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (0.0002921986261110885) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.0002921986261110885) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.0002921986261110885) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.0002921986261110885) [X6 Z7 X8 X9 Z10 X11]
+ (0.0004956762314917394) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.0004956762314917394) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.0006650070219499238) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.0006650070219499238) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.0006650070219499238) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.0006650070219499238) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.0008533856254125546) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.0008533856254125546) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.0016095313817213758) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.0016095313817213758) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.0016095313817213758) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.0016095313817213758) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.0016676041811440716) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.0016676041811440716) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.0016676041811440716) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.0016676041811440716) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.0017278753941369594) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.0017278753941369594) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.0017992194936630062) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.0017992194936630062) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.002293956611352471) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.002293956611352471) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.0024629170071339304) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.0024629170071339304) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.0024629170071339304) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.0024629170071339304) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.0039615607924965435) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.0039615607924965435) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.0039615607924965435) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.0039615607924965435) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.004424855449441867) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.004424855449441867) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.004575007626639207) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.004575007626639207) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.004668620318776307) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.004668620318776307) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.0046849033881552265) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.0046849033881552265) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.005324835234221685) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.005324835234221685) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.005324835234221685) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.005324835234221685) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.005368659358109498) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.005368659358109498) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.005368659358109498) [X2 X3 X7 Z8 Z9 X10]
+ (0.005368659358109498) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.007960880725921545) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.007960880725921545) [X4 Z5 X6 X10 Z11 X12]
+ (0.007960880725921545) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.007960880725921545) [X5 Z6 X7 X11 Z12 X13]
+ (0.008125251921381044) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.008125251921381044) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.00889073152269457) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.00889073152269457) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.00889073152269457) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.00889073152269457) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.01026341486815855) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.01026341486815855) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.01026341486815855) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.01026341486815855) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.010540425907671479) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.010540425907671479) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.010540425907671479) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.010540425907671479) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.010960074940542547) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.010960074940542547) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.010960074940542547) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.010960074940542547) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.011307274008848126) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.011307274008848126) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.014411099430130981) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.014411099430130981) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.014411099430130981) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.014411099430130981) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.015225630757226619) [Y3 Z4 Z5 X6 X10 Y11]
+ (0.015225630757226619) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (0.015225630757226619) [X3 Z4 Z5 X6 X10 X11]
+ (0.015225630757226619) [X3 Z4 Z5 Y6 Y10 X11]
+ (0.015588250102380236) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (0.015588250102380236) [X2 Z3 X4 X10 Z11 X12]
+ (0.015588250102380236) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (0.015588250102380236) [X3 Z4 X5 X11 Z12 X13]
+ (0.018266834869375484) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.018266834869375484) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.018266834869375484) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.018266834869375484) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.019020423173039938) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.019020423173039938) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.019020423173039938) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.019020423173039938) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.02017592172353543) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.02017592172353543) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.02017592172353543) [X4 Z5 Z6 X7 X11 X12]
+ (0.02017592172353543) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.02017592172353543) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.02017592172353543) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.02017592172353543) [X5 X6 X10 Z11 Z12 X13]
+ (0.02017592172353543) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.024353077678069053) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.024353077678069053) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.024353077678069053) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.024353077678069053) [X2 Z3 X4 X11 Z12 X13]
+ (0.024353077678069053) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.024353077678069053) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.024353077678069053) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.024353077678069053) [X3 Z4 X5 X10 Z11 X12]
+ (0.024389082531149436) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.024389082531149436) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.024389082531149436) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.024389082531149436) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.02510495713884449) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.02510495713884449) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.02510495713884449) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.02510495713884449) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.030787505389143887) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.030787505389143887) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.04587947078129816) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.04587947078129816) [X0 Z2 Z3 Z4 Z5 X6]
+ (0.05600733087780755) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.05600733087780755) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.05600733087780755) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.05600733087780755) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661346) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.05608468124661346) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661346) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.05608468124661346) [Z1 X7 Z8 Z9 Z10 X11]
+ (-6.631277928589928e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.631277928589928e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.631277928589924e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.631277928589924e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.5950860070720705e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.5950860070720705e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.5950860070720705e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.5950860070720705e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.042743277013783304) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.042743277013783304) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (0.042743277013783304) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.042743277013783304) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.04764261217638314) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.04764261217638314) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.04764261217638314) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.04764261217638314) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.04171881383982179) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.04171881383982179) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.04171881383982179) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.04171881383982179) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.0395644163228935) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.0395644163228935) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.0395644163228935) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0395644163228935) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.039359168022053206) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.039359168022053206) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.039359168022053206) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.039359168022053206) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.03931805194719763) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03931805194719763) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.03931805194719763) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03931805194719763) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03560837898831268) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.03560837898831268) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.029903789512624904) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.029903789512624904) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.029903789512624904) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.029903789512624904) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.02873077955190558) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.02873077955190558) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.02873077955190558) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.02873077955190558) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.02563723829602682) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.02563723829602682) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.02563723829602682) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.02563723829602682) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.024755463292891054) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.024755463292891054) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.024755463292891054) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.024755463292891054) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.02428211735469297) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.02428211735469297) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.023145130929529086) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.023145130929529086) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.022528440196013046) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.022528440196013046) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.021433810721601085) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.021433810721601085) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.021433810721601085) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.021433810721601085) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.01925750509525158) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.01925750509525158) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.01902824244384737) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.01902824244384737) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.018889030304942968) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-0.018889030304942968) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-0.018889030304942968) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.018889030304942968) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.016024603689179576) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.016024603689179576) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.015225630757226619) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.015225630757226619) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.014603704729162156) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.014603704729162156) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.014564531231173012) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.014564531231173012) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.011756013419819255) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.011756013419819255) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.011285190200840855) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-0.011285190200840855) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-0.009841749246962609) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.009841749246962609) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.009612634606847244) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.009612634606847244) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.009612634606847244) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.009612634606847244) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.008469978791023814) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.008469978791023814) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.007306759928832935) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.007306759928832935) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005923798336561346) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.005923798336561346) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.0056526209780173595) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.0056526209780173595) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.005368659358109497) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005368659358109497) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.00415879738184005) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.00415879738184005) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.003356670563832896) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.003356670563832896) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.003356670563832896) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.003356670563832896) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.0032675138544235593) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.0032675138544235593) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.0032675138544235593) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.0032675138544235593) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.0027790267990255727) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.0027790267990255727) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.0026860409778066167) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.0026860409778066167) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.0026860409778066167) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.0026860409778066167) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.002293956611352471) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.002293956611352471) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.002293956611352471) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.002293956611352471) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.0009581655836696577) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.0009581655836696577) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.0009581655836696577) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.0009581655836696577) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.0009581655836696577) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.0009581655836696577) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.0009581655836696577) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.0009581655836696577) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.0002463643756958719) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0002463643756958719) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303548336) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.00013840177303548336) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.00013840177303548336) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.00013840177303548336) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.735036880591424e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.735036880591424e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.610358530598194e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.610358530598194e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.610358530598194e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.610358530598194e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808795573467e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.5316808795573467e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808795573467e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5316808795573467e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.806102775383324e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.806102775383324e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.806102775383324e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.806102775383324e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.089799467599107e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.089799467599107e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.089799467599107e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.089799467599107e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.652209669747152e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.652209669747152e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.652209669747152e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.652209669747152e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851834205635e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.481851834205635e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851834205635e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.481851834205635e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.071480736558971e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.071480736558971e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.071480736558971e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.071480736558971e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.734622038824355e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.734622038824355e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.734622038824355e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.734622038824355e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.728843147285952e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.728843147285952e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.728843147285952e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.728843147285952e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.253224225625206e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.253224225625206e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.769659452124877e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.769659452124877e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.5443954293523777e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.5443954293523777e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.5443954293523777e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.5443954293523777e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.5443954293523777e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.5443954293523777e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.5443954293523777e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.5443954293523777e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.360956320313155e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.360956320313155e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.360956320313155e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.360956320313155e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-2.103215604798089e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.103215604798089e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.103215604798089e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.103215604798089e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220984013687e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.0111220984013687e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220984013687e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.0111220984013687e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9429468366816745e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.9429468366816745e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9429468366816745e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9429468366816745e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.654117477194449e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.654117477194449e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.654117477194449e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.654117477194449e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.5224930676914425e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.5224930676914425e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.5224930676914425e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.5224930676914425e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.5224930676914425e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930676914425e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930676914425e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.5224930676914425e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.228333782451171e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.228333782451171e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.228333782451171e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.228333782451171e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.988770288171871e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.988770288171871e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.988770288171871e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.988770288171871e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.867765104084685e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104084685e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104084685e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.867765104084685e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.189990975515358e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.189990975515358e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.175246207104578e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175246207104578e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.47164774476234e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.47164774476234e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.561447179568666e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.561447179568666e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.561447179568666e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.561447179568666e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.523389678081911e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.523389678081911e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.4273231086032047e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.4273231086032047e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.4273231086032047e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.4273231086032047e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.3281393505378226e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.3281393505378226e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.3281393505378226e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.3281393505378226e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.086826565055668e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.086826565055668e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.8882935948722535e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8882935948722535e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8882935948722535e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.8882935948722535e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.371328947674745e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.371328947674745e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.8394209153399687e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.8394209153399687e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-8.057446595998926e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.057446595998926e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (4.5371780971756974e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.5371780971756974e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.5371780971756974e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (4.5371780971756974e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (8.057446595998926e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.057446595998926e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (9.209350639672008e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350639672008e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350639672008e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.209350639672008e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783554151688e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783554151688e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783554151688e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783554151688e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.8394209153399687e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.8394209153399687e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.371328947674745e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.371328947674745e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (3.086826565055668e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.086826565055668e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (4.523389678081911e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.523389678081911e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (5.47164774476234e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.47164774476234e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.175246207104578e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175246207104578e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.189990975515358e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.189990975515358e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.330473188662619e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.330473188662619e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.330473188662619e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.330473188662619e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.6288532435550047e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.6288532435550047e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.6288532435550047e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.6288532435550047e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.689348951515911e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.689348951515911e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.689348951515911e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.689348951515911e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (2.7455184005351906e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.7455184005351906e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.7455184005351906e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.7455184005351906e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.7455184005351906e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.7455184005351906e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.7455184005351906e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.7455184005351906e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.2118420192073536e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420192073536e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.2118420192073536e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420192073536e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.2118420192073536e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420192073536e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.2118420192073536e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420192073536e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.3131455001448566e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.3131455001448566e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.3131455001448566e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.3131455001448566e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (3.3343312894514363e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.3343312894514363e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (4.183932559461352e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.183932559461352e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (7.735036880591424e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.735036880591424e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.0002463643756958719) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.0002463643756958719) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0004458535128840802) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.0004458535128840802) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.0004458535128840802) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.0004458535128840802) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.0005940221543005328) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005328) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005328) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005328) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005328) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005328) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.0005940221543005328) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005328) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.0008533856254125544) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.0008533856254125544) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.0008533856254125544) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.0008533856254125544) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.0010435246534907531) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.0010435246534907531) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.0010435246534907531) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.0010435246534907531) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.001280306097349668) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.001280306097349668) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.001280306097349668) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.001280306097349668) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.0013038004788127012) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (0.0013038004788127012) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (0.0013038004788127012) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (0.0013038004788127012) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (0.0022619660624823585) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (0.0022619660624823585) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (0.0022619660624823585) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (0.0022619660624823585) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (0.0022619660624823585) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (0.0022619660624823585) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (0.0022619660624823585) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (0.0022619660624823585) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (0.003989841456619317) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (0.003989841456619317) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (0.003989841456619317) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (0.003989841456619317) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (0.00415879738184005) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.00415879738184005) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.004311038507914312) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.004311038507914312) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.004311038507914312) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.004311038507914312) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.004636976661182564) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.004636976661182564) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.004636976661182564) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.004636976661182564) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.005114473831660387) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.005114473831660387) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.005114473831660387) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.005114473831660387) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.005114473831660387) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660387) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660387) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005114473831660387) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.0052415353828038835) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (0.0052415353828038835) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (0.0052415353828038835) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (0.0052415353828038835) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (0.005262642473076839) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.005262642473076839) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.005262642473076839) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.005262642473076839) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.005368659358109497) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005368659358109497) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005379937155839367) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.005379937155839367) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.005379937155839367) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.005379937155839367) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.0056526209780173595) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.0056526209780173595) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.005708495985960919) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.005708495985960919) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.005708495985960919) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.005708495985960919) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.005923798336561346) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.005923798336561346) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.007306759928832935) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.007306759928832935) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.008469978791023814) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.008469978791023814) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.009841749246962609) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.009841749246962609) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.011285190200840855) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (0.011285190200840855) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (0.011756013419819255) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.011756013419819255) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.014564531231173012) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.014564531231173012) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.014603704729162156) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.014603704729162156) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.015225630757226619) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.015225630757226619) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.016024603689179576) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.016024603689179576) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.01902824244384737) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.01902824244384737) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.01925750509525158) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.01925750509525158) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.04587947078129816) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04587947078129816) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.3693708936615617) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.3693708936615617) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.3693708936615617) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.3693708936615617) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.28164257767023076) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.28164257767023076) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.28164257767023065) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.28164257767023065) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.09065144207036492) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.09065144207036492) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.09065144207036492) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.09065144207036492) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863636) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.08684737589863636) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863636) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.08684737589863636) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.07635021950635039) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07635021950635039) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07635021950635039) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07635021950635039) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214052) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.06752385099214052) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214052) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.06752385099214052) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03560837898831268) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.03560837898831268) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0349033433736618) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0349033433736618) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0349033433736618) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0349033433736618) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883830034) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.024591860883830034) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883830034) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.024591860883830034) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.02428211735469297) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.02428211735469297) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.023145130929529086) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.023145130929529086) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.022528440196013046) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.022528440196013046) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.01953805031131481) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.01953805031131481) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.01953805031131481) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.01953805031131481) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.017091553155898914) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.017091553155898914) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.017091553155898914) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.017091553155898914) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.016024603689179576) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.016024603689179576) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.016024603689179576) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.016024603689179576) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.010311482489831762) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831762) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831762) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010311482489831762) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.00984174924696261) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.00984174924696261) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.00984174924696261) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.00984174924696261) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.00882636851420988) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.00882636851420988) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.00882636851420988) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.00882636851420988) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008541996625454846) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454846) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454846) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454846) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454846) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454846) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454846) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008541996625454846) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008469978791023814) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023814) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.008469978791023814) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023814) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.004668620318776307) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.004668620318776307) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.003876470899336964) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.003876470899336964) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.00380406617172855) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00380406617172855) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00380406617172855) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.00380406617172855) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.00348415730021789) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.00348415730021789) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003356670563832896) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.003356670563832896) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.0032675138544235593) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.0032675138544235593) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.002141361223101615) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.002141361223101615) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.0017278753941369594) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.0017278753941369594) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.0016407548553124148) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0016407548553124148) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214169378) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0014528843214169378) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214169378) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0014528843214169378) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870896771024526) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007870896771024526) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0005192743499487861) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.0005192743499487861) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.00019400857029756692) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00019400857029756692) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303548336) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.00013840177303548336) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.141625221157913e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.141625221157913e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.141625221157913e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.141625221157913e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-5.0714807365589715e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.0714807365589715e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.1513463112464465e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.1513463112464465e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0882507113955108e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.0882507113955108e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.988511706211644e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.988511706211644e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.874299071422323e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.874299071422323e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.360956320313155e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.360956320313155e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.3002946562450167e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.3002946562450167e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.1468376507771523e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.1468376507771523e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.1468376507771523e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.1468376507771523e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.352332103372282e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.352332103372282e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.352332103372282e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.352332103372282e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.091637199283708e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637199283708e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637199283708e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637199283708e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637199283708e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637199283708e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637199283708e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.091637199283708e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.07430598616548e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.07430598616548e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.07430598616548e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.07430598616548e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.900128986569727e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.900128986569727e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.900128986569727e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900128986569727e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867765104084685e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.867765104084685e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.560692465129666e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692465129666e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692465129666e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692465129666e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692465129666e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692465129666e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692465129666e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.560692465129666e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.997018422235079e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.997018422235079e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.997018422235079e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.997018422235079e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.997018422235079e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.997018422235079e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.997018422235079e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.997018422235079e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.568247521201797e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.568247521201797e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.568247521201797e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.568247521201797e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.376739308487817e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.376739308487817e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.376739308487817e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.376739308487817e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.376739308487817e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.376739308487817e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.376739308487817e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.376739308487817e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.8882935948722535e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.8882935948722535e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.68638154625359e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.68638154625359e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035783554151688e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7035783554151688e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.209350639672008e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.209350639672008e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244463679e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773244463679e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244463679e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773244463679e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244463679e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773244463679e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.379773244463679e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773244463679e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253796036187e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253796036187e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9742253796036187e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9742253796036187e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.047471655578785e-08) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.047471655578785e-08) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.047471655578785e-08) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.047471655578785e-08) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.209350639672008e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.209350639672008e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0717282184815394e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.0717282184815394e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.0717282184815394e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.0717282184815394e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.2004287494065095e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.2004287494065095e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.2004287494065095e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.2004287494065095e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.7035783554151688e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7035783554151688e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.3120943052827975e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.3120943052827975e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.3120943052827975e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.3120943052827975e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.68638154625359e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.68638154625359e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.8882935948722535e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.8882935948722535e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.092250616178313e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.092250616178313e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.092250616178313e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.092250616178313e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.092250616178313e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.092250616178313e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.092250616178313e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.092250616178313e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.44459785402775e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.44459785402775e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.44459785402775e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.44459785402775e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.6849150951555884e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.6849150951555884e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.6849150951555884e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.6849150951555884e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.246974425586118e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.246974425586118e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.246974425586118e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.246974425586118e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.246974425586118e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.246974425586118e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.246974425586118e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.246974425586118e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.867765104084685e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.867765104084685e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.3002946562450167e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.3002946562450167e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.360956320313155e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.360956320313155e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.874299071422323e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.874299071422323e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.8836765760746378e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.8836765760746378e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.947356011805966e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.947356011805966e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.947356011805966e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.947356011805966e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.988511706211644e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.988511706211644e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.0882507113955108e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.0882507113955108e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.1513463112464465e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.1513463112464465e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.8462016713134525e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.8462016713134525e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.8462016713134525e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.8462016713134525e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.0714807365589715e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.0714807365589715e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.10552672202721e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.10552672202721e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.10552672202721e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.10552672202721e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.146496327558469e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.146496327558469e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.146496327558469e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.146496327558469e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.1593505019239884e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.1593505019239884e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.1593505019239884e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.1593505019239884e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.427988656549348e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.427988656549348e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.427988656549348e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.427988656549348e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.935867718017609e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.935867718017609e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.935867718017609e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.935867718017609e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273348117323e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.253273348117323e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.979825793449534e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.979825793449534e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.979825793449534e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.979825793449534e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (4.205548411218788e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.205548411218788e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.205548411218788e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.205548411218788e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00013840177303548336) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.00013840177303548336) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.0001878705338954769) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0001878705338954769) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0001878705338954769) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0001878705338954769) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00019400857029756692) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00019400857029756692) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0002463643756958719) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0002463643756958719) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0002463643756958719) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0002463643756958719) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192743499487861) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.0005192743499487861) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.0007156734248908736) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007156734248908736) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0007156734248908736) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007156734248908736) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024526) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007870896771024526) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.0015324835230730138) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.0015324835230730138) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.0015324835230730138) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.0015324835230730138) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.0016407548553124148) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0016407548553124148) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0017278753941369594) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.0017278753941369594) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.002446497155415894) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.002446497155415894) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.002446497155415894) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.002446497155415894) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.0032675138544235593) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.0032675138544235593) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.003356670563832896) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.003356670563832896) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.00348415730021789) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.00348415730021789) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.003876470899336964) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.003876470899336964) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.004668620318776307) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.004668620318776307) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278093) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.004767272188278093) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.004767272188278093) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278093) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.005286546538226879) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.005286546538226879) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.005286546538226879) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.005286546538226879) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.005408954422409978) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.005408954422409978) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.005408954422409978) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.005408954422409978) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.005923798336561347) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561347) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.005923798336561347) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561347) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.010715508469796737) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010715508469796737) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010715508469796737) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010715508469796737) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.010757563953908927) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010757563953908927) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010757563953908927) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010757563953908927) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.014603704729162156) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.014603704729162156) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.014603704729162156) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.014603704729162156) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.01929956057936377) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01929956057936377) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01929956057936377) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01929956057936377) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01929956057936377) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01929956057936377) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.01929956057936377) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01929956057936377) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.05859198873386208) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.05859198873386208) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.775950527269744e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.775950527269744e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.775950527269745e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.775950527269745e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.07165035181002574) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.07165035181002574) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.07165035181002576) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.07165035181002576) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.01925750509525158) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.01925750509525158) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010311482489831762) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311482489831762) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00882636851420988) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.00882636851420988) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.007597464029770622) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.007597464029770622) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.007597464029770622) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.007597464029770622) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00573356974731189) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00573356974731189) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00573356974731189) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.00573356974731189) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.00573356974731189) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00573356974731189) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00573356974731189) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00573356974731189) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676622) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005348051582676622) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005348051582676622) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676622) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0038040661717285503) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0038040661717285503) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0029841661681219503) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.0029841661681219503) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.0029841661681219503) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.0029841661681219503) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.0024464971554158943) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.0024464971554158943) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.002249412447094) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.002249412447094) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.002249412447094) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002249412447094) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.002141361223101615) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.002141361223101615) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.0018638942824587322) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587322) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587322) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587322) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587322) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587322) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0018638942824587322) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587322) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0016407548553124148) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124148) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0016407548553124148) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124148) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0012223378081538411) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538411) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538411) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538411) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538411) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538411) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538411) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0012223378081538411) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.001028329237856274) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.001028329237856274) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.001028329237856274) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.001028329237856274) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1463061453273246e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1463061453273246e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.874299071422323e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.874299071422323e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.874299071422323e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.874299071422323e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.3002946562450167e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.3002946562450167e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.3002946562450167e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.3002946562450167e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.0444941298133449e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.0444941298133449e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.0444941298133449e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.0444941298133449e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.956079230103769e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.956079230103769e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.956079230103769e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.956079230103769e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105515037267749e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105515037267749e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105515037267749e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105515037267749e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661347213193509e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661347213193509e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661347213193509e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661347213193509e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540341413824405e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540341413824405e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.189990975515358e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.189990975515358e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.876621658263526e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876621658263526e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876621658263526e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876621658263526e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175246207104578e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175246207104578e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.523389678081911e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.523389678081911e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0767325317012985e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.0767325317012985e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0767325317012985e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.0767325317012985e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.013471458977798e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.013471458977798e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.9045998843090417e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.9045998843090417e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.9045998843090417e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.9045998843090417e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.6667317543425385e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.6667317543425385e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.6667317543425385e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6667317543425385e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641928360206e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.8505641928360206e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309314786426e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.6569309314786426e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309314786426e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.6569309314786426e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.8505641928360206e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.8505641928360206e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.68638154625359e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.68638154625359e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.68638154625359e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.68638154625359e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.013471458977798e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.013471458977798e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.523389678081911e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.523389678081911e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.670402390456441e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.670402390456441e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.670402390456441e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.670402390456441e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175246207104578e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175246207104578e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.189990975515358e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.189990975515358e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540341413824405e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540341413824405e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949476487587286e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949476487587286e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.792493957732328e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.792493957732328e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.792493957732328e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.792493957732328e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.883676576074638e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.883676576074638e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.988511706211644e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.988511706211644e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.988511706211644e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.988511706211644e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273348117323e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.253273348117323e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4017109735361499e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4017109735361499e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4017109735361499e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4017109735361499e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603693093827e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.5809603693093827e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603693093827e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.5809603693093827e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0005192743499487862) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487862) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.0005192743499487862) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487862) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.0007870896771024524) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024524) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024524) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024524) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0011726348316441852) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0011726348316441852) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0011726348316441852) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0011726348316441852) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.001236647801924512) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.001236647801924512) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.001236647801924512) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.001236647801924512) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.002200964069500459) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002200964069500459) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002200964069500459) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002200964069500459) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798026) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798026) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798026) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798026) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798026) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798026) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002394972639798026) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798026) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0024464971554158943) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.0024464971554158943) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.0038040661717285503) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0038040661717285503) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0038764708993369646) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.0038764708993369646) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.0038764708993369646) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.0038764708993369646) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.004220813970046462) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.004220813970046462) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.004220813970046462) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.004220813970046462) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.00882636851420988) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.00882636851420988) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.010311482489831762) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010311482489831762) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01925750509525158) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01925750509525158) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.05859198873386208) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.05859198873386208) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.3987009016256942e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.3987009016256942e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.3987009016256942e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.3987009016256942e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00348415730021789) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00348415730021789) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0029841661681219503) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.0029841661681219503) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.00019400857029756692) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00019400857029756692) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453273246e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061453273246e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.792493957732328e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.792493957732328e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540341413824405e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413824405e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540341413824405e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413824405e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.8505641928360206e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641928360206e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641928360206e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641928360206e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714589777986e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.0134714589777986e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714589777986e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.0134714589777986e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.949476487587286e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.949476487587286e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.792493957732328e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.792493957732328e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029756692) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029756692) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0029841661681219503) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.0029841661681219503) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
 </code>
 </pre>
 </details>

---

## 10. tutorial_local_cost_functions.html <a name="demo9"></a>

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

## 11. tutorial_tn_circuits.html <a name="demo10"></a>

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

## 12. tutorial_quantum_transfer_learning.html <a name="demo11"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_quantum_transfer_learning.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
 11%|#1        | 5.06M/44.7M [00:00<00:00, 52.9MB/s]
 23%|##2       | 10.1M/44.7M [00:00<00:00, 52.1MB/s]
 81%|########1 | 36.2M/44.7M [00:00<00:00, 152MB/s]
100%|##########| 44.7M/44.7M [00:00<00:00, 140MB/s]
Training started:
Phase: train Epoch: 1/3 Iter: 1/62 Batch time: 0.3744
Phase: train Epoch: 1/3 Iter: 2/62 Batch time: 0.3443
Phase: train Epoch: 1/3 Iter: 3/62 Batch time: 0.3462
Phase: train Epoch: 1/3 Iter: 4/62 Batch time: 0.3498
Phase: train Epoch: 1/3 Iter: 5/62 Batch time: 0.3479
Phase: train Epoch: 1/3 Iter: 6/62 Batch time: 0.3446
Phase: train Epoch: 1/3 Iter: 7/62 Batch time: 0.3462
Phase: train Epoch: 1/3 Iter: 8/62 Batch time: 0.3469
Phase: train Epoch: 1/3 Iter: 9/62 Batch time: 0.3482
Phase: train Epoch: 1/3 Iter: 10/62 Batch time: 0.3485
Phase: train Epoch: 1/3 Iter: 11/62 Batch time: 0.3475
Phase: train Epoch: 1/3 Iter: 12/62 Batch time: 0.3489
Phase: train Epoch: 1/3 Iter: 13/62 Batch time: 0.3444
Phase: train Epoch: 1/3 Iter: 14/62 Batch time: 0.3469
Phase: train Epoch: 1/3 Iter: 15/62 Batch time: 0.3475
Phase: train Epoch: 1/3 Iter: 16/62 Batch time: 0.3460
Phase: train Epoch: 1/3 Iter: 17/62 Batch time: 0.3462
Phase: train Epoch: 1/3 Iter: 18/62 Batch time: 0.3468
Phase: train Epoch: 1/3 Iter: 19/62 Batch time: 0.3497
Phase: train Epoch: 1/3 Iter: 20/62 Batch time: 0.3470
Phase: train Epoch: 1/3 Iter: 21/62 Batch time: 0.3457
Phase: train Epoch: 1/3 Iter: 22/62 Batch time: 0.3476
Phase: train Epoch: 1/3 Iter: 23/62 Batch time: 0.3487
Phase: train Epoch: 1/3 Iter: 24/62 Batch time: 0.3981
Phase: train Epoch: 1/3 Iter: 25/62 Batch time: 0.3445
Phase: train Epoch: 1/3 Iter: 26/62 Batch time: 0.3450
Phase: train Epoch: 1/3 Iter: 27/62 Batch time: 0.3475
Phase: train Epoch: 1/3 Iter: 28/62 Batch time: 0.3451
Phase: train Epoch: 1/3 Iter: 29/62 Batch time: 0.3466
Phase: train Epoch: 1/3 Iter: 30/62 Batch time: 0.3458
Phase: train Epoch: 1/3 Iter: 31/62 Batch time: 0.3433
Phase: train Epoch: 1/3 Iter: 32/62 Batch time: 0.3470
Phase: train Epoch: 1/3 Iter: 33/62 Batch time: 0.3454
Phase: train Epoch: 1/3 Iter: 34/62 Batch time: 0.3449
Phase: train Epoch: 1/3 Iter: 35/62 Batch time: 0.3444
Phase: train Epoch: 1/3 Iter: 36/62 Batch time: 0.3468
Phase: train Epoch: 1/3 Iter: 37/62 Batch time: 0.3451
Phase: train Epoch: 1/3 Iter: 38/62 Batch time: 0.3451
Phase: train Epoch: 1/3 Iter: 39/62 Batch time: 0.3456
Phase: train Epoch: 1/3 Iter: 40/62 Batch time: 0.3440
Phase: train Epoch: 1/3 Iter: 41/62 Batch time: 0.3456
Phase: train Epoch: 1/3 Iter: 42/62 Batch time: 0.3473
Phase: train Epoch: 1/3 Iter: 43/62 Batch time: 0.3440
Phase: train Epoch: 1/3 Iter: 44/62 Batch time: 0.3452
Phase: train Epoch: 1/3 Iter: 45/62 Batch time: 0.3475
Phase: train Epoch: 1/3 Iter: 46/62 Batch time: 0.3438
Phase: train Epoch: 1/3 Iter: 47/62 Batch time: 0.3451
Phase: train Epoch: 1/3 Iter: 48/62 Batch time: 0.3467
Phase: train Epoch: 1/3 Iter: 49/62 Batch time: 0.3425
Phase: train Epoch: 1/3 Iter: 50/62 Batch time: 0.3441
Phase: train Epoch: 1/3 Iter: 51/62 Batch time: 0.3459
Phase: train Epoch: 1/3 Iter: 52/62 Batch time: 0.3479
Phase: train Epoch: 1/3 Iter: 53/62 Batch time: 0.3496
Phase: train Epoch: 1/3 Iter: 54/62 Batch time: 0.3506
Phase: train Epoch: 1/3 Iter: 55/62 Batch time: 0.3487
Phase: train Epoch: 1/3 Iter: 56/62 Batch time: 0.3468
Phase: train Epoch: 1/3 Iter: 57/62 Batch time: 0.3502
Phase: train Epoch: 1/3 Iter: 58/62 Batch time: 0.3467
Phase: train Epoch: 1/3 Iter: 59/62 Batch time: 0.3456
Phase: train Epoch: 1/3 Iter: 60/62 Batch time: 0.3463
Phase: train Epoch: 1/3 Iter: 61/62 Batch time: 0.3496
Phase: train Epoch: 1/3 Loss: 0.6990 Acc: 0.5246
Phase: validation Epoch: 1/3 Iter: 1/39 Batch time: 0.2754
Phase: validation Epoch: 1/3 Iter: 2/39 Batch time: 0.2723
Phase: validation Epoch: 1/3 Iter: 3/39 Batch time: 0.2717
Phase: validation Epoch: 1/3 Iter: 4/39 Batch time: 0.2740
Phase: validation Epoch: 1/3 Iter: 5/39 Batch time: 0.2713
Phase: validation Epoch: 1/3 Iter: 6/39 Batch time: 0.2741
Phase: validation Epoch: 1/3 Iter: 7/39 Batch time: 0.2718
Phase: validation Epoch: 1/3 Iter: 8/39 Batch time: 0.2716
Phase: validation Epoch: 1/3 Iter: 9/39 Batch time: 0.2711
Phase: validation Epoch: 1/3 Iter: 10/39 Batch time: 0.2708
Phase: validation Epoch: 1/3 Iter: 11/39 Batch time: 0.2713
Phase: validation Epoch: 1/3 Iter: 12/39 Batch time: 0.2716
Phase: validation Epoch: 1/3 Iter: 13/39 Batch time: 0.2702
Phase: validation Epoch: 1/3 Iter: 14/39 Batch time: 0.2702
Phase: validation Epoch: 1/3 Iter: 15/39 Batch time: 0.2719
Phase: validation Epoch: 1/3 Iter: 16/39 Batch time: 0.2701
Phase: validation Epoch: 1/3 Iter: 17/39 Batch time: 0.2708
Phase: validation Epoch: 1/3 Iter: 18/39 Batch time: 0.2714
Phase: validation Epoch: 1/3 Iter: 19/39 Batch time: 0.2699
Phase: validation Epoch: 1/3 Iter: 20/39 Batch time: 0.2703
Phase: validation Epoch: 1/3 Iter: 21/39 Batch time: 0.2706
Phase: validation Epoch: 1/3 Iter: 22/39 Batch time: 0.2719
Phase: validation Epoch: 1/3 Iter: 23/39 Batch time: 0.2707
Phase: validation Epoch: 1/3 Iter: 24/39 Batch time: 0.2712
Phase: validation Epoch: 1/3 Iter: 25/39 Batch time: 0.2698
Phase: validation Epoch: 1/3 Iter: 26/39 Batch time: 0.2708
Phase: validation Epoch: 1/3 Iter: 27/39 Batch time: 0.2708
Phase: validation Epoch: 1/3 Iter: 28/39 Batch time: 0.2695
Phase: validation Epoch: 1/3 Iter: 29/39 Batch time: 0.2706
Phase: validation Epoch: 1/3 Iter: 30/39 Batch time: 0.2727
Phase: validation Epoch: 1/3 Iter: 31/39 Batch time: 0.2712
Phase: validation Epoch: 1/3 Iter: 32/39 Batch time: 0.2713
Phase: validation Epoch: 1/3 Iter: 33/39 Batch time: 0.2709
Phase: validation Epoch: 1/3 Iter: 34/39 Batch time: 0.2703
Phase: validation Epoch: 1/3 Iter: 35/39 Batch time: 0.2710
Phase: validation Epoch: 1/3 Iter: 36/39 Batch time: 0.2699
Phase: validation Epoch: 1/3 Iter: 37/39 Batch time: 0.2715
Phase: validation Epoch: 1/3 Iter: 38/39 Batch time: 0.2723
Phase: validation Epoch: 1/3 Iter: 39/39 Batch time: 0.0845
Phase: validation   Epoch: 1/3 Loss: 0.6429 Acc: 0.6536
Phase: train Epoch: 2/3 Iter: 1/62 Batch time: 0.3414
Phase: train Epoch: 2/3 Iter: 2/62 Batch time: 0.3450
Phase: train Epoch: 2/3 Iter: 3/62 Batch time: 0.3456
Phase: train Epoch: 2/3 Iter: 4/62 Batch time: 0.3457
Phase: train Epoch: 2/3 Iter: 5/62 Batch time: 0.3459
Phase: train Epoch: 2/3 Iter: 6/62 Batch time: 0.3447
Phase: train Epoch: 2/3 Iter: 7/62 Batch time: 0.3466
Phase: train Epoch: 2/3 Iter: 8/62 Batch time: 0.3445
Phase: train Epoch: 2/3 Iter: 9/62 Batch time: 0.3452
Phase: train Epoch: 2/3 Iter: 10/62 Batch time: 0.3446
Phase: train Epoch: 2/3 Iter: 11/62 Batch time: 0.3446
Phase: train Epoch: 2/3 Iter: 12/62 Batch time: 0.3467
Phase: train Epoch: 2/3 Iter: 13/62 Batch time: 0.3452
Phase: train Epoch: 2/3 Iter: 14/62 Batch time: 0.3459
Phase: train Epoch: 2/3 Iter: 15/62 Batch time: 0.3467
Phase: train Epoch: 2/3 Iter: 16/62 Batch time: 0.3455
Phase: train Epoch: 2/3 Iter: 17/62 Batch time: 0.3441
Phase: train Epoch: 2/3 Iter: 18/62 Batch time: 0.3462
Phase: train Epoch: 2/3 Iter: 19/62 Batch time: 0.3455
Phase: train Epoch: 2/3 Iter: 20/62 Batch time: 0.3451
Phase: train Epoch: 2/3 Iter: 21/62 Batch time: 0.3466
Phase: train Epoch: 2/3 Iter: 22/62 Batch time: 0.3453
Phase: train Epoch: 2/3 Iter: 23/62 Batch time: 0.3458
Phase: train Epoch: 2/3 Iter: 24/62 Batch time: 0.3463
Phase: train Epoch: 2/3 Iter: 25/62 Batch time: 0.3452
Phase: train Epoch: 2/3 Iter: 26/62 Batch time: 0.3461
Phase: train Epoch: 2/3 Iter: 27/62 Batch time: 0.3456
Phase: train Epoch: 2/3 Iter: 28/62 Batch time: 0.3475
Phase: train Epoch: 2/3 Iter: 29/62 Batch time: 0.3454
Phase: train Epoch: 2/3 Iter: 30/62 Batch time: 0.3462
Phase: train Epoch: 2/3 Iter: 31/62 Batch time: 0.3471
Phase: train Epoch: 2/3 Iter: 32/62 Batch time: 0.3458
Phase: train Epoch: 2/3 Iter: 33/62 Batch time: 0.3474
Phase: train Epoch: 2/3 Iter: 34/62 Batch time: 0.3462
Phase: train Epoch: 2/3 Iter: 35/62 Batch time: 0.3459
Phase: train Epoch: 2/3 Iter: 36/62 Batch time: 0.3466
Phase: train Epoch: 2/3 Iter: 37/62 Batch time: 0.3452
Phase: train Epoch: 2/3 Iter: 38/62 Batch time: 0.3474
Phase: train Epoch: 2/3 Iter: 39/62 Batch time: 0.3483
Phase: train Epoch: 2/3 Iter: 40/62 Batch time: 0.3473
Phase: train Epoch: 2/3 Iter: 41/62 Batch time: 0.3459
Phase: train Epoch: 2/3 Iter: 42/62 Batch time: 0.3466
Phase: train Epoch: 2/3 Iter: 43/62 Batch time: 0.3462
Phase: train Epoch: 2/3 Iter: 44/62 Batch time: 0.3467
Phase: train Epoch: 2/3 Iter: 45/62 Batch time: 0.3482
Phase: train Epoch: 2/3 Iter: 46/62 Batch time: 0.3464
Phase: train Epoch: 2/3 Iter: 47/62 Batch time: 0.3443
Phase: train Epoch: 2/3 Iter: 48/62 Batch time: 0.3464
Phase: train Epoch: 2/3 Iter: 49/62 Batch time: 0.3440
Phase: train Epoch: 2/3 Iter: 50/62 Batch time: 0.3455
Phase: train Epoch: 2/3 Iter: 51/62 Batch time: 0.3477
Phase: train Epoch: 2/3 Iter: 52/62 Batch time: 0.3460
Phase: train Epoch: 2/3 Iter: 53/62 Batch time: 0.3463
Phase: train Epoch: 2/3 Iter: 54/62 Batch time: 0.3450
Phase: train Epoch: 2/3 Iter: 55/62 Batch time: 0.3468
Phase: train Epoch: 2/3 Iter: 56/62 Batch time: 0.3455
Phase: train Epoch: 2/3 Iter: 57/62 Batch time: 0.3462
Phase: train Epoch: 2/3 Iter: 58/62 Batch time: 0.3440
Phase: train Epoch: 2/3 Iter: 59/62 Batch time: 0.3452
Phase: train Epoch: 2/3 Iter: 60/62 Batch time: 0.3463
Phase: train Epoch: 2/3 Iter: 61/62 Batch time: 0.3461
Phase: train Epoch: 2/3 Loss: 0.6134 Acc: 0.7008
Phase: validation Epoch: 2/3 Iter: 1/39 Batch time: 0.2745
Phase: validation Epoch: 2/3 Iter: 2/39 Batch time: 0.2716
Phase: validation Epoch: 2/3 Iter: 3/39 Batch time: 0.2714
Phase: validation Epoch: 2/3 Iter: 4/39 Batch time: 0.2714
Phase: validation Epoch: 2/3 Iter: 5/39 Batch time: 0.2724
Phase: validation Epoch: 2/3 Iter: 6/39 Batch time: 0.2717
Phase: validation Epoch: 2/3 Iter: 7/39 Batch time: 0.2719
Phase: validation Epoch: 2/3 Iter: 8/39 Batch time: 0.2722
Phase: validation Epoch: 2/3 Iter: 9/39 Batch time: 0.2718
Phase: validation Epoch: 2/3 Iter: 10/39 Batch time: 0.2720
Phase: validation Epoch: 2/3 Iter: 11/39 Batch time: 0.2737
Phase: validation Epoch: 2/3 Iter: 12/39 Batch time: 0.2730
Phase: validation Epoch: 2/3 Iter: 13/39 Batch time: 0.2729
Phase: validation Epoch: 2/3 Iter: 14/39 Batch time: 0.2732
Phase: validation Epoch: 2/3 Iter: 15/39 Batch time: 0.2715
Phase: validation Epoch: 2/3 Iter: 16/39 Batch time: 0.2733
Phase: validation Epoch: 2/3 Iter: 17/39 Batch time: 0.2744
Phase: validation Epoch: 2/3 Iter: 18/39 Batch time: 0.2716
Phase: validation Epoch: 2/3 Iter: 19/39 Batch time: 0.2714
Phase: validation Epoch: 2/3 Iter: 20/39 Batch time: 0.2715
Phase: validation Epoch: 2/3 Iter: 21/39 Batch time: 0.2703
Phase: validation Epoch: 2/3 Iter: 22/39 Batch time: 0.2712
Phase: validation Epoch: 2/3 Iter: 23/39 Batch time: 0.2726
Phase: validation Epoch: 2/3 Iter: 24/39 Batch time: 0.2716
Phase: validation Epoch: 2/3 Iter: 25/39 Batch time: 0.2715
Phase: validation Epoch: 2/3 Iter: 26/39 Batch time: 0.2708
Phase: validation Epoch: 2/3 Iter: 27/39 Batch time: 0.2716
Phase: validation Epoch: 2/3 Iter: 28/39 Batch time: 0.2717
Phase: validation Epoch: 2/3 Iter: 29/39 Batch time: 0.2732
Phase: validation Epoch: 2/3 Iter: 30/39 Batch time: 0.2717
Phase: validation Epoch: 2/3 Iter: 31/39 Batch time: 0.2736
Phase: validation Epoch: 2/3 Iter: 32/39 Batch time: 0.2728
Phase: validation Epoch: 2/3 Iter: 33/39 Batch time: 0.2720
Phase: validation Epoch: 2/3 Iter: 34/39 Batch time: 0.2717
Phase: validation Epoch: 2/3 Iter: 35/39 Batch time: 0.2717
Phase: validation Epoch: 2/3 Iter: 36/39 Batch time: 0.2712
Phase: validation Epoch: 2/3 Iter: 37/39 Batch time: 0.2720
Phase: validation Epoch: 2/3 Iter: 38/39 Batch time: 0.2726
Phase: validation Epoch: 2/3 Iter: 39/39 Batch time: 0.0794
Phase: validation   Epoch: 2/3 Loss: 0.5389 Acc: 0.8235
Phase: train Epoch: 3/3 Iter: 1/62 Batch time: 0.3401
Phase: train Epoch: 3/3 Iter: 2/62 Batch time: 0.3437
Phase: train Epoch: 3/3 Iter: 3/62 Batch time: 0.3431
Phase: train Epoch: 3/3 Iter: 4/62 Batch time: 0.3450
Phase: train Epoch: 3/3 Iter: 5/62 Batch time: 0.3449
Phase: train Epoch: 3/3 Iter: 6/62 Batch time: 0.3448
Phase: train Epoch: 3/3 Iter: 7/62 Batch time: 0.3445
Phase: train Epoch: 3/3 Iter: 8/62 Batch time: 0.3439
Phase: train Epoch: 3/3 Iter: 9/62 Batch time: 0.3452
Phase: train Epoch: 3/3 Iter: 10/62 Batch time: 0.3449
Phase: train Epoch: 3/3 Iter: 11/62 Batch time: 0.3460
Phase: train Epoch: 3/3 Iter: 12/62 Batch time: 0.3449
Phase: train Epoch: 3/3 Iter: 13/62 Batch time: 0.3455
Phase: train Epoch: 3/3 Iter: 14/62 Batch time: 0.3444
Phase: train Epoch: 3/3 Iter: 15/62 Batch time: 0.3450
Phase: train Epoch: 3/3 Iter: 16/62 Batch time: 0.3462
Phase: train Epoch: 3/3 Iter: 17/62 Batch time: 0.3450
Phase: train Epoch: 3/3 Iter: 18/62 Batch time: 0.3447
Phase: train Epoch: 3/3 Iter: 19/62 Batch time: 0.3450
Phase: train Epoch: 3/3 Iter: 20/62 Batch time: 0.3453
Phase: train Epoch: 3/3 Iter: 21/62 Batch time: 0.3453
Phase: train Epoch: 3/3 Iter: 22/62 Batch time: 0.3445
Phase: train Epoch: 3/3 Iter: 23/62 Batch time: 0.3459
Phase: train Epoch: 3/3 Iter: 24/62 Batch time: 0.3456
Phase: train Epoch: 3/3 Iter: 25/62 Batch time: 0.3454
Phase: train Epoch: 3/3 Iter: 26/62 Batch time: 0.3452
Phase: train Epoch: 3/3 Iter: 27/62 Batch time: 0.3455
Phase: train Epoch: 3/3 Iter: 28/62 Batch time: 0.3444
Phase: train Epoch: 3/3 Iter: 29/62 Batch time: 0.3455
Phase: train Epoch: 3/3 Iter: 30/62 Batch time: 0.3445
Phase: train Epoch: 3/3 Iter: 31/62 Batch time: 0.3447
Phase: train Epoch: 3/3 Iter: 32/62 Batch time: 0.3447
Phase: train Epoch: 3/3 Iter: 33/62 Batch time: 0.3453
Phase: train Epoch: 3/3 Iter: 34/62 Batch time: 0.3464
Phase: train Epoch: 3/3 Iter: 35/62 Batch time: 0.3453
Phase: train Epoch: 3/3 Iter: 36/62 Batch time: 0.3451
Phase: train Epoch: 3/3 Iter: 37/62 Batch time: 0.3464
Phase: train Epoch: 3/3 Iter: 38/62 Batch time: 0.3449
Phase: train Epoch: 3/3 Iter: 39/62 Batch time: 0.3451
Phase: train Epoch: 3/3 Iter: 40/62 Batch time: 0.3454
Phase: train Epoch: 3/3 Iter: 41/62 Batch time: 0.3458
Phase: train Epoch: 3/3 Iter: 42/62 Batch time: 0.3472
Phase: train Epoch: 3/3 Iter: 43/62 Batch time: 0.3457
Phase: train Epoch: 3/3 Iter: 44/62 Batch time: 0.3447
Phase: train Epoch: 3/3 Iter: 45/62 Batch time: 0.3454
Phase: train Epoch: 3/3 Iter: 46/62 Batch time: 0.3451
Phase: train Epoch: 3/3 Iter: 47/62 Batch time: 0.3456
Phase: train Epoch: 3/3 Iter: 48/62 Batch time: 0.3452
Phase: train Epoch: 3/3 Iter: 49/62 Batch time: 0.3459
Phase: train Epoch: 3/3 Iter: 50/62 Batch time: 0.3457
Phase: train Epoch: 3/3 Iter: 51/62 Batch time: 0.3455
Phase: train Epoch: 3/3 Iter: 52/62 Batch time: 0.3443
Phase: train Epoch: 3/3 Iter: 53/62 Batch time: 0.3454
Phase: train Epoch: 3/3 Iter: 54/62 Batch time: 0.3456
Phase: train Epoch: 3/3 Iter: 55/62 Batch time: 0.3444
Phase: train Epoch: 3/3 Iter: 56/62 Batch time: 0.3451
Phase: train Epoch: 3/3 Iter: 57/62 Batch time: 0.3461
Phase: train Epoch: 3/3 Iter: 58/62 Batch time: 0.3451
Phase: train Epoch: 3/3 Iter: 59/62 Batch time: 0.3460
Phase: train Epoch: 3/3 Iter: 60/62 Batch time: 0.3461
Phase: train Epoch: 3/3 Iter: 61/62 Batch time: 0.3459
Phase: train Epoch: 3/3 Loss: 0.5652 Acc: 0.7418
Phase: validation Epoch: 3/3 Iter: 1/39 Batch time: 0.2754
Phase: validation Epoch: 3/3 Iter: 2/39 Batch time: 0.2717
Phase: validation Epoch: 3/3 Iter: 3/39 Batch time: 0.2716
Phase: validation Epoch: 3/3 Iter: 4/39 Batch time: 0.2723
Phase: validation Epoch: 3/3 Iter: 5/39 Batch time: 0.2717
Phase: validation Epoch: 3/3 Iter: 6/39 Batch time: 0.2724
Phase: validation Epoch: 3/3 Iter: 7/39 Batch time: 0.2713
Phase: validation Epoch: 3/3 Iter: 8/39 Batch time: 0.2705
Phase: validation Epoch: 3/3 Iter: 9/39 Batch time: 0.2708
Phase: validation Epoch: 3/3 Iter: 10/39 Batch time: 0.2715
Phase: validation Epoch: 3/3 Iter: 11/39 Batch time: 0.2704
Phase: validation Epoch: 3/3 Iter: 12/39 Batch time: 0.2725
Phase: validation Epoch: 3/3 Iter: 13/39 Batch time: 0.2699
Phase: validation Epoch: 3/3 Iter: 14/39 Batch time: 0.2713
Phase: validation Epoch: 3/3 Iter: 15/39 Batch time: 0.2705
Phase: validation Epoch: 3/3 Iter: 16/39 Batch time: 0.2701
Phase: validation Epoch: 3/3 Iter: 17/39 Batch time: 0.2725
Phase: validation Epoch: 3/3 Iter: 18/39 Batch time: 0.2711
Phase: validation Epoch: 3/3 Iter: 19/39 Batch time: 0.2709
Phase: validation Epoch: 3/3 Iter: 20/39 Batch time: 0.2726
Phase: validation Epoch: 3/3 Iter: 21/39 Batch time: 0.2724
Phase: validation Epoch: 3/3 Iter: 22/39 Batch time: 0.2718
Phase: validation Epoch: 3/3 Iter: 23/39 Batch time: 0.2706
Phase: validation Epoch: 3/3 Iter: 24/39 Batch time: 0.2723
Phase: validation Epoch: 3/3 Iter: 25/39 Batch time: 0.2711
Phase: validation Epoch: 3/3 Iter: 26/39 Batch time: 0.2709
Phase: validation Epoch: 3/3 Iter: 27/39 Batch time: 0.2711
Phase: validation Epoch: 3/3 Iter: 28/39 Batch time: 0.2723
Phase: validation Epoch: 3/3 Iter: 29/39 Batch time: 0.2716
Phase: validation Epoch: 3/3 Iter: 30/39 Batch time: 0.2704
Phase: validation Epoch: 3/3 Iter: 31/39 Batch time: 0.2706
Phase: validation Epoch: 3/3 Iter: 32/39 Batch time: 0.2719
Phase: validation Epoch: 3/3 Iter: 33/39 Batch time: 0.2710
Phase: validation Epoch: 3/3 Iter: 34/39 Batch time: 0.2714
Phase: validation Epoch: 3/3 Iter: 35/39 Batch time: 0.2722
Phase: validation Epoch: 3/3 Iter: 36/39 Batch time: 0.2710
Phase: validation Epoch: 3/3 Iter: 37/39 Batch time: 0.2728
Phase: validation Epoch: 3/3 Iter: 38/39 Batch time: 0.2716
Phase: validation Epoch: 3/3 Iter: 39/39 Batch time: 0.0802
Phase: validation   Epoch: 3/3 Loss: 0.4484 Acc: 0.8497
Training completed in 1m 41s
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
 36%|###6      | 16.1M/44.7M [00:00<00:00, 169MB/s]
 90%|########9 | 40.1M/44.7M [00:00<00:00, 217MB/s]
100%|##########| 44.7M/44.7M [00:00<00:00, 216MB/s]
Training started:
Phase: train Epoch: 1/3 Iter: 1/62 Batch time: 0.3289
Phase: train Epoch: 1/3 Iter: 2/62 Batch time: 0.3023
Phase: train Epoch: 1/3 Iter: 3/62 Batch time: 0.2965
Phase: train Epoch: 1/3 Iter: 4/62 Batch time: 0.2950
Phase: train Epoch: 1/3 Iter: 5/62 Batch time: 0.2970
Phase: train Epoch: 1/3 Iter: 6/62 Batch time: 0.2945
Phase: train Epoch: 1/3 Iter: 7/62 Batch time: 0.2934
Phase: train Epoch: 1/3 Iter: 8/62 Batch time: 0.2968
Phase: train Epoch: 1/3 Iter: 9/62 Batch time: 0.2954
Phase: train Epoch: 1/3 Iter: 10/62 Batch time: 0.2938
Phase: train Epoch: 1/3 Iter: 11/62 Batch time: 0.2904
Phase: train Epoch: 1/3 Iter: 12/62 Batch time: 0.2912
Phase: train Epoch: 1/3 Iter: 13/62 Batch time: 0.2917
Phase: train Epoch: 1/3 Iter: 14/62 Batch time: 0.2896
Phase: train Epoch: 1/3 Iter: 15/62 Batch time: 0.2959
Phase: train Epoch: 1/3 Iter: 16/62 Batch time: 0.2913
Phase: train Epoch: 1/3 Iter: 17/62 Batch time: 0.2926
Phase: train Epoch: 1/3 Iter: 18/62 Batch time: 0.2963
Phase: train Epoch: 1/3 Iter: 19/62 Batch time: 0.2967
Phase: train Epoch: 1/3 Iter: 20/62 Batch time: 0.2939
Phase: train Epoch: 1/3 Iter: 21/62 Batch time: 0.2974
Phase: train Epoch: 1/3 Iter: 22/62 Batch time: 0.2960
Phase: train Epoch: 1/3 Iter: 23/62 Batch time: 0.2981
Phase: train Epoch: 1/3 Iter: 24/62 Batch time: 0.2962
Phase: train Epoch: 1/3 Iter: 25/62 Batch time: 0.2929
Phase: train Epoch: 1/3 Iter: 26/62 Batch time: 0.2942
Phase: train Epoch: 1/3 Iter: 27/62 Batch time: 0.2970
Phase: train Epoch: 1/3 Iter: 28/62 Batch time: 0.2953
Phase: train Epoch: 1/3 Iter: 29/62 Batch time: 0.2934
Phase: train Epoch: 1/3 Iter: 30/62 Batch time: 0.2907
Phase: train Epoch: 1/3 Iter: 31/62 Batch time: 0.2900
Phase: train Epoch: 1/3 Iter: 32/62 Batch time: 0.2941
Phase: train Epoch: 1/3 Iter: 33/62 Batch time: 0.2944
Phase: train Epoch: 1/3 Iter: 34/62 Batch time: 0.2942
Phase: train Epoch: 1/3 Iter: 35/62 Batch time: 0.2917
Phase: train Epoch: 1/3 Iter: 36/62 Batch time: 0.2915
Phase: train Epoch: 1/3 Iter: 37/62 Batch time: 0.2923
Phase: train Epoch: 1/3 Iter: 38/62 Batch time: 0.2900
Phase: train Epoch: 1/3 Iter: 39/62 Batch time: 0.2908
Phase: train Epoch: 1/3 Iter: 40/62 Batch time: 0.2976
Phase: train Epoch: 1/3 Iter: 41/62 Batch time: 0.2913
Phase: train Epoch: 1/3 Iter: 42/62 Batch time: 0.2933
Phase: train Epoch: 1/3 Iter: 43/62 Batch time: 0.2921
Phase: train Epoch: 1/3 Iter: 44/62 Batch time: 0.2912
Phase: train Epoch: 1/3 Iter: 45/62 Batch time: 0.2938
Phase: train Epoch: 1/3 Iter: 46/62 Batch time: 0.2947
Phase: train Epoch: 1/3 Iter: 47/62 Batch time: 0.2946
Phase: train Epoch: 1/3 Iter: 48/62 Batch time: 0.2914
Phase: train Epoch: 1/3 Iter: 49/62 Batch time: 0.2980
Phase: train Epoch: 1/3 Iter: 50/62 Batch time: 0.2924
Phase: train Epoch: 1/3 Iter: 51/62 Batch time: 0.2931
Phase: train Epoch: 1/3 Iter: 52/62 Batch time: 0.2946
Phase: train Epoch: 1/3 Iter: 53/62 Batch time: 0.2965
Phase: train Epoch: 1/3 Iter: 54/62 Batch time: 0.2934
Phase: train Epoch: 1/3 Iter: 55/62 Batch time: 0.2919
Phase: train Epoch: 1/3 Iter: 56/62 Batch time: 0.2936
Phase: train Epoch: 1/3 Iter: 57/62 Batch time: 0.2903
Phase: train Epoch: 1/3 Iter: 58/62 Batch time: 0.2973
Phase: train Epoch: 1/3 Iter: 59/62 Batch time: 0.2959
Phase: train Epoch: 1/3 Iter: 60/62 Batch time: 0.2944
Phase: train Epoch: 1/3 Iter: 61/62 Batch time: 0.2944
Phase: train Epoch: 1/3 Loss: 0.6990 Acc: 0.5246
Phase: validation Epoch: 1/3 Iter: 1/39 Batch time: 0.2215
Phase: validation Epoch: 1/3 Iter: 2/39 Batch time: 0.2185
Phase: validation Epoch: 1/3 Iter: 3/39 Batch time: 0.2185
Phase: validation Epoch: 1/3 Iter: 4/39 Batch time: 0.2191
Phase: validation Epoch: 1/3 Iter: 5/39 Batch time: 0.2192
Phase: validation Epoch: 1/3 Iter: 6/39 Batch time: 0.2289
Phase: validation Epoch: 1/3 Iter: 7/39 Batch time: 0.2195
Phase: validation Epoch: 1/3 Iter: 8/39 Batch time: 0.2194
Phase: validation Epoch: 1/3 Iter: 9/39 Batch time: 0.2188
Phase: validation Epoch: 1/3 Iter: 10/39 Batch time: 0.2220
Phase: validation Epoch: 1/3 Iter: 11/39 Batch time: 0.2233
Phase: validation Epoch: 1/3 Iter: 12/39 Batch time: 0.2251
Phase: validation Epoch: 1/3 Iter: 13/39 Batch time: 0.2204
Phase: validation Epoch: 1/3 Iter: 14/39 Batch time: 0.2263
Phase: validation Epoch: 1/3 Iter: 15/39 Batch time: 0.2185
Phase: validation Epoch: 1/3 Iter: 16/39 Batch time: 0.2237
Phase: validation Epoch: 1/3 Iter: 17/39 Batch time: 0.2193
Phase: validation Epoch: 1/3 Iter: 18/39 Batch time: 0.2210
Phase: validation Epoch: 1/3 Iter: 19/39 Batch time: 0.2210
Phase: validation Epoch: 1/3 Iter: 20/39 Batch time: 0.2219
Phase: validation Epoch: 1/3 Iter: 21/39 Batch time: 0.2175
Phase: validation Epoch: 1/3 Iter: 22/39 Batch time: 0.2233
Phase: validation Epoch: 1/3 Iter: 23/39 Batch time: 0.2202
Phase: validation Epoch: 1/3 Iter: 24/39 Batch time: 0.2197
Phase: validation Epoch: 1/3 Iter: 25/39 Batch time: 0.2205
Phase: validation Epoch: 1/3 Iter: 26/39 Batch time: 0.2178
Phase: validation Epoch: 1/3 Iter: 27/39 Batch time: 0.2205
Phase: validation Epoch: 1/3 Iter: 28/39 Batch time: 0.2197
Phase: validation Epoch: 1/3 Iter: 29/39 Batch time: 0.2195
Phase: validation Epoch: 1/3 Iter: 30/39 Batch time: 0.2261
Phase: validation Epoch: 1/3 Iter: 31/39 Batch time: 0.2179
Phase: validation Epoch: 1/3 Iter: 32/39 Batch time: 0.2185
Phase: validation Epoch: 1/3 Iter: 33/39 Batch time: 0.2177
Phase: validation Epoch: 1/3 Iter: 34/39 Batch time: 0.2206
Phase: validation Epoch: 1/3 Iter: 35/39 Batch time: 0.2165
Phase: validation Epoch: 1/3 Iter: 36/39 Batch time: 0.2199
Phase: validation Epoch: 1/3 Iter: 37/39 Batch time: 0.2187
Phase: validation Epoch: 1/3 Iter: 38/39 Batch time: 0.2221
Phase: validation Epoch: 1/3 Iter: 39/39 Batch time: 0.0699
Phase: validation   Epoch: 1/3 Loss: 0.6429 Acc: 0.6536
Phase: train Epoch: 2/3 Iter: 1/62 Batch time: 0.2890
Phase: train Epoch: 2/3 Iter: 2/62 Batch time: 0.2958
Phase: train Epoch: 2/3 Iter: 3/62 Batch time: 0.2970
Phase: train Epoch: 2/3 Iter: 4/62 Batch time: 0.2976
Phase: train Epoch: 2/3 Iter: 5/62 Batch time: 0.2958
Phase: train Epoch: 2/3 Iter: 6/62 Batch time: 0.2977
Phase: train Epoch: 2/3 Iter: 7/62 Batch time: 0.2977
Phase: train Epoch: 2/3 Iter: 8/62 Batch time: 0.3021
Phase: train Epoch: 2/3 Iter: 9/62 Batch time: 0.3001
Phase: train Epoch: 2/3 Iter: 10/62 Batch time: 0.3009
Phase: train Epoch: 2/3 Iter: 11/62 Batch time: 0.3022
Phase: train Epoch: 2/3 Iter: 12/62 Batch time: 0.3010
Phase: train Epoch: 2/3 Iter: 13/62 Batch time: 0.3044
Phase: train Epoch: 2/3 Iter: 14/62 Batch time: 0.2984
Phase: train Epoch: 2/3 Iter: 15/62 Batch time: 0.2931
Phase: train Epoch: 2/3 Iter: 16/62 Batch time: 0.2929
Phase: train Epoch: 2/3 Iter: 17/62 Batch time: 0.2892
Phase: train Epoch: 2/3 Iter: 18/62 Batch time: 0.2951
Phase: train Epoch: 2/3 Iter: 19/62 Batch time: 0.2991
Phase: train Epoch: 2/3 Iter: 20/62 Batch time: 0.2955
Phase: train Epoch: 2/3 Iter: 21/62 Batch time: 0.3017
Phase: train Epoch: 2/3 Iter: 22/62 Batch time: 0.2962
Phase: train Epoch: 2/3 Iter: 23/62 Batch time: 0.2940
Phase: train Epoch: 2/3 Iter: 24/62 Batch time: 0.2987
Phase: train Epoch: 2/3 Iter: 25/62 Batch time: 0.3009
Phase: train Epoch: 2/3 Iter: 26/62 Batch time: 0.2927
Phase: train Epoch: 2/3 Iter: 27/62 Batch time: 0.2963
Phase: train Epoch: 2/3 Iter: 28/62 Batch time: 0.2960
Phase: train Epoch: 2/3 Iter: 29/62 Batch time: 0.2973
Phase: train Epoch: 2/3 Iter: 30/62 Batch time: 0.2971
Phase: train Epoch: 2/3 Iter: 31/62 Batch time: 0.3026
Phase: train Epoch: 2/3 Iter: 32/62 Batch time: 0.3037
Phase: train Epoch: 2/3 Iter: 33/62 Batch time: 0.2963
Phase: train Epoch: 2/3 Iter: 34/62 Batch time: 0.2970
Phase: train Epoch: 2/3 Iter: 35/62 Batch time: 0.2986
Phase: train Epoch: 2/3 Iter: 36/62 Batch time: 0.2925
Phase: train Epoch: 2/3 Iter: 37/62 Batch time: 0.2926
Phase: train Epoch: 2/3 Iter: 38/62 Batch time: 0.2975
Phase: train Epoch: 2/3 Iter: 39/62 Batch time: 0.2950
Phase: train Epoch: 2/3 Iter: 40/62 Batch time: 0.2957
Phase: train Epoch: 2/3 Iter: 41/62 Batch time: 0.3069
Phase: train Epoch: 2/3 Iter: 42/62 Batch time: 0.2959
Phase: train Epoch: 2/3 Iter: 43/62 Batch time: 0.2964
Phase: train Epoch: 2/3 Iter: 44/62 Batch time: 0.3057
Phase: train Epoch: 2/3 Iter: 45/62 Batch time: 0.3012
Phase: train Epoch: 2/3 Iter: 46/62 Batch time: 0.2971
Phase: train Epoch: 2/3 Iter: 47/62 Batch time: 0.3051
Phase: train Epoch: 2/3 Iter: 48/62 Batch time: 0.3023
Phase: train Epoch: 2/3 Iter: 49/62 Batch time: 0.3022
Phase: train Epoch: 2/3 Iter: 50/62 Batch time: 0.3064
Phase: train Epoch: 2/3 Iter: 51/62 Batch time: 0.2985
Phase: train Epoch: 2/3 Iter: 52/62 Batch time: 0.3029
Phase: train Epoch: 2/3 Iter: 53/62 Batch time: 0.3014
Phase: train Epoch: 2/3 Iter: 54/62 Batch time: 0.2989
Phase: train Epoch: 2/3 Iter: 55/62 Batch time: 0.2965
Phase: train Epoch: 2/3 Iter: 56/62 Batch time: 0.2989
Phase: train Epoch: 2/3 Iter: 57/62 Batch time: 0.3011
Phase: train Epoch: 2/3 Iter: 58/62 Batch time: 0.2965
Phase: train Epoch: 2/3 Iter: 59/62 Batch time: 0.2986
Phase: train Epoch: 2/3 Iter: 60/62 Batch time: 0.2972
Phase: train Epoch: 2/3 Iter: 61/62 Batch time: 0.2959
Phase: train Epoch: 2/3 Loss: 0.6134 Acc: 0.7008
Phase: validation Epoch: 2/3 Iter: 1/39 Batch time: 0.2232
Phase: validation Epoch: 2/3 Iter: 2/39 Batch time: 0.2268
Phase: validation Epoch: 2/3 Iter: 3/39 Batch time: 0.2237
Phase: validation Epoch: 2/3 Iter: 4/39 Batch time: 0.2220
Phase: validation Epoch: 2/3 Iter: 5/39 Batch time: 0.2258
Phase: validation Epoch: 2/3 Iter: 6/39 Batch time: 0.2207
Phase: validation Epoch: 2/3 Iter: 7/39 Batch time: 0.2236
Phase: validation Epoch: 2/3 Iter: 8/39 Batch time: 0.2244
Phase: validation Epoch: 2/3 Iter: 9/39 Batch time: 0.2222
Phase: validation Epoch: 2/3 Iter: 10/39 Batch time: 0.2328
Phase: validation Epoch: 2/3 Iter: 11/39 Batch time: 0.2266
Phase: validation Epoch: 2/3 Iter: 12/39 Batch time: 0.2276
Phase: validation Epoch: 2/3 Iter: 13/39 Batch time: 0.2246
Phase: validation Epoch: 2/3 Iter: 14/39 Batch time: 0.2269
Phase: validation Epoch: 2/3 Iter: 15/39 Batch time: 0.2238
Phase: validation Epoch: 2/3 Iter: 16/39 Batch time: 0.2250
Phase: validation Epoch: 2/3 Iter: 17/39 Batch time: 0.2249
Phase: validation Epoch: 2/3 Iter: 18/39 Batch time: 0.2330
Phase: validation Epoch: 2/3 Iter: 19/39 Batch time: 0.2277
Phase: validation Epoch: 2/3 Iter: 20/39 Batch time: 0.2273
Phase: validation Epoch: 2/3 Iter: 21/39 Batch time: 0.2270
Phase: validation Epoch: 2/3 Iter: 22/39 Batch time: 0.2308
Phase: validation Epoch: 2/3 Iter: 23/39 Batch time: 0.2272
Phase: validation Epoch: 2/3 Iter: 24/39 Batch time: 0.2315
Phase: validation Epoch: 2/3 Iter: 25/39 Batch time: 0.2292
Phase: validation Epoch: 2/3 Iter: 26/39 Batch time: 0.2318
Phase: validation Epoch: 2/3 Iter: 27/39 Batch time: 0.2250
Phase: validation Epoch: 2/3 Iter: 28/39 Batch time: 0.2263
Phase: validation Epoch: 2/3 Iter: 29/39 Batch time: 0.2326
Phase: validation Epoch: 2/3 Iter: 30/39 Batch time: 0.2224
Phase: validation Epoch: 2/3 Iter: 31/39 Batch time: 0.2252
Phase: validation Epoch: 2/3 Iter: 32/39 Batch time: 0.2297
Phase: validation Epoch: 2/3 Iter: 33/39 Batch time: 0.2265
Phase: validation Epoch: 2/3 Iter: 34/39 Batch time: 0.2379
Phase: validation Epoch: 2/3 Iter: 35/39 Batch time: 0.2290
Phase: validation Epoch: 2/3 Iter: 36/39 Batch time: 0.2307
Phase: validation Epoch: 2/3 Iter: 37/39 Batch time: 0.2258
Phase: validation Epoch: 2/3 Iter: 38/39 Batch time: 0.2235
Phase: validation Epoch: 2/3 Iter: 39/39 Batch time: 0.0653
Phase: validation   Epoch: 2/3 Loss: 0.5389 Acc: 0.8235
Phase: train Epoch: 3/3 Iter: 1/62 Batch time: 0.2956
Phase: train Epoch: 3/3 Iter: 2/62 Batch time: 0.2969
Phase: train Epoch: 3/3 Iter: 3/62 Batch time: 0.3063
Phase: train Epoch: 3/3 Iter: 4/62 Batch time: 0.3022
Phase: train Epoch: 3/3 Iter: 5/62 Batch time: 0.3022
Phase: train Epoch: 3/3 Iter: 6/62 Batch time: 0.3011
Phase: train Epoch: 3/3 Iter: 7/62 Batch time: 0.3027
Phase: train Epoch: 3/3 Iter: 8/62 Batch time: 0.3018
Phase: train Epoch: 3/3 Iter: 9/62 Batch time: 0.3051
Phase: train Epoch: 3/3 Iter: 10/62 Batch time: 0.3029
Phase: train Epoch: 3/3 Iter: 11/62 Batch time: 0.3027
Phase: train Epoch: 3/3 Iter: 12/62 Batch time: 0.3028
Phase: train Epoch: 3/3 Iter: 13/62 Batch time: 0.3028
Phase: train Epoch: 3/3 Iter: 14/62 Batch time: 0.3008
Phase: train Epoch: 3/3 Iter: 15/62 Batch time: 0.3056
Phase: train Epoch: 3/3 Iter: 16/62 Batch time: 0.3015
Phase: train Epoch: 3/3 Iter: 17/62 Batch time: 0.3013
Phase: train Epoch: 3/3 Iter: 18/62 Batch time: 0.2990
Phase: train Epoch: 3/3 Iter: 19/62 Batch time: 0.2964
Phase: train Epoch: 3/3 Iter: 20/62 Batch time: 0.2975
Phase: train Epoch: 3/3 Iter: 21/62 Batch time: 0.2953
Phase: train Epoch: 3/3 Iter: 22/62 Batch time: 0.2994
Phase: train Epoch: 3/3 Iter: 23/62 Batch time: 0.2962
Phase: train Epoch: 3/3 Iter: 24/62 Batch time: 0.3004
Phase: train Epoch: 3/3 Iter: 25/62 Batch time: 0.3004
Phase: train Epoch: 3/3 Iter: 26/62 Batch time: 0.2997
Phase: train Epoch: 3/3 Iter: 27/62 Batch time: 0.3042
Phase: train Epoch: 3/3 Iter: 28/62 Batch time: 0.3052
Phase: train Epoch: 3/3 Iter: 29/62 Batch time: 0.3042
Phase: train Epoch: 3/3 Iter: 30/62 Batch time: 0.3057
Phase: train Epoch: 3/3 Iter: 31/62 Batch time: 0.3062
Phase: train Epoch: 3/3 Iter: 32/62 Batch time: 0.2991
Phase: train Epoch: 3/3 Iter: 33/62 Batch time: 0.3014
Phase: train Epoch: 3/3 Iter: 34/62 Batch time: 0.3094
Phase: train Epoch: 3/3 Iter: 35/62 Batch time: 0.3020
Phase: train Epoch: 3/3 Iter: 36/62 Batch time: 0.3032
Phase: train Epoch: 3/3 Iter: 37/62 Batch time: 0.3014
Phase: train Epoch: 3/3 Iter: 38/62 Batch time: 0.2978
Phase: train Epoch: 3/3 Iter: 39/62 Batch time: 0.2989
Phase: train Epoch: 3/3 Iter: 40/62 Batch time: 0.3042
Phase: train Epoch: 3/3 Iter: 41/62 Batch time: 0.2987
Phase: train Epoch: 3/3 Iter: 42/62 Batch time: 0.2993
Phase: train Epoch: 3/3 Iter: 43/62 Batch time: 0.3029
Phase: train Epoch: 3/3 Iter: 44/62 Batch time: 0.3004
Phase: train Epoch: 3/3 Iter: 45/62 Batch time: 0.2982
Phase: train Epoch: 3/3 Iter: 46/62 Batch time: 0.3064
Phase: train Epoch: 3/3 Iter: 47/62 Batch time: 0.3090
Phase: train Epoch: 3/3 Iter: 48/62 Batch time: 0.3046
Phase: train Epoch: 3/3 Iter: 49/62 Batch time: 0.3115
Phase: train Epoch: 3/3 Iter: 50/62 Batch time: 0.3095
Phase: train Epoch: 3/3 Iter: 51/62 Batch time: 0.3003
Phase: train Epoch: 3/3 Iter: 52/62 Batch time: 0.3032
Phase: train Epoch: 3/3 Iter: 53/62 Batch time: 0.2989
Phase: train Epoch: 3/3 Iter: 54/62 Batch time: 0.3054
Phase: train Epoch: 3/3 Iter: 55/62 Batch time: 0.3023
Phase: train Epoch: 3/3 Iter: 56/62 Batch time: 0.3072
Phase: train Epoch: 3/3 Iter: 57/62 Batch time: 0.3026
Phase: train Epoch: 3/3 Iter: 58/62 Batch time: 0.3029
Phase: train Epoch: 3/3 Iter: 59/62 Batch time: 0.3005
Phase: train Epoch: 3/3 Iter: 60/62 Batch time: 0.2986
Phase: train Epoch: 3/3 Iter: 61/62 Batch time: 0.3016
Phase: train Epoch: 3/3 Loss: 0.5652 Acc: 0.7418
Phase: validation Epoch: 3/3 Iter: 1/39 Batch time: 0.2283
Phase: validation Epoch: 3/3 Iter: 2/39 Batch time: 0.2248
Phase: validation Epoch: 3/3 Iter: 3/39 Batch time: 0.2243
Phase: validation Epoch: 3/3 Iter: 4/39 Batch time: 0.2249
Phase: validation Epoch: 3/3 Iter: 5/39 Batch time: 0.2293
Phase: validation Epoch: 3/3 Iter: 6/39 Batch time: 0.2248
Phase: validation Epoch: 3/3 Iter: 7/39 Batch time: 0.2240
Phase: validation Epoch: 3/3 Iter: 8/39 Batch time: 0.2217
Phase: validation Epoch: 3/3 Iter: 9/39 Batch time: 0.2227
Phase: validation Epoch: 3/3 Iter: 10/39 Batch time: 0.2218
Phase: validation Epoch: 3/3 Iter: 11/39 Batch time: 0.2251
Phase: validation Epoch: 3/3 Iter: 12/39 Batch time: 0.2248
Phase: validation Epoch: 3/3 Iter: 13/39 Batch time: 0.2291
Phase: validation Epoch: 3/3 Iter: 14/39 Batch time: 0.2221
Phase: validation Epoch: 3/3 Iter: 15/39 Batch time: 0.2248
Phase: validation Epoch: 3/3 Iter: 16/39 Batch time: 0.2236
Phase: validation Epoch: 3/3 Iter: 17/39 Batch time: 0.2274
Phase: validation Epoch: 3/3 Iter: 18/39 Batch time: 0.2232
Phase: validation Epoch: 3/3 Iter: 19/39 Batch time: 0.2222
Phase: validation Epoch: 3/3 Iter: 20/39 Batch time: 0.2261
Phase: validation Epoch: 3/3 Iter: 21/39 Batch time: 0.2278
Phase: validation Epoch: 3/3 Iter: 22/39 Batch time: 0.2282
Phase: validation Epoch: 3/3 Iter: 23/39 Batch time: 0.2256
Phase: validation Epoch: 3/3 Iter: 24/39 Batch time: 0.2289
Phase: validation Epoch: 3/3 Iter: 25/39 Batch time: 0.2286
Phase: validation Epoch: 3/3 Iter: 26/39 Batch time: 0.2265
Phase: validation Epoch: 3/3 Iter: 27/39 Batch time: 0.2233
Phase: validation Epoch: 3/3 Iter: 28/39 Batch time: 0.2252
Phase: validation Epoch: 3/3 Iter: 29/39 Batch time: 0.2220
Phase: validation Epoch: 3/3 Iter: 30/39 Batch time: 0.2216
Phase: validation Epoch: 3/3 Iter: 31/39 Batch time: 0.2235
Phase: validation Epoch: 3/3 Iter: 32/39 Batch time: 0.2242
Phase: validation Epoch: 3/3 Iter: 33/39 Batch time: 0.2236
Phase: validation Epoch: 3/3 Iter: 34/39 Batch time: 0.2226
Phase: validation Epoch: 3/3 Iter: 35/39 Batch time: 0.2232
Phase: validation Epoch: 3/3 Iter: 36/39 Batch time: 0.2230
Phase: validation Epoch: 3/3 Iter: 37/39 Batch time: 0.2277
Phase: validation Epoch: 3/3 Iter: 38/39 Batch time: 0.2218
Phase: validation Epoch: 3/3 Iter: 39/39 Batch time: 0.0658
Phase: validation   Epoch: 3/3 Loss: 0.4484 Acc: 0.8497
Training completed in 1m 27s
Best test loss: 0.4484 | Best test accuracy: 0.8497
 </code>
 </pre>
 </details>

---

## 13. tutorial_qnn_module_tf.html <a name="demo12"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_qnn_module_tf.html):

```
30/30 - 9s - loss: 0.4997 - accuracy: 0.5000 - val_loss: 0.5081 - val_accuracy: 0.4400 - 9s/epoch - 304ms/step
30/30 - 9s - loss: 0.4673 - accuracy: 0.6200 - val_loss: 0.4488 - val_accuracy: 0.6200 - 9s/epoch - 302ms/step
30/30 - 9s - loss: 0.3230 - accuracy: 0.8267 - val_loss: 0.2562 - val_accuracy: 0.8400 - 9s/epoch - 309ms/step
30/30 - 9s - loss: 0.2124 - accuracy: 0.8867 - val_loss: 0.1997 - val_accuracy: 0.8400 - 9s/epoch - 304ms/step
30/30 - 9s - loss: 0.1800 - accuracy: 0.8933 - val_loss: 0.1841 - val_accuracy: 0.8400 - 9s/epoch - 303ms/step
30/30 - 9s - loss: 0.1593 - accuracy: 0.8667 - val_loss: 0.2177 - val_accuracy: 0.8400 - 9s/epoch - 303ms/step
30/30 - 18s - loss: 0.5189 - accuracy: 0.4000 - val_loss: 0.4945 - val_accuracy: 0.5400 - 18s/epoch - 605ms/step
30/30 - 18s - loss: 0.4822 - accuracy: 0.6200 - val_loss: 0.4412 - val_accuracy: 0.7200 - 18s/epoch - 604ms/step
30/30 - 18s - loss: 0.3850 - accuracy: 0.7133 - val_loss: 0.2898 - val_accuracy: 0.7800 - 18s/epoch - 607ms/step
30/30 - 18s - loss: 0.2720 - accuracy: 0.7867 - val_loss: 0.2185 - val_accuracy: 0.8200 - 18s/epoch - 597ms/step
30/30 - 18s - loss: 0.2056 - accuracy: 0.8400 - val_loss: 0.1893 - val_accuracy: 0.8400 - 18s/epoch - 593ms/step
30/30 - 18s - loss: 0.1753 - accuracy: 0.8400 - val_loss: 0.1856 - val_accuracy: 0.8400 - 18s/epoch - 599ms/step
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_qnn_module_tf.html):

```
30/30 - 12s - loss: 0.4997 - accuracy: 0.5000 - val_loss: 0.5081 - val_accuracy: 0.4400 - 12s/epoch - 399ms/step
30/30 - 11s - loss: 0.4673 - accuracy: 0.6200 - val_loss: 0.4488 - val_accuracy: 0.6200 - 11s/epoch - 365ms/step
30/30 - 11s - loss: 0.3230 - accuracy: 0.8267 - val_loss: 0.2562 - val_accuracy: 0.8400 - 11s/epoch - 371ms/step
30/30 - 11s - loss: 0.2124 - accuracy: 0.8867 - val_loss: 0.1997 - val_accuracy: 0.8400 - 11s/epoch - 366ms/step
30/30 - 11s - loss: 0.1800 - accuracy: 0.8933 - val_loss: 0.1841 - val_accuracy: 0.8400 - 11s/epoch - 367ms/step
30/30 - 12s - loss: 0.1593 - accuracy: 0.8667 - val_loss: 0.2177 - val_accuracy: 0.8400 - 12s/epoch - 401ms/step
30/30 - 22s - loss: 0.5189 - accuracy: 0.4000 - val_loss: 0.4945 - val_accuracy: 0.5400 - 22s/epoch - 741ms/step
30/30 - 22s - loss: 0.4822 - accuracy: 0.6200 - val_loss: 0.4412 - val_accuracy: 0.7200 - 22s/epoch - 737ms/step
30/30 - 24s - loss: 0.3850 - accuracy: 0.7133 - val_loss: 0.2898 - val_accuracy: 0.7800 - 24s/epoch - 806ms/step
30/30 - 22s - loss: 0.2720 - accuracy: 0.7867 - val_loss: 0.2185 - val_accuracy: 0.8200 - 22s/epoch - 725ms/step
30/30 - 22s - loss: 0.2056 - accuracy: 0.8400 - val_loss: 0.1893 - val_accuracy: 0.8400 - 22s/epoch - 722ms/step
30/30 - 22s - loss: 0.1753 - accuracy: 0.8400 - val_loss: 0.1856 - val_accuracy: 0.8400 - 22s/epoch - 727ms/step
```

---

