Last update: 2021-12-04  23:24:55 (All times shown in Eastern time)
# List of differences in demonstration outputs

# Table of contents

1. [tutorial_adjoint_diff.html](#demo0)
2. [tutorial_multiclass_classification.html](#demo1)
3. [tutorial_quanvolution.html](#demo2)
4. [tutorial_kernel_based_training.html](#demo3)
5. [tutorial_error_mitigation.html](#demo4)
6. [tutorial_qaoa_intro.html](#demo5)
7. [tutorial_data_reuploading_classifier.html](#demo6)
8. [tutorial_ensemble_multi_qpu.html](#demo7)
9. [tutorial_quantum_metrology.html](#demo8)
10. [tutorial_qnn_module_tf.html](#demo9)
11. [tutorial_expressivity_fourier_series.html](#demo10)
12. [tutorial_vqe_parallel.html](#demo11)
13. [tutorial_measurement_optimize.html](#demo12)
14. [tutorial_general_parshift.html](#demo13)
15. [tutorial_vqt.html](#demo14)
16. [tutorial_quantum_analytic_descent.html](#demo15)
17. [tutorial_quantum_natural_gradient.html](#demo16)
18. [tutorial_rosalin.html](#demo17)
19. [tutorial_unitary_designs.html](#demo18)
20. [tutorial_doubly_stochastic.html](#demo19)
21. [tutorial_adaptive_circuits.html](#demo20)
22. [tutorial_qgrnn.html](#demo21)
23. [tutorial_quantum_chemistry.html](#demo22)
24. [tutorial_falqon.html](#demo23)
25. [tutorial_backprop.html](#demo24)
26. [tutorial_gbs.html](#demo25)
27. [tutorial_quantum_transfer_learning.html](#demo26)
28. [tutorial_classical_shadows.html](#demo27)
29. [tutorial_jax_transformations.html](#demo28)
30. [tutorial_barren_plateaus.html](#demo29)
31. [tutorial_qubit_rotation.html](#demo30)
32. [tutorial_QGAN.html](#demo31)


Number of demos different/all demos: 32/57

## 1. tutorial_adjoint_diff.html <a name="demo0"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_adjoint_diff.html):

```
vdot  :  (0.18884787122715618+3.634721684493463e-19j)
QNode :  0.18884787122715624
(0.18884787122715616+1.9739809094676298e-18j)
(0.18884787122715613+2.9931365520227565e-18j)
(0.18884787122715613+2.9931365520227565e-18j)
our calculation:  [-0.018947989233612104, 0.9316157966884513, -0.05841749223216957]
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_adjoint_diff.html):

```
vdot  :  (0.18884787122715618+0j)
QNode :  0.18884787122715618
(0.18884787122715616+3.4558944247975454e-18j)
(0.18884787122715616+1.6252868755895187e-18j)
(0.18884787122715616+1.6252868755895187e-18j)
our calculation:  [-0.018947989233612107, 0.9316157966884513, -0.05841749223216956]
```

---

## 2. tutorial_multiclass_classification.html <a name="demo1"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_multiclass_classification.html):

```
Iter:    98 | Cost: 0.0701063 | Acc train: 0.8839286 | Acc test: 0.8684211
Iter:    99 | Cost: 0.0827179 | Acc train: 0.9642857 | Acc test: 0.9473684
Iter:    77 | Cost: 0.0461330 | Acc train: 0.9464286 | Acc test: 0.9736842
Iter:    78 | Cost: 0.0674306 | Acc train: 0.9464286 | Acc test: 0.9736842
Iter:    83 | Cost: 0.0823874 | Acc train: 0.9464286 | Acc test: 0.9736842
Iter:    86 | Cost: 0.0964097 | Acc train: 0.9553571 | Acc test: 0.9736842
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_multiclass_classification.html):

```
Iter:    98 | Cost: 0.0701062 | Acc train: 0.8839286 | Acc test: 0.8684211
Iter:    99 | Cost: 0.0827178 | Acc train: 0.9642857 | Acc test: 0.9473684
Iter:    77 | Cost: 0.0461331 | Acc train: 0.9464286 | Acc test: 0.9736842
Iter:    78 | Cost: 0.0674305 | Acc train: 0.9464286 | Acc test: 0.9736842
Iter:    83 | Cost: 0.0823875 | Acc train: 0.9464286 | Acc test: 0.9736842
Iter:    86 | Cost: 0.0964098 | Acc train: 0.9553571 | Acc test: 0.9736842
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
   16384/11490434 [..............................] - ETA: 0s
 4202496/11490434 [=========>....................] - ETA: 0s
 8396800/11490434 [====================>.........] - ETA: 0s
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
13/13 - 0s - loss: 3.0160 - accuracy: 0.1000 - val_loss: 2.0646 - val_accuracy: 0.2000
Epoch 2/30
13/13 - 0s - loss: 2.2510 - accuracy: 0.1800 - val_loss: 1.9801 - val_accuracy: 0.3333
Epoch 3/30
13/13 - 0s - loss: 1.7851 - accuracy: 0.4000 - val_loss: 1.8177 - val_accuracy: 0.2667
Epoch 4/30
13/13 - 0s - loss: 1.3652 - accuracy: 0.5400 - val_loss: 1.6107 - val_accuracy: 0.4667
Epoch 5/30
13/13 - 0s - loss: 1.1317 - accuracy: 0.7800 - val_loss: 1.4723 - val_accuracy: 0.6000
Epoch 6/30
13/13 - 0s - loss: 0.9360 - accuracy: 0.8600 - val_loss: 1.4686 - val_accuracy: 0.5333
Epoch 7/30
13/13 - 0s - loss: 0.7383 - accuracy: 0.9400 - val_loss: 1.3536 - val_accuracy: 0.5667
Epoch 8/30
13/13 - 0s - loss: 0.5846 - accuracy: 0.9800 - val_loss: 1.2785 - val_accuracy: 0.6667
Epoch 9/30
13/13 - 0s - loss: 0.4987 - accuracy: 0.9800 - val_loss: 1.2253 - val_accuracy: 0.6333
Epoch 10/30
13/13 - 0s - loss: 0.3921 - accuracy: 1.0000 - val_loss: 1.2655 - val_accuracy: 0.6333
Epoch 11/30
13/13 - 0s - loss: 0.3617 - accuracy: 1.0000 - val_loss: 1.1555 - val_accuracy: 0.7000
Epoch 12/30
13/13 - 0s - loss: 0.3078 - accuracy: 1.0000 - val_loss: 1.2107 - val_accuracy: 0.6667
Epoch 13/30
13/13 - 0s - loss: 0.2618 - accuracy: 1.0000 - val_loss: 1.1166 - val_accuracy: 0.7333
Epoch 14/30
13/13 - 0s - loss: 0.2463 - accuracy: 1.0000 - val_loss: 1.0624 - val_accuracy: 0.7000
Epoch 15/30
13/13 - 0s - loss: 0.2033 - accuracy: 1.0000 - val_loss: 1.0904 - val_accuracy: 0.7333
Epoch 16/30
13/13 - 0s - loss: 0.1799 - accuracy: 1.0000 - val_loss: 1.0865 - val_accuracy: 0.7333
Epoch 17/30
13/13 - 0s - loss: 0.1682 - accuracy: 1.0000 - val_loss: 1.0385 - val_accuracy: 0.7333
Epoch 18/30
13/13 - 0s - loss: 0.1484 - accuracy: 1.0000 - val_loss: 1.0676 - val_accuracy: 0.7000
Epoch 19/30
13/13 - 0s - loss: 0.1349 - accuracy: 1.0000 - val_loss: 1.0447 - val_accuracy: 0.7000
Epoch 20/30
13/13 - 0s - loss: 0.1255 - accuracy: 1.0000 - val_loss: 0.9935 - val_accuracy: 0.7333
Epoch 21/30
13/13 - 0s - loss: 0.1135 - accuracy: 1.0000 - val_loss: 1.0451 - val_accuracy: 0.7333
Epoch 22/30
13/13 - 0s - loss: 0.1041 - accuracy: 1.0000 - val_loss: 1.0142 - val_accuracy: 0.7333
Epoch 23/30
13/13 - 0s - loss: 0.0983 - accuracy: 1.0000 - val_loss: 0.9893 - val_accuracy: 0.7333
Epoch 24/30
13/13 - 0s - loss: 0.0913 - accuracy: 1.0000 - val_loss: 0.9807 - val_accuracy: 0.7000
Epoch 25/30
13/13 - 0s - loss: 0.0868 - accuracy: 1.0000 - val_loss: 0.9715 - val_accuracy: 0.7333
Epoch 26/30
13/13 - 0s - loss: 0.0788 - accuracy: 1.0000 - val_loss: 0.9850 - val_accuracy: 0.7333
Epoch 27/30
13/13 - 0s - loss: 0.0749 - accuracy: 1.0000 - val_loss: 0.9750 - val_accuracy: 0.7333
Epoch 28/30
13/13 - 0s - loss: 0.0730 - accuracy: 1.0000 - val_loss: 0.9570 - val_accuracy: 0.7667
Epoch 29/30
13/13 - 0s - loss: 0.0681 - accuracy: 1.0000 - val_loss: 0.9895 - val_accuracy: 0.7333
Epoch 30/30
13/13 - 0s - loss: 0.0635 - accuracy: 1.0000 - val_loss: 0.9560 - val_accuracy: 0.7333
Epoch 1/30
13/13 - 0s - loss: 2.3619 - accuracy: 0.1400 - val_loss: 2.0567 - val_accuracy: 0.3667
Epoch 2/30
13/13 - 0s - loss: 1.9696 - accuracy: 0.4200 - val_loss: 1.9381 - val_accuracy: 0.4667
Epoch 3/30
13/13 - 0s - loss: 1.6671 - accuracy: 0.6400 - val_loss: 1.8300 - val_accuracy: 0.4333
Epoch 4/30
13/13 - 0s - loss: 1.4340 - accuracy: 0.7400 - val_loss: 1.7113 - val_accuracy: 0.4333
Epoch 5/30
13/13 - 0s - loss: 1.2342 - accuracy: 0.7600 - val_loss: 1.6044 - val_accuracy: 0.5000
Epoch 6/30
13/13 - 0s - loss: 1.0721 - accuracy: 0.8600 - val_loss: 1.5232 - val_accuracy: 0.5333
Epoch 7/30
13/13 - 0s - loss: 0.9348 - accuracy: 0.9000 - val_loss: 1.4596 - val_accuracy: 0.5667
Epoch 8/30
13/13 - 0s - loss: 0.8178 - accuracy: 0.9200 - val_loss: 1.3921 - val_accuracy: 0.6000
Epoch 9/30
13/13 - 0s - loss: 0.7223 - accuracy: 0.9400 - val_loss: 1.3404 - val_accuracy: 0.6333
Epoch 10/30
13/13 - 0s - loss: 0.6404 - accuracy: 0.9600 - val_loss: 1.3065 - val_accuracy: 0.6667
Epoch 11/30
13/13 - 0s - loss: 0.5772 - accuracy: 1.0000 - val_loss: 1.2644 - val_accuracy: 0.6333
Epoch 12/30
13/13 - 0s - loss: 0.5199 - accuracy: 1.0000 - val_loss: 1.2558 - val_accuracy: 0.6667
Epoch 13/30
13/13 - 0s - loss: 0.4695 - accuracy: 1.0000 - val_loss: 1.2258 - val_accuracy: 0.6667
Epoch 14/30
13/13 - 0s - loss: 0.4238 - accuracy: 1.0000 - val_loss: 1.1897 - val_accuracy: 0.6667
Epoch 15/30
13/13 - 0s - loss: 0.3848 - accuracy: 1.0000 - val_loss: 1.1651 - val_accuracy: 0.6667
Epoch 16/30
13/13 - 0s - loss: 0.3525 - accuracy: 1.0000 - val_loss: 1.1503 - val_accuracy: 0.7333
Epoch 17/30
13/13 - 0s - loss: 0.3245 - accuracy: 1.0000 - val_loss: 1.1374 - val_accuracy: 0.7000
Epoch 18/30
13/13 - 0s - loss: 0.2992 - accuracy: 1.0000 - val_loss: 1.1174 - val_accuracy: 0.6667
Epoch 19/30
13/13 - 0s - loss: 0.2745 - accuracy: 1.0000 - val_loss: 1.1119 - val_accuracy: 0.6667
Epoch 20/30
13/13 - 0s - loss: 0.2551 - accuracy: 1.0000 - val_loss: 1.0903 - val_accuracy: 0.7000
Epoch 21/30
13/13 - 0s - loss: 0.2370 - accuracy: 1.0000 - val_loss: 1.0877 - val_accuracy: 0.6667
Epoch 22/30
13/13 - 0s - loss: 0.2199 - accuracy: 1.0000 - val_loss: 1.0776 - val_accuracy: 0.6667
Epoch 23/30
13/13 - 0s - loss: 0.2050 - accuracy: 1.0000 - val_loss: 1.0675 - val_accuracy: 0.7000
Epoch 24/30
13/13 - 0s - loss: 0.1919 - accuracy: 1.0000 - val_loss: 1.0592 - val_accuracy: 0.7000
Epoch 25/30
13/13 - 0s - loss: 0.1812 - accuracy: 1.0000 - val_loss: 1.0554 - val_accuracy: 0.6667
Epoch 26/30
13/13 - 0s - loss: 0.1691 - accuracy: 1.0000 - val_loss: 1.0477 - val_accuracy: 0.7000
Epoch 27/30
13/13 - 0s - loss: 0.1599 - accuracy: 1.0000 - val_loss: 1.0377 - val_accuracy: 0.7000
Epoch 28/30
13/13 - 0s - loss: 0.1515 - accuracy: 1.0000 - val_loss: 1.0341 - val_accuracy: 0.6667
Epoch 29/30
13/13 - 0s - loss: 0.1426 - accuracy: 1.0000 - val_loss: 1.0291 - val_accuracy: 0.7000
Epoch 30/30
13/13 - 0s - loss: 0.1344 - accuracy: 1.0000 - val_loss: 1.0264 - val_accuracy: 0.7000
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
   16384/11490434 [..............................] - ETA: 2s
  196608/11490434 [..............................] - ETA: 3s
 1875968/11490434 [===>..........................] - ETA: 0s
 9068544/11490434 [======================>.......] - ETA: 0s
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
13/13 - 1s - loss: 3.0160 - accuracy: 0.1000 - val_loss: 2.0646 - val_accuracy: 0.2000
Epoch 2/30
13/13 - 0s - loss: 2.2510 - accuracy: 0.1800 - val_loss: 1.9801 - val_accuracy: 0.3333
Epoch 3/30
13/13 - 0s - loss: 1.7851 - accuracy: 0.4000 - val_loss: 1.8177 - val_accuracy: 0.2667
Epoch 4/30
13/13 - 0s - loss: 1.3652 - accuracy: 0.5400 - val_loss: 1.6107 - val_accuracy: 0.4667
Epoch 5/30
13/13 - 0s - loss: 1.1317 - accuracy: 0.7800 - val_loss: 1.4723 - val_accuracy: 0.6000
Epoch 6/30
13/13 - 0s - loss: 0.9360 - accuracy: 0.8600 - val_loss: 1.4686 - val_accuracy: 0.5333
Epoch 7/30
13/13 - 0s - loss: 0.7383 - accuracy: 0.9400 - val_loss: 1.3536 - val_accuracy: 0.5667
Epoch 8/30
13/13 - 0s - loss: 0.5846 - accuracy: 0.9800 - val_loss: 1.2785 - val_accuracy: 0.6667
Epoch 9/30
13/13 - 0s - loss: 0.4987 - accuracy: 0.9800 - val_loss: 1.2253 - val_accuracy: 0.6333
Epoch 10/30
13/13 - 0s - loss: 0.3921 - accuracy: 1.0000 - val_loss: 1.2655 - val_accuracy: 0.6333
Epoch 11/30
13/13 - 0s - loss: 0.3617 - accuracy: 1.0000 - val_loss: 1.1555 - val_accuracy: 0.7000
Epoch 12/30
13/13 - 0s - loss: 0.3078 - accuracy: 1.0000 - val_loss: 1.2107 - val_accuracy: 0.6667
Epoch 13/30
13/13 - 0s - loss: 0.2618 - accuracy: 1.0000 - val_loss: 1.1166 - val_accuracy: 0.7333
Epoch 14/30
13/13 - 0s - loss: 0.2463 - accuracy: 1.0000 - val_loss: 1.0624 - val_accuracy: 0.7000
Epoch 15/30
13/13 - 0s - loss: 0.2033 - accuracy: 1.0000 - val_loss: 1.0904 - val_accuracy: 0.7333
Epoch 16/30
13/13 - 0s - loss: 0.1799 - accuracy: 1.0000 - val_loss: 1.0865 - val_accuracy: 0.7333
Epoch 17/30
13/13 - 0s - loss: 0.1682 - accuracy: 1.0000 - val_loss: 1.0385 - val_accuracy: 0.7333
Epoch 18/30
13/13 - 0s - loss: 0.1484 - accuracy: 1.0000 - val_loss: 1.0676 - val_accuracy: 0.7000
Epoch 19/30
13/13 - 0s - loss: 0.1349 - accuracy: 1.0000 - val_loss: 1.0447 - val_accuracy: 0.7000
Epoch 20/30
13/13 - 0s - loss: 0.1255 - accuracy: 1.0000 - val_loss: 0.9935 - val_accuracy: 0.7333
Epoch 21/30
13/13 - 0s - loss: 0.1135 - accuracy: 1.0000 - val_loss: 1.0451 - val_accuracy: 0.7333
Epoch 22/30
13/13 - 0s - loss: 0.1041 - accuracy: 1.0000 - val_loss: 1.0142 - val_accuracy: 0.7333
Epoch 23/30
13/13 - 0s - loss: 0.0983 - accuracy: 1.0000 - val_loss: 0.9893 - val_accuracy: 0.7333
Epoch 24/30
13/13 - 0s - loss: 0.0913 - accuracy: 1.0000 - val_loss: 0.9807 - val_accuracy: 0.7000
Epoch 25/30
13/13 - 0s - loss: 0.0868 - accuracy: 1.0000 - val_loss: 0.9715 - val_accuracy: 0.7333
Epoch 26/30
13/13 - 0s - loss: 0.0788 - accuracy: 1.0000 - val_loss: 0.9850 - val_accuracy: 0.7333
Epoch 27/30
13/13 - 0s - loss: 0.0749 - accuracy: 1.0000 - val_loss: 0.9750 - val_accuracy: 0.7333
Epoch 28/30
13/13 - 0s - loss: 0.0730 - accuracy: 1.0000 - val_loss: 0.9570 - val_accuracy: 0.7667
Epoch 29/30
13/13 - 0s - loss: 0.0681 - accuracy: 1.0000 - val_loss: 0.9895 - val_accuracy: 0.7333
Epoch 30/30
13/13 - 0s - loss: 0.0635 - accuracy: 1.0000 - val_loss: 0.9560 - val_accuracy: 0.7333
Epoch 1/30
13/13 - 0s - loss: 2.3619 - accuracy: 0.1400 - val_loss: 2.0567 - val_accuracy: 0.3667
Epoch 2/30
13/13 - 0s - loss: 1.9696 - accuracy: 0.4200 - val_loss: 1.9381 - val_accuracy: 0.4667
Epoch 3/30
13/13 - 0s - loss: 1.6671 - accuracy: 0.6400 - val_loss: 1.8300 - val_accuracy: 0.4333
Epoch 4/30
13/13 - 0s - loss: 1.4340 - accuracy: 0.7400 - val_loss: 1.7113 - val_accuracy: 0.4333
Epoch 5/30
13/13 - 0s - loss: 1.2342 - accuracy: 0.7600 - val_loss: 1.6044 - val_accuracy: 0.5000
Epoch 6/30
13/13 - 0s - loss: 1.0721 - accuracy: 0.8600 - val_loss: 1.5232 - val_accuracy: 0.5333
Epoch 7/30
13/13 - 0s - loss: 0.9348 - accuracy: 0.9000 - val_loss: 1.4596 - val_accuracy: 0.5667
Epoch 8/30
13/13 - 0s - loss: 0.8178 - accuracy: 0.9200 - val_loss: 1.3921 - val_accuracy: 0.6000
Epoch 9/30
13/13 - 0s - loss: 0.7223 - accuracy: 0.9400 - val_loss: 1.3404 - val_accuracy: 0.6333
Epoch 10/30
13/13 - 0s - loss: 0.6404 - accuracy: 0.9600 - val_loss: 1.3065 - val_accuracy: 0.6667
Epoch 11/30
13/13 - 0s - loss: 0.5772 - accuracy: 1.0000 - val_loss: 1.2644 - val_accuracy: 0.6333
Epoch 12/30
13/13 - 0s - loss: 0.5199 - accuracy: 1.0000 - val_loss: 1.2558 - val_accuracy: 0.6667
Epoch 13/30
13/13 - 0s - loss: 0.4695 - accuracy: 1.0000 - val_loss: 1.2258 - val_accuracy: 0.6667
Epoch 14/30
13/13 - 0s - loss: 0.4238 - accuracy: 1.0000 - val_loss: 1.1897 - val_accuracy: 0.6667
Epoch 15/30
13/13 - 0s - loss: 0.3848 - accuracy: 1.0000 - val_loss: 1.1651 - val_accuracy: 0.6667
Epoch 16/30
13/13 - 0s - loss: 0.3525 - accuracy: 1.0000 - val_loss: 1.1503 - val_accuracy: 0.7333
Epoch 17/30
13/13 - 0s - loss: 0.3245 - accuracy: 1.0000 - val_loss: 1.1374 - val_accuracy: 0.7000
Epoch 18/30
13/13 - 0s - loss: 0.2992 - accuracy: 1.0000 - val_loss: 1.1174 - val_accuracy: 0.6667
Epoch 19/30
13/13 - 0s - loss: 0.2745 - accuracy: 1.0000 - val_loss: 1.1119 - val_accuracy: 0.6667
Epoch 20/30
13/13 - 0s - loss: 0.2551 - accuracy: 1.0000 - val_loss: 1.0903 - val_accuracy: 0.7000
Epoch 21/30
13/13 - 0s - loss: 0.2370 - accuracy: 1.0000 - val_loss: 1.0877 - val_accuracy: 0.6667
Epoch 22/30
13/13 - 0s - loss: 0.2199 - accuracy: 1.0000 - val_loss: 1.0776 - val_accuracy: 0.6667
Epoch 23/30
13/13 - 0s - loss: 0.2050 - accuracy: 1.0000 - val_loss: 1.0675 - val_accuracy: 0.7000
Epoch 24/30
13/13 - 0s - loss: 0.1919 - accuracy: 1.0000 - val_loss: 1.0592 - val_accuracy: 0.7000
Epoch 25/30
13/13 - 0s - loss: 0.1812 - accuracy: 1.0000 - val_loss: 1.0554 - val_accuracy: 0.6667
Epoch 26/30
13/13 - 0s - loss: 0.1691 - accuracy: 1.0000 - val_loss: 1.0477 - val_accuracy: 0.7000
Epoch 27/30
13/13 - 0s - loss: 0.1599 - accuracy: 1.0000 - val_loss: 1.0377 - val_accuracy: 0.7000
Epoch 28/30
13/13 - 0s - loss: 0.1515 - accuracy: 1.0000 - val_loss: 1.0341 - val_accuracy: 0.6667
Epoch 29/30
13/13 - 0s - loss: 0.1426 - accuracy: 1.0000 - val_loss: 1.0291 - val_accuracy: 0.7000
Epoch 30/30
 </code>
 </pre>
 </details>

---

## 4. tutorial_kernel_based_training.html <a name="demo3"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_kernel_based_training.html):

```
step 0 , loss 1.2128428849025235
step 10 , loss 0.8582750956106432
step 20 , loss 0.43849890579633255
step 30 , loss 0.645882927459064
step 40 , loss 0.5540116701446129
step 70 , loss 0.4694193423160378
step 80 , loss 0.48581457440211384
step 90 , loss 0.4196234621534021
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_kernel_based_training.html):

```
step 0 , loss 1.2128428849025232
step 10 , loss 0.8582750956106431
step 20 , loss 0.43849890579633233
step 30 , loss 0.6458829274590642
step 40 , loss 0.5540116701446127
step 70 , loss 0.469419342316038
step 80 , loss 0.4858145744021141
step 90 , loss 0.4196234621534023
```

---

## 5. tutorial_error_mitigation.html <a name="demo4"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_error_mitigation.html):

```
2: ──RY(4.05)──╭C──────────RY(3.32)────────────────────────╰Z──────────RY(1.07)──RY(-1.07)──╰Z──RY(-3.32)──╭C──RY(-4.05)──┤
3: ──RY(3.51)──╰Z──────────RY(3.66)───RY(-3.66)────────────────────────────────────────────────────────────╰Z──RY(-3.51)──┤
0: ──RY(4.56)──╭C──────────RY(5.93)───RY(-5.93)────────────────────────────────────────────────────────────╭C──RY(-4.56)──┤
1: ──RY(3.6)───╰Z──────────RY(5.9)─────────────────────────╭C──────────RY(5.18)──RY(-5.18)──╭C──RY(-5.9)───╰Z──RY(-3.6)───┤
2: ──RY(4.05)───RY(-4.05)──RY(4.05)──╭C──────────RY(3.32)──╰Z──────────RY(1.07)──RY(-1.07)──╰Z──RY(-3.32)──╭C──RY(-4.05)──┤
3: ──RY(3.51)────────────────────────╰Z──────────RY(3.66)───RY(-3.66)──────────────────────────────────────╰Z──RY(-3.51)──┤
0.8985196547410969
0.9589759497509437
ZNE result: 0.8985196547410969
0.9654528630154192
0: ──RY(4.56)──╭C──RY(5.93)───RY(-5.93)───────────────────────────────────────────────────────────╭C──RY(-4.56)──┤
1: ──RY(3.6)───╰Z──RY(5.9)───╭C──────────RY(5.18)──RY(-5.18)──RY(5.18)──RY(-5.18)──╭C──RY(-5.9)───╰Z──RY(-3.6)───┤
2: ──RY(4.05)──╭C──RY(3.32)──╰Z──────────RY(1.07)──RY(-1.07)───────────────────────╰Z──RY(-3.32)──╭C──RY(-4.05)──┤
3: ──RY(3.51)──╰Z──RY(3.66)───RY(-3.66)───────────────────────────────────────────────────────────╰Z──RY(-3.51)──┤
0: ──RY(4.56)───RY(-4.56)──RY(4.56)──╭C──────────RY(5.93)───RY(-5.93)──────────────────────────────────────╭C──RY(-4.56)──┤
1: ──RY(3.6)─────────────────────────╰Z──────────RY(5.9)───╭C──────────RY(5.18)──RY(-5.18)──╭C──RY(-5.9)───╰Z──RY(-3.6)───┤
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_error_mitigation.html):

```
2: ──RY(4.05)──╭C──RY(3.32)──╰Z──────────RY(1.07)──RY(-1.07)──╰Z──RY(-3.32)──╭C───RY(-4.05)─────────────────┤
3: ──RY(3.51)──╰Z──RY(3.66)───RY(-3.66)──────────────────────────────────────╰Z───RY(-3.51)─────────────────┤
0: ──RY(4.56)──╭C──RY(5.93)───RY(-5.93)──────────────────────────────────────╭C───RY(-4.56)─────────────────┤
1: ──RY(3.6)───╰Z──RY(5.9)───╭C──────────RY(5.18)──RY(-5.18)──╭C──RY(-5.9)───╰Z───RY(-3.6)──────────────────┤
2: ──RY(4.05)──╭C──RY(3.32)──╰Z──────────RY(1.07)──RY(-1.07)──╰Z──RY(-3.32)──╭C──╭C──────────╭C──RY(-4.05)──┤
3: ──RY(3.51)──╰Z──RY(3.66)───RY(-3.66)──────────────────────────────────────╰Z──╰Z──────────╰Z──RY(-3.51)──┤
0.8985196547410973
0.9589759451472933
ZNE result: 0.8985196547410973
0.9613227569563876
0: ──RY(4.56)──╭C──RY(5.93)───RY(-5.93)──────────────────────────────────────╭C─────────RY(-4.56)─────────────────┤
1: ──RY(3.6)───╰Z──RY(5.9)───╭C──────────RY(5.18)──RY(-5.18)──╭C──RY(-5.9)───╰Z─────────RY(-3.6)──────────────────┤
2: ──RY(4.05)──╭C──RY(3.32)──╰Z──────────RY(1.07)──RY(-1.07)──╰Z──RY(-3.32)───RY(3.32)──RY(-3.32)──╭C──RY(-4.05)──┤
3: ──RY(3.51)──╰Z──RY(3.66)───RY(-3.66)────────────────────────────────────────────────────────────╰Z──RY(-3.51)──┤
0: ──RY(4.56)──╭C──RY(5.93)───RY(-5.93)──────────────────────────────────────╭C──╭C──────────╭C──RY(-4.56)──┤
1: ──RY(3.6)───╰Z──RY(5.9)───╭C──────────RY(5.18)──RY(-5.18)──╭C──RY(-5.9)───╰Z──╰Z──────────╰Z──RY(-3.6)───┤
```

---

## 6. tutorial_qaoa_intro.html <a name="demo5"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_qaoa_intro.html):

```
0: ──H──────RZ(1)──H──H──╭RZ(0.5)──H──H──────RZ(1)──H──H──╭RZ(0.5)──H──┤ ⟨Z⟩
1: ──RZ(1)──H────────────╰RZ(0.5)──H──RZ(1)──H────────────╰RZ(0.5)──H──┤ ⟨Z⟩
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_qaoa_intro.html):

```
0: ──╭ApproxTimeEvolution(1, 1, 0.5, 1)──┤ ⟨Z⟩
1: ──╰ApproxTimeEvolution(1, 1, 0.5, 1)──┤ ⟨Z⟩
```

---

## 7. tutorial_data_reuploading_classifier.html <a name="demo6"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_data_reuploading_classifier.html):

```
Epoch:  1 | Loss: 0.200164 | Train accuracy: 0.675000 | Test accuracy: 0.704000
Epoch:  2 | Loss: 0.225628 | Train accuracy: 0.640000 | Test accuracy: 0.692500
Epoch:  3 | Loss: 0.164149 | Train accuracy: 0.750000 | Test accuracy: 0.746000
Epoch:  4 | Loss: 0.143985 | Train accuracy: 0.795000 | Test accuracy: 0.773000
Epoch:  5 | Loss: 0.116077 | Train accuracy: 0.860000 | Test accuracy: 0.827500
Epoch:  6 | Loss: 0.117629 | Train accuracy: 0.845000 | Test accuracy: 0.807000
Epoch:  7 | Loss: 0.103391 | Train accuracy: 0.890000 | Test accuracy: 0.853000
Epoch:  8 | Loss: 0.100581 | Train accuracy: 0.910000 | Test accuracy: 0.861500
Epoch:  9 | Loss: 0.106676 | Train accuracy: 0.870000 | Test accuracy: 0.821500
Epoch: 10 | Loss: 0.099787 | Train accuracy: 0.900000 | Test accuracy: 0.871000
Cost: 0.099787 | Train accuracy 0.900000 | Test Accuracy : 0.871000
Layer 0: [ 0.53841959  1.21036237 -0.08101526]
Layer 1: [-0.33445488  0.64181687 -0.59442591]
Layer 2: [-2.29400846 -1.18534645  0.32099705]
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_data_reuploading_classifier.html):

```
Epoch:  1 | Loss: 0.125417 | Train accuracy: 0.840000 | Test accuracy: 0.804000
Epoch:  2 | Loss: 0.154322 | Train accuracy: 0.775000 | Test accuracy: 0.756000
Epoch:  3 | Loss: 0.145234 | Train accuracy: 0.810000 | Test accuracy: 0.799000
Epoch:  4 | Loss: 0.126142 | Train accuracy: 0.805000 | Test accuracy: 0.781500
Epoch:  5 | Loss: 0.127102 | Train accuracy: 0.845000 | Test accuracy: 0.794500
Epoch:  6 | Loss: 0.128556 | Train accuracy: 0.825000 | Test accuracy: 0.807000
Epoch:  7 | Loss: 0.113327 | Train accuracy: 0.810000 | Test accuracy: 0.794500
Epoch:  8 | Loss: 0.109549 | Train accuracy: 0.895000 | Test accuracy: 0.857000
Epoch:  9 | Loss: 0.147936 | Train accuracy: 0.750000 | Test accuracy: 0.750000
Epoch: 10 | Loss: 0.104038 | Train accuracy: 0.890000 | Test accuracy: 0.847000
Cost: 0.104038 | Train accuracy 0.890000 | Test Accuracy : 0.847000
Layer 0: [-0.23838965  1.17081693 -0.19781887]
Layer 1: [0.64850867 0.71778245 0.46408056]
Layer 2: [ 2.39560597 -1.21404538  0.32099705]
```

---

## 8. tutorial_ensemble_multi_qpu.html <a name="demo7"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_ensemble_multi_qpu.html):

```
Training accuracy (ensemble): 0.808
Training accuracy (QPU1):  0.288
Choices: [0 0 0 1 0 0 1 0 0 0 1 1 0 0 0 0 0 1 1 0 0 1 0 1 1 0 0 0 1 0 0 1 0 1 0 0 0
 0 1 1 0 0 0 0 0 0 0 1 1 0 0 0 0 0 1 0 0 0 0 0 1 0 0 0 1 0 0 0 0 0 1 0 0 0
 0 0 0 0 0 1 1 1 1 1 0 0 1 0 1 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0
Choices counts: Counter({0: 110, 1: 40})
Counter({0: 55, 2: 55})
Counter({1: 37, 0: 3})
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_ensemble_multi_qpu.html):

```
Training accuracy (ensemble): 0.832
Training accuracy (QPU1):  0.296
Choices: [0 0 1 1 0 0 1 0 0 0 1 1 0 0 0 0 0 1 1 0 0 1 0 1 1 0 0 0 1 0 0 1 0 1 0 0 0
 0 1 1 0 0 0 0 0 0 0 1 1 0 0 0 0 0 1 0 0 0 0 0 1 0 1 0 1 0 0 0 0 0 1 0 0 0
 0 0 0 0 0 1 1 1 1 0 0 0 1 0 1 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0
Choices counts: Counter({0: 109, 1: 41})
Counter({0: 55, 2: 54})
Counter({1: 38, 0: 3})
```

---

## 9. tutorial_quantum_metrology.html <a name="demo8"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_quantum_metrology.html):

```
0: ──H──RZ(0)──H──RX(1.57)──RZ(1)──RX(-1.57)──H───────────────────────────────────────────────────────────╭X──RZ(8)──╭X──H──H─────────╭X──RZ(9)──╭X──H──────────H──╭X──RZ(10)──╭X──H──H─────────╭X──RZ(11)──╭X──H──────────H──────╭X──RZ(12)──╭X───H──H────────────────╭X──RZ(13)──╭X───H──RZ(0)──────PhaseDamp(0.2)──H───────────────RZ(14)──H───────RX(1.57)──RZ(15)────RX(-1.57)─────────────╭┤ Probs
1: ──H──RZ(2)──H──RX(1.57)──RZ(3)──RX(-1.57)──H──╭X──RZ(6)──╭X──H──H─────────╭X──RZ(7)──╭X──H──────────H──╰C─────────╰C──H──RX(1.57)──╰C─────────╰C──RX(-1.57)──H──│───────────│────────────────│───────────│─────────────────╭X──╰C──────────╰C──╭X──H──H─────────╭X──╰C──────────╰C──╭X──H──────────RZ(0)───────────PhaseDamp(0.2)──H───────RZ(16)──H─────────RX(1.57)──RZ(17)─────RX(-1.57)──├┤ Probs
2: ──H──RZ(4)──H──RX(1.57)──RZ(5)──RX(-1.57)──H──╰C─────────╰C──H──RX(1.57)──╰C─────────╰C──RX(-1.57)──H───────────────────────────────────────────────────────────╰C──────────╰C──H──RX(1.57)──╰C──────────╰C──RX(-1.57)──H──╰C──────────────────╰C──H──RX(1.57)──╰C──────────────────╰C──RX(-1.57)──RZ(0)───────────PhaseDamp(0.2)──H───────RZ(18)──H─────────RX(1.57)──RZ(19)─────RX(-1.57)──╰┤ Probs
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_quantum_metrology.html):

```
0: ──╭RX(0)──╭RY(1)──╭RI(2)──╭RI(3)──╭RI(4)──╭RI(5)──╭RI(6)──╭RI(7)──╭RX(8)──╭RX(9)──╭RX(10)──╭RX(11)──╭RX(12)──╭RX(13)──RZ(0)──PhaseDamp(0.2)──RX(14)──RY(15)──╭┤ Probs
1: ──├RI(0)──├RI(1)──├RX(2)──├RY(3)──├RI(4)──├RI(5)──├RX(6)──├RX(7)──├RX(8)──├RY(9)──├RI(10)──├RI(11)──├RX(12)──├RX(13)──RZ(0)──PhaseDamp(0.2)──RX(16)──RY(17)──├┤ Probs
2: ──╰RI(0)──╰RI(1)──╰RI(2)──╰RI(3)──╰RX(4)──╰RY(5)──╰RX(6)──╰RY(7)──╰RI(8)──╰RI(9)──╰RX(10)──╰RY(11)──╰RX(12)──╰RY(13)──RZ(0)──PhaseDamp(0.2)──RX(18)──RY(19)──╰┤ Probs
```

---

## 10. tutorial_qnn_module_tf.html <a name="demo9"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_qnn_module_tf.html):

```
30/30 - 10s - loss: 0.4997 - accuracy: 0.5000 - val_loss: 0.5081 - val_accuracy: 0.4400
30/30 - 10s - loss: 0.4673 - accuracy: 0.6200 - val_loss: 0.4488 - val_accuracy: 0.6200
30/30 - 10s - loss: 0.3230 - accuracy: 0.8267 - val_loss: 0.2562 - val_accuracy: 0.8400
30/30 - 10s - loss: 0.2124 - accuracy: 0.8867 - val_loss: 0.1997 - val_accuracy: 0.8400
30/30 - 10s - loss: 0.1800 - accuracy: 0.8933 - val_loss: 0.1841 - val_accuracy: 0.8400
30/30 - 10s - loss: 0.1593 - accuracy: 0.8667 - val_loss: 0.2177 - val_accuracy: 0.8400
30/30 - 19s - loss: 0.5189 - accuracy: 0.4000 - val_loss: 0.4945 - val_accuracy: 0.5400
30/30 - 19s - loss: 0.4822 - accuracy: 0.6200 - val_loss: 0.4412 - val_accuracy: 0.7200
30/30 - 20s - loss: 0.3850 - accuracy: 0.7133 - val_loss: 0.2898 - val_accuracy: 0.7800
30/30 - 19s - loss: 0.2720 - accuracy: 0.7867 - val_loss: 0.2185 - val_accuracy: 0.8200
30/30 - 20s - loss: 0.2056 - accuracy: 0.8400 - val_loss: 0.1893 - val_accuracy: 0.8400
30/30 - 19s - loss: 0.1753 - accuracy: 0.8400 - val_loss: 0.1856 - val_accuracy: 0.8400
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_qnn_module_tf.html):

```
30/30 - 16s - loss: 0.4997 - accuracy: 0.5000 - val_loss: 0.5081 - val_accuracy: 0.4400
30/30 - 16s - loss: 0.4673 - accuracy: 0.6200 - val_loss: 0.4488 - val_accuracy: 0.6200
30/30 - 16s - loss: 0.3230 - accuracy: 0.8267 - val_loss: 0.2562 - val_accuracy: 0.8400
30/30 - 16s - loss: 0.2124 - accuracy: 0.8867 - val_loss: 0.1997 - val_accuracy: 0.8400
30/30 - 16s - loss: 0.1800 - accuracy: 0.8933 - val_loss: 0.1841 - val_accuracy: 0.8400
30/30 - 16s - loss: 0.1593 - accuracy: 0.8667 - val_loss: 0.2177 - val_accuracy: 0.8400
30/30 - 33s - loss: 0.5189 - accuracy: 0.4000 - val_loss: 0.4945 - val_accuracy: 0.5400
30/30 - 32s - loss: 0.4822 - accuracy: 0.6200 - val_loss: 0.4412 - val_accuracy: 0.7200
30/30 - 32s - loss: 0.3850 - accuracy: 0.7133 - val_loss: 0.2898 - val_accuracy: 0.7800
30/30 - 33s - loss: 0.2720 - accuracy: 0.7867 - val_loss: 0.2185 - val_accuracy: 0.8200
30/30 - 32s - loss: 0.2056 - accuracy: 0.8400 - val_loss: 0.1893 - val_accuracy: 0.8400
30/30 - 32s - loss: 0.1753 - accuracy: 0.8400 - val_loss: 0.1856 - val_accuracy: 0.8400
```

---

## 11. tutorial_expressivity_fourier_series.html <a name="demo10"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_expressivity_fourier_series.html):

```
Cost at step  10: 0.04735694890119537
Cost at step  20: 0.04193426710325428
Cost at step  30: 0.0056074793613251055
Cost at step  40: 0.004608455923986289
Cost at step  50: 0.0016064517040624577
0: ──Rot(1.38, 4.29, 0.478)──╭C─────────────────────────────╭X──Rot(4.26, 3.55, 1.68)──╭C──╭X──────┤ ⟨I⟩
1: ──Rot(5.35, 3.11, 3.02)───╰X──╭C──Rot(5.52, 5.01, 4.14)──│──────────────────────────│───╰C──╭X──┤
2: ──Rot(3.72, 5.18, 2.19)───────╰X─────────────────────────╰C──Rot(5.34, 5.45, 4.45)──╰X──────╰C──┤
Cost at step  10: 0.022632957627087866
Cost at step  20: 0.001888527061988235
Cost at step  30: 0.0017982806807008216
Cost at step  40: 0.0007504225639151788
Cost at step  50: 0.0006664901287576844
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_expressivity_fourier_series.html):

```
Cost at step  10: 0.03212041720004563
Cost at step  20: 0.013853561883024695
Cost at step  30: 0.004049396436389428
Cost at step  40: 0.0005624933894468379
Cost at step  50: 8.145777333271188e-05
 0: ──╭StronglyEntanglingLayers(M0)──┤ ⟨I⟩
 1: ──├StronglyEntanglingLayers(M0)──┤
 2: ──╰StronglyEntanglingLayers(M0)──┤
M0 =
[[[1.38345186 4.29304142 0.4783443 ]
  [5.34829078 3.11109738 3.01961452]
  [3.72220789 5.18162333 2.1853497 ]]
 [[4.26010113 3.55459876 1.6777881 ]
```

---

## 12. tutorial_vqe_parallel.html <a name="demo11"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_vqe_parallel.html):

```
Speed up: 2.81
Evaluation time: 272.09 s
Evaluation time: 96.69 s
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_vqe_parallel.html):

```
Speed up: 3.19
Evaluation time: 320.88 s
Evaluation time: 100.68 s
```

---

## 13. tutorial_measurement_optimize.html <a name="demo12"></a>

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
Cost function value: -0.09248671036774045
   (-46.46390678868894) [I0]
+ (0.782966172595019) [Z11]
+ (0.7829661725950192) [Z10]
+ (0.8084581961720481) [Z12]
+ (0.8084581961720483) [Z13]
+ (1.2034402289145636) [Z4]
+ (1.2034402289145638) [Z5]
+ (1.3096862988615425) [Z6]
+ (1.3096862988615428) [Z7]
+ (1.369352563471818) [Z8]
+ (1.369352563471818) [Z9]
+ (1.65389422268317) [Z3]
+ (1.6538942226831703) [Z2]
+ (12.41263074211177) [Z0]
+ (12.41263074211177) [Z1]
+ (-8.194261372196947e-06) [Y10 Y12]
+ (-8.194261372196947e-06) [X10 X12]
+ (-1.854060857954981e-06) [Y5 Y7]
+ (-1.854060857954981e-06) [X5 X7]
+ (-7.764994118434258e-07) [Y3 Y5]
+ (-7.764994118434258e-07) [X3 X5]
+ (-5.929765815774793e-07) [Y4 Y6]
+ (-5.929765815774793e-07) [X4 X6]
+ (1.6021167406021185e-06) [Y2 Y4]
+ (1.6021167406021185e-06) [X2 X4]
+ (7.954413176110595e-06) [Y11 Y13]
+ (7.954413176110595e-06) [X11 X13]
+ (0.003276971931231638) [Y1 Y3]
+ (0.003276971931231638) [X1 X3]
+ (0.10433064780651397) [Y0 Y2]
+ (0.10433064780651397) [X0 X2]
+ (0.11270386920332226) [Z10 Z12]
+ (0.11270386920332226) [Z11 Z13]
+ (0.11383573679388677) [Z4 Z12]
+ (0.11383573679388677) [Z5 Z13]
+ (0.11952438964682688) [Z6 Z10]
+ (0.11952438964682688) [Z7 Z11]
+ (0.12489990917237609) [Z4 Z10]
+ (0.12489990917237609) [Z5 Z11]
+ (0.12495807739503226) [Z2 Z4]
+ (0.12495807739503226) [Z3 Z5]
+ (0.12799502492468412) [Z2 Z10]
+ (0.12799502492468412) [Z3 Z11]
+ (0.1340171526196373) [Z6 Z12]
+ (0.1340171526196373) [Z7 Z13]
+ (0.13734953064261346) [Z6 Z11]
+ (0.13734953064261346) [Z7 Z10]
+ (0.1373910476268324) [Z2 Z6]
+ (0.1373910476268324) [Z3 Z7]
+ (0.1376687264585258) [Z8 Z10]
+ (0.1376687264585258) [Z9 Z11]
+ (0.14011289865354817) [Z2 Z12]
+ (0.14011289865354817) [Z3 Z13]
+ (0.14138905291942816) [Z10 Z13]
+ (0.14138905291942816) [Z11 Z12]
+ (0.14257997712485765) [Z4 Z11]
+ (0.14257997712485765) [Z5 Z10]
+ (0.1472294321876617) [Z8 Z11]
+ (0.1472294321876617) [Z9 Z10]
+ (0.1489943057506557) [Z4 Z7]
+ (0.1489943057506557) [Z5 Z6]
+ (0.14926355147388895) [Z10 Z11]
+ (0.14973486803496935) [Z8 Z12]
+ (0.14973486803496935) [Z9 Z13]
+ (0.15071408121008292) [Z2 Z8]
+ (0.15071408121008292) [Z3 Z9]
+ (0.1513832716142887) [Z6 Z13]
+ (0.1513832716142887) [Z7 Z12]
+ (0.15215040708869068) [Z4 Z13]
+ (0.15215040708869068) [Z5 Z12]
+ (0.15337968243314146) [Z2 Z11]
+ (0.15337968243314146) [Z3 Z10]
+ (0.1543574865722365) [Z12 Z13]
+ (0.1556901067175247) [Z2 Z13]
+ (0.1556901067175247) [Z3 Z12]
+ (0.15582269051553121) [Z8 Z13]
+ (0.15582269051553121) [Z9 Z12]
+ (0.1607976453483857) [Z2 Z5]
+ (0.1607976453483857) [Z3 Z4]
+ (0.16756653265461283) [Z6 Z8]
+ (0.16756653265461283) [Z7 Z9]
+ (0.16853486561579967) [Z2 Z7]
+ (0.16853486561579967) [Z3 Z6]
+ (0.18143991440303892) [Z6 Z9]
+ (0.18143991440303892) [Z7 Z8]
+ (0.18189085790751366) [Z2 Z3]
+ (0.19299723935364238) [Z0 Z10]
+ (0.19299723935364238) [Z1 Z11]
+ (0.19392534613270224) [Z6 Z7]
+ (0.19661770890342148) [Z0 Z4]
+ (0.19661770890342148) [Z1 Z5]
+ (0.1993635453736083) [Z0 Z5]
+ (0.1993635453736083) [Z1 Z4]
+ (0.20072866460441763) [Z0 Z11]
+ (0.20072866460441763) [Z1 Z10]
+ (0.21102659849791527) [Z0 Z12]
+ (0.21102659849791527) [Z1 Z13]
+ (0.21631037498631822) [Z0 Z13]
+ (0.21631037498631822) [Z1 Z12]
+ (0.2367108078383043) [Z0 Z2]
+ (0.2367108078383043) [Z1 Z3]
+ (0.24164663936017214) [Z0 Z6]
+ (0.24164663936017214) [Z1 Z7]
+ (0.2485348337131427) [Z0 Z7]
+ (0.2485348337131427) [Z1 Z6]
+ (0.251294456745917) [Z0 Z3]
+ (0.251294456745917) [Z1 Z2]
+ (0.2723251830660568) [Z0 Z8]
+ (0.2723251830660568) [Z1 Z9]
+ (0.2788345442672341) [Z0 Z9]
+ (0.2788345442672341) [Z1 Z8]
+ (1.1861763734860495) [Z0 Z1]
+ (-1.2260484988963635e-05) [Y4 Z5 Y6]
+ (-1.2260484988963635e-05) [X4 Z5 X6]
+ (-1.2260484988963631e-05) [Y5 Z6 Y7]
+ (-1.2260484988963631e-05) [X5 Z6 X7]
+ (-1.0722312157832765e-05) [Y10 Z11 Y12]
+ (-1.0722312157832765e-05) [X10 Z11 X12]
+ (-1.0722312157832764e-05) [Y11 Z12 Y13]
+ (-1.0722312157832764e-05) [X11 Z12 X13]
+ (-3.887051673089415e-06) [Y3 Z4 Y5]
+ (-3.887051673089415e-06) [X3 Z4 X5]
+ (-3.8870516730894144e-06) [Y2 Z3 Y4]
+ (-3.8870516730894144e-06) [X2 Z3 X4]
+ (0.12507032579771862) [Y1 Z2 Y3]
+ (0.12507032579771862) [X1 Z2 X3]
+ (0.12507032579771865) [Y0 Z1 Y2]
+ (0.12507032579771865) [X0 Z1 X2]
+ (-0.03831467029480392) [Y4 Y5 X12 X13]
+ (-0.03831467029480392) [X4 X5 Y12 Y13]
+ (-0.03619412355904267) [Y2 Y3 X8 X9]
+ (-0.03619412355904267) [X2 X3 Y8 Y9]
+ (-0.03583956795335346) [Y2 Y3 X4 X5]
+ (-0.03583956795335346) [X2 X3 Y4 Y5]
+ (-0.031143817988967256) [Y2 Y3 X6 X7]
+ (-0.031143817988967256) [X2 X3 Y6 Y7]
+ (-0.028685183716105903) [Y10 Y11 X12 X13]
+ (-0.028685183716105903) [X10 X11 Y12 Y13]
+ (-0.025996177598021183) [Y3 Z4 Z5 Y7]
+ (-0.025996177598021183) [X3 Z4 Z5 X7]
+ (-0.025384657508457316) [Y2 Y3 X10 X11]
+ (-0.025384657508457316) [X2 X3 Y10 Y11]
+ (-0.019028242443847238) [Y3 Y4 X11 X12]
+ (-0.019028242443847238) [X3 X4 Y11 Y12]
+ (-0.01782514099578658) [Y6 Y7 X10 X11]
+ (-0.01782514099578658) [X6 X7 Y10 Y11]
+ (-0.017680067952481556) [Y4 Y5 X10 X11]
+ (-0.017680067952481556) [X4 X5 Y10 Y11]
+ (-0.017366118994651427) [Y6 Y7 X12 X13]
+ (-0.017366118994651427) [X6 X7 Y12 Y13]
+ (-0.015577208063976486) [Y2 Y3 X12 X13]
+ (-0.015577208063976486) [X2 X3 Y12 Y13]
+ (-0.01458364890761266) [Y0 Y1 X2 X3]
+ (-0.01458364890761266) [X0 X1 Y2 Y3]
+ (-0.013873381748426077) [Y6 Y7 X8 X9]
+ (-0.013873381748426077) [X6 X7 Y8 Y9]
+ (-0.011982389010248) [Y4 Y5 X6 X7]
+ (-0.011982389010248) [X4 X5 Y6 Y7]
+ (-0.011285190200840957) [Y5 X6 X11 Y12]
+ (-0.011285190200840957) [X5 Y6 Y11 X12]
+ (-0.00956070572913593) [Y8 Y9 X10 X11]
+ (-0.00956070572913593) [X8 X9 Y10 Y11]
+ (-0.008125251921381034) [Y1 X2 X8 Y9]
+ (-0.008125251921381034) [Y1 Y2 Y8 Y9]
+ (-0.008125251921381034) [X1 X2 X8 X9]
+ (-0.008125251921381034) [X1 Y2 Y8 X9]
+ (-0.0077314252507752635) [Y0 Y1 X10 X11]
+ (-0.0077314252507752635) [X0 X1 Y10 Y11]
+ (-0.007156934919856943) [Y4 Y5 X8 X9]
+ (-0.007156934919856943) [X4 X5 Y8 Y9]
+ (-0.006888194352970549) [Y0 Y1 X6 X7]
+ (-0.006888194352970549) [X0 X1 Y6 Y7]
+ (-0.006509361201177234) [Y0 Y1 X8 X9]
+ (-0.006509361201177234) [X0 X1 Y8 Y9]
+ (-0.006087822480561864) [Y8 Y9 X12 X13]
+ (-0.006087822480561864) [X8 X9 Y12 Y13]
+ (-0.005283776488402962) [Y0 Y1 X12 X13]
+ (-0.005283776488402962) [X0 X1 Y12 Y13]
+ (-0.005143391768825176) [Y3 X4 X5 Y6]
+ (-0.005143391768825176) [X3 Y4 Y5 X6]
+ (-0.004684903388155222) [Y1 X2 X6 Y7]
+ (-0.004684903388155222) [Y1 Y2 Y6 Y7]
+ (-0.004684903388155222) [X1 X2 X6 X7]
+ (-0.004684903388155222) [X1 Y2 Y6 X7]
+ (-0.004575007626639218) [Y1 X2 X12 Y13]
+ (-0.004575007626639218) [Y1 Y2 Y12 Y13]
+ (-0.004575007626639218) [X1 X2 X12 X13]
+ (-0.004575007626639218) [X1 Y2 Y12 X13]
+ (-0.004424855449441856) [Y1 X2 X4 Y5]
+ (-0.004424855449441856) [Y1 Y2 Y4 Y5]
+ (-0.004424855449441856) [X1 X2 X4 X5]
+ (-0.004424855449441856) [X1 Y2 Y4 X5]
+ (-0.0034795118903343633) [Y2 Z3 Z5 Y6]
+ (-0.0034795118903343633) [X2 Z3 Z5 X6]
+ (-0.0034795118903343633) [Y3 Z4 Z6 Y7]
+ (-0.0034795118903343633) [X3 Z4 Z6 X7]
+ (-0.0017992194936630318) [Y1 X2 X10 Y11]
+ (-0.0017992194936630318) [Y1 Y2 Y10 Y11]
+ (-0.0017992194936630318) [X1 X2 X10 X11]
+ (-0.0017992194936630318) [X1 Y2 Y10 X11]
+ (-0.0002921986261110223) [Y7 Y8 X9 X10]
+ (-0.0002921986261110223) [X7 X8 Y9 Y10]
+ (-8.194261372196947e-06) [Z10 Y11 Z12 Y13]
+ (-8.194261372196947e-06) [Z10 X11 Z12 X13]
+ (-7.801707500418598e-06) [Y2 Z3 Y4 Z11]
+ (-7.801707500418598e-06) [X2 Z3 X4 Z11]
+ (-7.801707500418598e-06) [Y3 Z4 Y5 Z10]
+ (-7.801707500418598e-06) [X3 Z4 X5 Z10]
+ (-4.643051068418438e-06) [Y3 X4 X10 Y11]
+ (-4.643051068418438e-06) [Y3 Y4 Y10 Y11]
+ (-4.643051068418438e-06) [X3 X4 X10 X11]
+ (-4.643051068418438e-06) [X3 Y4 Y10 X11]
+ (-4.5888551556194e-06) [Y4 Z5 Y6 Z13]
+ (-4.5888551556194e-06) [X4 Z5 X6 Z13]
+ (-4.5888551556194e-06) [Y5 Z6 Y7 Z12]
+ (-4.5888551556194e-06) [X5 Z6 X7 Z12]
+ (-4.556569218081705e-06) [Y5 X6 X12 Y13]
+ (-4.556569218081705e-06) [Y5 Y6 Y12 Y13]
+ (-4.556569218081705e-06) [X5 X6 X12 X13]
+ (-4.556569218081705e-06) [X5 Y6 Y12 X13]
+ (-3.6945132943914792e-06) [Y4 X5 X11 Y12]
+ (-3.6945132943914792e-06) [Y4 Y5 Y11 Y12]
+ (-3.6945132943914792e-06) [X4 X5 X11 X12]
+ (-3.6945132943914792e-06) [X4 Y5 Y11 X12]
+ (-3.344081556518191e-06) [Z0 Y5 Z6 Y7]
+ (-3.344081556518191e-06) [Z0 X5 Z6 X7]
+ (-3.344081556518191e-06) [Z1 Y4 Z5 Y6]
+ (-3.344081556518191e-06) [Z1 X4 Z5 X6]
+ (-3.1586564320001587e-06) [Y2 Z3 Y4 Z10]
+ (-3.1586564320001587e-06) [X2 Z3 X4 Z10]
+ (-3.1586564320001587e-06) [Y3 Z4 Y5 Z11]
+ (-3.1586564320001587e-06) [X3 Z4 X5 Z11]
+ (-3.09934924363716e-06) [Z0 Y4 Z5 Y6]
+ (-3.09934924363716e-06) [Z0 X4 Z5 X6]
+ (-3.09934924363716e-06) [Z1 Y5 Z6 Y7]
+ (-3.09934924363716e-06) [Z1 X5 Z6 X7]
+ (-2.8909678817554113e-06) [Z6 Y11 Z12 Y13]
+ (-2.8909678817554113e-06) [Z6 X11 Z12 X13]
+ (-2.8909678817554113e-06) [Z7 Y10 Z11 Y12]
+ (-2.8909678817554113e-06) [Z7 X10 Z11 X12]
+ (-2.177664605091903e-06) [Z0 Y10 Z11 Y12]
+ (-2.177664605091903e-06) [Z0 X10 Z11 X12]
+ (-2.177664605091903e-06) [Z1 Y11 Z12 Y13]
+ (-2.177664605091903e-06) [Z1 X11 Z12 X13]
+ (-1.88185018321663e-06) [Y4 Z5 Y6 Z9]
+ (-1.88185018321663e-06) [X4 Z5 X6 Z9]
+ (-1.88185018321663e-06) [Y5 Z6 Y7 Z8]
+ (-1.88185018321663e-06) [X5 Z6 X7 Z8]
+ (-1.8551201215636207e-06) [Z6 Y10 Z11 Y12]
+ (-1.8551201215636207e-06) [Z6 X10 Z11 X12]
+ (-1.8551201215636207e-06) [Z7 Y11 Z12 Y13]
+ (-1.8551201215636207e-06) [Z7 X11 Z12 X13]
+ (-1.8540608579549811e-06) [Y4 Z5 Y6 Z7]
+ (-1.8540608579549811e-06) [X4 Z5 X6 Z7]
+ (-1.8163031697426544e-06) [Z4 Y11 Z12 Y13]
+ (-1.8163031697426544e-06) [Z4 X11 Z12 X13]
+ (-1.8163031697426544e-06) [Z5 Y10 Z11 Y12]
+ (-1.8163031697426544e-06) [Z5 X10 Z11 X12]
+ (-1.6923978285386188e-06) [Y4 Z5 Y6 Z10]
+ (-1.6923978285386188e-06) [X4 Z5 X6 Z10]
+ (-1.6923978285386188e-06) [Y5 Z6 Y7 Z11]
+ (-1.6923978285386188e-06) [X5 Z6 X7 Z11]
+ (-1.6148794139672165e-06) [Z0 Y11 Z12 Y13]
+ (-1.6148794139672165e-06) [Z0 X11 Z12 X13]
+ (-1.6148794139672165e-06) [Z1 Y10 Z11 Y12]
+ (-1.6148794139672165e-06) [Z1 X10 Z11 X12]
+ (-1.597317197859398e-06) [Z8 Y10 Z11 Y12]
+ (-1.597317197859398e-06) [Z8 X10 Z11 X12]
+ (-1.597317197859398e-06) [Z9 Y11 Z12 Y13]
+ (-1.597317197859398e-06) [Z9 X11 Z12 X13]
+ (-1.454842449095295e-06) [Y3 X4 X6 Y7]
+ (-1.454842449095295e-06) [Y3 Y4 Y6 Y7]
+ (-1.454842449095295e-06) [X3 X4 X6 X7]
+ (-1.454842449095295e-06) [X3 Y4 Y6 X7]
+ (-1.3980449081247869e-06) [Y4 Z5 Y6 Z8]
+ (-1.3980449081247869e-06) [X4 Z5 X6 Z8]
+ (-1.3980449081247869e-06) [Y5 Z6 Y7 Z9]
+ (-1.3980449081247869e-06) [X5 Z6 X7 Z9]
+ (-1.1954890100102957e-06) [Y2 Z3 Y4 Z7]
+ (-1.1954890100102957e-06) [X2 Z3 X4 Z7]
+ (-1.1954890100102957e-06) [Y3 Z4 Y5 Z6]
+ (-1.1954890100102957e-06) [X3 Z4 X5 Z6]
+ (-1.1908508083533097e-06) [Z0 Y3 Z4 Y5]
+ (-1.1908508083533097e-06) [Z0 X3 Z4 X5]
+ (-1.1908508083533097e-06) [Z1 Y2 Z3 Y4]
+ (-1.1908508083533097e-06) [Z1 X2 Z3 X4]
+ (-1.1708301369938986e-06) [Z2 Y5 Z6 Y7]
+ (-1.1708301369938986e-06) [Z2 X5 Z6 X7]
+ (-1.1708301369938986e-06) [Z3 Y4 Z5 Y6]
+ (-1.1708301369938986e-06) [Z3 X4 Z5 X6]
+ (-1.063228342410461e-06) [Z2 Y10 Z11 Y12]
+ (-1.063228342410461e-06) [Z2 X10 Z11 X12]
+ (-1.063228342410461e-06) [Z3 Y11 Z12 Y13]
+ (-1.063228342410461e-06) [Z3 X11 Z12 X13]
+ (-1.0358477601917908e-06) [Y6 X7 X11 Y12]
+ (-1.0358477601917908e-06) [Y6 Y7 Y11 Y12]
+ (-1.0358477601917908e-06) [X6 X7 X11 X12]
+ (-1.0358477601917908e-06) [X6 Y7 Y11 X12]
+ (-9.509249751710963e-07) [Z2 Y4 Z5 Y6]
+ (-9.509249751710963e-07) [Z2 X4 Z5 X6]
+ (-9.509249751710963e-07) [Z3 Y5 Z6 Y7]
+ (-9.509249751710963e-07) [Z3 X5 Z6 X7]
+ (-9.344557777139389e-07) [Z8 Y11 Z12 Y13]
+ (-9.344557777139389e-07) [Z8 X11 Z12 X13]
+ (-9.344557777139389e-07) [Z9 Y10 Z11 Y12]
+ (-9.344557777139389e-07) [Z9 X10 Z11 X12]
+ (-8.337746754491219e-07) [Z0 Y2 Z3 Y4]
+ (-8.337746754491219e-07) [Z0 X2 Z3 X4]
+ (-8.337746754491219e-07) [Z1 Y3 Z4 Y5]
+ (-8.337746754491219e-07) [Z1 X3 Z4 X5]
+ (-7.956895372492603e-07) [Y3 X4 X8 Y9]
+ (-7.956895372492603e-07) [Y3 Y4 Y8 Y9]
+ (-7.956895372492603e-07) [X3 X4 X8 X9]
+ (-7.956895372492603e-07) [X3 Y4 Y8 X9]
+ (-7.764994118434257e-07) [Y2 Z3 Y4 Z5]
+ (-7.764994118434257e-07) [X2 Z3 X4 Z5]
+ (-5.929765815774793e-07) [Z4 Y5 Z6 Y7]
+ (-5.929765815774793e-07) [Z4 X5 Z6 X7]
+ (-5.770052995124601e-07) [Y2 Z3 Y4 Z9]
+ (-5.770052995124601e-07) [X2 Z3 X4 Z9]
+ (-5.770052995124601e-07) [Y3 Z4 Y5 Z8]
+ (-5.770052995124601e-07) [X3 Z4 X5 Z8]
+ (-5.47164774458336e-07) [Y1 Y2 X11 X12]
+ (-5.47164774458336e-07) [X1 X2 Y11 Y12]
+ (-4.838052750918428e-07) [Y5 X6 X8 Y9]
+ (-4.838052750918428e-07) [Y5 Y6 Y8 Y9]
+ (-4.838052750918428e-07) [X5 X6 X8 X9]
+ (-4.838052750918428e-07) [X5 Y6 Y8 X9]
+ (-3.570761329041879e-07) [Y0 X1 X3 Y4]
+ (-3.570761329041879e-07) [Y0 Y1 Y3 Y4]
+ (-3.570761329041879e-07) [X0 X1 X3 X4]
+ (-3.570761329041879e-07) [X0 Y1 Y3 X4]
+ (-2.447323128810316e-07) [Y0 X1 X5 Y6]
+ (-2.447323128810316e-07) [Y0 Y1 Y5 Y6]
+ (-2.447323128810316e-07) [X0 X1 X5 X6]
+ (-2.447323128810316e-07) [X0 Y1 Y5 X6]
+ (-2.1990516182280228e-07) [Y2 X3 X5 Y6]
+ (-2.1990516182280228e-07) [Y2 Y3 Y5 Y6]
+ (-2.1990516182280228e-07) [X2 X3 X5 X6]
+ (-2.1990516182280228e-07) [X2 Y3 Y5 X6]
+ (-1.9332412770879638e-07) [Y1 X2 X3 Y4]
+ (-1.9332412770879638e-07) [X1 Y2 Y3 X4]
+ (-1.2919694862691583e-07) [Y1 Z2 Z3 Y5]
+ (-1.2919694862691583e-07) [X1 Z2 Z3 X5]
+ (1.7379332623380152e-07) [Y0 Z1 Z3 Y4]
+ (1.7379332623380152e-07) [X0 Z1 Z3 X4]
+ (1.7379332623380152e-07) [Y1 Z2 Z4 Y5]
+ (1.7379332623380152e-07) [X1 Z2 Z4 X5]
+ (1.9332412770879638e-07) [Y1 Y2 X3 X4]
+ (1.9332412770879638e-07) [X1 X2 Y3 Y4]
+ (2.1868423773680033e-07) [Y2 Z3 Y4 Z8]
+ (2.1868423773680033e-07) [X2 Z3 X4 Z8]
+ (2.1868423773680033e-07) [Y3 Z4 Y5 Z9]
+ (2.1868423773680033e-07) [X3 Z4 X5 Z9]
+ (2.5935343908499963e-07) [Y2 Z3 Y4 Z6]
+ (2.5935343908499963e-07) [X2 Z3 X4 Z6]
+ (2.5935343908499963e-07) [Y3 Z4 Y5 Z7]
+ (2.5935343908499963e-07) [X3 Z4 X5 Z7]
+ (3.606071867841729e-07) [Y0 Z1 Z2 Y4]
+ (3.606071867841729e-07) [X0 Z1 Z2 X4]
+ (3.606071867841729e-07) [Y1 Z3 Z4 Y5]
+ (3.606071867841729e-07) [X1 Z3 Z4 X5]
+ (5.47164774458336e-07) [Y1 X2 X11 Y12]
+ (5.47164774458336e-07) [X1 Y2 Y11 X12]
+ (5.627851911246863e-07) [Y0 X1 X11 Y12]
+ (5.627851911246863e-07) [Y0 Y1 Y11 Y12]
+ (5.627851911246863e-07) [X0 X1 X11 X12]
+ (5.627851911246863e-07) [X0 Y1 Y11 X12]
+ (6.628614201454592e-07) [Y8 X9 X11 Y12]
+ (6.628614201454592e-07) [Y8 Y9 Y11 Y12]
+ (6.628614201454592e-07) [X8 X9 X11 X12]
+ (6.628614201454592e-07) [X8 Y9 Y11 X12]
+ (1.1094407590191028e-06) [Z2 Y11 Z12 Y13]
+ (1.1094407590191028e-06) [Z2 X11 Z12 X13]
+ (1.1094407590191028e-06) [Z3 Y10 Z11 Y12]
+ (1.1094407590191028e-06) [Z3 X10 Z11 X12]
+ (1.6021167406021185e-06) [Z2 Y3 Z4 Y5]
+ (1.6021167406021185e-06) [Z2 X3 Z4 X5]
+ (1.8782101246488253e-06) [Z4 Y10 Z11 Y12]
+ (1.8782101246488253e-06) [Z4 X10 Z11 X12]
+ (1.8782101246488253e-06) [Z5 Y11 Z12 Y13]
+ (1.8782101246488253e-06) [Z5 X11 Z12 X13]
+ (2.172669101429564e-06) [Y2 X3 X11 Y12]
+ (2.172669101429564e-06) [Y2 Y3 Y11 Y12]
+ (2.172669101429564e-06) [X2 X3 X11 X12]
+ (2.172669101429564e-06) [X2 Y3 Y11 X12]
+ (3.117447946025858e-06) [Y0 Z2 Z3 Y4]
+ (3.117447946025858e-06) [X0 Z2 Z3 X4]
+ (3.5390541844756488e-06) [Y2 Z3 Y4 Z12]
+ (3.5390541844756488e-06) [X2 Z3 X4 Z12]
+ (3.5390541844756488e-06) [Y3 Z4 Y5 Z13]
+ (3.5390541844756488e-06) [X3 Z4 X5 Z13]
+ (4.281913884827355e-06) [Y4 Z5 Y6 Z11]
+ (4.281913884827355e-06) [X4 Z5 X6 Z11]
+ (4.281913884827355e-06) [Y5 Z6 Y7 Z10]
+ (4.281913884827355e-06) [X5 Z6 X7 Z10]
+ (5.275883122026603e-06) [Y3 X4 X12 Y13]
+ (5.275883122026603e-06) [Y3 Y4 Y12 Y13]
+ (5.275883122026603e-06) [X3 X4 X12 X13]
+ (5.275883122026603e-06) [X3 Y4 Y12 X13]
+ (5.9743117133659734e-06) [Y5 X6 X10 Y11]
+ (5.9743117133659734e-06) [Y5 Y6 Y10 Y11]
+ (5.9743117133659734e-06) [X5 X6 X10 X11]
+ (5.9743117133659734e-06) [X5 Y6 Y10 X11]
+ (7.954413176110595e-06) [Y10 Z11 Y12 Z13]
+ (7.954413176110595e-06) [X10 Z11 X12 Z13]
+ (8.814937306502252e-06) [Y2 Z3 Y4 Z13]
+ (8.814937306502252e-06) [X2 Z3 X4 Z13]
+ (8.814937306502252e-06) [Y3 Z4 Y5 Z12]
+ (8.814937306502252e-06) [X3 Z4 X5 Z12]
+ (0.0002921986261110223) [Y7 X8 X9 Y10]
+ (0.0002921986261110223) [X7 Y8 Y9 X10]
+ (0.0004956762314915395) [Y2 Z4 Z5 Y6]
+ (0.0004956762314915395) [X2 Z4 Z5 X6]
+ (0.0011059037691896582) [Y0 Z1 Y2 Z5]
+ (0.0011059037691896582) [X0 Z1 X2 Z5]
+ (0.0011059037691896582) [Y1 Z2 Y3 Z4]
+ (0.0011059037691896582) [X1 Z2 X3 Z4]
+ (0.001663879878490812) [Y2 Z3 Z4 Y6]
+ (0.001663879878490812) [X2 Z3 Z4 X6]
+ (0.001663879878490812) [Y3 Z5 Z6 Y7]
+ (0.001663879878490812) [X3 Z5 Z6 X7]
+ (0.001756070701841216) [Y0 Z1 Y2 Z11]
+ (0.001756070701841216) [X0 Z1 X2 Z11]
+ (0.001756070701841216) [Y1 Z2 Y3 Z10]
+ (0.001756070701841216) [X1 Z2 X3 Z10]
+ (0.002326230623158049) [Y0 Z1 Y2 Z13]
+ (0.002326230623158049) [X0 Z1 X2 Z13]
+ (0.002326230623158049) [Y1 Z2 Y3 Z12]
+ (0.002326230623158049) [X1 Z2 X3 Z12]
+ (0.0029297686747510173) [Y0 Z1 Y2 Z9]
+ (0.0029297686747510173) [X0 Z1 X2 Z9]
+ (0.0029297686747510173) [Y1 Z2 Y3 Z8]
+ (0.0029297686747510173) [X1 Z2 X3 Z8]
+ (0.003276971931231638) [Y0 Z1 Y2 Z3]
+ (0.003276971931231638) [X0 Z1 X2 Z3]
+ (0.003347617530666143) [Y0 Z1 Y2 Z7]
+ (0.003347617530666143) [X0 Z1 X2 Z7]
+ (0.003347617530666143) [Y1 Z2 Y3 Z6]
+ (0.003347617530666143) [X1 Z2 X3 Z6]
+ (0.003555290195504248) [Y0 Z1 Y2 Z10]
+ (0.003555290195504248) [X0 Z1 X2 Z10]
+ (0.003555290195504248) [Y1 Z2 Y3 Z11]
+ (0.003555290195504248) [X1 Z2 X3 Z11]
+ (0.005143391768825176) [Y3 Y4 X5 X6]
+ (0.005143391768825176) [X3 X4 Y5 Y6]
+ (0.005283776488402962) [Y0 X1 X12 Y13]
+ (0.005283776488402962) [X0 Y1 Y12 X13]
+ (0.005530759218631514) [Y0 Z1 Y2 Z4]
+ (0.005530759218631514) [X0 Z1 X2 Z4]
+ (0.005530759218631514) [Y1 Z2 Y3 Z5]
+ (0.005530759218631514) [X1 Z2 X3 Z5]
+ (0.006087822480561864) [Y8 X9 X12 Y13]
+ (0.006087822480561864) [X8 Y9 Y12 X13]
+ (0.006509361201177234) [Y0 X1 X8 Y9]
+ (0.006509361201177234) [X0 Y1 Y8 X9]
+ (0.006888194352970549) [Y0 X1 X6 Y7]
+ (0.006888194352970549) [X0 Y1 Y6 X7]
+ (0.006901238249797265) [Y0 Z1 Y2 Z12]
+ (0.006901238249797265) [X0 Z1 X2 Z12]
+ (0.006901238249797265) [Y1 Z2 Y3 Z13]
+ (0.006901238249797265) [X1 Z2 X3 Z13]
+ (0.007156934919856943) [Y4 X5 X8 Y9]
+ (0.007156934919856943) [X4 Y5 Y8 X9]
+ (0.0077314252507752635) [Y0 X1 X10 Y11]
+ (0.0077314252507752635) [X0 Y1 Y10 X11]
+ (0.008032520918821364) [Y0 Z1 Y2 Z6]
+ (0.008032520918821364) [X0 Z1 X2 Z6]
+ (0.008032520918821364) [Y1 Z2 Y3 Z7]
+ (0.008032520918821364) [X1 Z2 X3 Z7]
+ (0.00956070572913593) [Y8 X9 X10 Y11]
+ (0.00956070572913593) [X8 Y9 Y10 X11]
+ (0.011055020596132052) [Y0 Z1 Y2 Z8]
+ (0.011055020596132052) [X0 Z1 X2 Z8]
+ (0.011055020596132052) [Y1 Z2 Y3 Z9]
+ (0.011055020596132052) [X1 Z2 X3 Z9]
+ (0.011285190200840957) [Y5 Y6 X11 X12]
+ (0.011285190200840957) [X5 X6 Y11 Y12]
+ (0.011307274008848234) [Y7 Z8 Z9 Y11]
+ (0.011307274008848234) [X7 Z8 Z9 X11]
+ (0.011982389010248) [Y4 X5 X6 Y7]
+ (0.011982389010248) [X4 Y5 Y6 X7]
+ (0.013873381748426077) [Y6 X7 X8 Y9]
+ (0.013873381748426077) [X6 Y7 Y8 X9]
+ (0.01458364890761266) [Y0 X1 X2 Y3]
+ (0.01458364890761266) [X0 Y1 Y2 X3]
+ (0.015577208063976486) [Y2 X3 X12 Y13]
+ (0.015577208063976486) [X2 Y3 Y12 X13]
+ (0.017366118994651427) [Y6 X7 X12 Y13]
+ (0.017366118994651427) [X6 Y7 Y12 X13]
+ (0.017680067952481556) [Y4 X5 X10 Y11]
+ (0.017680067952481556) [X4 Y5 Y10 X11]
+ (0.01782514099578658) [Y6 X7 X10 Y11]
+ (0.01782514099578658) [X6 Y7 Y10 X11]
+ (0.019028242443847238) [Y3 X4 X11 Y12]
+ (0.019028242443847238) [X3 Y4 Y11 X12]
+ (0.025384657508457316) [Y2 X3 X10 Y11]
+ (0.025384657508457316) [X2 Y3 Y10 X11]
+ (0.028685183716105903) [Y10 X11 X12 Y13]
+ (0.028685183716105903) [X10 Y11 Y12 X13]
+ (0.029812424517345892) [Y6 Z7 Z8 Y10]
+ (0.029812424517345892) [X6 Z7 Z8 X10]
+ (0.029812424517345892) [Y7 Z9 Z10 Y11]
+ (0.029812424517345892) [X7 Z9 Z10 X11]
+ (0.030104623143456914) [Y6 Z7 Z9 Y10]
+ (0.030104623143456914) [X6 Z7 Z9 X10]
+ (0.030104623143456914) [Y7 Z8 Z10 Y11]
+ (0.030104623143456914) [X7 Z8 Z10 X11]
+ (0.03078750538914403) [Y6 Z8 Z9 Y10]
+ (0.03078750538914403) [X6 Z8 Z9 X10]
+ (0.031143817988967256) [Y2 X3 X6 Y7]
+ (0.031143817988967256) [X2 Y3 Y6 X7]
+ (0.03583956795335346) [Y2 X3 X4 Y5]
+ (0.03583956795335346) [X2 Y3 Y4 X5]
+ (0.03619412355904267) [Y2 X3 X8 Y9]
+ (0.03619412355904267) [X2 Y3 Y8 X9]
+ (0.03831467029480392) [Y4 X5 X12 Y13]
+ (0.03831467029480392) [X4 Y5 Y12 X13]
+ (0.10433064780651397) [Z0 Y1 Z2 Y3]
+ (0.10433064780651397) [Z0 X1 Z2 X3]
+ (-0.12133276911042279) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133276911042279) [X2 Z3 Z4 Z5 X6]
+ (-0.12133276911042265) [Y3 Z4 Z5 Z6 Y7]
+ (-0.12133276911042265) [X3 Z4 Z5 Z6 X7]
+ (3.2020768795701924e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.2020768795701924e-06) [X0 Z1 Z2 Z3 X4]
+ (3.202076879570193e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.202076879570193e-06) [X1 Z2 Z3 Z4 X5]
+ (0.22848106564918946) [Y7 Z8 Z9 Z10 Y11]
+ (0.22848106564918946) [X7 Z8 Z9 Z10 X11]
+ (0.2284810656491895) [Y6 Z7 Z8 Z9 Y10]
+ (0.2284810656491895) [X6 Z7 Z8 Z9 X10]
+ (-0.032767657823290525) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.032767657823290525) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.032767657823290525) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.032767657823290525) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845273194) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.027115036845273194) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845273194) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.027115036845273194) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.025996177598021183) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.025996177598021183) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.01756120240964618) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.01756120240964618) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.01756120240964618) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.01756120240964618) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.014564531231173022) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.014564531231173022) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.014564531231173022) [X7 Z8 Z9 X10 X12 X13]
+ (-0.014564531231173022) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.01221504099761402) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.01221504099761402) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.01221504099761402) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.01221504099761402) [X4 Z5 X6 X11 Z12 X13]
+ (-0.01221504099761402) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.01221504099761402) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.01221504099761402) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.01221504099761402) [X5 Z6 X7 X10 Z11 X12]
+ (-0.011756013419819227) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.011756013419819227) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.011756013419819227) [X3 Z4 Z5 X6 X8 X9]
+ (-0.011756013419819227) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.008764827575688734) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.008764827575688734) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.008764827575688734) [X2 Z3 Z4 X5 X11 X12]
+ (-0.008764827575688734) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.008764827575688734) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.008764827575688734) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.008764827575688734) [X3 X4 X10 Z11 Z12 X13]
+ (-0.008764827575688734) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.008125251921381034) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.008125251921381034) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.00730675992883301) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.00730675992883301) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.00730675992883301) [X4 X5 X7 Z8 Z9 X10]
+ (-0.00730675992883301) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.005805188989826956) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.005805188989826956) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.005805188989826956) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.005805188989826956) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.005652620978017331) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.005652620978017331) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.005652620978017331) [X0 X1 X3 Z4 Z5 X6]
+ (-0.005652620978017331) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.005143391768825176) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.005143391768825176) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.005143391768825176) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.005143391768825176) [X2 Z3 X4 X5 Z6 X7]
+ (-0.004684903388155222) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.004684903388155222) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668620318776295) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.004668620318776295) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.004575007626639217) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.004575007626639217) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.004424855449441856) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.004424855449441856) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.00415879738184008) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.00415879738184008) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.00415879738184008) [X3 Z4 Z5 X6 X12 X13]
+ (-0.00415879738184008) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.0034937903598901685) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.0034937903598901685) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.0034937903598901685) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.0034937903598901685) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.002779026799025538) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.002779026799025538) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.002293956611352458) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.002293956611352458) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.0017992194936630316) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.0017992194936630316) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.0017278753941369536) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.0017278753941369536) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.0009298507967730621) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298507967730621) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298507967730621) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298507967730621) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533856254125401) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.0008533856254125401) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.0008145313270956694) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.0008145313270956694) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.0008145313270956694) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.0008145313270956694) [X3 Z4 Z5 Z6 X7 Z11]
+ (-7.735036880588172e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.735036880588172e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.735036880588172e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.735036880588172e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-8.774817864466953e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.774817864466953e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.774817864466953e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.774817864466953e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.518362215589984e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.518362215589984e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.518362215589984e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.518362215589984e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.444344675814849e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.444344675814849e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.444344675814849e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.444344675814849e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.524373848473977e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.524373848473977e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.524373848473977e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.524373848473977e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-6.290028433073248e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.290028433073248e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.290028433073248e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.290028433073248e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.9743117133659734e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (-5.9743117133659734e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (-5.275883122026603e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.275883122026603e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.643051068418438e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.643051068418438e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.556569218081705e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (-4.556569218081705e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (-4.25322422554244e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.25322422554244e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-3.7696594518838442e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.7696594518838442e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.694513294391479e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.694513294391479e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.610297130465756e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.610297130465756e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.610297130465756e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.610297130465756e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.313145500129672e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.313145500129672e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.2774831954067762e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.2774831954067762e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.2774831954067762e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.2774831954067762e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.211228348344305e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.211228348344305e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.211228348344305e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.211228348344305e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.151346311141735e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.151346311141735e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.0882507112194774e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.0882507112194774e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.172669101429564e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.172669101429564e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.454842449095295e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.454842449095295e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.330473188652104e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.330473188652104e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.2283337825167362e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.2283337825167362e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.0358477601917908e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.0358477601917908e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-7.956895372492603e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.956895372492603e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.733197742244438e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.733197742244438e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.733197742244438e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.733197742244438e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.628614201454592e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.628614201454592e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.55628191453065e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.55628191453065e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.55628191453065e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.55628191453065e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.4182915745273e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.4182915745273e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.4182915745273e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.4182915745273e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.92745308271015e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.92745308271015e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.92745308271015e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.92745308271015e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.627851911246863e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.627851911246863e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.287660624584057e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.287660624584057e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.287660624584057e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.287660624584057e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.287660624584057e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.287660624584057e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.287660624584057e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.287660624584057e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.838052750918428e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (-4.838052750918428e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (-3.570761329041879e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.570761329041879e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.3281393505897965e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.3281393505897965e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.0868265651379423e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.0868265651379423e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.0868265651379423e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.0868265651379423e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.447323128810316e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (-2.447323128810316e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (-2.3713289479011296e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.3713289479011296e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.3713289479011296e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.3713289479011296e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.1990516182280228e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-2.1990516182280228e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-1.9332412770879638e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.9332412770879638e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.9332412770879638e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.9332412770879638e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.8394209154939955e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.8394209154939955e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.8394209154939955e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.8394209154939955e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.5510539176649878e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (-1.5510539176649878e-07) [X0 Z1 X2 X4 Z5 X6]
+ (-1.5510539176649878e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (-1.5510539176649878e-07) [X1 Z2 X3 X5 Z6 X7]
+ (-1.3807781480650686e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (-1.3807781480650686e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (-1.3807781480650686e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (-1.3807781480650686e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (-1.3807781480650686e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (-1.3807781480650686e-07) [X0 Z1 X2 X5 Z6 X7]
+ (-1.3807781480650686e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (-1.3807781480650686e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (-1.3807781480650686e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (-1.3807781480650686e-07) [X1 Z2 X3 X4 Z5 X6]
+ (-1.3807781480650686e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (-1.3807781480650686e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (-1.2919694862691583e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.2919694862691583e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.107632559945267e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.107632559945267e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.107632559945267e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.107632559945267e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.107632559945267e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.107632559945267e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.107632559945267e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.107632559945267e-07) [X1 Z2 X3 X10 Z11 X12]
+ (8.057446595342888e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.057446595342888e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.057446595342888e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.057446595342888e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.649310133660303e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.649310133660303e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.649310133660303e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.649310133660303e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.8394209154939955e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.8394209154939955e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.8394209154939955e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.8394209154939955e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.1990516182280228e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (2.1990516182280228e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (2.447323128810316e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (2.447323128810316e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.2362599612671594e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.2362599612671594e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.2362599612671594e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.2362599612671594e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.3281393505897965e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.3281393505897965e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.570761329041879e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.570761329041879e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.838052750918428e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (4.838052750918428e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (5.627851911246863e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.627851911246863e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (6.628614201454592e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.628614201454592e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (7.956895372492603e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.956895372492603e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.306536651702712e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.306536651702712e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.306536651702712e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.306536651702712e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.0358477601917908e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.0358477601917908e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.2283337825167362e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.2283337825167362e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.2393363216840654e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.2393363216840654e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.2393363216840654e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.2393363216840654e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.330473188652104e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.330473188652104e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.454842449095295e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.454842449095295e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.172669101429564e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.172669101429564e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.0882507112194774e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.0882507112194774e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.117447946025858e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.117447946025858e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.151346311141735e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.151346311141735e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.313145500129672e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.313145500129672e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.3343312893981062e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.3343312893981062e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.694513294391479e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.694513294391479e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (4.183932559375859e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.183932559375859e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.556569218081705e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (4.556569218081705e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (4.643051068418438e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.643051068418438e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.275883122026603e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.275883122026603e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.9743117133659734e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (5.9743117133659734e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (0.0002921986261110223) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.0002921986261110223) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.0002921986261110223) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.0002921986261110223) [X6 Z7 X8 X9 Z10 X11]
+ (0.0004956762314915395) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.0004956762314915395) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.0006650070219499117) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.0006650070219499117) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.0006650070219499117) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.0006650070219499117) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.0008533856254125401) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.0008533856254125401) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.0016095313817213743) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.0016095313817213743) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.0016095313817213743) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.0016095313817213743) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.0016676041811440477) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.0016676041811440477) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.0016676041811440477) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.0016676041811440477) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.0017278753941369536) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.0017278753941369536) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.0017992194936630316) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.0017992194936630316) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.002293956611352458) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.002293956611352458) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.0024629170071339144) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.0024629170071339144) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.0024629170071339144) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.0024629170071339144) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.003961560792496506) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.003961560792496506) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.003961560792496506) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.003961560792496506) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.004424855449441856) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.004424855449441856) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.004575007626639217) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.004575007626639217) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.004668620318776295) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.004668620318776295) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.004684903388155222) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.004684903388155222) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.0053248352342216984) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.0053248352342216984) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.0053248352342216984) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.0053248352342216984) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.005368659358109599) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.005368659358109599) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.005368659358109599) [X2 X3 X7 Z8 Z9 X10]
+ (0.005368659358109599) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.00796088072592157) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.00796088072592157) [X4 Z5 X6 X10 Z11 X12]
+ (0.00796088072592157) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.00796088072592157) [X5 Z6 X7 X11 Z12 X13]
+ (0.008125251921381034) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.008125251921381034) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.00889073152269463) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.00889073152269463) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.00889073152269463) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.00889073152269463) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.010263414868158505) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.010263414868158505) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.010263414868158505) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.010263414868158505) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.010540425907671569) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.010540425907671569) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.010540425907671569) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.010540425907671569) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.01096007494054263) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.01096007494054263) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.01096007494054263) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.01096007494054263) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.011307274008848232) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.011307274008848232) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.014411099430130924) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.014411099430130924) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.014411099430130924) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.014411099430130924) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.015225630757226594) [Y3 Z4 Z5 X6 X10 Y11]
+ (0.015225630757226594) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (0.015225630757226594) [X3 Z4 Z5 X6 X10 X11]
+ (0.015225630757226594) [X3 Z4 Z5 Y6 Y10 X11]
+ (0.015588250102380206) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (0.015588250102380206) [X2 Z3 X4 X10 Z11 X12]
+ (0.015588250102380206) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (0.015588250102380206) [X3 Z4 X5 X11 Z12 X13]
+ (0.01826683486937564) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.01826683486937564) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.01826683486937564) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.01826683486937564) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.01902042317304003) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.01902042317304003) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.01902042317304003) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.01902042317304003) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.02017592172353559) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.02017592172353559) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.02017592172353559) [X4 Z5 Z6 X7 X11 X12]
+ (0.02017592172353559) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.02017592172353559) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.02017592172353559) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.02017592172353559) [X5 X6 X10 Z11 Z12 X13]
+ (0.02017592172353559) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.02435307767806894) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.02435307767806894) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.02435307767806894) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.02435307767806894) [X2 Z3 X4 X11 Z12 X13]
+ (0.02435307767806894) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.02435307767806894) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.02435307767806894) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.02435307767806894) [X3 Z4 X5 X10 Z11 X12]
+ (0.024389082531149627) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.024389082531149627) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.024389082531149627) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.024389082531149627) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.025104957138844593) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.025104957138844593) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.025104957138844593) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.025104957138844593) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.03078750538914403) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.03078750538914403) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.045879470781297824) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.045879470781297824) [X0 Z2 Z3 Z4 Z5 X6]
+ (0.056007330877807876) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.056007330877807876) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.056007330877807876) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.056007330877807876) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661375) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.05608468124661375) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661375) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.05608468124661375) [Z1 X7 Z8 Z9 Z10 X11]
+ (-6.631277928337395e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.631277928337395e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.631277928337393e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.631277928337393e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.5950860068615286e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.5950860068615286e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.595086006861528e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.595086006861528e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.04274327701378241) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.04274327701378241) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (0.04274327701378241) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04274327701378241) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.04764261217638313) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.04764261217638313) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.04764261217638313) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.04764261217638313) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.04171881383982178) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.04171881383982178) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.04171881383982178) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.04171881383982178) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.03956441632289332) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.03956441632289332) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.03956441632289332) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03956441632289332) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03935916802205296) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.03935916802205296) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.03935916802205296) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.03935916802205296) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.03931805194719759) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03931805194719759) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.03931805194719759) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03931805194719759) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03560837898831247) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.03560837898831247) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.02990378951262481) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.02990378951262481) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.02990378951262481) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.02990378951262481) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.02873077955190547) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.02873077955190547) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.02873077955190547) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.02873077955190547) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.025637238296026862) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.025637238296026862) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.025637238296026862) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.025637238296026862) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.02475546329289087) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.02475546329289087) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.02475546329289087) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.02475546329289087) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.024282117354692906) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.024282117354692906) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.02314513092952908) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.02314513092952908) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.022528440196012984) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.022528440196012984) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.02143381072160084) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.02143381072160084) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.02143381072160084) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.02143381072160084) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251627) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.019257505095251627) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019028242443847238) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.019028242443847238) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.018889030304942843) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-0.018889030304942843) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-0.018889030304942843) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.018889030304942843) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.016024603689179573) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.016024603689179573) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.015225630757226594) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.015225630757226594) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.014603704729162089) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.014603704729162089) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.014564531231173022) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.014564531231173022) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.011756013419819224) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.011756013419819224) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.011285190200840957) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-0.011285190200840957) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-0.00984174924696263) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.00984174924696263) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.00961263460684729) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.00961263460684729) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.00961263460684729) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.00961263460684729) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.00846997879102397) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.00846997879102397) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.00730675992883301) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.00730675992883301) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0059237983365613475) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.0059237983365613475) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.005652620978017331) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.005652620978017331) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.005368659358109599) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005368659358109599) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.00415879738184008) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.00415879738184008) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0033566705638328758) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.0033566705638328758) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.0033566705638328758) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.0033566705638328758) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.003267513854423547) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.003267513854423547) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.003267513854423547) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.003267513854423547) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.0027790267990255376) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.0027790267990255376) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.0026860409778066015) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.0026860409778066015) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.0026860409778066015) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.0026860409778066015) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.002293956611352458) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.002293956611352458) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.002293956611352458) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.002293956611352458) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.0009581655836696482) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.0009581655836696482) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.0009581655836696482) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.0009581655836696482) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.0009581655836696482) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.0009581655836696482) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.0009581655836696482) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.0009581655836696482) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.0002463643756957322) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0002463643756957322) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303549745) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.00013840177303549745) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.00013840177303549745) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.00013840177303549745) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.735036880588172e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.735036880588172e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.610358530546551e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.610358530546551e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.610358530546551e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.610358530546551e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808795060914e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.5316808795060914e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808795060914e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5316808795060914e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.806102775080644e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.806102775080644e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.806102775080644e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.806102775080644e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.089799467411524e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.089799467411524e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.089799467411524e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.089799467411524e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.652209669241448e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.652209669241448e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.652209669241448e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.652209669241448e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851833700101e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.481851833700101e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851833700101e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.481851833700101e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.071480736448518e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.071480736448518e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.071480736448518e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.071480736448518e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.7346220386321255e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.7346220386321255e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.7346220386321255e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.7346220386321255e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.728843147152603e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.728843147152603e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.728843147152603e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.728843147152603e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.253224225542441e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.253224225542441e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.7696594518838446e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.7696594518838446e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.544395429210305e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.544395429210305e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.544395429210305e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.544395429210305e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.544395429210305e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.544395429210305e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.544395429210305e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.544395429210305e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.3609563202589218e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563202589218e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563202589218e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.3609563202589218e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-2.10321560463351e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.10321560463351e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.10321560463351e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.10321560463351e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220981681073e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.0111220981681073e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220981681073e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.0111220981681073e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9429468365914244e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.9429468365914244e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9429468365914244e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9429468365914244e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174770377543e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.6541174770377543e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174770377543e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.6541174770377543e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.5224930676316185e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.5224930676316185e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.5224930676316185e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.5224930676316185e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.5224930676316185e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930676316185e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930676316185e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.5224930676316185e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.2283337825167362e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337825167362e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.2283337825167362e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337825167362e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.988770288690816e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.988770288690816e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.988770288690816e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.988770288690816e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.867765104045929e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104045929e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765104045929e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.867765104045929e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.18999097511814e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.18999097511814e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.175246206969021e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175246206969021e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.47164774458336e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.47164774458336e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.5614471799082687e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.5614471799082687e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.5614471799082687e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.5614471799082687e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.523389677771276e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.523389677771276e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.427323108782546e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.427323108782546e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.427323108782546e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.427323108782546e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.3281393505897965e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.3281393505897965e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.3281393505897965e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.3281393505897965e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.0868265651379423e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.0868265651379423e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.888293595536702e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.888293595536702e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.888293595536702e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.888293595536702e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.3713289479011296e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.3713289479011296e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.8394209154939955e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.8394209154939955e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-8.057446595342887e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.057446595342887e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (4.537178096643332e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.537178096643332e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.537178096643332e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (4.537178096643332e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (8.057446595342887e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.057446595342887e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (9.209350646540233e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350646540233e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350646540233e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.209350646540233e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783554134726e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783554134726e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783554134726e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783554134726e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.8394209154939955e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.8394209154939955e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.3713289479011296e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.3713289479011296e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (3.0868265651379423e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.0868265651379423e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (4.523389677771276e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.523389677771276e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (5.47164774458336e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.47164774458336e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.175246206969021e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175246206969021e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.18999097511814e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.18999097511814e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.330473188652104e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.330473188652104e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.330473188652104e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.330473188652104e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.6288532435101164e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.6288532435101164e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.6288532435101164e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.6288532435101164e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.6893489514663555e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.6893489514663555e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.6893489514663555e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.6893489514663555e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (2.7455184003412235e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.7455184003412235e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.7455184003412235e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.7455184003412235e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.7455184003412235e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.7455184003412235e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.7455184003412235e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.7455184003412235e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.2118420190979733e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420190979733e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.2118420190979733e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420190979733e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.2118420190979733e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420190979733e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.2118420190979733e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420190979733e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.313145500129672e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.313145500129672e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.313145500129672e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.313145500129672e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (3.3343312893981062e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.3343312893981062e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (4.183932559375859e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.183932559375859e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (7.735036880588172e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.735036880588172e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.0002463643756957322) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.0002463643756957322) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0004458535128840993) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.0004458535128840993) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.0004458535128840993) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.0004458535128840993) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.0005940221543005433) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005433) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005433) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005433) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005433) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005433) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.0005940221543005433) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005433) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.0008533856254125401) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.0008533856254125401) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.0008533856254125401) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.0008533856254125401) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.0010435246534907477) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.0010435246534907477) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.0010435246534907477) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.0010435246534907477) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.0012803060973496647) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.0012803060973496647) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.0012803060973496647) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.0012803060973496647) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.001303800478812696) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (0.001303800478812696) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (0.001303800478812696) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (0.001303800478812696) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (0.0022619660624823446) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (0.0022619660624823446) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (0.0022619660624823446) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (0.0022619660624823446) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (0.0022619660624823446) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (0.0022619660624823446) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (0.0022619660624823446) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (0.0022619660624823446) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (0.003989841456619299) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (0.003989841456619299) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (0.003989841456619299) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (0.003989841456619299) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (0.00415879738184008) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.00415879738184008) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.0043110385079142945) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.0043110385079142945) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.0043110385079142945) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.0043110385079142945) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.00463697666118254) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.00463697666118254) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.00463697666118254) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.00463697666118254) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.005114473831660395) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.005114473831660395) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.005114473831660395) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.005114473831660395) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.005114473831660395) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660395) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660395) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005114473831660395) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.005241535382803857) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (0.005241535382803857) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (0.005241535382803857) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (0.005241535382803857) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (0.005262642473076839) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.005262642473076839) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.005262642473076839) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.005262642473076839) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.005368659358109599) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005368659358109599) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005379937155839355) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.005379937155839355) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.005379937155839355) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.005379937155839355) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.005652620978017331) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.005652620978017331) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.005708495985960938) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.005708495985960938) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.005708495985960938) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.005708495985960938) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.0059237983365613475) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.0059237983365613475) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.00730675992883301) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.00730675992883301) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.00846997879102397) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.00846997879102397) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.00984174924696263) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.00984174924696263) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.011285190200840957) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (0.011285190200840957) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (0.011756013419819224) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.011756013419819224) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.014564531231173022) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.014564531231173022) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.014603704729162089) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.014603704729162089) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.015225630757226594) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.015225630757226594) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.016024603689179573) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.016024603689179573) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.019028242443847238) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.019028242443847238) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.019257505095251627) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.019257505095251627) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.045879470781297824) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.045879470781297824) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.36937089366156156) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.36937089366156156) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.36937089366156156) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.36937089366156156) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.2816425776702279) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.2816425776702279) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.28164257767022766) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.28164257767022766) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0906514420703648) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0906514420703648) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0906514420703648) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0906514420703648) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863627) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.08684737589863627) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863627) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.08684737589863627) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.07635021950634992) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07635021950634992) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07635021950634992) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07635021950634992) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214009) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.06752385099214009) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214009) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.06752385099214009) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03560837898831247) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.03560837898831247) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03490334337366187) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03490334337366187) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03490334337366187) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03490334337366187) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883829995) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.024591860883829995) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883829995) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.024591860883829995) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.024282117354692906) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.024282117354692906) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.023145130929529075) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.023145130929529075) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.022528440196012984) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.022528440196012984) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019538050311314652) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.019538050311314652) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.019538050311314652) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.019538050311314652) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.017091553155898827) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.017091553155898827) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.017091553155898827) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.017091553155898827) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.016024603689179573) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.016024603689179573) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.016024603689179573) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.016024603689179573) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.010311482489831877) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831877) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831877) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010311482489831877) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.00984174924696263) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.00984174924696263) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.00984174924696263) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.00984174924696263) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209822) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209822) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209822) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008826368514209822) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008541996625454823) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454823) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454823) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454823) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454823) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454823) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454823) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008541996625454823) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008469978791023972) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023972) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.008469978791023972) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023972) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.004668620318776295) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.004668620318776295) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.0038764708993369395) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.0038764708993369395) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.00380406617172854) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00380406617172854) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00380406617172854) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.00380406617172854) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178886) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178886) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0033566705638328758) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.0033566705638328758) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.0032675138544235468) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.0032675138544235468) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0021413612231016045) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.0021413612231016045) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.0017278753941369536) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.0017278753941369536) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.0016407548553124057) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0016407548553124057) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.001452884321416902) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.001452884321416902) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.001452884321416902) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.001452884321416902) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870896771024413) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007870896771024413) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0005192743499487719) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.0005192743499487719) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.0001940085702975688) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0001940085702975688) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303549745) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.00013840177303549745) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.141625221157317e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.141625221157317e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.141625221157317e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.141625221157317e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-5.071480736448518e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.071480736448518e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.151346311141735e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.151346311141735e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0882507112194774e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.0882507112194774e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.9885117063564172e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.9885117063564172e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.87429907131964e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.87429907131964e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.3609563202589218e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.3609563202589218e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.3002946562161126e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.3002946562161126e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.146837650727864e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.146837650727864e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.146837650727864e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.146837650727864e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.352332102821757e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.352332102821757e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.352332102821757e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.352332102821757e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.091637198893477e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637198893477e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637198893477e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637198893477e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637198893477e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637198893477e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637198893477e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.091637198893477e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.074305985705454e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.074305985705454e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.074305985705454e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.074305985705454e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.900128986156438e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.900128986156438e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.900128986156438e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900128986156438e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867765104045929e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.867765104045929e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.560692464611606e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692464611606e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692464611606e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692464611606e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692464611606e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692464611606e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692464611606e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.560692464611606e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.99701842209623e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.99701842209623e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.99701842209623e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.99701842209623e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.99701842209623e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.99701842209623e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.99701842209623e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.99701842209623e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.5682475211221987e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.5682475211221987e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.5682475211221987e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.5682475211221987e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393083851645e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393083851645e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393083851645e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393083851645e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393083851645e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393083851645e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.3767393083851645e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393083851645e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.888293595536702e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.888293595536702e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.686381544852535e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.686381544852535e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035783554134726e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7035783554134726e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.209350646540233e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.209350646540233e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773243987314e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773243987314e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773243987314e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773243987314e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773243987314e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773243987314e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.379773243987314e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773243987314e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253792071848e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253792071848e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9742253792071848e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9742253792071848e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0474716555080352e-08) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.0474716555080352e-08) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.0474716555080352e-08) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.0474716555080352e-08) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.209350646540233e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.209350646540233e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0717282183003911e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.0717282183003911e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.0717282183003911e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.0717282183003911e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.2004287493681124e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.2004287493681124e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.2004287493681124e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.2004287493681124e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.7035783554134726e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7035783554134726e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.312094305133736e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.312094305133736e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.312094305133736e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.312094305133736e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.686381544852535e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.686381544852535e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.888293595536702e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.888293595536702e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.0922506160172447e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506160172447e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.0922506160172447e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506160172447e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.0922506160172447e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506160172447e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.0922506160172447e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506160172447e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.4445978540786406e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.4445978540786406e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.4445978540786406e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.4445978540786406e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.6849150950932256e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.6849150950932256e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.6849150950932256e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.6849150950932256e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.246974425269414e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.246974425269414e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.246974425269414e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.246974425269414e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.246974425269414e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.246974425269414e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.246974425269414e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.246974425269414e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.867765104045929e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.867765104045929e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.3002946562161126e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.3002946562161126e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.3609563202589218e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.3609563202589218e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.87429907131964e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.87429907131964e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.8836765759258386e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.8836765759258386e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.947356011558074e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.947356011558074e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.947356011558074e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.947356011558074e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117063564172e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9885117063564172e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.0882507112194774e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.0882507112194774e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.151346311141735e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.151346311141735e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.846201671151076e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.846201671151076e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.846201671151076e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.846201671151076e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.071480736448518e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.071480736448518e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.105526721840849e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.105526721840849e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.105526721840849e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.105526721840849e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.146496327367189e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.146496327367189e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.146496327367189e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.146496327367189e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.1593505017912796e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.1593505017912796e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.1593505017912796e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.1593505017912796e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.427988656276533e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.427988656276533e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.427988656276533e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.427988656276533e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.935867717914491e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.935867717914491e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.935867717914491e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.935867717914491e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273347874336e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.253273347874336e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.979825793160489e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.979825793160489e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.979825793160489e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.979825793160489e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (4.205548411215579e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.205548411215579e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.205548411215579e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.205548411215579e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00013840177303549745) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.00013840177303549745) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.0001878705338955039) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0001878705338955039) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0001878705338955039) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0001878705338955039) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0001940085702975688) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.0001940085702975688) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0002463643756957322) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0002463643756957322) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0002463643756957322) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0002463643756957322) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192743499487719) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.0005192743499487719) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.0007156734248908682) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007156734248908682) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0007156734248908682) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007156734248908682) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024413) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007870896771024413) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.0015324835230730227) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.0015324835230730227) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.0015324835230730227) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.0015324835230730227) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.0016407548553124057) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0016407548553124057) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0017278753941369536) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.0017278753941369536) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.002446497155415826) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.002446497155415826) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.002446497155415826) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.002446497155415826) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.0032675138544235468) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.0032675138544235468) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.0033566705638328758) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.0033566705638328758) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.0034841573002178886) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0034841573002178886) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0038764708993369395) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.0038764708993369395) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.004668620318776295) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.004668620318776295) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278084) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.004767272188278084) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.004767272188278084) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278084) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.005286546538226855) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.005286546538226855) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.005286546538226855) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.005286546538226855) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.005408954422409962) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.005408954422409962) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.005408954422409962) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.005408954422409962) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.0059237983365613475) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.0059237983365613475) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.0059237983365613475) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.0059237983365613475) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.010715508469796802) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010715508469796802) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010715508469796802) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010715508469796802) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.010757563953908962) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010757563953908962) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010757563953908962) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010757563953908962) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.014603704729162089) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.014603704729162089) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.014603704729162089) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.014603704729162089) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.019299560579363786) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363786) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363786) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363786) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363786) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363786) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.019299560579363786) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363786) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.05859198873386171) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.05859198873386171) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.775950527047802e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.775950527047802e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.775950527047804e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.775950527047804e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.07165035181002517) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.07165035181002517) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0716503518100252) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0716503518100252) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.019257505095251627) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019257505095251627) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010311482489831877) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311482489831877) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008826368514209822) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209822) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0075974640297706) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0075974640297706) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0075974640297706) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0075974640297706) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311878) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311878) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311878) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311878) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311878) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311878) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311878) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311878) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676611) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005348051582676611) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005348051582676611) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676611) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0038040661717285394) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0038040661717285394) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0029841661681219286) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.0029841661681219286) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.0029841661681219286) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.0029841661681219286) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.0024464971554158258) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.0024464971554158258) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.0022494124470939882) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0022494124470939882) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0022494124470939882) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0022494124470939882) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0021413612231016045) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.0021413612231016045) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.0018638942824587236) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587236) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587236) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587236) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587236) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587236) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0018638942824587236) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587236) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0016407548553124059) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124059) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0016407548553124059) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124059) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0012223378081538247) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538247) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538247) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538247) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538247) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538247) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538247) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0012223378081538247) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0010283292378562559) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0010283292378562559) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0010283292378562559) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0010283292378562559) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1463061452749187e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1463061452749187e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.87429907131964e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.87429907131964e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.87429907131964e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.87429907131964e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.3002946562161126e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.3002946562161126e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.3002946562161126e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.3002946562161126e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.0444941297649691e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.0444941297649691e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.0444941297649691e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.0444941297649691e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.956079229682353e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.956079229682353e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.956079229682353e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.956079229682353e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105515036845376e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105515036845376e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105515036845376e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105515036845376e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661347212884969e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661347212884969e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661347212884969e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661347212884969e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.54034141354165e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.54034141354165e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.18999097511814e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.18999097511814e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.876621658040841e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876621658040841e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876621658040841e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876621658040841e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175246206969021e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175246206969021e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.523389677771276e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.523389677771276e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.076732531663776e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.076732531663776e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.076732531663776e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.076732531663776e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.0134714588502317e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0134714588502317e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.904599884108039e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.904599884108039e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.904599884108039e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.904599884108039e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.6667317545378977e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.6667317545378977e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.6667317545378977e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6667317545378977e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641928369785e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.8505641928369785e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.656930931685258e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.656930931685258e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.656930931685258e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.656930931685258e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.8505641928369785e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.8505641928369785e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815448525356e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815448525356e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.6863815448525356e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815448525356e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714588502317e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.0134714588502317e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.523389677771276e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.523389677771276e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.67040239053549e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.67040239053549e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.67040239053549e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.67040239053549e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175246206969021e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175246206969021e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.18999097511814e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.18999097511814e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.54034141354165e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.54034141354165e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949476487301752e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949476487301752e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.7924939576611075e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939576611075e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939576611075e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.7924939576611075e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.883676575925839e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.883676575925839e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.9885117063564172e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117063564172e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117063564172e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117063564172e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273347874336e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.253273347874336e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.40171097349144e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.40171097349144e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.40171097349144e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.40171097349144e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603692575507e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.5809603692575507e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603692575507e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.5809603692575507e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0005192743499487719) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487719) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.0005192743499487719) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487719) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.0007870896771024414) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024414) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024414) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024414) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0011726348316441922) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0011726348316441922) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0011726348316441922) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0011726348316441922) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.001236647801924518) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.001236647801924518) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.001236647801924518) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.001236647801924518) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.0022009640695004476) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0022009640695004476) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0022009640695004476) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0022009640695004476) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798017) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798017) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798017) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798017) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798017) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798017) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002394972639798017) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798017) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0024464971554158258) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.0024464971554158258) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.0038040661717285394) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0038040661717285394) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0038764708993369395) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.0038764708993369395) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.0038764708993369395) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.0038764708993369395) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.004220813970046446) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.004220813970046446) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.004220813970046446) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.004220813970046446) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.008826368514209822) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.008826368514209822) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.010311482489831877) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010311482489831877) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019257505095251627) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019257505095251627) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.05859198873386171) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.05859198873386171) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.3987009014222996e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.3987009014222996e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.3987009014222996e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.3987009014222996e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841573002178886) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841573002178886) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002984166168121929) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.002984166168121929) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.0001940085702975688) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0001940085702975688) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061452749187e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061452749187e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.7924939576611075e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7924939576611075e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.54034141354165e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.54034141354165e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.54034141354165e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.54034141354165e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.8505641928369785e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641928369785e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641928369785e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641928369785e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714588502317e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.0134714588502317e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714588502317e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.0134714588502317e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.949476487301752e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.949476487301752e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.7924939576611075e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7924939576611075e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0001940085702975688) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0001940085702975688) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002984166168121929) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.002984166168121929) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.0034841573002178886) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0034841573002178886) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
Expectation value of XIZ =  0.07715357869738948
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
  (-0.24274280513140462) [Z2]
+ (-0.24274280513140462) [Z3]
+ (-0.04207897647782277) [I0]
+ (0.1777128746513994) [Z1]
+ (0.17771287465139946) [Z0]
+ (0.12293305056183801) [Z0 Z2]
+ (0.12293305056183801) [Z1 Z3]
+ (0.1676831945771896) [Z0 Z3]
+ (0.1676831945771896) [Z1 Z2]
+ (0.17059738328801052) [Z0 Z1]
+ (0.1762764080431959) [Z2 Z3]
Cost function value: -0.5449960180902932
   (-46.463906788688924) [I0]
+ (0.7829661725950173) [Z10]
+ (0.7829661725950177) [Z11]
+ (0.8084581961720484) [Z12]
+ (0.8084581961720486) [Z13]
+ (1.2034402289145631) [Z4]
+ (1.2034402289145636) [Z5]
+ (1.3096862988615436) [Z6]
+ (1.3096862988615436) [Z7]
+ (1.369352563471818) [Z9]
+ (1.3693525634718182) [Z8]
+ (1.6538942226831699) [Z2]
+ (1.6538942226831703) [Z3]
+ (12.412630742111773) [Z0]
+ (12.412630742111773) [Z1]
+ (-8.194261371915928e-06) [Y10 Y12]
+ (-8.194261371915928e-06) [X10 X12]
+ (-1.8540608579051137e-06) [Y5 Y7]
+ (-1.8540608579051137e-06) [X5 X7]
+ (-7.764994117358654e-07) [Y3 Y5]
+ (-7.764994117358654e-07) [X3 X5]
+ (-5.929765815565687e-07) [Y4 Y6]
+ (-5.929765815565687e-07) [X4 X6]
+ (1.6021167407095646e-06) [Y2 Y4]
+ (1.6021167407095646e-06) [X2 X4]
+ (7.954413175767151e-06) [Y11 Y13]
+ (7.954413175767151e-06) [X11 X13]
+ (0.003276971931231585) [Y1 Y3]
+ (0.003276971931231585) [X1 X3]
+ (0.10433064780651398) [Y0 Y2]
+ (0.10433064780651398) [X0 X2]
+ (0.11270386920332202) [Z10 Z12]
+ (0.11270386920332202) [Z11 Z13]
+ (0.11383573679388667) [Z4 Z12]
+ (0.11383573679388667) [Z5 Z13]
+ (0.11952438964682675) [Z6 Z10]
+ (0.11952438964682675) [Z7 Z11]
+ (0.12489990917237599) [Z4 Z10]
+ (0.12489990917237599) [Z5 Z11]
+ (0.12495807739503233) [Z2 Z4]
+ (0.12495807739503233) [Z3 Z5]
+ (0.12799502492468415) [Z2 Z10]
+ (0.12799502492468415) [Z3 Z11]
+ (0.1340171526196372) [Z6 Z12]
+ (0.1340171526196372) [Z7 Z13]
+ (0.1373495306426131) [Z6 Z11]
+ (0.1373495306426131) [Z7 Z10]
+ (0.13739104762683246) [Z2 Z6]
+ (0.13739104762683246) [Z3 Z7]
+ (0.13766872645852563) [Z8 Z10]
+ (0.13766872645852563) [Z9 Z11]
+ (0.1401128986535482) [Z2 Z12]
+ (0.1401128986535482) [Z3 Z13]
+ (0.14138905291942802) [Z10 Z13]
+ (0.14138905291942802) [Z11 Z12]
+ (0.14257997712485762) [Z4 Z11]
+ (0.14257997712485762) [Z5 Z10]
+ (0.14722943218766155) [Z8 Z11]
+ (0.14722943218766155) [Z9 Z10]
+ (0.14899430575065561) [Z4 Z7]
+ (0.14899430575065561) [Z5 Z6]
+ (0.14926355147388892) [Z10 Z11]
+ (0.1497348680349693) [Z8 Z12]
+ (0.1497348680349693) [Z9 Z13]
+ (0.15071408121008295) [Z2 Z8]
+ (0.15071408121008295) [Z3 Z9]
+ (0.15138327161428852) [Z6 Z13]
+ (0.15138327161428852) [Z7 Z12]
+ (0.15215040708869057) [Z4 Z13]
+ (0.15215040708869057) [Z5 Z12]
+ (0.15337968243314157) [Z2 Z11]
+ (0.15337968243314157) [Z3 Z10]
+ (0.15435748657223627) [Z12 Z13]
+ (0.15569010671752465) [Z2 Z13]
+ (0.15569010671752465) [Z3 Z12]
+ (0.15582269051553116) [Z8 Z13]
+ (0.15582269051553116) [Z9 Z12]
+ (0.1607976453483858) [Z2 Z5]
+ (0.1607976453483858) [Z3 Z4]
+ (0.16756653265461288) [Z6 Z8]
+ (0.16756653265461288) [Z7 Z9]
+ (0.1685348656157995) [Z2 Z7]
+ (0.1685348656157995) [Z3 Z6]
+ (0.18143991440303905) [Z6 Z9]
+ (0.18143991440303905) [Z7 Z8]
+ (0.18189085790751386) [Z2 Z3]
+ (0.192997239353642) [Z0 Z10]
+ (0.192997239353642) [Z1 Z11]
+ (0.19392534613270265) [Z6 Z7]
+ (0.19661770890342142) [Z0 Z4]
+ (0.19661770890342142) [Z1 Z5]
+ (0.19936354537360826) [Z0 Z5]
+ (0.19936354537360826) [Z1 Z4]
+ (0.20072866460441727) [Z0 Z11]
+ (0.20072866460441727) [Z1 Z10]
+ (0.21102659849791522) [Z0 Z12]
+ (0.21102659849791522) [Z1 Z13]
+ (0.21631037498631817) [Z0 Z13]
+ (0.21631037498631817) [Z1 Z12]
+ (0.23671080783830428) [Z0 Z2]
+ (0.23671080783830428) [Z1 Z3]
+ (0.24164663936017244) [Z0 Z6]
+ (0.24164663936017244) [Z1 Z7]
+ (0.24853483371314306) [Z0 Z7]
+ (0.24853483371314306) [Z1 Z6]
+ (0.25129445674591694) [Z0 Z3]
+ (0.25129445674591694) [Z1 Z2]
+ (0.27232518306605674) [Z0 Z8]
+ (0.27232518306605674) [Z1 Z9]
+ (0.278834544267234) [Z0 Z9]
+ (0.278834544267234) [Z1 Z8]
+ (1.1861763734860493) [Z0 Z1]
+ (-1.226048498875555e-05) [Y4 Z5 Y6]
+ (-1.226048498875555e-05) [X4 Z5 X6]
+ (-1.226048498875555e-05) [Y5 Z6 Y7]
+ (-1.226048498875555e-05) [X5 Z6 X7]
+ (-1.0722312158457259e-05) [Y10 Z11 Y12]
+ (-1.0722312158457259e-05) [X10 Z11 X12]
+ (-1.0722312158457254e-05) [Y11 Z12 Y13]
+ (-1.0722312158457254e-05) [X11 Z12 X13]
+ (-3.887051671428606e-06) [Y2 Z3 Y4]
+ (-3.887051671428606e-06) [X2 Z3 X4]
+ (-3.887051671428605e-06) [Y3 Z4 Y5]
+ (-3.887051671428605e-06) [X3 Z4 X5]
+ (0.12507032579772023) [Y0 Z1 Y2]
+ (0.12507032579772023) [X0 Z1 X2]
+ (0.12507032579772026) [Y1 Z2 Y3]
+ (0.12507032579772026) [X1 Z2 X3]
+ (-0.03831467029480389) [Y4 Y5 X12 X13]
+ (-0.03831467029480389) [X4 X5 Y12 Y13]
+ (-0.036194123559042606) [Y2 Y3 X8 X9]
+ (-0.036194123559042606) [X2 X3 Y8 Y9]
+ (-0.03583956795335349) [Y2 Y3 X4 X5]
+ (-0.03583956795335349) [X2 X3 Y4 Y5]
+ (-0.031143817988967013) [Y2 Y3 X6 X7]
+ (-0.031143817988967013) [X2 X3 Y6 Y7]
+ (-0.028685183716105983) [Y10 Y11 X12 X13]
+ (-0.028685183716105983) [X10 X11 Y12 Y13]
+ (-0.025996177598021242) [Y3 Z4 Z5 Y7]
+ (-0.025996177598021242) [X3 Z4 Z5 X7]
+ (-0.025384657508457423) [Y2 Y3 X10 X11]
+ (-0.025384657508457423) [X2 X3 Y10 Y11]
+ (-0.01902824244384736) [Y3 Y4 X11 X12]
+ (-0.01902824244384736) [X3 X4 Y11 Y12]
+ (-0.017825140995786342) [Y6 Y7 X10 X11]
+ (-0.017825140995786342) [X6 X7 Y10 Y11]
+ (-0.017680067952481643) [Y4 Y5 X10 X11]
+ (-0.017680067952481643) [X4 X5 Y10 Y11]
+ (-0.017366118994651333) [Y6 Y7 X12 X13]
+ (-0.017366118994651333) [X6 X7 Y12 Y13]
+ (-0.015577208063976444) [Y2 Y3 X12 X13]
+ (-0.015577208063976444) [X2 X3 Y12 Y13]
+ (-0.014583648907612667) [Y0 Y1 X2 X3]
+ (-0.014583648907612667) [X0 X1 Y2 Y3]
+ (-0.013873381748426157) [Y6 Y7 X8 X9]
+ (-0.013873381748426157) [X6 X7 Y8 Y9]
+ (-0.011982389010247906) [Y4 Y5 X6 X7]
+ (-0.011982389010247906) [X4 X5 Y6 Y7]
+ (-0.011285190200840886) [Y5 X6 X11 Y12]
+ (-0.011285190200840886) [X5 Y6 Y11 X12]
+ (-0.009560705729135909) [Y8 Y9 X10 X11]
+ (-0.009560705729135909) [X8 X9 Y10 Y11]
+ (-0.00812525192138104) [Y1 X2 X8 Y9]
+ (-0.00812525192138104) [Y1 Y2 Y8 Y9]
+ (-0.00812525192138104) [X1 X2 X8 X9]
+ (-0.00812525192138104) [X1 Y2 Y8 X9]
+ (-0.007731425250775282) [Y0 Y1 X10 X11]
+ (-0.007731425250775282) [X0 X1 Y10 Y11]
+ (-0.0071569349198569426) [Y4 Y5 X8 X9]
+ (-0.0071569349198569426) [X4 X5 Y8 Y9]
+ (-0.006888194352970598) [Y0 Y1 X6 X7]
+ (-0.006888194352970598) [X0 X1 Y6 Y7]
+ (-0.0065093612011772415) [Y0 Y1 X8 X9]
+ (-0.0065093612011772415) [X0 X1 Y8 Y9]
+ (-0.006087822480561867) [Y8 Y9 X12 X13]
+ (-0.006087822480561867) [X8 X9 Y12 Y13]
+ (-0.005283776488402969) [Y0 Y1 X12 X13]
+ (-0.005283776488402969) [X0 X1 Y12 Y13]
+ (-0.005143391768825054) [Y3 X4 X5 Y6]
+ (-0.005143391768825054) [X3 Y4 Y5 X6]
+ (-0.0046849033881552204) [Y1 X2 X6 Y7]
+ (-0.0046849033881552204) [Y1 Y2 Y6 Y7]
+ (-0.0046849033881552204) [X1 X2 X6 X7]
+ (-0.0046849033881552204) [X1 Y2 Y6 X7]
+ (-0.00457500762663921) [Y1 X2 X12 Y13]
+ (-0.00457500762663921) [Y1 Y2 Y12 Y13]
+ (-0.00457500762663921) [X1 X2 X12 X13]
+ (-0.00457500762663921) [X1 Y2 Y12 X13]
+ (-0.004424855449441858) [Y1 X2 X4 Y5]
+ (-0.004424855449441858) [Y1 Y2 Y4 Y5]
+ (-0.004424855449441858) [X1 X2 X4 X5]
+ (-0.004424855449441858) [X1 Y2 Y4 X5]
+ (-0.0034795118903341894) [Y2 Z3 Z5 Y6]
+ (-0.0034795118903341894) [X2 Z3 Z5 X6]
+ (-0.0034795118903341894) [Y3 Z4 Z6 Y7]
+ (-0.0034795118903341894) [X3 Z4 Z6 X7]
+ (-0.0017992194936630012) [Y1 X2 X10 Y11]
+ (-0.0017992194936630012) [Y1 Y2 Y10 Y11]
+ (-0.0017992194936630012) [X1 X2 X10 X11]
+ (-0.0017992194936630012) [X1 Y2 Y10 X11]
+ (-0.00029219862611108607) [Y7 Y8 X9 X10]
+ (-0.00029219862611108607) [X7 X8 Y9 Y10]
+ (-8.194261371915928e-06) [Z10 Y11 Z12 Y13]
+ (-8.194261371915928e-06) [Z10 X11 Z12 X13]
+ (-7.801707500047804e-06) [Y2 Z3 Y4 Z11]
+ (-7.801707500047804e-06) [X2 Z3 X4 Z11]
+ (-7.801707500047804e-06) [Y3 Z4 Y5 Z10]
+ (-7.801707500047804e-06) [X3 Z4 X5 Z10]
+ (-4.6430510682894324e-06) [Y3 X4 X10 Y11]
+ (-4.6430510682894324e-06) [Y3 Y4 Y10 Y11]
+ (-4.6430510682894324e-06) [X3 X4 X10 X11]
+ (-4.6430510682894324e-06) [X3 Y4 Y10 X11]
+ (-4.588855155501e-06) [Y4 Z5 Y6 Z13]
+ (-4.588855155501e-06) [X4 Z5 X6 Z13]
+ (-4.588855155501e-06) [Y5 Z6 Y7 Z12]
+ (-4.588855155501e-06) [X5 Z6 X7 Z12]
+ (-4.55656921795634e-06) [Y5 X6 X12 Y13]
+ (-4.55656921795634e-06) [Y5 Y6 Y12 Y13]
+ (-4.55656921795634e-06) [X5 X6 X12 X13]
+ (-4.55656921795634e-06) [X5 Y6 Y12 X13]
+ (-3.6945132943900423e-06) [Y4 X5 X11 Y12]
+ (-3.6945132943900423e-06) [Y4 Y5 Y11 Y12]
+ (-3.6945132943900423e-06) [X4 X5 X11 X12]
+ (-3.6945132943900423e-06) [X4 Y5 Y11 X12]
+ (-3.3440815563651105e-06) [Z0 Y5 Z6 Y7]
+ (-3.3440815563651105e-06) [Z0 X5 Z6 X7]
+ (-3.3440815563651105e-06) [Z1 Y4 Z5 Y6]
+ (-3.3440815563651105e-06) [Z1 X4 Z5 X6]
+ (-3.158656431758373e-06) [Y2 Z3 Y4 Z10]
+ (-3.158656431758373e-06) [X2 Z3 X4 Z10]
+ (-3.158656431758373e-06) [Y3 Z4 Y5 Z11]
+ (-3.158656431758373e-06) [X3 Z4 X5 Z11]
+ (-3.0993492435101672e-06) [Z0 Y4 Z5 Y6]
+ (-3.0993492435101672e-06) [Z0 X4 Z5 X6]
+ (-3.0993492435101672e-06) [Z1 Y5 Z6 Y7]
+ (-3.0993492435101672e-06) [Z1 X5 Z6 X7]
+ (-2.8909678816365463e-06) [Z6 Y11 Z12 Y13]
+ (-2.8909678816365463e-06) [Z6 X11 Z12 X13]
+ (-2.8909678816365463e-06) [Z7 Y10 Z11 Y12]
+ (-2.8909678816365463e-06) [Z7 X10 Z11 X12]
+ (-2.1776646051139896e-06) [Z0 Y10 Z11 Y12]
+ (-2.1776646051139896e-06) [Z0 X10 Z11 X12]
+ (-2.1776646051139896e-06) [Z1 Y11 Z12 Y13]
+ (-2.1776646051139896e-06) [Z1 X11 Z12 X13]
+ (-1.881850183156339e-06) [Y4 Z5 Y6 Z9]
+ (-1.881850183156339e-06) [X4 Z5 X6 Z9]
+ (-1.881850183156339e-06) [Y5 Z6 Y7 Z8]
+ (-1.881850183156339e-06) [X5 Z6 X7 Z8]
+ (-1.8551201215656438e-06) [Z6 Y10 Z11 Y12]
+ (-1.8551201215656438e-06) [Z6 X10 Z11 X12]
+ (-1.8551201215656438e-06) [Z7 Y11 Z12 Y13]
+ (-1.8551201215656438e-06) [Z7 X11 Z12 X13]
+ (-1.854060857905114e-06) [Y4 Z5 Y6 Z7]
+ (-1.854060857905114e-06) [X4 Z5 X6 Z7]
+ (-1.8163031698084161e-06) [Z4 Y11 Z12 Y13]
+ (-1.8163031698084161e-06) [Z4 X11 Z12 X13]
+ (-1.8163031698084161e-06) [Z5 Y10 Z11 Y12]
+ (-1.8163031698084161e-06) [Z5 X10 Z11 X12]
+ (-1.6923978285595932e-06) [Y4 Z5 Y6 Z10]
+ (-1.6923978285595932e-06) [X4 Z5 X6 Z10]
+ (-1.6923978285595932e-06) [Y5 Z6 Y7 Z11]
+ (-1.6923978285595932e-06) [X5 Z6 X7 Z11]
+ (-1.614879413997043e-06) [Z0 Y11 Z12 Y13]
+ (-1.614879413997043e-06) [Z0 X11 Z12 X13]
+ (-1.614879413997043e-06) [Z1 Y10 Z11 Y12]
+ (-1.614879413997043e-06) [Z1 X10 Z11 X12]
+ (-1.5973171978417752e-06) [Z8 Y10 Z11 Y12]
+ (-1.5973171978417752e-06) [Z8 X10 Z11 X12]
+ (-1.5973171978417752e-06) [Z9 Y11 Z12 Y13]
+ (-1.5973171978417752e-06) [Z9 X11 Z12 X13]
+ (-1.454842448945275e-06) [Y3 X4 X6 Y7]
+ (-1.454842448945275e-06) [Y3 Y4 Y6 Y7]
+ (-1.454842448945275e-06) [X3 X4 X6 X7]
+ (-1.454842448945275e-06) [X3 Y4 Y6 X7]
+ (-1.3980449080999489e-06) [Y4 Z5 Y6 Z8]
+ (-1.3980449080999489e-06) [X4 Z5 X6 Z8]
+ (-1.3980449080999489e-06) [Y5 Z6 Y7 Z9]
+ (-1.3980449080999489e-06) [X5 Z6 X7 Z9]
+ (-1.195489009736364e-06) [Y2 Z3 Y4 Z7]
+ (-1.195489009736364e-06) [X2 Z3 X4 Z7]
+ (-1.195489009736364e-06) [Y3 Z4 Y5 Z6]
+ (-1.195489009736364e-06) [X3 Z4 X5 Z6]
+ (-1.1908508079827288e-06) [Z0 Y3 Z4 Y5]
+ (-1.1908508079827288e-06) [Z0 X3 Z4 X5]
+ (-1.1908508079827288e-06) [Z1 Y2 Z3 Y4]
+ (-1.1908508079827288e-06) [Z1 X2 Z3 X4]
+ (-1.1708301370095027e-06) [Z2 Y5 Z6 Y7]
+ (-1.1708301370095027e-06) [Z2 X5 Z6 X7]
+ (-1.1708301370095027e-06) [Z3 Y4 Z5 Y6]
+ (-1.1708301370095027e-06) [Z3 X4 Z5 X6]
+ (-1.0632283423803455e-06) [Z2 Y10 Z11 Y12]
+ (-1.0632283423803455e-06) [Z2 X10 Z11 X12]
+ (-1.0632283423803455e-06) [Z3 Y11 Z12 Y13]
+ (-1.0632283423803455e-06) [Z3 X11 Z12 X13]
+ (-1.0358477600709027e-06) [Y6 X7 X11 Y12]
+ (-1.0358477600709027e-06) [Y6 Y7 Y11 Y12]
+ (-1.0358477600709027e-06) [X6 X7 X11 X12]
+ (-1.0358477600709027e-06) [X6 Y7 Y11 X12]
+ (-9.509249751762266e-07) [Z2 Y4 Z5 Y6]
+ (-9.509249751762266e-07) [Z2 X4 Z5 X6]
+ (-9.509249751762266e-07) [Z3 Y5 Z6 Y7]
+ (-9.509249751762266e-07) [Z3 X5 Z6 X7]
+ (-9.344557777302516e-07) [Z8 Y11 Z12 Y13]
+ (-9.344557777302516e-07) [Z8 X11 Z12 X13]
+ (-9.344557777302516e-07) [Z9 Y10 Z11 Y12]
+ (-9.344557777302516e-07) [Z9 X10 Z11 X12]
+ (-8.337746751303545e-07) [Z0 Y2 Z3 Y4]
+ (-8.337746751303545e-07) [Z0 X2 Z3 X4]
+ (-8.337746751303545e-07) [Z1 Y3 Z4 Y5]
+ (-8.337746751303545e-07) [Z1 X3 Z4 X5]
+ (-7.956895371639311e-07) [Y3 X4 X8 Y9]
+ (-7.956895371639311e-07) [Y3 Y4 Y8 Y9]
+ (-7.956895371639311e-07) [X3 X4 X8 X9]
+ (-7.956895371639311e-07) [X3 Y4 Y8 X9]
+ (-7.764994117358654e-07) [Y2 Z3 Y4 Z5]
+ (-7.764994117358654e-07) [X2 Z3 X4 Z5]
+ (-5.929765815565687e-07) [Z4 Y5 Z6 Y7]
+ (-5.929765815565687e-07) [Z4 X5 Z6 X7]
+ (-5.77005299291543e-07) [Y2 Z3 Y4 Z9]
+ (-5.77005299291543e-07) [X2 Z3 X4 Z9]
+ (-5.77005299291543e-07) [Y3 Z4 Y5 Z8]
+ (-5.77005299291543e-07) [X3 Z4 X5 Z8]
+ (-5.471647744460724e-07) [Y1 Y2 X11 X12]
+ (-5.471647744460724e-07) [X1 X2 Y11 Y12]
+ (-4.838052750563902e-07) [Y5 X6 X8 Y9]
+ (-4.838052750563902e-07) [Y5 Y6 Y8 Y9]
+ (-4.838052750563902e-07) [X5 X6 X8 X9]
+ (-4.838052750563902e-07) [X5 Y6 Y8 X9]
+ (-3.5707613285237446e-07) [Y0 X1 X3 Y4]
+ (-3.5707613285237446e-07) [Y0 Y1 Y3 Y4]
+ (-3.5707613285237446e-07) [X0 X1 X3 X4]
+ (-3.5707613285237446e-07) [X0 Y1 Y3 X4]
+ (-2.447323128549443e-07) [Y0 X1 X5 Y6]
+ (-2.447323128549443e-07) [Y0 Y1 Y5 Y6]
+ (-2.447323128549443e-07) [X0 X1 X5 X6]
+ (-2.447323128549443e-07) [X0 Y1 Y5 X6]
+ (-2.1990516183327606e-07) [Y2 X3 X5 Y6]
+ (-2.1990516183327606e-07) [Y2 Y3 Y5 Y6]
+ (-2.1990516183327606e-07) [X2 X3 X5 X6]
+ (-2.1990516183327606e-07) [X2 Y3 Y5 X6]
+ (-1.9332412769828785e-07) [Y1 X2 X3 Y4]
+ (-1.9332412769828785e-07) [X1 Y2 Y3 X4]
+ (-1.2919694862656315e-07) [Y1 Z2 Z3 Y5]
+ (-1.2919694862656315e-07) [X1 Z2 Z3 X5]
+ (1.7379332621726185e-07) [Y0 Z1 Z3 Y4]
+ (1.7379332621726185e-07) [X0 Z1 Z3 X4]
+ (1.7379332621726185e-07) [Y1 Z2 Z4 Y5]
+ (1.7379332621726185e-07) [X1 Z2 Z4 X5]
+ (1.9332412769828785e-07) [Y1 Y2 X3 X4]
+ (1.9332412769828785e-07) [X1 X2 Y3 Y4]
+ (2.1868423787238815e-07) [Y2 Z3 Y4 Z8]
+ (2.1868423787238815e-07) [X2 Z3 X4 Z8]
+ (2.1868423787238815e-07) [Y3 Z4 Y5 Z9]
+ (2.1868423787238815e-07) [X3 Z4 X5 Z9]
+ (2.5935343920891113e-07) [Y2 Z3 Y4 Z6]
+ (2.5935343920891113e-07) [X2 Z3 X4 Z6]
+ (2.5935343920891113e-07) [Y3 Z4 Y5 Z7]
+ (2.5935343920891113e-07) [X3 Z4 X5 Z7]
+ (3.6060718675528685e-07) [Y0 Z1 Z2 Y4]
+ (3.6060718675528685e-07) [X0 Z1 Z2 X4]
+ (3.6060718675528685e-07) [Y1 Z3 Z4 Y5]
+ (3.6060718675528685e-07) [X1 Z3 Z4 X5]
+ (5.471647744460724e-07) [Y1 X2 X11 Y12]
+ (5.471647744460724e-07) [X1 Y2 Y11 X12]
+ (5.62785191116946e-07) [Y0 X1 X11 Y12]
+ (5.62785191116946e-07) [Y0 Y1 Y11 Y12]
+ (5.62785191116946e-07) [X0 X1 X11 X12]
+ (5.62785191116946e-07) [X0 Y1 Y11 X12]
+ (6.628614201115235e-07) [Y8 X9 X11 Y12]
+ (6.628614201115235e-07) [Y8 Y9 Y11 Y12]
+ (6.628614201115235e-07) [X8 X9 X11 X12]
+ (6.628614201115235e-07) [X8 Y9 Y11 X12]
+ (1.10944075896334e-06) [Z2 Y11 Z12 Y13]
+ (1.10944075896334e-06) [Z2 X11 Z12 X13]
+ (1.10944075896334e-06) [Z3 Y10 Z11 Y12]
+ (1.10944075896334e-06) [Z3 X10 Z11 X12]
+ (1.6021167407095646e-06) [Z2 Y3 Z4 Y5]
+ (1.6021167407095646e-06) [Z2 X3 Z4 X5]
+ (1.8782101245816266e-06) [Z4 Y10 Z11 Y12]
+ (1.8782101245816266e-06) [Z4 X10 Z11 X12]
+ (1.8782101245816266e-06) [Z5 Y11 Z12 Y13]
+ (1.8782101245816266e-06) [Z5 X11 Z12 X13]
+ (2.1726691013436857e-06) [Y2 X3 X11 Y12]
+ (2.1726691013436857e-06) [Y2 Y3 Y11 Y12]
+ (2.1726691013436857e-06) [X2 X3 X11 X12]
+ (2.1726691013436857e-06) [X2 Y3 Y11 X12]
+ (3.1174479456272065e-06) [Y0 Z2 Z3 Y4]
+ (3.1174479456272065e-06) [X0 Z2 Z3 X4]
+ (3.539054184440526e-06) [Y2 Z3 Y4 Z12]
+ (3.539054184440526e-06) [X2 Z3 X4 Z12]
+ (3.539054184440526e-06) [Y3 Z4 Y5 Z13]
+ (3.539054184440526e-06) [X3 Z4 X5 Z13]
+ (4.281913884590931e-06) [Y4 Z5 Y6 Z11]
+ (4.281913884590931e-06) [X4 Z5 X6 Z11]
+ (4.281913884590931e-06) [Y5 Z6 Y7 Z10]
+ (4.281913884590931e-06) [X5 Z6 X7 Z10]
+ (5.2758831218566915e-06) [Y3 X4 X12 Y13]
+ (5.2758831218566915e-06) [Y3 Y4 Y12 Y13]
+ (5.2758831218566915e-06) [X3 X4 X12 X13]
+ (5.2758831218566915e-06) [X3 Y4 Y12 X13]
+ (5.9743117131505255e-06) [Y5 X6 X10 Y11]
+ (5.9743117131505255e-06) [Y5 Y6 Y10 Y11]
+ (5.9743117131505255e-06) [X5 X6 X10 X11]
+ (5.9743117131505255e-06) [X5 Y6 Y10 X11]
+ (7.954413175767151e-06) [Y10 Z11 Y12 Z13]
+ (7.954413175767151e-06) [X10 Z11 X12 Z13]
+ (8.814937306297218e-06) [Y2 Z3 Y4 Z13]
+ (8.814937306297218e-06) [X2 Z3 X4 Z13]
+ (8.814937306297218e-06) [Y3 Z4 Y5 Z12]
+ (8.814937306297218e-06) [X3 Z4 X5 Z12]
+ (0.00029219862611108607) [Y7 X8 X9 Y10]
+ (0.00029219862611108607) [X7 Y8 Y9 X10]
+ (0.0004956762314918962) [Y2 Z4 Z5 Y6]
+ (0.0004956762314918962) [X2 Z4 Z5 X6]
+ (0.0011059037691896376) [Y0 Z1 Y2 Z5]
+ (0.0011059037691896376) [X0 Z1 X2 Z5]
+ (0.0011059037691896376) [Y1 Z2 Y3 Z4]
+ (0.0011059037691896376) [X1 Z2 X3 Z4]
+ (0.0016638798784908632) [Y2 Z3 Z4 Y6]
+ (0.0016638798784908632) [X2 Z3 Z4 X6]
+ (0.0016638798784908632) [Y3 Z5 Z6 Y7]
+ (0.0016638798784908632) [X3 Z5 Z6 X7]
+ (0.0017560707018411741) [Y0 Z1 Y2 Z11]
+ (0.0017560707018411741) [X0 Z1 X2 Z11]
+ (0.0017560707018411741) [Y1 Z2 Y3 Z10]
+ (0.0017560707018411741) [X1 Z2 X3 Z10]
+ (0.00232623062315804) [Y0 Z1 Y2 Z13]
+ (0.00232623062315804) [X0 Z1 X2 Z13]
+ (0.00232623062315804) [Y1 Z2 Y3 Z12]
+ (0.00232623062315804) [X1 Z2 X3 Z12]
+ (0.002929768674750999) [Y0 Z1 Y2 Z9]
+ (0.002929768674750999) [X0 Z1 X2 Z9]
+ (0.002929768674750999) [Y1 Z2 Y3 Z8]
+ (0.002929768674750999) [X1 Z2 X3 Z8]
+ (0.0032769719312315845) [Y0 Z1 Y2 Z3]
+ (0.0032769719312315845) [X0 Z1 X2 Z3]
+ (0.0033476175306661627) [Y0 Z1 Y2 Z7]
+ (0.0033476175306661627) [X0 Z1 X2 Z7]
+ (0.0033476175306661627) [Y1 Z2 Y3 Z6]
+ (0.0033476175306661627) [X1 Z2 X3 Z6]
+ (0.003555290195504176) [Y0 Z1 Y2 Z10]
+ (0.003555290195504176) [X0 Z1 X2 Z10]
+ (0.003555290195504176) [Y1 Z2 Y3 Z11]
+ (0.003555290195504176) [X1 Z2 X3 Z11]
+ (0.005143391768825054) [Y3 Y4 X5 X6]
+ (0.005143391768825054) [X3 X4 Y5 Y6]
+ (0.005283776488402969) [Y0 X1 X12 Y13]
+ (0.005283776488402969) [X0 Y1 Y12 X13]
+ (0.0055307592186314945) [Y0 Z1 Y2 Z4]
+ (0.0055307592186314945) [X0 Z1 X2 Z4]
+ (0.0055307592186314945) [Y1 Z2 Y3 Z5]
+ (0.0055307592186314945) [X1 Z2 X3 Z5]
+ (0.006087822480561867) [Y8 X9 X12 Y13]
+ (0.006087822480561867) [X8 Y9 Y12 X13]
+ (0.0065093612011772415) [Y0 X1 X8 Y9]
+ (0.0065093612011772415) [X0 Y1 Y8 X9]
+ (0.006888194352970598) [Y0 X1 X6 Y7]
+ (0.006888194352970598) [X0 Y1 Y6 X7]
+ (0.00690123824979725) [Y0 Z1 Y2 Z12]
+ (0.00690123824979725) [X0 Z1 X2 Z12]
+ (0.00690123824979725) [Y1 Z2 Y3 Z13]
+ (0.00690123824979725) [X1 Z2 X3 Z13]
+ (0.0071569349198569426) [Y4 X5 X8 Y9]
+ (0.0071569349198569426) [X4 Y5 Y8 X9]
+ (0.007731425250775282) [Y0 X1 X10 Y11]
+ (0.007731425250775282) [X0 Y1 Y10 X11]
+ (0.008032520918821381) [Y0 Z1 Y2 Z6]
+ (0.008032520918821381) [X0 Z1 X2 Z6]
+ (0.008032520918821381) [Y1 Z2 Y3 Z7]
+ (0.008032520918821381) [X1 Z2 X3 Z7]
+ (0.009560705729135909) [Y8 X9 X10 Y11]
+ (0.009560705729135909) [X8 Y9 Y10 X11]
+ (0.011055020596132038) [Y0 Z1 Y2 Z8]
+ (0.011055020596132038) [X0 Z1 X2 Z8]
+ (0.011055020596132038) [Y1 Z2 Y3 Z9]
+ (0.011055020596132038) [X1 Z2 X3 Z9]
+ (0.011285190200840886) [Y5 Y6 X11 X12]
+ (0.011285190200840886) [X5 X6 Y11 Y12]
+ (0.011307274008848001) [Y7 Z8 Z9 Y11]
+ (0.011307274008848001) [X7 Z8 Z9 X11]
+ (0.011982389010247906) [Y4 X5 X6 Y7]
+ (0.011982389010247906) [X4 Y5 Y6 X7]
+ (0.013873381748426157) [Y6 X7 X8 Y9]
+ (0.013873381748426157) [X6 Y7 Y8 X9]
+ (0.014583648907612667) [Y0 X1 X2 Y3]
+ (0.014583648907612667) [X0 Y1 Y2 X3]
+ (0.015577208063976444) [Y2 X3 X12 Y13]
+ (0.015577208063976444) [X2 Y3 Y12 X13]
+ (0.017366118994651333) [Y6 X7 X12 Y13]
+ (0.017366118994651333) [X6 Y7 Y12 X13]
+ (0.017680067952481643) [Y4 X5 X10 Y11]
+ (0.017680067952481643) [X4 Y5 Y10 X11]
+ (0.017825140995786342) [Y6 X7 X10 Y11]
+ (0.017825140995786342) [X6 Y7 Y10 X11]
+ (0.01902824244384736) [Y3 X4 X11 Y12]
+ (0.01902824244384736) [X3 Y4 Y11 X12]
+ (0.025384657508457423) [Y2 X3 X10 Y11]
+ (0.025384657508457423) [X2 Y3 Y10 X11]
+ (0.028685183716105983) [Y10 X11 X12 Y13]
+ (0.028685183716105983) [X10 Y11 Y12 X13]
+ (0.029812424517345622) [Y6 Z7 Z8 Y10]
+ (0.029812424517345622) [X6 Z7 Z8 X10]
+ (0.029812424517345622) [Y7 Z9 Z10 Y11]
+ (0.029812424517345622) [X7 Z9 Z10 X11]
+ (0.030104623143456702) [Y6 Z7 Z9 Y10]
+ (0.030104623143456702) [X6 Z7 Z9 X10]
+ (0.030104623143456702) [Y7 Z8 Z10 Y11]
+ (0.030104623143456702) [X7 Z8 Z10 X11]
+ (0.03078750538914391) [Y6 Z8 Z9 Y10]
+ (0.03078750538914391) [X6 Z8 Z9 X10]
+ (0.031143817988967013) [Y2 X3 X6 Y7]
+ (0.031143817988967013) [X2 Y3 Y6 X7]
+ (0.03583956795335349) [Y2 X3 X4 Y5]
+ (0.03583956795335349) [X2 Y3 Y4 X5]
+ (0.036194123559042606) [Y2 X3 X8 Y9]
+ (0.036194123559042606) [X2 Y3 Y8 X9]
+ (0.03831467029480389) [Y4 X5 X12 Y13]
+ (0.03831467029480389) [X4 Y5 Y12 X13]
+ (0.10433064780651398) [Z0 Y1 Z2 Y3]
+ (0.10433064780651398) [Z0 X1 Z2 X3]
+ (-0.12133276911042297) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133276911042297) [X2 Z3 Z4 Z5 X6]
+ (-0.12133276911042297) [Y3 Z4 Z5 Z6 Y7]
+ (-0.12133276911042297) [X3 Z4 Z5 Z6 X7]
+ (3.2020768780125086e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.2020768780125086e-06) [X1 Z2 Z3 Z4 X5]
+ (3.202076878012509e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.202076878012509e-06) [X0 Z1 Z2 Z3 X4]
+ (0.2284810656491865) [Y7 Z8 Z9 Z10 Y11]
+ (0.2284810656491865) [X7 Z8 Z9 Z10 X11]
+ (0.22848106564918663) [Y6 Z7 Z8 Z9 Y10]
+ (0.22848106564918663) [X6 Z7 Z8 Z9 X10]
+ (-0.03276765782329023) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.03276765782329023) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.03276765782329023) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.03276765782329023) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845272885) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.027115036845272885) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.027115036845272885) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.027115036845272885) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.025996177598021242) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.025996177598021242) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.017561202409646034) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.017561202409646034) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.017561202409646034) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.017561202409646034) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.01456453123117294) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.01456453123117294) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.01456453123117294) [X7 Z8 Z9 X10 X12 X13]
+ (-0.01456453123117294) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.012215040997613875) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.012215040997613875) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.012215040997613875) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.012215040997613875) [X4 Z5 X6 X11 Z12 X13]
+ (-0.012215040997613875) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.012215040997613875) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.012215040997613875) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.012215040997613875) [X5 Z6 X7 X10 Z11 X12]
+ (-0.011756013419819269) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.011756013419819269) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.011756013419819269) [X3 Z4 Z5 X6 X8 X9]
+ (-0.011756013419819269) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.008764827575688786) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.008764827575688786) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.008764827575688786) [X2 Z3 Z4 X5 X11 X12]
+ (-0.008764827575688786) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.008764827575688786) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.008764827575688786) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.008764827575688786) [X3 X4 X10 Z11 Z12 X13]
+ (-0.008764827575688786) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.00812525192138104) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.00812525192138104) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.007306759928832965) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.007306759928832965) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.007306759928832965) [X4 X5 X7 Z8 Z9 X10]
+ (-0.007306759928832965) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.005805188989826766) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.005805188989826766) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.005805188989826766) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.005805188989826766) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.005652620978017344) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.005652620978017344) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.005652620978017344) [X0 X1 X3 Z4 Z5 X6]
+ (-0.005652620978017344) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.005143391768825053) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.005143391768825053) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.005143391768825053) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.005143391768825053) [X2 Z3 X4 X5 Z6 X7]
+ (-0.00468490338815522) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.00468490338815522) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668620318776306) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.004668620318776306) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.00457500762663921) [Y0 Z1 Z2 Y3 X12 X13]
+ (-0.00457500762663921) [X0 Z1 Z2 X3 Y12 Y13]
+ (-0.004424855449441858) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.004424855449441858) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.004158797381840092) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.004158797381840092) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.004158797381840092) [X3 Z4 Z5 X6 X12 X13]
+ (-0.004158797381840092) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.003493790359890113) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.003493790359890113) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.003493790359890113) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.003493790359890113) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.0027790267990255805) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.0027790267990255805) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.0022939566113524806) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.0022939566113524806) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.0017992194936630014) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.0017992194936630014) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.0017278753941369625) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.0017278753941369625) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.0009298507967729903) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298507967729903) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298507967729903) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298507967729903) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533856254125563) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.0008533856254125563) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.0008145313270955783) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.0008145313270955783) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.0008145313270955783) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.0008145313270955783) [X3 Z4 Z5 Z6 X7 Z11]
+ (-7.73503688059188e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.73503688059188e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.73503688059188e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.73503688059188e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-8.774817864262598e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.774817864262598e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.774817864262598e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.774817864262598e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.518362215416034e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.518362215416034e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.518362215416034e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.518362215416034e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.4443446756618745e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.4443446756618745e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.4443446756618745e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.4443446756618745e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.5243738482331295e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.5243738482331295e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.5243738482331295e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.5243738482331295e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-6.290028433008552e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.290028433008552e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.290028433008552e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.290028433008552e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.974311713150524e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (-5.974311713150524e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (-5.275883121856691e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.275883121856691e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.643051068289432e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.643051068289432e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.556569217956339e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (-4.556569217956339e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (-4.253224225392209e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.253224225392209e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-3.769659451852755e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.769659451852755e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.6945132943900423e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.6945132943900423e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.610297130427475e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.610297130427475e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.610297130427475e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.610297130427475e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.3131454999448e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.3131454999448e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.2774831953989217e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.2774831953989217e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.2774831953989217e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.2774831953989217e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.2112283482883298e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.2112283482883298e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.2112283482883298e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.2112283482883298e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.1513463110288548e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.1513463110288548e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.0882507111723583e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.0882507111723583e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.1726691013436853e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.1726691013436853e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.4548424489452753e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.4548424489452753e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.3304731886007227e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.3304731886007227e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.2283337824074822e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.2283337824074822e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.0358477600709027e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.0358477600709027e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-7.956895371639313e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.956895371639313e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.73319774223291e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.73319774223291e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.73319774223291e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.73319774223291e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.628614201115235e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.628614201115235e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.556281914420754e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.556281914420754e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.556281914420754e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.556281914420754e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.418291574490642e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.418291574490642e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.418291574490642e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.418291574490642e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.927453082758287e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.927453082758287e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.927453082758287e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.927453082758287e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.62785191116946e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.62785191116946e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.287660624560351e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.287660624560351e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.287660624560351e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.287660624560351e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.287660624560351e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.287660624560351e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.287660624560351e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.287660624560351e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.838052750563902e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (-4.838052750563902e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (-3.5707613285237446e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.5707613285237446e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.328139350285532e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.328139350285532e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.08682656506743e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.08682656506743e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.08682656506743e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.08682656506743e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.447323128549443e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (-2.447323128549443e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (-2.3713289476593544e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.3713289476593544e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.3713289476593544e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.3713289476593544e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.1990516183327606e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-2.1990516183327606e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-1.9332412769828782e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.9332412769828782e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.9332412769828782e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.9332412769828782e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.8394209152973415e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.8394209152973415e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.8394209152973415e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.8394209152973415e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.5510539176012078e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (-1.5510539176012078e-07) [X0 Z1 X2 X4 Z5 X6]
+ (-1.5510539176012078e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (-1.5510539176012078e-07) [X1 Z2 X3 X5 Z6 X7]
+ (-1.3807781479930967e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (-1.3807781479930967e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (-1.3807781479930967e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (-1.3807781479930967e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (-1.3807781479930967e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (-1.3807781479930967e-07) [X0 Z1 X2 X5 Z6 X7]
+ (-1.3807781479930967e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (-1.3807781479930967e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (-1.3807781479930967e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (-1.3807781479930967e-07) [X1 Z2 X3 X4 Z5 X6]
+ (-1.3807781479930967e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (-1.3807781479930967e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (-1.2919694862656315e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.2919694862656315e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.107632559945161e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.107632559945161e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.107632559945161e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.107632559945161e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.107632559945161e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.107632559945161e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.107632559945161e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.107632559945161e-07) [X1 Z2 X3 X10 Z11 X12]
+ (8.057446594746224e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.057446594746224e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.057446594746224e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.057446594746224e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.649310131748746e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.649310131748746e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.649310131748746e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.649310131748746e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.8394209152973415e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.8394209152973415e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.8394209152973415e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.8394209152973415e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.1990516183327606e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (2.1990516183327606e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (2.447323128549443e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (2.447323128549443e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.2362599608342287e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.2362599608342287e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.2362599608342287e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.2362599608342287e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.328139350285532e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.328139350285532e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.5707613285237446e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.5707613285237446e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.838052750563902e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (4.838052750563902e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (5.62785191116946e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.62785191116946e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (6.628614201115235e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.628614201115235e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (7.956895371639313e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.956895371639313e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.306536651200704e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.306536651200704e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.306536651200704e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.306536651200704e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.0358477600709027e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.0358477600709027e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.2283337824074822e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.2283337824074822e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.2393363216268134e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.2393363216268134e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.2393363216268134e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.2393363216268134e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.3304731886007227e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.3304731886007227e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.4548424489452753e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.4548424489452753e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.1726691013436853e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.1726691013436853e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.0882507111723583e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.0882507111723583e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.1174479456272065e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.1174479456272065e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.1513463110288548e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.1513463110288548e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.3131454999448e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.3131454999448e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.334331289187871e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.334331289187871e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.6945132943900423e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.6945132943900423e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (4.183932559164548e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.183932559164548e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.556569217956339e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (4.556569217956339e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (4.643051068289432e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.643051068289432e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.275883121856691e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.275883121856691e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.974311713150524e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (5.974311713150524e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (0.00029219862611108607) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.00029219862611108607) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.00029219862611108607) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.00029219862611108607) [X6 Z7 X8 X9 Z10 X11]
+ (0.0004956762314918961) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.0004956762314918961) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.000665007021949979) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.000665007021949979) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.000665007021949979) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.000665007021949979) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.0008533856254125563) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.0008533856254125563) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.0016095313817213784) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.0016095313817213784) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.0016095313817213784) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.0016095313817213784) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.0016676041811440807) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.0016676041811440807) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.0016676041811440807) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.0016676041811440807) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.0017278753941369625) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.0017278753941369625) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.0017992194936630014) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.0017992194936630014) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.0022939566113524806) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.0022939566113524806) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.0024629170071339343) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.0024629170071339343) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.0024629170071339343) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.0024629170071339343) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.003961560792496561) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.003961560792496561) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.003961560792496561) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.003961560792496561) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.004424855449441858) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.004424855449441858) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.00457500762663921) [Y0 Z1 Z2 X3 X12 Y13]
+ (0.00457500762663921) [X0 Z1 Z2 Y3 Y12 X13]
+ (0.004668620318776306) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.004668620318776306) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.00468490338815522) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.00468490338815522) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.005324835234221668) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.005324835234221668) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.005324835234221668) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.005324835234221668) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.005368659358109504) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.005368659358109504) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.005368659358109504) [X2 X3 X7 Z8 Z9 X10]
+ (0.005368659358109504) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.007960880725921556) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.007960880725921556) [X4 Z5 X6 X10 Z11 X12]
+ (0.007960880725921556) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.007960880725921556) [X5 Z6 X7 X11 Z12 X13]
+ (0.00812525192138104) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.00812525192138104) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.008890731522694543) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.008890731522694543) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.008890731522694543) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.008890731522694543) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.010263414868158573) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.010263414868158573) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.010263414868158573) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.010263414868158573) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.010540425907671508) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.010540425907671508) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.010540425907671508) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.010540425907671508) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.010960074940542516) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.010960074940542516) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.010960074940542516) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.010960074940542516) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.011307274008848001) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.011307274008848001) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.014411099430130999) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.014411099430130999) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.014411099430130999) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.014411099430130999) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.01522563075722658) [Y3 Z4 Z5 X6 X10 Y11]
+ (0.01522563075722658) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (0.01522563075722658) [X3 Z4 Z5 X6 X10 X11]
+ (0.01522563075722658) [X3 Z4 Z5 Y6 Y10 X11]
+ (0.015588250102380245) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (0.015588250102380245) [X2 Z3 X4 X10 Z11 X12]
+ (0.015588250102380245) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (0.015588250102380245) [X3 Z4 X5 X11 Z12 X13]
+ (0.01826683486937548) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.01826683486937548) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.01826683486937548) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.01826683486937548) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.019020423173039865) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.019020423173039865) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.019020423173039865) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.019020423173039865) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.02017592172353543) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.02017592172353543) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.02017592172353543) [X4 Z5 Z6 X7 X11 X12]
+ (0.02017592172353543) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.02017592172353543) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.02017592172353543) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.02017592172353543) [X5 X6 X10 Z11 Z12 X13]
+ (0.02017592172353543) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.024353077678069025) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.024353077678069025) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.024353077678069025) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.024353077678069025) [X2 Z3 X4 X11 Z12 X13]
+ (0.024353077678069025) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.024353077678069025) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.024353077678069025) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.024353077678069025) [X3 Z4 X5 X10 Z11 X12]
+ (0.02438908253114937) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.02438908253114937) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.02438908253114937) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.02438908253114937) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.025104957138844447) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.025104957138844447) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.025104957138844447) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.025104957138844447) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.03078750538914391) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.03078750538914391) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.04587947078129811) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.04587947078129811) [X0 Z2 Z3 Z4 Z5 X6]
+ (0.0560073308778074) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.0560073308778074) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.0560073308778074) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.0560073308778074) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661331) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.05608468124661331) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661331) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.05608468124661331) [Z1 X7 Z8 Z9 Z10 X11]
+ (-6.631277928202286e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.631277928202286e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.631277928202283e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.631277928202283e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.5950860068828403e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.5950860068828403e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.5950860068828396e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.5950860068828396e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.042743277013783186) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.042743277013783186) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (0.042743277013783186) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.042743277013783186) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.0476426121763831) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.0476426121763831) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.0476426121763831) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.0476426121763831) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.04171881383982176) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.04171881383982176) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.04171881383982176) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.04171881383982176) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.03956441632289357) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.03956441632289357) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.03956441632289357) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03956441632289357) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03935916802205311) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.03935916802205311) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.03935916802205311) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.03935916802205311) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.03931805194719766) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03931805194719766) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.03931805194719766) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03931805194719766) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03560837898831263) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.03560837898831263) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.02990378951262484) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.02990378951262484) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.02990378951262484) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.02990378951262484) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.02873077955190555) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.02873077955190555) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.02873077955190555) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.02873077955190555) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.025637238296026814) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.025637238296026814) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.025637238296026814) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.025637238296026814) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.02475546329289101) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.02475546329289101) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.02475546329289101) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.02475546329289101) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.02428211735469279) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.02428211735469279) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.02314513092952911) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.02314513092952911) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.022528440196012967) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.022528440196012967) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.02143381072160104) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.02143381072160104) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.02143381072160104) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.02143381072160104) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.01925750509525152) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.01925750509525152) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.01902824244384736) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.01902824244384736) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.018889030304942836) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-0.018889030304942836) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-0.018889030304942836) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.018889030304942836) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.016024603689179674) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.016024603689179674) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.01522563075722658) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.01522563075722658) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.014603704729162104) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.014603704729162104) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.01456453123117294) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.01456453123117294) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.011756013419819269) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.011756013419819269) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.011285190200840886) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (-0.011285190200840886) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (-0.009841749246962716) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.009841749246962716) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.009612634606847144) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.009612634606847144) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.009612634606847144) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.009612634606847144) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.008469978791023803) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.008469978791023803) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.007306759928832965) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.007306759928832965) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005923798336561347) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.005923798336561347) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.005652620978017344) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.005652620978017344) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.005368659358109504) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005368659358109504) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.004158797381840092) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.004158797381840092) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0033566705638329013) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.0033566705638329013) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.0033566705638329013) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.0033566705638329013) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.003267513854423562) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.003267513854423562) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.003267513854423562) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.003267513854423562) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.002779026799025581) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.002779026799025581) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.002686040977806606) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.002686040977806606) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.002686040977806606) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.002686040977806606) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.0022939566113524806) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939566113524806) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.0022939566113524806) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.0022939566113524806) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.0009581655836696434) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.0009581655836696434) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.0009581655836696434) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.0009581655836696434) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.0009581655836696434) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.0009581655836696434) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.0009581655836696434) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.0009581655836696434) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.0002463643756959152) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0002463643756959152) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303549653) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.00013840177303549653) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.00013840177303549653) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.00013840177303549653) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.73503688059188e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.73503688059188e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585305065447e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585305065447e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.6103585305065447e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.6103585305065447e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808794701338e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.5316808794701338e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808794701338e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5316808794701338e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.806102774765383e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.806102774765383e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.806102774765383e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.806102774765383e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.089799467262292e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.089799467262292e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.089799467262292e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.089799467262292e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.652209669233543e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.652209669233543e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.652209669233543e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.652209669233543e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851833697793e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.481851833697793e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851833697793e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.481851833697793e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.071480736216837e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.071480736216837e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.071480736216837e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.071480736216837e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.7346220385485445e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.7346220385485445e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.7346220385485445e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.7346220385485445e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.728843147082269e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.728843147082269e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.728843147082269e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.728843147082269e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.253224225392209e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.253224225392209e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.769659451852755e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.769659451852755e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.5443954291129914e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.5443954291129914e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.5443954291129914e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.5443954291129914e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.5443954291129914e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.5443954291129914e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.5443954291129914e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.5443954291129914e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.360956320180023e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.360956320180023e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.360956320180023e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.360956320180023e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-2.103215604589685e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.103215604589685e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.103215604589685e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.103215604589685e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220981369225e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.0111220981369225e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220981369225e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.0111220981369225e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.942946836580832e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.942946836580832e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.942946836580832e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.942946836580832e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174770475996e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.6541174770475996e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174770475996e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.6541174770475996e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.5224930675897748e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.5224930675897748e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.5224930675897748e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.5224930675897748e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.5224930675897748e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930675897748e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930675897748e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.5224930675897748e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.2283337824074822e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337824074822e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.2283337824074822e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337824074822e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.98877028787881e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.98877028787881e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.98877028787881e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.98877028787881e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.867765103641068e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765103641068e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765103641068e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.867765103641068e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.189990974949324e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.189990974949324e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.175246206677877e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175246206677877e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.471647744460724e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.471647744460724e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.5614471794063334e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.5614471794063334e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.5614471794063334e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.5614471794063334e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.523389677786612e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.523389677786612e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.427323108472475e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.427323108472475e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.427323108472475e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.427323108472475e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.328139350285532e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350285532e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.328139350285532e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350285532e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.08682656506743e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.08682656506743e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.888293595332326e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.888293595332326e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.888293595332326e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.888293595332326e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.3713289476593544e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.3713289476593544e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.8394209152973415e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.8394209152973415e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-8.057446594746223e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.057446594746223e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (4.537178097481405e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.537178097481405e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.537178097481405e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (4.537178097481405e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (8.057446594746223e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.057446594746223e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (9.209350645276272e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350645276272e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350645276272e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.209350645276272e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.703578355357505e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.703578355357505e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.703578355357505e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.703578355357505e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.8394209152973415e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.8394209152973415e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.3713289476593544e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.3713289476593544e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (3.08682656506743e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.08682656506743e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (4.523389677786612e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.523389677786612e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (5.471647744460724e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.471647744460724e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.175246206677877e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175246206677877e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.189990974949324e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.189990974949324e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.3304731886007227e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.3304731886007227e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.3304731886007227e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.3304731886007227e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.6288532434390795e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.6288532434390795e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.6288532434390795e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.6288532434390795e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.6893489514055288e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.6893489514055288e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.6893489514055288e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.6893489514055288e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (2.7455184003251116e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.7455184003251116e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.7455184003251116e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.7455184003251116e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.7455184003251116e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.7455184003251116e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.7455184003251116e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.7455184003251116e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.2118420189953036e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420189953036e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.2118420189953036e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420189953036e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.2118420189953036e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420189953036e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.2118420189953036e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420189953036e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.3131454999448e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.3131454999448e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.3131454999448e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.3131454999448e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (3.334331289187871e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.334331289187871e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (4.183932559164549e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.183932559164549e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (7.73503688059188e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.73503688059188e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.0002463643756959152) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.0002463643756959152) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00044585351288406894) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.00044585351288406894) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.00044585351288406894) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.00044585351288406894) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.000594022154300526) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.000594022154300526) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.000594022154300526) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.000594022154300526) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.000594022154300526) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.000594022154300526) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.000594022154300526) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.000594022154300526) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.0008533856254125562) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.0008533856254125562) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.0008533856254125562) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.0008533856254125562) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.0010435246534907531) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.0010435246534907531) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.0010435246534907531) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.0010435246534907531) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.0012803060973496669) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.0012803060973496669) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.0012803060973496669) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.0012803060973496669) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.0013038004788127077) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (0.0013038004788127077) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (0.0013038004788127077) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (0.0013038004788127077) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (0.002261966062482351) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (0.002261966062482351) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (0.002261966062482351) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (0.002261966062482351) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (0.002261966062482351) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (0.002261966062482351) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (0.002261966062482351) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (0.002261966062482351) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (0.003989841456619314) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (0.003989841456619314) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (0.003989841456619314) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (0.003989841456619314) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (0.004158797381840092) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.004158797381840092) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.004311038507914315) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.004311038507914315) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.004311038507914315) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.004311038507914315) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.004636976661182568) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.004636976661182568) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.004636976661182568) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.004636976661182568) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.005114473831660375) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.005114473831660375) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.005114473831660375) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.005114473831660375) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.005114473831660375) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660375) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660375) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005114473831660375) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.005241535382803864) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (0.005241535382803864) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (0.005241535382803864) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (0.005241535382803864) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (0.005262642473076832) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.005262642473076832) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.005262642473076832) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.005262642473076832) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.005368659358109504) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005368659358109504) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005379937155839361) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.005379937155839361) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.005379937155839361) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.005379937155839361) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.005652620978017344) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.005652620978017344) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.005708495985960901) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.005708495985960901) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.005708495985960901) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.005708495985960901) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.005923798336561347) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.005923798336561347) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.007306759928832965) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.007306759928832965) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.008469978791023803) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.008469978791023803) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.009841749246962716) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.009841749246962716) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.011285190200840886) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (0.011285190200840886) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (0.011756013419819269) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.011756013419819269) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.01456453123117294) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.01456453123117294) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.014603704729162104) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.014603704729162104) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.01522563075722658) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.01522563075722658) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.016024603689179674) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.016024603689179674) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.01902824244384736) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.01902824244384736) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.01925750509525152) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.01925750509525152) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.04587947078129811) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04587947078129811) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.36937089366156206) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.36937089366156206) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.36937089366156195) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.36937089366156195) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.2816425776702287) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.2816425776702287) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.2816425776702286) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.2816425776702286) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.09065144207036478) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.09065144207036478) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.09065144207036478) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.09065144207036478) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863624) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.08684737589863624) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863624) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.08684737589863624) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.07635021950635024) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07635021950635024) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07635021950635024) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07635021950635024) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214037) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.06752385099214037) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214037) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.06752385099214037) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03560837898831263) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.03560837898831263) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.034903343373661744) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.034903343373661744) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.034903343373661744) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.034903343373661744) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.02459186088382996) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.02459186088382996) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.02459186088382996) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.02459186088382996) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.024282117354692788) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.024282117354692788) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.023145130929529113) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.023145130929529113) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.022528440196012967) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.022528440196012967) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0195380503113147) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.0195380503113147) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.0195380503113147) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.0195380503113147) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.01709155315589884) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.01709155315589884) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.01709155315589884) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.01709155315589884) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.016024603689179674) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.016024603689179674) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.016024603689179674) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.016024603689179674) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.010311482489831787) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831787) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831787) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010311482489831787) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.009841749246962716) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962716) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.009841749246962716) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962716) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209867) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209867) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209867) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008826368514209867) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008541996625454816) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454816) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454816) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454816) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454816) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454816) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454816) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008541996625454816) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008469978791023803) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023803) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.008469978791023803) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023803) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.004668620318776306) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.004668620318776306) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.003876470899336949) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.003876470899336949) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.003804066171728543) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728543) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728543) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003804066171728543) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178878) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178878) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0033566705638329013) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.0033566705638329013) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.003267513854423562) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.003267513854423562) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.002141361223101596) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.002141361223101596) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.0017278753941369625) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.0017278753941369625) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.001640754855312397) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.001640754855312397) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214169371) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0014528843214169371) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214169371) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0014528843214169371) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870896771024419) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007870896771024419) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0005192743499487755) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.0005192743499487755) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.00019400857029756754) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00019400857029756754) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303549653) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.00013840177303549653) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.141625221157337e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.141625221157337e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.141625221157337e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.141625221157337e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-5.071480736216836e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.071480736216836e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.1513463110288548e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.1513463110288548e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0882507111723583e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.0882507111723583e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.9885117060777137e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.9885117060777137e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.874299071209929e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.874299071209929e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.360956320180023e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.360956320180023e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.3002946561386982e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.3002946561386982e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.1468376506991562e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.1468376506991562e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.1468376506991562e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.1468376506991562e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.352332102822141e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.352332102822141e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.352332102822141e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.352332102822141e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.091637198713943e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637198713943e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637198713943e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637198713943e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637198713943e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637198713943e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637198713943e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.091637198713943e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.074305985613127e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.074305985613127e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.074305985613127e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.074305985613127e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.900128986064232e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.900128986064232e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.900128986064232e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900128986064232e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867765103641068e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.867765103641068e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.560692464564735e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692464564735e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692464564735e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692464564735e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692464564735e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692464564735e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692464564735e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.560692464564735e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.997018421846759e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.997018421846759e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.997018421846759e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.997018421846759e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.997018421846759e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.997018421846759e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.997018421846759e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.997018421846759e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.568247520927332e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.568247520927332e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.568247520927332e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.568247520927332e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393082776204e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393082776204e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393082776204e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.3767393082776204e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.3767393082776204e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393082776204e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.3767393082776204e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.3767393082776204e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.888293595332326e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.888293595332326e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6863815459494575e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.6863815459494575e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.703578355357505e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.703578355357505e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.209350645276272e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.209350645276272e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244606031e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773244606031e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244606031e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773244606031e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244606031e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773244606031e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.379773244606031e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773244606031e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.974225379619989e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.974225379619989e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.974225379619989e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.974225379619989e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.047471655581031e-08) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.047471655581031e-08) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.047471655581031e-08) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.047471655581031e-08) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.209350645276272e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.209350645276272e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0717282183516483e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.0717282183516483e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.0717282183516483e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.0717282183516483e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.2004287493260856e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.2004287493260856e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.2004287493260856e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.2004287493260856e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.703578355357505e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.703578355357505e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.3120943051055926e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.3120943051055926e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.3120943051055926e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.3120943051055926e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.6863815459494575e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815459494575e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.888293595332326e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.888293595332326e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.0922506158733577e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506158733577e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.0922506158733577e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506158733577e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.0922506158733577e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506158733577e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.0922506158733577e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506158733577e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.444597853789712e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.444597853789712e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.444597853789712e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.444597853789712e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.6849150948762925e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.6849150948762925e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.6849150948762925e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.6849150948762925e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.246974425029525e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.246974425029525e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.246974425029525e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.246974425029525e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.246974425029525e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.246974425029525e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.246974425029525e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.246974425029525e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.867765103641068e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.867765103641068e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.3002946561386982e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.3002946561386982e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.360956320180023e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.360956320180023e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.874299071209929e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.874299071209929e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.8836765758443253e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.8836765758443253e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.947356011531591e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.947356011531591e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.947356011531591e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.947356011531591e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117060777137e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9885117060777137e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.0882507111723583e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.0882507111723583e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.1513463110288548e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.1513463110288548e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.8462016710377314e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.8462016710377314e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.8462016710377314e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.8462016710377314e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.071480736216836e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.071480736216836e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.105526721681878e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.105526721681878e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.105526721681878e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.105526721681878e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.14649632717643e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.14649632717643e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.14649632717643e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.14649632717643e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.159350501571747e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.159350501571747e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.159350501571747e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.159350501571747e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.427988656166692e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.427988656166692e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.427988656166692e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.427988656166692e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.9358677176093046e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.9358677176093046e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.9358677176093046e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.9358677176093046e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273347659826e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.253273347659826e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.979825792891807e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.979825792891807e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.979825792891807e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.979825792891807e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (4.2055484112206096e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.2055484112206096e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.2055484112206096e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.2055484112206096e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00013840177303549653) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.00013840177303549653) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.00018787053389545966) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.00018787053389545966) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.00018787053389545966) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.00018787053389545966) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00019400857029756754) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00019400857029756754) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00024636437569591524) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024636437569591524) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00024636437569591524) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024636437569591524) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192743499487755) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.0005192743499487755) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.0007156734248908686) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007156734248908686) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0007156734248908686) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007156734248908686) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024419) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007870896771024419) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.0015324835230730244) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.0015324835230730244) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.0015324835230730244) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.0015324835230730244) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.001640754855312397) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.001640754855312397) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0017278753941369625) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.0017278753941369625) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.002446497155415856) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.002446497155415856) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.002446497155415856) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.002446497155415856) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.003267513854423562) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.003267513854423562) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.0033566705638329013) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.0033566705638329013) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.0034841573002178878) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0034841573002178878) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.003876470899336949) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.003876470899336949) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.004668620318776306) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.004668620318776306) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.0047672721882781165) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.0047672721882781165) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.0047672721882781165) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.0047672721882781165) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.0052865465382268915) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.0052865465382268915) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.0052865465382268915) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.0052865465382268915) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.005408954422409974) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.005408954422409974) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.005408954422409974) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.005408954422409974) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.005923798336561347) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561347) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.005923798336561347) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561347) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.010715508469796702) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010715508469796702) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010715508469796702) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010715508469796702) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.010757563953908903) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010757563953908903) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010757563953908903) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010757563953908903) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.014603704729162103) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.014603704729162103) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.014603704729162103) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.014603704729162103) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.01929956057936372) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01929956057936372) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01929956057936372) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01929956057936372) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01929956057936372) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01929956057936372) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.01929956057936372) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01929956057936372) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.05859198873386194) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.05859198873386194) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.7759505268264274e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.7759505268264274e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.7759505268264274e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.7759505268264274e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0716503518100276) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0716503518100276) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.07165035181002763) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.07165035181002763) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.01925750509525152) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.01925750509525152) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010311482489831787) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311482489831787) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008826368514209867) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209867) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.00759746402977061) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00759746402977061) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.00759746402977061) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00759746402977061) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311885) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311885) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311885) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311885) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311885) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311885) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311885) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311885) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676612) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005348051582676612) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005348051582676612) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676612) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0038040661717285425) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0038040661717285425) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0029841661681219434) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.0029841661681219434) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.0029841661681219434) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.0029841661681219434) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.0024464971554158557) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.0024464971554158557) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.0022494124470939974) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0022494124470939974) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0022494124470939974) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0022494124470939974) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.002141361223101596) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.002141361223101596) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.0018638942824587246) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587246) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587246) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587246) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587246) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587246) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0018638942824587246) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587246) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.001640754855312397) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.001640754855312397) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.001640754855312397) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.001640754855312397) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0012223378081538435) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538435) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538435) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538435) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538435) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538435) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538435) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0012223378081538435) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0010283292378562758) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0010283292378562758) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0010283292378562758) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0010283292378562758) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1463061452561745e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1463061452561745e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.874299071209929e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.874299071209929e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.874299071209929e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.874299071209929e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.3002946561386982e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.3002946561386982e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.3002946561386982e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.3002946561386982e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.0444941297534236e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.0444941297534236e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.0444941297534236e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.0444941297534236e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.956079229527808e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.956079229527808e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.956079229527808e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.956079229527808e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105515036828657e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105515036828657e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105515036828657e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105515036828657e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661347212793782e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661347212793782e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661347212793782e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661347212793782e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540341413266175e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540341413266175e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.189990974949324e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.189990974949324e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.876621657938544e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876621657938544e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876621657938544e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876621657938544e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175246206677877e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175246206677877e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.523389677786612e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.523389677786612e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0767325316306046e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.0767325316306046e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0767325316306046e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.0767325316306046e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.0134714587132056e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0134714587132056e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.9045998842680617e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.9045998842680617e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.9045998842680617e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.9045998842680617e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.6667317544447013e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.6667317544447013e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.6667317544447013e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.6667317544447013e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.85056419269915e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.85056419269915e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309315058546e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.6569309315058546e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309315058546e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.6569309315058546e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.85056419269915e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.85056419269915e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815459494575e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815459494575e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.6863815459494575e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815459494575e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714587132056e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.0134714587132056e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.523389677786612e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.523389677786612e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.67040239021906e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.67040239021906e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.67040239021906e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.67040239021906e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175246206677877e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175246206677877e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.189990974949324e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.189990974949324e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540341413266175e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540341413266175e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.94947648680591e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.94947648680591e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.792493957626609e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.792493957626609e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.792493957626609e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.792493957626609e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.8836765758443253e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.8836765758443253e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.9885117060777133e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117060777133e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117060777133e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117060777133e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273347659826e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.253273347659826e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4017109734463571e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4017109734463571e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4017109734463571e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4017109734463571e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.580960369209018e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.580960369209018e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.580960369209018e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.580960369209018e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0005192743499487754) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487754) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.0005192743499487754) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487754) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.000787089677102442) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.000787089677102442) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.000787089677102442) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.000787089677102442) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.001172634831644182) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.001172634831644182) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.001172634831644182) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.001172634831644182) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012366478019245272) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.0012366478019245272) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.0012366478019245272) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.0012366478019245272) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.002200964069500458) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002200964069500458) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002200964069500458) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002200964069500458) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980253) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980253) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0023949726397980253) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980253) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0023949726397980253) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0023949726397980253) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0023949726397980253) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0023949726397980253) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0024464971554158557) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.0024464971554158557) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.0038040661717285425) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0038040661717285425) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0038764708993369494) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.0038764708993369494) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.0038764708993369494) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.0038764708993369494) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.004220813970046471) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.004220813970046471) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.004220813970046471) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.004220813970046471) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.008826368514209867) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.008826368514209867) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.010311482489831787) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010311482489831787) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.01925750509525152) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01925750509525152) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.05859198873386194) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.05859198873386194) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.3987009016597809e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.3987009016597809e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.3987009016597809e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.3987009016597809e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841573002178878) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841573002178878) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0029841661681219434) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.0029841661681219434) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.00019400857029756754) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00019400857029756754) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061452561745e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061452561745e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.792493957626609e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.792493957626609e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540341413266175e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413266175e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540341413266175e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413266175e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.85056419269915e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.85056419269915e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.85056419269915e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.85056419269915e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714587132056e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.0134714587132056e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714587132056e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.0134714587132056e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.94947648680591e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.94947648680591e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.792493957626609e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.792493957626609e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029756754) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029756754) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0029841661681219434) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.0029841661681219434) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.0034841573002178878) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0034841573002178878) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
Expectation value of XIZ =  0.07715357869738937
 </code>
 </pre>
 </details>

---

## 14. tutorial_general_parshift.html <a name="demo13"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_general_parshift.html):

```
Second-order finite difference:    [ 0.26814   1.696853 -2.055918 -7.236954]
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_general_parshift.html):

```
Second-order finite difference:    [ 0.26814   1.696854 -2.055918 -7.236953]
```

---

## 15. tutorial_vqt.html <a name="demo14"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_vqt.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
 0: ──X──────RZ(1)──RY(1)──RX(1)──╭C─────────────────────────────╭RX(1)──RZ(1)──RY(1)──RX(1)──╭C─────────────────────────────╭RX(1)──RZ(1)──RY(1)──RX(1)──╭C─────────────────────────────╭RX(1)──RZ(1)──RY(1)──RX(1)──╭C──────────────────────╭RX(1)──╭┤ ⟨H0⟩
 1: ──RZ(1)──RY(1)──RX(1)─────────╰RX(1)──╭C───────RZ(1)──RY(1)──│───────RX(1)────────────────╰RX(1)──╭C───────RZ(1)──RY(1)──│───────RX(1)────────────────╰RX(1)──╭C───────RZ(1)──RY(1)──│───────RX(1)────────────────╰RX(1)──╭C──────────────│───────├┤ ⟨H0⟩
 2: ──X──────RZ(1)──RY(1)──RX(1)──────────╰RX(1)──╭C──────RZ(1)──│───────RY(1)──RX(1)─────────────────╰RX(1)──╭C──────RZ(1)──│───────RY(1)──RX(1)─────────────────╰RX(1)──╭C──────RZ(1)──│───────RY(1)──RX(1)─────────────────╰RX(1)──╭C──────│───────├┤ ⟨H0⟩
 3: ──RZ(1)──RY(1)──RX(1)─────────────────────────╰RX(1)─────────╰C──────RZ(1)──RY(1)──RX(1)──────────────────╰RX(1)─────────╰C──────RZ(1)──RY(1)──RX(1)──────────────────╰RX(1)─────────╰C──────RZ(1)──RY(1)──RX(1)──────────────────╰RX(1)──╰C──────╰┤ ⟨H0⟩
Cost at Step 0: -0.6605354666522008
Cost at Step 50: -2.869994162243927
Cost at Step 100: -4.637585392072052
Cost at Step 150: -5.656853560767198
Cost at Step 200: -6.402981632616908
Cost at Step 250: -7.194529795738805
Cost at Step 300: -8.549504336881128
Cost at Step 350: -9.49513193388411
Cost at Step 400: -11.2720655494965
Cost at Step 450: -11.41504314740235
Cost at Step 500: -11.730005054775035
Cost at Step 550: -12.146853224341926
Cost at Step 600: -12.707687382495887
Cost at Step 650: -12.897020710486425
Cost at Step 700: -13.135114176468683
Cost at Step 750: -13.336484236053463
Cost at Step 800: -13.609135353851714
Cost at Step 850: -13.857559868540363
Cost at Step 900: -13.956305337780474
Cost at Step 950: -14.140415777686899
Cost at Step 1000: -14.332958418542795
Cost at Step 1050: -14.429354831907656
Cost at Step 1100: -14.483183357691805
Cost at Step 1150: -14.566207701295353
Cost at Step 1200: -14.625828011364685
Cost at Step 1250: -14.671996831321566
Cost at Step 1300: -14.781784212856454
Cost at Step 1350: -14.81304729246651
Cost at Step 1400: -14.89227387332433
Cost at Step 1450: -14.969660667340055
Cost at Step 1500: -15.01775721812936
Cost at Step 1550: -15.073007676865677
Trace Distance: 0.05858340357994628
 </code>
 </pre>
 </details>

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_vqt.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
 0: ──╭BasisStatePreparation(M0)──╭AngleEmbedding(M1)──╭AngleEmbedding(M1)──╭AngleEmbedding(M1)──╭C──────────────────────╭RX(1)──╭AngleEmbedding(M1)──╭AngleEmbedding(M1)──╭AngleEmbedding(M1)──╭C──────────────────────╭RX(1)──╭AngleEmbedding(M1)──╭AngleEmbedding(M1)──╭AngleEmbedding(M1)──╭C──────────────────────╭RX(1)──╭AngleEmbedding(M1)──╭AngleEmbedding(M1)──╭AngleEmbedding(M1)──╭C──────────────────────╭RX(1)──╭┤ ⟨H0⟩
 1: ──├BasisStatePreparation(M0)──├AngleEmbedding(M1)──├AngleEmbedding(M1)──├AngleEmbedding(M1)──╰RX(1)──╭C──────────────│───────├AngleEmbedding(M1)──├AngleEmbedding(M1)──├AngleEmbedding(M1)──╰RX(1)──╭C──────────────│───────├AngleEmbedding(M1)──├AngleEmbedding(M1)──├AngleEmbedding(M1)──╰RX(1)──╭C──────────────│───────├AngleEmbedding(M1)──├AngleEmbedding(M1)──├AngleEmbedding(M1)──╰RX(1)──╭C──────────────│───────├┤ ⟨H0⟩
 2: ──├BasisStatePreparation(M0)──├AngleEmbedding(M1)──├AngleEmbedding(M1)──├AngleEmbedding(M1)──────────╰RX(1)──╭C──────│───────├AngleEmbedding(M1)──├AngleEmbedding(M1)──├AngleEmbedding(M1)──────────╰RX(1)──╭C──────│───────├AngleEmbedding(M1)──├AngleEmbedding(M1)──├AngleEmbedding(M1)──────────╰RX(1)──╭C──────│───────├AngleEmbedding(M1)──├AngleEmbedding(M1)──├AngleEmbedding(M1)──────────╰RX(1)──╭C──────│───────├┤ ⟨H0⟩
 3: ──╰BasisStatePreparation(M0)──╰AngleEmbedding(M1)──╰AngleEmbedding(M1)──╰AngleEmbedding(M1)──────────────────╰RX(1)──╰C──────╰AngleEmbedding(M1)──╰AngleEmbedding(M1)──╰AngleEmbedding(M1)──────────────────╰RX(1)──╰C──────╰AngleEmbedding(M1)──╰AngleEmbedding(M1)──╰AngleEmbedding(M1)──────────────────╰RX(1)──╰C──────╰AngleEmbedding(M1)──╰AngleEmbedding(M1)──╰AngleEmbedding(M1)──────────────────╰RX(1)──╰C──────╰┤ ⟨H0⟩
M0 =
[1, 0, 1, 0]
M1 =
[1 1 1 1]
Cost at Step 0: -0.660535466652201
Cost at Step 50: -2.869994162243927
Cost at Step 100: -4.642067184244081
Cost at Step 150: -5.127022428216539
Cost at Step 200: -6.5292479970263475
Cost at Step 250: -7.026618536189606
Cost at Step 300: -7.488102103496896
Cost at Step 350: -8.746171514554208
Cost at Step 400: -9.427226863807505
Cost at Step 450: -9.53755662322931
Cost at Step 500: -10.388988996560775
Cost at Step 550: -11.109731977694457
Cost at Step 600: -11.250258039584036
Cost at Step 650: -12.17341222302523
Cost at Step 700: -12.491587447630017
Cost at Step 750: -12.804392175167006
Cost at Step 800: -13.051998470031808
Cost at Step 850: -13.121929958625845
Cost at Step 900: -13.257288210600661
Cost at Step 950: -13.37946499688237
Cost at Step 1000: -13.564827616216016
Cost at Step 1050: -13.648718474396574
Cost at Step 1100: -13.83338377080432
Cost at Step 1150: -13.95826726516021
Cost at Step 1200: -14.088858350361166
Cost at Step 1250: -14.062345740176115
Cost at Step 1300: -14.193332801647184
Cost at Step 1350: -14.25625802821952
Cost at Step 1400: -14.26173795086304
 </code>
 </pre>
 </details>

---

## 16. tutorial_quantum_analytic_descent.html <a name="demo15"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_quantum_analytic_descent.html):

```
 E_C = [-0.1034281 -0.1034281]
 E_D = [[0.         0.29472535]
 [0.         0.        ]]
  Model:    0.15256055642369598
Epoch   50: Model cost = -0.313  at relative parameters [-0.9633  1.1863]
True energy at the minimum of the model: -0.06468825441574239
New reference parameters: [1.6986 5.2446]
Epoch   50: Model cost = -1.1834  at relative parameters [1.4673 1.3564]
True energy at the minimum of the model: -0.9496355879143927
New reference parameters: [3.166 6.601]
Epoch   50: Model cost = -1.0  at relative parameters [-0.0227 -0.3138]
True energy at the minimum of the model: -0.9999905112342359
New reference parameters: [3.1433 6.2872]
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_quantum_analytic_descent.html):

```
 E_C = [0.1034281 0.1034281]
 E_D = [[0. 0.]
 [0. 0.]]
  Model:    0.15227574516483558
Epoch   50: Model cost = -0.2055  at relative parameters [-0.5886  1.5739]
True energy at the minimum of the model: -0.3831433530585694
New reference parameters: [2.0733 5.6322]
Epoch   50: Model cost = -1.1044  at relative parameters [1.5128 0.2699]
True energy at the minimum of the model: -0.8380567686762479
New reference parameters: [3.5861 5.9021]
Epoch   50: Model cost = -1.2806  at relative parameters [-1.0475  0.692 ]
True energy at the minimum of the model: -0.7841668053265736
New reference parameters: [2.5386 6.5941]
```

---

## 17. tutorial_quantum_natural_gradient.html <a name="demo16"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_quantum_natural_gradient.html):

```
[[ 0.125       0.          0.          0.        ]
 [ 0.          0.1875      0.          0.        ]
 [ 0.          0.          0.24973433 -0.01524701]
 [ 0.          0.         -0.01524701  0.20293623]]
[[0.125      0.         0.         0.        ]
 [0.         0.1875     0.         0.        ]
 [0.         0.         0.24973433 0.        ]
 [0.         0.         0.         0.20293623]]
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_quantum_natural_gradient.html):

```
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/transforms/metric_tensor.py:303: UserWarning: The device does not have a wire that is not used by the tape.
Reverting to the block-diagonal approximation. It will often be much more efficient to request the block-diagonal approximation directly!
[[ 0.125       0.          0.          0.        ]
 [ 0.          0.1875      0.          0.        ]
 [ 0.          0.          0.24973433 -0.01524701]
 [ 0.          0.         -0.01524701  0.20293623]]
[[0.125      0.         0.         0.        ]
 [0.         0.1875     0.         0.        ]
```

---

## 18. tutorial_rosalin.html <a name="demo17"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_rosalin.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Step 1: cost = -1.6879973520840041 shots used = 8000
Step 2: cost = -2.437928256197112 shots used = 16000
Step 3: cost = -2.9300968884147647 shots used = 24000
Step 4: cost = -3.7779069617997116 shots used = 32000
Step 5: cost = -3.8889841568955115 shots used = 40000
Step 6: cost = -4.508059711766957 shots used = 48000
Step 7: cost = -4.71114219758592 shots used = 56000
Step 8: cost = -4.984457128293103 shots used = 64000
Step 9: cost = -5.597084424095087 shots used = 72000
Step 10: cost = -5.456976403687039 shots used = 80000
Step 11: cost = -5.736752824027413 shots used = 88000
Step 12: cost = -6.220317925041974 shots used = 96000
Step 13: cost = -6.45162161927903 shots used = 104000
Step 14: cost = -6.563539211112225 shots used = 112000
Step 15: cost = -6.487339064303318 shots used = 120000
Step 16: cost = -6.69261841162329 shots used = 128000
Step 17: cost = -6.909230576241427 shots used = 136000
Step 18: cost = -7.05156660241221 shots used = 144000
Step 19: cost = -7.163688069859358 shots used = 152000
Step 20: cost = -7.191791478058647 shots used = 160000
Step 21: cost = -7.191694602776715 shots used = 168000
Step 22: cost = -7.430122007574104 shots used = 176000
Step 23: cost = -7.245621601209081 shots used = 184000
Step 24: cost = -7.539044265851978 shots used = 192000
Step 25: cost = -7.532847998808006 shots used = 200000
Step 26: cost = -7.44257222073886 shots used = 208000
Step 27: cost = -7.439951968648378 shots used = 216000
Step 28: cost = -7.734568855081575 shots used = 224000
Step 29: cost = -7.618221322585628 shots used = 232000
Step 30: cost = -7.651544920606065 shots used = 240000
Step 31: cost = -7.5069088885777155 shots used = 248000
Step 32: cost = -7.780301321189146 shots used = 256000
Step 33: cost = -7.4456447455856445 shots used = 264000
Step 34: cost = -7.403560444278863 shots used = 272000
Step 35: cost = -7.666718876831026 shots used = 280000
Step 36: cost = -7.7178910518866415 shots used = 288000
Step 37: cost = -7.375680885292107 shots used = 296000
Step 38: cost = -7.665568049279896 shots used = 304000
Step 39: cost = -7.568101693343673 shots used = 312000
Step 40: cost = -7.524188200359864 shots used = 320000
Step 41: cost = -7.525528734255245 shots used = 328000
Step 42: cost = -7.57734861403185 shots used = 336000
Step 43: cost = -7.76844833198197 shots used = 344000
Step 44: cost = -7.797619087079373 shots used = 352000
Step 45: cost = -7.879148884805528 shots used = 360000
Step 46: cost = -7.744030492750696 shots used = 368000
Step 47: cost = -7.6484739221198765 shots used = 376000
Step 48: cost = -7.679623095926702 shots used = 384000
Step 49: cost = -7.607476988501242 shots used = 392000
Step 50: cost = -7.856041856821188 shots used = 400000
Step 51: cost = -7.644473030321983 shots used = 408000
Step 52: cost = -7.593159311741706 shots used = 416000
Step 53: cost = -7.606939212888227 shots used = 424000
Step 54: cost = -7.621128949485829 shots used = 432000
Step 55: cost = -7.743568287057952 shots used = 440000
Step 56: cost = -7.6325929460598525 shots used = 448000
Step 57: cost = -7.718256562367575 shots used = 456000
Step 58: cost = -7.861601938446393 shots used = 464000
Step 59: cost = -7.666115854972354 shots used = 472000
Step 60: cost = -7.644148944168839 shots used = 480000
Step 61: cost = -7.771569192260795 shots used = 488000
Step 62: cost = -7.776898446282362 shots used = 496000
Step 63: cost = -7.711006891533269 shots used = 504000
Step 64: cost = -7.748650044666392 shots used = 512000
Step 65: cost = -7.690723991927554 shots used = 520000
Step 66: cost = -7.694117031088106 shots used = 528000
Step 67: cost = -7.793250125674997 shots used = 536000
Step 68: cost = -7.926049735334674 shots used = 544000
Step 69: cost = -7.686292326080605 shots used = 552000
Step 70: cost = -7.745774212716911 shots used = 560000
Step 71: cost = -7.625346751584894 shots used = 568000
Step 72: cost = -7.846664469958039 shots used = 576000
Step 73: cost = -7.860275655123486 shots used = 584000
Step 74: cost = -7.593043619614097 shots used = 592000
Step 75: cost = -7.7969799318129045 shots used = 600000
Step 76: cost = -7.837545360539077 shots used = 608000
Step 77: cost = -7.845253964960701 shots used = 616000
Step 78: cost = -7.941652692590529 shots used = 624000
Step 79: cost = -7.967099906804574 shots used = 632000
Step 80: cost = -7.803163356121793 shots used = 640000
Step 81: cost = -7.665600401510319 shots used = 648000
Step 82: cost = -8.09158124610039 shots used = 656000
Step 83: cost = -7.774883584668083 shots used = 664000
Step 84: cost = -7.758175214036924 shots used = 672000
Step 85: cost = -7.9169924228411865 shots used = 680000
Step 86: cost = -7.670199051467696 shots used = 688000
Step 87: cost = -8.085682024006845 shots used = 696000
Step 88: cost = -7.8433919424579095 shots used = 704000
Step 89: cost = -7.755236580472145 shots used = 712000
Step 90: cost = -7.847624689390126 shots used = 720000
Step 91: cost = -8.122239105086607 shots used = 728000
Step 92: cost = -7.922374192271718 shots used = 736000
Step 93: cost = -7.904676929818973 shots used = 744000
Step 94: cost = -7.909417248833883 shots used = 752000
Step 95: cost = -8.06033491620787 shots used = 760000
Step 96: cost = -7.765636196903123 shots used = 768000
Step 97: cost = -7.801666008865329 shots used = 776000
Step 98: cost = -8.066513329432457 shots used = 784000
Step 99: cost = -7.8942080196569675 shots used = 792000
Step 0: cost = -0.38250000000000006 shots used = 0
Step 1: cost = -1.7450000000000003 shots used = 8000
Step 2: cost = -2.54875 shots used = 16000
Step 3: cost = -2.91 shots used = 24000
Step 4: cost = -3.4762500000000003 shots used = 32000
Step 5: cost = -4.08875 shots used = 40000
Step 6: cost = -4.586250000000001 shots used = 48000
Step 7: cost = -4.805 shots used = 56000
Step 8: cost = -4.925 shots used = 64000
Step 9: cost = -5.385 shots used = 72000
Step 10: cost = -5.4725 shots used = 80000
Step 11: cost = -5.63875 shots used = 88000
Step 12: cost = -5.79625 shots used = 96000
Step 13: cost = -6.30875 shots used = 104000
Step 14: cost = -6.2524999999999995 shots used = 112000
Step 15: cost = -6.706250000000001 shots used = 120000
Step 16: cost = -6.711250000000001 shots used = 128000
Step 17: cost = -6.803749999999999 shots used = 136000
Step 18: cost = -6.94375 shots used = 144000
Step 19: cost = -7.28375 shots used = 152000
Step 20: cost = -7.4 shots used = 160000
Step 21: cost = -7.38375 shots used = 168000
Step 22: cost = -7.40125 shots used = 176000
Step 23: cost = -7.4775 shots used = 184000
Step 24: cost = -7.58 shots used = 192000
Step 25: cost = -7.623750000000001 shots used = 200000
Step 26: cost = -7.49625 shots used = 208000
Step 27: cost = -7.58375 shots used = 216000
Step 28: cost = -7.6312500000000005 shots used = 224000
Step 29: cost = -7.13375 shots used = 232000
Step 30: cost = -7.47 shots used = 240000
Step 31: cost = -7.6075 shots used = 248000
Step 32: cost = -7.34875 shots used = 256000
Step 33: cost = -7.6525 shots used = 264000
Step 34: cost = -7.572500000000001 shots used = 272000
Step 35: cost = -7.390000000000001 shots used = 280000
Step 36: cost = -7.76375 shots used = 288000
Step 37: cost = -7.49 shots used = 296000
Step 38: cost = -7.61625 shots used = 304000
Step 39: cost = -7.694999999999999 shots used = 312000
Step 40: cost = -7.7025 shots used = 320000
Step 41: cost = -7.59625 shots used = 328000
Step 42: cost = -7.73375 shots used = 336000
Step 43: cost = -7.6875 shots used = 344000
Step 44: cost = -7.75875 shots used = 352000
Step 45: cost = -7.796250000000001 shots used = 360000
Step 46: cost = -7.7387500000000005 shots used = 368000
Step 47: cost = -7.92375 shots used = 376000
Step 48: cost = -7.6225000000000005 shots used = 384000
Step 49: cost = -7.8425 shots used = 392000
Step 50: cost = -7.74 shots used = 400000
Step 51: cost = -7.661250000000001 shots used = 408000
Step 52: cost = -7.786250000000001 shots used = 416000
Step 53: cost = -7.78875 shots used = 424000
Step 54: cost = -7.62375 shots used = 432000
Step 55: cost = -7.9375 shots used = 440000
Step 56: cost = -7.71625 shots used = 448000
Step 57: cost = -7.72375 shots used = 456000
Step 58: cost = -7.741250000000001 shots used = 464000
Step 59: cost = -7.811249999999999 shots used = 472000
Step 60: cost = -7.890000000000001 shots used = 480000
Step 61: cost = -7.74 shots used = 488000
Step 62: cost = -7.751250000000001 shots used = 496000
Step 63: cost = -7.71875 shots used = 504000
Step 64: cost = -7.695 shots used = 512000
Step 65: cost = -7.7325 shots used = 520000
Step 66: cost = -7.819999999999999 shots used = 528000
Step 67: cost = -7.98125 shots used = 536000
Step 68: cost = -7.8 shots used = 544000
Step 69: cost = -7.89 shots used = 552000
Step 70: cost = -7.7125 shots used = 560000
Step 71: cost = -7.99375 shots used = 568000
Step 72: cost = -7.772499999999999 shots used = 576000
Step 73: cost = -8.01125 shots used = 584000
Step 74: cost = -8.11625 shots used = 592000
Step 75: cost = -7.9662500000000005 shots used = 600000
Step 76: cost = -7.7125 shots used = 608000
Step 77: cost = -7.8925 shots used = 616000
Step 78: cost = -7.9675 shots used = 624000
Step 79: cost = -7.91375 shots used = 632000
Step 80: cost = -7.797499999999999 shots used = 640000
Step 81: cost = -7.9975000000000005 shots used = 648000
Step 82: cost = -7.990000000000001 shots used = 656000
Step 83: cost = -7.7124999999999995 shots used = 664000
Step 84: cost = -7.76875 shots used = 672000
Step 85: cost = -7.62 shots used = 680000
Step 86: cost = -7.822500000000001 shots used = 688000
Step 87: cost = -7.74625 shots used = 696000
Step 88: cost = -7.9137499999999985 shots used = 704000
Step 89: cost = -7.86125 shots used = 712000
Step 90: cost = -7.975 shots used = 720000
Step 91: cost = -7.89375 shots used = 728000
Step 92: cost = -8.1075 shots used = 736000
Step 93: cost = -7.775 shots used = 744000
Step 94: cost = -7.8999999999999995 shots used = 752000
Step 95: cost = -7.85625 shots used = 760000
Step 96: cost = -7.925000000000001 shots used = 768000
Step 97: cost = -8.0 shots used = 776000
Step 98: cost = -7.825000000000001 shots used = 784000
Step 99: cost = -7.999999999999999 shots used = 792000
Step 0: cost = -5.976611864639143, shots_used = 240
Step 1: cost = -3.9696542358660762, shots_used = 288
Step 2: cost = -4.960189727105252, shots_used = 360
Step 3: cost = -4.580003760087762, shots_used = 456
Step 4: cost = -2.2302167491286937, shots_used = 552
Step 5: cost = -3.639026220963565, shots_used = 696
Step 6: cost = -6.407579837465839, shots_used = 1050
Step 7: cost = -7.4366536874312565, shots_used = 1578
Step 8: cost = -7.2596043217789035, shots_used = 2250
Step 9: cost = -7.062132684694291, shots_used = 2970
Step 10: cost = -7.553938182352898, shots_used = 3738
Step 11: cost = -7.530120251217973, shots_used = 4866
Step 12: cost = -7.620064018172075, shots_used = 6474
Step 13: cost = -7.749105026853706, shots_used = 8288
Step 14: cost = -7.758466910010545, shots_used = 10388
Step 15: cost = -7.547668090788591, shots_used = 12404
Step 16: cost = -7.802606000681807, shots_used = 14660
Step 17: cost = -7.8193751054958875, shots_used = 17180
Step 18: cost = -7.813893056373781, shots_used = 19700
Step 19: cost = -7.818976697763793, shots_used = 22796
Step 20: cost = -7.847655565015216, shots_used = 26372
Step 21: cost = -7.854512274045719, shots_used = 30810
Step 22: cost = -7.8556658192540905, shots_used = 35538
Step 23: cost = -7.8432766666801905, shots_used = 40770
Step 24: cost = -7.828138957960688, shots_used = 45762
Step 25: cost = -7.796501914990248, shots_used = 51162
Step 26: cost = -7.871130124788933, shots_used = 56466
Step 27: cost = -7.866190872563945, shots_used = 62010
Step 28: cost = -7.780118268373546, shots_used = 68250
Step 29: cost = -7.843565291223451, shots_used = 74946
Step 30: cost = -7.840084824878833, shots_used = 81762
Step 31: cost = -7.863430860462216, shots_used = 88962
Step 32: cost = -7.863400771365604, shots_used = 96786
Step 33: cost = -7.828392469226824, shots_used = 104730
Step 34: cost = -7.845758777555815, shots_used = 114532
Step 35: cost = -7.862280441095795, shots_used = 122908
Step 36: cost = -7.866212335569503, shots_used = 131836
Step 37: cost = -7.859430128177041, shots_used = 140500
Step 38: cost = -7.856087432905532, shots_used = 150076
Step 39: cost = -7.850323433779112, shots_used = 159676
Step 40: cost = -7.834403598788762, shots_used = 170116
Step 41: cost = -7.849769789802027, shots_used = 181300
Step 42: cost = -7.866938413531174, shots_used = 192700
Step 43: cost = -7.865653895759862, shots_used = 204460
Step 44: cost = -7.853522061269165, shots_used = 217900
Step 45: cost = -7.885272132729721, shots_used = 231748
Step 46: cost = -7.8822439546786445, shots_used = 245644
Step 47: cost = -7.8843763496186225, shots_used = 259852
Step 48: cost = -7.880891178100384, shots_used = 275164
Step 49: cost = -7.881035167671661, shots_used = 292444
Step 50: cost = -7.8819311529035705, shots_used = 310300
Step 51: cost = -7.873486288144935, shots_used = 329452
Step 52: cost = -7.842973314288797, shots_used = 348532
Step 53: cost = -7.8710179479772915, shots_used = 368644
Step 54: cost = -7.880857865087545, shots_used = 388828
Step 55: cost = -7.884163217633472, shots_used = 409132
Step 56: cost = -7.866452206380504, shots_used = 429076
Step 57: cost = -7.876255345278054, shots_used = 451468
Step 58: cost = -7.873699840747662, shots_used = 475348
Step 59: cost = -7.8902435026301605, shots_used = 501460
Step 0: cost = -2.033768399727329 shots_used = 2400
Step 1: cost = -3.0397515887713924 shots_used = 4800
Step 2: cost = -3.8459175082365675 shots_used = 7200
Step 3: cost = -4.505506895275779 shots_used = 9600
Step 4: cost = -5.048810662370808 shots_used = 12000
Step 5: cost = -5.48216212954771 shots_used = 14400
Step 6: cost = -5.83880726147689 shots_used = 16800
Step 7: cost = -6.143933494222607 shots_used = 19200
Step 8: cost = -6.412317130720796 shots_used = 21600
Step 9: cost = -6.653466668269803 shots_used = 24000
Step 10: cost = -6.86746547637287 shots_used = 26400
Step 11: cost = -7.057043661341394 shots_used = 28800
Step 12: cost = -7.219548494479426 shots_used = 31200
Step 13: cost = -7.3445177518694456 shots_used = 33600
Step 14: cost = -7.4357539424205275 shots_used = 36000
Step 15: cost = -7.497138548636967 shots_used = 38400
Step 16: cost = -7.529946318655265 shots_used = 40800
Step 17: cost = -7.537070813893376 shots_used = 43200
Step 18: cost = -7.525225697166626 shots_used = 45600
Step 19: cost = -7.50482511597234 shots_used = 48000
Step 20: cost = -7.4814871712462105 shots_used = 50400
Step 21: cost = -7.461106527571477 shots_used = 52800
Step 22: cost = -7.449032577502402 shots_used = 55200
Step 23: cost = -7.444817343084729 shots_used = 57600
Step 24: cost = -7.44949135869375 shots_used = 60000
Step 25: cost = -7.462969617594352 shots_used = 62400
Step 26: cost = -7.484518392550573 shots_used = 64800
Step 27: cost = -7.509533957688122 shots_used = 67200
Step 28: cost = -7.535240804873657 shots_used = 69600
Step 29: cost = -7.560642729685868 shots_used = 72000
Step 30: cost = -7.586205677180159 shots_used = 74400
Step 31: cost = -7.612604754020482 shots_used = 76800
Step 32: cost = -7.637117815005765 shots_used = 79200
Step 33: cost = -7.661716123608455 shots_used = 81600
Step 34: cost = -7.685231918972718 shots_used = 84000
Step 35: cost = -7.708583289744084 shots_used = 86400
Step 36: cost = -7.729551671925802 shots_used = 88800
Step 37: cost = -7.746255812560462 shots_used = 91200
Step 38: cost = -7.758965992155234 shots_used = 93600
Step 39: cost = -7.764889692835299 shots_used = 96000
Step 40: cost = -7.77029881424766 shots_used = 98400
Step 41: cost = -7.771938304013665 shots_used = 100800
Step 42: cost = -7.771490419427762 shots_used = 103200
Step 43: cost = -7.77166593220399 shots_used = 105600
Step 44: cost = -7.771775966399094 shots_used = 108000
Step 45: cost = -7.772019786144455 shots_used = 110400
Step 46: cost = -7.774409408800272 shots_used = 112800
Step 47: cost = -7.777544198411681 shots_used = 115200
Step 48: cost = -7.780578424610067 shots_used = 117600
Step 49: cost = -7.786514622689886 shots_used = 120000
Step 50: cost = -7.7938392154542 shots_used = 122400
Step 51: cost = -7.802144039740556 shots_used = 124800
Step 52: cost = -7.8098590120818105 shots_used = 127200
Step 53: cost = -7.818330164675916 shots_used = 129600
Step 54: cost = -7.8269309939766645 shots_used = 132000
Step 55: cost = -7.834969848723968 shots_used = 134400
Step 56: cost = -7.84245439512367 shots_used = 136800
Step 57: cost = -7.8493351526751445 shots_used = 139200
Step 58: cost = -7.853951071633945 shots_used = 141600
Step 59: cost = -7.858296868696568 shots_used = 144000
Step 60: cost = -7.8628676721698305 shots_used = 146400
Step 61: cost = -7.86554008020274 shots_used = 148800
Step 62: cost = -7.8675776324852045 shots_used = 151200
Step 63: cost = -7.869035010771338 shots_used = 153600
Step 64: cost = -7.870496374034536 shots_used = 156000
Step 65: cost = -7.871678720443282 shots_used = 158400
Step 66: cost = -7.872542373444427 shots_used = 160800
Step 67: cost = -7.8737392996750195 shots_used = 163200
Step 68: cost = -7.8743142937383075 shots_used = 165600
Step 69: cost = -7.875793149514543 shots_used = 168000
Step 70: cost = -7.877051911492934 shots_used = 170400
Step 71: cost = -7.878207264678215 shots_used = 172800
Step 72: cost = -7.879198045006913 shots_used = 175200
Step 73: cost = -7.880726987471536 shots_used = 177600
Step 74: cost = -7.882055795432431 shots_used = 180000
Step 75: cost = -7.882152825150277 shots_used = 182400
Step 76: cost = -7.881947191378355 shots_used = 184800
Step 77: cost = -7.881566349945112 shots_used = 187200
Step 78: cost = -7.881659168988008 shots_used = 189600
Step 79: cost = -7.881276797156975 shots_used = 192000
Step 80: cost = -7.879976174007027 shots_used = 194400
Step 81: cost = -7.8787149186438725 shots_used = 196800
Step 82: cost = -7.877964404670647 shots_used = 199200
Step 83: cost = -7.877102201620369 shots_used = 201600
Step 84: cost = -7.875562772172705 shots_used = 204000
Step 85: cost = -7.87560235017497 shots_used = 206400
Step 86: cost = -7.877141380119035 shots_used = 208800
Step 87: cost = -7.879257885053647 shots_used = 211200
Step 88: cost = -7.88114476100938 shots_used = 213600
Step 89: cost = -7.882250363744701 shots_used = 216000
Step 90: cost = -7.881748113564449 shots_used = 218400
Step 91: cost = -7.883533319932512 shots_used = 220800
Step 92: cost = -7.884779159318075 shots_used = 223200
Step 93: cost = -7.8868911005436555 shots_used = 225600
Step 94: cost = -7.88852422448021 shots_used = 228000
Step 95: cost = -7.888123287772766 shots_used = 230400
Step 96: cost = -7.886780080146787 shots_used = 232800
Step 97: cost = -7.885310745063638 shots_used = 235200
Step 98: cost = -7.8835076740891346 shots_used = 237600
Step 99: cost = -7.881351067687096 shots_used = 240000
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
Step 1: cost = -1.9358195375281395 shots used = 8000
Step 2: cost = -2.429697191348172 shots used = 16000
Step 3: cost = -3.6308900629353356 shots used = 24000
Step 4: cost = -4.275894772648205 shots used = 32000
Step 5: cost = -5.2697645839743705 shots used = 40000
Step 6: cost = -5.414011202290673 shots used = 48000
Step 7: cost = -5.958619200612785 shots used = 56000
Step 8: cost = -6.589890076928924 shots used = 64000
Step 9: cost = -6.879689304585086 shots used = 72000
Step 10: cost = -7.11044038351491 shots used = 80000
Step 11: cost = -7.441618274754085 shots used = 88000
Step 12: cost = -7.343694305020567 shots used = 96000
Step 13: cost = -7.531129303076952 shots used = 104000
Step 14: cost = -7.3929094864194465 shots used = 112000
Step 15: cost = -7.41913469502728 shots used = 120000
Step 16: cost = -7.484772628807 shots used = 128000
Step 17: cost = -7.2470111788679805 shots used = 136000
Step 18: cost = -7.278243959604719 shots used = 144000
Step 19: cost = -7.21024168219871 shots used = 152000
Step 20: cost = -7.497562843426365 shots used = 160000
Step 21: cost = -7.300503461549561 shots used = 168000
Step 22: cost = -7.502226318725064 shots used = 176000
Step 23: cost = -7.683018234663317 shots used = 184000
Step 24: cost = -7.7122067730786 shots used = 192000
Step 25: cost = -7.502924268040441 shots used = 200000
Step 26: cost = -7.655288657867658 shots used = 208000
Step 27: cost = -7.606898557639433 shots used = 216000
Step 28: cost = -7.625269501080646 shots used = 224000
Step 29: cost = -7.669368776362655 shots used = 232000
Step 30: cost = -7.810865011077598 shots used = 240000
Step 31: cost = -7.679683881569233 shots used = 248000
Step 32: cost = -7.654476847955421 shots used = 256000
Step 33: cost = -7.5211521050495636 shots used = 264000
Step 34: cost = -7.751101012112956 shots used = 272000
Step 35: cost = -7.687077255130304 shots used = 280000
Step 36: cost = -7.736266201334557 shots used = 288000
Step 37: cost = -7.680984283808025 shots used = 296000
Step 38: cost = -7.718561744208679 shots used = 304000
Step 39: cost = -7.67028103083904 shots used = 312000
Step 40: cost = -7.710347919579339 shots used = 320000
Step 41: cost = -7.593136197480478 shots used = 328000
Step 42: cost = -7.810359527538487 shots used = 336000
Step 43: cost = -7.737769438457732 shots used = 344000
Step 44: cost = -7.853027575362733 shots used = 352000
Step 45: cost = -7.87316397259127 shots used = 360000
Step 46: cost = -7.918883946761729 shots used = 368000
Step 47: cost = -7.921546467893382 shots used = 376000
Step 48: cost = -7.71980941701694 shots used = 384000
Step 49: cost = -7.8308057919156155 shots used = 392000
Step 50: cost = -7.8469319373986925 shots used = 400000
Step 51: cost = -7.814094417970937 shots used = 408000
Step 52: cost = -7.805577964490593 shots used = 416000
Step 53: cost = -7.792672651319024 shots used = 424000
Step 54: cost = -7.859279576823298 shots used = 432000
Step 55: cost = -7.920898395551514 shots used = 440000
Step 56: cost = -8.109172247437503 shots used = 448000
Step 57: cost = -7.954949088064669 shots used = 456000
Step 58: cost = -7.840000679159047 shots used = 464000
Step 59: cost = -8.073897469157906 shots used = 472000
Step 60: cost = -8.016380156819022 shots used = 480000
Step 61: cost = -7.749354818376889 shots used = 488000
Step 62: cost = -7.9084244778219634 shots used = 496000
Step 63: cost = -7.911403066159991 shots used = 504000
Step 64: cost = -7.694258154002229 shots used = 512000
Step 65: cost = -8.135637132372215 shots used = 520000
Step 66: cost = -7.880710119461283 shots used = 528000
Step 67: cost = -7.958115647880543 shots used = 536000
Step 68: cost = -7.949005635306886 shots used = 544000
Step 69: cost = -7.815469675932031 shots used = 552000
Step 70: cost = -7.824093391711489 shots used = 560000
Step 71: cost = -7.867760065578605 shots used = 568000
Step 72: cost = -7.931110498889814 shots used = 576000
Step 73: cost = -7.879521348618871 shots used = 584000
Step 74: cost = -7.840075562849998 shots used = 592000
Step 75: cost = -7.866075581377627 shots used = 600000
Step 76: cost = -7.923529733995367 shots used = 608000
Step 77: cost = -7.934656834108851 shots used = 616000
Step 78: cost = -7.894211944790477 shots used = 624000
Step 79: cost = -7.8338507678458065 shots used = 632000
Step 80: cost = -7.7936236744612 shots used = 640000
Step 81: cost = -7.9924846833802 shots used = 648000
Step 82: cost = -7.767589081568244 shots used = 656000
Step 83: cost = -8.036426061542748 shots used = 664000
Step 84: cost = -8.085968623696424 shots used = 672000
Step 85: cost = -7.798674831445482 shots used = 680000
Step 86: cost = -7.771240866851645 shots used = 688000
Step 87: cost = -7.805259795070319 shots used = 696000
Step 88: cost = -7.892488048336198 shots used = 704000
Step 89: cost = -7.894970226892511 shots used = 712000
Step 90: cost = -7.881326529610687 shots used = 720000
Step 91: cost = -8.108673203429863 shots used = 728000
Step 92: cost = -7.907973083754514 shots used = 736000
Step 93: cost = -7.801569622806004 shots used = 744000
Step 94: cost = -7.933400493853428 shots used = 752000
Step 95: cost = -7.8470947907663415 shots used = 760000
Step 96: cost = -7.943261293748337 shots used = 768000
Step 97: cost = -7.818450593894962 shots used = 776000
Step 98: cost = -7.9210182454846425 shots used = 784000
Step 99: cost = -8.111046662059286 shots used = 792000
Step 0: cost = -0.6424999999999998 shots used = 0
Step 1: cost = -1.7650000000000001 shots used = 8000
Step 2: cost = -3.0875 shots used = 16000
Step 3: cost = -3.47875 shots used = 24000
Step 4: cost = -4.405 shots used = 32000
Step 5: cost = -5.126250000000001 shots used = 40000
Step 6: cost = -5.6625 shots used = 48000
Step 7: cost = -5.9837500000000015 shots used = 56000
Step 8: cost = -6.50375 shots used = 64000
Step 9: cost = -6.775 shots used = 72000
Step 10: cost = -6.90625 shots used = 80000
Step 11: cost = -7.165 shots used = 88000
Step 12: cost = -7.39375 shots used = 96000
Step 13: cost = -7.52 shots used = 104000
Step 14: cost = -7.547499999999999 shots used = 112000
Step 15: cost = -7.512499999999999 shots used = 120000
Step 16: cost = -7.155 shots used = 128000
Step 17: cost = -7.534999999999999 shots used = 136000
Step 18: cost = -7.350000000000001 shots used = 144000
Step 19: cost = -7.25625 shots used = 152000
Step 20: cost = -7.3987500000000015 shots used = 160000
Step 21: cost = -7.50875 shots used = 168000
Step 22: cost = -7.50375 shots used = 176000
Step 23: cost = -7.55625 shots used = 184000
Step 24: cost = -7.3374999999999995 shots used = 192000
Step 25: cost = -7.7524999999999995 shots used = 200000
Step 26: cost = -7.6499999999999995 shots used = 208000
Step 27: cost = -7.7524999999999995 shots used = 216000
Step 28: cost = -7.637500000000001 shots used = 224000
Step 29: cost = -7.5512500000000005 shots used = 232000
Step 30: cost = -7.7175 shots used = 240000
Step 31: cost = -7.65625 shots used = 248000
Step 32: cost = -7.998749999999999 shots used = 256000
Step 33: cost = -7.67375 shots used = 264000
Step 34: cost = -7.2962500000000015 shots used = 272000
Step 35: cost = -7.5874999999999995 shots used = 280000
Step 36: cost = -7.74875 shots used = 288000
Step 37: cost = -7.713749999999999 shots used = 296000
Step 38: cost = -7.735 shots used = 304000
Step 39: cost = -7.893750000000001 shots used = 312000
Step 40: cost = -7.57 shots used = 320000
Step 41: cost = -7.7787500000000005 shots used = 328000
Step 42: cost = -7.83 shots used = 336000
Step 43: cost = -7.8475 shots used = 344000
Step 44: cost = -7.82875 shots used = 352000
Step 45: cost = -7.819999999999999 shots used = 360000
Step 46: cost = -7.836250000000001 shots used = 368000
Step 47: cost = -7.786250000000001 shots used = 376000
Step 48: cost = -7.8625 shots used = 384000
Step 49: cost = -7.951250000000001 shots used = 392000
Step 50: cost = -7.958749999999999 shots used = 400000
Step 51: cost = -8.20375 shots used = 408000
Step 52: cost = -7.692500000000001 shots used = 416000
Step 53: cost = -7.8375 shots used = 424000
Step 54: cost = -7.6312500000000005 shots used = 432000
Step 55: cost = -7.828749999999999 shots used = 440000
Step 56: cost = -7.8625 shots used = 448000
Step 57: cost = -8.09125 shots used = 456000
Step 58: cost = -7.70625 shots used = 464000
Step 59: cost = -7.8237499999999995 shots used = 472000
Step 60: cost = -8.03625 shots used = 480000
Step 61: cost = -7.972499999999999 shots used = 488000
Step 62: cost = -7.81 shots used = 496000
Step 63: cost = -7.86375 shots used = 504000
Step 64: cost = -8.045 shots used = 512000
Step 65: cost = -7.80375 shots used = 520000
Step 66: cost = -7.90375 shots used = 528000
Step 67: cost = -7.9025 shots used = 536000
Step 68: cost = -8.01875 shots used = 544000
Step 69: cost = -7.9725 shots used = 552000
Step 70: cost = -8.03625 shots used = 560000
Step 71: cost = -8.09875 shots used = 568000
Step 72: cost = -7.7925 shots used = 576000
Step 73: cost = -7.68 shots used = 584000
Step 74: cost = -7.995 shots used = 592000
Step 75: cost = -7.93625 shots used = 600000
Step 76: cost = -7.74125 shots used = 608000
Step 77: cost = -7.72 shots used = 616000
Step 78: cost = -7.710000000000001 shots used = 624000
Step 79: cost = -7.7125 shots used = 632000
Step 80: cost = -7.7325 shots used = 640000
Step 81: cost = -7.93125 shots used = 648000
Step 82: cost = -7.785 shots used = 656000
Step 83: cost = -7.7625 shots used = 664000
Step 84: cost = -7.6937500000000005 shots used = 672000
Step 85: cost = -8.0525 shots used = 680000
Step 86: cost = -8.06125 shots used = 688000
Step 87: cost = -7.8812500000000005 shots used = 696000
Step 88: cost = -7.973750000000001 shots used = 704000
Step 89: cost = -7.89875 shots used = 712000
Step 90: cost = -7.88 shots used = 720000
Step 91: cost = -7.99875 shots used = 728000
Step 92: cost = -7.86375 shots used = 736000
Step 93: cost = -7.911250000000001 shots used = 744000
Step 94: cost = -7.842500000000001 shots used = 752000
Step 95: cost = -8.00875 shots used = 760000
Step 96: cost = -7.859999999999999 shots used = 768000
Step 97: cost = -7.96375 shots used = 776000
Step 98: cost = -7.772499999999999 shots used = 784000
Step 99: cost = -7.9475 shots used = 792000
Step 0: cost = -4.820380999693457, shots_used = 240
Step 1: cost = -4.937944875992972, shots_used = 336
Step 2: cost = -5.477016391676936, shots_used = 456
Step 3: cost = -5.878302378912975, shots_used = 624
Step 4: cost = -5.740983235298872, shots_used = 768
Step 5: cost = -5.868499507174453, shots_used = 960
Step 6: cost = -5.5725976187549255, shots_used = 1200
Step 7: cost = -6.03851112713168, shots_used = 1512
Step 8: cost = -7.187839850746338, shots_used = 1944
Step 9: cost = -7.244742040043006, shots_used = 2472
Step 10: cost = -6.955119947427081, shots_used = 3144
Step 11: cost = -7.324280331788419, shots_used = 4176
Step 12: cost = -7.3050991795603855, shots_used = 5400
Step 13: cost = -7.241339003277949, shots_used = 6528
Step 14: cost = -7.529075219254999, shots_used = 7632
Step 15: cost = -7.095276433317586, shots_used = 9048
Step 16: cost = -7.607336707435147, shots_used = 10584
Step 17: cost = -7.738585612241667, shots_used = 12624
Step 18: cost = -7.7927004721043005, shots_used = 15288
Step 19: cost = -7.802640154411867, shots_used = 18048
Step 20: cost = -7.7651962845268105, shots_used = 20976
Step 21: cost = -7.77937999521944, shots_used = 24264
Step 22: cost = -7.851415989752514, shots_used = 27576
Step 23: cost = -7.829808564028126, shots_used = 31728
Step 24: cost = -7.813743545091879, shots_used = 36264
Step 25: cost = -7.790635224170293, shots_used = 42024
Step 26: cost = -7.887177094048005, shots_used = 48096
Step 27: cost = -7.8778724953618555, shots_used = 54744
Step 28: cost = -7.875679215329842, shots_used = 62448
Step 29: cost = -7.8559020553306125, shots_used = 71352
Step 30: cost = -7.884434864101767, shots_used = 80832
Step 31: cost = -7.891637364212711, shots_used = 90336
Step 32: cost = -7.854723497132757, shots_used = 100680
Step 33: cost = -7.860436984930214, shots_used = 111984
Step 34: cost = -7.885736157831495, shots_used = 123312
Step 35: cost = -7.851714069232079, shots_used = 136632
Step 36: cost = -7.864848529367196, shots_used = 150744
Step 37: cost = -7.875581235645932, shots_used = 166152
Step 38: cost = -7.809686333605921, shots_used = 183240
Step 39: cost = -7.884245542198441, shots_used = 202752
Step 40: cost = -7.889534749964766, shots_used = 221448
Step 41: cost = -7.894948220964508, shots_used = 240192
Step 42: cost = -7.897425239891585, shots_used = 262368
Step 43: cost = -7.8799029002955, shots_used = 285024
Step 44: cost = -7.873122596846604, shots_used = 307704
Step 45: cost = -7.889285563108286, shots_used = 331272
Step 46: cost = -7.893112373227318, shots_used = 357552
Step 47: cost = -7.878308602523566, shots_used = 385320
Step 48: cost = -7.899236702757055, shots_used = 416208
Step 49: cost = -7.894296334408784, shots_used = 446808
Step 50: cost = -7.890494435194311, shots_used = 479976
Step 51: cost = -7.89229873961093, shots_used = 512928
Step 52: cost = -7.893708744149611, shots_used = 547176
Step 53: cost = -7.898823452831049, shots_used = 582960
Step 54: cost = -7.8988892291181925, shots_used = 621072
Step 55: cost = -7.881052733782683, shots_used = 661488
Step 56: cost = -7.891364379135187, shots_used = 703032
Step 57: cost = -7.8974256745741105, shots_used = 747480
Step 58: cost = -7.89322196381792, shots_used = 794808
Step 59: cost = -7.8964387922046235, shots_used = 842570
Step 0: cost = -2.12150804866895 shots_used = 2400
Step 1: cost = -3.4462874411421884 shots_used = 4800
Step 2: cost = -4.533723704599173 shots_used = 7200
Step 3: cost = -5.360324618255417 shots_used = 9600
Step 4: cost = -6.010958804727693 shots_used = 12000
Step 5: cost = -6.5450082323750856 shots_used = 14400
Step 6: cost = -6.960941130446836 shots_used = 16800
Step 7: cost = -7.248308512586621 shots_used = 19200
Step 8: cost = -7.398432638481054 shots_used = 21600
Step 9: cost = -7.43238804878266 shots_used = 24000
Step 10: cost = -7.374281342889537 shots_used = 26400
Step 11: cost = -7.2878455754894835 shots_used = 28800
Step 12: cost = -7.211212391636272 shots_used = 31200
Step 13: cost = -7.168136331225393 shots_used = 33600
Step 14: cost = -7.171989037816059 shots_used = 36000
Step 15: cost = -7.2153317942728545 shots_used = 38400
Step 16: cost = -7.278024044019375 shots_used = 40800
Step 17: cost = -7.361449931178438 shots_used = 43200
Step 18: cost = -7.442410269187307 shots_used = 45600
Step 19: cost = -7.5113154529689705 shots_used = 48000
Step 20: cost = -7.5608730559384805 shots_used = 50400
Step 21: cost = -7.591626968703286 shots_used = 52800
Step 22: cost = -7.608322534779552 shots_used = 55200
Step 23: cost = -7.607043067486307 shots_used = 57600
Step 24: cost = -7.5940769784963384 shots_used = 60000
Step 25: cost = -7.579153179578798 shots_used = 62400
Step 26: cost = -7.572266109391965 shots_used = 64800
Step 27: cost = -7.568439746440856 shots_used = 67200
Step 28: cost = -7.5819359681781515 shots_used = 69600
Step 29: cost = -7.610907153836914 shots_used = 72000
Step 30: cost = -7.651198088153218 shots_used = 74400
Step 31: cost = -7.697526604943236 shots_used = 76800
Step 32: cost = -7.7469033971024 shots_used = 79200
Step 33: cost = -7.787293318189747 shots_used = 81600
Step 34: cost = -7.820874827421242 shots_used = 84000
Step 35: cost = -7.840729913365395 shots_used = 86400
Step 36: cost = -7.858055514435193 shots_used = 88800
Step 37: cost = -7.868752507617257 shots_used = 91200
Step 38: cost = -7.877775001403506 shots_used = 93600
Step 39: cost = -7.884473822847104 shots_used = 96000
Step 40: cost = -7.887248861002906 shots_used = 98400
Step 41: cost = -7.8859305197679666 shots_used = 100800
Step 42: cost = -7.881002890765943 shots_used = 103200
Step 43: cost = -7.875074719805529 shots_used = 105600
Step 44: cost = -7.866129786750198 shots_used = 108000
Step 45: cost = -7.850581153251575 shots_used = 110400
Step 46: cost = -7.843337695237989 shots_used = 112800
Step 47: cost = -7.845453624375395 shots_used = 115200
Step 48: cost = -7.853444576995694 shots_used = 117600
Step 49: cost = -7.858018368114793 shots_used = 120000
Step 50: cost = -7.858043805938922 shots_used = 122400
Step 51: cost = -7.855559046474577 shots_used = 124800
Step 52: cost = -7.850626102015815 shots_used = 127200
Step 53: cost = -7.848969631273945 shots_used = 129600
Step 54: cost = -7.852176020039671 shots_used = 132000
Step 55: cost = -7.861427874163019 shots_used = 134400
Step 56: cost = -7.868403253322002 shots_used = 136800
Step 57: cost = -7.876241759094949 shots_used = 139200
Step 58: cost = -7.88046548748952 shots_used = 141600
Step 59: cost = -7.880026154757119 shots_used = 144000
Step 60: cost = -7.877251725772389 shots_used = 146400
Step 61: cost = -7.870289689150821 shots_used = 148800
Step 62: cost = -7.864543163588923 shots_used = 151200
Step 63: cost = -7.862715331323386 shots_used = 153600
Step 64: cost = -7.861607002909471 shots_used = 156000
Step 65: cost = -7.866735539612959 shots_used = 158400
Step 66: cost = -7.867386735902061 shots_used = 160800
Step 67: cost = -7.867196452121145 shots_used = 163200
Step 68: cost = -7.869567827264065 shots_used = 165600
Step 69: cost = -7.869618725213595 shots_used = 168000
Step 70: cost = -7.86657829219507 shots_used = 170400
Step 71: cost = -7.857706098037315 shots_used = 172800
Step 72: cost = -7.8558663452092246 shots_used = 175200
Step 73: cost = -7.858187887358694 shots_used = 177600
Step 74: cost = -7.861466111241276 shots_used = 180000
Step 75: cost = -7.864825877239139 shots_used = 182400
Step 76: cost = -7.863570824947599 shots_used = 184800
Step 77: cost = -7.863497614169011 shots_used = 187200
Step 78: cost = -7.860326845355642 shots_used = 189600
Step 79: cost = -7.855301086551648 shots_used = 192000
Step 80: cost = -7.85589006991893 shots_used = 194400
Step 81: cost = -7.857406216142363 shots_used = 196800
Step 82: cost = -7.863450868072559 shots_used = 199200
Step 83: cost = -7.870058142679088 shots_used = 201600
Step 84: cost = -7.8777699578190585 shots_used = 204000
Step 85: cost = -7.883842326350928 shots_used = 206400
Step 86: cost = -7.882633952688103 shots_used = 208800
Step 87: cost = -7.879224942149158 shots_used = 211200
Step 88: cost = -7.872184015334468 shots_used = 213600
Step 89: cost = -7.864992896022884 shots_used = 216000
Step 90: cost = -7.860606976666904 shots_used = 218400
Step 91: cost = -7.85981012759474 shots_used = 220800
Step 92: cost = -7.86306566068536 shots_used = 223200
Step 93: cost = -7.868586359942562 shots_used = 225600
Step 94: cost = -7.874757156105922 shots_used = 228000
Step 95: cost = -7.8798938081862495 shots_used = 230400
Step 96: cost = -7.8833173990872165 shots_used = 232800
Step 97: cost = -7.883690480348111 shots_used = 235200
Step 98: cost = -7.8820611003810965 shots_used = 237600
Step 99: cost = -7.8789076755031635 shots_used = 240000
 </code>
 </pre>
 </details>

---

## 19. tutorial_unitary_designs.html <a name="demo18"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_unitary_designs.html):

```
Mean fidelity = 0.9867904283600585
Haar-random mean fidelity = 0.9867904283600585
Clifford mean fidelity    = 0.9867892185247996
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_unitary_designs.html):

```
Mean fidelity = 0.986790428609261
Haar-random mean fidelity = 0.986790428609261
Clifford mean fidelity    = 0.9867892187454865
```

---

## 20. tutorial_doubly_stochastic.html <a name="demo19"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_doubly_stochastic.html):

```
Vanilla gradient descent min energy =  -4.605247234069294
Stochastic gradient descent (shots=1) min energy =  -4.457668962761635
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_doubly_stochastic.html):

```
Vanilla gradient descent min energy =  -4.605247234069292
Stochastic gradient descent (shots=1) min energy =  -4.457668962761634
```

---

## 21. tutorial_adaptive_circuits.html <a name="demo20"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_adaptive_circuits.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Excitation : [0, 1, 2, 3], Gradient: -0.012782175157668316
Excitation : [0, 1, 2, 5], Gradient: 5.421010862427523e-20
Excitation : [0, 1, 2, 7], Gradient: 1.2197274440461932e-19
Excitation : [0, 1, 2, 9], Gradient: 0.034264511701687594
Excitation : [0, 1, 3, 4], Gradient: 2.710505431213761e-20
Excitation : [0, 1, 3, 6], Gradient: 5.421010862427515e-20
Excitation : [0, 1, 3, 8], Gradient: -0.03426451170168762
Excitation : [0, 1, 4, 5], Gradient: -0.023581529020678203
Excitation : [0, 1, 5, 8], Gradient: -5.4210108624274824e-20
Excitation : [0, 1, 6, 7], Gradient: -0.023581529020678192
Excitation : [0, 1, 7, 8], Gradient: -1.2197274440461857e-19
Excitation : [0, 1, 8, 9], Gradient: -0.12362273485598638
[[0, 1, 2, 3], [0, 1, 2, 9], [0, 1, 3, 8], [0, 1, 4, 5], [0, 1, 6, 7], [0, 1, 8, 9]]
Excitation : [0, 2], Gradient: -0.005062536239328451
Excitation : [0, 4], Gradient: 1.5303223715824896e-17
Excitation : [0, 6], Gradient: 1.53813383544115e-18
Excitation : [0, 8], Gradient: -0.0009448044625784089
Excitation : [1, 3], Gradient: 0.004926616876996994
Excitation : [1, 5], Gradient: 1.9399488423539284e-18
Excitation : [1, 7], Gradient: 7.895062518189064e-19
Excitation : [1, 9], Gradient: 0.001453553485404492
[[0, 2], [0, 8], [1, 3], [1, 9]]
n = 0,  E = -7.86266587 H, t = 2.77 s
n = 1,  E = -7.87094621 H, t = 2.84 s
n = 2,  E = -7.87563100 H, t = 2.25 s
n = 3,  E = -7.87829146 H, t = 2.84 s
n = 4,  E = -7.87981705 H, t = 2.25 s
n = 5,  E = -7.88070477 H, t = 2.81 s
n = 6,  E = -7.88123143 H, t = 2.25 s
n = 7,  E = -7.88155161 H, t = 2.81 s
n = 8,  E = -7.88175217 H, t = 2.25 s
n = 9,  E = -7.88188237 H, t = 2.80 s
n = 10,  E = -7.88197041 H, t = 2.25 s
n = 11,  E = -7.88203267 H, t = 2.79 s
n = 12,  E = -7.88207879 H, t = 2.85 s
n = 13,  E = -7.88211452 H, t = 2.25 s
n = 14,  E = -7.88214335 H, t = 2.83 s
n = 15,  E = -7.88216743 H, t = 2.25 s
n = 16,  E = -7.88218814 H, t = 2.83 s
n = 17,  E = -7.88220634 H, t = 2.24 s
n = 18,  E = -7.88222261 H, t = 2.82 s
n = 19,  E = -7.88223734 H, t = 2.24 s
n = 0,  E = -7.86266587 H, t = 0.12 s
n = 1,  E = -7.87094621 H, t = 0.12 s
n = 2,  E = -7.87563100 H, t = 0.12 s
n = 3,  E = -7.87829146 H, t = 0.12 s
n = 4,  E = -7.87981705 H, t = 0.12 s
n = 5,  E = -7.88070477 H, t = 0.12 s
n = 6,  E = -7.88123143 H, t = 0.12 s
n = 7,  E = -7.88155161 H, t = 0.12 s
n = 8,  E = -7.88175217 H, t = 0.12 s
n = 9,  E = -7.88188237 H, t = 0.12 s
n = 10,  E = -7.88197041 H, t = 0.12 s
n = 11,  E = -7.88203267 H, t = 0.12 s
n = 12,  E = -7.88207879 H, t = 0.12 s
n = 13,  E = -7.88211452 H, t = 0.12 s
n = 14,  E = -7.88214335 H, t = 0.12 s
n = 15,  E = -7.88216743 H, t = 0.12 s
n = 16,  E = -7.88218814 H, t = 0.12 s
n = 17,  E = -7.88220634 H, t = 0.12 s
n = 18,  E = -7.88222261 H, t = 0.12 s
n = 19,  E = -7.88223734 H, t = 0.12 s
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
Excitation : [0, 1, 2, 3], Gradient: 0.0
Excitation : [0, 1, 2, 5], Gradient: 0.0
Excitation : [0, 1, 2, 7], Gradient: 0.0
Excitation : [0, 1, 2, 9], Gradient: 0.03426451170166862
Excitation : [0, 1, 3, 4], Gradient: 0.0
Excitation : [0, 1, 3, 6], Gradient: 0.0
Excitation : [0, 1, 3, 8], Gradient: -0.008566127925417156
Excitation : [0, 1, 4, 5], Gradient: 0.0
Excitation : [0, 1, 5, 8], Gradient: 0.0
Excitation : [0, 1, 6, 7], Gradient: 0.0
Excitation : [0, 1, 7, 8], Gradient: 0.0
Excitation : [0, 1, 8, 9], Gradient: 0.0
[[0, 1, 2, 9], [0, 1, 3, 8]]
Excitation : [0, 2], Gradient: -0.01336184379998598
Excitation : [0, 4], Gradient: 0.0
Excitation : [0, 6], Gradient: 0.0
Excitation : [0, 8], Gradient: 0.008127419311868169
Excitation : [1, 3], Gradient: 9.609881566904134e-06
Excitation : [1, 5], Gradient: -0.004875127086708152
Excitation : [1, 7], Gradient: -0.004875127086708153
Excitation : [1, 9], Gradient: -0.007509748822103645
[[0, 2], [0, 8], [1, 5], [1, 7], [1, 9]]
n = 0,  E = -7.85513767 H, t = 2.62 s
n = 1,  E = -7.85585993 H, t = 2.55 s
n = 2,  E = -7.85642249 H, t = 2.51 s
n = 3,  E = -7.85686535 H, t = 2.53 s
n = 4,  E = -7.85721832 H, t = 2.54 s
n = 5,  E = -7.85750361 H, t = 2.96 s
n = 6,  E = -7.85773773 H, t = 2.14 s
n = 7,  E = -7.85793296 H, t = 2.77 s
n = 8,  E = -7.85809846 H, t = 2.48 s
n = 9,  E = -7.85824102 H, t = 2.18 s
n = 10,  E = -7.85836572 H, t = 2.92 s
n = 11,  E = -7.85847636 H, t = 2.53 s
n = 12,  E = -7.85857579 H, t = 2.22 s
n = 13,  E = -7.85866614 H, t = 2.81 s
n = 14,  E = -7.85874902 H, t = 2.55 s
n = 15,  E = -7.85882566 H, t = 2.21 s
n = 16,  E = -7.85889701 H, t = 2.78 s
n = 17,  E = -7.85896378 H, t = 2.55 s
n = 18,  E = -7.85902654 H, t = 2.21 s
n = 19,  E = -7.85908573 H, t = 2.79 s
/opt/hostedtoolcache/Python/3.7.12/x64/lib/python3.7/site-packages/pennylane/math/multi_dispatch.py:66: UserWarning: Contains tensors of types {'autograd', 'scipy'}; dispatch will prioritize TensorFlow and PyTorch over autograd. Consider replacing Autograd with vanilla NumPy.
n = 0,  E = -7.86266587 H, t = 0.12 s
n = 1,  E = -7.86373056 H, t = 0.12 s
n = 2,  E = -7.86443636 H, t = 0.12 s
n = 3,  E = -7.86490587 H, t = 0.11 s
n = 4,  E = -7.86521992 H, t = 0.12 s
n = 5,  E = -7.86543166 H, t = 0.12 s
n = 6,  E = -7.86557597 H, t = 0.12 s
n = 7,  E = -7.86567575 H, t = 0.12 s
n = 8,  E = -7.86574604 H, t = 0.12 s
n = 9,  E = -7.86579669 H, t = 0.12 s
n = 10,  E = -7.86583418 H, t = 0.12 s
n = 11,  E = -7.86586277 H, t = 0.12 s
n = 12,  E = -7.86588528 H, t = 0.12 s
n = 13,  E = -7.86590357 H, t = 0.13 s
n = 14,  E = -7.86591886 H, t = 0.12 s
n = 15,  E = -7.86593199 H, t = 0.12 s
n = 16,  E = -7.86594350 H, t = 0.12 s
n = 17,  E = -7.86595377 H, t = 0.12 s
n = 18,  E = -7.86596307 H, t = 0.12 s
 </code>
 </pre>
 </details>

---

## 22. tutorial_qgrnn.html <a name="demo21"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_qgrnn.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Ground State Energy: -7.330689661291261
Weights at Step 0: [-0.22604317  0.4388776   0.85859736  0.69736712  0.09417674 -0.02437703]
Bias at Step 0: [-0.23885902 -0.21393414  0.12811164  0.45038514]
Cost at Step 5: -0.9806500098112143
Weights at Step 5: [-1.29827824  1.52426565  1.81163837  1.86438612  1.04288314 -0.98516982]
Bias at Step 5: [-1.27134815 -1.36193505  1.31543373  1.32653424]
Cost at Step 10: -0.9648857984236838
Weights at Step 10: [-1.41068173  1.67469055  1.64410873  2.23403518  0.87027277 -0.84569027]
Bias at Step 10: [-1.28108438 -1.67672193  1.74541519  0.99186816]
Cost at Step 15: -0.9909075076678548
Weights at Step 15: [-0.99423966  1.31509032  0.97714182  2.16814495  0.20032301 -0.2265963 ]
Bias at Step 15: [-0.74022191 -1.53032257  1.76965387  0.16183654]
Cost at Step 20: -0.9966008204823483
Weights at Step 20: [-0.46419217  0.84550568  0.43286203  1.9380202  -0.348272    0.25976054]
Bias at Step 20: [-0.12768221 -1.22298499  1.62611891 -0.4553839 ]
Cost at Step 25: -0.9926133497439016
Weights at Step 25: [-0.09015325  0.52906304  0.38274877  1.70919239 -0.41967469  0.26112821]
Bias at Step 25: [ 0.2326128  -0.93936924  1.45717265 -0.4540624 ]
Cost at Step 30: -0.9984946331046769
Weights at Step 30: [ 0.0123472   0.47133081  0.7428572   1.56943045 -0.11998299 -0.09999201]
Bias at Step 30: [ 0.22093363 -0.77889245  1.33551884 -0.01237726]
Cost at Step 35: -0.9987664875905384
Weights at Step 35: [-0.04974879  0.55838325  1.09528857  1.50649468  0.13247182 -0.39189813]
Bias at Step 35: [ 0.01173716 -0.72349491  1.25413089  0.33198678]
Cost at Step 40: -0.9976572253882189
Weights at Step 40: [-0.14927     0.66157373  1.19794921  1.48297288  0.10734093 -0.3831593 ]
Bias at Step 40: [-0.21739349 -0.72790187  1.18290627  0.29425225]
Cost at Step 45: -0.9996712273475133
Weights at Step 45: [-0.20674402  0.70360037  1.0807707   1.48455205 -0.12668579 -0.15106377]
Bias at Step 45: [-0.35389918 -0.76974032  1.11755141 -0.02878183]
Cost at Step 50: -0.9995584075846884
Weights at Step 50: [-0.21856857  0.69313289  0.9951204   1.51327524 -0.27695939 -0.00304774]
Bias at Step 50: [-0.39501335 -0.83717604  1.08135172 -0.25421503]
Cost at Step 55: -0.9993417338158986
Weights at Step 55: [-0.19958537  0.65456745  1.0540154   1.56588539 -0.22806521 -0.06304728]
Bias at Step 55: [-0.37502663 -0.92029696  1.08093403 -0.2300362 ]
Cost at Step 60: -0.9997702657469879
Weights at Step 60: [-0.15016116  0.59649054  1.17917889  1.61891613 -0.0895     -0.22355593]
Bias at Step 60: [-0.3186739  -0.99107611  1.09766839 -0.08892556]
Cost at Step 65: -0.9996436456556429
Weights at Step 65: [-0.08220889  0.53561352  1.22392446  1.64470734 -0.04968037 -0.29147776]
Bias at Step 65: [-0.26354431 -1.02050635  1.10560821 -0.06357961]
Cost at Step 70: -0.9997829725978746
Weights at Step 70: [-0.02812171  0.50207641  1.18712749  1.64861258 -0.12278163 -0.24907531]
Bias at Step 70: [-0.25578322 -1.01899708  1.10378897 -0.1755329 ]
Cost at Step 75: -0.9998161136786834
Weights at Step 75: [-0.01479171  0.51567978  1.17552912  1.65394365 -0.18664303 -0.21758165]
Bias at Step 75: [-0.32014432 -1.01791815  1.10519351 -0.27848449]
Cost at Step 80: -0.9998843655288554
Weights at Step 80: [-0.01884375  0.54281647  1.22344387  1.66582127 -0.17956552 -0.25497115]
Bias at Step 80: [-0.40362773 -1.02780124  1.10836139 -0.29496652]
Cost at Step 85: -0.999909835033078
Weights at Step 85: [-0.00379234  0.54313665  1.2733362   1.67338414 -0.15161841 -0.30903377]
Bias at Step 85: [-0.45061752 -1.03818189  1.10130953 -0.28564262]
---------------------------------------------
Cost at Step 90: -0.9999044415579312
Weights at Step 90: [ 0.04088165  0.50692558  1.28240681  1.67288196 -0.15109114 -0.33205828]
Bias at Step 90: [-0.4519275  -1.04504232  1.07941985 -0.3132239 ]
---------------------------------------------
Cost at Step 95: -0.9999061222838248
Weights at Step 95: [ 0.08554249  0.46656083  1.26840277  1.6755038  -0.16748469 -0.33495651]
Bias at Step 95: [-0.45081552 -1.0592657   1.05546404 -0.36819017]
---------------------------------------------
Cost at Step 100: -0.9999067038489897
Weights at Step 100: [ 0.10504756  0.45151576  1.27495676  1.68957857 -0.16364088 -0.35685113]
Bias at Step 100: [-0.48320092 -1.0860277   1.04231719 -0.40281373]
---------------------------------------------
Cost at Step 105: -0.9999258608983024
Weights at Step 105: [ 0.11286395  0.45096054  1.29489123  1.70583499 -0.14822871 -0.39062082]
Bias at Step 105: [-0.53290732 -1.11214824  1.03558616 -0.42231884]
---------------------------------------------
Cost at Step 110: -0.9999159642472002
Weights at Step 110: [ 0.13106534  0.44316269  1.30188589  1.7135758  -0.14449277 -0.41395624]
Bias at Step 110: [-0.57365278 -1.12589957  1.02588522 -0.45380079]
---------------------------------------------
Cost at Step 115: -0.9999295051359973
Weights at Step 115: [ 0.16456919  0.42229382  1.29615959  1.71331202 -0.14936482 -0.43044986]
Bias at Step 115: [-0.60049609 -1.12909729  1.01159669 -0.4958413 ]
---------------------------------------------
Cost at Step 120: -0.9999329105144759
Weights at Step 120: [ 0.19394551  0.40471337  1.2991934   1.71575281 -0.1428606  -0.45732438]
Bias at Step 120: [-0.63053342 -1.13507053  1.00043562 -0.52433156]
---------------------------------------------
Cost at Step 125: -0.999967215118185
Weights at Step 125: [ 0.21407342  0.39421863  1.30670262  1.72320127 -0.13102448 -0.48725764]
Bias at Step 125: [-0.6687698  -1.14749235  0.99304941 -0.54780452]
---------------------------------------------
Cost at Step 130: -0.9999498639111292
Weights at Step 130: [ 0.23251895  0.38323767  1.30565485  1.73043603 -0.12634796 -0.50754102]
Bias at Step 130: [-0.70519062 -1.16065043  0.9853521  -0.57984279]
Cost at Step 135: -0.9999460668283625
Weights at Step 135: [ 0.25315     0.36897848  1.30157699  1.73591416 -0.12368359 -0.52479079]
Bias at Step 135: [-0.73738032 -1.17210895  0.97628525 -0.6137189 ]
---------------------------------------------
Cost at Step 140: -0.9999760263168521
Weights at Step 140: [ 0.27319219  0.35494171  1.30465117  1.74054957 -0.11422205 -0.5482196 ]
Bias at Step 140: [-0.76767167 -1.18178289  0.96794962 -0.63664615]
---------------------------------------------
Cost at Step 145: -0.9999620825798219
Weights at Step 145: [ 0.29291312  0.34115888  1.30743903  1.7441703  -0.10602196 -0.56989588]
Bias at Step 145: [-0.79755643 -1.18957764  0.96017139 -0.65982932]
---------------------------------------------
Cost at Step 150: -0.9999528162713989
Weights at Step 150: [ 0.31193841  0.32791071  1.30767856  1.747724   -0.10220887 -0.58675715]
Bias at Step 150: [-0.8277568  -1.19660646  0.9533145  -0.68795875]
---------------------------------------------
Cost at Step 155: -0.9999676206975212
Weights at Step 155: [ 0.32929841  0.31576924  1.3101255   1.75246696 -0.0974324  -0.60396184]
Bias at Step 155: [-0.85776514 -1.2043529   0.94822294 -0.71402521]
---------------------------------------------
Cost at Step 160: -0.9999688897065236
Weights at Step 160: [ 0.34617048  0.30341272  1.31549083  1.75789129 -0.09127326 -0.62192276]
Bias at Step 160: [-0.88499311 -1.21222407  0.94414483 -0.73671672]
---------------------------------------------
Cost at Step 165: -0.9999678828349429
Weights at Step 165: [ 0.36424544  0.28904681  1.3183802   1.76299152 -0.08653562 -0.6379972 ]
Bias at Step 165: [-0.91076279 -1.21990516  0.93990101 -0.76174131]
Cost at Step 170: -0.9999718518505567
Weights at Step 170: [ 0.38058818  0.27530965  1.31811484  1.76729325 -0.08126922 -0.65326086]
Bias at Step 170: [-0.93762726 -1.22741419  0.93605274 -0.78587399]
Cost at Step 175: -0.9999699928840566
Weights at Step 175: [ 0.39531011  0.26293084  1.32005541  1.7712584  -0.0738556  -0.67032688]
Bias at Step 175: [-0.96623738 -1.23426398  0.9328937  -0.80654326]
Cost at Step 180: -0.9999810640341683
Weights at Step 180: [ 0.41111346  0.24939364  1.32150139  1.7744836  -0.06804758 -0.68598269]
Bias at Step 180: [-0.99446201 -1.23978732  0.92974762 -0.82904396]
Cost at Step 185: -0.9999732059499762
Weights at Step 185: [ 0.42663809  0.23569253  1.323561    1.77787964 -0.06331664 -0.7001218 ]
Bias at Step 185: [-1.02027229 -1.24495257  0.9273182  -0.8511393 ]
Cost at Step 190: -0.9999904686011621
Weights at Step 190: [ 0.4407972   0.22274166  1.32741124  1.78217538 -0.0581927  -0.71407865]
Bias at Step 190: [-1.04467037 -1.2507425   0.92600868 -0.87121712]
---------------------------------------------
Cost at Step 195: -0.9999828831078131
Weights at Step 195: [ 0.45318723  0.21066667  1.32928285  1.78607162 -0.05386645 -0.72584502]
Bias at Step 195: [-1.06684646 -1.25630331  0.92524928 -0.88990376]
---------------------------------------------
Cost at Step 200: -0.9999814691774112
Weights at Step 200: [ 0.46483484  0.19879592  1.33066437  1.78942676 -0.04940039 -0.73713324]
Bias at Step 200: [-1.08841247 -1.26129482  0.92474857 -0.90725602]
---------------------------------------------
Cost at Step 205: -0.9999831444629904
Weights at Step 205: [ 0.47584426  0.18719266  1.33154192  1.79202379 -0.04462708 -0.74822003]
Bias at Step 205: [-1.10977785 -1.265427    0.92453063 -0.92310712]
---------------------------------------------
Cost at Step 210: -0.9999790517608406
Weights at Step 210: [ 0.48584547  0.17597834  1.33028455  1.79378517 -0.03978138 -0.75852048]
Bias at Step 210: [-1.13101187 -1.26898531  0.92482346 -0.93780681]
---------------------------------------------
Cost at Step 215: -0.999984711195014
Weights at Step 215: [ 0.4946749   0.16562713  1.32915967  1.79528287 -0.03462316 -0.76851533]
Bias at Step 215: [-1.15132516 -1.27221485  0.92584491 -0.9503349 ]
---------------------------------------------
Cost at Step 220: -0.9999855424488904
Weights at Step 220: [ 0.50303178  0.15574802  1.32886423  1.79677025 -0.03058318 -0.77711746]
Bias at Step 220: [-1.16955409 -1.27502675  0.92729975 -0.96212831]
---------------------------------------------
Cost at Step 225: -0.9999863772257214
Weights at Step 225: [ 0.51097377  0.14603986  1.32919454  1.79836519 -0.02733105 -0.78463519]
Bias at Step 225: [-1.18628365 -1.27783948  0.92913454 -0.97340009]
---------------------------------------------
Cost at Step 230: -0.9999898484097556
Weights at Step 230: [ 0.51836647  0.13626055  1.32967368  1.80000495 -0.02394918 -0.79187192]
Bias at Step 230: [-1.20276401 -1.28103879  0.93145415 -0.98384786]
---------------------------------------------
Cost at Step 235: -0.999989065762445
Weights at Step 235: [ 0.52564729  0.12600301  1.32958604  1.80138672 -0.02082854 -0.79856878]
Bias at Step 235: [-1.2195148  -1.28431275  0.93422156 -0.9943289 ]
Cost at Step 240: -0.9999882976226053
Weights at Step 240: [ 0.53229253  0.11610906  1.32908336  1.80229993 -0.01749836 -0.80503539]
Bias at Step 240: [-1.23599684 -1.28731013  0.93747248 -1.00346756]
Cost at Step 245: -0.999988244093237
Weights at Step 245: [ 0.53847247  0.10723924  1.33070351  1.80329887 -0.01467023 -0.81108244]
Bias at Step 245: [-1.25069646 -1.28978222  0.94081211 -1.01123413]
Cost at Step 250: -0.9999868214455344
Weights at Step 250: [ 0.54460039  0.0980518   1.33056073  1.80410968 -0.01354454 -0.81496581]
Bias at Step 250: [-1.26394352 -1.29235223  0.94434983 -1.02022419]
---------------------------------------------
Cost at Step 255: -0.9999884214814982
Weights at Step 255: [ 0.54943379  0.08996909  1.33186446  1.8051759  -0.01056943 -0.82027142]
Bias at Step 255: [-1.27708148 -1.29545081  0.94840617 -1.02592635]
---------------------------------------------
Cost at Step 260: -0.9999905893984568
Weights at Step 260: [ 0.55392271  0.08211047  1.33002746  1.8051647  -0.00912692 -0.82330494]
Bias at Step 260: [-1.28877696 -1.29778044  0.9520992  -1.03219995]
---------------------------------------------
Cost at Step 265: -0.9999894660037794
Weights at Step 265: [ 0.55772452  0.07551996  1.33104998  1.80525639 -0.00672651 -0.82745976]
Bias at Step 265: [-1.29986939 -1.29978116  0.95585094 -1.03589225]
---------------------------------------------
Cost at Step 270: -0.9999873354125406
Weights at Step 270: [ 0.56170933  0.06903136  1.33175345  1.80537894 -0.00624539 -0.82966868]
Bias at Step 270: [-1.30907139 -1.30160912  0.9593855  -1.04092885]
---------------------------------------------
Cost at Step 275: -0.9999910126657188
Weights at Step 275: [ 0.56476346  0.06288961  1.33121415  1.80555827 -0.00483336 -0.83207027]
Bias at Step 275: [-1.3181107  -1.30426179  0.96340533 -1.04444787]
Cost at Step 280: -0.9999911807478907
Weights at Step 280: [ 0.56790545  0.05710628  1.33293783  1.80572737 -0.00337721 -0.83486294]
Bias at Step 280: [-1.32665878 -1.30655911  0.96725682 -1.04712595]
Cost at Step 285: -0.9999870533227305
Weights at Step 285: [ 0.57142315  0.05115447  1.33329561  1.80547158 -0.00412561 -0.83538511]
Bias at Step 285: [-1.33391608 -1.30836527  0.97068764 -1.05192006]
---------------------------------------------
Cost at Step 290: -0.9999918533581702
Weights at Step 290: [ 0.57314839  0.04641148  1.33277082  1.80515273 -0.00185471 -0.83794182]
Bias at Step 290: [-1.3420092  -1.3110848   0.97494496 -1.05274072]
---------------------------------------------
Cost at Step 295: -0.9999907581161834
Weights at Step 295: [ 5.75810409e-01  4.17373794e-02  1.33385607e+00  1.80469466e+00
 -1.37560830e-03 -8.39439431e-01]
Bias at Step 295: [-1.34861394 -1.31269908  0.97854738 -1.05489467]
---------------------------------------------
Target parameters     Learned parameters
Weights
-----------------------------------------
0.56                |  0.5782895244479015
1.24                |   1.335028329676284
1.67                |  1.8044804399858494
-0.79               | -0.8395497395039531
Bias
-----------------------------------------
-1.44               | -1.3529586931946351
-1.43               | -1.3138918025602138
1.18                |  0.9811173767093417
-0.93               |  -1.057933121200338
Non-Existing Edge Parameters: [0.037958492678915545, -0.0022991817976615774]
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
Ground State Energy: -7.330689661291242
Weights at Step 0: [-0.22603613  0.43887001  0.85859236  0.69735898  0.09417125 -0.02437147]
Bias at Step 0: [-0.23884748 -0.21392016  0.12809368  0.45037793]
Cost at Step 5: -0.9974589524428031
Weights at Step 5: [-0.75106068  1.078707    0.83766935  1.9741555   0.04982793 -0.06747815]
Bias at Step 5: [-0.50836435 -1.32708118  1.57468372  0.11442806]
Cost at Step 10: -0.9971878268304797
Weights at Step 10: [ 0.01577799  0.48771566  0.68379977  1.75747002 -0.21948418 -0.00484698]
Bias at Step 10: [ 0.22007905 -0.90282076  1.58989008 -0.1051542 ]
Cost at Step 15: -0.998187153312274
Weights at Step 15: [-0.06744249  0.65720464  1.31471457  1.47430241  0.05813038 -0.41315658]
Bias at Step 15: [-0.29045223 -0.67045595  1.19395446  0.24677711]
Cost at Step 20: -0.9995130692146865
Weights at Step 20: [-0.16225009  0.66208813  1.17758061  1.48254064 -0.35468207 -0.01648733]
Bias at Step 20: [-0.56832881 -0.87721581  0.86890622 -0.27734217]
Cost at Step 25: -0.9998181560069559
Weights at Step 25: [ 0.030689    0.40363412  1.32430282  1.80692972 -0.14761078 -0.27452275]
Bias at Step 25: [-0.33695681 -1.28646051  1.00509121 -0.19672993]
Cost at Step 30: -0.9997713453918133
Weights at Step 30: [ 0.22014178  0.38063046  1.34934589  1.82854127 -0.201276   -0.35610913]
Bias at Step 30: [-0.40515586 -1.19466624  1.13933672 -0.31753371]
Cost at Step 35: -0.9997858632135158
Weights at Step 35: [ 0.22310889  0.50896099  1.36029033  1.73588161 -0.30076632 -0.35201799]
Bias at Step 35: [-0.73605504 -1.06150682  1.07911402 -0.49331792]
Cost at Step 40: -0.9998587245201027
Weights at Step 40: [ 0.32580831  0.34293483  1.38813471  1.76261455 -0.16880133 -0.49831078]
Bias at Step 40: [-0.74534508 -1.20379495  0.91916721 -0.49099848]
Cost at Step 45: -0.9998796449154709
Weights at Step 45: [ 0.37151473  0.26329833  1.29149206  1.84754531 -0.16287055 -0.51793342]
Bias at Step 45: [-0.81082195 -1.35860008  0.88788716 -0.63818331]
Cost at Step 50: -0.9999381279674818
Weights at Step 50: [ 0.36839385  0.36200243  1.3398256   1.82138962 -0.10863003 -0.63499473]
Bias at Step 50: [-1.04909977 -1.27581779  0.93902928 -0.67286639]
Cost at Step 55: -0.9999252881391635
Weights at Step 55: [ 0.52717503  0.22669791  1.27477846  1.75733597 -0.1195999  -0.66147384]
Bias at Step 55: [-1.02458911 -1.19668936  0.89787258 -0.76851797]
Cost at Step 60: -0.9999298586216633
Weights at Step 60: [ 0.46426344  0.2377941   1.30579533  1.86309089 -0.05164041 -0.72495523]
Bias at Step 60: [-1.15189323 -1.3565901   0.92299357 -0.81716179]
Cost at Step 65: -0.9999633163685123
Weights at Step 65: [ 0.54814165  0.15321249  1.29421602  1.79468038 -0.06551864 -0.73442872]
Bias at Step 65: [-1.18093812 -1.2850757   0.8715214  -0.89498456]
Cost at Step 70: -0.9999692974112366
Weights at Step 70: [ 0.54968362  0.15363051  1.32636004  1.81799545 -0.03795287 -0.78176424]
Bias at Step 70: [-1.26004878 -1.29186174  0.93275685 -0.93151559]
Cost at Step 75: -0.9999839418035988
Weights at Step 75: [ 0.56926778  0.09899053  1.32784104  1.83800055 -0.02258999 -0.80100631]
Bias at Step 75: [-1.27844754 -1.32707392  0.95121289 -0.97776518]
Cost at Step 80: -0.9999900754964176
Weights at Step 80: [ 0.58436296  0.06469565  1.33177097  1.80638755 -0.0264562  -0.80504093]
Bias at Step 80: [-1.32003524 -1.30314422  0.93367031 -1.02093595]
Cost at Step 85: -0.9999892005586407
Weights at Step 85: [ 5.76118484e-01  5.92975577e-02  1.35458994e+00  1.82562544e+00
 -6.90843714e-04 -8.37859294e-01]
Bias at Step 85: [-1.35783384 -1.32173749  0.97374052 -1.02956436]
---------------------------------------------
Cost at Step 90: -0.9999860418671424
Weights at Step 90: [ 0.60406684  0.01793718  1.33218637  1.81633306 -0.0189833  -0.82575128]
Bias at Step 90: [-1.35074923 -1.31381294  0.98189172 -1.07270478]
---------------------------------------------
Cost at Step 95: -0.9999837519650678
Weights at Step 95: [ 0.58241323  0.02570679  1.35041725  1.81039596  0.00900262 -0.85399323]
Bias at Step 95: [-1.39511753 -1.32055157  0.98642234 -1.06137202]
---------------------------------------------
Cost at Step 100: -0.9999862374755372
Weights at Step 100: [ 0.59133649  0.00671959  1.33149619  1.80654252  0.00739735 -0.85271751]
Bias at Step 100: [-1.3928395  -1.32434117  0.99986843 -1.0746036 ]
---------------------------------------------
Cost at Step 105: -0.9999860417314501
Weights at Step 105: [ 5.98294601e-01 -1.63872117e-03  1.32435903e+00  1.79606830e+00
 -1.50600209e-03 -8.46854780e-01]
Bias at Step 105: [-1.39810406 -1.31716394  1.00632966 -1.08694727]
---------------------------------------------
Cost at Step 110: -0.9999889448376539
Weights at Step 110: [ 0.58790075  0.00252021  1.34070893  1.79969067  0.01353978 -0.86073241]
Bias at Step 110: [-1.41237809 -1.33237829  1.01344356 -1.07306401]
---------------------------------------------
Cost at Step 115: -0.9999907527375037
Weights at Step 115: [ 0.59511523 -0.00485992  1.33269226  1.79443191  0.00548165 -0.85246429]
Bias at Step 115: [-1.40713813 -1.33369881  1.02023634 -1.07811166]
---------------------------------------------
Cost at Step 120: -0.9999894991115473
Weights at Step 120: [ 5.96065192e-01 -3.92737036e-03  1.33288381e+00  1.78858855e+00
  1.75644333e-03 -8.48477853e-01]
Bias at Step 120: [-1.40998319 -1.33422301  1.02362295 -1.07680692]
---------------------------------------------
Cost at Step 125: -0.9999921833903868
Weights at Step 125: [ 5.91985767e-01 -1.57241251e-03  1.33882177e+00  1.79191152e+00
  8.89983556e-03 -8.53306759e-01]
Bias at Step 125: [-1.41164517 -1.34685868  1.0288417  -1.06603474]
Cost at Step 130: -0.9999895629847483
Weights at Step 130: [ 5.93396935e-01 -6.09903975e-04  1.32769014e+00  1.78340627e+00
  3.20989912e-03 -8.45936030e-01]
Bias at Step 130: [-1.41153989 -1.34372306  1.0306921  -1.06641934]
---------------------------------------------
Cost at Step 135: -0.9999893671898543
Weights at Step 135: [ 5.93933082e-01 -1.84879055e-04  1.32433785e+00  1.77929886e+00
  4.38714188e-03 -8.45503125e-01]
Bias at Step 135: [-1.4098866  -1.34497204  1.03415862 -1.06007092]
---------------------------------------------
Cost at Step 140: -0.999990999591533
Weights at Step 140: [ 0.5908483   0.00239647  1.33016899  1.78168215  0.00438378 -0.84478361]
Bias at Step 140: [-1.41145547 -1.35249113  1.0371159  -1.05630093]
---------------------------------------------
Cost at Step 145: -0.9999882781659929
Weights at Step 145: [ 5.95379262e-01 -3.85481762e-04  1.33209614e+00  1.77901613e+00
 -1.07423798e-03 -8.40357066e-01]
Bias at Step 145: [-1.40745547 -1.35117282  1.03764696 -1.05779208]
---------------------------------------------
Cost at Step 150: -0.9999882605157342
Weights at Step 150: [ 0.59753863 -0.00178702  1.33577445  1.78058366 -0.00414765 -0.83825161]
Bias at Step 150: [-1.40514789 -1.35375725  1.03826782 -1.05953418]
---------------------------------------------
Cost at Step 155: -0.9999873092858327
Weights at Step 155: [ 5.96677467e-01  6.17989620e-04  1.34466485e+00  1.78402267e+00
 -2.65741532e-03 -8.41450495e-01]
Bias at Step 155: [-1.40789298 -1.356471    1.0376034  -1.05877886]
Cost at Step 160: -0.9999872637771745
Weights at Step 160: [ 0.60194839 -0.00191478  1.35123357  1.7884694  -0.00603744 -0.84142449]
Bias at Step 160: [-1.40510396 -1.35668892  1.03451132 -1.06543631]
Cost at Step 165: -0.9999874624869765
Weights at Step 165: [ 0.6030402  -0.00193504  1.34611742  1.78890714 -0.00577061 -0.84238872]
Bias at Step 165: [-1.40613555 -1.35445251  1.02812253 -1.07090041]
Cost at Step 170: -0.9999886747321688
Weights at Step 170: [ 0.5975333   0.00282116  1.32780927  1.78277225  0.00466827 -0.84919749]
Bias at Step 170: [-1.41126245 -1.34857858  1.02684163 -1.0645541 ]
Cost at Step 175: -0.9999875562507218
Weights at Step 175: [ 0.59389571  0.00440434  1.32168971  1.78088367  0.00772899 -0.85019789]
Bias at Step 175: [-1.41279703 -1.34685714  1.02847119 -1.06186318]
Cost at Step 180: -0.9999914836764833
Weights at Step 180: [ 5.95476507e-01  1.04646713e-03  1.33253375e+00  1.78379408e+00
  1.79045580e-03 -8.45539923e-01]
Bias at Step 180: [-1.40924345 -1.34843997  1.03054894 -1.06465758]
---------------------------------------------
Cost at Step 185: -0.9999877384049591
Weights at Step 185: [ 0.60021817 -0.00344807  1.34837992  1.78919267 -0.00485391 -0.84211998]
Bias at Step 185: [-1.40406354 -1.35059434  1.03159602 -1.06906187]
---------------------------------------------
Cost at Step 190: -0.9999943811280623
Weights at Step 190: [ 0.60332647 -0.00502377  1.35580211  1.7937833  -0.00870162 -0.84089377]
Bias at Step 190: [-1.40248031 -1.35189812  1.02840358 -1.07489377]
---------------------------------------------
Cost at Step 195: -0.9999899532834509
Weights at Step 195: [ 6.01661213e-01 -1.40258189e-03  1.34845381e+00  1.79177947e+00
 -2.19330435e-03 -8.46882171e-01]
Bias at Step 195: [-1.40624231 -1.34827409  1.02488298 -1.07228167]
---------------------------------------------
Cost at Step 200: -0.9999880990041629
Weights at Step 200: [ 0.59781588  0.00262681  1.33622645  1.7903236   0.00491206 -0.85187394]
Bias at Step 200: [-1.41001483 -1.3468901   1.02276916 -1.06950677]
---------------------------------------------
Cost at Step 205: -0.9999882388738663
Weights at Step 205: [ 0.59525242  0.00459556  1.33045933  1.78843797  0.00647584 -0.85208423]
Bias at Step 205: [-1.41193406 -1.34513748  1.02230177 -1.06854427]
---------------------------------------------
Cost at Step 210: -0.9999848077040173
Weights at Step 210: [ 0.59407847  0.00331191  1.3261464   1.78462336  0.00583158 -0.84928496]
Bias at Step 210: [-1.40995186 -1.34363827  1.02517197 -1.06554306]
---------------------------------------------
Cost at Step 215: -0.9999886167575431
Weights at Step 215: [ 5.93598280e-01  1.27457027e-03  1.32894414e+00  1.78302258e+00
  3.44879382e-03 -8.45638843e-01]
Bias at Step 215: [-1.40701621 -1.34498189  1.03023459 -1.06215846]
---------------------------------------------
Cost at Step 220: -0.9999875727826187
Weights at Step 220: [ 5.94793970e-01 -8.57403942e-04  1.33609793e+00  1.78453102e+00
 -2.17937735e-03 -8.40640138e-01]
Bias at Step 220: [-1.40459538 -1.34840612  1.03328985 -1.06323676]
Cost at Step 225: -0.9999895237582315
Weights at Step 225: [ 0.59740767 -0.00202435  1.33963888  1.78462403 -0.00566848 -0.83842794]
Bias at Step 225: [-1.40352872 -1.34909283  1.03367034 -1.0646202 ]
Cost at Step 230: -0.9999922217151833
Weights at Step 230: [ 0.59830589 -0.00184753  1.33406652  1.78177535 -0.00330205 -0.84015047]
Bias at Step 230: [-1.40427031 -1.34839966  1.03168677 -1.06292291]
Cost at Step 235: -0.9999912692867915
Weights at Step 235: [ 5.95836686e-01  8.57240159e-04  1.32786532e+00  1.78035155e+00
  2.56153005e-03 -8.44403818e-01]
Bias at Step 235: [-1.40725255 -1.34955664  1.03260437 -1.05840103]
---------------------------------------------
Cost at Step 240: -0.999989935769989
Weights at Step 240: [ 0.59351612  0.00209534  1.32485259  1.77824259  0.00444528 -0.8447466 ]
Bias at Step 240: [-1.40903945 -1.35042747  1.03478287 -1.05550792]
---------------------------------------------
Cost at Step 245: -0.9999889869569204
Weights at Step 245: [ 5.95395532e-01  1.23721713e-03  1.33593185e+00  1.78146341e+00
 -1.83303336e-03 -8.40987115e-01]
Bias at Step 245: [-1.40856761 -1.35225776  1.03748333 -1.05978839]
---------------------------------------------
Cost at Step 250: -0.999988129317049
Weights at Step 250: [ 5.95495047e-01 -2.18886873e-04  1.33760142e+00  1.78252108e+00
 -1.04388862e-03 -8.41650674e-01]
Bias at Step 250: [-1.40788422 -1.35473503  1.03617744 -1.05980154]
---------------------------------------------
Cost at Step 255: -0.9999892613809337
Weights at Step 255: [ 5.95095996e-01  1.05345945e-03  1.33120954e+00  1.78110699e+00
  9.09621038e-04 -8.43176791e-01]
Bias at Step 255: [-1.40881827 -1.35271576  1.03732158 -1.05936317]
Cost at Step 260: -0.9999911696516827
Weights at Step 260: [ 0.59215329  0.00249312  1.32162048  1.77469111  0.00411753 -0.84385295]
Bias at Step 260: [-1.41164782 -1.34921326  1.03505213 -1.05565234]
Cost at Step 265: -0.9999897602169294
Weights at Step 265: [ 5.94219509e-01  1.15801805e-03  1.32981272e+00  1.77875998e+00
  2.00797690e-03 -8.43338459e-01]
Bias at Step 265: [-1.40879199 -1.35161558  1.04052305 -1.05548315]
---------------------------------------------
Cost at Step 270: -0.9999877865048058
Weights at Step 270: [ 0.5967199  -0.00228699  1.34015592  1.78207987 -0.00207472 -0.8407702 ]
Bias at Step 270: [-1.40654726 -1.35550984  1.03685756 -1.05892969]
---------------------------------------------
Cost at Step 275: -0.9999913148930893
Weights at Step 275: [ 5.95115179e-01  6.67458889e-04  1.33088691e+00  1.77908492e+00
  2.17950090e-03 -8.43784743e-01]
Bias at Step 275: [-1.40907272 -1.35232601  1.03787329 -1.05577319]
---------------------------------------------
Cost at Step 280: -0.9999912773773975
Weights at Step 280: [ 5.94534323e-01  1.49825127e-03  1.33221200e+00  1.78129607e+00
  1.80930063e-03 -8.43864609e-01]
Bias at Step 280: [-1.41032893 -1.35393432  1.03609549 -1.0579392 ]
---------------------------------------------
Cost at Step 285: -0.9999871922879656
Weights at Step 285: [ 5.96437497e-01  3.12450031e-04  1.33601315e+00  1.78353853e+00
 -7.60576311e-04 -8.42698914e-01]
Bias at Step 285: [-1.40896079 -1.3539148   1.03483345 -1.06189856]
---------------------------------------------
Cost at Step 290: -0.9999918144420525
Weights at Step 290: [ 0.5928268   0.00288728  1.32568915  1.77748739  0.00484885 -0.84553166]
Bias at Step 290: [-1.41115856 -1.34928842  1.03616776 -1.0560555 ]
 </code>
 </pre>
 </details>

---

## 23. tutorial_quantum_chemistry.html <a name="demo22"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_quantum_chemistry.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
(-46.463906788688966+0j) [] +
(-0.014583648907612672+0j) [X0 X1 Y2 Y3] +
(-3.570761328924765e-07+0j) [X0 X1 Y2 Z3 Z4 Y5] +
(-0.005652620978017333+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7] +
(-0.008826368514209815+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.7924939576408363e-06+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.570761328924765e-07+0j) [X0 X1 X3 X4] +
(-0.005652620978017333+0j) [X0 X1 X3 Z4 Z5 X6] +
(-0.008826368514209815+0j) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.7924939576408363e-06+0j) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.0027458364701868163+0j) [X0 X1 Y4 Y5] +
(-2.447323128779908e-07+0j) [X0 X1 Y4 Z5 Z6 Y7] +
(-7.867765103943425e-07+0j) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0038040661717285403+0j) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.447323128779908e-07+0j) [X0 X1 X5 X6] +
(-7.867765103943425e-07+0j) [X0 X1 X5 Z6 Z7 Z8 Z9 X10] +
(-0.0038040661717285403+0j) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.006888194352970551+0j) [X0 X1 Y6 Y7] +
(-7.735036880588467e-05+0j) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11] +
(1.7035783553262864e-07+0j) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.735036880588467e-05+0j) [X0 X1 X7 Z8 Z9 X10] +
(1.7035783553262864e-07+0j) [X0 X1 X7 Z8 Z9 Z10 Z11 X12] +
(-0.006509361201177231+0j) [X0 X1 Y8 Y9] +
(-0.007731425250775242+0j) [X0 X1 Y10 Y11] +
(5.627851911159139e-07+0j) [X0 X1 Y10 Z11 Z12 Y13] +
(5.627851911159139e-07+0j) [X0 X1 X11 X12] +
(-0.005283776488402946+0j) [X0 X1 Y12 Y13] +
(0.014583648907612672+0j) [X0 Y1 Y2 X3] +
(3.570761328924765e-07+0j) [X0 Y1 Y2 Z3 Z4 X5] +
(0.005652620978017333+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7] +
(0.008826368514209815+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-1.7924939576408363e-06+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.570761328924765e-07+0j) [X0 Y1 Y3 X4] +
(-0.005652620978017333+0j) [X0 Y1 Y3 Z4 Z5 X6] +
(-0.008826368514209815+0j) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.7924939576408363e-06+0j) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0027458364701868163+0j) [X0 Y1 Y4 X5] +
(2.447323128779908e-07+0j) [X0 Y1 Y4 Z5 Z6 X7] +
(7.867765103943425e-07+0j) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0038040661717285403+0j) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.447323128779908e-07+0j) [X0 Y1 Y5 X6] +
(-7.867765103943425e-07+0j) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.0038040661717285403+0j) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.006888194352970551+0j) [X0 Y1 Y6 X7] +
(7.735036880588467e-05+0j) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11] +
(-1.7035783553262864e-07+0j) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.735036880588467e-05+0j) [X0 Y1 Y7 Z8 Z9 X10] +
(1.7035783553262864e-07+0j) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12] +
(0.006509361201177231+0j) [X0 Y1 Y8 X9] +
(0.007731425250775242+0j) [X0 Y1 Y10 X11] +
(-5.627851911159139e-07+0j) [X0 Y1 Y10 Z11 Z12 X13] +
(5.627851911159139e-07+0j) [X0 Y1 Y11 X12] +
(0.005283776488402946+0j) [X0 Y1 Y12 X13] +
(0.12507032579772004+0j) [X0 Z1 X2] +
(-1.9332412770254549e-07+0j) [X0 Z1 X2 X3 Z4 X5] +
(-0.0022939566113524632+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 X7] +
(-0.0016407548553124178+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(3.013471458848925e-07+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.9332412770254549e-07+0j) [X0 Z1 X2 Y3 Z4 Y5] +
(-0.0022939566113524632+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7] +
(-0.0016407548553124178+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(3.013471458848925e-07+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.003276971931231693+0j) [X0 Z1 X2 Z3] +
(-1.551053917683672e-07+0j) [X0 Z1 X2 X4 Z5 X6] +
(-1.1468376507242277e-06+0j) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.00759746402977061+0j) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.380778148036923e-07+0j) [X0 Z1 X2 Y4 Z5 Y6] +
(-7.900128986109099e-07+0j) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.005348051582676628+0j) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.005530759218631569+0j) [X0 Z1 X2 Z4] +
(-1.3807781480369227e-07+0j) [X0 Z1 X2 X5 Z6 X7] +
(-3.376739308342664e-07+0j) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0018638942824587442+0j) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.3807781480369227e-07+0j) [X0 Z1 X2 Y5 Z6 Y7] +
(-3.376739308342664e-07+0j) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0018638942824587442+0j) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0011059037691897094+0j) [X0 Z1 X2 Z5] +
(-8.352332102794466e-07+0j) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
1.9742253791539885e-08j [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.005262642473076845+0j) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10] +
(-8.07430598570632e-07+0j) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.008032520918821411+0j) [X0 Z1 X2 Z6] +
(0.00059402215430055+0j) [X0 Z1 X2 X7 Z8 Z9 Z10 X11] +
(-8.37977324392156e-08+0j) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.00059402215430055+0j) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11] +
(-8.37977324392156e-08+0j) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0033476175306662017+0j) [X0 Z1 X2 Z7] +
(0.011055020596132108+0j) [X0 Z1 X2 Z8] +
(0.002929768674751085+0j) [X0 Z1 X2 Z9] +
(-6.418291574403467e-07+0j) [X0 Z1 X2 X10 Z11 X12] +
(-6.556281914468365e-07+0j) [X0 Z1 X2 Y10 Z11 Y12] +
(0.0035552901955042894+0j) [X0 Z1 X2 Z10] +
(-1.1076325599261146e-07+0j) [X0 Z1 X2 X11 Z12 X13] +
(-1.1076325599261146e-07+0j) [X0 Z1 X2 Y11 Z12 Y13] +
(0.0017560707018412587+0j) [X0 Z1 X2 Z11] +
(0.0069012382497972875+0j) [X0 Z1 X2 Z12] +
(0.002326230623158093+0j) [X0 Z1 X2 Z13] +
(-3.568247521133177e-07+0j) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0022494124470939826+0j) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.0474716554727828e-08+0j) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.00044585351288408033+0j) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10] +
(-1.9742253791035477e-08+0j) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.00442485544944186+0j) [X0 Z1 Z2 X3 Y4 Y5] +
(-4.5233896777664345e-07+0j) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0034841573002178847+0j) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-8.09163719889961e-07+0j) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10] +
(-0.005733569747311867+0j) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.00468490338815521+0j) [X0 Z1 Z2 X3 Y6 Y7] +
(0.004668620318776295+0j) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11] +
(-7.189990975130105e-07+0j) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005114473831660376+0j) [X0 Z1 Z2 X3 X7 Z8 Z9 X10] +
(-7.560692464586371e-07+0j) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12] +
(-0.008125251921381022+0j) [X0 Z1 Z2 X3 Y8 Y9] +
(-0.0017992194936630307+0j) [X0 Z1 Z2 X3 Y10 Y11] +
(-5.471647744547236e-07+0j) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13] +
(-5.287660624472366e-07+0j) [X0 Z1 Z2 X3 X11 X12] +
(-0.004575007626639194+0j) [X0 Z1 Z2 X3 Y12 Y13] +
(0.00442485544944186+0j) [X0 Z1 Z2 Y3 Y4 X5] +
(4.5233896777664345e-07+0j) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0034841573002178847+0j) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.09163719889961e-07+0j) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.005733569747311867+0j) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.00468490338815521+0j) [X0 Z1 Z2 Y3 Y6 X7] +
(-0.004668620318776295+0j) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11] +
(7.189990975130105e-07+0j) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005114473831660376+0j) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10] +
(-7.560692464586371e-07+0j) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12] +
(0.008125251921381022+0j) [X0 Z1 Z2 Y3 Y8 X9] +
(0.0017992194936630307+0j) [X0 Z1 Z2 Y3 Y10 X11] +
(5.471647744547236e-07+0j) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13] +
(-5.287660624472366e-07+0j) [X0 Z1 Z2 Y3 Y11 X12] +
(0.004575007626639194+0j) [X0 Z1 Z2 Y3 Y12 X13] +
(3.202076879976696e-06+0j) [X0 Z1 Z2 Z3 X4] +
(0.0008533856254125474+0j) [X0 Z1 Z2 Z3 X4 X5 Z6 X7] +
(0.0007870896771024459+0j) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0008533856254125474+0j) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7] +
(0.0007870896771024459+0j) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.2919694861583518e-07+0j) [X0 Z1 Z2 Z3 X4 Z5] +
(4.444597854009106e-07+0j) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10] +
(0.0011726348316441857+0j) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.6849150950302026e-07+0j) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10] +
(0.0022009640695004537+0j) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.8394209154321113e-07+0j) [X0 Z1 Z2 Z3 X4 Z6] +
(4.0922506159890957e-07+0j) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11] +
(0.002394972639798016+0j) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.0922506159890957e-07+0j) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11] +
(0.002394972639798016+0j) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.236259961200322e-07+0j) [X0 Z1 Z2 Z3 X4 Z8] +
(8.649310133879628e-08+0j) [X0 Z1 Z2 Z3 X4 Z9] +
(0.0013038004788126938+0j) [X0 Z1 Z2 Z3 X4 X10 Z11 X12] +
(0.003989841456619296+0j) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12] +
(-6.733197742201987e-07+0j) [X0 Z1 Z2 Z3 X4 Z10] +
(0.0022619660624823455+0j) [X0 Z1 Z2 Z3 X4 X11 Z12 X13] +
(0.0022619660624823455+0j) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13] +
(-5.927453082625384e-07+0j) [X0 Z1 Z2 Z3 X4 Z11] +
(1.239336321665889e-06+0j) [X0 Z1 Z2 Z3 X4 Z12] +
(9.3065366517069e-07+0j) [X0 Z1 Z2 Z3 X4 Z13] +
(-0.0010283292378562676+0j) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.002686040977806603+0j) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12] +
(-1.8394209154321113e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7] +
(-0.00019400857029756348+0j) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.001222337808153831+0j) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-2.3713289478123593e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9] +
(8.05744659576603e-08+0j) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11] +
(0.0017278753941369516+0j) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13] +
(-0.0009581655836696513+0j) [X0 Z1 Z2 Z3 Z4 X5 X11 X12] +
(-3.086826564951992e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13] +
(1.8394209154321113e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7] +
(0.00019400857029756348+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.001222337808153831+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(2.3713289478123593e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9] +
(-8.05744659576603e-08+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11] +
(-0.0017278753941369516+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13] +
(-0.0009581655836696513+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12] +
(3.086826564951992e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13] +
(0.04274327701378277+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6] +
(0.0005192743499487739+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(-1.850564192787724e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005192743499487739+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(-1.850564192787724e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0027790267990255276+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7] +
(0.0046369766611825515+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8] +
(0.0012803060973496769+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9] +
(2.3120943051349907e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12] +
(1.0717282183258482e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12] +
(0.005379937155839359+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10] +
(7.246974425293948e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13] +
(7.246974425293948e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13] +
(0.005241535382803861+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11] +
(0.004311038507914298+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12] +
(0.0010435246534907607+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13] +
(1.2004287493658403e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12] +
(-0.0033566705638328753+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9] +
(-0.00013840177303549843+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11] +
(-6.1752462069681e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(-4.997018422127146e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12] +
(-0.0032675138544235364+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13] +
(0.0033566705638328753+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9] +
(0.00013840177303549843+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11] +
(6.1752462069681e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(-4.997018422127146e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12] +
(0.0032675138544235364+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13] +
(0.003876470899336936+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(-7.540341413521469e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.003876470899336936+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(-7.540341413521469e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(0.0716503518100242+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.002141361223101599+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(0.004220813970046438+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(0.001236647801924517+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(-0.002984166168121921+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(0.002984166168121921+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-1.3987009012667615e-05+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(8.949476487416122e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-6.876621657853486e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(-7.66134721264732e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(0.001532483523073018+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10] +
(-2.9045998838753386e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(0.005408954422409955+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10] +
(-1.0444941297396807e-06+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(0.004767272188278083+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10] +
(-8.105515036627573e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(0.005286546538226856+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10] +
(-9.956079229415296e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0016095313817213778+0j) [X0 Z1 Z2 Z3 Z4 X6] +
(-7.141625221156237e-05+0j) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10] +
(-2.6667317543300093e-07+0j) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.002462917007133925+0j) [X0 Z1 Z2 Z3 Z5 X6] +
(0.0007156734248908832+0j) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10] +
(-3.0767325314809725e-07+0j) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.606071867821115e-07+0j) [X0 Z1 Z2 X4] +
(0.003961560792496526+0j) [X0 Z1 Z2 Z4 Z5 X6] +
(0.00018787053389549712+0j) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.656930931831413e-07+0j) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.737933262351634e-07+0j) [X0 Z1 Z3 X4] +
(0.001667604181144063+0j) [X0 Z1 Z3 Z4 Z5 X6] +
(-0.0014528843214169206+0j) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(4.670402390680338e-07+0j) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.10433064780651413+0j) [X0 X2] +
(3.1174479459374175e-06+0j) [X0 Z2 Z3 X4] +
(0.04587947078129791+0j) [X0 Z2 Z3 Z4 Z5 X6] +
(0.058591988733861636+0j) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-1.1463061452573606e-05+0j) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.014583648907612672+0j) [Y0 X1 X2 Y3] +
(3.570761328924765e-07+0j) [Y0 X1 X2 Z3 Z4 Y5] +
(0.005652620978017333+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7] +
(0.008826368514209815+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.7924939576408363e-06+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.570761328924765e-07+0j) [Y0 X1 X3 Y4] +
(-0.005652620978017333+0j) [Y0 X1 X3 Z4 Z5 Y6] +
(-0.008826368514209815+0j) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.7924939576408363e-06+0j) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0027458364701868163+0j) [Y0 X1 X4 Y5] +
(2.447323128779908e-07+0j) [Y0 X1 X4 Z5 Z6 Y7] +
(7.867765103943425e-07+0j) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.0038040661717285403+0j) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.447323128779908e-07+0j) [Y0 X1 X5 Y6] +
(-7.867765103943425e-07+0j) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.0038040661717285403+0j) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.006888194352970551+0j) [Y0 X1 X6 Y7] +
(7.735036880588467e-05+0j) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11] +
(-1.7035783553262864e-07+0j) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.735036880588467e-05+0j) [Y0 X1 X7 Z8 Z9 Y10] +
(1.7035783553262864e-07+0j) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12] +
(0.006509361201177231+0j) [Y0 X1 X8 Y9] +
(0.007731425250775242+0j) [Y0 X1 X10 Y11] +
(-5.627851911159139e-07+0j) [Y0 X1 X10 Z11 Z12 Y13] +
(5.627851911159139e-07+0j) [Y0 X1 X11 Y12] +
(0.005283776488402946+0j) [Y0 X1 X12 Y13] +
(-0.014583648907612672+0j) [Y0 Y1 X2 X3] +
(-3.570761328924765e-07+0j) [Y0 Y1 X2 Z3 Z4 X5] +
(-0.005652620978017333+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7] +
(-0.008826368514209815+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.7924939576408363e-06+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.570761328924765e-07+0j) [Y0 Y1 Y3 Y4] +
(-0.005652620978017333+0j) [Y0 Y1 Y3 Z4 Z5 Y6] +
(-0.008826368514209815+0j) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.7924939576408363e-06+0j) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.0027458364701868163+0j) [Y0 Y1 X4 X5] +
(-2.447323128779908e-07+0j) [Y0 Y1 X4 Z5 Z6 X7] +
(-7.867765103943425e-07+0j) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0038040661717285403+0j) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.447323128779908e-07+0j) [Y0 Y1 Y5 Y6] +
(-7.867765103943425e-07+0j) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.0038040661717285403+0j) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.006888194352970551+0j) [Y0 Y1 X6 X7] +
(-7.735036880588467e-05+0j) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11] +
(1.7035783553262864e-07+0j) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.735036880588467e-05+0j) [Y0 Y1 Y7 Z8 Z9 Y10] +
(1.7035783553262864e-07+0j) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.006509361201177231+0j) [Y0 Y1 X8 X9] +
(-0.007731425250775242+0j) [Y0 Y1 X10 X11] +
(5.627851911159139e-07+0j) [Y0 Y1 X10 Z11 Z12 X13] +
(5.627851911159139e-07+0j) [Y0 Y1 Y11 Y12] +
(-0.005283776488402946+0j) [Y0 Y1 X12 X13] +
(-3.568247521133177e-07+0j) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0022494124470939826+0j) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.00044585351288408033+0j) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10] +
(-1.9742253791035477e-08+0j) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.0474716554727828e-08+0j) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.12507032579772004+0j) [Y0 Z1 Y2] +
(-1.9332412770254549e-07+0j) [Y0 Z1 Y2 X3 Z4 X5] +
(-0.0022939566113524632+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7] +
(-0.0016407548553124178+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(3.013471458848925e-07+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.9332412770254549e-07+0j) [Y0 Z1 Y2 Y3 Z4 Y5] +
(-0.0022939566113524632+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7] +
(-0.0016407548553124178+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(3.013471458848925e-07+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.003276971931231693+0j) [Y0 Z1 Y2 Z3] +
(-1.380778148036923e-07+0j) [Y0 Z1 Y2 X4 Z5 X6] +
(-7.900128986109099e-07+0j) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.005348051582676628+0j) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.551053917683672e-07+0j) [Y0 Z1 Y2 Y4 Z5 Y6] +
(-1.1468376507242277e-06+0j) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.00759746402977061+0j) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.005530759218631569+0j) [Y0 Z1 Y2 Z4] +
(-1.3807781480369227e-07+0j) [Y0 Z1 Y2 X5 Z6 X7] +
(-3.376739308342664e-07+0j) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0018638942824587442+0j) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.3807781480369227e-07+0j) [Y0 Z1 Y2 Y5 Z6 Y7] +
(-3.376739308342664e-07+0j) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0018638942824587442+0j) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0011059037691897094+0j) [Y0 Z1 Y2 Z5] +
(0.005262642473076845+0j) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10] +
(-8.07430598570632e-07+0j) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
-1.9742253791539885e-08j [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-8.352332102794466e-07+0j) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.008032520918821411+0j) [Y0 Z1 Y2 Z6] +
(0.00059402215430055+0j) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11] +
(-8.37977324392156e-08+0j) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.00059402215430055+0j) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11] +
(-8.37977324392156e-08+0j) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0033476175306662017+0j) [Y0 Z1 Y2 Z7] +
(0.011055020596132108+0j) [Y0 Z1 Y2 Z8] +
(0.002929768674751085+0j) [Y0 Z1 Y2 Z9] +
(-6.556281914468365e-07+0j) [Y0 Z1 Y2 X10 Z11 X12] +
(-6.418291574403467e-07+0j) [Y0 Z1 Y2 Y10 Z11 Y12] +
(0.0035552901955042894+0j) [Y0 Z1 Y2 Z10] +
(-1.1076325599261146e-07+0j) [Y0 Z1 Y2 X11 Z12 X13] +
(-1.1076325599261146e-07+0j) [Y0 Z1 Y2 Y11 Z12 Y13] +
(0.0017560707018412587+0j) [Y0 Z1 Y2 Z11] +
(0.0069012382497972875+0j) [Y0 Z1 Y2 Z12] +
(0.002326230623158093+0j) [Y0 Z1 Y2 Z13] +
(0.00442485544944186+0j) [Y0 Z1 Z2 X3 X4 Y5] +
(4.5233896777664345e-07+0j) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.0034841573002178847+0j) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-8.09163719889961e-07+0j) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.005733569747311867+0j) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.00468490338815521+0j) [Y0 Z1 Z2 X3 X6 Y7] +
(-0.004668620318776295+0j) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11] +
(7.189990975130105e-07+0j) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005114473831660376+0j) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10] +
(-7.560692464586371e-07+0j) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12] +
(0.008125251921381022+0j) [Y0 Z1 Z2 X3 X8 Y9] +
(0.0017992194936630307+0j) [Y0 Z1 Z2 X3 X10 Y11] +
(5.471647744547236e-07+0j) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13] +
(-5.287660624472366e-07+0j) [Y0 Z1 Z2 X3 X11 Y12] +
(0.004575007626639194+0j) [Y0 Z1 Z2 X3 X12 Y13] +
(-0.00442485544944186+0j) [Y0 Z1 Z2 Y3 X4 X5] +
(-4.5233896777664345e-07+0j) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0034841573002178847+0j) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.09163719889961e-07+0j) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.005733569747311867+0j) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.00468490338815521+0j) [Y0 Z1 Z2 Y3 X6 X7] +
(0.004668620318776295+0j) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11] +
(-7.189990975130105e-07+0j) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005114473831660376+0j) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10] +
(-7.560692464586371e-07+0j) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.008125251921381022+0j) [Y0 Z1 Z2 Y3 X8 X9] +
(-0.0017992194936630307+0j) [Y0 Z1 Z2 Y3 X10 X11] +
(-5.471647744547236e-07+0j) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13] +
(-5.287660624472366e-07+0j) [Y0 Z1 Z2 Y3 Y11 Y12] +
(-0.004575007626639194+0j) [Y0 Z1 Z2 Y3 X12 X13] +
(-0.0010283292378562676+0j) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.002686040977806603+0j) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12] +
(3.202076879976696e-06+0j) [Y0 Z1 Z2 Z3 Y4] +
(0.0008533856254125474+0j) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7] +
(0.0007870896771024459+0j) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0008533856254125474+0j) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7] +
(0.0007870896771024459+0j) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.2919694861583518e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z5] +
(4.6849150950302026e-07+0j) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10] +
(0.0022009640695004537+0j) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.444597854009106e-07+0j) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10] +
(0.0011726348316441857+0j) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.8394209154321113e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z6] +
(4.0922506159890957e-07+0j) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11] +
(0.002394972639798016+0j) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.0922506159890957e-07+0j) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11] +
(0.002394972639798016+0j) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.236259961200322e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z8] +
(8.649310133879628e-08+0j) [Y0 Z1 Z2 Z3 Y4 Z9] +
(0.003989841456619296+0j) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12] +
(0.0013038004788126938+0j) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12] +
(-6.733197742201987e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z10] +
(0.0022619660624823455+0j) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13] +
(0.0022619660624823455+0j) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13] +
(-5.927453082625384e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z11] +
(1.239336321665889e-06+0j) [Y0 Z1 Z2 Z3 Y4 Z12] +
(9.3065366517069e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z13] +
(1.8394209154321113e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7] +
(0.00019400857029756348+0j) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.001222337808153831+0j) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(2.3713289478123593e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9] +
(-8.05744659576603e-08+0j) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11] +
(-0.0017278753941369516+0j) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13] +
(-0.0009581655836696513+0j) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12] +
(3.086826564951992e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13] +
(-1.8394209154321113e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7] +
(-0.00019400857029756348+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.001222337808153831+0j) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-2.3713289478123593e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9] +
(8.05744659576603e-08+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11] +
(0.0017278753941369516+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13] +
(-0.0009581655836696513+0j) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12] +
(-3.086826564951992e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13] +
(1.2004287493658403e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12] +
(0.04274327701378277+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6] +
(0.0005192743499487739+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(-1.850564192787724e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005192743499487739+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(-1.850564192787724e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0027790267990255276+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7] +
(0.0046369766611825515+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8] +
(0.0012803060973496769+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9] +
(1.0717282183258482e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12] +
(2.3120943051349907e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12] +
(0.005379937155839359+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10] +
(7.246974425293948e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13] +
(7.246974425293948e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13] +
(0.005241535382803861+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11] +
(0.004311038507914298+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12] +
(0.0010435246534907607+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13] +
(0.0033566705638328753+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9] +
(0.00013840177303549843+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11] +
(6.1752462069681e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(-4.997018422127146e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12] +
(0.0032675138544235364+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13] +
(-0.0033566705638328753+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9] +
(-0.00013840177303549843+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11] +
(-6.1752462069681e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(-4.997018422127146e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12] +
(-0.0032675138544235364+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13] +
(0.003876470899336936+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(-7.540341413521469e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.003876470899336936+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(-7.540341413521469e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(0.0716503518100242+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.002141361223101599+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(0.004220813970046438+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(0.001236647801924517+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(0.002984166168121921+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-0.002984166168121921+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-1.3987009012667615e-05+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(8.949476487416122e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-6.876621657853486e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(-7.66134721264732e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(0.001532483523073018+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10] +
(-2.9045998838753386e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(0.005408954422409955+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10] +
(-1.0444941297396807e-06+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(0.004767272188278083+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10] +
(-8.105515036627573e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(0.005286546538226856+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10] +
(-9.956079229415296e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0016095313817213778+0j) [Y0 Z1 Z2 Z3 Z4 Y6] +
(-7.141625221156237e-05+0j) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10] +
(-2.6667317543300093e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.002462917007133925+0j) [Y0 Z1 Z2 Z3 Z5 Y6] +
(0.0007156734248908832+0j) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10] +
(-3.0767325314809725e-07+0j) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(3.606071867821115e-07+0j) [Y0 Z1 Z2 Y4] +
(0.003961560792496526+0j) [Y0 Z1 Z2 Z4 Z5 Y6] +
(0.00018787053389549712+0j) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.656930931831413e-07+0j) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.737933262351634e-07+0j) [Y0 Z1 Z3 Y4] +
(0.001667604181144063+0j) [Y0 Z1 Z3 Z4 Z5 Y6] +
(-0.0014528843214169206+0j) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(4.670402390680338e-07+0j) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.10433064780651413+0j) [Y0 Y2] +
(3.1174479459374175e-06+0j) [Y0 Z2 Z3 Y4] +
(0.04587947078129791+0j) [Y0 Z2 Z3 Z4 Z5 Y6] +
(0.058591988733861636+0j) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-1.1463061452573606e-05+0j) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(12.412630742111773+0j) [Z0] +
(0.10433064780651413+0j) [Z0 X1 Z2 X3] +
(3.1174479459374175e-06+0j) [Z0 X1 Z2 Z3 Z4 X5] +
(0.04587947078129791+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7] +
(0.058591988733861636+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-1.1463061452573607e-05+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.10433064780651413+0j) [Z0 Y1 Z2 Y3] +
(3.1174479459374175e-06+0j) [Z0 Y1 Z2 Z3 Z4 Y5] +
(0.04587947078129791+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7] +
(0.058591988733861636+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.1463061452573607e-05+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-8.337746753243037e-07+0j) [Z0 X2 Z3 X4] +
(-0.02711503684527307+0j) [Z0 X2 Z3 Z4 Z5 X6] +
(-0.0675238509921402+0j) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.4017109734893847e-05+0j) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-8.337746753243037e-07+0j) [Z0 Y2 Z3 Y4] +
(-0.02711503684527307+0j) [Z0 Y2 Z3 Z4 Z5 Y6] +
(-0.0675238509921402+0j) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.4017109734893847e-05+0j) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.23671080783830434+0j) [Z0 Z2] +
(-1.19085080821678e-06+0j) [Z0 X3 Z4 X5] +
(-0.0327676578232904+0j) [Z0 X3 Z4 Z5 Z6 X7] +
(-0.07635021950635001+0j) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.580960369253468e-05+0j) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.19085080821678e-06+0j) [Z0 Y3 Z4 Y5] +
(-0.0327676578232904+0j) [Z0 Y3 Z4 Z5 Z6 Y7] +
(-0.07635021950635001+0j) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.580960369253468e-05+0j) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.251294456745917+0j) [Z0 Z3] +
(-3.0993492437070554e-06+0j) [Z0 X4 Z5 X6] +
(-1.5316808794928594e-05+0j) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.08684737589863627+0j) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-3.0993492437070554e-06+0j) [Z0 Y4 Z5 Y6] +
(-1.5316808794928594e-05+0j) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.08684737589863627+0j) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.1966177089034216+0j) [Z0 Z4] +
(-3.3440815565850464e-06+0j) [Z0 X5 Z6 X7] +
(-1.6103585305322935e-05+0j) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0906514420703648+0j) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.3440815565850464e-06+0j) [Z0 Y5 Z6 Y7] +
(-1.6103585305322935e-05+0j) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0906514420703648+0j) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.1993635453736084+0j) [Z0 Z5] +
(0.056084681246613664+0j) [Z0 X6 Z7 Z8 Z9 X10] +
(-6.6522096692344145e-06+0j) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.056084681246613664+0j) [Z0 Y6 Z7 Z8 Z9 Y10] +
(-6.6522096692344145e-06+0j) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.24164663936017197+0j) [Z0 Z6] +
(0.05600733087780778+0j) [Z0 X7 Z8 Z9 Z10 X11] +
(-6.481851833701785e-06+0j) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.05600733087780778+0j) [Z0 Y7 Z8 Z9 Z10 Y11] +
(-6.481851833701785e-06+0j) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.24853483371314253+0j) [Z0 Z7] +
(0.27232518306605685+0j) [Z0 Z8] +
(0.27883454426723403+0j) [Z0 Z9] +
(-2.1776646049897604e-06+0j) [Z0 X10 Z11 X12] +
(-2.1776646049897604e-06+0j) [Z0 Y10 Z11 Y12] +
(0.19299723935364224+0j) [Z0 Z10] +
(-1.6148794138738464e-06+0j) [Z0 X11 Z12 X13] +
(-1.6148794138738464e-06+0j) [Z0 Y11 Z12 Y13] +
(0.20072866460441746+0j) [Z0 Z11] +
(0.2110265984979149+0j) [Z0 Z12] +
(0.21631037498631786+0j) [Z0 Z13] +
(1.933241277025455e-07+0j) [X1 X2 Y3 Y4] +
(0.002293956611352463+0j) [X1 X2 Y3 Z4 Z5 Y6] +
(0.0016407548553124178+0j) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-3.013471458848925e-07+0j) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.00442485544944186+0j) [X1 X2 X4 X5] +
(-8.09163719889961e-07+0j) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.005733569747311867+0j) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-4.5233896777664345e-07+0j) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.0034841573002178847+0j) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.00468490338815521+0j) [X1 X2 X6 X7] +
(0.005114473831660376+0j) [X1 X2 X6 Z7 Z8 Z9 Z10 X11] +
(-7.560692464586371e-07+0j) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.004668620318776295+0j) [X1 X2 Y7 Z8 Z9 Y10] +
(-7.189990975130105e-07+0j) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.008125251921381022+0j) [X1 X2 X8 X9] +
(-0.0017992194936630307+0j) [X1 X2 X10 X11] +
(-5.287660624472366e-07+0j) [X1 X2 X10 Z11 Z12 X13] +
(-5.471647744547236e-07+0j) [X1 X2 Y11 Y12] +
(-0.004575007626639195+0j) [X1 X2 X12 X13] +
(-1.933241277025455e-07+0j) [X1 Y2 Y3 X4] +
(-0.002293956611352463+0j) [X1 Y2 Y3 Z4 Z5 X6] +
(-0.0016407548553124178+0j) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(3.013471458848925e-07+0j) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.00442485544944186+0j) [X1 Y2 Y4 X5] +
(-8.09163719889961e-07+0j) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.005733569747311867+0j) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.5233896777664345e-07+0j) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10] +
(0.0034841573002178847+0j) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.00468490338815521+0j) [X1 Y2 Y6 X7] +
(0.005114473831660376+0j) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11] +
(-7.560692464586371e-07+0j) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.004668620318776295+0j) [X1 Y2 Y7 Z8 Z9 X10] +
(7.189990975130105e-07+0j) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.008125251921381022+0j) [X1 Y2 Y8 X9] +
(-0.0017992194936630307+0j) [X1 Y2 Y10 X11] +
(-5.287660624472366e-07+0j) [X1 Y2 Y10 Z11 Z12 X13] +
(5.471647744547236e-07+0j) [X1 Y2 Y11 X12] +
(-0.004575007626639195+0j) [X1 Y2 Y12 X13] +
(0.12507032579772004+0j) [X1 Z2 X3] +
(-1.3807781480369227e-07+0j) [X1 Z2 X3 X4 Z5 X6] +
(-3.376739308342664e-07+0j) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0018638942824587442+0j) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.3807781480369227e-07+0j) [X1 Z2 X3 Y4 Z5 Y6] +
(-3.376739308342664e-07+0j) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0018638942824587442+0j) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0011059037691897094+0j) [X1 Z2 X3 Z4] +
(-1.551053917683672e-07+0j) [X1 Z2 X3 X5 Z6 X7] +
(-1.1468376507242277e-06+0j) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.00759746402977061+0j) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.380778148036923e-07+0j) [X1 Z2 X3 Y5 Z6 Y7] +
(-7.900128986109099e-07+0j) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.005348051582676628+0j) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005530759218631569+0j) [X1 Z2 X3 Z5] +
(0.00059402215430055+0j) [X1 Z2 X3 X6 Z7 Z8 Z9 X10] +
(-8.37977324392156e-08+0j) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.00059402215430055+0j) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10] +
(-8.37977324392156e-08+0j) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0033476175306662017+0j) [X1 Z2 X3 Z6] +
(-8.352332102794466e-07+0j) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
1.9742253791539885e-08j [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005262642473076845+0j) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11] +
(-8.07430598570632e-07+0j) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.008032520918821411+0j) [X1 Z2 X3 Z7] +
(0.002929768674751085+0j) [X1 Z2 X3 Z8] +
(0.011055020596132108+0j) [X1 Z2 X3 Z9] +
(-1.1076325599261146e-07+0j) [X1 Z2 X3 X10 Z11 X12] +
(-1.1076325599261146e-07+0j) [X1 Z2 X3 Y10 Z11 Y12] +
(0.0017560707018412587+0j) [X1 Z2 X3 Z10] +
(-6.418291574403467e-07+0j) [X1 Z2 X3 X11 Z12 X13] +
(-6.556281914468365e-07+0j) [X1 Z2 X3 Y11 Z12 Y13] +
(0.0035552901955042894+0j) [X1 Z2 X3 Z11] +
(0.002326230623158093+0j) [X1 Z2 X3 Z12] +
(0.0069012382497972875+0j) [X1 Z2 X3 Z13] +
(-3.568247521133177e-07+0j) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.0022494124470939826+0j) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.0474716554727828e-08+0j) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.00044585351288408033+0j) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11] +
(-1.9742253791035477e-08+0j) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0008533856254125474+0j) [X1 Z2 Z3 X4 Y5 Y6] +
(-0.0007870896771024459+0j) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10] +
(-1.8394209154321116e-07+0j) [X1 Z2 Z3 X4 X6 X7] +
(-0.001222337808153831+0j) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.00019400857029756348+0j) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12] +
(-2.3713289478123593e-07+0j) [X1 Z2 Z3 X4 X8 X9] +
(8.057446595766029e-08+0j) [X1 Z2 Z3 X4 X10 X11] +
(-0.0009581655836696513+0j) [X1 Z2 Z3 X4 X10 Z11 Z12 X13] +
(0.0017278753941369516+0j) [X1 Z2 Z3 X4 Y11 Y12] +
(-3.086826564951992e-07+0j) [X1 Z2 Z3 X4 X12 X13] +
(0.0008533856254125474+0j) [X1 Z2 Z3 Y4 Y5 X6] +
(0.0007870896771024459+0j) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10] +
(-1.8394209154321116e-07+0j) [X1 Z2 Z3 Y4 Y6 X7] +
(-0.001222337808153831+0j) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.00019400857029756348+0j) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12] +
(-2.3713289478123593e-07+0j) [X1 Z2 Z3 Y4 Y8 X9] +
(8.057446595766029e-08+0j) [X1 Z2 Z3 Y4 Y10 X11] +
(-0.0009581655836696513+0j) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13] +
(-0.0017278753941369516+0j) [X1 Z2 Z3 Y4 Y11 X12] +
(-3.086826564951992e-07+0j) [X1 Z2 Z3 Y4 Y12 X13] +
(3.202076879976695e-06+0j) [X1 Z2 Z3 Z4 X5] +
(4.0922506159890957e-07+0j) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10] +
(0.002394972639798016+0j) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.0922506159890957e-07+0j) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10] +
(0.002394972639798016+0j) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.444597854009106e-07+0j) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11] +
(0.0011726348316441857+0j) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.6849150950302026e-07+0j) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11] +
(0.0022009640695004537+0j) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.8394209154321113e-07+0j) [X1 Z2 Z3 Z4 X5 Z7] +
(8.649310133879628e-08+0j) [X1 Z2 Z3 Z4 X5 Z8] +
(3.236259961200322e-07+0j) [X1 Z2 Z3 Z4 X5 Z9] +
(0.0022619660624823455+0j) [X1 Z2 Z3 Z4 X5 X10 Z11 X12] +
(0.0022619660624823455+0j) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12] +
(-5.927453082625384e-07+0j) [X1 Z2 Z3 Z4 X5 Z10] +
(0.0013038004788126938+0j) [X1 Z2 Z3 Z4 X5 X11 Z12 X13] +
(0.003989841456619296+0j) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13] +
(-6.733197742201987e-07+0j) [X1 Z2 Z3 Z4 X5 Z11] +
(9.3065366517069e-07+0j) [X1 Z2 Z3 Z4 X5 Z12] +
(1.239336321665889e-06+0j) [X1 Z2 Z3 Z4 X5 Z13] +
(-0.0010283292378562676+0j) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.002686040977806603+0j) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13] +
(-0.0005192743499487739+0j) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10] +
(1.850564192787724e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.003356670563832875+0j) [X1 Z2 Z3 Z4 Z5 X6 X8 X9] +
(-0.00013840177303549843+0j) [X1 Z2 Z3 Z4 Z5 X6 X10 X11] +
(-4.997018422127146e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13] +
(-6.1752462069681e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12] +
(-0.0032675138544235364+0j) [X1 Z2 Z3 Z4 Z5 X6 X12 X13] +
(0.0005192743499487739+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10] +
(-1.850564192787724e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.003356670563832875+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9] +
(-0.00013840177303549843+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11] +
(-4.997018422127146e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13] +
(6.1752462069681e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12] +
(-0.0032675138544235364+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13] +
(0.04274327701378277+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7] +
(0.0012803060973496769+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8] +
(0.0046369766611825515+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9] +
(7.246974425293948e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12] +
(7.246974425293948e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12] +
(0.005241535382803861+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10] +
(2.3120943051349907e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13] +
(1.0717282183258482e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13] +
(0.005379937155839359+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11] +
(0.0010435246534907607+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12] +
(0.004311038507914298+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13] +
(1.2004287493658403e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13] +
(-0.003876470899336936+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10] +
(7.540341413521469e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(0.003876470899336936+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10] +
(-7.540341413521469e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(-0.002984166168121921+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-0.002984166168121921+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(0.07165035181002424+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.001236647801924517+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(0.004220813970046438+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(-1.3987009012667615e-05+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(8.949476487416123e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(-7.66134721264732e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(-0.0021413612231015993+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11] +
(-6.876621657853486e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(0.005408954422409955+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11] +
(-1.0444941297396807e-06+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(0.001532483523073018+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11] +
(-2.9045998838753386e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(0.005286546538226856+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11] +
(-9.956079229415296e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0027790267990255276+0j) [X1 Z2 Z3 Z4 Z5 X7] +
(0.004767272188278083+0j) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11] +
(-8.105515036627573e-07+0j) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.002462917007133925+0j) [X1 Z2 Z3 Z4 Z6 X7] +
(0.0007156734248908832+0j) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11] +
(-3.0767325314809725e-07+0j) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2919694861583518e-07+0j) [X1 Z2 Z3 X5] +
(0.0016095313817213778+0j) [X1 Z2 Z3 Z5 Z6 X7] +
(-7.141625221156237e-05+0j) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-2.6667317543300093e-07+0j) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.737933262351634e-07+0j) [X1 Z2 Z4 X5] +
(0.001667604181144063+0j) [X1 Z2 Z4 Z5 Z6 X7] +
(-0.0014528843214169206+0j) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(4.670402390680338e-07+0j) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.003276971931231693+0j) [X1 X3] +
(3.606071867821115e-07+0j) [X1 Z3 Z4 X5] +
(0.003961560792496526+0j) [X1 Z3 Z4 Z5 Z6 X7] +
(0.00018787053389549712+0j) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.656930931831413e-07+0j) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.933241277025455e-07+0j) [Y1 X2 X3 Y4] +
(-0.002293956611352463+0j) [Y1 X2 X3 Z4 Z5 Y6] +
(-0.0016407548553124178+0j) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(3.013471458848925e-07+0j) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.00442485544944186+0j) [Y1 X2 X4 Y5] +
(-8.09163719889961e-07+0j) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.005733569747311867+0j) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(4.5233896777664345e-07+0j) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10] +
(0.0034841573002178847+0j) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.00468490338815521+0j) [Y1 X2 X6 Y7] +
(0.005114473831660376+0j) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11] +
(-7.560692464586371e-07+0j) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.004668620318776295+0j) [Y1 X2 X7 Z8 Z9 Y10] +
(7.189990975130105e-07+0j) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.008125251921381022+0j) [Y1 X2 X8 Y9] +
(-0.0017992194936630307+0j) [Y1 X2 X10 Y11] +
(-5.287660624472366e-07+0j) [Y1 X2 X10 Z11 Z12 Y13] +
(5.471647744547236e-07+0j) [Y1 X2 X11 Y12] +
(-0.004575007626639195+0j) [Y1 X2 X12 Y13] +
(1.933241277025455e-07+0j) [Y1 Y2 X3 X4] +
(0.002293956611352463+0j) [Y1 Y2 X3 Z4 Z5 X6] +
(0.0016407548553124178+0j) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-3.013471458848925e-07+0j) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.00442485544944186+0j) [Y1 Y2 Y4 Y5] +
(-8.09163719889961e-07+0j) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.005733569747311867+0j) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-4.5233896777664345e-07+0j) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10] +
(-0.0034841573002178847+0j) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.00468490338815521+0j) [Y1 Y2 Y6 Y7] +
(0.005114473831660376+0j) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11] +
(-7.560692464586371e-07+0j) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.004668620318776295+0j) [Y1 Y2 X7 Z8 Z9 X10] +
(-7.189990975130105e-07+0j) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12] +
(-0.008125251921381022+0j) [Y1 Y2 Y8 Y9] +
(-0.0017992194936630307+0j) [Y1 Y2 Y10 Y11] +
(-5.287660624472366e-07+0j) [Y1 Y2 Y10 Z11 Z12 Y13] +
(-5.471647744547236e-07+0j) [Y1 Y2 X11 X12] +
(-0.004575007626639195+0j) [Y1 Y2 Y12 Y13] +
(-3.568247521133177e-07+0j) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.0022494124470939826+0j) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.00044585351288408033+0j) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11] +
(-1.9742253791035477e-08+0j) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.0474716554727828e-08+0j) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.12507032579772004+0j) [Y1 Z2 Y3] +
(-1.3807781480369227e-07+0j) [Y1 Z2 Y3 X4 Z5 X6] +
(-3.376739308342664e-07+0j) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0018638942824587442+0j) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.3807781480369227e-07+0j) [Y1 Z2 Y3 Y4 Z5 Y6] +
(-3.376739308342664e-07+0j) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0018638942824587442+0j) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0011059037691897094+0j) [Y1 Z2 Y3 Z4] +
(-1.380778148036923e-07+0j) [Y1 Z2 Y3 X5 Z6 X7] +
(-7.900128986109099e-07+0j) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.005348051582676628+0j) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.551053917683672e-07+0j) [Y1 Z2 Y3 Y5 Z6 Y7] +
(-1.1468376507242277e-06+0j) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.00759746402977061+0j) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005530759218631569+0j) [Y1 Z2 Y3 Z5] +
(0.00059402215430055+0j) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10] +
(-8.37977324392156e-08+0j) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.00059402215430055+0j) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10] +
(-8.37977324392156e-08+0j) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0033476175306662017+0j) [Y1 Z2 Y3 Z6] +
(0.005262642473076845+0j) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11] +
(-8.07430598570632e-07+0j) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
-1.9742253791539885e-08j [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.352332102794466e-07+0j) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.008032520918821411+0j) [Y1 Z2 Y3 Z7] +
(0.002929768674751085+0j) [Y1 Z2 Y3 Z8] +
(0.011055020596132108+0j) [Y1 Z2 Y3 Z9] +
(-1.1076325599261146e-07+0j) [Y1 Z2 Y3 X10 Z11 X12] +
(-1.1076325599261146e-07+0j) [Y1 Z2 Y3 Y10 Z11 Y12] +
(0.0017560707018412587+0j) [Y1 Z2 Y3 Z10] +
(-6.556281914468365e-07+0j) [Y1 Z2 Y3 X11 Z12 X13] +
(-6.418291574403467e-07+0j) [Y1 Z2 Y3 Y11 Z12 Y13] +
(0.0035552901955042894+0j) [Y1 Z2 Y3 Z11] +
(0.002326230623158093+0j) [Y1 Z2 Y3 Z12] +
(0.0069012382497972875+0j) [Y1 Z2 Y3 Z13] +
(0.0008533856254125474+0j) [Y1 Z2 Z3 X4 X5 Y6] +
(0.0007870896771024459+0j) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10] +
(-1.8394209154321116e-07+0j) [Y1 Z2 Z3 X4 X6 Y7] +
(-0.001222337808153831+0j) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.00019400857029756348+0j) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12] +
(-2.3713289478123593e-07+0j) [Y1 Z2 Z3 X4 X8 Y9] +
(8.057446595766029e-08+0j) [Y1 Z2 Z3 X4 X10 Y11] +
(-0.0009581655836696513+0j) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13] +
(-0.0017278753941369516+0j) [Y1 Z2 Z3 X4 X11 Y12] +
(-3.086826564951992e-07+0j) [Y1 Z2 Z3 X4 X12 Y13] +
(-0.0008533856254125474+0j) [Y1 Z2 Z3 Y4 X5 X6] +
(-0.0007870896771024459+0j) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10] +
(-1.8394209154321116e-07+0j) [Y1 Z2 Z3 Y4 Y6 Y7] +
(-0.001222337808153831+0j) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.00019400857029756348+0j) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12] +
(-2.3713289478123593e-07+0j) [Y1 Z2 Z3 Y4 Y8 Y9] +
(8.057446595766029e-08+0j) [Y1 Z2 Z3 Y4 Y10 Y11] +
(-0.0009581655836696513+0j) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13] +
(0.0017278753941369516+0j) [Y1 Z2 Z3 Y4 X11 X12] +
(-3.086826564951992e-07+0j) [Y1 Z2 Z3 Y4 Y12 Y13] +
(-0.0010283292378562676+0j) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.002686040977806603+0j) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13] +
(3.202076879976695e-06+0j) [Y1 Z2 Z3 Z4 Y5] +
(4.0922506159890957e-07+0j) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10] +
(0.002394972639798016+0j) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.0922506159890957e-07+0j) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10] +
(0.002394972639798016+0j) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.6849150950302026e-07+0j) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11] +
(0.0022009640695004537+0j) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.444597854009106e-07+0j) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11] +
(0.0011726348316441857+0j) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.8394209154321113e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z7] +
(8.649310133879628e-08+0j) [Y1 Z2 Z3 Z4 Y5 Z8] +
(3.236259961200322e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z9] +
(0.0022619660624823455+0j) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12] +
(0.0022619660624823455+0j) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12] +
(-5.927453082625384e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z10] +
(0.003989841456619296+0j) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13] +
(0.0013038004788126938+0j) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13] +
(-6.733197742201987e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z11] +
(9.3065366517069e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z12] +
(1.239336321665889e-06+0j) [Y1 Z2 Z3 Z4 Y5 Z13] +
(0.0005192743499487739+0j) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10] +
(-1.850564192787724e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.003356670563832875+0j) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9] +
(-0.00013840177303549843+0j) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11] +
(-4.997018422127146e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13] +
(6.1752462069681e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12] +
(-0.0032675138544235364+0j) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13] +
(-0.0005192743499487739+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10] +
(1.850564192787724e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-0.003356670563832875+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9] +
(-0.00013840177303549843+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11] +
(-4.997018422127146e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13] +
(-6.1752462069681e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12] +
(-0.0032675138544235364+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13] +
(1.2004287493658403e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13] +
(0.04274327701378277+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7] +
(0.0012803060973496769+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8] +
(0.0046369766611825515+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9] +
(7.246974425293948e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12] +
(7.246974425293948e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12] +
(0.005241535382803861+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10] +
(1.0717282183258482e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13] +
(2.3120943051349907e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13] +
(0.005379937155839359+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11] +
(0.0010435246534907607+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12] +
(0.004311038507914298+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13] +
(0.003876470899336936+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10] +
(-7.540341413521469e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-0.003876470899336936+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10] +
(7.540341413521469e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-0.002984166168121921+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(-0.002984166168121921+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(0.07165035181002424+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.001236647801924517+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(0.004220813970046438+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(-1.3987009012667615e-05+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(8.949476487416123e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(-7.66134721264732e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(-0.0021413612231015993+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11] +
(-6.876621657853486e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(0.005408954422409955+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11] +
(-1.0444941297396807e-06+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(0.001532483523073018+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11] +
(-2.9045998838753386e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(0.005286546538226856+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11] +
(-9.956079229415296e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0027790267990255276+0j) [Y1 Z2 Z3 Z4 Z5 Y7] +
(0.004767272188278083+0j) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11] +
(-8.105515036627573e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.002462917007133925+0j) [Y1 Z2 Z3 Z4 Z6 Y7] +
(0.0007156734248908832+0j) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11] +
(-3.0767325314809725e-07+0j) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.2919694861583518e-07+0j) [Y1 Z2 Z3 Y5] +
(0.0016095313817213778+0j) [Y1 Z2 Z3 Z5 Z6 Y7] +
(-7.141625221156237e-05+0j) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-2.6667317543300093e-07+0j) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.737933262351634e-07+0j) [Y1 Z2 Z4 Y5] +
(0.001667604181144063+0j) [Y1 Z2 Z4 Z5 Z6 Y7] +
(-0.0014528843214169206+0j) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(4.670402390680338e-07+0j) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.003276971931231693+0j) [Y1 Y3] +
(3.606071867821115e-07+0j) [Y1 Z3 Z4 Y5] +
(0.003961560792496526+0j) [Y1 Z3 Z4 Z5 Z6 Y7] +
(0.00018787053389549712+0j) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.656930931831413e-07+0j) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(12.412630742111773+0j) [Z1] +
(-1.19085080821678e-06+0j) [Z1 X2 Z3 X4] +
(-0.0327676578232904+0j) [Z1 X2 Z3 Z4 Z5 X6] +
(-0.07635021950635001+0j) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.580960369253468e-05+0j) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.19085080821678e-06+0j) [Z1 Y2 Z3 Y4] +
(-0.0327676578232904+0j) [Z1 Y2 Z3 Z4 Z5 Y6] +
(-0.07635021950635001+0j) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.580960369253468e-05+0j) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.251294456745917+0j) [Z1 Z2] +
(-8.337746753243037e-07+0j) [Z1 X3 Z4 X5] +
(-0.02711503684527307+0j) [Z1 X3 Z4 Z5 Z6 X7] +
(-0.0675238509921402+0j) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.4017109734893847e-05+0j) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.337746753243037e-07+0j) [Z1 Y3 Z4 Y5] +
(-0.02711503684527307+0j) [Z1 Y3 Z4 Z5 Z6 Y7] +
(-0.0675238509921402+0j) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.4017109734893847e-05+0j) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.23671080783830434+0j) [Z1 Z3] +
(-3.3440815565850464e-06+0j) [Z1 X4 Z5 X6] +
(-1.6103585305322935e-05+0j) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0906514420703648+0j) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-3.3440815565850464e-06+0j) [Z1 Y4 Z5 Y6] +
(-1.6103585305322935e-05+0j) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0906514420703648+0j) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.1993635453736084+0j) [Z1 Z4] +
(-3.0993492437070554e-06+0j) [Z1 X5 Z6 X7] +
(-1.5316808794928594e-05+0j) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.08684737589863627+0j) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.0993492437070554e-06+0j) [Z1 Y5 Z6 Y7] +
(-1.5316808794928594e-05+0j) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.08684737589863627+0j) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.1966177089034216+0j) [Z1 Z5] +
(0.05600733087780778+0j) [Z1 X6 Z7 Z8 Z9 X10] +
(-6.481851833701785e-06+0j) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.05600733087780778+0j) [Z1 Y6 Z7 Z8 Z9 Y10] +
(-6.481851833701785e-06+0j) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.24853483371314253+0j) [Z1 Z6] +
(0.056084681246613664+0j) [Z1 X7 Z8 Z9 Z10 X11] +
(-6.6522096692344145e-06+0j) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.056084681246613664+0j) [Z1 Y7 Z8 Z9 Z10 Y11] +
(-6.6522096692344145e-06+0j) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.24164663936017197+0j) [Z1 Z7] +
(0.27883454426723403+0j) [Z1 Z8] +
(0.27232518306605685+0j) [Z1 Z9] +
(-1.6148794138738464e-06+0j) [Z1 X10 Z11 X12] +
(-1.6148794138738464e-06+0j) [Z1 Y10 Z11 Y12] +
(0.20072866460441746+0j) [Z1 Z10] +
(-2.1776646049897604e-06+0j) [Z1 X11 Z12 X13] +
(-2.1776646049897604e-06+0j) [Z1 Y11 Z12 Y13] +
(0.19299723935364224+0j) [Z1 Z11] +
(0.21631037498631786+0j) [Z1 Z12] +
(0.2110265984979149+0j) [Z1 Z13] +
(-0.035839567953353434+0j) [X2 X3 Y4 Y5] +
(-2.1990516183020972e-07+0j) [X2 X3 Y4 Z5 Z6 Y7] +
(-2.360956320222924e-06+0j) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.01031148248983184+0j) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.1990516183020975e-07+0j) [X2 X3 X5 X6] +
(-2.360956320222924e-06+0j) [X2 X3 X5 Z6 Z7 Z8 Z9 X10] +
(-0.01031148248983184+0j) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.031143817988967173+0j) [X2 X3 Y6 Y7] +
(0.005368659358109595+0j) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11] +
(9.209350644760531e-08+0j) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005368659358109595+0j) [X2 X3 X7 Z8 Z9 X10] +
(9.209350644760531e-08+0j) [X2 X3 X7 Z8 Z9 Z10 Z11 X12] +
(-0.03619412355904266+0j) [X2 X3 Y8 Y9] +
(-0.02538465750845736+0j) [X2 X3 Y10 Y11] +
(2.1726691014281675e-06+0j) [X2 X3 Y10 Z11 Z12 Y13] +
(2.1726691014281675e-06+0j) [X2 X3 X11 X12] +
(-0.015577208063976443+0j) [X2 X3 Y12 Y13] +
(0.035839567953353434+0j) [X2 Y3 Y4 X5] +
(2.1990516183020972e-07+0j) [X2 Y3 Y4 Z5 Z6 X7] +
(2.360956320222924e-06+0j) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.01031148248983184+0j) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.1990516183020975e-07+0j) [X2 Y3 Y5 X6] +
(-2.360956320222924e-06+0j) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.01031148248983184+0j) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.031143817988967173+0j) [X2 Y3 Y6 X7] +
(-0.005368659358109595+0j) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11] +
(-9.209350644760531e-08+0j) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005368659358109595+0j) [X2 Y3 Y7 Z8 Z9 X10] +
(9.209350644760531e-08+0j) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12] +
(0.03619412355904266+0j) [X2 Y3 Y8 X9] +
(0.02538465750845736+0j) [X2 Y3 Y10 X11] +
(-2.1726691014281675e-06+0j) [X2 Y3 Y10 Z11 Z12 X13] +
(2.1726691014281675e-06+0j) [X2 Y3 Y11 X12] +
(0.015577208063976443+0j) [X2 Y3 Y12 X13] +
(-3.887051672348346e-06+0j) [X2 Z3 X4] +
(-0.0051433917688250876+0j) [X2 Z3 X4 X5 Z6 X7] +
(-0.009841749246962565+0j) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.9885117062666245e-06+0j) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0051433917688250876+0j) [X2 Z3 X4 Y5 Z6 Y7] +
(-0.009841749246962565+0j) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.9885117062666245e-06+0j) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.76499411792064e-07+0j) [X2 Z3 X4 Z5] +
(1.6893489514423537e-06+0j) [X2 Z3 X4 X6 Z7 Z8 Z9 X10] +
(0.010757563953908934+0j) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.537178096273229e-08+0j) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10] +
(4.2055484112176605e-05+0j) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.5935343913351927e-07+0j) [X2 Z3 X4 Z6] +
(3.211842019081199e-06+0j) [X2 Z3 X4 X7 Z8 Z9 Z10 X11] +
(0.01929956057936375+0j) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.211842019081199e-06+0j) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11] +
(0.01929956057936375+0j) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.1954890099195454e-06+0j) [X2 Z3 X4 Z7] +
(2.1868423778095717e-07+0j) [X2 Z3 X4 Z8] +
(-5.770052994325092e-07+0j) [X2 Z3 X4 Z9] +
(0.015588250102380184+0j) [X2 Z3 X4 X10 Z11 X12] +
(0.005324835234221682+0j) [X2 Z3 X4 Y10 Z11 Y12] +
(-3.158656431943763e-06+0j) [X2 Z3 X4 Z10] +
(0.024353077678068928+0j) [X2 Z3 X4 X11 Z12 X13] +
(0.024353077678068928+0j) [X2 Z3 X4 Y11 Z12 Y13] +
(-7.801707500332004e-06+0j) [X2 Z3 X4 Z11] +
(3.5390541844940366e-06+0j) [X2 Z3 X4 Z12] +
(8.81493730650821e-06+0j) [X2 Z3 X4 Z13] +
(1.6288532434901385e-06+0j) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10] +
(0.010715508469796757+0j) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.010263414868158502+0j) [X2 Z3 Y4 Y10 Z11 X12] +
(-1.4548424490530647e-06+0j) [X2 Z3 Z4 X5 Y6 Y7] +
(-3.1513463111289834e-06+0j) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11] +
(-0.01925750509525157+0j) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.5224930676388456e-06+0j) [X2 Z3 Z4 X5 X7 Z8 Z9 X10] +
(-0.00854199662545482+0j) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-7.956895372134663e-07+0j) [X2 Z3 Z4 X5 Y8 Y9] +
(-4.64305106838824e-06+0j) [X2 Z3 Z4 X5 Y10 Y11] +
(-0.019028242443847244+0j) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13] +
(-0.00876482757568874+0j) [X2 Z3 Z4 X5 X11 X12] +
(5.275883122014173e-06+0j) [X2 Z3 Z4 X5 Y12 Y13] +
(1.4548424490530647e-06+0j) [X2 Z3 Z4 Y5 Y6 X7] +
(3.1513463111289834e-06+0j) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11] +
(0.01925750509525157+0j) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.5224930676388456e-06+0j) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10] +
(-0.00854199662545482+0j) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(7.956895372134663e-07+0j) [X2 Z3 Z4 Y5 Y8 X9] +
(4.64305106838824e-06+0j) [X2 Z3 Z4 Y5 Y10 X11] +
(0.019028242443847244+0j) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13] +
(-0.00876482757568874+0j) [X2 Z3 Z4 Y5 Y11 X12] +
(-5.275883122014173e-06+0j) [X2 Z3 Z4 Y5 Y12 X13] +
(-0.1213327691104227+0j) [X2 Z3 Z4 Z5 X6] +
(-0.008469978791023914+0j) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(2.6863815451696547e-07+0j) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.008469978791023914+0j) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(2.6863815451696547e-07+0j) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.025996177598021142+0j) [X2 Z3 Z4 Z5 X6 Z7] +
(-0.005805188989826892+0j) [X2 Z3 Z4 Z5 X6 Z8] +
(-0.017561202409646114+0j) [X2 Z3 Z4 Z5 X6 Z9] +
(-7.988770288405068e-07+0j) [X2 Z3 Z4 Z5 X6 X10 Z11 X12] +
(-3.4273231086113966e-07+0j) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12] +
(-0.0008145313270956646+0j) [X2 Z3 Z4 Z5 X6 Z10] +
(2.7455184003597968e-06+0j) [X2 Z3 Z4 Z5 X6 X11 Z12 X13] +
(2.7455184003597968e-06+0j) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13] +
(0.01441109943013093+0j) [X2 Z3 Z4 Z5 X6 Z11] +
(0.0006650070219499674+0j) [X2 Z3 Z4 Z5 X6 Z12] +
(-0.003493790359890077+0j) [X2 Z3 Z4 Z5 X6 Z13] +
(-4.56144717979367e-07+0j) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12] +
(-0.011756013419819222+0j) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9] +
(0.015225630757226596+0j) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11] +
(-3.0882507112209364e-06+0j) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(-3.5443954292003035e-06+0j) [X2 Z3 Z4 Z5 Z6 X7 X11 X12] +
(-0.0041587973818400445+0j) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13] +
(0.011756013419819222+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9] +
(-0.015225630757226596+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11] +
(3.0882507112209364e-06+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(-3.5443954292003035e-06+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12] +
(0.0041587973818400445+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13] +
(0.014603704729162092+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(-2.874299071311125e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.014603704729162092+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(-2.874299071311125e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(-0.28164257767022804+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-1.3002946562197188e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(-1.3002946562197188e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(-0.024282117354692975+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(-0.01953805031131467+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(-0.017091553155898803+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(0.002446497155415865+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(-0.002446497155415865+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(5.775950527043989e-05+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(2.8836765759603235e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(5.146496327392735e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(3.846201671173017e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(-0.039359168022053054+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10] +
(7.979825793172219e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(-0.02475546329289096+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10] +
(5.105526721861094e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(-0.021433810721600915+0j) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10] +
(5.159350501798794e-06+0j) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(-0.02990378951262483+0j) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10] +
(5.42798865631576e-06+0j) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0016638798784907576+0j) [X2 Z3 Z4 X6] +
(-0.018889030304942957+0j) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10] +
(2.9473560116224046e-06+0j) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.00347951189033433+0j) [X2 Z3 Z5 X6] +
(-0.02873077955190552+0j) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10] +
(5.9358677178890296e-06+0j) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.602116740623028e-06+0j) [X2 X4] +
(0.0004956762314916191+0j) [X2 Z4 Z5 X6] +
(-0.035608378988312595+0j) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(7.253273347895439e-06+0j) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.035839567953353434+0j) [Y2 X3 X4 Y5] +
(2.1990516183020972e-07+0j) [Y2 X3 X4 Z5 Z6 Y7] +
(2.360956320222924e-06+0j) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.01031148248983184+0j) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.1990516183020975e-07+0j) [Y2 X3 X5 Y6] +
(-2.360956320222924e-06+0j) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.01031148248983184+0j) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.031143817988967173+0j) [Y2 X3 X6 Y7] +
(-0.005368659358109595+0j) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11] +
(-9.209350644760531e-08+0j) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005368659358109595+0j) [Y2 X3 X7 Z8 Z9 Y10] +
(9.209350644760531e-08+0j) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12] +
(0.03619412355904266+0j) [Y2 X3 X8 Y9] +
(0.02538465750845736+0j) [Y2 X3 X10 Y11] +
(-2.1726691014281675e-06+0j) [Y2 X3 X10 Z11 Z12 Y13] +
(2.1726691014281675e-06+0j) [Y2 X3 X11 Y12] +
(0.015577208063976443+0j) [Y2 X3 X12 Y13] +
(-0.035839567953353434+0j) [Y2 Y3 X4 X5] +
(-2.1990516183020972e-07+0j) [Y2 Y3 X4 Z5 Z6 X7] +
(-2.360956320222924e-06+0j) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.01031148248983184+0j) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.1990516183020975e-07+0j) [Y2 Y3 Y5 Y6] +
(-2.360956320222924e-06+0j) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.01031148248983184+0j) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.031143817988967173+0j) [Y2 Y3 X6 X7] +
(0.005368659358109595+0j) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11] +
(9.209350644760531e-08+0j) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005368659358109595+0j) [Y2 Y3 Y7 Z8 Z9 Y10] +
(9.209350644760531e-08+0j) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.03619412355904266+0j) [Y2 Y3 X8 X9] +
(-0.02538465750845736+0j) [Y2 Y3 X10 X11] +
(2.1726691014281675e-06+0j) [Y2 Y3 X10 Z11 Z12 X13] +
(2.1726691014281675e-06+0j) [Y2 Y3 Y11 Y12] +
(-0.015577208063976443+0j) [Y2 Y3 X12 X13] +
(1.6288532434901385e-06+0j) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10] +
(0.010715508469796757+0j) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.010263414868158502+0j) [Y2 Z3 X4 X10 Z11 Y12] +
(-3.887051672348346e-06+0j) [Y2 Z3 Y4] +
(-0.0051433917688250876+0j) [Y2 Z3 Y4 X5 Z6 X7] +
(-0.009841749246962565+0j) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.9885117062666245e-06+0j) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0051433917688250876+0j) [Y2 Z3 Y4 Y5 Z6 Y7] +
(-0.009841749246962565+0j) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.9885117062666245e-06+0j) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.76499411792064e-07+0j) [Y2 Z3 Y4 Z5] +
(4.537178096273229e-08+0j) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10] +
(4.2055484112176605e-05+0j) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.6893489514423537e-06+0j) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10] +
(0.010757563953908934+0j) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.5935343913351927e-07+0j) [Y2 Z3 Y4 Z6] +
(3.211842019081199e-06+0j) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11] +
(0.01929956057936375+0j) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.211842019081199e-06+0j) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11] +
(0.01929956057936375+0j) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.1954890099195454e-06+0j) [Y2 Z3 Y4 Z7] +
(2.1868423778095717e-07+0j) [Y2 Z3 Y4 Z8] +
(-5.770052994325092e-07+0j) [Y2 Z3 Y4 Z9] +
(0.005324835234221682+0j) [Y2 Z3 Y4 X10 Z11 X12] +
(0.015588250102380184+0j) [Y2 Z3 Y4 Y10 Z11 Y12] +
(-3.158656431943763e-06+0j) [Y2 Z3 Y4 Z10] +
(0.024353077678068928+0j) [Y2 Z3 Y4 X11 Z12 X13] +
(0.024353077678068928+0j) [Y2 Z3 Y4 Y11 Z12 Y13] +
(-7.801707500332004e-06+0j) [Y2 Z3 Y4 Z11] +
(3.5390541844940366e-06+0j) [Y2 Z3 Y4 Z12] +
(8.81493730650821e-06+0j) [Y2 Z3 Y4 Z13] +
(1.4548424490530647e-06+0j) [Y2 Z3 Z4 X5 X6 Y7] +
(3.1513463111289834e-06+0j) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11] +
(0.01925750509525157+0j) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.5224930676388456e-06+0j) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10] +
(-0.00854199662545482+0j) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(7.956895372134663e-07+0j) [Y2 Z3 Z4 X5 X8 Y9] +
(4.64305106838824e-06+0j) [Y2 Z3 Z4 X5 X10 Y11] +
(0.019028242443847244+0j) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13] +
(-0.00876482757568874+0j) [Y2 Z3 Z4 X5 X11 Y12] +
(-5.275883122014173e-06+0j) [Y2 Z3 Z4 X5 X12 Y13] +
(-1.4548424490530647e-06+0j) [Y2 Z3 Z4 Y5 X6 X7] +
(-3.1513463111289834e-06+0j) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11] +
(-0.01925750509525157+0j) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.5224930676388456e-06+0j) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10] +
(-0.00854199662545482+0j) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-7.956895372134663e-07+0j) [Y2 Z3 Z4 Y5 X8 X9] +
(-4.64305106838824e-06+0j) [Y2 Z3 Z4 Y5 X10 X11] +
(-0.019028242443847244+0j) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13] +
(-0.00876482757568874+0j) [Y2 Z3 Z4 Y5 Y11 Y12] +
(5.275883122014173e-06+0j) [Y2 Z3 Z4 Y5 X12 X13] +
(-4.56144717979367e-07+0j) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12] +
(-0.1213327691104227+0j) [Y2 Z3 Z4 Z5 Y6] +
(-0.008469978791023914+0j) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(2.6863815451696547e-07+0j) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.008469978791023914+0j) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(2.6863815451696547e-07+0j) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.025996177598021142+0j) [Y2 Z3 Z4 Z5 Y6 Z7] +
(-0.005805188989826892+0j) [Y2 Z3 Z4 Z5 Y6 Z8] +
(-0.017561202409646114+0j) [Y2 Z3 Z4 Z5 Y6 Z9] +
(-3.4273231086113966e-07+0j) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12] +
(-7.988770288405068e-07+0j) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12] +
(-0.0008145313270956646+0j) [Y2 Z3 Z4 Z5 Y6 Z10] +
(2.7455184003597968e-06+0j) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13] +
(2.7455184003597968e-06+0j) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13] +
(0.01441109943013093+0j) [Y2 Z3 Z4 Z5 Y6 Z11] +
(0.0006650070219499674+0j) [Y2 Z3 Z4 Z5 Y6 Z12] +
(-0.003493790359890077+0j) [Y2 Z3 Z4 Z5 Y6 Z13] +
(0.011756013419819222+0j) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9] +
(-0.015225630757226596+0j) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11] +
(3.0882507112209364e-06+0j) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(-3.5443954292003035e-06+0j) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12] +
(0.0041587973818400445+0j) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13] +
(-0.011756013419819222+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9] +
(0.015225630757226596+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11] +
(-3.0882507112209364e-06+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(-3.5443954292003035e-06+0j) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12] +
(-0.0041587973818400445+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13] +
(0.014603704729162092+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(-2.874299071311125e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.014603704729162092+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(-2.874299071311125e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(-0.28164257767022804+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-1.3002946562197188e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(-1.3002946562197188e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(-0.024282117354692975+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(-0.01953805031131467+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(-0.017091553155898803+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(-0.002446497155415865+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(0.002446497155415865+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(5.775950527043989e-05+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.8836765759603235e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(5.146496327392735e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(3.846201671173017e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(-0.039359168022053054+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10] +
(7.979825793172219e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(-0.02475546329289096+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10] +
(5.105526721861094e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(-0.021433810721600915+0j) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10] +
(5.159350501798794e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(-0.02990378951262483+0j) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10] +
(5.42798865631576e-06+0j) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0016638798784907576+0j) [Y2 Z3 Z4 Y6] +
(-0.018889030304942957+0j) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10] +
(2.9473560116224046e-06+0j) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.00347951189033433+0j) [Y2 Z3 Z5 Y6] +
(-0.02873077955190552+0j) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10] +
(5.9358677178890296e-06+0j) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.602116740623028e-06+0j) [Y2 Y4] +
(0.0004956762314916191+0j) [Y2 Z4 Z5 Y6] +
(-0.035608378988312595+0j) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(7.253273347895439e-06+0j) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.6538942226831712+0j) [Z2] +
(1.602116740623028e-06+0j) [Z2 X3 Z4 X5] +
(0.0004956762314916191+0j) [Z2 X3 Z4 Z5 Z6 X7] +
(-0.0356083789883126+0j) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(7.253273347895439e-06+0j) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.602116740623028e-06+0j) [Z2 Y3 Z4 Y5] +
(0.0004956762314916191+0j) [Z2 Y3 Z4 Z5 Z6 Y7] +
(-0.0356083789883126+0j) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(7.253273347895439e-06+0j) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.18189085790751375+0j) [Z2 Z3] +
(-9.509249752422925e-07+0j) [Z2 X4 Z5 X6] +
(-4.72884314709104e-06+0j) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.02459186088383003+0j) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-9.509249752422925e-07+0j) [Z2 Y4 Z5 Y6] +
(-4.72884314709104e-06+0j) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.02459186088383003+0j) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.12495807739503222+0j) [Z2 Z4] +
(-1.1708301370725025e-06+0j) [Z2 X5 Z6 X7] +
(-7.089799467313965e-06+0j) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.03490334337366187+0j) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.1708301370725025e-06+0j) [Z2 Y5 Z6 Y7] +
(-7.089799467313965e-06+0j) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.03490334337366187+0j) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.16079764534838567+0j) [Z2 Z5] +
(0.019020423173039983+0j) [Z2 X6 Z7 Z8 Z9 X10] +
(-2.1032156046071815e-06+0j) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.019020423173039983+0j) [Z2 Y6 Z7 Z8 Z9 Y10] +
(-2.1032156046071815e-06+0j) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.1373910476268323+0j) [Z2 Z6] +
(0.024389082531149575+0j) [Z2 X7 Z8 Z9 Z10 X11] +
(-2.0111220981595764e-06+0j) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.024389082531149575+0j) [Z2 Y7 Z8 Z9 Z10 Y11] +
(-2.0111220981595764e-06+0j) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.16853486561579945+0j) [Z2 Z7] +
(0.15071408121008287+0j) [Z2 Z8] +
(0.18690820476912556+0j) [Z2 Z9] +
(-1.0632283423606647e-06+0j) [Z2 X10 Z11 X12] +
(-1.0632283423606647e-06+0j) [Z2 Y10 Z11 Y12] +
(0.1279950249246841+0j) [Z2 Z10] +
(1.1094407590675027e-06+0j) [Z2 X11 Z12 X13] +
(1.1094407590675027e-06+0j) [Z2 Y11 Z12 Y13] +
(0.15337968243314146+0j) [Z2 Z11] +
(0.14011289865354803+0j) [Z2 Z12] +
(0.15569010671752448+0j) [Z2 Z13] +
(0.0051433917688250876+0j) [X3 X4 Y5 Y6] +
(0.009841749246962565+0j) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10] +
(-2.9885117062666245e-06+0j) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.4548424490530645e-06+0j) [X3 X4 X6 X7] +
(-1.5224930676388456e-06+0j) [X3 X4 X6 Z7 Z8 Z9 Z10 X11] +
(-0.00854199662545482+0j) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.1513463111289834e-06+0j) [X3 X4 Y7 Z8 Z9 Y10] +
(-0.01925750509525157+0j) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12] +
(-7.956895372134663e-07+0j) [X3 X4 X8 X9] +
(-4.6430510683882405e-06+0j) [X3 X4 X10 X11] +
(-0.00876482757568874+0j) [X3 X4 X10 Z11 Z12 X13] +
(-0.019028242443847244+0j) [X3 X4 Y11 Y12] +
(5.275883122014173e-06+0j) [X3 X4 X12 X13] +
(-0.0051433917688250876+0j) [X3 Y4 Y5 X6] +
(-0.009841749246962565+0j) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10] +
(2.9885117062666245e-06+0j) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.4548424490530645e-06+0j) [X3 Y4 Y6 X7] +
(-1.5224930676388456e-06+0j) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11] +
(-0.00854199662545482+0j) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.1513463111289834e-06+0j) [X3 Y4 Y7 Z8 Z9 X10] +
(0.01925750509525157+0j) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12] +
(-7.956895372134663e-07+0j) [X3 Y4 Y8 X9] +
(-4.6430510683882405e-06+0j) [X3 Y4 Y10 X11] +
(-0.00876482757568874+0j) [X3 Y4 Y10 Z11 Z12 X13] +
(0.019028242443847244+0j) [X3 Y4 Y11 X12] +
(5.275883122014173e-06+0j) [X3 Y4 Y12 X13] +
(-3.887051672348349e-06+0j) [X3 Z4 X5] +
(3.211842019081199e-06+0j) [X3 Z4 X5 X6 Z7 Z8 Z9 X10] +
(0.01929956057936375+0j) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.211842019081199e-06+0j) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10] +
(0.01929956057936375+0j) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.1954890099195454e-06+0j) [X3 Z4 X5 Z6] +
(1.6893489514423537e-06+0j) [X3 Z4 X5 X7 Z8 Z9 Z10 X11] +
(0.010757563953908934+0j) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.537178096273229e-08+0j) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11] +
(4.2055484112176605e-05+0j) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.5935343913351927e-07+0j) [X3 Z4 X5 Z7] +
(-5.770052994325092e-07+0j) [X3 Z4 X5 Z8] +
(2.1868423778095717e-07+0j) [X3 Z4 X5 Z9] +
(0.024353077678068928+0j) [X3 Z4 X5 X10 Z11 X12] +
(0.024353077678068928+0j) [X3 Z4 X5 Y10 Z11 Y12] +
(-7.801707500332004e-06+0j) [X3 Z4 X5 Z10] +
(0.015588250102380184+0j) [X3 Z4 X5 X11 Z12 X13] +
(0.005324835234221682+0j) [X3 Z4 X5 Y11 Z12 Y13] +
(-3.158656431943763e-06+0j) [X3 Z4 X5 Z11] +
(8.81493730650821e-06+0j) [X3 Z4 X5 Z12] +
(3.5390541844940366e-06+0j) [X3 Z4 X5 Z13] +
(1.6288532434901385e-06+0j) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11] +
(0.010715508469796757+0j) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.010263414868158502+0j) [X3 Z4 Y5 Y11 Z12 X13] +
(0.008469978791023914+0j) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10] +
(-2.6863815451696547e-07+0j) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.011756013419819224+0j) [X3 Z4 Z5 X6 X8 X9] +
(0.015225630757226596+0j) [X3 Z4 Z5 X6 X10 X11] +
(-3.5443954292003035e-06+0j) [X3 Z4 Z5 X6 X10 Z11 Z12 X13] +
(-3.0882507112209364e-06+0j) [X3 Z4 Z5 X6 Y11 Y12] +
(-0.0041587973818400445+0j) [X3 Z4 Z5 X6 X12 X13] +
(-0.008469978791023914+0j) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10] +
(2.6863815451696547e-07+0j) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.011756013419819224+0j) [X3 Z4 Z5 Y6 Y8 X9] +
(0.015225630757226596+0j) [X3 Z4 Z5 Y6 Y10 X11] +
(-3.5443954292003035e-06+0j) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13] +
(3.0882507112209364e-06+0j) [X3 Z4 Z5 Y6 Y11 X12] +
(-0.0041587973818400445+0j) [X3 Z4 Z5 Y6 Y12 X13] +
(-0.12133276911042266+0j) [X3 Z4 Z5 Z6 X7] +
(-0.017561202409646114+0j) [X3 Z4 Z5 Z6 X7 Z8] +
(-0.005805188989826892+0j) [X3 Z4 Z5 Z6 X7 Z9] +
(2.7455184003597968e-06+0j) [X3 Z4 Z5 Z6 X7 X10 Z11 X12] +
(2.7455184003597968e-06+0j) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12] +
(0.01441109943013093+0j) [X3 Z4 Z5 Z6 X7 Z10] +
(-7.988770288405068e-07+0j) [X3 Z4 Z5 Z6 X7 X11 Z12 X13] +
(-3.4273231086113966e-07+0j) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13] +
(-0.0008145313270956646+0j) [X3 Z4 Z5 Z6 X7 Z11] +
(-0.003493790359890077+0j) [X3 Z4 Z5 Z6 X7 Z12] +
(0.0006650070219499674+0j) [X3 Z4 Z5 Z6 X7 Z13] +
(-4.56144717979367e-07+0j) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13] +
(-0.014603704729162094+0j) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10] +
(2.874299071311125e-06+0j) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(0.014603704729162094+0j) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10] +
(-2.874299071311125e-06+0j) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(1.3002946562197188e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12] +
(0.002446497155415865+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-1.3002946562197188e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12] +
(0.002446497155415865+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(-0.2816425776702279+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.017091553155898803+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(-0.01953805031131467+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(5.775950527043989e-05+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(2.883676575960324e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(3.846201671173017e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(-0.024282117354692975+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11] +
(5.146496327392735e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(-0.02475546329289096+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11] +
(5.105526721861094e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(-0.039359168022053054+0j) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11] +
(7.979825793172219e-06+0j) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(-0.02990378951262483+0j) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11] +
(5.42798865631576e-06+0j) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.025996177598021142+0j) [X3 Z4 Z5 X7] +
(-0.021433810721600915+0j) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11] +
(5.159350501798794e-06+0j) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.00347951189033433+0j) [X3 Z4 Z6 X7] +
(-0.02873077955190552+0j) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11] +
(5.9358677178890296e-06+0j) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.764994117920641e-07+0j) [X3 X5] +
(0.0016638798784907576+0j) [X3 Z5 Z6 X7] +
(-0.018889030304942957+0j) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.9473560116224046e-06+0j) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0051433917688250876+0j) [Y3 X4 X5 Y6] +
(-0.009841749246962565+0j) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10] +
(2.9885117062666245e-06+0j) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.4548424490530645e-06+0j) [Y3 X4 X6 Y7] +
(-1.5224930676388456e-06+0j) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11] +
(-0.00854199662545482+0j) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.1513463111289834e-06+0j) [Y3 X4 X7 Z8 Z9 Y10] +
(0.01925750509525157+0j) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12] +
(-7.956895372134663e-07+0j) [Y3 X4 X8 Y9] +
(-4.6430510683882405e-06+0j) [Y3 X4 X10 Y11] +
(-0.00876482757568874+0j) [Y3 X4 X10 Z11 Z12 Y13] +
(0.019028242443847244+0j) [Y3 X4 X11 Y12] +
(5.275883122014173e-06+0j) [Y3 X4 X12 Y13] +
(0.0051433917688250876+0j) [Y3 Y4 X5 X6] +
(0.009841749246962565+0j) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10] +
(-2.9885117062666245e-06+0j) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.4548424490530645e-06+0j) [Y3 Y4 Y6 Y7] +
(-1.5224930676388456e-06+0j) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11] +
(-0.00854199662545482+0j) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.1513463111289834e-06+0j) [Y3 Y4 X7 Z8 Z9 X10] +
(-0.01925750509525157+0j) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12] +
(-7.956895372134663e-07+0j) [Y3 Y4 Y8 Y9] +
(-4.6430510683882405e-06+0j) [Y3 Y4 Y10 Y11] +
(-0.00876482757568874+0j) [Y3 Y4 Y10 Z11 Z12 Y13] +
(-0.019028242443847244+0j) [Y3 Y4 X11 X12] +
(5.275883122014173e-06+0j) [Y3 Y4 Y12 Y13] +
(1.6288532434901385e-06+0j) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11] +
(0.010715508469796757+0j) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.010263414868158502+0j) [Y3 Z4 X5 X11 Z12 Y13] +
(-3.887051672348349e-06+0j) [Y3 Z4 Y5] +
(3.211842019081199e-06+0j) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10] +
(0.01929956057936375+0j) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.211842019081199e-06+0j) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10] +
(0.01929956057936375+0j) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.1954890099195454e-06+0j) [Y3 Z4 Y5 Z6] +
(4.537178096273229e-08+0j) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11] +
(4.2055484112176605e-05+0j) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.6893489514423537e-06+0j) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11] +
(0.010757563953908934+0j) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.5935343913351927e-07+0j) [Y3 Z4 Y5 Z7] +
(-5.770052994325092e-07+0j) [Y3 Z4 Y5 Z8] +
(2.1868423778095717e-07+0j) [Y3 Z4 Y5 Z9] +
(0.024353077678068928+0j) [Y3 Z4 Y5 X10 Z11 X12] +
(0.024353077678068928+0j) [Y3 Z4 Y5 Y10 Z11 Y12] +
(-7.801707500332004e-06+0j) [Y3 Z4 Y5 Z10] +
(0.005324835234221682+0j) [Y3 Z4 Y5 X11 Z12 X13] +
(0.015588250102380184+0j) [Y3 Z4 Y5 Y11 Z12 Y13] +
(-3.158656431943763e-06+0j) [Y3 Z4 Y5 Z11] +
(8.81493730650821e-06+0j) [Y3 Z4 Y5 Z12] +
(3.5390541844940366e-06+0j) [Y3 Z4 Y5 Z13] +
(-0.008469978791023914+0j) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10] +
(2.6863815451696547e-07+0j) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.011756013419819224+0j) [Y3 Z4 Z5 X6 X8 Y9] +
(0.015225630757226596+0j) [Y3 Z4 Z5 X6 X10 Y11] +
(-3.5443954292003035e-06+0j) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13] +
(3.0882507112209364e-06+0j) [Y3 Z4 Z5 X6 X11 Y12] +
(-0.0041587973818400445+0j) [Y3 Z4 Z5 X6 X12 Y13] +
(0.008469978791023914+0j) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10] +
(-2.6863815451696547e-07+0j) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-0.011756013419819224+0j) [Y3 Z4 Z5 Y6 Y8 Y9] +
(0.015225630757226596+0j) [Y3 Z4 Z5 Y6 Y10 Y11] +
(-3.5443954292003035e-06+0j) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13] +
(-3.0882507112209364e-06+0j) [Y3 Z4 Z5 Y6 X11 X12] +
(-0.0041587973818400445+0j) [Y3 Z4 Z5 Y6 Y12 Y13] +
(-4.56144717979367e-07+0j) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13] +
(-0.12133276911042266+0j) [Y3 Z4 Z5 Z6 Y7] +
(-0.017561202409646114+0j) [Y3 Z4 Z5 Z6 Y7 Z8] +
(-0.005805188989826892+0j) [Y3 Z4 Z5 Z6 Y7 Z9] +
(2.7455184003597968e-06+0j) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12] +
(2.7455184003597968e-06+0j) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12] +
(0.01441109943013093+0j) [Y3 Z4 Z5 Z6 Y7 Z10] +
(-3.4273231086113966e-07+0j) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13] +
(-7.988770288405068e-07+0j) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13] +
(-0.0008145313270956646+0j) [Y3 Z4 Z5 Z6 Y7 Z11] +
(-0.003493790359890077+0j) [Y3 Z4 Z5 Z6 Y7 Z12] +
(0.0006650070219499674+0j) [Y3 Z4 Z5 Z6 Y7 Z13] +
(0.014603704729162094+0j) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10] +
(-2.874299071311125e-06+0j) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-0.014603704729162094+0j) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10] +
(2.874299071311125e-06+0j) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-1.3002946562197188e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12] +
(0.002446497155415865+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(1.3002946562197188e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12] +
(0.002446497155415865+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(-0.2816425776702279+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.017091553155898803+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(-0.01953805031131467+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(5.775950527043989e-05+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.883676575960324e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(3.846201671173017e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(-0.024282117354692975+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11] +
(5.146496327392735e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(-0.02475546329289096+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11] +
(5.105526721861094e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(-0.039359168022053054+0j) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11] +
(7.979825793172219e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(-0.02990378951262483+0j) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11] +
(5.42798865631576e-06+0j) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.025996177598021142+0j) [Y3 Z4 Z5 Y7] +
(-0.021433810721600915+0j) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11] +
(5.159350501798794e-06+0j) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.00347951189033433+0j) [Y3 Z4 Z6 Y7] +
(-0.02873077955190552+0j) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11] +
(5.9358677178890296e-06+0j) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.764994117920641e-07+0j) [Y3 Y5] +
(0.0016638798784907576+0j) [Y3 Z5 Z6 Y7] +
(-0.018889030304942957+0j) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.9473560116224046e-06+0j) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.6538942226831712+0j) [Z3] +
(-1.1708301370725025e-06+0j) [Z3 X4 Z5 X6] +
(-7.089799467313965e-06+0j) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.03490334337366187+0j) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.1708301370725025e-06+0j) [Z3 Y4 Z5 Y6] +
(-7.089799467313965e-06+0j) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.03490334337366187+0j) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.16079764534838567+0j) [Z3 Z4] +
(-9.509249752422925e-07+0j) [Z3 X5 Z6 X7] +
(-4.72884314709104e-06+0j) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.02459186088383003+0j) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-9.509249752422925e-07+0j) [Z3 Y5 Z6 Y7] +
(-4.72884314709104e-06+0j) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.02459186088383003+0j) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.12495807739503222+0j) [Z3 Z5] +
(0.024389082531149575+0j) [Z3 X6 Z7 Z8 Z9 X10] +
(-2.0111220981595764e-06+0j) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.024389082531149575+0j) [Z3 Y6 Z7 Z8 Z9 Y10] +
(-2.0111220981595764e-06+0j) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.16853486561579945+0j) [Z3 Z6] +
(0.019020423173039983+0j) [Z3 X7 Z8 Z9 Z10 X11] +
(-2.1032156046071815e-06+0j) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.019020423173039983+0j) [Z3 Y7 Z8 Z9 Z10 Y11] +
(-2.1032156046071815e-06+0j) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.1373910476268323+0j) [Z3 Z7] +
(0.18690820476912556+0j) [Z3 Z8] +
(0.15071408121008287+0j) [Z3 Z9] +
(1.1094407590675027e-06+0j) [Z3 X10 Z11 X12] +
(1.1094407590675027e-06+0j) [Z3 Y10 Z11 Y12] +
(0.15337968243314146+0j) [Z3 Z10] +
(-1.0632283423606647e-06+0j) [Z3 X11 Z12 X13] +
(-1.0632283423606647e-06+0j) [Z3 Y11 Z12 Y13] +
(0.1279950249246841+0j) [Z3 Z11] +
(0.15569010671752448+0j) [Z3 Z12] +
(0.14011289865354803+0j) [Z3 Z13] +
(-0.011982389010247955+0j) [X4 X5 Y6 Y7] +
(-0.007306759928832947+0j) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11] +
(-2.888293595502657e-07+0j) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.007306759928832947+0j) [X4 X5 X7 Z8 Z9 X10] +
(-2.888293595502657e-07+0j) [X4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-0.007156934919856957+0j) [X4 X5 Y8 Y9] +
(-0.017680067952481504+0j) [X4 X5 Y10 Y11] +
(-3.6945132943766862e-06+0j) [X4 X5 Y10 Z11 Z12 Y13] +
(-3.694513294376686e-06+0j) [X4 X5 X11 X12] +
(-0.03831467029480389+0j) [X4 X5 Y12 Y13] +
(0.011982389010247955+0j) [X4 Y5 Y6 X7] +
(0.007306759928832947+0j) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11] +
(2.888293595502657e-07+0j) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.007306759928832947+0j) [X4 Y5 Y7 Z8 Z9 X10] +
(-2.888293595502657e-07+0j) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(0.007156934919856957+0j) [X4 Y5 Y8 X9] +
(0.017680067952481504+0j) [X4 Y5 Y10 X11] +
(3.6945132943766862e-06+0j) [X4 Y5 Y10 Z11 Z12 X13] +
(-3.694513294376686e-06+0j) [X4 Y5 Y11 X12] +
(0.03831467029480389+0j) [X4 Y5 Y12 X13] +
(-1.2260484989582672e-05+0j) [X4 Z5 X6] +
(-1.2283337824929527e-06+0j) [X4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(0.0002463643756957804+0j) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2283337824929527e-06+0j) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(0.0002463643756957804+0j) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.85406085802419e-06+0j) [X4 Z5 X6 Z7] +
(-1.3980449081991513e-06+0j) [X4 Z5 X6 Z8] +
(-1.8818501832808126e-06+0j) [X4 Z5 X6 Z9] +
(0.00796088072592158+0j) [X4 Z5 X6 X10 Z11 X12] +
(-0.000929850796773043+0j) [X4 Z5 X6 Y10 Z11 Y12] +
(-1.692397828600888e-06+0j) [X4 Z5 X6 Z10] +
(-0.012215040997613962+0j) [X4 Z5 X6 X11 Z12 X13] +
(-0.012215040997613962+0j) [X4 Z5 X6 Y11 Z12 Y13] +
(4.281913884693185e-06+0j) [X4 Z5 X6 Z11] +
(-4.5888551556732865e-06+0j) [X4 Z5 X6 Z13] +
(0.008890731522694623+0j) [X4 Z5 Y6 Y10 Z11 X12] +
(-4.838052750816613e-07+0j) [X4 Z5 Z6 X7 Y8 Y9] +
(5.974311713294073e-06+0j) [X4 Z5 Z6 X7 Y10 Y11] +
(0.011285190200840919+0j) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(0.02017592172353554+0j) [X4 Z5 Z6 X7 X11 X12] +
(-4.556569218075857e-06+0j) [X4 Z5 Z6 X7 Y12 Y13] +
(4.838052750816613e-07+0j) [X4 Z5 Z6 Y7 Y8 X9] +
(-5.974311713294073e-06+0j) [X4 Z5 Z6 Y7 Y10 X11] +
(-0.011285190200840919+0j) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(0.02017592172353554+0j) [X4 Z5 Z6 Y7 Y11 X12] +
(4.556569218075857e-06+0j) [X4 Z5 Z6 Y7 Y12 X13] +
(1.3304731886332367e-06+0j) [X4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(0.005923798336561342+0j) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(1.3304731886332367e-06+0j) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(0.005923798336561342+0j) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(-6.631277928278836e-05+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.016024603689179566+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(-0.016024603689179566+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(3.334331289371026e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(-4.734622038607147e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(-9.806102775012614e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(-5.071480736405466e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(5.071480736405466e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-0.3693708936615612+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.023145130929528964+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-0.009612634606847276+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(-0.025637238296026838+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(-8.774817864374587e-06+0j) [X4 Z5 Z6 Z7 Z8 X10] +
(-0.04764261217638313+0j) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(-7.44434467574135e-06+0j) [X4 Z5 Z6 Z7 Z9 X10] +
(-0.04171881383982179+0j) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(-6.290028433041983e-06+0j) [X4 Z5 Z6 Z8 Z9 X10] +
(-0.03956441632289334+0j) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(-7.518362215534937e-06+0j) [X4 Z5 Z7 Z8 Z9 X10] +
(-0.03931805194719756+0j) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(-5.929765816535534e-07+0j) [X4 X6] +
(-4.253224225482298e-06+0j) [X4 Z6 Z7 Z8 Z9 X10] +
(-0.02252844019601312+0j) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.011982389010247955+0j) [Y4 X5 X6 Y7] +
(0.007306759928832947+0j) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11] +
(2.888293595502657e-07+0j) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.007306759928832947+0j) [Y4 X5 X7 Z8 Z9 Y10] +
(-2.888293595502657e-07+0j) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(0.007156934919856957+0j) [Y4 X5 X8 Y9] +
(0.017680067952481504+0j) [Y4 X5 X10 Y11] +
(3.6945132943766862e-06+0j) [Y4 X5 X10 Z11 Z12 Y13] +
(-3.694513294376686e-06+0j) [Y4 X5 X11 Y12] +
(0.03831467029480389+0j) [Y4 X5 X12 Y13] +
(-0.011982389010247955+0j) [Y4 Y5 X6 X7] +
(-0.007306759928832947+0j) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11] +
(-2.888293595502657e-07+0j) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.007306759928832947+0j) [Y4 Y5 Y7 Z8 Z9 Y10] +
(-2.888293595502657e-07+0j) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.007156934919856957+0j) [Y4 Y5 X8 X9] +
(-0.017680067952481504+0j) [Y4 Y5 X10 X11] +
(-3.6945132943766862e-06+0j) [Y4 Y5 X10 Z11 Z12 X13] +
(-3.694513294376686e-06+0j) [Y4 Y5 Y11 Y12] +
(-0.03831467029480389+0j) [Y4 Y5 X12 X13] +
(0.008890731522694623+0j) [Y4 Z5 X6 X10 Z11 Y12] +
(-1.2260484989582672e-05+0j) [Y4 Z5 Y6] +
(-1.2283337824929527e-06+0j) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(0.0002463643756957804+0j) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2283337824929527e-06+0j) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(0.0002463643756957804+0j) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.85406085802419e-06+0j) [Y4 Z5 Y6 Z7] +
(-1.3980449081991513e-06+0j) [Y4 Z5 Y6 Z8] +
(-1.8818501832808126e-06+0j) [Y4 Z5 Y6 Z9] +
(-0.000929850796773043+0j) [Y4 Z5 Y6 X10 Z11 X12] +
(0.00796088072592158+0j) [Y4 Z5 Y6 Y10 Z11 Y12] +
(-1.692397828600888e-06+0j) [Y4 Z5 Y6 Z10] +
(-0.012215040997613962+0j) [Y4 Z5 Y6 X11 Z12 X13] +
(-0.012215040997613962+0j) [Y4 Z5 Y6 Y11 Z12 Y13] +
(4.281913884693185e-06+0j) [Y4 Z5 Y6 Z11] +
(-4.5888551556732865e-06+0j) [Y4 Z5 Y6 Z13] +
(4.838052750816613e-07+0j) [Y4 Z5 Z6 X7 X8 Y9] +
(-5.974311713294073e-06+0j) [Y4 Z5 Z6 X7 X10 Y11] +
(-0.011285190200840919+0j) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(0.02017592172353554+0j) [Y4 Z5 Z6 X7 X11 Y12] +
(4.556569218075857e-06+0j) [Y4 Z5 Z6 X7 X12 Y13] +
(-4.838052750816613e-07+0j) [Y4 Z5 Z6 Y7 X8 X9] +
(5.974311713294073e-06+0j) [Y4 Z5 Z6 Y7 X10 X11] +
(0.011285190200840919+0j) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(0.02017592172353554+0j) [Y4 Z5 Z6 Y7 Y11 Y12] +
(-4.556569218075857e-06+0j) [Y4 Z5 Z6 Y7 X12 X13] +
(1.3304731886332367e-06+0j) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(0.005923798336561342+0j) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(1.3304731886332367e-06+0j) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(0.005923798336561342+0j) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(-6.631277928278836e-05+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.016024603689179566+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(-0.016024603689179566+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(3.334331289371026e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(-4.734622038607147e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(-9.806102775012614e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(5.071480736405466e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-5.071480736405466e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-0.3693708936615612+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.023145130929528964+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-0.009612634606847276+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(-0.025637238296026838+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(-8.774817864374587e-06+0j) [Y4 Z5 Z6 Z7 Z8 Y10] +
(-0.04764261217638313+0j) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(-7.44434467574135e-06+0j) [Y4 Z5 Z6 Z7 Z9 Y10] +
(-0.04171881383982179+0j) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(-6.290028433041983e-06+0j) [Y4 Z5 Z6 Z8 Z9 Y10] +
(-0.03956441632289334+0j) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(-7.518362215534937e-06+0j) [Y4 Z5 Z7 Z8 Z9 Y10] +
(-0.03931805194719756+0j) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(-5.929765816535534e-07+0j) [Y4 Y6] +
(-4.253224225482298e-06+0j) [Y4 Z6 Z7 Z8 Z9 Y10] +
(-0.02252844019601312+0j) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.2034402289145631+0j) [Z4] +
(-5.929765816535533e-07+0j) [Z4 X5 Z6 X7] +
(-4.253224225482298e-06+0j) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.02252844019601312+0j) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-5.929765816535533e-07+0j) [Z4 Y5 Z6 Y7] +
(-4.253224225482298e-06+0j) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.02252844019601312+0j) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.15755314797985667+0j) [Z4 Z5] +
(0.018266834869375623+0j) [Z4 X6 Z7 Z8 Z9 X10] +
(-1.6541174769819948e-06+0j) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.018266834869375623+0j) [Z4 Y6 Z7 Z8 Z9 Y10] +
(-1.6541174769819948e-06+0j) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.13701191674040758+0j) [Z4 Z6] +
(0.010960074940542677+0j) [Z4 X7 Z8 Z9 Z10 X11] +
(-1.9429468365322604e-06+0j) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.010960074940542677+0j) [Z4 Y7 Z8 Z9 Z10 Y11] +
(-1.9429468365322604e-06+0j) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.14899430575065553+0j) [Z4 Z7] +
(0.149607026844453+0j) [Z4 Z8] +
(0.15676396176430998+0j) [Z4 Z9] +
(1.8782101247376814e-06+0j) [Z4 X10 Z11 X12] +
(1.8782101247376814e-06+0j) [Z4 Y10 Z11 Y12] +
(0.124899909172376+0j) [Z4 Z10] +
(-1.8163031696390049e-06+0j) [Z4 X11 Z12 X13] +
(-1.8163031696390049e-06+0j) [Z4 Y11 Z12 Y13] +
(0.1425799771248575+0j) [Z4 Z11] +
(0.11383573679388653+0j) [Z4 Z12] +
(0.15215040708869043+0j) [Z4 Z13] +
(1.2283337824929525e-06+0j) [X5 X6 Y7 Z8 Z9 Y10] +
(-0.0002463643756957804+0j) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-4.838052750816613e-07+0j) [X5 X6 X8 X9] +
(5.974311713294073e-06+0j) [X5 X6 X10 X11] +
(0.02017592172353554+0j) [X5 X6 X10 Z11 Z12 X13] +
(0.011285190200840919+0j) [X5 X6 Y11 Y12] +
(-4.556569218075857e-06+0j) [X5 X6 X12 X13] +
(-1.2283337824929525e-06+0j) [X5 Y6 Y7 Z8 Z9 X10] +
(0.0002463643756957804+0j) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-4.838052750816613e-07+0j) [X5 Y6 Y8 X9] +
(5.974311713294073e-06+0j) [X5 Y6 Y10 X11] +
(0.02017592172353554+0j) [X5 Y6 Y10 Z11 Z12 X13] +
(-0.011285190200840919+0j) [X5 Y6 Y11 X12] +
(-4.556569218075857e-06+0j) [X5 Y6 Y12 X13] +
(-1.2260484989582675e-05+0j) [X5 Z6 X7] +
(-1.8818501832808126e-06+0j) [X5 Z6 X7 Z8] +
(-1.3980449081991513e-06+0j) [X5 Z6 X7 Z9] +
(-0.012215040997613962+0j) [X5 Z6 X7 X10 Z11 X12] +
(-0.012215040997613962+0j) [X5 Z6 X7 Y10 Z11 Y12] +
(4.281913884693185e-06+0j) [X5 Z6 X7 Z10] +
(0.00796088072592158+0j) [X5 Z6 X7 X11 Z12 X13] +
(-0.000929850796773043+0j) [X5 Z6 X7 Y11 Z12 Y13] +
(-1.692397828600888e-06+0j) [X5 Z6 X7 Z11] +
(-4.5888551556732865e-06+0j) [X5 Z6 X7 Z12] +
(0.008890731522694623+0j) [X5 Z6 Y7 Y11 Z12 X13] +
(-1.3304731886332367e-06+0j) [X5 Z6 Z7 X8 Y9 Y10] +
(-0.005923798336561342+0j) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(1.3304731886332367e-06+0j) [X5 Z6 Z7 Y8 Y9 X10] +
(0.005923798336561342+0j) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(0.016024603689179566+0j) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12] +
(-5.071480736405466e-06+0j) [X5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-0.016024603689179566+0j) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12] +
(-5.071480736405466e-06+0j) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(-6.631277928278838e-05+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-9.806102775012614e-06+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(-4.734622038607147e-06+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(-0.3693708936615612+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.023145130929528964+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(-0.025637238296026838+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(3.3343312893710257e-06+0j) [X5 Z6 Z7 Z8 Z9 X11] +
(-0.009612634606847276+0j) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(-7.44434467574135e-06+0j) [X5 Z6 Z7 Z8 Z10 X11] +
(-0.04171881383982179+0j) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(-8.774817864374587e-06+0j) [X5 Z6 Z7 Z9 Z10 X11] +
(-0.04764261217638313+0j) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(-7.518362215534937e-06+0j) [X5 Z6 Z8 Z9 Z10 X11] +
(-0.03931805194719756+0j) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.85406085802419e-06+0j) [X5 X7] +
(-6.290028433041983e-06+0j) [X5 Z7 Z8 Z9 Z10 X11] +
(-0.03956441632289334+0j) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2283337824929525e-06+0j) [Y5 X6 X7 Z8 Z9 Y10] +
(0.0002463643756957804+0j) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-4.838052750816613e-07+0j) [Y5 X6 X8 Y9] +
(5.974311713294073e-06+0j) [Y5 X6 X10 Y11] +
(0.02017592172353554+0j) [Y5 X6 X10 Z11 Z12 Y13] +
(-0.011285190200840919+0j) [Y5 X6 X11 Y12] +
(-4.556569218075857e-06+0j) [Y5 X6 X12 Y13] +
(1.2283337824929525e-06+0j) [Y5 Y6 X7 Z8 Z9 X10] +
(-0.0002463643756957804+0j) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-4.838052750816613e-07+0j) [Y5 Y6 Y8 Y9] +
(5.974311713294073e-06+0j) [Y5 Y6 Y10 Y11] +
(0.02017592172353554+0j) [Y5 Y6 Y10 Z11 Z12 Y13] +
(0.011285190200840919+0j) [Y5 Y6 X11 X12] +
(-4.556569218075857e-06+0j) [Y5 Y6 Y12 Y13] +
(0.008890731522694623+0j) [Y5 Z6 X7 X11 Z12 Y13] +
(-1.2260484989582675e-05+0j) [Y5 Z6 Y7] +
(-1.8818501832808126e-06+0j) [Y5 Z6 Y7 Z8] +
(-1.3980449081991513e-06+0j) [Y5 Z6 Y7 Z9] +
(-0.012215040997613962+0j) [Y5 Z6 Y7 X10 Z11 X12] +
(-0.012215040997613962+0j) [Y5 Z6 Y7 Y10 Z11 Y12] +
(4.281913884693185e-06+0j) [Y5 Z6 Y7 Z10] +
(-0.000929850796773043+0j) [Y5 Z6 Y7 X11 Z12 X13] +
(0.00796088072592158+0j) [Y5 Z6 Y7 Y11 Z12 Y13] +
(-1.692397828600888e-06+0j) [Y5 Z6 Y7 Z11] +
(-4.5888551556732865e-06+0j) [Y5 Z6 Y7 Z12] +
(1.3304731886332367e-06+0j) [Y5 Z6 Z7 X8 X9 Y10] +
(0.005923798336561342+0j) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-1.3304731886332367e-06+0j) [Y5 Z6 Z7 Y8 X9 X10] +
(-0.005923798336561342+0j) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-0.016024603689179566+0j) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12] +
(-5.071480736405466e-06+0j) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(0.016024603689179566+0j) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12] +
(-5.071480736405466e-06+0j) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(-6.631277928278838e-05+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-9.806102775012614e-06+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(-4.734622038607147e-06+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(-0.3693708936615612+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.023145130929528964+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(-0.025637238296026838+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(3.3343312893710257e-06+0j) [Y5 Z6 Z7 Z8 Z9 Y11] +
(-0.009612634606847276+0j) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(-7.44434467574135e-06+0j) [Y5 Z6 Z7 Z8 Z10 Y11] +
(-0.04171881383982179+0j) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(-8.774817864374587e-06+0j) [Y5 Z6 Z7 Z9 Z10 Y11] +
(-0.04764261217638313+0j) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(-7.518362215534937e-06+0j) [Y5 Z6 Z8 Z9 Z10 Y11] +
(-0.03931805194719756+0j) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.85406085802419e-06+0j) [Y5 Y7] +
(-6.290028433041983e-06+0j) [Y5 Z7 Z8 Z9 Z10 Y11] +
(-0.03956441632289334+0j) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.2034402289145634+0j) [Z5] +
(0.010960074940542677+0j) [Z5 X6 Z7 Z8 Z9 X10] +
(-1.9429468365322604e-06+0j) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.010960074940542677+0j) [Z5 Y6 Z7 Z8 Z9 Y10] +
(-1.9429468365322604e-06+0j) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.14899430575065553+0j) [Z5 Z6] +
(0.018266834869375623+0j) [Z5 X7 Z8 Z9 Z10 X11] +
(-1.6541174769819948e-06+0j) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.018266834869375623+0j) [Z5 Y7 Z8 Z9 Z10 Y11] +
(-1.6541174769819948e-06+0j) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.13701191674040758+0j) [Z5 Z7] +
(0.15676396176430998+0j) [Z5 Z8] +
(0.149607026844453+0j) [Z5 Z9] +
(-1.8163031696390049e-06+0j) [Z5 X10 Z11 X12] +
(-1.8163031696390049e-06+0j) [Z5 Y10 Z11 Y12] +
(0.1425799771248575+0j) [Z5 Z10] +
(1.8782101247376814e-06+0j) [Z5 X11 Z12 X13] +
(1.8782101247376814e-06+0j) [Z5 Y11 Z12 Y13] +
(0.124899909172376+0j) [Z5 Z11] +
(0.15215040708869043+0j) [Z5 Z12] +
(0.11383573679388653+0j) [Z5 Z13] +
(-0.013873381748426079+0j) [X6 X7 Y8 Y9] +
(-0.017825140995786512+0j) [X6 X7 Y10 Y11] +
(-1.035847760137098e-06+0j) [X6 X7 Y10 Z11 Z12 Y13] +
(-1.035847760137098e-06+0j) [X6 X7 X11 X12] +
(-0.01736611899465138+0j) [X6 X7 Y12 Y13] +
(0.013873381748426079+0j) [X6 Y7 Y8 X9] +
(0.017825140995786512+0j) [X6 Y7 Y10 X11] +
(1.035847760137098e-06+0j) [X6 Y7 Y10 Z11 Z12 X13] +
(-1.035847760137098e-06+0j) [X6 Y7 Y11 X12] +
(0.01736611899465138+0j) [X6 Y7 Y12 X13] +
(0.0002921986261110342+0j) [X6 Z7 X8 X9 Z10 X11] +
(-3.328139350544411e-07+0j) [X6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.0002921986261110342+0j) [X6 Z7 X8 Y9 Z10 Y11] +
(-3.328139350544411e-07+0j) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(0.2284810656491885+0j) [X6 Z7 Z8 Z9 X10] +
(3.313145500074632e-06+0j) [X6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(3.313145500074632e-06+0j) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(0.011307274008848197+0j) [X6 Z7 Z8 Z9 X10 Z11] +
(0.025104957138844544+0j) [X6 Z7 Z8 Z9 X10 Z12] +
(0.010540425907671517+0j) [X6 Z7 Z8 Z9 X10 Z13] +
(-0.014564531231173029+0j) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(0.014564531231173029+0j) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-2.5950860068752258e-05+0j) [X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.183932559412548e-06+0j) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-6.524373848371744e-06+0j) [X6 Z7 Z8 Z9 Z10 X12] +
(-3.211228348297112e-06+0j) [X6 Z7 Z8 Z9 Z11 X12] +
(0.029812424517345837+0j) [X6 Z7 Z8 X10] +
(-3.277483195385345e-06+0j) [X6 Z7 Z8 Z10 Z11 X12] +
(0.03010462314345687+0j) [X6 Z7 Z9 X10] +
(-3.610297130439786e-06+0j) [X6 Z7 Z9 Z10 Z11 X12] +
(0.030787505389143988+0j) [X6 Z8 Z9 X10] +
(-3.7696594518391573e-06+0j) [X6 Z8 Z9 Z10 Z11 X12] +
(0.013873381748426079+0j) [Y6 X7 X8 Y9] +
(0.017825140995786512+0j) [Y6 X7 X10 Y11] +
(1.035847760137098e-06+0j) [Y6 X7 X10 Z11 Z12 Y13] +
(-1.035847760137098e-06+0j) [Y6 X7 X11 Y12] +
(0.01736611899465138+0j) [Y6 X7 X12 Y13] +
(-0.013873381748426079+0j) [Y6 Y7 X8 X9] +
(-0.017825140995786512+0j) [Y6 Y7 X10 X11] +
(-1.035847760137098e-06+0j) [Y6 Y7 X10 Z11 Z12 X13] +
(-1.035847760137098e-06+0j) [Y6 Y7 Y11 Y12] +
(-0.01736611899465138+0j) [Y6 Y7 X12 X13] +
(0.0002921986261110342+0j) [Y6 Z7 Y8 X9 Z10 X11] +
(-3.328139350544411e-07+0j) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.0002921986261110342+0j) [Y6 Z7 Y8 Y9 Z10 Y11] +
(-3.328139350544411e-07+0j) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(0.2284810656491885+0j) [Y6 Z7 Z8 Z9 Y10] +
(3.313145500074632e-06+0j) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(3.313145500074632e-06+0j) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(0.011307274008848197+0j) [Y6 Z7 Z8 Z9 Y10 Z11] +
(0.025104957138844544+0j) [Y6 Z7 Z8 Z9 Y10 Z12] +
(0.010540425907671517+0j) [Y6 Z7 Z8 Z9 Y10 Z13] +
(0.014564531231173029+0j) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-0.014564531231173029+0j) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-2.5950860068752258e-05+0j) [Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.183932559412548e-06+0j) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-6.524373848371744e-06+0j) [Y6 Z7 Z8 Z9 Z10 Y12] +
(-3.211228348297112e-06+0j) [Y6 Z7 Z8 Z9 Z11 Y12] +
(0.029812424517345837+0j) [Y6 Z7 Z8 Y10] +
(-3.277483195385345e-06+0j) [Y6 Z7 Z8 Z10 Z11 Y12] +
(0.03010462314345687+0j) [Y6 Z7 Z9 Y10] +
(-3.610297130439786e-06+0j) [Y6 Z7 Z9 Z10 Z11 Y12] +
(0.030787505389143988+0j) [Y6 Z8 Z9 Y10] +
(-3.7696594518391573e-06+0j) [Y6 Z8 Z9 Z10 Z11 Y12] +
(1.3096862988615416+0j) [Z6] +
(0.030787505389143988+0j) [Z6 X7 Z8 Z9 Z10 X11] +
(-3.7696594518391573e-06+0j) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.030787505389143988+0j) [Z6 Y7 Z8 Z9 Z10 Y11] +
(-3.7696594518391573e-06+0j) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.19392534613270196+0j) [Z6 Z7] +
(0.16756653265461272+0j) [Z6 Z8] +
(0.18143991440303878+0j) [Z6 Z9] +
(-1.8551201214934251e-06+0j) [Z6 X10 Z11 X12] +
(-1.8551201214934251e-06+0j) [Z6 Y10 Z11 Y12] +
(0.11952438964682671+0j) [Z6 Z10] +
(-2.8909678816305235e-06+0j) [Z6 X11 Z12 X13] +
(-2.8909678816305235e-06+0j) [Z6 Y11 Z12 Y13] +
(0.13734953064261324+0j) [Z6 Z11] +
(0.13401715261963695+0j) [Z6 Z12] +
(0.15138327161428833+0j) [Z6 Z13] +
(-0.0002921986261110342+0j) [X7 X8 Y9 Y10] +
(3.328139350544411e-07+0j) [X7 X8 Y9 Z10 Z11 Y12] +
(0.0002921986261110342+0j) [X7 Y8 Y9 X10] +
(-3.328139350544411e-07+0j) [X7 Y8 Y9 Z10 Z11 X12] +
(-3.313145500074632e-06+0j) [X7 Z8 Z9 X10 Y11 Y12] +
(-0.014564531231173029+0j) [X7 Z8 Z9 X10 X12 X13] +
(3.313145500074632e-06+0j) [X7 Z8 Z9 Y10 Y11 X12] +
(-0.014564531231173029+0j) [X7 Z8 Z9 Y10 Y12 X13] +
(0.22848106564918844+0j) [X7 Z8 Z9 Z10 X11] +
(0.010540425907671517+0j) [X7 Z8 Z9 Z10 X11 Z12] +
(0.025104957138844544+0j) [X7 Z8 Z9 Z10 X11 Z13] +
(-2.595086006875227e-05+0j) [X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.183932559412547e-06+0j) [X7 Z8 Z9 Z10 Z11 X13] +
(-3.211228348297112e-06+0j) [X7 Z8 Z9 Z10 Z12 X13] +
(0.011307274008848199+0j) [X7 Z8 Z9 X11] +
(-6.524373848371744e-06+0j) [X7 Z8 Z9 Z11 Z12 X13] +
(0.03010462314345687+0j) [X7 Z8 Z10 X11] +
(-3.610297130439786e-06+0j) [X7 Z8 Z10 Z11 Z12 X13] +
(0.029812424517345837+0j) [X7 Z9 Z10 X11] +
(-3.277483195385345e-06+0j) [X7 Z9 Z10 Z11 Z12 X13] +
(0.0002921986261110342+0j) [Y7 X8 X9 Y10] +
(-3.328139350544411e-07+0j) [Y7 X8 X9 Z10 Z11 Y12] +
(-0.0002921986261110342+0j) [Y7 Y8 X9 X10] +
(3.328139350544411e-07+0j) [Y7 Y8 X9 Z10 Z11 X12] +
(3.313145500074632e-06+0j) [Y7 Z8 Z9 X10 X11 Y12] +
(-0.014564531231173029+0j) [Y7 Z8 Z9 X10 X12 Y13] +
(-3.313145500074632e-06+0j) [Y7 Z8 Z9 Y10 X11 X12] +
(-0.014564531231173029+0j) [Y7 Z8 Z9 Y10 Y12 Y13] +
(0.22848106564918844+0j) [Y7 Z8 Z9 Z10 Y11] +
(0.010540425907671517+0j) [Y7 Z8 Z9 Z10 Y11 Z12] +
(0.025104957138844544+0j) [Y7 Z8 Z9 Z10 Y11 Z13] +
(-2.595086006875227e-05+0j) [Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(4.183932559412547e-06+0j) [Y7 Z8 Z9 Z10 Z11 Y13] +
(-3.211228348297112e-06+0j) [Y7 Z8 Z9 Z10 Z12 Y13] +
(0.011307274008848199+0j) [Y7 Z8 Z9 Y11] +
(-6.524373848371744e-06+0j) [Y7 Z8 Z9 Z11 Z12 Y13] +
(0.03010462314345687+0j) [Y7 Z8 Z10 Y11] +
(-3.610297130439786e-06+0j) [Y7 Z8 Z10 Z11 Z12 Y13] +
(0.029812424517345837+0j) [Y7 Z9 Z10 Y11] +
(-3.277483195385345e-06+0j) [Y7 Z9 Z10 Z11 Z12 Y13] +
(1.309686298861541+0j) [Z7] +
(0.18143991440303878+0j) [Z7 Z8] +
(0.16756653265461272+0j) [Z7 Z9] +
(-2.8909678816305235e-06+0j) [Z7 X10 Z11 X12] +
(-2.8909678816305235e-06+0j) [Z7 Y10 Z11 Y12] +
(0.13734953064261324+0j) [Z7 Z10] +
(-1.8551201214934251e-06+0j) [Z7 X11 Z12 X13] +
(-1.8551201214934251e-06+0j) [Z7 Y11 Z12 Y13] +
(0.11952438964682671+0j) [Z7 Z11] +
(0.15138327161428833+0j) [Z7 Z12] +
(0.13401715261963695+0j) [Z7 Z13] +
(-0.009560705729135923+0j) [X8 X9 Y10 Y11] +
(6.62861420145345e-07+0j) [X8 X9 Y10 Z11 Z12 Y13] +
(6.628614201453451e-07+0j) [X8 X9 X11 X12] +
(-0.006087822480561846+0j) [X8 X9 Y12 Y13] +
(0.009560705729135923+0j) [X8 Y9 Y10 X11] +
(-6.62861420145345e-07+0j) [X8 Y9 Y10 Z11 Z12 X13] +
(6.628614201453451e-07+0j) [X8 Y9 Y11 X12] +
(0.006087822480561846+0j) [X8 Y9 Y12 X13] +
(0.009560705729135923+0j) [Y8 X9 X10 Y11] +
(-6.62861420145345e-07+0j) [Y8 X9 X10 Z11 Z12 Y13] +
(6.628614201453451e-07+0j) [Y8 X9 X11 Y12] +
(0.006087822480561846+0j) [Y8 X9 X12 Y13] +
(-0.009560705729135923+0j) [Y8 Y9 X10 X11] +
(6.62861420145345e-07+0j) [Y8 Y9 X10 Z11 Z12 X13] +
(6.628614201453451e-07+0j) [Y8 Y9 Y11 Y12] +
(-0.006087822480561846+0j) [Y8 Y9 X12 X13] +
(1.3693525634718189+0j) [Z8] +
(-1.5973171977755543e-06+0j) [Z8 X10 Z11 X12] +
(-1.5973171977755543e-06+0j) [Z8 Y10 Z11 Y12] +
(0.13766872645852574+0j) [Z8 Z10] +
(-9.344557776302092e-07+0j) [Z8 X11 Z12 X13] +
(-9.344557776302092e-07+0j) [Z8 Y11 Z12 Y13] +
(0.14722943218766166+0j) [Z8 Z11] +
(0.14973486803496922+0j) [Z8 Z12] +
(0.15582269051553105+0j) [Z8 Z13] +
(1.369352563471819+0j) [Z9] +
(-9.344557776302092e-07+0j) [Z9 X10 Z11 X12] +
(-9.344557776302092e-07+0j) [Z9 Y10 Z11 Y12] +
(0.14722943218766166+0j) [Z9 Z10] +
(-1.5973171977755543e-06+0j) [Z9 X11 Z12 X13] +
(-1.5973171977755543e-06+0j) [Z9 Y11 Z12 Y13] +
(0.13766872645852574+0j) [Z9 Z11] +
(0.15582269051553105+0j) [Z9 Z12] +
(0.14973486803496922+0j) [Z9 Z13] +
(-0.028685183716105983+0j) [X10 X11 Y12 Y13] +
(0.028685183716105983+0j) [X10 Y11 Y12 X13] +
(-1.0722312157807544e-05+0j) [X10 Z11 X12] +
(7.954413176205972e-06+0j) [X10 Z11 X12 Z13] +
(-8.194261372007567e-06+0j) [X10 X12] +
(0.028685183716105983+0j) [Y10 X11 X12 Y13] +
(-0.028685183716105983+0j) [Y10 Y11 X12 X13] +
(-1.0722312157807544e-05+0j) [Y10 Z11 Y12] +
(7.954413176205972e-06+0j) [Y10 Z11 Y12 Z13] +
(-8.194261372007567e-06+0j) [Y10 Y12] +
(0.7829661725950182+0j) [Z10] +
(-8.194261372007567e-06+0j) [Z10 X11 Z12 X13] +
(-8.194261372007567e-06+0j) [Z10 Y11 Z12 Y13] +
(0.14926355147388895+0j) [Z10 Z11] +
(0.11270386920332208+0j) [Z10 Z12] +
(0.14138905291942805+0j) [Z10 Z13] +
(-1.0722312157807542e-05+0j) [X11 Z12 X13] +
(7.954413176205972e-06+0j) [X11 X13] +
(-1.0722312157807542e-05+0j) [Y11 Z12 Y13] +
(7.954413176205972e-06+0j) [Y11 Y13] +
(0.7829661725950181+0j) [Z11] +
(0.14138905291942805+0j) [Z11 Z12] +
(0.11270386920332208+0j) [Z11 Z13] +
(0.8084581961720478+0j) [Z12] +
(0.15435748657223625+0j) [Z12 Z13] +
(0.8084581961720481+0j) [Z13]
  (-46.463906788688966) [I0]
+ (0.7829661725950186) [Z11]
+ (0.7829661725950187) [Z10]
+ (0.8084581961720491) [Z13]
+ (1.203440228914563) [Z4]
+ (1.2034402289145631) [Z5]
+ (1.309686298861544) [Z6]
+ (1.3096862988615445) [Z7]
+ (1.3693525634718187) [Z8]
+ (1.369352563471819) [Z9]
+ (1.6538942226831703) [Z2]
+ (1.6538942226831705) [Z3]
+ (12.412630742111755) [Z0]
+ (12.412630742111755) [Z1]
+ (-8.194261371899564e-06) [Y10 Y12]
+ (-8.194261371899564e-06) [X10 X12]
+ (-1.8540608579281016e-06) [Y5 Y7]
+ (-1.8540608579281016e-06) [X5 X7]
+ (-7.76499411826543e-07) [Y3 Y5]
+ (-7.76499411826543e-07) [X3 X5]
+ (-5.929765815515347e-07) [Y4 Y6]
+ (-5.929765815515347e-07) [X4 X6]
+ (1.6021167407267983e-06) [Y2 Y4]
+ (1.6021167407267983e-06) [X2 X4]
+ (7.954413175882256e-06) [Y11 Y13]
+ (7.954413175882256e-06) [X11 X13]
+ (0.0032769719312316058) [Y1 Y3]
+ (0.0032769719312316058) [X1 X3]
+ (0.10433064780651388) [Y0 Y2]
+ (0.10433064780651388) [X0 X2]
+ (0.1127038692033222) [Z10 Z12]
+ (0.1127038692033222) [Z11 Z13]
+ (0.11383573679388663) [Z4 Z12]
+ (0.11383573679388663) [Z5 Z13]
+ (0.11952438964682693) [Z6 Z10]
+ (0.11952438964682693) [Z7 Z11]
+ (0.12495807739503216) [Z2 Z4]
+ (0.12495807739503216) [Z3 Z5]
+ (0.12799502492468415) [Z2 Z10]
+ (0.12799502492468415) [Z3 Z11]
+ (0.13401715261963715) [Z6 Z12]
+ (0.13401715261963715) [Z7 Z13]
+ (0.13701191674040764) [Z4 Z6]
+ (0.13701191674040764) [Z5 Z7]
+ (0.13739104762683238) [Z2 Z6]
+ (0.13739104762683238) [Z3 Z7]
+ (0.13766872645852576) [Z8 Z10]
+ (0.13766872645852576) [Z9 Z11]
+ (0.14011289865354812) [Z2 Z12]
+ (0.14011289865354812) [Z3 Z13]
+ (0.14138905291942816) [Z10 Z13]
+ (0.14138905291942816) [Z11 Z12]
+ (0.14257997712485765) [Z4 Z11]
+ (0.14257997712485765) [Z5 Z10]
+ (0.14722943218766174) [Z8 Z11]
+ (0.14722943218766174) [Z9 Z10]
+ (0.1489943057506556) [Z4 Z7]
+ (0.1489943057506556) [Z5 Z6]
+ (0.14960702684445298) [Z4 Z8]
+ (0.14960702684445298) [Z5 Z9]
+ (0.1497348680349693) [Z8 Z12]
+ (0.1497348680349693) [Z9 Z13]
+ (0.15071408121008284) [Z2 Z8]
+ (0.15071408121008284) [Z3 Z9]
+ (0.15138327161428855) [Z6 Z13]
+ (0.15138327161428855) [Z7 Z12]
+ (0.1521504070886905) [Z4 Z13]
+ (0.1521504070886905) [Z5 Z12]
+ (0.15337968243314162) [Z2 Z11]
+ (0.15337968243314162) [Z3 Z10]
+ (0.15435748657223633) [Z12 Z13]
+ (0.15569010671752456) [Z2 Z13]
+ (0.15569010671752456) [Z3 Z12]
+ (0.15582269051553116) [Z8 Z13]
+ (0.15582269051553116) [Z9 Z12]
+ (0.15676396176430993) [Z4 Z9]
+ (0.15676396176430993) [Z5 Z8]
+ (0.15755314797985673) [Z4 Z5]
+ (0.16079764534838564) [Z2 Z5]
+ (0.16079764534838564) [Z3 Z4]
+ (0.16756653265461285) [Z6 Z8]
+ (0.16756653265461285) [Z7 Z9]
+ (0.16853486561579942) [Z2 Z7]
+ (0.16853486561579942) [Z3 Z6]
+ (0.18143991440303903) [Z6 Z9]
+ (0.18143991440303903) [Z7 Z8]
+ (0.18189085790751358) [Z2 Z3]
+ (0.1869082047691254) [Z2 Z9]
+ (0.1869082047691254) [Z3 Z8]
+ (0.1929972393536422) [Z0 Z10]
+ (0.1929972393536422) [Z1 Z11]
+ (0.1939253461327026) [Z6 Z7]
+ (0.1966177089034212) [Z0 Z4]
+ (0.1966177089034212) [Z1 Z5]
+ (0.199363545373608) [Z0 Z5]
+ (0.199363545373608) [Z1 Z4]
+ (0.20072866460441752) [Z0 Z11]
+ (0.20072866460441752) [Z1 Z10]
+ (0.21102659849791494) [Z0 Z12]
+ (0.21102659849791494) [Z1 Z13]
+ (0.2163103749863179) [Z0 Z13]
+ (0.2163103749863179) [Z1 Z12]
+ (0.23671080783830384) [Z0 Z2]
+ (0.23671080783830384) [Z1 Z3]
+ (0.2416466393601721) [Z0 Z6]
+ (0.2416466393601721) [Z1 Z7]
+ (0.24853483371314272) [Z0 Z7]
+ (0.24853483371314272) [Z1 Z6]
+ (0.2512944567459165) [Z0 Z3]
+ (0.2512944567459165) [Z1 Z2]
+ (0.2723251830660565) [Z0 Z8]
+ (0.2723251830660565) [Z1 Z9]
+ (0.27883454426723375) [Z0 Z9]
+ (0.27883454426723375) [Z1 Z8]
+ (1.1861763734860475) [Z0 Z1]
+ (-1.2260484988558262e-05) [Y4 Z5 Y6]
+ (-1.2260484988558262e-05) [X4 Z5 X6]
+ (-1.226048498855826e-05) [Y5 Z6 Y7]
+ (-1.226048498855826e-05) [X5 Z6 X7]
+ (-1.072231215731677e-05) [Y11 Z12 Y13]
+ (-1.072231215731677e-05) [X11 Z12 X13]
+ (-1.0722312157316766e-05) [Y10 Z11 Y12]
+ (-1.0722312157316766e-05) [X10 Z11 X12]
+ (-3.887051673164716e-06) [Y3 Z4 Y5]
+ (-3.887051673164716e-06) [X3 Z4 X5]
+ (-3.887051673164714e-06) [Y2 Z3 Y4]
+ (-3.887051673164714e-06) [X2 Z3 X4]
+ (0.12507032579772165) [Y0 Z1 Y2]
+ (0.12507032579772165) [X0 Z1 X2]
+ (0.12507032579772168) [Y1 Z2 Y3]
+ (0.12507032579772168) [X1 Z2 X3]
+ (-0.03831467029480388) [Y4 Y5 X12 X13]
+ (-0.03831467029480388) [X4 X5 Y12 Y13]
+ (-0.0358395679533535) [Y2 Y3 X4 X5]
+ (-0.0358395679533535) [X2 X3 Y4 Y5]
+ (-0.031143817988967044) [Y2 Y3 X6 X7]
+ (-0.031143817988967044) [X2 X3 Y6 Y7]
+ (-0.02868518371610596) [Y10 Y11 X12 X13]
+ (-0.02868518371610596) [X10 X11 Y12 Y13]
+ (-0.02599617759802126) [Y3 Z4 Z5 Y7]
+ (-0.02599617759802126) [X3 Z4 Z5 X7]
+ (-0.02538465750845747) [Y2 Y3 X10 X11]
+ (-0.02538465750845747) [X2 X3 Y10 Y11]
+ (-0.019028242443847366) [Y3 Y4 X11 X12]
+ (-0.019028242443847366) [X3 X4 Y11 Y12]
+ (-0.017825140995786356) [Y6 Y7 X10 X11]
+ (-0.017825140995786356) [X6 X7 Y10 Y11]
+ (-0.017680067952481556) [Y4 Y5 X10 X11]
+ (-0.017680067952481556) [X4 X5 Y10 Y11]
+ (-0.01736611899465139) [Y6 Y7 X12 X13]
+ (-0.01736611899465139) [X6 X7 Y12 Y13]
+ (-0.015577208063976456) [Y2 Y3 X12 X13]
+ (-0.015577208063976456) [X2 X3 Y12 Y13]
+ (-0.014583648907612655) [Y0 Y1 X2 X3]
+ (-0.014583648907612655) [X0 X1 Y2 Y3]
+ (-0.013873381748426157) [Y6 Y7 X8 X9]
+ (-0.013873381748426157) [X6 X7 Y8 Y9]
+ (-0.011982389010247934) [Y4 Y5 X6 X7]
+ (-0.011982389010247934) [X4 X5 Y6 Y7]
+ (-0.009560705729135956) [Y8 Y9 X10 X11]
+ (-0.009560705729135956) [X8 X9 Y10 Y11]
+ (-0.008125251921381034) [Y1 X2 X8 Y9]
+ (-0.008125251921381034) [Y1 Y2 Y8 Y9]
+ (-0.008125251921381034) [X1 X2 X8 X9]
+ (-0.008125251921381034) [X1 Y2 Y8 X9]
+ (-0.007731425250775303) [Y0 Y1 X10 X11]
+ (-0.007731425250775303) [X0 X1 Y10 Y11]
+ (-0.0068881943529705975) [Y0 Y1 X6 X7]
+ (-0.0068881943529705975) [X0 X1 Y6 Y7]
+ (-0.006509361201177239) [Y0 Y1 X8 X9]
+ (-0.006509361201177239) [X0 X1 Y8 Y9]
+ (-0.00608782248056186) [Y8 Y9 X12 X13]
+ (-0.00608782248056186) [X8 X9 Y12 Y13]
+ (-0.005143391768825088) [Y3 X4 X5 Y6]
+ (-0.005143391768825088) [X3 Y4 Y5 X6]
+ (-0.004684903388155205) [Y1 X2 X6 Y7]
+ (-0.004684903388155205) [Y1 Y2 Y6 Y7]
+ (-0.004684903388155205) [X1 X2 X6 X7]
+ (-0.004684903388155205) [X1 Y2 Y6 X7]
+ (-0.004575007626639201) [Y1 X2 X12 Y13]
+ (-0.004575007626639201) [Y1 Y2 Y12 Y13]
+ (-0.004575007626639201) [X1 X2 X12 X13]
+ (-0.004575007626639201) [X1 Y2 Y12 X13]
+ (-0.004424855449441857) [Y1 X2 X4 Y5]
+ (-0.004424855449441857) [Y1 Y2 Y4 Y5]
+ (-0.004424855449441857) [X1 X2 X4 X5]
+ (-0.004424855449441857) [X1 Y2 Y4 X5]
+ (-0.0034795118903342558) [Y2 Z3 Z5 Y6]
+ (-0.0034795118903342558) [X2 Z3 Z5 X6]
+ (-0.0034795118903342558) [Y3 Z4 Z6 Y7]
+ (-0.0034795118903342558) [X3 Z4 Z6 X7]
+ (-0.002745836470186808) [Y0 Y1 X4 X5]
+ (-0.002745836470186808) [X0 X1 Y4 Y5]
+ (-0.0017992194936630086) [Y1 X2 X10 Y11]
+ (-0.0017992194936630086) [Y1 Y2 Y10 Y11]
+ (-0.0017992194936630086) [X1 X2 X10 X11]
+ (-0.0017992194936630086) [X1 Y2 Y10 X11]
+ (-0.00029219862611109583) [Y7 Y8 X9 X10]
+ (-0.00029219862611109583) [X7 X8 Y9 Y10]
+ (-8.194261371899562e-06) [Z10 Y11 Z12 Y13]
+ (-8.194261371899562e-06) [Z10 X11 Z12 X13]
+ (-7.80170750009338e-06) [Y2 Z3 Y4 Z11]
+ (-7.80170750009338e-06) [X2 Z3 X4 Z11]
+ (-7.80170750009338e-06) [Y3 Z4 Y5 Z10]
+ (-7.80170750009338e-06) [X3 Z4 X5 Z10]
+ (-4.6430510682572266e-06) [Y3 X4 X10 Y11]
+ (-4.6430510682572266e-06) [Y3 Y4 Y10 Y11]
+ (-4.6430510682572266e-06) [X3 X4 X10 X11]
+ (-4.6430510682572266e-06) [X3 Y4 Y10 X11]
+ (-4.5888551554313205e-06) [Y4 Z5 Y6 Z13]
+ (-4.5888551554313205e-06) [X4 Z5 X6 Z13]
+ (-4.5888551554313205e-06) [Y5 Z6 Y7 Z12]
+ (-4.5888551554313205e-06) [X5 Z6 X7 Z12]
+ (-4.556569217885115e-06) [Y5 X6 X12 Y13]
+ (-4.556569217885115e-06) [Y5 Y6 Y12 Y13]
+ (-4.556569217885115e-06) [X5 X6 X12 X13]
+ (-4.556569217885115e-06) [X5 Y6 Y12 X13]
+ (-3.6945132942860533e-06) [Y4 X5 X11 Y12]
+ (-3.6945132942860533e-06) [Y4 Y5 Y11 Y12]
+ (-3.6945132942860533e-06) [X4 X5 X11 X12]
+ (-3.6945132942860533e-06) [X4 Y5 Y11 X12]
+ (-3.344081556455477e-06) [Z0 Y5 Z6 Y7]
+ (-3.344081556455477e-06) [Z0 X5 Z6 X7]
+ (-3.344081556455477e-06) [Z1 Y4 Z5 Y6]
+ (-3.344081556455477e-06) [Z1 X4 Z5 X6]
+ (-3.158656431836152e-06) [Y2 Z3 Y4 Z10]
+ (-3.158656431836152e-06) [X2 Z3 X4 Z10]
+ (-3.158656431836152e-06) [Y3 Z4 Y5 Z11]
+ (-3.158656431836152e-06) [X3 Z4 X5 Z11]
+ (-3.099349243570314e-06) [Z0 Y4 Z5 Y6]
+ (-3.099349243570314e-06) [Z0 X4 Z5 X6]
+ (-3.099349243570314e-06) [Z1 Y5 Z6 Y7]
+ (-3.099349243570314e-06) [Z1 X5 Z6 X7]
+ (-2.890967881599107e-06) [Z6 Y11 Z12 Y13]
+ (-2.890967881599107e-06) [Z6 X11 Z12 X13]
+ (-2.890967881599107e-06) [Z7 Y10 Z11 Y12]
+ (-2.890967881599107e-06) [Z7 X10 Z11 X12]
+ (-2.1776646048815418e-06) [Z0 Y10 Z11 Y12]
+ (-2.1776646048815418e-06) [Z0 X10 Z11 X12]
+ (-2.1776646048815418e-06) [Z1 Y11 Z12 Y13]
+ (-2.1776646048815418e-06) [Z1 X11 Z12 X13]
+ (-1.8818501831836816e-06) [Y4 Z5 Y6 Z9]
+ (-1.8818501831836816e-06) [X4 Z5 X6 Z9]
+ (-1.8818501831836816e-06) [Y5 Z6 Y7 Z8]
+ (-1.8818501831836816e-06) [X5 Z6 X7 Z8]
+ (-1.8551201214055254e-06) [Z6 Y10 Z11 Y12]
+ (-1.8551201214055254e-06) [Z6 X10 Z11 X12]
+ (-1.8551201214055254e-06) [Z7 Y11 Z12 Y13]
+ (-1.8551201214055254e-06) [Z7 X11 Z12 X13]
+ (-1.8540608579281016e-06) [Y4 Z5 Y6 Z7]
+ (-1.8540608579281016e-06) [X4 Z5 X6 Z7]
+ (-1.8163031696735858e-06) [Z4 Y11 Z12 Y13]
+ (-1.8163031696735858e-06) [Z4 X11 Z12 X13]
+ (-1.8163031696735858e-06) [Z5 Y10 Z11 Y12]
+ (-1.8163031696735858e-06) [Z5 X10 Z11 X12]
+ (-1.6923978284691957e-06) [Y4 Z5 Y6 Z10]
+ (-1.6923978284691957e-06) [X4 Z5 X6 Z10]
+ (-1.6923978284691957e-06) [Y5 Z6 Y7 Z11]
+ (-1.6923978284691957e-06) [X5 Z6 X7 Z11]
+ (-1.6148794137492804e-06) [Z0 Y11 Z12 Y13]
+ (-1.6148794137492804e-06) [Z0 X11 Z12 X13]
+ (-1.6148794137492804e-06) [Z1 Y10 Z11 Y12]
+ (-1.6148794137492804e-06) [Z1 X10 Z11 X12]
+ (-1.5973171977228496e-06) [Z8 Y10 Z11 Y12]
+ (-1.5973171977228496e-06) [Z8 X10 Z11 X12]
+ (-1.5973171977228496e-06) [Z9 Y11 Z12 Y13]
+ (-1.5973171977228496e-06) [Z9 X11 Z12 X13]
+ (-1.4548424490936176e-06) [Y3 X4 X6 Y7]
+ (-1.4548424490936176e-06) [Y3 Y4 Y6 Y7]
+ (-1.4548424490936176e-06) [X3 X4 X6 X7]
+ (-1.4548424490936176e-06) [X3 Y4 Y6 X7]
+ (-1.3980449080825064e-06) [Y4 Z5 Y6 Z8]
+ (-1.3980449080825064e-06) [X4 Z5 X6 Z8]
+ (-1.3980449080825064e-06) [Y5 Z6 Y7 Z9]
+ (-1.3980449080825064e-06) [X5 Z6 X7 Z9]
+ (-1.1954890099605172e-06) [Y2 Z3 Y4 Z7]
+ (-1.1954890099605172e-06) [X2 Z3 X4 Z7]
+ (-1.1954890099605172e-06) [Y3 Z4 Y5 Z6]
+ (-1.1954890099605172e-06) [X3 Z4 X5 Z6]
+ (-1.1908508084049275e-06) [Z0 Y3 Z4 Y5]
+ (-1.1908508084049275e-06) [Z0 X3 Z4 X5]
+ (-1.1908508084049275e-06) [Z1 Y2 Z3 Y4]
+ (-1.1908508084049275e-06) [Z1 X2 Z3 X4]
+ (-1.1708301369823714e-06) [Z2 Y5 Z6 Y7]
+ (-1.1708301369823714e-06) [Z2 X5 Z6 X7]
+ (-1.1708301369823714e-06) [Z3 Y4 Z5 Y6]
+ (-1.1708301369823714e-06) [Z3 X4 Z5 X6]
+ (-1.063228342271186e-06) [Z2 Y10 Z11 Y12]
+ (-1.063228342271186e-06) [Z2 X10 Z11 X12]
+ (-1.063228342271186e-06) [Z3 Y11 Z12 Y13]
+ (-1.063228342271186e-06) [Z3 X11 Z12 X13]
+ (-1.0358477601935817e-06) [Y6 X7 X11 Y12]
+ (-1.0358477601935817e-06) [Y6 Y7 Y11 Y12]
+ (-1.0358477601935817e-06) [X6 X7 X11 X12]
+ (-1.0358477601935817e-06) [X6 Y7 Y11 X12]
+ (-9.509249751411794e-07) [Z2 Y4 Z5 Y6]
+ (-9.509249751411794e-07) [Z2 X4 Z5 X6]
+ (-9.509249751411794e-07) [Z3 Y5 Z6 Y7]
+ (-9.509249751411794e-07) [Z3 X5 Z6 X7]
+ (-9.344557775824096e-07) [Z8 Y11 Z12 Y13]
+ (-9.344557775824096e-07) [Z8 X11 Z12 X13]
+ (-9.344557775824096e-07) [Z9 Y10 Z11 Y12]
+ (-9.344557775824096e-07) [Z9 X10 Z11 X12]
+ (-8.337746754796174e-07) [Z0 Y2 Z3 Y4]
+ (-8.337746754796174e-07) [Z0 X2 Z3 X4]
+ (-8.337746754796174e-07) [Z1 Y3 Z4 Y5]
+ (-8.337746754796174e-07) [Z1 X3 Z4 X5]
+ (-7.956895372909972e-07) [Y3 X4 X8 Y9]
+ (-7.956895372909972e-07) [Y3 Y4 Y8 Y9]
+ (-7.956895372909972e-07) [X3 X4 X8 X9]
+ (-7.956895372909972e-07) [X3 Y4 Y8 X9]
+ (-7.76499411826543e-07) [Y2 Z3 Y4 Z5]
+ (-7.76499411826543e-07) [X2 Z3 X4 Z5]
+ (-5.929765815515347e-07) [Z4 Y5 Z6 Y7]
+ (-5.929765815515347e-07) [Z4 X5 Z6 X7]
+ (-5.77005299510079e-07) [Y2 Z3 Y4 Z9]
+ (-5.77005299510079e-07) [X2 Z3 X4 Z9]
+ (-5.77005299510079e-07) [Y3 Z4 Y5 Z8]
+ (-5.77005299510079e-07) [X3 Z4 X5 Z8]
+ (-5.471647744418357e-07) [Y1 Y2 X11 X12]
+ (-5.471647744418357e-07) [X1 X2 Y11 Y12]
+ (-4.838052751011752e-07) [Y5 X6 X8 Y9]
+ (-4.838052751011752e-07) [Y5 Y6 Y8 Y9]
+ (-4.838052751011752e-07) [X5 X6 X8 X9]
+ (-4.838052751011752e-07) [X5 Y6 Y8 X9]
+ (-3.5707613292531e-07) [Y0 X1 X3 Y4]
+ (-3.5707613292531e-07) [Y0 Y1 Y3 Y4]
+ (-3.5707613292531e-07) [X0 X1 X3 X4]
+ (-3.5707613292531e-07) [X0 Y1 Y3 X4]
+ (-2.44732312885163e-07) [Y0 X1 X5 Y6]
+ (-2.44732312885163e-07) [Y0 Y1 Y5 Y6]
+ (-2.44732312885163e-07) [X0 X1 X5 X6]
+ (-2.44732312885163e-07) [X0 Y1 Y5 X6]
+ (-2.1990516184119171e-07) [Y2 X3 X5 Y6]
+ (-2.1990516184119171e-07) [Y2 Y3 Y5 Y6]
+ (-2.1990516184119171e-07) [X2 X3 X5 X6]
+ (-2.1990516184119171e-07) [X2 Y3 Y5 X6]
+ (-1.933241277199603e-07) [Y1 X2 X3 Y4]
+ (-1.933241277199603e-07) [X1 Y2 Y3 X4]
+ (-1.291969486276644e-07) [Y1 Z2 Z3 Y5]
+ (-1.291969486276644e-07) [X1 Z2 Z3 X5]
+ (1.7379332624652027e-07) [Y0 Z1 Z3 Y4]
+ (1.7379332624652027e-07) [X0 Z1 Z3 X4]
+ (1.7379332624652027e-07) [Y1 Z2 Z4 Y5]
+ (1.7379332624652027e-07) [X1 Z2 Z4 X5]
+ (1.933241277199603e-07) [Y1 Y2 X3 X4]
+ (1.933241277199603e-07) [X1 X2 Y3 Y4]
+ (2.1868423778091815e-07) [Y2 Z3 Y4 Z8]
+ (2.1868423778091815e-07) [X2 Z3 X4 Z8]
+ (2.1868423778091815e-07) [Y3 Z4 Y5 Z9]
+ (2.1868423778091815e-07) [X3 Z4 X5 Z9]
+ (2.593534391331005e-07) [Y2 Z3 Y4 Z6]
+ (2.593534391331005e-07) [X2 Z3 X4 Z6]
+ (2.593534391331005e-07) [Y3 Z4 Y5 Z7]
+ (2.593534391331005e-07) [X3 Z4 X5 Z7]
+ (3.606071868086414e-07) [Y0 Z1 Z2 Y4]
+ (3.606071868086414e-07) [X0 Z1 Z2 X4]
+ (3.606071868086414e-07) [Y1 Z3 Z4 Y5]
+ (3.606071868086414e-07) [X1 Z3 Z4 X5]
+ (5.471647744418357e-07) [Y1 X2 X11 Y12]
+ (5.471647744418357e-07) [X1 Y2 Y11 X12]
+ (5.627851911322608e-07) [Y0 X1 X11 Y12]
+ (5.627851911322608e-07) [Y0 Y1 Y11 Y12]
+ (5.627851911322608e-07) [X0 X1 X11 X12]
+ (5.627851911322608e-07) [X0 Y1 Y11 X12]
+ (6.628614201404398e-07) [Y8 X9 X11 Y12]
+ (6.628614201404398e-07) [Y8 Y9 Y11 Y12]
+ (6.628614201404398e-07) [X8 X9 X11 X12]
+ (6.628614201404398e-07) [X8 Y9 Y11 X12]
+ (1.1094407591813826e-06) [Z2 Y11 Z12 Y13]
+ (1.1094407591813826e-06) [Z2 X11 Z12 X13]
+ (1.1094407591813826e-06) [Z3 Y10 Z11 Y12]
+ (1.1094407591813826e-06) [Z3 X10 Z11 X12]
+ (1.6021167407267986e-06) [Z2 Y3 Z4 Y5]
+ (1.6021167407267986e-06) [Z2 X3 Z4 X5]
+ (1.878210124612467e-06) [Z4 Y10 Z11 Y12]
+ (1.878210124612467e-06) [Z4 X10 Z11 X12]
+ (1.878210124612467e-06) [Z5 Y11 Z12 Y13]
+ (1.878210124612467e-06) [Z5 X11 Z12 X13]
+ (2.1726691014525684e-06) [Y2 X3 X11 Y12]
+ (2.1726691014525684e-06) [Y2 Y3 Y11 Y12]
+ (2.1726691014525684e-06) [X2 X3 X11 X12]
+ (2.1726691014525684e-06) [X2 Y3 Y11 X12]
+ (3.1174479462121217e-06) [Y0 Z2 Z3 Y4]
+ (3.1174479462121217e-06) [X0 Z2 Z3 X4]
+ (3.5390541843636045e-06) [Y2 Z3 Y4 Z12]
+ (3.5390541843636045e-06) [X2 Z3 X4 Z12]
+ (3.5390541843636045e-06) [Y3 Z4 Y5 Z13]
+ (3.5390541843636045e-06) [X3 Z4 X5 Z13]
+ (4.281913884731525e-06) [Y4 Z5 Y6 Z11]
+ (4.281913884731525e-06) [X4 Z5 X6 Z11]
+ (4.281913884731525e-06) [Y5 Z6 Y7 Z10]
+ (4.281913884731525e-06) [X5 Z6 X7 Z10]
+ (5.2758831219240204e-06) [Y3 X4 X12 Y13]
+ (5.2758831219240204e-06) [Y3 Y4 Y12 Y13]
+ (5.2758831219240204e-06) [X3 X4 X12 X13]
+ (5.2758831219240204e-06) [X3 Y4 Y12 X13]
+ (5.9743117132007216e-06) [Y5 X6 X10 Y11]
+ (5.9743117132007216e-06) [Y5 Y6 Y10 Y11]
+ (5.9743117132007216e-06) [X5 X6 X10 X11]
+ (5.9743117132007216e-06) [X5 Y6 Y10 X11]
+ (7.954413175882256e-06) [Y10 Z11 Y12 Z13]
+ (7.954413175882256e-06) [X10 Z11 X12 Z13]
+ (8.814937306287622e-06) [Y2 Z3 Y4 Z13]
+ (8.814937306287622e-06) [X2 Z3 X4 Z13]
+ (8.814937306287622e-06) [Y3 Z4 Y5 Z12]
+ (8.814937306287622e-06) [X3 Z4 X5 Z12]
+ (0.00029219862611109583) [Y7 X8 X9 Y10]
+ (0.00029219862611109583) [X7 Y8 Y9 X10]
+ (0.0004956762314917958) [Y2 Z4 Z5 Y6]
+ (0.0004956762314917958) [X2 Z4 Z5 X6]
+ (0.0011059037691896441) [Y0 Z1 Y2 Z5]
+ (0.0011059037691896441) [X0 Z1 X2 Z5]
+ (0.0011059037691896441) [Y1 Z2 Y3 Z4]
+ (0.0011059037691896441) [X1 Z2 X3 Z4]
+ (0.0016638798784908318) [Y2 Z3 Z4 Y6]
+ (0.0016638798784908318) [X2 Z3 Z4 X6]
+ (0.0016638798784908318) [Y3 Z5 Z6 Y7]
+ (0.0016638798784908318) [X3 Z5 Z6 X7]
+ (0.0017560707018412008) [Y0 Z1 Y2 Z11]
+ (0.0017560707018412008) [X0 Z1 X2 Z11]
+ (0.0017560707018412008) [Y1 Z2 Y3 Z10]
+ (0.0017560707018412008) [X1 Z2 X3 Z10]
+ (0.0023262306231580515) [Y0 Z1 Y2 Z13]
+ (0.0023262306231580515) [X0 Z1 X2 Z13]
+ (0.0023262306231580515) [Y1 Z2 Y3 Z12]
+ (0.0023262306231580515) [X1 Z2 X3 Z12]
+ (0.002745836470186808) [Y0 X1 X4 Y5]
+ (0.002745836470186808) [X0 Y1 Y4 X5]
+ (0.0029297686747510165) [Y0 Z1 Y2 Z9]
+ (0.0029297686747510165) [X0 Z1 X2 Z9]
+ (0.0029297686747510165) [Y1 Z2 Y3 Z8]
+ (0.0029297686747510165) [X1 Z2 X3 Z8]
+ (0.003276971931231606) [Y0 Z1 Y2 Z3]
+ (0.003276971931231606) [X0 Z1 X2 Z3]
+ (0.0033476175306661714) [Y0 Z1 Y2 Z7]
+ (0.0033476175306661714) [X0 Z1 X2 Z7]
+ (0.0033476175306661714) [Y1 Z2 Y3 Z6]
+ (0.0033476175306661714) [X1 Z2 X3 Z6]
+ (0.0035552901955042096) [Y0 Z1 Y2 Z10]
+ (0.0035552901955042096) [X0 Z1 X2 Z10]
+ (0.0035552901955042096) [Y1 Z2 Y3 Z11]
+ (0.0035552901955042096) [X1 Z2 X3 Z11]
+ (0.005143391768825088) [Y3 Y4 X5 X6]
+ (0.005143391768825088) [X3 X4 Y5 Y6]
+ (0.0055307592186315015) [Y0 Z1 Y2 Z4]
+ (0.0055307592186315015) [X0 Z1 X2 Z4]
+ (0.0055307592186315015) [Y1 Z2 Y3 Z5]
+ (0.0055307592186315015) [X1 Z2 X3 Z5]
+ (0.00608782248056186) [Y8 X9 X12 Y13]
+ (0.00608782248056186) [X8 Y9 Y12 X13]
+ (0.006509361201177239) [Y0 X1 X8 Y9]
+ (0.006509361201177239) [X0 Y1 Y8 X9]
+ (0.0068881943529705975) [Y0 X1 X6 Y7]
+ (0.0068881943529705975) [X0 Y1 Y6 X7]
+ (0.006901238249797251) [Y0 Z1 Y2 Z12]
+ (0.006901238249797251) [X0 Z1 X2 Z12]
+ (0.006901238249797251) [Y1 Z2 Y3 Z13]
+ (0.006901238249797251) [X1 Z2 X3 Z13]
+ (0.007731425250775303) [Y0 X1 X10 Y11]
+ (0.007731425250775303) [X0 Y1 Y10 X11]
+ (0.008032520918821376) [Y0 Z1 Y2 Z6]
+ (0.008032520918821376) [X0 Z1 X2 Z6]
+ (0.008032520918821376) [Y1 Z2 Y3 Z7]
+ (0.008032520918821376) [X1 Z2 X3 Z7]
+ (0.009560705729135956) [Y8 X9 X10 Y11]
+ (0.009560705729135956) [X8 Y9 Y10 X11]
+ (0.01105502059613205) [Y0 Z1 Y2 Z8]
+ (0.01105502059613205) [X0 Z1 X2 Z8]
+ (0.01105502059613205) [Y1 Z2 Y3 Z9]
+ (0.01105502059613205) [X1 Z2 X3 Z9]
+ (0.01130727400884811) [Y7 Z8 Z9 Y11]
+ (0.01130727400884811) [X7 Z8 Z9 X11]
+ (0.011982389010247934) [Y4 X5 X6 Y7]
+ (0.011982389010247934) [X4 Y5 Y6 X7]
+ (0.013873381748426157) [Y6 X7 X8 Y9]
+ (0.013873381748426157) [X6 Y7 Y8 X9]
+ (0.014583648907612655) [Y0 X1 X2 Y3]
+ (0.014583648907612655) [X0 Y1 Y2 X3]
+ (0.015577208063976456) [Y2 X3 X12 Y13]
+ (0.015577208063976456) [X2 Y3 Y12 X13]
+ (0.01736611899465139) [Y6 X7 X12 Y13]
+ (0.01736611899465139) [X6 Y7 Y12 X13]
+ (0.017680067952481556) [Y4 X5 X10 Y11]
+ (0.017680067952481556) [X4 Y5 Y10 X11]
+ (0.017825140995786356) [Y6 X7 X10 Y11]
+ (0.017825140995786356) [X6 Y7 Y10 X11]
+ (0.019028242443847366) [Y3 X4 X11 Y12]
+ (0.019028242443847366) [X3 Y4 Y11 X12]
+ (0.02538465750845747) [Y2 X3 X10 Y11]
+ (0.02538465750845747) [X2 Y3 Y10 X11]
+ (0.02868518371610596) [Y10 X11 X12 Y13]
+ (0.02868518371610596) [X10 Y11 Y12 X13]
+ (0.029812424517345636) [Y6 Z7 Z8 Y10]
+ (0.029812424517345636) [X6 Z7 Z8 X10]
+ (0.029812424517345636) [Y7 Z9 Z10 Y11]
+ (0.029812424517345636) [X7 Z9 Z10 X11]
+ (0.030104623143456733) [Y6 Z7 Z9 Y10]
+ (0.030104623143456733) [X6 Z7 Z9 X10]
+ (0.030104623143456733) [Y7 Z8 Z10 Y11]
+ (0.030104623143456733) [X7 Z8 Z10 X11]
+ (0.03078750538914389) [Y6 Z8 Z9 Y10]
+ (0.03078750538914389) [X6 Z8 Z9 X10]
+ (0.031143817988967044) [Y2 X3 X6 Y7]
+ (0.031143817988967044) [X2 Y3 Y6 X7]
+ (0.0358395679533535) [Y2 X3 X4 Y5]
+ (0.0358395679533535) [X2 Y3 Y4 X5]
+ (0.03831467029480388) [Y4 X5 X12 Y13]
+ (0.03831467029480388) [X4 Y5 Y12 X13]
+ (0.1043306478065139) [Z0 Y1 Z2 Y3]
+ (0.1043306478065139) [Z0 X1 Z2 X3]
+ (-0.12133276911042369) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133276911042369) [X2 Z3 Z4 Z5 X6]
+ (-0.12133276911042369) [Y3 Z4 Z5 Z6 Y7]
+ (-0.12133276911042369) [X3 Z4 Z5 Z6 X7]
+ (3.2020768799670354e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.2020768799670354e-06) [X0 Z1 Z2 Z3 X4]
+ (3.2020768799670354e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.2020768799670354e-06) [X1 Z2 Z3 Z4 X5]
+ (0.22848106564918705) [Y6 Z7 Z8 Z9 Y10]
+ (0.22848106564918705) [X6 Z7 Z8 Z9 X10]
+ (0.22848106564918713) [Y7 Z8 Z9 Z10 Y11]
+ (0.22848106564918713) [X7 Z8 Z9 Z10 X11]
+ (-0.03276765782329039) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.03276765782329039) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.03276765782329039) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.03276765782329039) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.02711503684527303) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.02711503684527303) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.02711503684527303) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.02711503684527303) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.02599617759802126) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.02599617759802126) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.017561202409646107) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.017561202409646107) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.017561202409646107) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.017561202409646107) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.014564531231172993) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.014564531231172993) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.014564531231172993) [X7 Z8 Z9 X10 X12 X13]
+ (-0.014564531231172993) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.012215040997613894) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.012215040997613894) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.012215040997613894) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.012215040997613894) [X4 Z5 X6 X11 Z12 X13]
+ (-0.012215040997613894) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.012215040997613894) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.012215040997613894) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.012215040997613894) [X5 Z6 X7 X10 Z11 X12]
+ (-0.011756013419819281) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.011756013419819281) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.011756013419819281) [X3 Z4 Z5 X6 X8 X9]
+ (-0.011756013419819281) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.00876482757568882) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.00876482757568882) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.00876482757568882) [X2 Z3 Z4 X5 X11 X12]
+ (-0.00876482757568882) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.00876482757568882) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.00876482757568882) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.00876482757568882) [X3 X4 X10 Z11 Z12 X13]
+ (-0.00876482757568882) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.008125251921381034) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.008125251921381034) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.0073067599288329605) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.0073067599288329605) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.0073067599288329605) [X4 X5 X7 Z8 Z9 X10]
+ (-0.0073067599288329605) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.005805188989826827) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.005805188989826827) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.005805188989826827) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.005805188989826827) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.005652620978017363) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.005652620978017363) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.005652620978017363) [X0 X1 X3 Z4 Z5 X6]
+ (-0.005652620978017363) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.005143391768825088) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.005143391768825088) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.005143391768825088) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.005143391768825088) [X2 Z3 X4 X5 Z6 X7]
+ (-0.004684903388155205) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.004684903388155205) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668620318776307) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.004668620318776307) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.004424855449441857) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.004424855449441857) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.0041587973818400506) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.0041587973818400506) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.0041587973818400506) [X3 Z4 Z5 X6 X12 X13]
+ (-0.0041587973818400506) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.0034937903598901135) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.0034937903598901135) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.0034937903598901135) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.0034937903598901135) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.0027790267990255657) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.0027790267990255657) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.0022939566113524784) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.0022939566113524784) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.0017992194936630086) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.0017992194936630086) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.001727875394136962) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.001727875394136962) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.0009298507967730172) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298507967730172) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298507967730172) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298507967730172) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533856254125548) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.0008533856254125548) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.0008145313270956364) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.0008145313270956364) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.0008145313270956364) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.0008145313270956364) [X3 Z4 Z5 Z6 X7 Z11]
+ (-7.73503688059298e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.73503688059298e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.73503688059298e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.73503688059298e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-8.774817864204363e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.774817864204363e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.774817864204363e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.774817864204363e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.518362215374238e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.518362215374238e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.518362215374238e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.518362215374238e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.444344675570272e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.444344675570272e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.444344675570272e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.444344675570272e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.524373848253113e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.524373848253113e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.524373848253113e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.524373848253113e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-6.290028432856981e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.290028432856981e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.290028432856981e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.290028432856981e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.9743117132007216e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (-5.9743117132007216e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (-5.27588312192402e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.27588312192402e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.6430510682572266e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.6430510682572266e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.5565692178851166e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (-4.5565692178851166e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (-4.253224225426246e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.253224225426246e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-3.7696594516619783e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.7696594516619783e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.6945132942860533e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.6945132942860533e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.610297130268854e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.610297130268854e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.610297130268854e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.610297130268854e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.31314550007884e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.31314550007884e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.2774831952046873e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.2774831952046873e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.2774831952046873e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.2774831952046873e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.211228348174272e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.211228348174272e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.211228348174272e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.211228348174272e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.1513463109976467e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.1513463109976467e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.0882507111111165e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.0882507111111165e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.1726691014525684e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.1726691014525684e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.4548424490936176e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.4548424490936176e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.330473188634091e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.330473188634091e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.228333782517257e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.228333782517257e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.0358477601935817e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.0358477601935817e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-7.956895372909972e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.956895372909972e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.733197741801674e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.733197741801674e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.733197741801674e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.733197741801674e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.628614201404398e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.628614201404398e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.556281914314952e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.556281914314952e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.556281914314952e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.556281914314952e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.418291574356075e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.418291574356075e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.418291574356075e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.418291574356075e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.92745308237064e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.92745308237064e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.92745308237064e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.92745308237064e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.627851911322608e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.627851911322608e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.287660624479483e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.287660624479483e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.287660624479483e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.287660624479483e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.287660624479483e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.287660624479483e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.287660624479483e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.287660624479483e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.838052751011752e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (-4.838052751011752e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (-3.5707613292531e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.5707613292531e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.328139350641667e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.328139350641667e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.086826565289017e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.086826565289017e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.086826565289017e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.086826565289017e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.44732312885163e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (-2.44732312885163e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (-2.371328948003203e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.371328948003203e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.371328948003203e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.371328948003203e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.1990516184119171e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-2.1990516184119171e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-1.933241277199603e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.933241277199603e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.933241277199603e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.933241277199603e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.8394209155384293e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.8394209155384293e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.8394209155384293e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.8394209155384293e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.5510539176156709e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (-1.5510539176156709e-07) [X0 Z1 X2 X4 Z5 X6]
+ (-1.5510539176156709e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (-1.5510539176156709e-07) [X1 Z2 X3 X5 Z6 X7]
+ (-1.3807781481164848e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (-1.3807781481164848e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (-1.3807781481164848e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (-1.3807781481164848e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (-1.3807781481164848e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (-1.3807781481164848e-07) [X0 Z1 X2 X5 Z6 X7]
+ (-1.3807781481164848e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (-1.3807781481164848e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (-1.3807781481164848e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (-1.3807781481164848e-07) [X1 Z2 X3 X4 Z5 X6]
+ (-1.3807781481164848e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (-1.3807781481164848e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (-1.291969486276644e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.291969486276644e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.1076325598865934e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.1076325598865934e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.1076325598865934e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.1076325598865934e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.1076325598865934e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.1076325598865934e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.1076325598865934e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.1076325598865934e-07) [X1 Z2 X3 X10 Z11 X12]
+ (8.057446594310357e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.057446594310357e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.057446594310357e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.057446594310357e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.649310135446586e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.649310135446586e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.649310135446586e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.649310135446586e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.8394209155384293e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.8394209155384293e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.8394209155384293e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.8394209155384293e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.1990516184119171e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (2.1990516184119171e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (2.44732312885163e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (2.44732312885163e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.236259961547862e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.236259961547862e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.236259961547862e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.236259961547862e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.328139350641667e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.328139350641667e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.5707613292531e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.5707613292531e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.838052751011752e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (4.838052751011752e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (5.627851911322608e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.627851911322608e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (6.628614201404398e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.628614201404398e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (7.956895372909972e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.956895372909972e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.306536651566689e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.306536651566689e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.306536651566689e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.306536651566689e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.0358477601935817e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.0358477601935817e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.228333782517257e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.228333782517257e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.2393363216855703e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.2393363216855703e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.2393363216855703e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.2393363216855703e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.330473188634091e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.330473188634091e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.4548424490936176e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.4548424490936176e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.1726691014525684e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.1726691014525684e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.0882507111111165e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.0882507111111165e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.1174479462121217e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.1174479462121217e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.1513463109976467e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.1513463109976467e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.31314550007884e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.31314550007884e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.33433128927938e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.33433128927938e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.6945132942860533e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.6945132942860533e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (4.1839325592911285e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.1839325592911285e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.5565692178851166e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (4.5565692178851166e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (4.6430510682572266e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.6430510682572266e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.27588312192402e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.27588312192402e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.9743117132007216e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (5.9743117132007216e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (0.00029219862611109583) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.00029219862611109583) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.00029219862611109583) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.00029219862611109583) [X6 Z7 X8 X9 Z10 X11]
+ (0.0004956762314917957) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.0004956762314917957) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.0006650070219499374) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.0006650070219499374) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.0006650070219499374) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.0006650070219499374) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.0008533856254125548) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.0008533856254125548) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.0016095313817213789) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.0016095313817213789) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.0016095313817213789) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.0016095313817213789) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.0016676041811440772) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.0016676041811440772) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.0016676041811440772) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.0016676041811440772) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.001727875394136962) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.001727875394136962) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.0017992194936630086) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.0017992194936630086) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.0022939566113524784) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.0022939566113524784) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.0024629170071339334) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.0024629170071339334) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.0024629170071339334) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.0024629170071339334) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.003961560792496556) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.003961560792496556) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.003961560792496556) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.003961560792496556) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.004424855449441857) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.004424855449441857) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.004668620318776307) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.004668620318776307) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.004684903388155205) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.004684903388155205) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.005324835234221677) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.005324835234221677) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.005324835234221677) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.005324835234221677) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.005368659358109469) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.005368659358109469) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.005368659358109469) [X2 X3 X7 Z8 Z9 X10]
+ (0.005368659358109469) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.007960880725921543) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.007960880725921543) [X4 Z5 X6 X10 Z11 X12]
+ (0.007960880725921543) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.007960880725921543) [X5 Z6 X7 X11 Z12 X13]
+ (0.008125251921381034) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.008125251921381034) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.008890731522694558) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.008890731522694558) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.008890731522694558) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.008890731522694558) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.01026341486815854) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.01026341486815854) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.01026341486815854) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.01026341486815854) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.010540425907671442) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.010540425907671442) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.010540425907671442) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.010540425907671442) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.010960074940542486) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.010960074940542486) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.010960074940542486) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.010960074940542486) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.01130727400884811) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.01130727400884811) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.014411099430130967) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.014411099430130967) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.014411099430130967) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.014411099430130967) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.015225630757226603) [Y3 Z4 Z5 X6 X10 Y11]
+ (0.015225630757226603) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (0.015225630757226603) [X3 Z4 Z5 X6 X10 X11]
+ (0.015225630757226603) [X3 Z4 Z5 Y6 Y10 X11]
+ (0.015588250102380219) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (0.015588250102380219) [X2 Z3 X4 X10 Z11 X12]
+ (0.015588250102380219) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (0.015588250102380219) [X3 Z4 X5 X11 Z12 X13]
+ (0.018266834869375446) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.018266834869375446) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.018266834869375446) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.018266834869375446) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.019020423173039872) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.019020423173039872) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.019020423173039872) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.019020423173039872) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.020175921723535436) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.020175921723535436) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.020175921723535436) [X4 Z5 Z6 X7 X11 X12]
+ (0.020175921723535436) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.020175921723535436) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.020175921723535436) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.020175921723535436) [X5 X6 X10 Z11 Z12 X13]
+ (0.020175921723535436) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.024353077678069043) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.024353077678069043) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.024353077678069043) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.024353077678069043) [X2 Z3 X4 X11 Z12 X13]
+ (0.024353077678069043) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.024353077678069043) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.024353077678069043) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.024353077678069043) [X3 Z4 X5 X10 Z11 X12]
+ (0.02438908253114934) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.02438908253114934) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.02438908253114934) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.02438908253114934) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.02510495713884444) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.02510495713884444) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.02510495713884444) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.02510495713884444) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.03078750538914389) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.03078750538914389) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.04587947078129817) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.04587947078129817) [X0 Z2 Z3 Z4 Z5 X6]
+ (0.05600733087780737) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.05600733087780737) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.05600733087780737) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.05600733087780737) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.0560846812466133) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.0560846812466133) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.0560846812466133) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.0560846812466133) [Z1 X7 Z8 Z9 Z10 X11]
+ (-6.63127792809368e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.63127792809368e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-6.631277928093676e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.631277928093676e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.5950860067375375e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.5950860067375375e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.5950860067375372e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.5950860067375372e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.042743277013783665) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.042743277013783665) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (0.04274327701378368) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04274327701378368) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.04764261217638309) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.04764261217638309) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.04764261217638309) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.04764261217638309) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.04171881383982175) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.04171881383982175) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.04171881383982175) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.04171881383982175) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.03956441632289346) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.03956441632289346) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.03956441632289346) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03956441632289346) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03935916802205315) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.03935916802205315) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.03935916802205315) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.03935916802205315) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.0393180519471976) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0393180519471976) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0393180519471976) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0393180519471976) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03560837898831258) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.03560837898831258) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.02990378951262488) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.02990378951262488) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.02990378951262488) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.02990378951262488) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.028730779551905533) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.028730779551905533) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.028730779551905533) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.028730779551905533) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.02563723829602679) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.02563723829602679) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.02563723829602679) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.02563723829602679) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.024755463292891012) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.024755463292891012) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.024755463292891012) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.024755463292891012) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.02428211735469294) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.02428211735469294) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.023145130929529023) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.023145130929529023) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.02252844019601294) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.02252844019601294) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.02143381072160101) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.02143381072160101) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.02143381072160101) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.02143381072160101) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251582) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.019257505095251582) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019028242443847366) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.019028242443847366) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.01888903030494287) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-0.01888903030494287) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-0.01888903030494287) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.01888903030494287) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.01602460368917958) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.01602460368917958) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.015225630757226603) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.015225630757226603) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.014603704729162136) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.014603704729162136) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.014564531231172993) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.014564531231172993) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.01175601341981928) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.01175601341981928) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.00984174924696266) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.00984174924696266) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.00961263460684721) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.00961263460684721) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.00961263460684721) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.00961263460684721) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.008469978791023874) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.008469978791023874) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.0073067599288329605) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.0073067599288329605) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005923798336561343) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.005923798336561343) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.005652620978017363) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.005652620978017363) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.005368659358109469) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005368659358109469) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.0041587973818400506) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.0041587973818400506) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.003356670563832904) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.003356670563832904) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.003356670563832904) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.003356670563832904) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.003267513854423563) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.003267513854423563) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.003267513854423563) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.003267513854423563) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.0027790267990255657) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.0027790267990255657) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.0026860409778066132) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.0026860409778066132) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.0026860409778066132) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.0026860409778066132) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.002293956611352478) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.002293956611352478) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.002293956611352478) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.002293956611352478) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.0009581655836696521) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.0009581655836696521) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.0009581655836696521) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.0009581655836696521) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.0009581655836696521) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.0009581655836696521) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.0009581655836696521) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.0009581655836696521) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.00024636437569586445) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00024636437569586445) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303549398) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.00013840177303549398) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.00013840177303549398) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.00013840177303549398) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.73503688059298e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.73503688059298e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585304987906e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585304987906e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.6103585304987906e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.6103585304987906e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.531680879459254e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.531680879459254e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.531680879459254e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.531680879459254e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.806102774731865e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.806102774731865e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.806102774731865e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.806102774731865e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.089799467264372e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.089799467264372e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.089799467264372e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.089799467264372e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.652209668869263e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.652209668869263e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.652209668869263e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.652209668869263e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.48185183331274e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.48185183331274e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.48185183331274e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.48185183331274e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.071480736234577e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.071480736234577e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.071480736234577e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.071480736234577e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.7346220384972855e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.7346220384972855e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.7346220384972855e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.7346220384972855e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.728843147030733e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.728843147030733e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.728843147030733e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.728843147030733e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.253224225426246e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.253224225426246e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.7696594516619783e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.7696594516619783e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.5443954291250476e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.5443954291250476e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.5443954291250476e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.5443954291250476e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.5443954291250476e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.5443954291250476e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.5443954291250476e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.5443954291250476e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.36095632023364e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.36095632023364e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.36095632023364e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.36095632023364e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-2.103215604480481e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.103215604480481e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.103215604480481e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.103215604480481e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220979464383e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.0111220979464383e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220979464383e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.0111220979464383e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.942946836532737e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.942946836532737e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.942946836532737e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.942946836532737e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174769389102e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.6541174769389102e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174769389102e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.6541174769389102e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.5224930675412572e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.5224930675412572e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.5224930675412572e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.5224930675412572e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.5224930675412572e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930675412572e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930675412572e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.5224930675412572e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.228333782517257e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.228333782517257e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.228333782517257e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.228333782517257e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.98877028891089e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.98877028891089e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.98877028891089e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.98877028891089e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.867765103953693e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765103953693e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765103953693e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.867765103953693e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.189990974860698e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.189990974860698e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.175246206771483e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175246206771483e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.471647744418357e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.471647744418357e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.5614471801393086e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.5614471801393086e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.5614471801393086e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.5614471801393086e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.5233896774717475e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.5233896774717475e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.4273231087715816e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.4273231087715816e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.4273231087715816e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.4273231087715816e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.328139350641667e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350641667e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.328139350641667e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.328139350641667e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.086826565289017e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.086826565289017e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.8882935959382656e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8882935959382656e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8882935959382656e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.8882935959382656e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.371328948003203e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.371328948003203e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.839420915538429e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.839420915538429e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-8.057446594310357e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.057446594310357e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (4.5371780958975035e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.5371780958975035e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.5371780958975035e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (4.5371780958975035e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (8.057446594310357e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.057446594310357e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (9.2093506534043e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.2093506534043e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.2093506534043e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.2093506534043e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783555652185e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783555652185e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.7035783555652185e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.7035783555652185e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.839420915538429e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.839420915538429e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.371328948003203e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.371328948003203e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (3.086826565289017e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.086826565289017e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (4.5233896774717475e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.5233896774717475e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (5.471647744418357e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.471647744418357e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.175246206771483e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175246206771483e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.189990974860698e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.189990974860698e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.330473188634091e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.330473188634091e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.330473188634091e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.330473188634091e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.6288532434563897e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.6288532434563897e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.6288532434563897e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.6288532434563897e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.6893489514000125e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.6893489514000125e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.6893489514000125e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.6893489514000125e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (2.745518400233959e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.745518400233959e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.745518400233959e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.745518400233959e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.745518400233959e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.745518400233959e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.745518400233959e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.745518400233959e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.2118420189412692e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420189412692e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.2118420189412692e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420189412692e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.2118420189412692e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420189412692e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.2118420189412692e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420189412692e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.3131455000788397e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.3131455000788397e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.3131455000788397e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.3131455000788397e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (3.33433128927938e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.33433128927938e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (4.1839325592911285e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.1839325592911285e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (7.73503688059298e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.73503688059298e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.00024636437569586445) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00024636437569586445) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0004458535128840772) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.0004458535128840772) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.0004458535128840772) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.0004458535128840772) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.0005940221543005314) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005314) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005314) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005314) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005314) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005314) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.0005940221543005314) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005314) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.0008533856254125547) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.0008533856254125547) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.0008533856254125547) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.0008533856254125547) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.0010435246534907581) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.0010435246534907581) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.0010435246534907581) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.0010435246534907581) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.0012803060973496747) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.0012803060973496747) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.0012803060973496747) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.0012803060973496747) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.001303800478812702) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (0.001303800478812702) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (0.001303800478812702) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (0.001303800478812702) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (0.0022619660624823537) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (0.0022619660624823537) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (0.0022619660624823537) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (0.0022619660624823537) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (0.0022619660624823537) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (0.0022619660624823537) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (0.0022619660624823537) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (0.0022619660624823537) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (0.003989841456619315) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (0.003989841456619315) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (0.003989841456619315) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (0.003989841456619315) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (0.0041587973818400506) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.0041587973818400506) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.004311038507914321) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.004311038507914321) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.004311038507914321) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.004311038507914321) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.004636976661182579) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.004636976661182579) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.004636976661182579) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.004636976661182579) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.005114473831660384) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.005114473831660384) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.005114473831660384) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.005114473831660384) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.005114473831660384) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660384) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660384) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005114473831660384) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.005241535382803885) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (0.005241535382803885) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (0.005241535382803885) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (0.005241535382803885) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (0.005262642473076838) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.005262642473076838) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.005262642473076838) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.005262642473076838) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.005368659358109469) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005368659358109469) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005379937155839379) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.005379937155839379) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.005379937155839379) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.005379937155839379) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.005652620978017363) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.005652620978017363) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.005708495985960916) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.005708495985960916) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.005708495985960916) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.005708495985960916) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.005923798336561343) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.005923798336561343) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.0073067599288329605) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.0073067599288329605) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.008469978791023874) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.008469978791023874) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.00984174924696266) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.00984174924696266) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.01175601341981928) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.01175601341981928) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.014564531231172993) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.014564531231172993) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.014603704729162136) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.014603704729162136) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.015225630757226603) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.015225630757226603) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.01602460368917958) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.01602460368917958) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.019028242443847366) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.019028242443847366) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.019257505095251582) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.019257505095251582) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.04587947078129817) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04587947078129817) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.36937089366156256) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.36937089366156256) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.36937089366156256) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.36937089366156256) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.28164257767022965) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.28164257767022965) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.28164257767022954) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.28164257767022954) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.09065144207036463) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.09065144207036463) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.09065144207036463) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.09065144207036463) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0868473758986361) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0868473758986361) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0868473758986361) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0868473758986361) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.07635021950635018) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07635021950635018) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07635021950635018) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07635021950635018) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0675238509921403) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0675238509921403) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0675238509921403) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0675238509921403) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.03560837898831258) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.03560837898831258) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.034903343373661654) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.034903343373661654) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.034903343373661654) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.034903343373661654) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.02459186088382994) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.02459186088382994) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.02459186088382994) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.02459186088382994) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.02428211735469294) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.02428211735469294) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.023145130929529027) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.023145130929529027) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.022528440196012935) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.022528440196012935) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019538050311314753) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.019538050311314753) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.019538050311314753) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.019538050311314753) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.017091553155898852) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.017091553155898852) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.017091553155898852) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.017091553155898852) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.01602460368917958) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.01602460368917958) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.01602460368917958) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.01602460368917958) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.010311482489831714) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831714) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831714) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010311482489831714) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.009841749246962662) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962662) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.009841749246962662) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962662) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209868) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209868) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.008826368514209868) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008826368514209868) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.008541996625454852) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454852) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.008541996625454852) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454852) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.008541996625454852) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454852) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008541996625454852) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008541996625454852) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008469978791023874) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023874) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.008469978791023874) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023874) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.004668620318776307) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.004668620318776307) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.003876470899336958) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.003876470899336958) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.00380406617172854) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00380406617172854) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00380406617172854) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.00380406617172854) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178804) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178804) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003356670563832904) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.003356670563832904) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.003267513854423563) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.003267513854423563) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0021413612231016067) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.0021413612231016067) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.001727875394136962) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.001727875394136962) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.0016407548553124046) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0016407548553124046) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214169343) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0014528843214169343) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214169343) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0014528843214169343) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870896771024478) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007870896771024478) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0005192743499487746) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.0005192743499487746) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.0001940085702975657) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0001940085702975657) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303549398) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.00013840177303549398) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.141625221157713e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.141625221157713e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.141625221157713e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.141625221157713e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-5.071480736234578e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.071480736234578e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.1513463109976467e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.1513463109976467e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0882507111111165e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.0882507111111165e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.988511706340689e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.988511706340689e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8742990712620545e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.8742990712620545e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.36095632023364e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.36095632023364e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.3002946561872501e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.3002946561872501e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.146837650678852e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.146837650678852e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.146837650678852e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.146837650678852e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.352332102537661e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.352332102537661e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.352332102537661e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.352332102537661e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.091637198456606e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637198456606e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637198456606e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637198456606e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637198456606e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637198456606e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637198456606e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.091637198456606e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.074305985408553e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.074305985408553e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.074305985408553e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.074305985408553e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.900128985803661e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.900128985803661e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.900128985803661e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900128985803661e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867765103953693e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.867765103953693e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.560692464368689e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692464368689e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692464368689e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692464368689e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692464368689e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692464368689e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692464368689e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.560692464368689e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.99701842179659e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.99701842179659e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.99701842179659e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.99701842179659e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.99701842179659e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.99701842179659e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.99701842179659e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.99701842179659e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.568247520984858e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.568247520984858e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.568247520984858e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.568247520984858e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.376739308331912e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.376739308331912e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.376739308331912e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.376739308331912e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.376739308331912e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.376739308331912e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.376739308331912e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.376739308331912e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.8882935959382656e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.8882935959382656e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.686381544146694e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.686381544146694e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.7035783555652188e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7035783555652188e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.2093506534043e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.2093506534043e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773243584128e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773243584128e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773243584128e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773243584128e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773243584128e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773243584128e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.379773243584128e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773243584128e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253792115272e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253792115272e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9742253792115272e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9742253792115272e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0474716554644885e-08) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.0474716554644885e-08) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.0474716554644885e-08) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.0474716554644885e-08) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.2093506534043e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.2093506534043e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0717282182199522e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.0717282182199522e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.0717282182199522e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.0717282182199522e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.200428749360191e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.200428749360191e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.200428749360191e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.200428749360191e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.7035783555652188e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7035783555652188e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.3120943051100604e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.3120943051100604e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.3120943051100604e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.3120943051100604e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.686381544146694e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.686381544146694e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.8882935959382656e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.8882935959382656e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.092250615915996e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.092250615915996e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.092250615915996e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.092250615915996e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.092250615915996e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.092250615915996e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.092250615915996e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.092250615915996e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.4445978540698637e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.4445978540698637e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.4445978540698637e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.4445978540698637e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.6849150950527575e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.6849150950527575e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.6849150950527575e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.6849150950527575e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.246974424991437e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.246974424991437e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.246974424991437e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.246974424991437e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.246974424991437e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.246974424991437e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.246974424991437e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.246974424991437e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.867765103953693e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.867765103953693e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.3002946561872501e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.3002946561872501e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.36095632023364e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.36095632023364e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.8742990712620545e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.8742990712620545e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.8836765759527645e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.8836765759527645e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.9473560114986904e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9473560114986904e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.9473560114986904e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9473560114986904e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.988511706340689e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.988511706340689e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.0882507111111165e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.0882507111111165e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.1513463109976467e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.1513463109976467e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.846201671105387e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.846201671105387e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.846201671105387e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.846201671105387e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.071480736234578e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.071480736234578e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.10552672180211e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.10552672180211e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.10552672180211e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.10552672180211e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.146496327292638e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.146496327292638e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.146496327292638e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.146496327292638e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.159350501786219e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.159350501786219e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.159350501786219e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.159350501786219e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.427988656200889e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.427988656200889e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.427988656200889e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.427988656200889e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.935867717839379e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.935867717839379e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.935867717839379e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.935867717839379e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.2532733478168765e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.2532733478168765e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.979825793064165e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.979825793064165e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.979825793064165e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.979825793064165e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (4.2055484112201325e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.2055484112201325e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.2055484112201325e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.2055484112201325e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00013840177303549398) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.00013840177303549398) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.00018787053389547007) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.00018787053389547007) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.00018787053389547007) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.00018787053389547007) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0001940085702975657) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.0001940085702975657) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00024636437569586445) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024636437569586445) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00024636437569586445) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024636437569586445) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192743499487746) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.0005192743499487746) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.0007156734248908706) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007156734248908706) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0007156734248908706) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007156734248908706) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024478) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007870896771024478) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.0015324835230730227) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.0015324835230730227) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.0015324835230730227) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.0015324835230730227) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.0016407548553124046) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0016407548553124046) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.001727875394136962) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.001727875394136962) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.0024464971554158965) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.0024464971554158965) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.0024464971554158965) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.0024464971554158965) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.003267513854423563) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.003267513854423563) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.003356670563832904) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.003356670563832904) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.0034841573002178804) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0034841573002178804) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.003876470899336958) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.003876470899336958) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.004668620318776307) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.004668620318776307) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278118) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.004767272188278118) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.004767272188278118) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278118) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.005286546538226892) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.005286546538226892) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.005286546538226892) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.005286546538226892) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.00540895442240998) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.00540895442240998) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.00540895442240998) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.00540895442240998) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.005923798336561343) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561343) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.005923798336561343) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.005923798336561343) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.01071550846979673) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.01071550846979673) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.01071550846979673) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.01071550846979673) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.010757563953908929) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010757563953908929) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010757563953908929) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010757563953908929) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.014603704729162134) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.014603704729162134) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.014603704729162134) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.014603704729162134) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.019299560579363783) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363783) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363783) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363783) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363783) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363783) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.019299560579363783) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363783) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.058591988733861976) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.058591988733861976) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.775950527022736e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.775950527022736e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.775950527022736e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.775950527022736e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0716503518100273) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0716503518100273) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.07165035181002732) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.07165035181002732) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251582) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019257505095251582) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010311482489831714) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311482489831714) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.008826368514209868) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.008826368514209868) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.007597464029770604) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.007597464029770604) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.007597464029770604) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.007597464029770604) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311878) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311878) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311878) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311878) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311878) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311878) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311878) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311878) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676607) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005348051582676607) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005348051582676607) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676607) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00380406617172854) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00380406617172854) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0029841661681219434) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.0029841661681219434) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.0029841661681219434) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.0029841661681219434) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.002446497155415897) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.002446497155415897) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.002249412447093998) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.002249412447093998) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.002249412447093998) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002249412447093998) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0021413612231016067) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.0021413612231016067) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.0018638942824587268) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587268) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587268) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587268) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587268) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587268) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0018638942824587268) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587268) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0016407548553124046) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124046) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0016407548553124046) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124046) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0012223378081538429) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538429) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538429) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538429) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538429) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538429) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538429) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0012223378081538429) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.001028329237856277) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.001028329237856277) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.001028329237856277) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.001028329237856277) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1463061452650621e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1463061452650621e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.8742990712620545e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990712620545e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.8742990712620545e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990712620545e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.3002946561872501e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.3002946561872501e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.3002946561872501e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.3002946561872501e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.044494129760885e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.044494129760885e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.044494129760885e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.044494129760885e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.956079229586587e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.956079229586587e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.956079229586587e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.956079229586587e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105515036716724e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105515036716724e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105515036716724e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105515036716724e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661347212769297e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661347212769297e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661347212769297e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661347212769297e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540341413368254e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540341413368254e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.189990974860698e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.189990974860698e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.876621657880519e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876621657880519e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876621657880519e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876621657880519e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175246206771483e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175246206771483e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.5233896774717475e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.5233896774717475e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0767325319107607e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.0767325319107607e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0767325319107607e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.0767325319107607e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.0134714587380014e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0134714587380014e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.9045998842405945e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.9045998842405945e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.9045998842405945e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.9045998842405945e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.666731754734458e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.666731754734458e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.666731754734458e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.666731754734458e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641928698649e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.8505641928698649e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309316791776e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.6569309316791776e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.6569309316791776e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.6569309316791776e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.8505641928698649e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.8505641928698649e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.6863815441466935e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815441466935e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.6863815441466935e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.6863815441466935e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714587380014e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.0134714587380014e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.5233896774717475e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.5233896774717475e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.6704023904171796e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.6704023904171796e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.6704023904171796e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.6704023904171796e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175246206771483e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175246206771483e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.189990974860698e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.189990974860698e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540341413368254e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540341413368254e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949476486940308e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949476486940308e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.7924939576444087e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939576444087e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.7924939576444087e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.7924939576444087e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.8836765759527645e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.8836765759527645e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.988511706340689e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.988511706340689e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.988511706340689e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.988511706340689e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273347816876e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.253273347816876e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4017109734725097e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4017109734725097e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4017109734725097e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4017109734725097e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.580960369236951e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.580960369236951e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.580960369236951e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.580960369236951e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0005192743499487746) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487746) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.0005192743499487746) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487746) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.0007870896771024478) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024478) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024478) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024478) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0011726348316441857) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0011726348316441857) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0011726348316441857) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0011726348316441857) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.001236647801924521) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.001236647801924521) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.001236647801924521) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.001236647801924521) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.0022009640695004632) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0022009640695004632) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0022009640695004632) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0022009640695004632) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980288) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980288) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0023949726397980288) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0023949726397980288) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0023949726397980288) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0023949726397980288) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0023949726397980288) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0023949726397980288) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002446497155415897) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.002446497155415897) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.00380406617172854) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00380406617172854) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.003876470899336958) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.003876470899336958) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.003876470899336958) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.003876470899336958) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.0042208139700464645) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.0042208139700464645) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.0042208139700464645) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.0042208139700464645) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.008826368514209868) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.008826368514209868) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.010311482489831714) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010311482489831714) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019257505095251582) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019257505095251582) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.058591988733861976) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.058591988733861976) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.3987009016319313e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.3987009016319313e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.398700901631931e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.398700901631931e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0034841573002178804) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0034841573002178804) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002984166168121943) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.002984166168121943) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.0001940085702975657) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0001940085702975657) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061452650623e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061452650623e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.7924939576444087e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.7924939576444087e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540341413368254e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413368254e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540341413368254e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341413368254e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.8505641928698649e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641928698649e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641928698649e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641928698649e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714587380014e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.0134714587380014e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714587380014e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.0134714587380014e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.949476486940307e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.949476486940307e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.7924939576444087e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.7924939576444087e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0001940085702975657) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0001940085702975657) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002984166168121943) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.002984166168121943) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.0034841573002178804) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0034841573002178804) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
  (-73.13873231352528) [I0]
+ (-0.1806679265658332) [Z6]
+ (-0.18066792656583314) [Z7]
+ (-0.15961432501809827) [Z5]
+ (-0.15961432501809825) [Z4]
+ (0.17419956155055735) [Z2]
+ (0.1741995615505574) [Z3]
+ (0.22757269005453573) [Z0]
+ (0.2275726900545358) [Z1]
+ (-8.194261372628613e-06) [Y4 Y6]
+ (-8.194261372628613e-06) [X4 X6]
+ (7.95441317646776e-06) [Y5 Y7]
+ (7.95441317646776e-06) [X5 X7]
+ (0.11270386920332233) [Z4 Z6]
+ (0.11270386920332233) [Z5 Z7]
+ (0.11952438964682677) [Z0 Z4]
+ (0.11952438964682677) [Z1 Z5]
+ (0.1340171526196371) [Z0 Z6]
+ (0.1340171526196371) [Z1 Z7]
+ (0.1373495306426133) [Z0 Z5]
+ (0.1373495306426133) [Z1 Z4]
+ (0.13766872645852585) [Z2 Z4]
+ (0.13766872645852585) [Z3 Z5]
+ (0.1413890529194281) [Z4 Z7]
+ (0.1413890529194281) [Z5 Z6]
+ (0.14722943218766182) [Z2 Z5]
+ (0.14722943218766182) [Z3 Z4]
+ (0.14926355147388903) [Z4 Z5]
+ (0.1497348680349694) [Z2 Z6]
+ (0.1497348680349694) [Z3 Z7]
+ (0.15138327161428855) [Z0 Z7]
+ (0.15138327161428855) [Z1 Z6]
+ (0.1543574865722364) [Z6 Z7]
+ (0.15582269051553127) [Z2 Z7]
+ (0.15582269051553127) [Z3 Z6]
+ (0.16756653265461263) [Z0 Z2]
+ (0.16756653265461263) [Z1 Z3]
+ (0.18143991440303875) [Z0 Z3]
+ (0.18143991440303875) [Z1 Z2]
+ (0.19392534613270196) [Z0 Z1]
+ (0.2200397733437609) [Z2 Z3]
+ (-7.037887510145486e-06) [Y4 Z5 Y6]
+ (-7.037887510145486e-06) [X4 Z5 X6]
+ (-7.037887510145486e-06) [Y5 Z6 Y7]
+ (-7.037887510145486e-06) [X5 Z6 X7]
+ (-0.028685183716105764) [Y4 Y5 X6 X7]
+ (-0.028685183716105764) [X4 X5 Y6 Y7]
+ (-0.017825140995786512) [Y0 Y1 X4 X5]
+ (-0.017825140995786512) [X0 X1 Y4 Y5]
+ (-0.017366118994651455) [Y0 Y1 X6 X7]
+ (-0.017366118994651455) [X0 X1 Y6 Y7]
+ (-0.013873381748426115) [Y0 Y1 X2 X3]
+ (-0.013873381748426115) [X0 X1 Y2 Y3]
+ (-0.009560705729135968) [Y2 Y3 X4 X5]
+ (-0.009560705729135968) [X2 X3 Y4 Y5]
+ (-0.0060878224805618825) [Y2 Y3 X6 X7]
+ (-0.0060878224805618825) [X2 X3 Y6 Y7]
+ (-0.00029219862611106764) [Y1 Y2 X3 X4]
+ (-0.00029219862611106764) [X1 X2 Y3 Y4]
+ (-8.194261372628613e-06) [Z4 Y5 Z6 Y7]
+ (-8.194261372628613e-06) [Z4 X5 Z6 X7]
+ (-2.8909678818513014e-06) [Z0 Y5 Z6 Y7]
+ (-2.8909678818513014e-06) [Z0 X5 Z6 X7]
+ (-2.8909678818513014e-06) [Z1 Y4 Z5 Y6]
+ (-2.8909678818513014e-06) [Z1 X4 Z5 X6]
+ (-1.8551201215901665e-06) [Z0 Y4 Z5 Y6]
+ (-1.8551201215901665e-06) [Z0 X4 Z5 X6]
+ (-1.8551201215901665e-06) [Z1 Y5 Z6 Y7]
+ (-1.8551201215901665e-06) [Z1 X5 Z6 X7]
+ (-1.5973171978824618e-06) [Z2 Y4 Z5 Y6]
+ (-1.5973171978824618e-06) [Z2 X4 Z5 X6]
+ (-1.5973171978824618e-06) [Z3 Y5 Z6 Y7]
+ (-1.5973171978824618e-06) [Z3 X5 Z6 X7]
+ (-1.0358477602611349e-06) [Y0 X1 X5 Y6]
+ (-1.0358477602611349e-06) [Y0 Y1 Y5 Y6]
+ (-1.0358477602611349e-06) [X0 X1 X5 X6]
+ (-1.0358477602611349e-06) [X0 Y1 Y5 X6]
+ (-9.344557776711372e-07) [Z2 Y5 Z6 Y7]
+ (-9.344557776711372e-07) [Z2 X5 Z6 X7]
+ (-9.344557776711372e-07) [Z3 Y4 Z5 Y6]
+ (-9.344557776711372e-07) [Z3 X4 Z5 X6]
+ (6.628614202113245e-07) [Y2 X3 X5 Y6]
+ (6.628614202113245e-07) [Y2 Y3 Y5 Y6]
+ (6.628614202113245e-07) [X2 X3 X5 X6]
+ (6.628614202113245e-07) [X2 Y3 Y5 X6]
+ (7.95441317646776e-06) [Y4 Z5 Y6 Z7]
+ (7.95441317646776e-06) [X4 Z5 X6 Z7]
+ (0.00029219862611106764) [Y1 X2 X3 Y4]
+ (0.00029219862611106764) [X1 Y2 Y3 X4]
+ (0.0060878224805618825) [Y2 X3 X6 Y7]
+ (0.0060878224805618825) [X2 Y3 Y6 X7]
+ (0.009560705729135968) [Y2 X3 X4 Y5]
+ (0.009560705729135968) [X2 Y3 Y4 X5]
+ (0.011307274008848265) [Y1 Z2 Z3 Y5]
+ (0.011307274008848265) [X1 Z2 Z3 X5]
+ (0.013873381748426115) [Y0 X1 X2 Y3]
+ (0.013873381748426115) [X0 Y1 Y2 X3]
+ (0.017366118994651455) [Y0 X1 X6 Y7]
+ (0.017366118994651455) [X0 Y1 Y6 X7]
+ (0.017825140995786512) [Y0 X1 X4 Y5]
+ (0.017825140995786512) [X0 Y1 Y4 X5]
+ (0.028685183716105764) [Y4 X5 X6 Y7]
+ (0.028685183716105764) [X4 Y5 Y6 X7]
+ (0.029812424517345806) [Y0 Z1 Z2 Y4]
+ (0.029812424517345806) [X0 Z1 Z2 X4]
+ (0.029812424517345806) [Y1 Z3 Z4 Y5]
+ (0.029812424517345806) [X1 Z3 Z4 X5]
+ (0.03010462314345688) [Y0 Z1 Z3 Y4]
+ (0.03010462314345688) [X0 Z1 Z3 X4]
+ (0.03010462314345688) [Y1 Z2 Z4 Y5]
+ (0.03010462314345688) [X1 Z2 Z4 X5]
+ (0.03078750538914396) [Y0 Z2 Z3 Y4]
+ (0.03078750538914396) [X0 Z2 Z3 X4]
+ (0.04375263801066101) [Y0 Z1 Z2 Z3 Y4]
+ (0.04375263801066101) [X0 Z1 Z2 Z3 X4]
+ (0.04375263801066101) [Y1 Z2 Z3 Z4 Y5]
+ (0.04375263801066101) [X1 Z2 Z3 Z4 X5]
+ (-0.014564531231172975) [Y1 Z2 Z3 X4 X6 Y7]
+ (-0.014564531231172975) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-0.014564531231172975) [X1 Z2 Z3 X4 X6 X7]
+ (-0.014564531231172975) [X1 Z2 Z3 Y4 Y6 X7]
+ (-6.524373848845315e-06) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (-6.524373848845315e-06) [X0 Z1 Z2 Z3 Z4 X6]
+ (-6.524373848845315e-06) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (-6.524373848845315e-06) [X1 Z2 Z3 Z5 Z6 X7]
+ (-3.769659452008363e-06) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (-3.769659452008363e-06) [X0 Z2 Z3 Z4 Z5 X6]
+ (-3.6102971306253127e-06) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (-3.6102971306253127e-06) [X0 Z1 Z3 Z4 Z5 X6]
+ (-3.6102971306253127e-06) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (-3.6102971306253127e-06) [X1 Z2 Z4 Z5 Z6 X7]
+ (-3.3131455003703414e-06) [Y1 Z2 Z3 Y4 X5 X6]
+ (-3.3131455003703414e-06) [X1 Z2 Z3 X4 Y5 Y6]
+ (-3.27748319553009e-06) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (-3.27748319553009e-06) [X0 Z1 Z2 Z4 Z5 X6]
+ (-3.27748319553009e-06) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (-3.27748319553009e-06) [X1 Z3 Z4 Z5 Z6 X7]
+ (-3.2112283484749735e-06) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (-3.2112283484749735e-06) [X0 Z1 Z2 Z3 Z5 X6]
+ (-3.2112283484749735e-06) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (-3.2112283484749735e-06) [X1 Z2 Z3 Z4 Z6 X7]
+ (-1.0358477602611349e-06) [Y0 Y1 X4 Z5 Z6 X7]
+ (-1.0358477602611349e-06) [X0 X1 Y4 Z5 Z6 Y7]
+ (-6.628614202113245e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (-6.628614202113245e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (-3.3281393509522266e-07) [Y1 X2 X3 Z4 Z5 Y6]
+ (-3.3281393509522266e-07) [X1 Y2 Y3 Z4 Z5 X6]
+ (3.3281393509522266e-07) [Y1 Y2 X3 Z4 Z5 X6]
+ (3.3281393509522266e-07) [X1 X2 Y3 Z4 Z5 Y6]
+ (6.628614202113245e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (6.628614202113245e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (1.0358477602611349e-06) [Y0 X1 X4 Z5 Z6 Y7]
+ (1.0358477602611349e-06) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.3131455003703414e-06) [Y1 Z2 Z3 X4 X5 Y6]
+ (3.3131455003703414e-06) [X1 Z2 Z3 Y4 Y5 X6]
+ (4.183932559591734e-06) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (4.183932559591734e-06) [X1 Z2 Z3 Z4 Z5 X7]
+ (0.00029219862611106764) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (0.00029219862611106764) [Y0 Z1 Y2 X3 Z4 X5]
+ (0.00029219862611106764) [X0 Z1 X2 Y3 Z4 Y5]
+ (0.00029219862611106764) [X0 Z1 X2 X3 Z4 X5]
+ (0.010540425907671624) [Y0 Z1 Z2 Z3 Y4 Z7]
+ (0.010540425907671624) [X0 Z1 Z2 Z3 X4 Z7]
+ (0.010540425907671624) [Y1 Z2 Z3 Z4 Y5 Z6]
+ (0.010540425907671624) [X1 Z2 Z3 Z4 X5 Z6]
+ (0.011307274008848263) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (0.011307274008848263) [X0 Z1 Z2 Z3 X4 Z5]
+ (0.0251049571388446) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (0.0251049571388446) [X0 Z1 Z2 Z3 X4 Z6]
+ (0.0251049571388446) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (0.0251049571388446) [X1 Z2 Z3 Z4 X5 Z7]
+ (0.03078750538914396) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (0.03078750538914396) [Z0 X1 Z2 Z3 Z4 X5]
+ (-5.105396549655733e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (-5.105396549655733e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (-5.105396549655732e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-5.105396549655732e-06) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.014564531231172975) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-0.014564531231172975) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-3.769659452008363e-06) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-3.769659452008363e-06) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-3.3281393509522266e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-3.3281393509522266e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-3.3281393509522266e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-3.3281393509522266e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (3.313145500370342e-06) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (3.313145500370342e-06) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (3.313145500370342e-06) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (3.313145500370342e-06) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (4.183932559591733e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (4.183932559591733e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (0.014564531231172975) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (0.014564531231172975) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
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
(-46.46390678868893+0j) [] +
(-0.01458364890761271+0j) [X0 X1 Y2 Y3] +
(-3.5707613290133667e-07+0j) [X0 X1 Y2 Z3 Z4 Y5] +
(-0.0056526209780173664+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7] +
(-0.008826368514209862+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.7924939577540272e-06+0j) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.570761329013366e-07+0j) [X0 X1 X3 X4] +
(-0.0056526209780173664+0j) [X0 X1 X3 Z4 Z5 X6] +
(-0.008826368514209862+0j) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.7924939577540272e-06+0j) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.0027458364701868046+0j) [X0 X1 Y4 Y5] +
(-2.447323128752577e-07+0j) [X0 X1 Y4 Z5 Z6 Y7] +
(-7.867765104435319e-07+0j) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.003804066171728537+0j) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.447323128752577e-07+0j) [X0 X1 X5 X6] +
(-7.867765104435319e-07+0j) [X0 X1 X5 Z6 Z7 Z8 Z9 X10] +
(-0.003804066171728537+0j) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.0068881943529705905+0j) [X0 X1 Y6 Y7] +
(-7.735036880591797e-05+0j) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11] +
(1.7035783553760626e-07+0j) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.735036880591797e-05+0j) [X0 X1 X7 Z8 Z9 X10] +
(1.7035783553760623e-07+0j) [X0 X1 X7 Z8 Z9 Z10 Z11 X12] +
(-0.006509361201177232+0j) [X0 X1 Y8 Y9] +
(-0.007731425250775277+0j) [X0 X1 Y10 Y11] +
(5.627851911563885e-07+0j) [X0 X1 Y10 Z11 Z12 Y13] +
(5.627851911563885e-07+0j) [X0 X1 X11 X12] +
(-0.005283776488402962+0j) [X0 X1 Y12 Y13] +
(0.01458364890761271+0j) [X0 Y1 Y2 X3] +
(3.5707613290133667e-07+0j) [X0 Y1 Y2 Z3 Z4 X5] +
(0.0056526209780173664+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7] +
(0.008826368514209862+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-1.7924939577540272e-06+0j) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.570761329013366e-07+0j) [X0 Y1 Y3 X4] +
(-0.0056526209780173664+0j) [X0 Y1 Y3 Z4 Z5 X6] +
(-0.008826368514209862+0j) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.7924939577540272e-06+0j) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0027458364701868046+0j) [X0 Y1 Y4 X5] +
(2.447323128752577e-07+0j) [X0 Y1 Y4 Z5 Z6 X7] +
(7.867765104435319e-07+0j) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.003804066171728537+0j) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.447323128752577e-07+0j) [X0 Y1 Y5 X6] +
(-7.867765104435319e-07+0j) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.003804066171728537+0j) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0068881943529705905+0j) [X0 Y1 Y6 X7] +
(7.735036880591797e-05+0j) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11] +
(-1.7035783553760626e-07+0j) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.735036880591797e-05+0j) [X0 Y1 Y7 Z8 Z9 X10] +
(1.7035783553760623e-07+0j) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12] +
(0.006509361201177232+0j) [X0 Y1 Y8 X9] +
(0.007731425250775277+0j) [X0 Y1 Y10 X11] +
(-5.627851911563885e-07+0j) [X0 Y1 Y10 Z11 Z12 X13] +
(5.627851911563885e-07+0j) [X0 Y1 Y11 X12] +
(0.005283776488402962+0j) [X0 Y1 Y12 X13] +
(0.12507032579772262+0j) [X0 Z1 X2] +
(-1.9332412770431303e-07+0j) [X0 Z1 X2 X3 Z4 X5] +
(-0.0022939566113524814+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 X7] +
(-0.0016407548553124369+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(3.0134714590111076e-07+0j) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.9332412770431303e-07+0j) [X0 Z1 X2 Y3 Z4 Y5] +
(-0.0022939566113524814+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7] +
(-0.0016407548553124369+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(3.0134714590111076e-07+0j) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.003276971931231722+0j) [X0 Z1 X2 Z3] +
(-1.5510539176222496e-07+0j) [X0 Z1 X2 X4 Z5 X6] +
(-1.1468376508070932e-06+0j) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.00759746402977063+0j) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.380778148014566e-07+0j) [X0 Z1 X2 Y4 Z5 Y6] +
(-7.900128986672915e-07+0j) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.005348051582676642+0j) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.005530759218631583+0j) [X0 Z1 X2 Z4] +
(-1.3807781480145658e-07+0j) [X0 Z1 X2 X5 Z6 X7] +
(-3.3767393085546214e-07+0j) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.001863894282458761+0j) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.3807781480145658e-07+0j) [X0 Z1 X2 Y5 Z6 Y7] +
(-3.3767393085546214e-07+0j) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.001863894282458761+0j) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0011059037691897337+0j) [X0 Z1 X2 Z5] +
(-8.352332103590471e-07+0j) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
1.9742253798173834e-08j [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.005262642473076853+0j) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10] +
(-8.074305986352681e-07+0j) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.008032520918821468+0j) [X0 Z1 X2 Z6] +
(0.0005940221543005511+0j) [X0 Z1 X2 X7 Z8 Z9 Z10 X11] +
(-8.379773245175891e-08+0j) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005940221543005511+0j) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11] +
(-8.379773245175891e-08+0j) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0033476175306662607+0j) [X0 Z1 X2 Z7] +
(0.011055020596132148+0j) [X0 Z1 X2 Z8] +
(0.00292976867475113+0j) [X0 Z1 X2 Z9] +
(-6.418291574943104e-07+0j) [X0 Z1 X2 X10 Z11 X12] +
(-6.556281914841462e-07+0j) [X0 Z1 X2 Y10 Z11 Y12] +
(0.003555290195504292+0j) [X0 Z1 X2 Z10] +
(-1.1076325600034288e-07+0j) [X0 Z1 X2 X11 Z12 X13] +
(-1.1076325600034288e-07+0j) [X0 Z1 X2 Y11 Z12 Y13] +
(0.0017560707018412875+0j) [X0 Z1 X2 Z11] +
(0.0069012382497973465+0j) [X0 Z1 X2 Z12] +
(0.0023262306231581417+0j) [X0 Z1 X2 Z13] +
(-3.568247521398017e-07+0j) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.002249412447093989+0j) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.047471655652236e-08+0j) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.00044585351288407415+0j) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10] +
(-1.974225379764584e-08+0j) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004424855449441849+0j) [X0 Z1 Z2 X3 Y4 Y5] +
(-4.523389678118293e-07+0j) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.003484157300217879+0j) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-8.091637199516312e-07+0j) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10] +
(-0.005733569747311868+0j) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004684903388155206+0j) [X0 Z1 Z2 X3 Y6 Y7] +
(0.004668620318776303+0j) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11] +
(-7.189990975626834e-07+0j) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005114473831660377+0j) [X0 Z1 Z2 X3 X7 Z8 Z9 X10] +
(-7.560692465281139e-07+0j) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12] +
(-0.008125251921381017+0j) [X0 Z1 Z2 X3 Y8 Y9] +
(-0.001799219493663005+0j) [X0 Z1 Z2 X3 Y10 Y11] +
(-5.471647744820939e-07+0j) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13] +
(-5.287660624956771e-07+0j) [X0 Z1 Z2 X3 X11 X12] +
(-0.004575007626639206+0j) [X0 Z1 Z2 X3 Y12 Y13] +
(0.004424855449441849+0j) [X0 Z1 Z2 Y3 Y4 X5] +
(4.523389678118293e-07+0j) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.003484157300217879+0j) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.091637199516312e-07+0j) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.005733569747311868+0j) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.004684903388155206+0j) [X0 Z1 Z2 Y3 Y6 X7] +
(-0.004668620318776303+0j) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11] +
(7.189990975626834e-07+0j) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005114473831660377+0j) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10] +
(-7.560692465281139e-07+0j) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12] +
(0.008125251921381017+0j) [X0 Z1 Z2 Y3 Y8 X9] +
(0.001799219493663005+0j) [X0 Z1 Z2 Y3 Y10 X11] +
(5.471647744820939e-07+0j) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13] +
(-5.287660624956771e-07+0j) [X0 Z1 Z2 Y3 Y11 X12] +
(0.004575007626639206+0j) [X0 Z1 Z2 Y3 Y12 X13] +
(3.2020768810808515e-06+0j) [X0 Z1 Z2 Z3 X4] +
(0.0008533856254125504+0j) [X0 Z1 Z2 Z3 X4 X5 Z6 X7] +
(0.0007870896771024421+0j) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0008533856254125504+0j) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7] +
(0.0007870896771024421+0j) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.2919694861145907e-07+0j) [X0 Z1 Z2 Z3 X4 Z5] +
(4.444597854137022e-07+0j) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10] +
(0.0011726348316441822+0j) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.6849150953243406e-07+0j) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10] +
(0.002200964069500465+0j) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.8394209153712403e-07+0j) [X0 Z1 Z2 Z3 X4 Z6] +
(4.092250616235365e-07+0j) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11] +
(0.002394972639798023+0j) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.092250616235365e-07+0j) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11] +
(0.002394972639798023+0j) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.236259961416304e-07+0j) [X0 Z1 Z2 Z3 X4 Z8] +
(8.649310136046995e-08+0j) [X0 Z1 Z2 Z3 X4 Z9] +
(0.0013038004788126993+0j) [X0 Z1 Z2 Z3 X4 X10 Z11 X12] +
(0.003989841456619311+0j) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12] +
(-6.733197742605319e-07+0j) [X0 Z1 Z2 Z3 X4 Z10] +
(0.0022619660624823516+0j) [X0 Z1 Z2 Z3 X4 X11 Z12 X13] +
(0.0022619660624823516+0j) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13] +
(-5.927453082975489e-07+0j) [X0 Z1 Z2 Z3 X4 Z11] +
(1.2393363217466594e-06+0j) [X0 Z1 Z2 Z3 X4 Z12] +
(9.306536652226529e-07+0j) [X0 Z1 Z2 Z3 X4 Z13] +
(-0.0010283292378562825+0j) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.0026860409778066132+0j) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12] +
(-1.8394209153712403e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7] +
(-0.00019400857029755881+0j) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0012223378081538411+0j) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-2.371328947811605e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9] +
(8.057446596298295e-08+0j) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11] +
(0.0017278753941369603+0j) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13] +
(-0.0009581655836696526+0j) [X0 Z1 Z2 Z3 Z4 X5 X11 X12] +
(-3.0868265652400643e-07+0j) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13] +
(1.8394209153712403e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7] +
(0.00019400857029755881+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0012223378081538411+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(2.371328947811605e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9] +
(-8.057446596298295e-08+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11] +
(-0.0017278753941369603+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13] +
(-0.0009581655836696526+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12] +
(3.0868265652400643e-07+0j) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13] +
(0.04274327701378391+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6] +
(0.0005192743499487794+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(-1.8505641927581952e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005192743499487794+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(-1.8505641927581952e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.002779026799025543+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7] +
(0.00463697666118258+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8] +
(0.0012803060973496886+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9] +
(2.3120943053566103e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12] +
(1.0717282185093111e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12] +
(0.005379937155839381+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10] +
(7.246974425686464e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13] +
(7.246974425686464e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13] +
(0.0052415353828038835+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11] +
(0.00431103850791433+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12] +
(0.0010435246534907761+0j) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13] +
(1.2004287494347943e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12] +
(-0.0033566705638328927+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9] +
(-0.00013840177303549712+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11] +
(-6.175246207177152e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(-4.997018422236147e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12] +
(-0.003267513854423554+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13] +
(0.0033566705638328927+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9] +
(0.00013840177303549712+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11] +
(6.175246207177152e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(-4.997018422236147e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12] +
(0.003267513854423554+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13] +
(0.003876470899336946+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(-7.540341413838855e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.003876470899336946+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(-7.540341413838855e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(0.07165035181002585+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.0021413612231015694+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(0.004220813970046483+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(0.0012366478019245491+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(-0.002984166168121934+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(0.002984166168121934+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-1.3987009017082342e-05+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(8.949476487021924e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-6.876621658878703e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(-7.66134721372531e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(0.0015324835230730552+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10] +
(-2.904599884957336e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(0.005408954422410001+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10] +
(-1.044494129879619e-06+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(0.004767272188278136+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10] +
(-8.105515037949221e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(0.005286546538226915+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10] +
(-9.956079230707415e-07+0j) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0016095313817213938+0j) [X0 Z1 Z2 Z3 Z4 X6] +
(-7.141625221154273e-05+0j) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10] +
(-2.666731755118129e-07+0j) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0024629170071339443+0j) [X0 Z1 Z2 Z3 Z5 X6] +
(0.0007156734248908993+0j) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10] +
(-3.076732532482142e-07+0j) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.6060718680436896e-07+0j) [X0 Z1 Z2 X4] +
(0.003961560792496562+0j) [X0 Z1 Z2 Z4 Z5 X6] +
(0.00018787053389551555+0j) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.6569309311218813e-07+0j) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.7379332625094263e-07+0j) [X0 Z1 Z3 X4] +
(0.0016676041811440803+0j) [X0 Z1 Z3 Z4 Z5 X6] +
(-0.0014528843214169213+0j) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(4.670402390132989e-07+0j) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.10433064780651428+0j) [X0 X2] +
(3.11744794603745e-06+0j) [X0 Z2 Z3 X4] +
(0.04587947078129816+0j) [X0 Z2 Z3 Z4 Z5 X6] +
(0.0585919887338619+0j) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-1.1463061453460747e-05+0j) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.01458364890761271+0j) [Y0 X1 X2 Y3] +
(3.5707613290133667e-07+0j) [Y0 X1 X2 Z3 Z4 Y5] +
(0.0056526209780173664+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7] +
(0.008826368514209862+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.7924939577540272e-06+0j) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.570761329013366e-07+0j) [Y0 X1 X3 Y4] +
(-0.0056526209780173664+0j) [Y0 X1 X3 Z4 Z5 Y6] +
(-0.008826368514209862+0j) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.7924939577540272e-06+0j) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0027458364701868046+0j) [Y0 X1 X4 Y5] +
(2.447323128752577e-07+0j) [Y0 X1 X4 Z5 Z6 Y7] +
(7.867765104435319e-07+0j) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.003804066171728537+0j) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.447323128752577e-07+0j) [Y0 X1 X5 Y6] +
(-7.867765104435319e-07+0j) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.003804066171728537+0j) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0068881943529705905+0j) [Y0 X1 X6 Y7] +
(7.735036880591797e-05+0j) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11] +
(-1.7035783553760626e-07+0j) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.735036880591797e-05+0j) [Y0 X1 X7 Z8 Z9 Y10] +
(1.7035783553760623e-07+0j) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12] +
(0.006509361201177232+0j) [Y0 X1 X8 Y9] +
(0.007731425250775277+0j) [Y0 X1 X10 Y11] +
(-5.627851911563885e-07+0j) [Y0 X1 X10 Z11 Z12 Y13] +
(5.627851911563885e-07+0j) [Y0 X1 X11 Y12] +
(0.005283776488402962+0j) [Y0 X1 X12 Y13] +
(-0.01458364890761271+0j) [Y0 Y1 X2 X3] +
(-3.5707613290133667e-07+0j) [Y0 Y1 X2 Z3 Z4 X5] +
(-0.0056526209780173664+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7] +
(-0.008826368514209862+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.7924939577540272e-06+0j) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.570761329013366e-07+0j) [Y0 Y1 Y3 Y4] +
(-0.0056526209780173664+0j) [Y0 Y1 Y3 Z4 Z5 Y6] +
(-0.008826368514209862+0j) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.7924939577540272e-06+0j) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.0027458364701868046+0j) [Y0 Y1 X4 X5] +
(-2.447323128752577e-07+0j) [Y0 Y1 X4 Z5 Z6 X7] +
(-7.867765104435319e-07+0j) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.003804066171728537+0j) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.447323128752577e-07+0j) [Y0 Y1 Y5 Y6] +
(-7.867765104435319e-07+0j) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.003804066171728537+0j) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.0068881943529705905+0j) [Y0 Y1 X6 X7] +
(-7.735036880591797e-05+0j) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11] +
(1.7035783553760626e-07+0j) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.735036880591797e-05+0j) [Y0 Y1 Y7 Z8 Z9 Y10] +
(1.7035783553760623e-07+0j) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.006509361201177232+0j) [Y0 Y1 X8 X9] +
(-0.007731425250775277+0j) [Y0 Y1 X10 X11] +
(5.627851911563885e-07+0j) [Y0 Y1 X10 Z11 Z12 X13] +
(5.627851911563885e-07+0j) [Y0 Y1 Y11 Y12] +
(-0.005283776488402962+0j) [Y0 Y1 X12 X13] +
(-3.568247521398017e-07+0j) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.002249412447093989+0j) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.00044585351288407415+0j) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10] +
(-1.974225379764584e-08+0j) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.047471655652236e-08+0j) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.12507032579772262+0j) [Y0 Z1 Y2] +
(-1.9332412770431303e-07+0j) [Y0 Z1 Y2 X3 Z4 X5] +
(-0.0022939566113524814+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7] +
(-0.0016407548553124369+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(3.0134714590111076e-07+0j) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.9332412770431303e-07+0j) [Y0 Z1 Y2 Y3 Z4 Y5] +
(-0.0022939566113524814+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7] +
(-0.0016407548553124369+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(3.0134714590111076e-07+0j) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.003276971931231722+0j) [Y0 Z1 Y2 Z3] +
(-1.380778148014566e-07+0j) [Y0 Z1 Y2 X4 Z5 X6] +
(-7.900128986672915e-07+0j) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.005348051582676642+0j) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.5510539176222496e-07+0j) [Y0 Z1 Y2 Y4 Z5 Y6] +
(-1.1468376508070932e-06+0j) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.00759746402977063+0j) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.005530759218631583+0j) [Y0 Z1 Y2 Z4] +
(-1.3807781480145658e-07+0j) [Y0 Z1 Y2 X5 Z6 X7] +
(-3.3767393085546214e-07+0j) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.001863894282458761+0j) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.3807781480145658e-07+0j) [Y0 Z1 Y2 Y5 Z6 Y7] +
(-3.3767393085546214e-07+0j) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.001863894282458761+0j) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0011059037691897337+0j) [Y0 Z1 Y2 Z5] +
(0.005262642473076853+0j) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10] +
(-8.074305986352681e-07+0j) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
-1.9742253798173834e-08j [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(-8.352332103590471e-07+0j) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.008032520918821468+0j) [Y0 Z1 Y2 Z6] +
(0.0005940221543005511+0j) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11] +
(-8.379773245175891e-08+0j) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005940221543005511+0j) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11] +
(-8.379773245175891e-08+0j) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0033476175306662607+0j) [Y0 Z1 Y2 Z7] +
(0.011055020596132148+0j) [Y0 Z1 Y2 Z8] +
(0.00292976867475113+0j) [Y0 Z1 Y2 Z9] +
(-6.556281914841462e-07+0j) [Y0 Z1 Y2 X10 Z11 X12] +
(-6.418291574943104e-07+0j) [Y0 Z1 Y2 Y10 Z11 Y12] +
(0.003555290195504292+0j) [Y0 Z1 Y2 Z10] +
(-1.1076325600034288e-07+0j) [Y0 Z1 Y2 X11 Z12 X13] +
(-1.1076325600034288e-07+0j) [Y0 Z1 Y2 Y11 Z12 Y13] +
(0.0017560707018412875+0j) [Y0 Z1 Y2 Z11] +
(0.0069012382497973465+0j) [Y0 Z1 Y2 Z12] +
(0.0023262306231581417+0j) [Y0 Z1 Y2 Z13] +
(0.004424855449441849+0j) [Y0 Z1 Z2 X3 X4 Y5] +
(4.523389678118293e-07+0j) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.003484157300217879+0j) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-8.091637199516312e-07+0j) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.005733569747311868+0j) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.004684903388155206+0j) [Y0 Z1 Z2 X3 X6 Y7] +
(-0.004668620318776303+0j) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11] +
(7.189990975626834e-07+0j) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005114473831660377+0j) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10] +
(-7.560692465281139e-07+0j) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12] +
(0.008125251921381017+0j) [Y0 Z1 Z2 X3 X8 Y9] +
(0.001799219493663005+0j) [Y0 Z1 Z2 X3 X10 Y11] +
(5.471647744820939e-07+0j) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13] +
(-5.287660624956771e-07+0j) [Y0 Z1 Z2 X3 X11 Y12] +
(0.004575007626639206+0j) [Y0 Z1 Z2 X3 X12 Y13] +
(-0.004424855449441849+0j) [Y0 Z1 Z2 Y3 X4 X5] +
(-4.523389678118293e-07+0j) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.003484157300217879+0j) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.091637199516312e-07+0j) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.005733569747311868+0j) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004684903388155206+0j) [Y0 Z1 Z2 Y3 X6 X7] +
(0.004668620318776303+0j) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11] +
(-7.189990975626834e-07+0j) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005114473831660377+0j) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10] +
(-7.560692465281139e-07+0j) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.008125251921381017+0j) [Y0 Z1 Z2 Y3 X8 X9] +
(-0.001799219493663005+0j) [Y0 Z1 Z2 Y3 X10 X11] +
(-5.471647744820939e-07+0j) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13] +
(-5.287660624956771e-07+0j) [Y0 Z1 Z2 Y3 Y11 Y12] +
(-0.004575007626639206+0j) [Y0 Z1 Z2 Y3 X12 X13] +
(-0.0010283292378562825+0j) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.0026860409778066132+0j) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12] +
(3.2020768810808515e-06+0j) [Y0 Z1 Z2 Z3 Y4] +
(0.0008533856254125504+0j) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7] +
(0.0007870896771024421+0j) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0008533856254125504+0j) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7] +
(0.0007870896771024421+0j) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.2919694861145907e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z5] +
(4.6849150953243406e-07+0j) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10] +
(0.002200964069500465+0j) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.444597854137022e-07+0j) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10] +
(0.0011726348316441822+0j) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.8394209153712403e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z6] +
(4.092250616235365e-07+0j) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11] +
(0.002394972639798023+0j) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.092250616235365e-07+0j) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11] +
(0.002394972639798023+0j) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.236259961416304e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z8] +
(8.649310136046995e-08+0j) [Y0 Z1 Z2 Z3 Y4 Z9] +
(0.003989841456619311+0j) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12] +
(0.0013038004788126993+0j) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12] +
(-6.733197742605319e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z10] +
(0.0022619660624823516+0j) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13] +
(0.0022619660624823516+0j) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13] +
(-5.927453082975489e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z11] +
(1.2393363217466594e-06+0j) [Y0 Z1 Z2 Z3 Y4 Z12] +
(9.306536652226529e-07+0j) [Y0 Z1 Z2 Z3 Y4 Z13] +
(1.8394209153712403e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7] +
(0.00019400857029755881+0j) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0012223378081538411+0j) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(2.371328947811605e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9] +
(-8.057446596298295e-08+0j) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11] +
(-0.0017278753941369603+0j) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13] +
(-0.0009581655836696526+0j) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12] +
(3.0868265652400643e-07+0j) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13] +
(-1.8394209153712403e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7] +
(-0.00019400857029755881+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0012223378081538411+0j) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-2.371328947811605e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9] +
(8.057446596298295e-08+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11] +
(0.0017278753941369603+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13] +
(-0.0009581655836696526+0j) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12] +
(-3.0868265652400643e-07+0j) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13] +
(1.2004287494347943e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12] +
(0.04274327701378391+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6] +
(0.0005192743499487794+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(-1.8505641927581952e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0005192743499487794+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(-1.8505641927581952e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.002779026799025543+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7] +
(0.00463697666118258+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8] +
(0.0012803060973496886+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9] +
(1.0717282185093111e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12] +
(2.3120943053566103e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12] +
(0.005379937155839381+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10] +
(7.246974425686464e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13] +
(7.246974425686464e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13] +
(0.0052415353828038835+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11] +
(0.00431103850791433+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12] +
(0.0010435246534907761+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13] +
(0.0033566705638328927+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9] +
(0.00013840177303549712+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11] +
(6.175246207177152e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(-4.997018422236147e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12] +
(0.003267513854423554+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13] +
(-0.0033566705638328927+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9] +
(-0.00013840177303549712+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11] +
(-6.175246207177152e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(-4.997018422236147e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12] +
(-0.003267513854423554+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13] +
(0.003876470899336946+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(-7.540341413838855e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.003876470899336946+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(-7.540341413838855e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(0.07165035181002585+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.0021413612231015694+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(0.004220813970046483+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(0.0012366478019245491+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(0.002984166168121934+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-0.002984166168121934+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-1.3987009017082342e-05+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(8.949476487021924e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-6.876621658878703e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(-7.66134721372531e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(0.0015324835230730552+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10] +
(-2.904599884957336e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(0.005408954422410001+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10] +
(-1.044494129879619e-06+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(0.004767272188278136+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10] +
(-8.105515037949221e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(0.005286546538226915+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10] +
(-9.956079230707415e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0016095313817213938+0j) [Y0 Z1 Z2 Z3 Z4 Y6] +
(-7.141625221154273e-05+0j) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10] +
(-2.666731755118129e-07+0j) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0024629170071339443+0j) [Y0 Z1 Z2 Z3 Z5 Y6] +
(0.0007156734248908993+0j) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10] +
(-3.076732532482142e-07+0j) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(3.6060718680436896e-07+0j) [Y0 Z1 Z2 Y4] +
(0.003961560792496562+0j) [Y0 Z1 Z2 Z4 Z5 Y6] +
(0.00018787053389551555+0j) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.6569309311218813e-07+0j) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.7379332625094263e-07+0j) [Y0 Z1 Z3 Y4] +
(0.0016676041811440803+0j) [Y0 Z1 Z3 Z4 Z5 Y6] +
(-0.0014528843214169213+0j) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(4.670402390132989e-07+0j) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.10433064780651428+0j) [Y0 Y2] +
(3.11744794603745e-06+0j) [Y0 Z2 Z3 Y4] +
(0.04587947078129816+0j) [Y0 Z2 Z3 Z4 Z5 Y6] +
(0.0585919887338619+0j) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-1.1463061453460747e-05+0j) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(12.412630742111766+0j) [Z0] +
(0.10433064780651428+0j) [Z0 X1 Z2 X3] +
(3.11744794603745e-06+0j) [Z0 X1 Z2 Z3 Z4 X5] +
(0.045879470781298164+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7] +
(0.0585919887338619+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-1.1463061453460746e-05+0j) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.10433064780651428+0j) [Z0 Y1 Z2 Y3] +
(3.11744794603745e-06+0j) [Z0 Y1 Z2 Z3 Z4 Y5] +
(0.045879470781298164+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7] +
(0.0585919887338619+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-1.1463061453460746e-05+0j) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-8.337746752815771e-07+0j) [Z0 X2 Z3 X4] +
(-0.027115036845273062+0j) [Z0 X2 Z3 Z4 Z5 X6] +
(-0.06752385099214034+0j) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.4017109735623567e-05+0j) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-8.337746752815771e-07+0j) [Z0 Y2 Z3 Y4] +
(-0.027115036845273062+0j) [Z0 Y2 Z3 Z4 Z5 Y6] +
(-0.06752385099214034+0j) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.4017109735623567e-05+0j) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.2367108078383042+0j) [Z0 Z2] +
(-1.1908508081829136e-06+0j) [Z0 X3 Z4 X5] +
(-0.03276765782329043+0j) [Z0 X3 Z4 Z5 Z6 X7] +
(-0.07635021950635021+0j) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.5809603693377593e-05+0j) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.1908508081829136e-06+0j) [Z0 Y3 Z4 Y5] +
(-0.03276765782329043+0j) [Z0 Y3 Z4 Z5 Z6 Y7] +
(-0.07635021950635021+0j) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.5809603693377593e-05+0j) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.2512944567459169+0j) [Z0 Z3] +
(-3.099349243501555e-06+0j) [Z0 X4 Z5 X6] +
(-1.5316808795935526e-05+0j) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.08684737589863623+0j) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-3.099349243501555e-06+0j) [Z0 Y4 Z5 Y6] +
(-1.5316808795935526e-05+0j) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.08684737589863623+0j) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.19661770890342145+0j) [Z0 Z4] +
(-3.3440815563768127e-06+0j) [Z0 X5 Z6 X7] +
(-1.610358530637906e-05+0j) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.09065144207036475+0j) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.3440815563768127e-06+0j) [Z0 Y5 Z6 Y7] +
(-1.610358530637906e-05+0j) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.09065144207036475+0j) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.19936354537360823+0j) [Z0 Z5] +
(0.05608468124661341+0j) [Z0 X6 Z7 Z8 Z9 X10] +
(-6.652209669977058e-06+0j) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.05608468124661341+0j) [Z0 Y6 Z7 Z8 Z9 Y10] +
(-6.652209669977058e-06+0j) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.24164663936017242+0j) [Z0 Z6] +
(0.056007330877807494+0j) [Z0 X7 Z8 Z9 Z10 X11] +
(-6.481851834439452e-06+0j) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.056007330877807494+0j) [Z0 Y7 Z8 Z9 Z10 Y11] +
(-6.481851834439452e-06+0j) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.248534833713143+0j) [Z0 Z7] +
(0.2723251830660568+0j) [Z0 Z8] +
(0.278834544267234+0j) [Z0 Z9] +
(-2.1776646052089496e-06+0j) [Z0 X10 Z11 X12] +
(-2.1776646052089496e-06+0j) [Z0 Y10 Z11 Y12] +
(0.19299723935364232+0j) [Z0 Z10] +
(-1.614879414052561e-06+0j) [Z0 X11 Z12 X13] +
(-1.614879414052561e-06+0j) [Z0 Y11 Z12 Y13] +
(0.20072866460441763+0j) [Z0 Z11] +
(0.21102659849791539+0j) [Z0 Z12] +
(0.21631037498631833+0j) [Z0 Z13] +
(1.9332412770431303e-07+0j) [X1 X2 Y3 Y4] +
(0.0022939566113524814+0j) [X1 X2 Y3 Z4 Z5 Y6] +
(0.0016407548553124367+0j) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-3.0134714590111076e-07+0j) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004424855449441849+0j) [X1 X2 X4 X5] +
(-8.091637199516312e-07+0j) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.005733569747311868+0j) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-4.523389678118293e-07+0j) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.003484157300217879+0j) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.0046849033881552074+0j) [X1 X2 X6 X7] +
(0.005114473831660377+0j) [X1 X2 X6 Z7 Z8 Z9 Z10 X11] +
(-7.560692465281139e-07+0j) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.004668620318776303+0j) [X1 X2 Y7 Z8 Z9 Y10] +
(-7.189990975626834e-07+0j) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.008125251921381017+0j) [X1 X2 X8 X9] +
(-0.001799219493663005+0j) [X1 X2 X10 X11] +
(-5.287660624956771e-07+0j) [X1 X2 X10 Z11 Z12 X13] +
(-5.471647744820939e-07+0j) [X1 X2 Y11 Y12] +
(-0.004575007626639206+0j) [X1 X2 X12 X13] +
(-1.9332412770431303e-07+0j) [X1 Y2 Y3 X4] +
(-0.0022939566113524814+0j) [X1 Y2 Y3 Z4 Z5 X6] +
(-0.0016407548553124367+0j) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(3.0134714590111076e-07+0j) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004424855449441849+0j) [X1 Y2 Y4 X5] +
(-8.091637199516312e-07+0j) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.005733569747311868+0j) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.523389678118293e-07+0j) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10] +
(0.003484157300217879+0j) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.0046849033881552074+0j) [X1 Y2 Y6 X7] +
(0.005114473831660377+0j) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11] +
(-7.560692465281139e-07+0j) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.004668620318776303+0j) [X1 Y2 Y7 Z8 Z9 X10] +
(7.189990975626834e-07+0j) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.008125251921381017+0j) [X1 Y2 Y8 X9] +
(-0.001799219493663005+0j) [X1 Y2 Y10 X11] +
(-5.287660624956771e-07+0j) [X1 Y2 Y10 Z11 Z12 X13] +
(5.471647744820939e-07+0j) [X1 Y2 Y11 X12] +
(-0.004575007626639206+0j) [X1 Y2 Y12 X13] +
(0.1250703257977227+0j) [X1 Z2 X3] +
(-1.3807781480145658e-07+0j) [X1 Z2 X3 X4 Z5 X6] +
(-3.3767393085546214e-07+0j) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.001863894282458761+0j) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.3807781480145658e-07+0j) [X1 Z2 X3 Y4 Z5 Y6] +
(-3.3767393085546214e-07+0j) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.001863894282458761+0j) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0011059037691897337+0j) [X1 Z2 X3 Z4] +
(-1.5510539176222496e-07+0j) [X1 Z2 X3 X5 Z6 X7] +
(-1.1468376508070932e-06+0j) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.00759746402977063+0j) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.380778148014566e-07+0j) [X1 Z2 X3 Y5 Z6 Y7] +
(-7.900128986672915e-07+0j) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.005348051582676642+0j) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005530759218631583+0j) [X1 Z2 X3 Z5] +
(0.0005940221543005511+0j) [X1 Z2 X3 X6 Z7 Z8 Z9 X10] +
(-8.379773245175891e-08+0j) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0005940221543005511+0j) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10] +
(-8.379773245175891e-08+0j) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0033476175306662607+0j) [X1 Z2 X3 Z6] +
(-8.352332103590471e-07+0j) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
1.9742253798173834e-08j [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005262642473076853+0j) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11] +
(-8.074305986352681e-07+0j) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.008032520918821468+0j) [X1 Z2 X3 Z7] +
(0.00292976867475113+0j) [X1 Z2 X3 Z8] +
(0.011055020596132148+0j) [X1 Z2 X3 Z9] +
(-1.1076325600034288e-07+0j) [X1 Z2 X3 X10 Z11 X12] +
(-1.1076325600034288e-07+0j) [X1 Z2 X3 Y10 Z11 Y12] +
(0.0017560707018412875+0j) [X1 Z2 X3 Z10] +
(-6.418291574943104e-07+0j) [X1 Z2 X3 X11 Z12 X13] +
(-6.556281914841462e-07+0j) [X1 Z2 X3 Y11 Z12 Y13] +
(0.003555290195504292+0j) [X1 Z2 X3 Z11] +
(0.0023262306231581417+0j) [X1 Z2 X3 Z12] +
(0.0069012382497973465+0j) [X1 Z2 X3 Z13] +
(-3.568247521398017e-07+0j) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.002249412447093989+0j) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.047471655652236e-08+0j) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.00044585351288407415+0j) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11] +
(-1.974225379764584e-08+0j) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0008533856254125506+0j) [X1 Z2 Z3 X4 Y5 Y6] +
(-0.0007870896771024421+0j) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10] +
(-1.83942091537124e-07+0j) [X1 Z2 Z3 X4 X6 X7] +
(-0.0012223378081538411+0j) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.00019400857029755881+0j) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12] +
(-2.371328947811605e-07+0j) [X1 Z2 Z3 X4 X8 X9] +
(8.057446596298295e-08+0j) [X1 Z2 Z3 X4 X10 X11] +
(-0.0009581655836696526+0j) [X1 Z2 Z3 X4 X10 Z11 Z12 X13] +
(0.0017278753941369603+0j) [X1 Z2 Z3 X4 Y11 Y12] +
(-3.0868265652400643e-07+0j) [X1 Z2 Z3 X4 X12 X13] +
(0.0008533856254125506+0j) [X1 Z2 Z3 Y4 Y5 X6] +
(0.0007870896771024421+0j) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10] +
(-1.83942091537124e-07+0j) [X1 Z2 Z3 Y4 Y6 X7] +
(-0.0012223378081538411+0j) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.00019400857029755881+0j) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12] +
(-2.371328947811605e-07+0j) [X1 Z2 Z3 Y4 Y8 X9] +
(8.057446596298295e-08+0j) [X1 Z2 Z3 Y4 Y10 X11] +
(-0.0009581655836696526+0j) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13] +
(-0.0017278753941369603+0j) [X1 Z2 Z3 Y4 Y11 X12] +
(-3.0868265652400643e-07+0j) [X1 Z2 Z3 Y4 Y12 X13] +
(3.2020768810808507e-06+0j) [X1 Z2 Z3 Z4 X5] +
(4.092250616235365e-07+0j) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10] +
(0.002394972639798023+0j) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.092250616235365e-07+0j) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10] +
(0.002394972639798023+0j) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.444597854137022e-07+0j) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11] +
(0.0011726348316441822+0j) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.6849150953243406e-07+0j) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11] +
(0.002200964069500465+0j) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.8394209153712403e-07+0j) [X1 Z2 Z3 Z4 X5 Z7] +
(8.649310136046995e-08+0j) [X1 Z2 Z3 Z4 X5 Z8] +
(3.236259961416304e-07+0j) [X1 Z2 Z3 Z4 X5 Z9] +
(0.0022619660624823516+0j) [X1 Z2 Z3 Z4 X5 X10 Z11 X12] +
(0.0022619660624823516+0j) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12] +
(-5.927453082975489e-07+0j) [X1 Z2 Z3 Z4 X5 Z10] +
(0.0013038004788126993+0j) [X1 Z2 Z3 Z4 X5 X11 Z12 X13] +
(0.003989841456619311+0j) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13] +
(-6.733197742605319e-07+0j) [X1 Z2 Z3 Z4 X5 Z11] +
(9.306536652226529e-07+0j) [X1 Z2 Z3 Z4 X5 Z12] +
(1.2393363217466594e-06+0j) [X1 Z2 Z3 Z4 X5 Z13] +
(-0.0010283292378562825+0j) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0026860409778066132+0j) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13] +
(-0.0005192743499487794+0j) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10] +
(1.8505641927581952e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.0033566705638328927+0j) [X1 Z2 Z3 Z4 Z5 X6 X8 X9] +
(-0.00013840177303549712+0j) [X1 Z2 Z3 Z4 Z5 X6 X10 X11] +
(-4.997018422236147e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13] +
(-6.175246207177152e-07+0j) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12] +
(-0.003267513854423554+0j) [X1 Z2 Z3 Z4 Z5 X6 X12 X13] +
(0.0005192743499487794+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10] +
(-1.8505641927581952e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.0033566705638328927+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9] +
(-0.00013840177303549712+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11] +
(-4.997018422236147e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13] +
(6.175246207177152e-07+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12] +
(-0.003267513854423554+0j) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13] +
(0.04274327701378391+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7] +
(0.0012803060973496886+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8] +
(0.00463697666118258+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9] +
(7.246974425686464e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12] +
(7.246974425686464e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12] +
(0.0052415353828038835+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10] +
(2.3120943053566103e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13] +
(1.0717282185093111e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13] +
(0.005379937155839381+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11] +
(0.0010435246534907761+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12] +
(0.00431103850791433+0j) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13] +
(1.2004287494347943e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13] +
(-0.003876470899336946+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10] +
(7.540341413838855e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(0.003876470899336946+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10] +
(-7.540341413838855e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(-0.002984166168121934+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-0.002984166168121934+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(0.0716503518100258+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.0012366478019245491+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(0.004220813970046483+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(-1.3987009017082345e-05+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(8.949476487021924e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(-7.66134721372531e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(-0.0021413612231015694+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11] +
(-6.876621658878703e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(0.005408954422410001+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11] +
(-1.044494129879619e-06+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(0.0015324835230730552+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11] +
(-2.904599884957336e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(0.005286546538226915+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11] +
(-9.956079230707415e-07+0j) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.002779026799025543+0j) [X1 Z2 Z3 Z4 Z5 X7] +
(0.004767272188278136+0j) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11] +
(-8.105515037949221e-07+0j) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0024629170071339443+0j) [X1 Z2 Z3 Z4 Z6 X7] +
(0.0007156734248908993+0j) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11] +
(-3.076732532482142e-07+0j) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2919694861145907e-07+0j) [X1 Z2 Z3 X5] +
(0.0016095313817213938+0j) [X1 Z2 Z3 Z5 Z6 X7] +
(-7.141625221154273e-05+0j) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-2.666731755118129e-07+0j) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.7379332625094263e-07+0j) [X1 Z2 Z4 X5] +
(0.0016676041811440803+0j) [X1 Z2 Z4 Z5 Z6 X7] +
(-0.0014528843214169213+0j) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(4.670402390132989e-07+0j) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.0032769719312317224+0j) [X1 X3] +
(3.6060718680436896e-07+0j) [X1 Z3 Z4 X5] +
(0.003961560792496562+0j) [X1 Z3 Z4 Z5 Z6 X7] +
(0.00018787053389551555+0j) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.6569309311218813e-07+0j) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.9332412770431303e-07+0j) [Y1 X2 X3 Y4] +
(-0.0022939566113524814+0j) [Y1 X2 X3 Z4 Z5 Y6] +
(-0.0016407548553124367+0j) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(3.0134714590111076e-07+0j) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.004424855449441849+0j) [Y1 X2 X4 Y5] +
(-8.091637199516312e-07+0j) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.005733569747311868+0j) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(4.523389678118293e-07+0j) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10] +
(0.003484157300217879+0j) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.0046849033881552074+0j) [Y1 X2 X6 Y7] +
(0.005114473831660377+0j) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11] +
(-7.560692465281139e-07+0j) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.004668620318776303+0j) [Y1 X2 X7 Z8 Z9 Y10] +
(7.189990975626834e-07+0j) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.008125251921381017+0j) [Y1 X2 X8 Y9] +
(-0.001799219493663005+0j) [Y1 X2 X10 Y11] +
(-5.287660624956771e-07+0j) [Y1 X2 X10 Z11 Z12 Y13] +
(5.471647744820939e-07+0j) [Y1 X2 X11 Y12] +
(-0.004575007626639206+0j) [Y1 X2 X12 Y13] +
(1.9332412770431303e-07+0j) [Y1 Y2 X3 X4] +
(0.0022939566113524814+0j) [Y1 Y2 X3 Z4 Z5 X6] +
(0.0016407548553124367+0j) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-3.0134714590111076e-07+0j) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.004424855449441849+0j) [Y1 Y2 Y4 Y5] +
(-8.091637199516312e-07+0j) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.005733569747311868+0j) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-4.523389678118293e-07+0j) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10] +
(-0.003484157300217879+0j) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.0046849033881552074+0j) [Y1 Y2 Y6 Y7] +
(0.005114473831660377+0j) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11] +
(-7.560692465281139e-07+0j) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.004668620318776303+0j) [Y1 Y2 X7 Z8 Z9 X10] +
(-7.189990975626834e-07+0j) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12] +
(-0.008125251921381017+0j) [Y1 Y2 Y8 Y9] +
(-0.001799219493663005+0j) [Y1 Y2 Y10 Y11] +
(-5.287660624956771e-07+0j) [Y1 Y2 Y10 Z11 Z12 Y13] +
(-5.471647744820939e-07+0j) [Y1 Y2 X11 X12] +
(-0.004575007626639206+0j) [Y1 Y2 Y12 Y13] +
(-3.568247521398017e-07+0j) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.002249412447093989+0j) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.00044585351288407415+0j) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11] +
(-1.974225379764584e-08+0j) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.047471655652236e-08+0j) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.1250703257977227+0j) [Y1 Z2 Y3] +
(-1.3807781480145658e-07+0j) [Y1 Z2 Y3 X4 Z5 X6] +
(-3.3767393085546214e-07+0j) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.001863894282458761+0j) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.3807781480145658e-07+0j) [Y1 Z2 Y3 Y4 Z5 Y6] +
(-3.3767393085546214e-07+0j) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.001863894282458761+0j) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0011059037691897337+0j) [Y1 Z2 Y3 Z4] +
(-1.380778148014566e-07+0j) [Y1 Z2 Y3 X5 Z6 X7] +
(-7.900128986672915e-07+0j) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.005348051582676642+0j) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.5510539176222496e-07+0j) [Y1 Z2 Y3 Y5 Z6 Y7] +
(-1.1468376508070932e-06+0j) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.00759746402977063+0j) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005530759218631583+0j) [Y1 Z2 Y3 Z5] +
(0.0005940221543005511+0j) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10] +
(-8.379773245175891e-08+0j) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0005940221543005511+0j) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10] +
(-8.379773245175891e-08+0j) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0033476175306662607+0j) [Y1 Z2 Y3 Z6] +
(0.005262642473076853+0j) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11] +
(-8.074305986352681e-07+0j) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
-1.9742253798173834e-08j [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.352332103590471e-07+0j) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.008032520918821468+0j) [Y1 Z2 Y3 Z7] +
(0.00292976867475113+0j) [Y1 Z2 Y3 Z8] +
(0.011055020596132148+0j) [Y1 Z2 Y3 Z9] +
(-1.1076325600034288e-07+0j) [Y1 Z2 Y3 X10 Z11 X12] +
(-1.1076325600034288e-07+0j) [Y1 Z2 Y3 Y10 Z11 Y12] +
(0.0017560707018412875+0j) [Y1 Z2 Y3 Z10] +
(-6.556281914841462e-07+0j) [Y1 Z2 Y3 X11 Z12 X13] +
(-6.418291574943104e-07+0j) [Y1 Z2 Y3 Y11 Z12 Y13] +
(0.003555290195504292+0j) [Y1 Z2 Y3 Z11] +
(0.0023262306231581417+0j) [Y1 Z2 Y3 Z12] +
(0.0069012382497973465+0j) [Y1 Z2 Y3 Z13] +
(0.0008533856254125506+0j) [Y1 Z2 Z3 X4 X5 Y6] +
(0.0007870896771024421+0j) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10] +
(-1.83942091537124e-07+0j) [Y1 Z2 Z3 X4 X6 Y7] +
(-0.0012223378081538411+0j) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.00019400857029755881+0j) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12] +
(-2.371328947811605e-07+0j) [Y1 Z2 Z3 X4 X8 Y9] +
(8.057446596298295e-08+0j) [Y1 Z2 Z3 X4 X10 Y11] +
(-0.0009581655836696526+0j) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13] +
(-0.0017278753941369603+0j) [Y1 Z2 Z3 X4 X11 Y12] +
(-3.0868265652400643e-07+0j) [Y1 Z2 Z3 X4 X12 Y13] +
(-0.0008533856254125506+0j) [Y1 Z2 Z3 Y4 X5 X6] +
(-0.0007870896771024421+0j) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10] +
(-1.83942091537124e-07+0j) [Y1 Z2 Z3 Y4 Y6 Y7] +
(-0.0012223378081538411+0j) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.00019400857029755881+0j) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12] +
(-2.371328947811605e-07+0j) [Y1 Z2 Z3 Y4 Y8 Y9] +
(8.057446596298295e-08+0j) [Y1 Z2 Z3 Y4 Y10 Y11] +
(-0.0009581655836696526+0j) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13] +
(0.0017278753941369603+0j) [Y1 Z2 Z3 Y4 X11 X12] +
(-3.0868265652400643e-07+0j) [Y1 Z2 Z3 Y4 Y12 Y13] +
(-0.0010283292378562825+0j) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0026860409778066132+0j) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13] +
(3.2020768810808507e-06+0j) [Y1 Z2 Z3 Z4 Y5] +
(4.092250616235365e-07+0j) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10] +
(0.002394972639798023+0j) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.092250616235365e-07+0j) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10] +
(0.002394972639798023+0j) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.6849150953243406e-07+0j) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11] +
(0.002200964069500465+0j) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.444597854137022e-07+0j) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11] +
(0.0011726348316441822+0j) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.8394209153712403e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z7] +
(8.649310136046995e-08+0j) [Y1 Z2 Z3 Z4 Y5 Z8] +
(3.236259961416304e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z9] +
(0.0022619660624823516+0j) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12] +
(0.0022619660624823516+0j) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12] +
(-5.927453082975489e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z10] +
(0.003989841456619311+0j) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13] +
(0.0013038004788126993+0j) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13] +
(-6.733197742605319e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z11] +
(9.306536652226529e-07+0j) [Y1 Z2 Z3 Z4 Y5 Z12] +
(1.2393363217466594e-06+0j) [Y1 Z2 Z3 Z4 Y5 Z13] +
(0.0005192743499487794+0j) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10] +
(-1.8505641927581952e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.0033566705638328927+0j) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9] +
(-0.00013840177303549712+0j) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11] +
(-4.997018422236147e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13] +
(6.175246207177152e-07+0j) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12] +
(-0.003267513854423554+0j) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13] +
(-0.0005192743499487794+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10] +
(1.8505641927581952e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-0.0033566705638328927+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9] +
(-0.00013840177303549712+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11] +
(-4.997018422236147e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13] +
(-6.175246207177152e-07+0j) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12] +
(-0.003267513854423554+0j) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13] +
(1.2004287494347943e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13] +
(0.04274327701378391+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7] +
(0.0012803060973496886+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8] +
(0.00463697666118258+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9] +
(7.246974425686464e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12] +
(7.246974425686464e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12] +
(0.0052415353828038835+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10] +
(1.0717282185093111e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13] +
(2.3120943053566103e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13] +
(0.005379937155839381+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11] +
(0.0010435246534907761+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12] +
(0.00431103850791433+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13] +
(0.003876470899336946+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10] +
(-7.540341413838855e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-0.003876470899336946+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10] +
(7.540341413838855e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-0.002984166168121934+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(-0.002984166168121934+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(0.0716503518100258+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.0012366478019245491+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(0.004220813970046483+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(-1.3987009017082345e-05+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(8.949476487021924e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(-7.66134721372531e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(-0.0021413612231015694+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11] +
(-6.876621658878703e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(0.005408954422410001+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11] +
(-1.044494129879619e-06+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(0.0015324835230730552+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11] +
(-2.904599884957336e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(0.005286546538226915+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11] +
(-9.956079230707415e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.002779026799025543+0j) [Y1 Z2 Z3 Z4 Z5 Y7] +
(0.004767272188278136+0j) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11] +
(-8.105515037949221e-07+0j) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0024629170071339443+0j) [Y1 Z2 Z3 Z4 Z6 Y7] +
(0.0007156734248908993+0j) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11] +
(-3.076732532482142e-07+0j) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.2919694861145907e-07+0j) [Y1 Z2 Z3 Y5] +
(0.0016095313817213938+0j) [Y1 Z2 Z3 Z5 Z6 Y7] +
(-7.141625221154273e-05+0j) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-2.666731755118129e-07+0j) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.7379332625094263e-07+0j) [Y1 Z2 Z4 Y5] +
(0.0016676041811440803+0j) [Y1 Z2 Z4 Z5 Z6 Y7] +
(-0.0014528843214169213+0j) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(4.670402390132989e-07+0j) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.0032769719312317224+0j) [Y1 Y3] +
(3.6060718680436896e-07+0j) [Y1 Z3 Z4 Y5] +
(0.003961560792496562+0j) [Y1 Z3 Z4 Z5 Z6 Y7] +
(0.00018787053389551555+0j) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.6569309311218813e-07+0j) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(12.412630742111766+0j) [Z1] +
(-1.1908508081829136e-06+0j) [Z1 X2 Z3 X4] +
(-0.03276765782329043+0j) [Z1 X2 Z3 Z4 Z5 X6] +
(-0.07635021950635021+0j) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(1.5809603693377593e-05+0j) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.1908508081829136e-06+0j) [Z1 Y2 Z3 Y4] +
(-0.03276765782329043+0j) [Z1 Y2 Z3 Z4 Z5 Y6] +
(-0.07635021950635021+0j) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(1.5809603693377593e-05+0j) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.2512944567459169+0j) [Z1 Z2] +
(-8.337746752815771e-07+0j) [Z1 X3 Z4 X5] +
(-0.027115036845273062+0j) [Z1 X3 Z4 Z5 Z6 X7] +
(-0.06752385099214034+0j) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(1.4017109735623567e-05+0j) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-8.337746752815771e-07+0j) [Z1 Y3 Z4 Y5] +
(-0.027115036845273062+0j) [Z1 Y3 Z4 Z5 Z6 Y7] +
(-0.06752385099214034+0j) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(1.4017109735623567e-05+0j) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.2367108078383042+0j) [Z1 Z3] +
(-3.3440815563768127e-06+0j) [Z1 X4 Z5 X6] +
(-1.610358530637906e-05+0j) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.09065144207036475+0j) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-3.3440815563768127e-06+0j) [Z1 Y4 Z5 Y6] +
(-1.610358530637906e-05+0j) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.09065144207036475+0j) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.19936354537360823+0j) [Z1 Z4] +
(-3.099349243501555e-06+0j) [Z1 X5 Z6 X7] +
(-1.5316808795935526e-05+0j) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.08684737589863623+0j) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.099349243501555e-06+0j) [Z1 Y5 Z6 Y7] +
(-1.5316808795935526e-05+0j) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.08684737589863623+0j) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.19661770890342145+0j) [Z1 Z5] +
(0.056007330877807494+0j) [Z1 X6 Z7 Z8 Z9 X10] +
(-6.481851834439452e-06+0j) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.056007330877807494+0j) [Z1 Y6 Z7 Z8 Z9 Y10] +
(-6.481851834439452e-06+0j) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.248534833713143+0j) [Z1 Z6] +
(0.05608468124661341+0j) [Z1 X7 Z8 Z9 Z10 X11] +
(-6.652209669977058e-06+0j) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.05608468124661341+0j) [Z1 Y7 Z8 Z9 Z10 Y11] +
(-6.652209669977058e-06+0j) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.24164663936017242+0j) [Z1 Z7] +
(0.278834544267234+0j) [Z1 Z8] +
(0.2723251830660568+0j) [Z1 Z9] +
(-1.614879414052561e-06+0j) [Z1 X10 Z11 X12] +
(-1.614879414052561e-06+0j) [Z1 Y10 Z11 Y12] +
(0.20072866460441763+0j) [Z1 Z10] +
(-2.1776646052089496e-06+0j) [Z1 X11 Z12 X13] +
(-2.1776646052089496e-06+0j) [Z1 Y11 Z12 Y13] +
(0.19299723935364232+0j) [Z1 Z11] +
(0.21631037498631833+0j) [Z1 Z12] +
(0.21102659849791539+0j) [Z1 Z13] +
(-0.03583956795335349+0j) [X2 X3 Y4 Y5] +
(-2.1990516186075172e-07+0j) [X2 X3 Y4 Z5 Z6 Y7] +
(-2.3609563203844245e-06+0j) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.010311482489831783+0j) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.1990516186075172e-07+0j) [X2 X3 X5 X6] +
(-2.3609563203844245e-06+0j) [X2 X3 X5 Z6 Z7 Z8 Z9 X10] +
(-0.010311482489831783+0j) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.03114381798896712+0j) [X2 X3 Y6 Y7] +
(0.005368659358109507+0j) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11] +
(9.209350642367092e-08+0j) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005368659358109507+0j) [X2 X3 X7 Z8 Z9 X10] +
(9.209350642367092e-08+0j) [X2 X3 X7 Z8 Z9 Z10 Z11 X12] +
(-0.036194123559042606+0j) [X2 X3 Y8 Y9] +
(-0.02538465750845745+0j) [X2 X3 Y10 Y11] +
(2.1726691015626666e-06+0j) [X2 X3 Y10 Z11 Z12 Y13] +
(2.1726691015626666e-06+0j) [X2 X3 X11 X12] +
(-0.01557720806397646+0j) [X2 X3 Y12 Y13] +
(0.03583956795335349+0j) [X2 Y3 Y4 X5] +
(2.1990516186075172e-07+0j) [X2 Y3 Y4 Z5 Z6 X7] +
(2.3609563203844245e-06+0j) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(0.010311482489831783+0j) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.1990516186075172e-07+0j) [X2 Y3 Y5 X6] +
(-2.3609563203844245e-06+0j) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10] +
(-0.010311482489831783+0j) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.03114381798896712+0j) [X2 Y3 Y6 X7] +
(-0.005368659358109507+0j) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11] +
(-9.209350642367092e-08+0j) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005368659358109507+0j) [X2 Y3 Y7 Z8 Z9 X10] +
(9.209350642367092e-08+0j) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12] +
(0.036194123559042606+0j) [X2 Y3 Y8 X9] +
(0.02538465750845745+0j) [X2 Y3 Y10 X11] +
(-2.1726691015626666e-06+0j) [X2 Y3 Y10 Z11 Z12 X13] +
(2.1726691015626666e-06+0j) [X2 Y3 Y11 X12] +
(0.01557720806397646+0j) [X2 Y3 Y12 X13] +
(-3.887051672440412e-06+0j) [X2 Z3 X4] +
(-0.005143391768825092+0j) [X2 Z3 X4 X5 Z6 X7] +
(-0.00984174924696267+0j) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.988511706224744e-06+0j) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.005143391768825092+0j) [X2 Z3 X4 Y5 Z6 Y7] +
(-0.00984174924696267+0j) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.988511706224744e-06+0j) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.764994119155435e-07+0j) [X2 Z3 X4 Z5] +
(1.6893489515400427e-06+0j) [X2 Z3 X4 X6 Z7 Z8 Z9 X10] +
(0.010757563953908937+0j) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.5371780962314595e-08+0j) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10] +
(4.205548411218745e-05+0j) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.5935343916139047e-07+0j) [X2 Z3 X4 Z6] +
(3.211842019208348e-06+0j) [X2 Z3 X4 X7 Z8 Z9 Z10 X11] +
(0.019299560579363776+0j) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.211842019208348e-06+0j) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11] +
(0.019299560579363776+0j) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.1954890098126153e-06+0j) [X2 Z3 X4 Z7] +
(2.1868423782137136e-07+0j) [X2 Z3 X4 Z8] +
(-5.770052993949372e-07+0j) [X2 Z3 X4 Z9] +
(0.015588250102380243+0j) [X2 Z3 X4 X10 Z11 X12] +
(0.005324835234221695+0j) [X2 Z3 X4 Y10 Z11 Y12] +
(-3.1586564320842818e-06+0j) [X2 Z3 X4 Z10] +
(0.02435307767806904+0j) [X2 Z3 X4 X11 Z12 X13] +
(0.02435307767806904+0j) [X2 Z3 X4 Y11 Z12 Y13] +
(-7.801707500746608e-06+0j) [X2 Z3 X4 Z11] +
(3.5390541846291337e-06+0j) [X2 Z3 X4 Z12] +
(8.814937306924811e-06+0j) [X2 Z3 X4 Z13] +
(1.628853243590582e-06+0j) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10] +
(0.01071550846979675+0j) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.010263414868158547+0j) [X2 Z3 Y4 Y10 Z11 X12] +
(-1.454842448974006e-06+0j) [X2 Z3 Z4 X5 Y6 Y7] +
(-3.1513463112588878e-06+0j) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11] +
(-0.019257505095251592+0j) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.5224930676683058e-06+0j) [X2 Z3 Z4 X5 X7 Z8 Z9 X10] +
(-0.008541996625454839+0j) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-7.956895372163085e-07+0j) [X2 Z3 Z4 X5 Y8 Y9] +
(-4.643051068662326e-06+0j) [X2 Z3 Z4 X5 Y10 Y11] +
(-0.019028242443847345+0j) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13] +
(-0.008764827575688795+0j) [X2 Z3 Z4 X5 X11 X12] +
(5.275883122295677e-06+0j) [X2 Z3 Z4 X5 Y12 Y13] +
(1.454842448974006e-06+0j) [X2 Z3 Z4 Y5 Y6 X7] +
(3.1513463112588878e-06+0j) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11] +
(0.019257505095251592+0j) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.5224930676683058e-06+0j) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10] +
(-0.008541996625454839+0j) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(7.956895372163085e-07+0j) [X2 Z3 Z4 Y5 Y8 X9] +
(4.643051068662326e-06+0j) [X2 Z3 Z4 Y5 Y10 X11] +
(0.019028242443847345+0j) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13] +
(-0.008764827575688795+0j) [X2 Z3 Z4 Y5 Y11 X12] +
(-5.275883122295677e-06+0j) [X2 Z3 Z4 Y5 Y12 X13] +
(-0.1213327691104234+0j) [X2 Z3 Z4 Z5 X6] +
(-0.00846997879102387+0j) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(2.6863815468884714e-07+0j) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.00846997879102387+0j) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(2.6863815468884714e-07+0j) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.02599617759802128+0j) [X2 Z3 Z4 Z5 X6 Z7] +
(-0.005805188989826866+0j) [X2 Z3 Z4 Z5 X6 Z8] +
(-0.017561202409646134+0j) [X2 Z3 Z4 Z5 X6 Z9] +
(-7.988770288320358e-07+0j) [X2 Z3 Z4 Z5 X6 X10 Z11 X12] +
(-3.4273231086827337e-07+0j) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12] +
(-0.0008145313270956425+0j) [X2 Z3 Z4 Z5 X6 Z10] +
(2.7455184005914887e-06+0j) [X2 Z3 Z4 Z5 X6 X11 Z12 X13] +
(2.7455184005914887e-06+0j) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13] +
(0.014411099430130955+0j) [X2 Z3 Z4 Z5 X6 Z11] +
(0.0006650070219499201+0j) [X2 Z3 Z4 Z5 X6 Z12] +
(-0.0034937903598901516+0j) [X2 Z3 Z4 Z5 X6 Z13] +
(-4.5614471796376247e-07+0j) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12] +
(-0.01175601341981927+0j) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9] +
(0.015225630757226598+0j) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11] +
(-3.0882507114597616e-06+0j) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(-3.544395429423524e-06+0j) [X2 Z3 Z4 Z5 Z6 X7 X11 X12] +
(-0.004158797381840071+0j) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13] +
(0.01175601341981927+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9] +
(-0.015225630757226598+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11] +
(3.0882507114597616e-06+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(-3.544395429423524e-06+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12] +
(0.004158797381840071+0j) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13] +
(0.014603704729162123+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(-2.8742990714502686e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.014603704729162123+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(-2.8742990714502686e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(-0.281642577670229+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(-1.3002946562663052e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(-1.3002946562663052e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(-0.02428211735469295+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(-0.019538050311314767+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(-0.017091553155898886+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(0.002446497155415882+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(-0.002446497155415882+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(5.7759505273980086e-05+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(2.883676576139899e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(5.146496327675988e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(3.846201671409681e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(-0.03935916802205314+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10] +
(7.97982579361234e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(-0.02475546329289101+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10] +
(5.105526722162071e-06+0j) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(-0.021433810721601006+0j) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10] +
(5.159350502016612e-06+0j) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(-0.029903789512624866+0j) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10] +
(5.427988656705459e-06+0j) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(0.0016638798784908073+0j) [X2 Z3 Z4 X6] +
(-0.01888903030494289+0j) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10] +
(2.94735601187036e-06+0j) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.0034795118903342844+0j) [X2 Z3 Z5 X6] +
(-0.02873077955190556+0j) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10] +
(5.935867718095103e-06+0j) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.6021167407252466e-06+0j) [X2 X4] +
(0.0004956762314917412+0j) [X2 Z4 Z5 X6] +
(-0.03560837898831261+0j) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10] +
(7.253273348318091e-06+0j) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.03583956795335349+0j) [Y2 X3 X4 Y5] +
(2.1990516186075172e-07+0j) [Y2 X3 X4 Z5 Z6 Y7] +
(2.3609563203844245e-06+0j) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(0.010311482489831783+0j) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-2.1990516186075172e-07+0j) [Y2 X3 X5 Y6] +
(-2.3609563203844245e-06+0j) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10] +
(-0.010311482489831783+0j) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.03114381798896712+0j) [Y2 X3 X6 Y7] +
(-0.005368659358109507+0j) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11] +
(-9.209350642367092e-08+0j) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.005368659358109507+0j) [Y2 X3 X7 Z8 Z9 Y10] +
(9.209350642367092e-08+0j) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12] +
(0.036194123559042606+0j) [Y2 X3 X8 Y9] +
(0.02538465750845745+0j) [Y2 X3 X10 Y11] +
(-2.1726691015626666e-06+0j) [Y2 X3 X10 Z11 Z12 Y13] +
(2.1726691015626666e-06+0j) [Y2 X3 X11 Y12] +
(0.01557720806397646+0j) [Y2 X3 X12 Y13] +
(-0.03583956795335349+0j) [Y2 Y3 X4 X5] +
(-2.1990516186075172e-07+0j) [Y2 Y3 X4 Z5 Z6 X7] +
(-2.3609563203844245e-06+0j) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.010311482489831783+0j) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-2.1990516186075172e-07+0j) [Y2 Y3 Y5 Y6] +
(-2.3609563203844245e-06+0j) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10] +
(-0.010311482489831783+0j) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.03114381798896712+0j) [Y2 Y3 X6 X7] +
(0.005368659358109507+0j) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11] +
(9.209350642367092e-08+0j) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.005368659358109507+0j) [Y2 Y3 Y7 Z8 Z9 Y10] +
(9.209350642367092e-08+0j) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.036194123559042606+0j) [Y2 Y3 X8 X9] +
(-0.02538465750845745+0j) [Y2 Y3 X10 X11] +
(2.1726691015626666e-06+0j) [Y2 Y3 X10 Z11 Z12 X13] +
(2.1726691015626666e-06+0j) [Y2 Y3 Y11 Y12] +
(-0.01557720806397646+0j) [Y2 Y3 X12 X13] +
(1.628853243590582e-06+0j) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10] +
(0.01071550846979675+0j) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.010263414868158547+0j) [Y2 Z3 X4 X10 Z11 Y12] +
(-3.887051672440412e-06+0j) [Y2 Z3 Y4] +
(-0.005143391768825092+0j) [Y2 Z3 Y4 X5 Z6 X7] +
(-0.00984174924696267+0j) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.988511706224744e-06+0j) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.005143391768825092+0j) [Y2 Z3 Y4 Y5 Z6 Y7] +
(-0.00984174924696267+0j) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.988511706224744e-06+0j) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.764994119155435e-07+0j) [Y2 Z3 Y4 Z5] +
(4.5371780962314595e-08+0j) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10] +
(4.205548411218745e-05+0j) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(1.6893489515400427e-06+0j) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10] +
(0.010757563953908937+0j) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.5935343916139047e-07+0j) [Y2 Z3 Y4 Z6] +
(3.211842019208348e-06+0j) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11] +
(0.019299560579363776+0j) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.211842019208348e-06+0j) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11] +
(0.019299560579363776+0j) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.1954890098126153e-06+0j) [Y2 Z3 Y4 Z7] +
(2.1868423782137136e-07+0j) [Y2 Z3 Y4 Z8] +
(-5.770052993949372e-07+0j) [Y2 Z3 Y4 Z9] +
(0.005324835234221695+0j) [Y2 Z3 Y4 X10 Z11 X12] +
(0.015588250102380243+0j) [Y2 Z3 Y4 Y10 Z11 Y12] +
(-3.1586564320842818e-06+0j) [Y2 Z3 Y4 Z10] +
(0.02435307767806904+0j) [Y2 Z3 Y4 X11 Z12 X13] +
(0.02435307767806904+0j) [Y2 Z3 Y4 Y11 Z12 Y13] +
(-7.801707500746608e-06+0j) [Y2 Z3 Y4 Z11] +
(3.5390541846291337e-06+0j) [Y2 Z3 Y4 Z12] +
(8.814937306924811e-06+0j) [Y2 Z3 Y4 Z13] +
(1.454842448974006e-06+0j) [Y2 Z3 Z4 X5 X6 Y7] +
(3.1513463112588878e-06+0j) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11] +
(0.019257505095251592+0j) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.5224930676683058e-06+0j) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10] +
(-0.008541996625454839+0j) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(7.956895372163085e-07+0j) [Y2 Z3 Z4 X5 X8 Y9] +
(4.643051068662326e-06+0j) [Y2 Z3 Z4 X5 X10 Y11] +
(0.019028242443847345+0j) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13] +
(-0.008764827575688795+0j) [Y2 Z3 Z4 X5 X11 Y12] +
(-5.275883122295677e-06+0j) [Y2 Z3 Z4 X5 X12 Y13] +
(-1.454842448974006e-06+0j) [Y2 Z3 Z4 Y5 X6 X7] +
(-3.1513463112588878e-06+0j) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11] +
(-0.019257505095251592+0j) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.5224930676683058e-06+0j) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10] +
(-0.008541996625454839+0j) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-7.956895372163085e-07+0j) [Y2 Z3 Z4 Y5 X8 X9] +
(-4.643051068662326e-06+0j) [Y2 Z3 Z4 Y5 X10 X11] +
(-0.019028242443847345+0j) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13] +
(-0.008764827575688795+0j) [Y2 Z3 Z4 Y5 Y11 Y12] +
(5.275883122295677e-06+0j) [Y2 Z3 Z4 Y5 X12 X13] +
(-4.5614471796376247e-07+0j) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12] +
(-0.1213327691104234+0j) [Y2 Z3 Z4 Z5 Y6] +
(-0.00846997879102387+0j) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(2.6863815468884714e-07+0j) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.00846997879102387+0j) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(2.6863815468884714e-07+0j) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.02599617759802128+0j) [Y2 Z3 Z4 Z5 Y6 Z7] +
(-0.005805188989826866+0j) [Y2 Z3 Z4 Z5 Y6 Z8] +
(-0.017561202409646134+0j) [Y2 Z3 Z4 Z5 Y6 Z9] +
(-3.4273231086827337e-07+0j) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12] +
(-7.988770288320358e-07+0j) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12] +
(-0.0008145313270956425+0j) [Y2 Z3 Z4 Z5 Y6 Z10] +
(2.7455184005914887e-06+0j) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13] +
(2.7455184005914887e-06+0j) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13] +
(0.014411099430130955+0j) [Y2 Z3 Z4 Z5 Y6 Z11] +
(0.0006650070219499201+0j) [Y2 Z3 Z4 Z5 Y6 Z12] +
(-0.0034937903598901516+0j) [Y2 Z3 Z4 Z5 Y6 Z13] +
(0.01175601341981927+0j) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9] +
(-0.015225630757226598+0j) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11] +
(3.0882507114597616e-06+0j) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(-3.544395429423524e-06+0j) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12] +
(0.004158797381840071+0j) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13] +
(-0.01175601341981927+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9] +
(0.015225630757226598+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11] +
(-3.0882507114597616e-06+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(-3.544395429423524e-06+0j) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12] +
(-0.004158797381840071+0j) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13] +
(0.014603704729162123+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(-2.8742990714502686e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.014603704729162123+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(-2.8742990714502686e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(-0.281642577670229+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-1.3002946562663052e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(-1.3002946562663052e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(-0.02428211735469295+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(-0.019538050311314767+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(-0.017091553155898886+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(-0.002446497155415882+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(0.002446497155415882+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(5.7759505273980086e-05+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(2.883676576139899e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(5.146496327675988e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(3.846201671409681e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(-0.03935916802205314+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10] +
(7.97982579361234e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(-0.02475546329289101+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10] +
(5.105526722162071e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(-0.021433810721601006+0j) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10] +
(5.159350502016612e-06+0j) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(-0.029903789512624866+0j) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10] +
(5.427988656705459e-06+0j) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.0016638798784908073+0j) [Y2 Z3 Z4 Y6] +
(-0.01888903030494289+0j) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10] +
(2.94735601187036e-06+0j) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.0034795118903342844+0j) [Y2 Z3 Z5 Y6] +
(-0.02873077955190556+0j) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10] +
(5.935867718095103e-06+0j) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.6021167407252466e-06+0j) [Y2 Y4] +
(0.0004956762314917412+0j) [Y2 Z4 Z5 Y6] +
(-0.03560837898831261+0j) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10] +
(7.253273348318091e-06+0j) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.6538942226831694+0j) [Z2] +
(1.6021167407252466e-06+0j) [Z2 X3 Z4 X5] +
(0.0004956762314917412+0j) [Z2 X3 Z4 Z5 Z6 X7] +
(-0.03560837898831261+0j) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(7.253273348318091e-06+0j) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.6021167407252466e-06+0j) [Z2 Y3 Z4 Y5] +
(0.0004956762314917412+0j) [Z2 Y3 Z4 Z5 Z6 Y7] +
(-0.03560837898831261+0j) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(7.253273348318091e-06+0j) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.1818908579075138+0j) [Z2 Z3] +
(-9.509249751380598e-07+0j) [Z2 X4 Z5 X6] +
(-4.728843147417908e-06+0j) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.024591860883829964+0j) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-9.509249751380598e-07+0j) [Z2 Y4 Z5 Y6] +
(-4.728843147417908e-06+0j) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.024591860883829964+0j) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.12495807739503227+0j) [Z2 Z4] +
(-1.1708301369988116e-06+0j) [Z2 X5 Z6 X7] +
(-7.089799467802333e-06+0j) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.034903343373661744+0j) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.1708301369988116e-06+0j) [Z2 Y5 Z6 Y7] +
(-7.089799467802333e-06+0j) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.034903343373661744+0j) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.16079764534838575+0j) [Z2 Z5] +
(0.019020423173039924+0j) [Z2 X6 Z7 Z8 Z9 X10] +
(-2.103215604843574e-06+0j) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.019020423173039924+0j) [Z2 Y6 Z7 Z8 Z9 Y10] +
(-2.103215604843574e-06+0j) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.13739104762683246+0j) [Z2 Z6] +
(0.024389082531149433+0j) [Z2 X7 Z8 Z9 Z10 X11] +
(-2.011122098419903e-06+0j) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.024389082531149433+0j) [Z2 Y7 Z8 Z9 Z10 Y11] +
(-2.011122098419903e-06+0j) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.1685348656157996+0j) [Z2 Z7] +
(0.150714081210083+0j) [Z2 Z8] +
(0.18690820476912562+0j) [Z2 Z9] +
(-1.063228342433968e-06+0j) [Z2 X10 Z11 X12] +
(-1.063228342433968e-06+0j) [Z2 Y10 Z11 Y12] +
(0.1279950249246842+0j) [Z2 Z10] +
(1.1094407591286987e-06+0j) [Z2 X11 Z12 X13] +
(1.1094407591286987e-06+0j) [Z2 Y11 Z12 Y13] +
(0.15337968243314168+0j) [Z2 Z11] +
(0.14011289865354834+0j) [Z2 Z12] +
(0.1556901067175248+0j) [Z2 Z13] +
(0.005143391768825092+0j) [X3 X4 Y5 Y6] +
(0.00984174924696267+0j) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10] +
(-2.988511706224744e-06+0j) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.454842448974006e-06+0j) [X3 X4 X6 X7] +
(-1.5224930676683058e-06+0j) [X3 X4 X6 Z7 Z8 Z9 Z10 X11] +
(-0.008541996625454839+0j) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-3.1513463112588878e-06+0j) [X3 X4 Y7 Z8 Z9 Y10] +
(-0.019257505095251592+0j) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12] +
(-7.956895372163085e-07+0j) [X3 X4 X8 X9] +
(-4.643051068662327e-06+0j) [X3 X4 X10 X11] +
(-0.008764827575688795+0j) [X3 X4 X10 Z11 Z12 X13] +
(-0.019028242443847345+0j) [X3 X4 Y11 Y12] +
(5.275883122295677e-06+0j) [X3 X4 X12 X13] +
(-0.005143391768825092+0j) [X3 Y4 Y5 X6] +
(-0.00984174924696267+0j) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10] +
(2.988511706224744e-06+0j) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.454842448974006e-06+0j) [X3 Y4 Y6 X7] +
(-1.5224930676683058e-06+0j) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11] +
(-0.008541996625454839+0j) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(3.1513463112588878e-06+0j) [X3 Y4 Y7 Z8 Z9 X10] +
(0.019257505095251592+0j) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12] +
(-7.956895372163085e-07+0j) [X3 Y4 Y8 X9] +
(-4.643051068662327e-06+0j) [X3 Y4 Y10 X11] +
(-0.008764827575688795+0j) [X3 Y4 Y10 Z11 Z12 X13] +
(0.019028242443847345+0j) [X3 Y4 Y11 X12] +
(5.275883122295677e-06+0j) [X3 Y4 Y12 X13] +
(-3.887051672440412e-06+0j) [X3 Z4 X5] +
(3.211842019208348e-06+0j) [X3 Z4 X5 X6 Z7 Z8 Z9 X10] +
(0.019299560579363776+0j) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.211842019208348e-06+0j) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10] +
(0.019299560579363776+0j) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.1954890098126153e-06+0j) [X3 Z4 X5 Z6] +
(1.6893489515400427e-06+0j) [X3 Z4 X5 X7 Z8 Z9 Z10 X11] +
(0.010757563953908937+0j) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.5371780962314595e-08+0j) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11] +
(4.205548411218745e-05+0j) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.5935343916139047e-07+0j) [X3 Z4 X5 Z7] +
(-5.770052993949372e-07+0j) [X3 Z4 X5 Z8] +
(2.1868423782137136e-07+0j) [X3 Z4 X5 Z9] +
(0.02435307767806904+0j) [X3 Z4 X5 X10 Z11 X12] +
(0.02435307767806904+0j) [X3 Z4 X5 Y10 Z11 Y12] +
(-7.801707500746608e-06+0j) [X3 Z4 X5 Z10] +
(0.015588250102380243+0j) [X3 Z4 X5 X11 Z12 X13] +
(0.005324835234221695+0j) [X3 Z4 X5 Y11 Z12 Y13] +
(-3.1586564320842818e-06+0j) [X3 Z4 X5 Z11] +
(8.814937306924811e-06+0j) [X3 Z4 X5 Z12] +
(3.5390541846291337e-06+0j) [X3 Z4 X5 Z13] +
(1.628853243590582e-06+0j) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11] +
(0.01071550846979675+0j) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.010263414868158547+0j) [X3 Z4 Y5 Y11 Z12 X13] +
(0.00846997879102387+0j) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10] +
(-2.6863815468884714e-07+0j) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.011756013419819269+0j) [X3 Z4 Z5 X6 X8 X9] +
(0.015225630757226598+0j) [X3 Z4 Z5 X6 X10 X11] +
(-3.544395429423524e-06+0j) [X3 Z4 Z5 X6 X10 Z11 Z12 X13] +
(-3.0882507114597616e-06+0j) [X3 Z4 Z5 X6 Y11 Y12] +
(-0.004158797381840071+0j) [X3 Z4 Z5 X6 X12 X13] +
(-0.00846997879102387+0j) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10] +
(2.6863815468884714e-07+0j) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-0.011756013419819269+0j) [X3 Z4 Z5 Y6 Y8 X9] +
(0.015225630757226598+0j) [X3 Z4 Z5 Y6 Y10 X11] +
(-3.544395429423524e-06+0j) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13] +
(3.0882507114597616e-06+0j) [X3 Z4 Z5 Y6 Y11 X12] +
(-0.004158797381840071+0j) [X3 Z4 Z5 Y6 Y12 X13] +
(-0.12133276911042341+0j) [X3 Z4 Z5 Z6 X7] +
(-0.017561202409646134+0j) [X3 Z4 Z5 Z6 X7 Z8] +
(-0.005805188989826866+0j) [X3 Z4 Z5 Z6 X7 Z9] +
(2.7455184005914887e-06+0j) [X3 Z4 Z5 Z6 X7 X10 Z11 X12] +
(2.7455184005914887e-06+0j) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12] +
(0.014411099430130955+0j) [X3 Z4 Z5 Z6 X7 Z10] +
(-7.988770288320358e-07+0j) [X3 Z4 Z5 Z6 X7 X11 Z12 X13] +
(-3.4273231086827337e-07+0j) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13] +
(-0.0008145313270956425+0j) [X3 Z4 Z5 Z6 X7 Z11] +
(-0.0034937903598901516+0j) [X3 Z4 Z5 Z6 X7 Z12] +
(0.0006650070219499201+0j) [X3 Z4 Z5 Z6 X7 Z13] +
(-4.5614471796376247e-07+0j) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13] +
(-0.014603704729162122+0j) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10] +
(2.8742990714502686e-06+0j) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(0.014603704729162122+0j) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10] +
(-2.8742990714502686e-06+0j) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(1.3002946562663052e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12] +
(0.002446497155415882+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-1.3002946562663052e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12] +
(0.002446497155415882+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(-0.2816425776702291+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.017091553155898886+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(-0.019538050311314767+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(5.775950527398006e-05+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(2.883676576139899e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(3.846201671409681e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(-0.024282117354692948+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11] +
(5.146496327675988e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(-0.02475546329289101+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11] +
(5.105526722162071e-06+0j) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(-0.03935916802205314+0j) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11] +
(7.97982579361234e-06+0j) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(-0.029903789512624866+0j) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11] +
(5.427988656705459e-06+0j) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.02599617759802128+0j) [X3 Z4 Z5 X7] +
(-0.021433810721601006+0j) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11] +
(5.159350502016612e-06+0j) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.0034795118903342844+0j) [X3 Z4 Z6 X7] +
(-0.02873077955190556+0j) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11] +
(5.935867718095103e-06+0j) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-7.764994119155435e-07+0j) [X3 X5] +
(0.0016638798784908073+0j) [X3 Z5 Z6 X7] +
(-0.01888903030494289+0j) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11] +
(2.94735601187036e-06+0j) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.005143391768825092+0j) [Y3 X4 X5 Y6] +
(-0.00984174924696267+0j) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10] +
(2.988511706224744e-06+0j) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.454842448974006e-06+0j) [Y3 X4 X6 Y7] +
(-1.5224930676683058e-06+0j) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11] +
(-0.008541996625454839+0j) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(3.1513463112588878e-06+0j) [Y3 X4 X7 Z8 Z9 Y10] +
(0.019257505095251592+0j) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12] +
(-7.956895372163085e-07+0j) [Y3 X4 X8 Y9] +
(-4.643051068662327e-06+0j) [Y3 X4 X10 Y11] +
(-0.008764827575688795+0j) [Y3 X4 X10 Z11 Z12 Y13] +
(0.019028242443847345+0j) [Y3 X4 X11 Y12] +
(5.275883122295677e-06+0j) [Y3 X4 X12 Y13] +
(0.005143391768825092+0j) [Y3 Y4 X5 X6] +
(0.00984174924696267+0j) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10] +
(-2.988511706224744e-06+0j) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.454842448974006e-06+0j) [Y3 Y4 Y6 Y7] +
(-1.5224930676683058e-06+0j) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11] +
(-0.008541996625454839+0j) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-3.1513463112588878e-06+0j) [Y3 Y4 X7 Z8 Z9 X10] +
(-0.019257505095251592+0j) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12] +
(-7.956895372163085e-07+0j) [Y3 Y4 Y8 Y9] +
(-4.643051068662327e-06+0j) [Y3 Y4 Y10 Y11] +
(-0.008764827575688795+0j) [Y3 Y4 Y10 Z11 Z12 Y13] +
(-0.019028242443847345+0j) [Y3 Y4 X11 X12] +
(5.275883122295677e-06+0j) [Y3 Y4 Y12 Y13] +
(1.628853243590582e-06+0j) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11] +
(0.01071550846979675+0j) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.010263414868158547+0j) [Y3 Z4 X5 X11 Z12 Y13] +
(-3.887051672440412e-06+0j) [Y3 Z4 Y5] +
(3.211842019208348e-06+0j) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10] +
(0.019299560579363776+0j) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(3.211842019208348e-06+0j) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10] +
(0.019299560579363776+0j) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-1.1954890098126153e-06+0j) [Y3 Z4 Y5 Z6] +
(4.5371780962314595e-08+0j) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11] +
(4.205548411218745e-05+0j) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(1.6893489515400427e-06+0j) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11] +
(0.010757563953908937+0j) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.5935343916139047e-07+0j) [Y3 Z4 Y5 Z7] +
(-5.770052993949372e-07+0j) [Y3 Z4 Y5 Z8] +
(2.1868423782137136e-07+0j) [Y3 Z4 Y5 Z9] +
(0.02435307767806904+0j) [Y3 Z4 Y5 X10 Z11 X12] +
(0.02435307767806904+0j) [Y3 Z4 Y5 Y10 Z11 Y12] +
(-7.801707500746608e-06+0j) [Y3 Z4 Y5 Z10] +
(0.005324835234221695+0j) [Y3 Z4 Y5 X11 Z12 X13] +
(0.015588250102380243+0j) [Y3 Z4 Y5 Y11 Z12 Y13] +
(-3.1586564320842818e-06+0j) [Y3 Z4 Y5 Z11] +
(8.814937306924811e-06+0j) [Y3 Z4 Y5 Z12] +
(3.5390541846291337e-06+0j) [Y3 Z4 Y5 Z13] +
(-0.00846997879102387+0j) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10] +
(2.6863815468884714e-07+0j) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-0.011756013419819269+0j) [Y3 Z4 Z5 X6 X8 Y9] +
(0.015225630757226598+0j) [Y3 Z4 Z5 X6 X10 Y11] +
(-3.544395429423524e-06+0j) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13] +
(3.0882507114597616e-06+0j) [Y3 Z4 Z5 X6 X11 Y12] +
(-0.004158797381840071+0j) [Y3 Z4 Z5 X6 X12 Y13] +
(0.00846997879102387+0j) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10] +
(-2.6863815468884714e-07+0j) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-0.011756013419819269+0j) [Y3 Z4 Z5 Y6 Y8 Y9] +
(0.015225630757226598+0j) [Y3 Z4 Z5 Y6 Y10 Y11] +
(-3.544395429423524e-06+0j) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13] +
(-3.0882507114597616e-06+0j) [Y3 Z4 Z5 Y6 X11 X12] +
(-0.004158797381840071+0j) [Y3 Z4 Z5 Y6 Y12 Y13] +
(-4.5614471796376247e-07+0j) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13] +
(-0.12133276911042341+0j) [Y3 Z4 Z5 Z6 Y7] +
(-0.017561202409646134+0j) [Y3 Z4 Z5 Z6 Y7 Z8] +
(-0.005805188989826866+0j) [Y3 Z4 Z5 Z6 Y7 Z9] +
(2.7455184005914887e-06+0j) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12] +
(2.7455184005914887e-06+0j) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12] +
(0.014411099430130955+0j) [Y3 Z4 Z5 Z6 Y7 Z10] +
(-3.4273231086827337e-07+0j) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13] +
(-7.988770288320358e-07+0j) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13] +
(-0.0008145313270956425+0j) [Y3 Z4 Z5 Z6 Y7 Z11] +
(-0.0034937903598901516+0j) [Y3 Z4 Z5 Z6 Y7 Z12] +
(0.0006650070219499201+0j) [Y3 Z4 Z5 Z6 Y7 Z13] +
(0.014603704729162122+0j) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10] +
(-2.8742990714502686e-06+0j) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-0.014603704729162122+0j) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10] +
(2.8742990714502686e-06+0j) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-1.3002946562663052e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12] +
(0.002446497155415882+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(1.3002946562663052e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12] +
(0.002446497155415882+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(-0.2816425776702291+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.017091553155898886+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(-0.019538050311314767+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(5.775950527398006e-05+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(2.883676576139899e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(3.846201671409681e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(-0.024282117354692948+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11] +
(5.146496327675988e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(-0.02475546329289101+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11] +
(5.105526722162071e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(-0.03935916802205314+0j) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11] +
(7.97982579361234e-06+0j) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(-0.029903789512624866+0j) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11] +
(5.427988656705459e-06+0j) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.02599617759802128+0j) [Y3 Z4 Z5 Y7] +
(-0.021433810721601006+0j) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11] +
(5.159350502016612e-06+0j) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.0034795118903342844+0j) [Y3 Z4 Z6 Y7] +
(-0.02873077955190556+0j) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11] +
(5.935867718095103e-06+0j) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-7.764994119155435e-07+0j) [Y3 Y5] +
(0.0016638798784908073+0j) [Y3 Z5 Z6 Y7] +
(-0.01888903030494289+0j) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11] +
(2.94735601187036e-06+0j) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.6538942226831697+0j) [Z3] +
(-1.1708301369988116e-06+0j) [Z3 X4 Z5 X6] +
(-7.089799467802333e-06+0j) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.034903343373661744+0j) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-1.1708301369988116e-06+0j) [Z3 Y4 Z5 Y6] +
(-7.089799467802333e-06+0j) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.034903343373661744+0j) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.16079764534838575+0j) [Z3 Z4] +
(-9.509249751380598e-07+0j) [Z3 X5 Z6 X7] +
(-4.728843147417908e-06+0j) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.024591860883829964+0j) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-9.509249751380598e-07+0j) [Z3 Y5 Z6 Y7] +
(-4.728843147417908e-06+0j) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.024591860883829964+0j) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.12495807739503227+0j) [Z3 Z5] +
(0.024389082531149433+0j) [Z3 X6 Z7 Z8 Z9 X10] +
(-2.011122098419903e-06+0j) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.024389082531149433+0j) [Z3 Y6 Z7 Z8 Z9 Y10] +
(-2.011122098419903e-06+0j) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.1685348656157996+0j) [Z3 Z6] +
(0.019020423173039924+0j) [Z3 X7 Z8 Z9 Z10 X11] +
(-2.103215604843574e-06+0j) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.019020423173039924+0j) [Z3 Y7 Z8 Z9 Z10 Y11] +
(-2.103215604843574e-06+0j) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.13739104762683246+0j) [Z3 Z7] +
(0.18690820476912562+0j) [Z3 Z8] +
(0.150714081210083+0j) [Z3 Z9] +
(1.1094407591286987e-06+0j) [Z3 X10 Z11 X12] +
(1.1094407591286987e-06+0j) [Z3 Y10 Z11 Y12] +
(0.15337968243314168+0j) [Z3 Z10] +
(-1.063228342433968e-06+0j) [Z3 X11 Z12 X13] +
(-1.063228342433968e-06+0j) [Z3 Y11 Z12 Y13] +
(0.1279950249246842+0j) [Z3 Z11] +
(0.1556901067175248+0j) [Z3 Z12] +
(0.14011289865354834+0j) [Z3 Z13] +
(-0.011982389010247941+0j) [X4 X5 Y6 Y7] +
(-0.007306759928832978+0j) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11] +
(-2.8882935947807045e-07+0j) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.007306759928832979+0j) [X4 X5 X7 Z8 Z9 X10] +
(-2.8882935947807045e-07+0j) [X4 X5 X7 Z8 Z9 Z10 Z11 X12] +
(-0.0071569349198569426+0j) [X4 X5 Y8 Y9] +
(-0.017680067952481598+0j) [X4 X5 Y10 Y11] +
(-3.6945132947056683e-06+0j) [X4 X5 Y10 Z11 Z12 Y13] +
(-3.6945132947056688e-06+0j) [X4 X5 X11 X12] +
(-0.03831467029480392+0j) [X4 X5 Y12 Y13] +
(0.011982389010247941+0j) [X4 Y5 Y6 X7] +
(0.007306759928832978+0j) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11] +
(2.8882935947807045e-07+0j) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.007306759928832979+0j) [X4 Y5 Y7 Z8 Z9 X10] +
(-2.8882935947807045e-07+0j) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12] +
(0.0071569349198569426+0j) [X4 Y5 Y8 X9] +
(0.017680067952481598+0j) [X4 Y5 Y10 X11] +
(3.6945132947056683e-06+0j) [X4 Y5 Y10 Z11 Z12 X13] +
(-3.6945132947056688e-06+0j) [X4 Y5 Y11 X12] +
(0.03831467029480392+0j) [X4 Y5 Y12 X13] +
(-1.2260484988791038e-05+0j) [X4 Z5 X6] +
(-1.2283337824109085e-06+0j) [X4 Z5 X6 X7 Z8 Z9 Z10 X11] +
(0.00024636437569584726+0j) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2283337824109085e-06+0j) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11] +
(0.00024636437569584726+0j) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.8540608579131114e-06+0j) [X4 Z5 X6 Z7] +
(-1.3980449080777676e-06+0j) [X4 Z5 X6 Z8] +
(-1.8818501831512258e-06+0j) [X4 Z5 X6 Z9] +
(0.007960880725921545+0j) [X4 Z5 X6 X10 Z11 X12] +
(-0.0009298507967730161+0j) [X4 Z5 X6 Y10 Z11 Y12] +
(-1.692397828628194e-06+0j) [X4 Z5 X6 Z10] +
(-0.012215040997613932+0j) [X4 Z5 X6 X11 Z12 X13] +
(-0.012215040997613932+0j) [X4 Z5 X6 Y11 Z12 Y13] +
(4.28191388495378e-06+0j) [X4 Z5 X6 Z11] +
(-4.588855155791022e-06+0j) [X4 Z5 X6 Z13] +
(0.00889073152269456+0j) [X4 Z5 Y6 Y10 Z11 X12] +
(-4.838052750734583e-07+0j) [X4 Z5 Z6 X7 Y8 Y9] +
(5.9743117135819745e-06+0j) [X4 Z5 Z6 X7 Y10 Y11] +
(0.011285190200840917+0j) [X4 Z5 Z6 X7 Y10 Z11 Z12 Y13] +
(0.020175921723535477+0j) [X4 Z5 Z6 X7 X11 X12] +
(-4.556569218356426e-06+0j) [X4 Z5 Z6 X7 Y12 Y13] +
(4.838052750734583e-07+0j) [X4 Z5 Z6 Y7 Y8 X9] +
(-5.9743117135819745e-06+0j) [X4 Z5 Z6 Y7 Y10 X11] +
(-0.011285190200840917+0j) [X4 Z5 Z6 Y7 Y10 Z11 Z12 X13] +
(0.020175921723535477+0j) [X4 Z5 Z6 Y7 Y11 X12] +
(4.556569218356426e-06+0j) [X4 Z5 Z6 Y7 Y12 X13] +
(1.33047318870219e-06+0j) [X4 Z5 Z6 Z7 X8 X9 Z10 X11] +
(0.0059237983365613475+0j) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(1.33047318870219e-06+0j) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11] +
(0.0059237983365613475+0j) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(-6.631277928700954e-05+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10] +
(-0.016024603689179642+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(-0.016024603689179642+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(3.3343312894575205e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11] +
(-4.734622038946184e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12] +
(-9.806102775500193e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13] +
(-5.071480736554009e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(5.071480736554009e-06+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-0.3693708936615616+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(-0.02314513092952906+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-0.009612634606847185+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12] +
(-0.025637238296026824+0j) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12] +
(-8.774817864940294e-06+0j) [X4 Z5 Z6 Z7 Z8 X10] +
(-0.04764261217638309+0j) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12] +
(-7.444344676238104e-06+0j) [X4 Z5 Z6 Z7 Z9 X10] +
(-0.04171881383982175+0j) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12] +
(-6.290028433556012e-06+0j) [X4 Z5 Z6 Z8 Z9 X10] +
(-0.03956441632289347+0j) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12] +
(-7.51836221596692e-06+0j) [X4 Z5 Z7 Z8 Z9 X10] +
(-0.03931805194719762+0j) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12] +
(-5.929765814984023e-07+0j) [X4 X6] +
(-4.253224225664189e-06+0j) [X4 Z6 Z7 Z8 Z9 X10] +
(-0.02252844019601295+0j) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.011982389010247941+0j) [Y4 X5 X6 Y7] +
(0.007306759928832978+0j) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11] +
(2.8882935947807045e-07+0j) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.007306759928832979+0j) [Y4 X5 X7 Z8 Z9 Y10] +
(-2.8882935947807045e-07+0j) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12] +
(0.0071569349198569426+0j) [Y4 X5 X8 Y9] +
(0.017680067952481598+0j) [Y4 X5 X10 Y11] +
(3.6945132947056683e-06+0j) [Y4 X5 X10 Z11 Z12 Y13] +
(-3.6945132947056688e-06+0j) [Y4 X5 X11 Y12] +
(0.03831467029480392+0j) [Y4 X5 X12 Y13] +
(-0.011982389010247941+0j) [Y4 Y5 X6 X7] +
(-0.007306759928832978+0j) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11] +
(-2.8882935947807045e-07+0j) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.007306759928832979+0j) [Y4 Y5 Y7 Z8 Z9 Y10] +
(-2.8882935947807045e-07+0j) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12] +
(-0.0071569349198569426+0j) [Y4 Y5 X8 X9] +
(-0.017680067952481598+0j) [Y4 Y5 X10 X11] +
(-3.6945132947056683e-06+0j) [Y4 Y5 X10 Z11 Z12 X13] +
(-3.6945132947056688e-06+0j) [Y4 Y5 Y11 Y12] +
(-0.03831467029480392+0j) [Y4 Y5 X12 X13] +
(0.00889073152269456+0j) [Y4 Z5 X6 X10 Z11 Y12] +
(-1.2260484988791038e-05+0j) [Y4 Z5 Y6] +
(-1.2283337824109085e-06+0j) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11] +
(0.00024636437569584726+0j) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2283337824109085e-06+0j) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11] +
(0.00024636437569584726+0j) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.8540608579131114e-06+0j) [Y4 Z5 Y6 Z7] +
(-1.3980449080777676e-06+0j) [Y4 Z5 Y6 Z8] +
(-1.8818501831512258e-06+0j) [Y4 Z5 Y6 Z9] +
(-0.0009298507967730161+0j) [Y4 Z5 Y6 X10 Z11 X12] +
(0.007960880725921545+0j) [Y4 Z5 Y6 Y10 Z11 Y12] +
(-1.692397828628194e-06+0j) [Y4 Z5 Y6 Z10] +
(-0.012215040997613932+0j) [Y4 Z5 Y6 X11 Z12 X13] +
(-0.012215040997613932+0j) [Y4 Z5 Y6 Y11 Z12 Y13] +
(4.28191388495378e-06+0j) [Y4 Z5 Y6 Z11] +
(-4.588855155791022e-06+0j) [Y4 Z5 Y6 Z13] +
(4.838052750734583e-07+0j) [Y4 Z5 Z6 X7 X8 Y9] +
(-5.9743117135819745e-06+0j) [Y4 Z5 Z6 X7 X10 Y11] +
(-0.011285190200840917+0j) [Y4 Z5 Z6 X7 X10 Z11 Z12 Y13] +
(0.020175921723535477+0j) [Y4 Z5 Z6 X7 X11 Y12] +
(4.556569218356426e-06+0j) [Y4 Z5 Z6 X7 X12 Y13] +
(-4.838052750734583e-07+0j) [Y4 Z5 Z6 Y7 X8 X9] +
(5.9743117135819745e-06+0j) [Y4 Z5 Z6 Y7 X10 X11] +
(0.011285190200840917+0j) [Y4 Z5 Z6 Y7 X10 Z11 Z12 X13] +
(0.020175921723535477+0j) [Y4 Z5 Z6 Y7 Y11 Y12] +
(-4.556569218356426e-06+0j) [Y4 Z5 Z6 Y7 X12 X13] +
(1.33047318870219e-06+0j) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11] +
(0.0059237983365613475+0j) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(1.33047318870219e-06+0j) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11] +
(0.0059237983365613475+0j) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(-6.631277928700954e-05+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10] +
(-0.016024603689179642+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(-0.016024603689179642+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(3.3343312894575205e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11] +
(-4.734622038946184e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12] +
(-9.806102775500193e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13] +
(5.071480736554009e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-5.071480736554009e-06+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-0.3693708936615616+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(-0.02314513092952906+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-0.009612634606847185+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12] +
(-0.025637238296026824+0j) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12] +
(-8.774817864940294e-06+0j) [Y4 Z5 Z6 Z7 Z8 Y10] +
(-0.04764261217638309+0j) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12] +
(-7.444344676238104e-06+0j) [Y4 Z5 Z6 Z7 Z9 Y10] +
(-0.04171881383982175+0j) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12] +
(-6.290028433556012e-06+0j) [Y4 Z5 Z6 Z8 Z9 Y10] +
(-0.03956441632289347+0j) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12] +
(-7.51836221596692e-06+0j) [Y4 Z5 Z7 Z8 Z9 Y10] +
(-0.03931805194719762+0j) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12] +
(-5.929765814984023e-07+0j) [Y4 Y6] +
(-4.253224225664189e-06+0j) [Y4 Z6 Z7 Z8 Z9 Y10] +
(-0.02252844019601295+0j) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12] +
(1.2034402289145623+0j) [Z4] +
(-5.929765814984023e-07+0j) [Z4 X5 Z6 X7] +
(-4.253224225664189e-06+0j) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-0.02252844019601295+0j) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-5.929765814984023e-07+0j) [Z4 Y5 Z6 Y7] +
(-4.253224225664189e-06+0j) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-0.02252844019601295+0j) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.15755314797985678+0j) [Z4 Z5] +
(0.018266834869375498+0j) [Z4 X6 Z7 Z8 Z9 X10] +
(-1.6541174773247215e-06+0j) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.018266834869375498+0j) [Z4 Y6 Z7 Z8 Z9 Y10] +
(-1.6541174773247215e-06+0j) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.13701191674040772+0j) [Z4 Z6] +
(0.010960074940542517+0j) [Z4 X7 Z8 Z9 Z10 X11] +
(-1.942946836802792e-06+0j) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.010960074940542517+0j) [Z4 Y7 Z8 Z9 Z10 Y11] +
(-1.942946836802792e-06+0j) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.14899430575065567+0j) [Z4 Z7] +
(0.14960702684445304+0j) [Z4 Z8] +
(0.15676396176430996+0j) [Z4 Z9] +
(1.8782101247687923e-06+0j) [Z4 X10 Z11 X12] +
(1.8782101247687923e-06+0j) [Z4 Y10 Z11 Y12] +
(0.1248999091723761+0j) [Z4 Z10] +
(-1.8163031699368767e-06+0j) [Z4 X11 Z12 X13] +
(-1.8163031699368767e-06+0j) [Z4 Y11 Z12 Y13] +
(0.1425799771248577+0j) [Z4 Z11] +
(0.11383573679388677+0j) [Z4 Z12] +
(0.1521504070886907+0j) [Z4 Z13] +
(1.2283337824109085e-06+0j) [X5 X6 Y7 Z8 Z9 Y10] +
(-0.00024636437569584726+0j) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12] +
(-4.838052750734583e-07+0j) [X5 X6 X8 X9] +
(5.9743117135819745e-06+0j) [X5 X6 X10 X11] +
(0.020175921723535477+0j) [X5 X6 X10 Z11 Z12 X13] +
(0.011285190200840917+0j) [X5 X6 Y11 Y12] +
(-4.556569218356426e-06+0j) [X5 X6 X12 X13] +
(-1.2283337824109085e-06+0j) [X5 Y6 Y7 Z8 Z9 X10] +
(0.00024636437569584726+0j) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12] +
(-4.838052750734583e-07+0j) [X5 Y6 Y8 X9] +
(5.9743117135819745e-06+0j) [X5 Y6 Y10 X11] +
(0.020175921723535477+0j) [X5 Y6 Y10 Z11 Z12 X13] +
(-0.011285190200840917+0j) [X5 Y6 Y11 X12] +
(-4.556569218356426e-06+0j) [X5 Y6 Y12 X13] +
(-1.2260484988791041e-05+0j) [X5 Z6 X7] +
(-1.8818501831512258e-06+0j) [X5 Z6 X7 Z8] +
(-1.3980449080777676e-06+0j) [X5 Z6 X7 Z9] +
(-0.012215040997613932+0j) [X5 Z6 X7 X10 Z11 X12] +
(-0.012215040997613932+0j) [X5 Z6 X7 Y10 Z11 Y12] +
(4.28191388495378e-06+0j) [X5 Z6 X7 Z10] +
(0.007960880725921545+0j) [X5 Z6 X7 X11 Z12 X13] +
(-0.0009298507967730161+0j) [X5 Z6 X7 Y11 Z12 Y13] +
(-1.692397828628194e-06+0j) [X5 Z6 X7 Z11] +
(-4.588855155791022e-06+0j) [X5 Z6 X7 Z12] +
(0.00889073152269456+0j) [X5 Z6 Y7 Y11 Z12 X13] +
(-1.33047318870219e-06+0j) [X5 Z6 Z7 X8 Y9 Y10] +
(-0.0059237983365613475+0j) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12] +
(1.33047318870219e-06+0j) [X5 Z6 Z7 Y8 Y9 X10] +
(0.0059237983365613475+0j) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12] +
(0.016024603689179642+0j) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12] +
(-5.071480736554009e-06+0j) [X5 Z6 Z7 Z8 Z9 X10 X12 X13] +
(-0.016024603689179642+0j) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12] +
(-5.071480736554009e-06+0j) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13] +
(-6.631277928700953e-05+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11] +
(-9.806102775500193e-06+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12] +
(-4.734622038946184e-06+0j) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13] +
(-0.3693708936615616+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-0.023145130929529065+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13] +
(-0.025637238296026824+0j) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13] +
(3.3343312894575205e-06+0j) [X5 Z6 Z7 Z8 Z9 X11] +
(-0.009612634606847185+0j) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13] +
(-7.444344676238104e-06+0j) [X5 Z6 Z7 Z8 Z10 X11] +
(-0.04171881383982175+0j) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13] +
(-8.774817864940294e-06+0j) [X5 Z6 Z7 Z9 Z10 X11] +
(-0.04764261217638309+0j) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13] +
(-7.51836221596692e-06+0j) [X5 Z6 Z8 Z9 Z10 X11] +
(-0.03931805194719762+0j) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.8540608579131116e-06+0j) [X5 X7] +
(-6.290028433556012e-06+0j) [X5 Z7 Z8 Z9 Z10 X11] +
(-0.03956441632289347+0j) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13] +
(-1.2283337824109085e-06+0j) [Y5 X6 X7 Z8 Z9 Y10] +
(0.00024636437569584726+0j) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12] +
(-4.838052750734583e-07+0j) [Y5 X6 X8 Y9] +
(5.9743117135819745e-06+0j) [Y5 X6 X10 Y11] +
(0.020175921723535477+0j) [Y5 X6 X10 Z11 Z12 Y13] +
(-0.011285190200840917+0j) [Y5 X6 X11 Y12] +
(-4.556569218356426e-06+0j) [Y5 X6 X12 Y13] +
(1.2283337824109085e-06+0j) [Y5 Y6 X7 Z8 Z9 X10] +
(-0.00024636437569584726+0j) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12] +
(-4.838052750734583e-07+0j) [Y5 Y6 Y8 Y9] +
(5.9743117135819745e-06+0j) [Y5 Y6 Y10 Y11] +
(0.020175921723535477+0j) [Y5 Y6 Y10 Z11 Z12 Y13] +
(0.011285190200840917+0j) [Y5 Y6 X11 X12] +
(-4.556569218356426e-06+0j) [Y5 Y6 Y12 Y13] +
(0.00889073152269456+0j) [Y5 Z6 X7 X11 Z12 Y13] +
(-1.2260484988791041e-05+0j) [Y5 Z6 Y7] +
(-1.8818501831512258e-06+0j) [Y5 Z6 Y7 Z8] +
(-1.3980449080777676e-06+0j) [Y5 Z6 Y7 Z9] +
(-0.012215040997613932+0j) [Y5 Z6 Y7 X10 Z11 X12] +
(-0.012215040997613932+0j) [Y5 Z6 Y7 Y10 Z11 Y12] +
(4.28191388495378e-06+0j) [Y5 Z6 Y7 Z10] +
(-0.0009298507967730161+0j) [Y5 Z6 Y7 X11 Z12 X13] +
(0.007960880725921545+0j) [Y5 Z6 Y7 Y11 Z12 Y13] +
(-1.692397828628194e-06+0j) [Y5 Z6 Y7 Z11] +
(-4.588855155791022e-06+0j) [Y5 Z6 Y7 Z12] +
(1.33047318870219e-06+0j) [Y5 Z6 Z7 X8 X9 Y10] +
(0.0059237983365613475+0j) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12] +
(-1.33047318870219e-06+0j) [Y5 Z6 Z7 Y8 X9 X10] +
(-0.0059237983365613475+0j) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12] +
(-0.016024603689179642+0j) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12] +
(-5.071480736554009e-06+0j) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13] +
(0.016024603689179642+0j) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12] +
(-5.071480736554009e-06+0j) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13] +
(-6.631277928700953e-05+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11] +
(-9.806102775500193e-06+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12] +
(-4.734622038946184e-06+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13] +
(-0.3693708936615616+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(-0.023145130929529065+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13] +
(-0.025637238296026824+0j) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13] +
(3.3343312894575205e-06+0j) [Y5 Z6 Z7 Z8 Z9 Y11] +
(-0.009612634606847185+0j) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13] +
(-7.444344676238104e-06+0j) [Y5 Z6 Z7 Z8 Z10 Y11] +
(-0.04171881383982175+0j) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13] +
(-8.774817864940294e-06+0j) [Y5 Z6 Z7 Z9 Z10 Y11] +
(-0.04764261217638309+0j) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13] +
(-7.51836221596692e-06+0j) [Y5 Z6 Z8 Z9 Z10 Y11] +
(-0.03931805194719762+0j) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13] +
(-1.8540608579131116e-06+0j) [Y5 Y7] +
(-6.290028433556012e-06+0j) [Y5 Z7 Z8 Z9 Z10 Y11] +
(-0.03956441632289347+0j) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13] +
(1.203440228914563+0j) [Z5] +
(0.010960074940542517+0j) [Z5 X6 Z7 Z8 Z9 X10] +
(-1.942946836802792e-06+0j) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12] +
(0.010960074940542517+0j) [Z5 Y6 Z7 Z8 Z9 Y10] +
(-1.942946836802792e-06+0j) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(0.14899430575065567+0j) [Z5 Z6] +
(0.018266834869375498+0j) [Z5 X7 Z8 Z9 Z10 X11] +
(-1.6541174773247215e-06+0j) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.018266834869375498+0j) [Z5 Y7 Z8 Z9 Z10 Y11] +
(-1.6541174773247215e-06+0j) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.13701191674040772+0j) [Z5 Z7] +
(0.15676396176430996+0j) [Z5 Z8] +
(0.14960702684445304+0j) [Z5 Z9] +
(-1.8163031699368767e-06+0j) [Z5 X10 Z11 X12] +
(-1.8163031699368767e-06+0j) [Z5 Y10 Z11 Y12] +
(0.1425799771248577+0j) [Z5 Z10] +
(1.8782101247687923e-06+0j) [Z5 X11 Z12 X13] +
(1.8782101247687923e-06+0j) [Z5 Y11 Z12 Y13] +
(0.1248999091723761+0j) [Z5 Z11] +
(0.1521504070886907+0j) [Z5 Z12] +
(0.11383573679388677+0j) [Z5 Z13] +
(-0.013873381748426143+0j) [X6 X7 Y8 Y9] +
(-0.01782514099578641+0j) [X6 X7 Y10 Y11] +
(-1.0358477600514726e-06+0j) [X6 X7 Y10 Z11 Z12 Y13] +
(-1.0358477600514726e-06+0j) [X6 X7 X11 X12] +
(-0.017366118994651403+0j) [X6 X7 Y12 Y13] +
(0.013873381748426143+0j) [X6 Y7 Y8 X9] +
(0.01782514099578641+0j) [X6 Y7 Y10 X11] +
(1.0358477600514726e-06+0j) [X6 Y7 Y10 Z11 Z12 X13] +
(-1.0358477600514726e-06+0j) [X6 Y7 Y11 X12] +
(0.017366118994651403+0j) [X6 Y7 Y12 X13] +
(0.00029219862611107924+0j) [X6 Z7 X8 X9 Z10 X11] +
(-3.328139350435009e-07+0j) [X6 Z7 X8 X9 Z10 Z11 Z12 X13] +
(0.00029219862611107924+0j) [X6 Z7 X8 Y9 Z10 Y11] +
(-3.328139350435009e-07+0j) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13] +
(0.22848106564918694+0j) [X6 Z7 Z8 Z9 X10] +
(3.3131455001497706e-06+0j) [X6 Z7 Z8 Z9 X10 X11 Z12 X13] +
(3.3131455001497706e-06+0j) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13] +
(0.011307274008848067+0j) [X6 Z7 Z8 Z9 X10 Z11] +
(0.025104957138844496+0j) [X6 Z7 Z8 Z9 X10 Z12] +
(0.01054042590767148+0j) [X6 Z7 Z8 Z9 X10 Z13] +
(-0.014564531231173013+0j) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13] +
(0.014564531231173013+0j) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13] +
(-2.5950860071845866e-05+0j) [X6 Z7 Z8 Z9 Z10 Z11 X12] +
(4.1839325594337486e-06+0j) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13] +
(-6.5243738487661216e-06+0j) [X6 Z7 Z8 Z9 Z10 X12] +
(-3.2112283486163505e-06+0j) [X6 Z7 Z8 Z9 Z11 X12] +
(0.029812424517345677+0j) [X6 Z7 Z8 X10] +
(-3.2774831957911555e-06+0j) [X6 Z7 Z8 Z10 Z11 X12] +
(0.030104623143456757+0j) [X6 Z7 Z9 X10] +
(-3.6102971308346565e-06+0j) [X6 Z7 Z9 Z10 Z11 X12] +
(0.030787505389143925+0j) [X6 Z8 Z9 X10] +
(-3.7696594522402892e-06+0j) [X6 Z8 Z9 Z10 Z11 X12] +
(0.013873381748426143+0j) [Y6 X7 X8 Y9] +
(0.01782514099578641+0j) [Y6 X7 X10 Y11] +
(1.0358477600514726e-06+0j) [Y6 X7 X10 Z11 Z12 Y13] +
(-1.0358477600514726e-06+0j) [Y6 X7 X11 Y12] +
(0.017366118994651403+0j) [Y6 X7 X12 Y13] +
(-0.013873381748426143+0j) [Y6 Y7 X8 X9] +
(-0.01782514099578641+0j) [Y6 Y7 X10 X11] +
(-1.0358477600514726e-06+0j) [Y6 Y7 X10 Z11 Z12 X13] +
(-1.0358477600514726e-06+0j) [Y6 Y7 Y11 Y12] +
(-0.017366118994651403+0j) [Y6 Y7 X12 X13] +
(0.00029219862611107924+0j) [Y6 Z7 Y8 X9 Z10 X11] +
(-3.328139350435009e-07+0j) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13] +
(0.00029219862611107924+0j) [Y6 Z7 Y8 Y9 Z10 Y11] +
(-3.328139350435009e-07+0j) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13] +
(0.22848106564918694+0j) [Y6 Z7 Z8 Z9 Y10] +
(3.3131455001497706e-06+0j) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13] +
(3.3131455001497706e-06+0j) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13] +
(0.011307274008848067+0j) [Y6 Z7 Z8 Z9 Y10 Z11] +
(0.025104957138844496+0j) [Y6 Z7 Z8 Z9 Y10 Z12] +
(0.01054042590767148+0j) [Y6 Z7 Z8 Z9 Y10 Z13] +
(0.014564531231173013+0j) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13] +
(-0.014564531231173013+0j) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13] +
(-2.5950860071845866e-05+0j) [Y6 Z7 Z8 Z9 Z10 Z11 Y12] +
(4.1839325594337486e-06+0j) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13] +
(-6.5243738487661216e-06+0j) [Y6 Z7 Z8 Z9 Z10 Y12] +
(-3.2112283486163505e-06+0j) [Y6 Z7 Z8 Z9 Z11 Y12] +
(0.029812424517345677+0j) [Y6 Z7 Z8 Y10] +
(-3.2774831957911555e-06+0j) [Y6 Z7 Z8 Z10 Z11 Y12] +
(0.030104623143456757+0j) [Y6 Z7 Z9 Y10] +
(-3.6102971308346565e-06+0j) [Y6 Z7 Z9 Z10 Z11 Y12] +
(0.030787505389143925+0j) [Y6 Z8 Z9 Y10] +
(-3.7696594522402892e-06+0j) [Y6 Z8 Z9 Z10 Z11 Y12] +
(1.3096862988615423+0j) [Z6] +
(0.030787505389143925+0j) [Z6 X7 Z8 Z9 Z10 X11] +
(-3.7696594522402892e-06+0j) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13] +
(0.030787505389143925+0j) [Z6 Y7 Z8 Z9 Z10 Y11] +
(-3.7696594522402892e-06+0j) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(0.19392534613270265+0j) [Z6 Z7] +
(0.16756653265461294+0j) [Z6 Z8] +
(0.18143991440303908+0j) [Z6 Z9] +
(-1.8551201216625774e-06+0j) [Z6 X10 Z11 X12] +
(-1.8551201216625774e-06+0j) [Z6 Y10 Z11 Y12] +
(0.11952438964682696+0j) [Z6 Z10] +
(-2.8909678817140503e-06+0j) [Z6 X11 Z12 X13] +
(-2.8909678817140503e-06+0j) [Z6 Y11 Z12 Y13] +
(0.13734953064261338+0j) [Z6 Z11] +
(0.13401715261963743+0j) [Z6 Z12] +
(0.15138327161428883+0j) [Z6 Z13] +
(-0.00029219862611107924+0j) [X7 X8 Y9 Y10] +
(3.328139350435009e-07+0j) [X7 X8 Y9 Z10 Z11 Y12] +
(0.00029219862611107924+0j) [X7 Y8 Y9 X10] +
(-3.328139350435009e-07+0j) [X7 Y8 Y9 Z10 Z11 X12] +
(-3.3131455001497706e-06+0j) [X7 Z8 Z9 X10 Y11 Y12] +
(-0.014564531231173013+0j) [X7 Z8 Z9 X10 X12 X13] +
(3.3131455001497706e-06+0j) [X7 Z8 Z9 Y10 Y11 X12] +
(-0.014564531231173013+0j) [X7 Z8 Z9 Y10 Y12 X13] +
(0.22848106564918694+0j) [X7 Z8 Z9 Z10 X11] +
(0.01054042590767148+0j) [X7 Z8 Z9 Z10 X11 Z12] +
(0.025104957138844496+0j) [X7 Z8 Z9 Z10 X11 Z13] +
(-2.595086007184588e-05+0j) [X7 Z8 Z9 Z10 Z11 Z12 X13] +
(4.1839325594337486e-06+0j) [X7 Z8 Z9 Z10 Z11 X13] +
(-3.2112283486163505e-06+0j) [X7 Z8 Z9 Z10 Z12 X13] +
(0.011307274008848067+0j) [X7 Z8 Z9 X11] +
(-6.5243738487661216e-06+0j) [X7 Z8 Z9 Z11 Z12 X13] +
(0.030104623143456757+0j) [X7 Z8 Z10 X11] +
(-3.6102971308346565e-06+0j) [X7 Z8 Z10 Z11 Z12 X13] +
(0.029812424517345677+0j) [X7 Z9 Z10 X11] +
(-3.2774831957911555e-06+0j) [X7 Z9 Z10 Z11 Z12 X13] +
(0.00029219862611107924+0j) [Y7 X8 X9 Y10] +
(-3.328139350435009e-07+0j) [Y7 X8 X9 Z10 Z11 Y12] +
(-0.00029219862611107924+0j) [Y7 Y8 X9 X10] +
(3.328139350435009e-07+0j) [Y7 Y8 X9 Z10 Z11 X12] +
(3.3131455001497706e-06+0j) [Y7 Z8 Z9 X10 X11 Y12] +
(-0.014564531231173013+0j) [Y7 Z8 Z9 X10 X12 Y13] +
(-3.3131455001497706e-06+0j) [Y7 Z8 Z9 Y10 X11 X12] +
(-0.014564531231173013+0j) [Y7 Z8 Z9 Y10 Y12 Y13] +
(0.22848106564918694+0j) [Y7 Z8 Z9 Z10 Y11] +
(0.01054042590767148+0j) [Y7 Z8 Z9 Z10 Y11 Z12] +
(0.025104957138844496+0j) [Y7 Z8 Z9 Z10 Y11 Z13] +
(-2.595086007184588e-05+0j) [Y7 Z8 Z9 Z10 Z11 Z12 Y13] +
(4.1839325594337486e-06+0j) [Y7 Z8 Z9 Z10 Z11 Y13] +
(-3.2112283486163505e-06+0j) [Y7 Z8 Z9 Z10 Z12 Y13] +
(0.011307274008848067+0j) [Y7 Z8 Z9 Y11] +
(-6.5243738487661216e-06+0j) [Y7 Z8 Z9 Z11 Z12 Y13] +
(0.030104623143456757+0j) [Y7 Z8 Z10 Y11] +
(-3.6102971308346565e-06+0j) [Y7 Z8 Z10 Z11 Z12 Y13] +
(0.029812424517345677+0j) [Y7 Z9 Z10 Y11] +
(-3.2774831957911555e-06+0j) [Y7 Z9 Z10 Z11 Z12 Y13] +
(1.3096862988615423+0j) [Z7] +
(0.18143991440303908+0j) [Z7 Z8] +
(0.16756653265461294+0j) [Z7 Z9] +
(-2.8909678817140503e-06+0j) [Z7 X10 Z11 X12] +
(-2.8909678817140503e-06+0j) [Z7 Y10 Z11 Y12] +
(0.13734953064261338+0j) [Z7 Z10] +
(-1.8551201216625774e-06+0j) [Z7 X11 Z12 X13] +
(-1.8551201216625774e-06+0j) [Z7 Y11 Z12 Y13] +
(0.11952438964682696+0j) [Z7 Z11] +
(0.15138327161428883+0j) [Z7 Z12] +
(0.13401715261963743+0j) [Z7 Z13] +
(-0.009560705729135942+0j) [X8 X9 Y10 Y11] +
(6.628614201774483e-07+0j) [X8 X9 Y10 Z11 Z12 Y13] +
(6.628614201774483e-07+0j) [X8 X9 X11 X12] +
(-0.006087822480561867+0j) [X8 X9 Y12 Y13] +
(0.009560705729135942+0j) [X8 Y9 Y10 X11] +
(-6.628614201774483e-07+0j) [X8 Y9 Y10 Z11 Z12 X13] +
(6.628614201774483e-07+0j) [X8 Y9 Y11 X12] +
(0.006087822480561867+0j) [X8 Y9 Y12 X13] +
(0.009560705729135942+0j) [Y8 X9 X10 Y11] +
(-6.628614201774483e-07+0j) [Y8 X9 X10 Z11 Z12 Y13] +
(6.628614201774483e-07+0j) [Y8 X9 X11 Y12] +
(0.006087822480561867+0j) [Y8 X9 X12 Y13] +
(-0.009560705729135942+0j) [Y8 Y9 X10 X11] +
(6.628614201774483e-07+0j) [Y8 Y9 X10 Z11 Z12 X13] +
(6.628614201774483e-07+0j) [Y8 Y9 Y11 Y12] +
(-0.006087822480561867+0j) [Y8 Y9 X12 X13] +
(1.3693525634718169+0j) [Z8] +
(-1.5973171979018738e-06+0j) [Z8 X10 Z11 X12] +
(-1.5973171979018738e-06+0j) [Z8 Y10 Z11 Y12] +
(0.13766872645852585+0j) [Z8 Z10] +
(-9.344557777244257e-07+0j) [Z8 X11 Z12 X13] +
(-9.344557777244257e-07+0j) [Z8 Y11 Z12 Y13] +
(0.14722943218766177+0j) [Z8 Z11] +
(0.1497348680349694+0j) [Z8 Z12] +
(0.15582269051553127+0j) [Z8 Z13] +
(1.3693525634718173+0j) [Z9] +
(-9.344557777244257e-07+0j) [Z9 X10 Z11 X12] +
(-9.344557777244257e-07+0j) [Z9 Y10 Z11 Y12] +
(0.14722943218766177+0j) [Z9 Z10] +
(-1.5973171979018738e-06+0j) [Z9 X11 Z12 X13] +
(-1.5973171979018738e-06+0j) [Z9 Y11 Z12 Y13] +
(0.13766872645852585+0j) [Z9 Z11] +
(0.15582269051553127+0j) [Z9 Z12] +
(0.1497348680349694+0j) [Z9 Z13] +
(-0.028685183716106035+0j) [X10 X11 Y12 Y13] +
(0.028685183716106035+0j) [X10 Y11 Y12 X13] +
(-1.0722312158016717e-05+0j) [X10 Z11 X12] +
(7.954413176457233e-06+0j) [X10 Z11 X12 Z13] +
(-8.194261372454245e-06+0j) [X10 X12] +
(0.028685183716106035+0j) [Y10 X11 X12 Y13] +
(-0.028685183716106035+0j) [Y10 Y11 X12 X13] +
(-1.0722312158016717e-05+0j) [Y10 Z11 Y12] +
(7.954413176457233e-06+0j) [Y10 Z11 Y12 Z13] +
(-8.194261372454245e-06+0j) [Y10 Y12] +
(0.7829661725950187+0j) [Z10] +
(-8.194261372454245e-06+0j) [Z10 X11 Z12 X13] +
(-8.194261372454245e-06+0j) [Z10 Y11 Z12 Y13] +
(0.1492635514738893+0j) [Z10 Z11] +
(0.11270386920332233+0j) [Z10 Z12] +
(0.14138905291942835+0j) [Z10 Z13] +
(-1.0722312158016712e-05+0j) [X11 Z12 X13] +
(7.954413176457233e-06+0j) [X11 X13] +
(-1.0722312158016712e-05+0j) [Y11 Z12 Y13] +
(7.954413176457233e-06+0j) [Y11 Y13] +
(0.7829661725950172+0j) [Z11] +
(0.14138905291942835+0j) [Z11 Z12] +
(0.11270386920332233+0j) [Z11 Z13] +
(0.8084581961720488+0j) [Z12] +
(0.15435748657223666+0j) [Z12 Z13] +
(0.808458196172049+0j) [Z13]
  (-46.46390678868894) [I0]
+ (0.78296617259502) [Z10]
+ (0.7829661725950201) [Z11]
+ (0.8084581961720494) [Z13]
+ (1.203440228914564) [Z4]
+ (1.203440228914564) [Z5]
+ (1.3096862988615445) [Z7]
+ (1.3096862988615448) [Z6]
+ (1.3693525634718184) [Z8]
+ (1.3693525634718189) [Z9]
+ (1.6538942226831697) [Z2]
+ (1.65389422268317) [Z3]
+ (12.412630742111766) [Z0]
+ (12.412630742111766) [Z1]
+ (-8.194261373190638e-06) [Y10 Y12]
+ (-8.194261373190638e-06) [X10 X12]
+ (-1.8540608580450747e-06) [Y5 Y7]
+ (-1.8540608580450747e-06) [X5 X7]
+ (-7.764994120352228e-07) [Y3 Y5]
+ (-7.764994120352228e-07) [X3 X5]
+ (-5.929765815612772e-07) [Y4 Y6]
+ (-5.929765815612772e-07) [X4 X6]
+ (1.6021167406727115e-06) [Y2 Y4]
+ (1.6021167406727115e-06) [X2 X4]
+ (7.954413177118028e-06) [Y11 Y13]
+ (7.954413177118028e-06) [X11 X13]
+ (0.0032769719312316205) [Y1 Y3]
+ (0.0032769719312316205) [X1 X3]
+ (0.10433064780651377) [Y0 Y2]
+ (0.10433064780651377) [X0 X2]
+ (0.11270386920332223) [Z10 Z12]
+ (0.11270386920332223) [Z11 Z13]
+ (0.11383573679388664) [Z4 Z12]
+ (0.11383573679388664) [Z5 Z13]
+ (0.11952438964682684) [Z6 Z10]
+ (0.11952438964682684) [Z7 Z11]
+ (0.12495807739503198) [Z2 Z4]
+ (0.12495807739503198) [Z3 Z5]
+ (0.127995024924684) [Z2 Z10]
+ (0.127995024924684) [Z3 Z11]
+ (0.1340171526196372) [Z6 Z12]
+ (0.1340171526196372) [Z7 Z13]
+ (0.13701191674040752) [Z4 Z6]
+ (0.13701191674040752) [Z5 Z7]
+ (0.13739104762683213) [Z2 Z6]
+ (0.13739104762683213) [Z3 Z7]
+ (0.13766872645852582) [Z8 Z10]
+ (0.13766872645852582) [Z9 Z11]
+ (0.14011289865354803) [Z2 Z12]
+ (0.14011289865354803) [Z3 Z13]
+ (0.1413890529194282) [Z10 Z13]
+ (0.1413890529194282) [Z11 Z12]
+ (0.1425799771248576) [Z4 Z11]
+ (0.1425799771248576) [Z5 Z10]
+ (0.14722943218766182) [Z8 Z11]
+ (0.14722943218766182) [Z9 Z10]
+ (0.14899430575065548) [Z4 Z7]
+ (0.14899430575065548) [Z5 Z6]
+ (0.14960702684445296) [Z4 Z8]
+ (0.14960702684445296) [Z5 Z9]
+ (0.14973486803496935) [Z8 Z12]
+ (0.14973486803496935) [Z9 Z13]
+ (0.15071408121008267) [Z2 Z8]
+ (0.15071408121008267) [Z3 Z9]
+ (0.1513832716142886) [Z6 Z13]
+ (0.1513832716142886) [Z7 Z12]
+ (0.15215040708869054) [Z4 Z13]
+ (0.15215040708869054) [Z5 Z12]
+ (0.15337968243314143) [Z2 Z11]
+ (0.15337968243314143) [Z3 Z10]
+ (0.15435748657223655) [Z12 Z13]
+ (0.15569010671752453) [Z2 Z13]
+ (0.15569010671752453) [Z3 Z12]
+ (0.15582269051553124) [Z8 Z13]
+ (0.15582269051553124) [Z9 Z12]
+ (0.1567639617643099) [Z4 Z9]
+ (0.1567639617643099) [Z5 Z8]
+ (0.15755314797985662) [Z4 Z5]
+ (0.16079764534838548) [Z2 Z5]
+ (0.16079764534838548) [Z3 Z4]
+ (0.16756653265461272) [Z6 Z8]
+ (0.16756653265461272) [Z7 Z9]
+ (0.16853486561579925) [Z2 Z7]
+ (0.16853486561579925) [Z3 Z6]
+ (0.18143991440303883) [Z6 Z9]
+ (0.18143991440303883) [Z7 Z8]
+ (0.18189085790751316) [Z2 Z3]
+ (0.18690820476912523) [Z2 Z9]
+ (0.18690820476912523) [Z3 Z8]
+ (0.1929972393536426) [Z0 Z10]
+ (0.1929972393536426) [Z1 Z11]
+ (0.1939253461327021) [Z6 Z7]
+ (0.19661770890342134) [Z0 Z4]
+ (0.19661770890342134) [Z1 Z5]
+ (0.19936354537360812) [Z0 Z5]
+ (0.19936354537360812) [Z1 Z4]
+ (0.2007286646044179) [Z0 Z11]
+ (0.2007286646044179) [Z1 Z10]
+ (0.2110265984979153) [Z0 Z12]
+ (0.2110265984979153) [Z1 Z13]
+ (0.21631037498631828) [Z0 Z13]
+ (0.21631037498631828) [Z1 Z12]
+ (0.23671080783830378) [Z0 Z2]
+ (0.23671080783830378) [Z1 Z3]
+ (0.24164663936017208) [Z0 Z6]
+ (0.24164663936017208) [Z1 Z7]
+ (0.24853483371314267) [Z0 Z7]
+ (0.24853483371314267) [Z1 Z6]
+ (0.2512944567459164) [Z0 Z3]
+ (0.2512944567459164) [Z1 Z2]
+ (0.27232518306605674) [Z0 Z8]
+ (0.27232518306605674) [Z1 Z9]
+ (0.278834544267234) [Z0 Z9]
+ (0.278834544267234) [Z1 Z8]
+ (1.1861763734860495) [Z0 Z1]
+ (-1.2260484989138506e-05) [Y4 Z5 Y6]
+ (-1.2260484989138506e-05) [X4 Z5 X6]
+ (-1.2260484989138505e-05) [Y5 Z6 Y7]
+ (-1.2260484989138505e-05) [X5 Z6 X7]
+ (-1.0722312158017817e-05) [Y11 Z12 Y13]
+ (-1.0722312158017817e-05) [X11 Z12 X13]
+ (-1.0722312158017815e-05) [Y10 Z11 Y12]
+ (-1.0722312158017815e-05) [X10 Z11 X12]
+ (-3.887051674892871e-06) [Y2 Z3 Y4]
+ (-3.887051674892871e-06) [X2 Z3 X4]
+ (-3.88705167489287e-06) [Y3 Z4 Y5]
+ (-3.88705167489287e-06) [X3 Z4 X5]
+ (0.1250703257977187) [Y1 Z2 Y3]
+ (0.1250703257977187) [X1 Z2 X3]
+ (0.12507032579771873) [Y0 Z1 Y2]
+ (0.12507032579771873) [X0 Z1 X2]
+ (-0.038314670294803906) [Y4 Y5 X12 X13]
+ (-0.038314670294803906) [X4 X5 Y12 Y13]
+ (-0.03583956795335347) [Y2 Y3 X4 X5]
+ (-0.03583956795335347) [X2 X3 Y4 Y5]
+ (-0.031143817988967128) [Y2 Y3 X6 X7]
+ (-0.031143817988967128) [X2 X3 Y6 Y7]
+ (-0.028685183716105952) [Y10 Y11 X12 X13]
+ (-0.028685183716105952) [X10 X11 Y12 Y13]
+ (-0.025996177598021232) [Y3 Z4 Z5 Y7]
+ (-0.025996177598021232) [X3 Z4 Z5 X7]
+ (-0.02538465750845742) [Y2 Y3 X10 X11]
+ (-0.02538465750845742) [X2 X3 Y10 Y11]
+ (-0.019028242443847324) [Y3 Y4 X11 X12]
+ (-0.019028242443847324) [X3 X4 Y11 Y12]
+ (-0.017825140995786446) [Y6 Y7 X10 X11]
+ (-0.017825140995786446) [X6 X7 Y10 Y11]
+ (-0.017680067952481494) [Y4 Y5 X10 X11]
+ (-0.017680067952481494) [X4 X5 Y10 Y11]
+ (-0.017366118994651403) [Y6 Y7 X12 X13]
+ (-0.017366118994651403) [X6 X7 Y12 Y13]
+ (-0.015577208063976472) [Y2 Y3 X12 X13]
+ (-0.015577208063976472) [X2 X3 Y12 Y13]
+ (-0.014583648907612604) [Y0 Y1 X2 X3]
+ (-0.014583648907612604) [X0 X1 Y2 Y3]
+ (-0.013873381748426105) [Y6 Y7 X8 X9]
+ (-0.013873381748426105) [X6 X7 Y8 Y9]
+ (-0.01198238901024795) [Y4 Y5 X6 X7]
+ (-0.01198238901024795) [X4 X5 Y6 Y7]
+ (-0.009560705729135992) [Y8 Y9 X10 X11]
+ (-0.009560705729135992) [X8 X9 Y10 Y11]
+ (-0.008125251921381015) [Y1 X2 X8 Y9]
+ (-0.008125251921381015) [Y1 Y2 Y8 Y9]
+ (-0.008125251921381015) [X1 X2 X8 X9]
+ (-0.008125251921381015) [X1 Y2 Y8 X9]
+ (-0.0077314252507753155) [Y0 Y1 X10 X11]
+ (-0.0077314252507753155) [X0 X1 Y10 Y11]
+ (-0.0068881943529705645) [Y0 Y1 X6 X7]
+ (-0.0068881943529705645) [X0 X1 Y6 Y7]
+ (-0.006509361201177234) [Y0 Y1 X8 X9]
+ (-0.006509361201177234) [X0 X1 Y8 Y9]
+ (-0.006087822480561866) [Y8 Y9 X12 X13]
+ (-0.006087822480561866) [X8 X9 Y12 Y13]
+ (-0.005143391768825129) [Y3 X4 X5 Y6]
+ (-0.005143391768825129) [X3 Y4 Y5 X6]
+ (-0.0046849033881552074) [Y1 X2 X6 Y7]
+ (-0.0046849033881552074) [Y1 Y2 Y6 Y7]
+ (-0.0046849033881552074) [X1 X2 X6 X7]
+ (-0.0046849033881552074) [X1 Y2 Y6 X7]
+ (-0.004575007626639202) [Y1 X2 X12 Y13]
+ (-0.004575007626639202) [Y1 Y2 Y12 Y13]
+ (-0.004575007626639202) [X1 X2 X12 X13]
+ (-0.004575007626639202) [X1 Y2 Y12 X13]
+ (-0.0044248554494418476) [Y1 X2 X4 Y5]
+ (-0.0044248554494418476) [Y1 Y2 Y4 Y5]
+ (-0.0044248554494418476) [X1 X2 X4 X5]
+ (-0.0044248554494418476) [X1 Y2 Y4 X5]
+ (-0.003479511890334341) [Y2 Z3 Z5 Y6]
+ (-0.003479511890334341) [X2 Z3 Z5 X6]
+ (-0.003479511890334341) [Y3 Z4 Z6 Y7]
+ (-0.003479511890334341) [X3 Z4 Z6 X7]
+ (-0.0027458364701868072) [Y0 Y1 X4 X5]
+ (-0.0027458364701868072) [X0 X1 Y4 Y5]
+ (-0.001799219493663007) [Y1 X2 X10 Y11]
+ (-0.001799219493663007) [Y1 Y2 Y10 Y11]
+ (-0.001799219493663007) [X1 X2 X10 X11]
+ (-0.001799219493663007) [X1 Y2 Y10 X11]
+ (-0.0002921986261110777) [Y7 Y8 X9 X10]
+ (-0.0002921986261110777) [X7 X8 Y9 Y10]
+ (-8.194261373190638e-06) [Z10 Y11 Z12 Y13]
+ (-8.194261373190638e-06) [Z10 X11 Z12 X13]
+ (-7.801707501456448e-06) [Y2 Z3 Y4 Z11]
+ (-7.801707501456448e-06) [X2 Z3 X4 Z11]
+ (-7.801707501456448e-06) [Y3 Z4 Y5 Z10]
+ (-7.801707501456448e-06) [X3 Z4 X5 Z10]
+ (-4.64305106898011e-06) [Y3 X4 X10 Y11]
+ (-4.64305106898011e-06) [Y3 Y4 Y10 Y11]
+ (-4.64305106898011e-06) [X3 X4 X10 X11]
+ (-4.64305106898011e-06) [X3 Y4 Y10 X11]
+ (-4.588855156108793e-06) [Y4 Z5 Y6 Z13]
+ (-4.588855156108793e-06) [X4 Z5 X6 Z13]
+ (-4.588855156108793e-06) [Y5 Z6 Y7 Z12]
+ (-4.588855156108793e-06) [X5 Z6 X7 Z12]
+ (-4.556569218652214e-06) [Y5 X6 X12 Y13]
+ (-4.556569218652214e-06) [Y5 Y6 Y12 Y13]
+ (-4.556569218652214e-06) [X5 X6 X12 X13]
+ (-4.556569218652214e-06) [X5 Y6 Y12 X13]
+ (-3.6945132948867284e-06) [Y4 X5 X11 Y12]
+ (-3.6945132948867284e-06) [Y4 Y5 Y11 Y12]
+ (-3.6945132948867284e-06) [X4 X5 X11 X12]
+ (-3.6945132948867284e-06) [X4 Y5 Y11 X12]
+ (-3.3440815566618973e-06) [Z0 Y5 Z6 Y7]
+ (-3.3440815566618973e-06) [Z0 X5 Z6 X7]
+ (-3.3440815566618973e-06) [Z1 Y4 Z5 Y6]
+ (-3.3440815566618973e-06) [Z1 X4 Z5 X6]
+ (-3.1586564324763386e-06) [Y2 Z3 Y4 Z10]
+ (-3.1586564324763386e-06) [X2 Z3 X4 Z10]
+ (-3.1586564324763386e-06) [Y3 Z4 Y5 Z11]
+ (-3.1586564324763386e-06) [X3 Z4 X5 Z11]
+ (-3.0993492437584843e-06) [Z0 Y4 Z5 Y6]
+ (-3.0993492437584843e-06) [Z0 X4 Z5 X6]
+ (-3.0993492437584843e-06) [Z1 Y5 Z6 Y7]
+ (-3.0993492437584843e-06) [Z1 X5 Z6 X7]
+ (-2.8909678820250714e-06) [Z6 Y11 Z12 Y13]
+ (-2.8909678820250714e-06) [Z6 X11 Z12 X13]
+ (-2.8909678820250714e-06) [Z7 Y10 Z11 Y12]
+ (-2.8909678820250714e-06) [Z7 X10 Z11 X12]
+ (-2.17766460533369e-06) [Z0 Y10 Z11 Y12]
+ (-2.17766460533369e-06) [Z0 X10 Z11 X12]
+ (-2.17766460533369e-06) [Z1 Y11 Z12 Y13]
+ (-2.17766460533369e-06) [Z1 X11 Z12 X13]
+ (-1.881850183292263e-06) [Y4 Z5 Y6 Z9]
+ (-1.881850183292263e-06) [X4 Z5 X6 Z9]
+ (-1.881850183292263e-06) [Y5 Z6 Y7 Z8]
+ (-1.881850183292263e-06) [X5 Z6 X7 Z8]
+ (-1.8551201217853574e-06) [Z6 Y10 Z11 Y12]
+ (-1.8551201217853574e-06) [Z6 X10 Z11 X12]
+ (-1.8551201217853574e-06) [Z7 Y11 Z12 Y13]
+ (-1.8551201217853574e-06) [Z7 X11 Z12 X13]
+ (-1.8540608580450747e-06) [Y4 Z5 Y6 Z7]
+ (-1.8540608580450747e-06) [X4 Z5 X6 Z7]
+ (-1.8163031699882628e-06) [Z4 Y11 Z12 Y13]
+ (-1.8163031699882628e-06) [Z4 X11 Z12 X13]
+ (-1.8163031699882628e-06) [Z5 Y10 Z11 Y12]
+ (-1.8163031699882628e-06) [Z5 X10 Z11 X12]
+ (-1.6923978286838297e-06) [Y4 Z5 Y6 Z10]
+ (-1.6923978286838297e-06) [X4 Z5 X6 Z10]
+ (-1.6923978286838297e-06) [Y5 Z6 Y7 Z11]
+ (-1.6923978286838297e-06) [X5 Z6 X7 Z11]
+ (-1.6148794141356416e-06) [Z0 Y11 Z12 Y13]
+ (-1.6148794141356416e-06) [Z0 X11 Z12 X13]
+ (-1.6148794141356416e-06) [Z1 Y10 Z11 Y12]
+ (-1.6148794141356416e-06) [Z1 X10 Z11 X12]
+ (-1.59731719803349e-06) [Z8 Y10 Z11 Y12]
+ (-1.59731719803349e-06) [Z8 X10 Z11 X12]
+ (-1.59731719803349e-06) [Z9 Y11 Z12 Y13]
+ (-1.59731719803349e-06) [Z9 X11 Z12 X13]
+ (-1.454842449185479e-06) [Y3 X4 X6 Y7]
+ (-1.454842449185479e-06) [Y3 Y4 Y6 Y7]
+ (-1.454842449185479e-06) [X3 X4 X6 X7]
+ (-1.454842449185479e-06) [X3 Y4 Y6 X7]
+ (-1.3980449081607963e-06) [Y4 Z5 Y6 Z8]
+ (-1.3980449081607963e-06) [X4 Z5 X6 Z8]
+ (-1.3980449081607963e-06) [Y5 Z6 Y7 Z9]
+ (-1.3980449081607963e-06) [X5 Z6 X7 Z9]
+ (-1.1954890101481244e-06) [Y2 Z3 Y4 Z7]
+ (-1.1954890101481244e-06) [X2 Z3 X4 Z7]
+ (-1.1954890101481244e-06) [Y3 Z4 Y5 Z6]
+ (-1.1954890101481244e-06) [X3 Z4 X5 Z6]
+ (-1.1908508086679469e-06) [Z0 Y3 Z4 Y5]
+ (-1.1908508086679469e-06) [Z0 X3 Z4 X5]
+ (-1.1908508086679469e-06) [Z1 Y2 Z3 Y4]
+ (-1.1908508086679469e-06) [Z1 X2 Z3 X4]
+ (-1.1708301370667782e-06) [Z2 Y5 Z6 Y7]
+ (-1.1708301370667782e-06) [Z2 X5 Z6 X7]
+ (-1.1708301370667782e-06) [Z3 Y4 Z5 Y6]
+ (-1.1708301370667782e-06) [Z3 X4 Z5 X6]
+ (-1.063228342533819e-06) [Z2 Y10 Z11 Y12]
+ (-1.063228342533819e-06) [Z2 X10 Z11 X12]
+ (-1.063228342533819e-06) [Z3 Y11 Z12 Y13]
+ (-1.063228342533819e-06) [Z3 X11 Z12 X13]
+ (-1.035847760239714e-06) [Y6 X7 X11 Y12]
+ (-1.035847760239714e-06) [Y6 Y7 Y11 Y12]
+ (-1.035847760239714e-06) [X6 X7 X11 X12]
+ (-1.035847760239714e-06) [X6 Y7 Y11 X12]
+ (-9.509249751634036e-07) [Z2 Y4 Z5 Y6]
+ (-9.509249751634036e-07) [Z2 X4 Z5 X6]
+ (-9.509249751634036e-07) [Z3 Y5 Z6 Y7]
+ (-9.509249751634036e-07) [Z3 X5 Z6 X7]
+ (-9.344557777824648e-07) [Z8 Y11 Z12 Y13]
+ (-9.344557777824648e-07) [Z8 X11 Z12 X13]
+ (-9.344557777824648e-07) [Z9 Y10 Z11 Y12]
+ (-9.344557777824648e-07) [Z9 X10 Z11 X12]
+ (-8.337746757131147e-07) [Z0 Y2 Z3 Y4]
+ (-8.337746757131147e-07) [Z0 X2 Z3 X4]
+ (-8.337746757131147e-07) [Z1 Y3 Z4 Y5]
+ (-8.337746757131147e-07) [Z1 X3 Z4 X5]
+ (-7.956895373529558e-07) [Y3 X4 X8 Y9]
+ (-7.956895373529558e-07) [Y3 Y4 Y8 Y9]
+ (-7.956895373529558e-07) [X3 X4 X8 X9]
+ (-7.956895373529558e-07) [X3 Y4 Y8 X9]
+ (-7.764994120352229e-07) [Y2 Z3 Y4 Z5]
+ (-7.764994120352229e-07) [X2 Z3 X4 Z5]
+ (-5.929765815612772e-07) [Z4 Y5 Z6 Y7]
+ (-5.929765815612772e-07) [Z4 X5 Z6 X7]
+ (-5.7700529967594e-07) [Y2 Z3 Y4 Z9]
+ (-5.7700529967594e-07) [X2 Z3 X4 Z9]
+ (-5.7700529967594e-07) [Y3 Z4 Y5 Z8]
+ (-5.7700529967594e-07) [X3 Z4 X5 Z8]
+ (-5.471647745231822e-07) [Y1 Y2 X11 X12]
+ (-5.471647745231822e-07) [X1 X2 Y11 Y12]
+ (-4.838052751314664e-07) [Y5 X6 X8 Y9]
+ (-4.838052751314664e-07) [Y5 Y6 Y8 Y9]
+ (-4.838052751314664e-07) [X5 X6 X8 X9]
+ (-4.838052751314664e-07) [X5 Y6 Y8 X9]
+ (-3.57076132954832e-07) [Y0 X1 X3 Y4]
+ (-3.57076132954832e-07) [Y0 Y1 Y3 Y4]
+ (-3.57076132954832e-07) [X0 X1 X3 X4]
+ (-3.57076132954832e-07) [X0 Y1 Y3 X4]
+ (-2.447323129034128e-07) [Y0 X1 X5 Y6]
+ (-2.447323129034128e-07) [Y0 Y1 Y5 Y6]
+ (-2.447323129034128e-07) [X0 X1 X5 X6]
+ (-2.447323129034128e-07) [X0 Y1 Y5 X6]
+ (-2.199051619033749e-07) [Y2 X3 X5 Y6]
+ (-2.199051619033749e-07) [Y2 Y3 Y5 Y6]
+ (-2.199051619033749e-07) [X2 X3 X5 X6]
+ (-2.199051619033749e-07) [X2 Y3 Y5 X6]
+ (-1.933241277209509e-07) [Y1 X2 X3 Y4]
+ (-1.933241277209509e-07) [X1 Y2 Y3 X4]
+ (-1.2919694865037952e-07) [Y1 Z2 Z3 Y5]
+ (-1.2919694865037952e-07) [X1 Z2 Z3 X5]
+ (1.7379332624681835e-07) [Y0 Z1 Z3 Y4]
+ (1.7379332624681835e-07) [X0 Z1 Z3 X4]
+ (1.7379332624681835e-07) [Y1 Z2 Z4 Y5]
+ (1.7379332624681835e-07) [X1 Z2 Z4 X5]
+ (1.933241277209509e-07) [Y1 Y2 X3 X4]
+ (1.933241277209509e-07) [X1 X2 Y3 Y4]
+ (2.1868423767701593e-07) [Y2 Z3 Y4 Z8]
+ (2.1868423767701593e-07) [X2 Z3 X4 Z8]
+ (2.1868423767701593e-07) [Y3 Z4 Y5 Z9]
+ (2.1868423767701593e-07) [X3 Z4 X5 Z9]
+ (2.5935343903735456e-07) [Y2 Z3 Y4 Z6]
+ (2.5935343903735456e-07) [X2 Z3 X4 Z6]
+ (2.5935343903735456e-07) [Y3 Z4 Y5 Z7]
+ (2.5935343903735456e-07) [X3 Z4 X5 Z7]
+ (3.606071868096893e-07) [Y0 Z1 Z2 Y4]
+ (3.606071868096893e-07) [X0 Z1 Z2 X4]
+ (3.606071868096893e-07) [Y1 Z3 Z4 Y5]
+ (3.606071868096893e-07) [X1 Z3 Z4 X5]
+ (5.471647745231822e-07) [Y1 X2 X11 Y12]
+ (5.471647745231822e-07) [X1 Y2 Y11 X12]
+ (5.627851911980483e-07) [Y0 X1 X11 Y12]
+ (5.627851911980483e-07) [Y0 Y1 Y11 Y12]
+ (5.627851911980483e-07) [X0 X1 X11 X12]
+ (5.627851911980483e-07) [X0 Y1 Y11 X12]
+ (6.628614202510253e-07) [Y8 X9 X11 Y12]
+ (6.628614202510253e-07) [Y8 Y9 Y11 Y12]
+ (6.628614202510253e-07) [X8 X9 X11 X12]
+ (6.628614202510253e-07) [X8 Y9 Y11 X12]
+ (1.1094407592388426e-06) [Z2 Y11 Z12 Y13]
+ (1.1094407592388426e-06) [Z2 X11 Z12 X13]
+ (1.1094407592388426e-06) [Z3 Y10 Z11 Y12]
+ (1.1094407592388426e-06) [Z3 X10 Z11 X12]
+ (1.6021167406727115e-06) [Z2 Y3 Z4 Y5]
+ (1.6021167406727115e-06) [Z2 X3 Z4 X5]
+ (1.878210124898466e-06) [Z4 Y10 Z11 Y12]
+ (1.878210124898466e-06) [Z4 X10 Z11 X12]
+ (1.878210124898466e-06) [Z5 Y11 Z12 Y13]
+ (1.878210124898466e-06) [Z5 X11 Z12 X13]
+ (2.1726691017726613e-06) [Y2 X3 X11 Y12]
+ (2.1726691017726613e-06) [Y2 Y3 Y11 Y12]
+ (2.1726691017726613e-06) [X2 X3 X11 X12]
+ (2.1726691017726613e-06) [X2 Y3 Y11 X12]
+ (3.117447946401369e-06) [Y0 Z2 Z3 Y4]
+ (3.117447946401369e-06) [X0 Z2 Z3 X4]
+ (3.539054184814831e-06) [Y2 Z3 Y4 Z12]
+ (3.539054184814831e-06) [X2 Z3 X4 Z12]
+ (3.539054184814831e-06) [Y3 Z4 Y5 Z13]
+ (3.539054184814831e-06) [X3 Z4 X5 Z13]
+ (4.2819138853979055e-06) [Y4 Z5 Y6 Z11]
+ (4.2819138853979055e-06) [X4 Z5 X6 Z11]
+ (4.2819138853979055e-06) [Y5 Z6 Y7 Z10]
+ (4.2819138853979055e-06) [X5 Z6 X7 Z10]
+ (5.2758831227450436e-06) [Y3 X4 X12 Y13]
+ (5.2758831227450436e-06) [Y3 Y4 Y12 Y13]
+ (5.2758831227450436e-06) [X3 X4 X12 X13]
+ (5.2758831227450436e-06) [X3 Y4 Y12 X13]
+ (5.974311714081734e-06) [Y5 X6 X10 Y11]
+ (5.974311714081734e-06) [Y5 Y6 Y10 Y11]
+ (5.974311714081734e-06) [X5 X6 X10 X11]
+ (5.974311714081734e-06) [X5 Y6 Y10 X11]
+ (7.954413177118028e-06) [Y10 Z11 Y12 Z13]
+ (7.954413177118028e-06) [X10 Z11 X12 Z13]
+ (8.814937307559873e-06) [Y2 Z3 Y4 Z13]
+ (8.814937307559873e-06) [X2 Z3 X4 Z13]
+ (8.814937307559873e-06) [Y3 Z4 Y5 Z12]
+ (8.814937307559873e-06) [X3 Z4 X5 Z12]
+ (0.0002921986261110777) [Y7 X8 X9 Y10]
+ (0.0002921986261110777) [X7 Y8 Y9 X10]
+ (0.0004956762314916131) [Y2 Z4 Z5 Y6]
+ (0.0004956762314916131) [X2 Z4 Z5 X6]
+ (0.0011059037691896665) [Y0 Z1 Y2 Z5]
+ (0.0011059037691896665) [X0 Z1 X2 Z5]
+ (0.0011059037691896665) [Y1 Z2 Y3 Z4]
+ (0.0011059037691896665) [X1 Z2 X3 Z4]
+ (0.0016638798784907875) [Y2 Z3 Z4 Y6]
+ (0.0016638798784907875) [X2 Z3 Z4 X6]
+ (0.0016638798784907875) [Y3 Z5 Z6 Y7]
+ (0.0016638798784907875) [X3 Z5 Z6 X7]
+ (0.0017560707018412307) [Y0 Z1 Y2 Z11]
+ (0.0017560707018412307) [X0 Z1 X2 Z11]
+ (0.0017560707018412307) [Y1 Z2 Y3 Z10]
+ (0.0017560707018412307) [X1 Z2 X3 Z10]
+ (0.00232623062315806) [Y0 Z1 Y2 Z13]
+ (0.00232623062315806) [X0 Z1 X2 Z13]
+ (0.00232623062315806) [Y1 Z2 Y3 Z12]
+ (0.00232623062315806) [X1 Z2 X3 Z12]
+ (0.0027458364701868072) [Y0 X1 X4 Y5]
+ (0.0027458364701868072) [X0 Y1 Y4 X5]
+ (0.0029297686747510217) [Y0 Z1 Y2 Z9]
+ (0.0029297686747510217) [X0 Z1 X2 Z9]
+ (0.0029297686747510217) [Y1 Z2 Y3 Z8]
+ (0.0029297686747510217) [X1 Z2 X3 Z8]
+ (0.0032769719312316205) [Y0 Z1 Y2 Z3]
+ (0.0032769719312316205) [X0 Z1 X2 Z3]
+ (0.0033476175306661553) [Y0 Z1 Y2 Z7]
+ (0.0033476175306661553) [X0 Z1 X2 Z7]
+ (0.0033476175306661553) [Y1 Z2 Y3 Z6]
+ (0.0033476175306661553) [X1 Z2 X3 Z6]
+ (0.003555290195504238) [Y0 Z1 Y2 Z10]
+ (0.003555290195504238) [X0 Z1 X2 Z10]
+ (0.003555290195504238) [Y1 Z2 Y3 Z11]
+ (0.003555290195504238) [X1 Z2 X3 Z11]
+ (0.005143391768825129) [Y3 Y4 X5 X6]
+ (0.005143391768825129) [X3 X4 Y5 Y6]
+ (0.005530759218631514) [Y0 Z1 Y2 Z4]
+ (0.005530759218631514) [X0 Z1 X2 Z4]
+ (0.005530759218631514) [Y1 Z2 Y3 Z5]
+ (0.005530759218631514) [X1 Z2 X3 Z5]
+ (0.006087822480561866) [Y8 X9 X12 Y13]
+ (0.006087822480561866) [X8 Y9 Y12 X13]
+ (0.006509361201177234) [Y0 X1 X8 Y9]
+ (0.006509361201177234) [X0 Y1 Y8 X9]
+ (0.0068881943529705645) [Y0 X1 X6 Y7]
+ (0.0068881943529705645) [X0 Y1 Y6 X7]
+ (0.0069012382497972615) [Y0 Z1 Y2 Z12]
+ (0.0069012382497972615) [X0 Z1 X2 Z12]
+ (0.0069012382497972615) [Y1 Z2 Y3 Z13]
+ (0.0069012382497972615) [X1 Z2 X3 Z13]
+ (0.0077314252507753155) [Y0 X1 X10 Y11]
+ (0.0077314252507753155) [X0 Y1 Y10 X11]
+ (0.008032520918821362) [Y0 Z1 Y2 Z6]
+ (0.008032520918821362) [X0 Z1 X2 Z6]
+ (0.008032520918821362) [Y1 Z2 Y3 Z7]
+ (0.008032520918821362) [X1 Z2 X3 Z7]
+ (0.009560705729135992) [Y8 X9 X10 Y11]
+ (0.009560705729135992) [X8 Y9 Y10 X11]
+ (0.011055020596132035) [Y0 Z1 Y2 Z8]
+ (0.011055020596132035) [X0 Z1 X2 Z8]
+ (0.011055020596132035) [Y1 Z2 Y3 Z9]
+ (0.011055020596132035) [X1 Z2 X3 Z9]
+ (0.011307274008848275) [Y7 Z8 Z9 Y11]
+ (0.011307274008848275) [X7 Z8 Z9 X11]
+ (0.01198238901024795) [Y4 X5 X6 Y7]
+ (0.01198238901024795) [X4 Y5 Y6 X7]
+ (0.013873381748426105) [Y6 X7 X8 Y9]
+ (0.013873381748426105) [X6 Y7 Y8 X9]
+ (0.014583648907612604) [Y0 X1 X2 Y3]
+ (0.014583648907612604) [X0 Y1 Y2 X3]
+ (0.015577208063976472) [Y2 X3 X12 Y13]
+ (0.015577208063976472) [X2 Y3 Y12 X13]
+ (0.017366118994651403) [Y6 X7 X12 Y13]
+ (0.017366118994651403) [X6 Y7 Y12 X13]
+ (0.017680067952481494) [Y4 X5 X10 Y11]
+ (0.017680067952481494) [X4 Y5 Y10 X11]
+ (0.017825140995786446) [Y6 X7 X10 Y11]
+ (0.017825140995786446) [X6 Y7 Y10 X11]
+ (0.019028242443847324) [Y3 X4 X11 Y12]
+ (0.019028242443847324) [X3 Y4 Y11 X12]
+ (0.02538465750845742) [Y2 X3 X10 Y11]
+ (0.02538465750845742) [X2 Y3 Y10 X11]
+ (0.028685183716105952) [Y10 X11 X12 Y13]
+ (0.028685183716105952) [X10 Y11 Y12 X13]
+ (0.02981242451734576) [Y6 Z7 Z8 Y10]
+ (0.02981242451734576) [X6 Z7 Z8 X10]
+ (0.02981242451734576) [Y7 Z9 Z10 Y11]
+ (0.02981242451734576) [X7 Z9 Z10 X11]
+ (0.030104623143456834) [Y6 Z7 Z9 Y10]
+ (0.030104623143456834) [X6 Z7 Z9 X10]
+ (0.030104623143456834) [Y7 Z8 Z10 Y11]
+ (0.030104623143456834) [X7 Z8 Z10 X11]
+ (0.03078750538914394) [Y6 Z8 Z9 Y10]
+ (0.03078750538914394) [X6 Z8 Z9 X10]
+ (0.031143817988967128) [Y2 X3 X6 Y7]
+ (0.031143817988967128) [X2 Y3 Y6 X7]
+ (0.03583956795335347) [Y2 X3 X4 Y5]
+ (0.03583956795335347) [X2 Y3 Y4 X5]
+ (0.038314670294803906) [Y4 X5 X12 Y13]
+ (0.038314670294803906) [X4 Y5 Y12 X13]
+ (0.10433064780651377) [Z0 Y1 Z2 Y3]
+ (0.10433064780651377) [Z0 X1 Z2 X3]
+ (-0.12133276911042344) [Y2 Z3 Z4 Z5 Y6]
+ (-0.12133276911042344) [X2 Z3 Z4 Z5 X6]
+ (-0.1213327691104234) [Y3 Z4 Z5 Z6 Y7]
+ (-0.1213327691104234) [X3 Z4 Z5 Z6 X7]
+ (3.2020768803340124e-06) [Y1 Z2 Z3 Z4 Y5]
+ (3.2020768803340124e-06) [X1 Z2 Z3 Z4 X5]
+ (3.2020768803340137e-06) [Y0 Z1 Z2 Z3 Y4]
+ (3.2020768803340137e-06) [X0 Z1 Z2 Z3 X4]
+ (0.2284810656491887) [Y6 Z7 Z8 Z9 Y10]
+ (0.2284810656491887) [X6 Z7 Z8 Z9 X10]
+ (0.2284810656491887) [Y7 Z8 Z9 Z10 Y11]
+ (0.2284810656491887) [X7 Z8 Z9 Z10 X11]
+ (-0.03276765782329054) [Z0 Y3 Z4 Z5 Z6 Y7]
+ (-0.03276765782329054) [Z0 X3 Z4 Z5 Z6 X7]
+ (-0.03276765782329054) [Z1 Y2 Z3 Z4 Z5 Y6]
+ (-0.03276765782329054) [Z1 X2 Z3 Z4 Z5 X6]
+ (-0.0271150368452732) [Z0 Y2 Z3 Z4 Z5 Y6]
+ (-0.0271150368452732) [Z0 X2 Z3 Z4 Z5 X6]
+ (-0.0271150368452732) [Z1 Y3 Z4 Z5 Z6 Y7]
+ (-0.0271150368452732) [Z1 X3 Z4 Z5 Z6 X7]
+ (-0.025996177598021232) [Y2 Z3 Z4 Z5 Y6 Z7]
+ (-0.025996177598021232) [X2 Z3 Z4 Z5 X6 Z7]
+ (-0.017561202409646186) [Y2 Z3 Z4 Z5 Y6 Z9]
+ (-0.017561202409646186) [X2 Z3 Z4 Z5 X6 Z9]
+ (-0.017561202409646186) [Y3 Z4 Z5 Z6 Y7 Z8]
+ (-0.017561202409646186) [X3 Z4 Z5 Z6 X7 Z8]
+ (-0.014564531231173013) [Y7 Z8 Z9 X10 X12 Y13]
+ (-0.014564531231173013) [Y7 Z8 Z9 Y10 Y12 Y13]
+ (-0.014564531231173013) [X7 Z8 Z9 X10 X12 X13]
+ (-0.014564531231173013) [X7 Z8 Z9 Y10 Y12 X13]
+ (-0.012215040997613936) [Y4 Z5 Y6 Y11 Z12 Y13]
+ (-0.012215040997613936) [Y4 Z5 Y6 X11 Z12 X13]
+ (-0.012215040997613936) [X4 Z5 X6 Y11 Z12 Y13]
+ (-0.012215040997613936) [X4 Z5 X6 X11 Z12 X13]
+ (-0.012215040997613936) [Y5 Z6 Y7 Y10 Z11 Y12]
+ (-0.012215040997613936) [Y5 Z6 Y7 X10 Z11 X12]
+ (-0.012215040997613936) [X5 Z6 X7 Y10 Z11 Y12]
+ (-0.012215040997613936) [X5 Z6 X7 X10 Z11 X12]
+ (-0.011756013419819245) [Y3 Z4 Z5 X6 X8 Y9]
+ (-0.011756013419819245) [Y3 Z4 Z5 Y6 Y8 Y9]
+ (-0.011756013419819245) [X3 Z4 Z5 X6 X8 X9]
+ (-0.011756013419819245) [X3 Z4 Z5 Y6 Y8 X9]
+ (-0.008764827575688819) [Y2 Z3 Z4 X5 X11 Y12]
+ (-0.008764827575688819) [Y2 Z3 Z4 Y5 Y11 Y12]
+ (-0.008764827575688819) [X2 Z3 Z4 X5 X11 X12]
+ (-0.008764827575688819) [X2 Z3 Z4 Y5 Y11 X12]
+ (-0.008764827575688819) [Y3 X4 X10 Z11 Z12 Y13]
+ (-0.008764827575688819) [Y3 Y4 Y10 Z11 Z12 Y13]
+ (-0.008764827575688819) [X3 X4 X10 Z11 Z12 X13]
+ (-0.008764827575688819) [X3 Y4 Y10 Z11 Z12 X13]
+ (-0.008125251921381015) [Y0 Z1 Z2 Y3 X8 X9]
+ (-0.008125251921381015) [X0 Z1 Z2 X3 Y8 Y9]
+ (-0.007306759928832958) [Y4 X5 X7 Z8 Z9 Y10]
+ (-0.007306759928832958) [Y4 Y5 Y7 Z8 Z9 Y10]
+ (-0.007306759928832958) [X4 X5 X7 Z8 Z9 X10]
+ (-0.007306759928832958) [X4 Y5 Y7 Z8 Z9 X10]
+ (-0.0058051889898269385) [Y2 Z3 Z4 Z5 Y6 Z8]
+ (-0.0058051889898269385) [X2 Z3 Z4 Z5 X6 Z8]
+ (-0.0058051889898269385) [Y3 Z4 Z5 Z6 Y7 Z9]
+ (-0.0058051889898269385) [X3 Z4 Z5 Z6 X7 Z9]
+ (-0.005652620978017338) [Y0 X1 X3 Z4 Z5 Y6]
+ (-0.005652620978017338) [Y0 Y1 Y3 Z4 Z5 Y6]
+ (-0.005652620978017338) [X0 X1 X3 Z4 Z5 X6]
+ (-0.005652620978017338) [X0 Y1 Y3 Z4 Z5 X6]
+ (-0.005143391768825129) [Y2 Z3 Y4 Y5 Z6 Y7]
+ (-0.005143391768825129) [Y2 Z3 Y4 X5 Z6 X7]
+ (-0.005143391768825129) [X2 Z3 X4 Y5 Z6 Y7]
+ (-0.005143391768825129) [X2 Z3 X4 X5 Z6 X7]
+ (-0.004684903388155207) [Y0 Z1 Z2 Y3 X6 X7]
+ (-0.004684903388155207) [X0 Z1 Z2 X3 Y6 Y7]
+ (-0.004668620318776288) [Y1 X2 X7 Z8 Z9 Y10]
+ (-0.004668620318776288) [X1 Y2 Y7 Z8 Z9 X10]
+ (-0.0044248554494418476) [Y0 Z1 Z2 Y3 X4 X5]
+ (-0.0044248554494418476) [X0 Z1 Z2 X3 Y4 Y5]
+ (-0.004158797381840035) [Y3 Z4 Z5 X6 X12 Y13]
+ (-0.004158797381840035) [Y3 Z4 Z5 Y6 Y12 Y13]
+ (-0.004158797381840035) [X3 Z4 Z5 X6 X12 X13]
+ (-0.004158797381840035) [X3 Z4 Z5 Y6 Y12 X13]
+ (-0.0034937903598901525) [Y2 Z3 Z4 Z5 Y6 Z13]
+ (-0.0034937903598901525) [X2 Z3 Z4 Z5 X6 Z13]
+ (-0.0034937903598901525) [Y3 Z4 Z5 Z6 Y7 Z12]
+ (-0.0034937903598901525) [X3 Z4 Z5 Z6 X7 Z12]
+ (-0.0027790267990255553) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (-0.0027790267990255553) [X1 Z2 Z3 Z4 Z5 X7]
+ (-0.00229395661135245) [Y1 X2 X3 Z4 Z5 Y6]
+ (-0.00229395661135245) [X1 Y2 Y3 Z4 Z5 X6]
+ (-0.001799219493663007) [Y0 Z1 Z2 Y3 X10 X11]
+ (-0.001799219493663007) [X0 Z1 Z2 X3 Y10 Y11]
+ (-0.0017278753941369644) [Y1 Z2 Z3 X4 X11 Y12]
+ (-0.0017278753941369644) [X1 Z2 Z3 Y4 Y11 X12]
+ (-0.0009298507967730565) [Y4 Z5 Y6 X10 Z11 X12]
+ (-0.0009298507967730565) [X4 Z5 X6 Y10 Z11 Y12]
+ (-0.0009298507967730565) [Y5 Z6 Y7 X11 Z12 X13]
+ (-0.0009298507967730565) [X5 Z6 X7 Y11 Z12 Y13]
+ (-0.0008533856254125475) [Y1 Z2 Z3 Y4 X5 X6]
+ (-0.0008533856254125475) [X1 Z2 Z3 X4 Y5 Y6]
+ (-0.0008145313270956824) [Y2 Z3 Z4 Z5 Y6 Z10]
+ (-0.0008145313270956824) [X2 Z3 Z4 Z5 X6 Z10]
+ (-0.0008145313270956824) [Y3 Z4 Z5 Z6 Y7 Z11]
+ (-0.0008145313270956824) [X3 Z4 Z5 Z6 X7 Z11]
+ (-7.735036880591523e-05) [Y0 X1 X7 Z8 Z9 Y10]
+ (-7.735036880591523e-05) [Y0 Y1 Y7 Z8 Z9 Y10]
+ (-7.735036880591523e-05) [X0 X1 X7 Z8 Z9 X10]
+ (-7.735036880591523e-05) [X0 Y1 Y7 Z8 Z9 X10]
+ (-8.774817865547715e-06) [Y4 Z5 Z6 Z7 Z8 Y10]
+ (-8.774817865547715e-06) [X4 Z5 Z6 Z7 Z8 X10]
+ (-8.774817865547715e-06) [Y5 Z6 Z7 Z9 Z10 Y11]
+ (-8.774817865547715e-06) [X5 Z6 Z7 Z9 Z10 X11]
+ (-7.51836221648362e-06) [Y4 Z5 Z7 Z8 Z9 Y10]
+ (-7.51836221648362e-06) [X4 Z5 Z7 Z8 Z9 X10]
+ (-7.51836221648362e-06) [Y5 Z6 Z8 Z9 Z10 Y11]
+ (-7.51836221648362e-06) [X5 Z6 Z8 Z9 Z10 X11]
+ (-7.44434467672858e-06) [Y4 Z5 Z6 Z7 Z9 Y10]
+ (-7.44434467672858e-06) [X4 Z5 Z6 Z7 Z9 X10]
+ (-7.44434467672858e-06) [Y5 Z6 Z7 Z8 Z10 Y11]
+ (-7.44434467672858e-06) [X5 Z6 Z7 Z8 Z10 X11]
+ (-6.52437384933658e-06) [Y6 Z7 Z8 Z9 Z10 Y12]
+ (-6.52437384933658e-06) [X6 Z7 Z8 Z9 Z10 X12]
+ (-6.52437384933658e-06) [Y7 Z8 Z9 Z11 Z12 Y13]
+ (-6.52437384933658e-06) [X7 Z8 Z9 Z11 Z12 X13]
+ (-6.290028433913804e-06) [Y4 Z5 Z6 Z8 Z9 Y10]
+ (-6.290028433913804e-06) [X4 Z5 Z6 Z8 Z9 X10]
+ (-6.290028433913804e-06) [Y5 Z7 Z8 Z9 Z10 Y11]
+ (-6.290028433913804e-06) [X5 Z7 Z8 Z9 Z10 X11]
+ (-5.974311714081735e-06) [Y4 Z5 Z6 X7 X10 Y11]
+ (-5.974311714081735e-06) [X4 Z5 Z6 Y7 Y10 X11]
+ (-5.2758831227450436e-06) [Y2 Z3 Z4 X5 X12 Y13]
+ (-5.2758831227450436e-06) [X2 Z3 Z4 Y5 Y12 X13]
+ (-4.64305106898011e-06) [Y2 Z3 Z4 Y5 X10 X11]
+ (-4.64305106898011e-06) [X2 Z3 Z4 X5 Y10 Y11]
+ (-4.556569218652213e-06) [Y4 Z5 Z6 Y7 X12 X13]
+ (-4.556569218652213e-06) [X4 Z5 Z6 X7 Y12 Y13]
+ (-4.25322422604367e-06) [Y4 Z6 Z7 Z8 Z9 Y10]
+ (-4.25322422604367e-06) [X4 Z6 Z7 Z8 Z9 X10]
+ (-3.7696594524189586e-06) [Y6 Z8 Z9 Z10 Z11 Y12]
+ (-3.7696594524189586e-06) [X6 Z8 Z9 Z10 Z11 X12]
+ (-3.694513294886729e-06) [Y4 Y5 X10 Z11 Z12 X13]
+ (-3.694513294886729e-06) [X4 X5 Y10 Z11 Z12 Y13]
+ (-3.610297131015923e-06) [Y6 Z7 Z9 Z10 Z11 Y12]
+ (-3.610297131015923e-06) [X6 Z7 Z9 Z10 Z11 X12]
+ (-3.610297131015923e-06) [Y7 Z8 Z10 Z11 Z12 Y13]
+ (-3.610297131015923e-06) [X7 Z8 Z10 Z11 Z12 X13]
+ (-3.313145500508457e-06) [Y7 Z8 Z9 Y10 X11 X12]
+ (-3.313145500508457e-06) [X7 Z8 Z9 X10 Y11 Y12]
+ (-3.2774831959184443e-06) [Y6 Z7 Z8 Z10 Z11 Y12]
+ (-3.2774831959184443e-06) [X6 Z7 Z8 Z10 Z11 X12]
+ (-3.2774831959184443e-06) [Y7 Z9 Z10 Z11 Z12 Y13]
+ (-3.2774831959184443e-06) [X7 Z9 Z10 Z11 Z12 X13]
+ (-3.2112283488281236e-06) [Y6 Z7 Z8 Z9 Z11 Y12]
+ (-3.2112283488281236e-06) [X6 Z7 Z8 Z9 Z11 X12]
+ (-3.2112283488281236e-06) [Y7 Z8 Z9 Z10 Z12 Y13]
+ (-3.2112283488281236e-06) [X7 Z8 Z9 Z10 Z12 X13]
+ (-3.1513463114993e-06) [Y3 Y4 X7 Z8 Z9 X10]
+ (-3.1513463114993e-06) [X3 X4 Y7 Z8 Z9 Y10]
+ (-3.0882507116779086e-06) [Y3 Z4 Z5 Y6 X11 X12]
+ (-3.0882507116779086e-06) [X3 Z4 Z5 X6 Y11 Y12]
+ (-2.1726691017726618e-06) [Y2 X3 X10 Z11 Z12 Y13]
+ (-2.1726691017726618e-06) [X2 Y3 Y10 Z11 Z12 X13]
+ (-1.4548424491854792e-06) [Y2 Z3 Z4 Y5 X6 X7]
+ (-1.4548424491854792e-06) [X2 Z3 Z4 X5 Y6 Y7]
+ (-1.3304731888191346e-06) [Y5 Z6 Z7 Y8 X9 X10]
+ (-1.3304731888191346e-06) [X5 Z6 Z7 X8 Y9 Y10]
+ (-1.2283337825698148e-06) [Y5 X6 X7 Z8 Z9 Y10]
+ (-1.2283337825698148e-06) [X5 Y6 Y7 Z8 Z9 X10]
+ (-1.035847760239714e-06) [Y6 Y7 X10 Z11 Z12 X13]
+ (-1.035847760239714e-06) [X6 X7 Y10 Z11 Z12 Y13]
+ (-7.956895373529558e-07) [Y2 Z3 Z4 Y5 X8 X9]
+ (-7.956895373529558e-07) [X2 Z3 Z4 X5 Y8 Y9]
+ (-6.733197743234441e-07) [Y0 Z1 Z2 Z3 Y4 Z10]
+ (-6.733197743234441e-07) [X0 Z1 Z2 Z3 X4 Z10]
+ (-6.733197743234441e-07) [Y1 Z2 Z3 Z4 Y5 Z11]
+ (-6.733197743234441e-07) [X1 Z2 Z3 Z4 X5 Z11]
+ (-6.628614202510253e-07) [Y8 X9 X10 Z11 Z12 Y13]
+ (-6.628614202510253e-07) [X8 Y9 Y10 Z11 Z12 X13]
+ (-6.556281915255651e-07) [Y0 Z1 Y2 X10 Z11 X12]
+ (-6.556281915255651e-07) [X0 Z1 X2 Y10 Z11 Y12]
+ (-6.556281915255651e-07) [Y1 Z2 Y3 X11 Z12 X13]
+ (-6.556281915255651e-07) [X1 Z2 X3 Y11 Z12 Y13]
+ (-6.418291575299119e-07) [Y0 Z1 Y2 Y10 Z11 Y12]
+ (-6.418291575299119e-07) [X0 Z1 X2 X10 Z11 X12]
+ (-6.418291575299119e-07) [Y1 Z2 Y3 Y11 Z12 Y13]
+ (-6.418291575299119e-07) [X1 Z2 X3 X11 Z12 X13]
+ (-5.927453083565626e-07) [Y0 Z1 Z2 Z3 Y4 Z11]
+ (-5.927453083565626e-07) [X0 Z1 Z2 Z3 X4 Z11]
+ (-5.927453083565626e-07) [Y1 Z2 Z3 Z4 Y5 Z10]
+ (-5.927453083565626e-07) [X1 Z2 Z3 Z4 X5 Z10]
+ (-5.627851911980483e-07) [Y0 X1 X10 Z11 Z12 Y13]
+ (-5.627851911980483e-07) [X0 Y1 Y10 Z11 Z12 X13]
+ (-5.287660625292557e-07) [Y0 Z1 Z2 X3 X11 Y12]
+ (-5.287660625292557e-07) [Y0 Z1 Z2 Y3 Y11 Y12]
+ (-5.287660625292557e-07) [X0 Z1 Z2 X3 X11 X12]
+ (-5.287660625292557e-07) [X0 Z1 Z2 Y3 Y11 X12]
+ (-5.287660625292557e-07) [Y1 X2 X10 Z11 Z12 Y13]
+ (-5.287660625292557e-07) [Y1 Y2 Y10 Z11 Z12 Y13]
+ (-5.287660625292557e-07) [X1 X2 X10 Z11 Z12 X13]
+ (-5.287660625292557e-07) [X1 Y2 Y10 Z11 Z12 X13]
+ (-4.838052751314664e-07) [Y4 Z5 Z6 Y7 X8 X9]
+ (-4.838052751314664e-07) [X4 Z5 Z6 X7 Y8 Y9]
+ (-3.57076132954832e-07) [Y0 Y1 X2 Z3 Z4 X5]
+ (-3.57076132954832e-07) [X0 X1 Y2 Z3 Z4 Y5]
+ (-3.3281393509747905e-07) [Y7 X8 X9 Z10 Z11 Y12]
+ (-3.3281393509747905e-07) [X7 Y8 Y9 Z10 Z11 X12]
+ (-3.086826565463728e-07) [Y1 Z2 Z3 X4 X12 Y13]
+ (-3.086826565463728e-07) [Y1 Z2 Z3 Y4 Y12 Y13]
+ (-3.086826565463728e-07) [X1 Z2 Z3 X4 X12 X13]
+ (-3.086826565463728e-07) [X1 Z2 Z3 Y4 Y12 X13]
+ (-2.447323129034128e-07) [Y0 Y1 X4 Z5 Z6 X7]
+ (-2.447323129034128e-07) [X0 X1 Y4 Z5 Z6 Y7]
+ (-2.3713289481715648e-07) [Y1 Z2 Z3 X4 X8 Y9]
+ (-2.3713289481715648e-07) [Y1 Z2 Z3 Y4 Y8 Y9]
+ (-2.3713289481715648e-07) [X1 Z2 Z3 X4 X8 X9]
+ (-2.3713289481715648e-07) [X1 Z2 Z3 Y4 Y8 X9]
+ (-2.199051619033749e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (-2.199051619033749e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (-1.933241277209509e-07) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (-1.933241277209509e-07) [Y0 Z1 Y2 X3 Z4 X5]
+ (-1.933241277209509e-07) [X0 Z1 X2 Y3 Z4 Y5]
+ (-1.933241277209509e-07) [X0 Z1 X2 X3 Z4 X5]
+ (-1.8394209156656232e-07) [Y1 Z2 Z3 X4 X6 Y7]
+ (-1.8394209156656232e-07) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-1.8394209156656232e-07) [X1 Z2 Z3 X4 X6 X7]
+ (-1.8394209156656232e-07) [X1 Z2 Z3 Y4 Y6 X7]
+ (-1.5510539177252044e-07) [Y0 Z1 Y2 Y4 Z5 Y6]
+ (-1.5510539177252044e-07) [X0 Z1 X2 X4 Z5 X6]
+ (-1.5510539177252044e-07) [Y1 Z2 Y3 Y5 Z6 Y7]
+ (-1.5510539177252044e-07) [X1 Z2 X3 X5 Z6 X7]
+ (-1.3807781481655693e-07) [Y0 Z1 Y2 X4 Z5 X6]
+ (-1.3807781481655693e-07) [X0 Z1 X2 Y4 Z5 Y6]
+ (-1.3807781481655693e-07) [Y0 Z1 Y2 Y5 Z6 Y7]
+ (-1.3807781481655693e-07) [Y0 Z1 Y2 X5 Z6 X7]
+ (-1.3807781481655693e-07) [X0 Z1 X2 Y5 Z6 Y7]
+ (-1.3807781481655693e-07) [X0 Z1 X2 X5 Z6 X7]
+ (-1.3807781481655693e-07) [Y1 Z2 Y3 Y4 Z5 Y6]
+ (-1.3807781481655693e-07) [Y1 Z2 Y3 X4 Z5 X6]
+ (-1.3807781481655693e-07) [X1 Z2 X3 Y4 Z5 Y6]
+ (-1.3807781481655693e-07) [X1 Z2 X3 X4 Z5 X6]
+ (-1.3807781481655693e-07) [Y1 Z2 Y3 X5 Z6 X7]
+ (-1.3807781481655693e-07) [X1 Z2 X3 Y5 Z6 Y7]
+ (-1.2919694865037952e-07) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (-1.2919694865037952e-07) [X0 Z1 Z2 Z3 X4 Z5]
+ (-1.1076325600151951e-07) [Y0 Z1 Y2 Y11 Z12 Y13]
+ (-1.1076325600151951e-07) [Y0 Z1 Y2 X11 Z12 X13]
+ (-1.1076325600151951e-07) [X0 Z1 X2 Y11 Z12 Y13]
+ (-1.1076325600151951e-07) [X0 Z1 X2 X11 Z12 X13]
+ (-1.1076325600151951e-07) [Y1 Z2 Y3 Y10 Z11 Y12]
+ (-1.1076325600151951e-07) [Y1 Z2 Y3 X10 Z11 X12]
+ (-1.1076325600151951e-07) [X1 Z2 X3 Y10 Z11 Y12]
+ (-1.1076325600151951e-07) [X1 Z2 X3 X10 Z11 X12]
+ (8.05744659668815e-08) [Y1 Z2 Z3 X4 X10 Y11]
+ (8.05744659668815e-08) [Y1 Z2 Z3 Y4 Y10 Y11]
+ (8.05744659668815e-08) [X1 Z2 Z3 X4 X10 X11]
+ (8.05744659668815e-08) [X1 Z2 Z3 Y4 Y10 X11]
+ (8.649310134524579e-08) [Y0 Z1 Z2 Z3 Y4 Z9]
+ (8.649310134524579e-08) [X0 Z1 Z2 Z3 X4 Z9]
+ (8.649310134524579e-08) [Y1 Z2 Z3 Z4 Y5 Z8]
+ (8.649310134524579e-08) [X1 Z2 Z3 Z4 X5 Z8]
+ (1.8394209156656235e-07) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (1.8394209156656235e-07) [X0 Z1 Z2 Z3 X4 Z6]
+ (1.8394209156656235e-07) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (1.8394209156656235e-07) [X1 Z2 Z3 Z4 X5 Z7]
+ (2.199051619033749e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (2.199051619033749e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (2.447323129034128e-07) [Y0 X1 X4 Z5 Z6 Y7]
+ (2.447323129034128e-07) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.236259961624023e-07) [Y0 Z1 Z2 Z3 Y4 Z8]
+ (3.236259961624023e-07) [X0 Z1 Z2 Z3 X4 Z8]
+ (3.236259961624023e-07) [Y1 Z2 Z3 Z4 Y5 Z9]
+ (3.236259961624023e-07) [X1 Z2 Z3 Z4 X5 Z9]
+ (3.3281393509747905e-07) [Y7 Y8 X9 Z10 Z11 X12]
+ (3.3281393509747905e-07) [X7 X8 Y9 Z10 Z11 Y12]
+ (3.57076132954832e-07) [Y0 X1 X2 Z3 Z4 Y5]
+ (3.57076132954832e-07) [X0 Y1 Y2 Z3 Z4 X5]
+ (4.838052751314664e-07) [Y4 Z5 Z6 X7 X8 Y9]
+ (4.838052751314664e-07) [X4 Z5 Z6 Y7 Y8 X9]
+ (5.627851911980483e-07) [Y0 Y1 X10 Z11 Z12 X13]
+ (5.627851911980483e-07) [X0 X1 Y10 Z11 Z12 Y13]
+ (6.628614202510253e-07) [Y8 Y9 X10 Z11 Z12 X13]
+ (6.628614202510253e-07) [X8 X9 Y10 Z11 Z12 Y13]
+ (7.956895373529558e-07) [Y2 Z3 Z4 X5 X8 Y9]
+ (7.956895373529558e-07) [X2 Z3 Z4 Y5 Y8 X9]
+ (9.3065366528724e-07) [Y0 Z1 Z2 Z3 Y4 Z13]
+ (9.3065366528724e-07) [X0 Z1 Z2 Z3 X4 Z13]
+ (9.3065366528724e-07) [Y1 Z2 Z3 Z4 Y5 Z12]
+ (9.3065366528724e-07) [X1 Z2 Z3 Z4 X5 Z12]
+ (1.035847760239714e-06) [Y6 X7 X10 Z11 Z12 Y13]
+ (1.035847760239714e-06) [X6 Y7 Y10 Z11 Z12 X13]
+ (1.2283337825698148e-06) [Y5 Y6 X7 Z8 Z9 X10]
+ (1.2283337825698148e-06) [X5 X6 Y7 Z8 Z9 Y10]
+ (1.2393363218336128e-06) [Y0 Z1 Z2 Z3 Y4 Z12]
+ (1.2393363218336128e-06) [X0 Z1 Z2 Z3 X4 Z12]
+ (1.2393363218336128e-06) [Y1 Z2 Z3 Z4 Y5 Z13]
+ (1.2393363218336128e-06) [X1 Z2 Z3 Z4 X5 Z13]
+ (1.3304731888191346e-06) [Y5 Z6 Z7 X8 X9 Y10]
+ (1.3304731888191346e-06) [X5 Z6 Z7 Y8 Y9 X10]
+ (1.4548424491854792e-06) [Y2 Z3 Z4 X5 X6 Y7]
+ (1.4548424491854792e-06) [X2 Z3 Z4 Y5 Y6 X7]
+ (2.1726691017726618e-06) [Y2 Y3 X10 Z11 Z12 X13]
+ (2.1726691017726618e-06) [X2 X3 Y10 Z11 Z12 Y13]
+ (3.0882507116779086e-06) [Y3 Z4 Z5 X6 X11 Y12]
+ (3.0882507116779086e-06) [X3 Z4 Z5 Y6 Y11 X12]
+ (3.117447946401369e-06) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (3.117447946401369e-06) [Z0 X1 Z2 Z3 Z4 X5]
+ (3.1513463114993e-06) [Y3 X4 X7 Z8 Z9 Y10]
+ (3.1513463114993e-06) [X3 Y4 Y7 Z8 Z9 X10]
+ (3.313145500508457e-06) [Y7 Z8 Z9 X10 X11 Y12]
+ (3.313145500508457e-06) [X7 Z8 Z9 Y10 Y11 X12]
+ (3.3343312898181753e-06) [Y5 Z6 Z7 Z8 Z9 Y11]
+ (3.3343312898181753e-06) [X5 Z6 Z7 Z8 Z9 X11]
+ (3.694513294886729e-06) [Y4 X5 X10 Z11 Z12 Y13]
+ (3.694513294886729e-06) [X4 Y5 Y10 Z11 Z12 X13]
+ (4.183932559849535e-06) [Y7 Z8 Z9 Z10 Z11 Y13]
+ (4.183932559849535e-06) [X7 Z8 Z9 Z10 Z11 X13]
+ (4.556569218652213e-06) [Y4 Z5 Z6 X7 X12 Y13]
+ (4.556569218652213e-06) [X4 Z5 Z6 Y7 Y12 X13]
+ (4.64305106898011e-06) [Y2 Z3 Z4 X5 X10 Y11]
+ (4.64305106898011e-06) [X2 Z3 Z4 Y5 Y10 X11]
+ (5.2758831227450436e-06) [Y2 Z3 Z4 Y5 X12 X13]
+ (5.2758831227450436e-06) [X2 Z3 Z4 X5 Y12 Y13]
+ (5.974311714081735e-06) [Y4 Z5 Z6 Y7 X10 X11]
+ (5.974311714081735e-06) [X4 Z5 Z6 X7 Y10 Y11]
+ (0.0002921986261110777) [Y6 Z7 Y8 Y9 Z10 Y11]
+ (0.0002921986261110777) [Y6 Z7 Y8 X9 Z10 X11]
+ (0.0002921986261110777) [X6 Z7 X8 Y9 Z10 Y11]
+ (0.0002921986261110777) [X6 Z7 X8 X9 Z10 X11]
+ (0.0004956762314916131) [Z2 Y3 Z4 Z5 Z6 Y7]
+ (0.0004956762314916131) [Z2 X3 Z4 Z5 Z6 X7]
+ (0.0006650070219498819) [Y2 Z3 Z4 Z5 Y6 Z12]
+ (0.0006650070219498819) [X2 Z3 Z4 Z5 X6 Z12]
+ (0.0006650070219498819) [Y3 Z4 Z5 Z6 Y7 Z13]
+ (0.0006650070219498819) [X3 Z4 Z5 Z6 X7 Z13]
+ (0.0008533856254125475) [Y1 Z2 Z3 X4 X5 Y6]
+ (0.0008533856254125475) [X1 Z2 Z3 Y4 Y5 X6]
+ (0.0016095313817213672) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (0.0016095313817213672) [X0 Z1 Z2 Z3 Z4 X6]
+ (0.0016095313817213672) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (0.0016095313817213672) [X1 Z2 Z3 Z5 Z6 X7]
+ (0.0016676041811440438) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (0.0016676041811440438) [X0 Z1 Z3 Z4 Z5 X6]
+ (0.0016676041811440438) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (0.0016676041811440438) [X1 Z2 Z4 Z5 Z6 X7]
+ (0.0017278753941369644) [Y1 Z2 Z3 Y4 X11 X12]
+ (0.0017278753941369644) [X1 Z2 Z3 X4 Y11 Y12]
+ (0.001799219493663007) [Y0 Z1 Z2 X3 X10 Y11]
+ (0.001799219493663007) [X0 Z1 Z2 Y3 Y10 X11]
+ (0.00229395661135245) [Y1 Y2 X3 Z4 Z5 X6]
+ (0.00229395661135245) [X1 X2 Y3 Z4 Z5 Y6]
+ (0.002462917007133915) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (0.002462917007133915) [X0 Z1 Z2 Z3 Z5 X6]
+ (0.002462917007133915) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (0.002462917007133915) [X1 Z2 Z3 Z4 Z6 X7]
+ (0.003961560792496495) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (0.003961560792496495) [X0 Z1 Z2 Z4 Z5 X6]
+ (0.003961560792496495) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (0.003961560792496495) [X1 Z3 Z4 Z5 Z6 X7]
+ (0.0044248554494418476) [Y0 Z1 Z2 X3 X4 Y5]
+ (0.0044248554494418476) [X0 Z1 Z2 Y3 Y4 X5]
+ (0.004668620318776288) [Y1 Y2 X7 Z8 Z9 X10]
+ (0.004668620318776288) [X1 X2 Y7 Z8 Z9 Y10]
+ (0.004684903388155207) [Y0 Z1 Z2 X3 X6 Y7]
+ (0.004684903388155207) [X0 Z1 Z2 Y3 Y6 X7]
+ (0.0053248352342216854) [Y2 Z3 Y4 X10 Z11 X12]
+ (0.0053248352342216854) [X2 Z3 X4 Y10 Z11 Y12]
+ (0.0053248352342216854) [Y3 Z4 Y5 X11 Z12 X13]
+ (0.0053248352342216854) [X3 Z4 X5 Y11 Z12 Y13]
+ (0.005368659358109485) [Y2 X3 X7 Z8 Z9 Y10]
+ (0.005368659358109485) [Y2 Y3 Y7 Z8 Z9 Y10]
+ (0.005368659358109485) [X2 X3 X7 Z8 Z9 X10]
+ (0.005368659358109485) [X2 Y3 Y7 Z8 Z9 X10]
+ (0.007960880725921554) [Y4 Z5 Y6 Y10 Z11 Y12]
+ (0.007960880725921554) [X4 Z5 X6 X10 Z11 X12]
+ (0.007960880725921554) [Y5 Z6 Y7 Y11 Z12 Y13]
+ (0.007960880725921554) [X5 Z6 X7 X11 Z12 X13]
+ (0.008125251921381015) [Y0 Z1 Z2 X3 X8 Y9]
+ (0.008125251921381015) [X0 Z1 Z2 Y3 Y8 X9]
+ (0.00889073152269461) [Y4 Z5 X6 X10 Z11 Y12]
+ (0.00889073152269461) [X4 Z5 Y6 Y10 Z11 X12]
+ (0.00889073152269461) [Y5 Z6 X7 X11 Z12 Y13]
+ (0.00889073152269461) [X5 Z6 Y7 Y11 Z12 X13]
+ (0.01026341486815851) [Y2 Z3 X4 X10 Z11 Y12]
+ (0.01026341486815851) [X2 Z3 Y4 Y10 Z11 X12]
+ (0.01026341486815851) [Y3 Z4 X5 X11 Z12 Y13]
+ (0.01026341486815851) [X3 Z4 Y5 Y11 Z12 X13]
+ (0.010540425907671562) [Y6 Z7 Z8 Z9 Y10 Z13]
+ (0.010540425907671562) [X6 Z7 Z8 Z9 X10 Z13]
+ (0.010540425907671562) [Y7 Z8 Z9 Z10 Y11 Z12]
+ (0.010540425907671562) [X7 Z8 Z9 Z10 X11 Z12]
+ (0.010960074940542595) [Z4 Y7 Z8 Z9 Z10 Y11]
+ (0.010960074940542595) [Z4 X7 Z8 Z9 Z10 X11]
+ (0.010960074940542595) [Z5 Y6 Z7 Z8 Z9 Y10]
+ (0.010960074940542595) [Z5 X6 Z7 Z8 Z9 X10]
+ (0.011307274008848275) [Y6 Z7 Z8 Z9 Y10 Z11]
+ (0.011307274008848275) [X6 Z7 Z8 Z9 X10 Z11]
+ (0.014411099430130928) [Y2 Z3 Z4 Z5 Y6 Z11]
+ (0.014411099430130928) [X2 Z3 Z4 Z5 X6 Z11]
+ (0.014411099430130928) [Y3 Z4 Z5 Z6 Y7 Z10]
+ (0.014411099430130928) [X3 Z4 Z5 Z6 X7 Z10]
+ (0.015225630757226608) [Y3 Z4 Z5 X6 X10 Y11]
+ (0.015225630757226608) [Y3 Z4 Z5 Y6 Y10 Y11]
+ (0.015225630757226608) [X3 Z4 Z5 X6 X10 X11]
+ (0.015225630757226608) [X3 Z4 Z5 Y6 Y10 X11]
+ (0.01558825010238019) [Y2 Z3 Y4 Y10 Z11 Y12]
+ (0.01558825010238019) [X2 Z3 X4 X10 Z11 X12]
+ (0.01558825010238019) [Y3 Z4 Y5 Y11 Z12 Y13]
+ (0.01558825010238019) [X3 Z4 X5 X11 Z12 X13]
+ (0.018266834869375557) [Z4 Y6 Z7 Z8 Z9 Y10]
+ (0.018266834869375557) [Z4 X6 Z7 Z8 Z9 X10]
+ (0.018266834869375557) [Z5 Y7 Z8 Z9 Z10 Y11]
+ (0.018266834869375557) [Z5 X7 Z8 Z9 Z10 X11]
+ (0.019020423173039973) [Z2 Y6 Z7 Z8 Z9 Y10]
+ (0.019020423173039973) [Z2 X6 Z7 Z8 Z9 X10]
+ (0.019020423173039973) [Z3 Y7 Z8 Z9 Z10 Y11]
+ (0.019020423173039973) [Z3 X7 Z8 Z9 Z10 X11]
+ (0.020175921723535488) [Y4 Z5 Z6 X7 X11 Y12]
+ (0.020175921723535488) [Y4 Z5 Z6 Y7 Y11 Y12]
+ (0.020175921723535488) [X4 Z5 Z6 X7 X11 X12]
+ (0.020175921723535488) [X4 Z5 Z6 Y7 Y11 X12]
+ (0.020175921723535488) [Y5 X6 X10 Z11 Z12 Y13]
+ (0.020175921723535488) [Y5 Y6 Y10 Z11 Z12 Y13]
+ (0.020175921723535488) [X5 X6 X10 Z11 Z12 X13]
+ (0.020175921723535488) [X5 Y6 Y10 Z11 Z12 X13]
+ (0.024353077678069015) [Y2 Z3 Y4 Y11 Z12 Y13]
+ (0.024353077678069015) [Y2 Z3 Y4 X11 Z12 X13]
+ (0.024353077678069015) [X2 Z3 X4 Y11 Z12 Y13]
+ (0.024353077678069015) [X2 Z3 X4 X11 Z12 X13]
+ (0.024353077678069015) [Y3 Z4 Y5 Y10 Z11 Y12]
+ (0.024353077678069015) [Y3 Z4 Y5 X10 Z11 X12]
+ (0.024353077678069015) [X3 Z4 X5 Y10 Z11 Y12]
+ (0.024353077678069015) [X3 Z4 X5 X10 Z11 X12]
+ (0.02438908253114946) [Z2 Y7 Z8 Z9 Z10 Y11]
+ (0.02438908253114946) [Z2 X7 Z8 Z9 Z10 X11]
+ (0.02438908253114946) [Z3 Y6 Z7 Z8 Z9 Y10]
+ (0.02438908253114946) [Z3 X6 Z7 Z8 Z9 X10]
+ (0.025104957138844575) [Y6 Z7 Z8 Z9 Y10 Z12]
+ (0.025104957138844575) [X6 Z7 Z8 Z9 X10 Z12]
+ (0.025104957138844575) [Y7 Z8 Z9 Z10 Y11 Z13]
+ (0.025104957138844575) [X7 Z8 Z9 Z10 X11 Z13]
+ (0.03078750538914394) [Z6 Y7 Z8 Z9 Z10 Y11]
+ (0.03078750538914394) [Z6 X7 Z8 Z9 Z10 X11]
+ (0.045879470781297955) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (0.045879470781297955) [X0 Z2 Z3 Z4 Z5 X6]
+ (0.056007330877807605) [Z0 Y7 Z8 Z9 Z10 Y11]
+ (0.056007330877807605) [Z0 X7 Z8 Z9 Z10 X11]
+ (0.056007330877807605) [Z1 Y6 Z7 Z8 Z9 Y10]
+ (0.056007330877807605) [Z1 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661351) [Z0 Y6 Z7 Z8 Z9 Y10]
+ (0.05608468124661351) [Z0 X6 Z7 Z8 Z9 X10]
+ (0.05608468124661351) [Z1 Y7 Z8 Z9 Z10 Y11]
+ (0.05608468124661351) [Z1 X7 Z8 Z9 Z10 X11]
+ (-6.631277929147388e-05) [Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-6.631277929147388e-05) [X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.631277929147385e-05) [Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-6.631277929147385e-05) [X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.5950860072975324e-05) [Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.5950860072975324e-05) [X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.595086007297532e-05) [Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.595086007297532e-05) [X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.04274327701378253) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.04274327701378253) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (0.04274327701378254) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (0.04274327701378254) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (-0.04764261217638312) [Y4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-0.04764261217638312) [X4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-0.04764261217638312) [Y5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-0.04764261217638312) [X5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-0.04171881383982176) [Y4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-0.04171881383982176) [X4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-0.04171881383982176) [Y5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-0.04171881383982176) [X5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-0.03956441632289339) [Y4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-0.03956441632289339) [X4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-0.03956441632289339) [Y5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03956441632289339) [X5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.03935916802205311) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (-0.03935916802205311) [X2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (-0.03935916802205311) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (-0.03935916802205311) [X3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (-0.03931805194719759) [Y4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.03931805194719759) [X4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.03931805194719759) [Y5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.03931805194719759) [X5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.035608378988312456) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.035608378988312456) [X2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.029903789512624835) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (-0.029903789512624835) [X2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (-0.029903789512624835) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (-0.029903789512624835) [X3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (-0.028730779551905488) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.028730779551905488) [X2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.028730779551905488) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.028730779551905488) [X3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.02563723829602685) [Y4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-0.02563723829602685) [X4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-0.02563723829602685) [Y5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-0.02563723829602685) [X5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-0.024755463292890953) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (-0.024755463292890953) [X2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (-0.024755463292890953) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (-0.024755463292890953) [X3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (-0.024282117354693024) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.024282117354693024) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.023145130929529082) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (-0.023145130929529082) [X5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (-0.02252844019601298) [Y4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.02252844019601298) [X4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.021433810721600947) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (-0.021433810721600947) [X2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (-0.021433810721600947) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (-0.021433810721600947) [X3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251617) [Y3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.019257505095251617) [X3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.019028242443847324) [Y2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (-0.019028242443847324) [X2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (-0.01888903030494289) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-0.01888903030494289) [X2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-0.01888903030494289) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.01888903030494289) [X3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.016024603689179514) [Y5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-0.016024603689179514) [X5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-0.015225630757226608) [Y2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (-0.015225630757226608) [X2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (-0.01460370472916215) [Y3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.01460370472916215) [X3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.014564531231173017) [Y6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.014564531231173017) [X6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.011756013419819245) [Y2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.011756013419819245) [X2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.009841749246962597) [Y3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (-0.009841749246962597) [X3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (-0.009612634606847338) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-0.009612634606847338) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-0.009612634606847338) [Y5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-0.009612634606847338) [X5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-0.008469978791023887) [Y3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (-0.008469978791023887) [X3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (-0.007306759928832958) [Y4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-0.007306759928832958) [X4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0059237983365613475) [Y5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (-0.0059237983365613475) [X5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (-0.005652620978017338) [Y0 Y1 X2 Z3 Z4 Z5 Z6 X7]
+ (-0.005652620978017338) [X0 X1 Y2 Z3 Z4 Z5 Z6 Y7]
+ (-0.005368659358109485) [Y2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.005368659358109485) [X2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.004158797381840035) [Y2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.004158797381840035) [X2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0033566705638328875) [Y1 Z2 Z3 Z4 Z5 X6 X8 Y9]
+ (-0.0033566705638328875) [Y1 Z2 Z3 Z4 Z5 Y6 Y8 Y9]
+ (-0.0033566705638328875) [X1 Z2 Z3 Z4 Z5 X6 X8 X9]
+ (-0.0033566705638328875) [X1 Z2 Z3 Z4 Z5 Y6 Y8 X9]
+ (-0.003267513854423551) [Y1 Z2 Z3 Z4 Z5 X6 X12 Y13]
+ (-0.003267513854423551) [Y1 Z2 Z3 Z4 Z5 Y6 Y12 Y13]
+ (-0.003267513854423551) [X1 Z2 Z3 Z4 Z5 X6 X12 X13]
+ (-0.003267513854423551) [X1 Z2 Z3 Z4 Z5 Y6 Y12 X13]
+ (-0.0027790267990255553) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (-0.0027790267990255553) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (-0.002686040977806626) [Y0 Z1 Z2 Z3 X4 X10 Z11 Y12]
+ (-0.002686040977806626) [X0 Z1 Z2 Z3 Y4 Y10 Z11 X12]
+ (-0.002686040977806626) [Y1 Z2 Z3 Z4 X5 X11 Z12 Y13]
+ (-0.002686040977806626) [X1 Z2 Z3 Z4 Y5 Y11 Z12 X13]
+ (-0.00229395661135245) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-0.00229395661135245) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-0.00229395661135245) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-0.00229395661135245) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (-0.0009581655836696616) [Y0 Z1 Z2 Z3 Z4 X5 X11 Y12]
+ (-0.0009581655836696616) [Y0 Z1 Z2 Z3 Z4 Y5 Y11 Y12]
+ (-0.0009581655836696616) [X0 Z1 Z2 Z3 Z4 X5 X11 X12]
+ (-0.0009581655836696616) [X0 Z1 Z2 Z3 Z4 Y5 Y11 X12]
+ (-0.0009581655836696616) [Y1 Z2 Z3 X4 X10 Z11 Z12 Y13]
+ (-0.0009581655836696616) [Y1 Z2 Z3 Y4 Y10 Z11 Z12 Y13]
+ (-0.0009581655836696616) [X1 Z2 Z3 X4 X10 Z11 Z12 X13]
+ (-0.0009581655836696616) [X1 Z2 Z3 Y4 Y10 Z11 Z12 X13]
+ (-0.00024636437569579994) [Y5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00024636437569579994) [X5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303549303) [Y1 Z2 Z3 Z4 Z5 X6 X10 Y11]
+ (-0.00013840177303549303) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Y11]
+ (-0.00013840177303549303) [X1 Z2 Z3 Z4 Z5 X6 X10 X11]
+ (-0.00013840177303549303) [X1 Z2 Z3 Z4 Z5 Y6 Y10 X11]
+ (-7.735036880591525e-05) [Y0 Y1 X6 Z7 Z8 Z9 Z10 X11]
+ (-7.735036880591525e-05) [X0 X1 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585307542554e-05) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.6103585307542554e-05) [Z0 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.6103585307542554e-05) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.6103585307542554e-05) [Z1 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808797030116e-05) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.5316808797030116e-05) [Z0 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.5316808797030116e-05) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5316808797030116e-05) [Z1 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-9.80610277626595e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-9.80610277626595e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-9.80610277626595e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-9.80610277626595e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-7.089799468317785e-06) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.089799468317785e-06) [Z2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.089799468317785e-06) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.089799468317785e-06) [Z3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-6.652209670259161e-06) [Z0 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.652209670259161e-06) [Z0 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-6.652209670259161e-06) [Z1 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.652209670259161e-06) [Z1 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851834696979e-06) [Z0 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.481851834696979e-06) [Z0 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-6.481851834696979e-06) [Z1 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-6.481851834696979e-06) [Z1 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-5.07148073699972e-06) [Y5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-5.07148073699972e-06) [Y5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-5.07148073699972e-06) [X5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-5.07148073699972e-06) [X5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-4.7346220392662305e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-4.7346220392662305e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-4.7346220392662305e-06) [Y5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-4.7346220392662305e-06) [X5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-4.7288431477002475e-06) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-4.7288431477002475e-06) [Z2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-4.7288431477002475e-06) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.7288431477002475e-06) [Z3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.25322422604367e-06) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-4.25322422604367e-06) [Z4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.7696594524189586e-06) [Z6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.7696594524189586e-06) [Z6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.5443954297023175e-06) [Y2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-3.5443954297023175e-06) [Y2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-3.5443954297023175e-06) [X2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-3.5443954297023175e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-3.5443954297023175e-06) [Y3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-3.5443954297023175e-06) [Y3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-3.5443954297023175e-06) [X3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-3.5443954297023175e-06) [X3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-2.3609563206175377e-06) [Y2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563206175377e-06) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-2.3609563206175377e-06) [X2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-2.3609563206175377e-06) [X2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-2.103215604947479e-06) [Z2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.103215604947479e-06) [Z2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.103215604947479e-06) [Z3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.103215604947479e-06) [Z3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220985147153e-06) [Z2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.0111220985147153e-06) [Z2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.0111220985147153e-06) [Z3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.0111220985147153e-06) [Z3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9429468368813536e-06) [Z4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.9429468368813536e-06) [Z4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9429468368813536e-06) [Z5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9429468368813536e-06) [Z5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174773757497e-06) [Z4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.6541174773757497e-06) [Z4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.6541174773757497e-06) [Z5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.6541174773757497e-06) [Z5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.5224930677539133e-06) [Y2 Z3 Z4 X5 X7 Z8 Z9 Y10]
+ (-1.5224930677539133e-06) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Y10]
+ (-1.5224930677539133e-06) [X2 Z3 Z4 X5 X7 Z8 Z9 X10]
+ (-1.5224930677539133e-06) [X2 Z3 Z4 Y5 Y7 Z8 Z9 X10]
+ (-1.5224930677539133e-06) [Y3 X4 X6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930677539133e-06) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-1.5224930677539133e-06) [X3 X4 X6 Z7 Z8 Z9 Z10 X11]
+ (-1.5224930677539133e-06) [X3 Y4 Y6 Z7 Z8 Z9 Z10 X11]
+ (-1.2283337825698148e-06) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337825698148e-06) [Y4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-1.2283337825698148e-06) [X4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-1.2283337825698148e-06) [X4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-7.988770289422174e-07) [Y2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (-7.988770289422174e-07) [X2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (-7.988770289422174e-07) [Y3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (-7.988770289422174e-07) [X3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (-7.867765105124391e-07) [Y0 X1 X5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765105124391e-07) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-7.867765105124391e-07) [X0 X1 X5 Z6 Z7 Z8 Z9 X10]
+ (-7.867765105124391e-07) [X0 Y1 Y5 Z6 Z7 Z8 Z9 X10]
+ (-7.18999097615257e-07) [Y1 Y2 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.18999097615257e-07) [X1 X2 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-6.175246207788072e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X11 X12]
+ (-6.175246207788072e-07) [X1 Z2 Z3 Z4 Z5 X6 Y11 Y12]
+ (-5.471647745231822e-07) [Y0 Z1 Z2 Y3 X10 Z11 Z12 X13]
+ (-5.471647745231822e-07) [X0 Z1 Z2 X3 Y10 Z11 Z12 Y13]
+ (-4.5614471802440865e-07) [Y2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (-4.5614471802440865e-07) [X2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (-4.5614471802440865e-07) [Y3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (-4.5614471802440865e-07) [X3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (-4.523389678416751e-07) [Y1 Y2 X5 Z6 Z7 Z8 Z9 X10]
+ (-4.523389678416751e-07) [X1 X2 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-3.427323109178087e-07) [Y2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (-3.427323109178087e-07) [X2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (-3.427323109178087e-07) [Y3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (-3.427323109178087e-07) [X3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (-3.3281393509747905e-07) [Y6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-3.3281393509747905e-07) [Y6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-3.3281393509747905e-07) [X6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-3.3281393509747905e-07) [X6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-3.0868265654637287e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X12 X13]
+ (-3.0868265654637287e-07) [X0 Z1 Z2 Z3 Z4 X5 Y12 Y13]
+ (-2.888293595056035e-07) [Y4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-2.888293595056035e-07) [Y4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-2.888293595056035e-07) [X4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.888293595056035e-07) [X4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-2.3713289481715648e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X8 X9]
+ (-2.3713289481715648e-07) [X0 Z1 Z2 Z3 Z4 X5 Y8 Y9]
+ (-1.8394209156656232e-07) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-1.8394209156656232e-07) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-8.05744659668815e-08) [Y0 Z1 Z2 Z3 Z4 X5 X10 Y11]
+ (-8.05744659668815e-08) [X0 Z1 Z2 Z3 Z4 Y5 Y10 X11]
+ (4.537178095191372e-08) [X2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.537178095191372e-08) [Y2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.537178095191372e-08) [X3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (4.537178095191372e-08) [Y3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (8.05744659668815e-08) [Y0 Z1 Z2 Z3 Z4 Y5 X10 X11]
+ (8.05744659668815e-08) [X0 Z1 Z2 Z3 Z4 X5 Y10 Y11]
+ (9.209350643276405e-08) [Y2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350643276405e-08) [Y2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (9.209350643276405e-08) [X2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (9.209350643276405e-08) [X2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.703578355621829e-07) [Y0 X1 X7 Z8 Z9 Z10 Z11 Y12]
+ (1.703578355621829e-07) [Y0 Y1 Y7 Z8 Z9 Z10 Z11 Y12]
+ (1.703578355621829e-07) [X0 X1 X7 Z8 Z9 Z10 Z11 X12]
+ (1.703578355621829e-07) [X0 Y1 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.8394209156656232e-07) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (1.8394209156656232e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
+ (2.3713289481715648e-07) [Y0 Z1 Z2 Z3 Z4 X5 X8 Y9]
+ (2.3713289481715648e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y8 X9]
+ (3.0868265654637287e-07) [Y0 Z1 Z2 Z3 Z4 X5 X12 Y13]
+ (3.0868265654637287e-07) [X0 Z1 Z2 Z3 Z4 Y5 Y12 X13]
+ (4.523389678416751e-07) [Y1 X2 X5 Z6 Z7 Z8 Z9 Y10]
+ (4.523389678416751e-07) [X1 Y2 Y5 Z6 Z7 Z8 Z9 X10]
+ (5.471647745231822e-07) [Y0 Z1 Z2 X3 X10 Z11 Z12 Y13]
+ (5.471647745231822e-07) [X0 Z1 Z2 Y3 Y10 Z11 Z12 X13]
+ (6.175246207788072e-07) [Y1 Z2 Z3 Z4 Z5 X6 X11 Y12]
+ (6.175246207788072e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y11 X12]
+ (7.18999097615257e-07) [Y1 X2 X7 Z8 Z9 Z10 Z11 Y12]
+ (7.18999097615257e-07) [X1 Y2 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.3304731888191346e-06) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (1.3304731888191346e-06) [Y4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (1.3304731888191346e-06) [X4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (1.3304731888191346e-06) [X4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (1.6288532437453868e-06) [Y2 Z3 X4 X6 Z7 Z8 Z9 Y10]
+ (1.6288532437453868e-06) [X2 Z3 Y4 Y6 Z7 Z8 Z9 X10]
+ (1.6288532437453868e-06) [Y3 Z4 X5 X7 Z8 Z9 Z10 Y11]
+ (1.6288532437453868e-06) [X3 Z4 Y5 Y7 Z8 Z9 Z10 X11]
+ (1.6893489516804295e-06) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (1.6893489516804295e-06) [X2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (1.6893489516804295e-06) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (1.6893489516804295e-06) [X3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (2.7455184007601007e-06) [Y2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (2.7455184007601007e-06) [Y2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (2.7455184007601007e-06) [X2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (2.7455184007601007e-06) [X2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (2.7455184007601007e-06) [Y3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (2.7455184007601007e-06) [Y3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (2.7455184007601007e-06) [X3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (2.7455184007601007e-06) [X3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (3.2118420194343436e-06) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420194343436e-06) [Y2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (3.2118420194343436e-06) [X2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (3.2118420194343436e-06) [X2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (3.2118420194343436e-06) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420194343436e-06) [Y3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (3.2118420194343436e-06) [X3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (3.2118420194343436e-06) [X3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (3.3131455005084565e-06) [Y6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (3.3131455005084565e-06) [Y6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (3.3131455005084565e-06) [X6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (3.3131455005084565e-06) [X6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (3.334331289818175e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (3.334331289818175e-06) [X4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (4.183932559849535e-06) [Y6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (4.183932559849535e-06) [X6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (7.735036880591525e-05) [Y0 X1 X6 Z7 Z8 Z9 Z10 Y11]
+ (7.735036880591525e-05) [X0 Y1 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.00024636437569579994) [Y5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00024636437569579994) [X5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.0004458535128840983) [Y0 Z1 X2 X6 Z7 Z8 Z9 Y10]
+ (0.0004458535128840983) [X0 Z1 Y2 Y6 Z7 Z8 Z9 X10]
+ (0.0004458535128840983) [Y1 Z2 X3 X7 Z8 Z9 Z10 Y11]
+ (0.0004458535128840983) [X1 Z2 Y3 Y7 Z8 Z9 Z10 X11]
+ (0.0005940221543005327) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005327) [Y0 Z1 Y2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005327) [X0 Z1 X2 Y7 Z8 Z9 Z10 Y11]
+ (0.0005940221543005327) [X0 Z1 X2 X7 Z8 Z9 Z10 X11]
+ (0.0005940221543005327) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005327) [Y1 Z2 Y3 X6 Z7 Z8 Z9 X10]
+ (0.0005940221543005327) [X1 Z2 X3 Y6 Z7 Z8 Z9 Y10]
+ (0.0005940221543005327) [X1 Z2 X3 X6 Z7 Z8 Z9 X10]
+ (0.0008533856254125475) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (0.0008533856254125475) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (0.0008533856254125475) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (0.0008533856254125475) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (0.0010435246534907473) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z13]
+ (0.0010435246534907473) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z13]
+ (0.0010435246534907473) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z12]
+ (0.0010435246534907473) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z12]
+ (0.0012803060973496571) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z9]
+ (0.0012803060973496571) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z9]
+ (0.0012803060973496571) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z8]
+ (0.0012803060973496571) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z8]
+ (0.0013038004788126917) [Y0 Z1 Z2 Z3 Y4 Y10 Z11 Y12]
+ (0.0013038004788126917) [X0 Z1 Z2 Z3 X4 X10 Z11 X12]
+ (0.0013038004788126917) [Y1 Z2 Z3 Z4 Y5 Y11 Z12 Y13]
+ (0.0013038004788126917) [X1 Z2 Z3 Z4 X5 X11 Z12 X13]
+ (0.0022619660624823533) [Y0 Z1 Z2 Z3 Y4 Y11 Z12 Y13]
+ (0.0022619660624823533) [Y0 Z1 Z2 Z3 Y4 X11 Z12 X13]
+ (0.0022619660624823533) [X0 Z1 Z2 Z3 X4 Y11 Z12 Y13]
+ (0.0022619660624823533) [X0 Z1 Z2 Z3 X4 X11 Z12 X13]
+ (0.0022619660624823533) [Y1 Z2 Z3 Z4 Y5 Y10 Z11 Y12]
+ (0.0022619660624823533) [Y1 Z2 Z3 Z4 Y5 X10 Z11 X12]
+ (0.0022619660624823533) [X1 Z2 Z3 Z4 X5 Y10 Z11 Y12]
+ (0.0022619660624823533) [X1 Z2 Z3 Z4 X5 X10 Z11 X12]
+ (0.003989841456619317) [Y0 Z1 Z2 Z3 Y4 X10 Z11 X12]
+ (0.003989841456619317) [X0 Z1 Z2 Z3 X4 Y10 Z11 Y12]
+ (0.003989841456619317) [Y1 Z2 Z3 Z4 Y5 X11 Z12 X13]
+ (0.003989841456619317) [X1 Z2 Z3 Z4 X5 Y11 Z12 Y13]
+ (0.004158797381840035) [Y2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.004158797381840035) [X2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.004311038507914299) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z12]
+ (0.004311038507914299) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z12]
+ (0.004311038507914299) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z13]
+ (0.004311038507914299) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z13]
+ (0.004636976661182545) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z8]
+ (0.004636976661182545) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z8]
+ (0.004636976661182545) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z9]
+ (0.004636976661182545) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z9]
+ (0.005114473831660387) [Y0 Z1 Z2 X3 X7 Z8 Z9 Y10]
+ (0.005114473831660387) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Y10]
+ (0.005114473831660387) [X0 Z1 Z2 X3 X7 Z8 Z9 X10]
+ (0.005114473831660387) [X0 Z1 Z2 Y3 Y7 Z8 Z9 X10]
+ (0.005114473831660387) [Y1 X2 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660387) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.005114473831660387) [X1 X2 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005114473831660387) [X1 Y2 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.0052415353828038705) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z11]
+ (0.0052415353828038705) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z11]
+ (0.0052415353828038705) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z10]
+ (0.0052415353828038705) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z10]
+ (0.0052626424730768204) [Y0 Z1 Y2 X6 Z7 Z8 Z9 X10]
+ (0.0052626424730768204) [X0 Z1 X2 Y6 Z7 Z8 Z9 Y10]
+ (0.0052626424730768204) [Y1 Z2 Y3 X7 Z8 Z9 Z10 X11]
+ (0.0052626424730768204) [X1 Z2 X3 Y7 Z8 Z9 Z10 Y11]
+ (0.005368659358109485) [Y2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.005368659358109485) [X2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.0053799371558393635) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z10]
+ (0.0053799371558393635) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z10]
+ (0.0053799371558393635) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Z11]
+ (0.0053799371558393635) [X1 Z2 Z3 Z4 Z5 Z6 X7 Z11]
+ (0.005652620978017338) [Y0 X1 X2 Z3 Z4 Z5 Z6 Y7]
+ (0.005652620978017338) [X0 Y1 Y2 Z3 Z4 Z5 Z6 X7]
+ (0.005708495985960919) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Y10]
+ (0.005708495985960919) [X0 Z1 X2 X6 Z7 Z8 Z9 X10]
+ (0.005708495985960919) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Y11]
+ (0.005708495985960919) [X1 Z2 X3 X7 Z8 Z9 Z10 X11]
+ (0.0059237983365613475) [Y5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (0.0059237983365613475) [X5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (0.007306759928832958) [Y4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (0.007306759928832958) [X4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (0.008469978791023887) [Y3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (0.008469978791023887) [X3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (0.009841749246962597) [Y3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (0.009841749246962597) [X3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (0.011756013419819245) [Y2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.011756013419819245) [X2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.014564531231173017) [Y6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.014564531231173017) [X6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.01460370472916215) [Y3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.01460370472916215) [X3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.015225630757226608) [Y2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (0.015225630757226608) [X2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (0.016024603689179514) [Y5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (0.016024603689179514) [X5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (0.019028242443847324) [Y2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (0.019028242443847324) [X2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (0.019257505095251617) [Y3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.019257505095251617) [X3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.045879470781297955) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (0.045879470781297955) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-0.36937089366156195) [Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.36937089366156195) [X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.36937089366156184) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.36937089366156184) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.28164257767023054) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.28164257767023054) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.2816425776702304) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.2816425776702304) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.09065144207036477) [Z0 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.09065144207036477) [Z0 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.09065144207036477) [Z1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.09065144207036477) [Z1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863624) [Z0 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.08684737589863624) [Z0 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.08684737589863624) [Z1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.08684737589863624) [Z1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.07635021950635013) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.07635021950635013) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.07635021950635013) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.07635021950635013) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214027) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.06752385099214027) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.06752385099214027) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.06752385099214027) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.035608378988312456) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.035608378988312456) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0349033433736617) [Z2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0349033433736617) [Z2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0349033433736617) [Z3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0349033433736617) [Z3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883829968) [Z2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.024591860883829968) [Z2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.024591860883829968) [Z3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.024591860883829968) [Z3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.024282117354693024) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.024282117354693024) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.023145130929529086) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (-0.023145130929529086) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (-0.022528440196012984) [Z4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.022528440196012984) [Z4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.01953805031131474) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (-0.01953805031131474) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (-0.01953805031131474) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (-0.01953805031131474) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (-0.017091553155898845) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (-0.017091553155898845) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (-0.017091553155898845) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (-0.017091553155898845) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (-0.016024603689179514) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-0.016024603689179514) [Y4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-0.016024603689179514) [X4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-0.016024603689179514) [X4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-0.010311482489831736) [Y2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831736) [Y2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.010311482489831736) [X2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.010311482489831736) [X2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.009841749246962597) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962597) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.009841749246962597) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.009841749246962597) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.00882636851420985) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.00882636851420985) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.00882636851420985) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.00882636851420985) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.00854199662545485) [Y2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00854199662545485) [Y2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00854199662545485) [X2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00854199662545485) [X2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.00854199662545485) [Y3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00854199662545485) [Y3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00854199662545485) [X3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00854199662545485) [X3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.008469978791023887) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023887) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (-0.008469978791023887) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (-0.008469978791023887) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (-0.004668620318776288) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Y11]
+ (-0.004668620318776288) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 X11]
+ (-0.0038764708993369577) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 X10]
+ (-0.0038764708993369577) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Y10]
+ (-0.003804066171728539) [Y0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728539) [Y0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.003804066171728539) [X0 X1 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003804066171728539) [X0 Y1 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003484157300217878) [Y1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003484157300217878) [X1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0033566705638328875) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X8 X9]
+ (-0.0033566705638328875) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y8 Y9]
+ (-0.003267513854423551) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X12 X13]
+ (-0.003267513854423551) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y12 Y13]
+ (-0.0021413612231015785) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y11]
+ (-0.0021413612231015785) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X11]
+ (-0.0017278753941369644) [Y0 Z1 Z2 Z3 Z4 X5 X10 Z11 Z12 Y13]
+ (-0.0017278753941369644) [X0 Z1 Z2 Z3 Z4 Y5 Y10 Z11 Z12 X13]
+ (-0.0016407548553124104) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0016407548553124104) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214168786) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0014528843214168786) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-0.0014528843214168786) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0014528843214168786) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0007870896771024509) [Y1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 X10]
+ (-0.0007870896771024509) [X1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-0.0005192743499487768) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 X10]
+ (-0.0005192743499487768) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Y10]
+ (-0.00019400857029757058) [Y1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.00019400857029757058) [X1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.00013840177303549303) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 X11]
+ (-0.00013840177303549303) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Y11]
+ (-7.141625221153723e-05) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Y10]
+ (-7.141625221153723e-05) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 X10]
+ (-7.141625221153723e-05) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.141625221153723e-05) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-5.07148073699972e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-5.07148073699972e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-3.1513463114993e-06) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 X11]
+ (-3.1513463114993e-06) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Y11]
+ (-3.0882507116779086e-06) [Y2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-3.0882507116779086e-06) [X2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-2.9885117066681004e-06) [Y3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.9885117066681004e-06) [X3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.8742990717144027e-06) [Y3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-2.8742990717144027e-06) [X3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-2.3609563206175377e-06) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-2.3609563206175377e-06) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.3002946563775575e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Y12]
+ (-1.3002946563775575e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 X12]
+ (-1.146837650884536e-06) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-1.146837650884536e-06) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-1.146837650884536e-06) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-1.146837650884536e-06) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.352332103961421e-07) [Y0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.352332103961421e-07) [X0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.352332103961421e-07) [Y1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.352332103961421e-07) [X1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.091637200015722e-07) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637200015722e-07) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Y10]
+ (-8.091637200015722e-07) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637200015722e-07) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 X10]
+ (-8.091637200015722e-07) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637200015722e-07) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-8.091637200015722e-07) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.091637200015722e-07) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-8.074305986783706e-07) [Y0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.074305986783706e-07) [X0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.074305986783706e-07) [Y1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.074305986783706e-07) [X1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.900128987246384e-07) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-7.900128987246384e-07) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-7.900128987246384e-07) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.900128987246384e-07) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.867765105124391e-07) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-7.867765105124391e-07) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-7.560692465726185e-07) [Y0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692465726185e-07) [Y0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-7.560692465726185e-07) [X0 Z1 Z2 X3 X7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692465726185e-07) [X0 Z1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 X12]
+ (-7.560692465726185e-07) [Y1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692465726185e-07) [Y1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-7.560692465726185e-07) [X1 X2 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.560692465726185e-07) [X1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-4.997018422682816e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 Y12]
+ (-4.997018422682816e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Y12]
+ (-4.997018422682816e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X11 X12]
+ (-4.997018422682816e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 X12]
+ (-4.997018422682816e-07) [Y1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 Y13]
+ (-4.997018422682816e-07) [Y1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 Y13]
+ (-4.997018422682816e-07) [X1 Z2 Z3 Z4 Z5 X6 X10 Z11 Z12 X13]
+ (-4.997018422682816e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Z12 X13]
+ (-3.568247521598973e-07) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.568247521598973e-07) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.568247521598973e-07) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.568247521598973e-07) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.376739308829635e-07) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.376739308829635e-07) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.376739308829635e-07) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.376739308829635e-07) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-3.376739308829635e-07) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.376739308829635e-07) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-3.376739308829635e-07) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (-3.376739308829635e-07) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 X10]
+ (-2.888293595056035e-07) [Y4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-2.888293595056035e-07) [X4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.686381545752522e-07) [Y3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (-2.686381545752522e-07) [X3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-1.703578355621829e-07) [Y0 X1 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.703578355621829e-07) [X0 Y1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-9.209350643276405e-08) [Y2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.209350643276405e-08) [X2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244331875e-08) [Y0 Z1 Y2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773244331875e-08) [Y0 Z1 Y2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244331875e-08) [X0 Z1 X2 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.379773244331875e-08) [X0 Z1 X2 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.379773244331875e-08) [Y1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773244331875e-08) [Y1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-8.379773244331875e-08) [X1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-8.379773244331875e-08) [X1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253794761668e-08) [X0 Z1 Y2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-1.9742253794761668e-08) [Y0 Z1 X2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.9742253794761668e-08) [X1 Z2 Y3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.9742253794761668e-08) [Y1 Z2 X3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0474716555571406e-08) [Y0 Z1 X2 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.0474716555571406e-08) [X0 Z1 Y2 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.0474716555571406e-08) [Y1 Z2 X3 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.0474716555571406e-08) [X1 Z2 Y3 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (9.209350643276405e-08) [Y2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (9.209350643276405e-08) [X2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.0717282184504789e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X10 Z11 X12]
+ (1.0717282184504789e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y10 Z11 Y12]
+ (1.0717282184504789e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X11 Z12 X13]
+ (1.0717282184504789e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y11 Z12 Y13]
+ (1.2004287495473181e-07) [X0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 X12]
+ (1.2004287495473181e-07) [Y0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 Y12]
+ (1.2004287495473181e-07) [X1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 X13]
+ (1.2004287495473181e-07) [Y1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 Y13]
+ (1.703578355621829e-07) [Y0 Y1 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.703578355621829e-07) [X0 X1 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.3120943054945366e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X10 Z11 X12]
+ (2.3120943054945366e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y10 Z11 Y12]
+ (2.3120943054945366e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X11 Z12 X13]
+ (2.3120943054945366e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y11 Z12 Y13]
+ (2.686381545752522e-07) [Y3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (2.686381545752522e-07) [X3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (2.888293595056035e-07) [Y4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.888293595056035e-07) [X4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.0922506165386433e-07) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506165386433e-07) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 X11]
+ (4.0922506165386433e-07) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Y11]
+ (4.0922506165386433e-07) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 X11]
+ (4.0922506165386433e-07) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506165386433e-07) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 X10]
+ (4.0922506165386433e-07) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Y10]
+ (4.0922506165386433e-07) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 X10]
+ (4.444597854580653e-07) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Y10]
+ (4.444597854580653e-07) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 X10]
+ (4.444597854580653e-07) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Y11]
+ (4.444597854580653e-07) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 X11]
+ (4.684915095703966e-07) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 X10]
+ (4.684915095703966e-07) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Y10]
+ (4.684915095703966e-07) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 X11]
+ (4.684915095703966e-07) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Y11]
+ (7.246974426238551e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y11 Z12 Y13]
+ (7.246974426238551e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X11 Z12 X13]
+ (7.246974426238551e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y11 Z12 Y13]
+ (7.246974426238551e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X11 Z12 X13]
+ (7.246974426238551e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Y12]
+ (7.246974426238551e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 X12]
+ (7.246974426238551e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Y12]
+ (7.246974426238551e-07) [X1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 X12]
+ (7.867765105124391e-07) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (7.867765105124391e-07) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (1.3002946563775575e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 X12]
+ (1.3002946563775575e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Y12]
+ (2.3609563206175377e-06) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (2.3609563206175377e-06) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (2.8742990717144027e-06) [Y3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (2.8742990717144027e-06) [X3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (2.8836765763748783e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (2.8836765763748783e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (2.947356012025388e-06) [Y2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.947356012025388e-06) [X2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.947356012025388e-06) [Y3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.947356012025388e-06) [X3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117066681004e-06) [Y3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (2.9885117066681004e-06) [X3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (3.0882507116779086e-06) [Y2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (3.0882507116779086e-06) [X2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (3.1513463114993e-06) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Y11]
+ (3.1513463114993e-06) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 X11]
+ (3.8462016717283276e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (3.8462016717283276e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (3.8462016717283276e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (3.8462016717283276e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (5.07148073699972e-06) [Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (5.07148073699972e-06) [X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (5.10552672255154e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (5.10552672255154e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (5.10552672255154e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (5.10552672255154e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (5.146496328105885e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (5.146496328105885e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (5.146496328105885e-06) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (5.146496328105885e-06) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (5.1593505024873306e-06) [Y2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (5.1593505024873306e-06) [X2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (5.1593505024873306e-06) [Y3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.1593505024873306e-06) [X3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.427988657062582e-06) [Y2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.427988657062582e-06) [X2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.427988657062582e-06) [Y3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.427988657062582e-06) [X3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (5.935867718693488e-06) [Y2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.935867718693488e-06) [X2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.935867718693488e-06) [Y3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.935867718693488e-06) [X3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273348883681e-06) [Y2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (7.253273348883681e-06) [X2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (7.979825794265945e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (7.979825794265945e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (7.979825794265945e-06) [Y3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (7.979825794265945e-06) [X3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (4.2055484112167064e-05) [Y2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.2055484112167064e-05) [X2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.2055484112167064e-05) [Y3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (4.2055484112167064e-05) [X3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00013840177303549303) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Y11]
+ (0.00013840177303549303) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 X11]
+ (0.0001878705338955317) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0001878705338955317) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0001878705338955317) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0001878705338955317) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.00019400857029757058) [Y1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Y12]
+ (0.00019400857029757058) [X1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 X12]
+ (0.00024636437569579994) [Y4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024636437569579994) [Y4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.00024636437569579994) [X4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00024636437569579994) [X4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0005192743499487768) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Y10]
+ (0.0005192743499487768) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 X10]
+ (0.0007156734248909137) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007156734248909137) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0007156734248909137) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007156734248909137) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024509) [Y1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Y10]
+ (0.0007870896771024509) [X1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 X10]
+ (0.0015324835230730639) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Y10]
+ (0.0015324835230730639) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 X10]
+ (0.0015324835230730639) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Y11]
+ (0.0015324835230730639) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 X11]
+ (0.0016407548553124104) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.0016407548553124104) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.0017278753941369644) [Y0 Z1 Z2 Z3 Z4 Y5 X10 Z11 Z12 X13]
+ (0.0017278753941369644) [X0 Z1 Z2 Z3 Z4 X5 Y10 Z11 Z12 Y13]
+ (0.0024464971554158913) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (0.0024464971554158913) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (0.0024464971554158913) [X3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (0.0024464971554158913) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (0.003267513854423551) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X12 Y13]
+ (0.003267513854423551) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y12 X13]
+ (0.0033566705638328875) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X8 Y9]
+ (0.0033566705638328875) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y8 X9]
+ (0.003484157300217878) [Y1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.003484157300217878) [X1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0038764708993369577) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Y10]
+ (0.0038764708993369577) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 X10]
+ (0.004668620318776288) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 X11]
+ (0.004668620318776288) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278136) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Y10]
+ (0.004767272188278136) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 X10]
+ (0.004767272188278136) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Y11]
+ (0.004767272188278136) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 X11]
+ (0.005286546538226912) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Y10]
+ (0.005286546538226912) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 X10]
+ (0.005286546538226912) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Y11]
+ (0.005286546538226912) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 X11]
+ (0.0054089544224100216) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Y10]
+ (0.0054089544224100216) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 X10]
+ (0.0054089544224100216) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Y11]
+ (0.0054089544224100216) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 X11]
+ (0.0059237983365613475) [Y4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (0.0059237983365613475) [Y4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (0.0059237983365613475) [X4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (0.0059237983365613475) [X4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (0.010715508469796766) [Y2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010715508469796766) [X2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010715508469796766) [Y3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010715508469796766) [X3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.010757563953908932) [Y2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.010757563953908932) [X2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.010757563953908932) [Y3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010757563953908932) [X3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.014603704729162148) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.014603704729162148) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.014603704729162148) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.014603704729162148) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.019299560579363786) [Y2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363786) [Y2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363786) [X2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019299560579363786) [X2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019299560579363786) [Y3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363786) [Y3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.019299560579363786) [X3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.019299560579363786) [X3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.058591988733862066) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.058591988733862066) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (5.7759505279497496e-05) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (5.7759505279497496e-05) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (5.7759505279497523e-05) [Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (5.7759505279497523e-05) [X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.07165035181002642) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10]
+ (0.07165035181002642) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10]
+ (0.07165035181002642) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.07165035181002642) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.019257505095251617) [Y2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.019257505095251617) [X2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.010311482489831736) [Y2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.010311482489831736) [X2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.00882636851420985) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.00882636851420985) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.007597464029770588) [Y0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.007597464029770588) [X0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.007597464029770588) [Y1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.007597464029770588) [X1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311868) [Y0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311868) [Y0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005733569747311868) [X0 Z1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311868) [X0 Z1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005733569747311868) [Y1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311868) [Y1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.005733569747311868) [X1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005733569747311868) [X1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676599) [Y0 Z1 Y2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.005348051582676599) [X0 Z1 X2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.005348051582676599) [Y1 Z2 Y3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.005348051582676599) [X1 Z2 X3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.003804066171728539) [Y0 Y1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003804066171728539) [X0 X1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002984166168121939) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 Y13]
+ (-0.002984166168121939) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 Y13]
+ (-0.002984166168121939) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X12 X13]
+ (-0.002984166168121939) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y12 X13]
+ (-0.0024464971554158913) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (-0.0024464971554158913) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (-0.002249412447093991) [Y0 Z1 X2 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.002249412447093991) [X0 Z1 Y2 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.002249412447093991) [Y1 Z2 X3 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002249412447093991) [X1 Z2 Y3 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0021413612231015785) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z11]
+ (-0.0021413612231015785) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z11]
+ (-0.0018638942824587205) [Y0 Z1 Y2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587205) [Y0 Z1 Y2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587205) [X0 Z1 X2 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0018638942824587205) [X0 Z1 X2 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0018638942824587205) [Y1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587205) [Y1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0018638942824587205) [X1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0018638942824587205) [X1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0016407548553124104) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124104) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0016407548553124104) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-0.0016407548553124104) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-0.0012223378081538349) [Y0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538349) [Y0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0012223378081538349) [X0 Z1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538349) [X0 Z1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 X12]
+ (-0.0012223378081538349) [Y1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538349) [Y1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0012223378081538349) [X1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0012223378081538349) [X1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.0010283292378562641) [Y0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-0.0010283292378562641) [X0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.0010283292378562641) [Y1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.0010283292378562641) [X1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.1463061454294112e-05) [Y0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.1463061454294112e-05) [X0 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.8742990717144027e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990717144027e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-2.8742990717144027e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-2.8742990717144027e-06) [X2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.3002946563775575e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Y11 Z12 Y13]
+ (-1.3002946563775575e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 X11 Z12 X13]
+ (-1.3002946563775575e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Y11 Z12 Y13]
+ (-1.3002946563775575e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 X11 Z12 X13]
+ (-1.0444941299308184e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Y12]
+ (-1.0444941299308184e-06) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 X12]
+ (-1.0444941299308184e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 Y13]
+ (-1.0444941299308184e-06) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Z12 X13]
+ (-9.956079231193574e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-9.956079231193574e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 X12]
+ (-9.956079231193574e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-9.956079231193574e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Z12 X13]
+ (-8.105515038132544e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 Y12]
+ (-8.105515038132544e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z8 Z9 Z10 Z11 X12]
+ (-8.105515038132544e-07) [Y1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-8.105515038132544e-07) [X1 Z2 Z3 Z4 Z5 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.661347214025191e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Y12]
+ (-7.661347214025191e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 X12]
+ (-7.661347214025191e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 Y13]
+ (-7.661347214025191e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z12 X13]
+ (-7.540341414531721e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Y12]
+ (-7.540341414531721e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 X12]
+ (-7.18999097615257e-07) [Y0 Z1 Z2 Y3 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.18999097615257e-07) [X0 Z1 Z2 X3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-6.876621659100246e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y12]
+ (-6.876621659100246e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X12]
+ (-6.876621659100246e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 Y13]
+ (-6.876621659100246e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z11 Z12 X13]
+ (-6.175246207788072e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 X10 Z11 Z12 X13]
+ (-6.175246207788072e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 X7 Y10 Z11 Z12 Y13]
+ (-4.523389678416751e-07) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-4.523389678416751e-07) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (-3.076732532414349e-07) [Y0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-3.076732532414349e-07) [X0 Z1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.076732532414349e-07) [Y1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-3.076732532414349e-07) [X1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-3.0134714592964116e-07) [Y1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-3.0134714592964116e-07) [X1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.9045998847764625e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 Y12]
+ (-2.9045998847764625e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z10 Z11 X12]
+ (-2.9045998847764625e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 Y13]
+ (-2.9045998847764625e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z9 Z10 Z11 Z12 X13]
+ (-2.666731755107284e-07) [Y0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-2.666731755107284e-07) [X0 Z1 Z2 Z3 Z4 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-2.666731755107284e-07) [Y1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-2.666731755107284e-07) [X1 Z2 Z3 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641930610301e-07) [Y1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Y12]
+ (-1.8505641930610301e-07) [X1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 X12]
+ (1.656930931491598e-07) [Y0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.656930931491598e-07) [X0 Z1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.656930931491598e-07) [Y1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.656930931491598e-07) [X1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.8505641930610301e-07) [Y1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 X12]
+ (1.8505641930610301e-07) [X1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Y12]
+ (2.686381545752522e-07) [Y2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.686381545752522e-07) [Y2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.686381545752522e-07) [X2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.686381545752522e-07) [X2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714592964116e-07) [Y1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (3.0134714592964116e-07) [X1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.523389678416751e-07) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (4.523389678416751e-07) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (4.6704023907880096e-07) [Y0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (4.6704023907880096e-07) [X0 Z1 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (4.6704023907880096e-07) [Y1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (4.6704023907880096e-07) [X1 Z2 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (6.175246207788072e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 X7 X10 Z11 Z12 Y13]
+ (6.175246207788072e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Y7 Y10 Z11 Z12 X13]
+ (7.18999097615257e-07) [Y0 Z1 Z2 X3 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.18999097615257e-07) [X0 Z1 Z2 Y3 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.540341414531721e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 X12]
+ (7.540341414531721e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Y12]
+ (8.949476488220912e-07) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y13]
+ (8.949476488220912e-07) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X13]
+ (1.792493957884883e-06) [Y0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.792493957884883e-06) [Y0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.792493957884883e-06) [X0 X1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.792493957884883e-06) [X0 Y1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (2.8836765763748783e-06) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (2.8836765763748783e-06) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (2.9885117066681004e-06) [Y2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117066681004e-06) [Y2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (2.9885117066681004e-06) [X2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (2.9885117066681004e-06) [X2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (7.253273348883681e-06) [Z2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (7.253273348883681e-06) [Z2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.4017109736734273e-05) [Z0 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.4017109736734273e-05) [Z0 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (1.4017109736734273e-05) [Z1 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.4017109736734273e-05) [Z1 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603694619153e-05) [Z0 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (1.5809603694619153e-05) [Z0 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.5809603694619153e-05) [Z1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (1.5809603694619153e-05) [Z1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0005192743499487769) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487769) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 X11]
+ (0.0005192743499487769) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Y11]
+ (0.0005192743499487769) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 X11]
+ (0.0007870896771024508) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024508) [Y0 Z1 Z2 Z3 Y4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0007870896771024508) [X0 Z1 Z2 Z3 X4 Y5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.0007870896771024508) [X0 Z1 Z2 Z3 X4 X5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.0011726348316441863) [Y0 Z1 Z2 Z3 Y4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0011726348316441863) [X0 Z1 Z2 Z3 X4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0011726348316441863) [Y1 Z2 Z3 Z4 Y5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.0011726348316441863) [X1 Z2 Z3 Z4 X5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0012366478019245606) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z13]
+ (0.0012366478019245606) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z13]
+ (0.0012366478019245606) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z12]
+ (0.0012366478019245606) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z12]
+ (0.0022009640695004502) [Y0 Z1 Z2 Z3 Y4 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0022009640695004502) [X0 Z1 Z2 Z3 X4 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.0022009640695004502) [Y1 Z2 Z3 Z4 Y5 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0022009640695004502) [X1 Z2 Z3 Z4 X5 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798021) [Y0 Z1 Z2 Z3 Y4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798021) [Y0 Z1 Z2 Z3 Y4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798021) [X0 Z1 Z2 Z3 X4 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.002394972639798021) [X0 Z1 Z2 Z3 X4 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002394972639798021) [Y1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798021) [Y1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.002394972639798021) [X1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (0.002394972639798021) [X1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 X12]
+ (0.0024464971554158913) [Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (0.0024464971554158913) [X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (0.003804066171728539) [Y0 X1 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.003804066171728539) [X0 Y1 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.0038764708993369577) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Y11]
+ (0.0038764708993369577) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 X11]
+ (0.0038764708993369577) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Y11]
+ (0.0038764708993369577) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 X11]
+ (0.0042208139700465) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Y10 Z12]
+ (0.0042208139700465) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 X10 Z12]
+ (0.0042208139700465) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Z13]
+ (0.0042208139700465) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Z13]
+ (0.00882636851420985) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.00882636851420985) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (0.010311482489831736) [Y2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.010311482489831736) [X2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.019257505095251617) [Y2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.019257505095251617) [X2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.058591988733862066) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11]
+ (0.058591988733862066) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11]
+ (-1.3987009016740193e-05) [Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.3987009016740193e-05) [X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.398700901674019e-05) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12]
+ (-1.398700901674019e-05) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12]
+ (-0.003484157300217878) [Y0 Z1 Z2 Y3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.003484157300217878) [X0 Z1 Z2 X3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-0.002984166168121939) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 X12 X13]
+ (-0.002984166168121939) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 Y12 Y13]
+ (-0.00019400857029757058) [Y0 Z1 Z2 Z3 Z4 Y5 X6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-0.00019400857029757058) [X0 Z1 Z2 Z3 Z4 X5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061454294112e-05) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.1463061454294112e-05) [Z0 X1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.792493957884883e-06) [Y0 X1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.792493957884883e-06) [X0 Y1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-7.540341414531721e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341414531721e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Y8 X9 Z10 Z11 Z12 X13]
+ (-7.540341414531721e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 Y9 Z10 Z11 Z12 Y13]
+ (-7.540341414531721e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 X8 X9 Z10 Z11 Z12 X13]
+ (-1.8505641930610301e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641930610301e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (-1.8505641930610301e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 Y7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (-1.8505641930610301e-07) [X0 Z1 Z2 Z3 Z4 Z5 X6 X7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714592964116e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.0134714592964116e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (3.0134714592964116e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (3.0134714592964116e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (8.949476488220912e-07) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Y12 Z13]
+ (8.949476488220912e-07) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 X12 Z13]
+ (1.792493957884883e-06) [Y0 Y1 X2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (1.792493957884883e-06) [X0 X1 Y2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029757058) [Y0 Z1 Z2 Z3 Z4 X5 X6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.00019400857029757058) [X0 Z1 Z2 Z3 Z4 Y5 Y6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
+ (0.002984166168121939) [Y0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 X11 X12 Y13]
+ (0.002984166168121939) [X0 Z1 Z2 Z3 Z4 Z5 Z6 Z7 Z8 Z9 Z10 Y11 Y12 X13]
+ (0.003484157300217878) [Y0 Z1 Z2 X3 X4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 Y13]
+ (0.003484157300217878) [X0 Z1 Z2 Y3 Y4 Z5 Z6 Z7 Z8 Z9 Z10 Z11 Z12 X13]
  (-73.1387323135253) [I0]
+ (-0.18066792656583425) [Z6]
+ (-0.1806679265658342) [Z7]
+ (-0.15961432501809883) [Z4]
+ (-0.1596143250180988) [Z5]
+ (0.1741995615505567) [Z3]
+ (0.17419956155055671) [Z2]
+ (0.22757269005453518) [Z0]
+ (0.2275726900545352) [Z1]
+ (-8.194261372453012e-06) [Y4 Y6]
+ (-8.194261372453012e-06) [X4 X6]
+ (7.954413176290946e-06) [Y5 Y7]
+ (7.954413176290946e-06) [X5 X7]
+ (0.11270386920332215) [Z4 Z6]
+ (0.11270386920332215) [Z5 Z7]
+ (0.11952438964682675) [Z0 Z4]
+ (0.11952438964682675) [Z1 Z5]
+ (0.13401715261963718) [Z0 Z6]
+ (0.13401715261963718) [Z1 Z7]
+ (0.13734953064261324) [Z0 Z5]
+ (0.13734953064261324) [Z1 Z4]
+ (0.13766872645852563) [Z2 Z4]
+ (0.13766872645852563) [Z3 Z5]
+ (0.141389052919428) [Z4 Z7]
+ (0.141389052919428) [Z5 Z6]
+ (0.14722943218766155) [Z2 Z5]
+ (0.14722943218766155) [Z3 Z4]
+ (0.14926355147388876) [Z4 Z5]
+ (0.14973486803496924) [Z2 Z6]
+ (0.14973486803496924) [Z3 Z7]
+ (0.1513832716142886) [Z0 Z7]
+ (0.1513832716142886) [Z1 Z6]
+ (0.1543574865722363) [Z6 Z7]
+ (0.1558226905155311) [Z2 Z7]
+ (0.1558226905155311) [Z3 Z6]
+ (0.16756653265461272) [Z0 Z2]
+ (0.16756653265461272) [Z1 Z3]
+ (0.18143991440303883) [Z0 Z3]
+ (0.18143991440303883) [Z1 Z2]
+ (0.19392534613270224) [Z0 Z1]
+ (0.2200397733437608) [Z2 Z3]
+ (-7.037887510297245e-06) [Y5 Z6 Y7]
+ (-7.037887510297245e-06) [X5 Z6 X7]
+ (-7.037887510297242e-06) [Y4 Z5 Y6]
+ (-7.037887510297242e-06) [X4 Z5 X6]
+ (-0.028685183716105844) [Y4 Y5 X6 X7]
+ (-0.028685183716105844) [X4 X5 Y6 Y7]
+ (-0.017825140995786495) [Y0 Y1 X4 X5]
+ (-0.017825140995786495) [X0 X1 Y4 Y5]
+ (-0.017366118994651427) [Y0 Y1 X6 X7]
+ (-0.017366118994651427) [X0 X1 Y6 Y7]
+ (-0.013873381748426113) [Y0 Y1 X2 X3]
+ (-0.013873381748426113) [X0 X1 Y2 Y3]
+ (-0.009560705729135923) [Y2 Y3 X4 X5]
+ (-0.009560705729135923) [X2 X3 Y4 Y5]
+ (-0.006087822480561862) [Y2 Y3 X6 X7]
+ (-0.006087822480561862) [X2 X3 Y6 Y7]
+ (-0.00029219862611105273) [Y1 Y2 X3 X4]
+ (-0.00029219862611105273) [X1 X2 Y3 Y4]
+ (-8.194261372453012e-06) [Z4 Y5 Z6 Y7]
+ (-8.194261372453012e-06) [Z4 X5 Z6 X7]
+ (-2.8909678817879047e-06) [Z0 Y5 Z6 Y7]
+ (-2.8909678817879047e-06) [Z0 X5 Z6 X7]
+ (-2.8909678817879047e-06) [Z1 Y4 Z5 Y6]
+ (-2.8909678817879047e-06) [Z1 X4 Z5 X6]
+ (-1.855120121616294e-06) [Z0 Y4 Z5 Y6]
+ (-1.855120121616294e-06) [Z0 X4 Z5 X6]
+ (-1.855120121616294e-06) [Z1 Y5 Z6 Y7]
+ (-1.855120121616294e-06) [Z1 X5 Z6 X7]
+ (-1.597317197907464e-06) [Z2 Y4 Z5 Y6]
+ (-1.597317197907464e-06) [Z2 X4 Z5 X6]
+ (-1.597317197907464e-06) [Z3 Y5 Z6 Y7]
+ (-1.597317197907464e-06) [Z3 X5 Z6 X7]
+ (-1.0358477601716111e-06) [Y0 X1 X5 Y6]
+ (-1.0358477601716111e-06) [Y0 Y1 Y5 Y6]
+ (-1.0358477601716111e-06) [X0 X1 X5 X6]
+ (-1.0358477601716111e-06) [X0 Y1 Y5 X6]
+ (-9.344557777395828e-07) [Z2 Y5 Z6 Y7]
+ (-9.344557777395828e-07) [Z2 X5 Z6 X7]
+ (-9.344557777395828e-07) [Z3 Y4 Z5 Y6]
+ (-9.344557777395828e-07) [Z3 X4 Z5 X6]
+ (6.628614201678811e-07) [Y2 X3 X5 Y6]
+ (6.628614201678811e-07) [Y2 Y3 Y5 Y6]
+ (6.628614201678811e-07) [X2 X3 X5 X6]
+ (6.628614201678811e-07) [X2 Y3 Y5 X6]
+ (7.954413176290948e-06) [Y4 Z5 Y6 Z7]
+ (7.954413176290948e-06) [X4 Z5 X6 Z7]
+ (0.00029219862611105273) [Y1 X2 X3 Y4]
+ (0.00029219862611105273) [X1 Y2 Y3 X4]
+ (0.006087822480561862) [Y2 X3 X6 Y7]
+ (0.006087822480561862) [X2 Y3 Y6 X7]
+ (0.009560705729135923) [Y2 X3 X4 Y5]
+ (0.009560705729135923) [X2 Y3 Y4 X5]
+ (0.011307274008848168) [Y1 Z2 Z3 Y5]
+ (0.011307274008848168) [X1 Z2 Z3 X5]
+ (0.013873381748426113) [Y0 X1 X2 Y3]
+ (0.013873381748426113) [X0 Y1 Y2 X3]
+ (0.017366118994651427) [Y0 X1 X6 Y7]
+ (0.017366118994651427) [X0 Y1 Y6 X7]
+ (0.017825140995786495) [Y0 X1 X4 Y5]
+ (0.017825140995786495) [X0 Y1 Y4 X5]
+ (0.028685183716105844) [Y4 X5 X6 Y7]
+ (0.028685183716105844) [X4 Y5 Y6 X7]
+ (0.02981242451734576) [Y0 Z1 Z2 Y4]
+ (0.02981242451734576) [X0 Z1 Z2 X4]
+ (0.02981242451734576) [Y1 Z3 Z4 Y5]
+ (0.02981242451734576) [X1 Z3 Z4 X5]
+ (0.030104623143456813) [Y0 Z1 Z3 Y4]
+ (0.030104623143456813) [X0 Z1 Z3 X4]
+ (0.030104623143456813) [Y1 Z2 Z4 Y5]
+ (0.030104623143456813) [X1 Z2 Z4 X5]
+ (0.030787505389143956) [Y0 Z2 Z3 Y4]
+ (0.030787505389143956) [X0 Z2 Z3 X4]
+ (0.04375263801066032) [Y0 Z1 Z2 Z3 Y4]
+ (0.04375263801066032) [X0 Z1 Z2 Z3 X4]
+ (0.04375263801066033) [Y1 Z2 Z3 Z4 Y5]
+ (0.04375263801066033) [X1 Z2 Z3 Z4 X5]
+ (-0.01456453123117299) [Y1 Z2 Z3 X4 X6 Y7]
+ (-0.01456453123117299) [Y1 Z2 Z3 Y4 Y6 Y7]
+ (-0.01456453123117299) [X1 Z2 Z3 X4 X6 X7]
+ (-0.01456453123117299) [X1 Z2 Z3 Y4 Y6 X7]
+ (-6.524373848697595e-06) [Y0 Z1 Z2 Z3 Z4 Y6]
+ (-6.524373848697595e-06) [X0 Z1 Z2 Z3 Z4 X6]
+ (-6.524373848697595e-06) [Y1 Z2 Z3 Z5 Z6 Y7]
+ (-6.524373848697595e-06) [X1 Z2 Z3 Z5 Z6 X7]
+ (-3.769659452063997e-06) [Y0 Z2 Z3 Z4 Z5 Y6]
+ (-3.769659452063997e-06) [X0 Z2 Z3 Z4 Z5 X6]
+ (-3.610297130637156e-06) [Y0 Z1 Z3 Z4 Z5 Y6]
+ (-3.610297130637156e-06) [X0 Z1 Z3 Z4 Z5 X6]
+ (-3.610297130637156e-06) [Y1 Z2 Z4 Z5 Z6 Y7]
+ (-3.610297130637156e-06) [X1 Z2 Z4 Z5 Z6 X7]
+ (-3.31314550022649e-06) [Y1 Z2 Z3 Y4 X5 X6]
+ (-3.31314550022649e-06) [X1 Z2 Z3 X4 Y5 Y6]
+ (-3.277483195578502e-06) [Y0 Z1 Z2 Z4 Z5 Y6]
+ (-3.277483195578502e-06) [X0 Z1 Z2 Z4 Z5 X6]
+ (-3.277483195578502e-06) [Y1 Z3 Z4 Z5 Z6 Y7]
+ (-3.277483195578502e-06) [X1 Z3 Z4 Z5 Z6 X7]
+ (-3.2112283484711042e-06) [Y0 Z1 Z2 Z3 Z5 Y6]
+ (-3.2112283484711042e-06) [X0 Z1 Z2 Z3 Z5 X6]
+ (-3.2112283484711042e-06) [Y1 Z2 Z3 Z4 Z6 Y7]
+ (-3.2112283484711042e-06) [X1 Z2 Z3 Z4 Z6 X7]
+ (-1.0358477601716111e-06) [Y0 Y1 X4 Z5 Z6 X7]
+ (-1.0358477601716111e-06) [X0 X1 Y4 Z5 Z6 Y7]
+ (-6.628614201678811e-07) [Y2 X3 X4 Z5 Z6 Y7]
+ (-6.628614201678811e-07) [X2 Y3 Y4 Z5 Z6 X7]
+ (-3.3281393505865323e-07) [Y1 X2 X3 Z4 Z5 Y6]
+ (-3.3281393505865323e-07) [X1 Y2 Y3 Z4 Z5 X6]
+ (3.3281393505865323e-07) [Y1 Y2 X3 Z4 Z5 X6]
+ (3.3281393505865323e-07) [X1 X2 Y3 Z4 Z5 Y6]
+ (6.628614201678811e-07) [Y2 Y3 X4 Z5 Z6 X7]
+ (6.628614201678811e-07) [X2 X3 Y4 Z5 Z6 Y7]
+ (1.0358477601716111e-06) [Y0 X1 X4 Z5 Z6 Y7]
+ (1.0358477601716111e-06) [X0 Y1 Y4 Z5 Z6 X7]
+ (3.31314550022649e-06) [Y1 Z2 Z3 X4 X5 Y6]
+ (3.31314550022649e-06) [X1 Z2 Z3 Y4 Y5 X6]
+ (4.183932559437338e-06) [Y1 Z2 Z3 Z4 Z5 Y7]
+ (4.183932559437338e-06) [X1 Z2 Z3 Z4 Z5 X7]
+ (0.00029219862611105273) [Y0 Z1 Y2 Y3 Z4 Y5]
+ (0.00029219862611105273) [Y0 Z1 Y2 X3 Z4 X5]
+ (0.00029219862611105273) [X0 Z1 X2 Y3 Z4 Y5]
+ (0.00029219862611105273) [X0 Z1 X2 X3 Z4 X5]
+ (0.010540425907671534) [Y0 Z1 Z2 Z3 Y4 Z7]
+ (0.010540425907671534) [X0 Z1 Z2 Z3 X4 Z7]
+ (0.010540425907671534) [Y1 Z2 Z3 Z4 Y5 Z6]
+ (0.010540425907671534) [X1 Z2 Z3 Z4 X5 Z6]
+ (0.011307274008848166) [Y0 Z1 Z2 Z3 Y4 Z5]
+ (0.011307274008848166) [X0 Z1 Z2 Z3 X4 Z5]
+ (0.025104957138844527) [Y0 Z1 Z2 Z3 Y4 Z6]
+ (0.025104957138844527) [X0 Z1 Z2 Z3 X4 Z6]
+ (0.025104957138844527) [Y1 Z2 Z3 Z4 Y5 Z7]
+ (0.025104957138844527) [X1 Z2 Z3 Z4 X5 Z7]
+ (0.030787505389143956) [Z0 Y1 Z2 Z3 Z4 Y5]
+ (0.030787505389143956) [Z0 X1 Z2 Z3 Z4 X5]
+ (-5.105396549653351e-06) [Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-5.105396549653351e-06) [X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-5.105396549653347e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6]
+ (-5.105396549653347e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6]
+ (-0.01456453123117299) [Y0 Z1 Z2 Z3 Z4 Y5 X6 X7]
+ (-0.01456453123117299) [X0 Z1 Z2 Z3 Z4 X5 Y6 Y7]
+ (-3.769659452063997e-06) [Z0 Y1 Z2 Z3 Z4 Z5 Z6 Y7]
+ (-3.769659452063997e-06) [Z0 X1 Z2 Z3 Z4 Z5 Z6 X7]
+ (-3.3281393505865323e-07) [Y0 Z1 Y2 Y3 Z4 Z5 Z6 Y7]
+ (-3.3281393505865323e-07) [Y0 Z1 Y2 X3 Z4 Z5 Z6 X7]
+ (-3.3281393505865323e-07) [X0 Z1 X2 Y3 Z4 Z5 Z6 Y7]
+ (-3.3281393505865323e-07) [X0 Z1 X2 X3 Z4 Z5 Z6 X7]
+ (3.31314550022649e-06) [Y0 Z1 Z2 Z3 Y4 Y5 Z6 Y7]
+ (3.31314550022649e-06) [Y0 Z1 Z2 Z3 Y4 X5 Z6 X7]
+ (3.31314550022649e-06) [X0 Z1 Z2 Z3 X4 Y5 Z6 Y7]
+ (3.31314550022649e-06) [X0 Z1 Z2 Z3 X4 X5 Z6 X7]
+ (4.183932559437338e-06) [Y0 Z1 Z2 Z3 Z4 Z5 Y6 Z7]
+ (4.183932559437338e-06) [X0 Z1 Z2 Z3 Z4 Z5 X6 Z7]
+ (0.01456453123117299) [Y0 Z1 Z2 Z3 Z4 X5 X6 Y7]
+ (0.01456453123117299) [X0 Z1 Z2 Z3 Z4 Y5 Y6 X7]
 </code>
 </pre>
 </details>

---

## 24. tutorial_falqon.html <a name="demo23"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_falqon.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Step 1, Cost = -2.4265436197783443
Step 2, Cost = -5.451838418111181
Step 3, Cost = -5.058939064534096
Step 4, Cost = 0.6663779891077409
Step 5, Cost = -3.9617659191509684
Step 6, Cost = -6.012336027057502
Step 7, Cost = -6.383828240291033
Step 8, Cost = -6.568581722318139
Step 9, Cost = -6.652767426710391
Step 10, Cost = -6.718062615729144
Step 11, Cost = -6.763947743609292
Step 12, Cost = -6.804857466609769
Step 13, Cost = -6.839403058736172
Step 14, Cost = -6.871459263552874
Step 15, Cost = -6.899746975480946
Step 16, Cost = -6.925884328592724
Step 17, Cost = -6.949229507885626
Step 18, Cost = -6.970594125057243
Step 19, Cost = -6.989907329921352
Step 20, Cost = -7.0076231058226455
Step 21, Cost = -7.023986049880374
Step 22, Cost = -7.039304856521962
Step 23, Cost = -7.053894937286077
Step 24, Cost = -7.067988454154523
Step 25, Cost = -7.081842534715278
Step 26, Cost = -7.095617260802734
Step 27, Cost = -7.109472588274408
Step 29, Cost = -7.137684426026892
Step 30, Cost = -7.152041022693125
Step 31, Cost = -7.1664533102873
Step 32, Cost = -7.180748341609375
Step 33, Cost = -7.194694917926663
Step 34, Cost = -7.208028603663331
Step 35, Cost = -7.2204563958703645
Step 36, Cost = -7.2317273300321885
Step 37, Cost = -7.241565955502967
Step 38, Cost = -7.249767410209224
Step 39, Cost = -7.25578289566485
Step 40, Cost = -7.25898790701407
 </code>
 </pre>
 </details>

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_falqon.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Step 1, Cost = -2.4265436197783448
Step 2, Cost = -5.451838418111176
Step 3, Cost = -5.05893906453409
Step 4, Cost = 0.6663779891077449
Step 5, Cost = -3.9617659191509746
Step 6, Cost = -6.012336027057521
Step 7, Cost = -6.383828240291071
Step 8, Cost = -6.5685817223181155
Step 9, Cost = -6.652767426710387
Step 10, Cost = -6.718062615729132
Step 11, Cost = -6.763947743609312
Step 12, Cost = -6.80485746660975
Step 13, Cost = -6.8394030587362
Step 14, Cost = -6.871459263552881
Step 15, Cost = -6.899746975480981
Step 16, Cost = -6.925884328592719
Step 17, Cost = -6.949229507885613
Step 18, Cost = -6.970594125057218
Step 19, Cost = -6.989907329921348
Step 20, Cost = -7.007623105822645
Step 21, Cost = -7.023986049880396
Step 22, Cost = -7.03930485652197
Step 23, Cost = -7.053894937286083
Step 24, Cost = -7.067988454154528
Step 25, Cost = -7.081842534715245
Step 26, Cost = -7.095617260802699
Step 27, Cost = -7.109472588274404
Step 29, Cost = -7.137684426026871
Step 30, Cost = -7.152041022693112
Step 31, Cost = -7.166453310287315
Step 32, Cost = -7.1807483416093865
Step 33, Cost = -7.194694917926691
Step 34, Cost = -7.208028603663316
Step 35, Cost = -7.22045639587038
Step 36, Cost = -7.231727330032204
Step 37, Cost = -7.241565955502979
Step 38, Cost = -7.249767410209168
Step 39, Cost = -7.255782895664824
Step 40, Cost = -7.258987907014076
 </code>
 </pre>
 </details>

---

## 25. tutorial_backprop.html <a name="demo24"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_backprop.html):

```
Expectation value: -0.1197136570687156
-0.06518877224958129
-0.06518877224958129
[[-6.51887722e-02 -2.72891905e-02  1.38777878e-17 -9.33934621e-02
Forward pass (best of 3): 0.011305861799974082 sec per loop
Gradient computation (best of 3): 4.170899169899985 sec per loop
4.0701102479906694
0.9358535378025422
Forward pass (best of 3): 0.047940462099995786 sec per loop
Backward pass (best of 3): 0.10497711480002181 sec per loop
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_backprop.html):

```
Expectation value: -0.11971365706871569
-0.0651887722495813
-0.0651887722495813
[[-6.51887722e-02 -2.72891905e-02  0.00000000e+00 -9.33934621e-02
Forward pass (best of 3): 0.025541875700037055 sec per loop
Gradient computation (best of 3): 7.604229594299977 sec per loop
9.19507525201334
0.9358535378025427
Forward pass (best of 3): 0.07064138080004341 sec per loop
Backward pass (best of 3): 0.1398009409000224 sec per loop
```

---

## 26. tutorial_gbs.html <a name="demo25"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_gbs.html):

```
|1100>: 0.034732936494202844
|0101>: 0.011870900427255568
|1111>: 0.005957399165336124
|2000>: 0.029573843083205452
0.034732936494202844
0.011870900427255547
0.011870900427255568
0.0059573991653360855
0.005957399165336124
0.029573843083205372
0.029573843083205452
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_gbs.html):

```
|1100>: 0.034732936494202823
|0101>: 0.011870900427255577
|1111>: 0.005957399165336117
|2000>: 0.02957384308320544
0.034732936494202823
0.011870900427255558
0.011870900427255577
0.005957399165336081
0.005957399165336117
0.02957384308320539
0.02957384308320544
```

---

## 27. tutorial_quantum_transfer_learning.html <a name="demo26"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_quantum_transfer_learning.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
  2%|2         | 1.11M/44.7M [00:00<00:04, 10.3MB/s]
 14%|#4        | 6.45M/44.7M [00:00<00:01, 35.7MB/s]
 25%|##4       | 10.9M/44.7M [00:00<00:00, 40.3MB/s]
 36%|###6      | 16.1M/44.7M [00:00<00:00, 44.6MB/s]
 63%|######2   | 28.1M/44.7M [00:00<00:00, 72.6MB/s]
 80%|########  | 35.9M/44.7M [00:00<00:00, 75.8MB/s]
100%|##########| 44.7M/44.7M [00:00<00:00, 69.0MB/s]
Training started:
Phase: train Epoch: 1/3 Iter: 1/62 Batch time: 0.2179
Phase: train Epoch: 1/3 Iter: 2/62 Batch time: 0.1984
Phase: train Epoch: 1/3 Iter: 3/62 Batch time: 0.1979
Phase: train Epoch: 1/3 Iter: 4/62 Batch time: 0.2008
Phase: train Epoch: 1/3 Iter: 5/62 Batch time: 0.1998
Phase: train Epoch: 1/3 Iter: 6/62 Batch time: 0.2007
Phase: train Epoch: 1/3 Iter: 7/62 Batch time: 0.1976
Phase: train Epoch: 1/3 Iter: 8/62 Batch time: 0.2039
Phase: train Epoch: 1/3 Iter: 9/62 Batch time: 0.2029
Phase: train Epoch: 1/3 Iter: 10/62 Batch time: 0.2007
Phase: train Epoch: 1/3 Iter: 11/62 Batch time: 0.1960
Phase: train Epoch: 1/3 Iter: 12/62 Batch time: 0.2000
Phase: train Epoch: 1/3 Iter: 13/62 Batch time: 0.2013
Phase: train Epoch: 1/3 Iter: 14/62 Batch time: 0.1971
Phase: train Epoch: 1/3 Iter: 15/62 Batch time: 0.1961
Phase: train Epoch: 1/3 Iter: 16/62 Batch time: 0.1969
Phase: train Epoch: 1/3 Iter: 17/62 Batch time: 0.2003
Phase: train Epoch: 1/3 Iter: 18/62 Batch time: 0.1980
Phase: train Epoch: 1/3 Iter: 19/62 Batch time: 0.2032
Phase: train Epoch: 1/3 Iter: 20/62 Batch time: 0.1983
Phase: train Epoch: 1/3 Iter: 21/62 Batch time: 0.2071
Phase: train Epoch: 1/3 Iter: 22/62 Batch time: 0.2003
Phase: train Epoch: 1/3 Iter: 23/62 Batch time: 0.1988
Phase: train Epoch: 1/3 Iter: 24/62 Batch time: 0.1913
Phase: train Epoch: 1/3 Iter: 25/62 Batch time: 0.1954
Phase: train Epoch: 1/3 Iter: 26/62 Batch time: 0.1958
Phase: train Epoch: 1/3 Iter: 27/62 Batch time: 0.1949
Phase: train Epoch: 1/3 Iter: 28/62 Batch time: 0.1959
Phase: train Epoch: 1/3 Iter: 29/62 Batch time: 0.1947
Phase: train Epoch: 1/3 Iter: 30/62 Batch time: 0.2000
Phase: train Epoch: 1/3 Iter: 31/62 Batch time: 0.1999
Phase: train Epoch: 1/3 Iter: 32/62 Batch time: 0.2012
Phase: train Epoch: 1/3 Iter: 33/62 Batch time: 0.2018
Phase: train Epoch: 1/3 Iter: 34/62 Batch time: 0.1990
Phase: train Epoch: 1/3 Iter: 35/62 Batch time: 0.2000
Phase: train Epoch: 1/3 Iter: 36/62 Batch time: 0.2027
Phase: train Epoch: 1/3 Iter: 37/62 Batch time: 0.2004
Phase: train Epoch: 1/3 Iter: 38/62 Batch time: 0.1968
Phase: train Epoch: 1/3 Iter: 39/62 Batch time: 0.1988
Phase: train Epoch: 1/3 Iter: 40/62 Batch time: 0.1968
Phase: train Epoch: 1/3 Iter: 41/62 Batch time: 0.1952
Phase: train Epoch: 1/3 Iter: 42/62 Batch time: 0.1941
Phase: train Epoch: 1/3 Iter: 43/62 Batch time: 0.1959
Phase: train Epoch: 1/3 Iter: 44/62 Batch time: 0.1976
Phase: train Epoch: 1/3 Iter: 45/62 Batch time: 0.1979
Phase: train Epoch: 1/3 Iter: 46/62 Batch time: 0.1931
Phase: train Epoch: 1/3 Iter: 47/62 Batch time: 0.1917
Phase: train Epoch: 1/3 Iter: 48/62 Batch time: 0.1942
Phase: train Epoch: 1/3 Iter: 49/62 Batch time: 0.1943
Phase: train Epoch: 1/3 Iter: 50/62 Batch time: 0.1943
Phase: train Epoch: 1/3 Iter: 51/62 Batch time: 0.1990
Phase: train Epoch: 1/3 Iter: 52/62 Batch time: 0.1957
Phase: train Epoch: 1/3 Iter: 53/62 Batch time: 0.2000
Phase: train Epoch: 1/3 Iter: 54/62 Batch time: 0.1995
Phase: train Epoch: 1/3 Iter: 55/62 Batch time: 0.1945
Phase: train Epoch: 1/3 Iter: 56/62 Batch time: 0.1946
Phase: train Epoch: 1/3 Iter: 57/62 Batch time: 0.1932
Phase: train Epoch: 1/3 Iter: 58/62 Batch time: 0.1920
Phase: train Epoch: 1/3 Iter: 59/62 Batch time: 0.1888
Phase: train Epoch: 1/3 Iter: 60/62 Batch time: 0.1898
Phase: train Epoch: 1/3 Iter: 61/62 Batch time: 0.1930
Phase: train Epoch: 1/3 Loss: 0.6993 Acc: 0.5246
Phase: validation Epoch: 1/3 Iter: 1/39 Batch time: 0.1417
Phase: validation Epoch: 1/3 Iter: 2/39 Batch time: 0.1383
Phase: validation Epoch: 1/3 Iter: 3/39 Batch time: 0.1381
Phase: validation Epoch: 1/3 Iter: 4/39 Batch time: 0.1442
Phase: validation Epoch: 1/3 Iter: 5/39 Batch time: 0.1400
Phase: validation Epoch: 1/3 Iter: 6/39 Batch time: 0.1447
Phase: validation Epoch: 1/3 Iter: 7/39 Batch time: 0.1441
Phase: validation Epoch: 1/3 Iter: 8/39 Batch time: 0.1392
Phase: validation Epoch: 1/3 Iter: 9/39 Batch time: 0.1395
Phase: validation Epoch: 1/3 Iter: 10/39 Batch time: 0.1393
Phase: validation Epoch: 1/3 Iter: 11/39 Batch time: 0.1423
Phase: validation Epoch: 1/3 Iter: 12/39 Batch time: 0.1414
Phase: validation Epoch: 1/3 Iter: 13/39 Batch time: 0.1362
Phase: validation Epoch: 1/3 Iter: 14/39 Batch time: 0.1876
Phase: validation Epoch: 1/3 Iter: 15/39 Batch time: 0.2132
Phase: validation Epoch: 1/3 Iter: 16/39 Batch time: 0.1395
Phase: validation Epoch: 1/3 Iter: 17/39 Batch time: 0.1387
Phase: validation Epoch: 1/3 Iter: 18/39 Batch time: 0.1382
Phase: validation Epoch: 1/3 Iter: 19/39 Batch time: 0.1368
Phase: validation Epoch: 1/3 Iter: 20/39 Batch time: 0.1377
Phase: validation Epoch: 1/3 Iter: 21/39 Batch time: 0.1411
Phase: validation Epoch: 1/3 Iter: 22/39 Batch time: 0.1383
Phase: validation Epoch: 1/3 Iter: 23/39 Batch time: 0.1389
Phase: validation Epoch: 1/3 Iter: 24/39 Batch time: 0.1383
Phase: validation Epoch: 1/3 Iter: 25/39 Batch time: 0.1396
Phase: validation Epoch: 1/3 Iter: 26/39 Batch time: 0.1403
Phase: validation Epoch: 1/3 Iter: 27/39 Batch time: 0.1390
Phase: validation Epoch: 1/3 Iter: 28/39 Batch time: 0.1408
Phase: validation Epoch: 1/3 Iter: 29/39 Batch time: 0.1381
Phase: validation Epoch: 1/3 Iter: 30/39 Batch time: 0.1419
Phase: validation Epoch: 1/3 Iter: 31/39 Batch time: 0.1386
Phase: validation Epoch: 1/3 Iter: 32/39 Batch time: 0.1373
Phase: validation Epoch: 1/3 Iter: 33/39 Batch time: 0.1397
Phase: validation Epoch: 1/3 Iter: 34/39 Batch time: 0.1413
Phase: validation Epoch: 1/3 Iter: 35/39 Batch time: 0.1391
Phase: validation Epoch: 1/3 Iter: 36/39 Batch time: 0.1373
Phase: validation Epoch: 1/3 Iter: 37/39 Batch time: 0.1370
Phase: validation Epoch: 1/3 Iter: 38/39 Batch time: 0.1378
Phase: validation Epoch: 1/3 Iter: 39/39 Batch time: 0.0492
Phase: validation   Epoch: 1/3 Loss: 0.6432 Acc: 0.6536
Phase: train Epoch: 2/3 Iter: 1/62 Batch time: 0.1849
Phase: train Epoch: 2/3 Iter: 2/62 Batch time: 0.1904
Phase: train Epoch: 2/3 Iter: 3/62 Batch time: 0.1929
Phase: train Epoch: 2/3 Iter: 4/62 Batch time: 0.1918
Phase: train Epoch: 2/3 Iter: 5/62 Batch time: 0.1907
Phase: train Epoch: 2/3 Iter: 6/62 Batch time: 0.1957
Phase: train Epoch: 2/3 Iter: 7/62 Batch time: 0.1978
Phase: train Epoch: 2/3 Iter: 8/62 Batch time: 0.1954
Phase: train Epoch: 2/3 Iter: 9/62 Batch time: 0.1920
Phase: train Epoch: 2/3 Iter: 10/62 Batch time: 0.1921
Phase: train Epoch: 2/3 Iter: 11/62 Batch time: 0.1926
Phase: train Epoch: 2/3 Iter: 12/62 Batch time: 0.1927
Phase: train Epoch: 2/3 Iter: 13/62 Batch time: 0.1943
Phase: train Epoch: 2/3 Iter: 14/62 Batch time: 0.1950
Phase: train Epoch: 2/3 Iter: 15/62 Batch time: 0.1947
Phase: train Epoch: 2/3 Iter: 16/62 Batch time: 0.1967
Phase: train Epoch: 2/3 Iter: 17/62 Batch time: 0.1964
Phase: train Epoch: 2/3 Iter: 18/62 Batch time: 0.1894
Phase: train Epoch: 2/3 Iter: 19/62 Batch time: 0.1888
Phase: train Epoch: 2/3 Iter: 20/62 Batch time: 0.1899
Phase: train Epoch: 2/3 Iter: 21/62 Batch time: 0.1946
Phase: train Epoch: 2/3 Iter: 22/62 Batch time: 0.1916
Phase: train Epoch: 2/3 Iter: 23/62 Batch time: 0.1920
Phase: train Epoch: 2/3 Iter: 24/62 Batch time: 0.1921
Phase: train Epoch: 2/3 Iter: 25/62 Batch time: 0.1957
Phase: train Epoch: 2/3 Iter: 26/62 Batch time: 0.1952
Phase: train Epoch: 2/3 Iter: 27/62 Batch time: 0.1949
Phase: train Epoch: 2/3 Iter: 28/62 Batch time: 0.1939
Phase: train Epoch: 2/3 Iter: 29/62 Batch time: 0.1890
Phase: train Epoch: 2/3 Iter: 30/62 Batch time: 0.1884
Phase: train Epoch: 2/3 Iter: 31/62 Batch time: 0.1892
Phase: train Epoch: 2/3 Iter: 32/62 Batch time: 0.1929
Phase: train Epoch: 2/3 Iter: 33/62 Batch time: 0.1901
Phase: train Epoch: 2/3 Iter: 34/62 Batch time: 0.1866
Phase: train Epoch: 2/3 Iter: 35/62 Batch time: 0.1937
Phase: train Epoch: 2/3 Iter: 36/62 Batch time: 0.1971
Phase: train Epoch: 2/3 Iter: 37/62 Batch time: 0.1992
Phase: train Epoch: 2/3 Iter: 38/62 Batch time: 0.1941
Phase: train Epoch: 2/3 Iter: 39/62 Batch time: 0.1979
Phase: train Epoch: 2/3 Iter: 40/62 Batch time: 0.1911
Phase: train Epoch: 2/3 Iter: 41/62 Batch time: 0.1899
Phase: train Epoch: 2/3 Iter: 42/62 Batch time: 0.1899
Phase: train Epoch: 2/3 Iter: 43/62 Batch time: 0.1879
Phase: train Epoch: 2/3 Iter: 44/62 Batch time: 0.1939
Phase: train Epoch: 2/3 Iter: 45/62 Batch time: 0.1906
Phase: train Epoch: 2/3 Iter: 46/62 Batch time: 0.1991
Phase: train Epoch: 2/3 Iter: 47/62 Batch time: 0.2002
Phase: train Epoch: 2/3 Iter: 48/62 Batch time: 0.1932
Phase: train Epoch: 2/3 Iter: 49/62 Batch time: 0.1927
Phase: train Epoch: 2/3 Iter: 50/62 Batch time: 0.1923
Phase: train Epoch: 2/3 Iter: 51/62 Batch time: 0.1910
Phase: train Epoch: 2/3 Iter: 52/62 Batch time: 0.1963
Phase: train Epoch: 2/3 Iter: 53/62 Batch time: 0.1978
Phase: train Epoch: 2/3 Iter: 54/62 Batch time: 0.1947
Phase: train Epoch: 2/3 Iter: 55/62 Batch time: 0.1922
Phase: train Epoch: 2/3 Iter: 56/62 Batch time: 0.1947
Phase: train Epoch: 2/3 Iter: 57/62 Batch time: 0.1979
Phase: train Epoch: 2/3 Iter: 58/62 Batch time: 0.1994
Phase: train Epoch: 2/3 Iter: 59/62 Batch time: 0.2028
Phase: train Epoch: 2/3 Iter: 60/62 Batch time: 0.2036
Phase: train Epoch: 2/3 Iter: 61/62 Batch time: 0.2050
Phase: train Epoch: 2/3 Loss: 0.6141 Acc: 0.7049
Phase: validation Epoch: 2/3 Iter: 1/39 Batch time: 0.1479
Phase: validation Epoch: 2/3 Iter: 2/39 Batch time: 0.1408
Phase: validation Epoch: 2/3 Iter: 3/39 Batch time: 0.1452
Phase: validation Epoch: 2/3 Iter: 4/39 Batch time: 0.1425
Phase: validation Epoch: 2/3 Iter: 5/39 Batch time: 0.1389
Phase: validation Epoch: 2/3 Iter: 6/39 Batch time: 0.1396
Phase: validation Epoch: 2/3 Iter: 7/39 Batch time: 0.1416
Phase: validation Epoch: 2/3 Iter: 8/39 Batch time: 0.1411
Phase: validation Epoch: 2/3 Iter: 9/39 Batch time: 0.1399
Phase: validation Epoch: 2/3 Iter: 10/39 Batch time: 0.1394
Phase: validation Epoch: 2/3 Iter: 11/39 Batch time: 0.1395
Phase: validation Epoch: 2/3 Iter: 12/39 Batch time: 0.1394
Phase: validation Epoch: 2/3 Iter: 13/39 Batch time: 0.1400
Phase: validation Epoch: 2/3 Iter: 14/39 Batch time: 0.1364
Phase: validation Epoch: 2/3 Iter: 15/39 Batch time: 0.1388
Phase: validation Epoch: 2/3 Iter: 16/39 Batch time: 0.1389
Phase: validation Epoch: 2/3 Iter: 17/39 Batch time: 0.1432
Phase: validation Epoch: 2/3 Iter: 18/39 Batch time: 0.1419
Phase: validation Epoch: 2/3 Iter: 19/39 Batch time: 0.1426
Phase: validation Epoch: 2/3 Iter: 20/39 Batch time: 0.1413
Phase: validation Epoch: 2/3 Iter: 21/39 Batch time: 0.1400
Phase: validation Epoch: 2/3 Iter: 22/39 Batch time: 0.1391
Phase: validation Epoch: 2/3 Iter: 23/39 Batch time: 0.1429
Phase: validation Epoch: 2/3 Iter: 24/39 Batch time: 0.1419
Phase: validation Epoch: 2/3 Iter: 25/39 Batch time: 0.1395
Phase: validation Epoch: 2/3 Iter: 26/39 Batch time: 0.1413
Phase: validation Epoch: 2/3 Iter: 27/39 Batch time: 0.1416
Phase: validation Epoch: 2/3 Iter: 28/39 Batch time: 0.1411
Phase: validation Epoch: 2/3 Iter: 29/39 Batch time: 0.1419
Phase: validation Epoch: 2/3 Iter: 30/39 Batch time: 0.1408
Phase: validation Epoch: 2/3 Iter: 31/39 Batch time: 0.1382
Phase: validation Epoch: 2/3 Iter: 32/39 Batch time: 0.1403
Phase: validation Epoch: 2/3 Iter: 33/39 Batch time: 0.1419
Phase: validation Epoch: 2/3 Iter: 34/39 Batch time: 0.1415
Phase: validation Epoch: 2/3 Iter: 35/39 Batch time: 0.1408
Phase: validation Epoch: 2/3 Iter: 36/39 Batch time: 0.1396
Phase: validation Epoch: 2/3 Iter: 37/39 Batch time: 0.1406
Phase: validation Epoch: 2/3 Iter: 38/39 Batch time: 0.1436
Phase: validation Epoch: 2/3 Iter: 39/39 Batch time: 0.0457
Phase: validation   Epoch: 2/3 Loss: 0.5392 Acc: 0.8235
Phase: train Epoch: 3/3 Iter: 1/62 Batch time: 0.1917
Phase: train Epoch: 3/3 Iter: 2/62 Batch time: 0.1953
Phase: train Epoch: 3/3 Iter: 3/62 Batch time: 0.1975
Phase: train Epoch: 3/3 Iter: 4/62 Batch time: 0.1971
Phase: train Epoch: 3/3 Iter: 5/62 Batch time: 0.1974
Phase: train Epoch: 3/3 Iter: 6/62 Batch time: 0.2016
Phase: train Epoch: 3/3 Iter: 7/62 Batch time: 0.2074
Phase: train Epoch: 3/3 Iter: 8/62 Batch time: 0.1980
Phase: train Epoch: 3/3 Iter: 9/62 Batch time: 0.1981
Phase: train Epoch: 3/3 Iter: 10/62 Batch time: 0.2015
Phase: train Epoch: 3/3 Iter: 11/62 Batch time: 0.1991
Phase: train Epoch: 3/3 Iter: 12/62 Batch time: 0.1997
Phase: train Epoch: 3/3 Iter: 13/62 Batch time: 0.2001
Phase: train Epoch: 3/3 Iter: 14/62 Batch time: 0.1962
Phase: train Epoch: 3/3 Iter: 15/62 Batch time: 0.2008
Phase: train Epoch: 3/3 Iter: 16/62 Batch time: 0.1979
Phase: train Epoch: 3/3 Iter: 17/62 Batch time: 0.1976
Phase: train Epoch: 3/3 Iter: 18/62 Batch time: 0.1936
Phase: train Epoch: 3/3 Iter: 19/62 Batch time: 0.2021
Phase: train Epoch: 3/3 Iter: 20/62 Batch time: 0.1983
Phase: train Epoch: 3/3 Iter: 21/62 Batch time: 0.1949
Phase: train Epoch: 3/3 Iter: 22/62 Batch time: 0.1987
Phase: train Epoch: 3/3 Iter: 23/62 Batch time: 0.2040
Phase: train Epoch: 3/3 Iter: 24/62 Batch time: 0.2018
Phase: train Epoch: 3/3 Iter: 25/62 Batch time: 0.2009
Phase: train Epoch: 3/3 Iter: 26/62 Batch time: 0.1956
Phase: train Epoch: 3/3 Iter: 27/62 Batch time: 0.1965
Phase: train Epoch: 3/3 Iter: 28/62 Batch time: 0.1982
Phase: train Epoch: 3/3 Iter: 29/62 Batch time: 0.2011
Phase: train Epoch: 3/3 Iter: 30/62 Batch time: 0.1979
Phase: train Epoch: 3/3 Iter: 31/62 Batch time: 0.1975
Phase: train Epoch: 3/3 Iter: 32/62 Batch time: 0.1992
Phase: train Epoch: 3/3 Iter: 33/62 Batch time: 0.1962
Phase: train Epoch: 3/3 Iter: 34/62 Batch time: 0.2071
Phase: train Epoch: 3/3 Iter: 35/62 Batch time: 0.1981
Phase: train Epoch: 3/3 Iter: 36/62 Batch time: 0.1973
Phase: train Epoch: 3/3 Iter: 37/62 Batch time: 0.1983
Phase: train Epoch: 3/3 Iter: 38/62 Batch time: 0.1987
Phase: train Epoch: 3/3 Iter: 39/62 Batch time: 0.2023
Phase: train Epoch: 3/3 Iter: 40/62 Batch time: 0.1982
Phase: train Epoch: 3/3 Iter: 41/62 Batch time: 0.2009
Phase: train Epoch: 3/3 Iter: 42/62 Batch time: 0.2010
Phase: train Epoch: 3/3 Iter: 43/62 Batch time: 0.2013
Phase: train Epoch: 3/3 Iter: 44/62 Batch time: 0.2011
Phase: train Epoch: 3/3 Iter: 45/62 Batch time: 0.1981
Phase: train Epoch: 3/3 Iter: 46/62 Batch time: 0.2014
Phase: train Epoch: 3/3 Iter: 47/62 Batch time: 0.1989
Phase: train Epoch: 3/3 Iter: 48/62 Batch time: 0.2009
Phase: train Epoch: 3/3 Iter: 49/62 Batch time: 0.1980
Phase: train Epoch: 3/3 Iter: 50/62 Batch time: 0.2087
Phase: train Epoch: 3/3 Iter: 51/62 Batch time: 0.1979
Phase: train Epoch: 3/3 Iter: 52/62 Batch time: 0.2025
Phase: train Epoch: 3/3 Iter: 53/62 Batch time: 0.2014
Phase: train Epoch: 3/3 Iter: 54/62 Batch time: 0.2034
Phase: train Epoch: 3/3 Iter: 55/62 Batch time: 0.2005
Phase: train Epoch: 3/3 Iter: 56/62 Batch time: 0.1981
Phase: train Epoch: 3/3 Iter: 57/62 Batch time: 0.2018
Phase: train Epoch: 3/3 Iter: 58/62 Batch time: 0.1986
Phase: train Epoch: 3/3 Iter: 59/62 Batch time: 0.1936
Phase: train Epoch: 3/3 Iter: 60/62 Batch time: 0.1999
Phase: train Epoch: 3/3 Iter: 61/62 Batch time: 0.2005
Phase: train Epoch: 3/3 Loss: 0.5652 Acc: 0.7336
Phase: validation Epoch: 3/3 Iter: 1/39 Batch time: 0.1519
Phase: validation Epoch: 3/3 Iter: 2/39 Batch time: 0.1494
Phase: validation Epoch: 3/3 Iter: 3/39 Batch time: 0.1686
Phase: validation Epoch: 3/3 Iter: 4/39 Batch time: 0.1453
Phase: validation Epoch: 3/3 Iter: 5/39 Batch time: 0.1435
Phase: validation Epoch: 3/3 Iter: 6/39 Batch time: 0.1466
Phase: validation Epoch: 3/3 Iter: 7/39 Batch time: 0.1406
Phase: validation Epoch: 3/3 Iter: 8/39 Batch time: 0.1435
Phase: validation Epoch: 3/3 Iter: 9/39 Batch time: 0.1399
Phase: validation Epoch: 3/3 Iter: 10/39 Batch time: 0.1408
Phase: validation Epoch: 3/3 Iter: 11/39 Batch time: 0.1379
Phase: validation Epoch: 3/3 Iter: 12/39 Batch time: 0.1369
Phase: validation Epoch: 3/3 Iter: 13/39 Batch time: 0.1376
Phase: validation Epoch: 3/3 Iter: 14/39 Batch time: 0.1404
Phase: validation Epoch: 3/3 Iter: 15/39 Batch time: 0.1417
Phase: validation Epoch: 3/3 Iter: 16/39 Batch time: 0.1402
Phase: validation Epoch: 3/3 Iter: 17/39 Batch time: 0.1380
Phase: validation Epoch: 3/3 Iter: 18/39 Batch time: 0.1375
Phase: validation Epoch: 3/3 Iter: 19/39 Batch time: 0.1395
Phase: validation Epoch: 3/3 Iter: 20/39 Batch time: 0.1412
Phase: validation Epoch: 3/3 Iter: 21/39 Batch time: 0.1397
Phase: validation Epoch: 3/3 Iter: 22/39 Batch time: 0.1379
Phase: validation Epoch: 3/3 Iter: 23/39 Batch time: 0.1390
Phase: validation Epoch: 3/3 Iter: 24/39 Batch time: 0.1386
Phase: validation Epoch: 3/3 Iter: 25/39 Batch time: 0.1410
Phase: validation Epoch: 3/3 Iter: 26/39 Batch time: 0.1403
Phase: validation Epoch: 3/3 Iter: 27/39 Batch time: 0.1441
Phase: validation Epoch: 3/3 Iter: 28/39 Batch time: 0.1428
Phase: validation Epoch: 3/3 Iter: 29/39 Batch time: 0.1434
Phase: validation Epoch: 3/3 Iter: 30/39 Batch time: 0.1424
Phase: validation Epoch: 3/3 Iter: 31/39 Batch time: 0.1434
Phase: validation Epoch: 3/3 Iter: 32/39 Batch time: 0.1434
Phase: validation Epoch: 3/3 Iter: 33/39 Batch time: 0.1436
Phase: validation Epoch: 3/3 Iter: 34/39 Batch time: 0.1390
Phase: validation Epoch: 3/3 Iter: 35/39 Batch time: 0.1411
Phase: validation Epoch: 3/3 Iter: 36/39 Batch time: 0.1409
Phase: validation Epoch: 3/3 Iter: 37/39 Batch time: 0.1397
Phase: validation Epoch: 3/3 Iter: 38/39 Batch time: 0.1413
Phase: validation Epoch: 3/3 Iter: 39/39 Batch time: 0.0457
Phase: validation   Epoch: 3/3 Loss: 0.4484 Acc: 0.8497
Training completed in 1m 1s
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
  1%|1         | 528k/44.7M [00:00<00:08, 5.38MB/s]
 13%|#2        | 5.63M/44.7M [00:00<00:01, 33.7MB/s]
 28%|##7       | 12.4M/44.7M [00:00<00:00, 50.6MB/s]
 43%|####3     | 19.4M/44.7M [00:00<00:00, 59.6MB/s]
 59%|#####8    | 26.3M/44.7M [00:00<00:00, 64.3MB/s]
 73%|#######2  | 32.5M/44.7M [00:00<00:00, 39.7MB/s]
 85%|########5 | 38.0M/44.7M [00:00<00:00, 44.0MB/s]
100%|##########| 44.7M/44.7M [00:00<00:00, 47.8MB/s]
Training started:
Phase: train Epoch: 1/3 Iter: 1/62 Batch time: 0.3847
Phase: train Epoch: 1/3 Iter: 2/62 Batch time: 0.3819
Phase: train Epoch: 1/3 Iter: 3/62 Batch time: 0.3584
Phase: train Epoch: 1/3 Iter: 4/62 Batch time: 0.3765
Phase: train Epoch: 1/3 Iter: 5/62 Batch time: 0.3494
Phase: train Epoch: 1/3 Iter: 6/62 Batch time: 0.3624
Phase: train Epoch: 1/3 Iter: 7/62 Batch time: 0.3522
Phase: train Epoch: 1/3 Iter: 8/62 Batch time: 0.3666
Phase: train Epoch: 1/3 Iter: 9/62 Batch time: 0.3904
Phase: train Epoch: 1/3 Iter: 10/62 Batch time: 0.3744
Phase: train Epoch: 1/3 Iter: 11/62 Batch time: 0.3620
Phase: train Epoch: 1/3 Iter: 12/62 Batch time: 0.3702
Phase: train Epoch: 1/3 Iter: 13/62 Batch time: 0.3560
Phase: train Epoch: 1/3 Iter: 14/62 Batch time: 0.3702
Phase: train Epoch: 1/3 Iter: 15/62 Batch time: 0.3675
Phase: train Epoch: 1/3 Iter: 16/62 Batch time: 0.3652
Phase: train Epoch: 1/3 Iter: 17/62 Batch time: 0.3850
Phase: train Epoch: 1/3 Iter: 18/62 Batch time: 0.3728
Phase: train Epoch: 1/3 Iter: 19/62 Batch time: 0.3678
Phase: train Epoch: 1/3 Iter: 20/62 Batch time: 0.3805
Phase: train Epoch: 1/3 Iter: 21/62 Batch time: 0.3726
Phase: train Epoch: 1/3 Iter: 22/62 Batch time: 0.3709
Phase: train Epoch: 1/3 Iter: 23/62 Batch time: 0.3795
Phase: train Epoch: 1/3 Iter: 24/62 Batch time: 0.3717
Phase: train Epoch: 1/3 Iter: 25/62 Batch time: 0.3665
Phase: train Epoch: 1/3 Iter: 26/62 Batch time: 0.3605
Phase: train Epoch: 1/3 Iter: 27/62 Batch time: 0.3768
Phase: train Epoch: 1/3 Iter: 28/62 Batch time: 0.3771
Phase: train Epoch: 1/3 Iter: 29/62 Batch time: 0.3679
Phase: train Epoch: 1/3 Iter: 30/62 Batch time: 0.3678
Phase: train Epoch: 1/3 Iter: 31/62 Batch time: 0.3692
Phase: train Epoch: 1/3 Iter: 32/62 Batch time: 0.3698
Phase: train Epoch: 1/3 Iter: 33/62 Batch time: 0.3727
Phase: train Epoch: 1/3 Iter: 34/62 Batch time: 0.3798
Phase: train Epoch: 1/3 Iter: 35/62 Batch time: 0.3661
Phase: train Epoch: 1/3 Iter: 36/62 Batch time: 0.3870
Phase: train Epoch: 1/3 Iter: 37/62 Batch time: 0.3792
Phase: train Epoch: 1/3 Iter: 38/62 Batch time: 0.3702
Phase: train Epoch: 1/3 Iter: 39/62 Batch time: 0.3789
Phase: train Epoch: 1/3 Iter: 40/62 Batch time: 0.3605
Phase: train Epoch: 1/3 Iter: 41/62 Batch time: 0.3754
Phase: train Epoch: 1/3 Iter: 42/62 Batch time: 0.3645
Phase: train Epoch: 1/3 Iter: 43/62 Batch time: 0.3726
Phase: train Epoch: 1/3 Iter: 44/62 Batch time: 0.3645
Phase: train Epoch: 1/3 Iter: 45/62 Batch time: 0.3660
Phase: train Epoch: 1/3 Iter: 46/62 Batch time: 0.3909
Phase: train Epoch: 1/3 Iter: 47/62 Batch time: 0.3651
Phase: train Epoch: 1/3 Iter: 48/62 Batch time: 0.3686
Phase: train Epoch: 1/3 Iter: 49/62 Batch time: 0.3837
Phase: train Epoch: 1/3 Iter: 50/62 Batch time: 0.3930
Phase: train Epoch: 1/3 Iter: 51/62 Batch time: 0.3692
Phase: train Epoch: 1/3 Iter: 52/62 Batch time: 0.3782
Phase: train Epoch: 1/3 Iter: 53/62 Batch time: 0.3883
Phase: train Epoch: 1/3 Iter: 54/62 Batch time: 0.3737
Phase: train Epoch: 1/3 Iter: 55/62 Batch time: 0.4004
Phase: train Epoch: 1/3 Iter: 56/62 Batch time: 0.3760
Phase: train Epoch: 1/3 Iter: 57/62 Batch time: 0.3847
Phase: train Epoch: 1/3 Iter: 58/62 Batch time: 0.3790
Phase: train Epoch: 1/3 Iter: 59/62 Batch time: 0.3812
Phase: train Epoch: 1/3 Iter: 60/62 Batch time: 0.3662
Phase: train Epoch: 1/3 Iter: 61/62 Batch time: 0.3715
Phase: train Epoch: 1/3 Loss: 0.6993 Acc: 0.5246
Phase: validation Epoch: 1/3 Iter: 1/39 Batch time: 0.2925
Phase: validation Epoch: 1/3 Iter: 2/39 Batch time: 0.2942
Phase: validation Epoch: 1/3 Iter: 3/39 Batch time: 0.2960
Phase: validation Epoch: 1/3 Iter: 4/39 Batch time: 0.3098
Phase: validation Epoch: 1/3 Iter: 5/39 Batch time: 0.2853
Phase: validation Epoch: 1/3 Iter: 6/39 Batch time: 0.2890
Phase: validation Epoch: 1/3 Iter: 7/39 Batch time: 0.3052
Phase: validation Epoch: 1/3 Iter: 8/39 Batch time: 0.2977
Phase: validation Epoch: 1/3 Iter: 9/39 Batch time: 0.3148
Phase: validation Epoch: 1/3 Iter: 10/39 Batch time: 0.3018
Phase: validation Epoch: 1/3 Iter: 11/39 Batch time: 0.3204
Phase: validation Epoch: 1/3 Iter: 12/39 Batch time: 0.2920
Phase: validation Epoch: 1/3 Iter: 13/39 Batch time: 0.2920
Phase: validation Epoch: 1/3 Iter: 14/39 Batch time: 0.3059
Phase: validation Epoch: 1/3 Iter: 15/39 Batch time: 0.3169
Phase: validation Epoch: 1/3 Iter: 16/39 Batch time: 0.3197
Phase: validation Epoch: 1/3 Iter: 17/39 Batch time: 0.2856
Phase: validation Epoch: 1/3 Iter: 18/39 Batch time: 0.3176
Phase: validation Epoch: 1/3 Iter: 19/39 Batch time: 0.2917
Phase: validation Epoch: 1/3 Iter: 20/39 Batch time: 0.3002
Phase: validation Epoch: 1/3 Iter: 21/39 Batch time: 0.3129
Phase: validation Epoch: 1/3 Iter: 22/39 Batch time: 0.3167
Phase: validation Epoch: 1/3 Iter: 23/39 Batch time: 0.3046
Phase: validation Epoch: 1/3 Iter: 24/39 Batch time: 0.3057
Phase: validation Epoch: 1/3 Iter: 25/39 Batch time: 0.3026
Phase: validation Epoch: 1/3 Iter: 26/39 Batch time: 0.3059
Phase: validation Epoch: 1/3 Iter: 27/39 Batch time: 0.3060
Phase: validation Epoch: 1/3 Iter: 28/39 Batch time: 0.3022
Phase: validation Epoch: 1/3 Iter: 29/39 Batch time: 0.3068
Phase: validation Epoch: 1/3 Iter: 30/39 Batch time: 0.3017
Phase: validation Epoch: 1/3 Iter: 31/39 Batch time: 0.2951
Phase: validation Epoch: 1/3 Iter: 32/39 Batch time: 0.3079
Phase: validation Epoch: 1/3 Iter: 33/39 Batch time: 0.3018
Phase: validation Epoch: 1/3 Iter: 34/39 Batch time: 0.3006
Phase: validation Epoch: 1/3 Iter: 35/39 Batch time: 0.2976
Phase: validation Epoch: 1/3 Iter: 36/39 Batch time: 0.3060
Phase: validation Epoch: 1/3 Iter: 37/39 Batch time: 0.2928
Phase: validation Epoch: 1/3 Iter: 38/39 Batch time: 0.2975
Phase: validation Epoch: 1/3 Iter: 39/39 Batch time: 0.0902
Phase: validation   Epoch: 1/3 Loss: 0.6432 Acc: 0.6536
Phase: train Epoch: 2/3 Iter: 1/62 Batch time: 0.3644
Phase: train Epoch: 2/3 Iter: 2/62 Batch time: 0.3843
Phase: train Epoch: 2/3 Iter: 3/62 Batch time: 0.3695
Phase: train Epoch: 2/3 Iter: 4/62 Batch time: 0.3626
Phase: train Epoch: 2/3 Iter: 5/62 Batch time: 0.3706
Phase: train Epoch: 2/3 Iter: 6/62 Batch time: 0.3647
Phase: train Epoch: 2/3 Iter: 7/62 Batch time: 0.3727
Phase: train Epoch: 2/3 Iter: 8/62 Batch time: 0.3881
Phase: train Epoch: 2/3 Iter: 9/62 Batch time: 0.3730
Phase: train Epoch: 2/3 Iter: 10/62 Batch time: 0.3657
Phase: train Epoch: 2/3 Iter: 11/62 Batch time: 0.3473
Phase: train Epoch: 2/3 Iter: 12/62 Batch time: 0.3777
Phase: train Epoch: 2/3 Iter: 13/62 Batch time: 0.3772
Phase: train Epoch: 2/3 Iter: 14/62 Batch time: 0.3675
Phase: train Epoch: 2/3 Iter: 15/62 Batch time: 0.3731
Phase: train Epoch: 2/3 Iter: 16/62 Batch time: 0.3759
Phase: train Epoch: 2/3 Iter: 17/62 Batch time: 0.3667
Phase: train Epoch: 2/3 Iter: 18/62 Batch time: 0.3669
Phase: train Epoch: 2/3 Iter: 19/62 Batch time: 0.3654
Phase: train Epoch: 2/3 Iter: 20/62 Batch time: 0.3779
Phase: train Epoch: 2/3 Iter: 21/62 Batch time: 0.3730
Phase: train Epoch: 2/3 Iter: 22/62 Batch time: 0.3651
Phase: train Epoch: 2/3 Iter: 23/62 Batch time: 0.3713
Phase: train Epoch: 2/3 Iter: 24/62 Batch time: 0.3644
Phase: train Epoch: 2/3 Iter: 25/62 Batch time: 0.3705
Phase: train Epoch: 2/3 Iter: 26/62 Batch time: 0.3654
Phase: train Epoch: 2/3 Iter: 27/62 Batch time: 0.3926
Phase: train Epoch: 2/3 Iter: 28/62 Batch time: 0.3732
Phase: train Epoch: 2/3 Iter: 29/62 Batch time: 0.3707
Phase: train Epoch: 2/3 Iter: 30/62 Batch time: 0.3714
Phase: train Epoch: 2/3 Iter: 31/62 Batch time: 0.3876
Phase: train Epoch: 2/3 Iter: 32/62 Batch time: 0.3719
Phase: train Epoch: 2/3 Iter: 33/62 Batch time: 0.3724
Phase: train Epoch: 2/3 Iter: 34/62 Batch time: 0.3684
Phase: train Epoch: 2/3 Iter: 35/62 Batch time: 0.3723
Phase: train Epoch: 2/3 Iter: 36/62 Batch time: 0.3631
Phase: train Epoch: 2/3 Iter: 37/62 Batch time: 0.3776
Phase: train Epoch: 2/3 Iter: 38/62 Batch time: 0.3722
Phase: train Epoch: 2/3 Iter: 39/62 Batch time: 0.3770
Phase: train Epoch: 2/3 Iter: 40/62 Batch time: 0.3778
Phase: train Epoch: 2/3 Iter: 41/62 Batch time: 0.3682
Phase: train Epoch: 2/3 Iter: 42/62 Batch time: 0.3688
Phase: train Epoch: 2/3 Iter: 43/62 Batch time: 0.3672
Phase: train Epoch: 2/3 Iter: 44/62 Batch time: 0.3621
Phase: train Epoch: 2/3 Iter: 45/62 Batch time: 0.3606
Phase: train Epoch: 2/3 Iter: 46/62 Batch time: 0.3641
Phase: train Epoch: 2/3 Iter: 47/62 Batch time: 0.3643
Phase: train Epoch: 2/3 Iter: 48/62 Batch time: 0.3658
Phase: train Epoch: 2/3 Iter: 49/62 Batch time: 0.3586
Phase: train Epoch: 2/3 Iter: 50/62 Batch time: 0.3715
Phase: train Epoch: 2/3 Iter: 51/62 Batch time: 0.3761
Phase: train Epoch: 2/3 Iter: 52/62 Batch time: 0.3815
Phase: train Epoch: 2/3 Iter: 53/62 Batch time: 0.3681
Phase: train Epoch: 2/3 Iter: 54/62 Batch time: 0.3589
Phase: train Epoch: 2/3 Iter: 55/62 Batch time: 0.3649
Phase: train Epoch: 2/3 Iter: 56/62 Batch time: 0.3738
Phase: train Epoch: 2/3 Iter: 57/62 Batch time: 0.3764
Phase: train Epoch: 2/3 Iter: 58/62 Batch time: 0.3564
Phase: train Epoch: 2/3 Iter: 59/62 Batch time: 0.3644
Phase: train Epoch: 2/3 Iter: 60/62 Batch time: 0.3733
Phase: train Epoch: 2/3 Iter: 61/62 Batch time: 0.3705
Phase: train Epoch: 2/3 Loss: 0.6141 Acc: 0.7049
Phase: validation Epoch: 2/3 Iter: 1/39 Batch time: 0.3046
Phase: validation Epoch: 2/3 Iter: 2/39 Batch time: 0.3031
Phase: validation Epoch: 2/3 Iter: 3/39 Batch time: 0.3003
Phase: validation Epoch: 2/3 Iter: 4/39 Batch time: 0.3047
Phase: validation Epoch: 2/3 Iter: 5/39 Batch time: 0.3036
Phase: validation Epoch: 2/3 Iter: 6/39 Batch time: 0.3049
Phase: validation Epoch: 2/3 Iter: 7/39 Batch time: 0.2914
Phase: validation Epoch: 2/3 Iter: 8/39 Batch time: 0.3066
Phase: validation Epoch: 2/3 Iter: 9/39 Batch time: 0.2954
Phase: validation Epoch: 2/3 Iter: 10/39 Batch time: 0.2999
Phase: validation Epoch: 2/3 Iter: 11/39 Batch time: 0.2883
Phase: validation Epoch: 2/3 Iter: 12/39 Batch time: 0.2968
Phase: validation Epoch: 2/3 Iter: 13/39 Batch time: 0.3067
Phase: validation Epoch: 2/3 Iter: 14/39 Batch time: 0.3029
Phase: validation Epoch: 2/3 Iter: 15/39 Batch time: 0.3041
Phase: validation Epoch: 2/3 Iter: 16/39 Batch time: 0.2885
Phase: validation Epoch: 2/3 Iter: 17/39 Batch time: 0.2915
Phase: validation Epoch: 2/3 Iter: 18/39 Batch time: 0.3040
Phase: validation Epoch: 2/3 Iter: 19/39 Batch time: 0.3089
Phase: validation Epoch: 2/3 Iter: 20/39 Batch time: 0.2974
Phase: validation Epoch: 2/3 Iter: 21/39 Batch time: 0.2942
Phase: validation Epoch: 2/3 Iter: 22/39 Batch time: 0.2959
Phase: validation Epoch: 2/3 Iter: 23/39 Batch time: 0.2966
Phase: validation Epoch: 2/3 Iter: 24/39 Batch time: 0.2975
Phase: validation Epoch: 2/3 Iter: 25/39 Batch time: 0.2960
Phase: validation Epoch: 2/3 Iter: 26/39 Batch time: 0.3106
Phase: validation Epoch: 2/3 Iter: 27/39 Batch time: 0.3159
Phase: validation Epoch: 2/3 Iter: 28/39 Batch time: 0.3084
Phase: validation Epoch: 2/3 Iter: 29/39 Batch time: 0.3075
Phase: validation Epoch: 2/3 Iter: 30/39 Batch time: 0.3119
Phase: validation Epoch: 2/3 Iter: 31/39 Batch time: 0.3075
Phase: validation Epoch: 2/3 Iter: 32/39 Batch time: 0.2989
Phase: validation Epoch: 2/3 Iter: 33/39 Batch time: 0.3025
Phase: validation Epoch: 2/3 Iter: 34/39 Batch time: 0.3075
Phase: validation Epoch: 2/3 Iter: 35/39 Batch time: 0.3100
Phase: validation Epoch: 2/3 Iter: 36/39 Batch time: 0.3159
Phase: validation Epoch: 2/3 Iter: 37/39 Batch time: 0.3070
Phase: validation Epoch: 2/3 Iter: 38/39 Batch time: 0.3120
Phase: validation Epoch: 2/3 Iter: 39/39 Batch time: 0.0880
Phase: validation   Epoch: 2/3 Loss: 0.5392 Acc: 0.8235
Phase: train Epoch: 3/3 Iter: 1/62 Batch time: 0.3816
Phase: train Epoch: 3/3 Iter: 2/62 Batch time: 0.3803
Phase: train Epoch: 3/3 Iter: 3/62 Batch time: 0.3817
Phase: train Epoch: 3/3 Iter: 4/62 Batch time: 0.3863
Phase: train Epoch: 3/3 Iter: 5/62 Batch time: 0.3858
Phase: train Epoch: 3/3 Iter: 6/62 Batch time: 0.3769
Phase: train Epoch: 3/3 Iter: 7/62 Batch time: 0.3816
Phase: train Epoch: 3/3 Iter: 8/62 Batch time: 0.3764
Phase: train Epoch: 3/3 Iter: 9/62 Batch time: 0.3793
Phase: train Epoch: 3/3 Iter: 10/62 Batch time: 0.3710
Phase: train Epoch: 3/3 Iter: 11/62 Batch time: 0.3678
Phase: train Epoch: 3/3 Iter: 12/62 Batch time: 0.3646
Phase: train Epoch: 3/3 Iter: 13/62 Batch time: 0.3747
Phase: train Epoch: 3/3 Iter: 14/62 Batch time: 0.3755
Phase: train Epoch: 3/3 Iter: 15/62 Batch time: 0.3804
Phase: train Epoch: 3/3 Iter: 16/62 Batch time: 0.3744
Phase: train Epoch: 3/3 Iter: 17/62 Batch time: 0.3625
Phase: train Epoch: 3/3 Iter: 18/62 Batch time: 0.3827
Phase: train Epoch: 3/3 Iter: 19/62 Batch time: 0.3754
Phase: train Epoch: 3/3 Iter: 20/62 Batch time: 0.3677
Phase: train Epoch: 3/3 Iter: 21/62 Batch time: 0.3815
Phase: train Epoch: 3/3 Iter: 22/62 Batch time: 0.3806
Phase: train Epoch: 3/3 Iter: 23/62 Batch time: 0.3757
Phase: train Epoch: 3/3 Iter: 24/62 Batch time: 0.3743
Phase: train Epoch: 3/3 Iter: 25/62 Batch time: 0.3776
Phase: train Epoch: 3/3 Iter: 26/62 Batch time: 0.3596
Phase: train Epoch: 3/3 Iter: 27/62 Batch time: 0.3693
Phase: train Epoch: 3/3 Iter: 28/62 Batch time: 0.3812
Phase: train Epoch: 3/3 Iter: 29/62 Batch time: 0.3716
Phase: train Epoch: 3/3 Iter: 30/62 Batch time: 0.3848
Phase: train Epoch: 3/3 Iter: 31/62 Batch time: 0.3781
Phase: train Epoch: 3/3 Iter: 32/62 Batch time: 0.3830
Phase: train Epoch: 3/3 Iter: 33/62 Batch time: 0.3929
Phase: train Epoch: 3/3 Iter: 34/62 Batch time: 0.3771
Phase: train Epoch: 3/3 Iter: 35/62 Batch time: 0.3733
Phase: train Epoch: 3/3 Iter: 36/62 Batch time: 0.3707
Phase: train Epoch: 3/3 Iter: 37/62 Batch time: 0.3791
Phase: train Epoch: 3/3 Iter: 38/62 Batch time: 0.3775
Phase: train Epoch: 3/3 Iter: 39/62 Batch time: 0.3724
Phase: train Epoch: 3/3 Iter: 40/62 Batch time: 0.3630
Phase: train Epoch: 3/3 Iter: 41/62 Batch time: 0.4034
Phase: train Epoch: 3/3 Iter: 42/62 Batch time: 0.3839
Phase: train Epoch: 3/3 Iter: 43/62 Batch time: 0.3693
Phase: train Epoch: 3/3 Iter: 44/62 Batch time: 0.3717
Phase: train Epoch: 3/3 Iter: 45/62 Batch time: 0.3849
Phase: train Epoch: 3/3 Iter: 46/62 Batch time: 0.3824
Phase: train Epoch: 3/3 Iter: 47/62 Batch time: 0.3746
Phase: train Epoch: 3/3 Iter: 48/62 Batch time: 0.3755
Phase: train Epoch: 3/3 Iter: 49/62 Batch time: 0.3781
Phase: train Epoch: 3/3 Iter: 50/62 Batch time: 0.3835
Phase: train Epoch: 3/3 Iter: 51/62 Batch time: 0.3833
Phase: train Epoch: 3/3 Iter: 52/62 Batch time: 0.3814
Phase: train Epoch: 3/3 Iter: 53/62 Batch time: 0.3829
Phase: train Epoch: 3/3 Iter: 54/62 Batch time: 0.3842
Phase: train Epoch: 3/3 Iter: 55/62 Batch time: 0.3824
Phase: train Epoch: 3/3 Iter: 56/62 Batch time: 0.3788
Phase: train Epoch: 3/3 Iter: 57/62 Batch time: 0.3790
Phase: train Epoch: 3/3 Iter: 58/62 Batch time: 0.3777
Phase: train Epoch: 3/3 Iter: 59/62 Batch time: 0.3769
Phase: train Epoch: 3/3 Iter: 60/62 Batch time: 0.3746
Phase: train Epoch: 3/3 Iter: 61/62 Batch time: 0.3691
Phase: train Epoch: 3/3 Loss: 0.5652 Acc: 0.7336
Phase: validation Epoch: 3/3 Iter: 1/39 Batch time: 0.3094
Phase: validation Epoch: 3/3 Iter: 2/39 Batch time: 0.3101
Phase: validation Epoch: 3/3 Iter: 3/39 Batch time: 0.3070
Phase: validation Epoch: 3/3 Iter: 4/39 Batch time: 0.3029
Phase: validation Epoch: 3/3 Iter: 5/39 Batch time: 0.3003
Phase: validation Epoch: 3/3 Iter: 6/39 Batch time: 0.3045
Phase: validation Epoch: 3/3 Iter: 7/39 Batch time: 0.2986
Phase: validation Epoch: 3/3 Iter: 8/39 Batch time: 0.3305
Phase: validation Epoch: 3/3 Iter: 9/39 Batch time: 0.2987
Phase: validation Epoch: 3/3 Iter: 10/39 Batch time: 0.3076
Phase: validation Epoch: 3/3 Iter: 11/39 Batch time: 0.2988
Phase: validation Epoch: 3/3 Iter: 12/39 Batch time: 0.3064
Phase: validation Epoch: 3/3 Iter: 13/39 Batch time: 0.3105
Phase: validation Epoch: 3/3 Iter: 14/39 Batch time: 0.3023
Phase: validation Epoch: 3/3 Iter: 15/39 Batch time: 0.3113
Phase: validation Epoch: 3/3 Iter: 16/39 Batch time: 0.3065
Phase: validation Epoch: 3/3 Iter: 17/39 Batch time: 0.3083
Phase: validation Epoch: 3/3 Iter: 18/39 Batch time: 0.3072
Phase: validation Epoch: 3/3 Iter: 19/39 Batch time: 0.3173
Phase: validation Epoch: 3/3 Iter: 20/39 Batch time: 0.3012
Phase: validation Epoch: 3/3 Iter: 21/39 Batch time: 0.3046
Phase: validation Epoch: 3/3 Iter: 22/39 Batch time: 0.3051
Phase: validation Epoch: 3/3 Iter: 23/39 Batch time: 0.2981
Phase: validation Epoch: 3/3 Iter: 24/39 Batch time: 0.2996
Phase: validation Epoch: 3/3 Iter: 25/39 Batch time: 0.3041
Phase: validation Epoch: 3/3 Iter: 26/39 Batch time: 0.3085
Phase: validation Epoch: 3/3 Iter: 27/39 Batch time: 0.2970
Phase: validation Epoch: 3/3 Iter: 28/39 Batch time: 0.3206
Phase: validation Epoch: 3/3 Iter: 29/39 Batch time: 0.3049
Phase: validation Epoch: 3/3 Iter: 30/39 Batch time: 0.3112
Phase: validation Epoch: 3/3 Iter: 31/39 Batch time: 0.3028
Phase: validation Epoch: 3/3 Iter: 32/39 Batch time: 0.3104
Phase: validation Epoch: 3/3 Iter: 33/39 Batch time: 0.2965
Phase: validation Epoch: 3/3 Iter: 34/39 Batch time: 0.2977
Phase: validation Epoch: 3/3 Iter: 35/39 Batch time: 0.3107
Phase: validation Epoch: 3/3 Iter: 36/39 Batch time: 0.3058
Phase: validation Epoch: 3/3 Iter: 37/39 Batch time: 0.2988
Phase: validation Epoch: 3/3 Iter: 38/39 Batch time: 0.3010
Phase: validation Epoch: 3/3 Iter: 39/39 Batch time: 0.0825
Phase: validation   Epoch: 3/3 Loss: 0.4484 Acc: 0.8497
Training completed in 1m 53s
 </code>
 </pre>
 </details>

---

## 28. tutorial_classical_shadows.html <a name="demo27"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_classical_shadows.html):

```
(0.16156422871415568+0j)
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_classical_shadows.html):

```
(0.16156422871415568+9.967876155406687e-20j)
```

---

## 29. tutorial_jax_transformations.html <a name="demo28"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_jax_transformations.html):

```
No jit time: 0.0078 seconds
First run time: 0.0585 seconds
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_jax_transformations.html):

```
No jit time: 0.0112 seconds
First run time: 0.0812 seconds
```

---

## 30. tutorial_barren_plateaus.html <a name="demo29"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_barren_plateaus.html):

```
Mean of the gradients for 200 random circuits: -0.0010002268976521363
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_barren_plateaus.html):

```
Mean of the gradients for 200 random circuits: -0.0010002268976521357
```

---

## 31. tutorial_qubit_rotation.html <a name="demo30"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_qubit_rotation.html):

```
0.8515405859048368
```

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_qubit_rotation.html):

```
0.8515405859048366
```

---

## 32. tutorial_QGAN.html <a name="demo31"></a>

---

[Master](https://pennylane.ai/qml/demos/tutorial_QGAN.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Step 0: cost = -0.05727684497833252
Step 5: cost = -0.26348113268613815
Step 10: cost = -0.4273916110396385
Step 15: cost = -0.47261594980955124
Step 20: cost = -0.48406897485256195
Step 25: cost = -0.48946401476860046
Step 30: cost = -0.4928189888596535
Step 35: cost = -0.4949494078755379
Step 40: cost = -0.49627023935317993
Step 45: cost = -0.4970719963312149
Prob(real classified as real):  0.998587042093277
Prob(fake classified as real):  0.5011128261685371
Step 0: cost = -0.5833386331796646
Step 5: cost = -0.8915729522705078
Step 15: cost = -0.9946483373641968
Step 20: cost = -0.9984995424747467
Step 25: cost = -0.9995636343955994
Step 30: cost = -0.999871701002121
Step 45: cost = -0.9999965131282806
Prob(fake classified as real):  0.9999986290931702
Discriminator cost:  0.0014115869998931885
Real Bloch vector: [-0.21694186  0.4504844  -0.86602527]
Generator Bloch vector: [-0.28404635  0.41893214 -0.8624441 ]
 </code>
 </pre>
 </details>

[Dev](http://pennylane.ai-dev.s3-website-us-east-1.amazonaws.com/qml/demos/tutorial_QGAN.html):

<details> 
 <summary>
 More 
 </summary>
 <pre>
 <code>
Step 0: cost = -0.05727699398994446
Step 5: cost = -0.26348118484020233
Step 10: cost = -0.4273917078971863
Step 15: cost = -0.47261589020490646
Step 20: cost = -0.48406901210546494
Step 25: cost = -0.4894639030098915
Step 30: cost = -0.49281900376081467
Step 35: cost = -0.4949493855237961
Step 40: cost = -0.49627020210027695
Step 45: cost = -0.49707192927598953
Prob(real classified as real):  0.9985870718955994
Prob(fake classified as real):  0.5011127963662148
Step 0: cost = -0.583338625729084
Step 5: cost = -0.8915732204914093
Step 15: cost = -0.9946482479572296
Step 20: cost = -0.9984994232654572
Step 25: cost = -0.9995635747909546
Step 30: cost = -0.9998717308044434
Step 45: cost = -0.9999965727329254
Prob(fake classified as real):  0.9999985992908478
Discriminator cost:  0.001411527395248413
Real Bloch vector: [-0.21694186  0.45048442 -0.86602521]
Generator Bloch vector: [-0.28404653  0.41893214 -0.86244416]
 </code>
 </pre>
 </details>

---

